# -*- coding: utf-8 -*-
"""判别式探针: 用 LoRA 模型的 LM 末层隐状态训 50 类线性头, 替代类名似然排序.

动机: evaluate.py 把 50 个类名当文本逐条打分, 读出头太弱 (似然排序), 会把
模型本身的判别能力压低。这里冻结模型, 只抽每张图的隐状态, 训一个线性分类
头, 得到"模型特征本身"的分类上限。

用法:
  python scripts/probe_eval.py --ckpt outputs/e4_lora --split test
  python scripts/probe_eval.py --ckpt outputs/e4_lora --train_limit 3000   # 先小样本摸底
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_ROOT = os.path.join(ROOT, "data")
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
QUESTION = "What action is the person performing in this radar point cloud?"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, HERE)
from m4config import ACTION_NAMES, S2_TEST, S2_VAL


def gather(split, per_pair, limit, skip_flip):
    """按 (subject, action) 均匀取帧, 返回 [(path, action_id), ...]."""
    img_dir = os.path.join(DATA_ROOT, "images", split)
    keep = {"test": set(S2_TEST), "val": set(S2_VAL)}.get(split)
    by_pair = {}
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(".png"):
            continue
        if skip_flip and "flip" in fn:
            continue
        parts = fn[:-4].split("_")
        s = int(parts[0][1:])
        a = int(parts[1][1:])
        if keep is not None and s not in keep:
            continue
        by_pair.setdefault((s, a), []).append(fn)
    items = []
    for (s, a), lst in sorted(by_pair.items()):
        idx = np.linspace(0, len(lst) - 1, min(per_pair, len(lst))).round().astype(int)
        for i in sorted(set(idx)):
            items.append((os.path.join(img_dir, lst[i]), a))
    if limit:
        items = items[:limit]
    return items


def extract(model, proc, items, tag, batch_size):
    """逐批前向, 取最后一个非 pad token 的末层隐状态."""
    feats, labels = [], []
    n = len(items)
    for b0 in range(0, n, batch_size):
        chunk = items[b0:b0 + batch_size]
        imgs = [Image.open(p).convert("RGB") for p, _ in chunk]
        msgs = [[{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": QUESTION}]}] for im in imgs]
        pd = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True)
        pd = {k: v.to(DEVICE) for k, v in pd.items()}
        with torch.no_grad():
            out = model(**pd, output_hidden_states=True)
        hs = out.hidden_states[-1]                      # [B, L, D]
        am = pd["attention_mask"]
        last = am.sum(1) - 1                            # 最后一个有效 token
        idx = last.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        vec = hs.gather(1, idx).squeeze(1).float().cpu().numpy()
        feats.append(vec)
        labels.extend(a for _, a in chunk)
        done = min(b0 + batch_size, n)
        if done % (batch_size * 10) == 0 or done == n:
            print(f"[{tag}] {done}/{n} features", flush=True)
    return np.concatenate(feats), np.asarray(labels, dtype=np.int64)


def train_probe(x_tr, y_tr, epochs, lr, seed=0):
    """线性头: 特征标准化 + 单层 Linear, 交叉熵."""
    mu = x_tr.mean(0, keepdims=True)
    sd = x_tr.std(0, keepdims=True) + 1e-6
    x_tr = (x_tr - mu) / sd
    xt = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr - 1, dtype=torch.long, device=DEVICE)
    head = torch.nn.Linear(xt.size(1), 50).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    torch.manual_seed(seed)
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(xt.size(0), device=DEVICE)
        tot = 0.0
        for i in range(0, xt.size(0), bs):
            sel = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(head(xt[sel]), yt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * sel.numel()
        sched.step()
        print(f"[probe] epoch {ep+1}/{epochs} loss {tot/xt.size(0):.4f}", flush=True)
    return head, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e4_lora")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per_pair", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0,
                    help="训练特征只取前 N 张 (摸底用, 0=全量)")
    ap.add_argument("--train_per_pair", type=int, default=1,
                    help="每个 (subject, action) 对抽多少帧做训练特征")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reuse", action="store_true",
                    help="复用已有特征缓存")
    args = ap.parse_args()

    ckpt = os.path.abspath(os.path.join(ROOT, args.ckpt))
    tag = os.path.basename(ckpt.rstrip("/"))
    out_dir = os.path.join(ROOT, "outputs", "probe", tag)
    os.makedirs(out_dir, exist_ok=True)

    tr_items = gather("train", args.train_per_pair, args.train_limit, skip_flip=True)
    te_items = gather(args.split, args.per_pair, args.limit, skip_flip=True)
    print(f"train clips {len(tr_items)}, {args.split} clips {len(te_items)}", flush=True)

    f_tr = os.path.join(out_dir, f"feat_train_{len(tr_items)}.npz")
    f_te = os.path.join(out_dir, f"feat_{args.split}_{len(te_items)}.npz")

    proc = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ckpt).to(DEVICE).eval()

    if args.reuse and os.path.exists(f_tr):
        d = np.load(f_tr)
        x_tr, y_tr = d["x"], d["y"]
        print(f"reuse {f_tr}", flush=True)
    else:
        x_tr, y_tr = extract(model, proc, tr_items, "train", args.batch)
        np.savez_compressed(f_tr, x=x_tr, y=y_tr)

    if args.reuse and os.path.exists(f_te):
        d = np.load(f_te)
        x_te, y_te = d["x"], d["y"]
        print(f"reuse {f_te}", flush=True)
    else:
        x_te, y_te = extract(model, proc, te_items, args.split, args.batch)
        np.savez_compressed(f_te, x=x_te, y=y_te)

    head, mu, sd = train_probe(x_tr, y_tr, args.epochs, args.lr)
    x_te_n = torch.tensor((x_te - mu) / sd, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = head(x_te_n)
    order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
    gt = y_te - 1
    top1 = float((order[:, 0] == gt).mean())
    top5 = float((order[:, :5] == gt[:, None]).any(1).mean())
    per_act = {}
    for a in range(1, 51):
        m = gt == a - 1
        if m.any():
            per_act[ACTION_NAMES[a]] = float((order[m, 0] == a - 1).mean())
    res = {"ckpt": args.ckpt, "split": args.split, "n_train": len(y_tr),
           "n_test": len(y_te), "probe_top1": top1, "probe_top5": top5,
           "per_action": per_act}
    with open(os.path.join(out_dir, f"probe_{args.split}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"PROBE_TOP1 {top1:.4f} PROBE_TOP5 {top5:.4f}", flush=True)


if __name__ == "__main__":
    main()
