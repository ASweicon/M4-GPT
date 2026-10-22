# -*- coding: utf-8 -*-
"""A 方案: 特征增强 + 分组超参搜索的判别式探针 (最终版).

背景 (为什么要有这个脚本):
  E4/E5 的 50 类准确率长期停在 0.24~0.30, 但同一份权重换成"冻结模型 + 线性头"
  读出就有 0.34/0.755 (top5), 说明瓶颈在读出方式 (50 个类名逐条算似然), 不是特征。
  本脚本把读出做到该权重特征的上限, 用来回答"模型到底看懂了多少"。

相对 probe_eval.py 的改进:
  1. 训练特征掺入镜像样本, 标签用 swap_lr 互换 —— 镜像版外观对应镜像类, 监督一致;
     这直接抑制 61% 的左右镜像错误 (E5 混淆矩阵里最大的错误源);
  2. 特征 = 末 token 隐状态 || 图像 token 均值池化(末层) || 图像 token 均值池化(中层);
  3. 交叉验证按"被试"分组 (GroupKFold), 与 test 的跨被试口径一致, 避免用被试指纹
     选超参导致虚高; 只在 CV 里选一次超参, test 只碰一次;
  4. 多种子平均 + 特征标准化, 减少随机波动;
  5. 特征缓存落盘 (键含 train_per_pair / limit / 特征模式), 重跑只做头训练, 分钟级。

用法:
  python scripts/probe_aug.py --ckpt outputs/e4_lora --tag e4_aug --limit 0
  python scripts/probe_aug.py --ckpt outputs/e5_lora --tag e5_aug --limit 0
  python scripts/probe_ensemble.py --tags e4_aug e5_aug
"""
import argparse
import itertools
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
sys.path.insert(0, HERE)
from m4config import ACTION_NAMES, S2_TEST, S2_VAL

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
QUESTION = "What action is the person performing in this radar point cloud?"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 标签工具 ----------------

def swap_name(n):
    """文本级 left<->right 互换 (与 m4config.swap_lr 等价, 用占位符防连环替换)."""
    t = n.replace("left", "\x00").replace("right", "left").replace("\x00", "right")
    t = t.replace("Left", "\x01").replace("Right", "Left").replace("\x01", "Right")
    return t


def swap_id_map():
    name2id = {v: k for k, v in ACTION_NAMES.items()}
    return {i: name2id.get(swap_name(n), i) for i, n in ACTION_NAMES.items()}


def merge_group_ids():
    """左右镜像对合并后的组 id (50 类 -> 39 组)."""
    smap = swap_id_map()
    keys, gid = {}, np.zeros(51, dtype=np.int64)
    for i in sorted(ACTION_NAMES):
        pair = tuple(sorted([ACTION_NAMES[i], ACTION_NAMES[smap[i]]]))
        if pair not in keys:
            keys[pair] = len(keys)
        gid[i] = keys[pair]
    return gid, len(keys)


# ---------------- 样本收集 ----------------

def parse_name(fn):
    """s12_a03_f133flip.png -> (12, 3, True); 返回 None 表示命名不符."""
    stem = fn[:-4]
    is_flip = stem.endswith("flip")
    if is_flip:
        stem = stem[:-4]
    p = stem.split("_")
    if len(p) != 3 or not p[0].startswith("s") or not p[1].startswith("a"):
        return None
    try:
        return int(p[0][1:]), int(p[1][1:]), is_flip
    except ValueError:
        return None


def take_even(lst, k):
    """在 lst 里等间隔取 k 个, 保证覆盖整段帧序而不是只看开头."""
    if k >= len(lst):
        return list(lst)
    idx = np.linspace(0, len(lst) - 1, k).round().astype(int)
    return [lst[i] for i in sorted(set(idx))]


def gather_train(per_pair):
    """train: 每 (被试, 动作) 的原图/镜像各取 per_pair 张.

    返回 [(path, action_id, is_flip, subject)]。
    镜像文件是 build_dataset.py 写的 ``s{s}_a{a}_f{fc}flip.png``。
    若一张镜像都没找到, 直接报错 —— 静默丢掉增强会让整个实验结论失真。
    """
    img_dir = os.path.join(ROOT, "data", "images", "train")
    by_pair = {}
    n_flip = 0
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(".png"):
            continue
        meta = parse_name(fn)
        if meta is None:
            continue
        s, a, is_flip = meta
        n_flip += int(is_flip)
        by_pair.setdefault((s, a, is_flip), []).append(fn)
    n_orig = sum(len(v) for k, v in by_pair.items() if not k[2])
    items = []
    for (s, a, is_flip), lst in sorted(by_pair.items()):
        for fn in take_even(lst, per_pair):
            items.append((os.path.join(img_dir, fn), a, is_flip, s))
    share = n_flip / max(1, n_flip + n_orig)
    print(f"[gather] train files: orig={n_orig} flip={n_flip} "
          f"flip_share={share:.3f}, sampled={len(items)}", flush=True)
    if n_flip == 0:
        raise SystemExit(
            "!! data/images/train 里找不到镜像样本 (*flip.png)。\n"
            "   A 方案的镜像增强完全依赖它们, 静默继续会得到误导性结论。\n"
            "   请先确认 E6 数据重建已完成 (bash scripts/run_e6_data.sh), 或检查文件名规则。")
    if share < 0.35:
        print(f"!! 警告: 镜像样本占比仅 {share:.1%}, 预期约 50% (原始/镜像各半)", flush=True)
    return items


def gather_split(split, limit):
    """val/test: 不取镜像. limit=0 表示全量."""
    img_dir = os.path.join(ROOT, "data", "images", split)
    keep = {"test": set(S2_TEST), "val": set(S2_VAL)}.get(split)
    by_pair = {}
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(".png"):
            continue
        meta = parse_name(fn)
        if meta is None:
            continue
        s, a, is_flip = meta
        if is_flip or (keep is not None and s not in keep):
            continue
        by_pair.setdefault((s, a), []).append(fn)
    items = []
    for (s, a), lst in sorted(by_pair.items()):
        # limit=0 -> 全量; 否则按约 200 个 (被试,动作) 对估算每对帧数
        k = len(lst) if not limit else min(len(lst), max(1, int(np.ceil(limit / 200.0))))
        for fn in take_even(lst, k):
            items.append((os.path.join(img_dir, fn), a, False, s))
    if limit:
        items = items[:limit]
    return items


# ---------------- 特征抽取 ----------------

def extract(model, proc, items, tag, batch, img_tok_id, smap):
    """返回 (特征, 标签, 被试, 是否镜像). 特征维度见 load_or_extract 的说明."""
    feats, labels, subs, flips = [], [], [], []
    n = len(items)
    for b0 in range(0, n, batch):
        chunk = items[b0:b0 + batch]
        imgs = [Image.open(p).convert("RGB") for p, _, _, _ in chunk]
        msgs = [[{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": QUESTION}]}] for im in imgs]
        pd = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt", padding=True)
        pd = {k: v.to(DEV) for k, v in pd.items()}
        with torch.no_grad():
            out = model(**pd, output_hidden_states=True)
        am = pd["attention_mask"]
        mask = (pd["input_ids"] == img_tok_id).unsqueeze(-1)
        denom = mask.sum(1).clamp(min=1)
        blocks = []
        mid = len(out.hidden_states) // 2      # 中层特征: 线性探针常用做法, 比末层更"通用"
        for hs in (out.hidden_states[mid], out.hidden_states[-1]):
            last = am.sum(1) - 1
            blocks.append(hs.gather(
                1, last.view(-1, 1, 1).expand(-1, 1, hs.size(-1))).squeeze(1))
            blocks.append((hs * mask).sum(1) / denom)
        # 顺序固定为 [中层末token, 中层图均值, 末层末token, 末层图均值]
        feats.append(torch.cat(blocks, 1).float().cpu().numpy())
        for _, a, is_flip, s in chunk:
            labels.append(smap.get(a, a) if is_flip else a)
            subs.append(s)
            flips.append(int(is_flip))
        done = min(b0 + batch, n)
        if done % (batch * 20) == 0 or done == n:
            print(f"[{tag}] {done}/{n}", flush=True)
    return (np.concatenate(feats), np.asarray(labels, dtype=np.int64),
            np.asarray(subs, dtype=np.int64), np.asarray(flips, dtype=np.int8))


def load_or_extract(args, tag, out_dir, smap):
    """抽/读特征. 缓存键含 train_per_pair 与 limit, 避免不同采样设置互相覆盖.

    特征 = [中层末token || 中层图均值 || 末层末token || 末层图均值], 每块 1536 维.
    """
    f_tr = os.path.join(out_dir, f"feat_train_pp{args.train_per_pair}.npz")
    f_te = os.path.join(out_dir, f"feat_test_lim{args.limit}.npz")
    if os.path.exists(f_tr) and os.path.exists(f_te) and not args.no_cache:
        tr, te = np.load(f_tr), np.load(f_te)
        print(f"[cache] {f_tr} {tr['x'].shape} / {f_te} {te['x'].shape}", flush=True)
        return (tr["x"], tr["y"], tr["s"], tr["f"],
                te["x"], te["y"], te["s"], te["f"])

    tr_items = gather_train(args.train_per_pair)
    te_items = gather_split("test", args.limit)
    print(f"[gather] clips: train {len(tr_items)} test {len(te_items)}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    img_tok_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, os.path.join(ROOT, args.ckpt)).to(DEV).eval()
    print(f"[model] dev={DEV}", flush=True)

    x_tr, y_tr, s_tr, f_tr_flag = extract(
        model, proc, tr_items, "train", args.batch, img_tok_id, smap)
    np.savez_compressed(f_tr, x=x_tr, y=y_tr, s=s_tr, f=f_tr_flag)
    x_te, y_te, s_te, f_te_flag = extract(
        model, proc, te_items, "test", args.batch, img_tok_id, smap)
    np.savez_compressed(f_te, x=x_te, y=y_te, s=s_te, f=f_te_flag)
    return x_tr, y_tr, s_tr, f_tr_flag, x_te, y_te, s_te, f_te_flag


# ---------------- 头训练与评估 ----------------

class CosHead(torch.nn.Module):
    """余弦分类头 (特征与权重都 L2 归一化). VLM 特征上常优于普通线性头:
    不受特征范数随被试/视角漂移的影响, 跨被试更稳."""

    def __init__(self, dim, n_cls=50, scale=20.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n_cls, dim) * 0.02)
        self.scale = scale

    def forward(self, x):
        xn = torch.nn.functional.normalize(x, dim=1)
        wn = torch.nn.functional.normalize(self.w, dim=1)
        return self.scale * (xn @ wn.t())


def make_head(kind, dim):
    if kind == "linear":
        return torch.nn.Linear(dim, 50)
    if kind == "cos":
        return CosHead(dim)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(dim),
        torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 50))


def fit(x, y, kind, lr, wd, epochs, seed=0):
    torch.manual_seed(seed)
    head = make_head(kind, x.size(1)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    yt = torch.tensor(y - 1, dtype=torch.long, device=DEV)
    for _ in range(epochs):
        perm = torch.randperm(x.size(0), device=DEV)
        for i in range(0, x.size(0), 1024):
            sel = perm[i:i + 1024]
            loss = torch.nn.functional.cross_entropy(head(x[sel]), yt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    head.eval()
    return head


def predict(head, x):
    with torch.no_grad():
        return head(x).float().cpu().numpy()


def metrics(logits, y, gid, n_group):
    probs = torch.softmax(torch.tensor(logits), 1).numpy()
    gt = y - 1
    order = np.argsort(-probs, 1)
    out = {"fine_top1": float((order[:, 0] == gt).mean()),
           "fine_top5": float((order[:, :5] == gt[:, None]).any(1).mean())}
    gp = np.zeros((probs.shape[0], n_group))
    for j in range(50):
        gp[:, gid[j + 1]] += probs[:, j]
    g = gid[y]
    go = np.argsort(-gp, 1)
    out["merged_top1"] = float((go[:, 0] == g).mean())
    out["merged_top5"] = float((go[:, :5] == g[:, None]).any(1).mean())
    out["merged_n"] = int(n_group)
    return out, probs


def grouped_folds(subjects, k):
    """按被试分组切折: 折内被试互斥, 与 test 的跨被试口径一致."""
    uniq = np.unique(subjects)
    rng = np.random.RandomState(0)
    rng.shuffle(uniq)
    groups = np.array_split(uniq, min(k, len(uniq)))
    return [{i for i, s in enumerate(subjects) if s in g} for g in groups]


def random_folds(n, k):
    """同被试口径: 随机切折 (同一被试的样本会同时出现在训练折和验证折).
    与 grouped_folds 形成"同被试 vs 跨被试"对照, 其余条件完全相同."""
    idx = np.random.RandomState(1).permutation(n)
    return [set(a.tolist()) for a in np.array_split(idx, min(k, n))]


def cv_score(xt, y_tr, cfg, folds):
    """用固定超参跑一遍 CV, 返回 (平均准确率, 各折准确率)."""
    kind, lr, wd, eps = cfg
    accs = []
    for f in folds:
        hold = np.fromiter(f, dtype=np.int64, count=len(f))
        tr_idx = np.setdiff1d(np.arange(len(y_tr)), hold)
        head = fit(xt[tr_idx], y_tr[tr_idx], kind, lr, wd, eps)
        p = torch.softmax(torch.tensor(predict(head, xt[hold])), 1).numpy()
        accs.append(float((p.argmax(1) == (y_tr[hold] - 1)).mean()))
    return float(np.mean(accs)), accs


def evaluate_features(x_tr, y_tr, s_tr, x_te, y_te, epochs, epoch_grid, folds_n,
                      seeds, log_prefix=""):
    """标准化 -> 按被试分组 CV 选头/超参 -> 全量重训多种子平均 -> test 指标.

    返回 (指标 dict(含 best_cfg/cv_acc), test 概率矩阵)。被 probe_ensemble 复用。
    """
    mu = x_tr.mean(0, keepdims=True)
    sd = x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32, device=DEV)
    xv = torch.tensor((x_te - mu) / sd, dtype=torch.float32, device=DEV)

    folds = grouped_folds(s_tr, folds_n)
    ep_grid = ([int(x) for x in epoch_grid.split(",") if x.strip()]
               if epoch_grid else [epochs])
    grid = list(itertools.product(("linear", "cos", "mlp"),
                                  (3e-3, 1e-2), (1e-2, 1e-1), ep_grid))
    best, best_cfg = -1.0, None
    for kind, lr, wd, eps in grid:
        accs = []
        for f in folds:
            hold = np.fromiter(f, dtype=np.int64, count=len(f))
            tr_idx = np.setdiff1d(np.arange(len(y_tr)), hold)
            head = fit(xt[tr_idx], y_tr[tr_idx], kind, lr, wd, eps)
            p = torch.softmax(torch.tensor(predict(head, xt[hold])), 1).numpy()
            accs.append(float((p.argmax(1) == (y_tr[hold] - 1)).mean()))
        score = float(np.mean(accs))
        print(f"[{log_prefix}cv] {kind} lr={lr} wd={wd} ep={eps} acc={score:.4f} "
              f"folds={[round(a, 3) for a in accs]}", flush=True)
        if score > best:
            best, best_cfg = score, (kind, lr, wd, eps)
    kind, lr, wd, eps = best_cfg
    print(f"[{log_prefix}cv] best {best_cfg} acc={best:.4f}", flush=True)

    # 同被试对照: 同一组超参、同一批特征, 只把折改成随机切 (被试不互斥)
    cv_random, rnd_accs = cv_score(xt, y_tr, best_cfg,
                                   random_folds(len(y_tr), folds_n))
    print(f"[{log_prefix}cv] 同被试(随机折) acc={cv_random:.4f} "
          f"folds={[round(a, 3) for a in rnd_accs]} "
          f"(跨被试 {best:.4f}, 差 {100*(cv_random-best):+.1f}pp)", flush=True)

    # 用全部 train 特征训练最终头, 多种子概率平均
    prob_sum = np.zeros((len(y_te), 50), dtype=np.float64)
    for s in range(seeds):
        head_s = fit(xt, y_tr, kind, lr, wd, eps, seed=s)
        prob_sum += torch.softmax(torch.tensor(predict(head_s, xv)), 1).numpy()
        print(f"[{log_prefix}final] seed {s} done", flush=True)
    probs = prob_sum / seeds
    gid, n_group = merge_group_ids()
    res, probs = metrics(np.log(probs + 1e-12), y_te, gid, n_group)
    res.update({"best_cfg": {"head": kind, "lr": lr, "wd": wd, "epochs": eps},
                "cv_acc_grouped": best, "cv_acc_random_folds": cv_random,
                "seeds": seeds,
                "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                "feat_dim": int(x_tr.shape[1])})
    return res, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e4_lora")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="test 题数, 0=全量(约 6000)")
    ap.add_argument("--train_per_pair", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epoch_grid", default="40,100",
                    help="CV 里一起搜的 epochs 候选, 逗号分隔; 留空则只用 --epochs")
    ap.add_argument("--seeds", type=int, default=5, help="最终头训练用几个随机种子做平均")
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    tag = args.tag or (os.path.basename(args.ckpt.rstrip("/")) + "_aug")
    out_dir = os.path.join(ROOT, "outputs", "probe", tag)
    os.makedirs(out_dir, exist_ok=True)
    smap = swap_id_map()
    x_tr, y_tr, s_tr, f_tr_flag, x_te, y_te, s_te, _ = \
        load_or_extract(args, tag, out_dir, smap)
    print(f"[data] train {x_tr.shape} test {x_te.shape} "
          f"subjects tr={len(np.unique(s_tr))} te={len(np.unique(s_te))} "
          f"flip_share={f_tr_flag.mean():.3f}", flush=True)
    if f_tr_flag.mean() < 0.35:
        raise SystemExit("!! train 特征里镜像样本不足 35% —— 特征缓存或数据有问题, 中止。")

    res, probs = evaluate_features(x_tr, y_tr, s_tr, x_te, y_te, args.epochs,
                                   args.epoch_grid, args.folds, args.seeds)
    res.update({"tag": tag, "ckpt": args.ckpt,
                "flip_share_train": float(f_tr_flag.mean())})
    np.save(os.path.join(out_dir, "test_logits.npy"), np.log(probs + 1e-12))
    np.save(os.path.join(out_dir, "test_labels.npy"), y_te)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("AUG_RESULT " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
