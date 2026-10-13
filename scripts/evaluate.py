# -*- coding: utf-8 -*-
"""Evaluate HAR: vision tower runs once per clip, 50 action names scored at LLM."""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "data"))
CKPT = os.path.abspath(os.path.join(HERE, "..", "outputs", "lora"))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs", "eval"))
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
QUESTION = "What action is the person performing in this radar point cloud?"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

sys.path.insert(0, HERE)
from m4config import ACTION_NAMES, S2_TEST, S2_VAL, coarse_group


def gather_items(split, per_pair):
    img_dir = os.path.join(DATA_ROOT, "images", split)
    # train: 不过滤(诊断记忆用); test/val: 只保留对应被试
    keep = {"test": set(S2_TEST), "val": set(S2_VAL)}.get(split)
    by_pair = defaultdict(list)
    for fn in os.listdir(img_dir):
        parts = fn[:-4].split("_")
        s = int(parts[0][1:]); a = int(parts[1][1:])
        if keep is None or s in keep:
            by_pair[(s, a)].append(fn)
    items = []
    for (s, a), lst in sorted(by_pair.items()):
        lst.sort()
        idx = np.linspace(0, len(lst) - 1, min(per_pair, len(lst))).round().astype(int)
        for i in sorted(set(idx)):
            items.append((s, a, lst[i]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_pair", type=int, default=1)
    ap.add_argument("--cbatch", type=int, default=25)
    ap.add_argument("--split", default="test")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task", default="fine", choices=["fine", "coarse"],
                    help="fine: 50类动作名似然打分; coarse: 3类粗粒度打分")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.base_only:
        model = base.to(DEVICE).eval()
    else:
        model = PeftModel.from_pretrained(base, CKPT).to(DEVICE).eval()
    im_end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    image_pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    raw = model.get_base_model() if hasattr(model, "get_base_model") else model
    qmodel = raw.model
    visual, embed_layer = qmodel.visual, qmodel.language_model.embed_tokens

    items = gather_items(args.split, args.per_pair)
    if args.limit:
        items = items[:args.limit]
    if args.task == "coarse":
        names = ["in-place", "sit-in-place", "non-in-place"]
    else:
        names = [ACTION_NAMES[i] for i in range(1, 51)]
    ans_tokens = [proc.tokenizer(n, add_special_tokens=False).input_ids
                  for n in names]
    NC = len(names)

    top1 = top5 = 0
    confusion = np.zeros((NC, NC), dtype=int)
    per_act = defaultdict(lambda: [0, 0])

    for n_it, (s, gt_act, fn) in enumerate(items):
        img = Image.open(os.path.join(DATA_ROOT, "images", args.split, fn)).convert("RGB")
        pd = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": QUESTION}]}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        Lp = pd["input_ids"].shape[1]
        pid = pd["input_ids"][0].tolist()
        pmm = pd["mm_token_type_ids"][0].tolist()
        grid0 = pd["image_grid_thw"].to(DEVICE)

        with torch.no_grad():
            img_emb = visual(pd["pixel_values"].to(DEVICE),
                             grid0).pooler_output  # [196,1536]

        scores = np.zeros(NC)
        for c0 in range(0, NC, args.cbatch):
            ids, mm, attn, spans = [], [], [], []
            for c in range(c0, min(c0 + args.cbatch, NC)):
                a_ids = ans_tokens[c]
                ids.append(torch.tensor(pid + a_ids))
                mm.append(torch.tensor(pmm + [0] * len(a_ids)))
                attn.append(torch.ones(len(pid) + len(a_ids)))
                spans.append((Lp, Lp + len(a_ids)))
            B = len(ids); maxlen = max(x.shape[0] for x in ids)
            ids = torch.stack([torch.cat([x, torch.full((maxlen-x.shape[0],), im_end)]) for x in ids])
            mm = torch.stack([torch.cat([x, torch.zeros(maxlen-x.shape[0], dtype=x.dtype)]) for x in mm])
            attn = torch.stack([torch.cat([x, torch.zeros(maxlen-x.shape[0])]) for x in attn]).long()
            ids_t, mm_t, attn_t = ids.to(DEVICE), mm.to(DEVICE), attn.to(DEVICE)
            grid_rep = grid0.repeat(B, 1)
            with torch.no_grad():
                inputs_embeds = embed_layer(ids_t)
                mask = ids_t == image_pad
                for b in range(B):
                    pos = mask[b].nonzero(as_tuple=True)[0]
                    inputs_embeds[b, pos] = img_emb.to(inputs_embeds.dtype)
                logits = model(inputs_embeds=inputs_embeds, attention_mask=attn_t,
                               mm_token_type_ids=mm_t,
                               image_grid_thw=grid_rep).logits
            logp = torch.log_softmax(logits[:, :-1].float(), -1)
            tgt = ids_t[:, 1:]
            lp_tok = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for k, (a0, a1) in enumerate(spans):
                sel = lp_tok[k, a0-1:a1-1]
                scores[c0 + k] = (sel.sum() / sel.numel()).item()

        order = np.argsort(-scores)
        pred = int(order[0])
        if args.task == "coarse":
            key = coarse_group(gt_act)
            gt_idx = names.index(key)
        else:
            key = ACTION_NAMES[gt_act]
            gt_idx = gt_act - 1
        confusion[gt_idx, pred] += 1
        per_act[key][0] += int(pred == gt_idx)
        per_act[key][1] += 1
        top1 += int(pred == gt_idx)
        top5 += int(gt_idx in order[:5])
        print(f"{n_it+1}/{len(items)} running top1 {top1/(n_it+1):.3f} "
              f"top5 {top5/(n_it+1):.3f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    res = {"n": len(items), "task": args.task, "split": args.split,
           "top1": top1/len(items), "top5": top5/len(items),
           "per_action": {k: per_act[k][0]/per_act[k][1]
                          for k in sorted(per_act)}}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savetxt(os.path.join(OUT, "confusion.csv"), confusion, fmt="%d",
               delimiter=",")
    print("TOP1", f"{top1/len(items):.4f}", "TOP5", f"{top5/len(items):.4f}")


if __name__ == "__main__":
    main()
