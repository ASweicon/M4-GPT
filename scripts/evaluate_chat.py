# -*- coding: utf-8 -*-
"""E7 多轮对话评估: 逐轮生成 + 分轮型自动阅卷 (差异于单轮 exact match).

考什么:
  action          50 类动作名 (同时报 39 类镜像合并 / 3 粗类 / 平衡准确率)
  evidence        解释是否引用了图中可观测证据 (水平位置/深度/运动量/姿态/垂直尺寸)
                  同时对 action 轮做左右一致性检查 (Q1 说 right, Q2 不能说 left)
  quantity:xxx    数值题 (movement / height / dist), 容差 max(0.15, 20%)
  counterfactual  镜像反事实: 必须给出 swap_lr 后的动作名 (左右互换)

用法:
  python scripts/evaluate_chat.py --ckpt outputs/e7_lora --limit 300
  python scripts/evaluate_chat.py --ckpt outputs/e7_lora --limit 20 --demo 2
  # 冒烟: --limit 5

产物: outputs/eval/<ckpt名>_chat/{results.json, predictions.jsonl, demo.txt}
"""
import argparse, json, os, re, sys, time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m4config import ACTION_NAMES, coarse_group, swap_lr
from make_caption import POS_X, POS_Y, PATH_STILL, PATH_LARGE, Z_SIT, zone

DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "data"))
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
MAX_NEW = {"action": 24, "evidence": 128, "counterfactual": 96}
MAX_NEW_Q = 40

NAME2ID = {v: k for k, v in ACTION_NAMES.items()}
LOWER2ID = None          # norm(name) -> id, main() 里初始化 (依赖 norm)


def build_gid():
    """左右镜像合并分组 (39 类), 确定性构建, 不用 hash()."""
    keys, gid = {}, {}
    for i in sorted(ACTION_NAMES):
        pair = tuple(sorted([ACTION_NAMES[i], swap_lr(ACTION_NAMES[i])]))
        if pair not in keys:
            keys[pair] = len(keys)
        gid[i] = keys[pair]
    return gid


GID = build_gid()


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".")


def num_in(text):
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    return float(m.group()) if m else None


def side_set(text):
    return set(re.findall(r"\b(left|right)\b", norm(text)))


def match_action(text):
    """在生成文本里找最长命中的类名, 返回 id; 未命中返回 None."""
    p = norm(text)
    hits = [(len(n), i) for n, i in LOWER2ID.items() if n in p]
    return max(hits)[1] if hits else None


LEVEL_WORDS = {
    "still": ("still", "stationary", "barely", "hardly", "in place", "in-place"),
    "slight": ("slight", "small", "modest", "little"),
    "notable": ("notable", "large", "significant", "substantial", "considerable"),
}


def score_evidence(pred, meta, action_pred):
    f = meta["features"]
    p = norm(pred)
    lx = zone(f["centroid_x"], *POS_X, ("left", "center", "right"))
    ly = zone(f["centroid_y"], *POS_Y, ("close", "middle", "far"))
    pl = f["path_len"]
    lvl = ("still" if pl < PATH_STILL else
           "notable" if pl > PATH_LARGE else "slight")
    checks = {
        "horizontal": lx in p,
        "depth": ly in p,
        "movement": any(w in p for w in LEVEL_WORDS[lvl]),
        "posture": ("seated" in p or "sitting" in p or "low" in p)
                   if f["centroid_z"] < Z_SIT else ("standing" in p or "upright" in p),
        "extent": (lambda v: v is not None and abs(v - f["extent_z"]) <= max(0.5, 0.3 * f["extent_z"]))(num_in(pred)),
    }
    score = sum(checks.values()) / len(checks)
    # 左右一致性: action 轮说 right, evidence 轮不应只说 left
    sa, se = side_set(action_pred), side_set(pred)
    consistent = not (sa and se and not (sa & se))
    return score, {"checks": checks, "consistent": consistent}


def score_quantity(pred, ref):
    r, p = num_in(ref), num_in(pred)
    if r is None or p is None:
        return 0.0, {}
    tol = max(0.15, 0.2 * r)
    return float(abs(p - r) <= tol), {"ref_num": r, "pred_num": p, "tol": round(tol, 3)}


def score_counterfactual(pred, action):
    sw = swap_lr(action)
    p = norm(pred)
    if norm(sw) in p:
        return 1.0, {"expect": sw}
    if sw == action and ("same" in p or "unchanged" in p) and (
            "flip" in p or "mirror" in p or " x " in p or "x-" in p):
        return 1.0, {"expect": "same-action + x flip"}
    return 0.0, {"expect": sw}


def load_rows(split, limit):
    fp = os.path.join(DATA_ROOT, f"chat_{split}.jsonl")
    if not os.path.isfile(fp):
        sys.exit(f"!! 缺 {fp} —— 先运行 python scripts/synthesize_chat.py")
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    rows.sort(key=lambda r: r["image"])
    if limit and limit < len(rows):
        idx = np.linspace(0, len(rows) - 1, limit).round().astype(int)
        rows = [rows[i] for i in sorted(set(idx))]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e7_lora")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300,
                    help="抽样对话数 (每条约 2~4 轮生成), 默认 300")
    ap.add_argument("--demo", type=int, default=0, help="打印前 N 条完整对话")
    args = ap.parse_args()

    global LOWER2ID
    LOWER2ID = {norm(n): i for i, n in ACTION_NAMES.items()}

    ckpt = os.path.abspath(args.ckpt)
    rows = load_rows(args.split, args.limit)
    out_dir = os.path.join(HERE, "..", "outputs", "eval",
                           os.path.basename(ckpt.rstrip("/")) + "_chat")
    os.makedirs(out_dir, exist_ok=True)
    print(f"设备 {DEVICE} | ckpt {ckpt} | {len(rows)} 条对话", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ckpt).to(DEVICE).eval()

    scores = defaultdict(list)
    action_preds, action_refs = [], []
    all_recs = []
    inconsistent = 0
    demo_lines = []
    t0 = time.time()

    for i, row in enumerate(rows):
        img = Image.open(os.path.join(DATA_ROOT, row["image"])).convert("RGB")
        msgs, hist = row["messages"], []
        preds = []
        for t in range(0, len(msgs), 2):
            user = msgs[t]["content"]
            kind = row["meta"]["kinds"][t // 2]
            text = user.replace("<image>\n", "")
            u = ({"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text}]}
                 if "<image>" in user else {"role": "user", "content": text})
            batch = proc.apply_chat_template(
                hist + [u], add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(DEVICE)
            mn = 24 if kind == "action" else (
                128 if kind == "evidence" else
                96 if kind == "counterfactual" else MAX_NEW_Q)
            with torch.no_grad():
                out = model.generate(**batch, max_new_tokens=mn, do_sample=False)
            gen = proc.tokenizer.decode(
                out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            preds.append(gen)
            hist += [u, {"role": "assistant",
                         "content": [{"type": "text", "text": gen}]}]

        # ---- 分轮阅卷 ----
        rec = {"image": row["image"], "turns": []}
        for t, kind in enumerate(row["meta"]["kinds"]):
            ref, pred = row["meta"]["refs"][t], preds[t]
            if kind == "action":
                s = float(norm(ref) in norm(pred))
                action_preds.append(pred); action_refs.append(ref)
                extra = {}
            elif kind == "evidence":
                s, extra = score_evidence(pred, row["meta"], preds[0])
                inconsistent += int(not extra["consistent"])
            elif kind.startswith("quantity:"):
                s, extra = score_quantity(pred, ref)
            else:
                s, extra = score_counterfactual(pred, row["meta"]["action"])
            scores[kind.split(":")[0]].append(s)
            rec["turns"].append({"kind": kind, "q": msgs[2 * t]["content"],
                                 "ref": ref, "pred": pred, "score": s, **extra})

        if len(demo_lines) < args.demo:
            demo_lines.append(f"\n== {row['image']}  ({row['meta']['action']})")
            for d in rec["turns"]:
                demo_lines.append(f"Q: {d['q'].replace('<image>', '').strip()}")
                demo_lines.append(f"A: {d['pred']}")
                demo_lines.append(f"   ref: {str(d['ref'])[:120]}  score={d['score']:.2f}")
        all_recs.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.1f}s/dialogue",
                  flush=True)

    # ---- 汇总 ----
    res = {k: {"acc": round(float(np.mean(v)), 4), "n": len(v)}
           for k, v in scores.items()}
    # 动作轮: 50/39/粗类 + 平衡准确率 + 多数类基线
    pid = [match_action(p) for p in action_preds]
    rid = [NAME2ID[r] for r in action_refs]
    recs_hit = [p is not None and p == r for p, r in zip(pid, rid)]
    g39_hit = [p is not None and GID[p] == GID[r] for p, r in zip(pid, rid)]
    coarse_hit = [p is not None and coarse_group(p) == coarse_group(r)
                  for p, r in zip(pid, rid)]
    per_class = defaultdict(list)
    for r, h in zip(action_refs, recs_hit):
        per_class[r].append(h)
    bal = float(np.mean([np.mean(v) for v in per_class.values()]))
    maj = Counter(action_refs).most_common(1)[0][1] / len(action_refs)

    res["action_fine_50"] = round(float(np.mean(recs_hit)), 4)
    res["action_group39"] = round(float(np.mean(g39_hit)), 4)
    res["action_coarse3"] = round(float(np.mean(coarse_hit)), 4)
    res["action_balanced_acc"] = round(bal, 4)
    res["action_majority_baseline"] = round(maj, 4)
    res["evidence_inconsistent_sides"] = inconsistent

    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if demo_lines:
        with open(os.path.join(out_dir, "demo.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(demo_lines))
        print("\n".join(demo_lines))
    print("\n== results")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
