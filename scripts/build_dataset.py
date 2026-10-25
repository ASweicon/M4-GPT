# -*- coding: utf-8 -*-
"""Build rendered-image + instruction dataset for HAR (S2 cross-subject)."""
import argparse, glob, json, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m4config import (RPC, ACTION_NAMES, coarse_group, S2_TEST, S2_VAL,
                      swap_lr)
from render_samples import open_db, list_frames, render_clip

DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data"))

FINE_Q = [
    "What action is the person performing in this radar point cloud?",
    "Based on this mmWave point cloud, what is the person doing?",
    "Identify the action shown in this radar point cloud.",
    "根据这个毫米波雷达点云，判断人物在做什么动作？",
]
COARSE_Q = [
    "Is the person staying in place, sitting on a chair, or moving around?",
    "Classify this motion as in-place, sit-in-place, or non-in-place.",
    "判断该动作属于原地、坐姿还是移动？",
]
DESC_Q = [
    "Describe what the person is doing in this scene.",
    "Describe the person's motion based on the radar data.",
    "描述图中人物正在做的动作。",
]

MODES = {
    "smoke": dict(train=2, val=4, test=4),
    "mini": dict(train=6, val=12, test=12),
    "full": dict(train=16, val=30, test=30),
}


def center_frames(frames, k):
    valid = [f for f in frames if f >= 3]
    if len(valid) <= k:
        return valid
    idx = np.linspace(0, len(valid) - 1, k).round().astype(int)
    return [valid[i] for i in sorted(set(idx))]


def make_records(s, a, fc, img_rel, flip=False):
    name = ACTION_NAMES[a]; cg = coarse_group(a)
    if flip:
        name = swap_lr(name)      # 已验证互换后仍在 50 类内
    rng = random.Random(hash((s, a, fc, flip)) & 0xffffffff)
    return [
        {"image": img_rel, "messages": [
            {"role": "user", "content": "<image>\n" + rng.choice(FINE_Q)},
            {"role": "assistant", "content": name}]},
        {"image": img_rel, "messages": [
            {"role": "user", "content": "<image>\n" + rng.choice(COARSE_Q)},
            {"role": "assistant", "content": cg}]},
        {"image": img_rel, "messages": [
            {"role": "user", "content": "<image>\n" + rng.choice(DESC_Q)},
            {"role": "assistant", "content": f"The person is {name}."}]},
    ]


def merge_chunks(sp):
    """合并 {sp}_chunkN.jsonl -> {sp}.jsonl, 做全局重排后删除分块文件."""
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, f"{sp}_chunk*.jsonl")))
    if not paths:
        print(f"no {sp}_chunk*.jsonl to merge", flush=True)
        return
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f]
        os.remove(p)
    random.Random(0).shuffle(rows)
    jp = os.path.join(DATA_ROOT, f"{sp}.jsonl")
    with open(jp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"== merged {len(paths)} chunks -> {jp} ({len(rows)} records)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=list(MODES))
    ap.add_argument("--split", default="train,val,test",
                    help="逗号分隔: train,val,test")
    ap.add_argument("--chunk", type=int, default=0,
                    help=">0: 只渲染 --split 的第 N 个 subject, "
                         "写 {split}_chunk{N}.jsonl (进程隔离防内存泄漏, 支持并行)")
    ap.add_argument("--merge", default="",
                    help="非空: 合并 {值}_chunk*.jsonl 为 {值}.jsonl 后退出")
    args = ap.parse_args()
    K = MODES[args.mode]

    if args.merge:
        merge_chunks(args.merge)
        return

    train_sub = sorted(set(range(1, 21)) - set(S2_TEST) - set(S2_VAL))
    by_split = {"train": (train_sub, K["train"]),
                "val": (S2_VAL, K["val"]),
                "test": (S2_TEST, K["test"])}
    names = [x.strip() for x in args.split.split(",") if x.strip()]
    for sp in names:
        if sp not in by_split:
            sys.exit(f"unknown split: {sp}")
    splits = [(sp, sorted(by_split[sp][0]), by_split[sp][1]) for sp in names]

    suffix = ""
    if args.chunk:
        sp, subs, k = splits[0]
        if not 1 <= args.chunk <= len(subs):
            sys.exit(f"--chunk must be 1..{len(subs)} for split {sp}")
        splits = [(sp, [subs[args.chunk - 1]], k)]
        suffix = f"_chunk{args.chunk}"

    for sp, _, _ in splits:
        os.makedirs(os.path.join(DATA_ROOT, "images", sp), exist_ok=True)

    env = open_db(RPC, 8)
    counts = {}
    with env.begin() as txn:
        for sp, subs, k in splits:
            records = []
            for s in subs:
                for a in range(1, 51):
                    frames = list_frames(txn, s, a)
                    for fc in center_frames(frames, k):
                        fn = f"s{s}_a{a:02d}_f{fc}.png"
                        rel = f"images/{sp}/{fn}"
                        out = os.path.join(DATA_ROOT, "images", sp, fn)
                        render_clip(txn, s, a, fc, out)
                        records.extend(make_records(s, a, fc, rel))
                        if sp == "train":   # 左右翻转增强, 仅训练集
                            fn_f = f"s{s}_a{a:02d}_f{fc}flip.png"
                            rel_f = f"images/{sp}/{fn_f}"
                            render_clip(txn, s, a, fc,
                                        os.path.join(DATA_ROOT, rel_f),
                                        flip=True)
                            records.extend(make_records(s, a, fc, rel_f,
                                                        flip=True))
                print(f"[{sp}] subject {s} done, records={len(records)}",
                      flush=True)
            random.Random(0).shuffle(records)
            jp = os.path.join(DATA_ROOT, f"{sp}{suffix}.jsonl")
            with open(jp, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[sp] = len(records)
            print(f"== {sp}: {len(records)} records -> {jp}", flush=True)
    env.close()
    print("DONE", counts, flush=True)


if __name__ == "__main__":
    main()
