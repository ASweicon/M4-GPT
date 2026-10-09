# -*- coding: utf-8 -*-
"""Build rendered-image + instruction dataset for HAR (S2 cross-subject)."""
import argparse, json, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m4config import (RPC, ACTION_NAMES, coarse_group, S2_TEST, S2_VAL)
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


def make_records(s, a, fc, img_rel):
    name = ACTION_NAMES[a]; cg = coarse_group(a)
    rng = random.Random(hash((s, a, fc)) & 0xffffffff)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=list(MODES))
    args = ap.parse_args()
    K = MODES[args.mode]

    train_sub = sorted(set(range(1, 21)) - set(S2_TEST) - set(S2_VAL))
    splits = [("train", train_sub, K["train"]),
              ("val", S2_VAL, K["val"]),
              ("test", S2_TEST, K["test"])]

    for sp, subs, k in splits:
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
                print(f"[{sp}] subject {s} done, records={len(records)}",
                      flush=True)
            random.Random(0).shuffle(records)
            jp = os.path.join(DATA_ROOT, f"{sp}.jsonl")
            with open(jp, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[sp] = len(records)
            print(f"== {sp}: {len(records)} records -> {jp}", flush=True)
    env.close()
    print("DONE", counts, flush=True)


if __name__ == "__main__":
    main()
