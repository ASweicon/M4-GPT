# -*- coding: utf-8 -*-
"""Build manifest.json and splits.json for the I/Q dataset.

Run from the repo root:
    python3 radar_encoder/index.py --train 1,2 --val 17 --test 19
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def scan(root):
    iq_files = sorted(glob.glob(str(root / "data/iq/p*_iq_int16.lmdb")))
    subjects = {}
    for path in iq_files:
        m = re.search(r"p(\d+)_iq_int16\.lmdb$", path)
        if not m:
            continue
        sub = int(m.group(1))
        man = path + ".manifest.json"
        frames, actions = 0, []
        if os.path.exists(man):
            d = json.load(open(man))
            frames = d.get("frames_written", 0)
            actions = sorted(int(a) for a in d.get("actions", {}))
        params = root / f"data/params/p{sub}/parameters"
        params = sorted(glob.glob(str(params / "*.json")))
        subjects[str(sub)] = {
            "subject": sub,
            "lmdb": os.path.relpath(path, root),
            "frames": frames,
            "actions_iq": actions,
            "params_dir": os.path.relpath(root / f"data/params/p{sub}", root),
            "params_count": len(params),
            "params_actions": sorted(int(Path(p).stem) for p in params),
        }
    return subjects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--train", default="1,2")
    ap.add_argument("--val", default="17")
    ap.add_argument("--test", default="19")
    args = ap.parse_args()

    root = Path(args.root)
    meta = root / "data/meta"
    meta.mkdir(parents=True, exist_ok=True)

    subjects = scan(root)
    manifest = {
        "root": str(root),
        "representation": "iq_int16",
        "value_layout": "header <IIIIf, int16 (2,320,151), value = int16 * scale",
        "subjects": subjects,
        "total_frames": sum(s["frames"] for s in subjects.values()),
    }
    splits = {
        "train": [int(x) for x in args.train.split(",") if x],
        "val": [int(x) for x in args.val.split(",") if x],
        "test": [int(x) for x in args.test.split(",") if x],
        "official_s2": {"train": list(range(1, 21)), "val": [17], "test": [19, 5, 8, 10]},
    }

    (meta / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (meta / "splits.json").write_text(json.dumps(splits, indent=2, ensure_ascii=False))
    print(json.dumps({k: v["frames"] for k, v in subjects.items()}, indent=2))
    print("total frames:", manifest["total_frames"])
    print("splits:", splits["train"], splits["val"], splits["test"])


if __name__ == "__main__":
    main()
