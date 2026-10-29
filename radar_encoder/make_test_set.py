# -*- coding: utf-8 -*-
"""Build a small deterministic test JSONL: first N clips per action.

Example (from repo root):
    python3 radar_encoder/make_test_set.py --subject 17 --per-action 30
"""

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--subject", type=int, default=17)
    ap.add_argument("--per-action", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(glob.glob(str(root / f"data/text/p{args.subject}_*.jsonl")))
    out = Path(args.out) if args.out else root / f"data/text/test_p{args.subject}_{args.per_action}per.jsonl"
    n = 0
    with open(out, "w") as fo:
        for f in files:
            with open(f) as fi:
                for i, line in enumerate(fi):
                    if i >= args.per_action:
                        break
                    fo.write(line)
                    n += 1
    print("wrote", out, n, "clips")


if __name__ == "__main__":
    main()
