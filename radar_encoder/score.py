# -*- coding: utf-8 -*-
"""Score an evaluate.py output JSONL with normalized exact match, by question."""

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9.+-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/iq_eval.jsonl")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(Path(args.root) / args.pred) if x.strip()]
    by_q = collections.defaultdict(lambda: [0, 0])
    hit = 0
    for r in rows:
        ok = norm(r["pred"]) == norm(r["ref"])
        hit += ok
        q = r.get("question", "?")
        by_q[q][0] += int(ok)
        by_q[q][1] += 1
    print(f"overall exact: {hit}/{len(rows)} = {hit/max(len(rows),1):.4f}")
    for q, (h, n) in sorted(by_q.items(), key=lambda x: -x[1][1]):
        print(f"  {q[:60]:60s} {h}/{n} = {h/n:.4f}")


if __name__ == "__main__":
    main()
