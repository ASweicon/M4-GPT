# -*- coding: utf-8 -*-
"""A 方案的收尾: 把 E4/E5 两个探针合起来, 两种集成方式同时给出.

  1) 概率平均 (logit ensemble): 两个独立头的 softmax 取平均再判 —— 便宜, 稳;
  2) 特征拼接 (feature concat): [E4 特征 || E5 特征] 训一个头 —— 通常比概率平均
     再高 1~3pp, 因为两个权重看到的是同一张图的不同抽象层, 拼接后互补性被保留。

用法:
  python scripts/probe_ensemble.py --tags e4_aug e5_aug
  python scripts/probe_ensemble.py --tags e4_aug e5_aug --no_concat
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from probe_aug import evaluate_features, merge_group_ids, metrics


def load_tag(tag, per_pair, limit):
    d = os.path.join(ROOT, "outputs", "probe", tag)
    f_tr = os.path.join(d, f"feat_train_pp{per_pair}.npz")
    f_te = os.path.join(d, f"feat_test_lim{limit}.npz")
    for p in (f_tr, f_te):
        if not os.path.exists(p):
            raise SystemExit(f"!! 缺特征缓存 {p} —— 先跑 scripts/probe_aug.py --tag {tag}")
    tr, te = np.load(f_tr), np.load(f_te)
    return tr["x"], tr["y"], tr["s"], te["x"], te["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--train_per_pair", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no_concat", action="store_true")
    ap.add_argument("--out", default="outputs/probe/ensemble_result.json")
    args = ap.parse_args()

    if args.weights and len(args.weights) != len(args.tags):
        raise SystemExit("--weights 个数必须与 --tags 一致")
    w = np.asarray(args.weights if args.weights else [1.0] * len(args.tags))
    w = w / w.sum()

    gid, n_group = merge_group_ids()
    probs_list, labels, singles, feats = [], None, {}, []
    for tag in args.tags:
        x_tr, y_tr, s_tr, x_te, y_te = load_tag(tag, args.train_per_pair, args.limit)
        d = os.path.join(ROOT, "outputs", "probe", tag)
        p = torch.softmax(torch.tensor(np.load(os.path.join(d, "test_logits.npy"))), 1).numpy()
        if labels is None:
            labels, s_ref, y_tr_ref = y_te, s_tr, y_tr
        else:
            assert np.array_equal(labels, y_te), f"test 标签顺序不一致: {tag}"
            assert np.array_equal(y_tr_ref, y_tr) and np.array_equal(s_ref, s_tr), \
                f"train 特征顺序不一致: {tag} (train_per_pair 设置需相同)"
        probs_list.append(p)
        m, _ = metrics(np.log(p + 1e-12), labels, gid, n_group)
        singles[tag] = {k: v for k, v in m.items() if k != "merged_n"}
        print(f"[single] {tag} fine_top1={m['fine_top1']:.4f} "
              f"top5={m['fine_top5']:.4f} merged_top1={m['merged_top1']:.4f}", flush=True)
        feats.append((x_tr, x_te))

    # 1) 概率平均
    probs = np.zeros_like(probs_list[0])
    for wi, p in zip(w, probs_list):
        probs += wi * p
    res_prob, _ = metrics(np.log(probs + 1e-12), labels, gid, n_group)
    res_prob.update({"tags": args.tags, "weights": [float(x) for x in w],
                     "singles": singles})
    print("ENSEMBLE_PROB " + json.dumps(
        {k: res_prob[k] for k in ("fine_top1", "fine_top5",
                                  "merged_top1", "merged_top5")}), flush=True)

    # 2) 特征拼接 (只在 tag 数 >= 2 时才有意义)
    if len(feats) >= 2 and not args.no_concat:
        x_tr = np.concatenate([f[0] for f in feats], 1)
        x_te = np.concatenate([f[1] for f in feats], 1)
        print(f"[concat] dim={x_tr.shape[1]} train={x_tr.shape[0]}", flush=True)
        res_cat, probs_cat = evaluate_features(
            x_tr, y_tr_ref, s_ref, x_te, labels, 40, "40,100", 4, args.seeds,
            log_prefix="concat_")
        res_prob["concat"] = {k: v for k, v in res_cat.items()}
        np.save(os.path.join(ROOT, "outputs", "probe", "concat_logits.npy"),
                np.log(probs_cat + 1e-12))
        print("ENSEMBLE_CONCAT " + json.dumps(
            {k: res_cat[k] for k in ("fine_top1", "fine_top5",
                                     "merged_top1", "merged_top5", "best_cfg")}),
            flush=True)

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res_prob, f, indent=2, ensure_ascii=False)
    print("ENSEMBLE_RESULT " + json.dumps(res_prob, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
