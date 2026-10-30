# -*- coding: utf-8 -*-
"""NumPy linear probe: train on p1+p2, test on p17, compare I/Q vs RPC features.

This is a sanity check, not the LLM. It answers "is there learnable signal in
the I/Q front-end under a cross-subject split?" without torch.

Run from the repo root:
    python3 radar_encoder/probe.py --per-action 30
"""

import argparse
import collections
import importlib.util
import json
import random
import struct
from pathlib import Path

import lmdb
import numpy as np

try:
    from .dataset import IQClipDataset
    from .dsp import iq_to_rda, load_pairs
except ImportError:
    from dataset import IQClipDataset
    from dsp import iq_to_rda, load_pairs

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
RPC = ROOT / "radar_pc.lmdb"
RPC_HDR = struct.Struct("<III")


def load_m4config():
    path = CODE_ROOT / "scripts/m4config.py"
    spec = importlib.util.spec_from_file_location("m4config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coarse_id(action):
    if 1 <= action <= 30:
        return 0
    if 31 <= action <= 35:
        return 1
    return 2


def read_rpc(env, key):
    with env.begin() as txn:
        v = txn.get(key.encode())
    if v is None:
        return None
    _, n, cols = RPC_HDR.unpack(v[:12])
    return np.frombuffer(v[12:], dtype="<f4").reshape(n, cols)


def iq_feat(iq, pairs):
    x = iq_to_rda(iq, pairs)          # [T,2,256,20,16]
    lm = x[:, 0]
    rp = lm.mean(axis=(0, 2, 3))      # 256
    rp = rp.reshape(64, 4).mean(1)
    dp = lm.mean(axis=(0, 1, 3))      # 20
    ap = lm.mean(axis=(0, 1, 2))      # 16
    e = lm.mean(axis=(1, 2, 3))       # T
    ph = x[:, 1]
    stats = np.array([
        e.std(), np.abs(np.diff(e)).mean(),
        ph.std(), np.abs(np.diff(ph, axis=0)).mean(),
    ], dtype=np.float32)
    return np.concatenate([rp, dp, ap, stats]).astype(np.float32)


def rpc_feat(frames, action, subject, env, frame_to_idx):
    """I/Q 库用被试内全局帧号，点云库用动作内序号，这里按排序序号映射。"""
    vals = []
    for f in frames:
        idx = frame_to_idx.get(f)
        if idx is None:
            continue
        p = read_rpc(env, f"[{subject}, {action}, {idx}]")
        if p is None or len(p) == 0:
            continue
        xyz = p[:, :3]
        r = np.linalg.norm(xyz, axis=1)
        vals.append(np.array([
            len(p), xyz[:, 0].mean(), xyz[:, 1].mean(), xyz[:, 2].mean(),
            xyz[:, 0].max() - xyz[:, 0].min(),
            xyz[:, 1].max() - xyz[:, 1].min(),
            xyz[:, 2].max() - xyz[:, 2].min(),
            np.median(r), np.percentile(r, 90), p[:, 3].mean(),
        ], dtype=np.float32))
    if not vals:
        return None
    V = np.stack(vals)
    stats = np.array([V[:, 0].mean(), V[:, 0].std(),
                      V[:, 2].std(), V[:, 3].std()], dtype=np.float32)
    disp = np.linalg.norm(np.diff(V[:, 1:4], axis=0), axis=1) if len(V) > 1 else np.array([0.0])
    return np.concatenate([V.mean(0), V.std(0), stats, [float(disp.mean())]]).astype(np.float32)


def collect(subjects, per_action, t_frames, stride, pairs):
    rng = random.Random(0)
    iq_env = lmdb.open(str(RPC), subdir=False, readonly=True, lock=False, readahead=False)
    X_iq, X_rpc, Y, Yc = [], [], [], []
    for sub in subjects:
        ds = IQClipDataset(ROOT / f"data/iq/p{sub}_iq_int16.lmdb", sub, pairs,
                           t_frames=t_frames, stride=stride)
        by = collections.defaultdict(list)
        for a, frames in ds.clips:
            by[a].append(frames)
        order = {a: {f: i for i, f in enumerate(sorted(fs))}
                 for a, fs in ds.frames_by_action.items()}
        for a, clips in sorted(by.items()):
            rng.shuffle(clips)
            for frames in clips[:per_action]:
                iq = ds._read(a, frames)
                if iq is None:
                    continue
                fi = iq_feat(iq, pairs)
                fr = rpc_feat(frames, a, sub, iq_env, order[a])
                if fr is None:
                    continue
                X_iq.append(fi)
                X_rpc.append(fr)
                Y.append(a)
                Yc.append(coarse_id(a))
        print(f"p{sub}: {sum(1 for y in Y if True)} total samples so far", flush=True)
        ds.close()
    iq_env.close()
    return (np.stack(X_iq), np.stack(X_rpc), np.array(Y), np.array(Yc))


def ridge_fit(X, y, n_class, lam=1.0):
    Xb = np.c_[X, np.ones(len(X))]
    Y = np.eye(n_class)[y]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ Y)


def metrics(scores, y):
    top1 = float((scores.argmax(1) == y).mean())
    top5 = float(np.mean([y[i] in np.argsort(scores[i])[::-1][:5] for i in range(len(y))]))
    rec = []
    for c in np.unique(y):
        m = y == c
        if m.sum():
            rec.append(float((scores[m].argmax(1) == c).mean()))
    return top1, top5, float(np.mean(rec))


def run_probe(Xtr, Xte, ytr, yte, n_class, name):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    W = ridge_fit(Xtr, ytr, n_class)
    Xb = np.c_[Xte, np.ones(len(Xte))]
    s = Xb @ W
    top1, top5, bal = metrics(s, yte)
    maj = np.bincount(ytr).argmax()
    base = float((yte == maj).mean())
    print(f"{name}: top1={top1:.4f} top5={top5:.4f} balanced={bal:.4f} majority={base:.4f}")
    return {"top1": top1, "top5": top5, "balanced": bal, "majority": base}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--train", default="1,2")
    ap.add_argument("--test", default="17")
    ap.add_argument("--per-action", type=int, default=30)
    ap.add_argument("--t-frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="data/meta/probe_results.json")
    args = ap.parse_args()

    pairs = load_pairs(ROOT / "data/meta/p2_radar_config.json")
    train = [int(x) for x in args.train.split(",")]
    test = [int(x) for x in args.test.split(",")]

    Xi, Xr, Y, Yc = collect(train, args.per_action, args.t_frames, args.stride, pairs)
    Ti, Tr, TY, TYc = collect(test, args.per_action, args.t_frames, args.stride, pairs)
    print(f"train={len(Y)} test={len(TY)}", flush=True)

    res = {}
    res["iq_50"] = run_probe(Xi, Ti, Y, TY, 51, "I/Q 50-class")
    res["rpc_50"] = run_probe(Xr, Tr, Y, TY, 51, "RPC 50-class")
    res["iq_3"] = run_probe(Xi, Ti, Yc, TYc, 3, "I/Q 3-coarse")
    res["rpc_3"] = run_probe(Xr, Tr, Yc, TYc, 3, "RPC 3-coarse")
    res["meta"] = {"train": train, "test": test, "per_action": args.per_action,
                   "t_frames": args.t_frames, "stride": args.stride,
                   "n_train": int(len(Y)), "n_test": int(len(TY))}
    Path(ROOT, args.out).write_text(json.dumps(res, indent=2))
    print("saved", ROOT / args.out)


if __name__ == "__main__":
    main()
