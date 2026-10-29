# -*- coding: utf-8 -*-
"""NumPy-only DSP check: load a p2 clip, run iq_to_rda, compare range peak to RPC."""

import json
import struct

import lmdb
import numpy as np

from dsp import iq_to_rda, load_pairs, range_peak_m, range_profile

IQ = "/Users/weicongbing/Desktop/m4gpt/data/iq/p2_iq_int16.lmdb"
RPC = "/Users/weicongbing/Desktop/m4gpt/radar_pc.lmdb"
CFG = "/Users/weicongbing/Desktop/m4gpt/data/meta/p2_radar_config.json"
HDR = struct.Struct("<IIIIf")


def load_clip(env, subject, action, center, t_frames=16):
    prefix = f"[{subject}, {action}, "
    items = []
    with env.begin() as txn:
        cur = txn.cursor()
        cur.set_range(prefix.encode())
        for k, v in cur:
            if not k.decode().startswith(prefix):
                break
            frame = int(k.decode().strip("[]").split(",")[2].strip())
            items.append((frame, v))
    items.sort(key=lambda x: abs(x[0] - center))
    items = sorted(items[:t_frames], key=lambda x: x[0])

    frames = []
    for _, v in items:
        magic, h, w, c, scale = HDR.unpack_from(v, 0)
        data = np.frombuffer(v, dtype="<i2", offset=HDR.size).reshape(c, h, w)
        frames.append(data[0].astype(np.float32) * scale + 1j * (data[1].astype(np.float32) * scale))
    return np.stack(frames).astype(np.complex64), [f for f, _ in items]


def rpc_ranges(env, subject, action, frame):
    with env.begin() as txn:
        v = txn.get(f"[{subject}, {action}, {frame}]".encode())
    if v is None:
        return None
    ndim, n, cols = struct.unpack("<III", v[:12])
    p = np.frombuffer(v[12:], dtype="<f4").reshape(n, cols)
    return np.linalg.norm(p[:, :3], axis=1)


def main():
    pairs = load_pairs(CFG)
    env = lmdb.open(IQ, subdir=False, readonly=True, lock=False, readahead=False)
    iq, frames = load_clip(env, 2, 10, 300)
    rda = iq_to_rda(iq, pairs)
    peak_m, peak_bin = range_peak_m(rda)
    print("clip frames:", frames)
    print("iq:", iq.shape, iq.dtype, "abs max:", float(np.abs(iq).max()))
    print("rda:", rda.shape, rda.dtype, "min/max:", float(rda.min()), float(rda.max()))
    print(f"range peak (rda): bin={peak_bin} -> {peak_m:.2f} m")

    prof = range_profile(iq, pairs)
    prof[:4] = 0.0
    prof[-4:] = 0.0
    lin_peak = int(np.argmax(prof))
    lin_m = lin_peak * (151.0 / 256) * 0.09375
    print(f"range peak (linear profile): bin={lin_peak} -> {lin_m:.2f} m")
    top = np.argsort(prof)[::-1][:5]
    print("linear top5:", [(int(i), round(i * (151.0 / 256) * 0.09375, 2)) for i in top])

    rpc_env = lmdb.open(RPC, subdir=False, readonly=True, lock=False, readahead=False)
    r = rpc_ranges(rpc_env, 2, 10, frames[len(frames) // 2])
    if r is not None:
        print(f"rpc range median={np.median(r):.2f} m  p10={np.percentile(r,10):.2f}  "
              f"p90={np.percentile(r,90):.2f}  n={len(r)}")
    env.close()
    rpc_env.close()


if __name__ == "__main__":
    main()
