# -*- coding: utf-8 -*-
"""Inspect M4Human lmdb files: verify CUDA, dump keys and value serialization."""
import io, pickle, sys
import numpy as np
import lmdb

# ---- 1. CUDA check ----
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("cuda compute ok:", float((x @ x).sum()) != 0.0)

# ---- 2. lmdb inspection ----
RPC = r"C:\Users\Congbing\Downloads\radar_pc.lmdb\radar_pc.lmdb"
IND = r"C:\Users\Congbing\Downloads\indicator.lmdb\indicator.lmdb"


def summarize(obj):
    if isinstance(obj, np.ndarray):
        return f"ndarray shape={obj.shape} dtype={obj.dtype}"
    if isinstance(obj, (list, tuple)):
        parts = [summarize(x) for x in obj[:6]]
        return f"{type(obj).__name__}(len={len(obj)}) [{', '.join(parts)}]"
    if isinstance(obj, dict):
        return "dict keys=" + str(list(obj.keys())[:20])
    return repr(obj)[:200]


def decode_value(b):
    info = {"len": len(b), "head_hex": b[:16].hex()}
    if b[:6] == b"\x93NUMPY":
        try:
            arr = np.load(io.BytesIO(b), allow_pickle=True)
            info["npy"] = (getattr(arr, "shape", None), str(getattr(arr, "dtype", None)))
            if arr.dtype == object:
                info["npy_item0"] = summarize(arr.item() if arr.shape == () else arr.reshape(-1)[0])
        except Exception as e:
            info["npy_err"] = repr(e)[:200]
    try:
        obj = pickle.loads(b)
        info["pickle_type"] = str(type(obj))
        info["pickle_summary"] = summarize(obj)
    except Exception as e:
        info["pickle_err"] = repr(e)[:150]
    return info


def open_db(path, map_gb):
    return lmdb.open(path, subdir=False, readonly=True, lock=False,
                     readahead=False, max_readers=512,
                     map_size=int(map_gb * (1 << 30)))


for name, path, map_gb in [("RPC", RPC, 8), ("INDICATOR", IND, 1)]:
    print("=" * 24, name)
    print(path)
    try:
        env = open_db(path, map_gb)
    except Exception as e:
        print("OPEN ERROR:", repr(e))
        continue
    with env.begin() as txn:
        print("entries:", txn.stat()["entries"])
        cur = txn.cursor()
        for i, (k, v) in enumerate(cur):
            if i >= 3:
                break
            print(f"-- key[{i}]: {k.decode('utf-8','replace')!r}")
            print("   value:", decode_value(v))
    env.close()
