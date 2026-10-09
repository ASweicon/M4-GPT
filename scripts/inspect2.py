# -*- coding: utf-8 -*-
"""Second pass: parse RPC binary, open indicator, count action/subject distribution."""
import io, pickle, struct
from collections import Counter
import numpy as np
import lmdb

RPC = r"C:\Users\Congbing\Downloads\radar_pc.lmdb\radar_pc.lmdb"
IND = r"C:\Users\Congbing\Downloads\indicator.lmdb"


def open_db(path, map_gb):
    return lmdb.open(path, subdir=False, readonly=True, lock=False,
                     readahead=False, max_readers=512,
                     map_size=int(map_gb * (1 << 30)))


def parse_rpc(b):
    ndim, n, cols = struct.unpack("<III", b[:12])
    arr = np.frombuffer(b[12:], dtype="<f4").reshape(n, cols)
    return arr


# ---- 1. RPC samples + distribution ----
env = open_db(RPC, 8)
action_frames = Counter()
subj_frames = Counter()
sample_shown = 0
with env.begin() as txn:
    cur = txn.cursor()
    for k, v in cur:
        s, a, f = [int(x) for x in k.decode().strip("[]").split(",")]
        action_frames[a] += 1
        subj_frames[s] += 1
        if sample_shown < 3:
            arr = parse_rpc(v)
            print(f"key {k.decode()} -> points {arr.shape}")
            print("  x range", np.round(arr[:,0].min(),3), np.round(arr[:,0].max(),3),
                  "| y", np.round(arr[:,1].min(),3), np.round(arr[:,1].max(),3),
                  "| z", np.round(arr[:,2].min(),3), np.round(arr[:,2].max(),3))
            print("  intensity", np.round(arr[:,3].min(),3), np.round(arr[:,3].max(),3),
                  "| first point", np.round(arr[0],4))
            sample_shown += 1
env.close()

print("\nsubjects:", sorted(subj_frames.items()))
print("num actions:", len(action_frames))
print("action frames (sorted by id):")
for a in sorted(action_frames):
    print(f"  action {a}: {action_frames[a]}")

# ---- 2. indicator ----
print("=" * 24, "INDICATOR")
try:
    ienv = open_db(IND, 1)
except Exception as e:
    print("OPEN ERROR:", repr(e)); raise SystemExit
with ienv.begin() as txn:
    print("entries:", txn.stat()["entries"])
    cur = txn.cursor()
    for i, (k, v) in enumerate(cur):
        if i >= 3:
            break
        print(f"-- key {k.decode()!r} len={len(v)} head={v[:24].hex()}")
        # try same binary-array header
        if len(v) >= 12:
            ndim, n, cols = struct.unpack("<III", v[:12])
            body = len(v) - 12
            if body == n * cols * 4:
                arr = np.frombuffer(v[12:], dtype="<f4").reshape(
                    (n, cols) if ndim == 2 else (n,))
                print("   binary-array:", arr.shape, np.round(arr.reshape(-1)[:12], 4))
        # try pickle
        try:
            obj = pickle.loads(v)
            print("   pickle:", type(obj), repr(obj)[:300])
        except Exception:
            pass
        # try raw float32
        if len(v) % 4 == 0:
            fl = np.frombuffer(v, dtype="<f4")
            print("   raw f4 first12:", np.round(fl[:12], 4))
ienv.close()
