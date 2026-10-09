# -*- coding: utf-8 -*-
"""Dump indicator for consecutive frames of one (subject,action) to find pattern."""
import lmdb

IND = r"C:\Users\Congbing\Downloads\indicator.lmdb"
env = lmdb.open(IND, subdir=False, readonly=True, lock=False,
                readahead=False, map_size=int(1 << 30))

def get(s, a, f):
    with env.begin() as txn:
        v = txn.get(f"[{s}, {a}, {f}]".encode())
    return v.hex() if v else None

for label, frames in [("0..20", range(0, 21)),
                      ("250..272", range(250, 273)),
                      ("500..510", range(500, 511)),
                      ("680..697", range(680, 698))]:
    print("---- frames", label)
    for f in frames:
        print(f"f={f:<4} ind={get(1,10,f)}")
env.close()
