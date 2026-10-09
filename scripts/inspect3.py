# -*- coding: utf-8 -*-
"""Check what indicator.lmdb encodes."""
from collections import Counter
import lmdb

IND = r"C:\Users\Congbing\Downloads\indicator.lmdb"
env = lmdb.open(IND, subdir=False, readonly=True, lock=False,
                readahead=False, map_size=int(1 << 30))
prefix = Counter()
match = 0
mismatch = 0
n = 0
with env.begin() as txn:
    for k, v in txn.cursor():
        s, a, f = [int(x) for x in k.decode().strip("[]").split(",")]
        prefix[v[:3].hex()] += 1
        if len(v) == 4 and v[3] == (f % 256):
            match += 1
        else:
            mismatch += 1
        n += 1
env.close()
print("total", n)
print("prefix of first 3 bytes distribution:", prefix.most_common(10))
print("byte3 == frame%256 -> match", match, "mismatch", mismatch)
