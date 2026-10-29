# -*- coding: utf-8 -*-
"""Server-side smoke test (requires torch). Run from this directory:
    python3 smoke_test.py
"""

import lmdb
import numpy as np
import torch

from dsp import iq_to_rda, load_pairs
from encoder import build_encoder, count_params
from test_dsp import CFG, IQ, load_clip


def main():
    pairs = load_pairs(CFG)
    env = lmdb.open(IQ, subdir=False, readonly=True, lock=False, readahead=False)
    iq, frames = load_clip(env, 2, 10, 300)
    env.close()
    rda = iq_to_rda(iq, pairs)
    x = torch.from_numpy(rda).unsqueeze(0)  # [1, T, 2, R, D, A]

    model = build_encoder(llm_dim=4096)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print("frames:", frames)
    print("input:", tuple(x.shape))
    print("output:", tuple(out.shape))
    print("params:", f"{count_params(model):,}")
    print("finite:", bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    main()
