# -*- coding: utf-8 -*-
"""Render T=4 radar point clouds into a 3-view image (BEV/Front/Side) with time color."""
import os, struct, sys
import numpy as np
import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from m4config import RPC, LIM, ACTION_NAMES, coarse_group

OUT_SIZE = 784          # multiple of 28 (784/28=28 -> 784 visual tokens, 高清版)
T = 4
TIME_COLORS = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]  # old -> new


def open_db(path, map_gb=8):
    return lmdb.open(path, subdir=False, readonly=True, lock=False,
                     readahead=False, max_readers=512,
                     map_size=int(map_gb * (1 << 30)))


def get_frame(txn, s, a, f):
    v = txn.get(f"[{s}, {a}, {f}]".encode())
    if v is None:
        return None
    ndim, n, cols = struct.unpack("<III", v[:12])
    return np.frombuffer(v[12:], dtype="<f4").reshape(n, cols)


def list_frames(txn, s, a):
    prefix = f"[{s}, {a}, ".encode()
    frames = []
    cur = txn.cursor()
    if cur.set_range(prefix):
        for k, v in cur:
            if not k.startswith(prefix):
                break
            frames.append(int(k.decode().strip("[]").split(",")[2]))
    return sorted(frames)


def render_clip(txn, s, a, f_center, out_path):
    frames = [get_frame(txn, s, a, f_center - T + 1 + i) for i in range(T)]

    fig = plt.figure(figsize=(7.84, 7.84), dpi=100, facecolor="black")
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    views = [("BEV  (x-y)", 0, 1, ("x", "y")),
             ("Front (x-z)", 0, 2, ("x", "z")),
             ("Side (y-z)", 1, 2, ("y", "z"))]

    for ax, (title, i, j, (lx, ly)) in zip(axes[:3], views):
        ax.set_facecolor("black")
        for t, fr in enumerate(frames):
            if fr is None:
                continue
            ax.scatter(fr[:, i], fr[:, j], s=4.5, c=TIME_COLORS[t],
                       alpha=0.75, linewidths=0)
        ax.set_xlim(*LIM[lx]); ax.set_ylim(*LIM[ly])
        ax.set_title(title, color="white", fontsize=13)
        ax.set_xlabel(lx + " (m)", color="#cccccc", fontsize=11)
        ax.set_ylabel(ly + " (m)", color="#cccccc", fontsize=11)
        ax.tick_params(colors="#999999", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#666666")

    # legend panel ONLY: never render subject/action/coarse (would leak labels)
    ax = axes[3]; ax.set_facecolor("black"); ax.axis("off")
    ax.text(0.05, 0.88, "time order\n(old -> recent)", color="white",
            fontsize=12, va="top", family="monospace", transform=ax.transAxes)
    for t in range(T):
        y = 0.58 - t * 0.105
        ax.add_patch(plt.Rectangle((0.05, y), 0.12, 0.06,
                                   color=TIME_COLORS[t], transform=ax.transAxes))
        ax.text(0.22, y + 0.026, f"t-{T-1-t}" if t < T-1 else "t (now)",
                color="#dddddd", fontsize=11, transform=ax.transAxes)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.07,
                        hspace=0.28, wspace=0.28)
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, facecolor="black")
    plt.close(fig)
    Image.open(tmp).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS).save(out_path)
    os.remove(tmp)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "samples")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    samples = [(1, 7), (2, 22), (3, 47), (4, 48), (2, 43), (1, 31)]
    env = open_db(RPC, 8)
    with env.begin() as txn:
        for s, a in samples:
            fr = list_frames(txn, s, a)
            if not fr:
                print("missing", s, a); continue
            fc = fr[len(fr) // 2]
            out = os.path.join(out_dir, f"sub{s}_act{a:02d}.png")
            render_clip(txn, s, a, fc, out)
            print("saved", out, "| frames", len(fr), "| center", fc)
    env.close()


if __name__ == "__main__":
    main()
