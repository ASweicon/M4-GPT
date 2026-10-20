# -*- coding: utf-8 -*-
"""Render T=4 radar point clouds into a 3-view image (BEV/Front/Side) with time color.

第 4 格为质心轨迹图 (BEV 投影, 4 帧连线, 时间色), 帮助 movement/dist 类任务
从图中直接读出位移幅度 (只画线不标数值, 避免答案泄漏).

flip=True 时对点云 x 坐标取负后渲染 (左右翻转增强): 文字/坐标轴不镜像,
标签文本由上层同步 swap_lr 互换.
"""
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


def render_clip(txn, s, a, f_center, out_path, flip=False):
    frames = [get_frame(txn, s, a, f_center - T + 1 + i) for i in range(T)]
    if flip:
        frames = [None if f is None else
                  np.column_stack([-f[:, 0], f[:, 1], f[:, 2]])
                  for f in frames]

    fig = plt.figure(figsize=(7.84, 7.84), dpi=100, facecolor="black")
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    views = [("BEV  (x-y)", 0, 1, ("x", "y")),
             ("Front (x-z)", 0, 2, ("x", "z")),
             ("Side (y-z)", 1, 2, ("y", "z"))]

    # flip 后 x 范围镜像: (-3.0, 2.5) -> (-2.5, 3.0)
    xlim_x = (-LIM["x"][1], -LIM["x"][0]) if flip else LIM["x"]
    for ax, (title, i, j, (lx, ly)) in zip(axes[:3], views):
        ax.set_facecolor("black")
        for t, fr in enumerate(frames):
            if fr is None:
                continue
            ax.scatter(fr[:, i], fr[:, j], s=4.5, c=TIME_COLORS[t],
                       alpha=0.75, linewidths=0)
        ax.set_xlim(*(xlim_x if lx == "x" else LIM[lx]))
        ax.set_ylim(*LIM[ly])
        ax.set_title(title, color="white", fontsize=13)
        ax.set_xlabel(lx + " (m)", color="#cccccc", fontsize=11)
        ax.set_ylabel(ly + " (m)", color="#cccccc", fontsize=11)
        ax.tick_params(colors="#999999", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#666666")

    # 第 4 格: 质心轨迹图 (BEV 投影, 时间色连线, 只画线不标数值防泄漏)
    ax = axes[3]; ax.set_facecolor("black")
    ax.set_title("Centroid path (BEV)", color="white", fontsize=13)
    ax.set_xlabel("x (m)", color="#cccccc", fontsize=11)
    ax.set_ylabel("y (m)", color="#cccccc", fontsize=11)
    ax.tick_params(colors="#999999", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#666666")
    cents = [(fr[:, 0].mean(), fr[:, 1].mean())
             for fr in frames if fr is not None]
    if len(cents) >= 2:
        cx, cy = zip(*cents)
        ax.plot(cx, cy, color="#888888", lw=1.2, zorder=2)
        ax.scatter(cx, cy, c=TIME_COLORS[:len(cents)], s=34, zorder=3)
        ax.annotate("S", cents[0], color="#2b83ba", fontsize=12,
                    fontweight="bold", xytext=(4, 4),
                    textcoords="offset points")
        ax.annotate("E", cents[-1], color="#d7191c", fontsize=12,
                    fontweight="bold", xytext=(4, 4),
                    textcoords="offset points")
        # 自适应缩放: 窗口 >= 0.5 m (原地动作保持"几乎不动"的观感),
        # 走路类轨迹放大到可读, 刻度数值保留, 位移仍可估
        span = max(max(cx) - min(cx), max(cy) - min(cy), 0.5)
        mx, my = (min(cx) + max(cx)) / 2, (min(cy) + max(cy)) / 2
        ax.set_xlim(mx - span / 2 - 0.2, mx + span / 2 + 0.2)
        ax.set_ylim(my - span / 2 - 0.2, my + span / 2 + 0.2)
    else:
        ax.set_xlim(*xlim_x); ax.set_ylim(*LIM["y"])
    ax.text(0.02, 0.02, "point color: old (blue) -> now (red)",
            color="#999999", fontsize=8, transform=ax.transAxes)

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
