# -*- coding: utf-8 -*-
"""Feature 分支: 从 LMDB 为每张渲染图导出确定性元数据 caption.

对应 RF-GPT 的 deterministic captioning 环节: 对每张图 (s,a,fc) 读取与
render_clip 完全相同的 T=4 帧点云, 计算只含视觉可观测信息的数值特征
(点数/质心/轨迹长度/包围盒), 生成结构化英文 caption.

caption 本身不进训练, 它是下一环节 synthesize_instructions.py (LLM 指令合成)
的 grounding 输入; 图片渲染时不含任何标签, 无泄漏风险.

用法:
  python scripts/make_caption.py --limit 5    # 抽样打印前 5 条
  python scripts/make_caption.py              # 全量导出 data/captions.jsonl
  python scripts/make_caption.py --stats     # 按 coarse 组打印特征分布(校准阈值)
"""
import argparse, json, os, re, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m4config import RPC, ACTION_NAMES, coarse_group, swap_lr
from render_samples import open_db, get_frame, T

DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "data"))

FN_RE = re.compile(r"^s(\d+)_a(\d+)_f(\d+)(flip)?\.png$")

# ---- 描述阈值 (初版推测值, 待 --stats 分布校准后调整) ----
POS_X = (-0.5, 0.5)     # BEV 水平: 左 / 中 / 右 (x 全程 -3 ~ 2.5 m)
POS_Y = (2.0, 4.0)      # 深度: 近 / 中 / 远 (y 全程 0 ~ 6 m)
PATH_STILL = 0.25       # 4 帧质心轨迹 < 0.25 m 视为原地 (校准: 全量 p25)
PATH_LARGE = 0.5        # > 0.5 m 视为明显位移 (校准: 全量 p75)
Z_SIT = 0.85            # 质心高度 < 0.85 m 提示低/坐姿 (校准: sit组p75=0.83, 站组p25=0.87)


def clip_features(txn, s, a, fc):
    """读 T=4 帧点云, 返回确定性数值特征. 帧选择与 render_clip 完全一致."""
    frames = [get_frame(txn, s, a, fc - T + 1 + i) for i in range(T)]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    allpts = np.vstack(frames)
    cents = np.array([(f[:, 0].mean(), f[:, 1].mean()) for f in frames])
    path = float(np.hypot(*(np.diff(cents, axis=0).T)).sum()) if len(cents) > 1 else 0.0
    return {
        "points": int(len(allpts)),
        "centroid_x": round(float(allpts[:, 0].mean()), 3),
        "centroid_y": round(float(allpts[:, 1].mean()), 3),
        "centroid_z": round(float(allpts[:, 2].mean()), 3),
        "path_len": round(path, 3),
        "extent_x": round(float(np.ptp(allpts[:, 0])), 3),
        "extent_y": round(float(np.ptp(allpts[:, 1])), 3),
        "extent_z": round(float(np.ptp(allpts[:, 2])), 3),
        "frames_used": len(frames),
    }


def zone(v, lo, hi, names):
    return names[0] if v < lo else names[2] if v > hi else names[1]


def make_caption(feat, action, coarse):
    """由数值特征生成结构化 caption (只含图中可观测信息 + 动作名 grounding)."""
    art = "an" if coarse[0] in "aeiou" else "a"  # an in-place / a sit-in-place ...
    lx = zone(feat["centroid_x"], *POS_X, ("the left part", "the center", "the right part"))
    ly = zone(feat["centroid_y"], *POS_Y, ("close to the radar", "at middle depth", "far from the radar"))
    if feat["path_len"] < PATH_STILL:
        move = (f"the body centroid remains nearly still "
                f"(shifting only {feat['path_len']:.2f} m across four frames)")
    elif feat["path_len"] > PATH_LARGE:
        move = f"the body shows notable movement (a centroid path of {feat['path_len']:.2f} m)"
    else:
        move = f"the body moves slightly (centroid path of {feat['path_len']:.2f} m)"
    posture = ("a low, likely seated posture" if feat["centroid_z"] < Z_SIT
               else "an upright standing posture")
    return (f"Radar point cloud showing a person performing '{action}', "
            f"which is {art} {coarse} action. The person is located in {lx} of the scene, {ly}. "
            f"Across four consecutive frames, {move}. "
            f"The point cloud suggests {posture}, "
            f"with a vertical extent of about {feat['extent_z']:.1f} m.")


def collect_images(limit):
    """从三个 jsonl 提取 unique 图片 (split, s, a, fc, flip), 与训练管线一一对应."""
    seen, rows = set(), []
    for sp in ("train", "val", "test"):
        p = os.path.join(DATA_ROOT, f"{sp}.jsonl")
        with open(p, encoding="utf-8") as f:
            for line in f:
                img = json.loads(line)["image"]
                if img in seen:
                    continue
                seen.add(img)
                m = FN_RE.match(os.path.basename(img))
                if not m:
                    sys.exit(f"bad filename: {img}")
                s, a, fc = map(int, m.groups()[:3])
                rows.append((sp, img, s, a, fc, bool(m.group(4))))
                if limit and len(rows) >= limit:
                    break
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张图")
    ap.add_argument("--stats", action="store_true", help="只打印特征分布, 不写文件")
    args = ap.parse_args()

    rows = collect_images(args.limit or None)
    print(f"unique images: {len(rows)}", flush=True)

    env = open_db(RPC, 8)
    out, stats = [], defaultdict(lambda: defaultdict(list))
    with env.begin() as txn:
        for sp, img, s, a, fc, flip in rows:
            name, cg = ACTION_NAMES[a], coarse_group(a)
            feat = clip_features(txn, s, a, fc)
            if feat is None:
                print(f"!! no points for {img}, skipped", flush=True)
                continue
            if flip:
                # 左右翻转: x 取负 (zone 的 left/right 自动随 x 正确),
                # path_len/extent/points 与 x 无符号性, 不变
                name = swap_lr(name)
                feat = dict(feat, centroid_x=round(-feat["centroid_x"], 3))
            if args.stats:
                g = stats[cg]
                for k in ("points", "centroid_z", "path_len", "centroid_x", "centroid_y"):
                    g[k].append(feat[k])
                continue
            out.append({
                "image": img, "split": sp,
                "subject": s, "action_id": a,
                "action": name, "coarse": cg,
                "features": feat,
                "caption": make_caption(feat, name, cg),
            })
            if len(out) % 4000 == 0:
                print(f"  {len(out)} done", flush=True)
    env.close()

    if args.stats:
        for cg, g in stats.items():
            print(f"\n== {cg} (n={len(g['points'])})")
            for k, v in g.items():
                a = np.array(v)
                print(f"  {k:>11}: min {a.min():7.3f}  p25 {np.percentile(a,25):7.3f}  "
                      f"median {np.median(a):7.3f}  p75 {np.percentile(a,75):7.3f}  max {a.max():7.3f}")
        return

    jp = os.path.join(DATA_ROOT, "captions.jsonl")
    with open(jp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_sit_like = sum(1 for r in out if r["features"]["centroid_z"] < Z_SIT)
    print(f"== {len(out)} captions -> {jp}", flush=True)
    print(f"   (z<{Z_SIT}m 低姿态 {n_sit_like} 条, 占比 {n_sit_like/len(out):.1%} — "
          f"若与 sit-in-place 组占比偏差大, 需校准 Z_SIT)", flush=True)
    if args.limit:
        for r in out[:min(args.limit, 5)]:
            print("\n--", r["image"])
            print(r["caption"])


if __name__ == "__main__":
    main()
