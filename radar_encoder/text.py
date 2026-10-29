# -*- coding: utf-8 -*-
"""Generate caption / QA / multi-turn text from SMPL-X params + action names.

No hand-written posture thresholds: posture is taken from the official coarse
group, and geometric stats come from the MoCap joints.

Run from the repo root:
    python3 radar_encoder/text.py --subjects 1,2,17,19
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

try:
    from .dataset import IQClipDataset, parse_key
    from .dsp import load_pairs
except ImportError:
    from dataset import IQClipDataset, parse_key
    from dsp import load_pairs

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
FPS = 12.0


def load_m4config():
    path = CODE_ROOT / "scripts/m4config.py"
    spec = importlib.util.spec_from_file_location("m4config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clip_stats(params, frames, frame_to_pos):
    idx = [frame_to_pos[f] for f in frames if f in frame_to_pos]
    idx = [i for i in idx if i < len(params["joints"])]
    if not idx:
        return None
    joints = np.asarray([params["joints"][i] for i in idx], dtype=np.float32)  # [t,55,3]
    trans = np.asarray([params["trans"][i] for i in idx], dtype=np.float32)    # [t,3]
    z = joints[:, :, 2]
    height = float(z.max() - z.min())
    root_h = float(trans[:, 2].mean())
    dx = float(trans[-1, 0] - trans[0, 0])
    dy = float(trans[-1, 1] - trans[0, 1])
    disp = float(np.hypot(dx, dy))
    speed = disp / max(len(idx) / FPS, 1e-3)
    return {"height": height, "root_height": root_h, "displacement": disp, "speed": speed}


def make_messages(action_name, coarse, s):
    a = action_name.lower()
    cap = (f"A person is {a}. The body spans about {s['height']:.2f} m vertically, "
           f"and moves {s['displacement']:.2f} m horizontally during the clip.")
    msgs = [
        {"role": "user", "content": "What is the person doing?"},
        {"role": "assistant", "content": action_name},
        {"role": "user", "content": "Which coarse category is it?"},
        {"role": "assistant", "content": coarse},
        {"role": "user", "content": "How much horizontal movement is there?"},
        {"role": "assistant", "content": f"{s['displacement']:.2f} m"},
        {"role": "user", "content": "What is the approximate body height?"},
        {"role": "assistant", "content": f"{s['height']:.2f} m"},
    ]
    return cap, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--subjects", default="1,2,17,19")
    ap.add_argument("--t-frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    cfg = json.load(open(root / "data/meta/manifest.json"))
    pairs = load_pairs(root / "data/meta/p2_radar_config.json")
    m4 = load_m4config()
    out_dir = root / "data/text"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for sub in [int(x) for x in args.subjects.split(",") if x]:
        info = cfg["subjects"].get(str(sub))
        if not info:
            continue
        ds = IQClipDataset(root / info["lmdb"], sub, pairs,
                           t_frames=args.t_frames, stride=args.stride)
        params_dir = root / "data/params" / f"p{sub}" / "parameters"
        for action in sorted(ds.frames_by_action):
            pfile = params_dir / f"{action}.json"
            if not pfile.exists():
                continue
            params = json.load(open(pfile))
            frames = ds.frames_by_action[action]
            frame_to_pos = {f: i for i, f in enumerate(frames)}
            clips = [c for a, c in ds.clips if a == action]
            name = m4.ACTION_NAMES.get(action, f"action {action}")
            coarse = m4.coarse_group(action)
            path = out_dir / f"p{sub}_a{action}.jsonl"
            n = 0
            with open(path, "w") as fo:
                for frames_clip in clips:
                    center = frames_clip[len(frames_clip) // 2]
                    s = clip_stats(params, frames_clip, frame_to_pos)
                    if s is None:
                        continue
                    cap, msgs = make_messages(name, coarse, s)
                    rec = {
                        "key": f"[{sub}, {action}, {center}]",
                        "subject": sub,
                        "action": action,
                        "frames": frames_clip,
                        "center": center,
                        "stats": s,
                        "caption": cap,
                        "messages": msgs,
                    }
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            total += n
            print(f"p{sub} a{action}: {n} clips")
        ds.close()
    print("total text clips:", total)


if __name__ == "__main__":
    main()
