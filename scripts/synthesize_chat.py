# -*- coding: utf-8 -*-
"""E7: 从 captions.jsonl 合成 RF-GPT 式多轮对话数据 (追问 / 解释 / 反事实).

与 synthesize_instructions.py 的区别: 后者每张图是独立的单轮 QA (9 题型),
本脚本把同一张图的多个可观测事实串成 2~4 轮对话, 逼模型在"看到历史回答"的
前提下继续引用图中证据作答, 而不是背单题模板.

对话结构 (轮数由 md5 决定, 可复现):
  1 action        动作名 (细粒度, 与 E4/E5 的 fine 同源, 兼作分类 rehearsal)
  2 evidence      "你怎么看出来的" —— 必须引用质心位置/运动量/姿态/垂直尺寸
  3 quantity      位移或高度或垂直尺寸定量 (三选一)
  4 counterfactual "镜像后会怎样" —— 左右肢/位置互换, x -> -x, 高度与位移不变

答案全部由点云确定性特征生成 (阈值/措辞与 make_caption 完全一致);
图片本身无标签, 无泄漏. split 继承 captions.jsonl (按被试划分).

用法:
  python scripts/synthesize_chat.py --limit 5      # 每 split 前 5 图打印样例
  python scripts/synthesize_chat.py                # 全量 -> data/chat_{split}.jsonl
  python scripts/synthesize_chat.py --per_image 2  # 每图 2 条对话 (控制体量)
"""
import argparse, hashlib, json, os, random, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_caption import (DATA_ROOT, POS_X, POS_Y, PATH_STILL, PATH_LARGE,
                          Z_SIT, zone)
from m4config import swap_lr

ACTION_Q = [
    "What is the person doing in this radar point cloud?",
    "Which action does this radar clip show?",
    "Identify the motion captured in this radar frame.",
    "Look at the point cloud and tell me the action.",
]
EVIDENCE_Q = [
    "How can you tell?",
    "What in the image supports that?",
    "Which visual cues lead you to that answer?",
    "Explain what you observed in the point cloud.",
]
MOVEMENT_Q = [
    "How much does the body move during these frames?",
    "What is the body displacement across the four frames?",
    "Quantify the movement you see, in metres.",
]
HEIGHT_Q = [
    "How tall does the person appear?",
    "What is the vertical extent of the point cloud, in metres?",
    "Estimate the person's height from the point cloud.",
]
COUNTER_Q = [
    "If this motion were mirrored left to right, what would change?",
    "Suppose we flipped the scene horizontally; how would the answer change?",
    "What would a left-right mirrored version of this scene look like?",
]


def seeded_rng(*key):
    """md5 播种, 跨进程/跨机器可复现."""
    h = int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16)
    return random.Random(h)


def cues(f):
    """把特征翻译成四组可引用的观测事实."""
    lx = zone(f["centroid_x"], *POS_X,
              ("the left part", "the center", "the right part"))
    ly = zone(f["centroid_y"], *POS_Y,
              ("close to the radar", "at middle depth", "far from the radar"))
    pl = f["path_len"]
    if pl < PATH_STILL:
        level, mv = "still", f"the body stays nearly still (centroid path {pl:.2f} m)"
    elif pl > PATH_LARGE:
        level, mv = "notable", f"the body moves notably (centroid path {pl:.2f} m)"
    else:
        level, mv = "slight", f"the body moves slightly (centroid path {pl:.2f} m)"
    seated = f["centroid_z"] < Z_SIT
    posture = "a low, likely seated posture" if seated else "an upright standing posture"
    return {"lx": lx, "ly": ly, "level": level, "movement": mv,
            "posture": posture, "seated": seated}


def evidence_answer(r, c):
    f = r["features"]
    return (f"Three cues: the centroid sits in {c['lx']} of the scene, "
            f"{c['ly']}; {c['movement']}; and the posture looks like "
            f"{c['posture']}. The vertical extent is about "
            f"{f['extent_z']:.1f} m.")


def quantity_answer(kind, r):
    f = r["features"]
    if kind == "movement":
        return (f"About {f['path_len']:.2f} m of centroid path across the four "
                f"frames — {cues(f)['level']} movement.")
    if kind == "height":
        return f"About {f['extent_z']:.1f} m vertical extent."
    return f"About {f['path_len']:.2f} m."          # dist


def counterfactual_answer(action):
    sw = swap_lr(action)
    if sw != action:
        return (f"Mirroring the scene would turn '{action}' into '{sw}'. The "
                f"horizontal position and every left/right limb reference would "
                f"swap, and x becomes -x. Height, depth and the movement "
                f"magnitude would stay the same.")
    return (f"The action name would stay '{action}', but the scene's horizontal "
            f"position and the x-direction would flip; height, depth and the "
            f"movement magnitude are unchanged.")


def make_dialogue(r, d_idx):
    """生成一条 2~4 轮对话; 轮数与问法均由 (图片, 对话序号) 决定."""
    img, action = r["image"], r["action"]
    rng = seeded_rng(img, d_idx)
    n_turns = rng.choice([2, 3, 3, 4])          # 2 轮 25%, 3 轮 50%, 4 轮 25%
    kinds = ["action", "evidence",
             "quantity:" + rng.choice(["movement", "height", "dist"])]
    if n_turns == 4:
        kinds.append("counterfactual")
    kinds = kinds[:n_turns]

    c = cues(r["features"])
    msgs = []
    for t, kind in enumerate(kinds):
        if kind == "action":
            q = rng.choice(ACTION_Q)
            a = action
        elif kind == "evidence":
            q = rng.choice(EVIDENCE_Q)
            a = evidence_answer(r, c)
        elif kind.startswith("quantity:"):
            sub = kind.split(":")[1]
            q = rng.choice({"movement": MOVEMENT_Q, "height": HEIGHT_Q,
                            "dist": MOVEMENT_Q}[sub])
            a = quantity_answer(sub, r)
        else:
            q = rng.choice(COUNTER_Q)
            a = counterfactual_answer(action)
        if t == 0:
            q = "<image>\n" + q
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})

    # refs 与 kinds 一一对应, 供 evaluate_chat.py 自动阅卷
    refs = []
    for kind in kinds:
        if kind == "action":
            refs.append(action)
        elif kind == "evidence":
            refs.append(evidence_answer(r, c))
        elif kind.startswith("quantity:"):
            refs.append(quantity_answer(kind.split(":")[1], r))
        else:
            refs.append(counterfactual_answer(action))

    return {
        "image": img, "task": "chat",
        "messages": msgs,
        "meta": {"kinds": kinds, "refs": refs, "action": action,
                 "coarse": r["coarse"], "subject": r["subject"],
                 "features": r["features"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_image", type=int, default=3,
                    help="每张图生成几条对话 (默认3; train 建议3, test 1 即可)")
    ap.add_argument("--limit", type=int, default=0, help="每 split 只处理前 N 张图")
    args = ap.parse_args()

    by_split = defaultdict(list)
    n_img = Counter()
    with open(os.path.join(DATA_ROOT, "captions.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sp = r["split"]
            if args.limit and n_img[sp] >= args.limit:
                continue
            n_img[sp] += 1
            per = 1 if sp != "train" else args.per_image
            for d in range(per):
                by_split[sp].append(make_dialogue(r, d))

    for sp in ("train", "val", "test"):
        if not by_split[sp]:
            continue
        jp = os.path.join(DATA_ROOT, f"chat_{sp}.jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            for rec in by_split[sp]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        turns = Counter(len(rec["meta"]["kinds"]) for rec in by_split[sp])
        kinds = Counter(k.split(":")[0] for rec in by_split[sp]
                        for k in rec["meta"]["kinds"])
        print(f"== {sp}: {n_img[sp]} images -> {len(by_split[sp])} dialogues -> {jp}")
        print("   轮数: " + "  ".join(f"{k}轮:{v}" for k, v in sorted(turns.items())))
        print("   轮类型: " + "  ".join(f"{k}:{v}" for k, v in sorted(kinds.items())))

    if args.limit:
        for rec in by_split["train"][:2]:
            print("\n--", rec["image"])
            for m in rec["messages"]:
                print(f"{m['role']}: {m['content'][:200]}")


if __name__ == "__main__":
    main()
