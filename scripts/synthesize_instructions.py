# -*- coding: utf-8 -*-
"""Feature 分支: 从 captions.jsonl 合成 RF-GPT 式多任务指令数据 (方案1: 规则模板).

对应 RF-GPT 的 instruction synthesis 环节. 每张图生成:
  fine    (必出)  细粒度动作名 — E4 rehearsal, 防遗忘
  coarse  (必出)  3 类粗分类 — E2 验证过的探针题
  + 从 7 类新题中抽 --per_image 个:
    where     位置 (水平 + 深度), 与 caption 措辞一致
    posture   站姿/低姿态 (二分, from centroid_z)
    movement  运动量三档 (still/slight/notable, from path_len)
    height    垂直高度定量 (from extent_z, 模型须从图估读数值)
    dist      4 帧位移定量 (from path_len, 需理解时序颜色)
    describe  完整 caption 生成式描述
    json      结构化输出 (多字段, 便于 evaluate_gen 自动阅卷)

新题答案全部由点云数学特征确定性计算 (阈值/措辞 import 自 make_caption,
与 caption 完全一致); 图片本身无标签, 无泄漏风险. split 划分继承 captions.jsonl
(按被试), 不存在跨集泄漏.

下游用法 (E5 续训):
  python scripts/train.py --resume outputs/lora --data data/instruct_train.jsonl --lr 2e-5 ...

可复现性: 问法/抽题的随机种子来自 md5(图片名+任务名), 跨进程稳定
(不用内置 hash(), 其受 PYTHONHASHSEED 影响不可跨进程复现).

用法:
  python scripts/synthesize_instructions.py --limit 5   # 每 split 前 5 图, 打印样例
  python scripts/synthesize_instructions.py             # 全量 -> data/instruct_{split}.jsonl
"""
import argparse, hashlib, json, os, random, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_caption import (DATA_ROOT, POS_X, POS_Y, PATH_STILL, PATH_LARGE,
                          Z_SIT, zone)
from build_dataset import FINE_Q, COARSE_Q

# ---- 每图必出的 rehearsal 题 ----
CORE_TASKS = ("fine", "coarse")
# ---- 新题池 (每图抽 --per_image 个) ----
NEW_TASKS = ("where", "posture", "movement", "height", "dist",
             "describe", "json")

Q_POOLS = {
    "fine": FINE_Q,      # 4 问法 (3英1中), 与 E4 训练集同源
    "coarse": COARSE_Q,  # 3 问法
    "where": [
        "Where is the person located in the scene?",
        "In which part of the scene is the person, and how far from the radar?",
        "Tell me the person's position in the scene (side and depth).",
    ],
    "posture": [
        "Is the person standing upright or seated/low?",
        "What posture does the person appear to be in?",
        "Judging from the point cloud, is the person standing or sitting?",
    ],
    "movement": [
        "How much does the person move during the four frames?",
        "Does the person stay roughly in place or move notably across the frames?",
        "Describe the amount of body movement over the four frames.",
    ],
    "height": [
        "How tall does the person appear in this point cloud?",
        "What is the approximate height of the person's point cloud?",
        "Estimate the vertical extent of the person, in metres.",
    ],
    "dist": [
        "How far does the person's body move over the four frames?",
        "What distance does the person travel during the four frames?",
        "Estimate the total body displacement across the frames, in metres.",
    ],
    "describe": [
        "Describe what you see in this radar point cloud.",
        "Summarize this radar scene: who is where, doing what, and how.",
        "Give a full description of this radar point cloud scene.",
    ],
    "json": [
        "Analyze this radar point cloud and answer in JSON with fields: "
        "action, category, position (horizontal, depth), posture, "
        "movement_m, vertical_extent_m, points.",
        "Return a structured JSON summary of this radar scene "
        "(action, category, position, posture, movement_m, "
        "vertical_extent_m, points).",
    ],
}


def seeded_rng(*key):
    """md5 播种的可复现随机源 (跨进程/跨机器稳定)."""
    h = int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16)
    return random.Random(h)


def answer(task, r):
    """由 features 确定性生成答案, 措辞与 make_caption 完全一致."""
    f = r["features"]
    if task == "fine":
        return r["action"]
    if task == "coarse":
        return r["coarse"]
    if task == "where":
        lx = zone(f["centroid_x"], *POS_X,
                  ("the left part", "the center", "the right part"))
        ly = zone(f["centroid_y"], *POS_Y,
                  ("close to the radar", "at middle depth", "far from the radar"))
        return f"The person is located in {lx} of the scene, {ly}."
    if task == "posture":
        p = ("a low, likely seated posture" if f["centroid_z"] < Z_SIT
             else "an upright standing posture")
        return f"The person appears to be in {p}."
    if task == "movement":
        pl = f["path_len"]
        if pl < PATH_STILL:
            return f"The body remains nearly still (centroid path of {pl:.2f} m across four frames)."
        if pl > PATH_LARGE:
            return f"The body shows notable movement (centroid path of {pl:.2f} m)."
        return f"The body moves slightly (centroid path of {pl:.2f} m)."
    if task == "height":
        return f"About {f['extent_z']:.1f} m."
    if task == "dist":
        return f"About {f['path_len']:.2f} m."
    if task == "describe":
        return r["caption"]
    if task == "json":
        return json.dumps({
            "action": r["action"], "category": r["coarse"],
            "position": {"horizontal": zone(f["centroid_x"], *POS_X,
                                            ("left", "center", "right")),
                        "depth": zone(f["centroid_y"], *POS_Y,
                                      ("close", "middle", "far"))},
            "posture": "likely seated" if f["centroid_z"] < Z_SIT else "standing",
            "movement_m": f["path_len"],
            "vertical_extent_m": round(f["extent_z"], 2),
            "points": f["points"],
        }, ensure_ascii=False)
    raise ValueError(task)


def make_records(r, per_image):
    """每图: fine + coarse 必出, 再从 NEW_TASKS 抽 per_image 个新题."""
    img = r["image"]
    tasks = list(CORE_TASKS)
    if per_image < len(NEW_TASKS):
        tasks += seeded_rng("mix", img).sample(list(NEW_TASKS), per_image)
    else:
        tasks += list(NEW_TASKS)
    out = []
    for t in tasks:
        q = seeded_rng(img, t).choice(Q_POOLS[t])
        out.append({
            "image": img, "task": t,
            "messages": [
                {"role": "user", "content": "<image>\n" + q},
                {"role": "assistant", "content": answer(t, r)},
            ],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_image", type=int, default=3,
                    help="每图新增题型数 (1~7, 默认3: fine+coarse+3新题=5条/图)")
    ap.add_argument("--limit", type=int, default=0,
                    help="每 split 只处理前 N 张图 (debug)")
    args = ap.parse_args()
    if not 1 <= args.per_image <= len(NEW_TASKS):
        sys.exit(f"--per_image must be 1..{len(NEW_TASKS)}")

    by_split = defaultdict(list)
    n_img = Counter()
    with open(os.path.join(DATA_ROOT, "captions.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sp = r["split"]
            if args.limit and n_img[sp] >= args.limit:
                continue
            n_img[sp] += 1
            by_split[sp].extend(make_records(r, args.per_image))

    for sp in ("train", "val", "test"):
        if not by_split[sp]:
            continue
        jp = os.path.join(DATA_ROOT, f"instruct_{sp}.jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            for rec in by_split[sp]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dist = Counter(rec["task"] for rec in by_split[sp])
        print(f"== {sp}: {n_img[sp]} images -> {len(by_split[sp])} records -> {jp}")
        print("   " + "  ".join(f"{t}:{n}" for t, n in sorted(dist.items())))

    if args.limit:
        for rec in by_split["train"][:6]:
            print(f"\n-- [{rec['task']}] {rec['image']}")
            print("Q:", rec["messages"][0]["content"].replace("<image>\n", ""))
            print("A:", rec["messages"][1]["content"][:220])


if __name__ == "__main__":
    main()
