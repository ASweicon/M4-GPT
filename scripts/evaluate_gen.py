# -*- coding: utf-8 -*-
"""E5 生成式评估: 9 类题型自动阅卷器 (模型自由生成答案, 规则判分).

与 evaluate.py 互补: evaluate.py 用似然打分只能考选择题 (fine/coarse);
本脚本考 E5 真正教的能力 —— 让模型自己"说"出答案, 按题型自动判分:

  task      考什么                判分规则
  ----      ------                --------
  fine      50 类动作名            标准答案出现在生成文本中
  coarse    3 类粗分类            标签精确匹配 (最长命中, 防子串陷阱)
  where     水平位置 + 深度       两分量都对才得分
  posture   站姿/低姿态           二分关键词
  movement  运动量档位+路径数值    档位对 且 数值 ±20%
  height    垂直高度              数值 ±15% (至少 0.25 m)
  dist      4 帧位移              数值 ±20% (至少 0.20 m)
  json      8 字段结构化输出      逐字段比对, 得分 = 对字段占比
  describe  完整描述              动作名命中 (宽松参考) + 样本存档人审

用法:
  python scripts/evaluate_gen.py --ckpt outputs/e5_lora --per_task 30
  python scripts/evaluate_gen.py --ckpt outputs/e5_lora --tasks fine,where --per_task 10
  # 冒烟: python scripts/evaluate_gen.py --ckpt <adapter目录> --tasks fine,json --per_task 2

抽样: 每题型按 (图片, 问题) 排序后等距抽样, 跨运行/跨机器可复现.
结果写入 outputs/eval/<ckpt目录名>_gen/:
  results.json        按题型汇总
  predictions.jsonl   每题的 问题/标准答案/模型生成/得分 (逐条可复查)
  describe_samples.txt describe 样本前 10 条 (人工抽查用)
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "data"))
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

TASKS = ("fine", "coarse", "where", "posture", "movement",
         "height", "dist", "json", "describe")
MAX_NEW = {"fine": 24, "coarse": 8, "where": 48, "posture": 32,
           "movement": 48, "height": 16, "dist": 16,
           "json": 192, "describe": 128}
JSON_FIELDS = ("action", "category", "horizontal", "depth", "posture",
               "movement_m", "vertical_extent_m", "points")


# ---------- 判分工具 ----------

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".")


def num_in(text):
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    return float(m.group()) if m else None


def ok_num(pred, ref, abs_tol, rel_tol):
    """数值判分: 绝对容差与相对容差取宽者."""
    if pred is None or ref is None:
        return False
    return abs(pred - ref) <= max(abs_tol, rel_tol * abs(ref))


COARSE_LABELS = ("in-place", "sit-in-place", "non-in-place")


def parse_where(t):
    t = str(t).lower()
    h = next((k for k in ("left", "right", "center") if k in t), None)
    d = next((k for k in ("close", "far", "middle") if k in t), None)
    return h, d


def parse_posture(t):
    t = str(t).lower()
    seated = "seat" in t or "low" in t
    standing = "stand" in t or "upright" in t
    if seated != standing:          # 恰好命中一边才算数
        return "seated" if seated else "standing"
    return None


def parse_move_level(t):
    t = str(t).lower()
    if "still" in t:
        return "still"
    if "notable" in t:
        return "notable"
    if "slight" in t:
        return "slight"
    return None


def extract_json(text):
    """从生成文本中抠出第一个配平的 {...} 并 parse; 抠不出返回 None."""
    text = str(text)
    i = text.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    frag = text[i:j + 1]
                    try:
                        return json.loads(frag)
                    except json.JSONDecodeError:
                        fixed = frag.replace(",}", "}").replace(",]", "]")
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            break
        i = text.find("{", i + 1)
    return None


def as_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------- 各题型阅卷: 返回 (得分 0~1, 附加明细 dict) ----------

def grade_fine(ref, pred):
    return float(norm(ref) in norm(pred)), {}


def grade_coarse(ref, pred):
    t = norm(pred)
    hit = [lb for lb in COARSE_LABELS if lb in t]
    # 取最长命中, 避免 "in-place" 淹没 "sit-in-place"/"non-in-place"
    return float(bool(hit) and max(hit, key=len) == norm(ref)), {}


def grade_where(ref, pred):
    (rh, rd), (ph, pd_) = parse_where(ref), parse_where(pred)
    return float((rh, rd) == (ph, pd_)), {"h": int(rh == ph),
                                         "d": int(rd == pd_)}


def grade_posture(ref, pred):
    r, p = parse_posture(ref), parse_posture(pred)
    return float(r is not None and r == p), {}


def grade_movement(ref, pred):
    rl, pl = parse_move_level(ref), parse_move_level(pred)
    rn, pn = num_in(ref), num_in(pred)
    num_ok = ok_num(pn, rn, 0.20, 0.20)
    return float(rl == pl and num_ok), {"level": int(rl == pl),
                                        "num": int(num_ok)}


def grade_height(ref, pred):
    return float(ok_num(num_in(pred), num_in(ref), 0.25, 0.15)), {}


def grade_dist(ref, pred):
    return float(ok_num(num_in(pred), num_in(ref), 0.20, 0.20)), {}


def grade_json(ref, pred):
    rj, pj = extract_json(ref), extract_json(pred)
    if not isinstance(rj, dict) or not isinstance(pj, dict):
        return 0.0, {"fields": {k: 0 for k in JSON_FIELDS}, "parse": 0}
    rp, pp = rj.get("position"), pj.get("position")
    rp = rp if isinstance(rp, dict) else {}
    pp = pp if isinstance(pp, dict) else {}

    def sub(ref_v, pred_v):     # 单词级包含: 容忍 "the center" 这类写法
        return norm(ref_v) != "" and norm(ref_v) in norm(pred_v)

    ra = parse_posture(rj.get("posture", ""))
    pa = parse_posture(pj.get("posture", ""))
    f = {
        "action": sub(rj.get("action", ""), pj.get("action", "")),
        "category": norm(rj.get("category", "")) == norm(pj.get("category", "")),
        "horizontal": sub(rp.get("horizontal", ""), pp.get("horizontal", "")),
        "depth": sub(rp.get("depth", ""), pp.get("depth", "")),
        "posture": ra is not None and ra == pa,
        "movement_m": ok_num(as_num(pj.get("movement_m")),
                             as_num(rj.get("movement_m")), 0.20, 0.20),
        "vertical_extent_m": ok_num(as_num(pj.get("vertical_extent_m")),
                                    as_num(rj.get("vertical_extent_m")),
                                    0.25, 0.15),
        "points": ok_num(as_num(pj.get("points")),
                         as_num(rj.get("points")), 0.0, 0.40),
    }
    fields = {k: int(bool(v)) for k, v in f.items()}
    return sum(fields.values()) / len(fields), {"fields": fields, "parse": 1}


def grade_describe(ref, pred):
    m = re.search(r"'([^']+)'", str(ref))       # caption 里动作名带单引号
    hit = bool(m) and norm(m.group(1)) in norm(pred)
    return float(hit), {"action_hit": int(hit)}


GRADE = {"fine": grade_fine, "coarse": grade_coarse, "where": grade_where,
         "posture": grade_posture, "movement": grade_movement,
         "height": grade_height, "dist": grade_dist,
         "json": grade_json, "describe": grade_describe}


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/e5_lora",
                    help="待评估 adapter 目录 (相对项目根; 结果写 "
                         "outputs/eval/<目录名>_gen/)")
    ap.add_argument("--split", default="test",
                    choices=["test", "val", "train"],
                    help="用哪份题库 (默认 test = 跨被试)")
    ap.add_argument("--per_task", type=int, default=30,
                    help="每题型抽样题数 (默认 30, 全 9 类共 270 题)")
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="逗号分隔的题型子集, 默认全部 9 类")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in TASKS]
    if bad or not tasks:
        sys.exit(f"题型须为 {TASKS} 的子集, 收到: {tasks}")

    ckpt = os.path.abspath(os.path.join(HERE, "..", args.ckpt))
    if not os.path.isfile(os.path.join(ckpt, "adapter_model.safetensors")):
        sys.exit(f"!! {ckpt} 下无 adapter_model.safetensors")
    out_dir = os.path.abspath(os.path.join(
        HERE, "..", "outputs", "eval",
        os.path.basename(ckpt.rstrip("/")) + "_gen"))
    os.makedirs(out_dir, exist_ok=True)

    # ---- 抽题: 每题型按 (图片, 问题) 排序后等距抽样, 跨运行可复现 ----
    data_file = os.path.join(DATA_ROOT, f"instruct_{args.split}.jsonl")
    if not os.path.isfile(data_file):
        sys.exit(f"!! 缺 {data_file}")
    by_task = defaultdict(list)
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["task"] in tasks:
                by_task[r["task"]].append(r)
    items = []
    for t in tasks:
        rows = sorted(by_task[t],
                      key=lambda r: (r["image"], r["messages"][0]["content"]))
        if not rows:
            print(f"!! {args.split} 题库无 {t} 题, 跳过")
            continue
        n = min(args.per_task, len(rows))
        idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
        items += [(t, rows[i]) for i in sorted(set(idx))]
    if not items:
        sys.exit("!! 无可评估题目")

    plan = " ".join(f"{t}:{min(args.per_task, len(by_task[t]))}"
                    for t in tasks if by_task[t])
    print(f"设备 {DEVICE} | ckpt {args.ckpt} | 题库 instruct_{args.split} "
          f"| 共 {len(items)} 题 ({plan})", flush=True)

    # ---- 加载模型 ----
    proc = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ckpt).to(DEVICE).eval()

    # ---- 逐题生成 + 阅卷 ----
    preds = []
    for i, (task, rec) in enumerate(items):
        img = Image.open(os.path.join(DATA_ROOT, rec["image"])).convert("RGB")
        q = rec["messages"][0]["content"].replace("<image>\n", "")
        ref = rec["messages"][1]["content"]
        batch = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": q}]}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=MAX_NEW[task],
                                 do_sample=False)
        gen = proc.tokenizer.decode(
            out[0][batch["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
        score, extra = GRADE[task](ref, gen)
        row = {"task": task, "image": rec["image"], "q": q,
               "ref": ref, "pred": gen, "score": score}
        row.update(extra)
        preds.append(row)
        print(f"[{i+1}/{len(items)}] {task:<9} {score:>4.2f} | {gen[:60]!r}",
              flush=True)

    # ---- 汇总 ----
    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    tasks_done = [t for t in tasks if any(p["task"] == t for p in preds)]
    summary = {}
    for t in tasks_done:
        tp = [p for p in preds if p["task"] == t]
        entry = {"n": len(tp), "acc": mean([p["score"] for p in tp])}
        if t == "where":
            entry["h_acc"] = mean([p["h"] for p in tp])
            entry["d_acc"] = mean([p["d"] for p in tp])
        elif t == "movement":
            entry["level_acc"] = mean([p["level"] for p in tp])
            entry["num_acc"] = mean([p["num"] for p in tp])
        elif t == "json":
            entry["strict"] = mean([1.0 if p["score"] == 1.0 else 0.0
                                    for p in tp])
            entry["parse_rate"] = mean([p.get("parse", 0) for p in tp])
            entry["fields"] = {k: mean([p["fields"].get(k, 0) for p in tp])
                               for k in JSON_FIELDS}
        elif t == "describe":
            entry["metric"] = "action_hit (宽松参考, 需人审 describe_samples)"
        summary[t] = entry

    old = [t for t in tasks_done if t in ("fine", "coarse")]
    closed_new = [t for t in tasks_done
                 if t not in ("fine", "coarse", "describe")]
    overall = {
        "old_mean_fine_coarse": mean([summary[t]["acc"] for t in old]),
        "new_closed_mean_6tasks": mean([summary[t]["acc"] for t in closed_new]),
        "note": "fine/coarse 为生成式口径, 与 evaluate.py 似然打分 "
                "(E4 test top1=0.275) 不直接可比",
    }

    results = {"ckpt": args.ckpt, "split": args.split,
               "per_task": args.per_task, "n_items": len(preds),
               "tasks": summary, "overall": overall}
    with open(os.path.join(out_dir, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w",
              encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    desc_rows = [p for p in preds if p["task"] == "describe"][:10]
    with open(os.path.join(out_dir, "describe_samples.txt"), "w",
              encoding="utf-8") as f:
        for p in desc_rows:
            f.write(f"== {p['image']}\nQ: {p['q']}\n"
                    f"REF: {p['ref']}\nPRED: {p['pred']}\n\n")

    # ---- 控制台报表 ----
    print("\n" + "=" * 66)
    print(f"{'task':<10}{'n':>4}{'acc':>8}   备注")
    print("-" * 66)
    for t in tasks_done:
        e = summary[t]
        note = ""
        if t == "where":
            note = f"h {e['h_acc']:.2f} | d {e['d_acc']:.2f}"
        elif t == "movement":
            note = f"level {e['level_acc']:.2f} | num {e['num_acc']:.2f}"
        elif t == "json":
            note = (f"strict {e['strict']:.2f} | parse {e['parse_rate']:.2f} "
                    f"| 最弱字段 {min(e['fields'], key=e['fields'].get)}")
        elif t == "describe":
            note = "动作名命中(宽松), 样本见 describe_samples.txt"
        print(f"{t:<10}{e['n']:>4}{e['acc']:>8.3f}   {note}")
    print("-" * 66)
    if overall["old_mean_fine_coarse"] is not None:
        print(f"旧题均值 (fine+coarse, 生成式): {overall['old_mean_fine_coarse']}")
    if overall["new_closed_mean_6tasks"] is not None:
        print(f"新题均值 (where/posture/movement/height/dist/json): "
              f"{overall['new_closed_mean_6tasks']}  <- E5 新能力")
    print(f"结果已写入: {out_dir}")


if __name__ == "__main__":
    main()
