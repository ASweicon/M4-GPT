# -*- coding: utf-8 -*-
"""E3 二分类探针: 一条命令回答"模型到底看不看得见图"。

原理: 取两个视觉差异极大的动作 (Squat=Yes / Jumping up=No 各32张图),
问题固定、答案只有 Yes/No 两个单 token。让模型过拟合这 64 条数据:
  - 视觉通路正常 -> 几十轮后必然背下 (准确率接近 100%)
  - 图像没进 LLM  -> 只能抛硬币 (准确率卡在 50%)
判决不受学习率/数据量/类别数干扰: 记住 image->答案的映射,
物理上必须以"能区分不同图"为前提。

用法:
  python scripts/probe_binary.py                # 生成数据+训练+评测 (约15分钟)
  python scripts/probe_binary.py --skip_train  # 只用现有 adapter 评测
"""
import argparse, json, os, random, shutil, subprocess, sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_ROOT = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "outputs", "lora")
BACKUP = os.path.join(ROOT, "outputs", "lora_v1_300step")
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

ACT_YES, ACT_NO = "Squat", "Jumping up"   # 动作22/27, 蹲下vs起跳, 形态差异极大
PER_CLASS = 32
QUESTION = "Is the person squatting? Answer Yes or No."


def binary_jsonl(split):
    return os.path.join(DATA_ROOT, f"binary_{split}.jsonl")


def make_data(split):
    """从 {split}.jsonl 抽取二分类数据: Squat->Yes, Jumping up->No, 各最多32条.

    固定随机种子, 重复运行生成完全一致的文件 (保证评测的就是训练的那64条).
    split=test 时用未见过的被试 (泛化测试).
    """
    with open(os.path.join(DATA_ROOT, f"{split}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    def pick(name, ans):
        rs = [r for r in rows if r["messages"][1]["content"] == name]
        random.shuffle(rs)
        return [{"image": r["image"],
                 "messages": [
                     {"role": "user", "content": "<image>\n" + QUESTION},
                     {"role": "assistant", "content": ans}]}
                for r in rs[:PER_CLASS]]

    random.seed(0)
    data = pick(ACT_YES, "Yes") + pick(ACT_NO, "No")
    random.shuffle(data)
    out_path = binary_jsonl(split)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_yes = sum(1 for r in data if r["messages"][1]["content"] == "Yes")
    print(f"[probe] wrote {len(data)} records -> {out_path} "
          f"(Yes {n_yes} / No {len(data) - n_yes})", flush=True)


def run_train(epochs, split):
    # 训练会覆盖 outputs/lora, 若尚未备份旧 adapter 则先备份一次
    if os.path.isdir(CKPT) and not os.path.isdir(BACKUP):
        shutil.copytree(CKPT, BACKUP)
        print("[probe] 已备份现有 adapter ->", BACKUP, flush=True)
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--data", binary_jsonl(split), "--epochs", str(epochs)]
    print("[probe] 训练:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def run_eval(split):
    """逐条前向, 比较答案首 token (Yes vs No) 的 logit, 统计准确率."""
    proc = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, CKPT).to(DEVICE).eval()
    yes_tok = proc.tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_tok = proc.tokenizer("No", add_special_tokens=False).input_ids[0]

    with open(binary_jsonl(split), encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    correct = 0
    conf = {"Yes": 0, "No": 0}
    for i, r in enumerate(data):
        img = Image.open(os.path.join(DATA_ROOT, r["image"])).convert("RGB")
        pd = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": QUESTION}]}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        batch = {k: v.to(DEVICE) for k, v in pd.items()
                 if k in ("input_ids", "attention_mask", "pixel_values",
                          "image_grid_thw", "mm_token_type_ids")}
        with torch.no_grad():
            logits = model(**batch).logits
        last = logits[0, -1].float()
        pred = "Yes" if last[yes_tok] > last[no_tok] else "No"
        gt = r["messages"][1]["content"]
        conf[pred] += 1
        correct += int(pred == gt)
        if (i + 1) % 16 == 0 or i + 1 == len(data):
            print(f"[probe] {i+1}/{len(data)} acc {correct/(i+1):.3f}",
                  flush=True)

    acc = correct / len(data)
    print(f"\n[probe] 二分类准确率 ({split} split): {acc:.3f} "
          f"(预测分布 Yes:{conf['Yes']} No:{conf['No']})")
    if split == "test":
        if acc >= 0.8:
            print("[probe] 判决: 泛化出色 —— 未见过的被试上也能区分蹲/跳, E4 前景很好")
        elif acc >= 0.6:
            print("[probe] 判决: 部分泛化 —— 有迁移但不足, "
                  "符合仅64图训练的预期, 靠 E4 扩数据提升")
        else:
            print("[probe] 判决: 泛化失败 —— 记住的是训练被试的指纹而非动作特征; "
                  "E4 需靠数据量与多样性破局")
        return
    if acc >= 0.9:
        print("[probe] 判决: 视觉通路正常 —— 模型背下了64张图, "
              "失败原因是数据量/任务稀释, 可上 E4 正式重训")
    elif acc <= 0.6:
        print("[probe] 判决: 图像没有到达 LLM —— 通路存在 bug, "
              "先停止重训计划, 排查 pixel/mm_token 通路")
    else:
        print("[probe] 判决: 信号微弱 (0.6~0.9) —— "
              "把训练日志和本输出发回来进一步分析")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30,
                    help="过拟合训练轮数 (默认30)")
    ap.add_argument("--skip_train", action="store_true",
                    help="跳过训练, 直接用现有 adapter 评测")
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="从哪个split抽二分类数据 (test=未见过的被试, 测泛化)")
    args = ap.parse_args()

    make_data(args.split)
    if not args.skip_train:
        run_train(args.epochs, args.split)
    run_eval(args.split)


if __name__ == "__main__":
    main()
