#!/usr/bin/env bash
# E5: RF-GPT 式多任务指令续训 —— 从冻结的 E4 adapter 加载, 混合题库防遗忘.
#
#   起点: outputs/e4_lora (E4 成果, 只读, 不覆盖)
#   数据: data/instruct_train.jsonl (fine+coarse rehearsal + 7 类信号新题)
#   产物: outputs/e5_lora
#
# 额外参数透传给 train.py, 例: bash scripts/run_e5.sh --max_steps 1200
# 训完的 fine 抽查用于对照 E4 (监控遗忘); 9 类全量评估用 evaluate_gen.py.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d outputs/e4_lora ]; then
    echo "!! outputs/e4_lora 不存在 —— 先运行: bash scripts/run_e4.sh"; exit 1
fi
if [ ! -f data/instruct_train.jsonl ]; then
    echo "!! data/instruct_train.jsonl 不存在 —— 先运行: python scripts/synthesize_instructions.py"; exit 1
fi

python scripts/train.py --resume outputs/e4_lora \
    --data data/instruct_train.jsonl \
    --lr 2e-5 --max_steps 600 --ckpt outputs/e5_lora "$@"

# fine 快速抽查: 与 E4 同口径 (train top1≈0.40), 显著下滑说明遗忘需调整
python scripts/evaluate.py --ckpt outputs/e5_lora --split train \
    --per_pair 1 --limit 100 --cbatch 5
