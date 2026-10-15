#!/usr/bin/env bash
# E4: 50 类细粒度动作分类 LoRA 训练 (从零挂新 LoRA) + 训练集快速抽查.
#
# 产物: outputs/e4_lora —— 训完即"冻结", 供 E5 以 --resume 续训, 不再改动.
# 参考进度: step 3200 时 train top1≈0.40 / top5≈0.64 (RTX 4090, ~4h).
# 断点续训: 重跑本脚本 + 追加 --resume outputs/e4_lora 即可接续.
# 额外参数透传给 train.py, 例: bash scripts/run_e4.sh --max_steps 3200
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python scripts/train.py --task fine --lr 1e-4 --ckpt outputs/e4_lora "$@"

# 快速抽查 (100 张训练图, 与 E4 基线口径一致; 完整评估见 README)
python scripts/evaluate.py --ckpt outputs/e4_lora --split train \
    --per_pair 1 --limit 100 --cbatch 5
