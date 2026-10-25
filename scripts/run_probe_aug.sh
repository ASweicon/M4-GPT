#!/usr/bin/env bash
# A 方案一键跑: E4 增强探针 -> E5 增强探针 -> 双头概率集成.
#
#   LIMIT=0   全量 test 约 6000 题 (推荐, 200 题有 ±3pp 抖动)
#   LIMIT=200 快速版 (先验证流程是否通, 约 5 分钟)
#   PAIR=8    每个 (被试,动作) 取 8 张原图 + 8 张镜像 = 每类 16 个训练特征
#
# 依赖: E6 数据重建已完成 (data/images/train 下应有 *flip.png), E4/E5 adapter 在 outputs/
# 用法: bash scripts/run_probe_aug.sh
set -u
cd "$(dirname "$0")/.."
export PATH=/opt/conda/bin:$PATH
mkdir -p outputs

LIMIT=${LIMIT:-0}
PAIR=${PAIR:-8}
SEEDS=${SEEDS:-5}

for ck in outputs/e4_lora outputs/e5_lora; do
  [ -f "$ck/adapter_model.safetensors" ] || { echo "!! 缺 $ck/adapter_model.safetensors"; exit 1; }
done
n_flip=$(ls data/images/train 2>/dev/null | grep -c 'flip\.png$')
echo "[preflight] flip train images = $n_flip"
[ "${n_flip:-0}" -gt 0 ] || { echo "!! data/images/train 里没有 *flip.png —— 先跑 E6 数据重建"; exit 1; }

echo "=== [1/3] E4 增强探针 (limit=$LIMIT pair=$PAIR seeds=$SEEDS) ==="
python scripts/probe_aug.py --ckpt outputs/e4_lora --tag e4_aug \
  --limit "$LIMIT" --train_per_pair "$PAIR" --seeds "$SEEDS" 2>&1 | tee outputs/a_e4.log
echo "E4_EXIT=${PIPESTATUS[0]}"

echo "=== [2/3] E5 增强探针 ==="
python scripts/probe_aug.py --ckpt outputs/e5_lora --tag e5_aug \
  --limit "$LIMIT" --train_per_pair "$PAIR" --seeds "$SEEDS" 2>&1 | tee outputs/a_e5.log
echo "E5_EXIT=${PIPESTATUS[0]}"

echo "=== [3/3] 集成 (概率平均 + 特征拼接) ==="
python scripts/probe_ensemble.py --tags e4_aug e5_aug \
  --train_per_pair "$PAIR" --limit "$LIMIT" --seeds "$SEEDS" 2>&1 | tee outputs/a_ens.log
echo "ENS_EXIT=${PIPESTATUS[0]}"

grep -h 'AUG_RESULT\|ENSEMBLE_PROB\|ENSEMBLE_CONCAT\|ENSEMBLE_RESULT' \
  outputs/a_e4.log outputs/a_e5.log outputs/a_ens.log
echo "A_DONE"
