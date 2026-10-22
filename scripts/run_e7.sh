#!/usr/bin/env bash
# E7 一键跑: 多轮追问 + 解释型 QA 数据 -> 从 E5 续训 -> 多轮/单轮终评.
#
#   E7 学什么: RF-GPT 式的"理解 + 追问", 不是新分类器
#     1) 数据: synthesize_chat.py 把同一张图的可观测事实串成 2~4 轮对话
#        (动作 -> 你怎么看出来的 -> 定量 -> 镜像反事实)
#     2) 训练: 从 outputs/e5_lora 续训 (不覆盖), 数据 = 原 instruct + chat
#     3) 评测: 每 400 步快照各评一次 (轮次曲线) + 终评 9 题型 + 似然分类
#
# 用法:
#   bash scripts/run_e7.sh                 # 数据+训练+评测 全流程
#   SKIP_DATA=1 bash scripts/run_e7.sh     # 复用已有 chat_*.jsonl
#   SKIP_TRAIN=1 bash scripts/run_e7.sh    # 只评测 outputs/e7_lora
#   STEPS=1200 bash scripts/run_e7.sh      # 改训练步数
#   CURVE=0 bash scripts/run_e7.sh         # 不评中间快照 (省时间)
set -u
cd "$(dirname "$0")/.."
export PATH=/opt/conda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs data

E5=outputs/e5_lora
E7=outputs/e7_lora
STEPS=${STEPS:-1600}
CHAT_PER=${CHAT_PER:-3}
LIMIT=${LIMIT:-300}          # 终评对话数
CURVE=${CURVE:-1}
PY=${PY:-python}

[ -f "$E5/adapter_model.safetensors" ] || {
    echo "!! 缺 $E5/adapter_model.safetensors —— E7 起点是 E5 权重"; exit 1; }
[ -f data/captions.jsonl ] || {
    echo "!! 缺 data/captions.jsonl —— 先跑 E6 数据重建"; exit 1; }

# ---------- 1. 数据 ----------
if [ "${SKIP_DATA:-0}" != "1" ]; then
    echo "== [1/3] 合成多轮对话数据 (per_image=$CHAT_PER)"
    $PY scripts/synthesize_chat.py --per_image "$CHAT_PER" | tee outputs/e7_chat_data.log
    [ -s data/chat_train.jsonl ] || { echo "!! chat_train.jsonl 为空"; exit 1; }

    echo "== 混合 E7 训练集 (原 instruct + chat)"
    cat data/instruct_train.jsonl data/chat_train.jsonl > data/e7_train.jsonl
    echo "   instruct $(wc -l < data/instruct_train.jsonl) + chat $(wc -l < data/chat_train.jsonl) => $(wc -l < data/e7_train.jsonl)"
else
    echo "== [1/3] SKIP_DATA=1, 复用 data/e7_train.jsonl"
    [ -s data/e7_train.jsonl ] || { echo "!! data/e7_train.jsonl 不存在"; exit 1; }
fi

# ---------- 2. 训练 ----------
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "== [2/3] E7 续训 $STEPS 步 (起点 $E5, 每 400 步存快照)"
    $PY scripts/train.py --resume "$E5" --data data/e7_train.jsonl \
        --lr 2e-5 --max_steps "$STEPS" --ckpt "$E7" --save_every 400 \
        | tee outputs/e7_train.log
else
    echo "== [2/3] SKIP_TRAIN=1"
fi
[ -f "$E7/adapter_model.safetensors" ] || {
    echo "!! 缺 $E7/adapter_model.safetensors"; exit 1; }

# ---------- 3. 评测 ----------
echo "== [3/3] 评测"
# 3a. 轮次曲线: 每个快照抽少量对话
if [ "$CURVE" = "1" ]; then
    for snap in "$E7"/step*; do
        [ -d "$snap" ] || continue
        echo "-- 曲线: $(basename "$snap")"
        $PY scripts/evaluate_chat.py --ckpt "$snap" --limit 60 \
            >> outputs/e7_curve.log 2>&1 || echo "!! $(basename "$snap") 评测失败"
    done
    echo "   曲线日志 -> outputs/e7_curve.log"
fi

# 3b. 终评: 多轮对话 (含 50/39/粗类 + 平衡准确率 + 一致性)
echo "-- 终评: 多轮对话 (limit=$LIMIT)"
$PY scripts/evaluate_chat.py --ckpt "$E7" --limit "$LIMIT" --demo 2 \
    | tee outputs/e7_chat_eval.log

# 3c. 9 题型单轮 (与 E5 直接可比)
echo "-- 终评: 9 题型 x30"
$PY scripts/evaluate_gen.py --ckpt "$E7" --split test --per_task 30 \
    | tee outputs/e7_gen_eval.log

# 3d. 似然口径 fine 200 (与 E4/E5 直接可比)
echo "-- 终评: 似然 fine 200"
$PY scripts/evaluate.py --ckpt "$E7" --split test \
    --per_pair 1 --limit 200 --cbatch 5 | tee outputs/e7_lik_eval.log

echo ""
echo "== E7 完成"
echo "  多轮: outputs/e7_chat_eval.log"
echo "  9题型: outputs/e7_gen_eval.log"
echo "  似然: outputs/e7_lik_eval.log"
[ "$CURVE" = "1" ] && echo "  轮次曲线: outputs/e7_curve.log"
echo "  adapter: $E7 (记得 backup 回本地)"
