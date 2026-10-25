#!/usr/bin/env bash
# E6 一条龙: 数据重建 -> E4 从零重训 -> E5 续训 -> test 终评.
#
# 背景: E5 终评诊断出 fine 瓶颈 (镜像混淆 61% + 时序信息不足), E6 改渲染管线
# (质心轨迹图) + train 左右翻转增强, 数据全变, 故 E4/E5 必须从零重训.
#
# 步数对齐: train 图片翻倍 (flip), E4 max_steps 3200->6400, E5 600->1200,
# 保持每样本曝光次数与 E4/E5 一致 (控制变量, 本轮唯一变量是数据层).
#
# 用法 (建议 tmux 内): bash scripts/run_e6_all.sh [all|data|train]
#   all   = data + train (默认)
#   data  = 只重建数据 (run_e6_data.sh full)
#   train = 只训练评估 (假定数据已就绪)
#   PY=/opt/conda/bin/python bash scripts/run_e6_all.sh all
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PY:-python}
MODE="${1:-all}"
case "$MODE" in
    all|data|train) ;;
    *) echo "用法: bash scripts/run_e6_all.sh [all|data|train]"; exit 1;;
esac

if [ "$MODE" = "all" ] || [ "$MODE" = "data" ]; then
    bash scripts/run_e6_data.sh smoke
    bash scripts/run_e6_data.sh full
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "train" ]; then
    [ -f data/instruct_train.jsonl ] || {
        echo "!! 缺 data/instruct_train.jsonl —— 先跑: bash scripts/run_e6_all.sh data"; exit 1; }

    echo "== [E4] 从零重训 (新数据, 6400 步) =="
    echo yes | bash scripts/run_e4.sh fresh --max_steps 6400

    echo "== [E5] 多任务续训 (1200 步) =="
    bash scripts/run_e5.sh fresh --max_steps 1200

    echo "== [EVAL] E5 test 集跨被试终评 =="
    bash scripts/run_e5.sh eval
fi

echo "== E6_ALL_DONE =="
