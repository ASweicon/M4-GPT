#!/usr/bin/env bash
# E5: RF-GPT 式多任务指令续训 —— 从冻结的 E4 adapter 加载, 混合题库防遗忘.
#
#   起点: outputs/e4_lora (E4 成果, 只读, 不会被覆盖)
#   数据: data/instruct_train.jsonl (fine+coarse rehearsal + 7 类信号新题)
#   产物: outputs/e5_lora
#   训完自动 fine 抽查, 对照 E4 (train top1≈0.40) 监控遗忘;
#   9 类全量评估用 evaluate_gen.py.
#
# 用法: bash scripts/run_e5.sh [mode] [train.py 额外参数...]
#   mode: skip   跳过训练, 只做 fine 抽查 (E5 已完成时用)
#         resume 从 E5 现有 checkpoint 断点续训 (中断接续)
#         fresh  从 E4 重新开始 E5 (删除现有 E5 进度, 需输入 yes 确认)
#   不带 mode 时: 先报告产物状态 (有无 outputs/e5_lora), 再询问如何继续;
#               无论有无产物都由用户明确选择, 不默认直接开训;
#               非交互环境提示后退出, 需显式指定 mode.
#   DRY=1 时只打印将执行的命令, 不实际运行 (预览用).
#   例: bash scripts/run_e5.sh fresh --max_steps 1200
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
E4=outputs/e4_lora
E5=outputs/e5_lora

run() { echo "+ $*"; [ "${DRY:-0}" = "1" ] && return 0; "$@"; }

[ -f "$E4/adapter_model.safetensors" ] || {
    echo "!! 缺 $E4 —— 先运行: bash scripts/run_e4.sh"; exit 1; }
[ -f data/instruct_train.jsonl ] || {
    echo "!! 缺 data/instruct_train.jsonl —— 先运行: python scripts/synthesize_instructions.py"; exit 1; }

MODE="auto"
if [ $# -gt 0 ]; then
    case "$1" in skip|resume|fresh) MODE="$1"; shift;; esac
fi

if [ "$MODE" = "auto" ]; then
    if [ -d "$E5" ]; then
        echo "检测到 $E5 已存在 —— E5 可能已有训练进度."
        REPLY=""
        if [ -t 0 ]; then
            echo "  [1] skip   已完成, 跳过训练只做抽查"
            echo "  [2] resume 从 E5 现有 checkpoint 断点续训"
            echo "  [3] fresh  从 E4 重开 E5 (删除现有进度, 需确认)"
            echo "  [q] 退出"
            read -r -p "选择 [1/2/3/q]: " REPLY || REPLY=q
        else
            echo "  非交互环境无法弹出选择, 请显式指定后重跑:"
            echo "    bash scripts/run_e5.sh skip    # E5 已完成, 跳过训练只抽查"
            echo "    bash scripts/run_e5.sh resume  # 从 E5 现有 checkpoint 断点续训"
            echo "    bash scripts/run_e5.sh fresh   # 从 E4 重开 E5 (将再次要求确认)"
            exit 1
        fi
        case "$REPLY" in
            1) MODE=skip;;  2) MODE=resume;;  3) MODE=fresh;;
            *) echo "退出"; exit 0;;
        esac
    else
        echo "未检测到 $E5 —— 当前无 E5 训练进度 (起点: 冻结的 $E4)."
        REPLY=""
        if [ -t 0 ]; then
            echo "  [1] fresh 从 E4 起点开始 E5"
            echo "  [q] 退出"
            read -r -p "选择 [1/q]: " REPLY || REPLY=q
        else
            echo "  非交互环境无法弹出选择, 请显式指定后重跑:"
            echo "    bash scripts/run_e5.sh fresh   # 从 E4 起点开始 E5"
            exit 1
        fi
        case "$REPLY" in
            1) MODE=fresh;;  *) echo "退出"; exit 0;;
        esac
    fi
fi

# 显式指定但产物状态对不上时提前拦截, 不让 train.py 报晦涩错误
if [ "$MODE" = "resume" ] && [ ! -d "$E5" ]; then
    echo "!! $MODE 需要 $E5 存在, 但当前无 E5 进度"; exit 1
fi

EXTRA=(--data data/instruct_train.jsonl)
case "$MODE" in
    skip)
        [ -f "$E5/adapter_model.safetensors" ] || {
            echo "!! $E5 下无 adapter 权重, 不能跳过训练"; exit 1; }
        echo "== skip: E5 视为已完成, 仅 fine 抽查 =="
        EXTRA=();;
    resume)
        echo "== resume: 从 $E5 断点续训 =="
        EXTRA+=(--resume "$E5");;
    fresh)
        if [ -d "$E5" ]; then
            REPLY=""
            read -r -p "确认删除 $E5 并从 E4 重开? 输入 yes: " REPLY || REPLY=""
            [ "$REPLY" = "yes" ] || { echo "已取消"; exit 1; }
            rm -rf "$E5"
        fi
        echo "== fresh: 从 $E4 起点开始 E5 =="
        EXTRA+=(--resume "$E4");;
esac

if [ "$MODE" != "skip" ]; then
    EXTRA+=(--lr 2e-5 --max_steps 600 --ckpt "$E5")
    run python scripts/train.py "${EXTRA[@]}" "$@"
fi
run python scripts/evaluate.py --ckpt "$E5" --split train \
    --per_pair 1 --limit 100 --cbatch 5
