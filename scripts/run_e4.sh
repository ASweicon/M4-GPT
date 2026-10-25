#!/usr/bin/env bash
# E4: 50 类细粒度动作分类 LoRA 训练 (从零挂新 LoRA) + 训练集快速抽查.
#
# 产物: outputs/e4_lora —— 训完即"冻结", 供 E5 以 --resume 续训, 不再改动.
# 参考进度: step 3200 时 train top1≈0.40 / top5≈0.64 (RTX 4090, ~4h).
#
# 用法: bash scripts/run_e4.sh [mode] [train.py 额外参数...]
#   mode: skip   跳过训练, 只做抽查评估 (E4 已完成/已冻结时用)
#         resume 从现有 checkpoint 断点续训 (中断接续)
#         fresh  删除现有成果从零重训 (需输入 yes 确认)
#   不带 mode 时: 先报告产物状态 (有无 outputs/e4_lora), 再询问如何继续;
#               无论有无产物都由用户明确选择, 不默认直接开训;
#               非交互环境提示后退出, 需显式指定 mode.
#   DRY=1 时只打印将执行的命令, 不实际运行 (预览用).
#   额外参数透传给 train.py, 例: bash scripts/run_e4.sh resume --max_steps 3200
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
E4=outputs/e4_lora

run() { echo "+ $*"; [ "${DRY:-0}" = "1" ] && return 0; "$@"; }

MODE="auto"
if [ $# -gt 0 ]; then
    case "$1" in skip|resume|fresh) MODE="$1"; shift;; esac
fi

if [ "$MODE" = "auto" ]; then
    if [ -d "$E4" ]; then
        echo "检测到 $E4 已存在 —— E4 可能已训练完成 (冻结起点)."
        REPLY=""
        if [ -t 0 ]; then
            echo "  [1] skip   已完成, 跳过训练只做抽查评估"
            echo "  [2] resume 断点续训 (从现有 checkpoint 继续)"
            echo "  [3] fresh  删除成果从零重训 (危险, 需确认)"
            echo "  [q] 退出 (E4 已冻结想开 E5 请直接: bash scripts/run_e5.sh)"
            read -r -p "选择 [1/2/3/q]: " REPLY || REPLY=q
        else
            echo "  非交互环境无法弹出选择, 请显式指定后重跑:"
            echo "    bash scripts/run_e4.sh skip    # E4 已完成, 跳过训练只抽查"
            echo "    bash scripts/run_e4.sh resume  # 从现有 checkpoint 断点续训"
            echo "    bash scripts/run_e4.sh fresh   # 删除成果从零重训 (将再次要求确认)"
            exit 1
        fi
        case "$REPLY" in
            1) MODE=skip;;  2) MODE=resume;;  3) MODE=fresh;;
            *) echo "退出"; exit 0;;
        esac
    else
        echo "未检测到 $E4 —— 当前无已训练产物."
        REPLY=""
        if [ -t 0 ]; then
            echo "  [1] fresh 开始从零训练"
            echo "  [q] 退出"
            read -r -p "选择 [1/q]: " REPLY || REPLY=q
        else
            echo "  非交互环境无法弹出选择, 请显式指定后重跑:"
            echo "    bash scripts/run_e4.sh fresh   # 从零开始训练"
            exit 1
        fi
        case "$REPLY" in
            1) MODE=fresh;;  *) echo "退出"; exit 0;;
        esac
    fi
fi

# 显式指定但产物状态对不上时提前拦截, 不让 train.py 报晦涩错误
if [ "$MODE" = "resume" ] && [ ! -d "$E4" ]; then
    echo "!! $MODE 需要 $E4 存在, 但当前无产物"; exit 1
fi

EXTRA=()
case "$MODE" in
    skip)
        [ -f "$E4/adapter_model.safetensors" ] || {
            echo "!! $E4 下无 adapter 权重, 不能跳过训练"; exit 1; }
        echo "== skip: E4 视为已完成, 仅抽查评估 ==";;
    resume)
        echo "== resume: 从 $E4 断点续训 =="
        EXTRA+=(--resume "$E4");;
    fresh)
        if [ -d "$E4" ]; then
            REPLY=""
            read -r -p "确认删除 $E4 并从零重训? 输入 yes: " REPLY || REPLY=""
            [ "$REPLY" = "yes" ] || { echo "已取消"; exit 1; }
            rm -rf "$E4"
        fi
        echo "== fresh: 从零开始 E4 ==";;
esac

if [ "$MODE" != "skip" ]; then
    run python scripts/train.py --task fine --lr 1e-4 --ckpt "$E4" "${EXTRA[@]}" "$@"
fi
run python scripts/evaluate.py --ckpt "$E4" --split train \
    --per_pair 1 --limit 100 --cbatch 5
