#!/usr/bin/env bash
# E6 数据重建: 新渲染管线 (第4格质心轨迹图) + train 左右翻转增强 (x 镜像 + swap_lr).
#
# 前置: radar_pc.lmdb 已在项目根 (新容器从本地上传), 代码已 git clone/pull.
# 产物: data/images/{train,val,test}/*.png (train 翻倍含 flip) +
#       data/{train,val,test}.jsonl + data/captions.jsonl + data/instruct_*.jsonl
# 注意: 渲染公式变了 (轨迹图), 三个 split 全部重渲染, 与旧图不可混用.
#
# 用法: bash scripts/run_e6_data.sh [mode]
#   mode: smoke  端到端门禁: 每 split 渲染 2-4 张 + caption/instruct 限量冒烟
#         full   全量重建: train 16 chunk 并行渲染(flip 翻倍) + val/test,
#                merge 后重算 captions 与 instruct 题库 (约 1-2h, CPU 密集)
#   PY=/opt/conda/bin/python bash scripts/run_e6_data.sh full   # 指定解释器
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PY:-python}
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
PAR=$(( NPROC > 2 ? NPROC - 2 : 1 ))

MODE="${1:-}"
case "$MODE" in
    smoke|full) ;;
    *) echo "用法: bash scripts/run_e6_data.sh [smoke|full]"; exit 1;;
esac

[ -f radar_pc.lmdb ] || {
    echo "!! 缺 radar_pc.lmdb —— 从本地上传: rsync -e 'ssh -p <端口>' -avzP radar_pc.lmdb root@<host>:/root/M4-GPT/"; exit 1; }

if [ "$MODE" = "smoke" ]; then
    echo "== smoke: 端到端门禁 (渲染/轨迹图/flip/caption/instruct) =="
    "$PY" scripts/build_dataset.py --mode smoke --split train,val,test
    "$PY" scripts/make_caption.py --limit 3
    "$PY" scripts/synthesize_instructions.py --limit 2
    echo "== smoke OK — 确认无报错后运行: bash scripts/run_e6_data.sh full =="
    exit 0
fi

echo "== full: 全量重建 (并行度 $PAR) =="

echo "-- [1/5] train 渲染 (16 subjects x flip 翻倍, ${PAR} 并行) --"
seq 16 | xargs -P "$PAR" -I{} "$PY" scripts/build_dataset.py \
    --mode full --split train --chunk {}
"$PY" scripts/build_dataset.py --merge train

echo "-- [2/5] test 渲染 (4 subjects, 不翻转) --"
seq 4 | xargs -P "$PAR" -I{} "$PY" scripts/build_dataset.py \
    --mode full --split test --chunk {}
"$PY" scripts/build_dataset.py --merge test

echo "-- [3/5] val 渲染 (1 subject) --"
"$PY" scripts/build_dataset.py --mode full --split val

echo "-- [4/5] captions 重算 (含 flip 记录) --"
"$PY" scripts/make_caption.py

echo "-- [5/5] instruct 题库重合成 --"
"$PY" scripts/synthesize_instructions.py

echo "== E6_DATA_DONE =="
