# M4-GPT: 教大模型看懂毫米波雷达点云

基于 Qwen2-VL-2B-Instruct + LoRA 的多任务指令微调: 让大模型理解毫米波雷达点云的
三视图渲染图 (BEV/Front/Side + 4 帧时序着色), 不仅能识别 50 类动作 (分类探针),
还能生成式地回答关于信号本身的问题 (位置/姿态/运动量/高度估读/结构化 JSON).

路线参考 RF-GPT: 确定性 captioning → 指令合成 → 多任务 SFT.

## 环境

- Python 3.11 + PyTorch + transformers (最新, Qwen2-VL 处理器需要) + peft
- 模型: `Qwen/Qwen2-VL-2B-Instruct` (国内加速: `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`)
- 显存: 24GB 可训练 (LoRA r=16); 评估需 `--cbatch 5`
- 数据: `radar_pc.lmdb` (M4Human 点云) 放项目根目录

## 数据准备 (一次性)

`data/*.jsonl` 题库已在仓库中; **渲染图片**被 .gitignore 屏蔽, 需本地重建:

```bash
# full 模式分被试渲染: train 16 个 chunk, val 1 个, test 4 个 (并行度≤4)
python scripts/build_dataset.py --mode full --split train --chunk 1   # 1..16
python scripts/build_dataset.py --mode full --split val   --chunk 1
python scripts/build_dataset.py --mode full --split test  --chunk 1   # 1..4
python scripts/build_dataset.py --merge train
python scripts/build_dataset.py --merge val
python scripts/build_dataset.py --merge test
```

多任务题库若需重建 (可选, 仓库已含):

```bash
python scripts/make_caption.py            # 事实卡片 -> data/captions.jsonl
python scripts/synthesize_instructions.py # 多任务题库 -> data/instruct_{train,val,test}.jsonl
```

## 实验管线: E4 → E5

两个阶段用**同一个** `scripts/train.py`, 差别只在参数 (单变量可对比);
checkpoint 各自独立目录, E5 从冻结的 E4 出发续训, 不会覆盖 E4:

```bash
# 1) E4: 50 类分类训练, 产物冻结在 outputs/e4_lora
#    (每 100 步自动保存; Ctrl+C 中断后加 --resume outputs/e4_lora 续训)
bash scripts/run_e4.sh

# 2) E5: 从 E4 起点多任务指令续训, 新 adapter 写 outputs/e5_lora
bash scripts/run_e5.sh
```

| 阶段 | 数据 | 起点 | lr | 参考结果 |
|---|---|---|---|---|
| E4 | data/train.jsonl (fine 50 类) | 零 | 1e-4 | step 3200: train top1 0.40 / top5 0.64 |
| E5 | data/instruct_train.jsonl (9 类任务族) | E4 adapter | 2e-5 | 进行中 |

E5 题库每图 5 问答: fine + coarse 必出 (rehearsal 防遗忘), 另抽 3 类:
where (位置) / posture (站坐) / movement (运动量三档) / height (高度估读) /
dist (位移估读) / describe (生成式描述) / json (结构化输出).
新题答案全部由点云数值特征确定性计算, 自动阅卷, 无人工标注.

## 评估

```bash
# 50 类细粒度 top1/top5 (结果写入 outputs/eval/<ckpt名>/)
python scripts/evaluate.py --ckpt outputs/e4_lora --split test --cbatch 5
# 3 类粗粒度
python scripts/evaluate.py --ckpt outputs/e5_lora --split test --task coarse --cbatch 5
```

`--split test` 为跨被试泛化判决 (被试 5/8/10/19), `--split train` 用于遗忘监控.

## 实验记录 (main 分支)

| 实验 | 内容 | 结果 |
|---|---|---|
| E0 | 零样本直接问 | top1 0.02 |
| E1 | 小数据微调 (mini) | top1 0.035 |
| E2 | 3 类粗分类训练 | top1 0.30 |
| E3 | 二分类探针 (视觉通路验证) | 1.000 |
| E4 | full 50 类训练 | train top1 0.40 / top5 0.64 (step 3200) |
| E5 | 多任务指令续训 (feature 分支) | 进行中 |
