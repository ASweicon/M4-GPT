# -*- coding: utf-8 -*-
"""LoRA SFT of Qwen2-VL-2B on rendered radar point clouds (answer-only loss).

RTX 4090 优化版:
  - batch_size=4, accum=2 (等效 batch=8, 但 4 路并行更快)
  - 视觉塔也挂 LoRA (qkv/attn.proj/fc1/fc2), 让 ViT 学雷达图特征
  - warmup(3%) + cosine 衰减, 梯度裁剪 1.0, 训练更稳
  - DataLoader 4 workers + pin_memory, 数据加载不阻塞 GPU
  - TF32 矩阵乘加速 (Ada 架构支持)
"""
import argparse, json, os, random, sys, time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from PIL import Image
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from m4config import ACTION_NAMES

DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "data"))
CKPT = os.path.abspath(os.path.join(HERE, "..", "outputs", "lora"))
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

# TF32 加速 (Ampere/Ada 架构, CUDA only)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_jsonl(split):
    with open(os.path.join(DATA_ROOT, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class M4Dataset(Dataset):
    def __init__(self, split, proc):
        self.rows = load_jsonl(split)
        self.proc = proc

    def __len__(self):
        return len(self.rows)

    def encode(self, row):
        img = Image.open(os.path.join(DATA_ROOT, row["image"])).convert("RGB")
        q = row["messages"][0]["content"].replace("<image>\n", "")
        a = row["messages"][1]["content"]
        prompt = [{"role": "user",
                   "content": [{"type": "image", "image": img},
                               {"type": "text", "text": q}]}]
        full = prompt + [{"role": "assistant",
                          "content": [{"type": "text", "text": a}]}]
        pk = self.proc.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        fk = self.proc.apply_chat_template(
            full, tokenize=True, return_dict=True, return_tensors="pt")
        input_ids = fk["input_ids"][0]
        labels = input_ids.clone()
        labels[: pk["input_ids"].shape[1]] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": fk["attention_mask"][0],
            "mm_token_type_ids": fk["mm_token_type_ids"][0],
            "pixel_values": fk["pixel_values"],
            "image_grid_thw": fk["image_grid_thw"],
        }

    def __getitem__(self, i):
        return self.encode(self.rows[i])


def collate(batch, pad_id):
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    input_ids, labels, attn, mm = [], [], [], []
    for b in batch:
        pad = maxlen - b["input_ids"].shape[0]
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100)]))
        attn.append(torch.cat([b["attention_mask"], torch.zeros(pad)]))
        mm.append(torch.cat([b["mm_token_type_ids"], torch.zeros(pad,
                     dtype=b["mm_token_type_ids"].dtype)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn).long(),
        "mm_token_type_ids": torch.stack(mm).long(),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_gc", action="store_true",
                    help="disable gradient checkpointing (faster, more VRAM)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--task", default="all", choices=["all", "fine"],
                    help="all: 3种问答混合; fine: 只用细粒度动作名问答")
    ap.add_argument("--data", default="",
                    help="直接指定训练用jsonl路径 (覆盖train split, 跳过--task过滤)")
    ap.add_argument("--resume", default="",
                    help="从已有 adapter 目录继续训练 (如 outputs/lora); 默认挂全新 LoRA")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL)
    if args.debug:
        ds = M4Dataset("train", proc)
        ex = ds[0]
        print("input_ids", ex["input_ids"].shape)
        print("pixel_values", ex["pixel_values"].shape)
        print("image_grid_thw", ex["image_grid_thw"])
        supervised = ex["input_ids"][ex["labels"] != -100]
        print("answer tokens:", proc.tokenizer.decode(supervised))
        return

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.requires_grad_(False)
    model.config.use_cache = False
    if not args.no_gc:
        model.gradient_checkpointing_enable()
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            # 语言模型层 (7 个)
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # 视觉塔层 (4 个) —— 关键: 让 ViT 也学雷达图特征
            "qkv",         # visual.blocks.N.attn.qkv
            "attn.proj",   # visual.blocks.N.attn.proj (不匹配 self_attn.o_proj)
            "fc1",         # visual.blocks.N.mlp.fc1
            "fc2",         # visual.blocks.N.mlp.fc2
        ])
    if args.resume:
        # 阶段衔接/断点续训: 加载已训好的 LoRA 权重, 继续在新数据上训练
        # (r/target_modules 等 LoRA 结构自动读取自 adapter_config.json)
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"resume: 从 {args.resume} 加载 LoRA 权重继续训练", flush=True)
    else:
        model = get_peft_model(model, lcfg)
    model.to(DEVICE)
    model.print_trainable_parameters()

    train_ds = M4Dataset("train", proc)
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            train_ds.rows = [json.loads(l) for l in f]
        print(f"data override: {args.data} -> {len(train_ds.rows)} records",
              flush=True)
    elif args.task == "fine":
        names = set(ACTION_NAMES.values())
        before = len(train_ds.rows)
        train_ds.rows = [r for r in train_ds.rows
                         if r["messages"][1]["content"] in names]
        print(f"task=fine: {before} -> {len(train_ds.rows)} records",
              flush=True)
    if args.max_train:
        train_ds.rows = train_ds.rows[:args.max_train]
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, proc.tokenizer.pad_token_id),
                    num_workers=args.workers, pin_memory=(DEVICE == "cuda"),
                    drop_last=True)

    steps_target = round(args.epochs * len(dl) / args.accum)
    if args.max_steps:
        steps_target = min(steps_target, args.max_steps)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, int(args.warmup * steps_target)),
        num_training_steps=steps_target)
    os.makedirs(CKPT, exist_ok=True)
    model.train()
    step, micro, t0 = 0, 0, time.time()
    running = 0.0
    it = iter(dl)
    while step < steps_target:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        (out.loss / args.accum).backward()
        running += out.loss.item()
        micro += 1
        if micro % args.accum == 0:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); scheduler.step(); opt.zero_grad(); step += 1
            if step % 5 == 0:
                avg = running / (args.accum * 5)
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {step}/{steps_target} loss {avg:.4f} "
                      f"lr {lr_now:.2e} {(time.time()-t0)/step:.1f}s/step",
                      flush=True)
                running = 0.0
            if step % 100 == 0:
                model.save_pretrained(CKPT); proc.save_pretrained(CKPT)
    model.save_pretrained(CKPT)
    proc.save_pretrained(CKPT)
    print("SAVED to", CKPT, flush=True)


if __name__ == "__main__":
    main()
