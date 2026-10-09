# -*- coding: utf-8 -*-
"""Benchmark CPU encode vs GPU fwd/bwd vs vision tower cost."""
import os, sys, time
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(__file__))
from train import M4Dataset, MODEL, collate

N = 8
proc = AutoProcessor.from_pretrained(MODEL)
ds = M4Dataset("train", proc)

# ---- CPU encode timing ----
enc_times, samples = [], []
for i in range(N):
    t0 = time.time()
    samples.append(ds[i])
    enc_times.append(time.time() - t0)
print(f"CPU encode: mean {sum(enc_times)/N:.2f}s  {[f'{x:.2f}' for x in enc_times]}",
      flush=True)

# ---- model (same as train) ----
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.requires_grad_(False)
model.config.use_cache = False
model.gradient_checkpointing_enable()
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type=TaskType.CAUSAL_LM,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg).to("cuda")
model.train()
pad = proc.tokenizer.pad_token_id

# ---- vision tower only ----
visual = model.get_base_model().model.visual
vt = []
for ex in samples:
    pv = ex["pixel_values"].to("cuda"); grid = ex["image_grid_thw"].to("cuda")
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        visual(pv, grid)
    torch.cuda.synchronize(); vt.append(time.time() - t0)
print(f"VISUAL fwd only: mean {sum(vt)/N:.3f}s {[f'{x:.3f}' for x in vt]}",
      flush=True)

# ---- full fwd + bwd ----
ft = []
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
for ex in samples:
    b = {k: v.unsqueeze(0).to("cuda") if k in
         ("input_ids", "labels", "attention_mask", "mm_token_type_ids")
         else v.to("cuda") for k, v in ex.items()}
    torch.cuda.synchronize(); t0 = time.time()
    out = model(**b)
    out.loss.backward()
    torch.cuda.synchronize(); ft.append(time.time() - t0)
print(f"FULL fwd+bwd: mean {sum(ft)/N:.3f}s {[f'{x:.3f}' for x in ft]}",
      flush=True)
print("peak MB", torch.cuda.max_memory_allocated() / 1e6, flush=True)
