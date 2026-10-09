# -*- coding: utf-8 -*-
import os, sys
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

DATA = r"C:\Users\Congbing\Doubao\chats\2026-10-03\new-chat-1\data"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
m = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16).to("cuda")
visual = m.model.visual

fn = sorted(os.listdir(os.path.join(DATA, "images", "test")))[0]
img = Image.open(os.path.join(DATA, "images", "test", fn)).convert("RGB")
pd = proc.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What action?"}]}],
    add_generation_prompt=True, tokenize=True, return_dict=True,
    return_tensors="pt")
out = visual(pd["pixel_values"].to("cuda"), pd["image_grid_thw"].to("cuda"))
print("TYPE", type(out).__name__)
if hasattr(out, "keys"):
    for k in out.keys():
        v = out[k]
        print(k, None if v is None else getattr(v, "shape", v))
else:
    print(out.shape)
