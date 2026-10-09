# -*- coding: utf-8 -*-
"""Batch generative check: one model load, one image per action, GT vs generated."""
import argparse, json, os, sys, time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CKPT = os.path.join(ROOT, "outputs", "lora")
MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--base_only", action="store_true")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if not args.base_only:
        base = PeftModel.from_pretrained(base, CKPT)
    model = base.to("cuda").eval()

    rows = [json.loads(l) for l in open(
        os.path.join(ROOT, "data", "test.jsonl"), encoding="utf-8")]
    seen, picks = set(), []
    for r in rows:
        a = int(r["image"].split("_a")[1].split("_")[0])
        if a not in seen:
            seen.add(a); picks.append((a, r))
    picks = picks[:args.n]

    correct = 0
    for a, r in picks:
        img = Image.open(os.path.join(ROOT, "data", r["image"])).convert("RGB")
        q = r["messages"][0]["content"].replace("<image>\n", "").strip()
        gt = r["messages"][1]["content"].strip()
        d = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": q}]}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**d, max_new_tokens=20, do_sample=False)
        dt = time.time() - t0
        pred = proc.tokenizer.decode(
            out[0][d["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = pred.lower() == gt.lower()
        correct += ok
        print(f"a{a:02d} {'OK ' if ok else 'XX '} gt='{gt}' pred='{pred}' {dt:.1f}s",
              flush=True)
    print(f"exact-match {correct}/{len(picks)} = {correct/len(picks):.1%}",
          flush=True)


if __name__ == "__main__":
    main()
