# -*- coding: utf-8 -*-
"""Local inference: rendered image OR live (s,a,f) clip -> action label."""
import argparse, os, sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

HERE = os.path.dirname(__file__)
CKPT = os.path.abspath(os.path.join(HERE, "..", "outputs", "lora"))
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--s", type=int); ap.add_argument("--a", type=int)
    ap.add_argument("--f", type=int, default=-1)
    ap.add_argument("--question",
                    default="What action is the person performing in this radar point cloud?")
    args = ap.parse_args()

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        from render_samples import open_db, list_frames, render_clip
        from m4config import RPC
        env = open_db(RPC, 8)
        with env.begin() as txn:
            fr = list_frames(txn, args.s, args.a)
            fc = args.f if args.f >= 0 else fr[len(fr) // 2]
            tmp = os.path.abspath(os.path.join(HERE, "..", "outputs", "_live.png"))
            render_clip(txn, args.s, args.a, fc, tmp)
        env.close(); img = Image.open(tmp).convert("RGB")

    proc = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, CKPT).to("cuda").eval()

    d = proc.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": args.question}]}],
        add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**d, max_new_tokens=64, do_sample=False)
    n_in = d["input_ids"].shape[1]
    print("PREDICTION:", proc.tokenizer.decode(out[0][n_in:],
          skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
