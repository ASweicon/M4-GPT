# -*- coding: utf-8 -*-
"""Generate answers for I/Q clips (server-side, requires torch + transformers).

Example:
    python3 radar_encoder/evaluate.py --text data/text/p19_a10.jsonl \
        --encoder outputs/iq_encoder/encoder_sft.pt --lora outputs/iq_encoder/lora
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .dataset import IQClipDataset, parse_key
    from .dsp import load_pairs
    from .encoder import build_encoder
except ImportError:
    from dataset import IQClipDataset, parse_key
    from dsp import load_pairs
    from encoder import build_encoder

ROOT = Path(__file__).resolve().parents[2]
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--text", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--llm", default="Qwen/Qwen3-8B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/iq_eval.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.load(open(root / "data/meta/manifest.json"))
    pairs = load_pairs(root / "data/meta/p2_radar_config.json")

    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    encoder = build_encoder(llm_dim=model.config.hidden_size)
    encoder.load_state_dict(torch.load(args.encoder, map_location="cpu"))
    encoder.to(next(model.parameters()).device).eval()

    recs = [json.loads(x) for x in open(root / args.text) if x.strip()]
    if args.limit:
        recs = recs[:args.limit]
    ds_cache = {}
    out = open(root / args.out, "w")

    for rec in recs:
        s, a, _ = parse_key(rec["key"].encode())
        if s not in ds_cache:
            ds_cache[s] = IQClipDataset(root / manifest["subjects"][str(s)]["lmdb"],
                                       s, pairs, t_frames=16, stride=8)
        x = ds_cache[s].load_clip_at(a, rec["center"])
        if x is None:
            continue
        user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        prompt = f"{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n"
        with torch.no_grad():
            radar = encoder(torch.from_numpy(x).unsqueeze(0).to(next(model.parameters()).device))
            p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            p_ids = p_ids.to(radar.device)
            emb = torch.cat([radar, model.get_input_embeddings()(p_ids)], dim=1)
            gen = model.generate(inputs_embeds=emb, max_new_tokens=64, do_sample=False)
        text = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
        out.write(json.dumps({"key": rec["key"], "pred": text,
                              "ref": rec["messages"][1]["content"]}, ensure_ascii=False) + "\n")
        print(rec["key"], "->", text[:80])
    out.close()


if __name__ == "__main__":
    main()
