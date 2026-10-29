# -*- coding: utf-8 -*-
"""Train the I/Q encoder + projector (+ optional LoRA) on generated text.

Server-side only (requires torch + transformers). Examples:
    python3 radar_encoder/train.py --stage align --llm Qwen/Qwen3-8B
    python3 radar_encoder/train.py --stage sft   --llm Qwen/Qwen3-8B
"""

import argparse
import glob
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .dataset import IQClipDataset, parse_key
    from .dsp import load_pairs
    from .encoder import build_encoder, count_params
except ImportError:
    from dataset import IQClipDataset, parse_key
    from dsp import load_pairs
    from encoder import build_encoder, count_params

ROOT = Path(__file__).resolve().parents[2]
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class ClipTextDataset(Dataset):
    def __init__(self, jsonl_files, root, pairs, t_frames=16, subjects=(1, 2)):
        self.items = []
        for f in jsonl_files:
            with open(f) as fi:
                self.items.extend([json.loads(line) for line in fi if line.strip()])
        self.items = [x for x in self.items if int(x["subject"]) in subjects]
        self.root = Path(root)
        self.pairs = pairs
        self.t_frames = t_frames
        self.ds = {}
        manifest = json.load(open(self.root / "data/meta/manifest.json"))
        self.lmdb_rel = {int(k): v["lmdb"] for k, v in manifest["subjects"].items()}

    def _get_ds(self, subject, lmdb_rel):
        if subject not in self.ds:
            self.ds[subject] = IQClipDataset(
                self.root / lmdb_rel, subject, self.pairs, t_frames=self.t_frames, stride=8)
        return self.ds[subject]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rec = self.items[i]
        s, a, _ = parse_key(rec["key"].encode())
        ds = self._get_ds(s, self.lmdb_rel[s])
        x = ds.load_clip_at(a, rec["center"])
        if x is None:
            return None
        return {"x": x, "rec": rec}


def make_loader(items, root, pairs, batch_size, subjects):
    ds = ClipTextDataset(items, root, pairs, subjects=subjects)

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {
            "x": torch.from_numpy(np_stack([b["x"] for b in batch])),
            "rec": [b["rec"] for b in batch],
        }
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                      collate_fn=collate)


def np_stack(xs):
    import numpy as np
    return np.stack(xs)


def build_llm(args):
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.stage == "sft":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                         bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        model.train()
        if args.grad_checkpoint:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
    return tokenizer, model


def save_ckpt(out_dir, encoder, model, stage):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_dir / f"encoder_{stage}.pt")
    if stage == "sft":
        model.save_pretrained(out_dir / "lora")


def make_batch(tokenizer, model, radar_tokens, recs, stage):
    embed = model.get_input_embeddings()
    parts, labels = [], []
    for j, rec in enumerate(recs):
        if stage == "align":
            user, answer = "Describe the radar clip.", rec.get("caption") or rec["messages"][1]["content"]
        else:
            msgs = rec["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            answer = next(m["content"] for m in msgs if m["role"] == "assistant")
        prompt = f"{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n"
        p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(radar_tokens.device)
        a_ids = tokenizer(answer + IM_END, return_tensors="pt", add_special_tokens=False).input_ids.to(radar_tokens.device)
        p_emb = embed(p_ids)
        a_emb = embed(a_ids)
        seq = torch.cat([radar_tokens[j:j + 1], p_emb, a_emb], dim=1)
        lab = torch.cat([
            torch.full((1, radar_tokens.shape[1]), -100, device=radar_tokens.device, dtype=torch.long),
            torch.full((1, p_ids.shape[1]), -100, device=radar_tokens.device, dtype=torch.long),
            a_ids,
        ], dim=1)
        parts.append(seq)
        labels.append(lab)
    maxlen = max(p.shape[1] for p in parts)
    emb, lab, mask = [], [], []
    for p, l in zip(parts, labels):
        pad = maxlen - p.shape[1]
        emb.append(F.pad(p, (0, 0, 0, pad)))
        lab.append(F.pad(l, (0, pad), value=-100))
        mask.append(torch.cat([torch.ones(p.shape[1], device=p.device),
                               torch.zeros(pad, device=p.device)]))
    return torch.cat(emb, 0), torch.stack(mask), torch.cat(lab, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["align", "sft"], default="align")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--text-glob", default="data/text/*.jsonl")
    ap.add_argument("--llm", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/iq_encoder")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", default="1,2")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--save-every", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(glob.glob(str(root / args.text_glob)))
    if args.limit:
        files = files[:args.limit]
    pairs = load_pairs(root / "data/meta/p2_radar_config.json")

    tokenizer, model = build_llm(args)
    device = next(model.parameters()).device
    encoder = build_encoder(llm_dim=model.config.hidden_size).to(device)
    print("encoder params:", f"{count_params(encoder):,}")

    trainable = list(encoder.parameters())
    if args.stage == "sft":
        trainable += [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = tuple(int(x) for x in args.subjects.split(",") if x)
    global_step = 0
    for epoch in range(args.epochs):
        loader = make_loader(files, root, pairs, args.batch, subjects)
        for step, batch in enumerate(loader):
            if args.max_steps and global_step >= args.max_steps:
                break
            if batch is None:
                continue
            x = batch["x"].to(device)
            radar = encoder(x)
            emb, mask, labels = make_batch(tokenizer, model, radar, batch["rec"], args.stage)
            out = model(inputs_embeds=emb, attention_mask=mask, labels=labels, use_cache=False)
            if step % args.grad_accum == 0:
                opt.zero_grad(set_to_none=True)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                global_step += 1
                if global_step % 10 == 0:
                    print(f"epoch {epoch} step {global_step} loss {out.loss.item():.4f}", flush=True)
                if args.save_every and global_step % args.save_every == 0:
                    save_ckpt(out_dir, encoder, model, args.stage)
        if args.max_steps and global_step >= args.max_steps:
            break

    save_ckpt(out_dir, encoder, model, args.stage)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
