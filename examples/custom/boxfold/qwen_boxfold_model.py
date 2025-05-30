# qwen_boxfold_head.py
"""Minimal skeleton that augments **Qwen‑2.5 VL** with a multi‑task head
(Box‑Folding) and optional LoRA adapters.

▪ Tested with:
    transformers >= 4.40.0
    accelerate >= 0.28
    peft        >= 0.10.0  (for LoRA)

Usage (single‑GPU debug):
    python qwen_boxfold_head.py \
        --model qwen/Qwen-VL-Chat-Int4 \
        --train data/train.jsonl --val data/val.jsonl \
        --lora  --output ./ckpt_lora

Data format for each line in *.jsonl*::
    {"images": ["stem.png", "opt0.png", ...],
     "text":   "<|im_start|> ...",
     "target": "<extra_id_0> C",
     "mapping": {...}, "visible": {...}, "foldable": true}

You only need to implement Dataset & collator (place‑holders below) and adapt
loss mixing coefficients to your task.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# ───────────────────────────────────────────────────────────────
# 1.  Multi‑task head
# ───────────────────────────────────────────────────────────────


class BoxHead(nn.Module):
    """Predict foldable (1), mapping (6×6), visible (6) from pooled vector."""

    def __init__(self, hidden: int, n_faces: int = 6):
        super().__init__()
        self.foldable = nn.Linear(hidden, 1)
        self.mapping = nn.Linear(hidden, 6 * n_faces)
        self.visible = nn.Linear(hidden, n_faces)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "foldable": torch.sigmoid(self.foldable(pooled)),
            "mapping": self.mapping(pooled).view(-1, 6, 6),  # (B, face, cls)
            "visible": torch.sigmoid(self.visible(pooled)),
        }

# ───────────────────────────────────────────────────────────────
# 2.  Wrapper model
# ───────────────────────────────────────────────────────────────


class QwenBoxModel(nn.Module):
    def __init__(self, base_model: str, lora: bool = False, lora_r: int = 32, lora_alpha: int = 64):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True)
        hidden = self.model.config.hidden_size
        self.box_head = BoxHead(hidden)

        if lora:
            lora_cfg = LoraConfig(task_type="CAUSAL_LM", r=lora_r, lora_alpha=lora_alpha,
                                  target_modules=["q_proj", "v_proj", "k_proj"], lora_dropout=0.05)
            self.model = get_peft_model(self.model, lora_cfg)

    # ──────────────────────────────────────────
    def forward(self, batch: Dict[str, Any]):
        #  images already encoded as <img> tokens in batch["input_ids"]
        out = self.model(**{k: batch[k] for k in ["input_ids", "attention_mask"]},
                         labels=batch.get("labels"), return_dict=True)
        pooled = out.hidden_states[-1][:, -1]  # last token (</image>)
        multi = self.box_head(pooled)
        return out, multi


# ───────────────────────────────────────────────────────────────
# 3.  Loss utilities
# ───────────────────────────────────────────────────────────────
ce = nn.CrossEntropyLoss()
bce = nn.BCELoss()


def multitask_loss(multi_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   λ_fold=1.0, λ_map=1.0, λ_vis=0.5):
    loss_f = bce(multi_out["foldable"].squeeze(1), batch["foldable"].float())
    loss_m = ce(multi_out["mapping"].view(-1, 6), batch["mapping"].view(-1))
    loss_v = bce(multi_out["visible"], batch["visible"].float())
    return λ_fold * loss_f + λ_map * loss_m + λ_vis * loss_v

# ───────────────────────────────────────────────────────────────
# 4.  Dummy Dataset & collator (→ replace with real)
# ───────────────────────────────────────────────────────────────


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, tokenizer):
        ############  TODO: implement real image→<img> token pipeline  ############
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.tok = tokenizer

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tok(s["text"], return_tensors="pt")
        tgt = self.tok(s["target"], return_tensors="pt")
        # pack mapping/visible/foldable into tensors (placeholder)
        return {
            **enc,
            "labels": tgt["input_ids"],
            "foldable": torch.tensor([float(s["foldable"])], dtype=torch.float32),
            "mapping": torch.tensor([0] * 36).view(6, 6),   # fill real
            "visible": torch.zeros(6),
        }


def collate(batch: List[Dict[str, Any]]):
    # pad input_ids/labels etc.
    return {k: torch.nn.utils.rnn.pad_sequence([b[k].squeeze(0) for b in batch], batch_first=True, padding_value=0)
            if k in ["input_ids", "attention_mask", "labels"] else torch.stack([b[k] for b in batch])
            for k in batch[0]}

# ───────────────────────────────────────────────────────────────
# 5.  Train loop (simplified)
# ───────────────────────────────────────────────────────────────


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QwenBoxModel(args.model, lora=args.lora).to(device)
    ds = DummyDataset(args.train, model.tokenizer)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(optim, 500, len(dl) * args.epochs)
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out_text, out_multi = model(batch)
            loss_mt = multitask_loss(out_multi, batch)
            loss = out_text.loss + loss_mt
            loss.backward()
            optim.step()
            optim.zero_grad()
            sched.step()
            if step % 50 == 0:
                print(f"E{epoch} S{step} loss={loss.item():.3f}")
        model.save_pretrained(Path(args.output) / f"epoch{epoch}")


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lora", action="store_true")
    args = ap.parse_args()

    train(args)
