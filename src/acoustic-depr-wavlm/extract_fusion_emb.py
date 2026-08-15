#!/usr/bin/env python3
"""Extract WavLM-large penultimate embeddings (e_a, 1024-D) for late fusion.

e_a is the attention-pooled subject embedding fed to the final Linear classifier
in AcousticDepressionModel. Saved per split to the CTD project's fusion dir.
Run from this project root with the acoustic-depr env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SubjectFeatureDataset, collate_fn  # noqa: E402
from model.full_model import build_model  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "ctd" / "outputs" / "fusion"
CKPT = "checkpoints/linear_probe/best.pt"
SPLITS = ("train", "dev", "test")


@torch.no_grad()
def embed_split(model, split):
    ds = SubjectFeatureDataset(split, "manifests/", "cache/")
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    sids, embs, labels = [], [], []
    for batch in loader:
        u = model.layer_weighting(batch["features"])      # [B, L, 1024]
        e_a, _ = model.attention_pool(u, batch["mask"])   # [B, 1024]
        embs.append(e_a.numpy())
        sids.extend(int(p) for p in batch["pids"])
        labels.append(batch["labels"].numpy())
    return (np.array(sids, dtype=int),
            np.concatenate(embs).astype(np.float32),
            np.concatenate(labels).astype(int))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded wavlm best.pt (epoch {ckpt.get('epoch')}, dev_loss={ckpt.get('dev_loss')})")
    for split in SPLITS:
        sids, embs, labels = embed_split(model, split)
        np.savez(OUT / f"wavlm_{split}.npz", session_id=sids, emb=embs, label=labels)
        print(f"  wavlm {split}: {len(sids)} sessions, dim={embs.shape[1]}")
    print(f"Saved -> {OUT}/wavlm_{{train,dev,test}}.npz")


if __name__ == "__main__":
    main()
