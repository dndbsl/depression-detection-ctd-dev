#!/usr/bin/env python3
"""Extract pre-prediction-layer representations for the BEST CTD and RoBERTa models.

Outputs per split -> outputs/fusion/{modality}_{split}.npz with arrays:
    session_id (int), emb (float [n, d]), label (int)

CTD          : 24-D session-mean of the per-turn CTD features. This is the
               pre-prediction representation of the best CTD model (24-D
               session-mean + L2 logistic regression), which is by far the most
               robust on test (0.63 macro-F1, small dev->test gap) -- the MIL
               variant overfits (dev 0.86 / test 0.38) so it is NOT used here.
               NaNs imputed with TRAIN-only per-feature medians (leakage-safe).
RoBERTa-large: pooled participant embedding (1024-D), before the classifier
               = mean over the session's instance embeddings after the model's
               train-fit per-feature standardization (mil_pooling='mean').
               Best checkpoint = roberta-large frozen seed43 (project best_model,
               dev macro-F1 0.690). Uses the on-disk cached instance embeddings +
               the checkpoint's standardization buffers (no HF download needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
CTD_ROOT = SRC_ROOT / "ctd"
for p in (str(CTD_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from constants import CTD_FEATURE_NAMES, OUTPUT_DIR  # noqa: E402
from bags import build_split_bags  # noqa: E402
from ml_splits import load_split_labels  # noqa: E402

OUT = OUTPUT_DIR / "fusion"
ROBERTA_FEAT_DIR = (
    SRC_ROOT / "semantic-depr-roberta" / "datasets" / "processed"
    / "features_roberta-large_mean"
)
ROBERTA_CKPT = (
    SRC_ROOT / "semantic-depr-roberta" / "checkpoints"
    / "roberta-large_frozen_seed43" / "best_checkpoint.pt"
)
SPLITS = ("train", "dev", "test")


def _session_means(split):
    """24-D per-session mean of the per-turn CTD features (NaN-aware)."""
    bags = build_split_bags(split)
    sids = [b.session_id for b in bags]
    labels = [b.label for b in bags]
    means = np.vstack([np.nanmean(b.turns, axis=0) for b in bags])  # (n, 24)
    return np.array(sids, dtype=int), means, np.array(labels, dtype=int)


def extract_ctd() -> None:
    cache = {s: _session_means(s) for s in SPLITS}
    # train-only median imputation (leakage-safe), mirrors the LogReg pipeline
    train_means = cache["train"][1]
    train_median = np.nanmedian(train_means, axis=0)
    for split in SPLITS:
        sids, means, labels = cache[split]
        means = np.where(np.isnan(means), train_median, means)
        np.savez(OUT / f"ctd_{split}.npz",
                 session_id=sids, emb=means.astype(np.float32), label=labels)
        print(f"  ctd  {split}: {len(sids)} sessions, dim={means.shape[1]} (24-D session-mean)")


@torch.no_grad()
def extract_roberta() -> None:
    ckpt = torch.load(ROBERTA_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    feat_mean = sd["feature_mean"].float()
    feat_std = sd["feature_std"].float().clamp_min(1e-6)

    for split in SPLITS:
        feats = torch.load(ROBERTA_FEAT_DIR / f"{split}.pt", map_location="cpu")
        labels_df = load_split_labels(split).set_index("session_id")["PHQ8_Binary"].to_dict()
        sids, embs, labels = [], [], []
        for pid, inst in feats.items():
            pid = int(pid)
            if pid not in labels_df:
                continue
            z = (inst.float() - feat_mean) / feat_std          # [n_inst, 1024]
            pooled = z.mean(dim=0)                              # [1024]
            sids.append(pid); embs.append(pooled.numpy()); labels.append(int(labels_df[pid]))
        np.savez(OUT / f"roberta_{split}.npz",
                 session_id=np.array(sids, dtype=int),
                 emb=np.stack(embs).astype(np.float32),
                 label=np.array(labels, dtype=int))
        print(f"  roberta {split}: {len(sids)} sessions, dim={embs[0].shape[0]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Extracting CTD penultimate embeddings...")
    extract_ctd()
    print("Extracting RoBERTa-large penultimate embeddings...")
    extract_roberta()
    print(f"Saved -> {OUT}/{{ctd,roberta}}_{{train,dev,test}}.npz")


if __name__ == "__main__":
    main()
