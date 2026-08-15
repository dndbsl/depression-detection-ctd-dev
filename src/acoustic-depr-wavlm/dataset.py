"""Per-subject dataset over cached WavLM utterance features.

Each item is one subject: a bag of utterance features ``[num_utt, 25, 1024]`` plus a
binary label. ``collate_fn`` pads variable utterance counts and returns a boolean
mask so attention pooling can ignore padding.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset

# Cache files are trusted local artefacts; silence the weights_only deprecation notice.
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

from config import CACHE_DIR, MANIFEST_DIR


class SubjectFeatureDataset(Dataset):
    def __init__(self, split: str, manifest_dir: str = MANIFEST_DIR, cache_dir: str = CACHE_DIR):
        self.split = split
        self.cache_dir = cache_dir
        manifest_path = os.path.join(manifest_dir, f"manifest_{split}.csv")
        manifest = pd.read_csv(manifest_path)

        # One label per subject (constant across the subject's utterances).
        subj = manifest.drop_duplicates("participant_id")[["participant_id", "PHQ8_Binary"]]
        self.subjects: list[tuple[int, int]] = []
        missing = []
        for _, r in subj.iterrows():
            pid = int(r["participant_id"])
            if not os.path.exists(os.path.join(cache_dir, f"{pid}.pt")):
                missing.append(pid)
                continue
            self.subjects.append((pid, int(r["PHQ8_Binary"])))
        if missing:
            print(f"[dataset] {split}: {len(missing)} subjects missing cache, skipped: {missing}")
        if not self.subjects:
            raise RuntimeError(f"No cached features for split {split!r} in {cache_dir}")

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        pid, label = self.subjects[idx]
        blob = torch.load(os.path.join(self.cache_dir, f"{pid}.pt"), map_location="cpu",
                          weights_only=False)
        feats = blob["features"].float()                            # [num_utt, 25, 1024]
        return {"pid": pid, "features": feats, "label": float(label)}


def collate_fn(batch: list[dict]):
    """Pad to the max utterance count in the batch and build a validity mask."""
    lengths = [b["features"].shape[0] for b in batch]
    max_len = max(lengths)
    num_layers = batch[0]["features"].shape[1]
    dim = batch[0]["features"].shape[2]

    feats = torch.zeros(len(batch), max_len, num_layers, dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.zeros(len(batch))
    pids = []
    for i, b in enumerate(batch):
        n = b["features"].shape[0]
        feats[i, :n] = b["features"]
        mask[i, :n] = True
        labels[i] = b["label"]
        pids.append(b["pid"])

    return {"features": feats, "mask": mask, "labels": labels, "pids": pids}
