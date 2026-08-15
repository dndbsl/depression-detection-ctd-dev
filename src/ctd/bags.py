"""Per-session bags of raw per-turn 24-D CTD features.

Shared helper used by the fusion-embedding extraction: each session is a *bag*
whose instances are the Ask/Res turn pairs with their raw 24-D CTD vectors.
Turn-level features are cached to ``outputs/turn_level_{split}.csv`` on first
use so repeated runs do not re-parse the transcripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT.parent
for p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from constants import CTD_FEATURE_NAMES, DEFAULT_DATA_ROOT, OUTPUT_DIR  # noqa: E402
from common.transcript_preprocessing import load_transcript  # noqa: E402
from data_loading import transcript_path  # noqa: E402
from feature_extraction import extract_session_turn_features  # noqa: E402
from ml_splits import load_split_labels  # noqa: E402
from turn_pairing import build_turn_pairs  # noqa: E402


class Bag:
    __slots__ = ("session_id", "label", "turns")

    def __init__(self, session_id: int, label: int, turns: np.ndarray) -> None:
        self.session_id = session_id
        self.label = label
        self.turns = turns  # (n_turns, 24) float


def build_split_bags(split: str) -> list[Bag]:
    cache = OUTPUT_DIR / f"turn_level_{split}.csv"
    if cache.exists():
        turn_df = pd.read_csv(cache)
        labels = load_split_labels(split).set_index("session_id")["PHQ8_Binary"].to_dict()
    else:
        labels_df = load_split_labels(split)
        labels = labels_df.set_index("session_id")["PHQ8_Binary"].to_dict()
        rows = []
        for sid in labels_df["session_id"]:
            df = load_transcript(transcript_path(DEFAULT_DATA_ROOT, int(sid)))
            pairs = build_turn_pairs(int(sid), df)
            tdf = extract_session_turn_features(pairs)
            if len(tdf):
                rows.append(tdf)
        turn_df = pd.concat(rows, ignore_index=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        turn_df.to_csv(cache, index=False)
    bags = []
    for sid, g in turn_df.groupby("session_id"):
        g = g.sort_values("turn_index")
        bags.append(Bag(int(sid), int(labels[int(sid)]),
                        g[CTD_FEATURE_NAMES].to_numpy(dtype=float)))
    return bags
