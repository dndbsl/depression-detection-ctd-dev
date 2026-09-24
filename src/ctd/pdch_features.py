#!/usr/bin/env python3
"""Turn PDCH's cached VAD utterances into session-level CTD features.

This is the payoff of building an adapter (`MC-01`) rather than reimplementing
CTD maths per corpus: everything below the `list[Utterance]` boundary --
`turn_pairing.build_turn_pairs`, `feature_extraction.extract_turn_features`,
`functionals.compute_functionals` -- is the DAIC-WOZ code, called unchanged.
The only corpus-specific argument is which speaker string is the Ask side.

`build_pdch_features()` is parameterised by the intra-turn silence threshold so
`MC-08` can re-select it inside training folds instead of inheriting
DAIC-WOZ's 0.2 s constant unexamined.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import CTD_FEATURE_NAMES  # noqa: E402
from feature_extraction import extract_session_turn_features  # noqa: E402
from functionals import FUNCTIONAL_NAMES, compute_functionals  # noqa: E402
from pdch_adapter import PDCH_LABELS, subject_of  # noqa: E402
from turn_pairing import Utterance, build_turn_pairs  # noqa: E402

CACHE_DIR = REPO_ROOT / "output" / "pdch"
UTTERANCE_CACHE = CACHE_DIR / "pdch_utterances.csv"

MEAN_COLS = [f"{f}__amean" for f in CTD_FEATURE_NAMES]
# The intra-turn silence threshold grid re-selected within PDCH training folds.
# 0.2 s is DAIC-WOZ's deployed value and is kept in the grid as the reference.
SILENCE_THRESHOLD_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def load_utterance_cache(path: Path = UTTERANCE_CACHE) -> dict[str, list[Utterance]]:
    """session_id -> VAD-refined utterances, from `scripts/build_pdch_cache.py`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/build_pdch_cache.py` first."
        )
    df = pd.read_csv(path)
    out: dict[str, list[Utterance]] = {}
    for sid, g in df.groupby("session_id", sort=True):
        g = g.sort_values(["start", "end"])
        out[str(sid)] = [
            Utterance(start=float(s), end=float(e), speaker=str(spk))
            for s, e, spk in zip(g["start"], g["end"], g["speaker"])
        ]
    return out


def session_turn_features(
    utts: dict[str, list[Utterance]], silence_threshold: float,
) -> pd.DataFrame:
    """Turn-level 24-D CTD features for every session, at one silence threshold."""
    frames = []
    for sid, u in utts.items():
        pairs = build_turn_pairs(sid, u, PDCH_LABELS)
        if not pairs:
            continue
        frames.append(extract_session_turn_features(pairs, silence_threshold))
    return pd.concat(frames, ignore_index=True)


def build_pdch_features(
    utts: dict[str, list[Utterance]], silence_threshold: float = 0.2,
) -> pd.DataFrame:
    """One row per session: 24 CTD features x 10 eGeMAPS functionals.

    `__amean` is the session mean, i.e. the deployed 24-D descriptor.
    """
    turn_level = session_turn_features(utts, silence_threshold)
    rows = []
    for sid, g in turn_level.groupby("session_id", sort=True):
        g = g.sort_values("turn_index")
        row: dict = {
            "session_id": str(sid),
            "subject_id": subject_of(str(sid)),
            "n_turn_pairs": int(len(g)),
        }
        for feat in CTD_FEATURE_NAMES:
            for fname, val in compute_functionals(g[feat].to_numpy(float)).items():
                row[f"{feat}__{fname}"] = val
        rows.append(row)
    cols = ["session_id", "subject_id", "n_turn_pairs"] + [
        f"{f}__{fn}" for f in CTD_FEATURE_NAMES for fn in FUNCTIONAL_NAMES
    ]
    return pd.DataFrame(rows)[cols]


def build_feature_grid(
    utts: dict[str, list[Utterance]],
    thresholds: tuple[float, ...] = SILENCE_THRESHOLD_GRID,
) -> dict[float, pd.DataFrame]:
    """One session-level feature table per candidate silence threshold."""
    return {t: build_pdch_features(utts, t) for t in thresholds}


def degeneracy_report(features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature NaN rate and distinct-value count at the session-mean level.

    The point of `MC-06`: under raw 1-second turn labels 10 of the 24 features
    are constant or NaN. This is the check that VAD actually brought them back.
    """
    rows = []
    for feat in CTD_FEATURE_NAMES:
        v = features[f"{feat}__amean"].to_numpy(float)
        finite = v[np.isfinite(v)]
        rows.append({
            "feature": feat,
            "nan_rate": float(np.mean(~np.isfinite(v))),
            "n_distinct": int(len(np.unique(np.round(finite, 6)))),
            "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "degenerate": bool(finite.size < 2 or np.std(finite) < 1e-12),
        })
    return pd.DataFrame(rows)
