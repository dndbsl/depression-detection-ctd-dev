#!/usr/bin/env python3
"""eGeMAPS-style functionals applied to the 24 CTD turn-level features.

Motivation
----------
Each DAIC-WOZ session is a *sequence* of Ask/Res turn pairs, so every one of the
24 CTD features (``constants.CTD_FEATURE_NAMES``) is a short time series across
turns. Instead of collapsing each to a single per-session mean (the original
``stats_analysis.aggregate_to_session_level``), we summarise each turn-sequence
with the openSMILE eGeMAPS v02 *functional* bank.

eGeMAPS functional bank (Eyben et al., 2016; openSMILE
``config/egemaps/v02/eGeMAPSv02_core.func.conf.inc``)
-----------------------------------------------------
eGeMAPS applies, per low-level descriptor (LLD):

* to **every** LLD: arithmetic mean + coefficient of variation
  (``stddevNorm = stddev / |mean|``);
* to the **pitch / loudness** contours additionally: the 20th/50th/80th
  percentiles, the 20-80 percentile range, and the mean & standard deviation of
  the **rising** and **falling** slopes.

That is 10 distinct functional operators. CTD features are not split into
pitch-like vs. spectral-like groups, so we apply the full 10-operator bank
uniformly to all 24 features -> 240 session-level descriptors.

Slope adaptation: openSMILE computes slopes over monotonic rising/falling
segments of a contour. For the short per-turn sequences here we use the
first differences between consecutive turns (slope per turn-step); rising =
positive differences, falling = negative differences. This is the standard
discrete adaptation and is documented as such.

All functionals are NaN-aware: NaN turns (e.g. undefined silence ratios when a
turn has zero silence) are dropped before summarising a given feature's
sequence. Functionals that are undefined for the remaining length (CV with
<2 points or ~zero mean, slopes with no rising/falling step) are returned as
NaN and imputed downstream (train-fit median) by the ML pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import CTD_FEATURE_NAMES, OUTPUT_DIR  # noqa: E402

_EPS = 1e-8

# Canonical order of the eGeMAPS functional operators applied per CTD feature.
FUNCTIONAL_NAMES: list[str] = [
    "amean",
    "cv",
    "pctl20",
    "pctl50",
    "pctl80",
    "pctlrange20_80",
    "mean_rising_slope",
    "std_rising_slope",
    "mean_falling_slope",
    "std_falling_slope",
]


def _slopes(seq: np.ndarray) -> tuple[float, float, float, float]:
    """Mean/std of rising (positive) and falling (negative) first differences."""
    if seq.size < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    diffs = np.diff(seq)
    rising = diffs[diffs > 0]
    falling = diffs[diffs < 0]
    mean_rise = float(np.mean(rising)) if rising.size else 0.0
    std_rise = float(np.std(rising, ddof=0)) if rising.size > 1 else (0.0 if rising.size == 1 else np.nan)
    mean_fall = float(np.mean(falling)) if falling.size else 0.0
    std_fall = float(np.std(falling, ddof=0)) if falling.size > 1 else (0.0 if falling.size == 1 else np.nan)
    return (mean_rise, std_rise, mean_fall, std_fall)


def compute_functionals(seq: np.ndarray) -> dict[str, float]:
    """Apply the 10-operator eGeMAPS functional bank to one feature's turn sequence."""
    s = np.asarray(seq, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return {name: np.nan for name in FUNCTIONAL_NAMES}

    mean = float(np.mean(s))
    std = float(np.std(s, ddof=0))
    cv = std / abs(mean) if abs(mean) > _EPS else np.nan
    p20, p50, p80 = (float(v) for v in np.percentile(s, [20, 50, 80]))
    mean_rise, std_rise, mean_fall, std_fall = _slopes(s)

    return {
        "amean": mean,
        "cv": cv,
        "pctl20": p20,
        "pctl50": p50,
        "pctl80": p80,
        "pctlrange20_80": p80 - p20,
        "mean_rising_slope": mean_rise,
        "std_rising_slope": std_rise,
        "mean_falling_slope": mean_fall,
        "std_falling_slope": std_fall,
    }


def functional_feature_names() -> list[str]:
    return [f"{feat}__{func}" for feat in CTD_FEATURE_NAMES for func in FUNCTIONAL_NAMES]


def build_session_functionals(
    turn_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per session: 24 CTD features x 10 functionals = 240 columns."""
    rows = []
    label_lookup = labels_df.set_index("session_id")
    for session_id, group in turn_df.groupby("session_id"):
        group = group.sort_values("turn_index")
        row: dict[str, float] = {"session_id": int(session_id)}
        lab = label_lookup.loc[session_id]
        row["PHQ8_Binary"] = int(lab["PHQ8_Binary"])
        row["PHQ8_Score"] = int(lab["PHQ8_Score"])
        row["n_turns"] = int(len(group))
        for feat in CTD_FEATURE_NAMES:
            funcs = compute_functionals(group[feat].to_numpy(dtype=float))
            for fname, val in funcs.items():
                row[f"{feat}__{fname}"] = val
        rows.append(row)
    cols = ["session_id", "PHQ8_Binary", "PHQ8_Score", "n_turns", *functional_feature_names()]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    turn_df = pd.read_csv(OUTPUT_DIR / "turn_level_features.csv")
    session_simple = pd.read_csv(OUTPUT_DIR / "session_level_features.csv")
    labels_df = session_simple[["session_id", "PHQ8_Binary", "PHQ8_Score"]].copy()

    feats = build_session_functionals(turn_df, labels_df)
    out_path = OUTPUT_DIR / "session_functionals_features.csv"
    feats.to_csv(out_path, index=False)

    feat_cols = functional_feature_names()
    nan_rates = feats[feat_cols].isna().mean().sort_values(ascending=False)
    print(f"Sessions: {len(feats)}  |  functional features: {len(feat_cols)} "
          f"(= {len(CTD_FEATURE_NAMES)} CTD x {len(FUNCTIONAL_NAMES)} functionals)")
    print(f"Saved -> {out_path}")
    print(f"Turn counts per session: min={feats['n_turns'].min()}, "
          f"median={int(feats['n_turns'].median())}, max={feats['n_turns'].max()}")
    print("\nTop-10 functional columns by NaN rate (imputed downstream):")
    print(nan_rates.head(10).to_string())


if __name__ == "__main__":
    main()
