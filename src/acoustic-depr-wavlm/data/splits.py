"""Load and normalise the train/dev/test label CSVs and apply session exclusions.

This is the single shared place where exclusions and label-column normalisation
happen, so every downstream step sees an identical view of the splits.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXCLUDED_SESSIONS, LABEL_FILES, TEST_COLUMN_RENAME  # noqa: E402


def load_split_labels(split: str, labels_root: str) -> pd.DataFrame:
    """Return a DataFrame with columns [Participant_ID, PHQ8_Binary, PHQ8_Score, split].

    The test split CSV uses ``PHQ_Binary``/``PHQ_Score``; these are renamed to the
    train/dev schema. Excluded sessions are dropped here.
    """
    if split not in LABEL_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(LABEL_FILES)}")

    path = os.path.join(labels_root, LABEL_FILES[split])
    df = pd.read_csv(path)
    df = df.rename(columns=TEST_COLUMN_RENAME)

    required = {"Participant_ID", "PHQ8_Binary", "PHQ8_Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split {split!r} CSV {path} missing columns: {missing}")

    df = df[["Participant_ID", "PHQ8_Binary", "PHQ8_Score"]].copy()
    df["Participant_ID"] = df["Participant_ID"].astype(int)
    df["PHQ8_Binary"] = df["PHQ8_Binary"].astype(int)
    df["PHQ8_Score"] = df["PHQ8_Score"].astype(int)
    df["split"] = split

    before = len(df)
    df = df[~df["Participant_ID"].isin(EXCLUDED_SESSIONS)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        excluded_here = sorted(set(EXCLUDED_SESSIONS) & set(
            pd.read_csv(path).rename(columns=TEST_COLUMN_RENAME)["Participant_ID"].astype(int)))
        print(f"[splits] {split}: excluded {dropped} session(s): {excluded_here}")

    return df


def load_all_splits(labels_root: str) -> dict[str, pd.DataFrame]:
    """Load every split into a dict keyed by split name."""
    return {split: load_split_labels(split, labels_root) for split in LABEL_FILES}
