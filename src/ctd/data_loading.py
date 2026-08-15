"""Load DAIC-WOZ labels and transcripts for train+dev combined cohort."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from constants import DEFAULT_DATA_ROOT, DEPRESSION_ROOT

if str(DEPRESSION_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPRESSION_ROOT))

from common.daic_cleaning import apply_cleaning  # noqa: E402
from common.transcript_preprocessing import load_transcript  # noqa: E402

LABEL_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
}


def transcript_path(data_root: Path, pid: int) -> Path:
    return Path(data_root) / "data" / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"


def load_split_labels(data_root: Path, split: str) -> pd.DataFrame:
    path = Path(data_root) / "labels" / LABEL_FILES[split]
    df = pd.read_csv(path)
    df = df.rename(columns={"Participant_ID": "pid"})
    df = df[["pid", "PHQ8_Score", "PHQ8_Binary"]].copy()
    df["pid"] = df["pid"].astype(int)
    df["PHQ8_Score"] = df["PHQ8_Score"].astype(int)
    df["PHQ8_Binary"] = (df["PHQ8_Score"] >= 10).astype(int)
    return df.set_index("pid").sort_index()


def load_combined_train_dev_labels(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[pd.DataFrame, dict]:
    """Load and clean train+dev labels. Returns (labels_df, cleaning_reports)."""
    reports = {}
    frames = []
    for split in ("train", "dev"):
        raw = load_split_labels(data_root, split)
        cleaned, report = apply_cleaning(raw, split=split)
        reports[split] = report.to_dict()
        cleaned = cleaned.reset_index()
        cleaned["split"] = split
        frames.append(cleaned)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"pid": "session_id"})
    return combined, reports


def load_session_transcript(data_root: Path, session_id: int) -> pd.DataFrame:
    path = transcript_path(data_root, session_id)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    return load_transcript(path)
