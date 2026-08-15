"""Parse a session transcript into a clean, ordered list of participant utterances.

Cleaning rules (no audio signal processing here, transcript-driven only):
  * keep only ``speaker == "Participant"`` rows
  * drop empty / whitespace-only / NaN ``value``
  * drop rows with NaN/missing start or stop time
  * drop rows where duration <= 0 (malformed) or < MIN_UTTERANCE_SEC (too short)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, MIN_UTTERANCE_SEC, PARTICIPANT_SPEAKER  # noqa: E402


@dataclass
class Utterance:
    participant_id: int
    utterance_index: int
    start_time: float
    stop_time: float
    duration: float
    text_value: str


def transcript_path(participant_id: int, data_root: str = DATA_ROOT) -> str:
    return os.path.join(data_root, f"{participant_id}_P", f"{participant_id}_TRANSCRIPT.csv")


def parse_transcript(participant_id: int, data_root: str = DATA_ROOT) -> tuple[list[Utterance], dict]:
    """Return (utterances, stats) for one session.

    ``stats`` records how many rows were dropped and why, for transparent logging.
    """
    path = transcript_path(participant_id, data_root)
    stats = {
        "participant_id": participant_id,
        "raw_rows": 0,
        "participant_rows": 0,
        "dropped_empty_text": 0,
        "dropped_nan_time": 0,
        "dropped_bad_duration": 0,
        "dropped_too_short": 0,
        "kept": 0,
    }

    if not os.path.exists(path):
        stats["error"] = "transcript_missing"
        return [], stats

    df = pd.read_csv(path, sep="\t", dtype={"speaker": str, "value": str})
    df.columns = [c.strip() for c in df.columns]
    stats["raw_rows"] = len(df)

    df = df[df["speaker"].astype(str).str.strip() == PARTICIPANT_SPEAKER].copy()
    stats["participant_rows"] = len(df)

    # Transcripts are time-ordered already; assert it rather than re-sorting.
    starts = pd.to_numeric(df["start_time"], errors="coerce")
    assert starts.is_monotonic_increasing or starts.dropna().is_monotonic_increasing, (
        f"Transcript for {participant_id} is not time-ordered")

    utterances: list[Utterance] = []
    idx = 0
    for _, row in df.iterrows():
        text = row["value"]
        if not isinstance(text, str) or text.strip() == "" or pd.isna(text):
            stats["dropped_empty_text"] += 1
            continue
        text = text.strip()

        start = pd.to_numeric(row["start_time"], errors="coerce")
        stop = pd.to_numeric(row["stop_time"], errors="coerce")
        if pd.isna(start) or pd.isna(stop):
            stats["dropped_nan_time"] += 1
            continue

        duration = float(stop) - float(start)
        if duration <= 0:
            stats["dropped_bad_duration"] += 1
            continue
        if duration < MIN_UTTERANCE_SEC:
            stats["dropped_too_short"] += 1
            continue

        utterances.append(Utterance(
            participant_id=int(participant_id),
            utterance_index=idx,
            start_time=float(start),
            stop_time=float(stop),
            duration=float(duration),
            text_value=text,
        ))
        idx += 1

    stats["kept"] = len(utterances)
    return utterances, stats
