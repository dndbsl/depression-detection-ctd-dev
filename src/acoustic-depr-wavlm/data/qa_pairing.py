"""Question-answering (Ask/Response) segmentation of DAIC-WOZ transcripts.

This mirrors the CTD turn-pairing convention (src/ctd/turn_pairing.py) so the
acoustic stream is segmented *consistently with the CTD/text pipeline*:

  * read the FULL transcript (Ellie + Participant),
  * group consecutive same-speaker rows into speaker "runs",
  * discard any leading Participant run before the first Ellie turn,
  * pair each Ellie "ask" run with the immediately following Participant "response" run.

The acoustic unit returned is the **participant response turn** -- one segment per
answered question, spanning ``[res_first.start, res_last.stop]`` (the contiguous answer
window, internal micro-pauses included). This yields fewer, longer, semantically
coherent Q-A segments instead of raw per-row utterances.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, MIN_UTTERANCE_SEC  # noqa: E402

ELLIE = "Ellie"
PARTICIPANT = "Participant"


@dataclass
class ResponseTurn:
    participant_id: int
    utterance_index: int   # = Q-A turn index; name kept for downstream compatibility
    start_time: float
    stop_time: float
    duration: float
    text_value: str


def transcript_path(participant_id: int, data_root: str = DATA_ROOT) -> str:
    return os.path.join(data_root, f"{participant_id}_P", f"{participant_id}_TRANSCRIPT.csv")


def _speaker_runs(df: pd.DataFrame) -> list[tuple[str, list]]:
    """Group time-ordered rows into consecutive same-speaker runs."""
    runs: list[tuple[str, list]] = []
    cur_spk: str | None = None
    cur: list = []
    for row in df.itertuples(index=False):
        spk = row.speaker
        if spk == cur_spk:
            cur.append(row)
        else:
            if cur:
                runs.append((cur_spk, cur))
            cur_spk = spk
            cur = [row]
    if cur:
        runs.append((cur_spk, cur))
    return runs


def build_response_turns(
    participant_id: int,
    data_root: str = DATA_ROOT,
    min_sec: float = MIN_UTTERANCE_SEC,
) -> tuple[list[ResponseTurn], dict]:
    """Return (response_turns, stats) for one session using CTD ask/response pairing."""
    path = transcript_path(participant_id, data_root)
    stats = {
        "participant_id": participant_id,
        "raw_rows": 0,
        "ask_response_pairs": 0,
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

    df["speaker"] = df["speaker"].astype(str).str.strip()
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["stop_time"] = pd.to_numeric(df["stop_time"], errors="coerce")
    df = df.dropna(subset=["start_time", "stop_time"])
    df = df[df["speaker"].isin([ELLIE, PARTICIPANT])]
    df = df.sort_values("start_time").reset_index(drop=True)

    runs = _speaker_runs(df)
    while runs and runs[0][0] != ELLIE:   # drop leading Participant run(s)
        runs.pop(0)

    turns: list[ResponseTurn] = []
    idx = 0
    i = 0
    while i < len(runs):
        speaker, _ = runs[i]
        if speaker != ELLIE:
            i += 1
            continue
        if i + 1 >= len(runs) or runs[i + 1][0] != PARTICIPANT:
            break  # trailing incomplete Ask -> discard
        res_run = runs[i + 1][1]
        i += 2
        stats["ask_response_pairs"] += 1

        start = float(res_run[0].start_time)
        stop = float(res_run[-1].stop_time)
        duration = stop - start
        if duration <= 0:
            stats["dropped_bad_duration"] += 1
            continue
        if duration < min_sec:
            stats["dropped_too_short"] += 1
            continue

        text = " ".join(
            str(r.value).strip() for r in res_run
            if isinstance(r.value, str) and r.value.strip()
        )
        turns.append(ResponseTurn(
            participant_id=int(participant_id),
            utterance_index=idx,
            start_time=start,
            stop_time=stop,
            duration=duration,
            text_value=text,
        ))
        idx += 1

    stats["kept"] = len(turns)
    return turns, stats
