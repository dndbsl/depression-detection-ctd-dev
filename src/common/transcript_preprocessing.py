"""Row-level transcript preprocessing for the acoustic+semantic+timing fusion MIL.

Scope: turn grouping with marker-aware onset-latency anchoring. Session-level
exclusion/relabeling is fully delegated to `daic_cleaning` — do NOT add session
filtering logic here.

Transcript column spec (DAIC-WOZ): start_time, stop_time, speaker, value
(tab-separated, speaker values "Ellie" | "Participant").

Run as a script to execute Step 0 (bracket-token scan) followed by the full
turn-grouping report on the cleaned train split:

    python -m common.transcript_preprocessing \\
        [--data_root /path/to/DAIC-WOZ] [--split train]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Sys-path bootstrap: allow running from any cwd inside the repo
# ---------------------------------------------------------------------------
_COMMON_DIR = Path(__file__).resolve().parent
_DEPRESSION_ROOT = _COMMON_DIR.parent
if str(_DEPRESSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEPRESSION_ROOT))

from common.daic_cleaning import filter_session_ids  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_DATA_ROOT = Path(os.environ.get("DAIC_WOZ_ROOT", str(Path.home() / "datasets" / "DAIC-WOZ")))
_TRAIN_LABEL_FILE = "train_split_Depression_AVEC2017.csv"


def _transcript_path(data_root: Path, pid: int) -> Path:
    return Path(data_root) / "data" / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"


# ---------------------------------------------------------------------------
# Marker handling
# ---------------------------------------------------------------------------

# Single regex covering both bracket styles.  Intentionally permissive on
# whitespace inside tags (<clears throat>, <sharp inhale>) and bracket
# repetition.  Empty tags (<>, []) are also consumed.
_MARKER_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")


def strip_markers(value: str) -> str:
    """Remove all <...> and [...] markers from value, collapse whitespace.

    Returns the remaining text, stripped.  Returns "" if nothing remains.
    """
    cleaned = _MARKER_RE.sub("", str(value))
    return " ".join(cleaned.split())


def is_marker_only(value: str) -> bool:
    """True iff value contains only bracket markers and no real verbal content."""
    return strip_markers(value) == ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    session_id: int
    turn_index: int
    ellie_stop: Optional[float]     # stop_time of the preceding Ellie row
    # Acoustic instance span: [first_real_start, last_row_stop]
    # None if every participant row in this turn was marker-only.
    span_start: Optional[float]
    span_stop: float                # always set (last participant row stop)
    onset_latency: Optional[float]  # None if all-marker-only or no ellie anchor
    anchor_moved: bool              # True → leading marker-only rows were skipped
    skipped_marker_rows: list       # (start, stop, value) tuples that were skipped
    all_rows: list                  # (start, stop, value) tuples — full turn


@dataclass
class SessionPreTurnStats:
    session_id: int
    n_pre_turn_participant_rows: int
    pre_turn_rows: list             # (start, stop, value) tuples before first real Ellie


# ---------------------------------------------------------------------------
# Transcript loading (no Ellie-text isolation: we need Ellie text to detect
# the first substantive question for the pre-turn filter)
# ---------------------------------------------------------------------------

def load_transcript(path: Path) -> pd.DataFrame:
    """Load a DAIC-WOZ transcript, normalise columns, sort by (start, stop).

    Returns columns: start_time (float), stop_time (float), speaker (str),
    value (str).  Only rows with speaker in {"Ellie", "Participant"} are kept.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["speaker"] = df["speaker"].str.strip()
    df["value"] = df["value"].str.strip()
    df["start_time"] = df["start_time"].astype(float)
    df["stop_time"] = df["stop_time"].astype(float)
    df = df[df["speaker"].isin(["Ellie", "Participant"])].copy()
    return df.sort_values(["start_time", "stop_time"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 0: bracket-token scanner
# ---------------------------------------------------------------------------

def scan_bracket_tokens(data_root: Path, session_ids: list[int]) -> dict:
    """Scan all transcript files for <...> and [...] tokens.

    Returns a dict with per-style token frequency Counters and session-level
    style-presence counts.
    """
    angle_re = re.compile(r"<[^>]+>")
    square_re = re.compile(r"\[[^\]]+\]")

    angle_counter: Counter = Counter()
    square_counter: Counter = Counter()
    sessions_with_angle = 0
    sessions_with_square = 0
    sessions_with_both = 0
    sessions_scanned = 0
    missing = []

    for pid in sorted(session_ids):
        path = _transcript_path(data_root, pid)
        if not path.exists():
            missing.append(pid)
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        all_text = " ".join(df["value"].astype(str).tolist())
        angles = [t.lower() for t in angle_re.findall(all_text)]
        squares = [t.lower() for t in square_re.findall(all_text)]
        has_angle = bool(angles)
        has_square = bool(squares)
        if has_angle:
            sessions_with_angle += 1
            angle_counter.update(angles)
        if has_square:
            sessions_with_square += 1
            square_counter.update(squares)
        if has_angle and has_square:
            sessions_with_both += 1
        sessions_scanned += 1

    return {
        "sessions_scanned": sessions_scanned,
        "missing": missing,
        "sessions_with_angle_brackets": sessions_with_angle,
        "sessions_with_square_brackets": sessions_with_square,
        "sessions_with_both": sessions_with_both,
        "angle_tokens": angle_counter,
        "square_tokens": square_counter,
    }


def print_scan_report(scan: dict) -> None:
    n = scan["sessions_scanned"]
    print("=" * 70)
    print("STEP 0 — BRACKET TOKEN SCAN")
    print("=" * 70)
    print(f"Sessions scanned : {n}")
    if scan["missing"]:
        print(f"Missing files    : {scan['missing']}")
    print(f"Sessions with <...> : {scan['sessions_with_angle_brackets']} / {n}")
    print(f"Sessions with [...] : {scan['sessions_with_square_brackets']} / {n}")
    print(f"Sessions with both  : {scan['sessions_with_both']} / {n}")

    if scan["sessions_with_angle_brackets"] > 0 and scan["sessions_with_square_brackets"] > 0:
        style_str = "BOTH angle-bracket and square-bracket styles present"
    elif scan["sessions_with_angle_brackets"] > 0:
        style_str = "angle-bracket style only"
    else:
        style_str = "square-bracket style only"
    print(f"Bracket styles    : {style_str}")

    print()
    print("── Angle-bracket tokens (sorted by freq) ──")
    for tok, cnt in scan["angle_tokens"].most_common():
        print(f"  {cnt:5d}  {tok}")

    print()
    print("── Square-bracket tokens (sorted by freq) ──")
    for tok, cnt in scan["square_tokens"].most_common():
        print(f"  {cnt:5d}  {tok}")
    print()


# ---------------------------------------------------------------------------
# Core: turn grouping with marker-aware anchoring
# ---------------------------------------------------------------------------

def _finalize_turn(
    session_id: int,
    turn_index: int,
    p_rows: list,               # list of (start, stop, value)
    ellie_stop: Optional[float],
) -> Turn:
    """Build a Turn from a collected group of Participant rows."""
    first_real_idx = next(
        (i for i, (_, _, val) in enumerate(p_rows) if not is_marker_only(val)),
        None,
    )
    if first_real_idx is None:
        # Every participant row in this turn is marker-only.
        return Turn(
            session_id=session_id,
            turn_index=turn_index,
            ellie_stop=ellie_stop,
            span_start=None,
            span_stop=p_rows[-1][1],
            onset_latency=None,
            anchor_moved=False,
            skipped_marker_rows=[],
            all_rows=p_rows,
        )

    first_real_start = p_rows[first_real_idx][0]
    onset_latency = (
        first_real_start - ellie_stop if ellie_stop is not None else None
    )
    return Turn(
        session_id=session_id,
        turn_index=turn_index,
        ellie_stop=ellie_stop,
        span_start=first_real_start,
        span_stop=p_rows[-1][1],
        onset_latency=onset_latency,
        anchor_moved=first_real_idx > 0,
        skipped_marker_rows=p_rows[:first_real_idx],
        all_rows=p_rows,
    )


def group_turns_with_anchoring(
    df: pd.DataFrame,
    session_id: int,
) -> tuple[list[Turn], SessionPreTurnStats]:
    """Group a session transcript into marker-aware turns.

    Pre-turn rows (before the first Ellie row whose stripped value is non-empty)
    are discarded from turn grouping and reported separately.

    Returns (turns, pre_turn_stats).
    """
    rows = list(
        df[["start_time", "stop_time", "speaker", "value"]].itertuples(
            index=False, name=None
        )
    )

    # Locate the first real Ellie question (non-empty after strip_markers).
    first_ellie_idx: Optional[int] = None
    for i, (_, _, spk, val) in enumerate(rows):
        if spk == "Ellie" and not is_marker_only(val):
            first_ellie_idx = i
            break

    if first_ellie_idx is None:
        return [], SessionPreTurnStats(
            session_id=session_id,
            n_pre_turn_participant_rows=0,
            pre_turn_rows=[],
        )

    pre_turn_participant_rows = [
        (r[0], r[1], r[3])
        for r in rows[:first_ellie_idx]
        if r[2] == "Participant"
    ]
    pre_turn_stats = SessionPreTurnStats(
        session_id=session_id,
        n_pre_turn_participant_rows=len(pre_turn_participant_rows),
        pre_turn_rows=pre_turn_participant_rows,
    )

    turns: list[Turn] = []
    turn_index = 0
    last_ellie_stop: Optional[float] = None
    current_p_rows: list = []   # (start, stop, value)

    for start, stop, spk, val in rows[first_ellie_idx:]:
        if spk == "Ellie":
            if current_p_rows:
                turns.append(
                    _finalize_turn(session_id, turn_index, current_p_rows, last_ellie_stop)
                )
                turn_index += 1
                current_p_rows = []
            last_ellie_stop = stop
        else:  # Participant
            current_p_rows.append((start, stop, val))

    if current_p_rows:
        turns.append(
            _finalize_turn(session_id, turn_index, current_p_rows, last_ellie_stop)
        )

    return turns, pre_turn_stats


# ---------------------------------------------------------------------------
# Multi-session processing
# ---------------------------------------------------------------------------

def process_sessions(
    data_root: Path, session_ids: list[int]
) -> tuple[list[Turn], list[SessionPreTurnStats], list[int]]:
    """Process all sessions; return (all_turns, pre_turn_stats, missing_pids)."""
    all_turns: list[Turn] = []
    all_pre_stats: list[SessionPreTurnStats] = []
    missing: list[int] = []

    for pid in sorted(session_ids):
        path = _transcript_path(data_root, pid)
        if not path.exists():
            missing.append(pid)
            continue
        df = load_transcript(path)
        turns, pre_stats = group_turns_with_anchoring(df, session_id=pid)
        all_turns.extend(turns)
        all_pre_stats.append(pre_stats)

    return all_turns, all_pre_stats, missing


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_turn_report(
    turns: list[Turn],
    pre_stats: list[SessionPreTurnStats],
    missing: list[int],
    split: str,
) -> None:
    total = len(turns)
    anchor_moved = [t for t in turns if t.anchor_moved]
    all_marker_only = [t for t in turns if t.span_start is None]
    pre_turn_participant = sum(s.n_pre_turn_participant_rows for s in pre_stats)
    sessions_with_pre_rows = [s for s in pre_stats if s.n_pre_turn_participant_rows > 0]

    print("=" * 70)
    print(f"TURN-GROUPING REPORT  (split={split})")
    print("=" * 70)
    print(f"Sessions processed     : {len(pre_stats)}")
    if missing:
        print(f"Missing transcripts    : {missing}")
    print(f"Total turns found      : {total}")
    print()

    # --- Anchor-moved turns ---
    print(f"Turns w/ anchor moved past leading marker-only row(s): "
          f"{len(anchor_moved)}")
    if anchor_moved:
        sessions_am = sorted({t.session_id for t in anchor_moved})
        print(f"  Sessions affected: {sessions_am}")

    print()
    # --- All-marker-only turns ---
    print(f"Turns w/ onset_latency=None (all-marker-only participant turn): "
          f"{len(all_marker_only)}")
    if all_marker_only:
        sessions_mo = sorted({t.session_id for t in all_marker_only})
        print(f"  Sessions affected: {sessions_mo}")
        for t in all_marker_only:
            rows_str = "; ".join(
                f"[{s:.3f}-{e:.3f}] {repr(v)}" for s, e, v in t.all_rows
            )
            print(f"  pid={t.session_id} turn={t.turn_index}  rows: {rows_str}")

    print()
    # --- Pre-turn excluded rows ---
    print(f"Pre-turn participant rows excluded (session intro chatter): "
          f"{pre_turn_participant}  "
          f"({len(sessions_with_pre_rows)} sessions)")
    if sessions_with_pre_rows:
        for s in sessions_with_pre_rows:
            rows_str = "; ".join(
                f"[{r[0]:.3f}-{r[1]:.3f}] {repr(r[2])}" for r in s.pre_turn_rows
            )
            print(f"  pid={s.session_id}  n={s.n_pre_turn_participant_rows}  {rows_str}")

    print()
    # --- Spot-check sample of anchor-moved turns ---
    SAMPLE_N = 10
    sample = anchor_moved[:SAMPLE_N]
    print(f"── Spot-check sample: {len(sample)} anchor-moved turns ──")
    for t in sample:
        skipped_str = "; ".join(
            f"[{s:.3f}-{e:.3f}] {repr(v)}" for s, e, v in t.skipped_marker_rows
        )
        first_real = next(
            (r for r in t.all_rows if not is_marker_only(r[2])), None
        )
        real_str = (
            f"[{first_real[0]:.3f}] {repr(first_real[2])}"
            if first_real else "—"
        )
        print(
            f"  pid={t.session_id}  turn={t.turn_index}"
            f"  ellie_stop={t.ellie_stop:.3f}"
            f"  onset_latency={t.onset_latency:.3f}s"
        )
        print(f"    skipped : {skipped_str}")
        print(f"    anchor  : {real_str}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_split_pids(data_root: Path, split: str) -> list[int]:
    label_file = {
        "train": "train_split_Depression_AVEC2017.csv",
        "dev": "dev_split_Depression_AVEC2017.csv",
        "test": "full_test_split.csv",
    }[split]
    pid_col = "Participant_ID"
    df = pd.read_csv(Path(data_root) / "labels" / label_file)
    return df[pid_col].astype(int).tolist()


def main(data_root: Path = _DEFAULT_DATA_ROOT, split: str = "train") -> None:
    # --- Get cleaned session list ---
    raw_pids = _load_split_pids(data_root, split)
    valid_pids, cleaning_report = filter_session_ids(raw_pids, split=split)
    print(f"Session cleaning: {cleaning_report.n_before} → {cleaning_report.n_after} "
          f"(excluded: {cleaning_report.excluded})")
    print()

    # --- Step 0: bracket token scan ---
    scan = scan_bracket_tokens(data_root, valid_pids)
    print_scan_report(scan)

    # --- Turn grouping report ---
    all_turns, pre_stats, missing = process_sessions(data_root, valid_pids)
    print_turn_report(all_turns, pre_stats, missing, split=split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transcript preprocessing recon (Step 0 + turn report)")
    ap.add_argument("--data_root", type=Path, default=_DEFAULT_DATA_ROOT)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="train")
    args = ap.parse_args()
    main(args.data_root, args.split)
