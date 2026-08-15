"""Chou-style Ask/Res turn-pair construction from DAIC-WOZ transcripts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: str
    value: str = ""


@dataclass
class TurnPair:
    session_id: int
    turn_index: int
    ask_utts: list[Utterance]
    res_utts: list[Utterance]
    next_ask_utts: list[Utterance]


def _df_to_utterances(df: pd.DataFrame) -> list[Utterance]:
    has_value = "value" in df.columns
    return [
        Utterance(
            start=row.start_time,
            end=row.stop_time,
            speaker=row.speaker,
            value=(getattr(row, "value", "") if has_value else ""),
        )
        for row in df.itertuples(index=False)
    ]


def _speaker_runs(utts: list[Utterance]) -> list[tuple[str, list[Utterance]]]:
    if not utts:
        return []
    runs: list[tuple[str, list[Utterance]]] = []
    current_speaker = utts[0].speaker
    current_run = [utts[0]]
    for u in utts[1:]:
        if u.speaker == current_speaker:
            current_run.append(u)
        else:
            runs.append((current_speaker, current_run))
            current_speaker = u.speaker
            current_run = [u]
    runs.append((current_speaker, current_run))
    return runs


def build_turn_pairs(session_id: int, df: pd.DataFrame) -> list[TurnPair]:
    """Segment transcript into alternating Ellie/Participant runs and pair Ask/Res."""
    utts = _df_to_utterances(df)
    runs = _speaker_runs(utts)

    # Discard leading Participant run before any Ellie utterance.
    while runs and runs[0][0] != "Ellie":
        runs.pop(0)

    # Collect alternating Ask (Ellie) / Res (Participant) sequences.
    ask_res_sequences: list[tuple[list[Utterance], list[Utterance]]] = []
    i = 0
    while i < len(runs):
        speaker, run_utts = runs[i]
        if speaker != "Ellie":
            i += 1
            continue
        ask_utts = run_utts
        if i + 1 >= len(runs) or runs[i + 1][0] != "Participant":
            break  # trailing incomplete Ask — discard
        res_utts = runs[i + 1][1]
        ask_res_sequences.append((ask_utts, res_utts))
        i += 2

    pairs: list[TurnPair] = []
    for turn_index, (ask_utts, res_utts) in enumerate(ask_res_sequences):
        if turn_index + 1 < len(ask_res_sequences):
            next_ask_utts = ask_res_sequences[turn_index + 1][0]
        else:
            next_ask_utts = []
        pairs.append(
            TurnPair(
                session_id=session_id,
                turn_index=turn_index,
                ask_utts=ask_utts,
                res_utts=res_utts,
                next_ask_utts=next_ask_utts,
            )
        )
    return pairs
