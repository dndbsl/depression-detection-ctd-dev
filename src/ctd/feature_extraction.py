"""Extract 24 Chou-style CTD features per turn pair."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from constants import CTD_FEATURE_NAMES
from turn_pairing import TurnPair, Utterance


def turn_d(utts: list[Utterance]) -> float:
    return utts[-1].end - utts[0].start


def turn_u(utts: list[Utterance]) -> float:
    return sum(u.end - u.start for u in utts)


def turn_s(utts: list[Utterance]) -> float:
    if len(utts) < 2:
        return 0.0
    return sum(utts[i + 1].start - utts[i].end for i in range(len(utts) - 1))


def _safe_div(num: float, denom: float) -> float:
    if denom == 0:
        return float("nan")
    return num / denom


def _silence_gap_count(utts: list[Utterance], threshold: float = 0.2) -> int:
    if len(utts) < 2:
        return 0
    return sum(1 for i in range(len(utts) - 1) if utts[i + 1].start - utts[i].end > threshold)


def extract_turn_features(pair: TurnPair) -> dict[str, float]:
    ask_utts = sorted(pair.ask_utts, key=lambda u: u.start)
    res_utts = sorted(pair.res_utts, key=lambda u: u.start)
    next_ask_utts = sorted(pair.next_ask_utts, key=lambda u: u.start)

    ask_d = turn_d(ask_utts)
    res_d = turn_d(res_utts)
    ask_u = turn_u(ask_utts)
    res_u = turn_u(res_utts)
    ask_s = turn_s(ask_utts)
    res_s = turn_s(res_utts)

    assert abs(ask_d - ask_u - ask_s) < 1e-6, f"ask sanity failed: {ask_d}, {ask_u}, {ask_s}"
    assert abs(res_d - res_u - res_s) < 1e-6, f"res sanity failed: {res_d}, {res_u}, {res_s}"

    res_h = res_utts[0].start - ask_utts[-1].end

    if next_ask_utts:
        ask_bt = sum(1 for u in next_ask_utts if u.start < res_utts[-1].end)
    else:
        ask_bt = 0

    res_bt = sum(1 for u in res_utts if u.start < ask_utts[-1].end)
    ask_st = _silence_gap_count(ask_utts)
    res_st = _silence_gap_count(res_utts)

    feats = {
        "ask_d": ask_d,
        "res_d": res_d,
        "res_minus_ask": res_d - ask_d,
        "ask_minus_res": ask_d - res_d,
        "duration_sum": ask_d + res_d,
        "res_over_ask": _safe_div(res_d, ask_d),
        "ask_over_res": _safe_div(ask_d, res_d),
        "ask_ud": _safe_div(ask_u, ask_d),
        "ask_du": _safe_div(ask_d, ask_u),
        "res_ud": _safe_div(res_u, res_d),
        "res_du": _safe_div(res_d, res_u),
        "ask_sd": _safe_div(ask_s, ask_d),
        "ask_ds": _safe_div(ask_d, ask_s),
        "res_sd": _safe_div(res_s, res_d),
        "res_ds": _safe_div(res_d, res_s),
        "ask_su": _safe_div(ask_s, ask_u),
        "ask_us": _safe_div(ask_u, ask_s),
        "res_su": _safe_div(res_s, res_u),
        "res_us": _safe_div(res_u, res_s),
        "res_h": res_h,
        "ask_bt": float(ask_bt),
        "res_bt": float(res_bt),
        "ask_st": float(ask_st),
        "res_st": float(res_st),
    }
    assert set(feats.keys()) == set(CTD_FEATURE_NAMES)
    return feats


def extract_session_turn_features(pairs: list[TurnPair]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        feats = extract_turn_features(pair)
        row = {"session_id": pair.session_id, "turn_index": pair.turn_index}
        row.update(feats)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["session_id", "turn_index", *CTD_FEATURE_NAMES])
    return pd.DataFrame(rows)


def nan_rate_report(df: pd.DataFrame) -> dict[str, float]:
    rates = {}
    for col in CTD_FEATURE_NAMES:
        rates[col] = float(df[col].isna().mean())
    return rates


def aggregate_nan_counter(df: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    for col in CTD_FEATURE_NAMES:
        counter[col] += int(df[col].isna().sum())
    return counter
