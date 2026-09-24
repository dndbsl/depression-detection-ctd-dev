"""Authoritative Ask/Cross/Response grouping of the 24 CTD features.

Single source of truth for `SLT-04` (interpretability), `SLT-05` (ablation),
and later `MC-02` (multi-corpus feature tiers). Subsetting is always by
*column selection* against `constants.CTD_FEATURE_NAMES` — this module never
mutates that list.

Note on `ask_bt`: grouped ask-side because it measures *interviewer*
behaviour (interrupting the participant), but it is computed using the
response end time, so it is not derivable from ask timestamps alone. This
matters for the corpus tiers later (`MC-02`).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import CTD_FEATURE_NAMES  # noqa: E402

ASK_ONLY: list[str] = [
    "ask_d",
    "ask_ud",
    "ask_du",
    "ask_sd",
    "ask_ds",
    "ask_su",
    "ask_us",
    "ask_st",
    "ask_bt",
]

CROSS: list[str] = [
    "res_minus_ask",
    "ask_minus_res",
    "duration_sum",
    "res_over_ask",
    "ask_over_res",
    "res_h",
    "res_bt",
]

RES_ONLY: list[str] = [
    "res_d",
    "res_ud",
    "res_du",
    "res_sd",
    "res_ds",
    "res_su",
    "res_us",
    "res_st",
]

assert len(ASK_ONLY) == 9
assert len(CROSS) == 7
assert len(RES_ONLY) == 8
assert set(ASK_ONLY) | set(CROSS) | set(RES_ONLY) == set(CTD_FEATURE_NAMES)
assert not (set(ASK_ONLY) & set(CROSS) & set(RES_ONLY))
assert len(ASK_ONLY) + len(CROSS) + len(RES_ONLY) == len(CTD_FEATURE_NAMES)

# Named ablation configs. Order follows CTD_FEATURE_NAMES for reproducibility.
FEATURE_CONFIGS: dict[str, list[str]] = {
    "all24": list(CTD_FEATURE_NAMES),
    "no_ask": [f for f in CTD_FEATURE_NAMES if f in set(CROSS) | set(RES_ONLY)],
    "res_only": [f for f in CTD_FEATURE_NAMES if f in RES_ONLY],
    "ask_only": [f for f in CTD_FEATURE_NAMES if f in ASK_ONLY],
    "no_cross": [f for f in CTD_FEATURE_NAMES if f in set(ASK_ONLY) | set(RES_ONLY)],
}

assert len(FEATURE_CONFIGS["all24"]) == 24
assert len(FEATURE_CONFIGS["no_ask"]) == 15
assert len(FEATURE_CONFIGS["res_only"]) == 8
assert len(FEATURE_CONFIGS["ask_only"]) == 9
assert len(FEATURE_CONFIGS["no_cross"]) == 17


def group_of(feature_name: str) -> str:
    """Return 'ask', 'cross', or 'res' for a CTD feature name."""
    if feature_name in ASK_ONLY:
        return "ask"
    if feature_name in CROSS:
        return "cross"
    if feature_name in RES_ONLY:
        return "res"
    raise KeyError(f"{feature_name!r} is not one of the 24 CTD features")
