"""Shared constants for CTD feature analysis."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEPRESSION_ROOT = PROJECT_ROOT.parent
# DAIC-WOZ raw dataset (external, licensed). Override with the DAIC_WOZ_ROOT env var.
DEFAULT_DATA_ROOT = Path(os.environ.get("DAIC_WOZ_ROOT", str(Path.home() / "datasets" / "DAIC-WOZ")))
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Exactly 24 named features — single source of truth for all downstream code.
CTD_FEATURE_NAMES: list[str] = [
    "ask_d",
    "res_d",
    "res_minus_ask",
    "ask_minus_res",
    "duration_sum",
    "res_over_ask",
    "ask_over_res",
    "ask_ud",
    "ask_du",
    "res_ud",
    "res_du",
    "ask_sd",
    "ask_ds",
    "res_sd",
    "res_ds",
    "ask_su",
    "ask_us",
    "res_su",
    "res_us",
    "res_h",
    "ask_bt",
    "res_bt",
    "ask_st",
    "res_st",
]

assert len(CTD_FEATURE_NAMES) == 24
