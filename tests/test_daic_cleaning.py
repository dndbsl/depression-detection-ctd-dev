"""Guards the DAIC-WOZ session-count claim (SLT-13) against silent drift.

189 sessions total; nine documented integrity cases are excluded, one (409)
is relabeled and kept -> 180 sessions, split 102/33/45 across train/dev/test.
See docs/HANDOFF.md and common/daic_cleaning.py::KNOWN_ERRORS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT / "src" / "ctd"), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.daic_cleaning import excluded_ids, corrected_labels, union_exclusion_report

EXPECTED_EXCLUDED = {318, 321, 341, 362, 373, 444, 451, 458, 480}


def test_nine_sessions_excluded():
    assert len(excluded_ids()) == 9


def test_exact_excluded_set():
    assert excluded_ids() == EXPECTED_EXCLUDED


def test_409_relabeled_not_excluded():
    assert 409 in corrected_labels()
    assert 409 not in excluded_ids()


@pytest.mark.skipif(
    "DAIC_WOZ_ROOT" not in os.environ, reason="requires DAIC-WOZ dataset on disk"
)
def test_per_split_kept_counts():
    report = union_exclusion_report(Path(os.environ["DAIC_WOZ_ROOT"]))
    per_split = report["per_split"]
    assert per_split["train"]["n_sessions_after"] == 102
    assert per_split["dev"]["n_sessions_after"] == 33
    assert per_split["test"]["n_sessions_after"] == 45
    n_before_total = sum(s["n_sessions_before"] for s in per_split.values())
    assert n_before_total == 189
