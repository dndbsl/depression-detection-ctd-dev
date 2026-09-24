#!/usr/bin/env python3
"""PDCH label construction (`MC-07`) and subject grouping (`MC-08`).

Label decisions, fixed in `docs/preregistration-pdch.md` before any feature was
joined to any label:

* **Primary (classification):** ``HAMD17_total >= 17`` -- moderate-or-worse.
  27/62 = 44% prevalence, close to DAIC-WOZ's ~30%. The standard clinical
  cutoff of 8 is unusable here: this is an inpatient cohort, so it marks 85%
  of the sample positive and macro-F1 would hinge on 9 negatives.
* **Secondary (regression):** the HAMD-17 total itself.
* PDCH labels never share a column with DAIC-WOZ's PHQ-8. HAMD-17 is
  clinician-rated, PHQ-8 is self-report; pooling or rescaling them across
  instruments and raters would be unjustifiable.

Sessions with a blank total are dropped from both supervised tasks. They still
flow through the stitching/VAD pipeline, where they serve as unlabelled
validation of feature extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pdch_adapter import PDCH_ROOT, subject_of  # noqa: E402

HAMD_XLSX = PDCH_ROOT / "HAMD_annotation_en.xlsx"
HAMD_SHEET = "Sheet1"
HAMD17_CUTOFF = 17
ITEM_COLUMNS = list(range(1, 18))
NOT_ASSESSED_CODE = 9  # sentinel in item columns; the `total` column excludes it


def load_pdch_labels(path: Path = HAMD_XLSX) -> pd.DataFrame:
    """One row per PDCH session with a HAMD-17 total.

    Columns: ``session_id``, ``subject_id``, ``HAMD17_total``, ``HAMD17_ge17``.

    The shipped ``total`` column is authoritative and is used as-is. Note that
    7 of the 62 scored rows code item 14 (genital symptoms, a 0-2 item) as
    ``9``, a not-assessed sentinel; ``total`` already excludes those, which is
    why summing the item columns disagrees with ``total`` on exactly those 7
    rows. Re-deriving the total from items would silently add 9 points to each.
    """
    df = pd.read_excel(path, sheet_name=HAMD_SHEET)
    df = df.rename(columns={"Serial": "session_id", "total": "HAMD17_total"})
    df["session_id"] = df["session_id"].astype(str).str.strip()

    n_rows = len(df)
    df = df[df["HAMD17_total"].notna()].copy()
    df["HAMD17_total"] = df["HAMD17_total"].astype(float)
    df["HAMD17_ge17"] = (df["HAMD17_total"] >= HAMD17_CUTOFF).astype(int)
    df["subject_id"] = df["session_id"].map(subject_of)

    out = df[["session_id", "subject_id", "HAMD17_total", "HAMD17_ge17"]]
    out = out.sort_values("session_id").reset_index(drop=True)
    assert out["session_id"].is_unique
    assert len(out) <= n_rows
    return out


def label_summary(labels: pd.DataFrame) -> dict:
    t = labels["HAMD17_total"]
    subj = labels.groupby("subject_id").size()
    return {
        "n_sessions": int(len(labels)),
        "n_subjects": int(labels["subject_id"].nunique()),
        "n_subjects_with_two_sessions": int((subj == 2).sum()),
        "hamd17_total": {
            "min": float(t.min()), "max": float(t.max()),
            "mean": float(t.mean()), "sd": float(t.std(ddof=1)),
        },
        "primary_label": f"HAMD17_total >= {HAMD17_CUTOFF}",
        "n_positive": int(labels["HAMD17_ge17"].sum()),
        "prevalence": float(labels["HAMD17_ge17"].mean()),
        "prevalence_at_cutoff_8_for_reference": float((t >= 8).mean()),
    }


if __name__ == "__main__":
    lab = load_pdch_labels()
    import json
    print(json.dumps(label_summary(lab), indent=2))
