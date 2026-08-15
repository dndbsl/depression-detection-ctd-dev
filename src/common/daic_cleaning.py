"""Shared DAIC-WOZ session cleaning (single source of truth).

Provenance: documented errors in the upstream daic_woz_process README
(https://github.com/adbailey1/daic_woz_process), as vendored in
projects/depression/AVEC2016/daic_woz_process/README.md.

These are *our* operational decisions on how to handle each case — not a verbatim
copy of the upstream preprocessing pipeline (which trims/corrects rather than
drops). We exclude entire sessions where transcript timestamps are unreliable
for onset-silence work, or where Ellie turns are missing.

Fushimi2026 audio-proxy exclusions {338, 347, 354, 362} are documented
separately; only 362 overlaps README errors and is excluded via KNOWN_ERRORS.
338, 347, 354 are retained unless they appear in KNOWN_ERRORS (they do not).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

README_URL = "https://github.com/adbailey1/daic_woz_process"
PHQ8_DEPRESSED_THRESHOLD = 10

# session_id -> metadata. action: "exclude" | "relabel"
KNOWN_ERRORS: dict[int, dict[str, str]] = {
    318: {
        "reason": "transcript out of sync with audio",
        "action": "exclude",
        "source": "README",
    },
    321: {
        "reason": "transcript out of sync with audio",
        "action": "exclude",
        "source": "README",
    },
    341: {
        "reason": "transcript out of sync with audio",
        "action": "exclude",
        "source": "README",
    },
    362: {
        "reason": "transcript out of sync with audio (also Fushimi shortest-audio proxy)",
        "action": "exclude",
        "source": "README+Fushimi overlap",
    },
    451: {
        "reason": "virtual-agent (Ellie) transcript missing",
        "action": "exclude",
        "source": "README",
    },
    458: {
        "reason": "virtual-agent (Ellie) transcript missing",
        "action": "exclude",
        "source": "README",
    },
    480: {
        "reason": "virtual-agent (Ellie) transcript missing",
        "action": "exclude",
        "source": "README",
    },
    373: {
        "reason": "long interview interruption",
        "action": "exclude",
        "source": "README",
    },
    444: {
        "reason": "long interview interruption",
        "action": "exclude",
        "source": "README",
    },
    409: {
        "reason": "PHQ-8 score 10 but binary label 0 in official file",
        "action": "relabel",
        "source": "README",
    },
}

# Documented Fushimi proxy IDs not excluded unless overlapping README list.
FUSHIMI_AUDIO_PROXY_DOCUMENTED = [338, 347, 354, 362]

SPLIT_LABEL_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",
}


@dataclass
class CleaningReport:
    split: str
    n_before: int
    n_after: int
    excluded: list[int] = field(default_factory=list)
    excluded_reasons: dict[int, str] = field(default_factory=dict)
    relabeled: dict[int, dict] = field(default_factory=dict)
    n_dep_before: int = 0
    n_dep_after: int = 0

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "excluded": self.excluded,
            "excluded_reasons": self.excluded_reasons,
            "relabeled": self.relabeled,
            "n_dep_before": self.n_dep_before,
            "n_dep_after": self.n_dep_after,
        }


def excluded_ids(split: str | None = None) -> set[int]:
    """Session IDs to drop entirely. `split` is accepted for API symmetry (global set)."""
    _ = split
    return {sid for sid, meta in KNOWN_ERRORS.items() if meta["action"] == "exclude"}


def corrected_labels() -> dict[int, int]:
    """Explicit PHQ8_Binary corrections after cleaning."""
    return {
        sid: PHQ8_DEPRESSED_THRESHOLD  # binary 1
        for sid, meta in KNOWN_ERRORS.items()
        if meta["action"] == "relabel"
    }


def _load_split_index(data_root: Path, split: str) -> pd.Index:
    fname = SPLIT_LABEL_FILES[split]
    df = pd.read_csv(Path(data_root) / "labels" / fname)
    if split == "test":
        df = df.rename(columns={"Participant_ID": "pid"})
    else:
        df = df.rename(columns={"Participant_ID": "pid"})
    return pd.Index(df["pid"].astype(int).tolist())


def apply_label_corrections(labels: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit relabels on a labels frame indexed by participant id."""
    out = labels.copy()
    for pid, binary in corrected_labels().items():
        if pid in out.index:
            out.loc[pid, "PHQ8_Binary"] = int(binary >= PHQ8_DEPRESSED_THRESHOLD)
    # defensive: always follow PHQ-8 >= 10 rule
    if "PHQ8_Score" in out.columns:
        out["PHQ8_Binary"] = (out["PHQ8_Score"] >= PHQ8_DEPRESSED_THRESHOLD).astype(int)
    return out


def apply_cleaning(
    labels: pd.DataFrame,
    *,
    split: str,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop excluded sessions and apply relabels. `labels` indexed by participant id."""
    labels = apply_label_corrections(labels)
    excl = excluded_ids(split)
    present_excl = sorted(int(p) for p in labels.index if int(p) in excl)
    kept = labels[~labels.index.isin(excl)].copy()

    relabeled = {}
    for pid in corrected_labels():
        if pid in labels.index and pid in kept.index:
            relabeled[int(pid)] = {
                "PHQ8_Score": int(labels.loc[pid, "PHQ8_Score"]),
                "PHQ8_Binary": int(kept.loc[pid, "PHQ8_Binary"]),
            }

    report = CleaningReport(
        split=split,
        n_before=int(len(labels)),
        n_after=int(len(kept)),
        excluded=present_excl,
        excluded_reasons={int(p): KNOWN_ERRORS[int(p)]["reason"] for p in present_excl},
        relabeled=relabeled,
        n_dep_before=int((labels["PHQ8_Binary"] == 1).sum()),
        n_dep_after=int((kept["PHQ8_Binary"] == 1).sum()),
    )
    return kept, report


def filter_session_ids(pids: Iterable[int], *, split: str) -> tuple[list[int], CleaningReport]:
    """Filter a list of session IDs, returning kept IDs and a report."""
    pids = [int(p) for p in sorted(set(pids))]
    labels = pd.DataFrame(
        {"PHQ8_Binary": [0] * len(pids), "PHQ8_Score": [0] * len(pids)},
        index=pd.Index(pids, name="pid"),
    )
    _, report = apply_cleaning(labels, split=split)
    kept = [p for p in pids if p not in excluded_ids(split)]
    return kept, report


def union_exclusion_report(data_root: Path) -> dict:
    """Per-split exclusion summary for all official splits."""
    excl = excluded_ids()
    out = {"union_excluded": sorted(excl), "per_split": {}, "fushimi_not_excluded": [
        p for p in FUSHIMI_AUDIO_PROXY_DOCUMENTED if p not in excl
    ]}
    for split in ("train", "dev", "test"):
        idx = _load_split_index(data_root, split)
        in_split = sorted(int(p) for p in idx if int(p) in excl)
        out["per_split"][split] = {
            "excluded_in_split": in_split,
            "reasons": {int(p): KNOWN_ERRORS[int(p)]["reason"] for p in in_split},
            "n_sessions_before": len(idx),
            "n_sessions_after": len(idx) - len(in_split),
        }
    return out
