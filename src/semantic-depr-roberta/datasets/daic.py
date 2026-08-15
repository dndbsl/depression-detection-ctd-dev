"""DAIC-WOZ transcript preprocessing for text-only MIL bags."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from common.daic_cleaning import apply_cleaning
from training.config import DataConfig


SPLIT_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",
}


@dataclass(frozen=True)
class InstanceRecord:
    """One participant response segment."""

    participant_id: int
    split: str
    instance_index: int
    text: str
    label: int
    phq_score: int
    n_words: int
    n_chars: int


@dataclass(frozen=True)
class BagRecord:
    """One DAIC-WOZ participant interview represented as a MIL bag."""

    participant_id: int
    split: str
    label: int
    phq_score: int
    instances: list[str]

    @property
    def n_instances(self) -> int:
        """Return the number of response instances in the bag."""
        return len(self.instances)


def _normalize_label_frame(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "participant_ID": "Participant_ID",
        "PHQ_Score": "PHQ8_Score",
        "PHQ_Binary": "PHQ8_Binary",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"Participant_ID", "PHQ8_Score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "PHQ8_Binary" not in df.columns:
        df["PHQ8_Binary"] = (df["PHQ8_Score"] >= 10).astype(int)
    out = df[["Participant_ID", "PHQ8_Binary", "PHQ8_Score"]].copy()
    out["Participant_ID"] = out["Participant_ID"].astype(int)
    out["PHQ8_Score"] = out["PHQ8_Score"].astype(int)
    out["PHQ8_Binary"] = (out["PHQ8_Score"] >= 10).astype(int)
    out["split"] = split
    out = out.set_index("Participant_ID", drop=True)
    out.index.name = "pid"
    return out


def load_clean_labels(labels_root: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load official split labels and apply the shared DAIC cleaning policy."""
    labels_root = Path(labels_root)
    split_frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for split, filename in SPLIT_FILES.items():
        raw = _normalize_label_frame(labels_root / filename, split)
        cleaned, report = apply_cleaning(raw, split=split)
        cleaned["label"] = (cleaned["PHQ8_Score"] >= 10).astype(int)
        cleaned["participant_id"] = cleaned.index.astype(int)
        split_frames[split] = cleaned
        reports[split] = report.to_dict()
    _assert_disjoint_splits(split_frames)
    return split_frames, reports


def _assert_disjoint_splits(split_frames: dict[str, pd.DataFrame]) -> None:
    seen: dict[int, str] = {}
    overlaps: list[tuple[int, str, str]] = []
    for split, frame in split_frames.items():
        for pid in frame.index.astype(int):
            if pid in seen:
                overlaps.append((pid, seen[pid], split))
            seen[pid] = split
    if overlaps:
        raise ValueError(f"Participants appear in multiple splits: {overlaps}")


def transcript_path(data_root: str | Path, participant_id: int) -> Path:
    """Return the expected transcript path for a participant."""
    return Path(data_root) / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"


def segment_transcript(path: str | Path) -> list[str]:
    """Segment a transcript into participant-only response instances.

    Ellie rows are used strictly as instance boundaries and are never included
    in model input. Participant rows before the first Ellie row are discarded.
    """
    frame = pd.read_csv(path, sep="\t")
    required = {"speaker", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing transcript columns: {missing}")

    instances: list[str] = []
    buffer: list[str] = []
    seen_ellie = False

    def flush() -> None:
        nonlocal buffer
        text = " ".join(part.strip() for part in buffer if str(part).strip()).strip()
        if text:
            instances.append(text)
        buffer = []

    for row in frame.itertuples(index=False):
        speaker = str(getattr(row, "speaker", "")).strip()
        raw_value = getattr(row, "value", "")
        value = "" if pd.isna(raw_value) else str(raw_value).strip()
        if speaker == "Ellie":
            if seen_ellie:
                flush()
            seen_ellie = True
            buffer = []
        elif speaker == "Participant" and seen_ellie and value:
            buffer.append(value)

    if seen_ellie:
        flush()
    return instances


def prepare_dataset(config: DataConfig) -> dict:
    """Create cleaned MIL bags and preprocessing summaries on disk."""
    processed_dir = Path(config.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_frames, cleaning_reports = load_clean_labels(config.labels_root)

    bag_rows: list[dict] = []
    instance_rows: list[dict] = []
    stats_rows: list[dict] = []
    empty_bags: list[tuple[str, int]] = []

    for split, labels in split_frames.items():
        for row in labels.sort_index().itertuples():
            pid = int(row.Index)
            instances = segment_transcript(transcript_path(config.data_root, pid))
            if not instances:
                empty_bags.append((split, pid))
                continue
            label = int(row.label)
            phq_score = int(row.PHQ8_Score)
            bag = BagRecord(
                participant_id=pid,
                split=split,
                label=label,
                phq_score=phq_score,
                instances=instances,
            )
            bag_rows.append(asdict(bag))
            word_counts = [len(text.split()) for text in instances]
            stats_rows.append(
                {
                    "split": split,
                    "participant_id": pid,
                    "label": label,
                    "phq_score": phq_score,
                    "n_instances": len(instances),
                    "avg_instance_words": float(sum(word_counts) / len(word_counts)),
                    "total_words": int(sum(word_counts)),
                }
            )
            for idx, text in enumerate(instances):
                instance_rows.append(
                    asdict(
                        InstanceRecord(
                            participant_id=pid,
                            split=split,
                            instance_index=idx,
                            text=text,
                            label=label,
                            phq_score=phq_score,
                            n_words=len(text.split()),
                            n_chars=len(text),
                        )
                    )
                )

    if empty_bags:
        raise ValueError(
            "Empty bags remain after preprocessing despite shared cleaning: "
            f"{empty_bags}"
        )

    bags_path = processed_dir / "bags.jsonl"
    with bags_path.open("w", encoding="utf-8") as handle:
        for row in bag_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    instances = pd.DataFrame(instance_rows)
    stats = pd.DataFrame(stats_rows)
    instances.to_csv(processed_dir / "instances.csv", index=False)
    stats.to_csv(processed_dir / "transcript_statistics.csv", index=False)

    summary = {
        "data_root": str(config.data_root),
        "labels_root": str(config.labels_root),
        "max_length": config.max_length,
        "cleaning_reports": cleaning_reports,
        "split_summary": {},
        "n_instances_total": int(len(instances)),
    }
    for split in SPLIT_FILES:
        split_stats = stats[stats["split"] == split]
        split_instances = instances[instances["split"] == split]
        summary["split_summary"][split] = {
            "participants": int(len(split_stats)),
            "positives": int((split_stats["label"] == 1).sum()),
            "negatives": int((split_stats["label"] == 0).sum()),
            "instances": int(len(split_instances)),
            "avg_instances_per_participant": float(split_stats["n_instances"].mean()),
            "avg_instance_words": float(split_instances["n_words"].mean()),
        }
    with (processed_dir / "preprocessing_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_bags(processed_dir: str | Path, split: str | None = None) -> list[BagRecord]:
    """Load preprocessed bags from JSONL."""
    rows: list[BagRecord] = []
    with (Path(processed_dir) / "bags.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if split is None or raw["split"] == split:
                rows.append(
                    BagRecord(
                        participant_id=int(raw["participant_id"]),
                        split=str(raw["split"]),
                        label=int(raw["label"]),
                        phq_score=int(raw["phq_score"]),
                        instances=list(raw["instances"]),
                    )
                )
    return rows


def class_counts(records: Iterable[BagRecord]) -> tuple[int, int]:
    """Return negative and positive participant counts."""
    labels = [record.label for record in records]
    positives = int(sum(labels))
    negatives = int(len(labels) - positives)
    return negatives, positives

