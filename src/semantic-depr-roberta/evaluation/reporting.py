"""Result aggregation and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_COLUMNS = [
    "Experiment",
    "Encoder",
    "Training",
    "Seed",
    "Macro F1",
    "AUROC",
    "Accuracy",
    "Precision",
    "Recall",
    "Sensitivity",
    "Specificity",
    "Threshold",
]


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional tabulate dependency."""
    if frame.empty:
        return "_No rows._"
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_run_metrics(results_dir: str | Path) -> pd.DataFrame:
    """Load per-seed test metrics from run output directories."""
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(results_dir).glob("runs/*/test_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "Experiment": payload["experiment_name"],
                "Encoder": payload["encoder"],
                "Training": payload["training_strategy"],
                "Seed": payload["seed"],
                "Macro F1": payload["macro_f1"],
                "AUROC": payload["auroc"],
                "Accuracy": payload["accuracy"],
                "Precision": payload["precision"],
                "Recall": payload["recall"],
                "Sensitivity": payload["sensitivity"],
                "Specificity": payload["specificity"],
                "Threshold": payload["threshold"],
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def aggregate_results(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Create per-seed, mean/std, and ranking tables."""
    per_seed = load_run_metrics(results_dir)
    if per_seed.empty:
        mean_std = pd.DataFrame()
        ranking = pd.DataFrame()
    else:
        numeric_cols = [col for col in RESULT_COLUMNS if col not in {"Experiment", "Encoder", "Training", "Seed"}]
        grouped = per_seed.groupby(["Experiment", "Encoder", "Training"], dropna=False)
        rows = []
        for keys, group in grouped:
            row: dict[str, Any] = {
                "Experiment": keys[0],
                "Encoder": keys[1],
                "Training": keys[2],
                "Seeds": ",".join(str(seed) for seed in sorted(group["Seed"].astype(int))),
            }
            for col in numeric_cols:
                row[f"{col} Mean"] = float(group[col].mean())
                row[f"{col} Std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
                row[f"{col} Mean ± Std"] = f"{row[f'{col} Mean']:.4f} ± {row[f'{col} Std']:.4f}"
            rows.append(row)
        mean_std = pd.DataFrame(rows)
        ranking = mean_std.sort_values("Macro F1 Mean", ascending=False).reset_index(drop=True)
        ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return {"per_seed": per_seed, "mean_std": mean_std, "ranking": ranking}


def write_result_tables(results_dir: str | Path) -> dict[str, Path]:
    """Write publication-ready CSV and Markdown result tables."""
    results_dir = Path(results_dir)
    tables = aggregate_results(results_dir)
    out_paths: dict[str, Path] = {}
    for name, frame in tables.items():
        csv_path = results_dir / f"{name}_results.csv"
        md_path = results_dir / f"{name}_results.md"
        frame.to_csv(csv_path, index=False)
        md_path.write_text(markdown_table(frame), encoding="utf-8")
        out_paths[f"{name}_csv"] = csv_path
        out_paths[f"{name}_md"] = md_path
    return out_paths


def write_assumptions(project_root: str | Path) -> Path:
    """Write the implementation assumptions document."""
    path = Path(project_root) / "ASSUMPTIONS.md"
    path.write_text(
        "\n".join(
            [
                "# Assumptions",
                "",
                "- The `acoustic-depr` conda environment is the runtime target; code remains Python 3.10/3.11 compatible.",
                "- `full_test_split.csv` is the labeled test split; `test_split_Depression_AVEC2017.csv` is metadata-only.",
                "- `common/daic_cleaning.py` is the single source of truth for DAIC-WOZ exclusions and relabeling.",
                "- Binary labels are always derived from PHQ score using `PHQ >= 10` after label normalization.",
                "- Ellie text is never included in model input and is only used as a participant-response boundary.",
                "- The final checkpoint is evaluated on train/dev only; test is evaluated once with the best dev checkpoint.",
                "- No learning-rate scheduler is used; the learning rate is constant and logged every epoch.",
                "- Dropout is `0.1`, matching RoBERTa's common classifier-head default.",
                "- Instance probabilities are classifier scores on individual CLS embeddings for interpretation only.",
                "- W&B online logging is required for full runs; offline fallback is intentionally not used.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_report(project_root: str | Path, results_dir: str | Path, processed_dir: str | Path) -> Path:
    """Write REPORT.md from preprocessing and result artifacts."""
    project_root = Path(project_root)
    results_dir = Path(results_dir)
    processed_dir = Path(processed_dir)
    report_path = project_root / "REPORT.md"
    preprocessing = {}
    preprocessing_path = processed_dir / "preprocessing_report.json"
    if preprocessing_path.exists():
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    tables = aggregate_results(results_dir)
    split_summary = preprocessing.get("split_summary", {})
    lines = [
        "# Semantic RoBERTa MIL DAIC-WOZ Report",
        "",
        "## Project Overview",
        "",
        "Text-only participant-level binary depression detection using RoBERTa encoders and trainable attention MIL pooling.",
        "",
        "## Dataset Summary",
        "",
    ]
    for split, values in split_summary.items():
        lines.append(
            f"- {split}: {values.get('participants')} participants, "
            f"{values.get('positives')} positive, {values.get('negatives')} negative, "
            f"{values.get('instances')} instances."
        )
    lines.extend(
        [
            "",
            "## Preprocessing Pipeline",
            "",
            "Official train/dev/test splits are loaded, shared DAIC cleaning is applied, Ellie rows define boundaries, and only participant language is retained as model input.",
            "",
            "## Transcript Statistics",
            "",
        ]
    )
    for split, values in split_summary.items():
        lines.append(
            f"- {split}: average instances per participant "
            f"{values.get('avg_instances_per_participant', 0):.2f}; average instance length "
            f"{values.get('avg_instance_words', 0):.2f} words."
        )
    lines.extend(
        [
            "",
            "## Training Configuration",
            "",
            "All experiments use AdamW, learning rate 2e-5, weight decay 0.01, BCEWithLogitsLoss with train-split positive weighting, and 500 epochs without early stopping.",
            "",
            "## Experiment Results",
            "",
            markdown_table(tables["ranking"]),
            "",
            "## Discussion",
            "",
            "The ranking table summarizes the strongest text-only RoBERTa MIL baseline across seeds. Attention weights are saved for qualitative inspection.",
            "",
            "## Limitations",
            "",
            "This baseline intentionally excludes audio, video, timing, and handcrafted linguistic features. Results depend on the official split size and DAIC-WOZ label distribution.",
            "",
            "## Future Work",
            "",
            "Future multimodal branches can add audio or video encoders while reusing participant-level MIL outputs and reporting infrastructure.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

