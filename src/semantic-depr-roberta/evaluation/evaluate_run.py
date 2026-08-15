"""Checkpoint evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from datasets.daic import load_bags
from evaluation.metrics import compute_metrics
from training.config import ExperimentConfig
from training.runner import load_model_from_checkpoint, predict_records


def evaluate_checkpoint(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | Path,
    split: str,
    output_path: str | Path,
    threshold: float | None = None,
    allow_cpu: bool = False,
) -> dict:
    """Evaluate a saved checkpoint on one preprocessed split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.training.require_cuda and device.type != "cuda" and not allow_cpu:
        raise RuntimeError("CUDA is required by config; pass allow_cpu=True only for debug.")
    model, payload = load_model_from_checkpoint(checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained(config.encoder)
    records = load_bags(config.data.processed_dir, split)
    selected_threshold = float(payload["threshold"] if threshold is None else threshold)
    predictions, attention, instances, _ = predict_records(
        model,
        tokenizer,
        records,
        config,
        threshold=selected_threshold,
        device=device,
        criterion=None,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    attention.to_csv(output_path.with_name(output_path.stem + "_attention.csv"), index=False)
    instances.to_csv(output_path.with_name(output_path.stem + "_instances.csv"), index=False)
    metrics = compute_metrics(
        predictions["label"], predictions["probability"], threshold=selected_threshold
    ).to_dict()
    return metrics

