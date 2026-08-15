"""Classification metrics and confidence-interval helpers.

Each subject yields exactly one prediction, so bootstrapping over subjects needs no
special ``conditions`` argument (every sample is already its own condition).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
from confidence_intervals import evaluate_with_conf_int  # noqa: E402


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def sensitivity_score(labels: np.ndarray, preds: np.ndarray) -> float:
    """Recall of the positive (depressed) class."""
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def specificity_score(labels: np.ndarray, preds: np.ndarray) -> float:
    """Recall of the negative (non-depressed) class."""
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def best_threshold(labels: np.ndarray, probs: np.ndarray,
                   grid: np.ndarray | None = None) -> float:
    """Decision threshold that maximizes macro-F1 on the GIVEN split.

    Intended to be called on DEV only; the returned threshold is then applied
    unchanged to test (selection never sees test). Grid matches the fusion
    pipeline (0.05..0.95, 19 points).
    """
    labels = labels.astype(int)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    scores = [f1_score(labels, (probs >= t).astype(int), average="macro", zero_division=0)
              for t in grid]
    return float(grid[int(np.argmax(scores))])


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Point estimates (no CI) -- used for per-epoch W&B logging."""
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "sensitivity": sensitivity_score(labels, preds),
        "specificity": specificity_score(labels, preds),
        "auc": _safe_auc(labels, probs),
    }


def compute_metrics_with_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    num_bootstraps: int = 1000,
    alpha: int = 5,
) -> dict:
    """All six metrics, each as {center, ci_low, ci_high} via bootstrap over subjects."""
    labels = labels.astype(int)
    preds = (probs >= threshold).astype(int)

    # metric(labels, samples) -- 'samples' carries preds (for threshold metrics) or
    # probs (for AUC). Bootstrap resamples indices consistently within each call.
    metric_defs = {
        "accuracy": (preds, lambda y, s: accuracy_score(y, (s >= 0.5).astype(int))),
        "f1": (preds, lambda y, s: f1_score(y, (s >= 0.5).astype(int), pos_label=1, zero_division=0)),
        "macro_f1": (preds, lambda y, s: f1_score(y, (s >= 0.5).astype(int), average="macro", zero_division=0)),
        "sensitivity": (preds, lambda y, s: sensitivity_score(y, (s >= 0.5).astype(int))),
        "specificity": (preds, lambda y, s: specificity_score(y, (s >= 0.5).astype(int))),
        "auc": (probs, lambda y, s: _safe_auc(y, s)),
    }

    results = {}
    for name, (samples, metric) in metric_defs.items():
        center, (lo, hi) = evaluate_with_conf_int(
            samples=samples.astype(float),
            metric=metric,
            labels=labels,
            conditions=None,
            num_bootstraps=num_bootstraps,
            alpha=alpha,
        )
        results[name] = {"center": float(center), "ci_low": float(lo), "ci_high": float(hi)}
    return results


def format_metrics_table(results: dict) -> str:
    lines = [f"{'metric':<14}{'center':>10}{'ci_low':>10}{'ci_high':>10}"]
    lines.append("-" * 44)
    for name, v in results.items():
        lines.append(f"{name:<14}{v['center']:>10.4f}{v['ci_low']:>10.4f}{v['ci_high']:>10.4f}")
    return "\n".join(lines)
