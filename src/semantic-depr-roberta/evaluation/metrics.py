"""Metrics and threshold selection for binary depression detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class MetricResult:
    """Participant-level binary classification metrics."""

    macro_f1: float
    auroc: float
    accuracy: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    threshold: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, float | list[list[int]]]:
        """Return metrics as a serializable dictionary."""
        return {
            "macro_f1": self.macro_f1,
            "auroc": self.auroc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
        }


def select_threshold(labels: list[int] | np.ndarray, probabilities: list[float] | np.ndarray) -> float:
    """Select the dev threshold that maximizes Macro F1."""
    y_true = np.asarray(labels, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probs)))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score or (
            np.isclose(score, best_score)
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    threshold: float,
) -> MetricResult:
    """Compute all required participant-level metrics."""
    y_true = np.asarray(labels, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auroc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auroc = float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return MetricResult(
        macro_f1=float(f1_score(y_true, preds, average="macro", zero_division=0)),
        auroc=auroc,
        accuracy=float(accuracy_score(y_true, preds)),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        sensitivity=sensitivity,
        specificity=specificity,
        threshold=float(threshold),
        confusion_matrix=cm.astype(int).tolist(),
    )


def roc_points(labels: list[int], probabilities: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ROC curve points."""
    return roc_curve(np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float))


def pr_points(labels: list[int], probabilities: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision-recall curve points."""
    return precision_recall_curve(
        np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    )

