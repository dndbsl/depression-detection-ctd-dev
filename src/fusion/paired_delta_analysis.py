#!/usr/bin/env python3
"""Paired uncertainty/error analysis for the deployed RoBERTa+CTD fusion.

This is an analysis-only script: it does not train a new neural model or tune a
new fusion rule. It reconstructs the canonical CTD probabilities from the
train/dev/test CTD embeddings (written by ``fusion/extract_ctd_roberta.py``),
reads the deployed RoBERTa probabilities (written by the seed-43 frozen run in
``src/semantic-depr-roberta``), and evaluates the already-selected
wconvex[RoBERTa+CTD] rule (paired bootstrap deltas + prediction-change counts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


SRC_ROOT = Path(__file__).resolve().parents[1]
EMB = SRC_ROOT / "ctd" / "outputs" / "fusion"
ROBERTA = (SRC_ROOT / "semantic-depr-roberta" / "results" / "runs"
           / "roberta-large_frozen_seed43")

ROBERTA_THR = 0.5238006711006165
CTD_THR = 0.5
FUSION_THR = 0.55
ROBERTA_WEIGHT = 0.3
CTD_WEIGHT = 0.7
N_BOOT = 2000


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro"))


def read_roberta(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = "dev_predictions_best.csv" if split == "dev" else "test_predictions.csv"
    df = pd.read_csv(ROBERTA / name).sort_values("participant_id")
    return (
        df["participant_id"].to_numpy(dtype=int),
        df["label"].to_numpy(dtype=int),
        df["probability"].to_numpy(dtype=float),
    )


def ctd_probs() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train = np.load(EMB / "ctd_train.npz")
    dev = np.load(EMB / "ctd_dev.npz")
    test = np.load(EMB / "ctd_test.npz")

    x_train, y_train = train["emb"], train["label"].astype(int)
    x_dev, y_dev = dev["emb"], dev["label"].astype(int)

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
    clf.fit(scaler.transform(x_train), y_train)
    dev_prob = clf.predict_proba(scaler.transform(x_dev))[:, 1]

    x_train_dev = np.vstack([x_train, x_dev])
    y_train_dev = np.concatenate([y_train, y_dev])
    scaler_test = StandardScaler().fit(x_train_dev)
    clf_test = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
    clf_test.fit(scaler_test.transform(x_train_dev), y_train_dev)
    test_prob = clf_test.predict_proba(scaler_test.transform(test["emb"]))[:, 1]

    return {
        "dev": (
            dev["session_id"].astype(int),
            y_dev,
            dev_prob,
        ),
        "test": (
            test["session_id"].astype(int),
            test["label"].astype(int),
            test_prob,
        ),
    }


def align(
    rob: tuple[np.ndarray, np.ndarray, np.ndarray],
    ctd: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rob_ids, rob_y, rob_p = rob
    ctd_ids, ctd_y, ctd_p = ctd
    rob_map = {int(i): (int(y), float(p)) for i, y, p in zip(rob_ids, rob_y, rob_p)}
    ctd_map = {int(i): (int(y), float(p)) for i, y, p in zip(ctd_ids, ctd_y, ctd_p)}
    ids = np.array(sorted(set(rob_map) & set(ctd_map)), dtype=int)
    y = np.array([rob_map[int(i)][0] for i in ids], dtype=int)
    y_ctd = np.array([ctd_map[int(i)][0] for i in ids], dtype=int)
    if not np.array_equal(y, y_ctd):
        raise ValueError("RoBERTa and CTD labels do not align")
    p_rob = np.array([rob_map[int(i)][1] for i in ids], dtype=float)
    p_ctd = np.array([ctd_map[int(i)][1] for i in ids], dtype=float)
    return ids, y, p_rob, p_ctd


def paired_ci(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260702)
    n = len(y)
    point = macro_f1(y, pred_a) - macro_f1(y, pred_b)
    diffs = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        diffs[i] = macro_f1(y[idx], pred_a[idx]) - macro_f1(y[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def change_counts(y: np.ndarray, base: np.ndarray, fused: np.ndarray) -> tuple[int, int, int, int]:
    changed = base != fused
    corrected = changed & (base != y) & (fused == y)
    worsened = changed & (base == y) & (fused != y)
    neutral = changed & (base != y) & (fused != y)
    return int(changed.sum()), int(corrected.sum()), int(worsened.sum()), int(neutral.sum())


def summarize_split(split: str, ctd_by_split: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    ids, y, p_rob, p_ctd = align(read_roberta(split), ctd_by_split[split])
    p_fused = ROBERTA_WEIGHT * p_rob + CTD_WEIGHT * p_ctd

    pred_rob = (p_rob >= ROBERTA_THR).astype(int)
    pred_ctd = (p_ctd >= CTD_THR).astype(int)
    pred_fused = (p_fused >= FUSION_THR).astype(int)

    print(f"\n{split.upper()} n={len(ids)} depressed={int(y.sum())}")
    print(f"  RoBERTa macro-F1:      {macro_f1(y, pred_rob):.3f}")
    print(f"  CTD macro-F1:          {macro_f1(y, pred_ctd):.3f}")
    print(f"  RoBERTa+CTD macro-F1:  {macro_f1(y, pred_fused):.3f}")

    for label, base in (("vs RoBERTa", pred_rob), ("vs CTD", pred_ctd)):
        point, lo, hi = paired_ci(y, pred_fused, base)
        changed, corrected, worsened, neutral = change_counts(y, base, pred_fused)
        print(
            f"  Delta {label}: {point:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}], changed={changed}, "
            f"corrected={corrected}, worsened={worsened}, neutral={neutral}"
        )


def main() -> None:
    ctd_by_split = ctd_probs()
    summarize_split("dev", ctd_by_split)
    summarize_split("test", ctd_by_split)


if __name__ == "__main__":
    main()
