#!/usr/bin/env python3
"""Depression detection from CTD functionals using the STANDARD DAIC-WOZ splits.

Protocol (challenge-style, no leakage)
--------------------------------------
* **train** (102 sessions, cleaned): fit imputation + scaling and train models.
* **dev** (33 sessions): model / hyperparameter selection only.
* **test** (45 sessions): touched once, for the final report. 95% confidence
  intervals via 2000-resample bootstrapping (``confidence_intervals`` /
  Ferrer & Riera).

For the final test report each selected model is refit on **train+dev** (the
standard "use all development data for the deployed system" step); preprocessing
statistics are refit on train+dev as well. Dev numbers below are from
train-only fits and are what selection is based on.

Features: 24 CTD features x 10 eGeMAPS functionals = 240 per session
(``functionals.py``). A 24-D session-mean baseline is run under the identical
split protocol for comparison.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from confidence_intervals import evaluate_with_conf_int
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent
DEPRESSION_ROOT = PROJECT_ROOT.parent
for p in (str(PROJECT_ROOT), str(DEPRESSION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from constants import CTD_FEATURE_NAMES, DEFAULT_DATA_ROOT, OUTPUT_DIR  # noqa: E402
from common.daic_cleaning import apply_cleaning  # noqa: E402
from common.transcript_preprocessing import load_transcript  # noqa: E402
from data_loading import transcript_path  # noqa: E402
from feature_extraction import extract_session_turn_features  # noqa: E402
from functionals import build_session_functionals, functional_feature_names  # noqa: E402
from turn_pairing import build_turn_pairs  # noqa: E402

RANDOM_STATE = 42
N_BOOTSTRAPS = 2000
ALPHA = 5  # -> 95% CI

LABEL_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",
}


def load_split_labels(split: str) -> pd.DataFrame:
    path = Path(DEFAULT_DATA_ROOT) / "labels" / LABEL_FILES[split]
    df = pd.read_csv(path).rename(columns={"Participant_ID": "pid"})
    bincol = "PHQ_Binary" if "PHQ_Binary" in df.columns else "PHQ8_Binary"
    scorecol = "PHQ_Score" if "PHQ_Score" in df.columns else "PHQ8_Score"
    df = df[["pid", scorecol, bincol]].rename(
        columns={scorecol: "PHQ8_Score", bincol: "PHQ8_Binary"}
    )
    df["pid"] = df["pid"].astype(int)
    df["PHQ8_Score"] = df["PHQ8_Score"].astype(int)
    df["PHQ8_Binary"] = (df["PHQ8_Score"] >= 10).astype(int)
    df = df.set_index("pid").sort_index()
    cleaned, _ = apply_cleaning(df, split=split)
    return cleaned.reset_index().rename(columns={"pid": "session_id"})


def build_split_features(split: str) -> pd.DataFrame:
    labels = load_split_labels(split)
    turn_rows = []
    for sid in labels["session_id"]:
        df = load_transcript(transcript_path(DEFAULT_DATA_ROOT, int(sid)))
        pairs = build_turn_pairs(int(sid), df)
        turn_df = extract_session_turn_features(pairs)
        if len(turn_df):
            turn_rows.append(turn_df)
    turn_level = pd.concat(turn_rows, ignore_index=True)
    feats = build_session_functionals(turn_level, labels)
    feats["split"] = split
    return feats


def _make_pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def _model_grid() -> dict[str, list]:
    """model_name -> list of (label, estimator) candidates tuned on dev."""
    grid: dict[str, list] = {
        "logreg_l2": [
            (f"C={c}", LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                          random_state=RANDOM_STATE))
            for c in (0.01, 0.03, 0.1, 0.3, 1.0)
        ],
        "logreg_l1": [
            (f"C={c}", LogisticRegression(C=c, penalty="l1", solver="liblinear",
                                          class_weight="balanced", max_iter=5000,
                                          random_state=RANDOM_STATE))
            for c in (0.03, 0.1, 0.3, 1.0)
        ],
        "svm_rbf": [
            (f"C={c}", SVC(C=c, kernel="rbf", class_weight="balanced",
                           probability=True, random_state=RANDOM_STATE))
            for c in (0.1, 0.5, 1.0, 3.0)
        ],
        "random_forest": [
            (f"leaf={leaf}", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=leaf,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1))
            for leaf in (1, 2, 4)
        ],
    }
    return grid


def _scores(pipe: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    return pipe.decision_function(x)


def _metrics(y: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "f1_depressed": float(f1_score(y, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "accuracy": float((preds == y).mean()),
    }


def _test_metrics_with_ci(y: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    out: dict = {}
    for name, fn, samples in [
        ("balanced_accuracy", balanced_accuracy_score, preds),
        ("macro_f1", lambda yt, s: f1_score(yt, s, average="macro"), preds),
        ("f1_depressed", lambda yt, s: f1_score(yt, s, pos_label=1, zero_division=0), preds),
        ("roc_auc", roc_auc_score, scores),
    ]:
        point, (lo, hi) = evaluate_with_conf_int(
            np.asarray(samples), fn, np.asarray(y),
            conditions=None, num_bootstraps=N_BOOTSTRAPS, alpha=ALPHA,
        )
        out[name] = {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}
    out["accuracy"] = {"point": float((preds == y).mean())}
    return out


def run(feature_cols: list[str], data: dict[str, pd.DataFrame], tag: str) -> dict:
    xtr = data["train"][feature_cols].to_numpy(float)
    ytr = data["train"]["PHQ8_Binary"].to_numpy(int)
    xdv = data["dev"][feature_cols].to_numpy(float)
    ydv = data["dev"]["PHQ8_Binary"].to_numpy(int)
    xte = data["test"][feature_cols].to_numpy(float)
    yte = data["test"]["PHQ8_Binary"].to_numpy(int)
    xtrdv = np.vstack([xtr, xdv])
    ytrdv = np.concatenate([ytr, ydv])

    results: dict = {"tag": tag, "n_features": len(feature_cols), "models": {}}
    for model_name, candidates in _model_grid().items():
        best = None
        for label, clf in candidates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = _make_pipeline(clf).fit(xtr, ytr)
                dev_preds = pipe.predict(xdv)
                dev_scores = _scores(pipe, xdv)
            dev_m = _metrics(ydv, dev_preds, dev_scores)
            if best is None or dev_m["balanced_accuracy"] > best["dev"]["balanced_accuracy"]:
                best = {"hyperparam": label, "clf": clf, "dev": dev_m}

        # Refit selected config on train+dev, evaluate once on test (with CIs).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipe = _make_pipeline(best["clf"]).fit(xtrdv, ytrdv)
            test_preds = final_pipe.predict(xte)
            test_scores = _scores(final_pipe, xte)
        results["models"][model_name] = {
            "selected_hyperparam": best["hyperparam"],
            "dev": best["dev"],
            "test": _test_metrics_with_ci(yte, test_preds, test_scores),
        }
    return results


def _fmt_ci(m: dict) -> str:
    return f"{m['point']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def _print_block(res: dict) -> None:
    print(f"\n=== {res['tag']}  ({res['n_features']} features) ===")
    print(f"{'model':<16} {'sel':<8} {'dev_bAcc':>9} | "
          f"{'test bAcc [95% CI]':<24} {'test AUC [95% CI]':<24} {'test macroF1 [95% CI]':<24}")
    for name, r in res["models"].items():
        print(f"{name:<16} {r['selected_hyperparam']:<8} "
              f"{r['dev']['balanced_accuracy']:>9.3f} | "
              f"{_fmt_ci(r['test']['balanced_accuracy']):<24} "
              f"{_fmt_ci(r['test']['roc_auc']):<24} "
              f"{_fmt_ci(r['test']['macro_f1']):<24}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Extracting CTD + functionals per split (train/dev/test)...")
    data = {split: build_split_features(split) for split in ("train", "dev", "test")}
    for split, df in data.items():
        df.to_csv(OUTPUT_DIR / f"functionals_{split}.csv", index=False)
        n_pos = int(df["PHQ8_Binary"].sum())
        print(f"  {split}: {len(df)} sessions, {n_pos} depressed ({100*n_pos/len(df):.1f}%)")

    func_cols = functional_feature_names()
    # 24-D session-mean baseline: the `__amean` functional == per-feature session mean.
    mean_cols = [f"{feat}__amean" for feat in CTD_FEATURE_NAMES]

    res_func = run(func_cols, data, tag="CTD functionals (240-D)")
    res_mean = run(mean_cols, data, tag="CTD session-mean (24-D)")

    _print_block(res_mean)
    _print_block(res_func)

    payload = {
        "protocol": "train=fit, dev=select, test=report (refit on train+dev); "
                    f"{N_BOOTSTRAPS}-bootstrap {100-ALPHA}% CI",
        "cohort": {s: {"n": int(len(d)), "n_depressed": int(d["PHQ8_Binary"].sum())}
                   for s, d in data.items()},
        "session_mean_24d": res_mean,
        "functionals_240d": res_func,
    }
    out = OUTPUT_DIR / "ml_splits_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
