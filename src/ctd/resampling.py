#!/usr/bin/env python3
"""SLT-10: variability of the deployed CTD detector via data resampling.

rwao asked for "a similar sweep over CTD or the fusion weights" by analogy
with the neural baselines' seed variance. Per `docs/plan-preflight.md` Step
5a: the CTD detector is `SimpleImputer -> StandardScaler -> LogisticRegression
(lbfgs)`, a deterministic fit on a convex objective, so seed variance is
exactly 0.000 and reporting it would look evasive. The honest analogue is
**data** resampling: repeatedly re-partition train+dev, re-select C on the
new held-out portion, and see how much both the metrics and the selected C
move around. The real test set is touched only for reporting, never for
re-selection.

Protocol (R=100 repeats):
1. Pool train+dev (135 sessions). Split it into a resampled train/held-out
   pair with `StratifiedGroupKFold(shuffle=True, random_state=repeat)` --
   group-aware from the start (`groups=session_id`) so this splitter is
   directly reusable for PDCH, which has 2 sessions/subject (`MC-08`); on
   DAIC-WOZ (1 session/subject) grouping is a no-op.
2. Refit on the resampled train, select C on the resampled held-out portion
   (dev balanced accuracy -- the deployed selection metric). Record the
   held-out metrics and the selected C.
3. Refit on the *full* train+dev pool with that repeat's selected C, evaluate
   once on the untouched real test set. Record test macro-F1.

Head-to-head rate = fraction of repeats where step 3's test macro-F1 beats
RoBERTa's test macro-F1 (0.631, `RESULTS.md`) -- a direct, honest answer to
"is CTD really the best single modality" instead of resting on one dev split.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent
DEPRESSION_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DEPRESSION_ROOT.parent
for p in (str(PROJECT_ROOT), str(DEPRESSION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from constants import CTD_FEATURE_NAMES, OUTPUT_DIR  # noqa: E402
from ml_splits import RANDOM_STATE, _make_pipeline, _metrics, _scores, build_split_features  # noqa: E402

REPORT_DIR = REPO_ROOT / "output"
MEAN_COLS = [f"{feat}__amean" for feat in CTD_FEATURE_NAMES]
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
R_REPEATS = 100
N_SPLITS = 4  # 135/4 ~= 33.75 held out per repeat, matching the 102/33 ratio
ROBERTA_TEST_MACRO_F1 = 0.631  # RESULTS.md


def _load_split(split: str) -> pd.DataFrame:
    cache = OUTPUT_DIR / f"functionals_{split}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_split_features(split)
    df.to_csv(cache, index=False)
    return df


def _select_c(x_tr, y_tr, x_hd, y_hd) -> tuple[float, dict]:
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                  random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = _make_pipeline(clf).fit(x_tr, y_tr)
            hd_preds = pipe.predict(x_hd)
            hd_scores = _scores(pipe, x_hd)
        hd_m = _metrics(y_hd, hd_preds, hd_scores)
        if best is None or hd_m["balanced_accuracy"] > best["metrics"]["balanced_accuracy"]:
            best = {"C": c, "metrics": hd_m}
    return best["C"], best["metrics"]


def run() -> dict:
    trdv = pd.concat([_load_split("train"), _load_split("dev")], ignore_index=True)
    test = _load_split("test")

    x_pool = trdv[MEAN_COLS].to_numpy(float)
    y_pool = trdv["PHQ8_Binary"].to_numpy(int)
    groups = trdv["session_id"].to_numpy(int)  # 1 session/subject on DAIC-WOZ; group-aware for reuse
    x_test = test[MEAN_COLS].to_numpy(float)
    y_test = test["PHQ8_Binary"].to_numpy(int)

    held_out_metrics: list[dict] = []
    selected_cs: list[float] = []
    test_macro_f1s: list[float] = []

    for repeat in range(R_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        train_idx, held_idx = next(splitter.split(x_pool, y_pool, groups=groups))

        c, hd_metrics = _select_c(
            x_pool[train_idx], y_pool[train_idx], x_pool[held_idx], y_pool[held_idx],
        )
        held_out_metrics.append(hd_metrics)
        selected_cs.append(c)

        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                  random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = _make_pipeline(clf).fit(x_pool, y_pool)
            test_preds = pipe.predict(x_test)
        test_m = _metrics(y_test, test_preds, _scores(pipe, x_test))
        test_macro_f1s.append(test_m["macro_f1"])

    def _mean_sd(key: str) -> dict:
        vals = np.array([m[key] for m in held_out_metrics])
        return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}

    c_counts = Counter(selected_cs)
    test_arr = np.array(test_macro_f1s)
    head_to_head_rate = float((test_arr > ROBERTA_TEST_MACRO_F1).mean())

    max_count = max(c_counts.values())
    modal_cs = sorted(c for c, n in c_counts.items() if n == max_count)

    return {
        "protocol": (
            f"{R_REPEATS}x repeated stratified-group resampling of train+dev "
            f"(StratifiedGroupKFold, n_splits={N_SPLITS}, group=session_id); "
            "refit on resampled train, select C on resampled held-out portion "
            "(dev balanced accuracy); refit on full train+dev with that C, "
            "evaluate once on the untouched real test set."
        ),
        "held_out_metrics_over_repeats": {
            "balanced_accuracy": _mean_sd("balanced_accuracy"),
            "macro_f1": _mean_sd("macro_f1"),
            "roc_auc": _mean_sd("roc_auc"),
        },
        "hyperparameter_stability": {
            "selected_C_counts": {str(c): n for c, n in sorted(c_counts.items())},
            "modal_C": modal_cs if len(modal_cs) > 1 else modal_cs[0],
            "modal_C_fraction": max_count / R_REPEATS,
            "note": (
                "Deployed model uses C=0.3, selected on the single official "
                "dev split. If a different C is modal here, C=0.3 should be "
                "reported as one plausible choice among several rather than "
                "uniquely optimal."
            ),
        },
        "head_to_head_vs_roberta": {
            "roberta_test_macro_f1": ROBERTA_TEST_MACRO_F1,
            "ctd_test_macro_f1_over_repeats": {
                "mean": float(test_arr.mean()), "sd": float(test_arr.std(ddof=1)),
            },
            "fraction_repeats_ctd_beats_roberta": head_to_head_rate,
        },
        "seed_variance_note": (
            "Not reported: the CTD pipeline (SimpleImputer(median) -> "
            "StandardScaler -> LogisticRegression(lbfgs)) is a deterministic "
            "convex fit, so seed variance is exactly 0.000 for a fixed data "
            "split. Data resampling above answers the variability question "
            "rwao intended by analogy with the neural baselines' seed sweep."
        ),
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = ["# SLT-10: CTD variability via data resampling\n", payload["protocol"] + "\n"]
    lines.append(f"> {payload['seed_variance_note']}\n")

    lines.append("\n## Held-out metrics over 100 repeats (mean ± sd)\n")
    lines.append("| Metric | Mean | SD |")
    lines.append("|---|---:|---:|")
    for name, v in payload["held_out_metrics_over_repeats"].items():
        lines.append(f"| {name} | {v['mean']:.3f} | {v['sd']:.3f} |")

    hp = payload["hyperparameter_stability"]
    lines.append("\n## Hyperparameter (C) stability\n")
    lines.append("| C | Count |")
    lines.append("|---:|---:|")
    for c, n in hp["selected_C_counts"].items():
        lines.append(f"| {c} | {n} |")
    is_tie = isinstance(hp["modal_C"], list)
    modal_str = ", ".join(str(c) for c in hp["modal_C"]) if is_tie else str(hp["modal_C"])
    label = "Tied modal C values" if is_tie else "Modal C"
    suffix = "each" if is_tie else ""
    lines.append(f"\n{label} = **{modal_str}**, selected in {hp['modal_C_fraction']*100:.0f}% of repeats {suffix}.\n")
    lines.append(f"\n> {hp['note']}\n")

    h2h = payload["head_to_head_vs_roberta"]
    lines.append("\n## Head-to-head vs. RoBERTa on test\n")
    lines.append(
        f"CTD test macro-F1 across repeats: {h2h['ctd_test_macro_f1_over_repeats']['mean']:.3f} "
        f"± {h2h['ctd_test_macro_f1_over_repeats']['sd']:.3f} "
        f"(RoBERTa: {h2h['roberta_test_macro_f1']:.3f}).\n"
    )
    lines.append(
        f"CTD beats RoBERTa's test macro-F1 in **{h2h['fraction_repeats_ctd_beats_roberta']*100:.0f}%** "
        "of repeats.\n"
    )
    path.write_text("\n".join(lines))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {R_REPEATS} repeats of stratified-group resampling over train+dev...")
    payload = run()

    out_json = REPORT_DIR / "ctd_resampling.json"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, REPORT_DIR / "ctd_resampling.md")

    hom = payload["held_out_metrics_over_repeats"]
    hp = payload["hyperparameter_stability"]
    h2h = payload["head_to_head_vs_roberta"]
    print(f"\nHeld-out macro-F1 over {R_REPEATS} repeats: {hom['macro_f1']['mean']:.3f} ± {hom['macro_f1']['sd']:.3f}")
    print(f"Modal C: {hp['modal_C']} ({hp['modal_C_fraction']*100:.0f}% of repeats)")
    print(f"CTD beats RoBERTa test macro-F1 ({h2h['roberta_test_macro_f1']}) in "
          f"{h2h['fraction_repeats_ctd_beats_roberta']*100:.0f}% of repeats")
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {REPORT_DIR / 'ctd_resampling.md'}")


if __name__ == "__main__":
    main()
