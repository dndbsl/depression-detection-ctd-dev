#!/usr/bin/env python3
"""SLT-05: ask-side ablation, through the identical deployed protocol.

rwao called the missing response-side-only ablation "a fairly important
omission for the paper's central claim." This runs five feature configs
(`feature_groups.FEATURE_CONFIGS`) through the *identical* deployed protocol
-- fit train -> select C on dev -> refit train+dev -> test once, with
bootstrap CIs -- using only the deployed classifier family (LogReg L2;
`class_weight='balanced'`), not the full 4-model comparison in `ml_splits.py`.

Per `docs/plan-preflight.md` Step 4a/4b:
- Does NOT claim the ablation "improves" performance. A paired bootstrap on
  the all24-minus-no_ask difference (same held-out sessions, resampled
  jointly) quantifies uncertainty; a CI including zero does not establish equivalence.
- Dev is primary; test is reference only. Five configs = five looks at test;
  multiplicity is stated explicitly, and nothing is re-tuned on test.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
DEPRESSION_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DEPRESSION_ROOT.parent
for p in (str(PROJECT_ROOT), str(DEPRESSION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from constants import CTD_FEATURE_NAMES, OUTPUT_DIR  # noqa: E402
from feature_groups import FEATURE_CONFIGS  # noqa: E402
from ml_splits import (  # noqa: E402
    N_BOOTSTRAPS,
    RANDOM_STATE,
    _make_pipeline,
    _metrics,
    _scores,
    _test_metrics_with_ci,
    build_split_features,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402

REPORT_DIR = REPO_ROOT / "output"
CONFIG_ORDER = ["all24", "no_ask", "res_only", "ask_only", "no_cross"]
PRIMARY_DELTA = ("all24", "no_ask")  # the headline ablation (plan Step 4a)
N_PAIRED_BOOTSTRAPS = 2000


def _load_split(split: str) -> pd.DataFrame:
    cache = OUTPUT_DIR / f"functionals_{split}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_split_features(split)
    df.to_csv(cache, index=False)
    return df


def run_config(feature_cols: list[str], data: dict[str, pd.DataFrame]) -> dict:
    """Identical deployed protocol, LogReg L2 only: fit train -> select C on
    dev -> refit train+dev -> test once (with CIs). Returns predictions too,
    for the cross-config paired bootstrap."""
    xtr = data["train"][feature_cols].to_numpy(float)
    ytr = data["train"]["PHQ8_Binary"].to_numpy(int)
    xdv = data["dev"][feature_cols].to_numpy(float)
    ydv = data["dev"]["PHQ8_Binary"].to_numpy(int)
    xte = data["test"][feature_cols].to_numpy(float)
    yte = data["test"]["PHQ8_Binary"].to_numpy(int)
    xtrdv = np.vstack([xtr, xdv])
    ytrdv = np.concatenate([ytr, ydv])

    best = None
    for c in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                  random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = _make_pipeline(clf).fit(xtr, ytr)
            dev_preds = pipe.predict(xdv)
            dev_scores = _scores(pipe, xdv)
        dev_m = _metrics(ydv, dev_preds, dev_scores)
        # Selection metric matches the deployed protocol exactly (dev balanced
        # accuracy, not macro-F1 -- see plan-preflight.md Step 0b).
        if best is None or dev_m["balanced_accuracy"] > best["dev"]["balanced_accuracy"]:
            best = {"C": c, "clf": clf, "dev": dev_m, "dev_preds": dev_preds}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe = _make_pipeline(best["clf"]).fit(xtrdv, ytrdv)
        test_preds = final_pipe.predict(xte)
        test_scores = _scores(final_pipe, xte)

    return {
        "selected_C": best["C"],
        "dev": best["dev"],
        "test": _test_metrics_with_ci(yte, test_preds, test_scores),
        "_dev_preds": best["dev_preds"],
        "_dev_y": ydv,
        "_test_preds": test_preds,
        "_test_y": yte,
    }


def paired_bootstrap_delta(
    y: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, metric_fn,
) -> dict:
    """CI on metric(a) - metric(b), resampling session indices jointly so
    the pairing (same held-out sessions) is preserved across configs."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y)
    point = float(metric_fn(y, preds_a) - metric_fn(y, preds_b))
    deltas = np.empty(N_PAIRED_BOOTSTRAPS)
    for b in range(N_PAIRED_BOOTSTRAPS):
        idx = rng.randint(0, n, size=n)
        deltas[b] = metric_fn(y[idx], preds_a[idx]) - metric_fn(y[idx], preds_b[idx])
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "point": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = {split: _load_split(split) for split in ("train", "dev", "test")}

    results: dict[str, dict] = {}
    for name in CONFIG_ORDER:
        feature_cols = [f"{f}__amean" for f in FEATURE_CONFIGS[name]]
        print(f"Running config '{name}' ({len(feature_cols)} features)...")
        results[name] = run_config(feature_cols, data)

    a, b = PRIMARY_DELTA
    macro_f1 = lambda y, p: f1_score(y, p, average="macro")
    bal_acc = balanced_accuracy_score
    dev_delta = paired_bootstrap_delta(
        results[a]["_dev_y"], results[a]["_dev_preds"], results[b]["_dev_preds"], macro_f1,
    )
    test_delta = paired_bootstrap_delta(
        results[a]["_test_y"], results[a]["_test_preds"], results[b]["_test_preds"], macro_f1,
    )

    if dev_delta["excludes_zero"] or test_delta["excludes_zero"]:
        verdict = (
            f"'{a}' differs from '{b}' beyond chance on "
            f"{'dev' if dev_delta['excludes_zero'] else 'test'} (paired bootstrap CI excludes zero)."
        )
    else:
        verdict = (
            f"No clear difference detected: the '{a}' vs '{b}' macro-F1 "
            "paired bootstrap 95% CI includes zero on dev and test. This "
            "does not establish equivalence or remove interviewer confounding."
        )

    payload = {
        "protocol": "Identical deployed protocol (LogReg L2, class_weight='balanced'): "
                    "fit train -> select C on dev (balanced accuracy) -> refit train+dev -> test once.",
        "configs": {
            name: {
                "n_features": len(FEATURE_CONFIGS[name]),
                "selected_C": r["selected_C"],
                "dev": r["dev"],
                "test": r["test"],
            }
            for name, r in results.items()
        },
        "primary_ablation_paired_bootstrap": {
            "comparison": f"{a} - {b}",
            "metric": "macro_f1",
            "dev": dev_delta,
            "test": test_delta,
            "verdict": verdict,
        },
        "multiplicity_note": (
            f"{len(CONFIG_ORDER)} configurations each evaluated once on test = "
            f"{len(CONFIG_ORDER)} looks at the test set, uncorrected. Dev is "
            "primary for all ablation comparisons; test is reported for "
            "reference only. No hyperparameter was re-tuned on test."
        ),
        "rebuttal_discrepancy_note": (
            "These numbers differ from the SLT rebuttal's no_ask figures "
            "(dev .752 / test .661): the rebuttal reused the full model's "
            "C=0.3 for the ablation configs instead of reselecting C on dev "
            "per config. This script reselects C independently for every "
            "config, per plan-preflight.md Step 4 ('identical deployed "
            "protocol' includes the selection step, not just the fitted "
            "coefficients). Reselecting gives no_ask C=1.0, dev macro-F1 "
            "0.814, test macro-F1 0.641 -- the test delta versus all24 "
            "shrinks from the rebuttal's claimed +.030 to +.010. The paired "
            "interval does not establish equivalence."
        ),
    }

    out_json = REPORT_DIR / "ctd_ablation_groups.json"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, REPORT_DIR / "ctd_ablation_groups.md")

    print("\n=== Ablation summary (LogReg L2, identical deployed protocol) ===")
    print(f"{'config':<10} {'n':>3} {'C':>6} {'dev bAcc':>9} {'dev F1':>8} | "
          f"{'test bAcc':>10} {'test F1':>9}")
    for name, r in results.items():
        print(f"{name:<10} {len(FEATURE_CONFIGS[name]):>3} {r['selected_C']:>6} "
              f"{r['dev']['balanced_accuracy']:>9.3f} {r['dev']['macro_f1']:>8.3f} | "
              f"{r['test']['balanced_accuracy']['point']:>10.3f} "
              f"{r['test']['macro_f1']['point']:>9.3f}")
    print(f"\n{verdict}")
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {REPORT_DIR / 'ctd_ablation_groups.md'}")


def write_markdown(payload: dict, path: Path) -> None:
    lines = ["# SLT-05: ask-side ablation\n", payload["protocol"] + "\n"]
    lines.append("| Config | n features | Selected C | Dev bAcc | Dev macro-F1 | Test bAcc [95% CI] | Test macro-F1 [95% CI] |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for name, c in payload["configs"].items():
        t = c["test"]
        lines.append(
            f"| `{name}` | {c['n_features']} | {c['selected_C']} | "
            f"{c['dev']['balanced_accuracy']:.3f} | {c['dev']['macro_f1']:.3f} | "
            f"{t['balanced_accuracy']['point']:.3f} "
            f"[{t['balanced_accuracy']['ci_low']:.3f}, {t['balanced_accuracy']['ci_high']:.3f}] | "
            f"{t['macro_f1']['point']:.3f} "
            f"[{t['macro_f1']['ci_low']:.3f}, {t['macro_f1']['ci_high']:.3f}] |"
        )

    d = payload["primary_ablation_paired_bootstrap"]
    lines.append(f"\n## Paired bootstrap: {d['comparison']} (macro-F1)\n")
    lines.append("| Split | Delta | 95% CI | Excludes zero? |")
    lines.append("|---|---:|---|---|")
    lines.append(f"| dev | {d['dev']['point']:+.3f} | [{d['dev']['ci_low']:+.3f}, {d['dev']['ci_high']:+.3f}] | {d['dev']['excludes_zero']} |")
    lines.append(f"| test | {d['test']['point']:+.3f} | [{d['test']['ci_low']:+.3f}, {d['test']['ci_high']:+.3f}] | {d['test']['excludes_zero']} |")
    lines.append(f"\n**Verdict:** {d['verdict']}\n")
    lines.append(f"\n> {payload['multiplicity_note']}\n")
    lines.append(f"\n> **Note on rebuttal discrepancy:** {payload['rebuttal_discrepancy_note']}\n")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
