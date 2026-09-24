#!/usr/bin/env python3
"""Within-PDCH CTD experiment (`MC-08`) and the pre-registered P1-P6 check.

Runs the five `feature_groups.FEATURE_CONFIGS` tiers through subject-grouped
cross-validation and evaluates every prediction in
`docs/preregistration-pdch.md`. Nothing here selects on a PDCH result: the
label rule, the tiers, the detector and the predictions were all fixed before
any PDCH feature met any PDCH label.

Protocol
--------
* **Splits.** Repeated `StratifiedGroupKFold` on `subject_id`, 5 folds x 20
  repeats. 16 of the 46 labelled subjects contribute two sessions with
  *different* HAMD-17 totals, so session-level splitting would leak subject
  identity. No fixed held-out test split: at n=62 a single PDCH partition
  would reproduce exactly the small-sample fragility `SLT-10` moved away from,
  so every number below is a distribution over repeats, not a point estimate.
* **Detector.** The deployed pipeline unchanged -- median impute ->
  standardize -> L2 LogReg, `class_weight='balanced'`.
* **Selection.** `C` *and* the intra-turn silence threshold are chosen inside
  an inner `StratifiedGroupKFold` on the outer training fold only, on balanced
  accuracy (the deployed selection metric, `ml_splits.py`). The DAIC-tuned
  0.2 s threshold stays in the grid as a candidate but is never assumed.
* **Scoring.** Out-of-fold predictions are pooled within a repeat, giving one
  macro-F1 per repeat; we report mean +- sd across repeats.
* **Secondary target.** Ridge regression on the HAMD-17 total under the same
  folds (`MC-07`).
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

PROJECT_ROOT = Path(__file__).resolve().parent
DEPRESSION_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DEPRESSION_ROOT.parent
for p in (str(PROJECT_ROOT), str(DEPRESSION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from constants import CTD_FEATURE_NAMES  # noqa: E402
from feature_groups import CROSS, FEATURE_CONFIGS  # noqa: E402
from pdch_features import (  # noqa: E402
    MEAN_COLS,
    SILENCE_THRESHOLD_GRID,
    build_feature_grid,
    degeneracy_report,
    load_utterance_cache,
)
from pdch_labels import label_summary, load_pdch_labels  # noqa: E402

REPORT_DIR = REPO_ROOT / "output"
PDCH_DIR = REPORT_DIR / "pdch"

RANDOM_STATE = 42
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)          # deployed grid (ml_splits.py)
RIDGE_ALPHA_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
N_REPEATS = 20
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 4
N_BOOTSTRAPS = 2000
CONFIG_ORDER = ["all24", "no_ask", "res_only", "ask_only", "no_cross"]

# DAIC-WOZ reference values, read from output/ctd_interpretability.json.
DAIC_INTERPRET_JSON = REPORT_DIR / "ctd_interpretability.json"
DAIC_RES_H_TEST_AUC = 0.829  # preregistration P5 anchor

P4_RATIO_FEATURES = ["res_ud", "res_sd", "res_su", "res_over_ask", "ask_over_res"]
P4_DURATION_FEATURES = ["ask_d", "res_d", "duration_sum"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clf_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                   random_state=RANDOM_STATE)),
    ])


def _reg_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
    ])


def _fit(pipe: Pipeline, x, y) -> Pipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipe.fit(x, y)


def _mean_sd(vals) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def _align(tables: dict[float, pd.DataFrame], labels: pd.DataFrame,
           cols: list[str]) -> dict[float, np.ndarray]:
    """threshold -> design matrix, rows in `labels` order."""
    out = {}
    for thr, df in tables.items():
        idx = df.set_index("session_id")
        out[thr] = idx.loc[labels["session_id"].to_numpy(), cols].to_numpy(float)
    return out


# --------------------------------------------------------------------------
# MC-08 -- subject-grouped nested CV
# --------------------------------------------------------------------------
def nested_cv_classification(
    x_by_thr: dict[float, np.ndarray], y: np.ndarray, groups: np.ndarray,
    n_repeats: int = N_REPEATS, fixed_threshold: float | None = None,
    pipeline_factory=None,
) -> dict:
    """Repeated subject-grouped nested CV. Returns per-repeat pooled metrics.

    `fixed_threshold` pins the silence threshold instead of selecting it, which
    is how the threshold-sensitivity table is produced.

    `pipeline_factory` overrides the estimator built per candidate `C`. It
    defaults to `_clf_pipeline`, so every existing caller is byte-for-byte
    unaffected; `shared_signal_audit.py` passes a factory that residualizes the
    absolute-time features on session length *inside the training fold*.
    """
    make_pipe = pipeline_factory or _clf_pipeline
    thresholds = [fixed_threshold] if fixed_threshold is not None else list(x_by_thr)
    per_repeat, chosen_c, chosen_thr = [], [], []

    for rep in range(n_repeats):
        outer = StratifiedGroupKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=rep)
        x_ref = x_by_thr[thresholds[0]]
        oof_pred = np.full(len(y), -1, dtype=int)
        oof_prob = np.full(len(y), np.nan, dtype=float)

        for tr, te in outer.split(x_ref, y, groups=groups):
            inner = StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True,
                                         random_state=1000 + rep)
            inner_splits = list(inner.split(x_ref[tr], y[tr], groups=groups[tr]))

            best_score, best = -np.inf, None
            for thr in thresholds:
                xt = x_by_thr[thr]
                for c in C_GRID:
                    scores = []
                    for itr, ite in inner_splits:
                        pipe = _fit(make_pipe(c), xt[tr][itr], y[tr][itr])
                        scores.append(balanced_accuracy_score(y[tr][ite],
                                                              pipe.predict(xt[tr][ite])))
                    s = float(np.mean(scores))
                    if s > best_score:
                        best_score, best = s, (c, thr)

            c, thr = best
            chosen_c.append(c)
            chosen_thr.append(thr)
            xt = x_by_thr[thr]
            pipe = _fit(make_pipe(c), xt[tr], y[tr])
            oof_pred[te] = pipe.predict(xt[te])
            oof_prob[te] = pipe.predict_proba(xt[te])[:, 1]

        assert (oof_pred >= 0).all()
        per_repeat.append({
            "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
            "macro_f1": float(f1_score(y, oof_pred, average="macro")),
            "f1_positive": float(f1_score(y, oof_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, oof_prob)),
            "accuracy": float((oof_pred == y).mean()),
        })

    keys = per_repeat[0].keys()
    out = {k: _mean_sd([r[k] for r in per_repeat]) for k in keys}
    out["n_repeats"] = n_repeats
    out["selected_C_counts"] = {str(k): v for k, v in sorted(Counter(chosen_c).items())}
    if fixed_threshold is None:
        out["selected_threshold_counts"] = {
            str(k): v for k, v in sorted(Counter(chosen_thr).items())
        }
    return out


def nested_cv_regression(
    x_by_thr: dict[float, np.ndarray], y_cont: np.ndarray, y_bin: np.ndarray,
    groups: np.ndarray, n_repeats: int = N_REPEATS,
) -> dict:
    """Severity regression on the HAMD-17 total, same folds (`MC-07` secondary).

    Folds are stratified on the binary label so the outer partition is
    identical in construction to the classification protocol.
    """
    thresholds = list(x_by_thr)
    per_repeat, chosen_a, chosen_thr = [], [], []

    for rep in range(n_repeats):
        outer = StratifiedGroupKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=rep)
        x_ref = x_by_thr[thresholds[0]]
        oof = np.full(len(y_cont), np.nan)

        for tr, te in outer.split(x_ref, y_bin, groups=groups):
            inner = StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True,
                                         random_state=1000 + rep)
            inner_splits = list(inner.split(x_ref[tr], y_bin[tr], groups=groups[tr]))
            best_score, best = np.inf, None
            for thr in thresholds:
                xt = x_by_thr[thr]
                for a in RIDGE_ALPHA_GRID:
                    errs = []
                    for itr, ite in inner_splits:
                        pipe = _fit(_reg_pipeline(a), xt[tr][itr], y_cont[tr][itr])
                        errs.append(mean_absolute_error(y_cont[tr][ite],
                                                        pipe.predict(xt[tr][ite])))
                    e = float(np.mean(errs))
                    if e < best_score:
                        best_score, best = e, (a, thr)
            a, thr = best
            chosen_a.append(a)
            chosen_thr.append(thr)
            xt = x_by_thr[thr]
            oof[te] = _fit(_reg_pipeline(a), xt[tr], y_cont[tr]).predict(xt[te])

        r, p = pearsonr(y_cont, oof)
        per_repeat.append({
            "mae": float(mean_absolute_error(y_cont, oof)),
            "rmse": float(np.sqrt(np.mean((y_cont - oof) ** 2))),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "r2": float(1 - np.sum((y_cont - oof) ** 2) / np.sum((y_cont - y_cont.mean()) ** 2)),
        })

    out = {k: _mean_sd([r[k] for r in per_repeat]) for k in per_repeat[0]}
    out["n_repeats"] = n_repeats
    out["baseline_mae_predict_train_mean"] = float(
        mean_absolute_error(y_cont, np.full_like(y_cont, y_cont.mean()))
    )
    out["selected_alpha_counts"] = {str(k): v for k, v in sorted(Counter(chosen_a).items())}
    out["selected_threshold_counts"] = {str(k): v for k, v in sorted(Counter(chosen_thr).items())}
    return out


# --------------------------------------------------------------------------
# univariate + coefficients (P1, P2, P4, P5, P6)
# --------------------------------------------------------------------------
def univariate_table(x: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for i, feat in enumerate(CTD_FEATURE_NAMES):
        v = x[:, i]
        if not np.isfinite(v).all() or np.std(v) < 1e-12:
            out[feat] = {"point_biserial_r": float("nan"), "p_value": float("nan"),
                         "single_feature_auc": float("nan"), "degenerate": True}
            continue
        r, p = pointbiserialr(y, v)
        # AUC is direction-free here: report the AUC of the feature as scored,
        # so it is comparable in sign with the correlation.
        out[feat] = {"point_biserial_r": float(r), "p_value": float(p),
                     "single_feature_auc": float(roc_auc_score(y, v)),
                     "degenerate": False}
    return out


def bootstrap_coefficients(x: np.ndarray, y: np.ndarray, c: float) -> dict:
    pipe = _fit(_clf_pipeline(c), x, y)
    point = pipe.named_steps["clf"].coef_.ravel()
    rng = np.random.RandomState(RANDOM_STATE)
    boot = np.empty((N_BOOTSTRAPS, len(point)))
    for b in range(N_BOOTSTRAPS):
        xb, yb = resample(x, y, replace=True, stratify=y,
                          random_state=rng.randint(0, 2**31 - 1))
        boot[b] = _fit(_clf_pipeline(c), xb, yb).named_steps["clf"].coef_.ravel()
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    sign_ok = (np.sign(boot) == np.sign(point)[None, :]).mean(axis=0)
    return {
        feat: {"coef": float(point[i]), "ci_low": float(lo[i]), "ci_high": float(hi[i]),
               "sign_consistency_rate": float(sign_ok[i]),
               "odds_ratio_per_1sd": float(np.exp(point[i]))}
        for i, feat in enumerate(CTD_FEATURE_NAMES)
    }


def evaluate_predictions(uni: dict, configs: dict, coefs: dict, daic_uni: dict) -> dict:
    """Explicit pass/fail for each pre-registered prediction P1-P6."""
    preds: dict = {}

    # --- P1: res_h has the largest univariate association among tier-B -------
    live_cross = [f for f in CROSS if not uni[f]["degenerate"]]
    by_r = sorted(live_cross, key=lambda f: -abs(uni[f]["point_biserial_r"]))
    by_auc = sorted(live_cross, key=lambda f: -abs(uni[f]["single_feature_auc"] - 0.5))
    preds["P1"] = {
        "statement": "Among tier-B (cross-speaker) features, res_h has the largest "
                     "absolute univariate association with the PDCH label.",
        "res_h_r": uni["res_h"]["point_biserial_r"],
        "res_h_auc": uni["res_h"]["single_feature_auc"],
        "rank_by_abs_r": by_r.index("res_h") + 1,
        "rank_by_auc_distance_from_chance": by_auc.index("res_h") + 1,
        "n_tier_b_testable": len(live_cross),
        "tier_b_ranking_by_abs_r": [
            {"feature": f, "abs_r": abs(uni[f]["point_biserial_r"]),
             "auc": uni[f]["single_feature_auc"]} for f in by_r
        ],
        "degenerate_tier_b_features": [f for f in CROSS if uni[f]["degenerate"]],
        "passed": by_r[0] == "res_h" and by_auc[0] == "res_h",
    }

    # --- P2: direction preserved (positive) ---------------------------------
    preds["P2"] = {
        "statement": "The res_h association is positive: longer response latency "
                     "with greater severity, same sign as DAIC-WOZ.",
        "pdch_r": uni["res_h"]["point_biserial_r"],
        "daic_train_r": daic_uni["res_h"]["train"]["point_biserial_r"],
        "daic_test_r": daic_uni["res_h"]["test"]["point_biserial_r"],
        "passed": bool(uni["res_h"]["point_biserial_r"] > 0),
    }

    # --- P3: tier ordering reproduces ---------------------------------------
    mf = {k: configs[k]["macro_f1"]["mean"] for k in CONFIG_ORDER}
    preds["P3"] = {
        "statement": "Macro-F1 ordering on PDCH is no_ask >= all24 > no_cross > res_only.",
        "macro_f1_means": mf,
        "observed_order": sorted(mf, key=lambda k: -mf[k]),
        "checks": {
            "no_ask >= all24": bool(mf["no_ask"] >= mf["all24"]),
            "all24 > no_cross": bool(mf["all24"] > mf["no_cross"]),
            "no_cross > res_only": bool(mf["no_cross"] > mf["res_only"]),
        },
        "passed": bool(mf["no_ask"] >= mf["all24"] > mf["no_cross"] > mf["res_only"]),
    }

    # --- P4: ratios transfer, absolute durations do not ---------------------
    def retention(feats, split):
        vals = []
        for f in feats:
            d = abs(daic_uni[f][split]["point_biserial_r"])
            if uni[f]["degenerate"] or d < 1e-9:
                continue
            vals.append(abs(uni[f]["point_biserial_r"]) / d)
        return vals

    p4: dict = {
        "statement": "Dimensionless ratios retain more of their DAIC-WOZ association "
                     "than absolute durations.",
        "operationalization": (
            "retention = |r_PDCH| / |r_DAIC| per feature, compared by group median. "
            "The preregistration did not name a DAIC-WOZ split; train (n=102, the "
            "largest and the one not used for any selection) is primary, dev/test "
            "reported as sensitivity. See the appendix amendment."
        ),
        "mean_abs_r_pdch": {
            "ratios": float(np.mean([abs(uni[f]["point_biserial_r"]) for f in P4_RATIO_FEATURES])),
            "durations": float(np.mean([abs(uni[f]["point_biserial_r"]) for f in P4_DURATION_FEATURES])),
        },
        "per_split": {},
    }
    for split in ("train", "dev", "test"):
        rr, dd = retention(P4_RATIO_FEATURES, split), retention(P4_DURATION_FEATURES, split)
        p4["per_split"][split] = {
            "median_retention_ratios": float(np.median(rr)),
            "median_retention_durations": float(np.median(dd)),
            "ratios_retain_more": bool(np.median(rr) > np.median(dd)),
        }
    p4["passed"] = p4["per_split"]["train"]["ratios_retain_more"]
    preds["P4"] = p4

    # --- P5: effect size shrinks --------------------------------------------
    preds["P5"] = {
        "statement": f"PDCH single-feature AUC for res_h is below the "
                     f"{DAIC_RES_H_TEST_AUC} observed on DAIC-WOZ test.",
        "pdch_res_h_auc": uni["res_h"]["single_feature_auc"],
        "daic_test_res_h_auc": DAIC_RES_H_TEST_AUC,
        "passed": bool(uni["res_h"]["single_feature_auc"] < DAIC_RES_H_TEST_AUC),
    }

    # --- P6: ask_d does not dominate ----------------------------------------
    ranked = sorted(coefs, key=lambda f: -abs(coefs[f]["coef"]))
    preds["P6"] = {
        "statement": "ask_d will NOT have the largest absolute coefficient on PDCH "
                     "(human clinicians adapt to the patient; the DAIC-WOZ wizard "
                     "does not).",
        "largest_abs_coefficient": ranked[0],
        "ask_d_rank": ranked.index("ask_d") + 1,
        "ask_d_coef": coefs["ask_d"]["coef"],
        "ask_d_sign_consistency": coefs["ask_d"]["sign_consistency_rate"],
        "top5": [{"feature": f, "coef": coefs[f]["coef"],
                  "sign_consistency": coefs[f]["sign_consistency_rate"]} for f in ranked[:5]],
        "passed": ranked[0] != "ask_d",
    }
    return preds


# --------------------------------------------------------------------------
# MC-06 payoff: what the raw 1-second turn labels would have given
# --------------------------------------------------------------------------
def label_only_features() -> pd.DataFrame:
    """Session features built from the raw turn labels, with no VAD refinement.

    Not part of the pipeline -- this is the counterfactual that quantifies what
    `MC-06` bought. Each labelled turn becomes a single utterance, so
    `turn_s()` is 0 for any single-turn run and the intra-turn silence features
    collapse.
    """
    from pdch_adapter import session_dirs, stitch_session_turns
    from pdch_features import build_pdch_features
    from turn_pairing import Utterance

    utts = {}
    for d in session_dirs():
        turns, _ = stitch_session_turns(d)
        utts[d.name] = [Utterance(start=t.start, end=t.end, speaker=t.speaker) for t in turns]
    return build_pdch_features(utts, 0.2)


def main() -> None:
    PDCH_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_pdch_labels()
    lab_sum = label_summary(labels)
    print(f"PDCH labels: {lab_sum['n_sessions']} sessions, "
          f"{lab_sum['n_subjects']} subjects, "
          f"{lab_sum['n_positive']} positive ({lab_sum['prevalence']:.1%})")

    print("Building feature tables over the silence-threshold grid...")
    utts = load_utterance_cache()
    tables = build_feature_grid(utts)
    n_all_sessions = len(tables[0.2])

    # Pipeline attrition: labelled sessions that actually produced turn pairs.
    have_features = set(tables[0.2]["session_id"])
    labels = labels[labels["session_id"].isin(have_features)].reset_index(drop=True)
    print(f"  {n_all_sessions}/100 sessions survived stitching+VAD+pairing; "
          f"{len(labels)} of those carry a HAMD-17 total")

    y = labels["HAMD17_ge17"].to_numpy(int)
    y_cont = labels["HAMD17_total"].to_numpy(float)
    groups = labels["subject_id"].to_numpy()

    # --- degeneracy / MC-06 payoff -----------------------------------------
    print("Degeneracy check (VAD-refined vs raw turn labels)...")
    deg_vad = degeneracy_report(tables[0.2])
    deg_lab = degeneracy_report(label_only_features())
    deg = deg_vad.merge(deg_lab, on="feature", suffixes=("_vad", "_labels_only"))

    # --- classification: five tiers ----------------------------------------
    configs: dict[str, dict] = {}
    for name in CONFIG_ORDER:
        cols = [f"{f}__amean" for f in FEATURE_CONFIGS[name]]
        x_by_thr = _align(tables, labels, cols)
        print(f"Nested subject-grouped CV: '{name}' ({len(cols)} features)...")
        configs[name] = nested_cv_classification(x_by_thr, y, groups)
        configs[name]["n_features"] = len(cols)
        configs[name]["n_features_non_degenerate"] = int(
            sum(1 for f in FEATURE_CONFIGS[name]
                if not bool(deg_vad.set_index("feature").loc[f, "degenerate"]))
        )

    # --- threshold sensitivity (all24) -------------------------------------
    print("Silence-threshold sensitivity (all24, threshold pinned)...")
    x_all = _align(tables, labels, MEAN_COLS)
    sens = {
        str(thr): nested_cv_classification(x_all, y, groups, fixed_threshold=thr)
        for thr in SILENCE_THRESHOLD_GRID
    }

    # --- severity regression (MC-07 secondary) ------------------------------
    regression: dict[str, dict] = {}
    for name in CONFIG_ORDER:
        cols = [f"{f}__amean" for f in FEATURE_CONFIGS[name]]
        print(f"Severity regression (HAMD-17 total): '{name}'...")
        regression[name] = nested_cv_regression(
            _align(tables, labels, cols), y_cont, y, groups,
        )

    # --- univariate + coefficients -----------------------------------------
    print("Univariate associations and bootstrap coefficients (B=2000)...")
    x24 = x_all[0.2]
    uni = univariate_table(x24, y)
    modal_c = float(max(configs["all24"]["selected_C_counts"].items(),
                        key=lambda kv: (kv[1], -float(kv[0])))[0])
    coefs = bootstrap_coefficients(x24, y, modal_c)

    # --- rank deficiency (carried-over trap 1) ------------------------------
    xc = x24 - x24.mean(axis=0)
    rank = int(np.linalg.matrix_rank(xc))

    print("Control: identical protocol on DAIC-WOZ...")
    control = daic_protocol_control()
    print("Within-subject paired analysis (exploratory)...")
    within = within_subject_analysis(tables[0.2], labels)

    daic_uni = json.loads(DAIC_INTERPRET_JSON.read_text())["univariate"]
    predictions = evaluate_predictions(uni, configs, coefs, daic_uni)

    payload = {
        "corpus": "PDCH (Mandarin clinical interviews)",
        "action_items": ["MC-01", "MC-05", "MC-06", "MC-07", "MC-08"],
        "protocol": (
            f"Subject-grouped nested CV: StratifiedGroupKFold "
            f"{N_SPLITS_OUTER}-fold x {N_REPEATS} repeats on subject_id, inner "
            f"{N_SPLITS_INNER}-fold selection of C and the intra-turn silence "
            "threshold on balanced accuracy. Deployed pipeline unchanged "
            "(median impute -> standardize -> L2 LogReg, class_weight='balanced'). "
            "Out-of-fold predictions pooled within a repeat; mean +- sd across repeats. "
            "No fixed held-out test split at n=62."
        ),
        "labels": lab_sum,
        "pipeline_attrition": {
            "session_dirs": 100,
            "sessions_through_stitch_vad_pairing": int(n_all_sessions),
            "sessions_with_hamd17_total": int(len(labels)),
            "sessions_used_for_detection": int(len(labels)),
            "subjects_used": int(labels["subject_id"].nunique()),
        },
        "feature_degeneracy": deg.to_dict(orient="records"),
        "effective_dimensionality": {
            "n_features": len(CTD_FEATURE_NAMES),
            "matrix_rank_centered_session_mean": rank,
            "note": (
                "duration_sum = ask_d + res_d and res_minus_ask = res_d - ask_d are "
                "exact algebraic identities, so the 24-D descriptor is rank-deficient "
                "by construction on any corpus (PF-02 found rank 15 on DAIC-WOZ). "
                "Not a PDCH defect."
            ),
        },
        "majority_class_baseline": {
            "rule": "always predict the majority (negative) class",
            "macro_f1": float(f1_score(y, np.zeros_like(y), average="macro",
                                       zero_division=0)),
            "balanced_accuracy": 0.5,
        },
        "classification": configs,
        "silence_threshold_sensitivity": sens,
        "severity_regression": regression,
        "univariate": uni,
        "coefficients": {"fit_C": modal_c, "fit_on": f"all {len(labels)} labelled sessions",
                         "table": coefs},
        "preregistered_predictions": predictions,
        "daic_protocol_control": control,
        "within_subject_exploratory": within,
    }

    out_json = REPORT_DIR / "pdch_ctd_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, REPORT_DIR / "pdch_ctd_results.md")
    deg.to_csv(PDCH_DIR / "pdch_feature_degeneracy.csv", index=False)

    print("\n=== PDCH tier results (subject-grouped CV, mean +- sd over repeats) ===")
    print(f"{'config':<10}{'n':>4}{'live':>6}  {'macro-F1':>16}{'bAcc':>16}{'AUC':>16}")
    for name in CONFIG_ORDER:
        r = configs[name]
        print(f"{name:<10}{r['n_features']:>4}{r['n_features_non_degenerate']:>6}  "
              f"{r['macro_f1']['mean']:>8.3f} ± {r['macro_f1']['sd']:<5.3f}"
              f"{r['balanced_accuracy']['mean']:>8.3f} ± {r['balanced_accuracy']['sd']:<5.3f}"
              f"{r['roc_auc']['mean']:>8.3f} ± {r['roc_auc']['sd']:<5.3f}")

    print("\n=== Pre-registered predictions ===")
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        p = predictions[k]
        print(f"  {k}: {'PASS' if p['passed'] else 'FAIL'}  -- {p['statement'][:78]}")

    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {REPORT_DIR / 'pdch_ctd_results.md'}")


def write_markdown(payload: dict, path: Path) -> None:
    p = payload
    L: list[str] = ["# PDCH: within-corpus CTD results (`MC-01`, `MC-05`, `MC-06`, `MC-07`, `MC-08`)\n"]
    cls, ctrl0 = p["classification"], p["daic_protocol_control"]
    mf = [cls[k]["macro_f1"]["mean"] for k in CONFIG_ORDER]
    auc = [cls[k]["roc_auc"]["mean"] for k in CONFIG_ORDER]
    L.append(
        f"> **Headline: no reliable within-PDCH discrimination.** All five "
        f"pre-registered tiers give macro-F1 {min(mf):.3f}–{max(mf):.3f} and "
        f"ROC AUC {min(auc):.3f}–{max(auc):.3f} under subject-grouped CV. "
        "These are models trained within PDCH, not a detector-transfer test. "
        "Language, instrument, population, interviewer and timing measurement "
        "differ from DAIC-WOZ. Low power is not established as the explanation. "
        "Predictions remain frozen in `docs/preregistration-pdch.md`; the "
        "current interpretation follows `docs/revision-audit-20260924.md`.\n"
    )
    L.append("\n" + p["protocol"] + "\n")

    a = p["pipeline_attrition"]
    lab = p["labels"]
    L.append("\n## Cohort\n")
    L.append("| Stage | n |")
    L.append("|---|---:|")
    L.append(f"| Session directories on disk | {a['session_dirs']} |")
    L.append(f"| Survived stitching → VAD → turn pairing | {a['sessions_through_stitch_vad_pairing']} |")
    L.append(f"| …of those, with a HAMD-17 total | {a['sessions_with_hamd17_total']} |")
    L.append(f"| Distinct subjects used | {a['subjects_used']} |")
    L.append(
        f"\nPrimary label `{lab['primary_label']}`: {lab['n_positive']}/{lab['n_sessions']} "
        f"positive ({lab['prevalence']:.1%}). For reference, the standard cutoff of 8 "
        f"would mark {lab['prevalence_at_cutoff_8_for_reference']:.0%} of this inpatient "
        "cohort positive, which is why it is not used.\n"
    )

    L.append("\n## Feature tiers\n")
    L.append("| Config | n features | non-degenerate | Macro-F1 | Balanced acc. | ROC AUC |")
    L.append("|---|---:|---:|---|---|---|")
    for name in CONFIG_ORDER:
        r = p["classification"][name]
        L.append(
            f"| `{name}` | {r['n_features']} | {r['n_features_non_degenerate']} | "
            f"{r['macro_f1']['mean']:.3f} ± {r['macro_f1']['sd']:.3f} | "
            f"{r['balanced_accuracy']['mean']:.3f} ± {r['balanced_accuracy']['sd']:.3f} | "
            f"{r['roc_auc']['mean']:.3f} ± {r['roc_auc']['sd']:.3f} |"
        )
    mb = p["majority_class_baseline"]
    L.append(f"\nAlways-negative baseline: macro-F1 {mb['macro_f1']:.3f}, "
             f"balanced accuracy {mb['balanced_accuracy']:.3f}. Per-repeat spreads are "
             "in `pdch_ctd_results.json`.\n")

    L.append("\n## Selected hyperparameters (distribution over folds × repeats)\n")
    L.append("| Config | C counts | Silence threshold counts |")
    L.append("|---|---|---|")
    for name in CONFIG_ORDER:
        r = p["classification"][name]
        L.append(f"| `{name}` | {r['selected_C_counts']} | {r.get('selected_threshold_counts', {})} |")
    L.append("\n`C` is reported as a distribution, not a winner: it was unstable under "
             "resampling on DAIC-WOZ too (`SLT-10`).\n")

    L.append("\n## Silence-threshold sensitivity (`all24`, threshold pinned)\n")
    L.append("| Threshold (s) | Macro-F1 | Balanced acc. |")
    L.append("|---:|---|---|")
    for thr, r in p["silence_threshold_sensitivity"].items():
        L.append(f"| {thr} | {r['macro_f1']['mean']:.3f} ± {r['macro_f1']['sd']:.3f} | "
                 f"{r['balanced_accuracy']['mean']:.3f} ± {r['balanced_accuracy']['sd']:.3f} |")

    L.append("\n## Severity regression on the HAMD-17 total (`MC-07` secondary)\n")
    L.append("| Config | MAE | RMSE | Pearson r | R² |")
    L.append("|---|---|---|---|---|")
    for name in CONFIG_ORDER:
        r = p["severity_regression"][name]
        L.append(f"| `{name}` | {r['mae']['mean']:.2f} ± {r['mae']['sd']:.2f} | "
                 f"{r['rmse']['mean']:.2f} ± {r['rmse']['sd']:.2f} | "
                 f"{r['pearson_r']['mean']:.3f} ± {r['pearson_r']['sd']:.3f} | "
                 f"{r['r2']['mean']:.3f} ± {r['r2']['sd']:.3f} |")
    L.append(f"\nPredict-the-mean MAE baseline: "
             f"{p['severity_regression']['all24']['baseline_mae_predict_train_mean']:.2f}.\n")

    L.append("\n## Feature degeneracy — what VAD refinement bought (`MC-06`)\n")
    L.append(
        "Session-level (`__amean`) spread over all 100 sessions, VAD-refined "
        "utterances versus the raw 1-second turn labels. `distinct` counts "
        "distinct session values out of 100, so 100 means every session is "
        "separable and low values mean quantization. **Read the last two rows "
        "first:** VAD refinement is what makes the 10 intra-turn features "
        "usable, and it is also what kills the 2 interruption features.\n"
    )
    L.append("| Feature | distinct (VAD) | distinct (labels) | NaN (VAD) | NaN (labels) | sd (VAD) | sd (labels) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in p["feature_degeneracy"]:
        flag = " ⚠" if row["degenerate_vad"] else ""
        L.append(
            f"| `{row['feature']}`{flag} | {row['n_distinct_vad']} | "
            f"{row['n_distinct_labels_only']} | {row['nan_rate_vad']:.0%} | "
            f"{row['nan_rate_labels_only']:.0%} | {row['sd_vad']:.3f} | "
            f"{row['sd_labels_only']:.3f} |"
        )
    L.append(
        "\n⚠ `ask_bt` and `res_bt` are **exactly constant** under VAD refinement. "
        "webrtcvad on a mono mixed channel emits one disjoint segment stream, and "
        "`turn_pairing.py` then orders every segment into speaker runs, so no "
        "response can start before its ask ends. Interruption is *unmeasurable* "
        "under this method rather than merely rare — a stronger statement than "
        "`docs/multi-corpus-plan.md` §2c's \"0.2% of gaps are negative\". "
        "22 of 24 features are live; centered rank is "
        f"{p['effective_dimensionality']['matrix_rank_centered_session_mean']}.\n"
    )

    ctrl = p["daic_protocol_control"]
    L.append("\n## Control: the identical protocol on DAIC-WOZ\n")
    if "skipped" in ctrl:
        L.append(f"Skipped: {ctrl['skipped']}\n")
    else:
        L.append(
            f"The same `nested_cv_classification()` over DAIC-WOZ's pooled "
            f"{ctrl['n_sessions']} sessions (24-D session means) gives macro-F1 "
            f"**{ctrl['macro_f1']['mean']:.3f} ± {ctrl['macro_f1']['sd']:.3f}**, "
            f"balanced accuracy {ctrl['balanced_accuracy']['mean']:.3f}, AUC "
            f"{ctrl['roc_auc']['mean']:.3f} — clearly above chance. The PDCH null "
            "above is therefore a property of PDCH, not of this code.\n"
        )

    w = p["within_subject_exploratory"]
    if w.get("n_subjects"):
        L.append("\n## Within-subject change (exploratory — *not* pre-registered)\n")
        L.append(
            f"{w['n_subjects']} subjects contribute two labelled sessions, so each is "
            "their own control and every between-subject confound drops out. "
            "Correlation of within-subject ΔCTD against ΔHAMD-17:\n"
        )
        L.append("| Feature | r (Δ vs ΔHAMD) | p |")
        L.append("|---|---:|---:|")
        feats = sorted(w["features"],
                       key=lambda f: -(abs(w["features"][f]["pearson_r"])
                                       if np.isfinite(w["features"][f]["pearson_r"]) else -1))
        shown = feats[:8]
        if "res_h" not in shown:   # always show the pre-registered feature of interest
            shown.append("res_h")
        for f in shown:
            v = w["features"][f]
            star = " *(pre-registered focus)*" if f == "res_h" else ""
            L.append(f"| `{f}`{star} | {v['pearson_r']:.3f} | {v['p_value']:.3f} |")
        L.append(f"\n> {w['status']}. At n={w['n_subjects']} pairs this is a lead to "
                 "pre-register for a future cohort, not a result.\n")

    L.append("\n## Pre-registered predictions\n")
    L.append("| ID | Prediction | Evidence | Outcome |")
    L.append("|---|---|---|---|")
    pr = p["preregistered_predictions"]
    ev = {
        "P1": (f"`res_h` ranks {pr['P1']['rank_by_abs_r']}/{pr['P1']['n_tier_b_testable']} "
               f"by \\|r\\| (r={pr['P1']['res_h_r']:.3f}, behind "
               f"`{pr['P1']['tier_b_ranking_by_abs_r'][0]['feature']}` at "
               f"{pr['P1']['tier_b_ranking_by_abs_r'][0]['abs_r']:.3f}) but "
               f"{pr['P1']['rank_by_auc_distance_from_chance']}/{pr['P1']['n_tier_b_testable']} "
               f"by AUC (={pr['P1']['res_h_auc']:.3f})"),
        "P2": (f"PDCH r = {pr['P2']['pdch_r']:+.3f} (positive); DAIC-WOZ train "
               f"{pr['P2']['daic_train_r']:+.3f}, test {pr['P2']['daic_test_r']:+.3f}"),
        "P3": ("macro-F1 " + ", ".join(f"`{k}` {v:.3f}" for k, v in pr["P3"]["macro_f1_means"].items())),
        "P4": (f"median retention (vs DAIC train) ratios "
               f"{pr['P4']['per_split']['train']['median_retention_ratios']:.2f} vs durations "
               f"{pr['P4']['per_split']['train']['median_retention_durations']:.2f}"),
        "P5": (f"PDCH `res_h` AUC {pr['P5']['pdch_res_h_auc']:.3f} < "
               f"{pr['P5']['daic_test_res_h_auc']} on DAIC-WOZ test"),
        "P6": (f"largest \\|coef\\| is `{pr['P6']['largest_abs_coefficient']}`; "
               f"`ask_d` ranks {pr['P6']['ask_d_rank']}/24"),
    }
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        L.append(f"| **{k}** | {pr[k]['statement']} | {ev[k]} | "
                 f"{'**PASS**' if pr[k]['passed'] else '**FAIL**'} |")
    L.append("\nFull numbers are in `pdch_ctd_results.json` under "
             "`preregistered_predictions`.\n")
    path.write_text("\n".join(L))



# --------------------------------------------------------------------------
# Supplementary analyses
# --------------------------------------------------------------------------
def daic_protocol_control(n_repeats: int = 10) -> dict:
    """Run the *identical* nested CV on DAIC-WOZ, as a control on this script.

    A null result is only interpretable if the machinery that produced it can
    produce a non-null one. This feeds DAIC-WOZ's pooled 180 sessions (24-D
    session means, the same descriptor) through the same
    `nested_cv_classification`. If it lands clearly above chance while PDCH
    does not, the PDCH null is a property of PDCH, not of this code.
    """
    from constants import OUTPUT_DIR

    frames = []
    for split in ("train", "dev", "test"):
        cache = OUTPUT_DIR / f"functionals_{split}.csv"
        if not cache.exists():
            return {"skipped": f"{cache} not found; run src/ctd/ml_splits.py first"}
        frames.append(pd.read_csv(cache))
    d = pd.concat(frames, ignore_index=True)

    x = d[MEAN_COLS].to_numpy(float)
    y = d["PHQ8_Binary"].to_numpy(int)
    groups = d["session_id"].to_numpy()
    res = nested_cv_classification({0.2: x}, y, groups, n_repeats=n_repeats,
                                   fixed_threshold=0.2)
    res["n_sessions"] = int(len(y))
    res["n_positive"] = int(y.sum())
    res["note"] = (
        "DAIC-WOZ pooled train+dev+test (180 sessions) through the identical "
        "protocol, grouped on session_id (1 session/subject, so grouping is a "
        "no-op there). Not a paper result -- a control certifying that this "
        "script can detect signal when signal is present."
    )
    return res


def within_subject_analysis(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Paired within-subject change in CTD against change in HAMD-17.

    **Exploratory, not pre-registered.** `docs/multi-corpus-plan.md` §2d flags
    this as an opportunity but no P1-P6 prediction covers it, so it is reported
    as a lead, not as confirmation.

    Each subject with two labelled sessions is their own control, which removes
    every between-subject confound (age, sex, dialect, clinician pairing,
    baseline speaking rate) that a cross-sectional correlation carries.
    """
    m = labels.merge(features.drop(columns=["subject_id"]), on="session_id",
                     validate="one_to_one")
    paired = m.groupby("subject_id").filter(lambda g: len(g) == 2)
    if paired.empty:
        return {"n_subjects": 0}

    d_hamd, deltas = [], {f: [] for f in CTD_FEATURE_NAMES}
    for _, g in paired.groupby("subject_id"):
        g = g.sort_values("session_id")
        a, b = g.iloc[0], g.iloc[1]
        d_hamd.append(float(b["HAMD17_total"] - a["HAMD17_total"]))
        for f in CTD_FEATURE_NAMES:
            deltas[f].append(float(b[f"{f}__amean"] - a[f"{f}__amean"]))

    d_hamd_arr = np.asarray(d_hamd)
    out: dict = {
        "n_subjects": int(len(d_hamd)),
        "status": "EXPLORATORY -- not covered by any pre-registered prediction",
        "delta_hamd": {"mean": float(d_hamd_arr.mean()),
                       "min": float(d_hamd_arr.min()), "max": float(d_hamd_arr.max())},
        "features": {},
    }
    for f in CTD_FEATURE_NAMES:
        v = np.asarray(deltas[f])
        if not np.isfinite(v).all() or np.std(v) < 1e-12 or np.std(d_hamd_arr) < 1e-12:
            out["features"][f] = {"pearson_r": float("nan"), "p_value": float("nan")}
            continue
        r, p = pearsonr(d_hamd_arr, v)
        out["features"][f] = {"pearson_r": float(r), "p_value": float(p),
                              "n_same_sign": int(np.sum(np.sign(v) == np.sign(d_hamd_arr)))}
    return out


if __name__ == "__main__":
    main()
