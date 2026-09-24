#!/usr/bin/env python3
"""SLT-04: interpretability for the deployed 24-D CTD session-mean detector.

The rebuttal claims the largest coefficient is `ask_d` and that `res_h` is
direction-consistent across train/dev/test. This script makes that
reproducible -- and, per `docs/plan-preflight.md` Step 3a, reports three
triangulating views instead of a single naive coefficient table, because the
24 features contain 8 near-collinear reciprocal pairs by construction and L2
splits weight arbitrarily between them:

1. Standardized coefficients (bootstrap B=2000 over train) with 95% CI,
   sign-consistency rate, and odds ratio per 1 SD.
2. Permutation importance on dev (robust to within-model collinearity;
   measures what the deployed detector actually uses).
3. Univariate direction consistency: point-biserial correlation and
   single-feature AUC, computed separately per split.

Plus a collinearity diagnostic (correlation matrix + VIF) and pair-level
aggregate importance (summed |coef| within each reciprocal pair).

Does NOT change the deployed model: LogReg C=0.3, class_weight='balanced',
threshold 0.5, `SimpleImputer(median) -> StandardScaler -> LogisticRegression`.
Coefficients are fit on train alone (matching the B=2000 train resample used
for the CI), not the train+dev refit used for the final test report in
`ml_splits.py`.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).resolve().parent
DEPRESSION_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DEPRESSION_ROOT.parent
for p in (str(PROJECT_ROOT), str(DEPRESSION_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from constants import CTD_FEATURE_NAMES, OUTPUT_DIR  # noqa: E402
from feature_groups import ASK_ONLY, CROSS, RES_ONLY  # noqa: E402
from ml_splits import build_split_features  # noqa: E402

REPORT_DIR = REPO_ROOT / "output"
DEPLOYED_C = 0.3
RANDOM_STATE = 42
N_BOOTSTRAPS = 2000

MEAN_COLS = [f"{feat}__amean" for feat in CTD_FEATURE_NAMES]

# 8 near-collinear reciprocal pairs by construction (plan-preflight.md Step 3a).
RECIPROCAL_PAIRS = [
    ("ask_ud", "ask_du"),
    ("ask_sd", "ask_ds"),
    ("ask_su", "ask_us"),
    ("res_ud", "res_du"),
    ("res_sd", "res_ds"),
    ("res_su", "res_us"),
    ("res_over_ask", "ask_over_res"),
    ("res_minus_ask", "ask_minus_res"),
]


def _group_of(feature: str) -> str:
    if feature in ASK_ONLY:
        return "ask"
    if feature in CROSS:
        return "cross"
    if feature in RES_ONLY:
        return "res"
    raise KeyError(feature)


def _load_split(split: str) -> pd.DataFrame:
    cache = OUTPUT_DIR / f"functionals_{split}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_split_features(split)
    df.to_csv(cache, index=False)
    return df


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=DEPLOYED_C, class_weight="balanced", max_iter=5000,
                random_state=RANDOM_STATE,
            )),
        ]
    )


def fit_point_estimate(x_train: np.ndarray, y_train: np.ndarray) -> tuple[Pipeline, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = _make_pipeline().fit(x_train, y_train)
    return pipe, pipe.named_steps["clf"].coef_.ravel()


def bootstrap_coefficients(x_train: np.ndarray, y_train: np.ndarray, point_coef: np.ndarray) -> dict:
    rng = np.random.RandomState(RANDOM_STATE)
    boot = np.empty((N_BOOTSTRAPS, len(point_coef)))
    for b in range(N_BOOTSTRAPS):
        xb, yb = resample(
            x_train, y_train, replace=True, stratify=y_train,
            random_state=rng.randint(0, 2**31 - 1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe_b = _make_pipeline().fit(xb, yb)
        boot[b] = pipe_b.named_steps["clf"].coef_.ravel()

    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    point_sign = np.sign(point_coef)
    sign_consistency = (np.sign(boot) == point_sign[None, :]).mean(axis=0)

    return {
        feat: {
            "coef": float(point_coef[i]),
            "ci_low": float(ci_low[i]),
            "ci_high": float(ci_high[i]),
            "sign_consistency_rate": float(sign_consistency[i]),
            "odds_ratio_per_1sd": float(np.exp(point_coef[i])),
            "group": _group_of(feat),
        }
        for i, feat in enumerate(CTD_FEATURE_NAMES)
    }


def permutation_importance_on_dev(pipe: Pipeline, x_dev: np.ndarray, y_dev: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            pipe, x_dev, y_dev, scoring="f1_macro", n_repeats=100,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    return {
        feat: {
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i, feat in enumerate(CTD_FEATURE_NAMES)
    }


def univariate_direction_consistency(data: dict[str, pd.DataFrame]) -> dict:
    out: dict = {feat: {} for feat in CTD_FEATURE_NAMES}
    for split, df in data.items():
        y = df["PHQ8_Binary"].to_numpy(int)
        for feat in CTD_FEATURE_NAMES:
            x = df[f"{feat}__amean"].to_numpy(float)
            r, p = pointbiserialr(y, x)
            auc = roc_auc_score(y, x)
            out[feat][split] = {
                "point_biserial_r": float(r),
                "p_value": float(p),
                "single_feature_auc": float(auc),
            }
    for feat in CTD_FEATURE_NAMES:
        signs = {s: np.sign(out[feat][s]["point_biserial_r"]) for s in data}
        out[feat]["sign_consistent_across_splits"] = bool(len(set(signs.values())) == 1)
    return out


def collinearity_diagnostics(x_train: np.ndarray) -> dict:
    corr = pd.DataFrame(x_train, columns=CTD_FEATURE_NAMES).corr()

    # |r| > 0.999 means an exact algebraic identity at the session-mean level,
    # not mere correlation -- distinct from (and a superset of) the 8
    # documented reciprocal pairs. This is what makes standard VIF unstable
    # below: the design matrix is rank-deficient, not just ill-conditioned.
    exact_dependencies = [
        {"pair": [a, b], "r": float(corr.loc[a, b])}
        for i, a in enumerate(CTD_FEATURE_NAMES)
        for b in CTD_FEATURE_NAMES[i + 1:]
        if abs(corr.loc[a, b]) > 0.999
    ]

    x_std = StandardScaler().fit_transform(x_train)
    x_design = np.column_stack([np.ones(len(x_std)), x_std])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vif = [
            variance_inflation_factor(x_design, i + 1)
            for i in range(len(CTD_FEATURE_NAMES))
        ]
    vif_reliable = bool(max(vif) < 1000)

    return {
        "correlation_matrix": corr.round(4).to_dict(),
        "exact_linear_dependencies": exact_dependencies,
        "vif": {feat: float(v) for feat, v in zip(CTD_FEATURE_NAMES, vif)},
        "vif_reliable": vif_reliable,
        "vif_caveat": (
            None if vif_reliable else
            "VIF is numerically unstable / not meaningful here: the design "
            "matrix is rank-deficient because of the exact linear "
            "dependencies listed in exact_linear_dependencies (not just "
            "high correlation). Use the correlation matrix and pair-level "
            "importance below instead of VIF magnitudes."
        ),
    }


def pair_level_importance(coef_table: dict) -> dict:
    out = {}
    paired = set()
    for a, b in RECIPROCAL_PAIRS:
        paired.update((a, b))
        out[f"{a}+{b}"] = {
            "sum_abs_coef": abs(coef_table[a]["coef"]) + abs(coef_table[b]["coef"]),
            "members": [a, b],
        }
    unpaired = [f for f in CTD_FEATURE_NAMES if f not in paired]
    out["_unpaired_features"] = unpaired
    return out


def write_markdown(payload: dict, path: Path) -> None:
    lines = ["# SLT-04: CTD interpretability\n"]
    lines.append(
        "Deployed model: 24-D session-mean, L2 LogReg `C=0.3`, "
        "`class_weight='balanced'`, fit on train (n=102). "
        "8 features are near-collinear reciprocal pairs by construction; "
        "see the pair-aggregate table before reading single-feature ranks.\n"
    )

    lines.append("## 1. Standardized coefficients (bootstrap B=2000, 95% CI)\n")
    lines.append("| Feature | Group | Coef | 95% CI | Sign-consistency | Odds ratio /1SD |")
    lines.append("|---|---|---:|---|---:|---:|")
    coef_table = payload["coefficients"]
    for feat in sorted(coef_table, key=lambda f: -abs(coef_table[f]["coef"])):
        c = coef_table[feat]
        lines.append(
            f"| `{feat}` | {c['group']} | {c['coef']:.3f} | "
            f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}] | "
            f"{c['sign_consistency_rate']:.2f} | {c['odds_ratio_per_1sd']:.3f} |"
        )

    lines.append("\n## 2. Permutation importance on dev (f1_macro drop, 100 repeats)\n")
    lines.append("| Feature | Mean importance | Std |")
    lines.append("|---|---:|---:|")
    perm = payload["permutation_importance_dev"]
    for feat in sorted(perm, key=lambda f: -perm[f]["importance_mean"]):
        p = perm[feat]
        lines.append(f"| `{feat}` | {p['importance_mean']:.4f} | {p['importance_std']:.4f} |")

    lines.append("\n## 3. Univariate direction consistency (point-biserial r, single-feature AUC)\n")
    lines.append("| Feature | train r | dev r | test r | train AUC | dev AUC | test AUC | Sign-consistent |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    uni = payload["univariate"]
    for feat in CTD_FEATURE_NAMES:
        u = uni[feat]
        lines.append(
            f"| `{feat}` | {u['train']['point_biserial_r']:.3f} | "
            f"{u['dev']['point_biserial_r']:.3f} | {u['test']['point_biserial_r']:.3f} | "
            f"{u['train']['single_feature_auc']:.3f} | {u['dev']['single_feature_auc']:.3f} | "
            f"{u['test']['single_feature_auc']:.3f} | "
            f"{'yes' if u['sign_consistent_across_splits'] else 'no'} |"
        )

    lines.append("\n## 4. Pair-level aggregate importance (reciprocal pairs)\n")
    lines.append("| Pair | Sum |coef| |")
    lines.append("|---|---:|")
    pair = payload["pair_level_importance"]
    for name, v in pair.items():
        if name == "_unpaired_features":
            continue
        lines.append(f"| `{name}` | {v['sum_abs_coef']:.3f} |")
    lines.append(f"\nUnpaired features: {', '.join(f'`{f}`' for f in pair['_unpaired_features'])}\n")

    lines.append("\n## 5. Collinearity diagnostic\n")
    coll = payload["collinearity"]
    if coll["exact_linear_dependencies"]:
        lines.append(
            "**Exact linear dependencies found (|r| > 0.999)** -- these are "
            "algebraic identities at the session-mean level, not just high "
            "correlation, and go beyond the 8 documented reciprocal pairs:\n"
        )
        lines.append("| Pair | r |")
        lines.append("|---|---:|")
        for dep in coll["exact_linear_dependencies"]:
            lines.append(f"| `{dep['pair'][0]}` / `{dep['pair'][1]}` | {dep['r']:.4f} |")
    if not coll["vif_reliable"]:
        lines.append(f"\n> {coll['vif_caveat']}\n")
    else:
        lines.append("\n| Feature | VIF |")
        lines.append("|---|---:|")
        vif = coll["vif"]
        for feat in sorted(vif, key=lambda f: -vif[f]):
            lines.append(f"| `{feat}` | {vif[feat]:.2f} |")
    lines.append(
        "\nFull correlation matrix is in `ctd_interpretability.json` "
        "(`collinearity.correlation_matrix`).\n"
    )
    path.write_text("\n".join(lines))


def make_forest_plot(coef_table: dict, path: Path) -> None:
    feats = sorted(coef_table, key=lambda f: coef_table[f]["coef"])
    coefs = [coef_table[f]["coef"] for f in feats]
    lo = [coef_table[f]["coef"] - coef_table[f]["ci_low"] for f in feats]
    hi = [coef_table[f]["ci_high"] - coef_table[f]["coef"] for f in feats]
    colors = {"ask": "#4C72B0", "cross": "#55A868", "res": "#C44E52"}
    bar_colors = [colors[coef_table[f]["group"]] for f in feats]

    fig, ax = plt.subplots(figsize=(7, 9))
    y_pos = np.arange(len(feats))
    ax.errorbar(coefs, y_pos, xerr=[lo, hi], fmt="none", ecolor="black",
                elinewidth=1, capsize=2, zorder=1)
    ax.scatter(coefs, y_pos, c=bar_colors, zorder=2, s=30)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Standardized coefficient (95% bootstrap CI, B=2000)")
    ax.set_title("CTD 24-D session-mean detector: coefficient forest plot")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=g, markersize=8)
               for g, c in colors.items()]
    ax.legend(handles=handles, title="group", loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = {split: _load_split(split) for split in ("train", "dev", "test")}

    x_train = data["train"][MEAN_COLS].to_numpy(float)
    y_train = data["train"]["PHQ8_Binary"].to_numpy(int)
    x_dev = data["dev"][MEAN_COLS].to_numpy(float)
    y_dev = data["dev"]["PHQ8_Binary"].to_numpy(int)

    pipe, point_coef = fit_point_estimate(x_train, y_train)
    print("Bootstrapping coefficients (B=2000 over train)...")
    coef_table = bootstrap_coefficients(x_train, y_train, point_coef)

    print("Permutation importance on dev (100 repeats)...")
    perm = permutation_importance_on_dev(pipe, x_dev, y_dev)

    print("Univariate direction consistency per split...")
    uni = univariate_direction_consistency(data)

    print("Collinearity diagnostics (correlation + VIF)...")
    coll = collinearity_diagnostics(x_train)

    pair_imp = pair_level_importance(coef_table)

    top_by_abs_coef = max(coef_table, key=lambda f: abs(coef_table[f]["coef"]))
    res_h_sign_consistent = uni["res_h"]["sign_consistent_across_splits"]

    payload = {
        "deployed_model": {
            "features": "24-D session-mean (__amean)",
            "classifier": "LogisticRegression(C=0.3, class_weight='balanced')",
            "fit_on": "train (n=102) -- matches the B=2000 train-resample bootstrap population",
        },
        "coefficients": coef_table,
        "permutation_importance_dev": perm,
        "univariate": uni,
        "collinearity": coll,
        "pair_level_importance": pair_imp,
        "rebuttal_claims_check": {
            "largest_abs_coefficient": top_by_abs_coef,
            "matches_rebuttal_ask_d_claim": top_by_abs_coef == "ask_d",
            "res_h_sign_consistent_train_dev_test": res_h_sign_consistent,
        },
    }

    out_json = REPORT_DIR / "ctd_interpretability.json"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, REPORT_DIR / "ctd_interpretability.md")
    make_forest_plot(coef_table, REPORT_DIR / "ctd_coef_forest.png")

    print(f"\nLargest |coefficient|: {top_by_abs_coef} "
          f"(rebuttal claimed ask_d: {'match' if top_by_abs_coef == 'ask_d' else 'MISMATCH'})")
    print(f"res_h sign-consistent across train/dev/test: {res_h_sign_consistent}")
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {REPORT_DIR / 'ctd_interpretability.md'}")
    print(f"Saved -> {REPORT_DIR / 'ctd_coef_forest.png'}")


if __name__ == "__main__":
    main()
