#!/usr/bin/env python3
"""Shared-signal audit across DAIC-WOZ and PDCH (`SLT-01`, `SLT-03`, `MC-02`).

Runs the experiment family pre-declared in
`docs/exploratory-shared-signal-pdch-daic.md`, which was committed before any
number below existed. Read that file first: it fixes the estimation sets, the
four test families, the multiplicity policy, and the pass criteria, and it
discloses per-quantity what had already been seen.

The question is **not** whether the 24-D CTD detector can be made to work on
PDCH. `output/pdch_ctd_results.md` already answered that: it cannot. The
question is what, if anything, is commonly informative about depression-related
labels across the two corpora, and what fails for an identifiable reason.

Stages
------
* **E0** inventory and distribution shift on the 22 live features, no labels.
* **E1** confound checks: matched PDCH subset, within-fold residualization on
  session length (run on both corpora), and a power calibration that asks what
  DAIC-WOZ looks like at PDCH's cohort size.
* **E2** the univariate concordance map -- the headline artifact.
* **E3** parsimonious candidate detectors, run only on features F1 promotes.
* **E4** continuous and psychomotor item-level targets.

Nothing here pools labels: `PHQ8_Binary`/`PHQ8_Score` and
`HAMD17_ge17`/`HAMD17_total` never share a column, and every supervised fit is
within one corpus. Cross-corpus statements are comparisons of effects, ranks
and signs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
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

PROJECT_ROOT = Path(__file__).resolve().parent
DEPRESSION_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DEPRESSION_ROOT.parent
for _p in (str(PROJECT_ROOT), str(DEPRESSION_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import CTD_FEATURE_NAMES, DEFAULT_DATA_ROOT, OUTPUT_DIR  # noqa: E402
from data_loading import transcript_path  # noqa: E402
from feature_groups import FEATURE_CONFIGS, group_of  # noqa: E402
from pdch_experiment import (  # noqa: E402
    C_GRID,
    N_SPLITS_INNER,
    N_SPLITS_OUTER,
    RANDOM_STATE,
    _fit,
    _mean_sd,
    nested_cv_classification,
)
from pdch_features import build_pdch_features, load_utterance_cache  # noqa: E402
from pdch_labels import HAMD_SHEET, HAMD_XLSX, load_pdch_labels  # noqa: E402

sys.path.insert(0, str(DEPRESSION_ROOT))
from common.transcript_preprocessing import load_transcript  # noqa: E402

REPORT_DIR = REPO_ROOT / "output"

# ---------------------------------------------------------------------------
# Constants fixed by docs/exploratory-shared-signal-pdch-daic.md
# ---------------------------------------------------------------------------
DEAD_ON_PDCH = ("ask_bt", "res_bt")
LIVE_FEATURES: list[str] = [f for f in CTD_FEATURE_NAMES if f not in DEAD_ON_PDCH]
assert len(LIVE_FEATURES) == 22

SILENCE_THRESHOLD = 0.2          # pinned on both corpora, per the declaration
N_PERM = 5000                    # subject-grouped label permutations
PERM_SEED = 20260924
FDR_Q = 0.10
WEAK_ABS_R = 0.15                # weak-concordance tier floor
ABSOLUTE_TIME_FEATURES = (
    "ask_d", "res_d", "res_minus_ask", "ask_minus_res", "duration_sum", "res_h",
)
F4_ROBUST_COLS = ("res_h__pctl50", "res_h__cv", "res_h__pctlrange20_80")
F4_RESID_FEATURES = ("res_h", "ask_d", "res_d", "duration_sum")
HAMD_RETARDATION_ITEM = 8
HAMD_AGITATION_ITEM = 9

MEAN_COL = "{}__amean".format


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def corr_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r between every column of `x` and `y`, vectorised."""
    yc = y - y.mean()
    xc = x - x.mean(axis=0)
    den = np.sqrt((yc ** 2).sum() * (xc ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (yc @ xc) / den


def rank_cols(x: np.ndarray) -> np.ndarray:
    """Column-wise average ranks (for Spearman via Pearson-on-ranks)."""
    out = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        out[:, j] = pd.Series(x[:, j]).rank().to_numpy()
    return out


class SubjectPermuter:
    """Label permutations that move each subject's label vector as a unit.

    16 of PDCH's 46 labelled subjects contribute two sessions, so a plain
    session-level shuffle would destroy the within-subject dependence that the
    real data carries and make the null too narrow. Blocks are permuted only
    among blocks of the same size, which preserves both the label multiset and
    the within-subject structure exactly. On DAIC-WOZ every subject is one
    session, so this degenerates to an ordinary shuffle.
    """

    def __init__(self, subject_ids: np.ndarray, seed: int = PERM_SEED):
        self.rng = np.random.RandomState(seed)
        blocks: dict[str, list[int]] = {}
        for i, s in enumerate(subject_ids):
            blocks.setdefault(str(s), []).append(i)
        self.by_size: dict[int, list[np.ndarray]] = {}
        for idx in blocks.values():
            self.by_size.setdefault(len(idx), []).append(np.asarray(idx))
        self.n = len(subject_ids)

    def take(self) -> np.ndarray:
        """One permutation, as an index array to apply to the label vector."""
        take = np.empty(self.n, dtype=int)
        for size, blocks in self.by_size.items():
            order = self.rng.permutation(len(blocks))
            for dst, src in enumerate(order):
                take[blocks[dst]] = blocks[src]
        return take

    def describe(self) -> dict:
        return {
            "n_rows": int(self.n),
            "block_sizes": {str(k): len(v) for k, v in sorted(self.by_size.items())},
        }


def permutation_association(
    x: np.ndarray, y: np.ndarray, permuter: SubjectPermuter, n_perm: int = N_PERM,
    spearman: bool = False,
) -> dict:
    """Two-sided permutation p per column plus the max-|r| null distribution."""
    xx = rank_cols(x) if spearman else x
    yy = pd.Series(y).rank().to_numpy() if spearman else y.astype(float)
    r_obs = corr_vec(xx, yy)
    ge = np.zeros(x.shape[1], dtype=int)
    max_null = np.empty(n_perm)
    for b in range(n_perm):
        r_null = corr_vec(xx, yy[permuter.take()])
        ge += (np.abs(r_null) >= np.abs(r_obs) - 1e-12).astype(int)
        max_null[b] = np.nanmax(np.abs(r_null))
    p = (ge + 1.0) / (n_perm + 1.0)
    return {
        "r": r_obs,
        "p_perm": p,
        "max_abs_r_null": max_null,
        "max_abs_r_observed": float(np.nanmax(np.abs(r_obs))),
        "max_abs_r_p": float(
            (np.sum(max_null >= np.nanmax(np.abs(r_obs)) - 1e-12) + 1) / (n_perm + 1)
        ),
    }


# ---------------------------------------------------------------------------
# within-fold residualization (E1b, F4)
# ---------------------------------------------------------------------------
class ResidualizeOnTrailingConfounds(BaseEstimator, TransformerMixin):
    """Regress selected feature columns on trailing confound columns, drop them.

    The confounds (log dialogue span, log turn count) are appended as the last
    columns of `X`. The OLS coefficients are estimated in `fit`, i.e. on the
    training fold only, so no test-fold information reaches the residuals.
    """

    def __init__(self, n_confounds: int = 2, target_idx: tuple[int, ...] = ()):
        self.n_confounds = n_confounds
        self.target_idx = target_idx

    def _design(self, x: np.ndarray) -> np.ndarray:
        z = x[:, x.shape[1] - self.n_confounds:]
        return np.column_stack([np.ones(len(z)), z])

    def fit(self, x, y=None):
        d = self._design(np.asarray(x, dtype=float))
        xf = np.asarray(x, dtype=float)
        self.betas_ = {
            j: np.linalg.lstsq(d, xf[:, j], rcond=None)[0] for j in self.target_idx
        }
        return self

    def transform(self, x):
        xf = np.asarray(x, dtype=float)
        d = self._design(xf)
        out = xf[:, : xf.shape[1] - self.n_confounds].copy()
        for j, beta in self.betas_.items():
            out[:, j] = xf[:, j] - d @ beta
        return out


def residualizing_pipeline_factory(target_idx: tuple[int, ...]):
    """Deployed pipeline with a residualization step inserted after imputation."""

    def factory(c: float) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("resid", ResidualizeOnTrailingConfounds(2, target_idx)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                       random_state=RANDOM_STATE)),
        ])

    return factory


def plain_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                   random_state=RANDOM_STATE)),
    ])


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------
def load_daic() -> pd.DataFrame:
    """DAIC-WOZ session table: 240 functionals + labels + confounds, 180 rows."""
    frames = []
    for split in ("train", "dev", "test"):
        cache = OUTPUT_DIR / f"functionals_{split}.csv"
        if not cache.exists():
            raise FileNotFoundError(f"{cache} missing -- run src/ctd/ml_splits.py first.")
        frames.append(pd.read_csv(cache))
    d = pd.concat(frames, ignore_index=True)

    # Dialogue span, defined identically to PDCH: last utterance end - first start.
    spans = []
    for sid in d["session_id"]:
        t = load_transcript(transcript_path(DEFAULT_DATA_ROOT, int(sid)))
        spans.append(float(t["stop_time"].max() - t["start_time"].min()))
    d["span_s"] = spans
    d["n_turn_pairs"] = d["n_turns"].astype(int)

    # PHQ8_Moving: official AVEC2017 item column, read as-is (see F3).
    items = []
    for split, fname in (("train", "train_split_Depression_AVEC2017.csv"),
                         ("dev", "dev_split_Depression_AVEC2017.csv"),
                         ("test", "full_test_split.csv")):
        path = Path(DEFAULT_DATA_ROOT) / "labels" / fname
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw.columns = [c.strip() for c in raw.columns]
        idcol = "Participant_ID" if "Participant_ID" in raw.columns else raw.columns[0]
        if "PHQ8_Moving" in raw.columns:
            items.append(raw[[idcol, "PHQ8_Moving"]].rename(
                columns={idcol: "session_id", "PHQ8_Moving": "PHQ8_Moving"}))
    if items:
        it = pd.concat(items, ignore_index=True).drop_duplicates("session_id")
        d = d.merge(it, on="session_id", how="left")
    else:
        d["PHQ8_Moving"] = np.nan
    d["subject_id"] = d["session_id"].astype(str)
    return d


def load_pdch() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all 100 sessions with features, the 62 labelled ones with labels+items)."""
    utts = load_utterance_cache()
    feats = build_pdch_features(utts, SILENCE_THRESHOLD)
    span = {
        sid: float(max(u.end for u in us) - min(u.start for u in us))
        for sid, us in utts.items() if us
    }
    feats["span_s"] = feats["session_id"].map(span)

    labels = load_pdch_labels()
    raw = pd.read_excel(HAMD_XLSX, sheet_name=HAMD_SHEET)
    raw = raw.rename(columns={"Serial": "session_id"})
    raw["session_id"] = raw["session_id"].astype(str).str.strip()
    items = raw[["session_id", HAMD_RETARDATION_ITEM, HAMD_AGITATION_ITEM]].rename(
        columns={HAMD_RETARDATION_ITEM: "hamd_item8_retardation",
                 HAMD_AGITATION_ITEM: "hamd_item9_agitation"})
    labels = labels.merge(items, on="session_id", how="left")

    labelled = labels.merge(feats, on=["session_id", "subject_id"], how="inner")
    labelled = labelled.sort_values("session_id").reset_index(drop=True)
    return feats, labelled


# ---------------------------------------------------------------------------
# E0 -- inventory and distribution shift (no labels)
# ---------------------------------------------------------------------------
def e0_inventory(daic: pd.DataFrame, pdch_all: pd.DataFrame) -> dict:
    def qs(v: np.ndarray) -> dict:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "median": float(np.median(v)),
                "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                "mean": float(v.mean()), "sd": float(v.std(ddof=1))}

    session_level = {
        "dialogue_span_s": {"daic": qs(daic["span_s"]), "pdch": qs(pdch_all["span_s"])},
        "n_turn_pairs": {"daic": qs(daic["n_turn_pairs"]),
                         "pdch": qs(pdch_all["n_turn_pairs"])},
    }
    for feat in ("ask_d", "res_d", "res_over_ask", "ask_over_res", "res_h"):
        session_level[f"{feat}__amean"] = {
            "daic": qs(daic[MEAN_COL(feat)]), "pdch": qs(pdch_all[MEAN_COL(feat)])}

    rows = []
    for feat in LIVE_FEATURES:
        a = daic[MEAN_COL(feat)].to_numpy(float)
        b = pdch_all[MEAN_COL(feat)].to_numpy(float)
        af, bf = a[np.isfinite(a)], b[np.isfinite(b)]
        ks = ks_2samp(af, bf) if af.size and bf.size else None
        da, db = qs(a), qs(b)

        def near_degenerate(v: np.ndarray, raw: np.ndarray) -> bool:
            if v.size < 2:
                return True
            med = np.median(np.abs(v))
            scale = np.std(v, ddof=1) / (abs(med) if abs(med) > 1e-9 else 1.0)
            n_distinct = len(np.unique(np.round(v, 6)))
            return bool(scale < 0.02 or n_distinct < 0.25 * raw.size)

        rows.append({
            "feature": feat,
            "group": group_of(feat),
            "daic_median": da["median"], "daic_iqr": da["iqr"], "daic_sd": da["sd"],
            "daic_nan_rate": float(np.mean(~np.isfinite(a))),
            "daic_n_distinct": int(len(np.unique(np.round(af, 6)))),
            "pdch_median": db["median"], "pdch_iqr": db["iqr"], "pdch_sd": db["sd"],
            "pdch_nan_rate": float(np.mean(~np.isfinite(b))),
            "pdch_n_distinct": int(len(np.unique(np.round(bf, 6)))),
            "median_ratio_pdch_over_daic": (
                float(db["median"] / da["median"]) if abs(da["median"]) > 1e-9 else float("nan")),
            "ks_statistic": float(ks.statistic) if ks else float("nan"),
            "ks_p": float(ks.pvalue) if ks else float("nan"),
            "daic_near_degenerate": near_degenerate(af, a),
            "pdch_near_degenerate": near_degenerate(bf, b),
        })
    shift = pd.DataFrame(rows)

    # The intra-turn ratio features are algebraic transforms of ONE quantity --
    # the silence fraction of that speaker's turn -- because turn duration is
    # exactly speech + silence. This is why tier C collapses, and it holds on
    # both corpora by construction rather than by correlation.
    def redundancy(df: pd.DataFrame, side: str) -> dict:
        ud = df[MEAN_COL(f"{side}_ud")].to_numpy(float)
        sd = df[MEAN_COL(f"{side}_sd")].to_numpy(float)
        ok = np.isfinite(ud) & np.isfinite(sd)
        x = df[[MEAN_COL(f) for f in LIVE_FEATURES]].to_numpy(float)
        x = x[np.isfinite(x).all(axis=1)]
        return {
            "identity": f"{side}_ud + {side}_sd == 1",
            "max_abs_deviation": float(np.max(np.abs(ud[ok] + sd[ok] - 1.0))),
            "derived_from_silence_fraction": [
                f"{side}_ud", f"{side}_sd", f"{side}_du", f"{side}_su",
                f"{side}_ds", f"{side}_us",
            ],
            "matrix_rank_22_live_centered": int(np.linalg.matrix_rank(x - x.mean(axis=0))),
        }

    redundancy_note = {
        "note": (
            "Turn duration is exactly speech + silence, so *_ud, *_sd, *_du, "
            "*_su, *_ds and *_us are six monotone transforms of a single "
            "quantity: that speaker's within-turn silence fraction. Twelve of "
            "the 22 live features therefore carry two underlying numbers. This "
            "is a property of the feature definitions, not of either corpus, and "
            "it is why the 8-feature res_only tier behaves like a ~3-D "
            "descriptor (res_d, res_st, and the response silence fraction)."
        ),
        "daic": {side: redundancy(daic, side) for side in ("ask", "res")},
        "pdch": {side: redundancy(pdch_all, side) for side in ("ask", "res")},
    }
    return {
        "algebraic_redundancy": redundancy_note,
        "note": (
            "Unsupervised: no label is touched. DAIC-WOZ uses all 180 cleaned "
            "sessions and PDCH all 100 pipeline-complete sessions, including the "
            "38 without a HAMD-17 total, because nothing here is label-dependent. "
            "KS is computed on raw units and is therefore expected to reject for "
            "almost every absolute-duration feature -- the informative columns are "
            "the median ratio and the degeneracy flags, not the KS p."
        ),
        "session_level": session_level,
        "per_feature_shift": shift.to_dict(orient="records"),
        "dead_on_pdch": list(DEAD_ON_PDCH),
    }


# ---------------------------------------------------------------------------
# association machinery shared by E2 (F1) and E4 (F2, F3, F4)
# ---------------------------------------------------------------------------
def _design(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, int]:
    """Median-impute (the deployed imputer) and report how many cells were filled."""
    x = df[cols].to_numpy(float)
    n_missing = int(np.sum(~np.isfinite(x)))
    x = np.where(np.isfinite(x), x, np.nan)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(np.isnan(x))
    x[idx] = np.take(med, idx[1])
    return x, n_missing


def association_family(
    df: pd.DataFrame, feature_cols: list[str], target: str, permuter: SubjectPermuter,
    binary: bool, spearman: bool, n_perm: int = N_PERM,
) -> dict:
    """One (corpus, target) cell of a test family, FDR-corrected within itself."""
    keep = df[target].notna().to_numpy()
    sub = df.loc[keep]
    x, n_missing = _design(sub, feature_cols)
    y = sub[target].to_numpy(float)

    live = np.std(x, axis=0) > 1e-12
    perm = permutation_association(x[:, live], y, permuter, n_perm, spearman=spearman)
    q_live = bh_fdr(perm["p_perm"])

    out: dict = {}
    li = 0
    for j, col in enumerate(feature_cols):
        if not live[j]:
            out[col] = {"degenerate": True}
            continue
        rec = {
            "degenerate": False,
            "r": float(perm["r"][li]),
            "p_perm": float(perm["p_perm"][li]),
            "q_bh": float(q_live[li]),
        }
        if binary:
            rec["auc"] = float(roc_auc_score(y.astype(int), x[:, j]))
        out[col] = rec
        li += 1

    return {
        "target": target,
        "statistic": "spearman_rho" if spearman else "point_biserial_r",
        "n": int(keep.sum()),
        "n_features_tested": int(live.sum()),
        "n_cells_median_imputed": n_missing,
        "n_permutations": n_perm,
        "permutation_scheme": permuter.describe(),
        "max_abs_r_observed": perm["max_abs_r_observed"],
        "max_abs_r_permutation_p": perm["max_abs_r_p"],
        "max_abs_r_null_q95": float(np.percentile(perm["max_abs_r_null"], 95)),
        "per_feature": out,
    }


def concordance(daic_cell: dict, pdch_cell: dict, feats: list[str]) -> dict:
    """Apply the pre-declared classification rule to one pair of cells."""
    rows, counts = [], {"confirmed_shared": 0, "opposite_sign": 0,
                        "daic_only": 0, "pdch_only": 0, "null": 0,
                        "weak_concordant": 0}
    n_sign_agree = n_comparable = 0
    for f in feats:
        a, b = daic_cell["per_feature"][MEAN_COL(f)], pdch_cell["per_feature"][MEAN_COL(f)]
        if a.get("degenerate") or b.get("degenerate"):
            rows.append({"feature": f, "group": group_of(f), "verdict": "degenerate"})
            continue
        sig_a, sig_b = a["q_bh"] <= FDR_Q, b["q_bh"] <= FDR_Q
        same_sign = np.sign(a["r"]) == np.sign(b["r"])
        n_comparable += 1
        n_sign_agree += int(bool(same_sign))

        if sig_a and sig_b:
            verdict = "confirmed_shared" if same_sign else "opposite_sign"
        elif sig_a:
            verdict = "daic_only"
        elif sig_b:
            verdict = "pdch_only"
        else:
            verdict = "null"
        counts[verdict] += 1

        weak = bool(same_sign and abs(a["r"]) >= WEAK_ABS_R and abs(b["r"]) >= WEAK_ABS_R
                    and min(a["p_perm"], b["p_perm"]) < 0.05)
        counts["weak_concordant"] += int(weak)
        rows.append({
            "feature": f, "group": group_of(f),
            "daic_r": a["r"], "daic_auc": a.get("auc"), "daic_p": a["p_perm"],
            "daic_q": a["q_bh"],
            "pdch_r": b["r"], "pdch_auc": b.get("auc"), "pdch_p": b["p_perm"],
            "pdch_q": b["q_bh"],
            "same_sign": bool(same_sign),
            "verdict": verdict,
            "weak_concordant_exploratory": weak,
        })

    from scipy.stats import binomtest
    sign_p = float(binomtest(n_sign_agree, n_comparable, 0.5).pvalue) if n_comparable else float("nan")
    return {
        "rule": (
            f"confirmed_shared = BH-q <= {FDR_Q} in BOTH corpora and matching sign; "
            f"weak_concordant (EXPLORATORY) = matching sign, |r| >= {WEAK_ABS_R} in "
            "both, and permutation p < 0.05 in at least one."
        ),
        "table": rows,
        "counts": counts,
        "sign_agreement": {
            "n_agree": n_sign_agree, "n_comparable": n_comparable,
            "binomial_p_vs_half": sign_p,
            "caveat": (
                "Descriptive only. The 22 features are rank-deficient by "
                "construction (rank 13 for the 22 live features on both corpora), so these are not "
                "independent tests and the binomial p is anti-conservative."
            ),
        },
        "promoted_confirmed": [r["feature"] for r in rows
                               if r.get("verdict") == "confirmed_shared"],
        "promoted_weak_exploratory": [r["feature"] for r in rows
                                      if r.get("weak_concordant_exploratory")],
        "descriptive_sign_reversals": {
            "note": (
                "DESCRIPTIVE, not a promoted verdict. Features whose effect is "
                "non-trivial in both corpora (|r| >= "
                f"{WEAK_ABS_R}) but points the opposite way. The pre-declared rule "
                "only labels a feature 'opposite_sign' when it clears FDR in both "
                "corpora, which nothing does at these sample sizes; this list is "
                "reported because a reversal is diagnostic of an interpretation "
                "failure even when neither side is individually significant."
            ),
            "features": [
                {"feature": r["feature"], "group": r["group"],
                 "daic_r": r["daic_r"], "pdch_r": r["pdch_r"],
                 "daic_q": r["daic_q"], "pdch_q": r["pdch_q"]}
                for r in rows
                if r.get("verdict") not in (None, "degenerate")
                and not r["same_sign"]
                and abs(r["daic_r"]) >= WEAK_ABS_R and abs(r["pdch_r"]) >= WEAK_ABS_R
            ],
        },
    }


# ---------------------------------------------------------------------------
# E1 -- confound checks
# ---------------------------------------------------------------------------
def match_on_confounds(df: pd.DataFrame, ycol: str, seed: int = RANDOM_STATE) -> dict:
    """Greedy 1:1 nearest-neighbour match of positives to negatives on (span, turns)."""
    z = np.column_stack([
        np.log(df["span_s"].to_numpy(float)),
        np.log1p(df["n_turn_pairs"].to_numpy(float)),
    ])
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=0)
    y = df[ycol].to_numpy(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    # Optimal (not greedy) 1:1 assignment: minimises total matched distance and
    # is deterministic, which a greedy pass over a random ordering is not.
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(z[pos][:, None, :] - z[neg][None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(int(pos[a]), int(neg[b]), float(cost[a, b])) for a, b in zip(ri, ci)]

    keep = sorted([i for p in pairs for i in p[:2]])

    def smd(idx):
        a = z[[i for i in idx if y[i] == 1]]
        b = z[[i for i in idx if y[i] == 0]]
        pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2)
        return (a.mean(axis=0) - b.mean(axis=0)) / np.where(pooled > 1e-9, pooled, 1.0)

    all_idx = list(range(len(df)))
    return {
        "n_pairs": len(pairs),
        "kept_index": keep,
        "confounds": ["log_span_s", "log1p_n_turn_pairs"],
        "standardized_mean_diff_before": [float(v) for v in smd(all_idx)],
        "standardized_mean_diff_after": [float(v) for v in smd(keep)],
        "mean_match_distance": float(np.mean([p[2] for p in pairs])) if pairs else float("nan"),
    }


def confound_augmented_matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """[features | log span | log1p turn count] for the residualizing pipeline."""
    x = df[[MEAN_COL(f) for f in feats]].to_numpy(float)
    z = np.column_stack([
        np.log(df["span_s"].to_numpy(float)),
        np.log1p(df["n_turn_pairs"].to_numpy(float)),
    ])
    return np.column_stack([x, z])


def e1_confounds(daic: pd.DataFrame, pdch: pd.DataFrame, n_repeats: int = 20) -> dict:
    out: dict = {}

    # --- E1a: matched PDCH subset ------------------------------------------
    m = match_on_confounds(pdch, "HAMD17_ge17")
    sub = pdch.iloc[m["kept_index"]].reset_index(drop=True)
    y = sub["HAMD17_ge17"].to_numpy(int)
    g = sub["subject_id"].to_numpy()
    matched = {}
    for cfg in ("all24", "no_ask"):
        cols = [MEAN_COL(f) for f in FEATURE_CONFIGS[cfg]]
        x = sub[cols].to_numpy(float)
        matched[cfg] = nested_cv_classification({SILENCE_THRESHOLD: x}, y, g,
                                                n_repeats=n_repeats,
                                                fixed_threshold=SILENCE_THRESHOLD)
    out["e1a_matched_subset"] = {
        "matching": {k: v for k, v in m.items() if k != "kept_index"},
        "n_sessions": int(len(sub)),
        "n_subjects": int(sub["subject_id"].nunique()),
        "prevalence": float(y.mean()),
        "results": matched,
        "softens_criterion": "mean macro-F1 >= 0.60 AND mean ROC AUC >= 0.60",
        "softens": {
            cfg: bool(matched[cfg]["macro_f1"]["mean"] >= 0.60
                      and matched[cfg]["roc_auc"]["mean"] >= 0.60)
            for cfg in matched
        },
    }

    # --- E1b: within-fold residualization, both corpora ---------------------
    resid = {}
    for corpus, df, ycol, gcol in (("pdch", pdch, "HAMD17_ge17", "subject_id"),
                                   ("daic", daic, "PHQ8_Binary", "subject_id")):
        resid[corpus] = {}
        for cfg in ("all24", "no_ask"):
            feats = FEATURE_CONFIGS[cfg]
            tgt = tuple(i for i, f in enumerate(feats) if f in ABSOLUTE_TIME_FEATURES)
            xa = confound_augmented_matrix(df, feats)
            yv = df[ycol].to_numpy(int)
            gv = df[gcol].to_numpy()
            resid[corpus][cfg] = {
                "residualized": nested_cv_classification(
                    {SILENCE_THRESHOLD: xa}, yv, gv, n_repeats=n_repeats,
                    fixed_threshold=SILENCE_THRESHOLD,
                    pipeline_factory=residualizing_pipeline_factory(tgt)),
                "unresidualized_same_folds": nested_cv_classification(
                    {SILENCE_THRESHOLD: xa[:, :len(feats)]}, yv, gv,
                    n_repeats=n_repeats, fixed_threshold=SILENCE_THRESHOLD,
                    pipeline_factory=plain_pipeline),
                "n_features_residualized": len(tgt),
            }
    out["e1b_residualized"] = {
        "note": (
            "The six absolute-time features (ask_d, res_d, res_minus_ask, "
            "ask_minus_res, duration_sum, res_h) are regressed on log dialogue "
            "span and log turn count, with the OLS fitted on the training fold "
            "only. The DAIC-WOZ run is the control: if its signal also collapses, "
            "the deployed detector is partly a session-length detector."
        ),
        "results": resid,
    }

    # --- E1c: power calibration -- DAIC at PDCH's cohort shape --------------
    out["e1c_power_calibration"] = daic_at_pdch_size(daic, pdch)
    return out


def daic_at_pdch_size(daic: pd.DataFrame, pdch: pd.DataFrame, n_subsample: int = 100,
                      n_repeats: int = 3) -> dict:
    """What does a corpus with known signal look like at n=62, 43.5% positive?

    This sensitivity analysis matches session count and prevalence, not the
    repeated-subject structure, instrument, population or measurement pipeline.
    It cannot identify low power as the explanation for the PDCH null.
    """
    n_total = int(len(pdch))
    prevalence = float(pdch["HAMD17_ge17"].mean())
    n_pos = int(round(prevalence * n_total))
    n_neg = n_total - n_pos

    cols = [MEAN_COL(f) for f in FEATURE_CONFIGS["all24"]]
    x_all = daic[cols].to_numpy(float)
    y_all = daic["PHQ8_Binary"].to_numpy(int)
    pos = np.where(y_all == 1)[0]
    neg = np.where(y_all == 0)[0]
    rng = np.random.RandomState(RANDOM_STATE)

    per_sub = []
    for _ in range(n_subsample):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, n_neg, replace=False)])
        idx.sort()
        r = nested_cv_classification({SILENCE_THRESHOLD: x_all[idx]}, y_all[idx],
                                     daic["subject_id"].to_numpy()[idx],
                                     n_repeats=n_repeats,
                                     fixed_threshold=SILENCE_THRESHOLD)
        per_sub.append({"macro_f1": r["macro_f1"]["mean"],
                        "balanced_accuracy": r["balanced_accuracy"]["mean"],
                        "roc_auc": r["roc_auc"]["mean"]})

    # Descriptive location using the previously reported rounded PDCH result.
    # This is not a formal power analysis or an attribution of the null.
    pdch_observed = {"macro_f1": 0.520, "balanced_accuracy": 0.522, "roc_auc": 0.476}
    located = {
        k: {
            "pdch_observed": v,
            "percentile_within_daic_at_n62": float(
                100.0 * np.mean([p[k] <= v for p in per_sub])),
            "z_vs_daic_at_n62": float(
                (v - np.mean([p[k] for p in per_sub]))
                / np.std([p[k] for p in per_sub], ddof=1)),
        }
        for k, v in pdch_observed.items()
    }
    return {
        "pdch_located_in_daic_null": located,
        "pdch_observed_source": "output/pdch_ctd_results.md, all24 tier",
        "design": (
            f"DAIC-WOZ (180 sessions, 24-D session means) subsampled {n_subsample} "
            f"times to PDCH's session count and prevalence: n={n_total} at {prevalence:.1%} positive "
            f"({n_pos}/{n_neg}), each run through the identical subject-grouped "
            f"nested CV with {n_repeats} repeats. Repeated-subject structure and "
            "other corpus characteristics are not matched; this cannot identify "
            "low power as the explanation for the PDCH null."
        ),
        "n_subsamples": n_subsample,
        "macro_f1": _mean_sd([p["macro_f1"] for p in per_sub]),
        "balanced_accuracy": _mean_sd([p["balanced_accuracy"] for p in per_sub]),
        "roc_auc": _mean_sd([p["roc_auc"] for p in per_sub]),
        "macro_f1_percentiles": {
            str(q): float(np.percentile([p["macro_f1"] for p in per_sub], q))
            for q in (5, 25, 50, 75, 95)
        },
    }


# ---------------------------------------------------------------------------
# E2 -- the univariate concordance map (family F1)
# ---------------------------------------------------------------------------
def e2_concordance(daic: pd.DataFrame, pdch: pd.DataFrame) -> dict:
    est = daic[daic["split"].isin(["train", "dev"])].reset_index(drop=True)
    cols = [MEAN_COL(f) for f in LIVE_FEATURES]

    daic_cell = association_family(
        est, cols, "PHQ8_Binary",
        SubjectPermuter(est["subject_id"].to_numpy(), PERM_SEED),
        binary=True, spearman=False)
    pdch_cell = association_family(
        pdch, cols, "HAMD17_ge17",
        SubjectPermuter(pdch["subject_id"].to_numpy(), PERM_SEED),
        binary=True, spearman=False)

    return {
        "family": "F1 (primary) -- binary-label univariate concordance",
        "estimation_sets": {
            "daic": "train+dev pooled, n=135; the official test split is withheld",
            "pdch": "all labelled sessions, n=62 / 46 subjects",
        },
        "daic": daic_cell,
        "pdch": pdch_cell,
        "concordance": concordance(daic_cell, pdch_cell, LIVE_FEATURES),
    }


# ---------------------------------------------------------------------------
# E4 -- continuous severity (F2), psychomotor items (F3), variants (F4)
# ---------------------------------------------------------------------------
def e4_secondary_targets(daic: pd.DataFrame, pdch: pd.DataFrame,
                         n_repeats: int = 20) -> dict:
    est = daic[daic["split"].isin(["train", "dev"])].reset_index(drop=True)
    cols = [MEAN_COL(f) for f in LIVE_FEATURES]
    dperm = SubjectPermuter(est["subject_id"].to_numpy(), PERM_SEED)
    pperm = SubjectPermuter(pdch["subject_id"].to_numpy(), PERM_SEED)

    # --- F2: continuous severity -------------------------------------------
    f2_daic = association_family(est, cols, "PHQ8_Score", dperm, binary=False,
                                 spearman=True)
    f2_pdch = association_family(pdch, cols, "HAMD17_total", pperm, binary=False,
                                 spearman=True)
    f2 = {
        "family": "F2 (secondary) -- continuous-severity concordance (Spearman)",
        "daic": f2_daic, "pdch": f2_pdch,
        "concordance": concordance(f2_daic, f2_pdch, LIVE_FEATURES),
    }

    # --- F3: psychomotor items ---------------------------------------------
    f3: dict = {
        "family": "F3 (secondary) -- psychomotor item-level (Spearman)",
        "daic_note": (
            "PHQ8_Moving is read as-is from the official AVEC2017 train/dev label "
            "CSVs. No DAIC label is invented or re-derived. It is absent from "
            "full_test_split.csv, which is consistent with the pre-declared "
            "estimation set (train+dev only)."
        ),
        "pdch_note": (
            "HAMD-17 items 8 (retardation) and 9 (agitation) read directly from "
            "the shipped item columns. Totals are never re-derived from items "
            "(MC-07). Both items are empirically capped at 2 in this cohort "
            "despite a 0-4 range, so range restriction attenuates rho."
        ),
    }
    if est["PHQ8_Moving"].notna().any():
        f3["daic_phq8_moving"] = association_family(est, cols, "PHQ8_Moving", dperm,
                                                    binary=False, spearman=True)
    for item, key in ((f"hamd_item{HAMD_RETARDATION_ITEM}_retardation", "pdch_hamd_retardation"),
                      (f"hamd_item{HAMD_AGITATION_ITEM}_agitation", "pdch_hamd_agitation")):
        f3[key] = association_family(pdch, cols, item, pperm, binary=False, spearman=True)
        f3[key]["item_value_counts"] = {
            str(int(k)): int(v) for k, v in pdch[item].value_counts().sort_index().items()
        }
    if "daic_phq8_moving" in f3:
        f3["concordance_retardation"] = concordance(
            f3["daic_phq8_moving"], f3["pdch_hamd_retardation"], LIVE_FEATURES)

    # Items 8 and 9 are nominally opposite poles (retardation vs agitation). If
    # they correlate positively with each other and with the total, they are
    # behaving as severity indicators rather than as a direction contrast, and
    # any feature that tracks both is tracking severity, not retardation.
    i8 = pdch[f"hamd_item{HAMD_RETARDATION_ITEM}_retardation"].to_numpy(float)
    i9 = pdch[f"hamd_item{HAMD_AGITATION_ITEM}_agitation"].to_numpy(float)
    tot = pdch["HAMD17_total"].to_numpy(float)
    from scipy.stats import spearmanr
    f3["item_structure"] = {
        "note": (
            "HAMD-17 items 8 and 9 are nominally opposite poles (retardation vs "
            "agitation). These correlations say whether they behave that way in "
            "this cohort."
        ),
        "spearman_item8_item9": float(spearmanr(i8, i9).statistic),
        "spearman_item8_total": float(spearmanr(i8, tot).statistic),
        "spearman_item9_total": float(spearmanr(i9, tot).statistic),
    }

    # --- F4: robust and residualized variants (EXPLORATORY) ------------------
    def variant_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        v = pd.DataFrame(index=df.index)
        for c in F4_ROBUST_COLS:
            v[c] = df[c].to_numpy(float)
        z = np.column_stack([
            np.ones(len(df)),
            np.log(df["span_s"].to_numpy(float)),
            np.log1p(df["n_turn_pairs"].to_numpy(float)),
        ])
        for f in F4_RESID_FEATURES:
            col = df[MEAN_COL(f)].to_numpy(float)
            ok = np.isfinite(col)
            beta = np.linalg.lstsq(z[ok], col[ok], rcond=None)[0]
            v[f"{f}_resid"] = col - z @ beta
        return v, list(v.columns)

    vd, vcols = variant_frame(est)
    vp, _ = variant_frame(pdch)
    vd["PHQ8_Binary"] = est["PHQ8_Binary"].to_numpy()
    vp["HAMD17_ge17"] = pdch["HAMD17_ge17"].to_numpy()

    f4_daic = association_family(vd, vcols, "PHQ8_Binary", dperm, binary=True,
                                 spearman=False)
    f4_pdch = association_family(vp, vcols, "HAMD17_ge17", pperm, binary=True,
                                 spearman=False)
    f4_rows = []
    for c in vcols:
        a, b = f4_daic["per_feature"][c], f4_pdch["per_feature"][c]
        if a.get("degenerate") or b.get("degenerate"):
            f4_rows.append({"variant": c, "verdict": "degenerate"})
            continue
        f4_rows.append({
            "variant": c,
            "daic_r": a["r"], "daic_auc": a.get("auc"), "daic_p": a["p_perm"],
            "daic_q": a["q_bh"],
            "pdch_r": b["r"], "pdch_auc": b.get("auc"), "pdch_p": b["p_perm"],
            "pdch_q": b["q_bh"],
            "same_sign": bool(np.sign(a["r"]) == np.sign(b["r"])),
            "both_pass_fdr": bool(a["q_bh"] <= FDR_Q and b["q_bh"] <= FDR_Q),
        })
    f4 = {
        "family": "F4 (EXPLORATORY) -- robust and length-residualized variants",
        "label": "EXPLORATORY -- may not be reported as confirmation",
        "variants": {
            "column_selection_from_existing_functionals": list(F4_ROBUST_COLS),
            "newly_derived_exploratory": [f"{f}_resid" for f in F4_RESID_FEATURES],
            "residualization": "OLS on [1, log(span_s), log1p(n_turn_pairs)] within corpus",
        },
        "daic": f4_daic, "pdch": f4_pdch, "table": f4_rows,
    }

    # --- severity regression on DAIC, beside PDCH's known null ---------------
    from pdch_experiment import nested_cv_regression
    reg = {}
    for cfg in ("all24", "no_ask"):
        cols_cfg = [MEAN_COL(f) for f in FEATURE_CONFIGS[cfg]]
        reg[cfg] = nested_cv_regression(
            {SILENCE_THRESHOLD: daic[cols_cfg].to_numpy(float)},
            daic["PHQ8_Score"].to_numpy(float),
            daic["PHQ8_Binary"].to_numpy(int),
            daic["subject_id"].to_numpy(), n_repeats=n_repeats)
    f2["daic_severity_regression"] = {
        "note": (
            "DAIC-WOZ PHQ8_Score under the identical subject-grouped nested CV "
            "used for PDCH's HAMD-17 regression, so the two nulls are comparable. "
            "PDCH's numbers are in output/pdch_ctd_results.json."
        ),
        "results": reg,
    }
    return {"F2": f2, "F3": f3, "F4": f4}


# ---------------------------------------------------------------------------
# E3 -- parsimonious candidate detectors (conditional on E2)
# ---------------------------------------------------------------------------
def daic_deployed_protocol(daic: pd.DataFrame, feats: list[str]) -> dict:
    """Train fit, dev selects C on balanced accuracy, refit train+dev, test once."""
    cols = [MEAN_COL(f) for f in feats]
    tr = daic[daic["split"] == "train"]
    dv = daic[daic["split"] == "dev"]
    te = daic[daic["split"] == "test"]
    xtr, ytr = tr[cols].to_numpy(float), tr["PHQ8_Binary"].to_numpy(int)
    xdv, ydv = dv[cols].to_numpy(float), dv["PHQ8_Binary"].to_numpy(int)
    xte, yte = te[cols].to_numpy(float), te["PHQ8_Binary"].to_numpy(int)

    best_c, best = None, -np.inf
    dev_scores = {}
    for c in C_GRID:
        pipe = _fit(plain_pipeline(c), xtr, ytr)
        s = balanced_accuracy_score(ydv, pipe.predict(xdv))
        dev_scores[str(c)] = {
            "balanced_accuracy": float(s),
            "macro_f1": float(f1_score(ydv, pipe.predict(xdv), average="macro")),
        }
        if s > best:
            best, best_c = s, c

    refit = _fit(plain_pipeline(best_c),
                 np.vstack([xtr, xdv]), np.concatenate([ytr, ydv]))
    pred = refit.predict(xte)
    prob = refit.predict_proba(xte)[:, 1]
    return {
        "features": feats,
        "selected_C": best_c,
        "dev_grid": dev_scores,
        "dev": dev_scores[str(best_c)],
        "test": {
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
            "roc_auc": float(roc_auc_score(yte, prob)),
        },
    }


def e3_candidate_detectors(daic: pd.DataFrame, pdch: pd.DataFrame, e2: dict,
                           n_repeats: int = 20) -> dict:
    conc = e2["concordance"]
    confirmed = conc["promoted_confirmed"]
    weak = conc["promoted_weak_exploratory"]
    if not confirmed and not weak:
        return {
            "status": "skipped",
            "reason": (
                "F1 promoted no feature to either the confirmed or the "
                "weak-concordance tier, and the pre-declaration forbids "
                "broadening the search to recover a positive."
            ),
        }

    tier = "confirmed" if confirmed else "weak_concordance_EXPLORATORY"
    candidates = confirmed or weak
    runs: dict = {}
    sets: list[tuple[str, list[str]]] = [(f, [f]) for f in candidates]
    if len(candidates) > 1:
        sets.append(("k_feature_" + "+".join(candidates), list(candidates)))

    for name, feats in sets:
        cols = [MEAN_COL(f) for f in feats]
        runs[name] = {
            "daic_deployed_protocol": daic_deployed_protocol(daic, feats),
            "daic_nested_cv_pooled180": nested_cv_classification(
                {SILENCE_THRESHOLD: daic[cols].to_numpy(float)},
                daic["PHQ8_Binary"].to_numpy(int), daic["subject_id"].to_numpy(),
                n_repeats=n_repeats, fixed_threshold=SILENCE_THRESHOLD),
            "pdch_subject_grouped_cv": nested_cv_classification(
                {SILENCE_THRESHOLD: pdch[cols].to_numpy(float)},
                pdch["HAMD17_ge17"].to_numpy(int), pdch["subject_id"].to_numpy(),
                n_repeats=n_repeats, fixed_threshold=SILENCE_THRESHOLD),
        }

    def verdict(r: dict) -> str:
        pd_ok = r["pdch_subject_grouped_cv"]["macro_f1"]["mean"] >= 0.60 and \
            r["pdch_subject_grouped_cv"]["roc_auc"]["mean"] >= 0.60
        dc_ok = r["daic_deployed_protocol"]["test"]["macro_f1"] >= 0.60
        if pd_ok and dc_ok:
            return "PASS on both corpora"
        if dc_ok:
            return "FAIL -- DAIC-WOZ only"
        if pd_ok:
            return "FAIL -- PDCH only"
        return "FAIL on both corpora"

    return {
        "status": "run",
        "tier": tier,
        "candidates": candidates,
        "caveat": (
            "EXPLORATORY: promoted via the weak-concordance tier, which does not "
            "survive FDR. No result here may be described as transfer."
            if tier != "confirmed" else
            "Candidates passed BH-FDR in both corpora with matching sign."
        ),
        "criterion": ("PDCH mean macro-F1 >= 0.60 and mean AUC >= 0.60; "
                      "DAIC-WOZ test macro-F1 >= 0.60"),
        "runs": runs,
        "verdicts": {k: verdict(v) for k, v in runs.items()},
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _f(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def _ms(d: dict, key: str, nd=3) -> str:
    return f"{d[key]['mean']:.{nd}f} ± {d[key]['sd']:.{nd}f}"


def write_markdown(p: dict, path: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# Shared-signal audit — DAIC-WOZ × PDCH (`SLT-01`, `SLT-03`)")
    A("")
    A("> Pre-declared in [`docs/exploratory-shared-signal-pdch-daic.md`]"
      "(../docs/exploratory-shared-signal-pdch-daic.md), committed before any "
      "number here existed. That file also discloses, per quantity, what had "
      "already been seen — PDCH tier-B `|r|`/AUC were visible from "
      "`pdch_ctd_results.md`, so F1 is **re-analysis** for tier B and "
      "pre-declared for everything else. The DAIC-WOZ test split is withheld "
      "from every association estimate and is touched once, in E3.")
    A("")
    A(f"Generated by `src/ctd/shared_signal_audit.py`. Silence threshold pinned "
      f"at {SILENCE_THRESHOLD} s on both corpora; {N_PERM} subject-grouped label "
      f"permutations; BH-FDR at q = {FDR_Q} within family within corpus.")
    A("")

    # --- E0 ----------------------------------------------------------------
    e0 = p["E0_inventory"]
    A("## E0 — Inventory and distribution shift (no labels)")
    A("")
    A(e0["note"])
    A("")
    A("### Session level")
    A("")
    A("| Quantity | DAIC-WOZ median [IQR] | PDCH median [IQR] | PDCH / DAIC |")
    A("|---|---|---|---:|")
    for k, v in e0["session_level"].items():
        d, q = v["daic"], v["pdch"]
        ratio = q["median"] / d["median"] if abs(d["median"]) > 1e-9 else float("nan")
        A(f"| `{k}` | {_f(d['median'],2)} [{_f(d['q25'],2)}, {_f(d['q75'],2)}] | "
          f"{_f(q['median'],2)} [{_f(q['q25'],2)}, {_f(q['q75'],2)}] | {_f(ratio,2)} |")
    A("")
    A("### Per-feature shift, 22 live features")
    A("")
    A("| Feature | Grp | DAIC med [IQR] | PDCH med [IQR] | ratio | KS | KS p | "
      "PDCH distinct | flag |")
    A("|---|---|---|---|---:|---:|---:|---:|---|")
    for r in e0["per_feature_shift"]:
        flags = []
        if r["daic_near_degenerate"]:
            flags.append("DAIC~deg")
        if r["pdch_near_degenerate"]:
            flags.append("PDCH~deg")
        A(f"| `{r['feature']}` | {r['group']} | "
          f"{_f(r['daic_median'])} [{_f(r['daic_iqr'])}] | "
          f"{_f(r['pdch_median'])} [{_f(r['pdch_iqr'])}] | "
          f"{_f(r['median_ratio_pdch_over_daic'],2)} | {_f(r['ks_statistic'],2)} | "
          f"{r['ks_p']:.1e} | {r['pdch_n_distinct']} | {', '.join(flags) or '—'} |")
    A("")
    A(f"`{'`, `'.join(e0['dead_on_pdch'])}` are excluded throughout: they are "
      "exactly constant on PDCH under mono webrtcvad, so they are structurally "
      "unmeasurable there rather than weak.")
    A("")
    ar = e0["algebraic_redundancy"]
    A("### Effective dimensionality — 12 of the 22 features carry 2 numbers")
    A("")
    A(ar["note"])
    A("")
    A("| Corpus | Identity | Max abs deviation | Rank of the 22 live features |")
    A("|---|---|---:|---:|")
    for corpus in ("daic", "pdch"):
        for side in ("ask", "res"):
            r = ar[corpus][side]
            A(f"| {corpus.upper()} | `{r['identity']}` | "
              f"{r['max_abs_deviation']:.1e} | {r['matrix_rank_22_live_centered']} |")
    A("")

    # --- E2 (headline) ------------------------------------------------------
    e2 = p["E2_concordance"]
    c = e2["concordance"]
    A("## E2 — Univariate concordance map (family F1) — headline artifact")
    A("")
    A(f"DAIC-WOZ: {e2['estimation_sets']['daic']}, label `PHQ8_Binary`. "
      f"PDCH: {e2['estimation_sets']['pdch']}, label `HAMD17_ge17`. "
      "Labels are never pooled; the two columns are estimated separately and "
      "only their effects, ranks and signs are compared.")
    A("")
    A(f"Rule: {c['rule']}")
    A("")
    A("| Feature | Grp | DAIC r | DAIC AUC | DAIC q | PDCH r | PDCH AUC | PDCH q | "
      "Sign | Verdict |")
    A("|---|---|---:|---:|---:|---:|---:|---:|:--:|---|")
    order = {"confirmed_shared": 0, "opposite_sign": 1, "daic_only": 2,
             "pdch_only": 3, "null": 4, "degenerate": 5}
    for r in sorted(c["table"], key=lambda r: (order.get(r["verdict"], 9),
                                               -abs(r.get("daic_r") or 0))):
        if r["verdict"] == "degenerate":
            A(f"| `{r['feature']}` | {r['group']} | — | — | — | — | — | — | — | degenerate |")
            continue
        mark = "✔" if r["same_sign"] else "✘"
        star = " *" if r["weak_concordant_exploratory"] else ""
        A(f"| `{r['feature']}` | {r['group']} | {_f(r['daic_r'])} | {_f(r['daic_auc'])} | "
          f"{_f(r['daic_q'])} | {_f(r['pdch_r'])} | {_f(r['pdch_auc'])} | "
          f"{_f(r['pdch_q'])} | {mark} | {r['verdict']}{star} |")
    A("")
    A("`*` = weak-concordance tier (EXPLORATORY; does not survive FDR).")
    A("")
    sr = c["descriptive_sign_reversals"]
    if sr["features"]:
        A("#### Sign reversals (descriptive)")
        A("")
        A(sr["note"])
        A("")
        A("| Feature | Grp | DAIC r (q) | PDCH r (q) |")
        A("|---|---|---:|---:|")
        for r in sr["features"]:
            A(f"| `{r['feature']}` | {r['group']} | {_f(r['daic_r'])} "
              f"({_f(r['daic_q'],2)}) | {_f(r['pdch_r'])} ({_f(r['pdch_q'],2)}) |")
        A("")
    A("| Verdict | n |")
    A("|---|---:|")
    for k, v in c["counts"].items():
        A(f"| {k} | {v} |")
    A("")
    sa = c["sign_agreement"]
    A(f"Sign agreement: **{sa['n_agree']}/{sa['n_comparable']}** features share a "
      f"sign across corpora (binomial p = {_f(sa['binomial_p_vs_half'])}). "
      f"{sa['caveat']}")
    A("")
    for corpus in ("daic", "pdch"):
        cell = e2[corpus]
        A(f"Max |r| calibration, {corpus.upper()}: observed "
          f"{_f(cell['max_abs_r_observed'])}, permutation p = "
          f"{_f(cell['max_abs_r_permutation_p'])}, null 95th percentile "
          f"{_f(cell['max_abs_r_null_q95'])} "
          f"(n = {cell['n']}, {cell['n_features_tested']} features tested).")
    A("")

    # --- E1 ----------------------------------------------------------------
    e1 = p["E1_confounds"]
    A("## E1 — Confound checks")
    A("")
    a = e1["e1a_matched_subset"]
    A("### E1a — PDCH matched on session length and turn count")
    A("")
    A(f"1:1 nearest-neighbour matching on {a['matching']['confounds']} gives "
      f"{a['matching']['n_pairs']} pairs ({a['n_sessions']} sessions, "
      f"{a['n_subjects']} subjects, {a['prevalence']:.1%} positive). Standardized "
      f"mean difference before "
      f"{[round(v,3) for v in a['matching']['standardized_mean_diff_before']]} → after "
      f"{[round(v,3) for v in a['matching']['standardized_mean_diff_after']]}.")
    A("")
    A("| Config | Macro-F1 | Balanced acc. | ROC AUC | Softens? |")
    A("|---|---|---|---|---|")
    for cfg, r in a["results"].items():
        A(f"| `{cfg}` | {_ms(r,'macro_f1')} | {_ms(r,'balanced_accuracy')} | "
          f"{_ms(r,'roc_auc')} | {'yes' if a['softens'][cfg] else 'no'} |")
    A("")
    A(f"Criterion fixed in advance: {a['softens_criterion']}.")
    A("")
    b = e1["e1b_residualized"]
    A("### E1b — Absolute-time features residualized on session length, within folds")
    A("")
    A(b["note"])
    A("")
    A("| Corpus | Config | Residualized macro-F1 | Same folds, unresidualized | "
      "Residualized AUC | Unresid. AUC |")
    A("|---|---|---|---|---|---|")
    for corpus, cfgs in b["results"].items():
        for cfg, r in cfgs.items():
            A(f"| {corpus.upper()} | `{cfg}` | {_ms(r['residualized'],'macro_f1')} | "
              f"{_ms(r['unresidualized_same_folds'],'macro_f1')} | "
              f"{_ms(r['residualized'],'roc_auc')} | "
              f"{_ms(r['unresidualized_same_folds'],'roc_auc')} |")
    A("")
    pc = e1["e1c_power_calibration"]
    A("### E1c — Power calibration: DAIC-WOZ at PDCH's cohort size")
    A("")
    A(pc["design"])
    A("")
    A("| Metric | Mean ± sd | 5th | 50th | 95th |")
    A("|---|---|---:|---:|---:|")
    q = pc["macro_f1_percentiles"]
    A(f"| macro-F1 | {_ms(pc,'macro_f1')} | {_f(q['5'])} | {_f(q['50'])} | {_f(q['95'])} |")
    A(f"| balanced acc. | {_ms(pc,'balanced_accuracy')} | | | |")
    A(f"| ROC AUC | {_ms(pc,'roc_auc')} | | | |")
    A("")
    A("PDCH's observed `all24` result for comparison: macro-F1 0.520 ± 0.055, "
      "AUC 0.476 ± 0.066 (`output/pdch_ctd_results.md`).")
    A("")
    A("**Where PDCH sits inside this distribution** — the question the null "
      "actually turns on:")
    A("")
    A("| Metric | PDCH observed | Percentile within DAIC-at-n=62 | z |")
    A("|---|---:|---:|---:|")
    for k, v in pc["pdch_located_in_daic_null"].items():
        A(f"| {k} | {_f(v['pdch_observed'])} | "
          f"{v['percentile_within_daic_at_n62']:.0f}th | "
          f"{_f(v['z_vs_daic_at_n62'],2)} |")
    A("")

    # --- E3 ----------------------------------------------------------------
    e3 = p["E3_candidate_detectors"]
    A("## E3 — Parsimonious candidate detectors")
    A("")
    if e3["status"] == "skipped":
        A(f"**Skipped.** {e3['reason']}")
    else:
        A(f"Tier: **{e3['tier']}**. Candidates: "
          f"`{'`, `'.join(e3['candidates'])}`. {e3['caveat']}")
        A("")
        A(f"Criterion: {e3['criterion']}.")
        A("")
        A("| Candidate | DAIC dev mF1 | DAIC **test** mF1 | DAIC test AUC | "
          "DAIC pooled-CV mF1 | PDCH CV mF1 | PDCH CV AUC | Verdict |")
        A("|---|---|---|---|---|---|---|---|")
        for name, r in e3["runs"].items():
            d = r["daic_deployed_protocol"]
            A(f"| `{name}` | {_f(d['dev']['macro_f1'])} | {_f(d['test']['macro_f1'])} | "
              f"{_f(d['test']['roc_auc'])} | "
              f"{_ms(r['daic_nested_cv_pooled180'],'macro_f1')} | "
              f"{_ms(r['pdch_subject_grouped_cv'],'macro_f1')} | "
              f"{_ms(r['pdch_subject_grouped_cv'],'roc_auc')} | "
              f"{e3['verdicts'][name]} |")
    A("")

    # --- E4 ----------------------------------------------------------------
    e4 = p["E4_secondary_targets"]
    A("## E4 — Secondary targets")
    A("")
    f2 = e4["F2"]
    A("### F2 — Continuous severity (Spearman)")
    A("")
    A("DAIC-WOZ `PHQ8_Score` (train+dev) vs. PDCH `HAMD17_total`. Separate "
      "instruments, separate columns, separate FDR families.")
    A("")
    A("| Feature | Grp | DAIC ρ | DAIC q | PDCH ρ | PDCH q | Sign | Verdict |")
    A("|---|---|---:|---:|---:|---:|:--:|---|")
    for r in sorted(f2["concordance"]["table"],
                    key=lambda r: (order.get(r["verdict"], 9), -abs(r.get("daic_r") or 0))):
        if r["verdict"] == "degenerate":
            continue
        A(f"| `{r['feature']}` | {r['group']} | {_f(r['daic_r'])} | {_f(r['daic_q'])} | "
          f"{_f(r['pdch_r'])} | {_f(r['pdch_q'])} | "
          f"{'✔' if r['same_sign'] else '✘'} | {r['verdict']} |")
    A("")
    dsr = f2["daic_severity_regression"]
    A(dsr["note"])
    A("")
    A("| Corpus | Config | MAE | Pearson r | R² | Predict-the-mean MAE |")
    A("|---|---|---|---|---|---|")
    for cfg, r in dsr["results"].items():
        A(f"| DAIC-WOZ | `{cfg}` | {_ms(r,'mae',2)} | {_ms(r,'pearson_r')} | "
          f"{_ms(r,'r2')} | {_f(r['baseline_mae_predict_train_mean'],2)} |")
    A("| PDCH | `all24` | 6.65 ± 0.33 | 0.083 ± 0.120 | −0.072 ± 0.124 | 6.55 |")
    A("")
    f3 = e4["F3"]
    A("### F3 — Psychomotor item-level (Spearman)")
    A("")
    A(f3["daic_note"])
    A("")
    A(f3["pdch_note"])
    A("")
    cols_f3 = [("daic_phq8_moving", "DAIC PHQ8_Moving"),
               ("pdch_hamd_retardation", "PDCH HAMD item 8"),
               ("pdch_hamd_agitation", "PDCH HAMD item 9")]
    present = [(k, lab) for k, lab in cols_f3 if k in f3]
    A("| Feature | Grp | " + " | ".join(f"{lab} ρ (q)" for _, lab in present) + " |")
    A("|---|---|" + "---:|" * len(present))
    for f in LIVE_FEATURES:
        cells = []
        for k, _ in present:
            rec = f3[k]["per_feature"][MEAN_COL(f)]
            cells.append("—" if rec.get("degenerate")
                         else f"{_f(rec['r'])} ({_f(rec['q_bh'],2)})")
        A(f"| `{f}` | {group_of(f)} | " + " | ".join(cells) + " |")
    A("")
    if "item_structure" in f3:
        it = f3["item_structure"]
        A(it["note"])
        A("")
        A(f"Spearman ρ: item 8 × item 9 = **{_f(it['spearman_item8_item9'])}**, "
          f"item 8 × total = {_f(it['spearman_item8_total'])}, "
          f"item 9 × total = {_f(it['spearman_item9_total'])}.")
        A("")
    f4 = e4["F4"]
    A("### F4 — Robust and length-residualized variants — **EXPLORATORY**")
    A("")
    A(f"{f4['label']}. Column selection from the existing functional bank: "
      f"`{'`, `'.join(f4['variants']['column_selection_from_existing_functionals'])}`. "
      f"Newly derived and named as exploratory: "
      f"`{'`, `'.join(f4['variants']['newly_derived_exploratory'])}` "
      f"({f4['variants']['residualization']}).")
    A("")
    A("| Variant | DAIC r | DAIC AUC | DAIC q | PDCH r | PDCH AUC | PDCH q | Sign | "
      "Both pass FDR |")
    A("|---|---:|---:|---:|---:|---:|---:|:--:|:--:|")
    for r in f4["table"]:
        if r.get("verdict") == "degenerate":
            A(f"| `{r['variant']}` | — | — | — | — | — | — | — | — |")
            continue
        A(f"| `{r['variant']}` | {_f(r['daic_r'])} | {_f(r['daic_auc'])} | "
          f"{_f(r['daic_q'])} | {_f(r['pdch_r'])} | {_f(r['pdch_auc'])} | "
          f"{_f(r['pdch_q'])} | {'✔' if r['same_sign'] else '✘'} | "
          f"{'yes' if r['both_pass_fdr'] else 'no'} |")
    A("")
    A("---")
    A("")
    A("Machine-readable results: `output/shared_signal_audit.json`. "
      "Decision memo: [`docs/shared-signal-audit.md`](../docs/shared-signal-audit.md).")
    path.write_text("\n".join(L) + "\n")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading DAIC-WOZ session table...")
    daic = load_daic()
    print(f"  {len(daic)} sessions "
          f"({(daic['split'] == 'train').sum()}/{(daic['split'] == 'dev').sum()}/"
          f"{(daic['split'] == 'test').sum()})")
    print("Loading PDCH session table (VAD cache -> features at 0.2 s)...")
    pdch_all, pdch = load_pdch()
    print(f"  {len(pdch_all)} sessions with features; {len(pdch)} labelled "
          f"across {pdch['subject_id'].nunique()} subjects")

    print("E0: inventory and distribution shift...")
    e0 = e0_inventory(daic, pdch_all)
    print("E2: univariate concordance map (F1)...")
    e2 = e2_concordance(daic, pdch)
    cc = e2["concordance"]["counts"]
    print(f"  verdicts: {cc}")
    print("E1: confound checks (matching, residualization, power calibration)...")
    e1 = e1_confounds(daic, pdch)
    print("E4: secondary targets (F2, F3, F4)...")
    e4 = e4_secondary_targets(daic, pdch)
    print("E3: candidate detectors...")
    e3 = e3_candidate_detectors(daic, pdch, e2)
    print(f"  {e3['status']}")

    payload = {
        "title": "Shared-signal audit across DAIC-WOZ and PDCH",
        "action_items": ["SLT-01", "SLT-03", "MC-02"],
        "predeclaration": "docs/exploratory-shared-signal-pdch-daic.md",
        "settings": {
            "live_features": LIVE_FEATURES,
            "excluded_structurally_unmeasurable_on_pdch": list(DEAD_ON_PDCH),
            "silence_threshold_s": SILENCE_THRESHOLD,
            "n_permutations": N_PERM,
            "permutation_seed": PERM_SEED,
            "fdr_q": FDR_Q,
            "weak_concordance_abs_r_floor": WEAK_ABS_R,
            "no_label_pooling": (
                "PHQ8_* and HAMD17_* never share a column; every supervised fit "
                "is within one corpus."
            ),
        },
        "E0_inventory": e0,
        "E1_confounds": e1,
        "E2_concordance": e2,
        "E3_candidate_detectors": e3,
        "E4_secondary_targets": e4,
    }
    out = REPORT_DIR / "shared_signal_audit.json"
    out.write_text(json.dumps(_jsonable(payload), indent=2))
    write_markdown(payload, REPORT_DIR / "shared_signal_audit.md")
    print(f"Saved -> {out}")
    print(f"Saved -> {REPORT_DIR / 'shared_signal_audit.md'}")


if __name__ == "__main__":
    main()
