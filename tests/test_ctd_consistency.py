"""SLT-04/05 prerequisite (plan-preflight.md Step 0a): the CTD 24-D detector
is implemented twice -- `ml_splits.py::run()` (via `functionals.py`'s
`__amean` columns) and `fusion/fusion_late.py::ctd_probs()` (via
`np.nanmean` over `bags.py`, with train-median imputation applied in
`fusion/extract_ctd_roberta.py`). Both are NaN-aware means over the same
turn-pairing/feature-extraction pipeline and should agree exactly, but
nothing asserted that before ablation work started. This does.

Skipped when DAIC_WOZ_ROOT is unset (needs the real transcripts).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTD_ROOT = PROJECT_ROOT / "src" / "ctd"
for p in (str(CTD_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = pytest.mark.skipif(
    "DAIC_WOZ_ROOT" not in os.environ, reason="requires DAIC-WOZ dataset on disk"
)


@pytest.fixture(scope="module")
def splits_data():
    from ml_splits import build_split_features  # noqa: E402
    return {s: build_split_features(s) for s in ("train", "dev", "test")}


@pytest.fixture(scope="module")
def bags_by_split():
    from bags import build_split_bags  # noqa: E402
    return {s: build_split_bags(s) for s in ("train", "dev", "test")}


def test_session_means_agree(splits_data, bags_by_split):
    from constants import CTD_FEATURE_NAMES  # noqa: E402
    mean_cols = [f"{f}__amean" for f in CTD_FEATURE_NAMES]

    for split in ("train", "dev", "test"):
        df = splits_data[split].set_index("session_id")
        bags = {b.session_id: b for b in bags_by_split[split]}
        assert set(df.index) == set(bags.keys()), f"{split}: session sets differ"

        for sid, bag in bags.items():
            functionals_mean = df.loc[sid, mean_cols].to_numpy(dtype=float)
            bags_mean = np.nanmean(bag.turns, axis=0)
            np.testing.assert_allclose(
                functionals_mean, bags_mean, rtol=1e-9, atol=1e-9,
                err_msg=f"{split} session {sid}: functionals.__amean != bags.nanmean",
            )


def test_deployed_probabilities_agree(splits_data):
    """No missing __amean cells in this cohort (verified separately), so
    ml_splits' SimpleImputer(median) and extract_ctd_roberta's train-median
    imputation are both no-ops here -- the two pipelines should therefore
    give identical fitted probabilities, not just identical inputs."""
    from constants import CTD_FEATURE_NAMES  # noqa: E402
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mean_cols = [f"{f}__amean" for f in CTD_FEATURE_NAMES]
    xtr = splits_data["train"][mean_cols].to_numpy(float)
    ytr = splits_data["train"]["PHQ8_Binary"].to_numpy(int)
    xdv = splits_data["dev"][mean_cols].to_numpy(float)

    assert not np.isnan(xtr).any(), "test assumes no missing cells; imputation paths are otherwise not comparable"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ml_splits.py path: SimpleImputer(median) -> StandardScaler -> LogReg, in one Pipeline.
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000, random_state=42)),
        ]).fit(xtr, ytr)
        dev_probs_ml_splits = pipe.predict_proba(xdv)[:, 1]

        # fusion_late.py / extract_ctd_roberta.py path: manual train-median
        # impute, then a separately-fit StandardScaler + LogReg (no Pipeline).
        train_median = np.nanmedian(xtr, axis=0)
        xtr_imputed = np.where(np.isnan(xtr), train_median, xtr)
        xdv_imputed = np.where(np.isnan(xdv), train_median, xdv)
        sc = StandardScaler().fit(xtr_imputed)
        clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000, random_state=42)
        clf.fit(sc.transform(xtr_imputed), ytr)
        dev_probs_fusion = clf.predict_proba(sc.transform(xdv_imputed))[:, 1]

    np.testing.assert_allclose(dev_probs_ml_splits, dev_probs_fusion, rtol=1e-9, atol=1e-9)
