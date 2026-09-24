"""Unit tests for the statistics added by `src/ctd/shared_signal_audit.py`.

The audit's conclusions rest on three pieces of machinery that are easy to get
subtly wrong and that no existing test covers: Benjamini-Hochberg adjustment,
the subject-blocked label permuter (which is what makes PDCH's p-values valid
when 16 subjects contribute two sessions), and the within-fold residualizer
(which must not let test-fold rows influence the fitted coefficients).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "ctd"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared_signal_audit import (  # noqa: E402
    LIVE_FEATURES,
    ResidualizeOnTrailingConfounds,
    SubjectPermuter,
    bh_fdr,
    corr_vec,
)


def test_live_features_exclude_only_the_unmeasurable_pair():
    assert len(LIVE_FEATURES) == 22
    assert "ask_bt" not in LIVE_FEATURES and "res_bt" not in LIVE_FEATURES


def test_bh_fdr_matches_the_textbook_definition():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = bh_fdr(p)
    # q_i = min over j>=i of (p_j * n / j)
    expected = [min(p[j] * 5 / (j + 1) for j in range(i, 5)) for i in range(5)]
    assert np.allclose(q, expected)
    assert np.all(np.diff(q) >= -1e-12), "q-values must be monotone in p"


def test_bh_fdr_is_never_smaller_than_the_raw_p_and_is_capped_at_one():
    rng = np.random.RandomState(0)
    p = rng.uniform(size=50)
    q = bh_fdr(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1.0)


def test_bh_fdr_is_order_invariant():
    p = np.array([0.3, 0.001, 0.2, 0.04])
    order = np.array([2, 0, 3, 1])
    assert np.allclose(bh_fdr(p)[order], bh_fdr(p[order]))


def test_corr_vec_agrees_with_numpy_corrcoef():
    rng = np.random.RandomState(1)
    x = rng.normal(size=(40, 5))
    y = rng.normal(size=40)
    got = corr_vec(x, y)
    want = [np.corrcoef(x[:, j], y)[0, 1] for j in range(5)]
    assert np.allclose(got, want)


def test_permuter_moves_each_subject_block_as_a_unit():
    subjects = np.array(["a", "a", "b", "b", "c", "d"])
    y = np.array([1, 0, 1, 1, 0, 1])
    perm = SubjectPermuter(subjects, seed=7)
    for _ in range(200):
        yp = y[perm.take()]
        # Paired subjects can only receive another paired subject's vector.
        assert sorted([tuple(yp[0:2]), tuple(yp[2:4])]) == sorted([(1, 0), (1, 1)])
        # Singletons only swap among themselves.
        assert sorted(yp[4:6].tolist()) == [0, 1]
        assert sorted(yp.tolist()) == sorted(y.tolist())


def test_permuter_describes_the_block_structure():
    perm = SubjectPermuter(np.array(["a", "a", "b", "c"]))
    assert perm.describe() == {"n_rows": 4, "block_sizes": {"1": 2, "2": 1}}


def test_permuter_actually_permutes():
    subjects = np.array([str(i) for i in range(20)])
    perm = SubjectPermuter(subjects, seed=3)
    takes = {tuple(perm.take()) for _ in range(20)}
    assert len(takes) > 1


def test_residualizer_removes_the_confound_and_drops_its_columns():
    rng = np.random.RandomState(2)
    n = 200
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    f0 = 3.0 * z1 - 2.0 * z2 + rng.normal(scale=0.01, size=n)   # confounded
    f1 = rng.normal(size=n)                                      # left alone
    x = np.column_stack([f0, f1, z1, z2])

    t = ResidualizeOnTrailingConfounds(n_confounds=2, target_idx=(0,)).fit(x)
    out = t.transform(x)

    assert out.shape == (n, 2), "confound columns must be dropped"
    assert abs(np.corrcoef(out[:, 0], z1)[0, 1]) < 0.05
    assert abs(np.corrcoef(out[:, 0], z2)[0, 1]) < 0.05
    assert np.allclose(out[:, 1], f1), "untargeted columns pass through unchanged"


def test_residualizer_uses_only_fit_rows_for_its_coefficients():
    """The point of doing this inside a fold: test rows must not move the fit."""
    rng = np.random.RandomState(4)
    n = 120
    z = rng.normal(size=(n, 2))
    f = z @ np.array([1.5, -0.5]) + rng.normal(scale=0.1, size=n)
    x = np.column_stack([f, z])

    train, test = np.arange(60), np.arange(60, n)
    t = ResidualizeOnTrailingConfounds(2, (0,)).fit(x[train])
    beta = t.betas_[0].copy()

    perturbed = x.copy()
    perturbed[test, 0] += 100.0
    beta2 = ResidualizeOnTrailingConfounds(2, (0,)).fit(perturbed[train]).betas_[0]
    assert np.allclose(beta, beta2)

    # ...and transforming the test rows uses those train-fitted coefficients.
    d = np.column_stack([np.ones(len(test)), x[test, 1:]])
    assert np.allclose(t.transform(x[test])[:, 0], x[test, 0] - d @ beta)


@pytest.mark.parametrize("n_confounds", [1, 2, 3])
def test_residualizer_handles_different_confound_counts(n_confounds):
    rng = np.random.RandomState(5)
    x = rng.normal(size=(50, 4 + n_confounds))
    out = ResidualizeOnTrailingConfounds(n_confounds, (0, 1)).fit(x).transform(x)
    assert out.shape == (50, 4)
