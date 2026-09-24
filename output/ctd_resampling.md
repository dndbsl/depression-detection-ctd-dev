# SLT-10: CTD variability via data resampling

100x repeated stratified-group resampling of train+dev (StratifiedGroupKFold, n_splits=4, group=session_id); refit on resampled train, select C on resampled held-out portion (dev balanced accuracy); refit on full train+dev with that C, evaluate once on the untouched real test set.

> Not reported: the CTD pipeline (SimpleImputer(median) -> StandardScaler -> LogisticRegression(lbfgs)) is a deterministic convex fit, so seed variance is exactly 0.000 for a fixed data split. Data resampling above answers the variability question rwao intended by analogy with the neural baselines' seed sweep.


## Held-out metrics over 100 repeats (mean ± sd)

| Metric | Mean | SD |
|---|---:|---:|
| balanced_accuracy | 0.648 | 0.074 |
| macro_f1 | 0.630 | 0.069 |
| roc_auc | 0.679 | 0.073 |

## Hyperparameter (C) stability

| C | Count |
|---:|---:|
| 0.01 | 20 |
| 0.03 | 9 |
| 0.1 | 26 |
| 0.3 | 19 |
| 1.0 | 26 |

Tied modal C values = **0.1, 1.0**, selected in 26% of repeats each.


> Deployed model uses C=0.3, selected on the single official dev split. If a different C is modal here, C=0.3 should be reported as one plausible choice among several rather than uniquely optimal.


## Head-to-head vs. RoBERTa on test

CTD test macro-F1 across repeats: 0.611 ± 0.031 (RoBERTa: 0.631).

CTD beats RoBERTa's test macro-F1 in **0%** of repeats.
