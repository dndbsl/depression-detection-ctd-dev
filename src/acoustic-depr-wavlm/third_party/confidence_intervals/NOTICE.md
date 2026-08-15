# Vendored: ConfidenceIntervals

This directory is a minimal, self-contained vendored copy of the
**ConfidenceIntervals** library by Luciana Ferrer and Pablo Riera.

- Upstream repository: https://github.com/luferrer/ConfidenceIntervals
- Vendored version: 0.1.0 (see `VERSION`)
- License: MIT (see `LICENSE`)

## What was copied

Only the runtime package is vendored so that `OL-MIL-2/` needs no network access
and no `pip install` at run time:

- `confidence_intervals.py` — the bootstrap CI implementation (`evaluate_with_conf_int`, `Bootstrap`, `get_conf_int`, `get_bootstrap_indices`).
- `utils.py` — helper plotting / toy-data utilities (`barplot_with_ci`, `create_data`) used only for sanity-checking.
- `__init__.py`, `LICENSE`, `VERSION`.

The upstream test suite, notebook, and example images were **not** copied.

## How it is used in this project

Every confidence interval reported in `outputs/results/REPORT.md` is produced by
`evaluate_with_conf_int(...)` from this vendored copy — both (a) the CI around each
condition's metric and (b) the CI around the paired Δ (OL+ − OL−). No custom
bootstrap or t-test implementation is used to compute those CIs. The paired t-test
p-value and Cohen's d reported alongside the CIs are standard scipy/numpy
computations, as permitted by the project spec.

## How to cite

> Ferrer, L. and Riera, P. Confidence Intervals for evaluation in machine learning
> [Computer software]. https://github.com/luferrer/ConfidenceIntervals
