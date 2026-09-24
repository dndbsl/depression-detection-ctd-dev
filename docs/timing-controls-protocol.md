# Timing controls: analysis specification, 2026-09-24

This is a new exploratory follow-up after inspection of the existing DAIC and
PDCH results. It is not an independent confirmation or a preregistration of
the original study. The configuration below is recorded before computing
these new comparisons; all configurations will be reported.

Use only the cleaned DAIC train+dev cohort (135 subjects). Do not load or score
the official test cohort. Compare all24, no_ask, latency alone, interview
structure alone (log session span and log1p turn-pair count), and latency plus
interview structure. The controls test whether the full descriptor set adds
predictive value beyond simple timing and interview-length summaries.

Use the existing subject-grouped nested-CV implementation: five outer folds,
four inner folds, ten repeats (outer seeds 0–9), identical folds across all
configurations. Fit median imputation and standardization only on training
folds. Select class-balanced L2 logistic-regression C from
0.01/0.03/0.1/0.3/1.0 by mean inner balanced accuracy. Fix decision threshold
at 0.5. Pool out-of-fold predictions within each repeat. Report macro-F1 and
AUC mean and SD across repeats, not significance tests treating folds as
independent observations. No configuration will replace the deployed model.

Reuse transcript cleaning and timing definitions. Session span is last
utterance stop minus first utterance start, matching the prior shared-signal
audit. Record source hashes and package versions with aggregate results.
This experiment does not answer whether CTD adds value to text; that still
requires the missing raw frozen-encoder caches and nested head/fusion fitting.
