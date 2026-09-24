# SLT-05: ask-side ablation

Identical deployed protocol (LogReg L2, class_weight='balanced'): fit train -> select C on dev (balanced accuracy) -> refit train+dev -> test once.

| Config | n features | Selected C | Dev bAcc | Dev macro-F1 | Test bAcc [95% CI] | Test macro-F1 [95% CI] |
|---|---:|---:|---:|---:|---|---|
| `all24` | 24 | 0.3 | 0.756 | 0.746 | 0.641 [0.482, 0.791] | 0.631 [0.472, 0.771] |
| `no_ask` | 15 | 1.0 | 0.839 | 0.814 | 0.660 [0.500, 0.807] | 0.641 [0.482, 0.779] |
| `res_only` | 8 | 1.0 | 0.607 | 0.607 | 0.517 [0.367, 0.674] | 0.517 [0.364, 0.661] |
| `ask_only` | 9 | 0.01 | 0.595 | 0.594 | 0.608 [0.453, 0.757] | 0.593 [0.444, 0.729] |
| `no_cross` | 17 | 0.1 | 0.696 | 0.700 | 0.540 [0.381, 0.708] | 0.527 [0.380, 0.669] |

## Paired bootstrap: all24 - no_ask (macro-F1)

| Split | Delta | 95% CI | Excludes zero? |
|---|---:|---|---|
| dev | -0.068 | [-0.216, +0.083] | False |
| test | -0.010 | [-0.089, +0.054] | False |

**Verdict:** No clear difference detected: the 'all24' vs 'no_ask' macro-F1 paired bootstrap 95% CI includes zero on dev and test. This does not establish equivalence or remove interviewer confounding.


> 5 configurations each evaluated once on test = 5 looks at the test set, uncorrected. Dev is primary for all ablation comparisons; test is reported for reference only. No hyperparameter was re-tuned on test.


> **Note on rebuttal discrepancy:** These numbers differ from the SLT rebuttal's no_ask figures (dev .752 / test .661): the rebuttal reused the full model's C=0.3 for the ablation configs instead of reselecting C on dev per config. This script reselects C independently for every config, per plan-preflight.md Step 4 ('identical deployed protocol' includes the selection step, not just the fitted coefficients). Reselecting gives no_ask C=1.0, dev macro-F1 0.814, test macro-F1 0.641 -- the test delta versus all24 shrinks from the rebuttal's claimed +.030 to +.010. The paired interval does not establish equivalence.
