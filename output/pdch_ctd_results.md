# PDCH: within-corpus CTD results (`MC-01`, `MC-05`, `MC-06`, `MC-07`, `MC-08`)

> **Headline: no reliable within-PDCH discrimination.** All five pre-registered tiers give macro-F1 0.468–0.522 and ROC AUC 0.445–0.497 under subject-grouped CV. These are models trained within PDCH, not a detector-transfer test. Language, instrument, population, interviewer and timing measurement differ from DAIC-WOZ. Low power is not established as the explanation. Predictions remain frozen in `docs/preregistration-pdch.md`; the current interpretation follows `docs/revision-audit-20260924.md`.


Subject-grouped nested CV: StratifiedGroupKFold 5-fold x 20 repeats on subject_id, inner 4-fold selection of C and the intra-turn silence threshold on balanced accuracy. Deployed pipeline unchanged (median impute -> standardize -> L2 LogReg, class_weight='balanced'). Out-of-fold predictions pooled within a repeat; mean +- sd across repeats. No fixed held-out test split at n=62.


## Cohort

| Stage | n |
|---|---:|
| Session directories on disk | 100 |
| Survived stitching → VAD → turn pairing | 100 |
| …of those, with a HAMD-17 total | 62 |
| Distinct subjects used | 46 |

Primary label `HAMD17_total >= 17`: 27/62 positive (43.5%). For reference, the standard cutoff of 8 would mark 85% of this inpatient cohort positive, which is why it is not used.


## Feature tiers

| Config | n features | non-degenerate | Macro-F1 | Balanced acc. | ROC AUC |
|---|---:|---:|---|---|---|
| `all24` | 24 | 22 | 0.520 ± 0.055 | 0.522 ± 0.054 | 0.476 ± 0.066 |
| `no_ask` | 15 | 14 | 0.522 ± 0.054 | 0.524 ± 0.053 | 0.489 ± 0.058 |
| `res_only` | 8 | 8 | 0.468 ± 0.054 | 0.472 ± 0.053 | 0.445 ± 0.072 |
| `ask_only` | 9 | 8 | 0.521 ± 0.041 | 0.522 ± 0.041 | 0.497 ± 0.030 |
| `no_cross` | 17 | 16 | 0.522 ± 0.062 | 0.524 ± 0.060 | 0.488 ± 0.079 |

Always-negative baseline: macro-F1 0.361, balanced accuracy 0.500. Per-repeat spreads are in `pdch_ctd_results.json`.


## Selected hyperparameters (distribution over folds × repeats)

| Config | C counts | Silence threshold counts |
|---|---|---|
| `all24` | {'0.01': 25, '0.03': 13, '0.1': 21, '0.3': 24, '1.0': 17} | {'0.05': 20, '0.1': 7, '0.2': 7, '0.3': 15, '0.5': 37, '1.0': 14} |
| `no_ask` | {'0.01': 21, '0.03': 14, '0.1': 13, '0.3': 23, '1.0': 29} | {'0.05': 24, '0.1': 9, '0.2': 7, '0.3': 11, '0.5': 33, '1.0': 16} |
| `res_only` | {'0.01': 38, '0.03': 16, '0.1': 12, '0.3': 11, '1.0': 23} | {'0.05': 21, '0.1': 5, '0.2': 8, '0.3': 6, '0.5': 33, '1.0': 27} |
| `ask_only` | {'0.01': 19, '0.03': 14, '0.1': 17, '0.3': 23, '1.0': 27} | {'0.05': 42, '0.1': 13, '0.2': 8, '0.3': 5, '0.5': 18, '1.0': 14} |
| `no_cross` | {'0.01': 22, '0.03': 8, '0.1': 15, '0.3': 21, '1.0': 34} | {'0.05': 14, '0.1': 5, '0.2': 11, '0.3': 17, '0.5': 46, '1.0': 7} |

`C` is reported as a distribution, not a winner: it was unstable under resampling on DAIC-WOZ too (`SLT-10`).


## Silence-threshold sensitivity (`all24`, threshold pinned)

| Threshold (s) | Macro-F1 | Balanced acc. |
|---:|---|---|
| 0.05 | 0.520 ± 0.057 | 0.522 ± 0.056 |
| 0.1 | 0.521 ± 0.053 | 0.523 ± 0.052 |
| 0.2 | 0.519 ± 0.060 | 0.520 ± 0.058 |
| 0.3 | 0.521 ± 0.056 | 0.523 ± 0.055 |
| 0.5 | 0.513 ± 0.050 | 0.515 ± 0.049 |
| 1.0 | 0.509 ± 0.051 | 0.511 ± 0.050 |

## Severity regression on the HAMD-17 total (`MC-07` secondary)

| Config | MAE | RMSE | Pearson r | R² |
|---|---|---|---|---|
| `all24` | 6.65 ± 0.33 | 7.98 ± 0.45 | 0.083 ± 0.120 | -0.072 ± 0.124 |
| `no_ask` | 6.62 ± 0.30 | 7.92 ± 0.35 | 0.084 ± 0.120 | -0.053 ± 0.098 |
| `res_only` | 6.62 ± 0.18 | 7.77 ± 0.24 | 0.100 ± 0.129 | -0.013 ± 0.063 |
| `ask_only` | 6.68 ± 0.20 | 7.90 ± 0.39 | 0.104 ± 0.128 | -0.049 ± 0.111 |
| `no_cross` | 6.62 ± 0.25 | 7.79 ± 0.27 | 0.153 ± 0.116 | -0.018 ± 0.071 |

Predict-the-mean MAE baseline: 6.55.


## Feature degeneracy — what VAD refinement bought (`MC-06`)

Session-level (`__amean`) spread over all 100 sessions, VAD-refined utterances versus the raw 1-second turn labels. `distinct` counts distinct session values out of 100, so 100 means every session is separable and low values mean quantization. **Read the last two rows first:** VAD refinement is what makes the 10 intra-turn features usable, and it is also what kills the 2 interruption features.

| Feature | distinct (VAD) | distinct (labels) | NaN (VAD) | NaN (labels) | sd (VAD) | sd (labels) |
|---|---:|---:|---:|---:|---:|---:|
| `ask_d` | 100 | 100 | 0% | 0% | 1.296 | 1.087 |
| `res_d` | 100 | 100 | 0% | 0% | 3.458 | 3.087 |
| `res_minus_ask` | 100 | 100 | 0% | 0% | 3.974 | 3.306 |
| `ask_minus_res` | 100 | 100 | 0% | 0% | 3.974 | 3.306 |
| `duration_sum` | 100 | 100 | 0% | 0% | 3.390 | 3.240 |
| `res_over_ask` | 100 | 100 | 0% | 0% | 3.441 | 1.567 |
| `ask_over_res` | 100 | 100 | 0% | 0% | 4.218 | 0.726 |
| `ask_ud` | 100 | 67 | 0% | 0% | 0.112 | 0.015 |
| `ask_du` | 100 | 67 | 0% | 0% | 0.611 | 0.051 |
| `res_ud` | 100 | 56 | 0% | 0% | 0.101 | 0.014 |
| `res_du` | 100 | 56 | 0% | 0% | 0.652 | 0.162 |
| `ask_sd` | 100 | 67 | 0% | 0% | 0.112 | 0.015 |
| `ask_ds` | 100 | 66 | 0% | 34% | 4.112 | 2.383 |
| `res_sd` | 100 | 56 | 0% | 0% | 0.101 | 0.014 |
| `res_ds` | 100 | 54 | 0% | 45% | 3.055 | 7.389 |
| `ask_su` | 100 | 67 | 0% | 0% | 0.611 | 0.051 |
| `ask_us` | 100 | 66 | 0% | 34% | 4.112 | 2.383 |
| `res_su` | 100 | 56 | 0% | 0% | 0.652 | 0.162 |
| `res_us` | 100 | 54 | 0% | 45% | 3.055 | 7.389 |
| `res_h` | 100 | 98 | 0% | 0% | 0.796 | 0.773 |
| `ask_bt` ⚠ | 1 | 23 | 0% | 0% | 0.000 | 0.004 |
| `res_bt` ⚠ | 1 | 16 | 0% | 0% | 0.000 | 0.004 |
| `ask_st` | 99 | 67 | 0% | 0% | 0.772 | 0.065 |
| `res_st` | 100 | 56 | 0% | 0% | 2.021 | 0.144 |

⚠ `ask_bt` and `res_bt` are **exactly constant** under VAD refinement. webrtcvad on a mono mixed channel emits one disjoint segment stream, and `turn_pairing.py` then orders every segment into speaker runs, so no response can start before its ask ends. Interruption is *unmeasurable* under this method rather than merely rare — a stronger statement than `docs/multi-corpus-plan.md` §2c's "0.2% of gaps are negative". 22 of 24 features are live; centered rank is 13.


## Control: the identical protocol on DAIC-WOZ

The same `nested_cv_classification()` over DAIC-WOZ's pooled 180 sessions (24-D session means) gives macro-F1 **0.652 ± 0.015**, balanced accuracy 0.669, AUC 0.698 — clearly above chance. The PDCH null above is therefore a property of PDCH, not of this code.


## Within-subject change (exploratory — *not* pre-registered)

16 subjects contribute two labelled sessions, so each is their own control and every between-subject confound drops out. Correlation of within-subject ΔCTD against ΔHAMD-17:

| Feature | r (Δ vs ΔHAMD) | p |
|---|---:|---:|
| `ask_du` | 0.462 | 0.071 |
| `ask_su` | 0.462 | 0.071 |
| `ask_st` | 0.461 | 0.072 |
| `res_du` | 0.455 | 0.076 |
| `res_su` | 0.455 | 0.076 |
| `ask_ud` | -0.429 | 0.097 |
| `ask_sd` | 0.429 | 0.097 |
| `ask_ds` | -0.319 | 0.228 |
| `res_h` *(pre-registered focus)* | 0.306 | 0.248 |

> EXPLORATORY -- not covered by any pre-registered prediction. At n=16 pairs this is a lead to pre-register for a future cohort, not a result.


## Pre-registered predictions

| ID | Prediction | Evidence | Outcome |
|---|---|---|---|
| **P1** | Among tier-B (cross-speaker) features, res_h has the largest absolute univariate association with the PDCH label. | `res_h` ranks 2/6 by \|r\| (r=0.158, behind `ask_over_res` at 0.255) but 1/6 by AUC (=0.646) | **FAIL** |
| **P2** | The res_h association is positive: longer response latency with greater severity, same sign as DAIC-WOZ. | PDCH r = +0.158 (positive); DAIC-WOZ train +0.104, test +0.528 | **PASS** |
| **P3** | Macro-F1 ordering on PDCH is no_ask >= all24 > no_cross > res_only. | macro-F1 `all24` 0.520, `no_ask` 0.522, `res_only` 0.468, `ask_only` 0.521, `no_cross` 0.522 | **FAIL** |
| **P4** | Dimensionless ratios retain more of their DAIC-WOZ association than absolute durations. | median retention (vs DAIC train) ratios 1.94 vs durations 1.05 | **PASS** |
| **P5** | PDCH single-feature AUC for res_h is below the 0.829 observed on DAIC-WOZ test. | PDCH `res_h` AUC 0.646 < 0.829 on DAIC-WOZ test | **PASS** |
| **P6** | ask_d will NOT have the largest absolute coefficient on PDCH (human clinicians adapt to the patient; the DAIC-WOZ wizard does not). | largest \|coef\| is `ask_over_res`; `ask_d` ranks 2/24 | **PASS** |

Full numbers are in `pdch_ctd_results.json` under `preregistered_predictions`.
