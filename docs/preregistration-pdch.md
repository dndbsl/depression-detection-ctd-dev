# Pre-registration — CTD transfer to PDCH

**Committed before any PDCH label was joined to any feature.** The git commit
timestamp is the evidence. Nothing below may be edited after PDCH results
exist; corrections go in a dated appendix at the bottom.

## Why this file exists

On DAIC-WOZ, `res_h` (response latency) turned out to be the strongest single
feature on **test** (point-biserial r = .528, AUC = .829, sign-consistent across
train/dev/test — `output/ctd_interpretability.md` §3). That is a **post-hoc
test-set observation**. Acting on it without pre-registering would be exactly
the dev/test-fitting this paper was already criticized for (`SLT-02`), only
across corpora instead of splits.

Writing the predictions down first converts PDCH from curve-fitting into
confirmation.

## Predictions

Stated so they can fail.

**P1 — `res_h` dominates.** Among tier-B (cross-speaker) features, `res_h` will
have the largest absolute univariate association with the PDCH label
(point-biserial r, and single-feature AUC).

**P2 — Direction is preserved.** The association will be **positive**: longer
response latency with greater depression severity, the same sign as DAIC-WOZ.
A sign flip falsifies the psychomotor-retardation interpretation.

**P3 — Tier ordering reproduces.** Macro-F1 ordering on PDCH will be
`no_ask` ≥ `all24` > `no_cross` > `res_only`, i.e. the cross-speaker group
carries the signal, as on DAIC-WOZ (`output/ctd_ablation_groups.md`).

**P4 — Ratios transfer, absolute durations do not.** Dimensionless features
(`res_ud`, `res_sd`, `res_su`, `res_over_ask`, `ask_over_res`) will retain more
of their DAIC-WOZ association than absolute durations (`ask_d`, `res_d`,
`duration_sum`), because Mandarin is syllable-timed and English stress-timed.

**P5 — Effect size shrinks.** PDCH AUC for `res_h` will be **below** the .829
observed on DAIC-WOZ test. That number is a small-sample high-water mark
(n = 45) and the label construct differs (clinician-rated HAMD-17 vs.
self-report PHQ-8). Predicting shrinkage now prevents reporting any drop as
disappointing.

**P6 — `ask_d` will not dominate on PDCH.** On DAIC-WOZ `ask_d` has the largest
coefficient, but the interviewer there is a wizard-driven agent. PDCH
interviewers are human clinicians who adapt to the patient, so if `ask_d`
dominates on PDCH too, that is evidence the feature tracks *interviewer
behaviour*, not patient state — which would weaken, not strengthen, the
paper.

## Analysis plan, fixed in advance

- **Label:** HAMD-17 total ≥ 17 (moderate-or-worse), 27/62 positive. Secondary:
  regression on HAMD-17 total. **Not** HAMD ≥ 8 (85% prevalence, degenerate).
- **Splits:** subject-grouped cross-validation (`GroupKFold` on subject ID). 27
  of 72 subjects contribute two sessions; session-level splitting would leak.
- **Features:** the same 24, with the same tier definitions from
  `src/ctd/feature_groups.py`. No new features invented for PDCH.
- **Scaling:** fit within PDCH training folds only. No cross-corpus scaler.
- **Detector:** the deployed pipeline unchanged — median impute → standardize →
  L2 LogReg, `class_weight='balanced'`.
- **`C`:** selected within training folds. Reported as a distribution, since it
  was unstable on DAIC-WOZ (`SLT-10`: modal values tied at 0.1 and 1.0).
- **Silence threshold:** the DAIC-tuned 0.2 s constant in `_silence_gap_count`
  will be re-selected on PDCH training folds and its sensitivity reported.

## Declared dependency: VAD refinement

These predictions assume VAD recovers sub-second turn onsets from PDCH's
1-second turn labels (`MC-06`). If the feasibility spike shows it does not,
**P1, P2, P3 and P5 are untestable on PDCH** and that must be reported as a
limitation rather than worked around by substituting a different feature.

Recorded now so a negative VAD result cannot be quietly reframed.

---

## Appendix — dated amendments

### 2026-09-23 — Outcome of P1–P6

Run: `src/ctd/pdch_experiment.py` (results in `output/pdch_ctd_results.json`
and `output/pdch_ctd_results.md`), on all 62 labelled PDCH sessions / 46
subjects, subject-grouped `StratifiedGroupKFold` 5-fold × 20 repeats, deployed
pipeline unchanged. The predictions above are **unedited**; this entry only
records what happened.

#### Headline: CTD does not transfer to PDCH

All five pre-committed tiers land at chance:

| Config | n | Macro-F1 | Balanced acc. | ROC AUC |
|---|---:|---|---|---|
| `all24` | 24 | 0.520 ± 0.055 | 0.522 ± 0.054 | 0.476 ± 0.066 |
| `no_ask` | 15 | 0.522 ± 0.054 | 0.524 ± 0.053 | 0.489 ± 0.058 |
| `res_only` | 8 | 0.468 ± 0.054 | 0.472 ± 0.053 | 0.445 ± 0.072 |
| `ask_only` | 9 | 0.521 ± 0.041 | 0.522 ± 0.041 | 0.497 ± 0.030 |
| `no_cross` | 17 | 0.522 ± 0.062 | 0.524 ± 0.060 | 0.488 ± 0.079 |

Balanced accuracy is within noise of 0.50 and AUC is at or below it, so the
macro-F1 of ~0.52 (against an always-negative baseline of 0.361) reflects
`class_weight='balanced'` splitting predictions, not discrimination. Severity
regression is likewise null: MAE 6.65 ± 0.10 against a predict-the-mean
baseline of 6.55, Pearson r 0.083 ± 0.120.

**This is not a pipeline failure.** Three controls say so: (i) the identical
`nested_cv_classification()` on DAIC-WOZ's pooled 180 sessions gives macro-F1
0.652 ± 0.015 / AUC 0.698; (ii) `MC-06` succeeded on its own terms — VAD
recovers 5.10 utterances/turn and a `res_h` median of 0.60 s, matching the
spike; (iii) the feature/label join was verified against an independent
`pandas.merge` and spot-checked row-by-row against the source spreadsheet.

#### Verdicts

| ID | Outcome | Numbers |
|---|---|---|
| **P1** | **FAIL** | `res_h` ranks 2/6 among testable tier-B features by \|r\| (0.158, behind `ask_over_res` at 0.255) but 1/6 by single-feature AUC (0.646 vs 0.554). The prediction named *both* measures, so the conjunction fails. |
| **P2** | **PASS** | PDCH `res_h` r = **+0.158**, same sign as DAIC-WOZ (train +0.104, test +0.528). No sign flip. |
| **P3** | **FAIL** | Predicted `no_ask` ≥ `all24` > `no_cross` > `res_only`. Observed 0.522 / 0.520 / 0.522 / 0.468 — `no_ask ≥ all24` holds, `all24 > no_cross` does not. The ordering is untestable in practice: four of five tiers are separated by ≤0.002 with sd ≈ 0.055. |
| **P4** | **PASS** (primary reference only) | Median retention \|r_PDCH\|/\|r_DAIC\|: ratios 1.94 vs durations 1.05 against DAIC train; 0.64 vs 0.56 against dev; **0.61 vs 0.96 against test (fails)**. Mean \|r_PDCH\| is effectively tied (ratios 0.137, durations 0.143). Weak support at best. |
| **P5** | **PASS** | PDCH `res_h` AUC 0.646 < 0.829 on DAIC-WOZ test. |
| **P6** | **PASS** | Largest \|coefficient\| is `ask_over_res` (0.051); `ask_d` ranks 2/24 (0.047). `ask_d` does not dominate. |

#### Read these passes conservatively

P5 and P6 are predictions of *shrinkage* and *non-dominance*, and a null result
satisfies both trivially — nothing dominates when nothing predicts. P2 is a
sign test on an association that is not significant (p = 0.22). Counting four
of six as confirmation would invert the purpose of this file. The defensible
summary is: **the only tier-B signal that survived to PDCH is a weak,
correctly-signed `res_h` association (AUC 0.646, p = 0.22) that does not
support a multivariate detector at n = 62.**

#### Amendments

1. **P4's reference split was unspecified.** The prediction says "retain more
   of their DAIC-WOZ association" without naming a split or a retention
   statistic. Fixed after the fact as: retention = \|r_PDCH\|/\|r_DAIC\|,
   compared by group median, with DAIC-WOZ **train** primary (largest n, used
   for no selection). All three splits are reported because the verdict flips
   on test. This choice was made while the result was already visible and
   should be treated as such.
2. **P1's two measures disagree.** Recorded as FAIL on the strict conjunction,
   with both rankings reported rather than the favourable one.
3. **New limitation, not anticipated here: `ask_bt` and `res_bt` are
   structurally zero on PDCH.** webrtcvad on a mono mixed channel emits one
   disjoint segment stream, and the run-based pairing in `turn_pairing.py`
   then orders every segment, so no response can begin before its ask ends.
   Overlap/interruption is therefore *unmeasurable* under this method, not
   merely rare — stronger than the "0.2% of gaps are negative" caveat in
   `docs/multi-corpus-plan.md` §2c. 22 of 24 features are live; `all24` is
   effectively 22-D, `no_ask` 14-D, `ask_only` 8-D. Centered rank is 13.
4. **Corpus inventory differs from `multi-corpus-plan.md` §2a.** On disk today:
   100 session directories, 73 distinct subject IDs, and **167** wavs each with
   a matching timestamped transcript — so the "2 wavs to drop" is a no-op and
   0 wavs were dropped. 51 turns (0.16%) were dropped instead: 3 with an
   unparseable speaker and 48 whose start jumps backwards.
5. **The declared VAD dependency was satisfied.** P1/P2/P3/P5 were testable;
   the null is a fact about PDCH, not an untested prediction.

### 2026-09-24 — How the shared-signal audit changes the reading of P5 (predictions unedited)

The predictions above and the 2026-09-23 verdicts are unchanged. This entry
records one thing the audit
([`shared-signal-audit.md`](shared-signal-audit.md),
[`output/shared_signal_audit.md`](../output/shared_signal_audit.md)) establishes
that alters how a verdict should be *read*.

**P5 passed for the wrong reason.** P5 predicted PDCH's `res_h` AUC would fall
below the 0.829 seen on DAIC-WOZ test, and framed that as *shrinkage* — a
smaller true effect under a different instrument and a smaller sample. It did
fall (0.646 < 0.829), so P5 passed. The audit shows the anchor was the outlier:
the same single feature through the same pipeline scores macro-F1 **0.581 on
DAIC-WOZ dev** and **0.595 ± 0.006 under pooled subject-grouped CV** on all 180
sessions, against 0.779 / AUC 0.829 on the 45-session test split. The 0.829 is
not a high-water mark of a real effect, it is one split's noise.

So P5's pass says nothing about PDCH. The correct comparison — `res_h` against
*continuous* severity, which the audit's F2 family tests — gives Spearman
ρ = +0.245 on DAIC-WOZ (train+dev, BH-q = 0.051) and ρ = +0.267 on PDCH
(permutation p = 0.086, BH-q = 0.431). Same sign, near-identical magnitude,
different instruments. There is no shrinkage to explain; PDCH simply lacks the
power at n = 62 to clear FDR on an effect of that size.

This strengthens P2 (direction preserved) rather than P5, and it retires
`PF-05`'s premise. Recorded here because P5's pass would otherwise be cited as
evidence that a real DAIC-WOZ effect attenuated cross-lingually, which the data
do not support.

No prediction text above has been altered.

### 2026-09-24 — Bounded manuscript interpretation (predictions unchanged)

The full experiment and shared-signal audit were rerun for the ICASSP revision.
The earlier outcome prose is historical; the current interpretation is in
[`revision-audit-20260924.md`](revision-audit-20260924.md). Within-PDCH
cross-validation trains within PDCH and does not test detector transfer.
Weak directional latency–severity concordance fails PDCH FDR, accompanies
zero dual-corpus binary FDR hits, near-chance sign agreement, and no reliable
within-PDCH discrimination. Language, instrument, population, interviewer
and timing measurement differ. Downsampling does not identify low power as
the cause of the null; no construct validation is claimed. P1–P6 and their
original analysis specification above remain unchanged.
