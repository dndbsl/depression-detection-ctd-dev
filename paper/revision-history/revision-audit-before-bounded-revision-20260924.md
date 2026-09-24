# Revision audit — 2026-09-24

## Completed

- Rebuilt the manuscript around predictive value and robustness, with the
  cross-corpus negative and uncertainty visible in the abstract and conclusion.
- Added the existing coefficient evidence, five feature-group ablations,
  resampling results, and a protocol-qualified published text comparison.
- Corrected nine exclusions versus session 409's retained corrected label;
  documented balanced-accuracy selection for CTD and distinct threshold rules.
- Removed claims that nonsignificance proves equivalence, ask-only ablation
  eliminates interviewer confounding, PDCH proves replication/transfer, or
  downsampling identifies low power as the cause of a null.
- Disclosed prior test inspection and the WavLM test-based seed choice recorded
  in RESULTS.md. The T+CTD result has no acoustic term, but that does not make
  the rest of the evaluation independently confirmatory.
- Added verified PDCH, Hamilton-scale, Agarwal et al., and preprocessing
  references. Removed unresolved citation placeholders from the active paper.
- Preserved the user's exact pre-revision TeX in
  `paper/revision-history/main-before-review-20260924.tex`.

## New experiment

Protocol recorded in `docs/timing-controls-protocol.md` before running this
comparison. Script: `src/ctd/timing_controls.py`. Results:
`output/timing_controls.{json,md}`. This is exploratory work informed by prior
benchmark inspection, not a new untouched cohort.

Only train+dev (135 subjects, 41 positive) are used. Five outer folds, four
inner folds, ten identical repeats per configuration, with training-fold-only
imputation/scaling and inner balanced-accuracy selection of C. Threshold 0.5.

| Features | Macro-F1 mean ± SD | AUC mean ± SD |
|---|---:|---:|
| Full CTD | .635 ± .026 | .676 ± .031 |
| No ask-side group | .566 ± .025 | .610 ± .030 |
| Latency | .531 ± .008 | .562 ± .015 |
| Log span + log(1+turn count) | .499 ± .020 | .535 ± .027 |
| Latency + structure | .550 ± .025 | .576 ± .029 |

Full CTD has higher mean scores than the simple controls. Ask-side removal
reduces mean scores, contradicting a general claim of dispensability based
on the original split. Repeats are dependent; SD is not a confidence interval
and no independent-repeat significance test is warranted.

Reproduce:

```bash
DAIC_WOZ_ROOT=/home/exouser/data/DAIC OPENBLAS_NUM_THREADS=1 \
  /home/exouser/.venvs/ctd/bin/python src/ctd/timing_controls.py
```

## Verification

- Re-ran `src/ctd/ml_splits.py`: 102/33/45 subjects, C=0.3,
  dev macro-F1 .746154 and test .630946, matching the reported baseline.
- Real-data cleaning and CTD consistency tests: six passed.
- LaTeX compilation succeeds: `paper/main.pdf` is a seven-page working draft,
  with no undefined citations/references or overfull boxes in the final log.
  Venue-specific length reduction remains necessary before submission.
- New experiment completed for all five specified configurations. Aggregate
  JSON includes source transcript/cache hashes and package versions.

## Still required before a stronger resubmission

1. **Complete nested text/fusion evaluation.** RoBERTa frozen feature caches,
   checkpoints and per-session fusion inputs are absent from the expected
   project paths and were not found in the searched project/data/cache trees.
   Existing aggregate JSON cannot reconstruct participant predictions or
   support retraining. Restore raw frozen embeddings or regenerate them before
   refitting all heads and fusion in nested folds. Do not reuse embeddings
   standardized on the full original train set without recovering raw values.
   Preselect text-only, text+full CTD, text+no-ask, and text+simple-controls;
   use the same outer subject folds and inner-only model/threshold selection.
2. **Reproduce a strong published text baseline.** The literature comparison
   is explicitly contextual. A quotation of .80 is not a matched rerun.
3. **Validate PDCH timing.** Independent label-blind boundary annotation and
   timing-error assessment are still needed. VAD output alone does not provide
   ground truth. Keep sessions clustered by subject in uncertainty estimates.
4. **Fresh external evaluation.** Within-PDCH CV is not frozen transfer.
   E-DAIC overlaps DAIC and needs speaker labeling; use only verified disjoint
   subjects if pursuing it. Do not claim interviewer regime is isolated when
   language, population, instrument and measurement also differ.
5. **Release and submission check.** Verify every linked analysis is actually
   available to reviewers, audit neural selection/refit provenance, and check
   the chosen venue's current page and anonymity requirements. This is a
   working scientific revision, not a claim of submission readiness.

No commits, pushes, new encoder downloads, external messages, or changes to
the deployed model or canonical 24-feature definition were made.
