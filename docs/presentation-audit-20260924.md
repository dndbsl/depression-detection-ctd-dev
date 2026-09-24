# ICASSP presentation revision — 2026-09-24

> Subsequent user-requested title and second-author update: [title-author-update-20260924.md](title-author-update-20260924.md). The title and PDF hashes below describe this earlier presentation snapshot.

## Authorized scope and stance

Presentation only, following the user’s 2026-09-24 instructions. No new experiments, feature/label/protocol changes, or pushes. The title is preserved exactly:

```tex
\title{Conversational Timing for Depression Screening: A Bounded Empirical Study}
```

CTD carries DAIC-WOZ information beyond simple latency and interview-structure controls, with sensitivity to interviewer features and evaluation protocol. Fusion gains remain statistically inconclusive. PDCH is boundary evidence: no reliable discrimination, and weak directional latency–severity concordance accompanied by its FDR failure and the complete negative evidence. Neither detector transfer, construct validation, nor reliable multimodal advantage is claimed.

Base commit: `d793c0cef2d2ee4b4e0bdeaa7ff3d8dc977748f6`.

Pre-edit manuscript: `paper/revision-history/main-before-presentation-20260924.tex`. Initial source/result hashes and the backup hash are in `output/presentation_20260924/initial_state.json`. The experimental execution history remains in `docs/revision-audit-20260924.md`; this pass does not imply fresh experimental reruns.

## Presentation decisions

- Nested controls lead Results. Figure 1 replaces the five-row control table, retaining every configuration and both metrics. Text emphasizes 0.635 full CTD versus 0.550 latency+structure, decreasing to 0.566 after ask-side removal. Both the image and caption explicitly label whiskers as SD, not confidence intervals; no significance test is added.
- The full original-split ablation table remains, with C reselected per configuration. Nested evaluation is prioritized for feature-group robustness because selection occurs within outer training folds and performance is evaluated across partitions. Original-split estimates are sensitivity results under different partitions and training sizes. Paired intervals including zero mean no clear difference detected, not equivalence. Cross-speaker coordinates still contain interviewer timing.
- The complete PDCH evidence list appears in Section 5.3 only: within-corpus F1/AUC, both latency–severity correlations and q values, zero dual-corpus binary FDR hits, 12/22 sign agreement, max-|r| permutation result, and the language/instrument/population/interviewer/timing-measurement differences. Introduction and conclusion point to that section; the abstract reports the discrimination boundary without promoting directional concordance.
- Fusion is secondary. The modality table retains CTD, text, text+CTD, equal-weight fusion, WavLM seed 44 and the six-seed mean. A+T and A+CTD rows are omitted from the compact table but remain in unchanged recorded artifacts. The test-based WavLM seed selection, unavailable caches/checkpoints/predictions, and probe-specific interpretation of zero acoustic weight remain explicit.
- Agarwal is one context sentence retaining the published 0.80 value and the unmatched-protocol / not-rerun qualification. The official 107/35/47 CTD sensitivity and its missing-turn/imputation and cleaning caveats remain.
- Discussion and conclusion are combined. The broad no-transfer/no-construct-validation/no-reliable-multimodal-advantage statement appears once. Data integrity, nine exclusions, corrected/retained session 409, prior inspection, neural seed disclosure, and ethics checks remain.
- Rank is described as the observed centered rank of the 22 shared live coordinates, not a universal reduction of the full 24-dimensional model. Nonlinear ratio means need not collapse to the mean of one turn-level primitive. The historical audit’s stronger explanatory prose is not adopted; its numerical artifacts and scientific generator are unchanged.

## Existing sensitivity results promoted to text

The newly displayed downsampling values are copied from the existing committed `output/shared_signal_audit.json`, not recomputed. They are descriptive sample-size/prevalence sensitivity, not a formal power analysis or an attribution of the PDCH null.

The implementation draws 100 subsets from all 180 cleaned DAIC sessions, including the original test cohort; this differs from the primary timing-control study’s 135 train+dev subjects. Each subset has 62 distinct DAIC subjects, unlike PDCH’s 62 sessions from 46 subjects. It fixes the silence threshold to 0.2 s and uses three repeats, versus within-fold threshold selection and twenty repeats for the main PDCH result. Subsamples overlap. The manuscript discloses the full-cohort origin, unmatched repeated-subject structure, and protocol differences.

The stored percentiles compare previously rounded PDCH points (F1 0.520, AUC 0.476) with the subsample distribution. DAIC subsample AUC minimum is 0.4804232804, so “below all sampled values” is literal, not a p-value or proof of a residual corpus mechanism. Matching and residualization failed to recover discrimination; neither rules out confounding. The original `res_h` 0.829 test AUC is not revived, the ask-duration sign reversal is not promoted as a causal mechanism, and no “20% of accuracy” attribution is introduced.

## Number ledger

The complete current ledger is `output/paper_numbers.json` / `.md`, with source hashes, exact display strings, JSON keys, original generating commands, TeX line numbers and figure paths. It contains 152 distinct published keys (137 in TeX, 20 in the figure, with five shared). Only used TeX macros are emitted; other JSON/Markdown entries preserve audit-only checks.

| Manuscript value / location | Existing source file → key | Original generating command (not rerun in this pass) |
|---|---|---|
| Full CTD, Fig. 1 / headline: **0.635 / 0.026** | `output/timing_controls.json` → `results.all24.macro_f1.mean`; `output/timing_controls.json` → `results.all24.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| Latency + structure, Fig. 1 / headline: **0.550 / 0.025** | `output/timing_controls.json` → `results.latency_structure.macro_f1.mean`; `output/timing_controls.json` → `results.latency_structure.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| No ask-side group, Fig. 1 / headline: **0.566 / 0.025** | `output/timing_controls.json` → `results.no_ask.macro_f1.mean`; `output/timing_controls.json` → `results.no_ask.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| Latency, Fig. 1: **0.531 / 0.008** | `output/timing_controls.json` → `results.latency.macro_f1.mean`; `output/timing_controls.json` → `results.latency.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| Structure, Fig. 1: **0.499 / 0.020** | `output/timing_controls.json` → `results.structure.macro_f1.mean`; `output/timing_controls.json` → `results.structure.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| Original full CTD, dev / test: **0.746 / 0.631** | `output/ctd_ablation_groups.json` → `configs.all24.dev.macro_f1`; `output/ctd_ablation_groups.json` → `configs.all24.test.macro_f1.point` | `python src/ctd/ablation_groups.py` |
| Original no-ask, dev / test: **0.814 / 0.641** | `output/ctd_ablation_groups.json` → `configs.no_ask.dev.macro_f1`; `output/ctd_ablation_groups.json` → `configs.no_ask.test.macro_f1.point` | `python src/ctd/ablation_groups.py` |
| Original test all24 − no-ask, point / CI limits: **-0.010 / -0.089 / +0.054** | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.point`; `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.ci_low`; `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.ci_high` | `python src/ctd/ablation_groups.py` |
| Text+CTD test F1 / observed gain: **0.669 / 0.038** | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.point`; `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.point - results.single_roberta.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| WavLM six-seed mean / SD: **0.506 / 0.034** | `output/revision_review_context.json` → `SLT08.wavlm_test_seed_mean`; `output/revision_review_context.json` → `SLT08.wavlm_test_seed_sd` | `python scripts/revision_context.py` |
| PDCH F1 mean / SD: **0.520 / 0.055** | `output/pdch_ctd_results.json` → `classification.all24.macro_f1.mean`; `output/pdch_ctd_results.json` → `classification.all24.macro_f1.sd` | `python src/ctd/pdch_experiment.py` |
| PDCH AUC mean / SD: **0.476 / 0.066** | `output/pdch_ctd_results.json` → `classification.all24.roc_auc.mean`; `output/pdch_ctd_results.json` → `classification.all24.roc_auc.sd` | `python src/ctd/pdch_experiment.py` |
| Latency–severity rho, DAIC / PDCH / PDCH q: **0.245 / 0.267 / 0.431** | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.daic.per_feature.res_h__amean.r`; `output/shared_signal_audit.json` → `E4_secondary_targets.F2.pdch.per_feature.res_h__amean.r`; `output/shared_signal_audit.json` → `E4_secondary_targets.F2.pdch.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` |
| Binary dual-corpus hits / signs agreeing / live features: **0 / 12 / 22** | `output/shared_signal_audit.json` → `E2_concordance.concordance.counts.confirmed_shared`; `output/shared_signal_audit.json` → `E2_concordance.concordance.sign_agreement.n_agree`; `output/shared_signal_audit.json` → `E2_concordance.concordance.sign_agreement.n_comparable` | `python src/ctd/shared_signal_audit.py` |
| PHQ8_Moving rho / q: **0.107 / 0.520** | `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.per_feature.res_h__amean.r`; `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` |
| Downsampling repetitions / F1 percentile: **100 / 10** | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.n_subsamples`; `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.pdch_located_in_daic_null.macro_f1.percentile_within_daic_at_n62` | `python src/ctd/shared_signal_audit.py` |
| Downsampled DAIC F1 mean / SD: **0.605 / 0.063** | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.macro_f1.mean`; `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.macro_f1.sd` | `python src/ctd/shared_signal_audit.py` |
| Downsampled DAIC AUC mean / SD: **0.636 / 0.075** | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.roc_auc.mean`; `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.roc_auc.sd` | `python src/ctd/shared_signal_audit.py` |

Figure provenance is additionally recorded in `output/paper_timing_controls.json`: full-precision means/SDs, exact displayed strings, source and PDF hashes, and the rendering command. Axis ticks are plotting coordinates, not additional experimental estimates. The observed-rank and fixed-protocol numbers retain their original ledger sources.

## Commands and verification

Final commands and logs: `output/presentation_20260924/commands.json`. Package versions: `package_versions.json` in that directory. Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.9, pytest 9.1.1; TeX Live 2026.

| Check | Command / artifact | Result |
|---|---|---|
| Numbers and usage ledger | `python scripts/build_paper_numbers.py` | Published display strings match existing source artifacts; unused TeX macros pruned. |
| Figure | `python scripts/plot_paper_timing_controls.py` | All five configurations, both metrics, means and SDs rendered; no fitting or new analysis. |
| Source / number integrity | `python scripts/verify_paper_revision.py --record-dir output/presentation_20260924` | Pass; all 95 tracked scientific source/result files hashed at entry remain unchanged. |
| Full relevant tests | `DAIC_WOZ_ROOT=/home/exouser/data/DAIC OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg /home/exouser/.venvs/ctd/bin/python -m pytest -q` | 31 passed in 15.04 s. |
| PDF build | In `paper/`: `pdflatex -interaction=nonstopmode -halt-on-error main.tex`, `bibtex main`, then two more identical pdfLaTeX passes | Five pages; all technical content within the first four, fifth page references only. |
| Layout / PDF extraction | `pdfinfo`, `pdffonts`, `pdftotext`, rendered-page inspection | No overfull boxes or undefined references/citations; fonts embedded, no Type 3 fonts; all ten mean±SD labels verified in PDF text. |

The DAIC gate was **not rerun** in this presentation-only pass: no shared scientific code was edited. The existing gate artifact remains byte-identical (MD5 `ca0b9cb2dd4c799b3a7c12532ceafd7e`), preserving dev macro-F1 0.746154 / test 0.630946, C=0.3, cohort 102/33/45. The earlier before/after execution records remain the evidence for those runs.

Final PDF and figure hashes, page checks and source invariance are in `output/presentation_20260924/layout_validation.json`; the complete deliverable hash list is in `deliverable_manifest.json` there. `paper/main.pdf` remains a local build artifact under the existing ignore policy; the generated figure PDF is committed. Pre-edit TeX and historical experimental logs are preserved.

## Scope and remaining submission checks

E6 remains unrun because neural artifacts are unavailable. No experiments, E-DAIC work, or Chinese encoder work were started. SLT-01/02/03/05/06 status cells alone were updated for this presentation pass. The prior revision audit links here so its historical line-number ledger is not mistaken for the current manuscript.

Author confirmation of ethics/data-use wording, funding/conflicts, the submission window, and reviewer access to the intended artifacts remain external submission checks. No external messages or pushes were made. The changes are committed locally on `icassp2027` after verification.
