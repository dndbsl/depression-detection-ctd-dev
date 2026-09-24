# Bounded empirical revision audit — 2026-09-24

> Presentation follow-up: [presentation-audit-20260924.md](presentation-audit-20260924.md)
> documents the subsequent presentation-only pass. The exact title is retained;
> nested controls lead Results, and the full PDCH evidence list appears once,
> with pointers elsewhere, as subsequently requested by the user. The execution
> history and inline ledger below describe the earlier `d793c0c` manuscript;
> current text/figure usage and number provenance are in `output/paper_numbers.*`.

## Locked stance and scope

> Conversational timing (CTD) carries information on DAIC-WOZ beyond simple
> latency and interview-structure controls, but that signal is sensitive to
> interviewer-side features and evaluation protocol; fusion gains are observed
> but statistically inconclusive; on PDCH we find only a weak, directionally
> consistent latency–severity association and no reliable within-corpus
> discrimination. We do not claim detector transfer, construct validation, or
> a reliable multimodal advantage.

The abstract, introduction's three-part contribution, discussion, and conclusion
use this stance or a close paraphrase. Every mention of directional PDCH
concordance accompanies its FDR failure, binary null, near-chance signs,
classification null, and the language/instrument/population/interviewer/timing
measurement differences. SA-05 item exploration is not promoted. The retired
PF-05/SA-01 single-feature test headline is absent from the manuscript.

This audit's skeleton, gate/execution tables and number-ledger headers were
written **before any rerun or edit to the active TeX**. E1–E5 completed before
the rewrite; E6 was explicitly scoped out. The initial working tree already
contained a draft revision, official-template migration, timing-control code
and results. Those related changes were preserved and completed.

Pre-edit copies (never overwritten):

- `paper/revision-history/main-before-bounded-revision-20260924.tex`: exact
  active TeX at the start of this pass.
- `paper/revision-history/main-before-review-20260924.tex`: earlier user draft,
  retained unchanged.
- `paper/revision-history/revision-audit-before-bounded-revision-20260924.md`:
  prior same-day audit, retained rather than silently replaced.

Historical decision prose is not the current scientific interpretation. The
shared-signal decision memo now points here; preregistration P1–P6 and their
original specification are unchanged, with only a dated interpretation appendix
added. No new CTD features, feature groups, label pooling, E-DAIC experiments
(MC-03/04), or Chinese encoders (MC-09) were introduced.

## Environment, commands and hashes

Base commit: `49520dc65987cf1f686e4aa28e0fd742778f6b31` on `icassp2027`. The worktree was initially dirty;
`output/revision_20260924/initial_state.json` preserves the initial status and
SHA-256 hashes. Each run JSON records its command array, cwd, base commit,
UTC start/finish, duration, return code, package versions, source/input hashes
and output hashes. Thus dirty-source changes are identifiable without pretending
all runs used an untouched commit. No push to either remote was made.

Canonical environment prefix for scientific reruns:

```bash
DAIC_WOZ_ROOT=/home/exouser/data/DAIC OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg /home/exouser/.venvs/ctd/bin/python scripts/run_revision_experiment.py RUN_ID /home/exouser/.venvs/ctd/bin/python -u SCRIPT
```

The before gate used Python without `-u`; this affects output buffering only.
Exact commands, including the executable, are in the per-run JSON files.
Python 3.10.12; NumPy 2.2.6; pandas 2.3.3; SciPy 1.15.3; scikit-learn 1.7.2;
confidence-intervals 0.0.3; matplotlib 3.10.9; webrtcvad 2.0.10; pytest 9.1.1.
The feasibility check found CUDA unavailable and `transformers` absent.
Complete installed-package inventories are retained in each run record.

Hashes cover the CTD/common source, DAIC transcripts and labels, regenerated
DAIC functional caches, PDCH VAD utterance cache, spreadsheet labels, and
aggregate outputs. No raw audio, clinical text, embeddings, or per-session
prediction arrays were added to version control. Logs are `.txt` because the
repository excludes generic `.log` build/run products.

## DAIC gate before and after shared-code reporting edits

| Stage | Command | Cohort | C | Dev macro-F1 | Test macro-F1 | Artifact MD5 | Verdict |
|---|---|---|---|---|---|---|---|
| Before | `python src/ctd/ml_splits.py` | 102/33/45 | 0.3 | 0.746154 | 0.630946 | `ca0b9cb2dd4c799b3a7c12532ceafd7e` | pass |
| After | `python -u src/ctd/ml_splits.py` | 102/33/45 | 0.3 | 0.746154 | 0.630946 | `ca0b9cb2dd4c799b3a7c12532ceafd7e` | pass |

The artifact is `src/ctd/outputs/ml_splits_results.json`; its aggregate contents
are also copied byte-for-byte to `output/daic_gate_results.json` for the paper's
ledger. `gate_preexisting.json`, `gate_before.json`, and `gate_after.json` in the
run-record directory preserve the pre-existing and refreshed payloads/hashes.
No shared scientific code was edited after the after gate. Later edits concern
standalone reporting/verification scripts and manuscript/audit files.

The changed existing Python scripts only correct reporting/docstrings:
`ablation_groups.py` now describes the actual balanced-accuracy selection and
says no clear difference detected; `pdch_experiment.py` describes within-corpus
CV, not transfer; `shared_signal_audit.py` corrects the live-feature rank note
and bounds the downsampling interpretation. The canonical definitions,
ASK/CROSS/RES_ONLY memberships, feature extraction, pairing and deployed
`ml_splits.py` remain SHA-256 identical to the starting state.

## Experiment execution ledger

| ID | Command (after environment prefix) | Outputs | Status / limits |
|---|---|---|---|
| E1 | `src/ctd/ml_splits.py` (before and after) | `output/daic_gate_results.json`; ignored `src/ctd/outputs/functionals_*.csv` | Full existing model grid rerun; canonical gate byte-stable. Neural/fusion artifacts retained, not recomputed. |
| E2 | `src/ctd/timing_controls.py` | `output/timing_controls.{json,md}` | All five declared rows rerun; JSON byte-identical to initial artifact. |
| E3a | `src/ctd/ablation_groups.py` | `output/ctd_ablation_groups.{json,md}` | C reselected per configuration by dev balanced accuracy; paired bootstrap B=2000. |
| E3b | `src/ctd/interpret.py` | `output/ctd_interpretability.{json,md}`, `output/ctd_coef_forest.png` | Coefficients, bootstrap CIs, permutation importance and split associations rerun; canonical selected C=0.3. |
| E4a | `src/ctd/pdch_experiment.py` | `output/pdch_ctd_results.{json,md}`, `output/pdch/pdch_feature_degeneracy.csv` | Full existing five-tier, threshold-sensitivity, severity, coefficient, DAIC-control and exploratory within-subject protocol rerun. Uses hashed existing VAD cache. |
| E4b | `src/ctd/shared_signal_audit.py` | `output/shared_signal_audit.{json,md}` | Full existing inventory, binary/severity/item families, matching, residualization, downsampling and candidate checks rerun. No search expansion. |
| E5a | `scripts/official_split_sensitivity.py` | `output/official_split_ctd.{json,md}` | One official 107/35/47 CTD run, C=1.0, dev/test F1=0.782/0.639, AUC=0.654; integrity caveat below. |
| E5b/E6 | `scripts/revision_context.py` | `output/revision_review_context.{json,md}` | Recorded neural provenance and cache/compute inventory refreshed; published context verified; E6 scoped out. |
| Extraction check | `scripts/check_official_features.py` | `output/revision_20260924/official_feature_check.json` | All 180 canonical session means agree, max absolute difference 7.11e-15; no model refit. |
| Number rendering | `scripts/build_paper_numbers.py` | `output/paper_numbers.{json,md}`, `paper/generated-numbers.tex` | Single display-string source for TeX/JSON/Markdown, including source hashes and main.tex line locations. |
| Number verification | `scripts/verify_paper_revision.py` | `output/revision_20260924/number_verification.json` | All active manuscript numeric macros checked against source hashes and identical display strings; canonical and recorded neural artifacts unchanged. |

Per-run record IDs are `gate_before`, `gate_after`, `e2_timing_controls`,
`e3_ablation`, `e3_interpret`, `e4_pdch`, `e4_shared_signal`, `e5_official_split`,
`e5_e6_context_final`, `official_extraction_check`, and `paper_numbers_final`.
Their `.json` files contain metadata and `.txt` files contain complete stdout /
stderr. Intermediate context/number-rendering records are retained too.

The PDCH full run took about 565 s and the shared-signal audit about 342 s.
`numeric_refresh_check.json` verifies exact equality of every numeric leaf to
the base commit: ablations 108, interpretability 1026, PDCH 1079, shared audit
2370. The timing-control JSON is also byte-identical to the initial artifact.
The shared-audit log retains scikit-learn warnings about single-class fold
metrics in small grouped resamples; the declared protocol was not altered to
remove those warnings. All runs completed successfully.

## Key number ledger summary

| Manuscript quantity | Refreshed display | Source / command |
|---|---|---|
| Nested full CTD | F1 0.635 ± 0.026; AUC 0.676 ± 0.031 | `output/timing_controls.json` → `timing_controls.py` |
| Nested latency + structure | F1 0.550 ± 0.025; AUC 0.576 ± 0.029 | same |
| Nested no-ask | F1 0.566 ± 0.025; AUC 0.610 ± 0.030 | same |
| Other declared controls | Latency F1 0.531 ± 0.008 / AUC 0.562 ± 0.015; structure F1 0.499 ± 0.020 / AUC 0.535 ± 0.027 | same; all five rows retained |
| Reselected no-ask | C=1.0; dev/test F1 0.814/0.641 | `output/ctd_ablation_groups.json` → `ablation_groups.py` |
| Paired all24 − no-ask | dev −0.068 [−0.216,+0.083]; test −0.010 [−0.089,+0.054] | same; no clear difference detected, no equivalence claim |
| Recorded text–CTD fusion | dev/test 0.804/0.669; test delta 0.038; test marginal CI [0.509,0.806] | base-commit `output/fusion_wavlm_seed44.json`; not rerun |
| Recorded equal-weight fusion | dev/test 0.738/0.650 | base-commit `output/mean_prob_seed44.json`; threshold still dev-selected |
| Recorded WavLM sweep | test 0.506 ± 0.034, six seeds; deployed seed 44 selected on test | base-commit `RESULTS.md` and acoustic README; not recomputed from the two-seed fusion summary |
| Within-PDCH all24 | F1 0.520 ± 0.055; AUC 0.476 ± 0.066 | `output/pdch_ctd_results.json` → `pdch_experiment.py` |
| Latency–severity | DAIC rho 0.245, q 0.051; PDCH rho 0.267, q 0.431 | `output/shared_signal_audit.json` → `shared_signal_audit.py` |
| Binary family boundary | zero dual-corpus FDR hits; signs 12/22; PDCH max-abs-r permutation p=0.340 | same |
| PHQ8_Moving latency null | rho 0.107, q 0.520 | same; no psychomotor validation or clinician/self-report superiority inference |
| Official cohort sensitivity | 107/35/47; dev/test F1 0.782/0.639; test AUC 0.654 | `output/official_split_ctd.json` → `official_split_sensitivity.py` |

All nested SDs describe dependent repeats, not CIs. The original split's
no-ask result does not negate the nested ask-side sensitivity. No significance
test across repeats, equivalence test, causal decomposition, or stable fusion
claim was added. PHQ-8/HAMD-17 associations are estimated separately.

The manuscript reports the observed centered rank of the 22 live coordinates
(13 on both corpora). It does **not** turn nonlinear per-turn identities into
a claim that session means contain only two independent latent quantities.
The older aggregate-report discussion of that distinction is not used as a
manuscript claim.

## Scoped review decisions

- **SLT-06 — context only.** Verified Agarwal, Dias and Dollfus, CLPsych 2024,
  Tables 1–2: official 107/35/47, test macro-F1 0.80, both speakers' text.
  [Primary paper](https://aclanthology.org/2024.clpsych-1.9.pdf). A local matched
  implementation/checkpoint is absent; reproducing and training that baseline
  is outside this pass. Table 2 explicitly states non-matched protocols and
  that no published method was reproduced. No SOTA claim.
- **SLT-07 — CTD completed, neural scoped out.** Before computing scores, the
  audit specified restoring all nine exclusions, retaining PHQ-8 >=10 / the
  session-409 correction, keeping the deployed C grid and selection/refit
  rule, and imputing entirely missing descriptors for no-pair sessions.
  All official sessions are included. No-pair sessions are 451 and 458 (dev)
  and 480 (test); their features use training-fitted medians, with no invented
  interviewer turns. Timestamp/interruptions remain defective. T/T+CTD cannot
  be evaluated for the full cohort because session predictions and encoder
  artifacts are missing.
- **SLT-08 — disclosure completed; stronger model deferred.** The frozen
  acoustic probe, test-based seed-44 choice, and six-seed mean are retained.
  Zero selected acoustic weight is probe/search-specific. No fine-tuning or
  alternative acoustic model was run without its artifacts.
- **E6 — scoped out, no fabricated scores.** Searches covered project, data
  and cache trees for `.npz`, `.pt`, `.pth`, `.safetensors` and prediction CSVs;
  no raw RoBERTa/WavLM embeddings, selected checkpoints, model weights, or
  session predictions were found. CUDA is unavailable and `transformers` is
  absent. Rebuilding a large encoder cache from raw and then fitting nested
  heads/fusion is beyond this bounded CPU revision. Aggregate scores cannot
  reconstruct inputs. The declared text / text+CTD / text+no-ask /
  text+simple-controls comparison remains explicit future work on E2 folds.
  The paper makes no fusion-stability claim.

## What was not rerun, and exact provenance limits

1. Neural heads, encoders, seed sweeps, original fusion and their marginal
   bootstraps were **not** recomputed. The original JSON files are byte-identical
   to commit `49520dc`; their values are re-rendered in the ledger. The source
   hashes and selected thresholds are recorded. WavLM seed choice uses test.
2. No exact numerical paired-fusion-CI artifact or underlying prediction files
   were found. The original paper/reviews record that the paired intervals
   include or touch zero (`reviews/slt2026/reviews.md` at the base commit).
   The manuscript explicitly describes this as recorded evidence and says its
   exact limits cannot be refreshed. No paired limits were invented, and
   overlapping marginal CIs are not substituted for a paired test.
3. PDCH VAD was **not** regenerated from raw audio; existing hashed utterance
   boundaries were reused and all downstream requested analyses rerun.
   Independent label-blind manual boundary validation is still absent.
4. The older `resampling.py` selection-set analysis is omitted from the paper
   and was not rerun. It is not nested validation and is not a fair
   distribution-to-distribution neural comparison. The existing downsampling
   inside the shared audit was rerun, but is not used to attribute the PDCH
   null to low power. It does not match PDCH's repeated-subject structure.
5. The existing 240-functional model grid was rerun by `ml_splits.py` and is
   retained in `output/daic_gate_results.json`; the expanded descriptor claim
   is omitted for length. No model or search space was enlarged.
6. No E-DAIC diarization/experiment, Chinese semantic branch, subgroup
   experiment, or new multimodal model was started. Their absence bounds the
   manuscript rather than being concealed by a transfer or superiority claim.

## Tests, compilation and visual verification

- `DAIC_WOZ_ROOT=/home/exouser/data/DAIC ... python -m pytest -q`:
  **31 passed**, no skipped tests; includes real-data cleaning/feature-path
  consistency, PDCH adapter and shared-audit statistical tests.
- Authored changes pass `git diff --cached --check` with exact historical
  copies, upstream template files and verbatim logs excluded. Those preserved
  files retain their original trailing whitespace; they are not reformatted
  because doing so would destroy byte-exact history/log provenance.
- All standalone revision scripts compile; canonical features/groups and
  recorded neural JSON hashes remain unchanged. Official extraction was
  checked independently against the canonical session means.
- `python scripts/verify_paper_revision.py`: **155 unique active manuscript
  numeric keys** have identical display strings in TeX/JSON/Markdown, valid
  source hashes, and no unledgered empirical or protocol numerals. Mathematical
  notation, instrument names, author ORCIDs, layout dimensions, generated
  section/citation numbers and bibliography metadata are not empirical
  measurements. The scalar identity constant is also included in the ledger.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex`:
  **5 pages total; technical content ends on page 4; page 5 contains only references.**
  Official `spconf.sty` / `IEEEbib.bst`, US Letter, embedded Type-1 fonts,
  zero overfull boxes and zero undefined citations/references. Abstract:
  113 whitespace-delimited words after expanding numeric macros. All four
  content pages were rendered and inspected. PDF validation and font inventory
  are in `output/revision_20260924/pdf_validation.json` and `pdf_fonts.txt`.
- `paper/main.pdf` SHA-256: `3b5a12f95f27310897b955bc565cbd21def97ed30fe9cfbd5f274cb71a5e8e8a`.
  The PDF is delivered locally; it remains an ignored build artifact per the
  repository's existing policy. Source, aggregate artifacts and audit records
  are included in the local completion commit.

The [official ICASSP paper kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
permits four technical pages and a fifth page restricted to references,
funding acknowledgments and ethics compliance. The existing official-template
migration was kept; no margins or body font sizes were reduced to fit.

## Remaining submission blockers and bounded scientific limits

- **Author disclosures:** the [conference author policy](https://2027.ieeeicassp.org/sps-policies/)
  requires ethics-compliance and funding/conflict statements. Approval/waiver
  or the applicable secondary-data-use basis, funding and conflicts were
  requested asynchronously; no author facts were supplied during this pass.
  The paper includes a factual secondary-analysis statement and an explicit
  source comment for completion. It does not invent institutional approval or
  a no-conflicts declaration. Authors must confirm these before submission.
- **Submission access/deadline:** the [CFP](https://2027.ieeeicassp.org/call-for-papers/)
  lists September 23, 2026, while the paper kit still lists September 16.
  Confirm the applicable timezone and existing submission/revision access;
  this pass does not assume the portal is still accepting new submissions.
- **Reviewer access / release:** the new revision is local only. Verify a
  reviewer-accessible artifact release and applicable data-use terms before
  publishing new materials. No push, upload or external message was sent.

Nested fusion, a stronger acoustic probe, matched published-baseline reruns,
independent PDCH timing validation, and fresh external/subgroup evaluation
remain scientific limitations. They are not claimed as completed and are not
prerequisites for the bounded conclusions written here. A stronger claim would
require those additional studies.

## Complete manuscript number ledger

The exact display strings below are generated from `output/paper_numbers.json`;
`paper/generated-numbers.tex` supplies the active TeX values. Keys map directly
to `\R{key}` in `paper/main.tex`, with line locations for every occurrence.
Full precision and source SHA-256 hashes are retained in the JSON. The
companion `output/paper_numbers.md` also includes audited values omitted from
the paper. Original neural/seed figures identify their committed provenance;
all other experiment values are refreshed. BibTeX bibliographic numbers and
author ORCIDs are preserved bibliographic metadata, not experiment claims.

| Manuscript key / main.tex lines | Display value | Source file → key | Command / provenance |
|---|---|---|---|
| `ACTDDev` (80) | 0.775 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ACTDTestci_high` (80) | 0.750 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ACTDTestci_low` (80) | 0.458 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ACTDTestpoint` (80) | 0.612 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ACTDThreshold` (80) | 0.50 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ADev` (76) | 0.673 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATDev` (79) | 0.700 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATTestci_high` (79) | 0.709 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATTestci_low` (79) | 0.410 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATTestpoint` (79) | 0.568 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATThreshold` (79) | 0.60 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATestci_high` (76) | 0.681 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATestci_low` (76) | 0.392 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ATestpoint` (76) | 0.545 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `AThreshold` (76) | 0.65 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `CTDDev` (78) | 0.746 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `CTDTestci_high` (78) | 0.771 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `CTDTestci_low` (78) | 0.472 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `CTDTestpoint` (64, 78) | 0.631 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `CTDThreshold` (78) | 0.50 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TCTDDev` (81) | 0.804 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TCTDTestci_high` (81) | 0.806 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TCTDTestci_low` (81) | 0.509 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TCTDTestpoint` (64, 81) | 0.669 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TCTDThreshold` (81) | 0.55 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TDev` (77) | 0.690 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TTestci_high` (77) | 0.769 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TTestci_low` (77) | 0.474 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TTestpoint` (77) | 0.631 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `TThreshold` (77) | 0.5238 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `ablationdevci_high` (129) | +0.083 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.dev.ci_high` | `python src/ctd/ablation_groups.py` |
| `ablationdevci_low` (129) | -0.216 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.dev.ci_low` | `python src/ctd/ablation_groups.py` |
| `ablationdevpoint` (129) | -0.068 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.dev.point` | `python src/ctd/ablation_groups.py` |
| `ablationtestci_high` (129) | +0.054 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.ci_high` | `python src/ctd/ablation_groups.py` |
| `ablationtestci_low` (129) | -0.089 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.ci_low` | `python src/ctd/ablation_groups.py` |
| `ablationtestpoint` (129) | -0.010 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.point` | `python src/ctd/ablation_groups.py` |
| `acousticWeight` (89) | 0.0 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta+ctd].weights.wavlm` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `agarwalTest` (102) | 0.80 | `output/revision_review_context.json` → `SLT06.test_macro_f1` | `python scripts/revision_context.py` |
| `all24Amean` (109, 120) | 0.676 | `output/timing_controls.json` → `results.all24.roc_auc.mean` | `python src/ctd/timing_controls.py` |
| `all24Asd` (120) | 0.031 | `output/timing_controls.json` → `results.all24.roc_auc.sd` | `python src/ctd/timing_controls.py` |
| `all24C` (141) | 0.30 | `output/ctd_ablation_groups.json` → `configs.all24.selected_C` | `python src/ctd/ablation_groups.py` |
| `all24Dev` (141) | 0.746 | `output/ctd_ablation_groups.json` → `configs.all24.dev.macro_f1` | `python src/ctd/ablation_groups.py` |
| `all24Dim` (29, 141) | 24 | `output/ctd_ablation_groups.json` → `configs.all24.n_features` | `python src/ctd/ablation_groups.py` |
| `all24Fmean` (18, 109, 120) | 0.635 | `output/timing_controls.json` → `results.all24.macro_f1.mean` | `python src/ctd/timing_controls.py` |
| `all24Fsd` (120) | 0.026 | `output/timing_controls.json` → `results.all24.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| `all24Test` (141) | 0.631 | `output/ctd_ablation_groups.json` → `configs.all24.test.macro_f1.point` | `python src/ctd/ablation_groups.py` |
| `annotationSeconds` (40) | 1 | `src/ctd/pdch_adapter.py` → `parse_turns MM:SS annotation resolution` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `ask_dci_high` (150) | -0.440 | `output/ctd_interpretability.json` → `coefficients.ask_d.ci_high` | `python src/ctd/interpret.py` |
| `ask_dci_low` (150) | -1.290 | `output/ctd_interpretability.json` → `coefficients.ask_d.ci_low` | `python src/ctd/interpret.py` |
| `ask_dcoef` (150) | -0.867 | `output/ctd_interpretability.json` → `coefficients.ask_d.coef` | `python src/ctd/interpret.py` |
| `ask_onlyC` (145) | 0.01 | `output/ctd_ablation_groups.json` → `configs.ask_only.selected_C` | `python src/ctd/ablation_groups.py` |
| `ask_onlyDev` (145) | 0.594 | `output/ctd_ablation_groups.json` → `configs.ask_only.dev.macro_f1` | `python src/ctd/ablation_groups.py` |
| `ask_onlyDim` (45, 145) | 9 | `output/ctd_ablation_groups.json` → `configs.ask_only.n_features` | `python src/ctd/ablation_groups.py` |
| `ask_onlyTest` (145) | 0.593 | `output/ctd_ablation_groups.json` → `configs.ask_only.test.macro_f1.point` | `python src/ctd/ablation_groups.py` |
| `binaryHits` (29, 153, 158, 163) | 0 | `output/shared_signal_audit.json` → `E2_concordance.concordance.counts.confirmed_shared` | `python src/ctd/shared_signal_audit.py` |
| `bootstrapN` (53) | 2000 | `src/ctd/ml_splits.py` → `N_BOOTSTRAPS` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `cGrid` (49) | 0.01, 0.03, 0.1, 0.3, 1.0 | `src/ctd/pdch_experiment.py` → `C_GRID` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `confidence` (53, 68, 129) | 95 | `src/ctd/ml_splits.py` → `100-ALPHA` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `correctedSession` (38, 91) | 409 | `src/common/daic_cleaning.py` → `KNOWN_ERRORS.relabel` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `crossDim` (45) | 7 | `src/ctd/feature_groups.py` → `len(CROSS)` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `ctdWeight` (89) | 0.7 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].weights.ctd` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `daicLatencyq` (153) | 0.051 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.daic.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` |
| `daicLatencyr` (153) | 0.245 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.daic.per_feature.res_h__amean.r` | `python src/ctd/shared_signal_audit.py` |
| `devN` (38) | 33 | `output/daic_gate_results.json` → `cohort.dev.n` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` |
| `devPositive` (38) | 12 | `output/daic_gate_results.json` → `cohort.dev.n_depressed` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` |
| `excludedN` (38, 91) | 9 | `output/official_split_ctd.json` → `sum(cohort.*.integrity_exclusions_restored)` | `python scripts/official_split_sensitivity.py` |
| `fdr` (60) | 0.10 | `output/shared_signal_audit.json` → `settings.fdr_q` | `python src/ctd/shared_signal_audit.py` |
| `fitC` (49) | 0.3 | `output/daic_gate_results.json` → `session_mean_24d.models.logreg_l2.selected_hyperparam` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` |
| `fusionDelta` (64) | 0.038 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.point - results.single_roberta.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `hamdCutoff` (40) | 17 | `src/ctd/pdch_labels.py` → `primary label cutoff` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `innerFolds` (56, 58) | 4 | `src/ctd/pdch_experiment.py` → `N_SPLITS_INNER` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `latencyAmean` (122) | 0.562 | `output/timing_controls.json` → `results.latency.roc_auc.mean` | `python src/ctd/timing_controls.py` |
| `latencyAsd` (122) | 0.015 | `output/timing_controls.json` → `results.latency.roc_auc.sd` | `python src/ctd/timing_controls.py` |
| `latencyFmean` (122) | 0.531 | `output/timing_controls.json` → `results.latency.macro_f1.mean` | `python src/ctd/timing_controls.py` |
| `latencyFsd` (122) | 0.008 | `output/timing_controls.json` → `results.latency.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| `latency_structureAmean` (124) | 0.576 | `output/timing_controls.json` → `results.latency_structure.roc_auc.mean` | `python src/ctd/timing_controls.py` |
| `latency_structureAsd` (124) | 0.029 | `output/timing_controls.json` → `results.latency_structure.roc_auc.sd` | `python src/ctd/timing_controls.py` |
| `latency_structureFmean` (18, 109, 124) | 0.550 | `output/timing_controls.json` → `results.latency_structure.macro_f1.mean` | `python src/ctd/timing_controls.py` |
| `latency_structureFsd` (124) | 0.025 | `output/timing_controls.json` → `results.latency_structure.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| `liveDim` (18, 29, 40, 45, 153, 158, 163) | 22 | `output/shared_signal_audit.json` → `E2_concordance.concordance.sign_agreement.n_comparable` | `python src/ctd/shared_signal_audit.py` |
| `maxBinaryP` (153) | 0.340 | `output/shared_signal_audit.json` → `E2_concordance.pdch.max_abs_r_permutation_p` | `python src/ctd/shared_signal_audit.py` |
| `meanThreshold` (82) | 0.55 | `output/mean_prob_seed44.json` → `mean_prob[roberta+ctd].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `meandev` (64, 82) | 0.738 | `output/mean_prob_seed44.json` → `mean_prob[roberta+ctd].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `meantest` (64, 82) | 0.650 | `output/mean_prob_seed44.json` → `mean_prob[roberta+ctd].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `movingLatencyq` (155) | 0.520 | `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` |
| `movingLatencyr` (155) | 0.107 | `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.per_feature.res_h__amean.r` | `python src/ctd/shared_signal_audit.py` |
| `nestedN` (56) | 135 | `output/timing_controls.json` → `n_subjects` | `python src/ctd/timing_controls.py` |
| `nestedRepeats` (56, 113) | 10 | `output/timing_controls.json` → `results.all24.n_repeats` | `python src/ctd/timing_controls.py` |
| `noPairsN` (91) | 3 | `output/official_split_ctd.json` → `sum(length(sessions_without_turn_pairs.*))` | `python scripts/official_split_sensitivity.py` |
| `no_askAmean` (109, 121) | 0.610 | `output/timing_controls.json` → `results.no_ask.roc_auc.mean` | `python src/ctd/timing_controls.py` |
| `no_askAsd` (121) | 0.030 | `output/timing_controls.json` → `results.no_ask.roc_auc.sd` | `python src/ctd/timing_controls.py` |
| `no_askC` (142) | 1.00 | `output/ctd_ablation_groups.json` → `configs.no_ask.selected_C` | `python src/ctd/ablation_groups.py` |
| `no_askDev` (129, 142) | 0.814 | `output/ctd_ablation_groups.json` → `configs.no_ask.dev.macro_f1` | `python src/ctd/ablation_groups.py` |
| `no_askDim` (142) | 15 | `output/ctd_ablation_groups.json` → `configs.no_ask.n_features` | `python src/ctd/ablation_groups.py` |
| `no_askFmean` (18, 109, 121) | 0.566 | `output/timing_controls.json` → `results.no_ask.macro_f1.mean` | `python src/ctd/timing_controls.py` |
| `no_askFsd` (121) | 0.025 | `output/timing_controls.json` → `results.no_ask.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| `no_askTest` (129, 142) | 0.641 | `output/ctd_ablation_groups.json` → `configs.no_ask.test.macro_f1.point` | `python src/ctd/ablation_groups.py` |
| `no_crossC` (143) | 0.10 | `output/ctd_ablation_groups.json` → `configs.no_cross.selected_C` | `python src/ctd/ablation_groups.py` |
| `no_crossDev` (143) | 0.700 | `output/ctd_ablation_groups.json` → `configs.no_cross.dev.macro_f1` | `python src/ctd/ablation_groups.py` |
| `no_crossDim` (143) | 17 | `output/ctd_ablation_groups.json` → `configs.no_cross.n_features` | `python src/ctd/ablation_groups.py` |
| `no_crossTest` (143) | 0.527 | `output/ctd_ablation_groups.json` → `configs.no_cross.test.macro_f1.point` | `python src/ctd/ablation_groups.py` |
| `officialAuc` (91) | 0.654 | `output/official_split_ctd.json` → `result.test.roc_auc.point` | `python scripts/official_split_sensitivity.py` |
| `officialC` (91) | 1.0 | `output/official_split_ctd.json` → `result.selected_C` | `python scripts/official_split_sensitivity.py` |
| `officialDev` (91) | 0.782 | `output/official_split_ctd.json` → `result.dev.macro_f1` | `python scripts/official_split_sensitivity.py` |
| `officialN` (38) | 189 | `output/official_split_ctd.json` → `sum(cohort.*.n)` | `python scripts/official_split_sensitivity.py` |
| `officialTest` (91, 103) | 0.639 | `output/official_split_ctd.json` → `result.test.macro_f1.point` | `python scripts/official_split_sensitivity.py` |
| `officialdevN` (91, 102, 103) | 35 | `output/official_split_ctd.json` → `cohort.dev.n` | `python scripts/official_split_sensitivity.py` |
| `officialtestN` (91, 102, 103) | 47 | `output/official_split_ctd.json` → `cohort.test.n` | `python scripts/official_split_sensitivity.py` |
| `officialtrainN` (91, 102, 103) | 107 | `output/official_split_ctd.json` → `cohort.train.n` | `python scripts/official_split_sensitivity.py` |
| `outerFolds` (56, 58) | 5 | `src/ctd/pdch_experiment.py` → `N_SPLITS_OUTER` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `pdchAmean` (18, 29, 153, 158, 163) | 0.476 | `output/pdch_ctd_results.json` → `classification.all24.roc_auc.mean` | `python src/ctd/pdch_experiment.py` |
| `pdchAsd` (153) | 0.066 | `output/pdch_ctd_results.json` → `classification.all24.roc_auc.sd` | `python src/ctd/pdch_experiment.py` |
| `pdchFmean` (153) | 0.520 | `output/pdch_ctd_results.json` → `classification.all24.macro_f1.mean` | `python src/ctd/pdch_experiment.py` |
| `pdchFsd` (153) | 0.055 | `output/pdch_ctd_results.json` → `classification.all24.macro_f1.sd` | `python src/ctd/pdch_experiment.py` |
| `pdchLatencyq` (153) | 0.431 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.pdch.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` |
| `pdchLatencyr` (153) | 0.267 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.pdch.per_feature.res_h__amean.r` | `python src/ctd/shared_signal_audit.py` |
| `pdchN` (40) | 62 | `output/pdch_ctd_results.json` → `labels.n_sessions` | `python src/ctd/pdch_experiment.py` |
| `pdchPositive` (40) | 27 | `output/pdch_ctd_results.json` → `labels.n_positive` | `python src/ctd/pdch_experiment.py` |
| `pdchRepeated` (40) | 16 | `output/pdch_ctd_results.json` → `labels.n_subjects_with_two_sessions` | `python src/ctd/pdch_experiment.py` |
| `pdchRepeats` (58) | 20 | `output/pdch_ctd_results.json` → `classification.all24.n_repeats` | `python src/ctd/pdch_experiment.py` |
| `pdchSubjects` (40) | 46 | `output/pdch_ctd_results.json` → `labels.n_subjects` | `python src/ctd/pdch_experiment.py` |
| `permutations` (60) | 5000 | `output/shared_signal_audit.json` → `settings.n_permutations` | `python src/ctd/shared_signal_audit.py` |
| `phqCutoff` (38) | 10 | `src/common/daic_cleaning.py` → `PHQ8_DEPRESSED_THRESHOLD` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `rankLive` (45) | 13 | `output/shared_signal_audit.json` → `E0_inventory.algebraic_redundancy.daic.ask.matrix_rank_22_live_centered` | `python src/ctd/shared_signal_audit.py` |
| `res_hci_high` (150) | +0.704 | `output/ctd_interpretability.json` → `coefficients.res_h.ci_high` | `python src/ctd/interpret.py` |
| `res_hci_low` (150) | -0.322 | `output/ctd_interpretability.json` → `coefficients.res_h.ci_low` | `python src/ctd/interpret.py` |
| `res_hcoef` (150) | +0.210 | `output/ctd_interpretability.json` → `coefficients.res_h.coef` | `python src/ctd/interpret.py` |
| `res_onlyC` (144) | 1.00 | `output/ctd_ablation_groups.json` → `configs.res_only.selected_C` | `python src/ctd/ablation_groups.py` |
| `res_onlyDev` (144) | 0.607 | `output/ctd_ablation_groups.json` → `configs.res_only.dev.macro_f1` | `python src/ctd/ablation_groups.py` |
| `res_onlyDim` (45, 144) | 8 | `output/ctd_ablation_groups.json` → `configs.res_only.n_features` | `python src/ctd/ablation_groups.py` |
| `res_onlyTest` (144) | 0.517 | `output/ctd_ablation_groups.json` → `configs.res_only.test.macro_f1.point` | `python src/ctd/ablation_groups.py` |
| `robertaSeedmean` (64) | 0.604 | `output/revision_review_context.json` → `SLT08.roberta_test_seed_mean` | `python scripts/revision_context.py` |
| `robertaSeeds` (64) | 3 | `output/revision_review_context.json` → `SLT08.roberta_n_seeds` | `python scripts/revision_context.py` |
| `robertaSeedsd` (64) | 0.025 | `output/revision_review_context.json` → `SLT08.roberta_test_seed_sd` | `python scripts/revision_context.py` |
| `robertaWeight` (89) | 0.3 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].weights.roberta` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` |
| `signAgree` (18, 29, 153, 158, 163) | 12 | `output/shared_signal_audit.json` → `E2_concordance.concordance.sign_agreement.n_agree` | `python src/ctd/shared_signal_audit.py` |
| `silenceSeconds` (43) | 0.2 | `output/shared_signal_audit.json` → `settings.silence_threshold_s` | `python src/ctd/shared_signal_audit.py` |
| `structureAmean` (123) | 0.535 | `output/timing_controls.json` → `results.structure.roc_auc.mean` | `python src/ctd/timing_controls.py` |
| `structureAsd` (123) | 0.027 | `output/timing_controls.json` → `results.structure.roc_auc.sd` | `python src/ctd/timing_controls.py` |
| `structureFmean` (123) | 0.499 | `output/timing_controls.json` → `results.structure.macro_f1.mean` | `python src/ctd/timing_controls.py` |
| `structureFsd` (123) | 0.020 | `output/timing_controls.json` → `results.structure.macro_f1.sd` | `python src/ctd/timing_controls.py` |
| `testN` (38) | 45 | `output/daic_gate_results.json` → `cohort.test.n` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` |
| `testPositive` (38) | 14 | `output/daic_gate_results.json` → `cohort.test.n_depressed` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` |
| `threshold` (49, 56) | 0.5 | `src/ctd/ml_splits.py` → `LogisticRegression.predict decision rule` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `thresholdHigh` (53) | 0.95 | `src/fusion/run_fusion_wavlm_seeds.py` → `threshold grid stop` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `thresholdLow` (53) | 0.05 | `src/fusion/run_fusion_wavlm_seeds.py` → `threshold grid start` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `thresholdStep` (53) | 0.05 | `src/fusion/run_fusion_wavlm_seeds.py` → `threshold grid step` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
| `trainN` (38) | 102 | `output/daic_gate_results.json` → `cohort.train.n` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` |
| `trainPositive` (38) | 29 | `output/daic_gate_results.json` → `cohort.train.n_depressed` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` |
| `unitSum` (45) | 1 | `output/shared_signal_audit.json` → `E0_inventory.algebraic_redundancy.daic.ask.identity` | `python src/ctd/shared_signal_audit.py` |
| `wavlmSeed` (51, 76) | 44 | `output/revision_review_context.json` → `SLT08.wavlm_selected_seed` | `python scripts/revision_context.py` |
| `wavlmSeedmean` (84) | 0.506 | `output/revision_review_context.json` → `SLT08.wavlm_test_seed_mean` | `python scripts/revision_context.py` |
| `wavlmSeeds` (51, 84) | 6 | `output/revision_review_context.json` → `SLT08.wavlm_n_seeds` | `python scripts/revision_context.py` |
| `wavlmSeedsd` (84) | 0.034 | `output/revision_review_context.json` → `SLT08.wavlm_test_seed_sd` | `python scripts/revision_context.py` |
| `weightStep` (53) | 0.1 | `src/fusion/run_fusion_wavlm_seeds.py` → `weight grid` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` |
