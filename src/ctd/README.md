# CTD — Conversational Temporal Dynamics (DAIC-WOZ)

The timing modality of the paper: 24 dyadic Ask/Res turn-pair timing
descriptors (adopted a priori from prior turn-taking work, unchanged) turned
into a per-session depression detector.

## Pipeline

```
transcript -> Ask/Res turn pairing (turn_pairing.py)
           -> 24 per-turn-pair timing features (feature_extraction.py, constants.py)
           -> NaN-aware per-session mean (24-D vector)
           -> train-median imputation + StandardScaler (train-fit)
           -> L2 logistic regression, C=0.3, class_weight='balanced' (ml_splits.py)
```

Dev probabilities come from a train-only fit; test probabilities from a
train+dev refit (standard deployed-system step). Decision threshold 0.5.

## Run

```bash
export DAIC_WOZ_ROOT=/path/to/DAIC-WOZ
pip install -r requirements.txt confidence_intervals
python ml_splits.py     # -> outputs/ml_splits_results.json (+ console tables)
```

`ml_splits.py` evaluates both:

- the canonical **24-D session-mean** detector (dev macro-F1 0.746 / test 0.631), and
- the paper's post-hoc robustness variant — 24 features × 10 **eGeMAPS
  functionals** = 240-D (`functionals.py`), selecting among LogReg/SVM/RF on
  dev balanced accuracy. It did not beat the 24-D session-mean model on dev,
  so the simpler pre-committed detector is the deployed one.

## The 24 CTD features

Defined in `constants.py` (single source of truth). For each Ask/Res turn pair:
`d` = turn duration, `u` = voiced time, `s = d − u` = within-turn silence.

- **Durations (2):** `ask_d`, `res_d`
- **Diff/sum (3):** `res_minus_ask`, `ask_minus_res`, `duration_sum`
- **Duration ratios (2):** `res_over_ask`, `ask_over_res`
- **Ask voiced/silence ratios (6):** `ask_ud`, `ask_du`, `ask_sd`, `ask_ds`, `ask_su`, `ask_us`
- **Res voiced/silence ratios (6):** `res_ud`, `res_du`, `res_sd`, `res_ds`, `res_su`, `res_us`
- **Hesitation (1):** `res_h` (Ask-end → Res-start gap)
- **Counts (4):** `ask_bt`, `res_bt` (overlap/backchannel), `ask_st`, `res_st` (silence)

Undefined ratios (e.g. zero-silence turns) are NaN and imputed with
train-split medians (leakage-safe). Interviewer turns are used **only** as
timing anchors; their lexical content never enters any detector.

## Files

| File | Role |
|---|---|
| `constants.py` | Feature names, dataset root (`DAIC_WOZ_ROOT`), output dir |
| `data_loading.py` | Label loading + shared cleaning (via `src/common/`) |
| `turn_pairing.py` | Ask/Res speaker-run pairing from transcripts |
| `feature_extraction.py` | 24 per-turn-pair timing features |
| `functionals.py` | 10-operator eGeMAPS functional bank (robustness variant) |
| `ml_splits.py` | Canonical train/dev/test protocol + both detectors |
| `bags.py` | Per-session bags of raw per-turn features (used by `src/fusion/extract_ctd_roberta.py`) |
| `feature_groups.py` | Ask/Cross/Res grouping of the 24 features; named ablation configs (`all24`, `no_ask`, `res_only`, `ask_only`, `no_cross`) |
| `interpret.py` | SLT-04: bootstrap coefficients, dev permutation importance, per-split univariate direction, collinearity diagnostic |
| `ablation_groups.py` | SLT-05: ask-side ablation through the identical deployed protocol, paired-bootstrap verdict |
| `resampling.py` | SLT-10: variability via repeated data resampling (not seeds — the fit is deterministic) |
| `pdch_adapter.py` | MC-01/MC-05/MC-06: PDCH backend — chunk stitching, VAD refinement, `list[Utterance]` |
| `pdch_labels.py` | MC-07: HAMD-17 ≥ 17 label + severity target (never pooled with PHQ-8) |
| `pdch_features.py` | PDCH session-level CTD features via the unmodified DAIC-WOZ feature code |
| `pdch_experiment.py` | MC-08: subject-grouped nested CV over the five tiers + the pre-registered P1–P6 check |
| `shared_signal_audit.py` | SLT-01/SLT-03: cross-corpus shared-signal audit (E0–E4) — distribution shift, confound matching and within-fold residualization, the univariate concordance map, candidate detectors, and continuous/item-level targets |

Outputs are written to `outputs/` (git-ignored; regenerable) for cached
intermediates, and to `../../output/` for the paper-facing
`ctd_interpretability.*`, `ctd_ablation_groups.*`, `ctd_resampling.*` results.

Run after the reproduction gate:

```bash
python interpret.py          # -> ../../output/ctd_interpretability.{json,md}, ctd_coef_forest.png
python ablation_groups.py    # -> ../../output/ctd_ablation_groups.{json,md}
python resampling.py         # -> ../../output/ctd_resampling.{json,md}
python pdch_experiment.py    # -> ../../output/pdch_ctd_results.{json,md}   (needs scripts/build_pdch_cache.py first)
python shared_signal_audit.py  # -> ../../output/shared_signal_audit.{json,md}
```

`shared_signal_audit.py` runs the family pre-declared in
`docs/exploratory-shared-signal-pdch-daic.md`; its conclusions are in
`docs/shared-signal-audit.md`. It reads DAIC-WOZ's cached `outputs/functionals_*.csv`,
so run `ml_splits.py` first.
