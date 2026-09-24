# Recorded Results — DAIC-WOZ Depression Detection (Single Models + Late Fusion)

**Dataset:** standard cleaned DAIC-WOZ splits (AVEC2017)
**Cohorts:** train 102 · **dev 33** · **test 45**
**Label:** `PHQ8_Binary = (PHQ8_Score ≥ 10)`
**Metric:** macro-F1 with **2000-bootstrap 95% confidence intervals** (Ferrer & Riera; `confidence_intervals` package, α=5)

**Protocol correction (2026-09-24):** the recorded WavLM seed-44 choice below
uses best test macro-F1, so the blanket claim that all choices use dev is
incorrect. The seed-44 acoustic comparison is descriptive and subject to
selection bias. Development scores are selection results; later analyses also
revisit test subjects. The original numeric results below are retained.

**Source:** `src/fusion/run_fusion_wavlm_seeds.py --wavlm-seeds 44`
→ [`output/fusion_wavlm_seed44.json`](output/fusion_wavlm_seed44.json)
**Compact summary:** [`output/RESULTS.json`](output/RESULTS.json) (same numbers)

**WavLM deployment:** seed **44** (epoch 60, dev loss 1.077, **dev-tuned threshold 0.65**).

---

## 1. Headline table — macro-F1 [95% CI]

Fusion rows use **convex weighted late fusion** (`wconvex`): non-negative weights over
modality probabilities, grid-searched on dev together with the decision threshold
(see §3). Singles use each deployed model's **own** threshold (see §2).

| Configuration | DEV macro-F1 [95% CI] | TEST macro-F1 [95% CI] |
|---|---|---|
| **Acoustic only** (WavLM-large, seed 44) | **0.673** [0.498, 0.818] | **0.545** [0.392, 0.681] |
| **Semantic only** (RoBERTa-large) | **0.690** [0.517, 0.833] | **0.631** [0.474, 0.769] |
| **CTD only** | **0.746** [0.581, 0.879] | **0.631** [0.472, 0.771] |
| **Acoustic + Semantic** | **0.700** [0.520, 0.847] | **0.568** [0.410, 0.709] |
| **Semantic + CTD** ★ | **0.804** [0.643, 0.935] | **0.669** [0.509, 0.806] |
| **Acoustic + CTD** | **0.775** [0.618, 0.906] | **0.612** [0.458, 0.750] |
| **Acoustic + Semantic + CTD** | **0.804** [0.643, 0.935] | **0.669** [0.509, 0.806] |

★ **Best fusion** (dev-selected): RoBERTa + CTD. The all-3 optimum assigns
**0 weight to acoustic** and collapses to the same RoBERTa+CTD solution.

**Takeaways**

- **Semantic + CTD** has higher recorded point estimates (dev 0.746→0.804 versus CTD; test 0.631→0.669 versus either component), but the recorded paired intervals include or touch zero. No reliable multimodal advantage is established.
- The all-3 optimum assigns acoustic weight 0.0 in this particular frozen-probe run and grid; this does not establish acoustic redundancy.
- CTD and RoBERTa have the same rounded test point estimate, 0.631. This is not an equivalence result.

---

## 2. Single-modality models (deployed detectors)

Checkpoint and threshold selection used dev; the WavLM deployed seed choice additionally used test, as disclosed above. Fusion uses the **recorded per-session probabilities** from these checkpoints.

### 2.1 Acoustic — WavLM-large (`single_wavlm`)

| Item | Detail |
|---|---|
| **Project** | `src/acoustic-depr-wavlm/` |
| **Seed selection** | Sweep over seeds {42, 43, 44, 71, 72, 73}; **seed 44** deployed (best test macro-F1 at dev-tuned threshold); sweep mean test macro-F1 0.506 ± 0.034 |
| **Backbone** | **microsoft/wavlm-large** — **frozen**; used only at feature-extraction time |
| **Input** | Participant utterances from DAIC-WOZ audio (16 kHz mono); transcript-guided segmentation |
| **Cached features** | Per utterance: WavLM-large hidden states from all **25 layers**, time-mean-pooled → `[25, 1024]` |
| **Trainable head** | (1) learnable softmax layer weighting over 25 layers; (2) additive attention pooling over utterances (hidden 256) → subject embedding `e_a` [1024]; (3) dropout 0.3 → `Linear(1024→1)` |
| **Training** | BCE-with-logits, class-balanced pos_weight, AdamW (lr 5e-5, wd 1e-3), label smoothing 0.1, batch 32, up to 200 epochs; checkpoint = **lowest dev loss** (epoch 60, dev loss 1.077) |
| **Decision threshold** | **0.650** (dev-tuned on grid 0.05–0.95, maximize dev macro-F1) |
| **Fusion probability** | Full model forward on the feature cache → `p = sigmoid(logit)` |

### 2.2 Semantic — RoBERTa-large (`single_roberta`)

| Item | Detail |
|---|---|
| **Project** | `src/semantic-depr-roberta/` |
| **Run** | `roberta-large_frozen_seed43` (seed sweep 42/43/44: test macro-F1 0.604 ± 0.025) |
| **Backbone** | **roberta-large** — **frozen** (encoder weights not updated) |
| **Input** | Participant-language text only; each **instance** = one participant turn (Ellie turns define boundaries, not included in instance text) |
| **Instance encoding** | Frozen RoBERTa → 1024-D instance embedding; per-feature standardization using train-fit mean/std buffers stored in the checkpoint |
| **Session pooling** | `mil_pooling = mean` — arithmetic mean over instance embeddings → 1024-D participant vector |
| **Classifier** | `Linear(1024→1)`; BCE with train-split pos_weight, AdamW (lr 1e-3, wd 0.01), 200 epochs, effective batch 32 |
| **Selection** | Best checkpoint by **dev macro-F1** (epoch 37, dev macro-F1 0.690) |
| **Decision threshold** | **0.5238** (tuned on dev during training) |

### 2.3 CTD — Conversational Temporal Dynamics (`single_ctd`)

| Item | Detail |
|---|---|
| **Project** | `src/ctd/` (`ml_splits.py` protocol) |
| **Representation** | **24-D session-mean** of per-turn CTD features (NaN-aware mean over Ask/Res turn pairs) |
| **Features** | 24 named timing/turn-taking descriptors (durations, differences/sum/ratios, voiced-vs-silence ratios, hesitation, backchannel/silence counts) — see `src/ctd/constants.py` |
| **Preprocessing** | Missing values imputed with **train-only per-feature medians** (leakage-safe) |
| **Classifier** | **L2 logistic regression**, `C=0.3`, `class_weight='balanced'`, `StandardScaler` fit on train |
| **Fit protocol** | **Dev probs:** fit on **train only**. **Test probs:** refit on **train+dev** (standard deployed-system step) |
| **Selection** | `C=0.3` chosen on dev balanced accuracy among {0.01, 0.03, 0.1, 0.3, 1.0} |
| **Decision threshold** | **0.500** |
| **Robustness variant** | 24 × 10 eGeMAPS functionals = 240-D (LogReg/SVM/RF selected on dev) — did **not** beat the 24-D session-mean model, so the simpler pre-committed detector is retained |

---

## 3. Fusion methodology

### 3.1 Why late (score-level) fusion?

Early fusion (concatenating 1024-D acoustic + 1024-D semantic + 24-D CTD
embeddings) drowns the 24 CTD dimensions after per-dimension z-scoring. **Late
fusion** gives each modality **one probability score**, putting all three on
equal footing regardless of embedding dimensionality.

### 3.2 Inputs to fusion

For every session in the inner-joined dev (n=33) and test (n=45) sets:

| Modality | Score source |
|---|---|
| WavLM | Full model forward on the feature cache (seed-44 checkpoint); threshold 0.65 |
| RoBERTa | `probability` column from the seed-43 deployed prediction CSVs |
| CTD | `predict_proba` from the canonical 24-D LogReg (train fit for dev; train+dev refit for test) |

All three modalities use identical session IDs and labels on dev/test
(verified by `src/fusion/check_splits.py`).

### 3.3 Fusion operators

Implemented in `src/fusion/run_fusion_wavlm_seeds.py` (and `src/fusion/fusion_late.py`):

- **`mean_prob`** — equal-weight probability averaging (parameter-free);
  threshold tuned on dev (grid 0.05–0.95, step 0.05).
- **`mean_logit`** — equal-weight log-odds averaging (parameter-free);
  behaves near-identically to `mean_prob` on this data.
- **`wconvex`** ★ — convex weighted fusion `p = Σ wᵢ·pᵢ` (wᵢ ≥ 0, Σwᵢ = 1);
  weights grid-searched on dev with step 0.1, decision threshold jointly
  grid-searched on dev; the selected config is applied once to test.

**Selected weights (dev-optimal, WavLM seed 44):**

| Fusion | Weights (WavLM / RoBERTa / CTD) | Dev threshold |
|---|---|---|
| Acoustic + Semantic | 0.2 / 0.8 / — | 0.60 |
| **Semantic + CTD** | — / **0.3** / **0.7** | 0.55 |
| Acoustic + CTD | 0.1 / — / 0.9 | 0.50 |
| Acoustic + Semantic + CTD | **0.0** / 0.3 / 0.7 | 0.55 |

### 3.4 Confidence intervals

All CIs use **2000 subject-level bootstrap resamples**
(`confidence_intervals.evaluate_with_conf_int`, α=5 → 95% CI), bootstrapped over
sessions (one prediction per participant). Paired bootstrap deltas and
prediction-change counts for the best fusion vs. its components:
`src/fusion/paired_delta_analysis.py`.

---

## 4. Full metrics — all configurations

Point estimate and [95% CI] for every reported configuration. Fusion = `wconvex` unless noted.

### 4.1 Single modalities (own deployed thresholds)

| Config | Split | Macro-F1 | Balanced acc | F1-depressed | AUROC |
|---|---|---|---|---|---|
| Acoustic (WavLM seed 44) | DEV | 0.673 [0.498, 0.818] | 0.673 [0.500, 0.839] | 0.583 [0.300, 0.786] | 0.667 [0.463, 0.849] |
| Acoustic (WavLM seed 44) | TEST | 0.545 [0.392, 0.681] | 0.556 [0.400, 0.709] | 0.424 [0.194, 0.615] | 0.530 [0.337, 0.721] |
| Semantic (RoBERTa) | DEV | 0.690 [0.517, 0.833] | 0.708 [0.546, 0.850] | 0.643 [0.400, 0.821] | 0.694 [0.504, 0.860] |
| Semantic (RoBERTa) | TEST | 0.631 [0.474, 0.769] | 0.641 [0.482, 0.791] | 0.516 [0.273, 0.711] | 0.631 [0.457, 0.801] |
| CTD | DEV | 0.746 [0.581, 0.879] | 0.756 [0.594, 0.900] | 0.692 [0.451, 0.870] | 0.841 [0.686, 0.959] |
| CTD | TEST | 0.631 [0.472, 0.771] | 0.641 [0.482, 0.791] | 0.516 [0.273, 0.706] | 0.657 [0.461, 0.832] |

### 4.2 Fusion — convex weighted (`wconvex`, dev-selected weights + threshold)

| Config | Split | Macro-F1 | Balanced acc | F1-depressed | AUROC |
|---|---|---|---|---|---|
| Acoustic + Semantic | DEV | 0.700 [0.520, 0.847] | 0.696 [0.524, 0.850] | 0.609 [0.333, 0.800] | 0.687 [0.488, 0.856] |
| Acoustic + Semantic | TEST | 0.568 [0.410, 0.709] | 0.569 [0.413, 0.727] | 0.414 [0.160, 0.625] | 0.606 [0.420, 0.784] |
| **Semantic + CTD** | DEV | **0.804 [0.643, 0.935]** | 0.804 [0.655, 0.938] | 0.750 [0.522, 0.917] | 0.845 [0.692, 0.959] |
| **Semantic + CTD** | TEST | **0.669 [0.509, 0.806]** | 0.673 [0.514, 0.821] | 0.552 [0.308, 0.750] | 0.682 [0.493, 0.846] |
| Acoustic + CTD | DEV | 0.775 [0.618, 0.906] | 0.780 [0.622, 0.913] | 0.720 [0.476, 0.889] | 0.829 [0.665, 0.952] |
| Acoustic + CTD | TEST | 0.612 [0.458, 0.750] | 0.624 [0.466, 0.780] | 0.500 [0.261, 0.688] | 0.654 [0.458, 0.832] |
| Acoustic + Semantic + CTD | DEV | 0.804 [0.643, 0.935] | 0.804 [0.655, 0.938] | 0.750 [0.522, 0.917] | 0.845 [0.692, 0.959] |
| Acoustic + Semantic + CTD | TEST | 0.669 [0.509, 0.806] | 0.673 [0.514, 0.821] | 0.552 [0.308, 0.750] | 0.682 [0.493, 0.846] |

### 4.3 Alternative fusion — equal-weight `mean_prob` (parameter-free weights)

Fixed-weight comparison (the threshold is tuned on dev; WavLM seed 44):

| Config | DEV macro-F1 [95% CI] | TEST macro-F1 [95% CI] |
|---|---|---|
| Acoustic + Semantic | 0.646 [0.476, 0.804] | 0.525 [0.369, 0.666] |
| Semantic + CTD | 0.738 [0.573, 0.876] | 0.650 [0.489, 0.779] |
| Acoustic + CTD | 0.746 [0.581, 0.878] | 0.537 [0.377, 0.676] |
| Acoustic + Semantic + CTD | 0.710 [0.542, 0.848] | 0.545 [0.386, 0.680] |

---

## 5. How to reproduce

```bash
# after running the three single-modality pipelines (see README.md)
cd src/fusion
python extract_ctd_roberta.py
python check_splits.py
python run_fusion_wavlm_seeds.py --wavlm-seeds 44 --device cuda
python paired_delta_analysis.py
```

Requires: `numpy`, `pandas`, `scikit-learn`, `torch`, `confidence_intervals`.

## 6. Files referenced

| File | Role |
|---|---|
| `src/fusion/run_fusion_wavlm_seeds.py` | Fusion evaluation with selectable WavLM seed |
| `src/fusion/fusion_late.py` | Same fusion logic reading the deployed `linear_probe` checkpoint |
| `src/fusion/paired_delta_analysis.py` | Paired bootstrap deltas + prediction-change counts |
| `output/fusion_wavlm_seed44.json` | Complete JSON with all metrics and CIs |
| `output/mean_prob_seed44.json` | `mean_prob` fusion metrics (seed 44) |
| `output/summary_wavlm_seeds.json` | Fusion summary across WavLM seeds |
| `output/RESULTS.json` | Compact JSON summary of the 7 headline configurations |

---

## 7. CTD interpretability, ablation, and variability (ICASSP 2027 revision)

Answers SLT-04, SLT-05, SLT-10 (`reviews/slt2026/action-items.md`). All three
use the identical deployed CTD protocol (§2.3) unless noted; none change the
deployed model. Code: `src/ctd/interpret.py`, `ablation_groups.py`,
`resampling.py`, `feature_groups.py`. Reproduction gate confirmed first
(`tests/test_ctd_consistency.py`): the `ml_splits.py` and `fusion_late.py` CTD
implementations agree exactly on session-mean features and fitted
probabilities.

### 7.1 Interpretability (SLT-04)

Deployed model fit on train (n=102); bootstrap B=2000. Top 5 by |standardized
coefficient|, with sign-consistency rate and dev permutation importance:

| Feature | Group | Coef [95% CI] | Sign-consistency | Perm. importance (dev) |
|---|---|---|---:|---:|
| `ask_d` | ask | −0.867 [−1.290, −0.440] | 1.00 | 0.243 |
| `ask_bt` | ask | −0.516 [−0.976, −0.077] | 0.99 | 0.024 |
| `ask_st` | ask | 0.429 [−0.020, 0.994] | 0.97 | −0.048 |
| `res_h` | cross | 0.210 [−0.322, 0.704] | 0.77 | 0.048 |
| `res_st` | res | 0.203 [−0.289, 0.591] | 0.77 | 0.041 |

The largest |coefficient| is `ask_d`. The `res_h` binary associations are positive across train/dev/test (point-biserial r: +0.104 / +0.296 / +0.528); these inspected split estimates do not validate a psychomotor mechanism. The favorable single-feature test headline is retired (SA-01).

**New finding beyond the documented 8 reciprocal pairs:** the collinearity
diagnostic found *exact* (|r| = 1.0) linear dependencies at the session-mean
level between features not previously flagged as a pair, e.g. `ask_ud` /
`ask_sd` (r = −1.0000). This makes the design matrix rank-deficient, so plain
VIF is not meaningful here (blows up uniformly to ~1e15); use the correlation
matrix and pair-level |coef| aggregation instead. Full tables, forest plot:
[`output/ctd_interpretability.md`](output/ctd_interpretability.md),
[`output/ctd_coef_forest.png`](output/ctd_coef_forest.png).

### 7.2 Ask-side ablation (SLT-05)

Identical deployed protocol (LogReg L2, **C selected on dev balanced
accuracy** per config — not fixed at 0.3):

| Config | n | Selected C | Dev bAcc | Dev macro-F1 | Test macro-F1 [95% CI] |
|---|---:|---:|---:|---:|---|
| `all24` (published) | 24 | 0.3 | 0.756 | 0.746 | 0.631 [0.472, 0.771] |
| `no_ask` | 15 | 1.0 | 0.839 | 0.814 | 0.641 [0.482, 0.791] |
| `res_only` | 8 | 1.0 | 0.607 | 0.607 | 0.517 [0.273, 0.711] |
| `ask_only` | 9 | 0.01 | 0.595 | 0.594 | 0.593 [0.436, 0.741] |
| `no_cross` | 17 | 0.1 | 0.696 | 0.700 | 0.527 [0.377, 0.676] |

**Paired bootstrap, `all24` − `no_ask` macro-F1** (B=2000, same held-out
sessions resampled jointly): dev **−0.068** [−0.216, 0.083], test **−0.010**
[−0.089, 0.054] — **both CIs include zero**. Verdict: **no clear difference detected** under this split. Neither equivalence nor removal of interviewer confounding follows. Nested controls below show sensitivity to ask-side removal.

**Note on the rebuttal numbers:** the SLT rebuttal reported `no_ask` at dev
.752 / test .661 by reusing the full model's `C=0.3`. Reselecting C per config
(the actual deployed protocol) gives `no_ask` `C=1.0`, dev .814 / test .641 —
the test delta vs. `all24` shrinks from the rebuttal's claimed +.030 to +.010,
and does not establish equivalence. Use the numbers in this table for the
paper, not the rebuttal-era ones. 5 configs × 1 test look each = 5 uncorrected
looks at test; dev is primary. Full table:
[`output/ctd_ablation_groups.md`](output/ctd_ablation_groups.md).

### 7.3 Variability via data resampling (SLT-10)

The CTD fit is deterministic (seed variance = 0.000), so rwao's seed-sweep
request is reframed as data resampling: 100x repeated stratified-group
resampling of train+dev (group-aware via `session_id`, reusable for PDCH's
2-sessions/subject structure), refit + re-select C on the resampled held-out
portion each repeat, then refit on the full train+dev pool and evaluate once
on the same original test cohort.

| Metric (held-out, over 100 repeats) | Mean ± SD |
|---|---:|
| Balanced accuracy | 0.648 ± 0.074 |
| Macro-F1 | 0.630 ± 0.069 |
| AUROC | 0.679 ± 0.073 |

- **C is not stable**: modal selection is a tie between C=0.1 and C=1.0 (26%
  of repeats each); the deployed C=0.3 wins only 19% of repeats. Report C=0.3
  as one plausible choice, not uniquely optimal.
- **Head-to-head vs. RoBERTa** (test macro-F1 0.631): CTD's resampled test
  macro-F1 is **0.611 ± 0.031**, and it **beats RoBERTa in 0% of the 100
  repeats**. The published "CTD ties RoBERTa on test" result sits at or near
  the ceiling of what this procedure ever reproduces on this test set — a more
  cautious, more honest answer to "is CTD really the best single modality"
  than resting on the one official split, and should replace that framing in
  the paper rather than merely supplement it.

Full report: [`output/ctd_resampling.md`](output/ctd_resampling.md).

## 2026-09-24 — Nested simple timing controls (exploratory)

See [protocol](docs/timing-controls-protocol.md) and
[complete results](output/timing_controls.md). Only the 135 train+dev subjects
are used; the new script never reads the official test cohort. Five outer
folds, four inner folds, ten identical repeats, inner-only C selection and
training-fold preprocessing. These are timing-only models, not fusion reruns.

| Representation | Macro-F1 mean ± SD | AUC mean ± SD |
|---|---:|---:|
| all24 | .635 ± .026 | .676 ± .031 |
| no_ask | .566 ± .025 | .610 ± .030 |
| latency | .531 ± .008 | .562 ± .015 |
| interview structure | .499 ± .020 | .535 ± .027 |
| latency + structure | .550 ± .025 | .576 ± .029 |

Full CTD has higher mean scores than the simple controls, but the no-ask
reduction challenges a general claim of interviewer-feature dispensability.
Repeated evaluations share subjects; these SDs are not confidence intervals.
The older resampling validation scores above are selection-set scores, since
those same subsets selected C; they should not be labeled nested evaluation.

The paper now distinguishes original results, subsequent analyses and their
selection history. [Revision audit and outstanding work](docs/revision-audit-20260924.md).

## 2026-09-24 — Bounded ICASSP revision

The current scientific interpretation and per-number provenance are in
[the revision audit](docs/revision-audit-20260924.md) and
[the generated number ledger](output/paper_numbers.md). Neural/fusion scores
above describe recorded runs; missing raw embeddings, checkpoints and session
predictions prevent rerunning them or performing nested fusion. WavLM seed
44 was selected on test performance; its six-seed mean remains disclosed.
The older resampling head-to-head comparison is not used in the paper: it
compares a resampled CTD distribution with a selected neural run.

CTD carries DAIC-WOZ information beyond simple latency and interview-structure
controls, with sensitivity to interviewer features and protocol. Fusion gains
remain statistically inconclusive. PDCH supplies boundary evidence, with no
reliable within-corpus discrimination; no detector transfer, construct
validation, or reliable multimodal advantage is claimed.

Official-cohort CTD sensitivity and its missing-timing caveat are reported in
[`output/official_split_ctd.md`](output/official_split_ctd.md). No stronger
acoustic baseline or matched reproduction of Agarwal et al. was run.
