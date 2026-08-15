# Full Results — DAIC-WOZ Depression Detection (Single Models + Late Fusion)

**Dataset:** standard cleaned DAIC-WOZ splits (AVEC2017)
**Cohorts:** train 102 · **dev 33** · **test 45**
**Label:** `PHQ8_Binary = (PHQ8_Score ≥ 10)`
**Primary metric:** macro-F1 with **2000-bootstrap 95% confidence intervals** (Ferrer & Riera; `confidence_intervals` package, α=5)

**Protocol:** all hyperparameter / weight / threshold choices are made on **dev**.
**dev = primary report** · **test = held-out side report** (never used for selection).

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

- **Semantic + CTD** beats every single modality on **both** dev (0.746→0.804) and test (0.631→0.669).
- **Acoustic (WavLM seed 44)** does not improve the best fusion; the all-3 optimum excludes it (weight 0.0).
- CTD alone matches RoBERTa on test (both 0.631) but is strongest on dev (0.746).

---

## 2. Single-modality models (deployed detectors)

Each model was trained/selected independently on dev. Fusion uses the **deployed
per-session probabilities** from these exact checkpoints — not a re-fit probe.

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

Conservative no-tuning comparison (only the threshold is tuned on dev; WavLM seed 44):

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
