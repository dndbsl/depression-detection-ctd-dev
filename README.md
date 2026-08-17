# Can Conversational Temporal Dynamics Improve Depression Detection in Dyads?

Code for the paper [*"Can Conversational Temporal Dynamics Improve Depression
Detection in Dyads? A Preliminary Investigation in Multi-Modality Perspectives"*
(arXiv:2607.03744)](https://arxiv.org/abs/2607.03744). A copy of the PDF is in
[`paper/`](paper/).

We treat **conversational temporal dynamics (CTD)** — dyadic Ask/Res turn-pair
timing — as a first-class modality alongside frozen self-supervised encoders,
and fuse the three detectors at the score level on DAIC-WOZ (cleaned standard
splits: **train 102 / dev 33 / test 45**, binary label `PHQ8_Score >= 10`):

| Modality | Model | DEV macro-F1 [95% CI] | TEST macro-F1 [95% CI] |
|---|---|---|---|
| Acoustic (A) | frozen WavLM-large, 25-layer weighting + attention pool → `Linear(1024→1)`; thr 0.65 | **0.673** [0.498, 0.818] | **0.545** [0.392, 0.681] |
| Semantic (T) | frozen RoBERTa-large, mean MIL pooling → classifier; thr 0.524 | **0.690** [0.517, 0.833] | **0.631** [0.474, 0.769] |
| CTD | 24-D session-mean + L2 LogReg (C=0.3); thr 0.50 | **0.746** [0.581, 0.879] | **0.631** [0.472, 0.771] |
| **T + CTD (convex late fusion)** ★ | dev-tuned weights 0.3/0.7, thr 0.55 | **0.804** [0.643, 0.935] | **0.669** [0.509, 0.806] |

★ Dev-selected best fusion. The learned three-way (A+T+CTD) fusion assigns
**zero weight to acoustics** and coincides with T+CTD.

Protocol: all hyperparameter / weight / threshold choices are made on **dev**;
dev is the primary report and test is reported once, on the side; every metric
carries a 2000-bootstrap 95% CI. Full tables: [`RESULTS.md`](RESULTS.md);
machine-readable results: [`output/`](output/).

---

## Repository layout

```
depression-detection-ctd/
├── README.md / RESULTS.md          # this guide + full result tables
├── paper/                          # paper PDF
├── output/                         # published aggregate metrics (JSON)
│   ├── RESULTS.json                # 7 headline configurations
│   ├── fusion_wavlm_seed44.json    # all metrics + CIs (canonical WavLM seed 44)
│   ├── mean_prob_seed44.json       # parameter-free mean_prob fusion reference
│   └── summary_wavlm_seeds.json    # fusion summary across WavLM seeds
└── src/
    ├── common/                     # shared DAIC-WOZ cleaning + transcript loading
    ├── acoustic-depr-wavlm/        # A: frozen WavLM-large probe (GPU)
    ├── semantic-depr-roberta/      # T: frozen RoBERTa-large MIL (GPU)
    ├── ctd/                        # CTD: 24-D timing features + LogReg (CPU)
    └── fusion/                     # score-level late fusion + analyses
```

Generated artifacts (feature caches, checkpoints, manifests, per-session
predictions, `outputs/`) are git-ignored and recreated by the pipeline below.

## Prerequisites

1. **DAIC-WOZ dataset** (license-restricted; obtain from the [USC ICT
   distributor](https://dcapswoz.ict.usc.edu/)). Point the code at it with one
   environment variable:

   ```bash
   export DAIC_WOZ_ROOT=/path/to/DAIC-WOZ
   ```

   Expected layout: `DAIC-WOZ/data/{id}_P/{id}_{AUDIO.wav,TRANSCRIPT.csv}` and
   `DAIC-WOZ/labels/{train,dev}_split_Depression_AVEC2017.csv`, `full_test_split.csv`.
2. **GPU** (a single A100-class GPU was used originally) for WavLM feature
   extraction and RoBERTa training. CTD and fusion run on CPU.
3. **Environments**:
   - `acoustic-depr` (WavLM **and** RoBERTa; torch 2.5.1 + transformers 4.46.3):
     `conda env create -f src/acoustic-depr-wavlm/environment.yml`
   - CPU env for CTD + fusion:
     `pip install -r src/ctd/requirements.txt confidence_intervals`

## Pipeline (end to end)

Run each modality, then fusion. All commands assume `DAIC_WOZ_ROOT` is exported.

### 1. Acoustic — frozen WavLM-large

```bash
cd src/acoustic-depr-wavlm            # env: acoustic-depr
python preprocess.py --data-root "$DAIC_WOZ_ROOT/data" --labels-root "$DAIC_WOZ_ROOT/labels" --out-dir manifests/
python -m features.wavlm_extractor    # caches frozen WavLM-large features (GPU, slowest step)
python train.py --seed 44 --checkpoint-dir checkpoints/seed_44
python evaluate.py --checkpoint checkpoints/seed_44/best.pt --split test
```

The paper's seed sweep (42/43/44/71/72/73; test macro-F1 0.506 ± 0.034) is
`bash run_seed_sweep.sh`. The paper's **turn-level acoustic ablation** (dev
0.572 / test 0.333) uses the same probe on Ask/Res response-turn segments:
pass `segmentation="qa"` in `preprocess.py` (see `data/qa_pairing.py`).

### 2. Semantic — frozen RoBERTa-large

```bash
cd src/semantic-depr-roberta          # env: acoustic-depr
python prepare_dataset.py --config configs/roberta_large_frozen.yaml
python run_all_experiments.py --allow-download    # trains seeds 42/43/44
# the deployed detector is the seed-43 frozen run (test macro-F1 0.604 ± 0.025 across seeds)
```

### 3. CTD — 24-D session-mean + L2 logistic regression

```bash
cd src/ctd                            # CPU env
python ml_splits.py                   # canonical 24-D detector (dev 0.746 / test 0.631)
```

`ml_splits.py` also runs the paper's post-hoc robustness variant (24 features ×
10 eGeMAPS functionals = 240-D, LogReg/SVM/RF selected on dev), which did not
beat the pre-committed 24-D session-mean model.

### 4. Late fusion

```bash
cd src/fusion                         # env: acoustic-depr (GPU only for the WavLM forward)
python extract_ctd_roberta.py         # -> src/ctd/outputs/fusion/{ctd,roberta}_{train,dev,test}.npz
python check_splits.py                # verify identical sessions + labels across modalities
python run_fusion_wavlm_seeds.py --wavlm-seeds 44 --device cuda
python paired_delta_analysis.py       # paired bootstrap deltas + prediction-change counts
```

`run_fusion_wavlm_seeds.py` fuses the deployed per-modality probabilities,
grid-searches convex weights + decision threshold on dev, and reports dev
(primary) and test (side) with 95% CIs — this reproduces
[`output/fusion_wavlm_seed44.json`](output/fusion_wavlm_seed44.json).

## Reproducibility notes

- **Exactly reproducible:** the CTD detector, the fusion stage (given a fixed
  WavLM checkpoint), split checks, and the paired-delta analysis.
- **Similar, not bit-identical:** WavLM and RoBERTa retraining, due to
  GPU/cuDNN nondeterminism. Seeds are fixed (RoBERTa 42/43/44 with seed 43
  deployed; WavLM 42/43/44/71/72/73 with seed 44 deployed).
- **Selection discipline:** test is never used for any choice. Dev (n=33) is
  small; the convex weights are dev-tuned, so the parameter-free
  `mean_prob[T+CTD]` fusion (dev 0.738 / test 0.650) is reported as a
  no-tuning reference.
- Deployed-model details (seeds, epochs, thresholds, hyperparameters) are
  documented in [`RESULTS.md`](RESULTS.md) §2–3.

## Data and licensing

The DAIC-WOZ corpus is **not** included and cannot be redistributed. This
repository also ships **no** per-session artifacts derived from it (no labels,
manifests, embeddings, predictions, or model checkpoints) — only code and
aggregate metrics. All intermediate artifacts are regenerated locally from
your licensed copy by the pipeline above.

`src/acoustic-depr-wavlm/third_party/confidence_intervals` is vendored from
[luferrer/ConfidenceIntervals](https://github.com/luferrer/ConfidenceIntervals)
under its own license (see the bundled LICENSE/NOTICE).
