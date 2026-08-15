# Semantic Depression Detection (DAIC-WOZ + frozen RoBERTa-large)

Text-only participant-level depression detection with a frozen RoBERTa encoder
and MIL pooling over participant turns.

```
transcript -> participant turns (Ellie rows define boundaries; participant text only)
           -> frozen roberta-large, mean-pooled 1024-D instance embeddings (prepare_dataset.py)
           -> per-feature standardization (train-fit, stored in checkpoint)
           -> MIL mean pooling over instances -> dropout 0.1 -> Linear(1024->1) (models/roberta_mil.py)
           -> train head only; best checkpoint by dev macro-F1 (train.py)
           -> dev-tuned decision threshold stored with the run
```

The model input contains **participant language only**. Ellie turns define
instance boundaries and are never included in any instance text.

## Setup

Uses the same conda env as the acoustic project (torch 2.5.1 + transformers 4.46.3):

```bash
conda env create -f ../acoustic-depr-wavlm/environment.yml
conda activate acoustic-depr
export DAIC_WOZ_ROOT=/path/to/DAIC-WOZ    # expects DAIC-WOZ/data and DAIC-WOZ/labels
```

Dataset cleaning is shared with the other modalities (`src/common/daic_cleaning.py`).

## Run

```bash
# 1. Build processed text bags -> datasets/processed/
python prepare_dataset.py --config configs/roberta_large_frozen.yaml

# 2a. Train all three seeds (42, 43, 44), as in the paper grid:
python run_all_experiments.py --allow-download

# 2b. OR only seed 43 (the deployed detector):
python train.py --config configs/roberta_large_frozen.yaml --seed 43
```

W&B online logging is used for full runs; for local smoke tests add
`--disable-wandb --skip-preflight --allow-cpu --debug-epochs 1`.

## Deployed outputs (consumed by `src/fusion/`)

```
checkpoints/roberta-large_frozen_seed43/best_checkpoint.pt
results/runs/roberta-large_frozen_seed43/dev_predictions_best.csv
results/runs/roberta-large_frozen_seed43/test_predictions.csv
```

The published table uses **seed 43** (dev macro-F1 0.690, threshold 0.5238;
test macro-F1 0.604 ± 0.025 across seeds 42/43/44). To re-evaluate a
checkpoint manually:

```bash
python predict.py \
  --config configs/roberta_large_frozen.yaml \
  --checkpoint checkpoints/roberta-large_frozen_seed43/best_checkpoint.pt \
  --split test \
  --output results/runs/roberta-large_frozen_seed43/test_predictions.csv
```

All generated artifacts (`datasets/processed/`, `checkpoints/`, `results/`,
`figures/`, `logs/`, `wandb/`) are git-ignored and regenerable.
