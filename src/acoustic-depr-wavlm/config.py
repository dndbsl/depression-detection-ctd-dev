"""Central configuration: paths, hyperparameters, exclusions, label-column mapping.

All paths default to this machine's DAIC-WOZ layout but can be overridden via the
CLI flags exposed by each entry-point script.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# DAIC-WOZ raw dataset (external, licensed). Override with the DAIC_WOZ_ROOT env var.
_DAIC_ROOT = os.environ.get("DAIC_WOZ_ROOT", os.path.expanduser("~/datasets/DAIC-WOZ"))
DATA_ROOT = os.path.join(_DAIC_ROOT, "data")
LABELS_ROOT = os.path.join(_DAIC_ROOT, "labels")

MANIFEST_DIR = os.path.join(PROJECT_ROOT, "manifests")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Label CSV filenames per split.
LABEL_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",
}

# The test CSV uses different column names; normalise to the train/dev schema.
TEST_COLUMN_RENAME = {
    "PHQ_Binary": "PHQ8_Binary",
    "PHQ_Score": "PHQ8_Score",
}

# --------------------------------------------------------------------------- #
# Dataset facts
# --------------------------------------------------------------------------- #
# Sessions excluded entirely from every split (corrupt audio / labelling issues).
EXCLUDED_SESSIONS = {318, 321, 341, 362, 451, 458, 480, 373, 444, 409}

SAMPLE_RATE = 16000          # DAIC-WOZ audio is already 16 kHz mono PCM16.
MIN_UTTERANCE_SEC = 0.1      # Drop participant utterances shorter than 100 ms.
PARTICIPANT_SPEAKER = "Participant"

# --------------------------------------------------------------------------- #
# Model / feature dimensions
# --------------------------------------------------------------------------- #
WAVLM_MODEL_NAME = "microsoft/wavlm-large"
NUM_LAYERS = 25              # embedding output + 24 transformer layers
FEATURE_DIM = 1024          # WavLM-large hidden size


@dataclass
class TrainConfig:
    """Hyperparameters for the linear-probe head over cached WavLM features.

    Architecture: learnable layer-weighting -> additive attention pooling ->
    Dropout -> single Linear(1024->1) (~0.5M params, SUPERB-style). Deliberately
    tiny to match the ~100-subject training set and avoid memorisation. Model
    selection still keeps the lowest-dev-loss checkpoint.
    """
    # --- model architecture (linear probe) ---
    attn_hidden_dim: int = 256              # additive-attention projection size
    dropout: float = 0.3

    # --- optimisation ---
    lr: float = 5e-5
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1           # soft BCE targets -> [eps/2, 1-eps/2]
    batch_size: int = 32
    max_epochs: int = 200                   # overfits slowly now; best ckpt kept
    early_stopping: bool = False            # disabled: small dataset
    early_stop_patience: int = 30           # only used if early_stopping=True
    seed: int = 72
    pos_weight: float | None = None        # None -> class-balanced n_neg/n_pos
    num_bootstraps: int = 1000
    device: str = "cuda"


DEFAULT_TRAIN_CONFIG = TrainConfig()
