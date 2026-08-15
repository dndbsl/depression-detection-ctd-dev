"""Configuration loading for semantic RoBERTa MIL experiments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

# DAIC-WOZ raw dataset (external, licensed). Override with the DAIC_WOZ_ROOT env var.
_DAIC_ROOT = os.environ.get("DAIC_WOZ_ROOT", os.path.expanduser("~/datasets/DAIC-WOZ"))


@dataclass
class DataConfig:
    """Dataset paths and text tokenization settings."""

    data_root: str = os.path.join(_DAIC_ROOT, "data")
    labels_root: str = os.path.join(_DAIC_ROOT, "labels")
    processed_dir: str = str(PROJECT_ROOT / "datasets" / "processed")
    max_length: int = 512


@dataclass
class ModelConfig:
    """Encoder and MIL head settings."""

    encoder_name: str = "roberta-base"
    training_mode: str = "frozen"
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    pooling: str = "mean"
    mil_pooling: str = "attention"
    standardize_features: bool = False


@dataclass
class TrainingConfig:
    """Optimizer, batching, and reproducibility settings."""

    epochs: int = 500
    encoder_lr: float = 2e-5
    classifier_lr: float = 2e-5
    weight_decay: float = 0.01
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    physical_batch_size: int | str = "auto"
    max_physical_batch_size: int = 4
    min_effective_batch_size: int = 32
    num_workers: int = 0
    deterministic: bool = True
    require_cuda: bool = True
    eval_batch_size: int = 1


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    enabled: bool = True
    project: str = "semantic-depr-roberta"
    entity: str | None = None
    mode: str = "online"
    log_artifacts: bool = True


@dataclass
class OutputConfig:
    """Output directory settings."""

    output_root: str = str(PROJECT_ROOT)
    results_dir: str = str(PROJECT_ROOT / "results")
    checkpoints_dir: str = str(PROJECT_ROOT / "checkpoints")
    logs_dir: str = str(PROJECT_ROOT / "logs")
    figures_dir: str = str(PROJECT_ROOT / "figures")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_name: str = "roberta-base_frozen"
    encoder: str = "roberta-base"
    training_strategy: str = "frozen"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def normalized(self) -> "ExperimentConfig":
        """Synchronize legacy top-level fields with nested model fields."""
        self.model.encoder_name = self.encoder
        self.model.training_mode = self.training_strategy
        return self


def _coerce_dataclass(cls: type, value: dict[str, Any] | None) -> Any:
    if value is None:
        return cls()
    names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in value.items() if k in names}
    return cls(**filtered)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file into an ExperimentConfig."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = ExperimentConfig(
        experiment_name=raw.get("experiment_name", raw.get("name", "experiment")),
        encoder=raw.get("encoder", raw.get("encoder_name", "roberta-base")),
        training_strategy=raw.get(
            "training_strategy", raw.get("training_mode", "frozen")
        ),
        data=_coerce_dataclass(DataConfig, raw.get("data")),
        model=_coerce_dataclass(ModelConfig, raw.get("model")),
        training=_coerce_dataclass(TrainingConfig, raw.get("training")),
        wandb=_coerce_dataclass(WandbConfig, raw.get("wandb")),
        output=_coerce_dataclass(OutputConfig, raw.get("output")),
    ).normalized()
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Validate configuration values that would otherwise fail late."""
    if config.encoder not in {"roberta-base", "roberta-large"}:
        raise ValueError(f"Unsupported encoder: {config.encoder}")
    if config.training_strategy not in {"frozen", "finetune"}:
        raise ValueError(f"Unsupported training strategy: {config.training_strategy}")
    if config.model.pooling not in {"mean", "cls"}:
        raise ValueError(f"Unsupported pooling: {config.model.pooling} (expected 'mean' or 'cls')")
    if config.model.mil_pooling not in {"attention", "mean"}:
        raise ValueError(f"Unsupported mil_pooling: {config.model.mil_pooling} (expected 'attention' or 'mean')")
    if config.data.max_length != 512:
        raise ValueError("The project specification requires max_length = 512.")
    if config.training.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.training.min_effective_batch_size < 32:
        raise ValueError("effective batch size must be at least 32.")
    if config.wandb.mode != "online":
        raise ValueError("W&B mode must be online for this project.")


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert a dataclass config to plain nested dictionaries."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass")
    return asdict(config)


def ensure_output_dirs(config: ExperimentConfig) -> None:
    """Create output directories used by training and evaluation."""
    for path in (
        config.data.processed_dir,
        config.output.results_dir,
        config.output.checkpoints_dir,
        config.output.logs_dir,
        config.output.figures_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def experiment_slug(config: ExperimentConfig) -> str:
    """Return a filesystem-safe experiment slug."""
    encoder = config.encoder.replace("/", "-")
    return f"{encoder}_{config.training_strategy}"

