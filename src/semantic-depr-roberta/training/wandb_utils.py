"""Weights & Biases helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from training.config import ExperimentConfig, config_to_dict, experiment_slug

# Canonical per-split metric keys logged each epoch (train/dev/test).
# See fusion_release/WANDB_METRICS.md for the full listing and dashboard tips.
EPOCH_METRIC_SUFFIXES: tuple[str, ...] = (
    "loss",
    "accuracy",
    "f1",
    "macro_f1",
    "auroc",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
)
EPOCH_SPLITS: tuple[str, ...] = ("train", "dev", "test")
SUMMARY_PREFIXES: tuple[str, ...] = ("best", "final")


class WandbLogger:
    """Small wrapper around wandb with a disabled mode for tests."""

    def __init__(self, run: Any | None) -> None:
        self.run = run

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        """Log scalar or table payloads."""
        if self.run is not None:
            self.run.log(payload, step=step)

    def log_summary(self, payload: dict[str, Any]) -> None:
        """Write run-level summary scalars (best/final checkpoint metrics)."""
        if self.run is not None:
            for key, value in payload.items():
                self.run.summary[key] = value

    def log_artifacts(self, paths: Iterable[str | Path]) -> None:
        """Log files to the active W&B run."""
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(f"{self.run.name}-artifacts", type="run-output")
        for path in paths:
            p = Path(path)
            if p.exists() and p.is_file():
                artifact.add_file(str(p))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            self.run.finish()


def start_wandb_run(
    config: ExperimentConfig,
    *,
    seed: int,
    job_type: str,
    disabled: bool = False,
) -> WandbLogger:
    """Start an online W&B run for an experiment."""
    if disabled or not config.wandb.enabled:
        return WandbLogger(None)
    import wandb

    run_name = f"{config.encoder}_{config.training_strategy}_seed{seed}"
    group_name = experiment_slug(config)
    kwargs: dict[str, Any] = {
        "project": config.wandb.project,
        "name": run_name,
        "group": group_name,
        "job_type": job_type,
        "config": {**config_to_dict(config), "seed": seed},
        "tags": [config.encoder, config.training_strategy, f"seed:{seed}"],
        "mode": config.wandb.mode,
        "settings": wandb.Settings(init_timeout=120),
    }
    if config.wandb.entity:
        kwargs["entity"] = config.wandb.entity
    try:
        run = wandb.init(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(
            "W&B online logging is required. Run `wandb login` or set "
            "`WANDB_API_KEY` before launching experiments."
        ) from exc
    return WandbLogger(run)


def start_aggregate_run(config: ExperimentConfig, disabled: bool = False) -> WandbLogger:
    """Start the aggregate W&B run for summary tables."""
    if disabled or not config.wandb.enabled:
        return WandbLogger(None)
    import wandb

    try:
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name="aggregate_results",
            group="aggregate_results",
            job_type="aggregate",
            config=config_to_dict(config),
            tags=["aggregate", config.encoder, config.training_strategy],
            mode=config.wandb.mode,
            settings=wandb.Settings(init_timeout=120),
        )
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError("Unable to start required W&B aggregate run.") from exc
    return WandbLogger(run)
