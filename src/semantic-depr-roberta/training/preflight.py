"""Runtime preflight checks for full experiments."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from training.config import ExperimentConfig


REQUIRED_PACKAGES = ["torch", "transformers", "sklearn", "pandas", "numpy", "matplotlib", "yaml", "tqdm", "wandb"]


def run_preflight(configs: list[ExperimentConfig], *, allow_download: bool = False) -> dict:
    """Validate runtime dependencies, CUDA, W&B auth, and model availability."""
    report: dict[str, object] = {"packages": {}, "models": {}, "cuda": {}, "wandb": {}}
    errors: list[str] = []

    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            report["packages"][package] = getattr(module, "__version__", "ok")
        except Exception as exc:
            errors.append(f"Missing package `{package}`: {exc}")
            report["packages"][package] = "missing"

    require_cuda = any(config.training.require_cuda for config in configs)
    report["cuda"] = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if require_cuda and not torch.cuda.is_available():
        errors.append("CUDA is required by config but torch.cuda.is_available() is false.")

    netrc_exists = Path.home().joinpath(".netrc").exists()
    api_key_set = bool(os.environ.get("WANDB_API_KEY"))
    report["wandb"] = {"netrc_exists": netrc_exists, "api_key_set": api_key_set}
    if any(config.wandb.enabled for config in configs) and not (netrc_exists or api_key_set):
        errors.append("W&B online auth is required but no .netrc or WANDB_API_KEY was found.")

    for encoder in sorted({config.encoder for config in configs}):
        try:
            AutoConfig.from_pretrained(encoder, local_files_only=not allow_download)
            AutoTokenizer.from_pretrained(encoder, local_files_only=not allow_download)
            model = AutoModel.from_pretrained(
                encoder,
                add_pooling_layer=False,
                local_files_only=not allow_download,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report["models"][encoder] = "available"
        except Exception as exc:
            report["models"][encoder] = "missing"
            errors.append(
                f"`{encoder}` is not available locally"
                + (" or downloadable." if allow_download else ".")
                + f" Details: {exc}"
            )

    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return report
