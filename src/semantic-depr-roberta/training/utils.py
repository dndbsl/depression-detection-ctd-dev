"""General training utilities."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, PyTorch, and CUDA random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def effective_accumulation(physical_batch_size: int, min_effective_batch_size: int) -> int:
    """Return accumulation steps needed to reach the minimum effective batch size."""
    return max(1, math.ceil(min_effective_batch_size / physical_batch_size))


def sigmoid_to_float(logits: torch.Tensor) -> list[float]:
    """Convert logits to CPU probability floats."""
    return torch.sigmoid(logits.detach()).cpu().numpy().astype(float).tolist()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable indentation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gpu_info() -> dict[str, Any]:
    """Return basic GPU runtime information."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "total_memory_mb": int(props.total_memory / (1024 * 1024)),
    }

