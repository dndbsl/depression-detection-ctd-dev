#!/usr/bin/env python
"""Prepare DAIC-WOZ text-only MIL bags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap_paths

bootstrap_paths()

from datasets.daic import prepare_dataset
from training.config import DataConfig, load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Optional experiment YAML.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--labels-root", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run dataset preprocessing."""
    args = parse_args()
    if args.config:
        data = load_config(args.config).data
    else:
        data = DataConfig()
    if args.data_root:
        data.data_root = args.data_root
    if args.labels_root:
        data.labels_root = args.labels_root
    if args.processed_dir:
        data.processed_dir = args.processed_dir
    summary = prepare_dataset(data)
    print(json.dumps(summary["split_summary"], indent=2))
    print(f"Wrote preprocessed dataset to {Path(data.processed_dir)}")


if __name__ == "__main__":
    main()

