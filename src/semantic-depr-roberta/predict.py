#!/usr/bin/env python
"""Generate participant predictions from a saved checkpoint."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap_paths

bootstrap_paths()

from evaluation.evaluate_run import evaluate_checkpoint
from training.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Write prediction CSV files."""
    args = parse_args()
    config = load_config(args.config)
    evaluate_checkpoint(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_path=args.output,
        allow_cpu=args.allow_cpu,
    )
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()

