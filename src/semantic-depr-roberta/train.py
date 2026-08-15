#!/usr/bin/env python
"""Train one semantic RoBERTa MIL experiment."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_paths

bootstrap_paths()

from training.config import load_config
from training.runner import train_one_run


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--debug-epochs", type=int, default=None, help="Override epochs for smoke tests.")
    parser.add_argument("--debug-participants", type=int, default=None, help="Use a tiny stratified participant subset.")
    parser.add_argument("--debug-max-instances", type=int, default=None, help="Cap instances per participant for smoke tests.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B for local debugging.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution for smoke tests.")
    return parser.parse_args()


def main() -> None:
    """Train one run."""
    args = parse_args()
    config = load_config(args.config)
    payload = train_one_run(
        config,
        seed=args.seed,
        disable_wandb=args.disable_wandb,
        debug_epochs=args.debug_epochs,
        debug_participants=args.debug_participants,
        debug_max_instances=args.debug_max_instances,
        allow_cpu=args.allow_cpu,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
