#!/usr/bin/env python
"""Run all semantic RoBERTa MIL experiments and aggregate results."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_paths

bootstrap_paths()

from datasets.daic import prepare_dataset
from evaluation.figures import plot_attention, plot_roc_pr
from evaluation.reporting import write_assumptions, write_report, write_result_tables
from training.config import PROJECT_ROOT, load_config
from training.preflight import run_preflight
from training.runner import train_one_run
from training.wandb_utils import start_aggregate_run


# Only the deployed configuration used for the fusion table is shipped in this
# bundle: roberta-large frozen (the seed43 run supplies the fusion probabilities).
DEFAULT_CONFIGS = [
    PROJECT_ROOT / "configs" / "roberta_large_frozen.yaml",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=[str(path) for path in DEFAULT_CONFIGS])
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow HF model download during preflight.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B for debugging.")
    parser.add_argument("--debug-epochs", type=int, default=None)
    parser.add_argument("--debug-participants", type=int, default=None)
    parser.add_argument("--debug-max-instances", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the complete experimental grid."""
    args = parse_args()
    configs = [load_config(path) for path in args.configs]
    if not args.skip_preflight:
        run_preflight(configs, allow_download=args.allow_download)

    prepare_dataset(configs[0].data)
    for config in configs:
        seeds = args.seeds if args.seeds is not None else config.training.seeds
        for seed in seeds:
            train_one_run(
                config,
                seed=seed,
                disable_wandb=args.disable_wandb,
                debug_epochs=args.debug_epochs,
                debug_participants=args.debug_participants,
                debug_max_instances=args.debug_max_instances,
                allow_cpu=args.allow_cpu,
            )

    result_paths = write_result_tables(configs[0].output.results_dir)
    assumptions_path = write_assumptions(PROJECT_ROOT)
    report_path = write_report(PROJECT_ROOT, configs[0].output.results_dir, configs[0].data.processed_dir)

    ranking = Path(configs[0].output.results_dir) / "ranking_results.csv"
    if ranking.exists():
        import pandas as pd

        ranking_frame = pd.read_csv(ranking)
        if not ranking_frame.empty:
            best_experiment = ranking_frame.iloc[0]["Experiment"]
            best_run = sorted(Path(configs[0].output.results_dir).glob(f"runs/{best_experiment}_seed*/test_predictions.csv"))
            if best_run:
                run_dir = best_run[0].parent
                try:
                    plot_roc_pr(
                        run_dir / "test_predictions.csv",
                        Path(configs[0].output.figures_dir) / "best_model_roc.png",
                        Path(configs[0].output.figures_dir) / "best_model_pr.png",
                    )
                    plot_attention(
                        run_dir / "attention_weights.csv",
                        Path(configs[0].output.figures_dir) / "attention",
                    )
                except ImportError:
                    pass

    logger = start_aggregate_run(configs[0], disabled=args.disable_wandb)
    logger.log_artifacts([*result_paths.values(), assumptions_path, report_path])
    logger.finish()
    print(f"Wrote aggregate results to {configs[0].output.results_dir}")


if __name__ == "__main__":
    main()
