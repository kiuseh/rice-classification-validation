"""Command-line entry point for the reproducible rice experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from rice_model import run_experiment

DEFAULT_DATASET = Path(__file__).parent / "data" / "Rice_Cammeo_Osmancik.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the rice-variety classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the UCI rice workbook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
        help="Directory for metrics.json and evaluation-summary.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, _ = run_experiment(args.data, output_dir=args.output_dir)
    cross_validation = results["cross_validation_on_training_partition"]
    holdout = results["holdout"]

    print(
        "Training CV accuracy: "
        f"{cross_validation['accuracy']['mean']:.2%} "
        f"± {cross_validation['accuracy']['std']:.2%}"
    )
    print(
        "Training CV macro F1: "
        f"{cross_validation['macro_f1']['mean']:.2%} "
        f"± {cross_validation['macro_f1']['std']:.2%}"
    )
    print(f"Holdout accuracy: {holdout['accuracy']:.2%}")
    print(f"Holdout macro F1: {holdout['macro_f1']:.2%}")
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
