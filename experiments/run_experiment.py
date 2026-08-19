#!/usr/bin/env python3
"""Command-line entry point for every experiment.

Examples::

    python experiments/run_experiment.py --config configs/baseline.yaml
    python experiments/run_experiment.py --config configs/confounding.yaml
    python experiments/run_experiment.py --config configs/ablation.yaml --seeds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow execution as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.aggregate import aggregate_results, save_tables  # noqa: E402
from src.experiments.plots import generate_figures  # noqa: E402
from src.experiments.runner import run_experiment, save_results  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import ensure_dir, resolve  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a causal representation-learning experiment from a YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override the number of replicate seeds (uses 0..N-1).")
    parser.add_argument("--output-dir", default=None,
                        help="Override the config's output directory.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one experiment end to end and write results, tables and figures."""
    args = parse_args(argv)
    config = load_config(args.config)
    if args.seeds is not None:
        if args.seeds < 1:
            raise SystemExit("--seeds must be at least 1")
        config.seeds = list(range(args.seeds))
    if args.output_dir:
        config.output_dir = args.output_dir

    verbose = not args.quiet
    if verbose:
        print(f"Experiment : {config.name} ({config.experiment})")
        print(f"Dataset    : {config.dataset.get('name')}")
        print(f"Models     : {', '.join(config.models)}")
        print(f"Seeds      : {len(config.seeds)}")

    df = run_experiment(config, verbose=verbose)
    csv_path = save_results(df, config)

    n_err = int((df.get("error", "").fillna("") != "").sum()) if "error" in df else 0
    agg = aggregate_results(df)
    out_dir = ensure_dir(resolve(config.output_dir))
    paths = save_tables(df, agg, config.name, out_dir)

    figures = []
    if not args.no_figures:
        figures = generate_figures(df, agg, config.experiment, config.name, out_dir)

    if verbose:
        print(f"\nRuns       : {len(df)} ({n_err} failed)")
        print(f"Raw results: {csv_path}")
        for key, path in paths.items():
            print(f"{key:11s}: {path}")
        for fig in figures:
            print(f"figure     : {fig}")
    return 1 if n_err and n_err == len(df) else 0


if __name__ == "__main__":
    raise SystemExit(main())
