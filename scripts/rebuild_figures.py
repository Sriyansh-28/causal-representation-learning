#!/usr/bin/env python3
"""Regenerate every figure from the committed raw result CSVs.

Figures are a rendering of the results, not part of the experiment, so they can
be rebuilt without refitting a single model. Use this after changing anything
in ``src/experiments/plots.py``.

Usage::

    python scripts/rebuild_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.aggregate import aggregate_results  # noqa: E402
from src.experiments.plots import generate_figures  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> int:
    """Rebuild figures for every ``*_raw.csv`` present, and report the count."""
    raw_files = sorted((RESULTS_DIR / "raw").glob("*_raw.csv"))
    if not raw_files:
        print("no raw result files found; run an experiment first", file=sys.stderr)
        return 1

    total = 0
    for path in raw_files:
        name = path.name.removesuffix("_raw.csv")
        df = pd.read_csv(path)
        if "error" in df.columns:
            df = df[df["error"].fillna("") == ""]
        if df.empty:
            print(f"{name}: no successful runs, skipped")
            continue
        experiment = str(df["experiment"].iloc[0])
        figures = generate_figures(
            df, aggregate_results(df), experiment, name, RESULTS_DIR
        )
        total += len(figures)
        print(f"{name}: {len(figures)} figure(s)")
    print(f"total: {total} figures written to {RESULTS_DIR / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
