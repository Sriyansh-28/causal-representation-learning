#!/usr/bin/env python3
"""Audit script: verify the statistical protocol across all committed results.

Checks, per experiment file:
  * the expected replicate count is present and complete
  * every model was evaluated on the *same* seed set (valid pairing)
  * no failed runs silently reduce a group's n
  * Holm correction is applied and is at least as conservative as raw alpha
  * bootstrap CIs bracket the reported mean

Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.paths import PROJECT_ROOT  # noqa: E402

RAW = PROJECT_ROOT / "results" / "raw"
TABLES = PROJECT_ROOT / "results" / "tables"

# Experiments 1-5 were pre-declared at 30 seeds; Experiment 6 sweeps 20
# settings and was pre-declared at 10.
EXPECTED_SEEDS = {"sensitivity": 10}
DEFAULT_SEEDS = 30


def check(results: List[Tuple[str, bool, str]], name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def main() -> int:
    """Run every statistical-protocol check and print a pass/fail report."""
    results: List[Tuple[str, bool, str]] = []
    raw_files = sorted(RAW.glob("*_raw.csv"))
    if not raw_files:
        print("no result files found", file=sys.stderr)
        return 1

    for path in raw_files:
        name = path.name.removesuffix("_raw.csv")
        df = pd.read_csv(path)
        errors = (df["error"].fillna("") != "").sum() if "error" in df else 0
        df_ok = df[df["error"].fillna("") == ""] if "error" in df else df

        check(results, f"{name}: no failed runs", errors == 0, f"{errors} failures")

        expected = EXPECTED_SEEDS.get(name, DEFAULT_SEEDS)
        n_seeds = df_ok["seed"].nunique()
        check(results, f"{name}: {expected} seeds present", n_seeds == expected,
              f"found {n_seeds}")

        # Pairing validity: identical seed sets for every model within a condition.
        bad = []
        for (cond, val), grp in df_ok.groupby(["condition", "condition_value"]):
            seed_sets = grp.groupby("model")["seed"].apply(lambda s: frozenset(s))
            if len(set(seed_sets)) > 1:
                bad.append(f"{cond}={val}")
        check(results, f"{name}: all models share one seed set per condition",
              not bad, "; ".join(bad))

        # Balanced cells: every (condition, model) has the full replicate count.
        counts = df_ok.groupby(["condition", "condition_value", "model"])["seed"].nunique()
        check(results, f"{name}: all cells complete",
              bool((counts == expected).all()),
              f"min cell n={int(counts.min())}")

        # Aggregated CI must bracket the aggregated mean.
        agg_path = TABLES / f"{name}_aggregate.csv"
        if agg_path.exists():
            agg = pd.read_csv(agg_path)
            valid = agg.dropna(subset=["ci_low", "ci_high", "mean"])
            brackets = ((valid["ci_low"] <= valid["mean"]) & (valid["mean"] <= valid["ci_high"]))
            check(results, f"{name}: bootstrap CIs bracket the mean",
                  bool(brackets.all()), f"{(~brackets).sum()} violations")

        # Holm must never reject something raw alpha would not.
        for suffix in ("_paired_tests.csv", "_ablation_deltas.csv"):
            tpath = TABLES / f"{name}{suffix}"
            if not tpath.exists():
                continue
            t = pd.read_csv(tpath)
            col = "reject_h0_holm_0.05"
            if col not in t or "p_value" not in t:
                continue
            inconsistent = t[t[col] & (t["p_value"] > 0.05)]
            check(results, f"{name}{suffix}: Holm no less conservative than raw alpha",
                  inconsistent.empty, f"{len(inconsistent)} rows")

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for nm, ok, detail in results:
        if not ok:
            failed += 1
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"[{'PASS' if ok else 'FAIL'}] {nm:<{width}}{suffix}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
