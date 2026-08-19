"""Aggregate raw per-run results into summary tables.

Aggregation is deliberately separate from execution: tables can be rebuilt
from the committed raw CSVs without re-running any experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..evaluation.statistics import holm_bonferroni, paired_test, summarize
from ..utils.paths import ensure_dir

PRIMARY_METRICS = ["pehe", "abs_ate_error", "policy_regret", "policy_value"]


def _successful(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose run raised, so failures cannot contaminate summaries."""
    if "error" not in df.columns:
        return df
    return df[df["error"].fillna("") == ""].copy()


def aggregate_results(
    df: pd.DataFrame, metrics: Sequence[str] = tuple(PRIMARY_METRICS)
) -> pd.DataFrame:
    """Summarise metrics by (condition, condition value, model) across seeds.

    Returns one row per group and metric with mean, SD, SEM, replicate count
    and a bootstrap 95% confidence interval.
    """
    df = _successful(df)
    rows: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(rows)

    group_cols = ["experiment", "condition", "condition_value", "model"]
    for keys, grp in df.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in grp.columns or grp[metric].isna().all():
                continue
            stats = summarize(grp[metric].to_numpy())
            rows.append({**base, "metric": metric, **stats})
    return pd.DataFrame(rows)


def pivot_table(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a condition-by-model table of ``mean +/- SD`` strings for one metric."""
    sub = agg[agg["metric"] == metric]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.assign(
        summary=sub.apply(
            lambda r: f"{r['mean']:.3f} ± {r['std']:.3f} (med {r['median']:.3f})"
            if "median" in sub.columns and np.isfinite(r.get("median", float("nan")))
            else f"{r['mean']:.3f} ± {r['std']:.3f}",
            axis=1,
        )
    )
    return sub.pivot_table(
        index=["condition", "condition_value"], columns="model",
        values="summary", aggfunc="first",
    ).reset_index()


def paired_comparisons(
    df: pd.DataFrame, reference: str = "neural_rep", metric: str = "pehe"
) -> pd.DataFrame:
    """Compare ``reference`` against every other model on paired replicates.

    Models are evaluated on identical seeds and conditions, so the pairing is
    valid. p-values are Holm-Bonferroni corrected within each condition; a
    negative ``mean_diff`` means the reference model has lower error.
    """
    df = _successful(df)
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for (cond, cval), grp in df.groupby(["condition", "condition_value"], sort=False):
        wide = grp.pivot_table(index="seed", columns="model", values=metric)
        if reference not in wide.columns:
            continue
        others = [m for m in wide.columns if m != reference]
        block: List[Dict[str, object]] = []
        for other in others:
            pair = wide[[reference, other]].dropna()
            res = paired_test(pair[reference].to_numpy(), pair[other].to_numpy())
            block.append({
                "condition": cond, "condition_value": cval, "metric": metric,
                "reference": reference, "comparison": other, **res,
            })
        flags = holm_bonferroni([b["p_value"] for b in block])
        for b, f in zip(block, flags):
            b["reject_h0_holm_0.05"] = bool(f)
        records.extend(block)
    return pd.DataFrame(records)


def to_markdown(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    if df.empty:
        return "_(no rows)_\n"
    return df.to_markdown(index=False, floatfmt=float_fmt.replace("%", "")) + "\n"


def save_tables(
    df: pd.DataFrame, agg: pd.DataFrame, name: str, out_dir: Path
) -> Dict[str, Path]:
    """Write aggregated CSV and Markdown tables; return the written paths."""
    tables_dir = ensure_dir(out_dir / "tables")
    paths: Dict[str, Path] = {}

    agg_csv = tables_dir / f"{name}_aggregate.csv"
    agg.to_csv(agg_csv, index=False)
    paths["aggregate_csv"] = agg_csv

    md_parts = [f"# {name}: aggregated results\n"]
    for metric in PRIMARY_METRICS:
        pv = pivot_table(agg, metric)
        if pv.empty:
            continue
        md_parts.append(f"\n## {metric} (mean ± SD over seeds)\n\n")
        md_parts.append(to_markdown(pv))
    md_path = tables_dir / f"{name}_summary.md"
    md_path.write_text("".join(md_parts), encoding="utf-8")
    paths["summary_md"] = md_path

    cmp_df = paired_comparisons(df)
    if not cmp_df.empty:
        cmp_csv = tables_dir / f"{name}_paired_tests.csv"
        cmp_df.to_csv(cmp_csv, index=False)
        paths["paired_tests_csv"] = cmp_csv
    return paths
