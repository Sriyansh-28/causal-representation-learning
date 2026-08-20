#!/usr/bin/env python3
"""Generate the README's results section directly from committed result CSVs.

Every number in the README is produced by this script from files under
``results/raw/``. Nothing is typed by hand, and any experiment whose CSV is
absent is rendered as an explicit PENDING line rather than being omitted or
guessed at.

Usage::

    python scripts/build_results_section.py > results/tables/README_results.md
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.aggregate import (  # noqa: E402
    ablation_comparisons, aggregate_results, paired_comparisons,
)
from src.utils.paths import PROJECT_ROOT  # noqa: E402

RAW_DIR = PROJECT_ROOT / "results" / "raw"

# (config name, human heading, experiment number)
EXPERIMENTS: List[tuple[str, str, str]] = [
    ("baseline_ihdp", "Experiment 1a — baseline comparison (IHDP)", "1a"),
    ("baseline_synthetic", "Experiment 1b — baseline comparison (synthetic)", "1b"),
    ("confounding", "Experiment 2 — confounding strength (synthetic)", "2"),
    ("treatment_imbalance", "Experiment 3 — treatment imbalance (synthetic)", "3"),
    ("sample_size", "Experiment 4a — training sample size (IHDP)", "4a"),
    ("sample_size_synthetic", "Experiment 4b — training sample size (synthetic)", "4b"),
    ("ablation", "Experiment 5a — neural component ablation (IHDP)", "5a"),
    ("ablation_synthetic", "Experiment 5b — neural component ablation (synthetic)", "5b"),
    ("sensitivity", "Experiment 6 — hyper-parameter sensitivity (IHDP)", "6"),
]

MODEL_LABEL = {
    "s_learner": "S-Learner", "t_learner": "T-Learner",
    "x_learner": "X-Learner", "neural_rep": "NeuralRep",
}
MODEL_ORDER = ["s_learner", "t_learner", "x_learner", "neural_rep"]


def load_raw(name: str) -> Optional[pd.DataFrame]:
    """Load one experiment's raw CSV, or None if it has not been run."""
    path = RAW_DIR / f"{name}_raw.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""]
    return df if not df.empty else None


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def metric_table(agg: pd.DataFrame, metric: str, condition_label: str) -> str:
    """Render a condition-by-model table of ``mean ± SD (median)`` cells."""
    sub = agg[agg["metric"] == metric]
    if sub.empty:
        return ""
    models = [m for m in MODEL_ORDER if m in set(sub["model"])]
    models += sorted(set(sub["model"]) - set(models))

    values = sorted(sub["condition_value"].unique(), key=lambda v: (str(type(v)), str(v)))
    header = f"| {condition_label} | " + " | ".join(MODEL_LABEL.get(m, m) for m in models) + " |"
    rule = "|---" * (len(models) + 1) + "|"
    lines = [header, rule]
    for val in values:
        cells = []
        for m in models:
            row = sub[(sub["model"] == m) & (sub["condition_value"] == val)]
            if row.empty:
                cells.append("—")
                continue
            r = row.iloc[0]
            cells.append(f"{_fmt(r['mean'])} ± {_fmt(r['std'])} ({_fmt(r['median'])})")
        lines.append(f"| {val} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def significance_block(df: pd.DataFrame, metric: str = "pehe") -> str:
    """Render Holm-corrected paired Wilcoxon results for NeuralRep vs baselines."""
    cmp_df = paired_comparisons(df, reference="neural_rep", metric=metric)
    if cmp_df.empty:
        return ""
    lines = [
        "| condition | NeuralRep vs | mean PEHE diff | p (Wilcoxon) | reject H0 (Holm 0.05) |",
        "|---|---|---|---|---|",
    ]
    for _, r in cmp_df.iterrows():
        lines.append(
            f"| {r['condition_value']} | {MODEL_LABEL.get(r['comparison'], r['comparison'])} "
            f"| {_fmt(r['mean_diff'])} | {_fmt(r['p_value'], 4)} "
            f"| {'yes' if r['reject_h0_holm_0.05'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def condition_label(df: pd.DataFrame) -> str:
    """Human-readable name for whatever the experiment swept."""
    cond = str(df["condition"].iloc[0])
    return {
        "baseline": "condition",
        "confounding_strength": "confounding γ",
        "treated_fraction": "treated fraction",
        "train_fraction": "train fraction",
        "ablation": "variant",
    }.get(cond, cond)


def diagnostics_table(df: pd.DataFrame) -> str:
    """Per-condition design diagnostics, averaged over seeds and models."""
    cols = {
        "ate_true": "true ATE",
        "diag_treated_fraction": "treated frac",
        "diag_smd_max": "max SMD",
        "diag_ps_frac_below_0.1": "P(e<0.1)",
        "diag_ps_frac_above_0.9": "P(e>0.9)",
        "diag_ps_ks_treated_vs_control": "PS KS",
    }
    present = [c for c in cols if c in df.columns]
    if not present:
        return ""
    grp = df.groupby("condition_value")[present].mean().reset_index()
    header = "| condition | " + " | ".join(cols[c] for c in present) + " |"
    rule = "|---" * (len(present) + 1) + "|"
    lines = [header, rule]
    for _, r in grp.iterrows():
        lines.append(
            f"| {r['condition_value']} | "
            + " | ".join(_fmt(r[c]) for c in present) + " |"
        )
    return "\n".join(lines) + "\n"


def render(name: str, heading: str) -> str:
    """Render one experiment's full block, or a PENDING notice."""
    df = load_raw(name)
    if df is None:
        return (
            f"### {heading}\n\n"
            f"**PENDING — not yet run.** No results file at "
            f"`results/raw/{name}_raw.csv`.\n"
        )

    agg = aggregate_results(df)
    label = condition_label(df)
    n_seeds = df["seed"].nunique()
    parts = [
        f"### {heading}\n",
        f"\nReplicates: {n_seeds} seeds. "
        f"Cells are `mean ± SD (median)` across seeds; lower is better for "
        f"PEHE, ATE error and policy regret.\n\n",
    ]
    for metric, title in [
        ("pehe", "PEHE"),
        ("abs_ate_error", "Absolute ATE error"),
        ("policy_regret", "Policy regret"),
    ]:
        table = metric_table(agg, metric, label)
        if table:
            parts.append(f"**{title}**\n\n{table}\n")

    if df["model"].nunique() > 1:
        sig = significance_block(df)
        if sig:
            parts.append(
                "**Paired Wilcoxon signed-rank tests on PEHE** "
                "(negative difference favours NeuralRep; Holm-corrected within "
                f"each condition)\n\n{sig}\n"
            )

    diag = diagnostics_table(df)
    if diag and df["condition"].iloc[0] != "ablation":
        parts.append(
            "**Manipulation check / design diagnostics** (mean over seeds). "
            "`true ATE` should stay constant across conditions — if it moves, "
            "error differences would be confounded with a shifting estimand.\n\n"
            f"{diag}\n"
        )

    if df["condition"].iloc[0] == "ablation":
        parts.append(ablation_delta_block(df))
    return "".join(parts)


def ablation_delta_block(df: pd.DataFrame) -> str:
    """Render each ablation variant's change relative to the full model."""
    frames = []
    for metric in ("pehe", "abs_ate_error", "policy_regret"):
        cmp_df = ablation_comparisons(df, metric=metric)
        if not cmp_df.empty:
            frames.append(cmp_df)
    if not frames:
        return ""
    lines = [
        "**Change relative to the full model** (positive = worse than full, "
        "i.e. the removed component was contributing). Paired Wilcoxon over "
        "shared seeds, Holm-corrected within each metric.\n\n",
        "| metric | variant | mean | full | Δ vs full | % change | p | reject H0 (Holm) |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for frame in frames:
        for _, r in frame.iterrows():
            lines.append(
                f"| {r['metric']} | {r['variant']} | {_fmt(r['mean'])} | "
                f"{_fmt(r['full_mean'])} | {_fmt(r['delta_vs_full'])} | "
                f"{_fmt(r['pct_change_vs_full'], 1)}% | {_fmt(r['p_value'], 4)} | "
                f"{'yes' if r['reject_h0_holm_0.05'] else 'no'} |\n"
            )
    return "".join(lines) + "\n"


def sensitivity_block() -> str:
    """Render the hyper-parameter sensitivity sweep, one table per parameter."""
    df = load_raw("sensitivity")
    if df is None:
        return (
            "### Experiment 6 — hyper-parameter sensitivity (IHDP)\n\n"
            "**PENDING — not yet run.** No results file at "
            "`results/raw/sensitivity_raw.csv`.\n"
        )
    agg = aggregate_results(df)
    parts = [
        "### Experiment 6 — hyper-parameter sensitivity (IHDP)\n",
        f"\nReplicates: {df['seed'].nunique()} seeds. One parameter varied at a "
        "time around the baseline setting. Cells are PEHE `mean ± SD (median)`.\n",
    ]
    for param, grp in agg[agg["metric"] == "pehe"].groupby("condition"):
        grp = grp.sort_values("condition_value")
        parts.append(f"\n**{param}**\n\n| value | PEHE |\n|---|---|\n")
        for _, r in grp.iterrows():
            parts.append(
                f"| {r['condition_value']} | {_fmt(r['mean'])} ± "
                f"{_fmt(r['std'])} ({_fmt(r['median'])}) |\n"
            )
    return "".join(parts)


def main() -> int:
    """Write the assembled results section to stdout."""
    blocks: List[str] = []
    for name, heading, _ in EXPERIMENTS:
        blocks.append(sensitivity_block() if name == "sensitivity" else render(name, heading))
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
