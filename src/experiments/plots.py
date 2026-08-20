"""Publication-style figures generated from raw experiment results.

Every figure is produced from a results DataFrame; nothing is drawn by hand.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.paths import ensure_dir

MODEL_LABELS = {
    "s_learner": "S-Learner",
    "t_learner": "T-Learner",
    "x_learner": "X-Learner",
    "neural_rep": "NeuralRep",
}
MODEL_ORDER = ["s_learner", "t_learner", "x_learner", "neural_rep"]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
        "axes.spines.right": False, "figure.autolayout": True,
    })


def _ordered_models(df: pd.DataFrame) -> List[str]:
    present = [m for m in MODEL_ORDER if m in set(df["model"])]
    return present + sorted(set(df["model"]) - set(present))


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def plot_metric_bars(
    agg: pd.DataFrame, metric: str, out_path: Path, title: Optional[str] = None
) -> Optional[Path]:
    """Bar chart of one metric per model with SD error bars (single condition)."""
    _style()
    sub = agg[agg["metric"] == metric]
    if sub.empty:
        return None
    models = _ordered_models(sub)
    means = [float(sub[sub["model"] == m]["mean"].iloc[0]) for m in models]
    stds = [float(sub[sub["model"] == m]["std"].iloc[0]) for m in models]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(range(len(models)), means, yerr=stds, capsize=4,
           color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(models)])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_label(m) for m in models], rotation=15)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by model (mean ± SD across seeds)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metric_vs_condition(
    agg: pd.DataFrame, metric: str, out_path: Path,
    xlabel: str, title: Optional[str] = None, logx: bool = False,
) -> Optional[Path]:
    """Line plot of one metric against a swept condition, one line per model."""
    _style()
    sub = agg[agg["metric"] == metric]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for model in _ordered_models(sub):
        ms = sub[sub["model"] == model].copy()
        ms["condition_value"] = pd.to_numeric(ms["condition_value"], errors="coerce")
        ms = ms.dropna(subset=["condition_value"]).sort_values("condition_value")
        if ms.empty:
            continue
        x = ms["condition_value"].to_numpy()
        y = ms["mean"].to_numpy()
        lo = ms["ci_low"].to_numpy()
        hi = ms["ci_high"].to_numpy()
        ax.plot(x, y, marker="o", label=_label(model))
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ax.fill_between(x, lo, hi, alpha=0.15)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {xlabel}")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ablation(agg: pd.DataFrame, metric: str, out_path: Path) -> Optional[Path]:
    """Horizontal bars comparing neural-model ablation variants."""
    _style()
    sub = agg[(agg["metric"] == metric) & (agg["condition"] == "ablation")]
    if sub.empty:
        return None
    sub = sub.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(range(len(sub)), sub["mean"], xerr=sub["std"], capsize=4, color="#4C72B0")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["condition_value"])
    ax.set_xlabel(metric)
    ax.set_title(f"Neural model ablation: {metric} (mean ± SD)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sensitivity(agg: pd.DataFrame, metric: str, out_dir: Path) -> List[Path]:
    """One panel per swept hyper-parameter."""
    _style()
    sub = agg[agg["metric"] == metric]
    if sub.empty:
        return []
    written: List[Path] = []
    for param, grp in sub.groupby("condition"):
        grp = grp.copy()
        grp["condition_value"] = pd.to_numeric(grp["condition_value"], errors="coerce")
        grp = grp.dropna(subset=["condition_value"]).sort_values("condition_value")
        if grp.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.errorbar(grp["condition_value"], grp["mean"], yerr=grp["std"],
                    marker="o", capsize=4, color="#C44E52")
        if param in {"weight_decay", "lr"}:
            ax.set_xscale("log")
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f"Sensitivity to {param}")
        path = out_dir / f"sensitivity_{param}_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def plot_cate_error_distribution(df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """Box plot of per-replicate PEHE, showing spread rather than just the mean."""
    _style()
    if "pehe" not in df.columns or df.empty:
        return None
    models = _ordered_models(df)
    data = [df[df["model"] == m]["pehe"].dropna().to_numpy() for m in models]
    data = [d for d in data if d.size]
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    # `tick_labels` replaced `labels` in matplotlib 3.9; set them separately so
    # the figure code works on both.
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([_label(m) for m in models][: len(data)])
    ax.set_ylabel("PEHE (log scale)")
    # Per-replicate PEHE spans orders of magnitude on IHDP; on a linear axis the
    # outlier replicates flatten every box into an unreadable sliver.
    if np.min(np.concatenate(data)) > 0:
        ax.set_yscale("log")
    ax.set_title("Per-replicate PEHE distribution")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grouped_bars(
    agg: pd.DataFrame, metric: str, out_path: Path, title: Optional[str] = None
) -> Optional[Path]:
    """Grouped bars: one cluster per scenario, one bar per model, SD whiskers."""
    _style()
    sub = agg[agg["metric"] == metric]
    if sub.empty:
        return None
    scenarios = list(dict.fromkeys(sub["condition_value"]))
    models = _ordered_models(sub)
    width = 0.8 / max(len(models), 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(scenarios)), 4.0))
    for i, model in enumerate(models):
        means, stds = [], []
        for scen in scenarios:
            row = sub[(sub["model"] == model) & (sub["condition_value"] == scen)]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["std"].iloc[0]) if not row.empty else np.nan)
        positions = np.arange(len(scenarios)) + i * width - 0.4 + width / 2
        ax.bar(positions, means, width=width, yerr=stds, capsize=3,
               label=_label(model), color=colors[i % len(colors)])
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by scenario")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_figures(
    df: pd.DataFrame, agg: pd.DataFrame, experiment: str, name: str, out_dir: Path
) -> List[Path]:
    """Produce the figure set appropriate to ``experiment``."""
    fig_dir = ensure_dir(out_dir / "figures")
    written: List[Path] = []

    def _add(p: Optional[Path]) -> None:
        if p is not None:
            written.append(p)

    if experiment == "baseline":
        _add(plot_metric_bars(agg, "pehe", fig_dir / f"{name}_pehe.png"))
        _add(plot_metric_bars(agg, "abs_ate_error", fig_dir / f"{name}_ate_error.png"))
        _add(plot_metric_bars(agg, "policy_regret", fig_dir / f"{name}_policy_regret.png"))
        _add(plot_cate_error_distribution(df, fig_dir / f"{name}_pehe_distribution.png"))
    elif experiment == "confounding":
        for metric in ("pehe", "abs_ate_error"):
            _add(plot_metric_vs_condition(
                agg, metric, fig_dir / f"{name}_{metric}.png",
                xlabel="confounding strength (gamma)"))
    elif experiment == "treatment_imbalance":
        for metric in ("pehe", "abs_ate_error"):
            _add(plot_metric_vs_condition(
                agg, metric, fig_dir / f"{name}_{metric}.png",
                xlabel="treated fraction P(T=1)"))
    elif experiment == "sample_size":
        for metric in ("pehe", "abs_ate_error"):
            _add(plot_metric_vs_condition(
                agg, metric, fig_dir / f"{name}_{metric}.png",
                xlabel="training-set fraction"))
    elif experiment == "ablation":
        _add(plot_ablation(agg, "pehe", fig_dir / f"{name}_pehe.png"))
        _add(plot_ablation(agg, "abs_ate_error", fig_dir / f"{name}_ate_error.png"))
    elif experiment == "fair_selection":
        for metric in ("pehe", "abs_ate_error", "policy_regret"):
            _add(plot_grouped_bars(
                agg, metric, fig_dir / f"{name}_{metric}.png",
                title=f"{metric} under equal-budget model selection"))
    elif experiment == "sensitivity":
        written.extend(plot_sensitivity(agg, "pehe", fig_dir))
    return written
