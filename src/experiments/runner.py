"""Central experiment runner.

One code path executes all six experiments; they differ only in which
conditions they sweep. Every row of the output records the experiment, the
condition, the model, the replicate seed and the full metric suite, so results
can be re-aggregated later without re-running anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

from ..data.preprocessing import standardize_split, subsample_train
from ..evaluation.diagnostics import overlap_diagnostics
from ..evaluation.metrics import evaluate_cate, subgroup_errors
from ..utils.config import ExperimentConfig
from ..utils.paths import ensure_dir, resolve
from ..utils.seeding import set_seed
from .registry import ABLATION_VARIANTS, build_dataset, build_learner

# A condition is (label, value, dataset_overrides, learner_overrides, models_or_None).
Condition = Tuple[str, Any, Dict[str, Any], Dict[str, Any], List[str] | None]


def _conditions(config: ExperimentConfig) -> Iterator[Condition]:
    """Expand a config into the list of conditions its experiment sweeps."""
    exp, cond = config.experiment, config.conditions

    if exp == "baseline":
        yield ("baseline", "default", {}, {}, None)

    elif exp == "confounding":
        for gamma in cond.get("confounding_strengths", [0.0, 1.0, 2.0, 3.0]):
            yield ("confounding_strength", gamma, {"confounding_strength": gamma}, {}, None)

    elif exp == "treatment_imbalance":
        for frac in cond.get("treated_fractions", [0.5, 0.25, 0.1]):
            yield ("treated_fraction", frac, {"target_treated_fraction": frac}, {}, None)

    elif exp == "sample_size":
        for frac in cond.get("train_fractions", [0.25, 0.5, 0.75, 1.0]):
            yield ("train_fraction", frac, {}, {}, None)

    elif exp == "ablation":
        for variant in cond.get("variants", list(ABLATION_VARIANTS)):
            if variant not in ABLATION_VARIANTS:
                raise ValueError(
                    f"unknown ablation variant '{variant}'; "
                    f"expected one of {sorted(ABLATION_VARIANTS)}"
                )
            yield ("ablation", variant, {}, dict(ABLATION_VARIANTS[variant]), ["neural_rep"])

    elif exp == "sensitivity":
        sweeps: Dict[str, List[Any]] = cond.get("sweeps", {})
        if not sweeps:
            raise ValueError("sensitivity experiment requires a non-empty 'sweeps' block")
        for param, values in sweeps.items():
            for value in values:
                overrides: Dict[str, Any] = {param: value}
                # Epoch sweeps must not be short-circuited by early stopping,
                # or the number of epochs would not actually vary.
                if param == "epochs":
                    overrides["early_stopping"] = False
                yield (param, value, {}, overrides, ["neural_rep"])
    else:
        raise ValueError(f"unsupported experiment '{exp}'")


def run_single(
    config: ExperimentConfig,
    model_name: str,
    seed: int,
    condition_name: str,
    condition_value: Any,
    dataset_overrides: Dict[str, Any],
    learner_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one (model, seed, condition) cell and return a result row.

    Covariates are standardized on training statistics, and the outcome is
    standardized too; predicted effects are rescaled back to the original
    outcome scale before evaluation so all metrics are comparable across models
    and datasets.
    """
    set_seed(seed)
    split = build_dataset(config.dataset, seed, **dataset_overrides)

    if config.experiment == "sample_size":
        split = subsample_train(split, float(condition_value), seed=seed)

    raw_test = split.test
    scaled = standardize_split(split)
    y_std = scaled.meta["y_std"]

    started = time.time()
    learner = build_learner(model_name, seed, config.neural, **learner_overrides)
    learner.fit(scaled.train.x, scaled.train.t, scaled.train.y)
    # Undo outcome scaling: y' = (y - m)/s implies tau = tau' * s.
    tau_hat = learner.predict_cate(scaled.test.x) * y_std
    fit_seconds = time.time() - started

    row: Dict[str, Any] = {
        "experiment": config.experiment,
        "config_name": config.name,
        "dataset": config.dataset.get("name"),
        "condition": condition_name,
        "condition_value": condition_value,
        "model": model_name,
        "seed": seed,
        "n_train": int(scaled.train.n),
        "n_test": int(raw_test.n),
        "treated_fraction_train": float(scaled.train.treated_fraction),
        "fit_seconds": fit_seconds,
    }
    row.update(
        evaluate_cate(tau_hat, raw_test.tau_true, raw_test.mu0, raw_test.mu1)
    )
    row.update(subgroup_errors(tau_hat, raw_test.tau_true))
    row.update({f"diag_{k}": v for k, v in overlap_diagnostics(split.train, seed=seed).items()})
    return row


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> pd.DataFrame:
    """Execute every (condition, model, seed) cell defined by ``config``.

    A failure in one cell is recorded in the ``error`` column rather than
    aborting the run, so a single bad condition cannot destroy a long sweep.
    """
    rows: List[Dict[str, Any]] = []
    conditions = list(_conditions(config))
    total = sum(
        len(models or config.models) * len(config.seeds)
        for _, _, _, _, models in conditions
    )
    done = 0

    for cname, cvalue, ds_over, lr_over, models in conditions:
        for model_name in (models or config.models):
            for seed in config.seeds:
                done += 1
                try:
                    row = run_single(
                        config, model_name, seed, cname, cvalue, ds_over, lr_over
                    )
                    row["error"] = ""
                except Exception as exc:  # recorded, not swallowed
                    row = {
                        "experiment": config.experiment, "config_name": config.name,
                        "dataset": config.dataset.get("name"), "condition": cname,
                        "condition_value": cvalue, "model": model_name, "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if verbose:
                        print(f"  [ERROR] {model_name} seed={seed} {cname}={cvalue}: {exc}")
                rows.append(row)
                if verbose and (done % 10 == 0 or done == total):
                    print(f"  [{done}/{total}] {cname}={cvalue} {model_name} seed={seed}")

    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, config: ExperimentConfig) -> Path:
    """Write raw per-run results plus a provenance sidecar; return the CSV path."""
    out_dir = ensure_dir(resolve(config.output_dir) / "raw")
    csv_path = out_dir / f"{config.name}_raw.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "config": config.raw,
        "n_rows": int(len(df)),
        "n_errors": int((df.get("error", pd.Series(dtype=str)).fillna("") != "").sum()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (out_dir / f"{config.name}_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return csv_path
