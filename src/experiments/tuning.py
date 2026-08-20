"""Experiment 7: equal-budget model selection for every estimator.

Experiments 1-6 gave the neural model adaptive capacity control (early
stopping on a validation split) while the meta-learners ran on fixed default
hyper-parameters. That asymmetry means those results compare *configured
estimators* rather than isolating representation learning. This module removes
the asymmetry by putting all four estimators through one identical protocol.

Protocol
--------
For each (scenario, seed):

1. Split the training set once into a fitting part and a validation part, with
   the **same split** handed to every model.
2. Fit each of ``N_CONFIGS`` candidate configurations on the fitting part.
   ``N_CONFIGS`` is identical for all four estimators.
3. Score every candidate by **factual validation MSE** -- the mean squared
   error of the predicted outcome under each unit's *observed* treatment.
4. Select the lowest-scoring configuration.
5. Refit that configuration on the **full** training set and evaluate on test.

Why factual validation MSE
--------------------------
PEHE cannot be used to select models: it requires counterfactual outcomes,
which no practitioner has. Selecting on PEHE would be oracle selection and
would leak the evaluation target into training. Factual outcome error is the
only criterion computable from observable data, so it is what every model is
selected on here. It is an imperfect proxy for CATE accuracy -- that
limitation applies equally to all four estimators, which is the point.

Fairness controls
-----------------
* Identical candidate count per model (``N_CONFIGS``).
* Identical validation split, seed, and selection criterion.
* The neural model's internal early stopping is **disabled**; its epoch budget
  becomes a tuned hyper-parameter like any other. No model gets adaptive
  stopping that the others lack.
* Wall-clock fit time is recorded per model so the realised compute budget can
  be compared rather than assumed equal.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

from ..learners.base import BaseCATELearner
from .registry import build_learner

# Every estimator receives exactly this many candidate configurations.
N_CONFIGS = 6

# Gradient-boosting grids for the meta-learners. All three share one grid so
# differences between them remain attributable to the meta-learning strategy.
# Ordered by roughly increasing capacity. A pilot run selected the smallest
# configuration in every scenario, i.e. the grid's boundary; a lower-capacity
# option was added so the selected region is bracketed rather than clipped.
# A grid whose optimum sits at its edge would under-serve these models and
# undermine the equal-budget claim.
_GBM_GRID: List[Dict[str, Any]] = [
    {"base_learner": "gbm", "n_estimators": 50, "max_depth": 2, "learning_rate": 0.05},
    {"base_learner": "gbm", "n_estimators": 100, "max_depth": 2, "learning_rate": 0.10},
    {"base_learner": "gbm", "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    {"base_learner": "gbm", "n_estimators": 200, "max_depth": 2, "learning_rate": 0.05},
    {"base_learner": "gbm", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.10},
    {"base_learner": "gbm", "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05},
]

# Neural grid spanning the ranges explored in Experiment 6. Early stopping is
# off throughout, so ``epochs`` is a genuine tuned hyper-parameter.
_NEURAL_GRID: List[Dict[str, Any]] = [
    {"latent_dim": 16, "lr": 0.001, "weight_decay": 1e-4, "epochs": 100, "dropout": 0.1},
    {"latent_dim": 32, "lr": 0.001, "weight_decay": 1e-4, "epochs": 200, "dropout": 0.1},
    {"latent_dim": 16, "lr": 0.005, "weight_decay": 1e-3, "epochs": 100, "dropout": 0.1},
    {"latent_dim": 32, "lr": 0.005, "weight_decay": 1e-3, "epochs": 200, "dropout": 0.0},
    {"latent_dim": 8, "lr": 0.001, "weight_decay": 1e-2, "epochs": 200, "dropout": 0.2},
    {"latent_dim": 8, "lr": 0.0005, "weight_decay": 1e-2, "epochs": 400, "dropout": 0.3},
]

TUNING_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "s_learner": _GBM_GRID,
    "t_learner": _GBM_GRID,
    "x_learner": _GBM_GRID,
    "neural_rep": _NEURAL_GRID,
}

for _name, _grid in TUNING_GRIDS.items():
    if len(_grid) != N_CONFIGS:  # pragma: no cover - guards a config edit mistake
        raise ValueError(f"{_name} has {len(_grid)} configs, expected {N_CONFIGS}")


def factual_val_mse(
    learner: BaseCATELearner, x: np.ndarray, t: np.ndarray, y: np.ndarray
) -> float:
    """Mean squared error of predicted outcomes under the observed treatment."""
    pred = learner.predict_factual(x, t)
    return float(np.mean((pred - np.asarray(y, dtype=np.float64).ravel()) ** 2))


def _neural_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Neural candidate settings with adaptive stopping disabled."""
    out = dict(cfg)
    out["early_stopping"] = False
    out["val_fraction"] = 0.0
    return out


def split_train_val(
    n: int, seed: int, val_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fit_idx, val_idx) for one replicate.

    Depends only on ``n`` and ``seed``, so every model in a given cell receives
    exactly the same split.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(2, int(round(val_fraction * n)))
    return idx[n_val:], idx[:n_val]


def tune_and_fit(
    model_name: str,
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    seed: int,
    neural_cfg: Dict[str, Any],
) -> Tuple[BaseCATELearner, Dict[str, Any]]:
    """Select a configuration by factual validation MSE, then refit on all data.

    Args:
        model_name: Registry name of the estimator.
        x, t, y: Training covariates, treatment and factual outcome.
        seed: Replicate seed; also fixes the shared train/validation split.
        neural_cfg: Base neural settings that candidates override.

    Returns:
        The refitted learner and a record of the tuning run: the selected
        configuration, its validation score, every candidate's score, the
        number of candidates evaluated and total tuning wall-clock seconds.

    Raises:
        RuntimeError: If every candidate configuration failed to fit.
    """
    grid = TUNING_GRIDS.get(model_name)
    if grid is None:
        raise ValueError(f"no tuning grid defined for model '{model_name}'")

    fit_idx, val_idx = split_train_val(x.shape[0], seed)
    x_fit, t_fit, y_fit = x[fit_idx], t[fit_idx], y[fit_idx]
    x_val, t_val, y_val = x[val_idx], t[val_idx], y[val_idx]

    scores: List[float] = []
    started = time.time()
    for cfg in grid:
        overrides = _neural_overrides(cfg) if model_name == "neural_rep" else dict(cfg)
        try:
            cand = build_learner(model_name, seed, neural_cfg, **overrides)
            cand.fit(x_fit, t_fit, y_fit)
            scores.append(factual_val_mse(cand, x_val, t_val, y_val))
        except Exception:
            scores.append(float("inf"))  # candidate is unusable; never selected
    tuning_seconds = time.time() - started

    if not np.isfinite(scores).any():
        raise RuntimeError(f"all {len(grid)} candidates failed for {model_name}")

    best = int(np.argmin(scores))
    chosen = grid[best]
    overrides = _neural_overrides(chosen) if model_name == "neural_rep" else dict(chosen)

    refit_started = time.time()
    learner = build_learner(model_name, seed, neural_cfg, **overrides)
    learner.fit(x, t, y)
    refit_seconds = time.time() - refit_started

    record = {
        "selected_config": chosen,
        "selected_index": best,
        "val_mse_selected": scores[best],
        "val_mse_all": scores,
        "n_configs_evaluated": len(grid),
        "tuning_seconds": tuning_seconds,
        "refit_seconds": refit_seconds,
    }
    return learner, record
