"""Treatment-effect evaluation metrics.

This is an estimation problem, not a classification problem: the primary
metrics are error in the estimated individual and average treatment effects.
Classification accuracy is deliberately absent.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _check(tau_hat: np.ndarray, tau_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau_hat = np.asarray(tau_hat, dtype=np.float64).ravel()
    tau_true = np.asarray(tau_true, dtype=np.float64).ravel()
    if tau_hat.shape != tau_true.shape:
        raise ValueError(
            f"shape mismatch: tau_hat {tau_hat.shape} vs tau_true {tau_true.shape}"
        )
    if tau_hat.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    return tau_hat, tau_true


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Precision in Estimation of Heterogeneous Effect (root form).

    ``sqrt( mean_i (tau_hat(x_i) - tau(x_i))^2 )``. Lower is better. This is
    the square root of the usual "PEHE" quantity, which is the convention used
    by the IHDP literature.
    """
    tau_hat, tau_true = _check(tau_hat, tau_true)
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def abs_ate_error(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Absolute error of the average treatment effect, ``|E[tau_hat] - E[tau]|``."""
    tau_hat, tau_true = _check(tau_hat, tau_true)
    return float(abs(np.mean(tau_hat) - np.mean(tau_true)))


def cate_mae(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Mean absolute error of individual-level effect estimates."""
    tau_hat, tau_true = _check(tau_hat, tau_true)
    return float(np.mean(np.abs(tau_hat - tau_true)))


def cate_bias(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Signed mean error, ``E[tau_hat - tau]``; reveals systematic over/under-estimation."""
    tau_hat, tau_true = _check(tau_hat, tau_true)
    return float(np.mean(tau_hat - tau_true))


def policy_value(
    tau_hat: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, threshold: float = 0.0
) -> float:
    """Expected outcome of the treat-if-``tau_hat > threshold`` policy.

    With known response surfaces the value of policy ``pi`` is
    ``E[ pi(x) mu1(x) + (1 - pi(x)) mu0(x) ]`` -- computed exactly, not
    estimated by inverse propensity weighting. Higher is better.
    """
    tau_hat = np.asarray(tau_hat, dtype=np.float64).ravel()
    mu0 = np.asarray(mu0, dtype=np.float64).ravel()
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()
    if not (tau_hat.shape == mu0.shape == mu1.shape):
        raise ValueError("tau_hat, mu0 and mu1 must have the same shape")
    treat = (tau_hat > threshold).astype(np.float64)
    return float(np.mean(treat * mu1 + (1.0 - treat) * mu0))


def optimal_policy_value(mu0: np.ndarray, mu1: np.ndarray) -> float:
    """Value of the oracle policy that treats exactly when ``mu1 > mu0``."""
    mu0 = np.asarray(mu0, dtype=np.float64).ravel()
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()
    return float(np.mean(np.maximum(mu0, mu1)))


def policy_regret(tau_hat: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> float:
    """Value gap to the oracle policy. Zero is optimal; always non-negative."""
    return optimal_policy_value(mu0, mu1) - policy_value(tau_hat, mu0, mu1)


def evaluate_cate(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    mu0: Optional[np.ndarray] = None,
    mu1: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full metric suite for one model on one dataset.

    Policy metrics are included only when the response surfaces are supplied.
    """
    out = {
        "pehe": pehe(tau_hat, tau_true),
        "abs_ate_error": abs_ate_error(tau_hat, tau_true),
        "cate_mae": cate_mae(tau_hat, tau_true),
        "cate_bias": cate_bias(tau_hat, tau_true),
        "ate_true": float(np.mean(tau_true)),
        "ate_hat": float(np.mean(tau_hat)),
    }
    if mu0 is not None and mu1 is not None:
        out["policy_value"] = policy_value(tau_hat, mu0, mu1)
        out["policy_regret"] = policy_regret(tau_hat, mu0, mu1)
        out["optimal_policy_value"] = optimal_policy_value(mu0, mu1)
    return out


def subgroup_errors(
    tau_hat: np.ndarray, tau_true: np.ndarray, n_bins: int = 4
) -> Dict[str, float]:
    """PEHE within quantile bins of the *true* effect.

    Reveals whether a model is accurate on average but fails on the units with
    the largest (or smallest) true effects -- the subgroups a decision-maker
    most cares about.
    """
    tau_hat, tau_true = _check(tau_hat, tau_true)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.quantile(tau_true, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = np.inf
    out: Dict[str, float] = {}
    for b in range(n_bins):
        mask = (tau_true >= edges[b]) & (tau_true < edges[b + 1])
        out[f"pehe_tau_q{b + 1}"] = (
            pehe(tau_hat[mask], tau_true[mask]) if mask.sum() > 0 else float("nan")
        )
    return out
