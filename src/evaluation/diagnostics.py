"""Propensity, overlap and covariate-balance diagnostics.

These characterise the *design* of each experimental condition -- how hard the
causal problem is -- independently of any outcome model.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from ..data.base import CausalDataset


def estimate_propensity(x: np.ndarray, t: np.ndarray, seed: int = 0) -> np.ndarray:
    """Estimate ``P(T=1 | X)`` with L2-regularised logistic regression."""
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    model.fit(x, t)
    return model.predict_proba(x)[:, 1]


def standardized_mean_differences(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-covariate standardized mean difference between treated and control.

    ``|mean_1 - mean_0| / sqrt((var_1 + var_0) / 2)``. Values above ~0.1 are
    conventionally read as meaningful imbalance.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t).ravel()
    x1, x0 = x[t == 1], x[t == 0]
    if x1.shape[0] < 2 or x0.shape[0] < 2:
        return np.full(x.shape[1], np.nan)
    pooled = np.sqrt((x1.var(axis=0, ddof=1) + x0.var(axis=0, ddof=1)) / 2.0)
    pooled = np.where(pooled < 1e-12, np.nan, pooled)
    return np.abs(x1.mean(axis=0) - x0.mean(axis=0)) / pooled


def overlap_diagnostics(dataset: CausalDataset, seed: int = 0) -> Dict[str, float]:
    """Summarise positivity/overlap and covariate balance for one dataset.

    Returns:
        Dictionary with the treated fraction, estimated-propensity summaries,
        the fraction of units in the extreme propensity tails (a positivity
        violation indicator), a KS statistic comparing the treated and control
        propensity distributions, and mean/max standardized mean differences.
    """
    ps = estimate_propensity(dataset.x, dataset.t, seed=seed)
    smd = standardized_mean_differences(dataset.x, dataset.t)
    treated_ps, control_ps = ps[dataset.t == 1], ps[dataset.t == 0]
    ks = (
        float(stats.ks_2samp(treated_ps, control_ps).statistic)
        if treated_ps.size > 1 and control_ps.size > 1
        else float("nan")
    )
    out = {
        "treated_fraction": dataset.treated_fraction,
        "ps_min": float(ps.min()),
        "ps_max": float(ps.max()),
        "ps_frac_below_0.1": float(np.mean(ps < 0.1)),
        "ps_frac_above_0.9": float(np.mean(ps > 0.9)),
        "ps_ks_treated_vs_control": ks,
        "smd_mean": float(np.nanmean(smd)),
        "smd_max": float(np.nanmax(smd)),
        "smd_frac_above_0.1": float(np.nanmean(smd > 0.1)),
    }
    if dataset.propensity is not None:
        out["true_ps_min"] = float(dataset.propensity.min())
        out["true_ps_max"] = float(dataset.propensity.max())
    return out
