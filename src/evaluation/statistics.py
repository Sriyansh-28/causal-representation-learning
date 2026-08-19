"""Statistical aggregation: dispersion, confidence intervals, paired tests.

Nothing here reports "significance" unless a test was actually run, and every
test used is paired, because all models are evaluated on identical replicates.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(nan, nan)`` for fewer than two observations, since dispersion is
    undefined there.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def summarize(values: Sequence[float], seed: int = 0) -> Dict[str, float]:
    """Mean, SD, SEM, n and a bootstrap 95% CI for a set of replicate values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "n": 0,
                "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = bootstrap_ci(v, seed=seed)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "sem": float(stats.sem(v)) if v.size > 1 else 0.0,
        "n": int(v.size),
        "ci_low": lo,
        "ci_high": hi,
    }


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired replicates ``a`` vs ``b``.

    The Wilcoxon test is used rather than a paired t-test because replicate
    counts are small and per-replicate errors are right-skewed. Returns NaNs
    when there are too few pairs or all differences are zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired_test requires equal-length inputs")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    out = {
        "n_pairs": int(a.size),
        "mean_diff": float(np.mean(a - b)) if a.size else float("nan"),
        "median_diff": float(np.median(a - b)) if a.size else float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
    }
    if a.size < 5 or np.allclose(a, b):
        return out
    try:
        res = stats.wilcoxon(a, b)
        out["statistic"] = float(res.statistic)
        out["p_value"] = float(res.pvalue)
    except ValueError:
        pass
    return out


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down correction.

    Returns a per-hypothesis rejection flag controlling the family-wise error
    rate at ``alpha``. NaN p-values are never rejected.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    reject = [False] * m
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            break
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject
