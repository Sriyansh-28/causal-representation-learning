"""Controlled synthetic DGP with tunable confounding and treatment imbalance.

IHDP fixes both its confounding level and its treatment mechanism, so neither
can serve as an experimental factor. This module supplies a fully specified
generator in which confounding strength and the marginal treated fraction are
independent, documented knobs.

Data-generating process
-----------------------
Let ``d`` be the covariate dimension and ``k`` the number of confounders.

1. ``X ~ N(0, I_d)``
2. Propensity:   ``e(x) = sigmoid(a0 + gamma * (x[:, :k] @ w))``
3. Treatment:    ``T ~ Bernoulli(e(x))``          (SUTVA assumed)
4. Baseline:     ``mu0(x) = x @ beta + 0.5 * sin(pi * x[:, 0])``
5. Effect:       ``tau(x) = 1 + 0.5 * x[:, 0] + 0.25 * (x[:, 1] ** 2 - 1)``
6. ``mu1(x) = mu0(x) + tau(x)``
7. Outcome:      ``Y = mu_T(x) + N(0, sigma^2)``

``beta`` is non-zero on the first ``k`` covariates, so those covariates drive
*both* treatment and outcome: they are genuine confounders.

* ``gamma`` (``confounding_strength``) scales how strongly the confounders
  drive treatment. ``gamma = 0`` gives a randomized trial (no confounding, and
  perfect overlap); larger ``gamma`` pushes propensities toward 0/1, creating
  selection bias and degrading positivity.
* ``a0`` (``propensity_intercept``) shifts the marginal treated fraction
  without changing which covariates confound. It is solved numerically for a
  requested target fraction by :func:`solve_intercept_for_treated_fraction`,
  so imbalance can be varied while holding ``gamma`` fixed.

Because ``mu0`` and ``mu1`` are known in closed form, PEHE and ATE error are
exact rather than estimated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import CausalDataset, DataSplit


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class SyntheticConfig:
    """Parameters of the synthetic DGP.

    Attributes:
        n_train: Training units.
        n_test: Test units.
        d: Covariate dimension.
        n_confounders: Number of covariates affecting both T and Y.
        confounding_strength: ``gamma`` in the propensity model.
        target_treated_fraction: Desired marginal ``P(T=1)``; the intercept is
            solved to hit it. Ignored if ``propensity_intercept`` is given.
        propensity_intercept: Explicit ``a0`` (overrides the target fraction).
        noise_std: Outcome noise standard deviation ``sigma``.
        seed: Seed controlling both the fixed coefficients and the draw.
    """

    n_train: int = 2000
    n_test: int = 1000
    d: int = 20
    n_confounders: int = 5
    confounding_strength: float = 1.0
    target_treated_fraction: float = 0.5
    propensity_intercept: Optional[float] = None
    noise_std: float = 1.0
    seed: int = 0
    _validated: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_confounders > self.d:
            raise ValueError(
                f"n_confounders ({self.n_confounders}) cannot exceed d ({self.d})"
            )
        if self.confounding_strength < 0:
            raise ValueError("confounding_strength must be non-negative")
        if not 0.0 < self.target_treated_fraction < 1.0:
            raise ValueError("target_treated_fraction must lie in (0, 1)")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


def _coefficients(cfg: SyntheticConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(w, beta)``: treatment-model and outcome-model coefficients.

    Coefficients depend only on ``d``/``n_confounders`` and a fixed structural
    seed, so different replicate seeds share the same DGP and differ only in
    the sampled data. That is what makes replicates comparable.
    """
    rng = np.random.default_rng(12345)  # structural seed, deliberately fixed
    k = cfg.n_confounders
    w = np.zeros(cfg.d)
    w[:k] = rng.normal(0.0, 1.0, size=k)
    w[:k] /= np.linalg.norm(w[:k])  # unit norm so gamma alone sets the scale
    beta = np.zeros(cfg.d)
    beta[:k] = rng.uniform(0.5, 1.5, size=k)          # confounders affect Y
    beta[k:] = rng.uniform(-0.5, 0.5, size=cfg.d - k)  # pure outcome predictors
    return w, beta


def solve_intercept_for_treated_fraction(
    cfg: SyntheticConfig, tol: float = 1e-4, max_iter: int = 200
) -> float:
    """Bisect for the intercept ``a0`` giving the target marginal ``P(T=1)``.

    The expectation is evaluated on a large Monte-Carlo sample drawn from the
    covariate distribution, independent of the experiment sample.
    """
    w, _ = _coefficients(cfg)
    rng = np.random.default_rng(999)
    x = rng.normal(size=(200_000, cfg.d))
    logits = cfg.confounding_strength * (x @ w)
    lo, hi = -20.0, 20.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        frac = float(np.mean(sigmoid(logits + mid)))
        if abs(frac - cfg.target_treated_fraction) < tol:
            return mid
        if frac < cfg.target_treated_fraction:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _draw(cfg: SyntheticConfig, n: int, a0: float, rng: np.random.Generator) -> CausalDataset:
    w, beta = _coefficients(cfg)
    x = rng.normal(size=(n, cfg.d))
    e = sigmoid(a0 + cfg.confounding_strength * (x @ w))
    t = (rng.uniform(size=n) < e).astype(np.float64)
    mu0 = x @ beta + 0.5 * np.sin(np.pi * x[:, 0])
    tau = 1.0 + 0.5 * x[:, 0] + 0.25 * (x[:, 1] ** 2 - 1.0)
    mu1 = mu0 + tau
    y = np.where(t == 1.0, mu1, mu0) + rng.normal(0.0, cfg.noise_std, size=n)
    return CausalDataset(x=x, t=t, y=y, mu0=mu0, mu1=mu1, propensity=e)


def make_synthetic(cfg: SyntheticConfig) -> DataSplit:
    """Generate a train/test split from the synthetic DGP.

    Both splits come from the same distribution; only the random draw differs.
    """
    a0 = (
        cfg.propensity_intercept
        if cfg.propensity_intercept is not None
        else solve_intercept_for_treated_fraction(cfg)
    )
    rng = np.random.default_rng(cfg.seed)
    train = _draw(cfg, cfg.n_train, a0, rng)
    test = _draw(cfg, cfg.n_test, a0, rng)
    meta = {
        "dataset": "synthetic",
        "seed": cfg.seed,
        "d": cfg.d,
        "n_confounders": cfg.n_confounders,
        "confounding_strength": cfg.confounding_strength,
        "propensity_intercept": float(a0),
        "target_treated_fraction": cfg.target_treated_fraction,
        "treated_fraction_train": train.treated_fraction,
        "n_train": train.n,
        "n_test": test.n,
    }
    return DataSplit(train=train, test=test, meta=meta)
