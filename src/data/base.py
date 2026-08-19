"""Core data containers shared by every benchmark.

A benchmark is any source that can produce a :class:`CausalDataset`. Adding a
new benchmark (e.g. ACIC) therefore requires implementing a loader that returns
these containers -- no change to models, metrics or the experiment runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CausalDataset:
    """Observed data plus (for semi-synthetic benchmarks) ground truth.

    Attributes:
        x: Covariates, shape ``(n, d)``.
        t: Binary treatment indicator, shape ``(n,)``.
        y: Factual outcome ``Y = Y(T)``, shape ``(n,)``.
        mu0: True ``E[Y(0) | X]``, shape ``(n,)``. None for real data.
        mu1: True ``E[Y(1) | X]``, shape ``(n,)``. None for real data.
        propensity: True ``P(T=1 | X)`` when the DGP is known.
    """

    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    mu0: Optional[np.ndarray] = None
    mu1: Optional[np.ndarray] = None
    propensity: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        if self.x.ndim != 2:
            raise ValueError(f"x must be 2-D (n, d), got shape {self.x.shape}")
        self.t = np.asarray(self.t, dtype=np.float64).ravel()
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        n = self.x.shape[0]
        if self.t.shape[0] != n or self.y.shape[0] != n:
            raise ValueError(
                f"x ({n}), t ({self.t.shape[0]}) and y ({self.y.shape[0]}) "
                "must have the same number of rows"
            )
        if not np.all(np.isin(self.t, (0.0, 1.0))):
            raise ValueError("t must be binary (0/1)")
        for name in ("mu0", "mu1", "propensity"):
            val = getattr(self, name)
            if val is not None:
                val = np.asarray(val, dtype=np.float64).ravel()
                if val.shape[0] != n:
                    raise ValueError(f"{name} must have length {n}, got {val.shape[0]}")
                setattr(self, name, val)

    @property
    def n(self) -> int:
        """Number of units."""
        return self.x.shape[0]

    @property
    def d(self) -> int:
        """Number of covariates."""
        return self.x.shape[1]

    @property
    def has_ground_truth(self) -> bool:
        """True when individual-level potential outcomes are known."""
        return self.mu0 is not None and self.mu1 is not None

    @property
    def tau_true(self) -> np.ndarray:
        """True individual treatment effect ``mu1(x) - mu0(x)``."""
        if not self.has_ground_truth:
            raise ValueError("dataset has no ground-truth potential outcomes")
        return self.mu1 - self.mu0

    @property
    def ate_true(self) -> float:
        """True sample average treatment effect."""
        return float(np.mean(self.tau_true))

    @property
    def treated_fraction(self) -> float:
        """Empirical P(T=1)."""
        return float(np.mean(self.t))

    def subset(self, idx: np.ndarray) -> "CausalDataset":
        """Return the sub-dataset selected by index array ``idx``."""
        idx = np.asarray(idx)
        return CausalDataset(
            x=self.x[idx],
            t=self.t[idx],
            y=self.y[idx],
            mu0=None if self.mu0 is None else self.mu0[idx],
            mu1=None if self.mu1 is None else self.mu1[idx],
            propensity=None if self.propensity is None else self.propensity[idx],
        )


@dataclass
class DataSplit:
    """A train/test pair drawn from the same data-generating process."""

    train: CausalDataset
    test: CausalDataset
    meta: dict

    def __post_init__(self) -> None:
        if self.train.d != self.test.d:
            raise ValueError(
                f"train ({self.train.d}) and test ({self.test.d}) covariate "
                "dimensions must match"
            )
