"""Preprocessing utilities.

All statistics are estimated on the training split only and then applied to the
test split, so no test information leaks into training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import CausalDataset, DataSplit


@dataclass
class Standardizer:
    """Zero-mean/unit-variance scaler fitted on training covariates."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        """Fit column-wise mean and standard deviation.

        Constant columns get scale 1.0 so they map to 0 instead of NaN.
        """
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"expected {self.mean.shape[0]} columns, got {x.shape[1]}"
            )
        return (x - self.mean) / self.scale


def standardize_split(split: DataSplit, scale_outcome: bool = True) -> DataSplit:
    """Standardize covariates (and optionally the outcome) of a split.

    The outcome scaling is affine, so treatment effects are rescaled by the same
    factor; metrics are computed on the original outcome scale by inverting it
    inside the runner. Here we only rescale ``y``; ``mu0``/``mu1`` are left on
    the original scale and the factor is recorded in ``meta``.
    """
    scaler = Standardizer.fit(split.train.x)
    y_mean, y_std = 0.0, 1.0
    if scale_outcome:
        y_mean = float(split.train.y.mean())
        y_std = float(split.train.y.std())
        if y_std < 1e-12:
            y_std = 1.0

    def _apply(ds: CausalDataset) -> CausalDataset:
        return CausalDataset(
            x=scaler.transform(ds.x),
            t=ds.t,
            y=(ds.y - y_mean) / y_std,
            mu0=ds.mu0,
            mu1=ds.mu1,
            propensity=ds.propensity,
        )

    meta = dict(split.meta)
    meta.update({"y_mean": y_mean, "y_std": y_std, "standardized": True})
    return DataSplit(train=_apply(split.train), test=_apply(split.test), meta=meta)


def subsample_train(
    split: DataSplit, fraction: float, seed: int
) -> DataSplit:
    """Return a split whose training set is a random ``fraction`` of the original.

    The test set is untouched so that all sample-size conditions are evaluated
    on identical held-out units.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return split
    rng = np.random.default_rng(seed)
    n = split.train.n
    k = max(2, int(round(fraction * n)))
    idx = rng.choice(n, size=k, replace=False)
    meta = dict(split.meta)
    meta.update({"train_fraction": fraction, "n_train": k})
    return DataSplit(train=split.train.subset(idx), test=split.test, meta=meta)
