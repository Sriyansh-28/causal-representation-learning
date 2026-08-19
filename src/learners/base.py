"""Common interface for every CATE estimator in this project.

The experiment runner only ever sees this interface, so a new estimator
(meta-learner or neural) becomes available to all six experiments by
implementing ``fit`` and ``predict_cate``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseCATELearner(ABC):
    """Abstract base class for conditional average treatment effect estimators."""

    name: str = "base"

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.seed = seed
        self.params: Dict[str, Any] = dict(kwargs)
        self._fitted = False

    @abstractmethod
    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> "BaseCATELearner":
        """Fit the estimator on observed ``(X, T, Y)``."""

    @abstractmethod
    def predict_cate(self, x: np.ndarray) -> np.ndarray:
        """Return estimated ``tau_hat(x)``, shape ``(n,)``."""

    def predict_ate(self, x: np.ndarray) -> float:
        """Average the predicted individual effects over ``x``."""
        return float(np.mean(self.predict_cate(x)))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fitted before prediction")

    @staticmethod
    def _validate_inputs(
        x: np.ndarray, t: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if not (x.shape[0] == t.shape[0] == y.shape[0]):
            raise ValueError("x, t and y must have the same number of rows")
        if t.sum() < 2 or (1 - t).sum() < 2:
            raise ValueError(
                "both treatment arms need at least 2 units "
                f"(treated={int(t.sum())}, control={int((1 - t).sum())})"
            )
        return x, t, y
