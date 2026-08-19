"""Conventional causal-ML meta-learners: S-, T- and X-learner.

All three wrap an arbitrary scikit-learn regressor, so differences between
them reflect the *meta-learning strategy* rather than the base model class.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from .base import BaseCATELearner

BASE_LEARNERS: dict[str, Callable[..., Any]] = {
    "gbm": GradientBoostingRegressor,
    "rf": RandomForestRegressor,
    "ridge": Ridge,
}


def make_base_learner(kind: str, seed: int, **overrides: Any) -> Any:
    """Instantiate a base regressor by name with reproducible settings."""
    if kind not in BASE_LEARNERS:
        raise ValueError(
            f"unknown base learner '{kind}'; expected one of {sorted(BASE_LEARNERS)}"
        )
    cls = BASE_LEARNERS[kind]
    kwargs: dict[str, Any] = {}
    if kind == "gbm":
        kwargs = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,
                  "random_state": seed}
    elif kind == "rf":
        kwargs = {"n_estimators": 300, "min_samples_leaf": 5,
                  "random_state": seed, "n_jobs": 1}
    elif kind == "ridge":
        kwargs = {"alpha": 1.0}
    kwargs.update(overrides)
    return cls(**kwargs)


class SLearner(BaseCATELearner):
    """Single model on ``[X, T]``; effect is the difference of two predictions.

    Treatment is one feature among many, so a flexible base learner can shrink
    the treatment signal toward zero -- the classic S-learner failure mode.
    """

    name = "S-Learner"

    def __init__(self, base_learner: str = "gbm", seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, base_learner=base_learner, **kwargs)
        self.model = make_base_learner(base_learner, seed)

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> "SLearner":
        x, t, y = self._validate_inputs(x, t, y)
        self.model.fit(np.column_stack([x, t]), y)
        self._fitted = True
        return self

    def predict_cate(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        ones, zeros = np.ones(x.shape[0]), np.zeros(x.shape[0])
        return self.model.predict(np.column_stack([x, ones])) - self.model.predict(
            np.column_stack([x, zeros])
        )


class TLearner(BaseCATELearner):
    """Separate outcome models per arm; effect is their difference.

    Fully flexible per arm, but each model sees only its own arm's data, so the
    smaller arm is estimated from few units -- the failure mode under imbalance.
    """

    name = "T-Learner"

    def __init__(self, base_learner: str = "gbm", seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, base_learner=base_learner, **kwargs)
        self.model0 = make_base_learner(base_learner, seed)
        self.model1 = make_base_learner(base_learner, seed + 1)

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> "TLearner":
        x, t, y = self._validate_inputs(x, t, y)
        self.model0.fit(x[t == 0], y[t == 0])
        self.model1.fit(x[t == 1], y[t == 1])
        self._fitted = True
        return self

    def predict_cate(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return self.model1.predict(x) - self.model0.predict(x)


class XLearner(BaseCATELearner):
    """Kuenzel et al. (2019) X-learner.

    Stage 1 fits per-arm outcome models. Stage 2 imputes individual effects by
    crossing each unit against the *other* arm's model and regresses them on X.
    Stage 3 combines the two effect models with propensity weights, which is
    what makes the X-learner robust when one arm is small.
    """

    name = "X-Learner"

    def __init__(self, base_learner: str = "gbm", seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, base_learner=base_learner, **kwargs)
        self.model0 = make_base_learner(base_learner, seed)
        self.model1 = make_base_learner(base_learner, seed + 1)
        self.tau0 = make_base_learner(base_learner, seed + 2)
        self.tau1 = make_base_learner(base_learner, seed + 3)
        self.propensity_model = LogisticRegression(max_iter=2000, random_state=seed)

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> "XLearner":
        x, t, y = self._validate_inputs(x, t, y)
        x1, y1 = x[t == 1], y[t == 1]
        x0, y0 = x[t == 0], y[t == 0]

        # Stage 1: outcome models per arm.
        self.model0.fit(x0, y0)
        self.model1.fit(x1, y1)

        # Stage 2: imputed treatment effects, crossing arms.
        d1 = y1 - self.model0.predict(x1)   # treated: observed - imputed control
        d0 = self.model1.predict(x0) - y0   # control: imputed treated - observed
        self.tau1.fit(x1, d1)
        self.tau0.fit(x0, d0)

        # Stage 3: propensity weights.
        self.propensity_model.fit(x, t)
        self._fitted = True
        return self

    def predict_cate(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        g = self.propensity_model.predict_proba(x)[:, 1]
        return g * self.tau0.predict(x) + (1.0 - g) * self.tau1.predict(x)
