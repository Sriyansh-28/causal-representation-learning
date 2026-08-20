"""Neural representation learner exposed through the common CATE interface."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from ..models.representation import RepresentationNet
from ..models.trainer import TrainConfig, train_model
from .base import BaseCATELearner


class NeuralRepresentationLearner(BaseCATELearner):
    """Shared-representation network with treatment-specific heads.

    Args:
        seed: Random seed for initialisation, batching and the train/val split.
        **kwargs: Any field of :class:`TrainConfig`.
    """

    name = "NeuralRep"

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        kwargs.pop("seed", None)
        self.config = TrainConfig(seed=seed, **kwargs)
        self.model: Optional[RepresentationNet] = None
        self.history: dict = {}

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> "NeuralRepresentationLearner":
        x, t, y = self._validate_inputs(x, t, y)
        self.model, self.history = train_model(x, t, y, self.config)
        self._fitted = True
        return self

    def predict_cate(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.model is not None
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            return self.model.predict_tau(xt).cpu().numpy().astype(np.float64)

    def predict_factual(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict observed-treatment outcomes from the matching head."""
        self._check_fitted()
        assert self.model is not None
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        tt = torch.from_numpy(np.asarray(t, dtype=np.float32).ravel())
        with torch.no_grad():
            return self.model.predict_factual(xt, tt).cpu().numpy().astype(np.float64)

    def predict_potential_outcomes(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return estimated ``(Y(0), Y(1))`` for each row of ``x``."""
        self._check_fitted()
        assert self.model is not None
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            y0, y1 = self.model(xt)
        return y0.cpu().numpy().astype(np.float64), y1.cpu().numpy().astype(np.float64)

    def save_checkpoint(self, path: str) -> None:
        """Persist model weights and the training configuration."""
        self._check_fitted()
        assert self.model is not None
        torch.save(
            {"state_dict": self.model.state_dict(), "config": vars(self.config)}, path
        )
