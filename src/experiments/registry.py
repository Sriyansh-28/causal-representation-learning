"""Factories mapping config strings to datasets and learners.

Adding a benchmark or an estimator means adding one entry here; the runner and
all six experiments pick it up unchanged.
"""
from __future__ import annotations

from typing import Any, Dict

from ..data.base import DataSplit
from ..data.ihdp import induce_imbalance, load_ihdp
from ..data.synthetic import SyntheticConfig, make_synthetic
from ..learners.base import BaseCATELearner
from ..learners.meta import SLearner, TLearner, XLearner
from ..learners.neural import NeuralRepresentationLearner

LEARNER_REGISTRY: Dict[str, type[BaseCATELearner]] = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "neural_rep": NeuralRepresentationLearner,
}

# Ablation variants for the neural model. Each changes exactly one component.
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_representation": {"use_representation": False},
    "no_treatment_heads": {"treatment_specific_heads": False},
    "no_regularization": {"dropout": 0.0, "weight_decay": 0.0},
}


def build_dataset(dataset_cfg: Dict[str, Any], seed: int, **overrides: Any) -> DataSplit:
    """Construct a train/test split for one replicate.

    Args:
        dataset_cfg: The config's ``dataset`` block; must contain ``name``.
        seed: Replicate index. For IHDP this selects the outcome realization;
            for the synthetic DGP it seeds the draw.
        **overrides: Per-condition overrides (e.g. ``confounding_strength``).

    Raises:
        ValueError: For an unknown dataset name.
    """
    cfg = {**dataset_cfg, **overrides}
    name = cfg.pop("name")

    if name == "ihdp":
        target_fraction = cfg.pop("target_treated_fraction", None)
        cfg.pop("confounding_strength", None)  # not tunable on IHDP
        split = load_ihdp(realization=seed % 100, **cfg)
        if target_fraction is not None:
            split = induce_imbalance(split, target_fraction, seed=seed)
        return split

    if name == "synthetic":
        valid = set(SyntheticConfig.__dataclass_fields__) - {"_validated"}
        unknown = set(cfg) - valid
        if unknown:
            raise ValueError(f"unknown synthetic dataset keys: {sorted(unknown)}")
        return make_synthetic(SyntheticConfig(seed=seed, **cfg))

    raise ValueError(f"unknown dataset '{name}'; expected 'ihdp' or 'synthetic'")


def build_learner(
    model_name: str, seed: int, neural_cfg: Dict[str, Any], **overrides: Any
) -> BaseCATELearner:
    """Instantiate a learner by registry name.

    Neural-model keyword arguments come from the config's ``neural`` block and
    can be overridden per condition (used by ablation and sensitivity runs).
    """
    base_name = model_name.split(":")[0]
    if base_name not in LEARNER_REGISTRY:
        raise ValueError(
            f"unknown model '{model_name}'; expected one of {sorted(LEARNER_REGISTRY)}"
        )
    cls = LEARNER_REGISTRY[base_name]
    if base_name == "neural_rep":
        kwargs = {**neural_cfg, **overrides}
        return cls(seed=seed, **kwargs)
    kwargs = {k: v for k, v in overrides.items() if k == "base_learner"}
    return cls(seed=seed, **kwargs)
