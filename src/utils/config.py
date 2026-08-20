"""YAML experiment-configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

VALID_EXPERIMENTS = {
    "baseline", "confounding", "treatment_imbalance",
    "sample_size", "ablation", "sensitivity", "fair_selection",
}


@dataclass
class ExperimentConfig:
    """Validated view over a YAML experiment configuration.

    Attributes:
        name: Human-readable experiment name, used for output filenames.
        experiment: Experiment family; must be in ``VALID_EXPERIMENTS``.
        dataset: Dataset block (``name`` plus dataset-specific keys).
        models: Names of learners to evaluate.
        seeds: Replicate seeds / dataset realizations.
        neural: Hyper-parameters for the neural representation learner.
        conditions: Experiment-specific sweep definition.
        output_dir: Root directory for results.
        raw: The unparsed dictionary, for provenance logging.
    """

    name: str
    experiment: str
    dataset: Dict[str, Any]
    models: List[str]
    seeds: List[int]
    neural: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results"
    raw: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML config file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required keys are missing or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping, got {type(raw).__name__}")
    return parse_config(raw)


def parse_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """Validate an already-loaded config mapping."""
    for key in ("name", "experiment", "dataset", "models", "seeds"):
        if key not in raw:
            raise ValueError(f"config missing required key: '{key}'")

    experiment = raw["experiment"]
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            f"unknown experiment '{experiment}'; expected one of {sorted(VALID_EXPERIMENTS)}"
        )

    dataset = raw["dataset"]
    if not isinstance(dataset, dict) or "name" not in dataset:
        raise ValueError("config 'dataset' must be a mapping containing 'name'")

    models = raw["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("config 'models' must be a non-empty list")

    seeds = raw["seeds"]
    if isinstance(seeds, int):
        seeds = list(range(seeds))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("config 'seeds' must be a non-empty list or an integer count")
    if any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("config 'seeds' must contain non-negative integers")

    return ExperimentConfig(
        name=raw["name"],
        experiment=experiment,
        dataset=dataset,
        models=models,
        seeds=seeds,
        neural=raw.get("neural", {}) or {},
        conditions=raw.get("conditions", {}) or {},
        output_dir=raw.get("output_dir", "results"),
        raw=raw,
    )
