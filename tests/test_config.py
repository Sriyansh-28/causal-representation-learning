"""Tests for YAML configuration loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import ExperimentConfig, load_config, parse_config
from src.utils.paths import PROJECT_ROOT, ensure_dir, resolve

CONFIG_DIR = PROJECT_ROOT / "configs"

MINIMAL = {
    "name": "t", "experiment": "baseline",
    "dataset": {"name": "synthetic"}, "models": ["t_learner"], "seeds": [0, 1],
}


def test_parse_minimal_config():
    cfg = parse_config(dict(MINIMAL))
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.seeds == [0, 1]
    assert cfg.output_dir == "results"


def test_integer_seeds_expand_to_a_range():
    cfg = parse_config({**MINIMAL, "seeds": 4})
    assert cfg.seeds == [0, 1, 2, 3]


@pytest.mark.parametrize("missing", ["name", "experiment", "dataset", "models", "seeds"])
def test_missing_required_keys_are_rejected(missing):
    raw = {k: v for k, v in MINIMAL.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        parse_config(raw)


def test_unknown_experiment_is_rejected():
    with pytest.raises(ValueError, match="unknown experiment"):
        parse_config({**MINIMAL, "experiment": "nonsense"})


def test_dataset_must_contain_a_name():
    with pytest.raises(ValueError, match="dataset"):
        parse_config({**MINIMAL, "dataset": {"n_train": 10}})


def test_empty_models_list_is_rejected():
    with pytest.raises(ValueError, match="models"):
        parse_config({**MINIMAL, "models": []})


def test_negative_seeds_are_rejected():
    with pytest.raises(ValueError, match="seeds"):
        parse_config({**MINIMAL, "seeds": [0, -3]})


def test_load_config_reports_a_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_load_config_rejects_a_non_mapping(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_round_trip_from_disk(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    assert load_config(p).name == "t"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_is_valid(path: Path):
    """Every config in configs/ must load, so no shipped config is broken."""
    cfg = load_config(path)
    assert cfg.models and cfg.seeds
    assert cfg.dataset["name"] in {"ihdp", "synthetic"}


def test_resolve_makes_relative_paths_project_relative():
    assert resolve("results").is_absolute()
    assert resolve("/tmp").as_posix() == "/tmp"


def test_ensure_dir_creates_nested_directories(tmp_path: Path):
    target = tmp_path / "a" / "b"
    assert ensure_dir(target).is_dir()
