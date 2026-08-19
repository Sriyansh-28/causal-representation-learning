"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import SyntheticConfig, make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def small_split():
    """A small synthetic split, fast enough to reuse across the suite."""
    return make_synthetic(
        SyntheticConfig(n_train=400, n_test=200, d=8, n_confounders=3, seed=0)
    )
