"""Deterministic seeding utilities shared by every experiment."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: Non-negative integer seed.
        deterministic_torch: If True, request deterministic kernels from
            PyTorch. Some ops have no deterministic implementation; those
            raise at call time rather than silently varying.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for pure-metric usage
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def new_rng(seed: Optional[int]) -> np.random.Generator:
    """Return a fresh NumPy Generator (preferred over global RNG state)."""
    return np.random.default_rng(seed)
