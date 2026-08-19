"""Tests that seeding makes every stochastic component reproducible."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.synthetic import SyntheticConfig, make_synthetic
from src.learners.meta import XLearner
from src.learners.neural import NeuralRepresentationLearner
from src.utils.seeding import new_rng, set_seed

FAST = dict(hidden_dims=(16,), latent_dim=8, head_hidden_dims=(8,), epochs=6)


def test_set_seed_makes_numpy_draws_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    assert np.allclose(a, np.random.rand(5))


def test_set_seed_rejects_negative_seeds():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_new_rng_is_independent_of_global_state():
    a = new_rng(7).normal(size=4)
    np.random.seed(999)  # perturb the global RNG
    assert np.allclose(a, new_rng(7).normal(size=4))


def test_same_seed_reproduces_the_same_dataset():
    a = make_synthetic(SyntheticConfig(n_train=100, n_test=50, seed=3))
    b = make_synthetic(SyntheticConfig(n_train=100, n_test=50, seed=3))
    assert np.allclose(a.train.x, b.train.x)
    assert np.allclose(a.train.t, b.train.t)
    assert np.allclose(a.train.y, b.train.y)


def test_different_seeds_produce_different_datasets():
    a = make_synthetic(SyntheticConfig(n_train=100, n_test=50, seed=1))
    b = make_synthetic(SyntheticConfig(n_train=100, n_test=50, seed=2))
    assert not np.allclose(a.train.x, b.train.x)


def test_neural_learner_is_deterministic_under_a_fixed_seed(small_split):
    preds = []
    for _ in range(2):
        set_seed(0)
        m = NeuralRepresentationLearner(seed=0, **FAST)
        m.fit(small_split.train.x, small_split.train.t, small_split.train.y)
        preds.append(m.predict_cate(small_split.test.x))
    assert np.allclose(preds[0], preds[1])


def test_neural_learner_varies_across_seeds(small_split):
    preds = []
    for seed in (0, 1):
        set_seed(seed)
        m = NeuralRepresentationLearner(seed=seed, **FAST)
        m.fit(small_split.train.x, small_split.train.t, small_split.train.y)
        preds.append(m.predict_cate(small_split.test.x))
    assert not np.allclose(preds[0], preds[1])


def test_meta_learner_is_deterministic_under_a_fixed_seed(small_split):
    preds = [
        XLearner(seed=0)
        .fit(small_split.train.x, small_split.train.t, small_split.train.y)
        .predict_cate(small_split.test.x)
        for _ in range(2)
    ]
    assert np.allclose(preds[0], preds[1])
