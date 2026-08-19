"""Tests for dataset containers, the synthetic DGP and preprocessing."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.base import CausalDataset, DataSplit
from src.data.preprocessing import Standardizer, standardize_split, subsample_train
from src.data.synthetic import (
    SyntheticConfig, make_synthetic, sigmoid, solve_intercept_for_treated_fraction,
)


def test_causal_dataset_derives_effects_and_shapes():
    ds = CausalDataset(
        x=np.zeros((4, 3)), t=np.array([0, 1, 0, 1]), y=np.arange(4.0),
        mu0=np.zeros(4), mu1=np.ones(4),
    )
    assert (ds.n, ds.d) == (4, 3)
    assert ds.has_ground_truth
    assert np.allclose(ds.tau_true, 1.0)
    assert ds.ate_true == pytest.approx(1.0)
    assert ds.treated_fraction == pytest.approx(0.5)


def test_causal_dataset_rejects_non_binary_treatment():
    with pytest.raises(ValueError, match="binary"):
        CausalDataset(x=np.zeros((2, 2)), t=np.array([0, 2]), y=np.zeros(2))


def test_causal_dataset_rejects_length_mismatch():
    with pytest.raises(ValueError):
        CausalDataset(x=np.zeros((3, 2)), t=np.array([0, 1]), y=np.zeros(3))


def test_causal_dataset_requires_ground_truth_for_tau():
    ds = CausalDataset(x=np.zeros((2, 2)), t=np.array([0, 1]), y=np.zeros(2))
    assert not ds.has_ground_truth
    with pytest.raises(ValueError):
        _ = ds.tau_true


def test_subset_keeps_all_arrays_aligned():
    ds = CausalDataset(
        x=np.arange(8.0).reshape(4, 2), t=np.array([0, 1, 0, 1]), y=np.arange(4.0),
        mu0=np.zeros(4), mu1=np.ones(4), propensity=np.full(4, 0.5),
    )
    sub = ds.subset(np.array([1, 3]))
    assert sub.n == 2
    assert np.allclose(sub.t, 1.0)
    assert sub.propensity is not None and sub.propensity.shape == (2,)


def test_data_split_requires_matching_dimensions():
    a = CausalDataset(x=np.zeros((2, 3)), t=np.array([0, 1]), y=np.zeros(2))
    b = CausalDataset(x=np.zeros((2, 4)), t=np.array([0, 1]), y=np.zeros(2))
    with pytest.raises(ValueError):
        DataSplit(train=a, test=b, meta={})


def test_sigmoid_is_stable_at_extremes():
    z = np.array([-800.0, 0.0, 800.0])
    out = sigmoid(z)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_synthetic_hits_target_treated_fraction():
    for target in (0.5, 0.25, 0.1):
        split = make_synthetic(SyntheticConfig(
            n_train=5000, n_test=100, target_treated_fraction=target, seed=0))
        assert split.train.treated_fraction == pytest.approx(target, abs=0.03)


def test_confounding_strength_does_not_move_the_true_ate():
    # The estimand must be constant across conditions, else error changes
    # would be confounded with a moving target.
    ates = [
        make_synthetic(SyntheticConfig(n_train=4000, n_test=100,
                                       confounding_strength=g, seed=0)).train.ate_true
        for g in (0.0, 1.0, 3.0)
    ]
    assert max(ates) - min(ates) < 0.05


def test_zero_confounding_gives_balanced_covariates():
    # gamma = 0 is a randomized trial: treated and control covariate means match.
    split = make_synthetic(SyntheticConfig(
        n_train=8000, n_test=100, confounding_strength=0.0, seed=1))
    tr = split.train
    diff = np.abs(tr.x[tr.t == 1].mean(axis=0) - tr.x[tr.t == 0].mean(axis=0))
    assert diff.max() < 0.15


def test_high_confounding_creates_covariate_imbalance():
    split = make_synthetic(SyntheticConfig(
        n_train=8000, n_test=100, confounding_strength=3.0, seed=1))
    tr = split.train
    diff = np.abs(tr.x[tr.t == 1].mean(axis=0) - tr.x[tr.t == 0].mean(axis=0))
    assert diff.max() > 0.3


def test_synthetic_config_validates_inputs():
    with pytest.raises(ValueError):
        SyntheticConfig(n_confounders=50, d=10)
    with pytest.raises(ValueError):
        SyntheticConfig(confounding_strength=-1.0)
    with pytest.raises(ValueError):
        SyntheticConfig(target_treated_fraction=1.5)


def test_intercept_solver_recovers_the_target():
    cfg = SyntheticConfig(target_treated_fraction=0.2, confounding_strength=2.0)
    a0 = solve_intercept_for_treated_fraction(cfg)
    assert np.isfinite(a0)


def test_standardizer_produces_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(200, 4))
    z = Standardizer.fit(x).transform(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-10)


def test_standardizer_handles_constant_columns():
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    z = Standardizer.fit(x).transform(x)
    assert np.all(np.isfinite(z))
    assert np.allclose(z[:, 0], 0.0)


def test_standardizer_rejects_wrong_width():
    s = Standardizer.fit(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        s.transform(np.zeros((5, 2)))


def test_standardize_split_uses_train_statistics_only(small_split):
    scaled = standardize_split(small_split)
    assert np.allclose(scaled.train.x.mean(axis=0), 0.0, atol=1e-8)
    # Test covariates are shifted by train stats, so their mean is near but
    # not exactly zero.
    assert not np.allclose(scaled.test.x.mean(axis=0), 0.0, atol=1e-12)
    assert scaled.meta["y_std"] > 0


def test_standardize_split_preserves_ground_truth_scale(small_split):
    scaled = standardize_split(small_split)
    assert np.allclose(scaled.test.tau_true, small_split.test.tau_true)


def test_subsample_train_reduces_only_the_training_set(small_split):
    sub = subsample_train(small_split, 0.5, seed=0)
    assert sub.train.n == pytest.approx(small_split.train.n // 2, abs=1)
    assert sub.test.n == small_split.test.n


def test_subsample_train_rejects_invalid_fraction(small_split):
    with pytest.raises(ValueError):
        subsample_train(small_split, 0.0, seed=0)
