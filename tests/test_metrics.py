"""Tests for treatment-effect metrics, checked against hand calculations."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    abs_ate_error, cate_bias, cate_mae, evaluate_cate, optimal_policy_value,
    pehe, policy_regret, policy_value, subgroup_errors,
)


def test_pehe_matches_closed_form():
    tau_hat = np.array([1.0, 2.0, 3.0])
    tau_true = np.array([1.0, 1.0, 1.0])
    assert pehe(tau_hat, tau_true) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_pehe_is_zero_for_perfect_estimates():
    tau = np.array([-1.0, 0.5, 2.0])
    assert pehe(tau, tau) == pytest.approx(0.0)


def test_pehe_is_symmetric_in_its_arguments():
    a, b = np.array([1.0, 4.0]), np.array([2.0, 2.0])
    assert pehe(a, b) == pytest.approx(pehe(b, a))


def test_abs_ate_error_uses_means_not_individuals():
    # Individual errors are large but cancel exactly in the mean.
    tau_hat = np.array([0.0, 2.0])
    tau_true = np.array([2.0, 0.0])
    assert abs_ate_error(tau_hat, tau_true) == pytest.approx(0.0)
    assert pehe(tau_hat, tau_true) == pytest.approx(2.0)


def test_cate_mae_and_bias():
    tau_hat = np.array([2.0, 4.0])
    tau_true = np.array([1.0, 1.0])
    assert cate_mae(tau_hat, tau_true) == pytest.approx(2.0)
    assert cate_bias(tau_hat, tau_true) == pytest.approx(2.0)


def test_cate_bias_is_signed():
    assert cate_bias(np.array([0.0]), np.array([1.0])) == pytest.approx(-1.0)


def test_policy_value_selects_the_right_arm():
    # Unit 0: treat (tau_hat > 0) -> mu1 = 5. Unit 1: do not treat -> mu0 = 3.
    tau_hat = np.array([1.0, -1.0])
    mu0 = np.array([0.0, 3.0])
    mu1 = np.array([5.0, 0.0])
    assert policy_value(tau_hat, mu0, mu1) == pytest.approx(4.0)
    assert optimal_policy_value(mu0, mu1) == pytest.approx(4.0)
    assert policy_regret(tau_hat, mu0, mu1) == pytest.approx(0.0)


def test_policy_regret_is_positive_for_a_wrong_policy():
    tau_hat = np.array([-1.0, 1.0])   # exactly the wrong decisions
    mu0 = np.array([0.0, 3.0])
    mu1 = np.array([5.0, 0.0])
    assert policy_regret(tau_hat, mu0, mu1) == pytest.approx(4.0)


def test_policy_regret_is_never_negative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu0, mu1 = rng.normal(size=50), rng.normal(size=50)
        assert policy_regret(rng.normal(size=50), mu0, mu1) >= -1e-12


def test_evaluate_cate_includes_policy_metrics_only_with_ground_truth():
    tau_hat, tau_true = np.array([1.0, 2.0]), np.array([1.0, 1.0])
    assert "policy_value" not in evaluate_cate(tau_hat, tau_true)
    full = evaluate_cate(tau_hat, tau_true, np.zeros(2), np.ones(2))
    assert {"policy_value", "policy_regret"} <= set(full)


def test_subgroup_errors_cover_all_bins():
    rng = np.random.default_rng(0)
    tau_true = rng.normal(size=200)
    out = subgroup_errors(tau_true + 0.1, tau_true, n_bins=4)
    assert len(out) == 4
    assert all(np.isfinite(v) for v in out.values())


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        pehe(np.array([1.0, 2.0]), np.array([1.0]))


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        pehe(np.array([]), np.array([]))
