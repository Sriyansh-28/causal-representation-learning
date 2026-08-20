"""Focused tests for the policy-value / policy-regret metric family.

These exist because a strong Experiment 7 result hinged entirely on policy
regret, and the metric needed independent verification of its direction,
label handling and baseline behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    always_treat_value, evaluate_cate, never_treat_value, optimal_policy_value,
    policy_regret, policy_value, trivial_policy_regret,
)

# Two units. Unit 0 benefits from treatment (+5), unit 1 is harmed (-3).
MU0 = np.array([0.0, 3.0])
MU1 = np.array([5.0, 0.0])
TAU = MU1 - MU0            # [+5, -3]


def test_decision_direction_positive_tau_means_treat():
    # Correct decisions: treat unit 0, withhold from unit 1 -> 5 and 3.
    assert policy_value(TAU, MU0, MU1) == pytest.approx(4.0)


def test_decision_direction_is_not_inverted():
    # Exactly wrong decisions must score strictly worse than correct ones.
    assert policy_value(-TAU, MU0, MU1) == pytest.approx(0.0)
    assert policy_value(TAU, MU0, MU1) > policy_value(-TAU, MU0, MU1)


def test_threshold_is_strict_greater_than_zero():
    # tau_hat exactly 0 must NOT trigger treatment.
    assert policy_value(np.array([0.0, 0.0]), MU0, MU1) == pytest.approx(
        never_treat_value(MU0, MU1)
    )


def test_policy_value_formula_matches_definition():
    rng = np.random.default_rng(0)
    mu0, mu1 = rng.normal(size=50), rng.normal(size=50)
    tau_hat = rng.normal(size=50)
    treat = (tau_hat > 0).astype(float)
    expected = float(np.mean(treat * mu1 + (1 - treat) * mu0))
    assert policy_value(tau_hat, mu0, mu1) == pytest.approx(expected)


def test_regret_equals_oracle_minus_policy_value():
    rng = np.random.default_rng(1)
    mu0, mu1, tau_hat = rng.normal(size=40), rng.normal(size=40), rng.normal(size=40)
    assert policy_regret(tau_hat, mu0, mu1) == pytest.approx(
        optimal_policy_value(mu0, mu1) - policy_value(tau_hat, mu0, mu1)
    )


def test_oracle_dominates_every_policy():
    rng = np.random.default_rng(2)
    for _ in range(50):
        mu0, mu1 = rng.normal(size=30), rng.normal(size=30)
        opt = optimal_policy_value(mu0, mu1)
        assert opt >= policy_value(rng.normal(size=30), mu0, mu1) - 1e-12
        assert opt >= always_treat_value(mu0, mu1) - 1e-12
        assert opt >= never_treat_value(mu0, mu1) - 1e-12


def test_regret_is_never_negative():
    rng = np.random.default_rng(3)
    for _ in range(50):
        mu0, mu1 = rng.normal(size=30), rng.normal(size=30)
        assert policy_regret(rng.normal(size=30), mu0, mu1) >= -1e-12


def test_perfect_predictions_give_zero_regret():
    assert policy_regret(TAU, MU0, MU1) == pytest.approx(0.0)


def test_treatment_labels_are_consistent_under_relabelling():
    # Swapping the arms and the sign of the estimate must give the same value.
    assert policy_value(TAU, MU0, MU1) == pytest.approx(
        policy_value(-TAU, MU1, MU0)
    )


def test_always_and_never_treat_baselines():
    assert always_treat_value(MU0, MU1) == pytest.approx(2.5)
    assert never_treat_value(MU0, MU1) == pytest.approx(1.5)
    triv = trivial_policy_regret(MU0, MU1)
    assert triv["always_treat_regret"] == pytest.approx(1.5)
    assert triv["never_treat_regret"] == pytest.approx(2.5)


def test_constant_positive_predictor_scores_exactly_as_always_treat():
    """The degenerate case behind the Experiment 7 policy-regret result.

    A model that predicts a positive effect for every unit performs *identically*
    to always-treat, no matter how inaccurate its individual estimates are.
    """
    rng = np.random.default_rng(4)
    mu0, mu1 = rng.normal(size=200), rng.normal(size=200)
    constant = np.full(200, 7.3)  # wildly wrong magnitudes, all positive
    assert policy_value(constant, mu0, mu1) == pytest.approx(always_treat_value(mu0, mu1))
    assert policy_regret(constant, mu0, mu1) == pytest.approx(
        trivial_policy_regret(mu0, mu1)["always_treat_regret"]
    )


def test_perturbing_predictions_only_matters_across_the_threshold():
    # Scaling by a positive constant preserves every decision, hence the value.
    rng = np.random.default_rng(5)
    mu0, mu1, tau_hat = rng.normal(size=60), rng.normal(size=60), rng.normal(size=60)
    assert policy_value(tau_hat * 10.0, mu0, mu1) == pytest.approx(
        policy_value(tau_hat, mu0, mu1)
    )
    # A large positive shift pushes everything above threshold -> always-treat.
    assert policy_value(tau_hat + 1e6, mu0, mu1) == pytest.approx(
        always_treat_value(mu0, mu1)
    )


def test_evaluate_cate_exposes_baselines_and_treated_fraction():
    out = evaluate_cate(TAU, TAU, MU0, MU1)
    assert out["always_treat_value"] == pytest.approx(2.5)
    assert out["never_treat_value"] == pytest.approx(1.5)
    assert out["always_treat_regret"] == pytest.approx(1.5)
    assert out["policy_treated_fraction"] == pytest.approx(0.5)
