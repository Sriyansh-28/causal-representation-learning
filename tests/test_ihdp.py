"""Tests for the IHDP loader.

These require the raw files. They download on first run and are cached under
``data/raw``; if the download is unavailable the tests skip rather than fail,
so the suite still runs offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.ihdp import N_REALIZATIONS, induce_imbalance, load_ihdp


@pytest.fixture(scope="module")
def ihdp():
    try:
        return load_ihdp(realization=0)
    except Exception as exc:  # network or checksum failure
        pytest.skip(f"IHDP data unavailable: {exc}")


def test_ihdp_has_the_documented_shape(ihdp):
    assert (ihdp.train.n, ihdp.train.d) == (672, 25)
    assert (ihdp.test.n, ihdp.test.d) == (75, 25)


def test_ihdp_provides_ground_truth_effects(ihdp):
    assert ihdp.train.has_ground_truth
    assert np.all(np.isfinite(ihdp.train.tau_true))


def test_ihdp_is_treatment_imbalanced_by_construction(ihdp):
    # The benchmark's non-random treatment assignment leaves ~19% treated.
    assert 0.15 < ihdp.train.treated_fraction < 0.25


def test_ihdp_realizations_draw_from_one_shared_unit_pool():
    """Each realization re-partitions the same 747 units and re-simulates outcomes.

    The train split therefore differs between realizations, but the pooled
    train+test covariate set is identical -- which is what makes realizations
    valid replicates of the same population.
    """
    try:
        a, b = load_ihdp(0), load_ihdp(1)
    except Exception as exc:
        pytest.skip(f"IHDP data unavailable: {exc}")

    def pooled(split):
        x = np.vstack([split.train.x, split.test.x])
        return x[np.lexsort(x.T)]

    assert np.allclose(pooled(a), pooled(b))
    assert not np.allclose(a.train.x, b.train.x)   # partition differs
    assert not np.allclose(a.train.y, b.train.y)   # outcomes differ


def test_ihdp_pooled_treated_fraction_is_constant_across_realizations():
    try:
        splits = [load_ihdp(i) for i in (0, 1, 2)]
    except Exception as exc:
        pytest.skip(f"IHDP data unavailable: {exc}")
    fracs = [
        float(np.concatenate([s.train.t, s.test.t]).mean()) for s in splits
    ]
    assert max(fracs) - min(fracs) < 1e-9


def test_ihdp_rejects_out_of_range_realizations():
    with pytest.raises(ValueError, match="realization"):
        load_ihdp(realization=N_REALIZATIONS)


def test_induce_imbalance_lowers_the_treated_fraction(ihdp):
    out = induce_imbalance(ihdp, target_treated_fraction=0.1, seed=0)
    assert out.train.treated_fraction == pytest.approx(0.1, abs=0.02)
    assert out.train.n < ihdp.train.n
    assert out.test.n == ihdp.test.n


def test_induce_imbalance_rejects_invalid_targets(ihdp):
    with pytest.raises(ValueError):
        induce_imbalance(ihdp, target_treated_fraction=0.0, seed=0)
