"""Tests for the Experiment 7 equal-budget model-selection protocol."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.tuning import (
    N_CONFIGS, TUNING_GRIDS, factual_val_mse, split_train_val, tune_and_fit,
)
from src.learners.meta import TLearner

FAST_NEURAL = {"hidden_dims": [16], "head_hidden_dims": [8], "batch_size": 64}


def test_every_model_has_the_same_number_of_candidates():
    """The core fairness guarantee: equal tuning budget for all estimators."""
    assert len(TUNING_GRIDS) == 4
    assert {len(g) for g in TUNING_GRIDS.values()} == {N_CONFIGS}


def test_meta_learners_share_one_grid():
    # Differences between S/T/X must reflect strategy, not candidate sets.
    assert TUNING_GRIDS["s_learner"] == TUNING_GRIDS["t_learner"]
    assert TUNING_GRIDS["t_learner"] == TUNING_GRIDS["x_learner"]


def test_split_is_identical_for_the_same_seed():
    a_fit, a_val = split_train_val(500, seed=3)
    b_fit, b_val = split_train_val(500, seed=3)
    assert np.array_equal(a_fit, b_fit) and np.array_equal(a_val, b_val)


def test_split_differs_across_seeds():
    a_fit, _ = split_train_val(500, seed=1)
    b_fit, _ = split_train_val(500, seed=2)
    assert not np.array_equal(a_fit, b_fit)


def test_split_is_disjoint_and_covers_all_rows():
    fit, val = split_train_val(200, seed=0)
    assert set(fit).isdisjoint(set(val))
    assert len(set(fit) | set(val)) == 200


def test_factual_mse_is_zero_for_a_perfect_predictor(small_split):
    class Perfect(TLearner):
        def predict_factual(self, x, t):
            return y_ref

    tr = small_split.train
    y_ref = tr.y
    m = Perfect(seed=0).fit(tr.x, tr.t, tr.y)
    assert factual_val_mse(m, tr.x, tr.t, tr.y) == pytest.approx(0.0)


def test_factual_mse_is_positive_for_a_real_model(small_split):
    tr = small_split.train
    m = TLearner(seed=0).fit(tr.x, tr.t, tr.y)
    assert factual_val_mse(m, tr.x, tr.t, tr.y) > 0


@pytest.mark.parametrize("model", ["s_learner", "t_learner", "x_learner", "neural_rep"])
def test_tune_and_fit_returns_a_fitted_model_and_record(model, small_split):
    tr = small_split.train
    learner, rec = tune_and_fit(model, tr.x, tr.t, tr.y, seed=0, neural_cfg=FAST_NEURAL)
    assert learner.predict_cate(small_split.test.x).shape == (small_split.test.n,)
    assert rec["n_configs_evaluated"] == N_CONFIGS
    assert 0 <= rec["selected_index"] < N_CONFIGS
    assert np.isfinite(rec["val_mse_selected"])


def test_selected_config_is_the_lowest_scoring_candidate(small_split):
    tr = small_split.train
    _, rec = tune_and_fit("t_learner", tr.x, tr.t, tr.y, seed=0, neural_cfg={})
    assert rec["val_mse_selected"] == pytest.approx(min(rec["val_mse_all"]))
    assert rec["selected_config"] == TUNING_GRIDS["t_learner"][rec["selected_index"]]


def test_neural_candidates_never_use_early_stopping(small_split):
    # Adaptive stopping is exactly the advantage Experiment 7 removes.
    tr = small_split.train
    learner, _ = tune_and_fit("neural_rep", tr.x, tr.t, tr.y, seed=0, neural_cfg=FAST_NEURAL)
    assert learner.config.early_stopping is False
    assert learner.config.val_fraction == 0.0


def test_tuning_is_deterministic_for_a_fixed_seed(small_split):
    tr = small_split.train
    a, ra = tune_and_fit("neural_rep", tr.x, tr.t, tr.y, 0, FAST_NEURAL)
    b, rb = tune_and_fit("neural_rep", tr.x, tr.t, tr.y, 0, FAST_NEURAL)
    assert ra["selected_index"] == rb["selected_index"]
    assert np.allclose(a.predict_cate(small_split.test.x),
                       b.predict_cate(small_split.test.x))


def test_unknown_model_is_rejected(small_split):
    tr = small_split.train
    with pytest.raises(ValueError, match="no tuning grid"):
        tune_and_fit("nope", tr.x, tr.t, tr.y, seed=0, neural_cfg={})


# --- Regression tests for the hyper-parameter forwarding bug -------------------
# A first run of Experiment 7 silently dropped meta-learner hyper-parameters:
# build_learner filtered them out and the learners never forwarded them to the
# base regressor. Every candidate was therefore identical, selection was a
# no-op, and only the neural model was genuinely tuned.

@pytest.mark.parametrize("model", ["s_learner", "t_learner", "x_learner"])
def test_meta_learner_hyperparameters_reach_the_base_regressor(model):
    from src.experiments.registry import build_learner

    small = build_learner(model, 0, {}, base_learner="gbm", n_estimators=50, max_depth=2)
    large = build_learner(model, 0, {}, base_learner="gbm", n_estimators=400, max_depth=3)
    inner = (lambda m: m.model) if model == "s_learner" else (lambda m: m.model0)
    assert inner(small).n_estimators == 50 and inner(small).max_depth == 2
    assert inner(large).n_estimators == 400 and inner(large).max_depth == 3


@pytest.mark.parametrize("model", ["s_learner", "t_learner", "x_learner"])
def test_different_configs_produce_different_predictions(model, small_split):
    from src.experiments.registry import build_learner

    tr = small_split.train
    a = build_learner(model, 0, {}, base_learner="gbm", n_estimators=50, max_depth=2,
                      learning_rate=0.05).fit(tr.x, tr.t, tr.y)
    b = build_learner(model, 0, {}, base_learner="gbm", n_estimators=400, max_depth=3,
                      learning_rate=0.05).fit(tr.x, tr.t, tr.y)
    assert not np.allclose(a.predict_cate(small_split.test.x),
                           b.predict_cate(small_split.test.x))


@pytest.mark.parametrize("model", ["s_learner", "t_learner", "x_learner", "neural_rep"])
def test_tuning_candidates_are_not_all_identical(model, small_split):
    """The grid must actually discriminate between candidates for every model."""
    tr = small_split.train
    _, rec = tune_and_fit(model, tr.x, tr.t, tr.y, seed=0, neural_cfg=FAST_NEURAL)
    finite = [v for v in rec["val_mse_all"] if np.isfinite(v)]
    assert len(set(finite)) > 1, f"{model}: all candidates scored identically"


def test_degenerate_grid_raises_rather_than_silently_passing(monkeypatch, small_split):
    import src.experiments.tuning as tuning_mod

    dup = [dict(tuning_mod.TUNING_GRIDS["t_learner"][0]) for _ in range(N_CONFIGS)]
    monkeypatch.setitem(tuning_mod.TUNING_GRIDS, "t_learner", dup)
    tr = small_split.train
    with pytest.raises(RuntimeError, match="no effect"):
        tuning_mod.tune_and_fit("t_learner", tr.x, tr.t, tr.y, seed=0, neural_cfg={})


def test_base_learner_key_is_not_passed_to_sklearn():
    from src.learners.meta import make_base_learner

    m = make_base_learner("gbm", seed=0, base_learner="gbm", n_estimators=77)
    assert m.n_estimators == 77
