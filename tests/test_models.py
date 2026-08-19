"""Tests for learner interfaces, model shapes and the ablation flags."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.learners.meta import SLearner, TLearner, XLearner, make_base_learner
from src.learners.neural import NeuralRepresentationLearner
from src.models.representation import RepresentationNet
from src.models.trainer import TrainConfig, train_model

ALL_LEARNERS = [SLearner, TLearner, XLearner]


@pytest.fixture(scope="module")
def fast_neural_kwargs():
    """Small, quick settings so model tests stay fast."""
    return dict(hidden_dims=(16,), latent_dim=8, head_hidden_dims=(8,), epochs=5)


@pytest.mark.parametrize("cls", ALL_LEARNERS)
def test_meta_learners_return_one_effect_per_row(cls, small_split):
    m = cls(seed=0).fit(small_split.train.x, small_split.train.t, small_split.train.y)
    tau = m.predict_cate(small_split.test.x)
    assert tau.shape == (small_split.test.n,)
    assert np.all(np.isfinite(tau))


@pytest.mark.parametrize("cls", ALL_LEARNERS)
def test_learners_raise_before_fitting(cls, small_split):
    with pytest.raises(RuntimeError, match="fitted"):
        cls(seed=0).predict_cate(small_split.test.x)


@pytest.mark.parametrize("cls", ALL_LEARNERS)
def test_learners_reject_single_arm_data(cls):
    x = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ValueError, match="arm"):
        cls(seed=0).fit(x, np.zeros(20), np.zeros(20))


def test_predict_ate_is_the_mean_of_predict_cate(small_split):
    m = TLearner(seed=0).fit(small_split.train.x, small_split.train.t, small_split.train.y)
    assert m.predict_ate(small_split.test.x) == pytest.approx(
        float(np.mean(m.predict_cate(small_split.test.x)))
    )


def test_make_base_learner_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown base learner"):
        make_base_learner("not_a_model", seed=0)


def test_representation_net_output_shapes():
    net = RepresentationNet(input_dim=6, hidden_dims=(8,), latent_dim=4,
                            head_hidden_dims=(4,))
    x = torch.randn(11, 6)
    y0, y1 = net(x)
    assert y0.shape == (11,) and y1.shape == (11,)
    assert net.predict_tau(x).shape == (11,)
    assert net.represent(x).shape == (11, 4)


def test_representation_net_tau_equals_head_difference():
    net = RepresentationNet(input_dim=5, hidden_dims=(8,), latent_dim=4, dropout=0.0)
    net.eval()
    x = torch.randn(7, 5)
    with torch.no_grad():
        y0, y1 = net(x)
        assert torch.allclose(net.predict_tau(x), y1 - y0)


def test_predict_factual_selects_the_observed_arm():
    net = RepresentationNet(input_dim=4, hidden_dims=(8,), latent_dim=4, dropout=0.0)
    net.eval()
    x = torch.randn(6, 4)
    t = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    with torch.no_grad():
        y0, y1 = net(x)
        assert torch.allclose(net.predict_factual(x, t), torch.where(t > 0.5, y1, y0))


def test_ablation_without_representation_uses_identity_encoder():
    net = RepresentationNet(input_dim=9, latent_dim=3, use_representation=False)
    assert net.latent_dim == 9
    assert net.represent(torch.randn(4, 9)).shape == (4, 9)


def test_ablation_without_treatment_heads_shares_one_head():
    net = RepresentationNet(input_dim=5, latent_dim=4, treatment_specific_heads=False)
    assert net.shared_head is not None and net.head0 is None
    y0, y1 = net(torch.randn(6, 5))
    assert y0.shape == y1.shape == (6,)


def test_full_model_has_two_distinct_heads():
    net = RepresentationNet(input_dim=5, latent_dim=4)
    assert net.head0 is not None and net.head1 is not None
    assert net.head0 is not net.head1
    assert net.shared_head is None


def test_representation_net_validates_construction_args():
    with pytest.raises(ValueError):
        RepresentationNet(input_dim=0)
    with pytest.raises(ValueError):
        RepresentationNet(input_dim=4, latent_dim=0)
    with pytest.raises(ValueError):
        RepresentationNet(input_dim=4, dropout=1.5)


def test_representation_net_rejects_wrong_input_width():
    net = RepresentationNet(input_dim=4, latent_dim=2)
    with pytest.raises(ValueError, match="expected 4 covariates"):
        net(torch.randn(3, 7))


def test_train_config_validates_hyperparameters():
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(lr=0.0)
    with pytest.raises(ValueError):
        TrainConfig(val_fraction=1.0)


def test_training_reduces_the_training_loss(small_split):
    cfg = TrainConfig(hidden_dims=(16,), latent_dim=8, head_hidden_dims=(8,),
                      epochs=40, early_stopping=False, seed=0)
    _, history = train_model(
        small_split.train.x, small_split.train.t, small_split.train.y, cfg
    )
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_neural_learner_predicts_potential_outcomes(small_split, fast_neural_kwargs):
    m = NeuralRepresentationLearner(seed=0, **fast_neural_kwargs)
    m.fit(small_split.train.x, small_split.train.t, small_split.train.y)
    y0, y1 = m.predict_potential_outcomes(small_split.test.x)
    assert y0.shape == y1.shape == (small_split.test.n,)
    assert np.allclose(y1 - y0, m.predict_cate(small_split.test.x), atol=1e-5)


def test_neural_checkpoint_round_trip(tmp_path, small_split, fast_neural_kwargs):
    m = NeuralRepresentationLearner(seed=0, **fast_neural_kwargs)
    m.fit(small_split.train.x, small_split.train.t, small_split.train.y)
    path = tmp_path / "ckpt.pt"
    m.save_checkpoint(str(path))
    assert path.exists()
    ckpt = torch.load(path, weights_only=False)
    assert "state_dict" in ckpt and "config" in ckpt
