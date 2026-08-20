"""Tests for the registry, runner and aggregation layers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.diagnostics import overlap_diagnostics, standardized_mean_differences
from src.evaluation.statistics import bootstrap_ci, holm_bonferroni, paired_test, summarize
from src.experiments.aggregate import (
    ablation_comparisons, aggregate_results, paired_comparisons, pivot_table,
)
from src.experiments.registry import ABLATION_VARIANTS, build_dataset, build_learner
from src.experiments.runner import run_experiment
from src.utils.config import parse_config

FAST_NEURAL = {"hidden_dims": [16], "latent_dim": 8, "head_hidden_dims": [8], "epochs": 5}
SMALL_DATASET = {"name": "synthetic", "n_train": 300, "n_test": 150, "d": 6,
                 "n_confounders": 2}


def _config(**overrides):
    base = {
        "name": "unit_test", "experiment": "baseline", "dataset": dict(SMALL_DATASET),
        "models": ["t_learner", "neural_rep"], "seeds": [0, 1], "neural": dict(FAST_NEURAL),
    }
    base.update(overrides)
    return parse_config(base)


def test_build_dataset_dispatches_on_name():
    split = build_dataset({"name": "synthetic", "n_train": 50, "n_test": 20}, seed=0)
    assert split.train.n == 50


def test_build_dataset_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset({"name": "not_a_dataset"}, seed=0)


def test_build_dataset_rejects_unknown_synthetic_keys():
    with pytest.raises(ValueError, match="unknown synthetic"):
        build_dataset({"name": "synthetic", "bogus_key": 1}, seed=0)


def test_build_dataset_applies_condition_overrides():
    split = build_dataset(dict(SMALL_DATASET), seed=0, confounding_strength=3.0)
    assert split.meta["confounding_strength"] == 3.0


def test_build_learner_returns_registered_types():
    assert build_learner("t_learner", 0, {}).name == "T-Learner"
    assert build_learner("neural_rep", 0, FAST_NEURAL).name == "NeuralRep"


def test_build_learner_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_learner("magic_learner", 0, {})


def test_ablation_variants_each_change_one_component():
    assert ABLATION_VARIANTS["full"] == {}
    assert ABLATION_VARIANTS["no_representation"] == {"use_representation": False}
    assert ABLATION_VARIANTS["no_treatment_heads"] == {"treatment_specific_heads": False}
    assert ABLATION_VARIANTS["no_regularization"] == {"dropout": 0.0, "weight_decay": 0.0}


def test_baseline_run_produces_one_row_per_model_and_seed():
    df = run_experiment(_config(), verbose=False)
    assert len(df) == 4
    assert (df["error"] == "").all()
    assert {"pehe", "abs_ate_error", "policy_value"} <= set(df.columns)
    assert df["pehe"].notna().all()


def test_runner_records_diagnostics_for_each_run():
    df = run_experiment(_config(), verbose=False)
    assert "diag_treated_fraction" in df.columns
    assert df["diag_treated_fraction"].between(0, 1).all()


def test_ablation_run_covers_every_variant():
    cfg = _config(experiment="ablation", models=["neural_rep"],
                  conditions={"variants": ["full", "no_representation"]})
    df = run_experiment(cfg, verbose=False)
    assert set(df["condition_value"]) == {"full", "no_representation"}


def test_unknown_ablation_variant_is_rejected():
    cfg = _config(experiment="ablation", conditions={"variants": ["nope"]})
    with pytest.raises(ValueError, match="unknown ablation variant"):
        run_experiment(cfg, verbose=False)


def test_sample_size_run_varies_the_training_size():
    cfg = _config(experiment="sample_size", models=["t_learner"],
                  conditions={"train_fractions": [0.5, 1.0]})
    df = run_experiment(cfg, verbose=False)
    sizes = df.groupby("condition_value")["n_train"].first()
    assert sizes[0.5] < sizes[1.0]


def test_sensitivity_requires_a_sweeps_block():
    cfg = _config(experiment="sensitivity", models=["neural_rep"], conditions={})
    with pytest.raises(ValueError, match="sweeps"):
        run_experiment(cfg, verbose=False)


def test_runner_records_failures_without_aborting():
    # A single-unit test split makes metric computation fail for that cell only.
    cfg = _config(models=["t_learner"], dataset={**SMALL_DATASET, "n_train": 4})
    df = run_experiment(cfg, verbose=False)
    assert len(df) == 2  # both seeds still produce a row


def test_aggregate_reports_mean_std_and_ci():
    df = run_experiment(_config(), verbose=False)
    agg = aggregate_results(df)
    assert {"mean", "std", "ci_low", "ci_high", "n"} <= set(agg.columns)
    assert (agg["n"] == 2).all()


def test_aggregate_ignores_failed_runs():
    df = pd.DataFrame([
        {"experiment": "baseline", "condition": "c", "condition_value": 1,
         "model": "m", "seed": 0, "pehe": 1.0, "error": ""},
        {"experiment": "baseline", "condition": "c", "condition_value": 1,
         "model": "m", "seed": 1, "pehe": 99.0, "error": "boom"},
    ])
    agg = aggregate_results(df, metrics=["pehe"])
    assert agg["mean"].iloc[0] == pytest.approx(1.0)


def test_pivot_table_formats_mean_and_sd():
    df = run_experiment(_config(), verbose=False)
    pv = pivot_table(aggregate_results(df), "pehe")
    assert not pv.empty and "±" in str(pv.iloc[0, -1])


def test_paired_comparisons_pairs_models_on_shared_seeds():
    df = run_experiment(_config(), verbose=False)
    cmp_df = paired_comparisons(df, reference="neural_rep", metric="pehe")
    assert set(cmp_df["comparison"]) == {"t_learner"}
    assert cmp_df["n_pairs"].iloc[0] == 2


def test_summarize_reports_dispersion():
    out = summarize([1.0, 2.0, 3.0])
    assert out["mean"] == pytest.approx(2.0)
    assert out["std"] == pytest.approx(1.0)
    assert out["n"] == 3


def test_summarize_reports_median_and_iqr():
    # A single large outlier moves the mean but not the median.
    out = summarize([1.0, 2.0, 3.0, 100.0])
    assert out["median"] == pytest.approx(2.5)
    assert out["iqr"] == pytest.approx(25.5)
    assert out["mean"] > out["median"]


def test_summarize_handles_a_single_observation():
    out = summarize([5.0])
    assert out["mean"] == pytest.approx(5.0)
    assert np.isnan(out["ci_low"])


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(10.0, 1.0, size=100)
    lo, hi = bootstrap_ci(v, n_boot=2000, seed=0)
    assert lo < v.mean() < hi


def test_paired_test_detects_a_consistent_shift():
    a = np.arange(10, dtype=float)
    res = paired_test(a, a + 1.0)
    assert res["mean_diff"] == pytest.approx(-1.0)
    assert res["p_value"] < 0.05


def test_paired_test_returns_nan_for_identical_inputs():
    a = np.arange(10, dtype=float)
    assert np.isnan(paired_test(a, a)["p_value"])


def test_paired_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_test([1.0, 2.0], [1.0])


def test_holm_bonferroni_is_more_conservative_than_raw_alpha():
    # 0.04 alone would be rejected at 0.05, but not as the second of three tests.
    assert holm_bonferroni([0.001, 0.04, 0.9]) == [True, False, False]


def test_holm_bonferroni_handles_nan_p_values():
    assert holm_bonferroni([float("nan"), 0.001]) == [False, True]


def test_standardized_mean_differences_are_zero_under_random_assignment():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 5))
    t = rng.integers(0, 2, size=4000).astype(float)
    assert np.nanmax(standardized_mean_differences(x, t)) < 0.15


def test_overlap_diagnostics_report_expected_keys():
    from src.data.synthetic import SyntheticConfig, make_synthetic
    split = make_synthetic(SyntheticConfig(n_train=500, n_test=100, seed=0))
    diag = overlap_diagnostics(split.train)
    assert {"treated_fraction", "ps_min", "ps_max", "smd_mean"} <= set(diag)
    assert 0.0 <= diag["ps_min"] <= diag["ps_max"] <= 1.0


def _ablation_frame(full_pehe: float, variant_pehe: float, n: int = 10) -> pd.DataFrame:
    """Build a synthetic ablation result frame with a fixed per-seed offset."""
    rows = []
    for seed in range(n):
        jitter = 0.01 * seed
        rows.append({"condition": "ablation", "condition_value": "full", "seed": seed,
                     "model": "neural_rep", "pehe": full_pehe + jitter, "error": ""})
        rows.append({"condition": "ablation", "condition_value": "no_representation",
                     "seed": seed, "model": "neural_rep",
                     "pehe": variant_pehe + jitter, "error": ""})
    return pd.DataFrame(rows)


def test_ablation_comparisons_reports_delta_and_percent_change():
    out = ablation_comparisons(_ablation_frame(1.0, 1.2))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["variant"] == "no_representation"
    assert row["delta_vs_full"] == pytest.approx(0.2)
    # Full-model mean is 1.0 + mean(0.01 * seed) = 1.045, so 0.2 / 1.045.
    assert row["full_mean"] == pytest.approx(1.045)
    assert row["pct_change_vs_full"] == pytest.approx(100 * 0.2 / 1.045, rel=1e-6)


def test_ablation_comparisons_positive_delta_means_variant_is_worse():
    # Removing a contributing component should raise error above the full model.
    out = ablation_comparisons(_ablation_frame(1.0, 1.5))
    assert out.iloc[0]["delta_vs_full"] > 0
    assert bool(out.iloc[0]["reject_h0_holm_0.05"]) is True


def test_ablation_comparisons_negative_delta_when_variant_is_better():
    out = ablation_comparisons(_ablation_frame(1.5, 1.0))
    assert out.iloc[0]["delta_vs_full"] < 0


def test_ablation_comparisons_excludes_the_reference_variant():
    out = ablation_comparisons(_ablation_frame(1.0, 1.2))
    assert "full" not in set(out["variant"])


def test_ablation_comparisons_returns_empty_without_the_reference():
    df = _ablation_frame(1.0, 1.2)
    df = df[df["condition_value"] != "full"]
    assert ablation_comparisons(df).empty


def test_ablation_comparisons_pairs_by_seed():
    out = ablation_comparisons(_ablation_frame(1.0, 1.2, n=7))
    assert out.iloc[0]["n_pairs"] == 7
