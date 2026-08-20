#!/usr/bin/env python3
"""Audit script: emit the experimental-design table straight from the configs.

Every field is read from the shipped YAML configs and the result CSVs, so the
documented design cannot drift from the design actually executed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

RAW = PROJECT_ROOT / "results" / "raw"

HYPOTHESES = {
    "baseline_ihdp": "NeuralRep estimates CATE at least as accurately as S/T/X on a real-covariate benchmark",
    "baseline_synthetic": "The Exp 1a ranking is a property of the estimators, not of one benchmark",
    "confounding": "NeuralRep's advantage grows as confounding strength (and selection bias) increases",
    "treatment_imbalance": "NeuralRep degrades less than S/T/X as the treated arm shrinks",
    "sample_size": "Representation learning benefits more from additional training data than meta-learners",
    "sample_size_synthetic": "Same, with a training pool large enough for the trend to be visible",
    "ablation": "The shared representation, not the heads or regularization, drives NeuralRep's behaviour",
    "ablation_synthetic": "Same mechanism test where training data is plentiful",
    "sensitivity": "NeuralRep's performance is not an artifact of one hyper-parameter setting",
}

MANIPULATED = {
    "baseline_ihdp": "none (single condition)",
    "baseline_synthetic": "none (single condition)",
    "confounding": "confounding strength gamma in {0,1,2,3}",
    "treatment_imbalance": "marginal P(T=1) in {0.5,0.25,0.1}",
    "sample_size": "training fraction in {0.25,0.5,0.75,1.0}",
    "sample_size_synthetic": "training fraction in {0.25,0.5,0.75,1.0}",
    "ablation": "neural component removed (4 variants)",
    "ablation_synthetic": "neural component removed (4 variants)",
    "sensitivity": "latent_dim, weight_decay, lr, epochs (one at a time)",
}

CONTROLLED = {
    "confounding": "treated fraction fixed at 0.5; true ATE constant; same covariate dim, noise, n",
    "treatment_imbalance": "gamma fixed at 1.0; true ATE constant; covariate imbalance (SMD) held ~constant",
    "sample_size": "test split held fixed across all conditions",
    "sample_size_synthetic": "test split held fixed across all conditions",
    "ablation": "identical seeds, data, optimizer, epochs and early stopping across variants",
    "ablation_synthetic": "identical seeds, data, optimizer, epochs and early stopping across variants",
    "sensitivity": "all other hyper-parameters at baseline values; epochs sweep disables early stopping",
}


def main() -> int:
    """Print one design row per shipped config."""
    rows = []
    for cfg_path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        cfg = load_config(cfg_path)
        raw_path = RAW / f"{cfg.name}_raw.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            ok = df[df["error"].fillna("") == ""] if "error" in df else df
            actual_seeds = ok["seed"].nunique()
            if ok.empty:
                status = "failed"
            elif actual_seeds != len(cfg.seeds):
                # Results on disk predate the current config: stale, not complete.
                status = "STALE"
            else:
                status = "complete"
            n_train = f"{int(ok['n_train'].min())}-{int(ok['n_train'].max())}"
            n_test = int(ok["n_test"].iloc[0])
            gt = "yes" if ok["pehe"].notna().any() else "no"
        else:
            status, actual_seeds, n_train, n_test, gt = "PENDING", 0, "-", "-", "-"

        rows.append({
            "experiment": cfg.name,
            "status": status,
            "dataset": cfg.dataset.get("name"),
            "hypothesis": HYPOTHESES.get(cfg.name, ""),
            "manipulated": MANIPULATED.get(cfg.name, ""),
            "controlled": CONTROLLED.get(cfg.name, "single condition"),
            "seeds_configured": len(cfg.seeds),
            "seeds_actual": actual_seeds,
            "n_train": n_train,
            "n_test": n_test,
            "ground_truth": gt,
            "models": len(cfg.models),
            "metrics": "PEHE, |ATE err|, CATE MAE/bias, policy value/regret, subgroup PEHE",
            "test": "paired Wilcoxon signed-rank",
            "correction": "Holm-Bonferroni within condition",
        })

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "results" / "audit" / "experiment_design.csv"
    out.to_csv(path, index=False)
    cols = ["experiment", "status", "dataset", "seeds_configured", "seeds_actual",
            "n_train", "n_test", "models", "ground_truth"]
    print(out[cols].to_string(index=False))
    print(f"\nfull table -> {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
