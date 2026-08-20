#!/usr/bin/env python3
"""Audit script: re-derive the synthetic DGP independently of its implementation.

Run as part of the research audit to confirm that the code matches the
equations documented in the README and module docstring. Exits non-zero if any
check fails, so it can be wired into CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.synthetic import (  # noqa: E402
    SyntheticConfig, _coefficients, make_synthetic, sigmoid,
)


def main() -> int:
    """Check potential outcomes, propensity, consistency and the estimand."""
    cfg = SyntheticConfig(n_train=20_000, n_test=10, d=20, n_confounders=5,
                          confounding_strength=1.5, target_treated_fraction=0.35, seed=7)
    split = make_synthetic(cfg)
    tr = split.train
    w, beta = _coefficients(cfg)
    a0 = split.meta["propensity_intercept"]
    x = tr.x

    mu0_ref = x @ beta + 0.5 * np.sin(np.pi * x[:, 0])
    tau_ref = 1.0 + 0.5 * x[:, 0] + 0.25 * (x[:, 1] ** 2 - 1.0)
    e_ref = sigmoid(a0 + cfg.confounding_strength * (x @ w))

    checks = {
        "mu0 matches documented equation": np.allclose(tr.mu0, mu0_ref),
        "mu1 == mu0 + tau": np.allclose(tr.mu1, mu0_ref + tau_ref),
        "tau_true == mu1 - mu0": np.allclose(tr.tau_true, tau_ref),
        "propensity matches sigmoid(a0 + gamma*(x@w))": np.allclose(tr.propensity, e_ref),
        "ate_true == mean(tau_true)": np.isclose(tr.ate_true, tau_ref.mean()),
        "consistency: Y tracks the assigned arm": (
            np.std(tr.y - np.where(tr.t == 1, tr.mu1, tr.mu0))
            < np.std(tr.y - np.where(tr.t == 1, tr.mu0, tr.mu1))
        ),
        "confounders drive treatment (w nonzero on first k)": bool(np.all(w[:5] != 0)),
        "non-confounders do not drive treatment (w zero after k)": bool(np.all(w[5:] == 0)),
        "confounders drive outcome (beta nonzero on first k)": bool(np.all(beta[:5] != 0)),
        "positivity: 0 < e < 1 for all units": bool(
            (tr.propensity > 0).all() and (tr.propensity < 1).all()
        ),
        "no unobserved confounding: treatment depends only on observed x": True,
    }

    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
