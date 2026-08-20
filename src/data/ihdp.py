"""IHDP semi-synthetic benchmark loader.

Uses the standard NPCI replication files (100 outcome realizations over the
real IHDP covariates) distributed with Johansson, Shalit & Sontag (2016) and
used by essentially all subsequent CATE benchmarks. Each realization supplies
noiseless response surfaces ``mu0``/``mu1``, so PEHE is exactly computable.

Files are downloaded once, checksum-verified and cached under ``data/raw/``.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..utils.paths import RAW_DATA_DIR, ensure_dir
from .base import CausalDataset, DataSplit

IHDP_FILES: Dict[str, Dict[str, str]] = {
    "train": {
        "url": "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
        "sha256": "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240",
        "filename": "ihdp_npci_1-100.train.npz",
    },
    "test": {
        "url": "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
        "sha256": "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c",
        "filename": "ihdp_npci_1-100.test.npz",
    },
}

N_REALIZATIONS = 100


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ihdp(data_dir: Optional[Path] = None, verify: bool = True) -> Dict[str, Path]:
    """Download (if absent) and checksum-verify the IHDP replication files.

    Args:
        data_dir: Destination directory; defaults to ``data/raw``.
        verify: If True, compare SHA-256 against the pinned digests.

    Returns:
        Mapping of ``{"train": path, "test": path}``.

    Raises:
        RuntimeError: If a downloaded file fails checksum verification.
    """
    data_dir = ensure_dir(data_dir or RAW_DATA_DIR)
    paths: Dict[str, Path] = {}
    for split, spec in IHDP_FILES.items():
        dest = data_dir / spec["filename"]
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(spec["url"], tmp)
            tmp.rename(dest)
        if verify:
            digest = _sha256(dest)
            if digest != spec["sha256"]:
                raise RuntimeError(
                    f"checksum mismatch for {dest.name}: expected "
                    f"{spec['sha256']}, got {digest}. Delete the file and retry."
                )
        paths[split] = dest
    return paths


def _realization(npz: np.lib.npyio.NpzFile, i: int) -> CausalDataset:
    """Extract realization ``i`` from a loaded NPZ archive."""
    return CausalDataset(
        x=npz["x"][:, :, i],
        t=npz["t"][:, i],
        y=npz["yf"][:, i],
        mu0=npz["mu0"][:, i],
        mu1=npz["mu1"][:, i],
    )


def load_ihdp(
    realization: int = 0,
    data_dir: Optional[Path] = None,
    verify: bool = True,
) -> DataSplit:
    """Load one IHDP outcome realization as a train/test split.

    Args:
        realization: Index in ``[0, 100)``. Different realizations share the
            same covariates but have independently simulated outcomes, so they
            act as replicates for variance estimation.
        data_dir: Cache directory for the raw files.
        verify: Whether to checksum-verify the cached files.

    Returns:
        A :class:`DataSplit` with 672 training and 75 test units.
    """
    if not 0 <= realization < N_REALIZATIONS:
        raise ValueError(
            f"realization must be in [0, {N_REALIZATIONS}), got {realization}"
        )
    paths = download_ihdp(data_dir=data_dir, verify=verify)
    with np.load(paths["train"]) as tr, np.load(paths["test"]) as te:
        train = _realization(tr, realization)
        test = _realization(te, realization)
    meta = {
        "dataset": "ihdp",
        "realization": realization,
        "n_train": train.n,
        "n_test": test.n,
        "d": train.d,
        "treated_fraction_train": train.treated_fraction,
    }
    return DataSplit(train=train, test=test, meta=meta)


def induce_imbalance(split: DataSplit, target_treated_fraction: float, seed: int) -> DataSplit:
    """Reduce the treated fraction of the training split by dropping treated units.

    Units are dropped uniformly at random among the treated, which changes the
    marginal treated fraction while leaving ``P(X | T=1)`` unchanged. The test
    split is untouched.
    """
    if not 0.0 < target_treated_fraction < 1.0:
        raise ValueError("target_treated_fraction must lie in (0, 1)")
    train = split.train
    rng = np.random.default_rng(seed)
    treated = np.flatnonzero(train.t == 1.0)
    control = np.flatnonzero(train.t == 0.0)
    # Solve n_t / (n_t + n_c) = target for n_t, holding the controls fixed.
    n_keep = int(round(target_treated_fraction * control.size / (1.0 - target_treated_fraction)))
    n_keep = max(2, min(n_keep, treated.size))
    keep = np.concatenate([rng.choice(treated, size=n_keep, replace=False), control])
    keep.sort()
    meta = dict(split.meta)
    meta.update({
        "target_treated_fraction": target_treated_fraction,
        "achieved_treated_fraction": float(np.mean(train.t[keep])),
        "n_train": int(keep.size),
    })
    return DataSplit(train=train.subset(keep), test=split.test, meta=meta)
