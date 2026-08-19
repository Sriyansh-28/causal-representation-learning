# Data

Raw data files are **not committed**; they are downloaded on first use and
cached under `data/raw/` (git-ignored). Every download is verified against a
pinned SHA-256 digest, so a corrupted or substituted file fails loudly instead
of silently changing results.

## IHDP (semi-synthetic)

The Infant Health and Development Program benchmark, in the NPCI form
distributed with Johansson, Shalit & Sontag (2016) and used by most subsequent
CATE papers.

| Property | Value |
|---|---|
| Source | `https://www.fredjo.com/files/ihdp_npci_1-100.{train,test}.npz` |
| Units | 747 (672 train / 75 test) |
| Covariates | 25 (6 continuous, 19 binary) |
| Realizations | 100 independent outcome simulations over fixed covariates |
| Treated fraction | 0.183 (train) |
| Ground truth | `mu0`, `mu1` response surfaces — so PEHE is exact |

**Why it is semi-synthetic.** The covariates and the treatment assignment come
from a real randomized experiment, but the outcomes are simulated. Hill (2011)
then *removed a non-random subset of the treated units* — specifically all
treated children of non-white mothers — which breaks the original
randomization and manufactures confounding. That is what makes IHDP a causal
benchmark rather than an RCT, and it is also why its treated group is only
~18% of the sample.

**What varies across the 100 realizations** (verified directly against the
files, not assumed): the pooled train+test covariate matrix is *identical* in
every realization, and the pooled treated fraction is exactly 0.1861
throughout. What changes per realization is (a) which 672 of the 747 units land
in train versus test, and (b) the simulated response surfaces. Realizations are
therefore genuine replicates of one population, differing in both the sample
split and the outcome draw. This project uses seed *i* to select realization
*i*, so across-seed variation reflects real variation in the data-generating
process rather than model initialisation alone.

Automatic download:

```bash
python scripts/download_data.py
```

## Synthetic (fully controlled)

Generated on the fly by `src/data/synthetic.py`; nothing to download. This DGP
exists because IHDP's confounding level and treatment mechanism are *fixed* and
therefore cannot serve as experimental factors. The generator exposes both as
independent, documented knobs. See the module docstring and the README's
"Experimental design" section for the exact equations.

## Adding another benchmark (e.g. ACIC)

Implement a loader that returns a `DataSplit` of `CausalDataset` objects
(`src/data/base.py`) and register it in `src/data/../experiments/registry.py`
under `build_dataset`. No model, metric, or experiment code needs to change.
