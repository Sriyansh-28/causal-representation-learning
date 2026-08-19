# Notebooks

Notebooks here are for **exploration and visualization only**. Every result
reported in the project README is produced by the executable scripts under
`experiments/`, not by a notebook, so that all findings are reproducible from
the command line.

- `01_data_exploration.ipynb` — IHDP covariate distributions, treatment
  imbalance and overlap diagnostics; the synthetic DGP's confounding knob.
- `02_results_exploration.ipynb` — loads the committed CSVs under
  `results/raw/` and re-plots them interactively.

Run `python experiments/run_experiment.py --config configs/baseline.yaml`
before opening `02_...`, or it will find no results to read.
