#!/usr/bin/env bash
# Reproduce every experiment in this project, then rebuild the README results.
#
# Runtime is dominated by the synthetic sweeps; on a 4-core CPU machine the
# full set takes roughly 1-2 hours. Pass a config name to run just one, e.g.
#   bash scripts/run_all.sh baseline
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
  baseline
  baseline_synthetic
  confounding
  treatment_imbalance
  sample_size
  sample_size_synthetic
  ablation
  ablation_synthetic
  sensitivity
)

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
fi

for cfg in "${CONFIGS[@]}"; do
  echo "=== ${cfg} ($(date -u +%H:%M:%S) UTC)"
  python3 experiments/run_experiment.py --config "configs/${cfg}.yaml" --quiet
done

echo "=== rebuilding README results section"
python3 scripts/build_readme.py
echo "=== done"
