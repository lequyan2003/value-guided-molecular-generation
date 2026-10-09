#!/bin/bash
# Submit the sweeps in dependency order: B1/B2 are independent; A1 and A2 train value functions on demand.
# Submit only after gate G3 and with the user's go-ahead (see docs/PLAN.md).
set -euo pipefail
for cfg in b1_unguided a1_m_sweep b2_best_of_n a2_estimator extra_seeds_svdd extra_seeds_b2 h2_selection; do
  sbatch job_scripts/run_sweep.sh "configs/sweeps/${cfg}.yaml"
done
