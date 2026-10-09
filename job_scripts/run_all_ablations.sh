#!/bin/bash
# Submit the sweeps in dependency order: B1/B2 are independent; A1 and A2 train value functions on demand.
# Submit only after gate G3 and with the user's go-ahead (see docs/PLAN.md).
set -euo pipefail
for cfg in b1_unguided b2_best_of_n a1_m_sweep a2_estimator h2_selection; do
  sbatch job_scripts/run_sweep.sh "configs/sweeps/${cfg}.yaml"
done
