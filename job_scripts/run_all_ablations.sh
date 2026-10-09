#!/bin/bash
# Submits one ablation sweep per config; each sweep point is a separately logged run.
set -euo pipefail
for cfg in configs/ablation_m.yaml configs/ablation_rewards.yaml configs/ablation_value_arch.yaml; do
  sbatch --gres=gpu:1 --time=08:00:00 --wrap "python scripts/run_ablations.py $cfg"
done
