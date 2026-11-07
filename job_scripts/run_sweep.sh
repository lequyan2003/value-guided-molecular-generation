#!/bin/bash
# Run one sweep file on GAIVI (class partition first, see docs/PLAN.md). Safe to resubmit: finished runs are skipped.
# usage: sbatch job_scripts/run_sweep.sh configs/sweeps/a1_m_sweep.yaml
#SBATCH --job-name=vgmg_sweep
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
python scripts/run_ablations.py "${1:?sweep yaml required}"
