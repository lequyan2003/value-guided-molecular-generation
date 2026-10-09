#!/bin/bash
#SBATCH --job-name=vgmg_infer
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/slurm_%j.out
set -euo pipefail
python scripts/run_inference.py --config "${1:-configs/default.yaml}" "${@:2}"
