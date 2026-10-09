#!/bin/bash
#SBATCH --job-name=vgmg_train_vf
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/slurm_%j.out
set -euo pipefail
python scripts/train_value_function.py --config "${1:-configs/default.yaml}" "${@:2}"
