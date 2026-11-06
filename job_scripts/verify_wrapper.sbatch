#!/bin/bash -l
# Exactness tests (a)-(c) of the DiGress wrapper on a checkpoint (scripts/verify_digress_wrapper.py). Submit from the project root on GAIVI:
#   sbatch --export=ALL,CKPT=results/digress_qm9/ckpt/last.ckpt,N=2000,OUT=results/digress_checks.json job_scripts/verify_wrapper.sbatch
# CKPT is a project-relative path; N = molecules per sampler for test (b); OUT is the JSON report. Exit 1 if any test fails.
#SBATCH --job-name=verify-wrapper
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail

: "${CKPT:?set --export=ALL,CKPT=<project-relative checkpoint path>}"
N=${N:-2000}
OUT=${OUT:-results/digress_checks.json}
PROJ=$(pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/anquyle/conda_envs/vgmg-digress
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ:$PROJ/third_party/DiGress:$PROJ/third_party/DiGress/src:$PROJ/scripts/stubs${PYTHONPATH:+:$PYTHONPATH}"
echo "verify job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is) commit=$(cat .sync_stamp 2>/dev/null || echo unknown) ckpt=$CKPT n=$N"
python scripts/verify_digress_wrapper.py --checkpoint "$CKPT" --n "$N" --out "$OUT"
