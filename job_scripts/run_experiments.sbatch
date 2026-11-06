#!/bin/bash -l
# Run sweep files (scripts/run_ablations.py) with the real DiGress model, inside the vgmg-digress env. Finished runs are
# skipped, so a preempted or timed-out job is simply resubmitted. Submit from the project root on GAIVI:
#   sbatch --export=ALL,SWEEPS="configs/sweeps/slot_a_b1.yaml configs/sweeps/slot_a_b2.yaml",OVERRIDES="model.digress_checkpoint=<path> output_dir=./results" job_scripts/run_experiments.sbatch
# SWEEPS: space-separated sweep files, run in order. OVERRIDES: space-separated key=value pairs (no spaces inside a value)
# applied after each sweep's own settings. The checkpoint's SHA-256 is written into every run record.
#SBATCH --job-name=vgmg-exp
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail

: "${SWEEPS:?set --export=ALL,SWEEPS=\"<sweep yaml> ...\"}"
: "${OVERRIDES:?set --export=ALL,OVERRIDES=\"model.digress_checkpoint=<path> ...\" (the checkpoint is never implicit)}"
PROJ=$(pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/anquyle/conda_envs/vgmg-digress
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ:$PROJ/third_party/DiGress:$PROJ/third_party/DiGress/src:$PROJ/scripts/stubs${PYTHONPATH:+:$PYTHONPATH}"
echo "experiments job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is) commit=$(cat .sync_stamp 2>/dev/null || echo unknown)"
echo "sweeps: $SWEEPS"
echo "overrides: $OVERRIDES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rc=0
for sweep in $SWEEPS; do
  echo "=== $sweep $(date -Is)"
  # shellcheck disable=SC2086  (word splitting of OVERRIDES is intended)
  python scripts/run_ablations.py "$sweep" $OVERRIDES || { rc=$?; echo "FAILED: $sweep (rc=$rc)"; break; }
done
echo "done rc=$rc $(date -Is)"
exit $rc
