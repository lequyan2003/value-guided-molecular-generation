#!/bin/bash
# Run sweep files (scripts/run_ablations.py) with the real DiGress model on a CSAIL workstation; the sbatch twin for GAIVI is
# run_experiments.sbatch. Run from the project root inside the vgmg-digress env (run_job.sh sets it up and pins one GPU).
# Finished runs are skipped, so the same command resumes after an interruption:
#   bash job_scripts/run_experiments_ws.sh "<sweep.yaml> [<sweep.yaml> ...]" "model.digress_checkpoint=<path> key=value ..."
set -uo pipefail
SWEEPS=${1:?usage: run_experiments_ws.sh "<sweep yaml> ..." "<key=value> ..."}
OVERRIDES=${2:?second argument: overrides, including model.digress_checkpoint=<path> (never implicit)}
PROJ=$(pwd)
export PYTHONNOUSERSITE=1 PYTHONPATH="$PROJ:$PROJ/third_party/DiGress:$PROJ/third_party/DiGress/src:$PROJ/scripts/stubs${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1     # see digress_ws.sh (only matters for torch >= 2.6, i.e. the 5090 env)
echo "experiments host=$(hostname) $(date -Is) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "sweeps: $SWEEPS"; echo "overrides: $OVERRIDES"
rc=0
for sweep in $SWEEPS; do
  echo "=== $sweep $(date -Is)"
  # shellcheck disable=SC2086  (word splitting of OVERRIDES is intended)
  python scripts/run_ablations.py "$sweep" $OVERRIDES || { rc=$?; echo "FAILED: $sweep (rc=$rc)"; break; }
done
echo "done rc=$rc $(date -Is)"
exit $rc
