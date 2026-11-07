#!/bin/bash
# Wrapper verification (tests (a)-(c), scripts/verify_digress_wrapper.py) on a CSAIL workstation; the sbatch twin for GAIVI is
# verify_wrapper.sbatch. Run from the project root inside the vgmg-digress env (the runner's run_job.sh sets it up and pins one GPU):
#   bash job_scripts/verify_wrapper_ws.sh <checkpoint> [n] [out.json]
set -uo pipefail
CKPT=${1:?usage: verify_wrapper_ws.sh <checkpoint> [n] [out.json]}
N=${2:-2000}
OUT=${3:-results/digress_checks.json}
PROJ=$(pwd)
export PYTHONNOUSERSITE=1 PYTHONPATH="$PROJ:$PROJ/third_party/DiGress:$PROJ/third_party/DiGress/src:$PROJ/scripts/stubs${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1     # see digress_ws.sh: torch >= 2.6 (the 5090 env) refuses our own pickled files otherwise
echo "verify host=$(hostname) $(date -Is) gpu=${CUDA_VISIBLE_DEVICES:-unset} ckpt=$CKPT n=$N"
python scripts/verify_digress_wrapper.py --checkpoint "$CKPT" --n "$N" --out "$OUT"
