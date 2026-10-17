#!/bin/bash
# DiGress environment on GAIVI. Run BY THE USER in a login-node shell (agents never create environments), from the project root:
#   bash scripts/setup_digress_env.sh
# Versions follow third_party/DiGress/README.md (tested with torch 2.0.1, torch_geometric 2.3.1). GAIVI drivers cap CUDA at 12.1;
# the cu118 wheel runs on them. Nothing is imported here: the login node cannot import numpy/torch (x86-64-v2), verify with
# job_scripts/digress_run.sbatch MODE=check. orca is not compiled: only the SBM/planar metrics use it.
set -euo pipefail
ENV=/data/anquyle/conda_envs/vgmg-digress
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$ENV" -c conda-forge python=3.9 rdkit=2023.03.2 graph-tool=2.45   # one solve; if graph-tool fails see scripts/stubs
conda activate "$ENV"
export PYTHONNOUSERSITE=1
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r third_party/DiGress/requirements.txt
pip install -e third_party/DiGress
pip install pytest   # our vgmg tests; torch/numpy/pandas/rdkit/PyYAML already come from the lines above
# Freeze for the report (plan Stage 1 item 7). Commit both files after review.
conda env export -p "$ENV" --no-builds > environment-gaivi.yml
pip freeze > environment-gaivi.pip.txt
echo "env ready at $ENV; next: submit job_scripts/digress_run.sbatch with MODE=check"
