#!/bin/bash
# DiGress on QM9 on a CSAIL workstation (the sbatch twin for GAIVI is digress_run.sbatch). Run from the project root, inside the
# vgmg-digress env (the runner's run_job.sh puts it on PATH and pins one GPU via CUDA_VISIBLE_DEVICES):
#   bash job_scripts/digress_ws.sh check            imports, GPU visible, DiGress debug run (also processes QM9 once)
#   bash job_scripts/digress_ws.sh timing           3 epochs, one validation + 512-sample generation
#   EPOCHS=<n> bash job_scripts/digress_ws.sh train full training, resumable (checkpoints every 5 epochs)
# graph-tool is not installable with pip, so scripts/stubs (an empty stand-in) is put on PYTHONPATH; QM9 never uses it.
set -uo pipefail
MODE=${1:-check}
PROJ=$(pwd)
QM9=$PROJ/data/digress_qm9/qm9_pyg/
RUN=$PROJ/results/digress_qm9/$MODE
NAME=qm9_no_h
# third_party/DiGress on the path stands in for `pip install -e` (main.py does `from src import utils`)
export PYTHONNOUSERSITE=1 PYTHONPATH="$PROJ/third_party/DiGress:$PROJ/scripts/stubs${PYTHONPATH:+:$PYTHONPATH}"
# torch >= 2.6 loads with weights_only=True by default, which rejects DiGress's own cached PyG file and Lightning checkpoints
# (both written by us). This switch restores the old behaviour without touching the pinned DiGress source; torch 2.0.1 ignores it.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "mode=$MODE host=$(hostname) $(date -Is) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch, torch_geometric, rdkit, graph_tool, pytorch_lightning, numpy, pandas
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), "arch", torch.cuda.get_arch_list(),
      "| pyg", torch_geometric.__version__, "| rdkit", rdkit.__version__, "| pl", pytorch_lightning.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__)
assert torch.cuda.is_available(), "no GPU visible"
x = torch.randn(256, 256, device="cuda"); print("cuda matmul ok", float((x @ x).sum().isfinite()))
PY
[ $? -eq 0 ] || exit 1
mkdir -p "$RUN"
cd "$PROJ/third_party/DiGress/src" || exit 1
COMMON=(general.wandb=disabled general.gpus=1 train.seed=0 dataset.datadir="$QM9" dataset.num_workers=8 hydra.run.dir="$RUN/run")
case "$MODE" in
  check)
    python main.py +experiment=debug.yaml general.wandb=disabled general.gpus=1 dataset.datadir="$QM9" hydra.run.dir="$RUN/run" ;;
  timing)
    python main.py +experiment=$NAME.yaml "${COMMON[@]}" general.name=timing train.n_epochs=3 train.save_model=False \
      general.check_val_every_n_epochs=1 general.sample_every_val=1 general.samples_to_generate=512 \
      general.final_model_samples_to_generate=512 general.final_model_chains_to_save=0 ;;
  train)
    : "${EPOCHS:?set EPOCHS=<n> from the timing run}"
    CK=$RUN/run/checkpoints; RESUME=()
    for d in "$CK/${NAME}_resume" "$CK/$NAME"; do
      [ -f "$d/last.ckpt" ] && { RESUME=(general.resume="$d/last.ckpt"); break; }
    done
    python main.py +experiment=$NAME.yaml "${COMMON[@]}" train.n_epochs="$EPOCHS" ${RESUME[@]+"${RESUME[@]}"} ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
