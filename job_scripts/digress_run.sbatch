#!/bin/bash -l
# DiGress on QM9 (qm9_no_h). One script, three modes (--export=ALL,MODE=...); submit from the project root on GAIVI:
#   MODE=check   (default) imports, GPU visible, DiGress debug run (also processes QM9 once). 0:30
#   MODE=timing  3 epochs with one validation + 512-sample generation: s/epoch, VRAM, s per sampling pass. 0:45
#   MODE=train   full training, resumable; EPOCHS from the timing run (plan G1: ETA <= 10 GPU-h). Jobs <= 8 h; chain with
#                --dependency=afterok (the job exits 0 when it stops at its own time budget, so afterok still fires).
# Needs: environment from scripts/setup_digress_env.sh and data staged by scripts/stage_digress_qm9.py (synced to data/digress_qm9).
# Run `mkdir -p logs` in the project root BEFORE sbatch: Slurm does not create the --output directory and the job would fail silently.
#SBATCH --job-name=digress-qm9
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail

MODE=${MODE:-check}
PROJ=$(pwd)
ENV=/data/anquyle/conda_envs/vgmg-digress
QM9=$PROJ/data/digress_qm9/qm9_pyg/           # absolute: DiGress joins it to its own base path, so an absolute path wins
RUN=$PROJ/results/digress_qm9/$MODE            # one folder per mode; hydra, checkpoints and samples all live here
NAME=qm9_no_h
BUDGET_S=${BUDGET_S:-27000}                      # 7h30m: stop before the 8 h limit; checkpoints are saved every 5 epochs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
export PYTHONNOUSERSITE=1
echo "mode=$MODE job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is) commit=$(cat .sync_stamp 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY'
import torch, torch_geometric, rdkit, graph_tool, pytorch_lightning, numpy, pandas
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| pyg", torch_geometric.__version__,
      "| rdkit", rdkit.__version__, "| pl", pytorch_lightning.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
assert torch.cuda.is_available(), "no GPU visible"
PY
[ $? -eq 0 ] || exit 1

mkdir -p "$RUN"
cd "$PROJ/third_party/DiGress/src"
COMMON=(general.wandb=disabled general.gpus=1 train.seed=0 dataset.datadir="$QM9" dataset.num_workers=8 hydra.run.dir="$RUN/run")

case "$MODE" in
  check)
    python main.py +experiment=debug.yaml general.wandb=disabled general.gpus=1 dataset.datadir="$QM9" hydra.run.dir="$RUN/run"
    ;;
  timing)
    python main.py +experiment=$NAME.yaml "${COMMON[@]}" general.name=timing train.n_epochs=3 train.save_model=False \
      general.check_val_every_n_epochs=1 general.sample_every_val=1 general.samples_to_generate=512 \
      general.final_model_samples_to_generate=512 general.final_model_chains_to_save=0
    ;;
  train)
    [ -f "$RUN/DONE" ] && { echo "already finished"; exit 0; }
    : "${EPOCHS:?set --export=ALL,MODE=train,EPOCHS=<n> from the timing run}"
    CK=$RUN/run/checkpoints
    RESUME=()
    # the first job writes checkpoints/$NAME; every resumed job writes checkpoints/${NAME}_resume (DiGress renames the run)
    for d in "$CK/${NAME}_resume" "$CK/$NAME"; do
      [ -f "$d/last.ckpt" ] && { RESUME=(general.resume="$d/last.ckpt"); break; }
    done
    echo "resume: ${RESUME[*]:-none (fresh start)}  epochs=$EPOCHS budget=${BUDGET_S}s"
    START=$SECONDS
    timeout --signal=TERM --kill-after=120 "$BUDGET_S" \
      python main.py +experiment=$NAME.yaml "${COMMON[@]}" train.n_epochs="$EPOCHS" ${RESUME[@]+"${RESUME[@]}"}
    rc=$?
    if [ $rc -eq 0 ]; then touch "$RUN/DONE"; echo "training finished"; exit 0; fi
    # timeout exits 124 after TERM, but 137 if it had to KILL (PyTorch Lightning only flags SIGTERM and keeps training), so judge by elapsed time
    if [ $rc -ne 0 ] && [ $((SECONDS - START)) -ge "$BUDGET_S" ]; then echo "stopped at the time budget (rc=$rc); resubmit (chained job continues from last.ckpt)"; exit 0; fi
    exit $rc
    ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
