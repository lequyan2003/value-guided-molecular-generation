# Value-Guided Molecular Generation (SVDD on DiGress)

CIS 6930 (Generative AI, Fall 2026, USF). We apply Soft Value-Based Decoding (SVDD, Li et al., 2024) to a frozen
discrete graph diffusion model (DiGress, Vignac et al., ICLR 2023) trained on QM9, to steer molecule generation toward a
reward (synthetic accessibility, QED) **without retraining the generator**. At each reverse step we draw M candidates
from the denoiser, score them with a learned value function, and keep one. We compare against best-of-N sampling.

## Main result (QM9, 2000 molecules per run, frozen DiGress, M = 10, mean ± std over seeds)

SA reward = (10 - SA)/9, invalid molecules score 0. Every number comes from the run records in
[release_results/](release_results/) (`make reproduce` regenerates this table with no GPU).

| Method | n seeds | SA reward | validity | QED reward | validity | forward passes / molecule |
|---|---|---|---|---|---|---|
| Unguided DiGress (B1) | 5 | 0.636 ± 0.003 | 0.992 | 0.463 ± 0.002 | 0.992 | 500 |
| Best-of-10, selects on the true reward (B2) | 5 | 0.800 ± 0.002 | 1.000 | 0.568 ± 0.000 | 1.000 | 5000 |
| Random selection among 10 (control) | 3 | 0.636 ± 0.002 | 0.991 | 0.463 ± 0.003 | 0.991 | 500 |
| **SVDD-MLP, argmax (pre-registered primary)** | 5 | **0.029 ± 0.052** | **0.045** | **0.007 ± 0.011** | **0.018** | 500 |
| SVDD-GNN, argmax (ablation A2) | 3 | 0.140 ± 0.131 | 0.227 | 0.027 ± 0.029 | 0.056 | 500 |
| SVDD-PM, no learned value (B3) | 3 | 0.862 ± 0.005 | 0.977 | 0.590 ± 0.001 | 0.980 | 5500 |
| *Exploratory:* SVDD-MLP softmax, alpha = 0.1 | 3 | 0.669 ± 0.001 | 0.987 | not run | | 500 |
| *Exploratory:* SVDD-MLP refit on its own guided rollouts (H4) | 3 | 0.617 ± 0.077 | 0.817 | 0.374 ± 0.188 | 0.750 | 500 |

**Verdict (pre-registered rule, `docs/PREREGISTRATION.md`): the hypothesis is rejected.** SVDD-MLP does not beat
best-of-10 for either reward (SA gain -0.770 vs noise 0.052; QED gain -0.560 vs noise 0.011; validity 0.947 / 0.974 below
unguided, 5 paired seeds; `release_results/tables/verdict.json`). The learned value function is exploited by argmax
selection: it ranks the children of one parent barely better than chance (within-parent Spearman between -0.06 and 0.31) and the
guided chains leave the data distribution, so the molecules become invalid. The collapse worsens with M (A1: M = 1 equals
unguided, M = 5 and 20 collapse). Refitting the value function on its own guided rollouts (H4, exploratory) removes most of the
collapse, which supports the distribution-shift explanation, but it still does not beat best-of-10 or unguided.
The training-free SVDD-PM is the strongest method but spends (M+1)T passes and loses diversity (uniqueness 0.54).
Caveats: one DiGress training run (all conclusions are conditional on that model), the best-of-N selector uses the true
reward (an oracle), and wall-clock is only comparable within one GPU type (`release_results/tables/wall_clock_by_gpu.csv`).

## Install

Python 3.10+ (verified on 3.12), PyTorch 2.x. DiGress itself needs Python 3.9 and torch 2.0.1
(`scripts/setup_digress_env.sh`); the pinned submodule is `third_party/DiGress`
(`git submodule update --init`). The official QM9 checkpoint link is dead, so DiGress is trained with
`job_scripts/digress_run.sbatch` and frozen by SHA-256 before any guided run.

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yml
```

## Reproduce the headline result (no GPU, about 1 minute)

```bash
make reproduce      # regenerates results table, pre-registered verdict, per-GPU wall-clock and figures from release_results/
```

`make reproduce` reads the committed run records (`summary.json`, `molecules.csv`, config, seed, commit, GPU and the checkpoint
SHA-256 of every run) and writes `release_results/tables/` and `release_results/figures/`. Re-running the experiments needs the
frozen checkpoint (below) and a GPU: `make headline` runs every proposal experiment, `make exploratory` the two extra arms;
`job_scripts/` has the Slurm and workstation launchers used.

**Frozen DiGress checkpoint.** It is 128 MB, so it is not in git: download it from the
[`checkpoint-v1` release](https://github.com/lequyan2003/value-guided-molecular-generation/releases/tag/checkpoint-v1) into
`results/digress_qm9/ckpt/` and check `sha256sum -c SHA256SUMS.txt`. It is our own training run (seed 0, 1000 epochs, the repository's
unmodified `qm9_no_h` config, torch 2.0.1, validity 99.2 %), SHA-256
`ef376a81e7433089a037412a8653b88168934a3279bfdb4f4068810c550974d2`; every run record stores this hash. To recreate it, run
`job_scripts/digress_run.sbatch` (or `job_scripts/digress_ws.sh train`; about 6.4 h on one RTX 4090). Training is not bit-identical
across machines, so a retrained model has a different hash and numbers that agree only approximately.

## Quick checks (CPU, no download)

```bash
make test           # unit tests: evaluation correctness, statistics, determinism, DiGress wrapper shapes
make smoke-test     # all five methods on a toy generator (not a generative model; checks the pipeline only)
```

Each run writes `config.yaml`, `seed.txt`, `git_commit.txt`, `metadata.json` (versions, GPU, wall-clock),
`summary.json` and `molecules.csv` to a run folder; any config key can be overridden on the command line, e.g.
`python scripts/smoke_test.py svdd.num_candidates_m=10 seed=1`.

## Full experiments

```bash
make splits       # QM9 download and seeded 1k splits
make reference    # reward statistics of real QM9 molecules
python scripts/run_inference.py method=svdd_pm svdd.reward_type=sa seed=1 model.name=digress
make sweep SWEEP=configs/sweeps/<sweep>.yaml   # finished runs are skipped
make results      # aggregate runs into summary table and pre-registered verdict
```

## Layout

`vgmg/` package (`data`, `models`, `rl`, `eval`, `utils`, `tests`) - `scripts/` entry points - `configs/` YAML
(`sweeps/` is the experiment matrix) - `job_scripts/` Slurm and workstation launchers - `third_party/DiGress` pinned
submodule. Provenance of every component is in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
