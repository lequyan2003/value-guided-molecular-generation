# Value-Guided Molecular Generation (SVDD on DiGress)

CIS 6930 (Generative AI, Fall 2026, USF). We apply Soft Value-Based Decoding (SVDD, Li et al., 2024) to a frozen
discrete graph diffusion model (DiGress, Vignac et al., ICLR 2023) trained on QM9, to steer molecule generation toward a
reward (synthetic accessibility, QED) **without retraining the generator**. At each reverse step we draw M candidates
from the denoiser, score them with a learned value function, and keep one. We compare against best-of-N sampling.

## Main result (SA reward, QM9, 2000 molecules per run, seeds 0-2, mean over seeds)

| Method | SA reward | Strict validity | DiGress forward passes / molecule |
|---|---|---|---|
| Unguided | 0.635 | 0.992 | 500 |
| Best-of-10 (selects on the true reward) | 0.799 | 1.000 | 5000 |
| SVDD-MLP, argmax (pre-registered) | 0.008 | 0.015 | 500 |
| SVDD-MLP, softmax alpha = 0.1 / 0.03 (exploratory) | 0.669 / 0.674 | 0.987 / 0.925 | 500 |
| SVDD-PM (posterior-mean value, no learned model) | 0.862 | 0.977 | 5500 |

The pre-registered SVDD-MLP rule fails: the learned value function is exploited by argmax selection and the guided
chains leave the data distribution (validity 0.015). Tempering the selection recovers validity but gives only a small
gain over unguided. Findings and caveats: one DiGress training seed, three sampling seeds, SA reward measured for the
SVDD arms only (QED for unguided and best-of-10).

## Install

Python 3.10+ (verified on 3.12), PyTorch 2.x. DiGress itself needs Python 3.9 and torch 2.0.1
(`scripts/setup_digress_env.sh`); the pinned submodule is `third_party/DiGress`
(`git submodule update --init`). The official QM9 checkpoint link is dead, so DiGress is trained with
`job_scripts/digress_run.sbatch` and frozen by SHA-256 before any guided run.

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yml
```

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
