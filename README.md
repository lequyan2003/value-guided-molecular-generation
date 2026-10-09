# Value-Guided Molecular Generation (SVDD on DiGress)

CIS 6930 (Generative AI, Fall 2026, USF). We apply Soft Value-Based Decoding (SVDD, Li et al. 2024)
to a pretrained discrete graph diffusion model (DiGress, Vignac et al., ICLR 2023) to steer molecule
generation toward a reward (synthetic accessibility, QED, docking proxy) **without retraining the
generator**. At each reverse step we draw M candidates, score them with a learned value function, and
keep the best. Contribution: sample-efficiency analysis (M) and molecular reward design.

## Status (read this first)

| Piece | State |
|---|---|
| Samplers (unguided, best-of-N, SVDD-PM, SVDD-value, random-select), value fn (MLP + timestep), rewards (SA, QED), metrics, stats, sweeps, tests | implemented, run on a toy generator |
| DiGress wrapper (`vgmg/models/digress.py`) | **not yet** - stub raises `NotImplementedError` (week 2) |
| GNN value fn | **not yet** - planned Nov 2-8, dropped if it does not beat the MLP on validation MSE by Nov 6 |
| Docking proxy | **not yet** - stretch, dropped if not started by Nov 6 |

The **smoke test uses `ToyDenoiser`, a stand-in that is not a generative model**; its numbers only show the
pipeline runs and is reproducible. They are not research results.

Project plan, gates and rubric mapping: [docs/PLAN.md](docs/PLAN.md).

## Install

Python 3.10+ (verified on 3.12), PyTorch 2.x.

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yml
```

## Smoke test (~3 s on CPU, no download, no GPU)

```bash
make smoke-test          # all five methods on the toy generator, ~3 s
make test                # 27 unit tests (evaluation correctness, stats, determinism)
```

Same seed gives identical output. Each run writes `results/logs/<name>_<timestamp>/` with `config.yaml`,
`seed.txt`, `git_commit.txt`, `metadata.json` (versions, GPU, wall-clock), `summary.json`, and the value-function checkpoint.
Override any config key: `python scripts/smoke_test.py svdd.num_candidates_m=10 seed=1`.

## Data and full experiments

```bash
make splits                                   # QM9 download + seeded 1k sanity splits (data/splits, committed)
make reference                                # reward stats of real QM9 test molecules
python scripts/run_inference.py method=svdd_pm svdd.reward_type=qed seed=1     # one run
make sweep SWEEP=configs/sweeps/a1_m_sweep.yaml                                # a sweep; finished runs are skipped
make results                                  # -> results/tables/{summary.csv,summary.tex,verdict.json}
```
Sweeps: `b1_unguided`, `b2_best_of_n`, `a1_m_sweep`, `a2_estimator`, `h2_selection`, `t3_gnn`. Real runs need the DiGress
wrapper (`model.name=digress`). The analysis rule is fixed in [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md).

## Environments
Dev/CI: `requirements.txt` (pinned, CPU torch). DiGress was tested on Python 3.9, PyTorch 2.0.1 (CUDA 11.8), torch_geometric 2.3.1,
rdkit 2023.03.2; this package avoids 3.10-only syntax so it can run in that environment (CI job `compat-digress-env`). The cluster
environment is built and verified at gate G1 (see `docs/PLAN.md`); record the exact versions here once it works.

## Layout

`vgmg/` package (data, models, rl, eval, utils, tests) - `scripts/` entry points - `configs/` YAML (`sweeps/` = experiment matrix) -
`job_scripts/` Slurm (GAIVI) - `docs/` plan, pre-registration, proposal, report, `docs/report/` LaTeX (`make compile-proposal`, `make compile-report`).
Provenance of every component is in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
