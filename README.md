# Value-Guided Molecular Generation (SVDD on DiGress)

CIS 6930 (Generative AI, Fall 2026, USF). We apply Soft Value-Based Decoding (SVDD, Li et al. 2024)
to a pretrained discrete graph diffusion model (DiGress, Vignac et al., ICLR 2023) to steer molecule
generation toward a reward (synthetic accessibility, QED, docking proxy) **without retraining the
generator**. At each reverse step we draw M candidates, score them with a learned value function, and
keep the best. Contribution: sample-efficiency analysis (M) and molecular reward design.

## Status (read this first)

| Piece | State |
|---|---|
| SVDD core, value fn (MLP), rewards (SA, QED), metrics, logging, tests | implemented |
| DiGress wrapper (`vgmg/models/digress.py`) | **not yet** - stub raises `NotImplementedError` |
| GNN value fn, docking proxy reward | **not yet** - stubs |
| `train_value_function.py`, `run_inference.py`, `run_ablations.py` | skeletons (need DiGress) |

The **smoke test currently uses `ToyDenoiser`, a stand-in that is not a generative model**; its numbers
only verify that the pipeline runs and is reproducible. They are not research results.

## Install

Python 3.10+ (verified on 3.12), PyTorch 2.x.

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yml
```

## Smoke test (~3 s on CPU, no download, no GPU)

```bash
make smoke-test          # or: python scripts/smoke_test.py
make test                # unit tests
```

Same seed gives identical output. Each run writes `results/logs/<name>_<timestamp>/` with `config.yaml`,
`seed.txt`, `git_commit.txt`, `metadata.json` (versions, GPU, wall-clock), `summary.json`, and the value-function checkpoint.
Override any config key: `python scripts/smoke_test.py svdd.num_candidates_m=10 seed=1`.

## Data and full experiments

```bash
make splits              # downloads QM9, writes data/splits/*.txt (seed 42)
python scripts/run_ablations.py configs/ablation_m.yaml      # needs the DiGress wrapper
python scripts/generate_results_table.py                     # -> results/tables/summary.tex
```

## Layout

`vgmg/` package (data, models, rl, eval, utils, tests) - `scripts/` entry points - `configs/` YAML -
`job_scripts/` Slurm (GAIVI) - `docs/proposal/`, `docs/report/` LaTeX (`make compile-proposal`, `make compile-report`).
Provenance of every component is in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
