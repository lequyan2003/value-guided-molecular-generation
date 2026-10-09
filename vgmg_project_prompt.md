# Claude Code Project Setup Prompt: SVDD-Guided Molecular Generation (CIS 6930)

## Context
I am starting a **high-standard research project** for CIS 6930 (Generative AI, Fall 2026, USF) on **Soft Value-Based Decoding (SVDD) for guided molecular generation**. I want to set up the project repository to meet:
- **Publication-ready code quality** (as if submitting to NeurIPS/ICML)
- **Reproducibility as first priority** (every run logged, seeds fixed, results regenerable)
- **Proposal in LaTeX** (proper academic formatting)
- **Documentation and code organization** matching top-tier labs (DeepMind, Meta, OpenAI, academic leaders)
- **Prof. Karimian's grading rubric** (CIS6930_Project_Guidelines.pdf):
  - Clean repo with smoke test running in <10 minutes on one GPU
  - Reproducible from README with pinned seeds and configs
  - CONTRIBUTIONS.md if using existing code
  - Real commit history (work over weeks, not one upload on Dec 11)
  - Provenance and evaluation correctness

## Project Summary
**Method:** Implement Soft Value-Based Decoding (SVDD, NeurIPS 2024) on top of a pretrained discrete graph diffusion model (DiGress, ICLR 2023) to guide molecular generation toward desired properties (QED, synthetic accessibility, docking score) *without retraining the generator*. Novel contribution: sample-efficiency analysis and molecular-specific reward design.

**Primary Papers:** SVDD (Li et al., NeurIPS 2024), DiGress (Vignac et al., ICLR 2023), DDPO (Black et al., ICLR 2024)

**Key Constraints:**
- ~50–70 GPU-hours total (inference-only, no heavy training)
- Fallback: <2 hours on single GPU (1k molecules, M ∈ {1, 10})
- 2 days/week active development, 8 weeks total (Oct 12 – Dec 11)
- Proposal due Oct 5 (2 pages + refs, PDF, submitted to Canvas)
- Report + code due Dec 11 (4–6 pages, PDF, GitHub repo link)

## Desired Repository Structure

```
value-guided-molecular-generation/
├── README.md                           # Entry point: what the project does, how to install, smoke test
├── CONTRIBUTIONS.md                    # Contribution table (base repo vs. novel work)
├── CIS6930_Project_Guidelines.pdf      # Course rubric (already provided)
├── Makefile                            # Common tasks: install, test, run-smoke-test, train, eval
├── requirements.txt                    # Pinned dependencies (exact versions)
├── environment.yml                     # Conda alternative (optional)
├── .gitignore                          # Proper ignore rules
├── .env.example                        # Config template
│
├── vgmg/                               # Main package (importable as `import vgmg`)
│   ├── __init__.py
│   ├── config.py                       # Centralized config management (argparse or hydra)
│   ├── constants.py                    # Hyperparameters, paths, model names
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py                  # PyTorch DataLoader(s) for QM9, ZINC subset
│   │   ├── preprocessing.py            # Tokenization, canonicalization (if needed)
│   │   └── download.py                 # Script to fetch QM9/ZINC (one command)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── digress.py                  # Load pretrained DiGress (wrapper around official checkpoint)
│   │   └── value_function.py           # Value function: MLPValueFunction, GNNValueFunction classes
│   │
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── svdd.py                     # Core SVDD logic: candidate sampling, value ranking, selection
│   │   ├── rewards.py                  # Reward functions: SA (synthetic accessibility), QED, docking_proxy
│   │   └── value_trainer.py            # Value function training loop (gradient descent)
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Metrics: validity, novelty, reward_mean, reward_std, etc.
│   │   ├── baselines.py                # Baseline sampling (unconditional DiGress, greedy M=1)
│   │   └── comparison.py               # Ablation table generation
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py                  # Logging setup (seed, run_id, config saved to logs/)
│   │   ├── reproducibility.py          # set_seed(), log_run_metadata()
│   │   ├── rdkit_utils.py              # RDKit wrappers (validity, novelty, QED, SA, etc.)
│   │   └── checkpointing.py            # Checkpoint/restore value function
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_svdd.py                # Unit test: SVDD forward pass
│       ├── test_rewards.py             # Unit test: reward functions
│       └── test_value_function.py      # Unit test: value function learning
│
├── scripts/
│   ├── train_value_function.py         # Main entry: load DiGress, data, train value function
│   ├── run_inference.py                # Inference: sample molecules, rank with value fn
│   ├── run_ablations.py                # Run all ablations (M, reward type, value arch) → CSV results
│   ├── generate_results_table.py       # Parse results → LaTeX tables for report
│   └── smoke_test.py                   # Quick sanity check: 50 molecules, M=5, single reward
│
├── configs/
│   ├── default.yaml                    # Default config (M, batch_size, reward weights, etc.)
│   ├── smoke_test.yaml                 # Minimal config for <2 hour run
│   ├── ablation_m.yaml                 # Sweep M ∈ {1, 5, 10, 20}
│   ├── ablation_rewards.yaml           # Sweep over {SA, QED, docking_proxy}
│   └── ablation_value_arch.yaml        # Sweep value function architectures
│
├── data/
│   ├── .gitkeep
│   ├── README.md                       # Instructions: where to download QM9, how to subsample
│   └── splits/                         # Train/val/test splits (small, check into git for reproducibility)
│       ├── qm9_indices_train_1k.txt
│       ├── qm9_indices_val_1k.txt
│       └── qm9_indices_test_1k.txt
│
├── results/
│   ├── .gitkeep
│   ├── logs/                           # Timestamped experiment logs (git-ignored, for local only)
│   │   └── run_20261005_1430/
│   │       ├── config.yaml             # Exact config used
│   │       ├── seed.txt                # Random seed logged
│   │       ├── metrics.csv             # Per-molecule results
│   │       ├── summary.json            # Mean/std of key metrics
│   │       └── value_fn_checkpoint.pt  # Value function weights
│   └── tables/                         # Final results tables (LaTeX format)
│       ├── ablation_m.tex
│       ├── ablation_rewards.tex
│       └── summary.tex
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory analysis: data, baseline results
│   ├── 02_results_visualize.ipynb      # Plot ablation curves, trade-offs
│   └── 03_failure_analysis.ipynb       # Analyze where value function fails
│
├── proposals/
│   ├── svdd_proposal.tex               # LaTeX source for Oct 5 proposal (2 pages + refs)
│   ├── svdd_proposal.pdf               # Compiled PDF (generated by Makefile)
│   ├── figures/                        # Proposal figures (block diagram, etc.)
│   └── refs.bib                        # BibTeX references (SVDD, DiGress, DDPO, etc.)
│
├── reports/
│   ├── svdd_report.tex                 # LaTeX source for Dec 11 report (4–6 pages)
│   ├── svdd_report.pdf                 # Compiled PDF
│   ├── figures/                        # Report figures (results plots, ablations)
│   ├── tables/                         # Generated from results/tables/
│   └── refs.bib                        # Same or extended reference list
│
├── job_scripts/                        # Cluster submission scripts (GAIVI/Slurm)
│   ├── train_value_fn.sh               # sbatch script for value function training
│   ├── run_inference.sh                # sbatch script for inference/evaluation
│   └── run_all_ablations.sh            # Master script launching all ablations
│
└── .github/workflows/                  # CI/CD (optional, for rigor)
    └── test.yml                        # Run unit tests on push
```

## Core Implementation Requirements

### 1. **Reproducibility (Non-Negotiable)**
- `reproducibility.py`: `set_seed(seed)` → torch, numpy, random, cuda all seeded
- Every run logs:
  - Random seed
  - Git commit hash (in stderr)
  - Python version, PyTorch version, CUDA version
  - Config used (saved as YAML)
  - Wall-clock time, GPU used
- Results table must include seed (or note "N=1 seed" if single run; report variance across ≥3 seeds if possible)
- **Smoke test must be reproducible**: same M, same reward, same seed → same metric values

### 2. **Configuration Management**
- Use **Hydra** or **argparse** (I suggest Hydra for top-tier projects)
- Config in `configs/default.yaml`:
  ```yaml
  seed: 42
  device: "cuda"
  model:
    digress_checkpoint: "pretrained_digress.pt"  # or path to huggingface
  data:
    dataset: "qm9"
    subset_size: 10000
  svdd:
    num_candidates_m: [1, 5, 10, 20]
    reward_type: "sa"  # or "qed", "docking_proxy"
    value_fn_arch: "mlp"  # or "gnn"
    value_train_steps: 5
  output_dir: "./results"
  ```
- Each run generates a timestamped log with config used

### 3. **Code Quality**
- **Type hints** throughout (e.g., `def sample(n: int) -> torch.Tensor`)
- **Docstrings** for all functions (NumPy style or Google style, consistent)
- **Modular design**: each file has one responsibility
  - `svdd.py` does SVDD
  - `rewards.py` does rewards
  - `value_trainer.py` does value function training
  - Main scripts call these, not re-implement
- **No Jupyter notebooks in root or scripts/** (notebooks live in `notebooks/`)
- **No dead code**: remove all debugging print statements before commit

### 4. **Testing**
- `tests/` directory with at least 3 unit tests:
  - `test_svdd.py`: forward pass on 10 molecules returns candidates ranked by value
  - `test_rewards.py`: SA/QED rewards return sensible values for simple molecules
  - `test_value_function.py`: value function training step reduces loss
- Run with `pytest tests/` (add to Makefile)

### 5. **Data Handling**
- **No large files in git**: data lives locally or on cluster
- `data/README.md` explains: "Run `python svdd/data/download.py` to fetch QM9 (1GB), then `scripts/prepare_splits.py` to create 1k/10k subsets"
- Splits (indices) committed to git for reproducibility
- Preprocessing idempotent: running twice gives same results

### 6. **Evaluation**
- `eval/metrics.py` computes:
  - **Validity**: % of generated molecules that are valid (RDKit can parse)
  - **Novelty**: % not in training set
  - **Reward mean/std**: average reward per condition
  - **Diversity**: Tanimoto distance (fingerprints)
- `eval/baselines.py`: unconditional DiGress, greedy (M=1), random selection
- `eval/comparison.py`: ablation table (M vs. reward type vs. value arch)
  - Output: CSV → LaTeX table via `generate_results_table.py`

### 7. **Logging & Tracking**
- Each run:
  ```
  results/logs/run_20261015_1430/
  ├── config.yaml
  ├── seed.txt
  ├── git_commit.txt
  ├── metrics.csv           # One row per molecule
  ├── summary.json          # {"reward_mean": ..., "validity": ..., ...}
  └── value_fn_checkpoint.pt
  ```
- `utils/logging.py`: setup function + context manager
  ```python
  with setup_logging("svdd_inference", config) as logger:
      logger.info(f"Seed: {seed}, Commit: {git_hash}")
      # run experiment
      save_results(logger.run_dir)
  ```

### 8. **Smoke Test**
- `scripts/smoke_test.py`:
  - Load DiGress (pretrained, small checkpoint)
  - Load 50 molecules from QM9
  - Run SVDD with M=5, single reward (SA)
  - Value function: MLP with 1 hidden layer
  - 1 gradient step
  - Compute metrics, print summary
  - **Runs in <2 minutes on GPU, <30 seconds on CPU** (if using tiny checkpoint)
- **This is your fallback if main experiments fail.**
- Makefile: `make smoke-test` runs it

### 9. **README**
- What the project does (1 paragraph)
- Requirements: Python 3.10+, PyTorch 2.0+, CUDA 11.8+
- Installation:
  ```bash
  git clone <repo>
  cd svdd-molecular-guidance
  pip install -r requirements.txt
  python -m svdd.data.download  # Fetch QM9
  ```
- **Smoke test**:
  ```bash
  python scripts/smoke_test.py
  # Or: make smoke-test
  ```
- Full inference (ablation):
  ```bash
  python scripts/run_ablations.py --config configs/default.yaml
  ```
- Results: `results/tables/summary.tex` and `results/logs/*/summary.json`

### 10. **Git & Provenance**
- `CONTRIBUTIONS.md`: 
  ```markdown
  | Component | Origin | What I Changed |
  |-----------|--------|---|
  | DiGress backbone | cvignac/DiGress (commit abc123) | Unmodified (inference only) |
  | SVDD core logic | SVDD paper (arXiv:2408.08252) | Implemented from scratch + molecular rewards |
  | Value function | Paper + standard PyTorch | Implemented MLP + GNN variants |
  | Eval metrics | RDKit, novelty check | Implemented per ablation design |
  ```
- Real commit history:
  - Week 1: `git add svdd/data/, svdd/models/, requirements.txt` → "Setup: data loaders and DiGress wrapper"
  - Week 2: `git add svdd/rl/svdd.py` → "Implement SVDD candidate sampling"
  - Week 3: `git add svdd/rl/value_trainer.py` → "Add value function training"
  - ...etc
  - **NOT**: One commit on Dec 11 with everything

### 11. **LaTeX Proposal & Report**
- **Proposal** (`proposals/svdd_proposal.tex`):
  - Preamble with proper packages (geometry, hyperref, natbib)
  - 2 pages max (excluding refs)
  - Sections: Question | Primary Paper & Background | Method | Data | Evaluation Plan | Compute Plan | Milestones | Risk & Fallback | Contribution Table | AI Tool Declaration
  - Compile: `make compile-proposal` → `svdd_proposal.pdf`
  
- **Report** (`reports/svdd_report.tex`):
  - 4–6 pages max (excluding refs, appendices)
  - Sections: Abstract | Intro & Research Question | Related Work | Method | Experimental Setup | Results | Ablations | Limitations & Failure Analysis | Conclusion | References | Contribution Table | AI Tool Declaration
  - Figures/tables auto-generated from `results/tables/`
  - Compile: `make compile-report`

### 12. **Makefile**
```makefile
.PHONY: help install test smoke-test ablations compile-proposal compile-report clean

help:
	@echo "Available targets: install, test, smoke-test, ablations, compile-proposal, compile-report, clean"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

smoke-test:
	python scripts/smoke_test.py

ablations:
	python scripts/run_ablations.py --config configs/default.yaml

compile-proposal:
	cd proposals && pdflatex -interaction=nonstopmode vgmg_proposal.tex && bibtex vgmg_proposal && pdflatex vgmg_proposal.tex && pdflatex vgmg_proposal.tex

compile-report:
	cd reports && pdflatex -interaction=nonstopmode vgmg_report.tex && bibtex vgmg_report && pdflatex vgmg_report.tex && pdflatex vgmg_report.tex

clean:
	rm -rf results/logs/* results/tables/*.pdf *.aux *.log *.bbl *.blg
```

---

## Tasks for Claude Code

1. **Create the directory structure** (all folders, __init__.py files, .gitkeep)
2. **Stub all Python files** with:
   - Proper docstrings (module-level)
   - Type hints on function signatures
   - Placeholder implementations (pass or return None)
   - Brief comments explaining what each function will do
3. **Create config files**:
   - `requirements.txt` (PyTorch, RDKit, hydra, pytest, pandas, numpy pinned versions)
   - `environment.yml` (conda equivalent)
   - `.gitignore` (Python, PyTorch, Jupyter, results/logs/*)
   - `configs/default.yaml` (full example config)
4. **Create LaTeX templates**:
   - `proposals/vgmg_proposal.tex` (skeleton: title, sections, refs)
   - `reports/vgmg_report.tex` (skeleton: title, sections, refs)
   - `proposals/refs.bib` (entries for SVDD, DiGress, DDPO, DDIM, etc.)
5. **Create scripts**:
   - `scripts/smoke_test.py` (complete, runnable)
   - `scripts/train_value_function.py` (skeleton)
   - `scripts/run_inference.py` (skeleton)
   - `scripts/run_ablations.py` (skeleton)
6. **Create documentation**:
   - `README.md` (complete, with install + smoke test instructions)
   - `data/README.md` (download instructions)
   - `CONTRIBUTIONS.md` (template)
7. **Makefile** (complete)
8. **Tests** (3 minimal unit tests with assertions)
9. **Workflow files** (optional): `.github/workflows/test.yml`

---

## Notes for Claude Code

- **Language**: Python 3.10+
- **Framework**: PyTorch 2.0+ (no TensorFlow)
- **Config management**: Hydra or argparse (I recommend Hydra for a top-tier project)
- **Linting**: Please add type hints and docstrings but don't enforce black/flake8 yet (can add CI later)
- **Git**: Initialize repo with `git init` but DON'T make initial commits yet (user will do that)
- **Testing**: pytest framework
- **Documentation**: Markdown for code, LaTeX for papers
- **Reproducibility obsession**: Every placeholder should remind future-me to log, seed, and pin.

---

## Success Criteria for This Setup

✅ User can run `make smoke-test` immediately and get results in <2 minutes  
✅ Directory structure is self-documenting (no guessing where code lives)  
✅ Proposal skeleton is ready to fill in by Oct 5  
✅ Each ablation (M, reward, value arch) has its own config  
✅ Results are auto-generated into LaTeX tables  
✅ README is complete enough that Prof. Karimian can clone and verify smoke test  
✅ CONTRIBUTIONS.md exists and is truthful  
✅ Zero dead code; every file has a purpose  
✅ All functions have type hints and docstrings  
✅ Logging is built-in from day 1  

---

## Submit This Prompt to Claude Code As-Is

You can paste this entire prompt directly into Claude Code's chat and it will scaffold the repo. Then, week by week, you fill in the implementations.
