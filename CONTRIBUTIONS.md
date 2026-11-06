# Contributions

| Component | Origin | What I changed |
|---|---|---|
| DiGress backbone | cvignac/DiGress @ bdfa10c (MIT), git submodule `third_party/DiGress` (added 2026-10-17, pinned) | Code and the `qm9_no_h` config files unmodified; training uses only command-line overrides (`job_scripts/digress_run.sbatch`: wandb off, seed 0, absolute data path, 8 loader workers, `train.n_epochs` set from the measured s/epoch within the 10 GPU-h budget). Known upstream issue (README, 2026-09-10): `dim_ffy` is ignored, so the network uses 2048 instead of the paper's value; we train `main` as is. No official checkpoint exists any more (link 404, issue #109), so **I train it once (seed 0) and freeze it**; the wrapper (`vgmg/models/digress.py`, **not yet written**) adapts the posterior computation of `sample_p_zs_given_zt` |
| SVDD algorithm | Li et al., arXiv:2408.08252; hyanan16/SVDD-molecule @ 455e4d4 read for reference only | Own batched implementation (`vgmg/rl/svdd.py`) written from the paper; ported to discrete graph diffusion |
| Denoiser interface + NFE accounting | Mine | Batched interface, per-forward-pass counter used for compute-matched comparison (`vgmg/models/denoiser.py`) |
| Value function (MLP, timestep input) | Standard PyTorch | Own; GNN variant planned Nov 14-19 (go/no-go Nov 20: dropped if it does not beat the MLP on validation MSE), not implemented yet |
| Value training + diagnostics | Mine | Chain-level train/val split, validation model selection, per-t diagnostics, sibling-ranking test (`vgmg/rl/value_trainer.py`) |
| Baselines | Mine | Unguided, best-of-N (oracle selector), random selection, SVDD-PM (`vgmg/eval/baselines.py`, `vgmg/rl/svdd.py`) |
| Rewards | RDKit QED; Ertl & Schuffenhauer SA (RDKit Contrib) | Wrappers + [0,1] rescaling, connected-only validity; docking proxy not implemented |
| Metrics, statistics, verdict rule | Mine (RDKit for chemistry) | validity/uniqueness/novelty/diversity, paired bootstrap, pre-registered rule (`vgmg/eval/`) |
| Experiment runner, sweeps, aggregation | Mine | Deterministic run ids, resumable sweeps, summary tables (`vgmg/experiments.py`, `scripts/`) |
| `ToyDenoiser` | Mine | Smoke-test stand-in only; not a model of molecules |
| Scaffold, tests, docs drafting | Claude Code (declared) | Reviewed by me; see AI tool declaration in the report |

Update this table (with commit hashes) as components land.
