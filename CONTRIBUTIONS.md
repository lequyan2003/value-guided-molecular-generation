# Contributions

| Component | Origin | What I changed |
|---|---|---|
| DiGress backbone + QM9 checkpoint | cvignac/DiGress @ bdfa10c (MIT) | Unmodified, inference only. **Not yet integrated** (`vgmg/models/digress.py` is a stub) |
| SVDD algorithm | Li et al., arXiv:2408.08252; hyanan16/SVDD-molecule @ 455e4d4 read for reference only | Own batched implementation (`vgmg/rl/svdd.py`) written from the paper; ported to discrete graph diffusion |
| Denoiser interface + NFE accounting | Mine | Batched interface, per-forward-pass counter used for compute-matched comparison (`vgmg/models/denoiser.py`) |
| Value function (MLP, timestep input) | Standard PyTorch | Own; GNN variant planned Nov 2-8 (dropped if it fails the Nov 6 rule), not implemented yet |
| Value training + diagnostics | Mine | Chain-level train/val split, validation model selection, per-t diagnostics, sibling-ranking test (`vgmg/rl/value_trainer.py`) |
| Baselines | Mine | Unguided, best-of-N (oracle selector), random selection, SVDD-PM (`vgmg/eval/baselines.py`, `vgmg/rl/svdd.py`) |
| Rewards | RDKit QED; Ertl & Schuffenhauer SA (RDKit Contrib) | Wrappers + [0,1] rescaling, connected-only validity; docking proxy not implemented |
| Metrics, statistics, verdict rule | Mine (RDKit for chemistry) | validity/uniqueness/novelty/diversity, paired bootstrap, pre-registered rule (`vgmg/eval/`) |
| Experiment runner, sweeps, aggregation | Mine | Deterministic run ids, resumable sweeps, summary tables (`vgmg/experiments.py`, `scripts/`) |
| `ToyDenoiser` | Mine | Smoke-test stand-in only; not a model of molecules |
| Scaffold, tests, docs drafting | Claude Code (declared) | Reviewed by me; see AI tool declaration in the report |

Update this table (with commit hashes) as components land.
