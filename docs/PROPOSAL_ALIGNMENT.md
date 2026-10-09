# Proposal alignment (what I promised the professor vs. what the repo does)

Source of truth for promises: `docs/proposal/vgmg_proposal.pdf`. Every row says where the promise is implemented
and whether anything differs. **Rule:** no row may silently differ; a difference is either (a) a strengthening
that I declare in the report, or (b) a correction that I mention to the professor if the proposal must be revised.
Topic changes after Oct 12 need a documented blocker and an email *before* the change.

| # | Proposal commitment | Where in repo | Status / difference |
|---|---|---|---|
| 1 | Question: SVDD-MLP vs best-of-N, mean SA/QED reward, validity, scaling in M | `docs/PREREGISTRATION.md` | Same. **Wording correction:** "equal denoiser evaluations" -> "equal candidates (N = M)" plus reported cost curves, because M children share one forward pass |
| 2 | Primary paper SVDD + DiGress, DDPO, Gao et al. | CONTRIBUTIONS, report | Same; report needs 12-20 refs (6930) |
| 3 | Frozen DiGress, QM9 heavy-atom checkpoint, T=500 | `vgmg/models/digress.py` (stub) | Pending G1/G2. Risk (i) unchanged |
| 4 | Value fn: MLP on atom/bond histograms + timestep embedding | `models/value_function.py` | MLP has a sinusoidal timestep embedding (4 frequencies + t). Histogram featurizer is part of the DiGress wrapper |
| 5 | Value fn: 3-layer message-passing GNN | `configs/sweeps/a2_gnn.yaml` | Scheduled Nov 2-8 as promised; dropped if it does not beat the MLP on validation MSE by Nov 6 (risk ii). Not implemented yet |
| 6 | MSE regression onto r(G0), 10k unguided rollouts, per reward | `rl/value_trainer.py`, `value_fn.train_chains: 10000` | Same. Adds: separate validation chains (1k), model selection on validation MSE |
| 7 | Selection argmax or softmax; SVDD-PM | `rl/svdd.py` | Same |
| 8 | Rewards SA (10-SA)/9 and QED, invalid -> 0 | `rl/rewards.py` | Same; invalid also covers disconnected molecules (proposal's validity criterion) |
| 9 | Docking proxy: stretch, dropped if not started by Nov 6 | `rewards.py` stub | Same |
| 10 | Data: QM9 133,885; three seeded (42) disjoint 1k splits to sanity-check reward code and value-fn predictions at t=0 | `data/splits`, `scripts/qm9_reference.py`, `value_fn_t0_check` | Same. Reward code checked on the test split; t=0 value-fn check implemented, runs once the wrapper exists. Splits never train or tune anything |
| 11 | N=2000 molecules per run, seeds 0,1,2 | `configs/default.yaml`, sweeps | Same for all sweeps. **Strengthening:** seeds 3-4 for the headline point only (M=10), pre-registered, run regardless of seeds 0-2 outcome |
| 12 | Metrics: mean + top-10% reward, validity, uniqueness, novelty vs QM9, ECFP4 diversity, denoiser calls, wall-clock | `eval/metrics.py`, summary.json | Same; denoiser calls = network forward passes (`nfe`); GPU recorded per run |
| 13 | Baselines B1 unguided, B2 best-of-N (cost = SVDD with same M), B3 SVDD-PM | `eval/baselines.py`, sweeps | Same. B2 implemented as one pool of 20 draws/slot per seed (nested best-of-5/10/20), valid because chains are reward-independent. Extra control: random selection |
| 14 | A1: M in {1,5,10,20}, SA and QED | `a1_m_sweep.yaml` | Same |
| 15 | A2: estimator PM / MLP / GNN at M=10 | `a1_m_sweep` (MLP), `a2_estimator`, `a2_gnn` | Same |
| 16 | Not run: DDPO fine-tuning, DiGress regressor guidance | PREREGISTRATION "Not run" | Same |
| 17 | Falsifiable outcome: gain over B2 at M=10 <= 2x across-seed std, or validity > 5 pts below B1 | `eval/stats.py::preregistered_verdict` | Same; "across-seed std" made precise (max of the two methods' stds) |
| 18 | Compute: GAIVI, ~50 GPU-h, arithmetic shown | `scripts/estimate_gpu_hours.py` | With proposal assumptions (0.1 s, 1.5x) the corrected accounting gives ~32 h (40 h with margin), under the 50 h budget; re-run with the measured s/step at G1 |
| 19 | Single-GPU fallback < 2 h (N=1000, M in {1,10}, 1 seed, MLP, SA) | `configs/sweeps/fallback.yaml`, `make fallback` | Config exists; first run in the week of Oct 26 (milestone) |
| 20 | Checkpoints every 50 batches; log config, seed, commit, metrics | `experiments.py::sample_molecules` (`partial.pt` after every chunk of 512), `utils/logging.py` | Same or finer: resume gives bit-identical molecules (tested) |
| 21 | Milestones Oct 12 -> Dec 4 | `docs/PLAN.md` section 4 | Same weeks and gates; B2 may start earlier than Nov 16-22 (cheap) |
| 22 | Risks (i)-(iv) with dates | `docs/PLAN.md` section 8 | Same; shrink order follows the proposal's rule (iii) after dropping the extra seeds |
| 23 | Contribution table | `CONTRIBUTIONS.md` | Same, more detailed |
| 24 | AI declaration: scaffolding, tests, proposal editing; "I will write and understand the DiGress wrapper and SVDD code" | report | **Must be updated honestly in the report:** Claude Code also drafted the implementation of SVDD, baselines, value trainer, metrics and the experiment runner; design, hypotheses, analysis and the ability to explain all code are mine. If the proposal is resubmitted, change that sentence now |

## Things to tell the professor (only if a revision/email is needed)
1. The compute-match wording (row 1) and that the corrected accounting lowers the GPU estimate (row 18).
2. Extra headline seeds (row 11) and the A2/B2 implementation detail (row 13) are strengthenings, not scope changes.
Nothing else departs from the approved scope.
