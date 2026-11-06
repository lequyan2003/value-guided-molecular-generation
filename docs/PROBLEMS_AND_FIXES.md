# Problems we met and how we fixed them

A reference for the report and for questions after the talk. Each row: what went wrong, why, what we did, and where the evidence is.
Entries from 2026-11-06 come from the real runs; earlier design problems (the compute-match, leakage, missing timestep, ...) are listed in
`docs/PLAN.md`, "Critique rounds", and are only summarised in Section 4. Numbers marked **pilot** come from an early DiGress checkpoint (epoch 139), tiny value functions and
128 molecules. They test the pipeline and are **not results**; results come only from the frozen model's runs.

## 1. Model and environment

| # | Problem | Cause | Fix | Evidence |
|---|---|---|---|---|
| 1 | The official DiGress QM9 checkpoint cannot be downloaded | link is dead (404, DiGress issue #109), no mirror | we train DiGress ourselves with the repo's unmodified `qm9_no_h` config (seed 0, 1000 epochs), plus resumable job scripts | `docs/PLAN.md` Sec. 1; `job_scripts/digress_ws.sh`, `digress_run.sbatch` |
| 2 | `import graph_tool` fails (DiGress `main.py` line 1) | graph-tool is not pip-installable; only the SBM/planar datasets use it, QM9 does not | an empty stand-in module in `scripts/stubs` on `PYTHONPATH` | `scripts/stubs`; all debug and training runs started |
| 3 | `from src import utils` fails | the DiGress repo is not an installable package | `third_party/DiGress` on `PYTHONPATH` instead of `pip install -e` | job scripts |
| 4 | Doubt: does torch 2.0.1+cu118 run on the RTX 4090 (sm_89)? | the cu118 wheel lists no `sm_89` kernel | it runs through its sm_80 code; verified by a full DiGress debug run and then the real training (about 21 s/epoch, no errors) | handbacks; GPU_ISSUES.md |
| 5 | The RTX 5090 (sm_120) cannot run torch 2.0.1 | Blackwell needs CUDA 12.8 builds | a separate env with torch 2.7.1+cu128 on the 5090 only. **Disclosure:** if the 5090 checkpoint is the frozen one, the report must say it was trained on torch 2.7.1, not the pinned 2.0.1 | `docs/PLAN.md`; second training run on the 5090 |
| 6 | Loading DiGress files fails on torch >= 2.6 (`UnpicklingError`) | `torch.load` defaults to `weights_only=True`, which rejects DiGress's cached PyG file and Lightning checkpoints | workstation jobs set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; the wrapper loads with `weights_only=False` by hand (our own checkpoint, so full unpickling is safe) | `vgmg/models/digress.py::load_digress_checkpoint` |
| 7 | **First verification job died at checkpoint load** (`unexpected keys sampling_metrics.*`) | a real Lightning checkpoint holds buffers of DiGress's metric modules (target node/edge/valency distributions); our model is built without metric objects, and a random-weight fixture never has them | `network_state` drops only the metric-object prefixes (`sampling_metrics.`, `train_metrics.`, `val_metrics.`, `test_metrics.`); everything else is still loaded strictly. A new test reproduces the failure and checks a non-metric extra key still raises | commit 61594d5; `test_checkpoint_with_metric_buffers_loads` |
| 8 | A job failed after 11 s (`third_party/DiGress/src: No such file`) | the code folders were not synced to GAIVI | sync `third_party/DiGress`, `scripts/stubs`, `job_scripts` before submitting; logged for the tooling | `.claude/gpu/GPU_ISSUES.md` |
| 9 | Different validity numbers for "the same" model | DiGress counts the largest fragment as valid; we require one connected molecule | we report both (`validity_digress` and the strict `validity`); the pre-registered verdict uses the strict one | `vgmg/eval/metrics.py`; wrapper test (b) prints both |
| 10 | The proposal's compute match was ill-posed | the M children of one parent share a single network pass | count real network passes: T (unguided, SVDD-MLP), M*T (best-of-N), (M+1)*T (SVDD-PM); test (c) checks the counter against a forward hook | `docs/PLAN.md`; `check_nfe_accounting` |

## 2. Wrapper correctness (the part a reviewer will probe)

| # | Question | How we know it is right |
|---|---|---|
| 11 | Is our reverse step exactly DiGress's? | test (a): our p(G_{t-1}\|G_t) against the probabilities DiGress's own `sample_p_zs_given_zt` builds from the same input, at 5 timesteps. Epoch-139 checkpoint: max difference 6e-8 (atoms) and 1.2e-7 (bonds) against a 1e-6 limit. The test can fail: injecting a one-step time error gave 1e-4 |
| 12 | Is the sample distribution the same as DiGress's sampler? | test (b): 1000 samples each, DiGress's own metric code. Validity 0.987 vs 0.989, uniqueness 0.995 vs 0.991 (limit 0.015). Run time 81 s vs 80 s (about 0.08 s per batch-pass, matching the proposal's 0.1 s) |
| 13 | Does `nfe` count what the GPU did? | test (c): 49 counted vs 49 network graphs, chunking respected |
| 14 | Children are drawn from the caller's generator | inverse-CDF sampling; tested for frequency (5 sigma), determinism, independence from the global RNG |
| 15 | `encode` fails on some molecules in a round-trip test | aromatic N-H rings cannot be written without explicit hydrogens (DiGress's QM9 convention). The test allows exactly those cases; `encode` is used only for the t=0 sanity check |

Tests (a)-(c) are re-run on every frozen checkpoint (`job_scripts/verify_wrapper.sbatch`); until they pass on the frozen model, no result counts.

## 3. Findings from the first real-DiGress pilots (pilot, 2026-11-06)

| # | Observation | Diagnosis | What we do |
|---|---|---|---|
| 16 | Order of methods is as predicted: B1 SA 0.628 < random selection 0.645 < best-of-10 0.796 < SVDD-PM 0.853; pass counts exact (500 / 500 / 5000 / 5500) | the pipeline works end to end on real DiGress | nothing to fix; numbers are for debugging only |
| 17 | **SVDD-MLP with argmax collapses: validity 0.000** (also with a 40k-row value function: strict 0.000, fragments-allowed 0.203) | not a code bug (random selection through the same sampler gives 0.984). The value function is trained on unguided rollouts, where aromatic bonds are about 1 % of edges; greedy argmax over 500 steps follows its output into states it never saw (about 14 aromatic bonds on 9 atoms by t = 300). This is reward hacking by distribution shift: the proposal's risk R7 | argmax stays the pre-registered primary and is reported as it comes out. Added an exploratory softmax arm (`slot_a_svdd_softmax.yaml`) with alpha on the scale of the value spread; next remedy if time allows is retraining the value function on guided rollouts (the proposal's H4) |
| 18 | Softmax needs a matching temperature | children's values differ by only 0.01-0.05, so the earlier grid (alpha 0.1, 1.0) is nearly random selection. Early check on CPU (12 chains each): alpha 0.1 validity 1.00 / SA 0.61; 0.03 validity 0.92 / SA 0.66; 0.01 validity 0.08 | grid {0.1, 0.03, 0.01} for the exploratory arm |
| 19 | Value function is weak early: validation MSE 0.0168 vs 0.0177 for a constant (40k rows); t=0 MSE worse than a constant although the ranking is good (Spearman 0.59) | lossy histogram features, 20 timesteps per chain, distribution shift | reported as is: sibling ranking (H1) and the t=0 check run automatically; the 10k-chain value function is better data |
| 20 | SVDD-PM uniqueness is 0.825 | not diagnosed; a plausible reason is that PM scores children by a decoded prediction of the final molecule, which concentrates the choices | watch item: report diversity and novelty next to reward (risk R7) |

## 4. Design problems fixed earlier (details in `docs/PLAN.md`, Critique rounds 1-3)

Value function lacked the timestep (added an embedding); a row-level train/validation split leaked (now chain-level with separate RNG streams); invalid molecules were excluded from the primary reward (now all N, invalid = 0); the pre-registered rule was under-specified (gain > 2 x noise and validity drop <= 0.05, implemented); the featurizer belongs to the denoiser; stale caches and missing provenance (now hashed, with commit, GPU and checkpoint SHA-256 in every record); the SVDD paper already has best-of-N, so the novelty claim was rewritten; `pm_scorer` documented m*T passes instead of (m+1)*T (fixed in code); the GNN dates in comments were wrong (Nov 14-19, go/no-go Nov 20). Per-chain timesteps: each rollout chain draws its own random timesteps (the proposal is silent; declared in `docs/PROPOSAL_ALIGNMENT.md`).

## 5. Compute and tooling

| # | Problem | Fix |
|---|---|---|
| 21 | Guards refuse `$(...)`, shell variables and some forms of `ssh` | use the documented helper scripts with literal arguments; logged in `.claude/gpu/GPU_ISSUES.md` |
| 22 | `storage_report` on GAIVI takes more than 10 minutes (the `/data` scan) | read its output once it finished; logged |
| 23 | Fetching a checkpoint folder from a running job failed (`last-v1.ckpt` rotated mid-transfer) | used the epoch snapshots that arrived (epoch 139) and copied one to `results/digress_qm9/ckpt/`; a final fetch runs after the job ends |
| 24 | The GAIVI insurance training chain was redundant (the workstations run 15-21 s/epoch, all 1000 epochs in 4-6 h) | cancelled by hand (the permission system denied `scancel` to the agent); ledger updated |
| 25 | Schedule slipped: no GPU step ran between Oct 17 and Nov 5 | re-baselined to Slot A (dates only, content unchanged), recorded in `docs/PROPOSAL_ALIGNMENT.md` |

## 6. Known limitations to state in the report

One DiGress training seed (conclusions are conditional on it). The 5090 run uses torch 2.7.1. Value-function features are lossy histograms (the GNN arm tests this, go/no-go Nov 20). Best-of-N uses the true reward as its selector (an oracle), which is why it is the strong baseline. The wrapper is verified on an early checkpoint so far; tests (a)-(c) on the frozen model are still to be run.
