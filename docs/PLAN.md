# Project plan v3: SVDD on frozen DiGress (QM9), presenting in Slot A

_Owner: Abel Le (6930). Rewritten 2026-10-09 after claim verification (`docs/VERIFICATION.md`). The proposal text and this plan
are kept aligned line by line in `docs/PROPOSAL_ALIGNMENT.md`. Update the gate log (end of file) at every gate._

## 0. Summary

**Slot status (2026-11-05): Slot A is chosen (talk Fri Nov 13, slides due Thu Nov 12 11:59 PM).** The professor's Slot A bar (email): "a working pipeline, your main result, and one baseline; ablations described rather than shown"; Slot C "needs the complete set". Nothing GPU-side ran between Oct 17 and Nov 5, so the dates of Stages 1-4 below were **re-baselined on 2026-11-05 (Sec. 4a)**. Only dates, gates and the Slot A scope floor changed; every technical item (method, baselines, ablations, metrics, rule, seeds, N) is exactly the proposal's. The original dates are kept in each stage's text for the record and marked "was". The proposal's own risk (v) (ask for Slot C) remains the escape hatch, now triggered at gate D2 (Sec. 4a).

**Objective.** Measure whether SVDD (value-guided decoding) on a frozen DiGress model beats best-of-N on QM9 SA/QED reward at
matched candidates and at measured cost, and explain the result.
**Where it is graded.** 65/100 live talk and Q&A (Slot A, Fri Nov 13), 15 report, 15 code, 5 proposal (Dec 11).
**Strategy.** Slot A rewards a working pipeline, one main result and one baseline by Nov 8; everything else (full ablations, GNN,
extra seeds, failure analysis) lands before Nov 29 and goes into the report. Slot C (Dec 4) is the escape hatch if the Oct 30 gate fails.
**Critical path.** (1) GAIVI environment for DiGress, (2) train DiGress on QM9 (the official checkpoint is gone), (3) wrapper with
exactness tests, (4) value function, (5) main comparison. Everything else is off the critical path.
**Three rules.** (a) Every number comes from a run record and a script, from a clean commit. (b) The decision rule is pre-registered
(`docs/PREREGISTRATION.md`, frozen before the first real run). (c) Every module is explained in `docs/qa_notes.md` by its due date.

## 1. Facts that shaped this plan (from the verification pass)
1. The DiGress QM9 checkpoint link is dead (404, issue #109, no mirror). **We train DiGress ourselves** with the repo's unmodified
   `qm9_no_h` config; the paper reports about 1 h to 99% validity (hardware unknown). This is proposal risk (i), now the baseline path.
2. One DiGress forward pass yields `p(x0|x_t)`, from which all M children are sampled. SVDD-MLP costs about T passes, SVDD-PM about
   (M+1)T, best-of-M costs MT. So `N = M` matches **candidates**, not network passes; cost curves are reported.
3. SVDD's paper already compares against best-of-N, but on a continuous model (GDSS, ZINC-250k) and defers DiGress to future work.
   TreeDiff compares them on GraphDiT. Our contribution is the **new domain** (discrete graph diffusion, DiGress/QM9) plus a
   pre-registered, multi-seed, cost-counted comparison and value-function diagnostics (guidelines Sec. 1.4: new domain, new evaluation).
4. DiGress's validity keeps the largest fragment; ours requires a connected molecule. Both are reported.
5. The cluster path is Titan RTX (sm_75, CUDA 11.8 works). The CSAIL 5090 (Blackwell, sm_120) cannot run torch 2.0.1/cu118, so it is **not** a drop-in fallback. The CSAIL 4090 (Ada, sm_89) **should** run it (CUDA 11.8 supports sm_89; not yet tested: confirm with the arch-list check, 2026-11-05) and is free without a queue, which is why Sec. 4a recommends it as the primary training machine for the compressed Slot A schedule. The `vgmg` (Python 3.12, torch >= 2.5) envs built on 2026-11-05 are for our own package only; DiGress needs its pinned stack.
6. DiGress's README (2026-09-10) reports a bug: `dim_ffy` is ignored and 2048 is used, so the paper's hyperparameter is not what the code runs and the effect is unknown. We train `main` as is (the published numbers come from this code) and state it in the report; `fixed_bug` stays the R1 fallback.

## 2. Rubric map
| Item | Pts | How it is secured | Evidence |
|---|---|---|---|
| Live Q&A | 30 | `docs/qa_notes.md` complete by Tue Nov 10 (sections due Sat/Sun/Mon/Tue, Sec. 7); code walk-through drill Thu Nov 12 | notes, rehearsal log |
| Exposition | 15 | derive soft value + explain why M children share a pass | slides |
| Results/evidence | 12 | seeds, std, CI, cost axes, pre-registered rule, labeled preliminary | `results/tables/` |
| Delivery | 8 | 10 min, two timed rehearsals, engage in others' talks | rehearsal log |
| Report | 15 | two-column, 4-6 pp, 12-20 refs, honest limitations | `docs/report/` |
| Code | 15 | clean clone + smoke test, no leakage (tests), commits over weeks, pinned DiGress | CI, git log |
| Proposal | 5 | submitted/emailed by Oct 12 | Canvas / email |

6930 bar: >= 3 seeds (3, plus 5 at the headline), >= 2 ablations (A1, A2), several baselines (B1, B2, B3, random-selection control),
reasons for comparisons not run (pre-registration), 12-20 references (report).

## 3. Scope tiers and what Slot A shows
| Tier | Contents | Due |
|---|---|---|
| T1 (talk) | trained DiGress, B1, B2, SVDD-PM, SVDD-MLP at M=10 for SA and QED, seeds 0-2, verdict labeled preliminary, H1 diagnostics on one seed | frozen Nov 8 |
| T2 (the proposal) | + A1 (M in {1,5,10,20}), A2 (PM, MLP, GNN), seeds 3-4 at M=10, failure analysis, figures | frozen Nov 29 |
| T3 (not promised) | docking proxy, H2 selection-rule study, H4 retrain the value fn on guided rollouts | only if T2 is done by Nov 22 |

The talk has about 8 slides: problem and gap; SVDD mechanism (soft value, importance resampling); DiGress mechanism and why children
share a pass; setup (baselines, cost accounting, metrics, seeds); main table (mean ± std over 3 seeds); why it works or not (value-function
ranking by timestep); limitations; planned ablations.

## 4a. Re-baselined schedule to Slot A (2026-11-05; supersedes the dates in Stages 1-5 and in the gate log)
Today is Thu Nov 5 (night); the talk is Fri Nov 13; slides are due Thu Nov 12 11:59 PM (upload Wed Nov 11 night).
| Day | Work (the stage numbers are the stages below) | Gate |
|---|---|---|
| Thu Nov 5 | choose the DiGress training machine; start the DiGress env (`vgmg-digress`, Python 3.9, torch 2.0.1/cu118; **CSAIL 4090 recommended** (sm_89 runs cu118; confirm with the arch-list check), GAIVI `ClsGenAI` as insurance; the 5090 cannot run it); recreate the local test env | env build started |
| Fri Nov 6 | `MODE=check`, `MODE=timing`, **launch `MODE=train` by 18:00**; wrapper v1 (`init/propose/take/decode`) + histogram featurizer; optional office-hours check-in (ask: slot change allowed? feedback) | **D1 Fri 18:00: training running.** If not: tell the professor Sat morning (R17) |
| Sat Nov 7 | `predict_x0`, `encode`; exactness tests (a)-(c) on early checkpoints; demo skeleton on the toy | wrapper tests green on an early checkpoint |
| Sun Nov 8 | **freeze DiGress at 12:00** whatever validity is reached (<= 10 GPU-h total); DiGress-metric validity; B1 seeds 0-2; B2 pool; SA rollouts start | **D2 Sun 12:00 (replaces G2/G3 go/no-go):** tests (a)-(c) pass and validity >= 90 % (target >= 97 %). Else NO-GO: ask for Slot C and keep the same order of work (Sec. 8) |
| Mon Nov 9 | SA value function, S2 t=0 check, S3; SVDD-MLP at M=10 seeds 0-2 (SA); SVDD-PM; random-selection control; QED rollouts | S2 Spearman >= 0.5 (an alarm, not a result) |
| Tue Nov 10 | QED runs; H1 `sibling_ranking` on seed 0; `make results`; **freeze 18:00**, tag `talk-snapshot`; demo cache from the snapshot | **D3 Tue 18:00 (replaces G4, was Nov 8): results frozen** |
| Wed Nov 11 | slides from `talk-snapshot`; two timed rehearsals; Q&A drill; **slides uploaded Wed night** | G5a |
| Thu Nov 12 | code walk-through drill; demo dry run (live / replay / toy); no new experiments | |
| Fri Nov 13 | present (Slot A) | G5 |
**Slot A scope tiers under this schedule (all inside the proposal):** floor = the proposal's single-GPU fallback scale (N = 1000, B1 + B2 + SVDD-MLP at M = 10, SA, seed 0); target = T1 of Sec. 3 (N = 2000, seeds 0-2, SA, B1, B2, SVDD-MLP, SVDD-PM, random control; the verdict is computed only with >= 3 paired seeds and labeled preliminary); stretch = QED, H1 sibling ranking, partial A1. If the measured s/batch-pass is > 3x the proposal's 0.1 s, apply proposal rule (iii) in this order: seeds 1-2 come after the talk, N to 1000, M grid later.
**Unchanged after the talk:** Nov 14-22 A1/A2/seeds 3-4, Nov 20 GNN go/no-go, Nov 29 results frozen, Dec 10-11 report. **Demo (not in the proposal, no scientific content):** `scripts/demo.py`, three views (denoising movie, one SVDD step with scores, 16-molecule head-to-head B1/B2/SVDD) in modes live / replay / toy; built Sat-Tue from the same code path as the experiments; replay mode is the default on stage.

## 4b. Progress against Slot A (2026-11-07, Sat): experiments ahead of schedule
The schedule of Sec. 4a assumed the training would end Nov 8 and the runs Nov 9-10. In fact DiGress finished on Nov 6 (both workstations), the wrapper tests passed on the frozen model on Nov 6 (4090) and Nov 6 night (5090), and **all Slot A runs finished on Nov 7 01:12** (GAIVI 578860 + 4090 + a 5090 replicate). Gates D1, D2 and G3 are met; D3 (freeze Tue Nov 10 18:00) can be met early.
| T1 item (Sec. 3) | Status 2026-11-07 |
|---|---|
| trained DiGress, wrapper tests (a)-(c) on the frozen model | done (4090; 5090 as a check) |
| B1, B2 (N = 10), SA and QED | done, 3 seeds |
| SVDD-MLP argmax at M = 10, SA, seeds 0-2 | done: pre-registered verdict **no win** (validity collapse 0.015) |
| SVDD-PM, random control, SA, 3 seeds | done: PM SA 0.862 (uniqueness 0.54), control = B1 |
| exploratory softmax arm (alpha 0.1 / 0.03 / 0.01) | done: alpha 0.1 SA 0.669 at validity 0.987 |
| H1 `sibling_ranking`, t=0 check | done on all 3 value functions: weak (0.04-0.13 early) |
| `make results`, `make_figures.py` | run on the real data (softmax arm not drawn yet) |
| SVDD-MLP / PM / control under QED; B2 wall-clock-matched N | **not run** (stretch tier; held back by the "no new experiments" decision of 2026-11-06) |
Remaining for Friday (no experiments needed): `talk-snapshot` tag (by Nov 10 18:00), qa_notes (all items still open), slides (upload Wed Nov 11 night), figure fixes, demo cache from the real model, rehearsals, code walk-through drill. Everything after the talk (A1, GNN, seeds 3-4, QED for all arms, H4) is Stage 6 and unchanged. Details and the validation of the numbers: `Abel_progress.md` Sec. 8.9-8.13 (private), `docs/PROBLEMS_AND_FIXES.md`.

## 4. Chronological execution plan
(Dates in the stage headings below are the **original** dates ("was"); the live dates are in Sec. 4a.)
Each stage has tasks, an exit milestone with objective criteria, the evidence artifact, and the action if it fails.
"Approval" = a GAIVI submission that needs your explicit go-ahead (the standing job limits stays off); agents never commit.

### Stage 0. Oct 9-12 (Fri-Mon): lock the proposal and the plan  **[DONE 2026-10-17]**
_Closed: the revised proposal (`Abel_proposal.pdf`) was submitted and approved, so no email was needed (item 1); prereg frozen and all docs committed (item 2, last commit `7d41052`); item 3 reading is the user's own and continues with Stage 1 notes; item 4 (Slot A sign-up) is **deferred**: the class has not posted the sheet. Decision (user, 2026-10-17): finish the plan as early as possible and pick the slot when the sheet opens, so gate dates below are targets, not deadlines, until a slot is chosen._
1. Decide the proposal action (`docs/PROPOSAL_ALIGNMENT.md` has the decision tree and the email text): if feedback said "revise", resubmit the new PDF by Oct 12 11:59 PM; if "approved", email the revised PDF as a documented technical blocker (dead checkpoint) and ask whether a slot can be changed after sign-up.
2. Freeze `docs/PREREGISTRATION.md` (commit). Commit the new docs.
3. Read: SVDD Sec. 4 + App. C; DiGress Sec. 4-5 and `sample_p_zs_given_zt`; start `docs/qa_notes.md` (SVDD theory).
4. Oct 13 (sheet opens): sign up for **Slot A** the same day.
- **M0 (Mon Oct 12, 11:59 PM):** proposal resubmitted or emailed; prereg frozen; Slot A claimed on Oct 13. _Result: met except the slot claim (sheet not posted)._ _If the professor will not allow the revision:_ keep the approved text and use this plan unchanged (it already treats the dead checkpoint as the baseline path, which is the original proposal's risk (i)); say so in the report.

### Stage 1. Oct 12-18: environment, launch DiGress training, timing (gate G1)
**Status 2026-10-17: the non-GPU part of Stage 1 is closed and reviewed (all of it committed); the GPU/GAIVI part is deliberately deferred.** The user decided to leave every GPU/GAIVI step until they say the access is fixed and they come back; nothing below that touches GAIVI starts before then, and Stage 2 is not started either. Resume list: `Abel_progress.md` Sec. 3c. Ownership: agents cannot create environments or run python on the login node (guard rules), so items 2-3 are user-run steps with scripts prepared.
**Status 2026-11-06: compute access is restored and verified.** GAIVI, the CSAIL 5090 and the CSAIL 4090 are all reachable (VPN on). A conda env `vgmg` (Python 3.12, for our own package only, **not** the DiGress env) is built and verified on all three: GAIVI `/data/anquyle/conda_envs/vgmg` (torch 2.5.1+cu121), 5090 and 4090 (torch 2.11.0+cu128). A 3000-step smoke job ran to completion on GAIVI (A100 on `Quick`, TITAN RTX on `ClsGenAI`), the 5090 and the 4090 (all `ok`, peak VRAM about 1 GB, under 2 min each). GAIVI project folder: `/data/anquyle/projects/value-guided-molecular-generation`; workstation copies live under their `projects/` folders. The DiGress env `vgmg-digress` (item 2) is still **not** built and DiGress training has not started, so G1 stays open. The 5090 (sm_120) cannot run torch 2.0.1, so DiGress trains on GAIVI or the 4090, decided by the timing run.
**Update 2026-11-06 (night): the DiGress env `vgmg-digress` is built and verified on all three machines** (Python 3.9, DiGress's `requirements.txt`, rdkit 2023.3.2 from pip, no graph-tool: the empty stub in `scripts/stubs` is on `PYTHONPATH`, which QM9 never touches; `third_party/DiGress` is also put on `PYTHONPATH` instead of `pip install -e`). Per machine: GAIVI `/data/anquyle/conda_envs/vgmg-digress` and the 4090 `.../aicriminology/envs/vgmg-digress` use the pinned **torch 2.0.1+cu118** (the cu118 wheel has no sm_89 kernel but runs on the 4090 through its sm_80 code); the 5090 `.../abel_team/envs/vgmg-digress` uses **torch 2.7.1+cu128** (the only way to run on sm_120) with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch >= 2.6 refuses DiGress's own cached PyG file otherwise). DiGress's own `debug` config (one step, validation, 4-molecule sampling) ran to exit 0 on a real GPU on each: GAIVI TITAN RTX (ClsGenAI, job 576626), 4090, 5090 (`job_scripts/digress_run.sbatch` MODE=check on GAIVI, `job_scripts/digress_ws.sh check` on the workstations). Not yet done: the timing run (MODE=timing), the training launch, and the choice of training machine. Any training or evaluation on the 5090 must be reported as torch 2.7.1, not the tested torch 2.0.1 stack, and the report must say so.
1. `check_connection`, storage report (VPN on). **Done 2026-11-06:** GAIVI connection OK, `/data` 63.5 % used.
2. DiGress pinned as submodule `third_party/DiGress` @ bdfa10c (**done**). Environment: the **user** runs `bash scripts/setup_digress_env.sh` in a login-node shell (creates `/data/anquyle/conda_envs/vgmg-digress`: Python 3.9, rdkit 2023.03.2, graph-tool 2.45, torch 2.0.1+cu118, torch_geometric 2.3.1 via DiGress's requirements, `pip install -e`). orca is **not** compiled (only SBM/planar metrics use it; QM9 imports neither it nor `spectre_utils`); graph-tool is needed only for DiGress's `import graph_tool` on `main.py` line 1, fallback `PYTHONPATH=scripts/stubs` (empty module, documented). The login node cannot import torch, so the import check runs as a job (`MODE=check` below).
3. QM9: DiGress's loader cannot run on the login node (it imports torch), and if only the raw files exist it skips `download()` and then fails on the missing split CSVs (checked in the source). So `python scripts/stage_digress_qm9.py` (**done and verified locally**: raw files from the deepchem zip, sha256 pinned, plus `uncharacterized.txt` from figshare which needs a User-Agent header; splits 100000/20497/13388, identical to DiGress's own `np.split(..., random_state=42)` run under pandas 1.4) writes `data/digress_qm9/qm9_pyg/raw`; sync it to GAIVI; `process()` then runs inside the first compute job. The PyG fallback URL in DiGress's loader (`qm9_v3.zip`) holds a processed `.pt`, not raw files, so it is not a substitute.
4. Job script `job_scripts/digress_run.sbatch` (written, **not submitted**), three modes: `check` (imports, GPU, DiGress `debug` run), `timing` (3 epochs with one validation and a 512-sample generation: s/epoch, VRAM, s per sampling pass) and `train`. **Approval:** launch `train` on `ClsGenAI`: `qm9_no_h`, `wandb=disabled`, `gpus=1`, `seed=0`, `train.n_epochs` = what fits 10 GPU-h from the timing run (the learning rate is constant, so a shorter run is just earlier stopping), resumable. Resume mechanics read from `main.py`: the first job writes `checkpoints/qm9_no_h`, resumed jobs write `checkpoints/qm9_no_h_resume`, the resume path must be absolute, and checkpoints land only at validation epochs (every 5), so a stop loses <= 5 epochs (confirm at timing). Chained jobs: a job that hits its Slurm limit ends TIMEOUT and `afterok` dependents would never start, so the script stops itself after `BUDGET_S=27000` s (7h30m) via `timeout` and exits 0 (a `DONE` file marks completion). `timeout` returns 137 instead of 124 when it has to KILL (Lightning only flags SIGTERM and keeps training), so the script judges the stop by elapsed time, not by the exit code. Run `mkdir -p logs` before `sbatch` (Slurm does not create the `--output` directory). `MODE=check` runs DiGress's debug config with `general.gpus=1` so the GPU path is exercised. The 10 GPU-h budget is two jobs at most. The final `trainer.test` generates 10000 samples inside the last job; keep it, it gives DiGress's own metric at no extra cost.
5. Python 3.9 compatibility **done locally**: conda env with Python 3.9, numpy 1.23.5, pandas 1.4.4, torch 2.0.1 (CPU), rdkit 2023.03.2: 32 tests and `make smoke-test` pass. The GAIVI re-run (inside the real env) remains. There is no GitHub remote, so the CI `compat-digress-env` job has not run; do not claim it.
6. Measure unguided sampling s/batch-pass with an early checkpoint (the `timing` mode gives a first value); re-run `python scripts/estimate_gpu_hours.py <s>`.
7. Env freeze: `setup_digress_env.sh` writes `environment-gaivi.yml` and `environment-gaivi.pip.txt`; commit them after review. The training command is the job script.
- **G1 (target Oct 18, floats; open until the GPU part is done) passes if:** training is running or done with ETA <= 10 GPU-h; env documented; smoke test passes in it; s/batch-pass measured and the GPU-hour table re-run (total within about 50 GPU-h, or proposal rule (iii) applied).
- **If it fails:** Oct 19-21 emergency (graph-tool stub, ask GAIVI admins, try the `general` partition); if training has not started by Oct 21, pre-warn the professor about Slot C.
- **Understanding:** qa_notes "DiGress" section drafted.

### Stage 2. Oct 19-25: wrapper with exactness tests, B1 (gate G2)
1. `DiGressDenoiser`: `init` (node counts from DiGress's distribution), `propose` (adapted from `sample_p_zs_given_zt`: one forward, M samples from the same probabilities), `take`, `features` (atom/bond histograms + node count), `predict_x0`, `decode` (DiGress `build_molecule`), `encode`. Develop against intermediate checkpoints. Mark GPU tests `@pytest.mark.digress` (skipped without the env).
2. Exactness tests: (a) wrapper posterior probabilities equal DiGress's internal `prob_X/prob_E` on the same noisy input (tolerance 1e-6); (b) M=1 unguided samples match the official sampler's validity and uniqueness within 1.5 points on 2000 molecules; (c) our NFE counter equals the number of network forward calls counted with a hook.
3. Evaluate the trained model with DiGress's own metric (expect about 99.0 valid, 96.2 unique) and ours; add the `validity_digress` (largest-fragment) metric to `metrics.py`; record the checkpoint SHA-256 in `summary.json`.
4. B1 for seeds 0-2 (3-4 are cheap). Measure and record GPU nondeterminism by rerunning one run.
5. qa_notes: SVDD theory and DiGress sections **complete**.
- **G2 (Sun Oct 25) passes if:** DiGress-metric validity >= 97% and uniqueness >= 90%; tests (a)-(c) pass; B1 table with 3 seeds committed.
- **If it fails (R1):** train longer; try the `fixed_bug` branch (<= 10 GPU-h total; if used, pin its commit and update `CONTRIBUTIONS.md` and the report: "unmodified main" would no longer be true); if validity stays < 90% freeze the best model and report B1's lower validity honestly (the 5-point rule is relative to B1); in parallel ask the author of issue #59 for the default-config checkpoint (an untrusted pickle: inspect and load in a sandbox).

### Stage 3. Oct 26-30: value function, PM, B2, fallback run (gate G3 = Slot A go/no-go)
1. **Approval:** `python scripts/train_value_function.py` for SA and QED, seed 0 (10k train + 1k val chains, about 0.45 h rollouts + fit). Check: validation MSE below the constant predictor; Spearman by timestep bin; **S2 t=0 check** on real molecules from the committed split (Spearman >= 0.5, an alarm and not a result); S3 (M=1 equals unguided).
2. SVDD-PM end to end; B2 pool; `make fallback` (< 2 h) on a GPU.
3. First preliminary look at the effect (fallback numbers). Sketch the talk outline.
4. qa_notes: experimental design section complete.
- **G3 (Fri Oct 30) passes (decision: GO for Slot A) if:** fallback run finished with `summary.json`; S2 passes; B1, B2 and PM numbers exist for SA seed 0; no unresolved evaluation bug.
- **If it fails:** record NO-GO and ask for Slot C (the Oct 12 email already asked whether a switch is possible); keep the same order of work with the Plan C dates in Sec. 8.

### Stage 4. Oct 31-Nov 8: main comparison (gate G4, results frozen Nov 8)
Priority order, each with the exact command (no new files needed: extra `key=value` arguments override every grid point):
1. B1: `python scripts/run_ablations.py configs/sweeps/b1_unguided.yaml`
2. SVDD-MLP at M=10, SA+QED, seeds 0-2: `python scripts/run_ablations.py configs/sweeps/a1_m_sweep.yaml svdd.num_candidates_m=10`
3. B2 (N=10, plus smaller N for a wall-clock-matched point chosen from the measured timing): `python scripts/run_ablations.py configs/sweeps/b2_best_of_n.yaml "best_of_n.m_values=[2,5,10]"`
4. SVDD-PM and random-selection control at M=10: `python scripts/run_ablations.py configs/sweeps/a2_estimator.yaml`
5. As time permits: A1 other M, SA first (`a1_m_sweep.yaml` without the override).
6. `make results` (table + verdict on seeds 0-2, labeled preliminary) and `python scripts/make_figures.py`.
7. H1 diagnostics on seed 0 (`sibling_ranking` at three timesteps) so a mechanism is ready whatever the sign.
8. Optional Nov 6 office-hours check-in with G1-G3 numbers. Tag the commit `talk-snapshot`; slide numbers come only from it.
- **G4 (Sun Nov 8) passes if:** B1, B2, PM and SVDD-MLP at M=10 are complete for at least SA (QED if finished), seeds 0-2, verdict computed, H1 on one seed.
- **If it fails:** present what is complete, labeled preliminary (Slot A explicitly allows it); do not change the protocol.
- **Understanding:** qa_notes counterfactuals section complete by Nov 10.

### Stage 5. Nov 9-13: talk (gate G5)
Mon-Tue slides from `talk-snapshot`; Wed two timed rehearsals + Q&A drill, **slides uploaded Wed night** (due Thu Nov 12, 11:59 PM; late costs 2 pts); Thu code walk-through drill (open every file, explain it); **Fri Nov 13 present**. Attend and ask questions in classmates' talks (Delivery criterion includes audience engagement; Nov 13, and Nov 20 / Dec 4 when possible). Afterwards log every question asked in `docs/talk_feedback.md`.
- **G5:** talk delivered; questions logged.

### Stage 6. Nov 14-29: complete the proposal's experiments (gates G6, G7)
1. Nov 14-15: switch the report to the two-column CVPR/NeurIPS template, fill the skeleton, build the 12-20 reference bibliography (re-check every DOI).
2. **Approvals:** finish A1 (all M, both rewards), seeds 3-4 at M=10 (`extra_seeds_*.yaml`), A2 (PM, MLP; GNN arm if kept).
3. Nov 14-19: implement the GNN value function; **Nov 20 go/no-go** by the proposal's rule (keep only if it beats the MLP on validation MSE); docking proxy dropped unless already started.
4. Address talk feedback. H1-H3 analyses and failure analysis (where the value function mis-ranks siblings, reward headroom, diversity).
- **G6 (Sun Nov 22) passes if:** all promised runs have `summary.json`; first full table. After G6 only experiments that explain results (T3 only if G6 is met).
- **G7 (Sun Nov 29):** results frozen, tag `results-final`, `make results` and figures regenerated.

### Stage 7. Nov 30-Dec 10: report and repository (gate G8)
Report from frozen numbers (every number generated by script); repo hygiene (README one-command headline, `CONTRIBUTIONS.md` final with pinned commits, env files, DiGress training script and config, checkpoint hosted outside git with a link and SHA-256); fresh-clone check on CPU; self-audit of the evaluation code (same `compute_metrics` for all methods, no leakage, no tuning on test); AI declaration written from what was actually done (guidelines Sec. 4.5); submit **Dec 10** (deadline Dec 11 11:59 PM).
- **G8:** report PDF and repo link on Canvas.

## 5. Critical path and dependencies
`env (G1)` -> `DiGress training (G1-G2)` -> `wrapper + tests (G2)` -> `value fn (G3)` -> `main comparison (G4)` -> `talk (G5)`.
Off the path (do in waiting time): qa_notes, talk outline, report template, bibliography, GNN code skeleton, figure scripts.
Training and wrapper overlap in Stage 2: the wrapper is developed against intermediate checkpoints.

## 6. Compute (proposal formula, measured s/pass at G1)
`python scripts/estimate_gpu_hours.py <s>`: at the proposal's 0.1 s and 1.5x: 30.7 GPU-h (38 with margin) plus DiGress training <= 10 = about 48, under the proposal's 50.
Slot A needs only about 15 GPU-h of experiments (B1 0.2, SVDD-MLP at M=10 incl. value functions 6.2, B2 pool of 10 about 2.4, PM 5.4, random control 0.5) between Sun Nov 8 12:00 and Tue Nov 10 18:00 (about 54 h of wall-clock), runnable as 2 parallel 1-GPU jobs (4090 + GAIVI), plus DiGress training <= 10 GPU-h before the freeze. These are the proposal's own estimates (0.1 s per batch-pass assumed): re-run `estimate_gpu_hours.py` with the s/pass from `MODE=timing` on Fri Nov 6; if the total no longer fits, apply the Sec. 4a scope tiers (floor = the proposal's fallback scale).
Limits (the standing job limits): <= 8 h per job, <= 20 GPU-h per task, <= 12 queued jobs; log every job in the GPU job ledger.
Shrink order: seeds 3-4, then N=1000 and M in {1,10,20} (proposal rule iii).

## 7. Understanding plan (qa_notes sections and due dates)
| Section | Due |
|---|---|
| SVDD theory (soft value, tilted kernel, importance resampling, argmax limit, MC vs soft value) | Sun Nov 8 (was Oct 25) |
| DiGress (forward process, denoiser outputs, posterior sampling, why one pass serves M children, loss, noise schedule) | Sat Nov 7 (was Oct 25) |
| Experimental design (baselines, pass counting, seeds, rule, leakage controls, splits) | Mon Nov 9 (was Oct 30) |
| Counterfactuals and limits | Tue Nov 10 |
| Code walk-through of every file | Thu Nov 12 |

## 8. Plan C (if D2 fails on Sun Nov 8; was: G3 on Oct 30): Slot C, Fri Dec 4
Same stages, same gates, dates shifted: main comparison frozen Nov 20 (G4'), slides Dec 3, talk Dec 4, T2 frozen Nov 29 as before, report Dec 10. Nothing in the proposal changes except the slot. Whether a slot can be changed after sign-up is **unknown** (ask at the Nov 6 check-in or by email on Fri Nov 6); if it is not allowed, D2 becomes "present what exists, labeled preliminary, with the pipeline and B1 only" instead of a move.

## 9. Risk register
| ID | Risk | Trigger / date | Response |
|---|---|---|---|
| R1 | DiGress training slow or below target validity | ETA > 10 GPU-h at G1; < 97% at G2 | fewer epochs with periodic sampling, `fixed_bug` branch, accept >= 90% and report, parallel ask for a checkpoint |
| R2 | environment fails (graph-tool, torch 2.0.1/cu118, PyG) | not built or `MODE=check` fails (was Oct 16, floats with G1) | graph_tool stub, GAIVI admins, `general` partition; 5090 only as a last resort (needs newer torch) |
| R3 | queue delays on `ClsGenAI` | job pending > 24 h | submit early, parallel jobs, resumable runs, `general`/`Quick` with a stated reason |
| R4 | wrapper bug (masks, extra features, probabilities) | exactness test fails | fix before any experiment; no results from an unverified wrapper |
| R5 | value fn no better than the constant predictor | G3 diagnostics | check features, S2, more rollouts; PM vs MLP still reportable |
| R6 | SVDD does not beat B2 | G4 | negative-result path: H1 (ranking), H3 (headroom); diagnostics already prepared on seed 0 |
| R7 | reward hacking / collapse to tiny molecules | reward above real QM9 top-10% or diversity collapse | inspect samples; report diversity and novelty next to reward |
| R8 | PM ties at early steps (x̂0 invalid -> reward 0) | PM about equals random selection | random tie-break; report tie fraction |
| R9 | evaluation bug or leakage | any time; final audit Dec 7 | tests, single `compute_metrics`, separate RNG streams, audit checklist |
| R10 | Slot A time pressure (8 days from 2026-11-05) | D1 Fri 18:00, D2 Sun 12:00, D3 Tue 18:00 (Sec. 4a) | NO-GO at D2 to Plan C; otherwise present what is frozen at D3, labeled preliminary; never extend the freeze |
| R18 | DiGress env on the 4090 fails (torch 2.0.1/cu118 on sm_89, graph-tool, PyG 2.3.1) | `MODE=check` fails Fri | graph_tool stub, GAIVI env as the second path; the `vgmg` (Python 3.12) analysis env is **not** a substitute: DiGress needs the pinned stack |
| R19 | wrapper bug under time pressure | exactness test (a), (b) or (c) fails Sat/Sun | no result from an unverified wrapper (R4); if unfixed at D2, NO-GO |
| R20 | demo fails on stage | any | replay mode from `talk-snapshot`; toy mode labeled "pipeline only"; a failed demo is not penalized if explained (guidelines 4.1) |
| R11 | ~~proposal revision not accepted~~ **closed 2026-10-17** (revised proposal approved) | n/a | none |
| R17 | GAIVI access not restored, so no GPU step can start (G1 open). **Resolved 2026-11-06: access works, see Stage 1 status** | access still broken when the user returns | user fixes it first; every day of delay moves G1-G3, so tell the professor early if training has not started within ~4 days of access (Slot C is the escape hatch); the CSAIL 5090 cannot run torch 2.0.1/cu118 |
| R12 | provenance holes (dirty commit, mixed GPUs) | `make results` warnings | run from a clean commit; compare wall-clock only on one GPU type |
| R13 | checkpoint > 100 MB cannot be committed | Stage 7 | host outside git, link + SHA-256, retrain script |
| R14 | one DiGress training seed only | report | state as a limitation: conclusions are conditional on one trained model |
| R15 | GPU nondeterminism | G1 rerun | report the measured tolerance |
| R16 | QM9 has duplicate SMILES (133,802 distinct of 133,885) | novelty | sets of canonical SMILES; documented |

## 10. Definition of done (Dec 10)
- [ ] Fresh clone, README only, `make smoke-test` passes (< 10 min); one command regenerates the headline table from run records.
- [ ] `CONTRIBUTIONS.md` accurate (DiGress code unmodified and pinned, weights trained by me with the stated config and seed, wrapper adapted from `sample_p_zs_given_zt`).
- [ ] Every report number traceable to a `summary.json` (GPU, commit, checkpoint hash recorded).
- [ ] Limitations name real failures and the next experiment.
- [ ] `docs/qa_notes.md` complete; every file opened and explained.

## 11. Critique log
Rounds 1-2: 1. Compute match ill-posed (children share a pass) -> pass counting, cost curves, wording fix in the proposal. 2. Value fn lacked the timestep -> added. 3. Invalid molecules excluded from the primary reward -> all N, invalid = 0, connected. 4. Row-level split leaks -> chain-level, separate RNG streams. 5. Rule underspecified -> precise, implemented. 6. Featurizer belongs to the denoiser. 7. Environment mismatch -> Python 3.9-compatible code, env stage. 8. Broken sweeps -> resumable. 9. "Write it yourself" rule -> explain-it rule. 10. Plan drifted from the proposal (GNN, fallback, resume, seeds) -> alignment map, fixed. 11. Stale cache and missing provenance -> hash, commit, GPU recorded.
**Round 3 (verification):** 12. Official checkpoint gone -> train DiGress (new critical path, G1-G2 gates). 13. Novelty claim overstated (SVDD paper has best-of-N; TreeDiff exists) -> gap rewritten around the new domain. 14. Proposal misdescribed SVDD's molecule setup (MOSES) -> corrected. 15. DiGress validity convention differs -> report both. 16. CSAIL is not a fallback for DiGress (sm_120). 17. Slot A compresses the schedule -> main comparison first, GNN after the talk, Oct 30 go/no-go. 18. One DiGress seed -> limitation. 19. Checkpoint > 100 MB and retraining must be reproducible -> hosting + script.

## Gate log
| Gate | Date | Status | Notes |
|---|---|---|---|
| M0 proposal/email, prereg frozen, Slot A signed | Oct 12 | **done 2026-10-17** (slot claim deferred) | revised proposal approved; prereg frozen; sheet not posted yet |
| G1 env + training launched + timing | was Oct 18; **now D1 Fri Nov 6 18:00** | **done 2026-11-06** | `vgmg-digress` env built on all 3 machines; DiGress trained on the 4090 (6.4 h) and the 5090 (4.7 h), validity 99.2 / 99.1 % |
| G2 DiGress validity + wrapper tests + B1 | was Oct 25; **now D2 Sun Nov 8 12:00 (also the Slot A go/no-go)** | **done 2026-11-06 (GO)** | frozen = 4090 checkpoint (SHA-256 `ef376a81...`); tests (a)-(c) ALL PASS on the 4090 and the 5090; B1 done (validity 0.992, SA 0.635) |
| G3 value fn + fallback run | was Oct 30; **now Mon Nov 9** | **done 2026-11-07** | 3 value fns trained (val MSE 9-11 % below constant, Spearman 0.34-0.39); full Slot A run instead of the fallback |
| G4 main comparison frozen | was Nov 8; **now D3 Tue Nov 10 18:00** | runs **done 2026-11-07**; freeze + tag pending | SA complete (B1, B2, SVDD-MLP, SVDD-PM, control, softmax arm); QED only B1/B2 |
| G5 talk | Nov 13 (unchanged) | pending | slides uploaded Wed Nov 11 night |
| G6 all promised runs | Nov 22 | pending | |
| G7 results frozen | Nov 29 | pending | |
| G8 submission | Dec 10 | pending | |
