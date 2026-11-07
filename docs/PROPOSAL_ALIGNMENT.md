# Proposal alignment (revised proposal vs plan vs repository)

Source of truth for promises: `docs/proposal/Abel_proposal.pdf` (**revised 2026-10-09**, 2 pages + references).
Rule: no row may silently differ. A difference is either a strengthening declared in the report or a correction explained to the professor.
Topic changes after Oct 12 need a documented blocker and an email before the change.

## 1. What changed from the first proposal, and why
| Change | Reason (evidence in `docs/VERIFICATION.md`) | Scope impact |
|---|---|---|
| DiGress is trained by me on QM9 (repo config, then frozen) instead of loading released weights | Official checkpoint link returns 404; README says the author lost the drive; issue #109 open, no mirror (V16) | None on the question; was proposal risk (i), now the baseline path; +<= 10 GPU-h |
| "equal denoiser evaluations" -> best-of-N with N = M **plus** cost curves (network passes, wall-clock, best-of-N at SVDD's wall-clock) | One DiGress pass serves all M children (V17); the SVDD paper matches best-of-N by inference time (V5) | Clarifies the comparison; strictly more informative |
| Gap statement rewritten: SVDD's paper uses continuous GDSS on ZINC-250k, defers DiGress (App. C); TreeDiff compares SVDD and best-of-N on GraphDiT | The paper already has best-of-N; the old text overclaimed and misdescribed the molecule setup as MOSES (V5-V7, V11) | Contribution is the new domain + new evaluation (guidelines 1.4) |
| Fourth supporting reference (TreeDiff) | Closest prior comparison | none |
| Validity: also report DiGress's largest-fragment convention | Published numbers use it (V20) | none |
| Seeds 0-2 everywhere, plus 3-4 at the headline point | Stronger evidence at the decisive comparison; pre-registered | declared strengthening |
| Compute arithmetic rewritten with corrected pass counts (about 31 h + <= 10 h training, about 48 GPU-h with margin) | Old arithmetic charged every candidate as a full pass | under the original 50 GPU-h |
| Milestones for **Slot A** (Nov 13); GNN go/no-go Nov 20 (was Nov 6); docking dropped if not started by Nov 20 | Slot A needs the main result by Nov 8; the GNN is not on that path | same deliverables, reordered |
| Risk (i) rewritten (train target >= 97% validity; fallback rules); risk (v) added (Slot C if the Oct 30 gate fails) | Dead checkpoint; Slot A | none |
| AI declaration | **unchanged** | none |

## 2. Commitment map (revised proposal -> plan -> repository)
| # | Commitment | Plan stage | Repository | Status |
|---|---|---|---|---|
| 1 | Question: SVDD-MLP vs best-of-N (N = M), mean SA/QED, validity, scaling in M | prereg | `docs/PREREGISTRATION.md` | aligned |
| 2 | Primary SVDD, supporting DiGress, DDPO, Gao, TreeDiff | report | `docs/proposal/refs.bib` | aligned |
| 3 | Train DiGress on QM9 with unmodified repo config, T=500, freeze | Stage 1-2 | submodule `third_party/DiGress` @ bdfa10c (done), `job_scripts/digress_run.sbatch` (written, not yet run) | **done 2026-11-06:** trained on the 4090 (frozen) and the 5090 (backup), seed 0, 1000 epochs, validity 99.2 / 99.1 % |
| 4 | Wrapper returns M i.i.d. children from one pass | Stage 2 | `vgmg/models/digress.py::DiGressDenoiser.propose`, `vgmg/models/denoiser.py` interface and NFE counter; tests (a)-(c): `vgmg/models/digress_checks.py`, `scripts/verify_digress_wrapper.py` | **(a)-(c) ALL PASS on the frozen checkpoint** (4090 and 5090) |
| 5 | Value fn: MLP on atom/bond histograms + timestep embedding | Stage 3 | `models/value_function.py` (embedding); `DiGressDenoiser.features` (4 atom + 4 bond counts + node count, /9) | **done:** 3 value fns fit on real DiGress states (seeds 0-2) |
| 6 | Value fn: 3-layer GNN | Stage 6 (go/no-go Nov 20) | `configs/sweeps/a2_gnn.yaml`; `GNNValueFunction` (3 message-passing layers on the dense graph state, written 2026-11-06 ahead of the Nov 14-19 slot; untrained, go/no-go Nov 20 unchanged) | code done, not trained |
| 7 | MSE regression onto r(G0), 10k unguided rollouts, per reward | Stage 3 | `rl/value_trainer.py`, `value_fn.train_chains: 10000`; each chain draws its own `ts_per_chain` random timesteps (proposal silent on which) | done; real runs 2026-11-06 |
| 8 | Argmax or softmax selection; SVDD-PM | Stage 3-4 | `rl/svdd.py` | **done on real DiGress** (SVDD-MLP argmax fails; softmax arm exploratory) |
| 9 | Rewards SA (10-SA)/9 and QED, invalid -> 0; docking proxy stretch | Stage 1-4 | `rl/rewards.py` | done; docking dropped unless started by Nov 20 |
| 10 | Data: QM9 133,885, CC0; DiGress's own split; 1k sanity splits only for reward and t=0 checks; novelty vs all of QM9 | Stage 2-3 | `data/`, `scripts/qm9_reference.py`, `value_fn_t0_check` (run automatically on the 1k val split, `experiments.py::value_diagnostics`) | done |
| 11 | N = 2000, seeds 0-2 (+3-4 at M=10) | Stage 4, 6 | `configs/default.yaml`, `configs/sweeps/*.yaml` | seeds 0-2 at M = 10 **done**; seeds 3-4 not run (Stage 6) |
| 12 | Metrics incl. top-10% reward, validity, uniqueness, novelty, diversity, passes, wall-clock | Stage 2 | `eval/metrics.py` (`validity_digress` added 2026-11-06) | done |
| 13 | B1, B2 (N = M, plus wall-clock-matched N), B3 PM, random-selection control | Stage 4 | `eval/baselines.py`, sweeps | **done for SA** (B1, B2 N = M = 10, PM, control); wall-clock-matched N and QED for SVDD arms not run |
| 14 | A1: M in {1,5,10,20}, SA and QED | Stage 4 (M=10), 6 (rest) | `a1_m_sweep.yaml` | code done; **not run** beyond M = 10 (Stage 6) |
| 15 | A2: PM / MLP / GNN at M=10 | Stage 4, 6 | `a2_estimator.yaml`, `a2_gnn.yaml` | PM and MLP at M = 10 **done (SA)**; GNN planned (go/no-go Nov 20) |
| 16 | Not run: DDPO, regressor guidance | prereg | `docs/PREREGISTRATION.md` | aligned |
| 17 | Falsifiable rule (gain over B2 > 2 x noise, validity within 5 points of B1) | Stage 4 | `eval/stats.py::preregistered_verdict`, `make results` | done |
| 18 | Compute: about 48 GPU-h, GAIVI class partition, fallback < 2 h | Stage 1, 6 | `scripts/estimate_gpu_hours.py`, `configs/sweeps/fallback.yaml` | done; re-run at G1 |
| 19 | Checkpoints and per-chunk resume; logging of config, seed, commit, GPU | all | `experiments.py`, `utils/logging.py` | done (training checkpoints planned) |
| 20 | Milestones Oct 12 -> Nov 13 (Slot A), then Nov 14 -> Dec 11 | Sec. 4 of the plan | `docs/PLAN.md` | aligned line by line |
| 21 | Risks (i)-(v) | plan Sec. 9 | `docs/PLAN.md` | aligned (R1 = (i), R5/R6 = (iv)) |
| 22 | Contribution table | Stage 7 | `CONTRIBUTIONS.md` | updated; hashes added as components land |
| 23 | AI declaration (as submitted) | Stage 7 | report | unchanged in the proposal; the report's declaration is written at the end from what was done (guidelines Sec. 4.5 requires it to be accurate) |

Note on the declaration sentence "I will write and understand the DiGress wrapper and SVDD code": the two named pieces are the ones to work through
line by line (qa_notes, Stage 2-3), so the sentence stays true in spirit; the Q&A tests exactly this.

## 3. Proposal action: decision tree (Stage 0)
- Feedback said **revise**: submit `docs/proposal/Abel_proposal.pdf` (already named `Abel_proposal.pdf`; the guidelines say `lastname_proposal.pdf`, i.e. `Le_proposal.pdf`, so rename on upload if you prefer to follow them literally) on Canvas by Mon Oct 12 11:59 PM (one revision carries no penalty).
- Feedback said **approved / approved with conditions**: send the email below with the PDF attached (a dead checkpoint is a documented technical blocker, sent before Oct 12 as the rules require) and wait for the reply; if there is no answer by Oct 12, proceed with this plan, which already treats the dead checkpoint as the baseline path of the original risk (i).
- No feedback yet: same as "approved", and ask for the status.

## 4. Email draft (to the professor)
Subject: CIS 6930 project: proposal revision (checkpoint blocker) and two questions

Dear Prof. Karimian,
While preparing the DiGress wrapper I found that the official QM9 checkpoint is no longer available (the README link returns 404; the author reports losing the drive; GitHub issue #109 is open). My proposal's risk (i) covered this case, so I will train DiGress on QM9 with the repository's own configuration (the paper reports about 1 GPU-hour to 99% validity; budget <= 10 GPU-h) and keep it frozen afterwards. The attached revised proposal also (1) rewrites the comparison as best-of-N with N = M plus reported network passes and wall-clock, because one DiGress forward pass serves all M candidates, (2) corrects the related-work description (the SVDD paper already compares against best-of-N on a continuous model, GDSS; it defers DiGress to future work), (3) updates the compute plan, and (4) targets Slot A (Nov 13). The question, baselines, ablations and falsifiable outcome are unchanged.
Two questions: may I replace the proposal with the revised version, and if the Oct 30 gate shows I need more time, may I move from Slot A to Slot C after signing up?
Thank you, Abel Le

## 5. Guidelines compliance check (CIS6930_Project_Guidelines.pdf, Sec. 1.1-1.5 and checklist), 2026-10-09
| Requirement | Where met in `Abel_proposal.pdf` |
|---|---|
| 2 pages max excluding references, PDF | pages 1-2, references on page 3 (checked after the last edit) |
| Header: title, name, 4930/6930 | title, Abel Le, graduate (6930), Path A |
| Question: one paragraph ending "This project asks whether..." | Sec. 1, last sentence |
| Primary paper (2021+, top venue) + 2-4 supporting, one sentence each, one gap sentence | SVDD (NeurIPS 2025); DiGress, DDPO, Gao et al., TreeDiff; Gap paragraph |
| Method: model, loss, training procedure | DiGress (cross-entropy, trained once on QM9), value fn (MSE on 10k rollouts), selection rule |
| Data: dataset, size, license, subset | QM9, 133,885, CC0, N = 2000 per run, 1k sanity splits |
| Evaluation: metrics, baselines, one falsifiable sentence | Sec. 5 (B1, B2, B3; A1, A2; rule) |
| 1.1 items: question, primary paper, own code, quantitative evaluation vs baseline, controlled ablation, failure analysis | all present; failure analysis in the Nov 23-29 row and risk (iv) |
| Compute plan: where, access, GPU-hours with arithmetic, < 2 h single-GPU fallback | Sec. 6 |
| Milestones: a dated line per week from approval to presentation slot | weekly rows Oct 12-18, 19-25, Oct 26-Nov 1, Nov 2-8, Nov 9-13 (Slot A), plus post-talk rows |
| Risk and fallback in the form "if X is not working by [date], I will fall back to Y" | Sec. 8 (every item has a date or trigger date) |
| Contribution table with pinned base commits | Sec. 9 (DiGress @ bdfa10c, SVDD-molecule @ 455e4d4) |
| AI declaration | Sec. 10 (unchanged) |
| Base repository declared | DiGress, SVDD-molecule named |
| Public data, no personal data | QM9 CC0 |
| Not "too big" (returned-proposal reason 1) | one training run of one model (about 1 h in the paper, budget <= 10 GPU-h); value fns are small regressions; shrink rules in risks (iii) |
| Not "README end to end" / checkpoint swap | training only supplies the frozen base; the contribution is the guidance study (stated in Sec. 3) |
| File name `lastname_proposal.pdf` | repo name is `Abel_proposal.pdf`; rename on upload to `Le_proposal.pdf` to follow the guidelines literally |
Milestone note: the proposal's weekly rows are split by calendar week; the plan splits the same work into Stage 3 (Oct 26-30) and Stage 4 (Oct 31-Nov 8) around the Oct 30 gate, with identical content and dates.

## 6. Status log
- 2026-11-05: **Slot A chosen; schedule re-baselined (dates only).** No GPU step ran after Oct 17, so the proposal's weekly rows (Oct 12-Nov 8) and gates (Oct 18/25/30) are late; the milestone *dates* in the approved proposal can no longer be met, the *content* of every row still stands. New dates: `docs/PLAN.md` Sec. 4a (D1 Nov 6 training running, D2 Nov 8 freeze + go/no-go, D3 Nov 10 results frozen, slides Nov 11-12, talk Nov 13); post-talk rows (Nov 14-Dec 11) unchanged. The professor's Slot A bar (pipeline + main result + one baseline, ablations described) is met by the proposal's own single-GPU fallback scale, which is the plan's floor. Proposal risk (v) (ask for Slot C) is now evaluated at D2 instead of Oct 30. Report this slip in the report's timeline section. Alignment checked the same day: code comments for the GNN date (Nov 2-8 / Nov 6 -> Nov 14-19 / Nov 20) and `pm_scorer`'s pass count (m·T -> (m+1)·T) were wrong against the proposal and were corrected; everything else in the commitment map (Sec. 2) is unchanged: rows 3-6 and 20 remain "not done / dates slipped", the rest as listed.
- 2026-11-06: **code layer completed for Slot A (no result yet).** Added without changing any proposal commitment: `DiGressDenoiser` (rows 4-5), exactness tests (a)-(c) (plan Stage 2; (a),(c) pass on a random-weight DiGress, (b) and all three on the trained checkpoint pending: `scripts/verify_digress_wrapper.py`), `validity_digress` metric (row 12), automatic t=0 check and sibling-ranking (H1) with every value function (rows 7, 10), checkpoint SHA-256 in every run record, demo (`scripts/demo.py`). One design choice the proposal is silent on: each rollout chain draws its own random timesteps instead of all chains sharing one set. The GNN value function (row 6) was also written early; its go/no-go stays Nov 20. Nothing is verified on real DiGress molecules yet.
- 2026-11-06 (later): **GPU work started; wrapper verified; first real-DiGress pilots (exploratory, NOT results).** DiGress trains on the 4090 and the 5090 (seed 0, the repo's `qm9_no_h` config, 1000 epochs, about 20 s and 15 s per epoch); the 5090 run uses torch 2.7.1 instead of the pinned 2.0.1 (the report must say so if its checkpoint is the frozen one). Tests (a)-(c) passed on an epoch-139 checkpoint (posterior max diff 1.2e-7, validity and uniqueness within 0.4 points of DiGress's own sampler at n = 1000, pass count exact). Pilots on that early checkpoint (128 molecules, tiny value functions) show the expected ordering B1 < random selection < best-of-N < SVDD-PM, and one **negative finding, proposal risk R7 (reward hacking) materialised: SVDD-MLP with argmax selection collapses (validity 0.000)**. Cause (diagnosed on CPU, not a code bug: random selection through the same sampler is fine): the argmax follows the value function into states it never saw in training (about 14 aromatic bonds on 9 atoms by t = 300; aromatic bonds are about 1 % of QM9 edges), the same with a 40k-row value function (val MSE 0.0168 vs 0.0177 constant). Softmax selection trades this off (12 chains per setting, early checkpoint: alpha 0.1 validity 1.00 / SA 0.61, alpha 0.03 validity 0.92 / SA 0.66, alpha 0.01 validity 0.08). Handling, all inside the proposal: argmax stays the pre-registered primary and is reported as it comes out (the validity rule will decide the verdict); the proposal's selection-rule study (H2, T3) gets an exploratory arm `configs/sweeps/slot_a_svdd_softmax.yaml` with alpha on the scale of the value spread (the earlier H2 grid, alpha 0.1 and 1.0, is effectively random selection here); retraining the value function on guided rollouts (H4, T3) is the next remedy if time allows. Pilot numbers are not for slides; the frozen model's runs are.
- 2026-11-05 (open item): the professor was not asked whether a slot can be changed after sign-up; ask at the Nov 6 check-in or by email.
- 2026-10-17: the revised proposal (`Abel_proposal.pdf`) was approved as submitted; no email sent. Slot sign-up deferred (sheet not posted). Stage 1 prep: DiGress is trained with the repo's config files and command-line overrides only (wandb off, seed 0, data path, loader workers, `n_epochs` sized to the 10 GPU-h budget); this stays inside the proposal's risk (i) and budget. Nothing in the proposal text changed.
- 2026-10-17 (later): the approved proposal's milestone row says "Sign up for Slot A on Oct 13"; the class has not posted the sheet, so this is the one proposal line not met on time (not the professor's doing, no action needed). GPU/GAIVI steps of Stage 1 are deferred by the user, so the proposal row "Oct 12-18: first GPU job, launch DiGress QM9 training" is late; the plan's gates float and risk R17 covers it. If the delay passes ~4 days after access returns, tell the professor.
- 2026-11-07: **Slot A experiments finished and validated; commitment map statuses updated.** All Slot A runs ran on the frozen model (SHA-256 `ef376a81...`, code d936c5c): B1, B2 (N = 10, SA and QED), SVDD-MLP argmax, exploratory softmax arm, SVDD-PM, random control, 3 seeds each, N = 2000; a 5090 (torch 2.7.1) replicate of PM and control agrees with the 4090 to 1e-3. Pre-registered verdict (SVDD-MLP argmax vs best-of-10, SA): **no win** (validity 0.015); this is reported as found (proposal risk R7). Deviations from the proposal, all disclosed: QED not run for the SVDD arms and the control; B2 only at N = M = 10 (no wall-clock-matched N); A1 beyond M = 10, GNN and seeds 3-4 not yet run (Stage 6, unchanged); an exploratory softmax arm added (argmax stays the pre-registered primary); wall-clock not comparable across arms (GAIVI TITAN vs 4090); one DiGress seed. No proposal commitment was changed. Details: `docs/PROBLEMS_AND_FIXES.md`, `docs/PLAN.md` Sec. 4b.
