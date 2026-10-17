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
| 3 | Train DiGress on QM9 with unmodified repo config, T=500, freeze | Stage 1-2 | submodule `third_party/DiGress` @ bdfa10c (done), `job_scripts/digress_run.sbatch` (written, not yet run) | prepared; training not started (GPU deferred) |
| 4 | Wrapper returns M i.i.d. children from one pass | Stage 2 | `vgmg/models/digress.py` (stub), `vgmg/models/denoiser.py` interface and NFE counter | interface done, wrapper planned |
| 5 | Value fn: MLP on atom/bond histograms + timestep embedding | Stage 3 | `models/value_function.py` (embedding done); histogram featurizer inside the wrapper | partly done |
| 6 | Value fn: 3-layer GNN | Stage 6 (go/no-go Nov 20) | `configs/sweeps/a2_gnn.yaml`; class stub | planned |
| 7 | MSE regression onto r(G0), 10k unguided rollouts, per reward | Stage 3 | `rl/value_trainer.py`, `value_fn.train_chains: 10000` | done on toy |
| 8 | Argmax or softmax selection; SVDD-PM | Stage 3-4 | `rl/svdd.py` | done on toy |
| 9 | Rewards SA (10-SA)/9 and QED, invalid -> 0; docking proxy stretch | Stage 1-4 | `rl/rewards.py` | done; docking dropped unless started by Nov 20 |
| 10 | Data: QM9 133,885, CC0; DiGress's own split; 1k sanity splits only for reward and t=0 checks; novelty vs all of QM9 | Stage 2-3 | `data/`, `scripts/qm9_reference.py`, `value_fn_t0_check` | done / planned |
| 11 | N = 2000, seeds 0-2 (+3-4 at M=10) | Stage 4, 6 | `configs/default.yaml`, `configs/sweeps/*.yaml` | done |
| 12 | Metrics incl. top-10% reward, validity, uniqueness, novelty, diversity, passes, wall-clock | Stage 2 | `eval/metrics.py`; `validity_digress` metric to add | partly done |
| 13 | B1, B2 (N = M, plus wall-clock-matched N), B3 PM, random-selection control | Stage 4 | `eval/baselines.py`, sweeps | done on toy |
| 14 | A1: M in {1,5,10,20}, SA and QED | Stage 4 (M=10), 6 (rest) | `a1_m_sweep.yaml` | done on toy |
| 15 | A2: PM / MLP / GNN at M=10 | Stage 4, 6 | `a2_estimator.yaml`, `a2_gnn.yaml` | done on toy; GNN planned |
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
- 2026-10-17: the revised proposal (`Abel_proposal.pdf`) was approved as submitted; no email sent. Slot sign-up deferred (sheet not posted). Stage 1 prep: DiGress is trained with the repo's config files and command-line overrides only (wandb off, seed 0, data path, loader workers, `n_epochs` sized to the 10 GPU-h budget); this stays inside the proposal's risk (i) and budget. Nothing in the proposal text changed.
- 2026-10-17 (later): the approved proposal's milestone row says "Sign up for Slot A on Oct 13"; the class has not posted the sheet, so this is the one proposal line not met on time (not the professor's doing, no action needed). GPU/GAIVI steps of Stage 1 are deferred by the user, so the proposal row "Oct 12-18: first GPU job, launch DiGress QM9 training" is late; the plan's gates float and risk R17 covers it. If the delay passes ~4 days after access returns, tell the professor.
