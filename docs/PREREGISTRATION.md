# Pre-registration (fixed before any DiGress result exists)

Commit this file before the first main-sweep run. After that, changes need a dated entry in the
"Amendments" section with the reason; nothing is silently edited. The analysis is mechanical:
`make results` applies this rule through `vgmg/eval/stats.py::preregistered_verdict`.

## Question
Does SVDD with a learned value function on a frozen DiGress model trained on QM9 raise mean SA / QED reward more
than best-of-N with the same number of candidates (N = M), without lowering validity, and how does
the gain scale with M? (Matches the revised proposal; see `docs/PROPOSAL_ALIGNMENT.md`.)

## Definitions
- **Validity**: RDKit-sanitizable and a single connected fragment (`vgmg/utils/rdkit_utils.py`).
  Stricter than DiGress's own metric, which sanitizes and then keeps the largest fragment (verified in
  `analysis/rdkit_functions.py`); the DiGress-convention validity is also reported (`validity_digress`, to be added at Stage 2)
  so published numbers (99.0% valid, 96.2% unique) can be compared.
- **Primary metric**: `reward_mean` = mean reward over all N=2000 generated molecules, invalid = 0.
  Secondary: `reward_top10`, `reward_mean_valid`, uniqueness, novelty (vs full QM9), diversity (ECFP4).
- **SA reward** = (10 - SA)/9; **QED** from RDKit. Both in [0,1].
- **Seed** controls value-function rollouts, network init and evaluation sampling through independent
  streams (`seed_streams`). Sweeps use seeds 0-2 (as proposed); the headline point (M=10: SVDD-MLP, best-of-10, B1) additionally runs seeds 3-4, decided now and run whatever seeds 0-2 show. The value function is retrained per seed.

## Comparisons
| Id | Method | Role |
|---|---|---|
| B1 | unguided DiGress | reference |
| B2 | best-of-N (N = M), true reward as selector | **primary baseline** |
| B3 | SVDD-PM (reward of the denoiser's own x0 prediction; no learned value) | training-free guidance |
| C  | random selection among M children per step | isolates the value signal from extra candidates |
| A1 | SVDD-MLP, M in {1,5,10,20} | scaling |
| A2 | estimator: PM vs MLP (GNN only if tier T3) at M=10 | isolates the value function |

## Cost accounting (the revised proposal's wording)
The first proposal said "equal denoiser evaluations". Verified in DiGress's `sample_p_zs_given_zt` (one `self.forward` call, then
sampling from the posterior): one network pass yields p(x0|x_t) from which all M children are sampled, so
- SVDD-MLP ≈ T forward passes per molecule (+ M·T cheap value evaluations),
- SVDD-PM ≈ (M+1)·T, best-of-N = M·T.
So "N = M" matches **candidates**, and gives B2 up to M× more network passes than SVDD-MLP.
Every run records `nfe_per_molecule` and sampling wall-clock. Reported: reward vs NFE and vs
wall-clock curves (B2 at N in {1,5,10,20} against SVDD at M in {1,5,10,20}). The primary test below
is candidate-matched (M = N = 10); a secondary point compares SVDD with best-of-N at the **same measured wall-clock**
(the protocol of the SVDD paper's Best-N baseline), taking N from the pool prefix; the cost-axis plots decide how it is worded in the report.
Value-function training cost (rollouts + fitting) is reported separately and amortised over
the 2000 molecules of that run.

## Primary decision rule (per reward, M = 10, paired by seed, all seeds 0-4)
SVDD-MLP **beats** B2 iff
1. mean over seeds of (SVDD reward_mean - B2 reward_mean) > 2 x noise, where
   noise = max(std over seeds of SVDD per-seed means, std over seeds of B2 per-seed means), AND
2. mean validity of SVDD >= mean validity of B1 - 0.05 (absolute).

The verdict is also reported on seeds 0-2 alone (the proposal's design); if the two verdicts differ, both are shown and the 5-seed one is primary. Reported alongside (not part of the rule): a paired bootstrap 95% CI over molecules within seeds
(seeds fixed), and all per-seed values. A claim is made only if rule 1 and 2 hold; if the CI
includes 0 while rule 1 holds, the report says so. Failing the rule is reported as a
negative result and analysed with H1-H3.

## Slot A reporting
At the talk (Nov 13) only seeds 0-2 exist: the verdict is shown for seeds 0-2 and labeled **preliminary**. The final verdict (all
paired seeds, 0-4 at M=10) is computed by the same code on the frozen Nov 29 data; if it differs from the talk's, both are reported.
Effect size is reported next to the rule: gain over B2 in reward units and as a fraction of the B2-over-B1 gain.

## Wall-clock
Comparable only among runs on the same GPU type (recorded per run; `make results` warns on mixed GPUs). Headline runs target
the ClsGenAI Titan RTX. GPU kernels are not bit-deterministic: CPU runs reproduce exactly, GPU reruns reproduce up to
floating-point nondeterminism (reported tolerance to be measured at G1 by rerunning one run).

## Reward-hacking check
A guided method whose mean reward exceeds the top-10% of real QM9 (`qm9_reference.json`: SA 0.825, QED 0.578) or whose diversity/novelty
collapses is inspected (smallest-molecule collapse) before any claim; diversity and novelty are always reported next to reward.

## Sanity checks that gate the experiments
S1 reward code on real QM9 test split (done: SA mean 0.631, QED 0.466). S2 value function at t=0 on held-out real molecules
(`value_fn_t0_check`: Spearman with true reward must be clearly positive, else the featurizer is wrong). S3 M=1 SVDD equals unguided.

## Mechanism hypotheses (for explaining the result either way)
- **H1** value function ranks siblings poorly at large t: report validation MSE vs the constant
  predictor, Spearman per t-bin, and `sibling_ranking` (within-parent Spearman against
  continuation-averaged reward).
- **H2** greedy argmax reduces diversity/novelty: compare argmax vs softmax (alpha in {0.1, 1}), M=10.
- **H3** limited headroom: reward spread of B1 and of real QM9 (`results/tables/qm9_reference.json`).

## Approximations that must be stated in the report
1. The value function is fitted by MSE to r(x0), which estimates E[r(x0)|x_t,t], not the soft
   value alpha log E[exp(r/alpha)|x_t]; with argmax selection only ranking matters. (Re-check the
   paper's statement of its estimator before the talk.)
2. It is trained on unguided rollouts but used on guided trajectories (distribution shift).
2b. The frozen DiGress is a single training run (seed 0): all conclusions are conditional on that model. Seeds vary the value
   function, rollouts and sampling, not the generator. (There is no official checkpoint; ours is trained with the repo's unmodified
   QM9 config.)
2c. SVDD-PM scores a child by the reward of the argmax-decoded denoiser prediction; an invalid decode scores 0, so ties at early
   steps are broken at random and the tie fraction is reported.
3. Features for the MLP are histograms of atom/bond types of a noisy graph, a lossy summary.
4. Rewards are cheap and exact for B2 (oracle selector), which favours B2.

## Not run, with reasons
DDPO-style fine-tuning (needs training; the point of SVDD is to avoid it); DiGress regressor
guidance (needs a retrained property regressor; different question).

## Amendments
**2026-11-05 (schedule only, before any real run exists; no change to the question, rule, seeds, N, metrics or comparisons).**
1. Dates re-baselined to the Slot A talk (Nov 13): DiGress frozen Sun Nov 8 12:00, results frozen Tue Nov 10 18:00 (was Nov 8), slides uploaded Nov 11
   (`docs/PLAN.md` Sec. 4a). The "Slot A reporting" paragraph above still holds: only seeds 0-2 can exist at the talk, the verdict is labeled preliminary.
2. If time forces a smaller Slot A scope, the order of cuts is fixed now: seeds 1-2 and QED come after the talk, then N = 1000 (the proposal's own
   single-GPU fallback scale). The verdict is computed only with >= 3 paired seeds (already enforced in `eval/comparison.py::headline_verdict`);
   with fewer, the talk reports means with their std and says "no verdict yet". The decision rule itself is untouched.
3. The frozen DiGress is whatever the training run has reached at the freeze time (proposal risk (i): >= 97 % target, < 90 % freeze and report);
   its DiGress-metric validity is reported next to every result.
