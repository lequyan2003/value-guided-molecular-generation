# Pre-registration (fixed before any DiGress result exists)

Commit this file before the first main-sweep run. After that, changes need a dated entry in the
"Amendments" section with the reason; nothing is silently edited. The analysis is mechanical:
`make results` applies this rule through `vgmg/eval/stats.py::preregistered_verdict`.

## Question
Does SVDD with a learned value function on frozen DiGress (QM9) raise mean SA / QED reward more
than best-of-N with the same number of candidates (N = M), without lowering validity, and how does
the gain scale with M?

## Definitions
- **Validity**: RDKit-sanitizable and a single connected fragment (`vgmg/utils/rdkit_utils.py`).
  Stricter than DiGress's own metric, which keeps the largest fragment.
- **Primary metric**: `reward_mean` = mean reward over all N=2000 generated molecules, invalid = 0.
  Secondary: `reward_top10`, `reward_mean_valid`, uniqueness, novelty (vs full QM9), diversity (ECFP4).
- **SA reward** = (10 - SA)/9; **QED** from RDKit. Both in [0,1].
- **Seed** controls value-function rollouts, network init and evaluation sampling through independent
  streams (`seed_streams`). Sweeps use seeds 0-4; the value function is retrained per seed.

## Comparisons
| Id | Method | Role |
|---|---|---|
| B1 | unguided DiGress | reference |
| B2 | best-of-N (N = M), true reward as selector | **primary baseline** |
| B3 | SVDD-PM (reward of the denoiser's own x0 prediction; no learned value) | training-free guidance |
| C  | random selection among M children per step | isolates the value signal from extra candidates |
| A1 | SVDD-MLP, M in {1,5,10,20} | scaling |
| A2 | estimator: PM vs MLP (GNN only if tier T3) at M=10 | isolates the value function |

## Cost accounting (corrects the proposal's wording)
The proposal says "equal denoiser evaluations". With DiGress one network forward pass yields the
distribution p(x0|x_t) from which all M children are sampled, so (verify at gate G1 by reading
`sample_p_zs_given_zt`):
- SVDD-MLP ≈ T forward passes per molecule (+ M·T cheap value evaluations),
- SVDD-PM ≈ (M+1)·T, best-of-N = M·T.
So "N = M" matches **candidates**, and gives B2 up to M× more network passes than SVDD-MLP.
Every run records `nfe_per_molecule` and sampling wall-clock. Reported: reward vs NFE and vs
wall-clock curves (B2 at N in {1,5,10,20} against SVDD at M in {1,5,10,20}). The primary test below
is candidate-matched (M = N = 10); the cost-axis plots decide how it is worded in the report.
Value-function training cost (rollouts + fitting) is reported separately and amortised over
the 2000 molecules of that run.

## Primary decision rule (per reward, M = 10, paired by seed)
SVDD-MLP **beats** B2 iff
1. mean over seeds of (SVDD reward_mean - B2 reward_mean) > 2 x noise, where
   noise = max(std over seeds of SVDD per-seed means, std over seeds of B2 per-seed means), AND
2. mean validity of SVDD >= mean validity of B1 - 0.05 (absolute).

Reported alongside (not part of the rule): a paired bootstrap 95% CI over molecules within seeds
(seeds fixed), and all per-seed values. A claim is made only if rule 1 and 2 hold; if the CI
includes 0 while rule 1 holds, the report says so. Failing the rule is reported as a
negative result and analysed with H1-H3.

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
3. Features for the MLP are histograms of atom/bond types of a noisy graph, a lossy summary.
4. Rewards are cheap and exact for B2 (oracle selector), which favours B2.

## Not run, with reasons
DDPO-style fine-tuning (needs training; the point of SVDD is to avoid it); DiGress regressor
guidance (needs a retrained property regressor; different question).

## Amendments
(none yet)
