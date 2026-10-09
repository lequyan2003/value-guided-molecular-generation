# Q&A notes (write the answers in your own words; this file is for you, not for grading)

Rule: after each module lands, write the answer to its questions here **before** moving on. If you cannot,
read the code again until you can. Questions are the ones the grading guide says will be asked.

## SVDD / theory
- [ ] Derive v_t(x_t) = alpha log E[exp(r(x0)/alpha) | x_t] and the tilted kernel p*(x_{t-1}|x_t).
- [ ] What does M=1 give? M→∞? How does argmax relate to the alpha→0 limit?
- [ ] Why does MSE regression onto r(x0) not estimate the soft value, and when does that matter?
- [ ] Why is the value function trained on unguided rollouts, and what breaks when it is used on guided ones?
- [ ] Why does the value function need the timestep? What if I remove it?

## DiGress
- [ ] Why the official checkpoint is not used, what I trained instead (config, seed, epochs, validity reached) and why one training seed limits the conclusions.
- [ ] Validity conventions: DiGress (sanitize, largest fragment) vs mine (connected). Which numbers are comparable?
- [ ] Forward process (marginal transition matrices), what the network predicts, how x_{t-1} is sampled.
- [ ] Why one forward pass serves all M children (NFE accounting).
- [ ] Training loss and noise schedule; why the QM9 checkpoint is not retrained.

## Experimental design
- [ ] Why best-of-N is the right baseline, and where "N = M" is unfair in either direction.
- [ ] What `nfe_per_molecule` is for SVDD-MLP, SVDD-PM, best-of-N, and why they differ.
- [ ] What a seed controls here; why seeds 0-4; what the noise estimate is; what the pre-registered rule says.
- [ ] Where is the train/val/eval separation? (`seed_streams`, chain-level split.) Where could leakage still hide?
- [ ] Why invalid = 0 in the primary reward; what changes with reward over valid only?
- [ ] What are the 1k QM9 splits for, and what are they NOT used for?

## Counterfactuals to be able to reason about
- [ ] Double M. Halve alpha. Swap SA for QED. Use softmax instead of argmax. Drop the validity constraint.
- [ ] Reward spread is tiny: what happens to best-of-N and SVDD?
- [ ] Strongest objection to the result, and the next experiment.

## Code walk-through checklist (open the file, explain every function)
- [ ] `vgmg/rl/svdd.py` · [ ] `vgmg/models/denoiser.py` · [ ] `vgmg/models/digress.py` · [ ] `vgmg/rl/value_trainer.py`
- [ ] `vgmg/eval/baselines.py` · [ ] `vgmg/eval/metrics.py` · [ ] `vgmg/eval/stats.py` · [ ] `vgmg/experiments.py`
- [ ] `vgmg/utils/rdkit_utils.py` · [ ] one config · [ ] one job script
