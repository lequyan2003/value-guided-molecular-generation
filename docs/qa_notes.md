# Q&A notes (write the answers in your own words; this file is for you, not for grading)

Rule: after each module lands, write the answer to its questions here **before** moving on. If you cannot,
read the code again until you can. Questions are the ones the grading guide says will be asked.

> **STATUS: Claude's first draft (2026-11-09).** Every answer under "Draft" is a starting point for you to read,
> check against the code, and rewrite in your own words. Tick a box only after you can answer without looking.
> `[VERIFY]` marks claims I derived from the repo notes or general knowledge and did not re-check in the code or logs.
> Numbers come from the `talk-snapshot` tag (`results/tables/summary.csv`).

## SVDD / theory
- [ ] Derive v_t(x_t) = alpha log E[exp(r(x0)/alpha) | x_t] and the tilted kernel p*(x_{t-1}|x_t).

  **Draft.** Target: the distribution that maximises E[r(x0)] - alpha * KL(q || p) over the whole reverse chain. Its
  optimum is p*(x0) ∝ p(x0) exp(r(x0)/alpha). Define the soft value v_t(x_t) = alpha log E_p[exp(r(x0)/alpha) | x_t].
  Then p*(x_{t-1}|x_t) = p(x_{t-1}|x_t) exp(v_{t-1}(x_{t-1})/alpha) / exp(v_t(x_t)/alpha). The denominator is the
  normaliser, because E[exp(v_{t-1}/alpha) | x_t] = exp(v_t/alpha) (tower property). So the optimal kernel is the pretrained
  kernel reweighted by exp(v/alpha). Code: docstring at the top of `vgmg/rl/svdd.py`.

- [ ] What does M=1 give? M→∞? How does argmax relate to the alpha→0 limit?

  **Draft.** M=1: one child, nothing to choose, so it is plain unguided sampling (the code returns the child directly).
  M→∞ with softmax at temperature alpha: importance resampling becomes exact sampling from p*. For finite M it is an
  approximation, and its quality depends on how good v is. Argmax is the alpha→0 limit of softmax(v/alpha): all
  probability goes to the highest-value child. It is greedy, so it trusts v completely, which is why it exploits v's errors.

- [ ] Why does MSE regression onto r(x0) not estimate the soft value, and when does that matter?

  **Draft.** MSE regression of r(x0) on x_t estimates the plain conditional mean E[r(x0)|x_t] (verified: the `value_trainer.py` docstring says it regresses the final reward with MSE), which is the alpha→∞
  limit of the soft value (log E exp(r/alpha) → E r when alpha is large). For finite alpha the soft value weights high
  rewards more (log-sum-exp is above the mean). It matters when alpha is small or the reward spread given x_t is wide:
  then the mean undervalues states with a chance of a high reward. In our runs the reward spread is small, so the two are
  close (an argument, not measured).

- [ ] Why is the value function trained on unguided rollouts, and what breaks when it is used on guided ones?

  **Draft.** v is defined by the pretrained model p (the expectation is over p's future), so its training data are
  unguided rollouts with their final rewards. Guided sampling visits states that were rare or absent in the training data.
  There the MLP extrapolates and its errors are not corrected. Argmax then selects exactly the children where v is wrongly high.
  This is our diagnosis for the argmax collapse (validity 0.015).

- [ ] Why does the value function need the timestep? What if I remove it?

  **Draft.** The same noisy graph means different things at different noise levels: the expected final reward given x_t
  depends on how much of x_t is still noise. At high noise v ≈ the average reward, and at low noise it is close to the
  reward of x_t itself. Without t, v would have to average over all noise levels and get worse at ranking. The Spearman
  plot (`value_fn_spearman_by_t.pdf`) shows this: about 0.6 near t/T=0.2, about 0.08 near pure noise.

## DiGress
- [ ] Why the official checkpoint is not used, what I trained instead (config, seed, epochs, validity reached) and why one training seed limits the conclusions.

  **Draft.** The official QM9 checkpoint link is dead (404, DiGress issue #109), so we trained DiGress ourselves: seed 0,
  1000 epochs, validity about 99.1 %, uniqueness about 96.1 %, matching the paper. Two runs exist (4090, 5090); the frozen
  model is the 4090 run (torch 2.0.1), checkpoint SHA-256 `ef376a81…`. One training seed means every result is conditional
  on this one model: a different trained DiGress might have a different value-function landscape, so "SVDD fails on DiGress"
  is really "on this model". The sampling seeds (0-2) only cover sampling noise.

- [ ] Validity conventions: DiGress (sanitize, largest fragment) vs mine (connected). Which numbers are comparable?

  **Draft.** Verified in `vgmg/utils/rdkit_utils.py`: our single definition of valid (`mol_from_smiles`) is RDKit-sanitizable AND one
  connected fragment, and every reward and metric goes through it, so no method gains from disconnected graphs. DiGress's convention
  (`is_sanitizable`) parses with fragments allowed and keeps the largest fragment, so it is more lenient. The table's `validity` is ours
  (strict); `validity_digress` is the lenient one and is the one comparable with the paper's 99.1 %. They differ a lot for argmax SVDD
  (0.015 strict vs about 0.19 lenient), so say which one every time.

- [ ] Forward process (marginal transition matrices), what the network predicts, how x_{t-1} is sampled.

  **Draft.** Forward: each node type and each edge type is corrupted independently with a transition matrix that mixes
  towards the marginal distribution of types in the training set (marginal-preserving noise). The network sees the noisy graph
  and t and predicts the clean graph (class probabilities per node and edge). Reverse: combine that prediction with the known
  forward posterior q(x_{t-1}|x_t, x0) and marginalise over x0 to get p(x_{t-1}|x_t), then sample nodes and edges. [VERIFY wording against the DiGress paper, Sec. 3-4]

- [ ] Why one forward pass serves all M children (NFE accounting).

  **Draft.** One pass gives the whole distribution p(x_{t-1}|x_t) for a parent. The M children are M independent samples
  from that one distribution, so no extra network calls. So: unguided T = 500; best-of-10 = M*T = 5000 (ten full chains);
  SVDD-MLP = T = 500 (the value MLP is cheap and not counted as a diffusion pass); SVDD-PM = (M+1)*T = 5500 (one extra
  `predict_x0` pass per child to get its posterior-mean molecule).

- [ ] Training loss and noise schedule; why the QM9 checkpoint is not retrained.

  **Draft.** Cross-entropy between the predicted and true clean graph (nodes and edges, edges weighted higher in DiGress),
  with a cosine-style noise schedule, T = 500 for QM9 [VERIFY schedule in `third_party/DiGress` config]. The model is frozen on
  purpose: SVDD is an inference-time method, and changing the model would mix a model effect into the comparison. (We did train
  it once, because the official checkpoint is dead.)

## Experimental design
- [ ] Why best-of-N is the right baseline, and where "N = M" is unfair in either direction.

  **Draft.** Best-of-N is the simplest way to spend more compute for a better reward, so SVDD must beat it to be worth the
  complexity. N = M matches the number of candidates, not the compute: best-of-10 spends 5000 passes, SVDD-MLP 500. That is
  unfair to best-of-N when the comparison is at equal passes (best-of-N would get only 1 sample) and unfair to SVDD when the
  comparison is at equal candidates (SVDD only sees partial states, best-of-N sees finished molecules and the true reward).
  Best-of-N also uses the true reward on complete molecules; SVDD-MLP only a learned estimate.

- [ ] What `nfe_per_molecule` is for SVDD-MLP, SVDD-PM, best-of-N, and why they differ.

  **Draft.** 500 / 5500 / 5000. See the NFE answer above. SVDD-MLP is cheapest because children share a pass and the
  scorer is not a diffusion pass. Wall-clock differs from NFE: SVDD-PM took about 1559 s per 1k molecules vs about 388 s for
  best-of-10 (table column `wall_s_per_1k`), because predict_x0 plus RDKit rewards per child is expensive.

- [ ] What a seed controls here; why seeds 0-4; what the noise estimate is; what the pre-registered rule says.

  **Draft.** A seed fixes the random streams (`seed_streams`: separate train / validation / eval streams) for the value-function
  data and the sampling noise; the DiGress weights stay fixed. We ran 3 seeds (0-2) for the talk; 3-4 are planned. Noise = max of the
  seed-to-seed std of the SVDD and baseline means. Pre-registered rule (`vgmg/eval/stats.py`): SVDD wins iff mean gain > 2 * noise
  AND validity drop <= 0.05 (absolute) versus the baseline at equal passes. Result: SVDD-MLP argmax vs best-of-10 gives
  gain -0.79 (about -60 noise units) and validity drop 0.98, so it fails both parts.

- [ ] Where is the train/val/eval separation? (`seed_streams`, chain-level split.) Where could leakage still hide?

  **Draft.** `vgmg/experiments.py` (around line 148-163): value-function training chains and validation chains come from
  different streams and are disjoint; evaluation molecules come from the `eval` stream. Possible leakage: the DiGress model
  was trained on QM9 itself, so generated molecules can be near or identical to training molecules (novelty about 0.15 for
  unguided, i.e. most generated molecules are in QM9). The value function is trained on rollouts from the same frozen model that is
  later guided, which is intended. The SA reward is fixed (no learning). [VERIFY the stream layout in `reproducibility.py`]

- [ ] Why invalid = 0 in the primary reward; what changes with reward over valid only?

  **Draft.** An invalid molecule has no SA or QED score, and counting it as 0 makes the reward punish invalid outputs, so a
  method cannot look good by producing few but high-reward valid molecules. Reward over valid only would hide the argmax collapse:
  the few valid molecules SVDD-MLP produces could have decent SA. With invalid = 0 the reward is validity-weighted, and the
  pre-registered rule also checks validity directly. [VERIFY: the `reward` column definition for SA in `rewards.py`]

- [ ] What are the 1k QM9 splits for, and what are they NOT used for?

  **Draft.** [VERIFY the exact split sizes in `scripts/train_value_function.py`.] From `docs/PLAN.md`: the value function is trained on
  10k rollout chains with 1k validation chains (disjoint from training chains) to check MSE against the constant predictor and Spearman per
  timestep. They are NOT used to train DiGress (it uses its own 100000/20497/13388 QM9 split) and NOT used to evaluate methods (evaluation
  molecules come from the separate `eval` stream). `qm9_reference.json` is a separate reference statistic file.

## Counterfactuals to be able to reason about
- [ ] Double M. Halve alpha. Swap SA for QED. Use softmax instead of argmax. Drop the validity constraint.

  **Draft.**
  - Double M (10 → 20): more candidates per step, so best-of-N improves (more draws) but costs 2x; SVDD argmax would exploit v more (more chances to find a child where v is wrongly high), so likely worse validity. Softmax gets closer to p* (better importance-resampling approximation). Not measured: the M sweep (A1) is planned.
  - Halve alpha: sharper selection. We measured it: 0.1 → 0.03 → 0.01 gives SA 0.669, 0.674, 0.127 and validity 0.99, 0.93, 0.20. Small alpha approaches argmax and collapses.
  - Swap SA for QED: QED exists only for unguided and best-of-10 (QED 0.463 → 0.568). SVDD arms are SA only. I cannot claim what happens.
  - Softmax instead of argmax: measured, it fixes validity (0.99 at alpha 0.1) at a small gain (SA 0.669 vs 0.635), still below best-of-10 (0.799) but at 10x fewer passes. Softmax was chosen after the pilot, so it is exploratory.
  - Drop the validity constraint: [VERIFY what "validity constraint" means in the rule]; the reward is already 0 for invalid molecules, and dropping the rule's validity check would let argmax look like a partial win on reward over valid molecules only.

- [ ] Reward spread is tiny: what happens to best-of-N and SVDD?

  **Draft.** With a tiny spread, picking the best of N gains little (the max of N draws is close to the mean) and the value
  function has almost nothing to learn (low signal-to-noise: Spearman near 0.04-0.13 for siblings supports this). SVDD then
  behaves close to random selection. Our random-select control sits at 0.636, equal to unguided, showing that extra candidates alone do nothing.
  The gain from best-of-10 (0.635 → 0.799) shows the SA spread is not zero, just hard for a learned v to rank among siblings.

- [ ] Strongest objection to the result, and the next experiment.

  **Draft.** Objection: "your value function is just badly trained, not SVDD at fault." Answer: SVDD-PM, which uses no learned
  value function, reaches SA 0.862 (above best-of-10) but collapses uniqueness to 0.54, so the mechanism works; the failure is
  in the learned v plus greedy selection. Another objection: one DiGress seed, SA only. Next experiments (not run, by decision): GNN value
  function; M sweep and compute-matched curves; more seeds; QED for all arms; H4. A second objection: softmax alpha was picked after seeing results.

## Likely extra questions (not on your original list)
- Why does SVDD-PM reach 0.862 with uniqueness 0.54? **Draft:** the posterior-mean score is the true reward of the network's current guess, an accurate signal, so greedy selection reliably finds the same easy-to-score small molecules (cyclononane-like) again and again: reward hacking, mode collapse. Reward goes up, diversity goes down.
- Is the argmax failure a bug? **Draft:** no. Evidence: tests pass (73), the toy generator works, the 5090 replicate of PM and random agrees to about 1e-3, the softmax arm with the same code works, and `docs/PROBLEMS_AND_FIXES.md` records the pilot diagnosis. Still, there is no independent reimplementation.
- Why did you add the softmax arm after the pre-registered rule failed? **Draft:** to diagnose, not to rescue: it tests the hypothesis that greedy selection exploits v. The pre-registered verdict stays "SVDD-MLP argmax does not beat best-of-10".
- What would you change if you had a week? **Draft:** more DiGress seeds, QED for all arms, GNN value function, train v on guided rollouts (iterating), and the M sweep.

## Code walk-through checklist (open the file, explain every function)
- [ ] `vgmg/rl/svdd.py` (`select_candidates`, `value_scorer`, `pm_scorer`, `svdd_sample`: the core, start here)
- [ ] `vgmg/models/denoiser.py` · [ ] `vgmg/models/digress.py` · [ ] `vgmg/rl/value_trainer.py`
- [ ] `vgmg/eval/baselines.py` · [ ] `vgmg/eval/metrics.py` · [ ] `vgmg/eval/stats.py` · [ ] `vgmg/experiments.py`
- [ ] `vgmg/utils/rdkit_utils.py` · [ ] one config · [ ] one job script
