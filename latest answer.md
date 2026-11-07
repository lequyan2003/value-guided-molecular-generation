**1. Are the results good, with no problems?** They hold up, with caveats. I recomputed every run from its saved molecules:
- **Counts and validity:** all 27 runs have 2000 molecules, and the strict validity recount matches each summary exactly.
- **Configs:** every run has the same config, commit (d936c5c) and checkpoint hash. Only the method, seed and softmax alpha differ.
- **5090 vs 4090:** the replicate of SVDD-PM and random selection matches the 4090 to about 0.001 in SA, with validity and uniqueness within 0.0005.
- **Control and baseline:** random selection equals unguided (0.636 vs 0.635), and unguided SA matches real QM9 (0.635 vs 0.631).
- **Best-of-10:** its SA of 0.799 is about what the order statistic predicts.
- **Collapses explained:** SVDD-MLP argmax's collapse reproduces the pilot diagnosis (the value function is exploited off its training data). SVDD-PM's low uniqueness traces to its most common sample, cyclononane (SA exactly 1.0), appearing 114–134 times in 2000. SA rewards simplicity, so this is reward hacking of the metric, not a logic error.

Two things I found:
- **SA reward not exactly reproducible (minor).** The SA reward can't be reproduced exactly from the saved SMILES for 0.1–1 % of molecules, and some runs differ by up to 0.2 per molecule. The effect on any run mean is 2e-4 or less, ten times smaller than seed noise, so no conclusion changes. I'll state it in the report.
- **Single-machine arms.** B1, SVDD-MLP and the softmax arm ran on one machine only (GAIVI), so they have no duplicate to compare. I also haven't tested on the real model that SVDD with M=1 equals unguided.

**2. Section 8 of `Abel_progress.md`:** yes. It now has:
- 8.9 the final results;
- 8.10 a code map;
- 8.11 the validation above and its gaps;
- 8.12 the proposal alignment;
- 8.13 progress towards Slot A.

**3. Alignment with the proposal:** the method, model, baselines, rewards, pass accounting, N, seeds and falsifiable rule match. These don't match yet:
- QED was run only for B1 and B2, not for the SVDD arms or the control.
- B2 ran only at N = M = 10, with no wall-clock-matched smaller N.
- A1 beyond M=10, the GNN and seeds 3–4 are not run yet, as planned.
- The softmax arm is an addition, labelled exploratory.
- Wall-clock isn't comparable across arms because of the different GPUs.

The pre-registered primary is not supported, and the proposal's H1 is weak.

**4. Progress against Friday:** the experiments are ahead of schedule and already exceed the professor's bar. These are left:
1. The `talk-snapshot` tag by Tue Nov 10 18:00.
2. `docs/qa_notes.md`, all items still unticked, which is yours to write.
3. Slides, uploaded Wed night.
4. Figure fixes: the softmax arm isn't drawn, and I'd add uniqueness.
5. The demo on the real model, not yet run.
6. Rehearsals Wed and Thu.

The plan documents are updated and pushed (722508f): the gate log, the new Sec. 4b in `docs/PLAN.md`, and the proposal alignment map and status log.

**Questions for you:**
- **Lift the no-new-experiments rule?** There's time before the freeze. My order would be: QED for SVDD-MLP, PM and the control (about 3.5 GPU-h), B2 at smaller N, then H4. H4 retrains the value function on guided rollouts and is the proposal's remedy for the argmax collapse. Which do you want, if any?
- **Slides:** should I draft them, starting Monday, from the table and figures?
