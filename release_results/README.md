# Run records behind every reported number

`logs/<run_id>/` holds, for each finished run: `summary.json` (metrics under every reported reward, value-function
diagnostics, git commit, GPU, torch/model identity incl. the checkpoint SHA-256, pass counts, sampling wall-clock),
`config.yaml`, `seed.txt`, `git_commit.txt`, `metadata.json` and `molecules.csv` (the generated molecules with their rewards).
`tables/` is regenerated from them with `make reproduce` (`summary.csv`, `verdict.json` = the pre-registered test,
`verdict_exploratory.json` = the same rule applied to the exploratory arms, `wall_clock_by_gpu.csv`).

Run ids: `<method>_[<arch>_]<reward>_M<M>[_h4r<rounds>|_softmax<alpha>]_s<seed>`; `unguided_s<seed>`; `best_of_n_pool_s<seed>`
is the pooled draw behind `best_of_n_<reward>_M{5,10,20}_s<seed>`.

Provenance notes (all runs use the same frozen checkpoint, SHA-256 `ef376a81...`, and the same sampling/SVDD/evaluation code):
- Commits: `d936c5c` (talk runs), `914a326` and `60b19f2` (later runs). The diffs between them touch no model, SVDD or sampling code
  (only result aggregation, demo/figure scripts and the opt-in H4 refit).
- Ten talk-era runs (B2 seeds 1-2, SVDD-PM, random control, all run on the CSAIL 4090 from a copy without a git folder) record
  `git_commit: unknown`; their sync stamp was `d936c5c`.
- Wall-clock is only comparable within one GPU type; see `tables/wall_clock_by_gpu.csv`.
- Same-seed re-runs on a different commit agree to about 6e-4 in mean reward (GPU nondeterminism).
