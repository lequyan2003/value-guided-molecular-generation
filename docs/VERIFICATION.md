# Verification log (every factual claim the proposal and plan rely on)

Checked 2026-10-09 against primary sources (paper PDFs, repository source, APIs). Status: **V** verified, **C** corrected
(proposal/plan changed), **U** cannot be verified until a stated gate. Re-verify anything marked U at its gate and amend here.

## SVDD (Li et al., arXiv 2408.08252)
| # | Claim | Source checked | Status |
|---|---|---|---|
| 1 | NeurIPS 2025 main track; authors Li, Zhao, Wang, Scalia, Eraslan, Nair, Biancalani, Ji, Regev, Levine, Uehara | NeurIPS proceedings page; arXiv metadata | V |
| 2 | Soft value `v_{t-1}(x) = alpha log E[exp(r(x0)/alpha) | x_{t-1}=x]`; soft-optimal kernel `p*(x_{t-1}|x_t) ∝ p_pre(x_{t-1}|x_t) exp(v_{t-1}/alpha)` | Sec. 4.1 | V |
| 3 | Algorithm 1: draw M candidates from the pretrained kernel, weight by `exp(v̂/alpha)`, sample an index; `alpha = 0` is argmax and is the recommended/default choice; M in 5-20, default M=10 for non-image domains | Sec. 4.2, 7.1 | V |
| 4 | SVDD-MC regresses `r(x0)` onto `(x_t, t)` (Alg. 2), which estimates `E[r(x0)|x_t]`, an acknowledged approximation of the soft value (eq. 1); SVDD-PM uses `r(x̂0(x_t))` (Alg. 3) | Sec. 4.3 | V (matches our pre-registered approximation 1) |
| 5 | The paper already compares against **Best-of-N** (N chosen so inference *time* is about equal to SVDD) on images, molecules and sequences | Sec. 7.1, Table 2 | **C**: proposal previously implied no SVDD-vs-best-of-N comparison exists |
| 6 | Paper's molecule experiments use **GDSS (continuous) trained on ZINC-250k, T=1000**, rewards QED, SA, docking; value fn is a GIN with hyperparameters chosen from grids; not MOSES | Sec. 7.1, F.1 | **C**: proposal said "MOSES/docking-based pipeline" |
| 7 | "A detailed empirical comparison between our method and [DiGress guidance] is left for future work"; Remark 4: molecule experiments use GDSS, which differs from DiGress | App. C | V (basis of the new gap statement) |
| 8 | SVDD cost: "if Line 3 is not computed in parallel, time is about M times standard inference"; Fig. 3: GPU time grows sublinearly in M | Sec. 4.2, 7.2 | V; cost depends on the value network size (ours is a small MLP) |
| 9 | SVDD lowers uniqueness (0.9375 vs 1.0) on GDSS | Table 3 | V (supports hypothesis H2) |
| 10 | `hyanan16/SVDD-molecule`: exists, commit 455e4d4 exists (merge, 2024-10-07), implements the molecule tasks with `sample_zinc250k` config, installs the MOSES package for metrics, **no license**; DNA/RNA code is `masa-ue/SVDD` | GitHub API, README | **C** (description) / V (commit). No license means we read for reference only and copy nothing (already our policy) |

## Related work found by search (positioning only)
| # | Claim | Source | Status |
|---|---|---|---|
| 11 | TreeDiff (Zhao et al., WWW 2026, arXiv 2510.10402) benchmarks Best-of-N, beam search, TAGMol and SVDD on GraphDiT with matched compute budgets (BACE/BBBP/HIV tasks; QM9 also used) | arXiv PDF text | V; cited as 4th supporting reference |
| 12 | Kerby & Moon (arXiv 2409.07359): training-free guidance on DiGress with simple guidance functions (atom-type proportion, molecular weight) | arXiv abstract | V; report-level related work |
| 13 | I found no study of SVDD on DiGress/QM9 with counted network passes and several seeds | web searches 2026-10-09 | U (absence claim; worded as "I found no") |

## DiGress (Vignac et al., ICLR 2023; repo cvignac/DiGress)
| # | Claim | Source | Status |
|---|---|---|---|
| 14 | Commit `bdfa10c` exists and is the current `main` head (2026-09-10) | GitHub API | V |
| 15 | README bug notice (2026-09-10): `dim_ffy` is ignored in the GraphTransformer (default 2048 used), impact unknown; `fixed_bug` branch exists but is "not evaluated" | README | V; we use `main` as-is because published numbers come from this code |
| 16 | **Official QM9 checkpoint link `drive.switch.ch/.../8IhyGE4giIW1wV3` returns HTTP 404**; README says the author lost the drive account; issue #109 (open, 2025-09), #99, #101 report the same; no mirror found; one user (issue #59) trained the default config and offers to share by email | curl, GitHub issues, web search | **C**: proposal said "released QM9 weights"; plan now trains DiGress ourselves |
| 17 | `sample_p_zs_given_zt`: ONE network forward (`self.forward`) per call, then samples `x_{t-1}` from `sum_{x0} pred(x0) q(x_{t-1}|x_t,x0)`; extra features computed from the noisy graph in the same call | source lines 597-655 | V (basis of the pass-count accounting: M children cost one pass) |
| 18 | `diffusion_steps: 500`, cosine schedule, marginal transition; QM9 experiment `qm9_no_h`: 1000 epochs, batch 1024, 9 layers, `lambda_train [5,0]`, heavy atoms only | configs | V |
| 19 | QM9 results without validity correction: valid **99.0 ± 0.1**, unique **96.2 ± 0.1**, "training time to reach 99% validity" **1.0 h** (hardware not stated in the table) | Table 2 | V; training time on our GPU is **U until G1** |
| 20 | DiGress validity = RDKit sanitization, then the **largest fragment** is kept; ours additionally requires a single connected fragment | `analysis/rdkit_functions.py` | V (our metric is stricter; we also report the DiGress convention) |
| 21 | Environment: Python 3.9, PyTorch 2.0.1 + CUDA 11.8, torch_geometric 2.3.1, rdkit 2023.03.2, graph-tool 2.45; `main.py` imports `graph_tool` at import time; orca must be compiled | README, `main.py` | V; installability on GAIVI is **U until G1** |
| 22 | DiGress QM9 split: 100,000 train, 10% of the data test, rest val, `random_state=42`; raw data from the MoleculeNet S3 copy | `qm9_dataset.py` | V |

## Data and references
| # | Claim | Source | Status |
|---|---|---|---|
| 23 | QM9 CSV has 133,885 rows (133,802 distinct SMILES strings); figshare "Data for 133885 GDB-9 molecules" is CC0 | local file; figshare API | V (duplicates matter only for set-based novelty, which uses canonical sets) |
| 24 | Gao, Schulman, Hilton, ICML 2023, pp. 10835-10866 | PMLR page | V |
| 25 | DDPO (Black et al.) is ICLR 2024 | ICLR proceedings | V |
| 26 | Ertl & Schuffenhauer 2009 (J. Cheminf. 1:8), Bickerton 2012 (Nat. Chem. 4:90-98), Ramakrishnan 2014 (Sci. Data 1:140022) | standard references, DOI of the QM9 collection matches | V for QM9; others from memory, **re-check DOIs when the report bibliography is built** |

## Course rules (guidelines PDF)
Slot A = Fri Nov 13: bar is "working pipeline, main result, one baseline; ablations may be planned; preliminary numbers labeled".
Slides due 11:59 PM the night before (Nov 12). Slot sign-up opens Oct 13, closes Oct 23 (default Slot C). Timing may be re-announced
two weeks before Nov 13 (about Oct 30). Report/code due Dec 11. 6930: >= 3 seeds, >= 2 ablations, 12-20 references.
Not stated in the guidelines (ask the professor): whether a slot can be changed after sign-up; whether a revised proposal is
accepted if it was approved.

## Still unverified (assumptions with owners and gates)
| Assumption | Gate | If wrong |
|---|---|---|
| 0.1 s per batch-pass of 512 graphs | G1 | rerun `scripts/estimate_gpu_hours.py`; proposal risk (iii) |
| DiGress trains to 99% validity within the 10 GPU-h budget on a Titan RTX | G1 (ETA), G2 | risk R1 in the plan |
| graph-tool/PyG/torch 2.0.1 build on GAIVI | G1 | stub `graph_tool`, ask admins |
| Compute nodes have no internet (data/ckpts downloaded on the login node) | G1 | stage data first |
| Python 3.9 CI job passes | next push | fix compatibility |
| GPU reruns reproduce up to a small tolerance | G1 | report the tolerance |
