# Contributions

| Component | Origin | What I changed |
|---|---|---|
| DiGress backbone | cvignac/DiGress (commit: bdfa10c, MIT) | Unmodified; inference only. **Not yet integrated.** |
| SVDD algorithm | Li et al., arXiv:2408.08252; hyanan16/SVDD-molecule @ 455e4d4 read for reference only | Own implementation (`vgmg/rl/svdd.py`); written from the paper, not copied from the authors' code |
| Value function | Standard PyTorch | Own MLP; GNN variant TODO |
| Rewards | RDKit QED; Ertl & Schuffenhauer SA score (RDKit Contrib) | Wrappers + [0,1] rescaling; docking proxy TODO |
| Eval metrics | RDKit | Own implementation (validity, uniqueness, novelty, reward, diversity) |
| `ToyDenoiser` | Own | Smoke-test stand-in only; not a model of molecules |

Update this table (and commit hashes) as components land.
