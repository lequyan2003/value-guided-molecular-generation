# Data

QM9 (~134k small molecules) is **not** committed. Fetch it with:

```bash
python -m vgmg.data.download      # -> data/qm9.csv (MoleculeNet copy)
python scripts/prepare_splits.py  # -> data/splits/qm9_indices_{train,val,test}_1k.txt
```

Splits are a seeded (42) permutation, disjoint, and small enough to commit once generated, so every run
uses identical molecules. Both scripts are idempotent. The smoke test needs none of this.
