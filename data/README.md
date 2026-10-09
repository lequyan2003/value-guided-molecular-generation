# Data

QM9 (~134k small molecules) is **not** committed. Fetch it with:

```bash
python -m vgmg.data.download      # -> data/qm9.csv (MoleculeNet copy)
python scripts/prepare_splits.py  # -> data/splits/qm9_indices_{train,val,test}_1k.txt
```

The 1k splits are a seeded (42) permutation, disjoint, committed once generated. **They are sanity splits only:**
`scripts/qm9_reference.py` uses the test split to check the reward code on real molecules and to give a headroom
reference. They are never used to train or tune anything; DiGress is pretrained on its own QM9 split and the value
function is trained on generated rollouts. Novelty is measured against the **full** QM9 set (stricter than DiGress's
train-split novelty). Both scripts are idempotent. The smoke test needs none of this.
