"""Create committed, seeded QM9 split-index files (1k each) in data/splits/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.constants import DEFAULT_SEED, SPLITS_DIR  # noqa: E402
from vgmg.data.loaders import load_qm9_smiles  # noqa: E402
from vgmg.data.preprocessing import make_split_indices  # noqa: E402


def main() -> None:
    """Write qm9_indices_{train,val,test}_1k.txt (deterministic given the seed)."""
    n = len(load_qm9_smiles())
    splits = make_split_indices(n, (1000, 1000, 1000), DEFAULT_SEED)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, idx in zip(("train", "val", "test"), splits):
        (SPLITS_DIR / f"qm9_indices_{name}_1k.txt").write_text("\n".join(map(str, idx)) + "\n")


if __name__ == "__main__":
    main()
