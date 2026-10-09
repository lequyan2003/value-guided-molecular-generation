"""Load QM9 SMILES and committed split indices."""

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from vgmg.constants import DATA_DIR, QM9_FILENAME, SPLITS_DIR


def load_qm9_smiles(path: Path = DATA_DIR / QM9_FILENAME) -> list[str]:
    """Return all QM9 SMILES in file order (run ``python -m vgmg.data.download`` first)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m vgmg.data.download`")
    return pd.read_csv(path)["smiles"].tolist()


def load_split(name: str, size: str = "1k", splits_dir: Path = SPLITS_DIR) -> list[int]:
    """Read ``qm9_indices_<name>_<size>.txt`` (one index per line)."""
    path = splits_dir / f"qm9_indices_{name}_{size}.txt"
    return [int(x) for x in path.read_text().split()]


def smiles_dataloader(smiles: list[str], batch_size: int, shuffle: bool = False) -> DataLoader:
    """Batches of SMILES strings (default collate keeps them as lists of str)."""
    return DataLoader(smiles, batch_size=batch_size, shuffle=shuffle)
