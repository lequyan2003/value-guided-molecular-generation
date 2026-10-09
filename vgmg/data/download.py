"""Fetch QM9 (SMILES CSV from MoleculeNet). Usage: ``python -m vgmg.data.download``."""

import urllib.request
from pathlib import Path

from vgmg.constants import DATA_DIR, QM9_FILENAME, QM9_URL


def download_qm9(dest_dir: Path = DATA_DIR) -> Path:
    """Download QM9 to ``dest_dir``; idempotent (skips if the file exists)."""
    dest = dest_dir / QM9_FILENAME
    if not dest.exists():
        urllib.request.urlretrieve(QM9_URL, dest)
    return dest


if __name__ == "__main__":
    print(download_qm9())
