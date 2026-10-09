"""Project-wide constants: paths, defaults, and the tiny smoke-test molecule pool."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = REPO_ROOT / "configs"
DATA_DIR: Path = REPO_ROOT / "data"
SPLITS_DIR: Path = DATA_DIR / "splits"
RESULTS_DIR: Path = REPO_ROOT / "results"

DEFAULT_SEED: int = 42
FINGERPRINT_BITS: int = 128

QM9_URL: str = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
QM9_FILENAME: str = "qm9.csv"

# Small, valid, QM9-style molecules (<= 9 heavy atoms). Used ONLY by the smoke
# test and unit tests so they run with no download and no pretrained checkpoint.
SMOKE_SMILES: tuple[str, ...] = (
    "C", "CC", "CCC", "CCO", "CCN", "CC=O", "CC#N", "C=CC", "C#CC", "CCCC",
    "CC(C)C", "CC(C)O", "CC(C)N", "CCOC", "CCNC", "CC(=O)C", "CC(=O)O",
    "CC(=O)N", "C1CC1", "C1CCC1", "C1CCCC1", "C1CCOC1", "C1CCNC1", "C1COC1",
    "C1CNC1", "C1=CC=CC=C1", "C1=CC=NC=C1", "C1=CNC=C1", "C1=COC=C1",
    "OC1CC1", "NC1CC1", "OCC1CC1", "CC1CCO1", "CC1(C)CC1", "CC12CC1C2",
    "C1CC2CC1C2", "C1CC2CCC1O2", "OC1C2CC1C2", "NC1C2CC1C2O", "CC#CC",
    "C=CC=C", "N#CC#N", "O=CC=O", "OCC(O)CO", "NCC(N)CN", "CC(O)C(N)=O",
    "O=C1CCC1", "O=C1CCCO1", "N#CC1CC1", "C1=CC2CC1C2", "C1OC2CC1C2",
    "OC12CC1CO2", "CC1=NC=CN1", "CC1=CN=CO1", "O=C1NC=CO1", "CC(C)(C)O",
    "CC(C)(C)N", "OC1CCOC1", "NC1CCNC1",
)
