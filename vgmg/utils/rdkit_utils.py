"""RDKit wrappers: parsing, canonicalization, QED, SA score, fingerprints, similarity."""

from __future__ import annotations

import os
import sys

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, AllChem, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402  (ships with RDKit Contrib)

RDLogger.DisableLog("rdApp.*")


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse SMILES; return None if unparsable, empty, or disconnected.

    This is the single definition of "valid" for the whole project: RDKit-sanitizable
    AND one connected fragment (the proposal's validity criterion). Every reward and
    metric goes through here, so no method can gain from disconnected graphs.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return mol


def is_valid(smiles: str) -> bool:
    """True if RDKit can parse and sanitize ``smiles``."""
    return mol_from_smiles(smiles) is not None


def canonicalize(smiles: str) -> str | None:
    """Canonical SMILES, or None if invalid."""
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def qed_score(smiles: str) -> float:
    """QED in [0, 1]; 0.0 for invalid molecules."""
    mol = mol_from_smiles(smiles)
    return float(QED.qed(mol)) if mol is not None else 0.0


def sa_score(smiles: str) -> float:
    """Ertl & Schuffenhauer SA score in [1, 10] (lower = easier); 10.0 if invalid."""
    mol = mol_from_smiles(smiles)
    return float(sascorer.calculateScore(mol)) if mol is not None else 10.0


def morgan_fingerprint(smiles: str, n_bits: int = 128, radius: int = 2) -> np.ndarray:
    """Morgan bit-vector as float32 array; all zeros if invalid."""
    out = np.zeros(n_bits, dtype=np.float32)
    mol = mol_from_smiles(smiles)
    if mol is not None:
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), out
        )
    return out


def tanimoto_diversity(smiles_list: list[str], n_bits: int = 1024) -> float:
    """1 - mean pairwise Tanimoto similarity over valid molecules (0.0 if < 2 valid)."""
    fps = []
    for s in smiles_list:
        mol = mol_from_smiles(s)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    if len(fps) < 2:
        return 0.0
    sims = [
        s for i in range(1, len(fps)) for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    ]
    return 1.0 - float(np.mean(sims))
