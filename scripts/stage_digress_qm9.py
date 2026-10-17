"""Stage the raw QM9 files in the layout DiGress's loader expects, without importing torch or rdkit.

usage: python scripts/stage_digress_qm9.py [--out data/digress_qm9/qm9_pyg]
Why: DiGress downloads in ``QM9Dataset.download`` (needs rdkit + torch_geometric; GAIVI's login node cannot import torch), and it
only writes the split CSVs inside that method. If the raw files exist, PyG skips ``download`` and DiGress would fail on the
missing ``train.csv``. So we fetch the same files ourselves and write the splits with DiGress's own rule (shuffle with
``random_state=42``, 100000 train, int(0.1*n) test, rest val). Pass ``dataset.datadir=<abs out>`` to DiGress; ``process()`` then runs
on the first compute-node job and writes ``processed/``.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAW_ZIP = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
RAW_ZIP_SHA256 = "f11d3f8ecc3097a72656a35c4847767852e6f346bf99a54619a1a3b6389ddc02"  # checked 2026-10-16
UNCHARACTERIZED = "https://ndownloader.figshare.com/files/3195404"  # 403 without a User-Agent header
N_MOLECULES = 133885


def fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with a browser-like User-Agent (figshare rejects the default one)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def main() -> None:
    """Download, verify, extract and split."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/digress_qm9/qm9_pyg")
    raw = Path(p.parse_args().out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    zpath = raw / "qm9.zip"
    if not (raw / "gdb9.sdf").exists():
        fetch(RAW_ZIP, zpath)
        digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
        assert digest == RAW_ZIP_SHA256, f"qm9.zip sha256 changed: {digest}"
        with zipfile.ZipFile(zpath) as z:
            for name in ("gdb9.sdf", "gdb9.sdf.csv"):
                z.extract(name, raw)
        zpath.unlink()
    if not (raw / "uncharacterized.txt").exists():
        fetch(UNCHARACTERIZED, raw / "uncharacterized.txt")

    df = pd.read_csv(raw / "gdb9.sdf.csv")
    assert len(df) == N_MOLECULES, len(df)
    n_train, n_test = 100000, int(0.1 * len(df))
    n_val = len(df) - (n_train + n_test)
    shuffled = df.sample(frac=1, random_state=42)  # iloc, not np.split (which returns arrays on pandas >= 2)
    train, val, test = shuffled.iloc[:n_train], shuffled.iloc[n_train:n_train + n_val], shuffled.iloc[n_train + n_val:]
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(raw / f"{name}.csv")
    print(f"staged {raw}: train {len(train)}, val {len(val)}, test {len(test)}")


if __name__ == "__main__":
    main()
