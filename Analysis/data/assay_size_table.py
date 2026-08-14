"""
assay_size_table.py — Print a table of how many assays qualify for each support size.

Usage:
    python Analysis/data/assay_size_table.py

For each split (train/valid/test), counts assays where molecule count >= 2*n_support
for n_support in {16, 32, 64, 128, 256, 512}.

Rule of thumb: an episode needs n_support support molecules + at least n_support query
molecules, so the assay must have >= 2*n_support total molecules.
"""
import gzip
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import FSMOL_TRAIN, FSMOL_VAL, FSMOL_TEST

THRESHOLDS = [16, 32, 64, 128, 256, 512]


def mol_counts(split_dir: str) -> np.ndarray:
    """Count molecules (lines) in each assay file."""
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".jsonl.gz"))
    counts = []
    for fname in files:
        n = 0
        with gzip.open(os.path.join(split_dir, fname), "rb") as fh:
            for _ in fh:
                n += 1
        counts.append(n)
    return np.array(counts)


def print_table(counts: np.ndarray, label: str) -> None:
    total = len(counts)
    print(f"\n{'=' * 60}")
    print(f"{label}  ({total} assays)")
    print(f"  mol counts:  median={np.median(counts):.0f}  mean={counts.mean():.1f}"
          f"  min={counts.min()}  max={counts.max()}")
    print(f"\n  {'n_support':<12} {'need (2n)':<12} {'qualifying':<14} {'%'}")
    print(f"  {'-'*50}")
    for n in THRESHOLDS:
        need = 2 * n
        q = (counts >= need).sum()
        print(f"  {n:<12} {need:<12} {q:<14} {q / total * 100:.1f}%")


if __name__ == "__main__":
    for split_dir, label in [
        (FSMOL_TRAIN, "TRAIN"),
        (FSMOL_VAL,   "VALID"),
        (FSMOL_TEST,  "TEST"),
    ]:
        print(f"\nReading {split_dir} ...", flush=True)
        counts = mol_counts(split_dir)
        print_table(counts, label)
    print()
