"""
Per-Task Scaffold Diversity — FS-Mol All Splits
================================================
For each assay (task), computes:
  - n_unique_scaffolds       : distinct Bemis-Murcko scaffolds
  - scaffold_diversity_ratio : n_unique_scaffolds / n_molecules

Runs on test + valid (full scan) and train (sampled, set TRAIN_SAMPLE below).
Only exact-measurement molecules (Relation == "=") are counted.

Outputs saved to FIGURES_DIR / RESULTS_DIR (from config.py):
    scaffold_diversity_per_task_all_splits.csv
    fig_scaffold_diversity_per_task.png

Usage:
    python analysis/data/scaffold_analysis.py
"""

import gzip
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem                          # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FSMOL_DIR, FIGURES_DIR, RESULTS_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_TASK_SIZE = 32
TRAIN_SAMPLE  = 2000   # train files to sample (None = all ~26k, very slow)


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def scaffold_diversity_for_split(
    split_dir: str,
    split_name: str,
    max_files: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each assay passing the size filter, compute unique Bemis-Murcko scaffold count
    and diversity ratio. Only exact-measurement molecules are counted.
    """
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".jsonl.gz"))

    if max_files is not None and len(files) > max_files:
        rng   = np.random.RandomState(seed)
        files = list(rng.choice(files, size=max_files, replace=False))
        print(f"  [{split_name}] Sampled {max_files} of {len(os.listdir(split_dir))} files.")

    rows = []
    for i, fname in enumerate(files):
        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            print(f"  [{split_name}] {i+1}/{len(files)}...", end="\r")

        smiles_list: list[str] = []
        with gzip.open(os.path.join(split_dir, fname), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mol = json.loads(line)
                if mol.get("Relation", "=") != "=":
                    continue
                smi = mol.get("SMILES")
                if smi:
                    smiles_list.append(smi)

        if len(smiles_list) < MIN_TASK_SIZE:
            continue

        scaffolds: set[str] = set()
        for smi in smiles_list:
            mol_obj = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
            if mol_obj is None:
                continue
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(  # type: ignore[attr-defined]
                    mol=mol_obj, includeChirality=False
                )
                scaffolds.add(sc)
            except Exception:
                pass

        n_mols = len(smiles_list)
        n_sc   = len(scaffolds)
        rows.append({
            "assay_id":                fname.replace(".jsonl.gz", ""),
            "split":                   split_name,
            "n_molecules":             n_mols,
            "n_unique_scaffolds":      n_sc,
            "scaffold_diversity_ratio": n_sc / n_mols if n_mols > 0 else 0.0,
        })

    print(f"\n  [{split_name}] Done. {len(rows)} assays processed.")
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, label: str) -> None:
    d  = df["n_unique_scaffolds"].describe()
    dr = df["scaffold_diversity_ratio"].describe()
    print(f"\n  === {label} — scaffold diversity ({len(df)} assays) ===")
    print(f"  n_unique_scaffolds : mean={d['mean']:.1f}, median={d['50%']:.0f}, "
          f"min={d['min']:.0f}, max={d['max']:.0f}, std={d['std']:.1f}")
    print(f"  diversity ratio    : mean={dr['mean']:.3f}, median={dr['50%']:.3f}")
    print(f"  Tasks with only 1 scaffold       : {(df['n_unique_scaffolds']==1).mean()*100:.1f}%")
    print(f"  Tasks with diversity ratio > 0.5 : {(df['scaffold_diversity_ratio']>0.5).mean()*100:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    dfs = []
    for split, max_f in [("test", None), ("valid", None), ("train", TRAIN_SAMPLE)]:
        split_dir = os.path.join(FSMOL_DIR, split)
        label     = split if max_f is None else f"{split} (sample n={max_f})"
        print(f"\nProcessing {label}...")
        df = scaffold_diversity_for_split(split_dir, split, max_files=max_f)
        dfs.append(df)
        print_summary(df, label)

    combined = pd.concat(dfs, ignore_index=True)
    out_csv  = os.path.join(RESULTS_DIR, "scaffold_diversity_per_task_all_splits.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Figure: 2-row × 3-col histogram grid ─────────────────────────────────
    split_colors   = {"train": "steelblue", "valid": "orange", "test": "tomato"}
    splits_present = [s for s in ("train", "valid", "test") if s in combined["split"].values]

    fig, axes = plt.subplots(2, len(splits_present), figsize=(6 * len(splits_present), 8))
    if len(splits_present) == 1:
        axes = axes.reshape(2, 1)

    for col_i, split in enumerate(splits_present):
        sub   = combined[combined.split == split]
        color = split_colors[split]
        label = split if TRAIN_SAMPLE is None or split != "train" \
                else f"train (sample n={TRAIN_SAMPLE})"

        ax = axes[0, col_i]
        ax.hist(sub["n_unique_scaffolds"], bins=50, color=color, edgecolor="none", alpha=0.85)
        med = sub["n_unique_scaffolds"].median()
        ax.axvline(med, color="black", linestyle="--", linewidth=1.2, label=f"median = {med:.0f}")
        ax.set_xlabel("Unique scaffolds per task", fontsize=10)
        ax.set_ylabel("Number of tasks", fontsize=10)
        ax.set_title(f"FS-Mol {label}\n({len(sub):,} assays)", fontsize=11)
        ax.legend(fontsize=8)

        ax = axes[1, col_i]
        ax.hist(sub["scaffold_diversity_ratio"], bins=40, color=color, edgecolor="none", alpha=0.85)
        med_r = sub["scaffold_diversity_ratio"].median()
        ax.axvline(med_r, color="black", linestyle="--", linewidth=1.2, label=f"median = {med_r:.3f}")
        ax.set_xlabel("Diversity ratio (unique scaffolds / n_molecules)", fontsize=10)
        ax.set_ylabel("Number of tasks", fontsize=10)
        ax.set_title(f"Diversity ratio — {label}", fontsize=11)
        ax.legend(fontsize=8)

    plt.suptitle("Per-Task Scaffold Diversity — FS-Mol", fontsize=13, y=1.01)
    plt.tight_layout()
    out_fig = os.path.join(FIGURES_DIR, "fig_scaffold_diversity_per_task.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {out_fig}")
