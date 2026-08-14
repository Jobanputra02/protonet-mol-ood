"""
Per-Task Scaffold Diversity - FS-Mol All Splits
================================================
ONE QUESTION: how scaffold-diverse is each task (assay)? This sets how much
scaffold variety a support set can actually span, which bounds how meaningful a
scaffold-split episode is.

For each assay, computes:
  - n_unique_scaffolds       : distinct Bemis-Murcko scaffolds
  - scaffold_diversity_ratio : n_unique_scaffolds / n_molecules

Runs on test + valid (full scan) and train (sampled, set TRAIN_SAMPLE below).
Only exact-measurement molecules (Relation == "=") are counted.

Outputs saved to FIGURES_DIR / RESULTS_DIR (from config.py):
    scaffold_diversity_per_task_all_splits.csv    - per-assay stats (all numbers; summary printed to stdout)
    fig_scaffold_distributions.png                - 4-row × 3-col histograms (molecules, scaffolds, diversity ratio, mols/scaffold); x-axes clipped at 99th pct for count metrics

Usage:
    python Analysis/data/scaffold_diversity.py
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
from config import FSMOL_DIR, DATA_ANALYSIS_FIGURES_DIR, DATA_ANALYSIS_RESULTS_DIR

FIGURES_DIR = DATA_ANALYSIS_FIGURES_DIR
RESULTS_DIR = DATA_ANALYSIS_RESULTS_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== CONFIG - edit these, then run (no arguments) =====
MIN_TASK_SIZE = 32     # assays with fewer exact molecules are skipped
TRAIN_SAMPLE  = None   # train files to sample (None = all ~26k, very slow)
# ========================================================


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

        scaffold_counter: dict[str, int] = {}
        for smi in smiles_list:
            mol_obj = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
            if mol_obj is None:
                continue
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(  # type: ignore[attr-defined]
                    mol=mol_obj, includeChirality=False
                )
                scaffold_counter[sc] = scaffold_counter.get(sc, 0) + 1
            except Exception:
                pass

        n_mols       = len(smiles_list)
        n_sc         = len(scaffold_counter)
        group_sizes  = list(scaffold_counter.values())
        rows.append({
            "assay_id":                fname.replace(".jsonl.gz", ""),
            "split":                   split_name,
            "n_molecules":             n_mols,
            "n_unique_scaffolds":      n_sc,
            "scaffold_diversity_ratio": n_sc / n_mols if n_mols > 0 else 0.0,
            "mean_mols_per_scaffold":  np.mean(group_sizes) if group_sizes else 0.0,
            "median_mols_per_scaffold": np.median(group_sizes) if group_sizes else 0.0,
            "max_mols_per_scaffold":   max(group_sizes) if group_sizes else 0,
        })

    print(f"\n  [{split_name}] Done. {len(rows)} assays processed.")
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, label: str) -> None:
    dm  = df["n_molecules"].describe()
    ds  = df["n_unique_scaffolds"].describe()
    dr  = df["scaffold_diversity_ratio"].describe()
    dg  = df["mean_mols_per_scaffold"].describe()
    print(f"\n  === {label} ===")
    print(f"  Assays                  : {len(df):,}")
    print(f"  Total molecules         : {df['n_molecules'].sum():,}")
    print(f"  Total unique scaffolds  : {df['n_unique_scaffolds'].sum():,}  (sum across assays; scaffolds shared across assays counted per assay)")
    print(f"  Molecules / assay       : mean={dm['mean']:.1f}, median={dm['50%']:.0f}, min={dm['min']:.0f}, max={dm['max']:.0f}")
    print(f"  Unique scaffolds / assay: mean={ds['mean']:.1f}, median={ds['50%']:.0f}, min={ds['min']:.0f}, max={ds['max']:.0f}")
    print(f"  Diversity ratio         : mean={dr['mean']:.3f}, median={dr['50%']:.3f}")
    print(f"  Molecules / scaffold    : mean={dg['mean']:.1f}, median={df['median_mols_per_scaffold'].median():.1f}, max(per-assay max)={df['max_mols_per_scaffold'].max():.0f}")
    print(f"  Tasks with 1 scaffold   : {(df['n_unique_scaffolds']==1).mean()*100:.1f}%")
    print(f"  Tasks diversity > 0.5   : {(df['scaffold_diversity_ratio']>0.5).mean()*100:.1f}%")


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

    split_colors   = {"train": "steelblue", "valid": "orange", "test": "tomato"}
    splits_present = [s for s in ("train", "valid", "test") if s in combined["split"].values]

    # ── Figure 1: distribution histograms (4 rows × 3 cols) ──────────────────
    # x-axes clipped at 99th percentile for count metrics to avoid outlier squish.
    CLIP_PCT = 99
    metrics_hist = [
        ("n_molecules",              "Molecules / assay",          True),
        ("n_unique_scaffolds",       "Unique scaffolds / assay",   True),
        ("scaffold_diversity_ratio", "Diversity ratio (scaffolds / molecules)", False),
        ("mean_mols_per_scaffold",   "Mean molecules / scaffold",  True),
    ]

    n_rows = len(metrics_hist)
    n_cols = len(splits_present)
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows))
    if n_cols == 1:
        axes1 = axes1.reshape(n_rows, 1)

    for row_i, (col, title, do_clip) in enumerate(metrics_hist):
        for col_i, split in enumerate(splits_present):
            sub   = combined[combined["split"] == split]
            vals  = sub[col].dropna()
            color = split_colors[split]
            ax    = axes1[row_i, col_i]

            if do_clip:
                xmax   = np.percentile(vals, CLIP_PCT)
                n_clip = int((vals > xmax).sum())
                pvals  = vals[vals <= xmax]
            else:
                n_clip = 0
                pvals  = vals

            ax.hist(pvals, bins=40, color=color, edgecolor="none", alpha=0.85)
            med = vals.median()
            fmt = f"{med:.3f}" if col == "scaffold_diversity_ratio" else f"{med:.0f}"
            ax.axvline(med, color="black", linestyle="--", linewidth=1.2, label=f"median = {fmt}")
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            ax.set_axisbelow(True)
            ax.set_ylabel("Tasks", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

            if row_i == 0:
                sp_label = split if (split != "train" or TRAIN_SAMPLE is None) \
                           else f"train (n={TRAIN_SAMPLE})"
                ax.set_title(f"FS-Mol {sp_label}\n({len(sub):,} assays)", fontsize=11)

            xlabel = title
            if n_clip > 0:
                xlabel += f"\n(99th-pct clip; {n_clip} outlier{'s' if n_clip > 1 else ''} hidden)"
            ax.set_xlabel(xlabel, fontsize=9)

    fig1.suptitle("Per-Task Scaffold Diversity - FS-Mol", fontsize=13)
    fig1.tight_layout()
    out_fig1 = os.path.join(FIGURES_DIR, "fig_scaffold_distributions.png")
    fig1.savefig(out_fig1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Figure saved → {out_fig1}")
