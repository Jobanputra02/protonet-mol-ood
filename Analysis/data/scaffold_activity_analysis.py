"""
Scaffold-Activity Degeneracy Analysis
======================================
For each FS-Mol training assay, answers two questions:

  1. Context/query assignment: split molecules by scaffold group and add a
     'role' column (context / query). This mirrors the scaffold-split evaluation
     protocol used during model testing.

  2. Scaffold-activity degeneracy: does the binary activity label (0/1) correlate
     with scaffold group membership within each assay? Measured with Cramér's V
     (chi-squared effect size). V=0 means scaffold does not predict activity;
     V=1 means scaffold perfectly determines activity.

Outputs:
    results/data_analysis/scaffold_context_query.csv   - molecule-level, role column
    results/data_analysis/scaffold_degeneracy.csv      - per-assay Cramér's V stats
    figures/data_analysis/fig_scaffold_activity_corr.png

Usage:
    python Analysis/data/scaffold_activity_analysis.py
"""

import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem                            # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold   # type: ignore
from scipy.stats import chi2_contingency          # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FSMOL_TRAIN, DATA_ANALYSIS_FIGURES_DIR, DATA_ANALYSIS_RESULTS_DIR

os.makedirs(DATA_ANALYSIS_FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_ANALYSIS_RESULTS_DIR, exist_ok=True)

MIN_TASK_SIZE = 32
N_SUPPORT     = 16     # molecules assigned to context
TRAIN_SAMPLE  = 2000   # training assays to sample (None = all ~16k, slow)
SEED          = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def _get_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "__none__"
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return sc if sc else "__none__"
    except Exception:
        return "__none__"


def load_assay(filepath: str) -> pd.DataFrame | None:
    """
    Load one assay. Returns DataFrame with columns:
        mol_idx  smiles  scaffold  binary_label
    Only exact-measurement molecules (Relation == '=') with valid 0/1 labels.
    Returns None if fewer than MIN_TASK_SIZE molecules survive.
    """
    rows = []
    with gzip.open(filepath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            mol = json.loads(line)
            if mol.get("Relation", "=") != "=":
                continue
            smi  = mol.get("SMILES", None)
            prop = mol.get("Property", None)
            if smi is None or prop is None:
                continue
            try:
                label = int(float(prop))
                if label not in (0, 1):
                    continue
            except (ValueError, TypeError):
                continue
            rows.append({"smiles": smi, "binary_label": label})

    if len(rows) < MIN_TASK_SIZE:
        return None

    df = pd.DataFrame(rows)
    df.insert(0, "mol_idx", range(len(df)))
    df["scaffold"] = df["smiles"].apply(_get_scaffold)
    return df


# =============================================================================
# CONTEXT / QUERY SPLIT
# =============================================================================

def assign_context_query(df: pd.DataFrame, n_support: int,
                         rng: np.random.RandomState) -> pd.DataFrame | None:
    """
    Scaffold-based split: shuffle scaffold groups, assign groups to context
    until context has >= n_support molecules, remaining groups → query.
    Returns df with 'role' column, or None if query has < 8 molecules.
    """
    groups = df.groupby("scaffold")["mol_idx"].apply(list).to_dict()
    keys   = list(groups.keys())
    rng.shuffle(keys)

    ctx: set[int] = set()
    qry: set[int] = set()
    for k in keys:
        if len(ctx) < n_support:
            ctx.update(groups[k])
        else:
            qry.update(groups[k])

    if len(qry) < 8:
        return None

    out = df.copy()
    out["role"] = out["mol_idx"].map(lambda i: "context" if i in ctx else "query")
    return out


# =============================================================================
# CRAMÉR'S V - scaffold vs activity label
# =============================================================================

def cramers_v(df: pd.DataFrame) -> float:
    """
    Chi-squared effect size between scaffold group and binary_label.
    V = sqrt(chi2 / (n * (min(rows, cols) - 1)))
    """
    ct = pd.crosstab(df["scaffold"], df["binary_label"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(ct)
    n = len(df)
    k = min(ct.shape) - 1
    if k == 0:
        return np.nan
    return float(np.clip(np.sqrt(chi2 / (n * k)), 0.0, 1.0))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    all_files = sorted(
        os.path.join(FSMOL_TRAIN, f)
        for f in os.listdir(FSMOL_TRAIN)
        if f.endswith(".jsonl.gz")
    )
    print(f"Training files found: {len(all_files)}")

    if TRAIN_SAMPLE and TRAIN_SAMPLE < len(all_files):
        rng0  = np.random.RandomState(SEED)
        idx   = rng0.choice(len(all_files), size=TRAIN_SAMPLE, replace=False)
        files = [all_files[i] for i in sorted(idx)]
        print(f"Sampled {len(files)} files")
    else:
        files = all_files

    per_assay_rows = []
    mol_rows       = []
    skipped        = 0

    for i, fpath in enumerate(files):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} ...", end="\r", flush=True)

        df = load_assay(fpath)
        if df is None:
            skipped += 1
            continue

        assay_id = os.path.basename(fpath).replace(".jsonl.gz", "")

        # --- 1. Context/query assignment ---
        rng    = np.random.RandomState(SEED + i)
        df_split = assign_context_query(df, N_SUPPORT, rng)

        if df_split is not None:
            samp = df_split[["mol_idx", "smiles", "scaffold",
                              "binary_label", "role"]].copy()
            samp.insert(0, "assay_id", assay_id)
            mol_rows.append(samp)

        # --- 2. Cramér's V (computed on full assay, not split) ---
        cv = cramers_v(df)

        per_assay_rows.append({
            "assay_id":           assay_id,
            "n_mols":             len(df),
            "n_scaffold_groups":  df["scaffold"].nunique(),
            "frac_active":        round(df["binary_label"].mean(), 4),
            "cramers_v":          round(cv, 4) if not np.isnan(cv) else np.nan,
            "split_ok":           df_split is not None,
        })

    print(f"\nDone. Analyzed: {len(per_assay_rows)}  Skipped (<{MIN_TASK_SIZE} mols): {skipped}")

    # --- Save CSVs ---
    per_assay_df = pd.DataFrame(per_assay_rows)
    deg_path = os.path.join(DATA_ANALYSIS_RESULTS_DIR, "scaffold_degeneracy.csv")
    per_assay_df.to_csv(deg_path, index=False)
    print(f"Saved → {deg_path}")

    if mol_rows:
        mol_df   = pd.concat(mol_rows, ignore_index=True)
        mol_path = os.path.join(DATA_ANALYSIS_RESULTS_DIR, "scaffold_context_query.csv")
        mol_df.to_csv(mol_path, index=False)
        print(f"Saved → {mol_path}  ({len(mol_df):,} molecule rows, "
              f"{mol_df['assay_id'].nunique()} assays)")

    # --- Summary ---
    cv = per_assay_df["cramers_v"].dropna()
    print(f"\n{'='*55}")
    print(f"Cramér's V  (scaffold → activity label correlation)")
    print(f"{'='*55}")
    print(f"  Assays with valid V : {len(cv)}")
    print(f"  mean   : {cv.mean():.4f}")
    print(f"  median : {cv.median():.4f}")
    print(f"  std    : {cv.std():.4f}")
    print(f"  V > 0.10 : {(cv > 0.10).sum():>4d}  ({(cv > 0.10).mean()*100:.1f}%)")
    print(f"  V > 0.20 : {(cv > 0.20).sum():>4d}  ({(cv > 0.20).mean()*100:.1f}%)")
    print(f"  V > 0.30 : {(cv > 0.30).sum():>4d}  ({(cv > 0.30).mean()*100:.1f}%)")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cv, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(cv.median(), color="tomato",  linestyle="--", linewidth=1.5,
               label=f"median = {cv.median():.3f}")
    ax.axvline(cv.mean(),   color="orange",  linestyle="--", linewidth=1.5,
               label=f"mean   = {cv.mean():.3f}")
    ax.set_xlabel("Cramér's V  (scaffold group → activity label)", fontsize=11)
    ax.set_ylabel("Number of assays", fontsize=11)
    ax.set_title(
        f"Scaffold-Activity Degeneracy - FS-Mol Training Tasks  (n={len(cv)} assays)\n"
        "V = 0: scaffold uninformative about activity   "
        "V = 1: scaffold perfectly predicts label",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig_path = os.path.join(DATA_ANALYSIS_FIGURES_DIR, "fig_scaffold_activity_corr.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {fig_path}")


if __name__ == "__main__":
    main()
