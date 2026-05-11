"""
Structural Variability Analysis
================================
For each unique compound, computes:
    - Generic Murcko scaffold
    - Molecular mass
    - Number of heavy (non-hydrogen) atoms
    - Number of rotatable bonds
    - Number of aromatic rings

Written as a generic, reusable script.
Input:  list of SMILES strings (from any source — FS-Mol, DrugOOD, etc.)
Output: pandas DataFrame, one row per unique compound

Usage:
    from structural_variability import compute_structural_variability
    df = compute_structural_variability(smiles_list)
    df.to_csv("structural_variability.csv", index=False)
"""

import pandas as pd
import numpy as np
from rdkit import Chem  # type: ignore
from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
from typing import Optional


# =============================================================================
# PER-MOLECULE FEATURE COMPUTATION
# =============================================================================

def compute_mol_features(smiles: str) -> Optional[dict]:
    """
    Compute structural features for a single molecule.

    Args:
        smiles: SMILES string

    Returns:
        dict of features, or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None

    # Generic Murcko scaffold — ring systems + linkers, no side chains
    # "Generic" means all atoms replaced by carbons, all bonds by single bonds
    # This groups molecules by topology rather than exact chemistry
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(  # type: ignore[attr-defined]
            mol=mol, includeChirality=False
        )
        generic_scaffold = MurckoScaffold.MakeScaffoldGeneric(  # type: ignore[attr-defined]
            Chem.MolFromSmiles(scaffold)  # type: ignore[attr-defined]
        )
        generic_scaffold_smi = Chem.MolToSmiles(generic_scaffold)  # type: ignore[attr-defined]
    except Exception:
        generic_scaffold_smi = None

    return {
        "smiles":              smiles,
        "generic_scaffold":    generic_scaffold_smi,
        "molecular_mass":      Descriptors.ExactMolWt(mol),
        "n_heavy_atoms":       mol.GetNumHeavyAtoms(),
        "n_rotatable_bonds":   rdMolDescriptors.CalcNumRotatableBonds(mol),  # type: ignore[attr-defined]
        "n_aromatic_rings":    rdMolDescriptors.CalcNumAromaticRings(mol),   # type: ignore[attr-defined]
    }


# =============================================================================
# BATCH COMPUTATION
# =============================================================================

def compute_structural_variability(
    smiles_list: list[str],
    deduplicate: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute structural features for a list of SMILES strings.

    Args:
        smiles_list:  List of SMILES strings
        deduplicate:  If True, compute only once per unique canonical SMILES
        verbose:      Print progress

    Returns:
        pd.DataFrame with columns:
            smiles, canonical_smiles, generic_scaffold,
            molecular_mass, n_heavy_atoms, n_rotatable_bonds, n_aromatic_rings
    """
    if deduplicate:
        # Canonicalize and deduplicate — different SMILES can represent same molecule
        canonical_map = {}
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)  # type: ignore[attr-defined]
                canonical_map[canonical] = smi  # keep one original per canonical
        unique_smiles = list(canonical_map.keys())
        if verbose:
            n_dupes = len(smiles_list) - len(unique_smiles)
            print(f"  Deduplication: {len(smiles_list)} → {len(unique_smiles)} unique "
                  f"({n_dupes} duplicates removed)")
    else:
        unique_smiles = smiles_list

    rows = []
    n_invalid = 0
    for i, smi in enumerate(unique_smiles):
        if verbose and i % 5000 == 0:
            print(f"  Processing {i+1}/{len(unique_smiles)}...", end="\r")
        features = compute_mol_features(smi)
        if features is None:
            n_invalid += 1
            continue
        rows.append(features)

    if verbose:
        print(f"  Done. {len(rows)} valid, {n_invalid} invalid SMILES.        ")

    df = pd.DataFrame(rows)
    return df


def compute_per_task_scaffold_diversity(
    assay_datasets,
    verbose: bool = True
) -> pd.DataFrame:
    """
    For each assay (task), compute the number of unique generic Murcko scaffolds
    and the diversity ratio (unique scaffolds / n_molecules).

    Args:
        assay_datasets: list of AssayDataset objects (from load_fsmol_split)
        verbose:        print progress

    Returns:
        DataFrame with columns:
            assay_id, n_molecules, n_unique_scaffolds, scaffold_diversity_ratio
        One row per task.
    """
    rows = []
    for i, dataset in enumerate(assay_datasets):
        if verbose and i % 1000 == 0:
            print(f"  Processing task {i+1}/{len(assay_datasets)}...", end="\r")

        # dataset.scaffolds stores raw SMILES (set in load_fsmol_assay via __new__ bypass)
        smiles_list = dataset.scaffolds
        n_mols = len(dataset)

        scaffolds = set()
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
            if mol is None:
                continue
            try:
                sc_smi = MurckoScaffold.MurckoScaffoldSmiles(  # type: ignore[attr-defined]
                    mol=mol, includeChirality=False
                )
                generic = MurckoScaffold.MakeScaffoldGeneric(  # type: ignore[attr-defined]
                    Chem.MolFromSmiles(sc_smi)  # type: ignore[attr-defined]
                )
                scaffolds.add(Chem.MolToSmiles(generic))  # type: ignore[attr-defined]
            except Exception:
                continue

        rows.append({
            "assay_id":               dataset.assay_id,
            "n_molecules":            n_mols,
            "n_unique_scaffolds":     len(scaffolds),
            "scaffold_diversity_ratio": len(scaffolds) / max(n_mols, 1),
        })

    if verbose:
        print(f"\n  Done. Processed {len(rows)} tasks.")

    return pd.DataFrame(rows)


def summarize_per_task_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print summary statistics for per-task scaffold diversity and return the summary.
    Reports distribution of unique scaffold counts across tasks.
    """
    col = "n_unique_scaffolds"
    summary = df[col].describe().round(2)

    print(f"\n  Per-task scaffold diversity ({len(df)} tasks):")
    print(f"  {'Statistic':<12} {'Value':>10}")
    print(f"  {'-'*24}")
    for stat, val in summary.items():
        print(f"  {stat:<12} {val:>10.2f}")

    pct_single = (df[col] == 1).mean() * 100
    pct_high   = (df["scaffold_diversity_ratio"] > 0.5).mean() * 100
    print(f"\n  Tasks with only 1 unique scaffold : {pct_single:.1f}%")
    print(f"  Tasks with diversity ratio > 0.5  : {pct_high:.1f}%")
    return summary


def summarize_variability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return summary statistics for the structural variability DataFrame.
    Useful for comparing FS-Mol vs DrugOOD distributions.
    """
    numeric_cols = ["molecular_mass", "n_heavy_atoms", "n_rotatable_bonds", "n_aromatic_rings"]
    summary = df[numeric_cols].describe().round(2)

    n_unique_scaffolds = df["generic_scaffold"].nunique()
    print(f"\n  Unique generic scaffolds: {n_unique_scaffolds} / {len(df)} molecules "
          f"({100 * n_unique_scaffolds / len(df):.1f}%)")
    print(f"\n{summary}")
    return summary