"""
Chemical Diversity — FS-Mol vs DrugOOD
========================================
Three complementary views of how different the two datasets are chemically:

  Section 1 — Molecular properties (mass, heavy atoms, rotatable bonds, aromatic rings)
               Side-by-side comparison: FS-Mol train/test vs DrugOOD train/ood_test
               Uses structural_variability.py for per-molecule feature computation.

  Section 2 — Tanimoto distance distributions
               Internal FS-Mol, internal DrugOOD, and cross-dataset pairwise distances.
               Higher cross-distance = more OOD shift between the two corpora.

  Section 3 — t-SNE visualization (ECFP4, Tanimoto/Jaccard metric)
               5000 molecules per dataset, combined embedding coloured by source.

Outputs saved to FIGURES_DIR / RESULTS_DIR (from config.py):
    structural_var_comparison.csv      — mean molecular properties per dataset
    tanimoto_distances.png
    tsne_fsmol_vs_drugood.png

Usage:
    python analysis/data/chemical_diversity.py
"""

import gzip
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from rdkit import Chem                                      # type: ignore
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FSMOL_DIR, DRUGOOD_DIR, FIGURES_DIR, RESULTS_DIR
from structural_variability import compute_structural_variability, summarize_variability

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)  # type: ignore

DRUGOOD_SCAFFOLD_JSON = os.path.join(DRUGOOD_DIR, "lbap_core_ic50_scaffold.json")


# =============================================================================
# SMILES LOADERS
# =============================================================================

def load_fsmol_smiles(split_dir: str) -> list[str]:
    """Load all SMILES from a FS-Mol split directory (all molecules, no filter)."""
    smiles = []
    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith(".jsonl.gz"):
            continue
        with gzip.open(os.path.join(split_dir, fname), "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    mol = json.loads(line)
                    smi = mol.get("SMILES")
                    if smi:
                        smiles.append(smi)
    return smiles


def load_drugood_smiles(json_path: str, split: str) -> list[str]:
    """Load all SMILES from one DrugOOD split."""
    with open(json_path) as f:
        data = json.load(f)
    return [e["smiles"] for e in data["split"].get(split, []) if e.get("smiles")]


# =============================================================================
# FINGERPRINT UTILITIES
# =============================================================================

def smiles_to_fp(smi: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
    if mol is None:
        return None
    return _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32)  # type: ignore[attr-defined]


def compute_fps(smiles_list: list[str], label: str) -> np.ndarray:
    fps = []
    for i, smi in enumerate(smiles_list):
        if i % 5000 == 0:
            print(f"  [{label}] {i}/{len(smiles_list)}...", end="\r")
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
    print(f"  [{label}] {len(fps)} fingerprints computed.     ")
    return np.array(fps)


def tanimoto_distance_matrix(fps_a: np.ndarray, fps_b: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto distance (1 - similarity) between two fingerprint matrices."""
    intersection = fps_a @ fps_b.T
    sum_a = fps_a.sum(axis=1, keepdims=True)
    sum_b = fps_b.sum(axis=1, keepdims=True).T
    union = sum_a + sum_b - intersection
    similarity = np.where(union > 0, intersection / union, 0.0)
    return 1.0 - similarity


# =============================================================================
# SECTION 1: MOLECULAR PROPERTIES
# =============================================================================

def run_molecular_properties(
    fsmol_train: list[str],
    fsmol_test: list[str],
    drugood_train: list[str],
    drugood_ood: list[str],
) -> None:
    print("\n" + "=" * 60)
    print("SECTION 1: Molecular property comparison")
    print("=" * 60)

    datasets = {
        "fsmol_train":   fsmol_train,
        "fsmol_test":    fsmol_test,
        "drugood_train": drugood_train,
        "drugood_ood":   drugood_ood,
    }

    dfs: dict[str, pd.DataFrame] = {}
    for name, smiles_list in datasets.items():
        print(f"\n--- {name} ({len(smiles_list):,} molecules) ---")
        dfs[name] = compute_structural_variability(smiles_list)
        summarize_variability(dfs[name])

    # Side-by-side mean comparison
    numeric_cols = ["molecular_mass", "n_heavy_atoms", "n_rotatable_bonds", "n_aromatic_rings"]
    comparison = pd.DataFrame(
        {name: df[numeric_cols].mean() for name, df in dfs.items()}
    ).round(2)
    print("\n=== Side-by-side mean comparison ===")
    print(comparison)

    # Scaffold overlap
    fs_scaffolds  = set(dfs["fsmol_train"]["generic_scaffold"].dropna())
    do_scaffolds  = set(dfs["drugood_ood"]["generic_scaffold"].dropna())
    overlap       = fs_scaffolds & do_scaffolds
    print(f"\nGeneric scaffold overlap (FS-Mol train → DrugOOD ood_test):")
    print(f"  FS-Mol train  : {len(fs_scaffolds):,} unique")
    print(f"  DrugOOD ood   : {len(do_scaffolds):,} unique")
    print(f"  Shared        : {len(overlap):,}  ({100*len(overlap)/max(len(do_scaffolds),1):.1f}% of DrugOOD)")

    out_csv = os.path.join(RESULTS_DIR, "structural_var_comparison.csv")
    comparison.to_csv(out_csv)
    print(f"\nSaved → {out_csv}")


# =============================================================================
# SECTION 2: TANIMOTO DISTANCE DISTRIBUTIONS
# =============================================================================

def run_tanimoto(fsmol_fps: np.ndarray, drugood_fps: np.ndarray, rng: np.random.RandomState) -> None:
    print("\n" + "=" * 60)
    print("SECTION 2: Tanimoto distance distributions")
    print("=" * 60)

    N    = 10000
    half = min(N, len(fsmol_fps) // 2, len(drugood_fps) // 2)

    fs_idx = rng.choice(len(fsmol_fps),   size=min(2 * half, len(fsmol_fps)),   replace=False)
    do_idx = rng.choice(len(drugood_fps), size=min(2 * half, len(drugood_fps)), replace=False)
    fs_s   = fsmol_fps[fs_idx]
    do_s   = drugood_fps[do_idx]

    print("  Computing FS-Mol internal distances...")
    dist_fs = tanimoto_distance_matrix(fs_s[:half], fs_s[half:2*half])
    print("  Computing DrugOOD internal distances...")
    dist_do = tanimoto_distance_matrix(do_s[:half], do_s[half:2*half])
    print("  Computing cross-dataset distances...")
    dist_cross = tanimoto_distance_matrix(fs_s[:half], do_s[:half])

    print(f"\n  Mean Tanimoto distance:")
    print(f"    FS-Mol internal   : {dist_fs.mean():.4f}")
    print(f"    DrugOOD internal  : {dist_do.mean():.4f}")
    print(f"    FS-Mol vs DrugOOD : {dist_cross.mean():.4f}")
    print("  Higher cross-distance = more OOD shift between datasets")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dist_fs.flatten(),    bins=80, alpha=0.6, label="FS-Mol internal",   color="steelblue", density=True)
    ax.hist(dist_do.flatten(),    bins=80, alpha=0.6, label="DrugOOD internal",  color="orange",    density=True)
    ax.hist(dist_cross.flatten(), bins=80, alpha=0.6, label="FS-Mol vs DrugOOD", color="green",     density=True)
    for val, col in [(dist_fs.mean(), "steelblue"), (dist_do.mean(), "orange"), (dist_cross.mean(), "green")]:
        ax.axvline(val, color=col, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Tanimoto Distance (1 − similarity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pairwise Tanimoto Distance Distributions", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tanimoto_distances.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out}")


# =============================================================================
# SECTION 3: t-SNE
# =============================================================================

def run_tsne(fsmol_fps: np.ndarray, drugood_fps: np.ndarray, rng: np.random.RandomState) -> None:
    print("\n" + "=" * 60)
    print("SECTION 3: t-SNE (this takes a few minutes)")
    print("=" * 60)

    N_TSNE  = 5000
    fs_idx  = rng.choice(len(fsmol_fps),   size=min(N_TSNE, len(fsmol_fps)),   replace=False)
    do_idx  = rng.choice(len(drugood_fps), size=min(N_TSNE, len(drugood_fps)), replace=False)
    all_fps = np.vstack([fsmol_fps[fs_idx], drugood_fps[do_idx]])
    labels  = ["FS-Mol"] * len(fs_idx) + ["DrugOOD"] * len(do_idx)

    # metric="jaccard" = Tanimoto on binary vectors (correct for ECFP)
    tsne = TSNE(n_components=2, metric="jaccard", random_state=42,
                perplexity=50, max_iter=1000, verbose=1)
    coords = tsne.fit_transform(all_fps)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = {"FS-Mol": "steelblue", "DrugOOD": "tomato"}
    for dataset in ["FS-Mol", "DrugOOD"]:
        mask = np.array([l == dataset for l in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[dataset], alpha=0.4, s=6, label=dataset, rasterized=True)
    ax.set_title("t-SNE: FS-Mol Train vs DrugOOD ood_test\n(ECFP4, Tanimoto/Jaccard distance)",
                 fontsize=13)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.legend(markerscale=4, fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tsne_fsmol_vs_drugood.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    rng = np.random.RandomState(42)

    # ── Load SMILES ───────────────────────────────────────────────────────────
    print("Loading SMILES...")
    fsmol_train_smi  = load_fsmol_smiles(os.path.join(FSMOL_DIR, "train"))
    fsmol_test_smi   = load_fsmol_smiles(os.path.join(FSMOL_DIR, "test"))
    drugood_train_smi = load_drugood_smiles(DRUGOOD_SCAFFOLD_JSON, "train")
    drugood_ood_smi   = load_drugood_smiles(DRUGOOD_SCAFFOLD_JSON, "ood_test")
    print(f"  FS-Mol train: {len(fsmol_train_smi):,}  test: {len(fsmol_test_smi):,}")
    print(f"  DrugOOD train: {len(drugood_train_smi):,}  ood_test: {len(drugood_ood_smi):,}")

    # Section 1: molecular properties (uses SMILES directly, no fingerprints)
    run_molecular_properties(fsmol_train_smi, fsmol_test_smi, drugood_train_smi, drugood_ood_smi)

    # Sections 2+3 need fingerprints — compute once, share
    print("\nComputing ECFP4 fingerprints...")
    fsmol_fps   = compute_fps(fsmol_train_smi, "FS-Mol train")
    drugood_fps = compute_fps(drugood_ood_smi, "DrugOOD ood_test")

    run_tanimoto(fsmol_fps, drugood_fps, rng)
    run_tsne(fsmol_fps, drugood_fps, rng)

    print("\nDone.")
