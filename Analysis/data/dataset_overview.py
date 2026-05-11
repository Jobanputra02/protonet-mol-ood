"""
Dataset Overview — FS-Mol & DrugOOD
=====================================
Single-pass scan of all assay files produces a unified stats table that drives:

  Data loss report:
    - Molecules dropped (inexact Relation, bad fields, task too small)
    - Correlation: assay size vs fraction of exact measurements

  Dataset statistics (Fig 1 equivalents):
    - Fig 1(a): compounds per assay, sorted by size (FS-Mol + DrugOOD)
    - Fig 1(b): fraction of active compounds per assay
    - Fig 1(c): DrugOOD domain group sizes

Outputs saved to FIGURES_DIR / RESULTS_DIR (from config.py):
    data_loss_per_assay.csv
    assay_sizes_fsmol.csv
    assay_sizes_drugood_by_assay.csv
    assay_sizes_drugood_by_domain.csv
    fig1a_assay_sizes.png
    fig1b_fraction_actives.png
    fig1c_domain_sizes.png
    fig_size_vs_fraction_exact.png

Usage:
    python analysis/data/dataset_overview.py

WARNING: scanning the train split (~26k files) takes ~10-20 minutes.
"""

import gzip
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FSMOL_DIR, DRUGOOD_DIR, FIGURES_DIR, RESULTS_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_TASK_SIZE = 32
DRUGOOD_FILES = {
    "scaffold": "lbap_core_ic50_scaffold.json",
    "size":     "lbap_core_ic50_size.json",
    "assay":    "lbap_core_ic50_assay.json",
}


# =============================================================================
# UNIFIED SCANNER — one pass per file, all columns at once
# =============================================================================

def scan_fsmol_assay(filepath: str) -> dict:
    """
    Single-pass scan of one .jsonl.gz file.
    Returns all stats needed for both data-loss reporting and dataset statistics.
    """
    n_total = n_inexact = n_invalid = n_exact = n_actives = n_has_binary = 0

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mol = json.loads(line)
            n_total += 1

            if mol.get("Relation", "=") != "=":
                n_inexact += 1
                continue

            fp    = mol.get("fingerprints")
            label = mol.get("LogRegressionProperty",
                            mol.get("RegressionProperty", mol.get("Property")))
            smi   = mol.get("SMILES")

            if fp is None or label is None or smi is None or len(fp) != 2048:
                n_invalid += 1
                continue

            try:
                float(label)
            except (ValueError, TypeError):
                n_invalid += 1
                continue

            n_exact += 1
            _prop = mol.get("Property")
            if _prop is not None:
                n_has_binary += 1
                if int(float(_prop)) == 1:
                    n_actives += 1

    return {
        "n_total":         n_total,
        "n_inexact":       n_inexact,
        "n_invalid":       n_invalid,
        "n_exact":         n_exact,
        "fraction_exact":  n_exact / n_total if n_total > 0 else 0.0,
        "n_actives":       n_actives       if n_has_binary > 0 else None,
        "n_has_binary":    n_has_binary,
        "fraction_active": n_actives / n_has_binary if n_has_binary > 0 else None,
    }


def scan_fsmol_split(split_dir: str, split_name: str) -> pd.DataFrame:
    """Scan all assay files in a split directory. Returns one row per assay."""
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".jsonl.gz"))
    rows  = []

    for i, fname in enumerate(files):
        if (i + 1) % 1000 == 0 or (i + 1) == len(files):
            print(f"  [{split_name}] {i+1}/{len(files)} files...", end="\r")

        stats = scan_fsmol_assay(os.path.join(split_dir, fname))
        rows.append({
            "assay_id":       fname.replace(".jsonl.gz", ""),
            "split":          split_name,
            "passes_filter":  int(stats["n_exact"] >= MIN_TASK_SIZE),
            **stats,
        })

    print(f"\n  [{split_name}] Done. {len(rows)} files scanned.")
    return pd.DataFrame(rows)


# =============================================================================
# DRUGOOD LOADER
# =============================================================================

def _group_counts(entries: list, group_key: str) -> pd.DataFrame:
    groups = defaultdict(lambda: {"n": 0, "actives": 0, "has_cls": 0})
    for entry in entries:
        gid = entry.get(group_key, "unknown")
        groups[gid]["n"] += 1
        cls = entry.get("cls_label")
        if cls is not None:
            groups[gid]["has_cls"] += 1
            if int(cls) == 1:
                groups[gid]["actives"] += 1

    rows = []
    for gid, counts in groups.items():
        has_cls = counts["has_cls"]
        rows.append({
            group_key:         gid,
            "n_molecules":     counts["n"],
            "n_actives":       counts["actives"]           if has_cls > 0 else None,
            "fraction_active": counts["actives"] / has_cls if has_cls > 0 else None,
        })
    return pd.DataFrame(rows)


def load_drugood_stats(json_path: str, shift_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (assay_df, domain_df) grouped by assay_id and domain_id respectively."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assay_rows, domain_rows = [], []
    for split_name, entries in data["split"].items():
        if split_name not in ("train", "ood_test", "iid_test"):
            continue
        for df_rows, key in ((assay_rows, "assay_id"), (domain_rows, "domain_id")):
            grp = _group_counts(entries, key)
            grp["shift_type"]     = shift_type
            grp["internal_split"] = split_name
            df_rows.append(grp)

    assay_df  = pd.concat(assay_rows,  ignore_index=True) if assay_rows  else pd.DataFrame()
    domain_df = pd.concat(domain_rows, ignore_index=True) if domain_rows else pd.DataFrame()
    return assay_df, domain_df


# =============================================================================
# REPORTING
# =============================================================================

def print_data_loss_table(df: pd.DataFrame, split_name: str) -> None:
    total_mols    = int(df["n_total"].sum())
    total_inexact = int(df["n_inexact"].sum())
    total_invalid = int(df["n_invalid"].sum())
    kept_df       = df[df.passes_filter == 1]
    dropped_df    = df[df.passes_filter == 0]
    mols_in_kept    = int(kept_df["n_exact"].sum())
    mols_in_dropped = int(dropped_df["n_exact"].sum())
    total_lost      = total_inexact + total_invalid + mols_in_dropped

    print(f"\n{'='*60}")
    print(f"  Data-loss: {split_name}/")
    print(f"{'='*60}")
    print(f"  Molecules in raw files        : {total_mols:>10,}")
    print(f"  -- Dropped (inexact Relation) : {total_inexact:>10,}  "
          f"({100*total_inexact/max(total_mols,1):.1f}%)")
    print(f"  -- Dropped (bad/missing)      : {total_invalid:>10,}  "
          f"({100*total_invalid/max(total_mols,1):.1f}%)")
    print(f"  -- Dropped (task too small)   : {mols_in_dropped:>10,}  "
          f"({100*mols_in_dropped/max(total_mols,1):.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total lost                    : {total_lost:>10,}  "
          f"({100*total_lost/max(total_mols,1):.1f}%)")
    print(f"  Molecules used                : {mols_in_kept:>10,}  "
          f"({100*mols_in_kept/max(total_mols,1):.1f}%)")
    print(f"  Tasks in dir                  : {len(df):>10,}")
    print(f"  Tasks dropped (<{MIN_TASK_SIZE})           : {len(dropped_df):>10,}  "
          f"({100*len(dropped_df)/max(len(df),1):.1f}%)")
    print(f"  Tasks kept                    : {len(kept_df):>10,}  "
          f"({100*len(kept_df)/max(len(df),1):.1f}%)")
    print(f"{'='*60}")


def print_correlation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("CORRELATION: assay size vs fraction of exact measurements")
    print("=" * 60)
    for split in ("train", "valid", "test"):
        sub = df[df.split == split]
        if len(sub) < 3:
            continue
        r,   p_r   = stats.pearsonr(sub["n_total"], sub["fraction_exact"])
        rho, p_rho = stats.spearmanr(sub["n_total"], sub["fraction_exact"])
        print(f"\n  {split}/ ({len(sub):,} assays):")
        print(f"    Pearson  r = {r:+.4f}  (p = {p_r:.3e})")
        print(f"    Spearman ρ = {rho:+.4f}  (p = {p_rho:.3e})")
        fe = sub["fraction_exact"]
        print(f"    fraction_exact: mean={fe.mean():.3f}, std={fe.std():.3f}, "
              f"range=[{fe.min():.3f}, {fe.max():.3f}]")


# =============================================================================
# FIGURES
# =============================================================================

split_colors  = {"train": "steelblue", "valid": "orange",    "test": "tomato"}
do_colors     = {"scaffold": "steelblue", "size": "orange",  "assay": "tomato"}
do_int_colors = {"train": "steelblue", "ood_test": "tomato", "iid_test": "seagreen"}


def plot_fig1a(fsmol_df: pd.DataFrame, drugood_assay_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    offset = 0
    for split in ("train", "valid", "test"):
        sub = fsmol_df[fsmol_df.split == split].sort_values("n_exact")
        x   = np.arange(offset, offset + len(sub))
        ax.scatter(x, sub["n_exact"].values, s=2, alpha=0.5,
                   color=split_colors[split], rasterized=True,
                   label=f"{split} ({len(sub):,} assays)")
        offset += len(sub)
    ax.axhline(MIN_TASK_SIZE, color="gray", linestyle="--", linewidth=1,
               label=f"Min size = {MIN_TASK_SIZE}")
    ax.set_yscale("log")
    ax.set_xlabel("Assay index (sorted by size within split)", fontsize=11)
    ax.set_ylabel("Number of compounds (log scale)", fontsize=11)
    ax.set_title("FS-Mol: compounds per assay", fontsize=12)
    ax.legend(markerscale=4, fontsize=9)

    ax = axes[1]
    do_train = drugood_assay_df[drugood_assay_df.internal_split == "train"]
    offset   = 0
    for shift in ("scaffold", "size", "assay"):
        sub = do_train[do_train.shift_type == shift].sort_values("n_molecules")
        x   = np.arange(offset, offset + len(sub))
        ax.scatter(x, sub["n_molecules"].values, s=6, alpha=0.7,
                   color=do_colors[shift], rasterized=True,
                   label=f"IC50 {shift} ({len(sub):,} assays)")
        offset += len(sub)
    ax.set_yscale("log")
    ax.set_xlabel("Assay index (sorted by size within shift type)", fontsize=11)
    ax.set_ylabel("Number of compounds (log scale)", fontsize=11)
    ax.set_title("DrugOOD train: compounds per assay_id", fontsize=12)
    ax.legend(markerscale=2, fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig1a_assay_sizes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_fig1b(fsmol_df: pd.DataFrame, drugood_assay_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for split in ("train", "valid", "test"):
        sub = fsmol_df[(fsmol_df.split == split) & fsmol_df.fraction_active.notna()]
        if sub.empty:
            continue
        ax.hist(sub["fraction_active"], bins=30, alpha=0.6, density=True,
                color=split_colors[split], edgecolor="none",
                label=f"{split} (n={len(sub):,})")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="50% active")
    ax.set_xlabel("Fraction of active compounds per assay", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("FS-Mol: fraction active per assay", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    do_ood = drugood_assay_df[
        (drugood_assay_df.internal_split == "ood_test") &
        drugood_assay_df.fraction_active.notna()
    ]
    for shift in ("scaffold", "size", "assay"):
        sub = do_ood[do_ood.shift_type == shift]
        if sub.empty:
            continue
        ax.hist(sub["fraction_active"], bins=20, alpha=0.6, density=True,
                color=do_colors[shift], edgecolor="none",
                label=f"IC50 {shift} (n={len(sub):,})")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="50% active")
    ax.set_xlabel("Fraction of active compounds per assay_id", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("DrugOOD ood_test: fraction active per assay_id", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig1b_fraction_actives.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_fig1c(drugood_domain_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, shift in zip(axes, ("scaffold", "size", "assay")):
        sub = drugood_domain_df[drugood_domain_df.shift_type == shift]
        for int_split in ("train", "ood_test", "iid_test"):
            s = sub[sub.internal_split == int_split].sort_values("n_molecules", ascending=False)
            if s.empty:
                continue
            ax.bar(np.arange(len(s)), s["n_molecules"].values,
                   alpha=0.7, color=do_int_colors[int_split],
                   label=f"{int_split} ({len(s)} domains)")
        ax.set_xlabel("Domain index (sorted by size)", fontsize=10)
        ax.set_ylabel("Molecules per domain", fontsize=10)
        ax.set_title(f"DrugOOD IC50 {shift}: domain sizes", fontsize=11)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig1c_domain_sizes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_size_vs_fraction_exact(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, split in zip(axes, ("train", "valid", "test")):
        sub = df[df.split == split]
        ax.scatter(sub["n_total"], sub["fraction_exact"],
                   alpha=0.3, s=3, color=split_colors[split], rasterized=True)
        ax.set_xlabel("Total molecules per assay", fontsize=10)
        ax.set_ylabel("Fraction exact (n_exact / n_total)", fontsize=10)
        ax.set_title(f"FS-Mol {split}: size vs fraction exact\n({len(sub):,} assays)", fontsize=11)
        ax.set_xscale("log")
        if len(sub) >= 3:
            r,   _ = stats.pearsonr(sub["n_total"], sub["fraction_exact"])
            rho, _ = stats.spearmanr(sub["n_total"], sub["fraction_exact"])
            ax.text(0.05, 0.95, f"Pearson r = {r:+.3f}\nSpearman ρ = {rho:+.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_size_vs_fraction_exact.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ── Scan FS-Mol splits ────────────────────────────────────────────────────
    all_dfs = []
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(FSMOL_DIR, split)
        print(f"\nScanning {split}/  (train takes ~10-20 min)...")
        df = scan_fsmol_split(split_dir, split)
        print_data_loss_table(df, split)
        all_dfs.append(df)

    fsmol_df = pd.concat(all_dfs, ignore_index=True)

    # Keep only tasks that pass size filter for statistics plots
    fsmol_stats_df = fsmol_df[fsmol_df.passes_filter == 1].copy()

    print_correlation(fsmol_df)

    # ── Load DrugOOD ─────────────────────────────────────────────────────────
    print("\nLoading DrugOOD statistics...")
    assay_dfs, domain_dfs = [], []
    for shift_type, fname in DRUGOOD_FILES.items():
        fpath = os.path.join(DRUGOOD_DIR, fname)
        print(f"  {shift_type}", end=" ", flush=True)
        a_df, d_df = load_drugood_stats(fpath, shift_type)
        print(f"→ {a_df['assay_id'].nunique()} assays, {d_df['domain_id'].nunique()} domains")
        assay_dfs.append(a_df)
        domain_dfs.append(d_df)
    drugood_assay_df  = pd.concat(assay_dfs,  ignore_index=True)
    drugood_domain_df = pd.concat(domain_dfs, ignore_index=True)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    fsmol_df.to_csv(os.path.join(RESULTS_DIR, "data_loss_per_assay.csv"), index=False)
    fsmol_stats_df.to_csv(os.path.join(RESULTS_DIR, "assay_sizes_fsmol.csv"), index=False)
    drugood_assay_df.to_csv(os.path.join(RESULTS_DIR, "assay_sizes_drugood_by_assay.csv"), index=False)
    drugood_domain_df.to_csv(os.path.join(RESULTS_DIR, "assay_sizes_drugood_by_domain.csv"), index=False)
    print("\nCSVs saved to", RESULTS_DIR)

    # ── Summary tables ────────────────────────────────────────────────────────
    print("\n=== FS-Mol assay size summary (after filtering, ≥32 molecules) ===")
    print(fsmol_stats_df.groupby("split")["n_exact"].describe().round(1))

    print("\n=== DrugOOD per-assay_id size summary ===")
    print(drugood_assay_df.groupby(["shift_type","internal_split"])["n_molecules"].describe().round(1))

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_fig1a(fsmol_stats_df, drugood_assay_df)
    plot_fig1b(fsmol_stats_df, drugood_assay_df)
    plot_fig1c(drugood_domain_df)
    plot_size_vs_fraction_exact(fsmol_df)

    print("\nDone.")
