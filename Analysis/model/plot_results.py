"""
Results Plotting — FS-Mol & DrugOOD
=====================================
Reads CSV files produced by the evaluation pipeline and generates:

  Figure 2(a): FS-Mol — mean ΔAUPRC vs support size, one line per split type
               (random / scaffold / size) with ±1 std error bars
  Figure 2(b): FS-Mol — per-assay ΔAUPRC and Spearman boxplots across support sizes
  Figure 3:    DrugOOD — Spearman and ΔAUPRC vs context size, faceted by shift type

Input CSVs (from outputs/results/ via config.py):
    fsmol_test_results.csv    — produced by evaluate_fsmol_test in main.py
    drugood_results.csv       — produced by evaluate_drugood_multiscale in main.py

Usage:
    python analysis/model/plot_results.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FIGURES_DIR, RESULTS_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

FSMOL_CSV   = os.path.join(RESULTS_DIR, "fsmol_test_results.csv")
DRUGOOD_CSV = os.path.join(RESULTS_DIR, "drugood_results.csv")


# =============================================================================
# FIGURE 2(a): FS-Mol three-curve line plot
# =============================================================================

def plot_fsmol_line(df: pd.DataFrame) -> None:
    """
    Three split types (random / scaffold / size) as separate lines.
    Three panels: ΔAUPRC, Spearman ρ, RMSE vs support size.
    Error bars = ±1 std across assays (shows task variability, not repeat noise).
    """
    split_colors = {"random": "steelblue", "scaffold": "tomato", "size": "seagreen"}
    split_types  = sorted(df["split_type"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, ylabel) in zip(axes, [
        ("delta_auprc", "Mean ΔAUPRC"),
        ("spearman",    "Mean Spearman ρ"),
        ("rmse",        "Mean RMSE"),
    ]):
        for stype in split_types:
            sub = df[df.split_type == stype]
            grouped = sub.groupby("support_size").agg(
                mean=(metric, "mean"),
                std=(metric,  "std"),
                n=("assay_id", "count"),
            ).reset_index()

            ax.errorbar(
                grouped["support_size"], grouped["mean"], yerr=grouped["std"],
                marker="o", linewidth=2, capsize=5, capthick=1.5,
                color=split_colors.get(stype, "gray"),
                ecolor=split_colors.get(stype, "gray"), elinewidth=1,
                label=stype,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Support set size", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"FS-Mol Test: {ylabel} vs Support Size", fontsize=12)
        if ax is axes[0]:
            ax.legend(fontsize=9)

    plt.suptitle("FS-Mol Test Evaluation — Prototypical Network (ECFP4, shift-aware)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig2a_fsmol_line_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

    print("\nMean ΔAUPRC by split_type × support_size:")
    print(df.groupby(["split_type", "support_size"])["delta_auprc"].mean().round(4))


# =============================================================================
# FIGURE 2(b): FS-Mol boxplot (per-assay distribution)
# =============================================================================

def plot_fsmol_boxplot(df: pd.DataFrame) -> None:
    """
    Boxplot showing distribution of per-assay ΔAUPRC and Spearman ρ across support sizes.
    One panel per metric; boxes coloured by split type.
    """
    split_types   = sorted(df["split_type"].unique())
    support_sizes = sorted(df["support_size"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    split_colors = {"random": "steelblue", "scaffold": "tomato", "size": "seagreen"}

    for ax, (metric, ylabel) in zip(axes, [
        ("delta_auprc", "ΔAUPRC per assay"),
        ("spearman",    "Spearman ρ per assay"),
    ]):
        positions = []
        all_data  = []
        all_colors = []
        n_types   = len(split_types)
        gap       = 0.8

        for si, size in enumerate(support_sizes):
            for ti, stype in enumerate(split_types):
                pos = si * (n_types + gap) + ti
                sub = df[(df.support_size == size) & (df.split_type == stype)]
                all_data.append(sub[metric].dropna().values)
                positions.append(pos)
                all_colors.append(split_colors.get(stype, "gray"))

        bp = ax.boxplot(
            all_data, positions=positions,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            widths=0.6,
        )
        for patch, color in zip(bp["boxes"], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Random baseline")

        # x-tick labels at group centres
        centres = [si * (n_types + gap) + (n_types - 1) / 2 for si in range(len(support_sizes))]
        ax.set_xticks(centres)
        ax.set_xticklabels([str(s) for s in support_sizes])
        ax.set_xlabel("Support set size", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"FS-Mol Test: Per-assay {ylabel}", fontsize=12)

        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=split_colors.get(s, "gray"), alpha=0.6, label=s)
                   for s in split_types]
        handles.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Random baseline"))
        ax.legend(handles=handles, fontsize=8)

    plt.suptitle("FS-Mol Test: Per-Assay Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig2b_fsmol_boxplot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# =============================================================================
# FIGURE 3: DrugOOD line plot
# =============================================================================

def plot_drugood_line(df: pd.DataFrame) -> None:
    """
    For each DrugOOD shift type: Spearman and ΔAUPRC vs context size.
    ood_test and iid_test as separate lines with error bars.
    """
    split_types = sorted(df["split_type"].unique())
    n_splits    = len(split_types)
    colors      = {"ood_test": "tomato", "iid_test": "steelblue"}
    markers     = {"ood_test": "o", "iid_test": "s"}

    fig, axes = plt.subplots(2, n_splits, figsize=(6 * n_splits, 10), squeeze=False)

    for col_i, split_type in enumerate(split_types):
        sub        = df[df.split_type == split_type]
        short_name = split_type.replace("lbap_core_ic50_", "IC50 ")

        for row_i, (metric, std_col, ylabel) in enumerate([
            ("spearman",    "spearman_std",    "Spearman ρ"),
            ("delta_auprc", "delta_auprc_std", "ΔAUPRC"),
        ]):
            ax = axes[row_i, col_i]
            for qset in ("ood_test", "iid_test"):
                qsub = sub[sub.query_set == qset].sort_values("context_set_size")
                if qsub.empty:
                    continue
                ax.errorbar(
                    qsub["context_set_size"], qsub[metric], yerr=qsub[std_col],
                    marker=markers[qset], linewidth=2, capsize=4, capthick=1.5,
                    color=colors[qset], ecolor=colors[qset], elinewidth=1,
                    label=qset.replace("_", " "),
                )
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_xlabel("Context set size", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{short_name} — {ylabel}", fontsize=11)
            ax.set_xticks(sorted(sub["context_set_size"].unique()))
            if col_i == 0:
                ax.legend(fontsize=8)

    plt.suptitle("DrugOOD Evaluation — Prototypical Network (ECFP4, shift-aware)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3_drugood_line_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(FSMOL_CSV):
        print(f"FS-Mol CSV not found: {FSMOL_CSV}")
        print("Run main.py first to generate fsmol_test_results.csv")
    else:
        print(f"Loading FS-Mol results: {FSMOL_CSV}")
        fsmol_df = pd.read_csv(FSMOL_CSV)
        print(f"  {len(fsmol_df)} rows, {fsmol_df['assay_id'].nunique()} assays, "
              f"split types: {sorted(fsmol_df['split_type'].unique())}, "
              f"support sizes: {sorted(fsmol_df['support_size'].unique())}")
        print("\nFigure 2(a): FS-Mol line plot...")
        plot_fsmol_line(fsmol_df)
        print("\nFigure 2(b): FS-Mol boxplot...")
        plot_fsmol_boxplot(fsmol_df)

    if not os.path.exists(DRUGOOD_CSV):
        print(f"\nDrugOOD CSV not found: {DRUGOOD_CSV} — skipping Figure 3.")
    else:
        print(f"\nLoading DrugOOD results: {DRUGOOD_CSV}")
        drugood_df = pd.read_csv(DRUGOOD_CSV)
        print(f"  {len(drugood_df)} rows, splits: {sorted(drugood_df['split_type'].unique())}")
        print("\nFigure 3: DrugOOD line plot...")
        plot_drugood_line(drugood_df)

    print("\nDone.")
