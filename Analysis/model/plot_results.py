"""
Results Plotting — FS-Mol & DrugOOD
=====================================
Reads CSV files produced by the evaluation pipeline and generates:

  Figure 2(a): FS-Mol — mean ΔAUPRC vs support size, one line per split type
               (random / scaffold / size) with ±1 std error bars
  Figure 2(b): FS-Mol — per-assay ΔAUPRC and Spearman boxplots across support sizes
  Figure 3:    DrugOOD — Spearman and ΔAUPRC vs context size, faceted by shift type

Input CSVs (from outputs/results/ via config.py):
    fsmol_test_results_{run_tag}.csv  — produced by main.py
    drugood_results_{run_tag}.csv     — produced by main.py

Usage:
    python analysis/model/plot_results.py --run_tag ecfp_regression_shift_aware
    python analysis/model/plot_results.py --encoder gnn --head classification --split shift_aware
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display required (works on HPC/server)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FIGURES_DIR, RESULTS_DIR

# ---------------------------------------------------------------------------
# Parse run tag from CLI
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Plot FS-Mol and DrugOOD results")
_parser.add_argument("--run_tag", type=str, default=None,
                     help="Full run tag, e.g. ecfp_regression_shift_aware")
_parser.add_argument("--encoder", type=str, default="ecfp", choices=["ecfp", "gnn"])
_parser.add_argument("--head",    type=str, default="regression",
                     choices=["regression", "classification"])
_parser.add_argument("--split",   type=str, default="shift_aware",
                     choices=["shift_aware", "random"])
_args = _parser.parse_args()

RUN_TAG = _args.run_tag or f"{_args.encoder}_{_args.head}_{_args.split}"
# Human-readable label for figure titles, e.g. "ECFP | regression | shift_aware"
_parts  = RUN_TAG.split("_", 2)
RUN_LABEL = f"{_parts[0].upper()} | {'_'.join(_parts[1:])}" if len(_parts) >= 2 else RUN_TAG

RUN_RESULTS_DIR = os.path.join(RESULTS_DIR, RUN_TAG)
RUN_FIGURES_DIR = os.path.join(FIGURES_DIR, RUN_TAG)
os.makedirs(RUN_FIGURES_DIR, exist_ok=True)

FSMOL_CSV   = os.path.join(RUN_RESULTS_DIR, "fsmol_test_results.csv")
DRUGOOD_CSV = os.path.join(RUN_RESULTS_DIR, "drugood_results.csv")


# =============================================================================
# FIGURE 2(a): FS-Mol three-curve line plot
# =============================================================================

def plot_fsmol_line(df: pd.DataFrame) -> None:
    """
    Three split types (random / scaffold / size) as separate lines.
    Panels: ΔAUPRC, Spearman ρ, RMSE vs support size.
    Panels with all-NaN values (e.g. classification has no Spearman/RMSE) are skipped.
    Error bars = ±1 std across assays (shows task variability, not repeat noise).
    """
    split_colors = {"random": "steelblue", "scaffold": "tomato", "size": "seagreen"}
    split_types  = sorted(df["split_type"].unique())

    all_metrics = [
        ("delta_auprc", "Mean ΔAUPRC"),
        ("spearman",    "Mean Spearman ρ"),
        ("rmse",        "Mean RMSE"),
    ]
    metrics = [(m, y) for m, y in all_metrics if df[m].notna().any()]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, metrics):
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

        # annotate N assays (from random split as representative) below each x-tick
        n_per_size = (df[df.split_type == split_types[0]]
                      .groupby("support_size")["assay_id"].nunique())
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        sizes = sorted(df["support_size"].unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s}\n(N={n_per_size.get(s, '?')})" for s in sizes],
                           fontsize=8)
        ax.set_xlabel("Support set size", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"FS-Mol Test: {ylabel} vs Support Size", fontsize=12)
        if ax is axes[0]:
            ax.legend(fontsize=9)


    plt.suptitle(f"FS-Mol Test Evaluation — Prototypical Network ({RUN_LABEL})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(RUN_FIGURES_DIR, "fig2a_fsmol_line_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")

    print("\nMean dAUPRC by split_type x support_size:")
    print(df.groupby(["split_type", "support_size"])["delta_auprc"].mean().round(4))


# =============================================================================
# FIGURE 2(b): FS-Mol boxplot (per-assay distribution)
# =============================================================================

def plot_fsmol_boxplot(df: pd.DataFrame) -> None:
    """
    Boxplot showing distribution of per-assay ΔAUPRC and Spearman ρ across support sizes.
    One panel per metric; panels with all-NaN values (e.g. classification) are skipped.
    """
    split_types   = sorted(df["split_type"].unique())
    support_sizes = sorted(df["support_size"].unique())
    split_colors  = {"random": "steelblue", "scaffold": "tomato", "size": "seagreen"}

    all_metrics = [
        ("delta_auprc", "ΔAUPRC per assay"),
        ("spearman",    "Spearman ρ per assay"),
    ]
    metrics = [(m, y) for m, y in all_metrics if df[m].notna().any()]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, metrics):
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
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

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

    plt.suptitle(f"FS-Mol Test: Per-Assay Distribution ({RUN_LABEL})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(RUN_FIGURES_DIR, "fig2b_fsmol_boxplot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# =============================================================================
# FIGURE 3: DrugOOD line plot
# =============================================================================

def plot_drugood_line(df: pd.DataFrame) -> None:
    """
    For each DrugOOD shift type: ΔAUPRC (and Spearman if available) vs context size.
    ood_test and iid_test as separate lines with error bars.
    Rows with all-NaN values (e.g. Spearman for classification) are skipped.
    """
    split_types = sorted(df["split_type"].unique())
    n_splits    = len(split_types)
    colors      = {"ood_test": "tomato", "iid_test": "steelblue"}
    markers     = {"ood_test": "o", "iid_test": "s"}

    all_row_metrics = [
        ("spearman",    "spearman_std",    "Spearman ρ"),
        ("delta_auprc", "delta_auprc_std", "ΔAUPRC"),
    ]
    row_metrics = [(m, s, y) for m, s, y in all_row_metrics if df[m].notna().any()]
    n_rows = len(row_metrics)

    fig, axes = plt.subplots(n_rows, n_splits, figsize=(6 * n_splits, 5 * n_rows), squeeze=False)

    for col_i, split_type in enumerate(split_types):
        sub        = df[df.split_type == split_type]
        short_name = split_type.replace("lbap_core_ic50_", "IC50 ")

        for row_i, (metric, std_col, ylabel) in enumerate(row_metrics):
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
            ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel("Context set size", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{short_name} — {ylabel}", fontsize=11)
            ax.set_xticks(sorted(sub["context_set_size"].unique()))
            if col_i == 0:
                ax.legend(fontsize=8)

    plt.suptitle(f"DrugOOD Evaluation — Prototypical Network ({RUN_LABEL})",
                 fontsize=13, y=1.02 if n_rows == 1 else 1.01)
    plt.tight_layout()
    out = os.path.join(RUN_FIGURES_DIR, "fig3_drugood_line_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(FSMOL_CSV):
        print(f"FS-Mol CSV not found: {FSMOL_CSV}")
        print(f"Run main.py with matching config first, or pass --run_tag {RUN_TAG}")
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
