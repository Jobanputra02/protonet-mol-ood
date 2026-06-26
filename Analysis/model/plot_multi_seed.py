"""
Multi-Seed Figure Generation + Model Comparison
================================================
Loads seed0/1/2 result CSVs for one or all classification models, averages
per (assay_id, support_size, split_type), and generates publication-quality figures:

  Per-model figures (same format as plot_results.py):
    fig2a_fsmol_line_plot.png  — ΔAUPRC vs support size, 3 split curves
    fig2b_fsmol_boxplot.png    — per-assay distribution across support sizes
    fig3_drugood_line_plot.png — ΔAUPRC vs context size, faceted by shift type

  Comparison figures (all models overlaid):
    fig_comparison_random.png   — all models on random split
    fig_comparison_scaffold.png — all models on scaffold split (thesis main finding)
    fig_comparison_drugood.png  — all models on DrugOOD assay OOD

Usage:
    # single model
    python Analysis/model/plot_multi_seed.py --model fsmol_gnn_classification_random

    # all models + comparison figures
    python Analysis/model/plot_multi_seed.py --all

    # only comparison figures (assumes per-model CSVs already merged)
    python Analysis/model/plot_multi_seed.py --comparison-only
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FIGURES_DIR, RESULTS_DIR, DATA_ANALYSIS_FIGURES_DIR, DATA_ANALYSIS_RESULTS_DIR

os.makedirs(DATA_ANALYSIS_FIGURES_DIR, exist_ok=True)

# =============================================================================
# Models
# =============================================================================

ALL_MODELS = [
    "ecfp_classification_random",
    "ecfp_classification_shift_aware",
    "fsmol_gnn_classification_random",
    "fsmol_gnn_classification_shift_aware",
]

MODEL_LABELS = {
    "ecfp_classification_random":          "ECFP ProtoNet (random)",
    "ecfp_classification_shift_aware":     "ECFP ProtoNet (shift-aware)",
    "fsmol_gnn_classification_random":     "GNN ProtoNet (random)",
    "fsmol_gnn_classification_shift_aware":"GNN ProtoNet (shift-aware)",
}

MODEL_COLORS = {
    "ecfp_classification_random":          "tomato",
    "ecfp_classification_shift_aware":     "salmon",
    "fsmol_gnn_classification_random":     "steelblue",
    "fsmol_gnn_classification_shift_aware":"royalblue",
}

MODEL_MARKERS = {
    "ecfp_classification_random":          "^",
    "ecfp_classification_shift_aware":     "v",
    "fsmol_gnn_classification_random":     "o",
    "fsmol_gnn_classification_shift_aware":"s",
}

MODEL_LINESTYLES = {
    "ecfp_classification_random":          "--",
    "ecfp_classification_shift_aware":     "-.",
    "fsmol_gnn_classification_random":     "-",
    "fsmol_gnn_classification_shift_aware":":",
}


# =============================================================================
# Data loading — merge seeds
# =============================================================================

def load_fsmol(model: str) -> pd.DataFrame:
    """Load and seed-average fsmol_test_results.csv for a model."""
    dfs = []
    for seed in range(3):
        path = os.path.join(RESULTS_DIR, f"{model}_seed{seed}", "fsmol_test_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["seed"] = seed
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No seed CSVs found for {model}")

    combined = pd.concat(dfs, ignore_index=True)
    # Average per-assay across seeds, then return per-assay rows
    avg = (combined
           .groupby(["assay_id", "split_type", "support_size"])
           [["delta_auprc", "spearman", "rmse"]]
           .mean()
           .reset_index())
    print(f"  {model}: {len(dfs)} seeds, {avg['assay_id'].nunique()} assays")
    return avg


def load_drugood(model: str) -> pd.DataFrame:
    """Load and seed-average drugood_results.csv for a model."""
    dfs = []
    for seed in range(3):
        path = os.path.join(RESULTS_DIR, f"{model}_seed{seed}", "drugood_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["seed"] = seed
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    avg = (combined
           .groupby(["split_type", "query_set", "context_set_size"])
           [["delta_auprc", "delta_auprc_std", "spearman", "spearman_std"]]
           .mean()
           .reset_index())
    return avg


# =============================================================================
# Per-model figures (Fig 2a, 2b, 3)
# =============================================================================

def plot_fsmol_line(df: pd.DataFrame, run_label: str, out_dir: str) -> None:
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

        n_per_size = (df[df.split_type == split_types[0]]
                      .groupby("support_size")["assay_id"].nunique())
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        sizes = sorted(df["support_size"].unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s}\n(N={n_per_size.get(s, '?')})" for s in sizes], fontsize=8)
        ax.set_xlabel("Support set size", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"FS-Mol Test: {ylabel} vs Support Size", fontsize=12)
        if ax is axes[0]:
            ax.legend(fontsize=9)

    plt.suptitle(f"FS-Mol Test Evaluation — Prototypical Network ({run_label})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig2a_fsmol_line_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_fsmol_boxplot(df: pd.DataFrame, run_label: str, out_dir: str) -> None:
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
        positions, all_data, all_colors = [], [], []
        n_types = len(split_types)
        gap = 0.8

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
        centres = [si * (n_types + gap) + (n_types - 1) / 2 for si in range(len(support_sizes))]
        ax.set_xticks(centres)
        ax.set_xticklabels([str(s) for s in support_sizes])
        ax.set_xlabel("Support set size", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"FS-Mol Test: Per-assay {ylabel}", fontsize=12)

        handles = [mpatches.Patch(facecolor=split_colors.get(s, "gray"), alpha=0.6, label=s)
                   for s in split_types]
        handles.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Random baseline"))
        ax.legend(handles=handles, fontsize=8)

    plt.suptitle(f"FS-Mol Test: Per-Assay Distribution ({run_label})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig2b_fsmol_boxplot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_drugood_line(df: pd.DataFrame, run_label: str, out_dir: str) -> None:
    if df.empty:
        print("  No DrugOOD data — skipping fig3")
        return

    split_types = sorted(df["split_type"].unique())
    n_splits    = len(split_types)
    colors  = {"ood_test": "tomato", "iid_test": "steelblue"}
    markers = {"ood_test": "o", "iid_test": "s"}

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

    plt.suptitle(f"DrugOOD Evaluation — Prototypical Network ({run_label})",
                 fontsize=13, y=1.02 if n_rows == 1 else 1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig3_drugood_line_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# =============================================================================
# Generate all per-model figures
# =============================================================================

def generate_model_figures(model: str) -> pd.DataFrame:
    """Generate fig2a/2b/3 for one model. Returns fsmol df for comparison use."""
    print(f"\n[{model}]")
    fsmol_df = load_fsmol(model)
    drugood_df = load_drugood(model)

    out_dir = os.path.join(FIGURES_DIR, model)
    os.makedirs(out_dir, exist_ok=True)

    label = MODEL_LABELS.get(model, model)
    plot_fsmol_line(fsmol_df, label, out_dir)
    plot_fsmol_boxplot(fsmol_df, label, out_dir)
    plot_drugood_line(drugood_df, label, out_dir)

    return fsmol_df


# =============================================================================
# Comparison figures (all models overlaid)
# =============================================================================

def plot_comparison_fsmol(model_dfs: dict[str, pd.DataFrame], split_type: str) -> None:
    """All models on one split type — one figure per split."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model, df in model_dfs.items():
        sub = df[df["split_type"] == split_type]
        if sub.empty:
            continue
        agg = sub.groupby("support_size")["delta_auprc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["support_size"], agg["mean"], yerr=agg["std"],
            marker=MODEL_MARKERS[model], linewidth=2,
            linestyle=MODEL_LINESTYLES[model],
            capsize=4, capthick=1.5,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model], elinewidth=1,
            label=MODEL_LABELS[model],
        )

    # RF baseline (from CSV if present, else skip)
    rf_csv = os.path.join(DATA_ANALYSIS_RESULTS_DIR, "rf_baseline_results.csv")
    if os.path.exists(rf_csv):
        rf = pd.read_csv(rf_csv)
        rf_sub = rf[rf["split_type"] == split_type]
        agg_rf = rf_sub.groupby("support_size")["delta_auprc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg_rf["support_size"], agg_rf["mean"], yerr=agg_rf["std"],
            marker="D", linewidth=2, linestyle=":",
            capsize=4, capthick=1.5,
            color="dimgray", ecolor="dimgray", elinewidth=1,
            label="RF baseline (non-episodic)",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Use assay counts from one of the models
    ref_df = next(iter(model_dfs.values()))
    ref_sub = ref_df[ref_df["split_type"] == split_type]
    n_per_size = ref_sub.groupby("support_size")["assay_id"].nunique()
    sizes = sorted(ref_sub["support_size"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}\n(N={n_per_size.get(s, '?')})" for s in sizes], fontsize=8)

    ax.set_xlabel("Support set size", fontsize=12)
    ax.set_ylabel("Mean ΔAUPRC (3-seed avg ± std)", fontsize=12)
    stype_label = {"random": "Random", "scaffold": "Scaffold OOD", "size": "Size OOD"}.get(split_type, split_type)
    ax.set_title(f"FS-Mol Test — {stype_label} Split: All Models", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out = os.path.join(DATA_ANALYSIS_FIGURES_DIR, f"fig_comparison_{split_type}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_comparison_drugood(model_dfs_drugood: dict[str, pd.DataFrame]) -> None:
    """Assay OOD ΔAUPRC vs context size for all models on one figure."""
    shift = "lbap_core_ic50_assay"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, df in model_dfs_drugood.items():
        if df.empty:
            continue
        sub = df[(df["split_type"] == shift) & (df["query_set"] == "ood_test")]
        if sub.empty:
            continue
        sub = sub.sort_values("context_set_size")
        ax.errorbar(
            sub["context_set_size"], sub["delta_auprc"], yerr=sub["delta_auprc_std"],
            marker=MODEL_MARKERS[model], linewidth=2,
            linestyle=MODEL_LINESTYLES[model],
            capsize=4, capthick=1.5,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model], elinewidth=1,
            label=MODEL_LABELS[model],
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("Context set size", fontsize=12)
    ax.set_ylabel("Mean ΔAUPRC (ood_test)", fontsize=12)
    ax.set_title("DrugOOD — Assay OOD: All Models", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out = os.path.join(DATA_ANALYSIS_FIGURES_DIR, "fig_comparison_drugood_assay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_comparison_scaffold_vs_random(model_dfs: dict[str, pd.DataFrame]) -> None:
    """Side-by-side: random split vs scaffold split for all models. Thesis main figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, split_type in zip(axes, ["random", "scaffold"]):
        for model, df in model_dfs.items():
            sub = df[df["split_type"] == split_type]
            if sub.empty:
                continue
            agg = sub.groupby("support_size")["delta_auprc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["support_size"], agg["mean"], yerr=agg["std"],
                marker=MODEL_MARKERS[model], linewidth=2,
                linestyle=MODEL_LINESTYLES[model],
                capsize=4, capthick=1.5,
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model], elinewidth=1,
                label=MODEL_LABELS[model],
            )

        # RF baseline
        rf_csv = os.path.join(DATA_ANALYSIS_RESULTS_DIR, "rf_baseline_results.csv")
        if os.path.exists(rf_csv):
            rf = pd.read_csv(rf_csv)
            rf_sub = rf[rf["split_type"] == split_type]
            agg_rf = rf_sub.groupby("support_size")["delta_auprc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg_rf["support_size"], agg_rf["mean"], yerr=agg_rf["std"],
                marker="D", linewidth=2, linestyle=":",
                capsize=4, capthick=1.5,
                color="dimgray", ecolor="dimgray", elinewidth=1,
                label="RF baseline",
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        ref_sub = next(iter(model_dfs.values()))
        ref_sub = ref_sub[ref_sub["split_type"] == split_type]
        n_per_size = ref_sub.groupby("support_size")["assay_id"].nunique()
        sizes = sorted(ref_sub["support_size"].unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s}\n(N={n_per_size.get(s,'?')})" for s in sizes], fontsize=8)
        ax.set_xlabel("Support set size", fontsize=12)
        ax.set_ylabel("Mean ΔAUPRC", fontsize=12)
        stype_label = "Random split" if split_type == "random" else "Scaffold OOD split"
        ax.set_title(stype_label, fontsize=13)
        if split_type == "random":
            ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("FS-Mol Test — All ProtoNet Models: Random vs Scaffold OOD (3-seed avg)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(DATA_ANALYSIS_FIGURES_DIR, "fig_comparison_random_vs_scaffold.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to regenerate figures for")
    parser.add_argument("--all", action="store_true",
                        help="Regenerate figures for all models + comparison")
    parser.add_argument("--comparison-only", action="store_true",
                        help="Generate only comparison figures (loads existing seed CSVs)")
    args = parser.parse_args()

    if args.model:
        generate_model_figures(args.model)
        return

    if not args.all and not args.comparison_only:
        parser.print_help()
        return

    # Load all models
    model_dfs = {}
    model_dfs_drugood = {}
    for model in ALL_MODELS:
        try:
            if not args.comparison_only:
                df = generate_model_figures(model)
            else:
                print(f"\nLoading {model} ...")
                df = load_fsmol(model)
            model_dfs[model] = df
            model_dfs_drugood[model] = load_drugood(model)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    print("\n[Comparison figures]")
    plot_comparison_fsmol(model_dfs, "random")
    plot_comparison_fsmol(model_dfs, "scaffold")
    plot_comparison_scaffold_vs_random(model_dfs)
    plot_comparison_drugood(model_dfs_drugood)
    print("\nAll done.")


if __name__ == "__main__":
    main()
