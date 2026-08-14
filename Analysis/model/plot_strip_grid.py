"""
Configurable per-assay strip plot
===================================
Shows EVERY individual assay's ΔAUPRC — not the mean, not a box summary.
Each dot is one assay (averaged across seeds). Jittered horizontally within
each (support_size × head) group so overlapping points are visible.

Complements plot_boxplot_grid.py (boxes) and plot_line_grid.py (means).
Use this when you want to see outlier assays, task-level spread, and whether
a head's advantage is broad or driven by a few tasks.

Layout: one subplot per eval split (side by side).
Within each subplot: grouped by support size; within each size, one jittered
column per head. A horizontal line marks the median.

Color = head (same palette as the other plotting scripts).

Output: outputs/{run_tag}/figures/strip_{heads_tag}__{splits_tag}.png

Usage
-----
    python Analysis/model/plot_strip_grid.py
Edit CONFIG block below to change what is plotted.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import run_csv_dir, run_fig_dir, BASELINE_CSV_DIR

# =============================================================================
# CONFIG - edit these, then run
# =============================================================================
ENCODER        = "gnn"              # "ecfp" | "gnn"
TRAINING_SPLIT = "random"           # "random" | "scaffold" | "similarity"
TRAIN_DISTANCE = "euclidean"        # "euclidean" | "mahalanobis"
N_SUPPORT      = 64                 # int or list[int] used during training, e.g. [16, 32, 64]
SEEDS          = [0, 1, 2]

# Eval split is Butina@0.70 — fixed design choice (config.SCAFFOLD_OOD_CUTOFF).

ECFP_HEADS = [
    "ecfp_rf",
    "ecfp_logreg",
    "ecfp_proto_euclid",
    "ecfp_proto_tanimoto",
    "ecfp_gp_tanimoto",
    ]
EMB_HEADS  = [
    # "emb_proto_mahalanobis",
    # "emb_proto_euclid",
    # "emb_logreg",
    # "emb_knn",
    ]
HEADS = EMB_HEADS + ECFP_HEADS

EVAL_SPLITS = [
    "random",
    # "scaffold",
    # "similarity",
    # "size",
    ]

# Support sizes to include. Fewer sizes → less crowded x-axis.
SUPPORT_SIZES = [16, 32, 64, 128, 256, 512]

# Dot appearance
DOT_SIZE   = 8      # marker size in points²  (scatter s= parameter)
DOT_ALPHA  = 0.45   # transparency; lower = easier to see dense regions
JITTER     = 0.30   # horizontal jitter width (fraction of per-head slot)

# Draw a horizontal line at median per (head × support_size) group?
SHOW_MEDIAN = True
# =============================================================================

# ── Style maps (shared palette with the other plotting scripts) ───────────────
HEAD_COLOR = {
    "emb_proto_mahalanobis": "#2a78d6",  # blue
    "emb_proto_euclid":      "#1baf7a",  # aqua
    "emb_logreg":            "#008300",  # green
    "emb_knn":               "#eda100",  # amber
    "ecfp_rf":               "#e34948",  # red
    "ecfp_logreg":           "#eb6834",  # orange
    "ecfp_proto_euclid":     "#4a3aa7",  # violet
    "ecfp_proto_tanimoto":   "#c2185b",  # deep pink
    "ecfp_gp_tanimoto":      "#00838f",  # teal
}
HEAD_LABEL = {
    "emb_proto_mahalanobis": "PN-M (emb)",
    "emb_proto_euclid":      "PN-E (emb)",
    "emb_logreg":            "LogReg (emb)",
    "emb_knn":               "kNN (emb)",
    "ecfp_rf":               "RF (ecfp)",
    "ecfp_logreg":           "LogReg (ecfp)",
    "ecfp_proto_euclid":     "PN-E (ecfp)",
    "ecfp_proto_tanimoto":   "PN-T (ecfp)",
    "ecfp_gp_tanimoto":      "GP-T (ecfp)",
}
HEAD_SHORT = {
    "emb_proto_mahalanobis": "PNM", "emb_proto_euclid": "PNE",
    "emb_logreg": "LR",             "emb_knn": "kNN",
    "ecfp_rf": "RF",                "ecfp_logreg": "eLR",
    "ecfp_proto_euclid": "ePN",     "ecfp_proto_tanimoto": "ePNT",
    "ecfp_gp_tanimoto": "eGPT",
}
SPLIT_LABEL = {"random": "Random", "scaffold": "Scaffold (Murcko)", "similarity": "Similarity (Butina@0.70)", "size": "Size"}
SPLIT_SHORT = {"random": "rand", "scaffold": "scaf", "similarity": "sim", "size": "size"}

# ── Load data ─────────────────────────────────────────────────────────────────
from config import make_run_tag   # noqa: E402
base_csv = os.path.join(BASELINE_CSV_DIR, "fsmol_test.csv")
base_df  = pd.read_csv(base_csv) if os.path.exists(base_csv) else None
if base_df is None:
    print(f"[WARN] No baselines CSV at {base_csv} - ECFP heads will be absent.")

dfs = []
if EMB_HEADS:
    for seed in SEEDS:
        tag = make_run_tag(ENCODER, "classification", TRAINING_SPLIT, TRAIN_DISTANCE, N_SUPPORT, seed)
        csv = os.path.join(run_csv_dir(tag), "fsmol_test.csv")
        if not os.path.exists(csv):
            print(f"[WARN] Missing: {csv}")
            continue
        df = pd.read_csv(csv)
        df["seed"] = seed
        if base_df is not None:
            b = base_df.copy(); b["seed"] = seed
            df = pd.concat([df, b], ignore_index=True)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No fsmol_test.csv found for ENCODER={ENCODER} TRAINING_SPLIT={TRAINING_SPLIT} seeds={SEEDS}")
    run_tag = make_run_tag(ENCODER, "classification", TRAINING_SPLIT, TRAIN_DISTANCE, N_SUPPORT, SEEDS[0])
else:
    if base_df is None:
        raise FileNotFoundError(f"No baselines CSV at {base_csv}")
    base_df["seed"] = 0
    dfs = [base_df]
    run_tag = "baselines"

combined = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(dfs)} seed(s) | run_tag={run_tag}")

# ── Per-assay means across seeds ─────────────────────────────────────────────
per_assay = (
    combined[combined["head"].isin(HEADS) & combined["split_type"].isin(EVAL_SPLITS)]
    .groupby(["assay_id", "head", "split_type", "support_size"])["delta_auprc"]
    .mean()
    .reset_index()
)

# ── Layout ────────────────────────────────────────────────────────────────────
n_splits = len(EVAL_SPLITS)
n_heads  = len(HEADS)
n_sizes  = len(SUPPORT_SIZES)

group_gap  = 1.0                              # distance between support-size groups
slot_width = group_gap / (n_heads + 1)        # width per head within a group
x_centers  = np.arange(n_sizes) * group_gap
panel_width = max(6.0, 1.5 + n_heads * n_sizes * 0.28)

rng = np.random.RandomState(0)

fig, axes = plt.subplots(1, n_splits, figsize=(panel_width * n_splits, 5), sharey=True)
if n_splits == 1:
    axes = [axes]

for ax, split in zip(axes, EVAL_SPLITS):
    for hi, head in enumerate(HEADS):
        color  = HEAD_COLOR[head]
        offset = (hi - (n_heads - 1) / 2) * slot_width

        for si, sz in enumerate(SUPPORT_SIZES):
            vals = per_assay[
                (per_assay["head"] == head) &
                (per_assay["split_type"] == split) &
                (per_assay["support_size"] == sz)
            ]["delta_auprc"].dropna().values

            if len(vals) == 0:
                continue

            cx = x_centers[si] + offset
            jit = rng.uniform(-JITTER * slot_width / 2,
                               JITTER * slot_width / 2, size=len(vals))
            ax.scatter(
                cx + jit, vals,
                s=DOT_SIZE, color=color, alpha=DOT_ALPHA, linewidths=0,
            )
            if SHOW_MEDIAN:
                med = float(np.median(vals))
                ax.plot(
                    [cx - slot_width * 0.35, cx + slot_width * 0.35],
                    [med, med],
                    color=color, linewidth=1.6, solid_capstyle="round",
                )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{SPLIT_LABEL[split]} split", fontsize=11)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(s) for s in SUPPORT_SIZES])
    ax.set_xlabel("Support size (n)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)

axes[0].set_ylabel("ΔAUPRC (per assay)", fontsize=10)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=HEAD_COLOR[h], alpha=0.85, label=HEAD_LABEL[h])
    for h in HEADS
]
fig.legend(handles=legend_handles, title="Head", loc="lower center",
           ncol=n_heads, fontsize=8, title_fontsize=8,
           bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

encoder_label = "GNN" if "gnn" in ENCODER else "ECFP"
split_label   = TRAINING_SPLIT.replace("_", "-")
seed_str      = f"seeds {SEEDS}" if len(SEEDS) > 1 else f"seed {SEEDS[0]}"
median_note   = "  |  tick = median" if SHOW_MEDIAN else ""
fig.suptitle(
    f"FS-Mol Test: Per-Assay ΔAUPRC (individual dots) — {encoder_label} + {split_label}  ({seed_str}{median_note})",
    fontsize=11, y=1.02,
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
heads_tag  = "_".join(HEAD_SHORT[h] for h in HEADS)
splits_tag = "_".join(SPLIT_SHORT[s] for s in EVAL_SPLITS)
fname      = f"strip_{heads_tag}__{splits_tag}.png"

out_dir = run_fig_dir(run_tag)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, fname)

plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")
