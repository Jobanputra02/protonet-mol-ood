"""
Configurable baseline-grid line plot
=====================================
ONE FIGURE per model run. Each line = (head, eval_split) combination.
- Color  encodes head  (consistent palette across all runs)
- Linestyle encodes eval split (solid/dashed/dotted)
- Error bars = +/- 1 std across seeds

Covers all 7 heads:
  emb_*  : emb_proto_mahalanobis, emb_proto_euclid, emb_logreg, emb_knn
  ecfp_* : ecfp_rf, ecfp_logreg, ecfp_proto_euclid

To reproduce fig_ecfp_baseline_curves.png: set HEADS = ECFP_HEADS, EVAL_SPLITS = all 3.
To reproduce fig2a-style single-head plot:  set HEADS = ["emb_proto_mahalanobis"], EVAL_SPLITS = all 3.

Output: outputs/figures/{run_tag}/lineplot_{heads_tag}__{splits_tag}.png

Usage
-----
    python Analysis/model/plot_line_grid.py
Edit CONFIG block below to change what is plotted.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RESULTS_DIR, FIGURES_DIR

# =============================================================================
# CONFIG - edit these, then run
# =============================================================================
ENCODER        = "fsmol_gnn"        # "ecfp" | "fsmol_gnn"
TRAINING_SPLIT = "random"           # "random" | "shift_aware"
SEEDS          = [0, 1, 2]

# Pick any subset of the 7 available heads:
ECFP_HEADS = [
    # "ecfp_rf",
    # "ecfp_logreg",
    # "ecfp_proto_euclid"
    ]
EMB_HEADS  = [
    "emb_proto_mahalanobis",
    # "emb_proto_euclid",
    # "emb_logreg",
    # "emb_knn"
    ]

HEADS = ECFP_HEADS + EMB_HEADS          # change to ECFP_HEADS, EMB_HEADS, or ECFP_HEADS + EMB_HEADS

# Pick any subset of: "random", "scaffold", "size"
EVAL_SPLITS = [
    "random",
    "scaffold",
    "size"
    ]
# =============================================================================

SUPPORT_SIZES = [16, 32, 64, 128, 256, 512]

# ── Style maps ────────────────────────────────────────────────────────────────
HEAD_COLOR = {
    "emb_proto_mahalanobis": "#1f77b4",  # blue
    "emb_proto_euclid":      "#17becf",  # teal
    "emb_logreg":            "#2ca02c",  # green
    "emb_knn":               "#8c564b",  # brown
    "ecfp_rf":               "#d62728",  # red
    "ecfp_logreg":           "#ff7f0e",  # orange
    "ecfp_proto_euclid":     "#9467bd",  # purple
}
HEAD_MARKER = {
    "emb_proto_mahalanobis": "o",
    "emb_proto_euclid":      "s",
    "emb_logreg":            "^",
    "emb_knn":               "D",
    "ecfp_rf":               "o",
    "ecfp_logreg":           "s",
    "ecfp_proto_euclid":     "^",
}
HEAD_LABEL = {
    "emb_proto_mahalanobis": "PN-M (emb)",
    "emb_proto_euclid":      "PN-E (emb)",
    "emb_logreg":            "LogReg (emb)",
    "emb_knn":               "kNN (emb)",
    "ecfp_rf":               "RF (ecfp)",
    "ecfp_logreg":           "LogReg (ecfp)",
    "ecfp_proto_euclid":     "PN-E (ecfp)",
}
SPLIT_LSTYLE = {"random": "-", "scaffold": "--", "size": ":"}
SPLIT_LABEL  = {"random": "Random", "scaffold": "Scaffold", "size": "Size"}

HEAD_SHORT  = {
    "emb_proto_mahalanobis": "PNM", "emb_proto_euclid": "PNE",
    "emb_logreg": "LR", "emb_knn": "kNN",
    "ecfp_rf": "RF", "ecfp_logreg": "eLR", "ecfp_proto_euclid": "ePN",
}
SPLIT_SHORT = {"random": "rand", "scaffold": "scaf", "size": "size"}

# ── Load data ─────────────────────────────────────────────────────────────────
run_tag = f"{ENCODER}_classification_{TRAINING_SPLIT}"
dfs = []
for seed in SEEDS:
    tag = f"{run_tag}_seed{seed}"
    csv = os.path.join(RESULTS_DIR, tag, "baseline_grid.csv")
    if not os.path.exists(csv):
        print(f"[WARN] Missing: {csv}")
        continue
    df = pd.read_csv(csv)
    df["seed"] = seed
    dfs.append(df)

if not dfs:
    raise FileNotFoundError(f"No baseline_grid.csv found for '{run_tag}' seeds {SEEDS}")

combined = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(dfs)} seed(s) for {run_tag}")

# ── Compute mean +/- std across seeds ────────────────────────────────────────
records = []
for head in HEADS:
    for split in EVAL_SPLITS:
        for sz in SUPPORT_SIZES:
            per_seed = []
            for seed in SEEDS:
                sub = combined[
                    (combined["seed"] == seed) &
                    (combined["head"] == head) &
                    (combined["split_type"] == split) &
                    (combined["support_size"] == sz)
                ]
                if not sub.empty:
                    per_seed.append(sub["delta_auprc"].mean())
            if per_seed:
                records.append({
                    "head": head, "split": split, "support_size": sz,
                    "mean":    np.mean(per_seed),
                    "std":     np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0,
                    "n_seeds": len(per_seed),
                })

stats = pd.DataFrame(records)
if stats.empty:
    raise ValueError("No data after filtering - check HEADS / EVAL_SPLITS / SEEDS.")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for head in HEADS:
    color  = HEAD_COLOR[head]
    marker = HEAD_MARKER[head]
    for split in EVAL_SPLITS:
        sub = stats[(stats["head"] == head) & (stats["split"] == split)].sort_values("support_size")
        if sub.empty:
            continue
        x  = sub["support_size"].values
        y  = sub["mean"].values
        ye = sub["std"].values
        ax.errorbar(
            x, y,
            yerr=ye,
            color=color,
            linestyle=SPLIT_LSTYLE[split],
            marker=marker,
            markersize=5,
            linewidth=1.8,
            capsize=3,
            elinewidth=1.0,
        )

# ── Reference line ─────────────────────────────────────────────────────────────
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# ── Two-part legend ───────────────────────────────────────────────────────────
head_handles = [
    Line2D([0], [0], color=HEAD_COLOR[h], marker=HEAD_MARKER[h],
           markersize=5, linewidth=1.8, label=HEAD_LABEL[h])
    for h in HEADS
]
split_handles = [
    Line2D([0], [0], color="gray", linestyle=SPLIT_LSTYLE[s],
           linewidth=1.8, label=SPLIT_LABEL[s])
    for s in EVAL_SPLITS
]

leg1 = ax.legend(handles=head_handles, title="Head", loc="upper left",
                  fontsize=8, title_fontsize=8, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=split_handles, title="Eval split", loc="lower right",
          fontsize=8, title_fontsize=8, framealpha=0.9)

# ── Labels & formatting ───────────────────────────────────────────────────────
encoder_label = "GNN" if "gnn" in ENCODER else "ECFP"
split_label   = TRAINING_SPLIT.replace("_", "-")
seed_str      = f"seeds {SEEDS}" if len(SEEDS) > 1 else f"seed {SEEDS[0]}"

ax.set_title(
    f"FS-Mol Test: Mean ΔAUPRC - {encoder_label} + {split_label}  ({seed_str})",
    fontsize=11,
)
ax.set_xlabel("Support size (n)", fontsize=11)
ax.set_ylabel("Mean ΔAUPRC", fontsize=11)
ax.set_xticks(SUPPORT_SIZES)
ax.set_xticklabels([str(s) for s in SUPPORT_SIZES])
ax.grid(True, alpha=0.3)
plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
heads_tag  = "_".join(HEAD_SHORT[h] for h in HEADS)
splits_tag = "_".join(SPLIT_SHORT[s] for s in EVAL_SPLITS)
fname      = f"lineplot_{heads_tag}__{splits_tag}.png"

out_dir = os.path.join(FIGURES_DIR, run_tag)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, fname)

plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")
