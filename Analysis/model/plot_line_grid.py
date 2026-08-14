"""
Configurable baseline-grid line plot
=====================================
ONE FIGURE per model run. Each line = (head, eval_split) combination.
- Color  encodes head  (consistent palette across all runs)
- Linestyle encodes eval split (solid/dashed/dotted)
- Error bars = +/- 1 std across seeds

Covers all 9 heads:
  emb_*  : emb_proto_mahalanobis, emb_proto_euclid, emb_logreg, emb_knn
  ecfp_* : ecfp_rf, ecfp_logreg, ecfp_proto_euclid, ecfp_proto_tanimoto, ecfp_gp_tanimoto

To reproduce fig_ecfp_baseline_curves.png: set HEADS = ECFP_HEADS, EVAL_SPLITS = all 3.
To reproduce fig2a-style single-head plot:  set HEADS = ["emb_proto_mahalanobis"], EVAL_SPLITS = all 3.

Output: outputs/{run_tag}/figures/lineplot_{heads_tag}__{splits_tag}.png

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

# Pick any subset of the 7 available heads:
ECFP_HEADS = [
    # "ecfp_rf",
    # "ecfp_logreg",
    # "ecfp_proto_euclid",
    # "ecfp_proto_tanimoto",
    "ecfp_gp_tanimoto",
    ]
EMB_HEADS  = [
    "emb_proto_mahalanobis",
    "emb_proto_euclid",
    # "emb_logreg",
    # "emb_knn"
    ]

HEADS = ECFP_HEADS + EMB_HEADS          # change to ECFP_HEADS, EMB_HEADS, or ECFP_HEADS + EMB_HEADS

# Pick any subset of: "random", "scaffold", "size"
EVAL_SPLITS = [
    "random",
    # "scaffold",
    # "similarity",
    # "size",
    ]

# Error bar mode:
#   "assays" - std across assays (standard for few-shot benchmarks; answers "how consistent across tasks?")
#   "seeds"  - std across seeds  (reproducibility check; answers "is the result stable?")
#   None     - no error bars (clean lines only)
ERROR_BARS = "assays"
# =============================================================================

# if ERROR_BARS == "None":  # allow writing ERROR_BARS = "None" as well as None
#     ERROR_BARS = None

SUPPORT_SIZES = [16, 32, 64, 128, 256, 512]

# ── Style maps ────────────────────────────────────────────────────────────────
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
HEAD_MARKER = {
    "emb_proto_mahalanobis": "o",
    "emb_proto_euclid":      "s",
    "emb_logreg":            "^",
    "emb_knn":               "D",
    "ecfp_rf":               "o",
    "ecfp_logreg":           "s",
    "ecfp_proto_euclid":     "^",
    "ecfp_proto_tanimoto":   "v",
    "ecfp_gp_tanimoto":      "P",
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
SPLIT_LSTYLE = {"random": "-", "scaffold": "--", "similarity": "-.", "size": ":"}
SPLIT_LABEL  = {"random": "Random", "scaffold": "Scaffold (Murcko)", "similarity": "Similarity (Butina@0.70)", "size": "Size"}

HEAD_SHORT  = {
    "emb_proto_mahalanobis": "PNM", "emb_proto_euclid": "PNE",
    "emb_logreg": "LR", "emb_knn": "kNN",
    "ecfp_rf": "RF", "ecfp_logreg": "eLR", "ecfp_proto_euclid": "ePN",
    "ecfp_proto_tanimoto": "ePNT", "ecfp_gp_tanimoto": "eGPT",
}
SPLIT_SHORT = {"random": "rand", "scaffold": "scaf", "similarity": "sim", "size": "size"}

# ── Load data ─────────────────────────────────────────────────────────────────
# emb heads live in <run_tag>/csvs/fsmol_test.csv; the model-free ECFP baselines
# live once in baselines/csvs/fsmol_test.csv. Merge them per seed.
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
    # ECFP-only mode: no model needed, load baselines directly
    if base_df is None:
        raise FileNotFoundError(f"No baselines CSV at {base_csv}")
    base_df["seed"] = 0
    dfs = [base_df]
    run_tag = "baselines"

combined = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(dfs)} seed(s) | run_tag={run_tag}")

# ── Compute mean +/- std ─────────────────────────────────────────────────────
records = []
for head in HEADS:
    for split in EVAL_SPLITS:
        for sz in SUPPORT_SIZES:
            sub = combined[
                (combined["head"] == head) &
                (combined["split_type"] == split) &
                (combined["support_size"] == sz)
            ]
            if sub.empty:
                continue

            if ERROR_BARS == "assays":
                # Average seeds per assay first, then std across assays
                per_assay = sub.groupby("assay_id")["delta_auprc"].mean()
                mean_val = per_assay.mean()
                std_val  = per_assay.std(ddof=1) if len(per_assay) > 1 else 0.0
            else:  # "seeds"
                per_seed = [
                    sub[sub["seed"] == s]["delta_auprc"].mean()
                    for s in SEEDS if not sub[sub["seed"] == s].empty
                ]
                mean_val = np.mean(per_seed)
                std_val  = np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0

            records.append({
                "head": head, "split": split, "support_size": sz,
                "mean": mean_val, "std": std_val,
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
            yerr=ye if ERROR_BARS is not None else None,
            color=color,
            linestyle=SPLIT_LSTYLE[split],
            marker=marker,
            markersize=5,
            linewidth=1.8,
            capsize=3 if ERROR_BARS is not None else 0,
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
ax.legend(handles=split_handles, title="Eval split", loc="upper right",
          fontsize=8, title_fontsize=8, framealpha=0.9)

# ── Labels & formatting ───────────────────────────────────────────────────────
encoder_label = "GNN" if "gnn" in ENCODER else "ECFP"
split_label   = TRAINING_SPLIT.replace("_", "-")
eb_note = (
    "error bars = ±1 std across assays" if ERROR_BARS == "assays" else
    "error bars = ±1 std across seeds"  if ERROR_BARS == "seeds"  else
    None
)

ax.set_title(
    f"FS-Mol Test: Mean ΔAUPRC - {encoder_label} + {split_label}",
    fontsize=11,
)
ax.set_xlabel("Support size (n)", fontsize=11)
ax.set_ylabel("Mean ΔAUPRC", fontsize=11)
# # N = number of unique assays qualifying at each support size (use first split as reference)
# ref_split = EVAL_SPLITS[0]
# n_assays = {
#     sz: combined[(combined["split_type"] == ref_split) & (combined["support_size"] == sz)]["assay_id"].nunique()
#     for sz in SUPPORT_SIZES
# }
ax.set_xticks(SUPPORT_SIZES)
# ax.set_xticklabels([f"{sz} (N={n_assays[sz]})" for sz in SUPPORT_SIZES], rotation=45, ha="center")
ax.set_xticklabels([str(s) for s in SUPPORT_SIZES])
ax.grid(True, alpha=0.3)
if eb_note:
    ax.annotate(eb_note, xy=(1, 0), xycoords="axes fraction",
                fontsize=7, color="gray", ha="right", va="bottom",
                xytext=(0, -36), textcoords="offset points")
plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
heads_tag  = "_".join(HEAD_SHORT[h] for h in HEADS)
splits_tag = "_".join(SPLIT_SHORT[s] for s in EVAL_SPLITS)
fname      = f"lineplot_{heads_tag}__{splits_tag}.png"

out_dir = run_fig_dir(run_tag)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, fname)

plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")
