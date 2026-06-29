"""
Intervention Comparison Plot
=============================
Grouped bar chart at n=128 comparing scaffold and random split ΔAUPRC
across all training strategies and test-time interventions.

Usage (from PTN/ root):
    python Analysis/model/plot_interventions.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import RESULTS_DIR, DATA_ANALYSIS_FIGURES_DIR

N_SUPPORT = 128

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

# Hard-coded 3-seed aggregate results (from cluster run, 2026-06-28).
# These are used as fallback when the aggregate CSVs are not present locally.
_TTPA_FALLBACK = {
    # (model_name, method, split_type): (mean, std)
    ("fsmol_gnn_classification_random",      "standard", "scaffold"): (0.0495, 0.0010),
    ("fsmol_gnn_classification_random",      "standard", "random"):   (0.2166, 0.0010),
    ("fsmol_gnn_classification_random",      "ttpa",     "scaffold"): (0.0499, 0.0010),
    ("fsmol_gnn_classification_random",      "ttpa",     "random"):   (0.2160, 0.0010),
    ("fsmol_gnn_classification_shift_aware", "standard", "scaffold"): (0.0499, 0.0030),
    ("fsmol_gnn_classification_shift_aware", "standard", "random"):   (0.2203, 0.0000),
    ("fsmol_gnn_classification_shift_aware", "ttpa",     "scaffold"): (0.0502, 0.0010),
    ("fsmol_gnn_classification_shift_aware", "ttpa",     "random"):   (0.2196, 0.0010),
}


def _load_ttpa_agg(model_name: str, method: str, split_type: str) -> tuple[float, float]:
    """Return (mean, std) from ttpa aggregate CSV; fall back to hard-coded constants."""
    path = os.path.join(RESULTS_DIR, f"{model_name}_ttpa_aggregate.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        row = df[(df["method"] == method) &
                 (df["split_type"] == split_type) &
                 (df["support_size"] == N_SUPPORT)]
        if not row.empty:
            return float(row["mean"].iloc[0]), float(row["std"].iloc[0])
    key = (model_name, method, split_type)
    return _TTPA_FALLBACK.get(key, (float("nan"), float("nan")))


def _load_anneal_seed0(split_type: str) -> float:
    """Load anneal seed0 result from its fsmol_test_results.csv."""
    path = os.path.join(RESULTS_DIR,
                        "fsmol_gnn_classification_anneal_seed0",
                        "fsmol_test_results.csv")
    if not os.path.exists(path):
        return float("nan")
    df = pd.read_csv(path)
    row = df[(df["split_type"] == split_type) & (df["support_size"] == N_SUPPORT)]
    return float(row["delta_auprc"].mean()) if not row.empty else float("nan")


# RF reference (from README, n=128)
RF_SCAFFOLD = 0.182
RF_RANDOM   = 0.194

# Build rows: (label, scaffold_mean, scaffold_std, random_mean, random_std, anneal_only)
rows = []

gnn_models = {
    "GNN random":      "fsmol_gnn_classification_random",
    "GNN shift-aware": "fsmol_gnn_classification_shift_aware",
}

for label, model in gnn_models.items():
    sc_m, sc_s = _load_ttpa_agg(model, "standard", "scaffold")
    rd_m, rd_s = _load_ttpa_agg(model, "standard", "random")
    rows.append((label, sc_m, sc_s, rd_m, rd_s, False))

for label, model in gnn_models.items():
    sc_m, sc_s = _load_ttpa_agg(model, "ttpa", "scaffold")
    rd_m, rd_s = _load_ttpa_agg(model, "ttpa", "random")
    rows.append((f"{label} + TTPA", sc_m, sc_s, rd_m, rd_s, False))

# Anneal seed 0 (no std - single seed)
anneal_sc = _load_anneal_seed0("scaffold")
anneal_rd = _load_anneal_seed0("random")
rows.append(("GNN anneal*", anneal_sc, 0.0, anneal_rd, 0.0, True))

# RF reference
rows.append(("RF baseline", RF_SCAFFOLD, 0.0, RF_RANDOM, 0.0, False))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

labels   = [r[0] for r in rows]
sc_means = np.array([r[1] for r in rows])
sc_stds  = np.array([r[2] for r in rows])
rd_means = np.array([r[3] for r in rows])
rd_stds  = np.array([r[4] for r in rows])
is_seed0 = [r[5] for r in rows]

n = len(labels)
x = np.arange(n)
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

bar_sc = ax.bar(x - w/2, sc_means, w, yerr=sc_stds, capsize=3,
                color="#d62728", alpha=0.85, label="Scaffold split")
bar_rd = ax.bar(x + w/2, rd_means, w, yerr=rd_stds, capsize=3,
                color="#1f77b4", alpha=0.85, label="Random split")

# Hatch anneal bars to signal single-seed
for i, seed0 in enumerate(is_seed0):
    if seed0:
        bar_sc[i].set_hatch("//")
        bar_rd[i].set_hatch("//")

# Reference line at scaffold ceiling
ax.axhline(0.050, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.5,
           label="Scaffold ceiling (~0.050)")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Mean ΔAUPRC (n=128)", fontsize=10)
ax.set_title(f"Scaffold OOD Interventions - FS-Mol Test, n={N_SUPPORT}", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.28)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.text(0.99, 0.97, "* anneal = seed 0 only, seeds 1-2 pending",
        transform=ax.transAxes, ha="right", va="top", fontsize=7, color="gray")

plt.tight_layout()

os.makedirs(DATA_ANALYSIS_FIGURES_DIR, exist_ok=True)
out = os.path.join(DATA_ANALYSIS_FIGURES_DIR, "fig_interventions.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
