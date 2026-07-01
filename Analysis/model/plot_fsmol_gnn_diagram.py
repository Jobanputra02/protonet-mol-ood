"""
Generates a diagram of the FSMolGNN architecture in the same visual style
as the ECFP/GNN slide — white background, gray palette, minimal.

Output: outputs/figures/data_analysis/fsmol_gnn_diagram.png
        (drop-in replacement for the GNN right panel in the slide)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_ANALYSIS_FIGURES_DIR

# ── colours matching the slide ──────────────────────────────────────────────
WHITE      = "#ffffff"
DARK_GRAY  = "#3a3a3a"   # text, atoms, bonds
MID_GRAY   = "#8a8a8a"   # labels, arrows
BOX_EDGE   = "#9a9a9a"
BOX_FACE   = "#e8e8e8"
BOX_FACE2  = "#d4d4d4"   # slightly darker for distinction

fig, ax = plt.subplots(figsize=(5.2, 8.2))
ax.set_xlim(0, 5.2)
ax.set_ylim(0, 8.2)
ax.axis("off")
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

CX = 2.5   # horizontal center
CR_LABEL_X = 4.30  # x for CombinedReadout label (inside figure)

# ── helper: draw arrow ───────────────────────────────────────────────────────
def arrow(y_from, y_to, x=CX):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="->", color=MID_GRAY, lw=1.4))

# ── helper: draw a box ───────────────────────────────────────────────────────
def box(y_center, height, width=3.2, label="", sublabel="",
        face=BOX_FACE, edge=BOX_EDGE):
    x0 = CX - width / 2
    y0 = y_center - height / 2
    rect = FancyBboxPatch((x0, y0), width, height,
                          boxstyle="round,pad=0.04",
                          linewidth=0.9, edgecolor=edge, facecolor=face)
    ax.add_patch(rect)
    dy = 0.06 if sublabel else 0
    ax.text(CX, y_center + dy, label, ha="center", va="center",
            fontsize=8.5, color=DARK_GRAY, fontfamily="sans-serif")
    if sublabel:
        ax.text(CX, y_center - 0.11, sublabel, ha="center", va="center",
                fontsize=6.5, color=MID_GRAY, fontfamily="sans-serif")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Molecule graph
# ══════════════════════════════════════════════════════════════════════════════
gx, gy = CX, 7.75
r = 0.38
angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
atoms = [(gx + r * np.cos(a), gy + r * np.sin(a)) for a in angles]

# benzene ring bonds
for i in range(6):
    ax.plot([atoms[i][0], atoms[(i + 1) % 6][0]],
            [atoms[i][1], atoms[(i + 1) % 6][1]],
            color=DARK_GRAY, lw=1.6, zorder=1, solid_capstyle="round")

# side chain (two extra atoms off atom index 0)
side1 = (atoms[0][0] + 0.28, atoms[0][1] + 0.10)
side2 = (atoms[0][0] + 0.52, atoms[0][1] + 0.30)
ax.plot([atoms[0][0], side1[0]], [atoms[0][1], side1[1]], color=DARK_GRAY, lw=1.6)
ax.plot([side1[0], side2[0]],   [side1[1], side2[1]],   color=DARK_GRAY, lw=1.6)

# draw atoms
for a in atoms + [side1, side2]:
    ax.add_patch(plt.Circle(a, 0.068, color=DARK_GRAY, zorder=2))

# extra bond to indicate multiple connections on GNN side
ax.plot([atoms[2][0], atoms[2][0] - 0.30],
        [atoms[2][1], atoms[2][1] + 0.28], color=DARK_GRAY, lw=1.6)
ax.add_patch(plt.Circle((atoms[2][0] - 0.30, atoms[2][1] + 0.28),
                         0.068, color=DARK_GRAY, zorder=2))

# ── feature labels on atoms / bonds ─────────────────────────────────────────
# small annotation boxes to hint at feature vectors
ax.text(gx - 0.78, gy + 0.45, "40", ha="center", va="center",
        fontsize=6, color=MID_GRAY,
        bbox=dict(boxstyle="round,pad=0.15", fc=BOX_FACE, ec=BOX_EDGE, lw=0.7))
ax.text(gx - 0.78, gy + 0.20, "feat", ha="center", va="center",
        fontsize=5.5, color=MID_GRAY)

ax.text(gx + 0.85, gy - 0.05, "10", ha="center", va="center",
        fontsize=6, color=MID_GRAY,
        bbox=dict(boxstyle="round,pad=0.15", fc=BOX_FACE, ec=BOX_EDGE, lw=0.7))
ax.text(gx + 0.85, gy - 0.30, "feat", ha="center", va="center",
        fontsize=5.5, color=MID_GRAY)

ax.text(CX, 7.12, "atom-bond graph", ha="center", fontsize=8.5,
        color=DARK_GRAY, fontfamily="sans-serif")
ax.text(CX, 6.88, "(40 atom features · 10 bond features)",
        ha="center", fontsize=6.8, color=MID_GRAY, fontfamily="sans-serif")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Arrow → 10 PNA layers
# ══════════════════════════════════════════════════════════════════════════════
arrow(6.75, 6.55)

# ── layer positions ──────────────────────────────────────────────────────────
LAYER_H  = 0.40
LAYER_GAP = 0.08
LAYER_W  = 3.2
CR_X     = CX + LAYER_W / 2 + 0.14   # x of CombinedReadout vertical bar

# show layers 1, 2, ellipsis, 10
layer_labels = [
    ("PNA Layer 1", "sum · mean · std · max   +  BOOM  +  ReZero"),
    ("PNA Layer 2", "sum · mean · std · max   +  BOOM  +  ReZero"),
    None,   # ellipsis
    ("PNA Layer 10", "sum · mean · std · max   +  BOOM  +  ReZero"),
]

layer_centers = []
y = 6.55 - LAYER_H / 2
for entry in layer_labels:
    if entry is None:
        # ellipsis row
        ax.text(CX, y, "· · ·", ha="center", va="center",
                fontsize=13, color=MID_GRAY)
        layer_centers.append(None)
        y -= (LAYER_H + LAYER_GAP)
    else:
        layer_centers.append(y)
        box(y, LAYER_H, width=LAYER_W,
            label=entry[0], sublabel=entry[1],
            face=BOX_FACE if entry[0] != "PNA Layer 10" else BOX_FACE2)
        y -= (LAYER_H + LAYER_GAP)

# ── CombinedReadout vertical bar + ticks ─────────────────────────────────────
real_centers = [c for c in layer_centers if c is not None]
bar_top = real_centers[0]  + LAYER_H / 2
bar_bot = real_centers[-1] - LAYER_H / 2

ax.plot([CR_X, CR_X], [bar_bot, bar_top], color=MID_GRAY, lw=1.4)

# ticks from each real layer
for c in real_centers:
    ax.plot([CX + LAYER_W / 2, CR_X], [c, c],
            color=MID_GRAY, lw=0.9, linestyle="--", dashes=(4, 3))

# label
mid_y = (bar_top + bar_bot) / 2
ax.text(CR_X + 0.10, mid_y + 0.13, "Combined", ha="left", va="center",
        fontsize=6.5, color=MID_GRAY, fontfamily="sans-serif")
ax.text(CR_X + 0.10, mid_y - 0.04, "Readout", ha="left", va="center",
        fontsize=6.5, color=MID_GRAY, fontfamily="sans-serif")
ax.text(CR_X + 0.10, mid_y - 0.22, "(all layers)", ha="left", va="center",
        fontsize=5.5, color=MID_GRAY, fontfamily="sans-serif")

# arrow from bar down to fusion box
readout_y_bot = bar_bot - 0.05
arrow_y_end   = readout_y_bot - 0.28
ax.annotate("", xy=(CX, arrow_y_end), xytext=(CR_X, readout_y_bot),
            arrowprops=dict(arrowstyle="->", color=MID_GRAY, lw=1.4,
                            connectionstyle="arc3,rad=0.0"))

# ══════════════════════════════════════════════════════════════════════════════
# 3. ECFP fusion box
# ══════════════════════════════════════════════════════════════════════════════
fuse_y = arrow_y_end - 0.22
box(fuse_y, 0.38, width=3.2,
    label="⊕  ECFP fingerprint fusion",
    sublabel="concat learned embedding + 2048-bit ECFP",
    face="#ddeeff", edge="#99bbdd")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Arrow → 512-dim embedding
# ══════════════════════════════════════════════════════════════════════════════
emb_y = fuse_y - 0.55
arrow(fuse_y - 0.19, emb_y + 0.19)

box(emb_y, 0.38, width=3.2,
    label="512-dim embedding",
    sublabel="(used as molecule representation for ProtoNet)",
    face=BOX_FACE2, edge=BOX_EDGE)

# ══════════════════════════════════════════════════════════════════════════════
# save
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(DATA_ANALYSIS_FIGURES_DIR, exist_ok=True)
out = os.path.join(DATA_ANALYSIS_FIGURES_DIR, "fsmol_gnn_diagram.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
print(f"Saved -> {out}")
