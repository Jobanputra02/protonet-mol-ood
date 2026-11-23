"""
chembl36 assay-size distribution - how many tasks have how many molecules.
==========================================================================

One bar per exact molecule count, linear x-axis.

Why the x-axis stops at 100 by default: the counts decay smoothly and 98.3% of tasks
have <= 100 molecules. Past that the tallest bar is 31 tasks, which is invisible next to
the 12,853 at x=5, so the remaining 900 px of axis would be blank. The number of tasks
beyond the cut is printed on the figure so nothing is hidden. Use --max-x to change it.

A task is one (assay_chembl_id, standard_type) pair, counting only rows whose
`target_transformed_type` is a p-scale transform (pchembl_value / pmic / pmic_nm) -
mixing transform types inside a task would mix scales (TASKS.md 4.6.1).

Usage
-----
    python Analysis/data/chembl36_assay_sizes.py
    python Analysis/data/chembl36_assay_sizes.py --max-x 200
    python Analysis/data/chembl36_assay_sizes.py --all-types
"""

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.ticker import FuncFormatter           # noqa: E402

from config import (CUSTOM_DIR, DATA_ANALYSIS_FIGURES_DIR,   # noqa: E402
                    DATA_ANALYSIS_RESULTS_DIR)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
BLUE = "#2a78d6"
GRID = "#deddd8"
MARK = "#8a8880"

P_SCALE = {"pchembl_value", "pmic", "pmic_nm"}


def assay_sizes(all_types=False):
    counter = collections.Counter()
    cols = ["assay_chembl_id", "standard_type", "target_transformed",
            "target_transformed_type"]
    for ch in pd.read_csv(os.path.join(CUSTOM_DIR, "chembl36_curated.csv"),
                          usecols=cols, chunksize=500_000, low_memory=False):
        st = ch["standard_type"].fillna("NA")
        if all_types:
            counter.update(zip(ch["assay_chembl_id"], st))
            continue
        tt = ch["target_transformed_type"].fillna("")
        val = pd.to_numeric(ch["target_transformed"], errors="coerce")
        m = tt.isin(P_SCALE) & val.notna()
        if m.any():
            counter.update(zip(ch.loc[m, "assay_chembl_id"], st[m]))
    return np.array(sorted(counter.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-x", type=int, default=100)
    ap.add_argument("--all-types", action="store_true")
    args = ap.parse_args()

    print("scanning chembl36 ...", flush=True)
    s = assay_sizes(args.all_types)
    tag = "all_types" if args.all_types else "pscale"
    lo, hi = int(s.min()), int(s.max())
    print(f"  {len(s):,} tasks | min {lo} | max {hi:,} | median {int(np.median(s))}")

    xs = np.arange(lo, args.max_x + 1)
    ys = np.array([(s == x).sum() for x in xs])
    beyond = int((s > args.max_x).sum())

    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(xs, ys, width=0.82, color=BLUE, linewidth=0, zorder=3)

    # the one threshold that drives a decision: an episode at n=16 needs 32 molecules
    ax.axvline(32, color=MARK, lw=1.2, ls=(0, (4, 3)), zorder=4)
    n16 = int((s >= 32).sum())
    ax.annotate(f"32 molecules = the minimum for a support size of 16\n"
                f"{n16:,} tasks ({100*n16/len(s):.0f}%) are at or past this line",
                xy=(32, ys.max() * 0.80), xytext=(14, 0), textcoords="offset points",
                ha="left", va="center", color=INK, fontsize=10.5)

    ax.set_xlim(lo - 1, args.max_x + 1)
    ax.set_ylim(0, ys.max() * 1.08)
    ax.set_xlabel("molecules in the task", color=INK_2, fontsize=11)
    ax.set_ylabel("number of tasks", color=INK_2, fontsize=11)
    ax.set_title(f"chembl36: how many tasks have how many molecules\n"
                 f"{len(s):,} tasks  |  smallest {lo}, median {int(np.median(s))}, "
                 f"largest {hi:,}",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=3)

    if beyond:
        ax.annotate(f"a further {beyond:,} tasks ({100*beyond/len(s):.1f}%) have more "
                    f"than {args.max_x} molecules, up to {hi:,} - off the right of this chart",
                    xy=(0.995, -0.165), xycoords="axes fraction", ha="right", va="top",
                    color=INK_2, fontsize=9.5)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.175)

    os.makedirs(DATA_ANALYSIS_FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_ANALYSIS_RESULTS_DIR, exist_ok=True)
    png = os.path.join(DATA_ANALYSIS_FIGURES_DIR, f"chembl36_assay_sizes_{tag}.png")
    pdf = os.path.join(DATA_ANALYSIS_FIGURES_DIR, f"chembl36_assay_sizes_{tag}.pdf")
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    fig.savefig(pdf, facecolor=SURFACE)
    plt.close(fig)

    # full per-value table, including the part beyond the cut
    allx = np.arange(lo, hi + 1)
    pd.DataFrame({"molecules": allx,
                  "tasks": [(s == x).sum() for x in allx]}).to_csv(
        os.path.join(DATA_ANALYSIS_RESULTS_DIR,
                     f"chembl36_assay_sizes_{tag}_per_value.csv"), index=False)

    print(f"\nfigure -> {png}\n          {pdf}")


if __name__ == "__main__":
    main()
