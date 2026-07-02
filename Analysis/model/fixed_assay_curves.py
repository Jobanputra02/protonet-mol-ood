"""
Fixed-Assay-Set Curves
=======================
The support-size sweep is reported over different assay populations at each x-tick
(only ~11 assays survive at n=512 vs ~154 at n=16). Statements like "GNN holds at
large n, ECFP collapses" then compare different assays at each point.

This helper recomputes the curve TWICE:
  * naive       - mean over whatever assays qualify at each support size (current behaviour)
  * fixed_set   - mean over only the assays present at EVERY support size in the range
                  (the honest within-assay trend)

Works on any long-form CSV with columns:
    assay_id, split_type, support_size, delta_auprc   (+ optional head/representation)

HOW TO RUN: edit the CONFIG block at the bottom (point CSV at a baseline_grid.csv or a
main.py fsmol_test_results.csv), then:  python Analysis/model/fixed_assay_curves.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fixed_set_summary(df: pd.DataFrame, value="delta_auprc",
                      extra_group=None) -> pd.DataFrame:
    """Return long-form summary with both naive and fixed-assay-set means.

    extra_group: optional extra column (e.g. 'head') to compute curves per group.
    """
    group_keys = ["split_type"] + ([extra_group] if extra_group else [])
    out = []

    for keys, sub in df.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sizes = sorted(sub["support_size"].unique())

        # assays present at EVERY support size in this group
        per_size_assays = {s: set(sub[sub["support_size"] == s]["assay_id"])
                           for s in sizes}
        common = set.intersection(*per_size_assays.values()) if per_size_assays else set()

        for s in sizes:
            ss = sub[sub["support_size"] == s]
            naive = ss.groupby("assay_id")[value].mean()
            fixed = ss[ss["assay_id"].isin(common)].groupby("assay_id")[value].mean()
            row = dict(zip(group_keys, keys))
            row.update({
                "support_size":     s,
                "naive_mean":       round(float(naive.mean()), 5) if len(naive) else np.nan,
                "naive_n_assays":   int(naive.shape[0]),
                "fixed_set_mean":   round(float(fixed.mean()), 5) if len(fixed) else np.nan,
                "fixed_set_n":      int(len(common)),
            })
            out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    # ===== CONFIG - edit these, then run (no arguments) =====
    CSV         = r"D:\Thesis\PTN\outputs\results\fsmol_gnn_classification_random_seed0\baseline_grid.csv"
    VALUE       = "delta_auprc"   # column to summarise
    EXTRA_GROUP = "head"          # extra column to split curves by (e.g. "head"); None for none
    OUT         = None            # path to save the summary CSV, or None to only print
    # =======================================================
    df = pd.read_csv(CSV)
    summ = fixed_set_summary(df, value=VALUE, extra_group=EXTRA_GROUP)
    pd.set_option("display.width", 160)
    print(summ.to_string(index=False))
    if OUT:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        summ.to_csv(OUT, index=False)
        print(f"\nSaved -> {OUT}")
