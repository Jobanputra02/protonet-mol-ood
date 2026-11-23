"""
Split audit for curated_mic: M4 (compound leakage), M5 (time pivot), and the
per-split count of assays that can actually form an episode.
=====================================================================

ONE QUESTION: are splits A-E as out-of-distribution as their names suggest, and how many
usable tasks does each actually leave?

Three things, all from `curated_mic_assays_balanced.csv` + `mic_splits.json`:

  M4  COMPOUND LEAKAGE.  Splits A-E keep *assays* disjoint, but 42.5% of curated_mic
      compounds appear in more than one assay (the same antibiotic tested against
      several organisms). So a test assay can be full of molecules the model already
      saw during training, under a different assay id. If that is common, the split is
      far weaker OOD than its name claims, and splits D/E (organism / Gram stain)
      become the only credible OOD evidence.

  M5  TIME PIVOT.  Split C is "year < 2022 -> train, 2022-2023 -> val, >= 2024 -> test"
      (data/custom/mic_splits.py). A time split is only credible if the pivot is
      justified by the year distribution rather than picked, so this reports the
      histogram and what alternative pivots would give.

  QUALIFYING COUNTS.  The split sizes recorded in mic_splits.json count *all* assays.
      TASKS.md 4.6.5 showed only 1,302 of 2,323 curated_mic assays can form a balanced
      support set at n=16 (and 376 at n=32). So the usable size of each split is much
      smaller than its nominal size. This recomputes it per split.

Labels: the adaptive per-assay rule from TASKS.md 4.7.7 (median pMIC, window [4,6] -
curated_mic targets are whole bacteria, not enzymes - with FS-Mol's fixed-5.0 fallback).
Pass --fixed to use the shipped is_potent column instead, for comparison.

Usage
-----
    python Analysis/data/split_audit.py
    python Analysis/data/split_audit.py --n-support 32
    python Analysis/data/split_audit.py --fixed
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import CUSTOM_DIR, DATA_ANALYSIS_RESULTS_DIR   # noqa: E402

WINDOW = (4.0, 6.0)       # curated_mic targets are bacteria -> non-enzyme window


def adaptive_labels(v):
    """TASKS.md 4.7.7: per-assay median, window [4,6], fixed-5.0 fallback."""
    med = float(np.median(v))
    thr = med if WINDOW[0] <= med <= WINDOW[1] else 5.0
    return (v >= thr).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-support", type=int, default=16)
    ap.add_argument("--fixed", action="store_true",
                    help="use the shipped is_potent column instead of the adaptive rule")
    args = ap.parse_args()
    n = args.n_support

    df = pd.read_csv(os.path.join(CUSTOM_DIR, "curated_mic_assays_balanced.csv"),
                     usecols=["assay_chembl_id", "compound_chembl_id",
                              "document_chembl_id", "year", "organism_canonical",
                              "l2", "pmic", "is_potent"])
    splits = json.load(open(os.path.join(CUSTOM_DIR, "mic_splits.json")))

    # ---------- per-assay label counts ---------------------------------
    qual = {}
    for aid, g in df.groupby("assay_chembl_id"):
        if args.fixed:
            y = g["is_potent"].dropna().astype(int).to_numpy()
        else:
            v = g["pmic"].dropna().to_numpy(float)
            y = adaptive_labels(v) if v.size >= 2 else np.array([], int)
        if y.size:
            qual[aid] = (int((y == 1).sum()), int((y == 0).sum()))

    compounds = df.groupby("assay_chembl_id")["compound_chembl_id"].apply(set).to_dict()

    # ================= M4: compound leakage ============================
    print("=" * 104)
    print(f"M4  COMPOUND LEAKAGE  (assays are disjoint by construction; are the "
          f"MOLECULES?)   n_support={n}")
    print("=" * 104)
    rows4 = []
    for name in ("A", "B", "C", "D", "E"):
        sp = splits[name]
        tr = set().union(*[compounds.get(a, set()) for a in sp["train"]]) \
            if sp["train"] else set()
        te_assays = [a for a in sp["test"] if a in compounds]
        te = set().union(*[compounds[a] for a in te_assays]) if te_assays else set()
        shared = tr & te
        # per test assay: what share of its molecules were seen in training?
        per = [len(compounds[a] & tr) / len(compounds[a]) for a in te_assays]
        per = np.array(per) if per else np.array([0.0])
        rows4.append({
            "split": name, "name": sp.get("name", ""),
            "train_compounds": len(tr), "test_compounds": len(te),
            "shared_compounds": len(shared),
            "pct_test_compounds_seen": round(100 * len(shared) / max(1, len(te)), 1),
            "jaccard": round(len(shared) / max(1, len(tr | te)), 3),
            "median_pct_of_test_assay_seen": round(100 * float(np.median(per)), 1),
            "test_assays_fully_seen": int((per >= 0.999).sum()),
            "test_assays_mostly_seen_ge50pct": int((per >= 0.5).sum()),
            "n_test_assays": len(te_assays),
        })
    t4 = pd.DataFrame(rows4)
    print(t4[["split", "name", "n_test_assays", "test_compounds", "shared_compounds",
              "pct_test_compounds_seen", "median_pct_of_test_assay_seen",
              "test_assays_mostly_seen_ge50pct", "test_assays_fully_seen"]]
          .to_string(index=False))

    # ================= M5: time pivot ==================================
    print("\n" + "=" * 104)
    print("M5  TIME PIVOT for split C   (mic_splits.py uses: <2022 train | 2022-23 val "
          "| >=2024 test)")
    print("=" * 104)
    ay = df.groupby("assay_chembl_id")["year"].min()
    yr = ay.dropna().astype(int)
    print(f"  assays with a year: {len(yr):,} of {df.assay_chembl_id.nunique():,} "
          f"({df.assay_chembl_id.nunique() - len(yr)} have none)")
    print(f"  range {yr.min()}-{yr.max()}, median {int(yr.median())}\n")
    print("  year     assays   cumulative%")
    dec = yr.value_counts().sort_index()
    run = 0
    for y_, c in dec.items():
        if y_ < 2008:
            continue
        run = int((yr <= y_).sum())
        bar = "#" * max(1, int(c / 8))
        print(f"  {y_}   {c:>6,}   {100*run/len(yr):>5.1f}%  {bar}")

    rows5 = []
    for pivot in (2012, 2015, 2018, 2020, 2021, 2022, 2023, 2024):
        tr = yr[yr < pivot].index
        te = yr[yr >= pivot].index
        q_tr = sum(1 for a in tr if qual.get(a, (0, 0))[0] >= n and qual.get(a, (0, 0))[1] >= n)
        q_te = sum(1 for a in te if qual.get(a, (0, 0))[0] >= n and qual.get(a, (0, 0))[1] >= n)
        rows5.append({"pivot": pivot, "train_assays": len(tr), "test_assays": len(te),
                      "test_pct": round(100 * len(te) / len(yr), 1),
                      f"train_qualifying_n{n}": q_tr, f"test_qualifying_n{n}": q_te})
    t5 = pd.DataFrame(rows5)
    print("\n  what each pivot would give (two-way, no val):")
    print(t5.to_string(index=False))

    # ================= qualifying per split ============================
    print("\n" + "=" * 104)
    print(f"USABLE SIZE OF EACH SPLIT  (nominal vs assays that can form a balanced "
          f"support at n={n})")
    print("=" * 104)
    rows6 = []
    for name in ("A", "B", "C", "D", "E"):
        sp = splits[name]
        r = {"split": name, "name": sp.get("name", "")}
        for part in ("train", "val", "test"):
            ids = sp[part]
            q = sum(1 for a in ids
                    if qual.get(a, (0, 0))[0] >= n and qual.get(a, (0, 0))[1] >= n)
            r[f"{part}_nominal"] = len(ids)
            r[f"{part}_usable"] = q
        rows6.append(r)
    t6 = pd.DataFrame(rows6)
    print(t6.to_string(index=False))

    out = DATA_ANALYSIS_RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    sfx = "fixed" if args.fixed else "adaptive"
    t4.to_csv(os.path.join(out, f"split_audit_M4_leakage_{sfx}_n{n}.csv"), index=False)
    t5.to_csv(os.path.join(out, f"split_audit_M5_pivot_{sfx}_n{n}.csv"), index=False)
    t6.to_csv(os.path.join(out, f"split_audit_usable_{sfx}_n{n}.csv"), index=False)
    print(f"\nCSVs -> {out}")


if __name__ == "__main__":
    main()
