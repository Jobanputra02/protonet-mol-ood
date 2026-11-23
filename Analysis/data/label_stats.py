"""
Label statistics and thresholding feasibility  -  measurements M1, M2, M7, M17.
==============================================================================

ONE QUESTION: for each dataset, can we binarise the activity labels sensibly, and how
many usable few-shot tasks survive if we do?

This is the evidence base for TASKS.md section 4 (thresholding) and section 5 (sampling).
Nothing here trains anything; it reads raw data only.

Produces five tables per dataset:

  T1  MEASUREMENT TYPES (M17, chembl36 only)
      standard_type x standard_units x target_transformed_type, with how many rows have
      a pChEMBL value. Decides which threshold rules we need at all.

  T2  SCALE AUDIT (chembl36 only)
      Value range and correlation with pchembl_value per (type, transform). This is the
      table that shows `target_transformed` is NOT one scale: IC50/log10_transformed
      correlates -0.99 with pchembl, i.e. its direction is INVERTED.

  T3  ACTIVE FRACTION PER ASSAY (M2)
      Distribution of the per-assay active fraction under each thresholding rule.
      Answers "how balanced is the dataset", and tells us whether proportional support
      sampling (TASKS.md 5.3) differs much from 50/50 in practice.

  T4  THRESHOLD FEASIBILITY
      Per assay, the median p-value and whether it falls inside the allowed window.
      FS-Mol (Stanley et al. 2021) uses median pXC50 clamped to [5,7] for enzyme targets
      and [4,6] otherwise, substituting a fixed 5.0 when the median falls outside.
      DrugOOD uses the simpler [4,6]-or-5.0. We report both, plus the clamp-vs-fixed
      difference (M6), because that choice changes the class balance.

  T5  QUALIFYING TASKS (M1, M7)
      How many assays support an episode at n_support = 16/32/64/128, under
        strict:  pos >= n AND neg >= n     <- what a class-balanced support set needs
        loose:   total >= 2n               <- what Analysis/data/assay_size_table.py used
      The gap between these two is why the old "24,009 FS-Mol train assays at n=16"
      figure was wrong; the strict count is ~11,900.
      Also reports retention under FS-Mol's published active-ratio filter [0.30, 0.70].

Usage
-----
    python Analysis/data/label_stats.py                      # MIC + chembl36 (fast)
    python Analysis/data/label_stats.py --fsmol              # add FS-Mol (slow)
    python Analysis/data/label_stats.py --fsmol-sample 3000  # cap FS-Mol train files

CSVs are written to outputs/data_analysis/csvs/.
"""

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import CUSTOM_DIR, DATA_ANALYSIS_RESULTS_DIR, FSMOL_TRAIN, FSMOL_VAL, FSMOL_TEST  # noqa: E402

N_SUPPORTS = (16, 32, 64, 128)
CHUNK = 500_000

# ---------------------------------------------------------------------------
# Threshold rules
# ---------------------------------------------------------------------------
# A p-scale value (pChEMBL, pMIC, pXC50) is "higher = more active". A percent-inhibition
# value is also "higher = more active" but lives on a 0-100 scale where the [4,6] window
# is meaningless, so it gets a plain median.
#
# target_transformed_type tells us which scale a chembl36 row is on. Only the p-scale
# transforms are comparable; `log10_transformed` is the raw log10 of the concentration,
# so it points the WRONG WAY, and `standard_value` is the raw measurement.
P_SCALE_TRANSFORMS = {"pchembl_value", "pmic", "pmic_nm"}
PERCENT_TRANSFORMS = {"standard_value"}      # only meaningful for % types

PERCENT_TYPES = {"Inhibition", "Activity", "Residual Activity", "GI", "Emax",
                 "Stability", "Efficacy"}
# Not target-activity measures: ADME/PK properties. A "task" of predicting clearance is
# a different prediction problem and does not belong in an activity benchmark.
NON_ACTIVITY_TYPES = {"CL", "T1/2", "Solubility", "Stability", "Vdss", "Papp",
                      "Cmax", "AUC", "F", "Tmax", "MRT", "Clearance"}


def fsmol_style_threshold(vals, is_enzyme=False, mode="fixed"):
    """FS-Mol / DrugOOD adaptive threshold for a single assay.

    vals : p-scale values for that assay
    is_enzyme : FS-Mol widens the window to [5,7] for enzyme targets
    mode : what to do when the median falls outside the window
           "fixed" - substitute 5.0   (what FS-Mol and DrugOOD both publish)
           "clamp" - clamp to the nearest bound
    Returns (threshold, in_window) so callers can count how often the fallback fires.
    """
    lo, hi = (5.0, 7.0) if is_enzyme else (4.0, 6.0)
    med = float(np.median(vals))
    if lo <= med <= hi:
        return med, True
    return (5.0 if mode == "fixed" else float(np.clip(med, lo, hi))), False


def summarise_fractions(fracs, label):
    """Percentile summary of a list of per-assay active fractions (M2)."""
    a = np.asarray([f for f in fracs if np.isfinite(f)])
    if a.size == 0:
        return {"group": label, "n_assays": 0}
    return {
        "group": label, "n_assays": int(a.size),
        "mean": round(float(a.mean()), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.median(a)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        # FS-Mol's published retention filter
        "frac_in_030_070": round(float(np.mean((a >= 0.30) & (a <= 0.70))), 4),
        "frac_single_class": round(float(np.mean((a == 0) | (a == 1))), 4),
    }


def qualifying_table(per_assay, label):
    """M1 / M7: assays supporting an episode at each n_support, strict vs loose."""
    rows = []
    n_tot = len(per_assay)
    for n in N_SUPPORTS:
        strict = sum(1 for p, q in per_assay if p >= n and q >= n)
        loose = sum(1 for p, q in per_assay if (p + q) >= 2 * n)
        rows.append({
            "group": label, "n_support": n, "n_assays_total": n_tot,
            "strict_pos_and_neg_ge_n": strict,
            "strict_pct": round(100 * strict / n_tot, 2) if n_tot else 0.0,
            "loose_total_ge_2n": loose,
            "loose_pct": round(100 * loose / n_tot, 2) if n_tot else 0.0,
            # how much the loose criterion overstates usable tasks
            "loose_overstates_by": loose - strict,
        })
    return rows


# ===========================================================================
# chembl36
# ===========================================================================

def analyse_chembl36(out):
    path = os.path.join(CUSTOM_DIR, "chembl36_curated.csv")
    if not os.path.exists(path):
        print(f"  skipped: {path} not found")
        return
    print("\n" + "=" * 78)
    print("chembl36_curated.csv")
    print("=" * 78)

    cols = ["assay_chembl_id", "standard_type", "standard_units",
            "target_transformed", "target_transformed_type", "pchembl_value",
            "assay_organism", "group"]

    # ---- T1 + T2: one pass for measurement types and scale audit ----------
    type_rows = collections.Counter()
    scale_acc = collections.defaultdict(list)
    pchembl_avail = collections.Counter()
    # ---- per-(assay, standard_type) p-values for T3/T4/T5 -----------------
    assay_vals = collections.defaultdict(list)
    assay_pct = collections.defaultdict(list)
    n_rows = 0

    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        n_rows += len(ch)
        st = ch["standard_type"].fillna("<NA>")
        tt = ch["target_transformed_type"].fillna("<none>")
        un = ch["standard_units"].fillna("<none>")
        type_rows.update(collections.Counter(zip(st, un, tt)))
        for t, g in ch.groupby(st):
            pchembl_avail[t] += int(g["pchembl_value"].notna().sum())

        val = pd.to_numeric(ch["target_transformed"], errors="coerce")
        # scale audit sample
        for k, g in ch.assign(_st=st, _tt=tt, _v=val).groupby(["_st", "_tt"]):
            if sum(len(x) for x in scale_acc[k]) < 40_000:
                scale_acc[k].append(g[["_v", "pchembl_value"]])

        # p-scale rows only, keyed by (assay, standard_type) - the task definition
        # recommended in TASKS.md 3.3: one task has one unit and one threshold rule.
        m_p = tt.isin(P_SCALE_TRANSFORMS) & val.notna() & ~st.isin(NON_ACTIVITY_TYPES)
        if m_p.any():
            sub = ch.loc[m_p, ["assay_chembl_id"]].assign(_st=st[m_p], _v=val[m_p])
            for k, g in sub.groupby(["assay_chembl_id", "_st"]):
                assay_vals[k].extend(g["_v"].to_numpy())

        # percent-inhibition rows
        m_c = (st.isin(PERCENT_TYPES) & tt.isin(PERCENT_TRANSFORMS)
               & val.notna() & (un == "%"))
        if m_c.any():
            sub = ch.loc[m_c, ["assay_chembl_id"]].assign(_st=st[m_c], _v=val[m_c])
            for k, g in sub.groupby(["assay_chembl_id", "_st"]):
                assay_pct[k].extend(g["_v"].to_numpy())

    print(f"\nrows: {n_rows:,}")

    # ---- T1 ----------------------------------------------------------------
    t1 = pd.DataFrame(
        [{"standard_type": t, "standard_units": u, "target_transformed_type": v,
          "rows": n} for (t, u, v), n in type_rows.items()]
    ).sort_values("rows", ascending=False)
    t1["pct_of_all"] = (100 * t1["rows"] / n_rows).round(3)
    t1.to_csv(os.path.join(out, "chembl36_T1_measurement_types.csv"), index=False)
    print("\n-- T1 measurement types (top 15 of "
          f"{t1['standard_type'].nunique()} types) --")
    print(t1.head(15).to_string(index=False))

    # ---- T2 ----------------------------------------------------------------
    t2 = []
    for k, parts in scale_acc.items():
        d = pd.concat(parts)
        v = pd.to_numeric(d["_v"], errors="coerce").dropna()
        if len(v) < 50:
            continue
        sub = d.dropna(subset=["_v", "pchembl_value"])
        corr = np.nan
        if len(sub) > 50:
            corr = float(np.corrcoef(
                pd.to_numeric(sub["_v"], errors="coerce"),
                pd.to_numeric(sub["pchembl_value"], errors="coerce"))[0, 1])
        t2.append({
            "standard_type": k[0], "target_transformed_type": k[1],
            "n_sampled": len(v),
            "p01": round(float(np.percentile(v, 1)), 3),
            "median": round(float(np.median(v)), 3),
            "p99": round(float(np.percentile(v, 99)), 3),
            "corr_with_pchembl": round(corr, 4) if np.isfinite(corr) else None,
            "p_scale": k[1] in P_SCALE_TRANSFORMS,
        })
    t2 = pd.DataFrame(t2).sort_values("n_sampled", ascending=False)
    t2.to_csv(os.path.join(out, "chembl36_T2_scale_audit.csv"), index=False)
    print("\n-- T2 scale audit: is target_transformed one scale? --")
    print(t2.head(14).to_string(index=False))
    bad = t2[(t2["corr_with_pchembl"].notna()) & (t2["corr_with_pchembl"] < 0)]
    if len(bad):
        n_bad = int(bad["n_sampled"].sum())
        print(f"\n  !! {len(bad)} (type, transform) combinations correlate NEGATIVELY "
              f"with pchembl_value.\n     Their target_transformed is direction-INVERTED: "
              f"thresholding it as if it were\n     a p-value flips active and inactive. "
              f"Affected combinations: "
              f"{', '.join(f'{r.standard_type}/{r.target_transformed_type}' for r in bad.itertuples())}")

    # ---- T3/T4/T5 ----------------------------------------------------------
    _assay_tables(assay_vals, assay_pct, "chembl36", out)


# ===========================================================================
# curated_mic
# ===========================================================================

def analyse_curated_mic(out):
    path = os.path.join(CUSTOM_DIR, "curated_mic_assays_balanced.csv")
    if not os.path.exists(path):
        print(f"  skipped: {path} not found")
        return
    print("\n" + "=" * 78)
    print("curated_mic_assays_balanced.csv")
    print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    print(f"rows: {len(df):,}   assays: {df['assay_chembl_id'].nunique():,}")

    vals = {(a, "MIC"): g["pmic"].dropna().to_numpy()
            for a, g in df.groupby("assay_chembl_id")}

    # The shipped is_potent column uses a FIXED pMIC >= 5, which contradicts the
    # adaptive-threshold decision. Quantify how the two differ.
    fixed = []
    for a, g in df.groupby("assay_chembl_id"):
        y = g["is_potent"].dropna()
        if len(y):
            fixed.append(float(y.mean()))
    print("\n-- shipped is_potent (FIXED pMIC >= 5) vs adaptive --")
    print(pd.DataFrame([summarise_fractions(fixed, "fixed pMIC>=5 (shipped)")]
                       ).to_string(index=False))

    _assay_tables(vals, {}, "curated_mic", out)


# ===========================================================================
# FS-Mol  (labels are already binary in the JSONL)
# ===========================================================================

def analyse_fsmol(out, sample):
    from data import load_fsmol_assay
    print("\n" + "=" * 78)
    print("FS-Mol  (labels already binary; no thresholding needed)")
    print("=" * 78)
    frac_rows, qual_rows = [], []
    for name, d in (("train", FSMOL_TRAIN), ("valid", FSMOL_VAL), ("test", FSMOL_TEST)):
        if not os.path.isdir(d):
            continue
        files = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if f.endswith(".jsonl.gz"))
        if name == "train" and sample and len(files) > sample:
            rng = np.random.RandomState(0)
            files = [files[i] for i in sorted(rng.choice(len(files), sample, replace=False))]
            print(f"  {name}: sampling {sample:,} of the full file list")
        fr, pa = [], []
        for i, f in enumerate(files):
            try:
                ds, _ = load_fsmol_assay(f)
            except Exception:
                continue
            if ds.binary_labels is None or len(ds) == 0:
                continue
            y = np.asarray(ds.binary_labels)
            y = y[np.isin(y, [0, 1])]
            if y.size == 0:
                continue
            fr.append(float(y.mean()))
            pa.append((int((y == 1).sum()), int((y == 0).sum())))
            if (i + 1) % 2000 == 0:
                print(f"    {name}: {i + 1:,}/{len(files):,}", flush=True)
        print(f"  {name}: {len(fr):,} assays with binary labels")
        frac_rows.append(summarise_fractions(fr, f"fsmol_{name}"))
        qual_rows += qualifying_table(pa, f"fsmol_{name}")
    _write_and_show(frac_rows, qual_rows, [], "fsmol", out)


# ===========================================================================
# shared table builder
# ===========================================================================

def _assay_tables(assay_vals, assay_pct, name, out):
    """T3/T4/T5 from per-assay p-scale values (and optional percent values)."""
    frac_rows, qual_rows, thr_rows = [], [], []

    for rule, enzyme, mode in (("drugood_4_6_fixed5", False, "fixed"),
                               ("drugood_4_6_clamp", False, "clamp"),
                               ("fsmol_enzyme_5_7_fixed5", True, "fixed")):
        fracs, per_assay, in_win = [], [], []
        for k, v in assay_vals.items():
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v)]
            if v.size < 2:
                continue
            thr, ok = fsmol_style_threshold(v, is_enzyme=enzyme, mode=mode)
            y = (v >= thr).astype(int)
            fracs.append(float(y.mean()))
            per_assay.append((int((y == 1).sum()), int((y == 0).sum())))
            in_win.append(ok)
        if not per_assay:
            continue
        frac_rows.append(summarise_fractions(fracs, f"{name} | {rule}"))
        qual_rows += qualifying_table(per_assay, f"{name} | {rule}")
        thr_rows.append({
            "group": name, "rule": rule, "n_assays": len(in_win),
            "median_inside_window_pct": round(100 * float(np.mean(in_win)), 2),
            "fallback_fired_pct": round(100 * (1 - float(np.mean(in_win))), 2),
        })

    # percent-inhibition assays get a plain median threshold
    if assay_pct:
        fracs, per_assay = [], []
        for k, v in assay_pct.items():
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v)]
            if v.size < 2:
                continue
            y = (v >= float(np.median(v))).astype(int)
            fracs.append(float(y.mean()))
            per_assay.append((int((y == 1).sum()), int((y == 0).sum())))
        if per_assay:
            frac_rows.append(summarise_fractions(fracs, f"{name} | percent_median"))
            qual_rows += qualifying_table(per_assay, f"{name} | percent_median")

    _write_and_show(frac_rows, qual_rows, thr_rows, name, out)


def _write_and_show(frac_rows, qual_rows, thr_rows, name, out):
    if frac_rows:
        d = pd.DataFrame(frac_rows)
        d.to_csv(os.path.join(out, f"{name}_T3_active_fraction.csv"), index=False)
        print(f"\n-- T3 active fraction per assay (M2) [{name}] --")
        print(d.to_string(index=False))
    if thr_rows:
        d = pd.DataFrame(thr_rows)
        d.to_csv(os.path.join(out, f"{name}_T4_threshold_feasibility.csv"), index=False)
        print(f"\n-- T4 does the median fall inside the window? [{name}] --")
        print(d.to_string(index=False))
    if qual_rows:
        d = pd.DataFrame(qual_rows)
        d.to_csv(os.path.join(out, f"{name}_T5_qualifying.csv"), index=False)
        print(f"\n-- T5 qualifying tasks, strict vs loose (M1/M7) [{name}] --")
        print(d.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsmol", action="store_true", help="include FS-Mol (slow)")
    ap.add_argument("--fsmol-sample", type=int, default=3000,
                    help="cap FS-Mol train files (0 = all 26,868)")
    ap.add_argument("--skip-chembl36", action="store_true")
    args = ap.parse_args()

    out = DATA_ANALYSIS_RESULTS_DIR
    os.makedirs(out, exist_ok=True)

    analyse_curated_mic(out)
    if not args.skip_chembl36:
        analyse_chembl36(out)
    if args.fsmol:
        analyse_fsmol(out, args.fsmol_sample)

    print(f"\nCSVs -> {out}")


if __name__ == "__main__":
    main()
