"""
Thresholding grid: 4 datasets x 8 options.
=========================================

ONE QUESTION: for every dataset and every defensible way of turning continuous activity
values into active/inactive labels, how balanced are the resulting tasks, and how many
are usable for few-shot episodes?

Option 0 is the label the dataset already ships (see TASKS.md 4.0 for who made it).
Options 1-7 are computed by us from the dataset's continuous column, so every dataset
with a continuous value can be compared on equal terms.

WHAT MAKES AN ASSAY UNUSABLE
----------------------------
An episode at support size n needs n actives AND n inactives in the support set, plus
query molecules left over. So an assay is usable at n only if
    n_active >= n  AND  n_inactive >= n        (the "strict" criterion)
Failure modes, reported separately because they have different causes and fixes:

  no_value     no continuous value exists, so we cannot threshold it at all
               (some FS-Mol assays ship only a binary Property)
  rule_dropped the thresholding option itself discarded the assay
               (option 5 drops assays whose median is outside the window)
  dead         single-class after thresholding: 0 actives or 0 inactives.
               Unusable at EVERY n. The threshold sits outside the data's range.
  too_skewed   both classes present, but the minority class has < n members.
               The assay has signal, the threshold is just badly placed.
  too_small    fewer than 2n molecules in total. No threshold can fix this;
               the assay is simply too small.
  usable       n_active >= n and n_inactive >= n

`too_skewed` is the interesting one: it is the failure mode that a better threshold
actually repairs. `dead` is the catastrophic version of the same thing, and `too_small`
is a property of the data that no rule can change.

Usage
-----
    python Analysis/data/threshold_grid.py                   # all 4 datasets
    python Analysis/data/threshold_grid.py --fsmol-sample 0  # all 26,868 FS-Mol train files (slow)
    python Analysis/data/threshold_grid.py --n-support 32
"""

import argparse
import collections
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import (CUSTOM_DIR, DATA_ANALYSIS_RESULTS_DIR, DRUGOOD_DIR,  # noqa: E402
                    FSMOL_TRAIN)

MIN_MOL = 20          # assays smaller than this are excluded from the grid entirely
BAND = 0.3            # option 6: log units dropped either side of the threshold
TARGET_ACTIVE = 0.30  # option 7: target hit rate


# ---------------------------------------------------------------------------
# The eight options
# ---------------------------------------------------------------------------

def _lab(v, thr):
    return (v >= thr).astype(int)


def _windowed(v, lo, hi, mode):
    med = float(np.median(v))
    if lo <= med <= hi:
        thr = med
    elif mode == "fixed":
        thr = 5.0
    elif mode == "clamp":
        thr = float(np.clip(med, lo, hi))
    else:
        return None, None               # drop the assay
    return _lab(v, thr), np.ones(len(v), bool)


def o1_fixed(v):
    return _lab(v, 5.0), np.ones(len(v), bool)


def o2_pure_median(v):
    return _lab(v, float(np.median(v))), np.ones(len(v), bool)


def o3_win_fixed(v):
    return _windowed(v, 4, 6, "fixed")


def o4_win_clamp(v):
    return _windowed(v, 4, 6, "clamp")


def o5_win_drop(v):
    return _windowed(v, 4, 6, "drop")


def o6_band(v):
    thr = float(np.median(v))
    keep = np.abs(v - thr) > BAND
    if keep.sum() < 4:
        return None, None
    return _lab(v[keep], thr), keep


def o7_quantile(v):
    return _lab(v, float(np.quantile(v, 1 - TARGET_ACTIVE))), np.ones(len(v), bool)


def o8_fsmol_enzyme(v):
    """FS-Mol's enzyme window. Its [4,6] non-enzyme counterpart is option 3.
    Replaced an earlier invented [3.5,7.5] window: pXC50 3.5 is 316 uM, which has no
    basis in the literature for affinity data (TASKS.md 4.7).
    """
    return _windowed(v, 5, 7, "fixed")


OPTIONS = [
    ("1 fixed 5.0",                o1_fixed),
    ("2 pure median (no window)",  o2_pure_median),
    ("3 median+[4,6] -> fixed 5",  o3_win_fixed),
    ("4 median+[4,6] -> clamp",    o4_win_clamp),
    ("5 median+[4,6] -> drop",     o5_win_drop),
    ("6 median, drop +/-0.3 band", o6_band),
    ("7 quantile @30% active",     o7_quantile),
    ("8 FS-Mol enzyme [5,7]->fix5", o8_fsmol_enzyme),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(assays, n_support):
    """assays: list of (continuous_values | None, provided_binary | None)."""
    rows = []

    def tally(labels_per_assay, n_no_value, n_rule_dropped, n_mol_kept, n_mol_total):
        fr, cls = [], collections.Counter()
        for y in labels_per_assay:
            npos, nneg = int((y == 1).sum()), int((y == 0).sum())
            fr.append(npos / (npos + nneg))
            if npos == 0 or nneg == 0:
                cls["dead"] += 1
            elif (npos + nneg) < 2 * n_support:
                cls["too_small"] += 1
            elif npos < n_support or nneg < n_support:
                cls["too_skewed"] += 1
            else:
                cls["usable"] += 1
        a = np.array(fr) if fr else np.array([0.0])
        tot = len(labels_per_assay) + n_no_value + n_rule_dropped
        return {
            "assays_scored": len(labels_per_assay),
            "no_value": n_no_value,
            "rule_dropped": n_rule_dropped,
            "median_active_frac": round(float(np.median(a)), 3) if fr else None,
            "mean_active_frac": round(float(a.mean()), 3) if fr else None,
            "p10_active_frac": round(float(np.percentile(a, 10)), 3) if fr else None,
            "p90_active_frac": round(float(np.percentile(a, 90)), 3) if fr else None,
            "balanced_30_70_pct": round(100 * float(np.mean((a >= .3) & (a <= .7))), 1) if fr else 0.0,
            "dead": cls["dead"],
            "too_small": cls["too_small"],
            "too_skewed": cls["too_skewed"],
            "usable": cls["usable"],
            "usable_pct_of_all": round(100 * cls["usable"] / tot, 1) if tot else 0.0,
            "molecules_kept_pct": round(100 * n_mol_kept / n_mol_total, 1) if n_mol_total else 0.0,
        }

    # ---- option 0: the shipped label ------------------------------------
    labs, no_val, kept, total = [], 0, 0, 0
    for v, prov in assays:
        total += len(v) if v is not None else (len(prov) if prov is not None else 0)
        if prov is None:
            no_val += 1
            continue
        labs.append(np.asarray(prov, int))
        kept += len(prov)
    r = {"option": "0 shipped label (default)"}
    r.update(tally(labs, no_val, 0, kept, total))
    r["no_value"] = no_val          # here it means "dataset ships no binary label"
    rows.append(r)

    # ---- options 1-8 ----------------------------------------------------
    for name, fn in OPTIONS:
        labs, no_val, dropped, kept, total = [], 0, 0, 0, 0
        for v, prov in assays:
            n_here = len(v) if v is not None else (len(prov) if prov is not None else 0)
            total += n_here
            if v is None:
                no_val += 1
                continue
            y, keep = fn(v)
            if y is None:
                dropped += 1
                continue
            labs.append(y)
            kept += int(keep.sum())
        r = {"option": name}
        r.update(tally(labs, no_val, dropped, kept, total))
        rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loaders: each returns list of (continuous p-scale values, provided binary)
# ---------------------------------------------------------------------------

# Field-targeted extraction instead of json.loads: each record carries a full molecular
# graph and a 2048-bit fingerprint array, so parsing the whole record costs ~22x more
# than the two fields needed here. Values are JSON *strings* ("Property": "1.0").
# Verified byte-identical to json.loads before adoption; see fsmol_rule_funnel.py.
_NUM = rb'(?:"[^"]*"|[-\d.eE+]+|null)'
_RE_PROP = re.compile(rb'"Property":\s*(' + _NUM + rb')')
_RE_LOGREG = re.compile(rb'"LogRegressionProperty":\s*(' + _NUM + rb')')


def _num(g):
    g = g.strip(b'"')
    if g in (b"", b"null"):
        return None
    try:
        return float(g)
    except ValueError:
        return None


def load_fsmol(sample):
    files = sorted(os.path.join(FSMOL_TRAIN, f) for f in os.listdir(FSMOL_TRAIN)
                   if f.endswith(".jsonl.gz"))
    if sample and len(files) > sample:
        rng = np.random.RandomState(0)
        files = [files[i] for i in sorted(rng.choice(len(files), sample, replace=False))]
        print(f"  FS-Mol: sampling {sample:,} files")
    else:
        print(f"  FS-Mol: reading ALL {len(files):,} train files")
    out = []
    for i, f in enumerate(files):
        P, L = [], []
        try:
            with gzip.open(f, "rb") as fh:
                for line in fh:
                    m = _RE_PROP.search(line)
                    if m is None:
                        continue
                    pv = _num(m.group(1))
                    if pv is None:
                        continue
                    P.append(pv)
                    m2 = _RE_LOGREG.search(line)
                    if m2 is not None:
                        lv = _num(m2.group(1))
                        if lv is not None:
                            L.append(lv)
        except Exception:
            continue
        if len(P) < MIN_MOL:
            continue
        # keep the continuous column only when it covers the whole assay
        v = np.asarray(L, float) if len(L) == len(P) else None
        out.append((v, np.asarray(P, int)))
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,}/{len(files):,}", flush=True)
    return out


def load_drugood():
    out = []
    for fn in ("lbap_core_ic50_scaffold.json",):
        p = os.path.join(DRUGOOD_DIR, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        by = collections.defaultdict(lambda: ([], []))
        for part in ("train", "ood_val", "ood_test", "iid_val", "iid_test"):
            for e in d["split"].get(part, []):
                reg, cls = by[e["assay_id"]]
                reg.append(float(e["reg_label"]))
                cls.append(int(e["cls_label"]))
        for reg, cls in by.values():
            if len(reg) >= MIN_MOL:
                out.append((np.asarray(reg, float), np.asarray(cls, int)))
    return out


def load_curated_mic():
    df = pd.read_csv(os.path.join(CUSTOM_DIR, "curated_mic_assays_balanced.csv"),
                     usecols=["assay_chembl_id", "pmic", "is_potent"]).dropna()
    out = []
    for _, g in df.groupby("assay_chembl_id"):
        if len(g) >= MIN_MOL:
            out.append((g["pmic"].to_numpy(float), g["is_potent"].to_numpy(int)))
    return out


def load_chembl36():
    P_SCALE = {"pchembl_value", "pmic", "pmic_nm"}
    acc = collections.defaultdict(list)
    for ch in pd.read_csv(os.path.join(CUSTOM_DIR, "chembl36_curated.csv"),
                          usecols=["assay_chembl_id", "standard_type",
                                   "target_transformed", "target_transformed_type"],
                          chunksize=500_000, low_memory=False):
        tt = ch["target_transformed_type"].fillna("")
        val = pd.to_numeric(ch["target_transformed"], errors="coerce")
        m = tt.isin(P_SCALE) & val.notna()
        if not m.any():
            continue
        sub = ch.loc[m, ["assay_chembl_id"]].assign(_st=ch.loc[m, "standard_type"], _v=val[m])
        for k, g in sub.groupby(["assay_chembl_id", "_st"]):
            acc[k].extend(g["_v"].to_numpy())
    # chembl36 ships NO binary label -> provided is None
    return [(np.asarray(v, float), None) for v in acc.values() if len(v) >= MIN_MOL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-support", type=int, default=16)
    ap.add_argument("--fsmol-sample", type=int, default=0,
                    help="0 = all 26,868 train files (default)")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    loaders = [
        ("FS-Mol (train)", lambda: load_fsmol(args.fsmol_sample)),
        ("DrugOOD (ic50_scaffold)", load_drugood),
        ("curated_mic", load_curated_mic),
        ("chembl36 (p-scale)", load_chembl36),
    ]

    out = DATA_ANALYSIS_RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    allres = []
    for name, fn in loaders:
        if name.split()[0] in args.skip:
            continue
        print(f"\nloading {name} ...", flush=True)
        assays = fn()
        if not assays:
            print("  nothing loaded, skipped")
            continue
        df = score(assays, args.n_support)
        df.insert(0, "dataset", name)
        allres.append(df)
        print(f"\n{'=' * 118}\n{name}  "
              f"({len(assays):,} assays with >={MIN_MOL} molecules, n_support={args.n_support})\n{'=' * 118}")
        show = df[["option", "assays_scored", "median_active_frac", "mean_active_frac",
                   "balanced_30_70_pct", "dead", "too_small", "too_skewed", "usable",
                   "no_value", "rule_dropped", "molecules_kept_pct"]]
        print(show.to_string(index=False))

    if allres:
        full = pd.concat(allres, ignore_index=True)
        p = os.path.join(out, f"threshold_grid_n{args.n_support}.csv")
        full.to_csv(p, index=False)
        print(f"\nfull grid -> {p}")


if __name__ == "__main__":
    main()
