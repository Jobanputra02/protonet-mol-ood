"""
FS-Mol's published thresholding rule, applied to all four datasets: where do assays go?
======================================================================================

ONE QUESTION: if we adopt FS-Mol's own published rule (Stanley et al. 2021) rather than
inventing one, what active/inactive ratio do we get, how many assays survive, and at
which stage are the rest lost?

THE RULE (Stanley et al. 2021, NeurIPS Datasets & Benchmarks)
-------------------------------------------------------------
  1. threshold = median pXC50 of the compounds in that assay
  2. the median must lie inside a target-class window:
        enzyme targets      [5, 7]
        other protein targets [4, 6]
  3. if the median lies outside the window, substitute a FIXED pXC50 = 5.0
     (FS-Mol substitutes; it does not clamp - verified empirically in TASKS.md 4.1)
  4. assays whose median is outside the window AND which end up strongly
     class-imbalanced are filtered out; the published set keeps assays with an
     active ratio in [0.30, 0.70]

CAVEAT ON STEP 2
----------------
The enzyme/non-enzyme split needs a protein-family classification that none of our four
files carries (chembl36 has target_chembl_id but no family; FS-Mol's AssayType is
B/A/F, i.e. assay format, not target class). So this script reports the rule under BOTH
windows as a range, and treats curated_mic as non-enzyme on chemical grounds: its
targets are whole bacterial cells, not purified enzymes, so [4, 6] is the correct window
there and the [5, 7] row is shown only for completeness.

THE FUNNEL
----------
Stages are applied in order, so each loss is counted exactly once and the numbers add
up. Order matters: a single-class assay is also skewed, but it is reported as `dead`
because that is the more fundamental failure.

  start            assays with at least 2 labelled molecules
  -no_value        no continuous p-value exists -> cannot threshold at all
  -too_small       fewer than 2*n molecules -> no episode is possible at this n,
                   whatever the threshold. Irreducible: a data property.
  -dead            single-class after thresholding -> the threshold fell outside the
                   data's range. Unusable at EVERY n. Threshold placement is the cause.
  -ratio_filter    active ratio outside [0.30, 0.70] -> FS-Mol's own retention filter.
                   These assays ARE usable, they are just not balanced enough for the
                   published protocol. This is a deliberate quality/quantity trade.
  -minority_lt_n   both classes present and ratio acceptable, but the minority class
                   still has < n members (small assays at the edge of the ratio band)
  = retained       n_active >= n AND n_inactive >= n

Usage
-----
    python Analysis/data/fsmol_rule_funnel.py
    python Analysis/data/fsmol_rule_funnel.py --n-support 32 --fsmol-sample 0
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

RATIO_LO, RATIO_HI = 0.30, 0.70


def fsmol_threshold(v, window):
    """Return (threshold, median_was_inside_window). FS-Mol substitutes a fixed 5.0."""
    lo, hi = window
    med = float(np.median(v))
    if lo <= med <= hi:
        return med, True
    return 5.0, False


def funnel(assays, window, n_support, label):
    """assays: list of (continuous values | None, n_molecules)."""
    start = len(assays)
    c = collections.Counter()
    ratios_before_filter, ratios_retained = [], []
    fallback_fired = 0

    for v, n_mol in assays:
        if v is None:
            c["no_value"] += 1
            continue
        if n_mol < 2 * n_support:
            c["too_small"] += 1
            continue
        thr, inside = fsmol_threshold(v, window)
        if not inside:
            fallback_fired += 1
        y = (v >= thr).astype(int)
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos == 0 or nneg == 0:
            c["dead"] += 1
            continue
        r = npos / (npos + nneg)
        ratios_before_filter.append(r)
        if not (RATIO_LO <= r <= RATIO_HI):
            c["ratio_filter"] += 1
            continue
        if npos < n_support or nneg < n_support:
            c["minority_lt_n"] += 1
            continue
        c["retained"] += 1
        ratios_retained.append(r)

    rb = np.array(ratios_before_filter) if ratios_before_filter else np.array([np.nan])
    rr = np.array(ratios_retained) if ratios_retained else np.array([np.nan])
    return {
        "dataset": label,
        "window": f"[{window[0]},{window[1]}]",
        "start_assays": start,
        "no_value": c["no_value"],
        "too_small": c["too_small"],
        "dead": c["dead"],
        "ratio_filter": c["ratio_filter"],
        "minority_lt_n": c["minority_lt_n"],
        "retained": c["retained"],
        "retained_pct": round(100 * c["retained"] / start, 1) if start else 0.0,
        "lost_pct": round(100 * (start - c["retained"]) / start, 1) if start else 0.0,
        "fallback_fired_pct": round(100 * fallback_fired / max(1, start - c["no_value"]), 1),
        "median_ratio_pre_filter": round(float(np.nanmedian(rb)), 3),
        "median_ratio_retained": round(float(np.nanmedian(rr)), 3),
        "mean_ratio_retained": round(float(np.nanmean(rr)), 3),
    }


# ---------------------------------------------------------------------------
# Loaders -> list of (continuous p-scale values | None, n_molecules)
# ---------------------------------------------------------------------------

# Field-targeted extraction instead of json.loads. Each FS-Mol record carries a full
# molecular `graph` and a 2048-bit `fingerprints` array; parsing those costs ~22x more
# than the two fields we need, and the full train pool is 26,868 files. Values are
# stored as JSON *strings* ("Property": "1.0"), hence the quoted-or-bare pattern.
# Verified byte-identical to json.loads on 150 files before being adopted.
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
    allf = sorted(os.path.join(FSMOL_TRAIN, f) for f in os.listdir(FSMOL_TRAIN)
                  if f.endswith(".jsonl.gz"))
    files = allf
    if sample and len(allf) > sample:
        rng = np.random.RandomState(0)
        files = [allf[i] for i in sorted(rng.choice(len(allf), sample, replace=False))]
        print(f"  sampling {sample:,} of {len(allf):,} train files "
              f"(scale counts by {len(allf)/sample:.2f} for the full pool)")
    else:
        print(f"  reading ALL {len(allf):,} train files")
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
        if len(P) < 2:
            continue
        out.append((np.asarray(L, float) if len(L) == len(P) else None, len(P)))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1:,}/{len(files):,}", flush=True)
    return out, (len(allf) / len(files) if files else 1.0)


def load_drugood():
    by = collections.defaultdict(list)
    p = os.path.join(DRUGOOD_DIR, "lbap_core_ic50_scaffold.json")
    if not os.path.exists(p):
        return [], 1.0
    d = json.load(open(p))
    for part in ("train", "ood_val", "ood_test", "iid_val", "iid_test"):
        for e in d["split"].get(part, []):
            by[e["assay_id"]].append(float(e["reg_label"]))
    return [(np.asarray(v, float), len(v)) for v in by.values() if len(v) >= 2], 1.0


def load_curated_mic():
    df = pd.read_csv(os.path.join(CUSTOM_DIR, "curated_mic_assays_balanced.csv"),
                     usecols=["assay_chembl_id", "pmic"]).dropna()
    out = [(g.to_numpy(float), len(g)) for _, g in df.groupby("assay_chembl_id")["pmic"]
           if len(g) >= 2]
    return out, 1.0


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
    return [(np.asarray(v, float), len(v)) for v in acc.values() if len(v) >= 2], 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-support", type=int, default=16)
    ap.add_argument("--fsmol-sample", type=int, default=0,
                    help="0 = all 26,868 train files (default)")
    args = ap.parse_args()
    n = args.n_support

    jobs = [
        ("FS-Mol (train)", lambda: load_fsmol(args.fsmol_sample), [(4, 6), (5, 7)]),
        ("DrugOOD (ic50_scaffold)", load_drugood, [(4, 6), (5, 7)]),
        ("curated_mic", load_curated_mic, [(4, 6), (5, 7)]),
        ("chembl36 (p-scale)", load_chembl36, [(4, 6), (5, 7)]),
    ]

    rows = []
    for name, loader, windows in jobs:
        print(f"\nloading {name} ...", flush=True)
        assays, scale = loader()
        if not assays:
            print("  nothing loaded")
            continue
        print(f"  {len(assays):,} assays with >=2 labelled molecules")
        for w in windows:
            r = funnel(assays, w, n, name)
            r["fsmol_sample_scale"] = round(scale, 2)
            rows.append(r)

    df = pd.DataFrame(rows)
    out = DATA_ANALYSIS_RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"fsmol_rule_funnel_n{n}.csv")
    df.to_csv(p, index=False)

    print(f"\n{'=' * 120}")
    print(f"FS-MOL PUBLISHED RULE, n_support={n}   (median pXC50, window, fixed-5.0 "
          f"fallback, then the [0.30,0.70] retention filter)")
    print(f"{'=' * 120}\n")
    show = df[["dataset", "window", "start_assays", "no_value", "too_small", "dead",
               "ratio_filter", "minority_lt_n", "retained", "retained_pct",
               "median_ratio_retained", "fallback_fired_pct"]]
    print(show.to_string(index=False))
    print(f"\nfunnel -> {p}")


if __name__ == "__main__":
    main()
