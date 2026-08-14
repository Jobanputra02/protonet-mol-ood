"""
Split OOD Characterization  (model-free, raw data only)
=======================================================
ONE QUESTION: for the FS-Mol test data, which support/query split construction produces a
genuine out-of-distribution gap, and at what cost in usable data?

No model, no training, no evaluation - this operates purely on SMILES/fingerprints. For
each candidate split it reports two things per support size:

  * OOD severity  = query->support nearest-neighbour Tanimoto (ECFP), the standard
                    applicability-domain measure. LOWER = more OOD.
                    (median across assays, plus the fraction of query molecules whose
                     nearest support neighbour is < 0.3, i.e. genuinely far.)
  * Data retention = how many assays still yield a valid split (both classes in support,
                     non-empty query) and the median support/query sizes that survive.

The right choice is the split that reaches genuine OOD (median NN-Tanimoto well below the
random-split value) WHILE retaining enough assays/molecules - not simply the most OOD,
which is trivially maximised by discarding almost all the data.

Candidates:
  random          IID reference (least OOD)
  murcko          Bemis-Murcko scaffold groups (fine-grained)
  generic         generic (graph-framework) scaffold groups (coarser)
  butina@c        Taylor-Butina sphere-exclusion clustering at distance cutoff c
  agglomerative   average-linkage hierarchical clustering on Tanimoto distance
  maxmin          MaxMin diversity-picked support (RDKit MaxMinPicker), query = remainder
  sim_cutoff@t    random support, then keep only query molecules with NN-Tanimoto < t
  size            support = small molecules (heavy-atom count), query = large
  datasail        DataSAIL cold single-molecule split (optional; needs `pip install datasail`)

Every candidate is measured under the SAME constraint: the support must contain both
classes (so the split is usable downstream). Class-balance / stratified sampling is a
constraint applied to all candidates - it is not itself a candidate.

HOW TO RUN:  python Analysis/data/split_ood_characterization.py
Output: prints a table and writes it to outputs/data_analysis/csvs/split_ood_characterization.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import DataStructs                                     # type: ignore
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker      # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FSMOL_TEST, DATA_ANALYSIS_RESULTS_DIR                       # noqa: E402
from data import (load_fsmol_split, build_scaffold_groups, build_fair_split_indices)  # type: ignore # noqa: E402


# =============================================================================
# CONFIG
# =============================================================================
SUPPORT_SIZES   = (64, 128)
N_REPEATS       = 3          # for the randomised candidates (random / maxmin / sim_cutoff)
BUTINA_CUTOFFS  = (0.4, 0.55, 0.70)
AGGLO_THRESHOLD = 0.65       # Tanimoto-distance merge threshold (average linkage)
SIM_CUTOFFS     = (0.4, 0.3, 0.25)
FAR_THRESHOLD   = 0.30       # a query molecule is "genuinely far" if NN-Tanimoto < this
MAX_ASSAYS      = None       # None = all 154; small int for a quick test
USE_DATASAIL    = True       # attempt DataSAIL (skipped automatically if not installed)
BASE_SEED       = 42
# =============================================================================


# ---- fingerprint helpers ----------------------------------------------------
def _bitvects(fingerprints):
    out = []
    for f in fingerprints:
        f = np.asarray(f)
        bv = DataStructs.ExplicitBitVect(len(f))
        for b in np.nonzero(f > 0)[0].tolist():
            bv.SetBit(int(b))
        out.append(bv)
    return out


def _tanimoto_matrix(sup_fp, qry_fp):
    """Full (n_query, n_support) Tanimoto similarity matrix (binary ECFP)."""
    s = (sup_fp > 0).astype(np.float32)
    q = (qry_fp > 0).astype(np.float32)
    inter = q @ s.T
    return inter / (q.sum(1, keepdims=True) + s.sum(1, keepdims=True).T - inter + 1e-8)


def _nn_tanimoto(sup_fp, qry_fp):
    """Top-1 NN: for each query, its MAX similarity to any support mol."""
    return _tanimoto_matrix(sup_fp, qry_fp).max(1)


def _topk_tanimoto(sup_fp, qry_fp, k=3):
    """Top-k NN: for each query, mean of its k most-similar support mols.
    Robust to one lucky outlier while still capturing near-neighbourhood."""
    T = _tanimoto_matrix(sup_fp, qry_fp)  # (n_query, n_support)
    k = min(k, T.shape[1])
    top = np.sort(T, axis=1)[:, -k:]      # k largest per query row
    return top.mean(1)                     # (n_query,)


def _mean_all_tanimoto(sup_fp, qry_fp):
    """Mean of ALL (query, support) pairs — most stable, least sensitive to support size."""
    return _tanimoto_matrix(sup_fp, qry_fp).mean(1)  # (n_query,) row means


# ---- grouping builders (return {group_id: [indices]}) -----------------------
def _butina_groups(bitvects, cutoff):
    from rdkit.ML.Cluster import Butina  # type: ignore
    n = len(bitvects)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return {ci: list(m) for ci, m in enumerate(clusters)}


def _agglomerative_groups(bitvects, threshold):
    from sklearn.cluster import AgglomerativeClustering
    n = len(bitvects)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
        for j, s in enumerate(sims):
            D[i, j] = D[j, i] = 1.0 - s
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                 metric="precomputed", linkage="average")
    labels = cl.fit_predict(D)
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    return groups


DATASAIL_MAX_MOLS = 300   # DataSAIL's ILP solver does not scale; skip larger assays
DATASAIL_MAX_SEC  = 20    # hard per-assay solver time limit (seconds)


def _datasail_groups(smiles_list):
    """DataSAIL cold single-molecule split -> two groups. None if unavailable / too large.
    DataSAIL produces one fixed ~50/50 similarity-disjoint partition (not size-parameterised),
    so it appears mainly at the smaller support size where one half reaches n_support. Its ILP
    solver does not scale, so it is capped to assays with <= DATASAIL_MAX_MOLS molecules and a
    per-assay time limit."""
    import os as _os
    import logging
    import contextlib
    if len(smiles_list) > DATASAIL_MAX_MOLS:
        return None
    try:
        from datasail.sail import datasail  # type: ignore
    except Exception:
        return None
    e_data = {str(i): smi for i, smi in enumerate(smiles_list)}
    logging.disable(logging.CRITICAL)   # silence CVXPY/DataSAIL logging chatter
    try:
        with open(_os.devnull, "w") as _dn, \
                contextlib.redirect_stdout(_dn), contextlib.redirect_stderr(_dn):
            out = datasail(techniques=["C1e"], splits=[1, 1], names=["A", "B"], runs=1,
                           max_sec=DATASAIL_MAX_SEC, e_type="M", e_data=e_data)
        assign = out[0]["C1e"][0] if isinstance(out, tuple) else out["C1e"][0]
    except Exception:
        return None
    finally:
        logging.disable(logging.NOTSET)
    groups = {}
    for i_str, name in assign.items():
        groups.setdefault(name, []).append(int(i_str))
    return groups if len(groups) >= 2 else None


# ---- split builders: return (sup_idx, qry_idx) or None ----------------------
def _split_from_groups(n, groups, y, n_sup, rng):
    return build_fair_split_indices(n, groups, y, n_sup, "scaffold", rng,
                                    require_both_classes=True)


def _split_random(n, y, n_sup, rng):
    return build_fair_split_indices(n, {}, y, n_sup, "random", rng, require_both_classes=True)


def _split_size(n, y, n_sup, rng, mol_sizes):
    return build_fair_split_indices(n, {}, y, n_sup, "size", rng, mol_sizes=mol_sizes,
                                    require_both_classes=True)


def _split_maxmin(n, y, n_sup, rng, bitvects):
    """Diverse support via MaxMin picking; query = remainder. Enforce both classes."""
    if n <= n_sup:
        return None
    picker = MaxMinPicker()
    seed = int(rng.randint(1 << 30))
    picks = list(picker.LazyBitVectorPick(bitvects, n, n_sup, seed=seed))
    sup = np.array(picks, dtype=np.int64)
    if len(np.unique(y[sup])) < 2:
        return None
    qry = np.array([i for i in range(n) if i not in set(picks)], dtype=np.int64)
    return sup, qry


def _split_sim_cutoff(n, y, n_sup, rng, fps, tau):
    """Random support, then keep only query molecules with NN-Tanimoto < tau."""
    base = _split_random(n, y, n_sup, rng)
    if base is None:
        return None
    sup, qry = base
    nn = _nn_tanimoto(fps[sup], fps[qry])
    keep = qry[nn < tau]
    if len(keep) == 0:
        return None
    return sup, keep


# =============================================================================
# main
# =============================================================================
def characterize(assays):
    records = []
    for ai, a in enumerate(assays):
        if a.binary_labels is None:
            continue
        n = len(a)
        y = np.asarray(a.binary_labels)
        fps = np.stack(a.fingerprints).astype(np.float32)
        smiles = a.scaffolds
        bvs = _bitvects(a.fingerprints)
        mol_sizes = None

        # pre-build deterministic groupings once per assay
        groupings = {
            "murcko":  build_scaffold_groups(smiles, "murcko"),
            "generic": build_scaffold_groups(smiles, "generic"),
            "agglomerative": _agglomerative_groups(bvs, AGGLO_THRESHOLD),
        }
        for c in BUTINA_CUTOFFS:
            groupings[f"butina@{c}"] = _butina_groups(bvs, c)
        ds = _datasail_groups(smiles) if USE_DATASAIL else None
        if ds is not None:
            groupings["datasail"] = ds

        for n_sup in SUPPORT_SIZES:
            def _record(name, sup, qry):
                nn1  = _nn_tanimoto(fps[sup], fps[qry])      # top-1 NN per query
                nn3  = _topk_tanimoto(fps[sup], fps[qry], k=3)  # top-3 mean per query
                all_ = _mean_all_tanimoto(fps[sup], fps[qry])   # mean over all pairs
                records.append({
                    "candidate": name, "support_size": n_sup, "assay_id": a.assay_id,
                    "n_support": len(sup), "n_query": len(qry),
                    "nn_top1_mean":   float(nn1.mean()),   # current metric
                    "nn_top3_mean":   float(nn3.mean()),   # prof suggestion 2
                    "mean_all_mean":  float(all_.mean()),  # prof suggestion 1
                    "frac_far":       float((nn1 < FAR_THRESHOLD).mean()),
                })

            # deterministic-ish group candidates + random/size, averaged over repeats
            for rep in range(N_REPEATS):
                rng = np.random.RandomState(BASE_SEED + 101 * rep + 7 * n_sup)

                sp = _split_random(n, y, n_sup, rng)
                if sp is not None:
                    _record("random", *sp)

                if mol_sizes is None:
                    from evaluate import _get_mol_sizes
                    mol_sizes = _get_mol_sizes(a)
                sp = _split_size(n, y, n_sup, rng, mol_sizes)
                if sp is not None:
                    _record("size", *sp)

                for name, groups in groupings.items():
                    if groups is None or len(groups) < 2:
                        continue
                    sp = _split_from_groups(n, groups, y, n_sup, rng)
                    if sp is not None:
                        _record(name, *sp)

                sp = _split_maxmin(n, y, n_sup, rng, bvs)
                if sp is not None:
                    _record("maxmin", *sp)

                for tau in SIM_CUTOFFS:
                    sp = _split_sim_cutoff(n, y, n_sup, rng, fps, tau)
                    if sp is not None:
                        _record(f"sim_cutoff@{tau}", *sp)

        if (ai + 1) % 20 == 0:
            print(f"  {ai + 1}/{len(assays)} assays...", flush=True)
    return pd.DataFrame(records)


def summarize(df, n_total_assays):
    """One row per (candidate, support_size): three OOD severity metrics + retention.

    nn_top1_median   : current metric - each query's nearest support neighbour
    nn_top3_median   : each query's mean of its 3 most-similar support mols (prof suggestion)
    mean_all_median  : mean over ALL (query, support) pairs (prof suggestion)
    All three: lower = more OOD. Ranking should agree; if it does, the choice is justified.
    """
    rows = []
    for (cand, n_sup), sub in df.groupby(["candidate", "support_size"]):
        per_assay = sub.groupby("assay_id").agg(
            t1=("nn_top1_mean", "mean"), t3=("nn_top3_mean", "mean"),
            ta=("mean_all_mean", "mean"), far=("frac_far", "mean"),
            nq=("n_query", "mean"), ns=("n_support", "mean"))
        rows.append({
            "candidate":       cand,
            "support_size":    n_sup,
            "nn_top1_median":  round(per_assay["t1"].median(), 3),
            "nn_top3_median":  round(per_assay["t3"].median(), 3),
            "mean_all_median": round(per_assay["ta"].median(), 3),
            "frac_far_median": round(per_assay["far"].median(), 3),
            "n_assays_valid":  int(per_assay.shape[0]),
            "pct_assays_valid":round(100 * per_assay.shape[0] / n_total_assays, 1),
            "median_support":  int(per_assay["ns"].median()),
            "median_query":    int(per_assay["nq"].median()),
        })
    out = pd.DataFrame(rows).sort_values(["support_size", "nn_top1_median"])
    return out


if __name__ == "__main__":
    print("Loading FS-Mol test assays ...")
    assays = load_fsmol_split(FSMOL_TEST, max_assays=MAX_ASSAYS)
    n_total = sum(1 for a in assays if a.binary_labels is not None)
    print(f"  {n_total} assays with both classes.\n")
    if USE_DATASAIL:
        try:
            import datasail  # noqa: F401
            print("DataSAIL: available.\n")
        except Exception:
            print("DataSAIL: NOT installed - that candidate will be skipped "
                  "(pip/conda install datasail on Linux to include it).\n")

    df = characterize(assays)
    summary = summarize(df, n_total)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("SPLIT OOD CHARACTERIZATION  (lower = more OOD for all three similarity columns)")
    print("  nn_top1   : each query's nearest support mol (original metric)")
    print("  nn_top3   : each query's mean of 3 nearest support mols")
    print("  mean_all  : mean over all (query, support) pairs — most stable")
    print("=" * 100)
    print(summary.to_string(index=False))

    os.makedirs(DATA_ANALYSIS_RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(DATA_ANALYSIS_RESULTS_DIR, "split_ood_characterization.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")
