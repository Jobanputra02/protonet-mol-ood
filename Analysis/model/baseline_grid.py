"""
Baseline Grid - Representation x Head, on IDENTICAL splits
==========================================================
ONE QUESTION: what actually drives scaffold-OOD performance - the representation
(raw ECFP vs frozen learned embedding) or the head (mean-prototype vs adaptive)?

    representation  in  {raw ECFP fingerprint, frozen learned embedding}
    head            in  {mean-prototype (averaging), adaptive per-task (LogReg/kNN/RF)}

    2x2:               | mean-prototype          | adaptive (LogReg/kNN/RF)
    -------------------+-------------------------+--------------------------
    raw ECFP           | ecfp_proto_euclid       | ecfp_logreg / ecfp_rf
    learned embedding  | emb_proto_euclid (model)| emb_logreg / emb_knn

KEY PROPERTY: for every (assay, support_size, repeat) all heads receive the SAME
support and query molecules (one fair split, shared). The frozen checkpoint is used
ONLY as an encoder; every head is computed here. No retraining.

This is to baseline_grid what model.py is to main.py: the per-task heads (fit fresh
on each support set, no meta-training) are defined here; the trainable ProtoNet
encoders/heads live in ../../model.py.

HOW TO RUN: edit the CONFIG block below (same ENCODER / MODEL_HEAD / TRAINING_SPLIT /
SEED variables as main.py - they pick which checkpoint's encoder to use), then:
    python Analysis/model/baseline_grid.py
Results are written to outputs/results/<run_tag>/baseline_grid.csv  (same run-tag
folder convention as main.py). Plot them with:
    python Analysis/model/plot.py --grid_csv outputs/results/<run_tag>/baseline_grid.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FSMOL_TEST, CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402
from data import build_fair_split_indices  # noqa: E402
from main import load_fsmol_split  # noqa: E402
from model import _mahalanobis_dists  # noqa: E402
from evaluate import _encode_assay, _is_gnn, _is_fsmol_gnn, _load_model_from_checkpoint  # noqa: E402

RF_PARAMS = dict(n_estimators=100, max_depth=10, max_features="sqrt",
                 min_samples_leaf=2, n_jobs=-1, random_state=0)


# =============================================================================
# PREDICTION HEADS - uniform signature head(Xs, ys, Xq) -> P(active)
# Xs/Xq are either raw ECFP fingerprints OR frozen embeddings; the head is agnostic.
# (These were a separate baselines.py - inlined here as this is their only user.)
# =============================================================================

def proto_euclidean(Xs, ys, Xq):
    """Mean-prototype + squared-Euclidean distance (the distance ProtoNet trains with)."""
    pa, pi = Xs[ys == 1].mean(0), Xs[ys == 0].mean(0)
    da = ((Xq - pa) ** 2).sum(1)
    di = ((Xq - pi) ** 2).sum(1)
    return 1.0 / (1.0 + np.exp(np.clip(da - di, -30, 30)))   # sigmoid(d_inactive - d_active)


def proto_mahalanobis(Xs, ys, Xq):
    """Mean-prototype + FS-Mol shrinkage Mahalanobis (the paper's eval head; embedding space)."""
    sup = torch.tensor(np.asarray(Xs), dtype=torch.float32)
    qry = torch.tensor(np.asarray(Xq), dtype=torch.float32)
    active = torch.tensor(ys.astype(bool))
    protos = torch.stack([sup[active].mean(0), sup[~active].mean(0)], dim=0)
    dists = _mahalanobis_dists(qry, protos, sup, active)
    return torch.softmax(-dists, dim=1)[:, 0].numpy()


def _sklearn(clf, Xs, ys, Xq):
    clf.fit(Xs, ys)
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.zeros(len(Xq), dtype=np.float32)
    return clf.predict_proba(Xq)[:, classes.index(1)]


def logreg(Xs, ys, Xq):
    return _sklearn(make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000, C=1.0)), Xs, ys, Xq)


def knn(Xs, ys, Xq, k=5):
    kk = max(1, min(k, len(ys) - 1))
    return _sklearn(make_pipeline(StandardScaler(),
                                  KNeighborsClassifier(n_neighbors=kk)), Xs, ys, Xq)


def random_forest(Xs, ys, Xq):
    return _sklearn(RandomForestClassifier(**RF_PARAMS), Xs, ys, Xq)


def mean_label(Xs, ys, Xq):
    """Trivial floor: predict the support active-fraction for every query molecule."""
    return np.full(len(Xq), float(ys.mean()), dtype=np.float32)


def kr_tanimoto(Xs, ys, Xq, eps=1e-8):
    """Tanimoto-kernel weighted vote on BINARY fingerprints (ECFP only)."""
    s = (np.asarray(Xs) > 0).astype(np.float32)
    q = (np.asarray(Xq) > 0).astype(np.float32)
    inter = q @ s.T
    tani = inter / (q.sum(1, keepdims=True) + s.sum(1, keepdims=True).T - inter + eps)
    return (tani @ ys.astype(np.float32)) / (tani.sum(1) + eps)


# name -> (callable, representation, head_family).  representation: "ecfp" | "embedding"
REGISTRY = {
    "emb_proto_euclid":      (proto_euclidean,   "embedding", "mean_proto"),
    "emb_proto_mahalanobis": (proto_mahalanobis, "embedding", "mean_proto"),
    "emb_logreg":            (logreg,            "embedding", "adaptive"),
    "emb_knn":               (knn,               "embedding", "adaptive"),
    "ecfp_proto_euclid":     (proto_euclidean,   "ecfp",      "mean_proto"),
    "ecfp_logreg":           (logreg,            "ecfp",      "adaptive"),
    "ecfp_rf":               (random_forest,     "ecfp",      "adaptive"),
    "ecfp_knn":              (knn,               "ecfp",      "adaptive"),
    "ecfp_kr_tanimoto":      (kr_tanimoto,       "ecfp",      "adaptive"),
    "ecfp_mean_label":       (mean_label,        "ecfp",      "adaptive"),
}

ECFP_HEADS = ["ecfp_proto_euclid", "ecfp_logreg", "ecfp_rf"]
EMB_HEADS  = ["emb_proto_euclid", "emb_proto_mahalanobis", "emb_logreg", "emb_knn"]
DEFAULT_HEADS = ECFP_HEADS + EMB_HEADS


# =============================================================================
# Metric + evaluation loop
# =============================================================================

def delta_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score) - float(np.mean(y_true)))


def _run_heads(Es, Fs, ys, Eq, Fq, head_names):
    """{head: P(active)} for one shared split; route each head to its representation."""
    out = {}
    for name in head_names:
        fn, representation, _ = REGISTRY[name]
        out[name] = fn(Fs, ys, Fq) if representation == "ecfp" else fn(Es, ys, Eq)
    return out


def run(checkpoint, splits, support_sizes, n_repeats, max_assays, base_seed,
        head_names=DEFAULT_HEADS, test_assays=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model_from_checkpoint(checkpoint, device)
    model.eval()
    gnn, fsmol = _is_gnn(model), _is_fsmol_gnn(model)

    if test_assays is None:
        print(f"Loading FS-Mol test assays (max_assays={max_assays}) ...")
        test_assays = load_fsmol_split(FSMOL_TEST, max_assays=max_assays)
        print(f"  {len(test_assays)} assays loaded.")

    need_sizes = "size" in splits
    rows = []
    for ai, assay in enumerate(test_assays):
        if assay.binary_labels is None:
            continue
        n_total = len(assay)
        y_all = np.asarray(assay.binary_labels)
        E_all = _encode_assay(model, assay, device, gnn, fsmol).detach().cpu().numpy()
        F_all = np.stack(assay.fingerprints).astype(np.float32)
        mol_sizes = None
        if need_sizes:
            from evaluate import _get_mol_sizes
            mol_sizes = _get_mol_sizes(assay)

        for split in splits:
            for n_sup in support_sizes:
                for rep in range(n_repeats):
                    _split_int = {"random": 1, "scaffold": 2, "size": 3}.get(split, 0)
                    rng = np.random.RandomState(
                        base_seed + 1009 * rep + 7919 * n_sup + 10000 * _split_int)
                    sp = build_fair_split_indices(
                        n_total, assay.scaffold_groups, y_all, n_sup, split, rng,
                        mol_sizes=mol_sizes, require_both_classes=True)
                    if sp is None:
                        continue
                    s_idx, q_idx = sp
                    ys, yq = y_all[s_idx], y_all[q_idx]
                    if len(np.unique(yq)) < 2:
                        continue
                    preds = _run_heads(E_all[s_idx], F_all[s_idx], ys,
                                       E_all[q_idx], F_all[q_idx], head_names)
                    for head, p in preds.items():
                        _, rep_repr, fam = REGISTRY[head]
                        rows.append({
                            "assay_id": assay.assay_id, "split_type": split,
                            "support_size": n_sup, "support_actual": int(len(s_idx)),
                            "n_query": int(len(q_idx)), "repeat": rep,
                            "representation": rep_repr, "head": head, "head_family": fam,
                            "delta_auprc": delta_auprc(yq, p),
                        })
        if (ai + 1) % 10 == 0:
            print(f"  {ai + 1}/{len(test_assays)} assays done ...", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows produced - check support sizes / scaffold diversity.")
        return
    order = ["ecfp_proto_euclid", "ecfp_logreg", "ecfp_rf", "ecfp_knn",
             "ecfp_kr_tanimoto", "ecfp_mean_label",
             "emb_proto_euclid", "emb_proto_mahalanobis", "emb_logreg", "emb_knn"]
    for split in sorted(df["split_type"].unique()):
        sub = df[df["split_type"] == split]
        piv = (sub.groupby(["head", "support_size"])["delta_auprc"].mean().reset_index()
                  .pivot(index="head", columns="support_size", values="delta_auprc").round(4))
        piv = piv.reindex([h for h in order if h in piv.index]
                          + [h for h in piv.index if h not in order])
        n_assays = sub.groupby("support_size")["assay_id"].nunique().to_dict()
        print(f"\n{'='*70}\nSPLIT = {split}   (mean delta-AUPRC; N assays per size: "
              f"{ {k: n_assays[k] for k in sorted(n_assays)} })\n{'='*70}")
        print(piv.to_string())


# =============================================================================
# CONFIG  -  edit these to run one or many configurations
# =============================================================================
if __name__ == "__main__":

    # -- list of (encoder, model_head, training_split, seed) to run ----------
    # Remove or comment out entries you don't want. Already-done runs are
    # skipped automatically (output CSV already exists).
    RUNS = [
        # encoder         model_head         training_split   seed
        ("ecfp",          "classification",  "random",        0),
        ("ecfp",          "classification",  "random",        1),
        ("ecfp",          "classification",  "random",        2),
        ("ecfp",          "classification",  "shift_aware",   0),
        ("ecfp",          "classification",  "shift_aware",   1),
        ("ecfp",          "classification",  "shift_aware",   2),
        ("fsmol_gnn",     "classification",  "random",        0),
        ("fsmol_gnn",     "classification",  "random",        1),
        ("fsmol_gnn",     "classification",  "random",        2),
        ("fsmol_gnn",     "classification",  "shift_aware",   0),
        ("fsmol_gnn",     "classification",  "shift_aware",   1),
        ("fsmol_gnn",     "classification",  "shift_aware",   2),
    ]

    HEADS         = DEFAULT_HEADS
    SPLITS        = ["random", "scaffold", "size"]
    SUPPORT_SIZES = [16, 32, 64, 128, 256, 512]
    N_REPEATS     = 5
    BASE_SEED     = 42    # split-construction seed (not the model seed)
    MAX_ASSAYS    = None  # None = all 154; set small (e.g. 6) for a quick test
    # -------------------------------------------------------------------------

    print("Loading FS-Mol test assays (shared across all runs) ...")
    test_assays = load_fsmol_split(FSMOL_TEST, max_assays=MAX_ASSAYS)
    print(f"  {len(test_assays)} assays loaded.\n")

    # -- run ECFP heads once (encoder-independent) ----------------------------
    ecfp_csv = os.path.join(RESULTS_DIR, "ecfp_baseline_heads.csv")
    if os.path.exists(ecfp_csv):
        print(f"[SKIP] ECFP heads - already exists at {ecfp_csv}")
        df_ecfp = pd.read_csv(ecfp_csv)
    else:
        first_ckpt = next(
            os.path.join(CHECKPOINT_DIR, f"ptn_{enc}_{mh}_{ts}_seed{s}.pt")
            for enc, mh, ts, s in RUNS
            if os.path.exists(os.path.join(CHECKPOINT_DIR, f"ptn_{enc}_{mh}_{ts}_seed{s}.pt"))
        )
        print(f"Running ECFP heads once using: {os.path.basename(first_ckpt)}")
        df_ecfp = run(first_ckpt, SPLITS, SUPPORT_SIZES, N_REPEATS, MAX_ASSAYS, BASE_SEED,
                      head_names=ECFP_HEADS, test_assays=test_assays)
        df_ecfp.to_csv(ecfp_csv, index=False)
        print(f"Saved ECFP heads -> {ecfp_csv}\n")

    # -- run emb heads per checkpoint -----------------------------------------
    for ENCODER, MODEL_HEAD, TRAINING_SPLIT, SEED in RUNS:
        run_tag    = f"{ENCODER}_{MODEL_HEAD}_{TRAINING_SPLIT}_seed{SEED}"
        checkpoint = os.path.join(CHECKPOINT_DIR, f"ptn_{run_tag}.pt")
        out_dir    = os.path.join(RESULTS_DIR, run_tag)
        out_csv    = os.path.join(out_dir, "baseline_grid.csv")

        if os.path.exists(out_csv):
            print(f"[SKIP] {run_tag} - already exists")
            continue
        if not os.path.exists(checkpoint):
            print(f"[SKIP] {run_tag} - checkpoint not found: {checkpoint}")
            continue

        print("=" * 60)
        print(f"Baseline grid  |  run_tag = {run_tag}")
        print(f"  checkpoint => {checkpoint}")
        print(f"  output     => {out_csv}")
        print("=" * 60)

        os.makedirs(out_dir, exist_ok=True)
        df_emb = run(checkpoint, SPLITS, SUPPORT_SIZES, N_REPEATS, MAX_ASSAYS, BASE_SEED,
                     head_names=EMB_HEADS, test_assays=test_assays)
        df = pd.concat([df_ecfp, df_emb], ignore_index=True)
        summarize(df)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df):,} rows -> {out_csv}")
