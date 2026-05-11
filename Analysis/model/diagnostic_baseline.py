"""
Diagnostic Baseline Comparison
================================
Supervisor diagnostic: compare PTN vs simple baselines on the exact same
context/query episodes used during training.

Baselines:
    mean          — predict mean(support_labels) for all queries      [trivial]
    kNN (k=1,3,5) — sklearn KNeighborsRegressor on raw ECFPs
    KR-Tanimoto   — kernel ridge regression with Tanimoto kernel      [most direct]
    PTN           — loaded from checkpoint

KR-Tanimoto is the key comparison: it does kernel regression in raw ECFP space
(no learned embedding). PTN does kernel regression in *learned* embedding space.
If PTN < KR-Tanimoto → learned embedding is better than raw fingerprints → training works.
If PTN >> KR-Tanimoto → training is broken.
If all ≈ mean-label → ECFP fingerprints carry no signal for these episodes.

Two episode split types are tested:
    random   — support/query both sampled randomly from the assay  [easy base case]
    scaffold — support from one scaffold family, query from another [hard, what PTN trains on]

Usage:
    python analysis/model/diagnostic_baseline.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

# Repo root on sys.path so we can import model and data modules
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO_ROOT)

from config import FSMOL_TRAIN, MODEL_SAVE_PATH, RESULTS_DIR
from data import _load_assay_file, EpisodeSampler
from model import PrototypicalNetworkRegression

os.makedirs(RESULTS_DIR, exist_ok=True)

N_ASSAYS   = 100   # train assays to sample
N_EPISODES = 10    # episodes per assay per split type
N_SUPPORT  = 16
N_QUERY    = 16
SEED       = 42


# =============================================================================
# TANIMOTO KERNEL
# =============================================================================

def tanimoto_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Tanimoto (Jaccard) similarity between rows of X and Y."""
    XY = X @ Y.T
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    return XY / (XX + YY.T - XY + 1e-10)


# =============================================================================
# PER-EPISODE EVALUATION
# =============================================================================

def run_one_episode(
    sup_fp:  np.ndarray,
    sup_lbl: np.ndarray,
    qry_fp:  np.ndarray,
    qry_lbl: np.ndarray,
    model:   PrototypicalNetworkRegression,
    device:  torch.device,
) -> dict:
    """Run all baselines + PTN on one episode. Returns dict of MSE values."""
    results: dict[str, float] = {}

    # Mean-label baseline
    mean_pred = np.full(len(qry_lbl), sup_lbl.mean())
    results["mean"] = float(np.mean((mean_pred - qry_lbl) ** 2))
    results["support_label_var"] = float(np.var(sup_lbl))

    # kNN baselines
    for k in [1, 3, 5]:
        actual_k = min(k, len(sup_fp))
        knn = KNeighborsRegressor(n_neighbors=actual_k, metric="euclidean")
        knn.fit(sup_fp, sup_lbl)
        preds = knn.predict(qry_fp)
        results[f"knn_k{k}"] = float(np.mean((preds - qry_lbl) ** 2))

    # Tanimoto kernel ridge (PTN analogue in raw ECFP space)
    K_train = tanimoto_kernel(sup_fp, sup_fp)
    K_test  = tanimoto_kernel(qry_fp, sup_fp)
    for alpha in [0.01, 0.1, 1.0]:
        try:
            kr = KernelRidge(alpha=alpha, kernel="precomputed")
            kr.fit(K_train, sup_lbl)
            preds = kr.predict(K_test)
            results[f"kr_tanimoto_a{alpha}"] = float(np.mean((preds - qry_lbl) ** 2))
        except Exception:
            results[f"kr_tanimoto_a{alpha}"] = float("nan")

    # PTN
    with torch.no_grad():
        s_fp  = torch.tensor(sup_fp).to(device)
        s_lbl = torch.tensor(sup_lbl).to(device)
        q_fp  = torch.tensor(qry_fp).to(device)
        ptn_preds = model.forward(s_fp, s_lbl, q_fp).cpu().numpy()
    results["ptn"] = float(np.mean((ptn_preds - qry_lbl) ** 2))

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_diagnostics() -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {MODEL_SAVE_PATH}")
    ckpt  = torch.load(MODEL_SAVE_PATH, map_location=device)
    cfg   = ckpt["config"]
    model = PrototypicalNetworkRegression(
        input_dim=2048, hidden_dim=cfg["hidden_dim"], embedding_dim=cfg["embedding_dim"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']}, Val RMSE {ckpt['val_rmse']:.4f}\n")

    all_files = sorted([
        os.path.join(FSMOL_TRAIN, f)
        for f in os.listdir(FSMOL_TRAIN)
        if f.endswith(".jsonl.gz")
    ])
    chosen_files = rng.choice(all_files, size=min(N_ASSAYS, len(all_files)), replace=False)

    sampler  = EpisodeSampler(N_SUPPORT, N_QUERY)
    min_len  = N_SUPPORT + N_QUERY
    min_grp  = max(2, N_SUPPORT // 4)

    rows                = []
    n_usable            = 0
    n_skipped           = 0
    n_scaffold_eligible = 0

    print(f"Running diagnostic on up to {len(chosen_files)} train assays "
          f"× {N_EPISODES} episodes × 2 split types...")

    for i, fpath in enumerate(chosen_files):
        assay = _load_assay_file(fpath)
        if len(assay) < min_len:
            n_skipped += 1
            continue
        n_usable += 1

        usable_groups    = [k for k, v in assay.scaffold_groups.items() if len(v) >= min_grp]
        has_scaffold     = len(usable_groups) >= 2
        if has_scaffold:
            n_scaffold_eligible += 1

        for ep_i in range(N_EPISODES):
            for split_type in ["random", "scaffold"]:
                if split_type == "scaffold" and not has_scaffold:
                    continue

                s_fp, s_lbl, q_fp, q_lbl = sampler.sample_episode(
                    assay, shift_aware=(split_type == "scaffold")
                )
                metrics = run_one_episode(
                    s_fp.numpy(), s_lbl.numpy(), q_fp.numpy(), q_lbl.numpy(),
                    model, device,
                )
                rows.append({
                    "assay_id":          assay.assay_id,
                    "split_type":        split_type,
                    "episode":           ep_i,
                    "n_molecules":       len(assay),
                    "n_scaffold_groups": len(assay.scaffold_groups),
                    "n_usable_groups":   len(usable_groups),
                    **metrics,
                })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chosen_files)} files  |  {n_usable} usable  "
                  f"| {n_scaffold_eligible} scaffold-eligible", end="\r")

    print(f"\n  Done. {n_usable} usable, {n_skipped} skipped, "
          f"{n_scaffold_eligible} scaffold-eligible.\n")

    df = pd.DataFrame(rows)

    # ── Report ────────────────────────────────────────────────────────────────
    method_cols = ["mean", "knn_k1", "knn_k3", "knn_k5",
                   "kr_tanimoto_a0.01", "kr_tanimoto_a0.1", "kr_tanimoto_a1.0", "ptn"]
    method_labels = {
        "mean":               "Mean-label",
        "knn_k1":             "kNN  (k=1)",
        "knn_k3":             "kNN  (k=3)",
        "knn_k5":             "kNN  (k=5)",
        "kr_tanimoto_a0.01":  "KR-Tanimoto (α=0.01)",
        "kr_tanimoto_a0.1":   "KR-Tanimoto (α=0.10)",
        "kr_tanimoto_a1.0":   "KR-Tanimoto (α=1.00)",
        "ptn":                "PTN (checkpoint)",
    }

    print("=" * 70)
    print(f"{'Method':<28} {'Mean MSE':>10} {'RMSE':>10} {'Median MSE':>12}")

    for split in ["random", "scaffold"]:
        sub = df[df["split_type"] == split]
        if sub.empty:
            continue
        n_ep = len(sub)
        n_as = sub["assay_id"].nunique()
        mean_lbl_var = sub["support_label_var"].mean()
        print(f"\n── Split: {split}  ({n_ep} episodes, {n_as} assays, "
              f"mean support-label var = {mean_lbl_var:.4f}) ──")
        print("-" * 70)
        for col in method_cols:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if vals.empty:
                continue
            mean_mse   = vals.mean()
            median_mse = vals.median()
            rmse       = float(np.sqrt(mean_mse))
            label      = method_labels.get(col, col)
            print(f"  {label:<26} {mean_mse:>10.4f} {rmse:>10.4f} {median_mse:>12.4f}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  PTN MSE >> best-KR-Tanimoto  →  training is still broken")
    print("  PTN MSE ≈ best-KR-Tanimoto   →  learned metric ≈ raw fingerprint kernel")
    print("  PTN MSE <  best-KR-Tanimoto  →  PTN has improved over baseline  ✓")
    print("  All methods ≈ mean-label MSE  →  ECFP fingerprints carry no signal for this split")
    print("=" * 70)

    print("\n── Per-assay mean MSE (random split, top-10 worst for PTN) ──")
    if not df[df["split_type"] == "random"].empty:
        per_assay = (df[df["split_type"] == "random"]
                     .groupby("assay_id")[["mean", "knn_k3", "kr_tanimoto_a0.1", "ptn"]]
                     .mean()
                     .sort_values("ptn", ascending=False)
                     .head(10)
                     .round(4))
        print(per_assay.to_string())

    out_path = os.path.join(RESULTS_DIR, "diagnostic_baseline.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results → {out_path}")
    return df


if __name__ == "__main__":
    run_diagnostics()
