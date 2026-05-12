"""
Diagnostic Baseline Comparison
================================
Compares PTN against simple baselines on the same episodes.

Baselines:
    mean          — predict mean(support_labels) for all queries      [trivial]
    kNN (k=1,3,5) — sklearn KNeighborsRegressor on raw ECFPs
    KR-Tanimoto   — kernel ridge regression with Tanimoto kernel

Two modes:

  --mode train  (default)
      100 sampled train assays × 10 episodes × 2 split types (random/scaffold).
      Sanity check: verifies training is working. N_QUERY=16 (fixed small query).

  --mode test
      All 154 FS-Mol test assays × 5 repeats × 3 split types (random/scaffold/size)
      × 6 support sizes (16/32/64/128/256/512).
      Directly comparable to evaluate_fsmol_test results in main.py.
      Query = all remaining molecules (not just 16), matching evaluate_fsmol_test.
      Metrics: MSE, RMSE, Spearman ρ, ΔAUPRC.

  --mode both
      Runs train mode then test mode.

Usage:
    python Analysis/model/diagnostic_baseline.py --mode test
    python Analysis/model/diagnostic_baseline.py --mode both
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO_ROOT)

from config import FSMOL_TRAIN, FSMOL_TEST, MODEL_SAVE_PATH, RESULTS_DIR
from data import _load_assay_file, EpisodeSampler
from model import PrototypicalNetworkRegression

os.makedirs(RESULTS_DIR, exist_ok=True)

# Train-mode constants
N_ASSAYS_TRAIN = 100
N_EPISODES     = 10
N_SUPPORT      = 16
N_QUERY        = 16
SEED           = 42

# Test-mode constants (match evaluate_fsmol_test)
SUPPORT_SIZES  = [16, 32, 64, 128, 256, 512]
N_REPEATS      = 5
MIN_TASK_SIZE  = 32


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def tanimoto_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    XY = X @ Y.T
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    return XY / (XX + YY.T - XY + 1e-10)


def _load_model(device: torch.device) -> PrototypicalNetworkRegression:
    print(f"Loading checkpoint: {MODEL_SAVE_PATH}")
    ckpt  = torch.load(MODEL_SAVE_PATH, map_location=device)
    cfg   = ckpt["config"]
    model = PrototypicalNetworkRegression(
        input_dim=2048, hidden_dim=cfg["hidden_dim"], embedding_dim=cfg["embedding_dim"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']}, Val RMSE {ckpt['val_rmse']:.4f}\n")
    return model


def _get_mol_sizes(assay) -> np.ndarray:
    from rdkit import Chem  # type: ignore
    sizes = []
    for smi in assay.scaffolds:
        mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
        sizes.append(mol.GetNumHeavyAtoms() if mol is not None else 0)
    return np.array(sizes, dtype=np.int32)


def _delta_auprc(preds: np.ndarray, binary_labels) -> float:
    if binary_labels is None:
        return float("nan")
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        return float("nan")
    return float(average_precision_score(binary_labels, preds) - binary_labels.mean())


def _spearman(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 3:
        return float("nan")
    return float(stats.spearmanr(preds, targets).statistic)  # type: ignore[union-attr]


def _all_metrics(preds: np.ndarray, targets: np.ndarray, binary_labels) -> dict:
    mse = float(np.mean((preds - targets) ** 2))
    return {
        "mse":       mse,
        "rmse":      float(np.sqrt(mse)),
        "spearman":  _spearman(preds, targets),
        "delta_auprc": _delta_auprc(preds, binary_labels),
    }


# =============================================================================
# PER-EPISODE EVALUATION (used by both modes)
# =============================================================================

def run_one_episode(
    sup_fp:  np.ndarray,
    sup_lbl: np.ndarray,
    qry_fp:  np.ndarray,
    qry_lbl: np.ndarray,
    bin_lbl,
    model:   PrototypicalNetworkRegression,
    device:  torch.device,
    full_metrics: bool = False,   # True → compute Spearman + ΔAUPRC as well
) -> dict:
    """Run all baselines + PTN on one episode."""
    results: dict[str, float] = {}
    results["support_label_var"] = float(np.var(sup_lbl))

    def _m(preds):
        if full_metrics:
            return _all_metrics(preds, qry_lbl, bin_lbl)
        return {"mse": float(np.mean((preds - qry_lbl) ** 2))}

    # Mean-label baseline
    results["mean"] = float(np.mean((np.full(len(qry_lbl), sup_lbl.mean()) - qry_lbl) ** 2))

    # kNN baselines
    for k in [1, 3, 5]:
        knn = KNeighborsRegressor(n_neighbors=min(k, len(sup_fp)), metric="euclidean")
        knn.fit(sup_fp, sup_lbl)
        m = _m(knn.predict(qry_fp))
        for key, val in m.items():
            results[f"knn_k{k}_{key}" if full_metrics else f"knn_k{k}"] = val

    # KR-Tanimoto
    K_train = tanimoto_kernel(sup_fp, sup_fp)
    K_test  = tanimoto_kernel(qry_fp, sup_fp)
    for alpha in [0.01, 0.1, 1.0]:
        tag = f"kr_tanimoto_a{alpha}"
        try:
            kr = KernelRidge(alpha=alpha, kernel="precomputed")
            kr.fit(K_train, sup_lbl)
            m = _m(kr.predict(K_test))
            for key, val in m.items():
                results[f"{tag}_{key}" if full_metrics else tag] = val
        except Exception:
            suffix = ["_mse", "_rmse", "_spearman", "_delta_auprc"] if full_metrics else [""]
            for s in suffix:
                results[f"{tag}{s}"] = float("nan")

    # PTN
    with torch.no_grad():
        ptn_preds = model.forward(
            torch.tensor(sup_fp).to(device),
            torch.tensor(sup_lbl).to(device),
            torch.tensor(qry_fp).to(device),
        ).cpu().numpy()
    m = _m(ptn_preds)
    for key, val in m.items():
        results[f"ptn_{key}" if full_metrics else "ptn"] = val

    return results


# =============================================================================
# TRAIN MODE — sanity check on 100 train assays
# =============================================================================

def _print_train_report(df: pd.DataFrame) -> None:
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
        print(f"\n── Split: {split}  ({len(sub)} episodes, {sub['assay_id'].nunique()} assays, "
              f"mean support-label var = {sub['support_label_var'].mean():.4f}) ──")
        print("-" * 70)
        for col in method_cols:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if vals.empty:
                continue
            mean_mse = vals.mean()
            print(f"  {method_labels.get(col, col):<26} {mean_mse:>10.4f} "
                  f"{np.sqrt(mean_mse):>10.4f} {vals.median():>12.4f}")
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  PTN MSE <  best-KR-Tanimoto  →  PTN improved over baseline  ✓")
    print("  PTN MSE ≈ best-KR-Tanimoto   →  learned metric ≈ raw fingerprint kernel")
    print("  All methods ≈ mean-label MSE  →  ECFP fingerprints carry no signal for this split")
    print("=" * 70)


def run_train_diagnostics() -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model  = _load_model(device)

    all_files    = sorted([os.path.join(FSMOL_TRAIN, f)
                           for f in os.listdir(FSMOL_TRAIN) if f.endswith(".jsonl.gz")])
    chosen_files = rng.choice(all_files, size=min(N_ASSAYS_TRAIN, len(all_files)), replace=False)
    sampler      = EpisodeSampler(N_SUPPORT, N_QUERY)
    min_len      = N_SUPPORT + N_QUERY
    min_grp      = max(2, N_SUPPORT // 4)

    rows, n_usable, n_skipped, n_scaffold_eligible = [], 0, 0, 0
    print(f"Running train diagnostic on up to {len(chosen_files)} assays "
          f"× {N_EPISODES} episodes × 2 split types...")

    for i, fpath in enumerate(chosen_files):
        assay = _load_assay_file(fpath)
        if len(assay) < min_len:
            n_skipped += 1
            continue
        n_usable += 1
        usable_groups = [k for k, v in assay.scaffold_groups.items() if len(v) >= min_grp]
        has_scaffold  = len(usable_groups) >= 2
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
                    None, model, device, full_metrics=False,
                )
                rows.append({
                    "assay_id": assay.assay_id, "split_type": split_type,
                    "episode": ep_i, "n_molecules": len(assay),
                    "n_scaffold_groups": len(assay.scaffold_groups),
                    "n_usable_groups": len(usable_groups),
                    **metrics,
                })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chosen_files)} files | {n_usable} usable "
                  f"| {n_scaffold_eligible} scaffold-eligible", end="\r")

    print(f"\n  Done. {n_usable} usable, {n_skipped} skipped, "
          f"{n_scaffold_eligible} scaffold-eligible.\n")
    df = pd.DataFrame(rows)
    _print_train_report(df)

    out_path = os.path.join(RESULTS_DIR, "diagnostic_baseline_train.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results → {out_path}")
    return df


# =============================================================================
# TEST MODE — all 154 test assays, directly comparable to evaluate_fsmol_test
# =============================================================================

def run_test_diagnostics() -> pd.DataFrame:
    """
    Mirrors evaluate_fsmol_test exactly:
      - All 154 FS-Mol test assays
      - Support sizes: 16 / 32 / 64 / 128 / 256 / 512
      - Split types: random / scaffold / size
      - N_REPEATS=5 per (assay, support_size)
      - Query = all remaining molecules (not just 16)
      - Metrics: MSE, RMSE, Spearman ρ, ΔAUPRC for every baseline + PTN
    """
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model  = _load_model(device)

    test_files = sorted([os.path.join(FSMOL_TEST, f)
                         for f in os.listdir(FSMOL_TEST) if f.endswith(".jsonl.gz")])
    print(f"Test assay files: {len(test_files)}")

    rows = []
    n_skipped = 0

    for file_idx, fpath in enumerate(test_files):
        assay   = _load_assay_file(fpath)
        n_total = len(assay)
        if n_total < MIN_TASK_SIZE:
            n_skipped += 1
            continue

        # Pre-compute split structures (once per assay)
        min_grp       = max(2, min(SUPPORT_SIZES) // 4)
        usable_groups = [k for k, v in assay.scaffold_groups.items() if len(v) >= min_grp]
        has_scaffold  = len(usable_groups) >= 2
        mol_sizes     = _get_mol_sizes(assay)
        sorted_idx    = np.argsort(mol_sizes)
        mid           = n_total // 2
        small_pool    = sorted_idx[:mid]
        large_pool    = sorted_idx[mid:]
        has_size      = len(small_pool) >= 1 and len(large_pool) >= 1

        for n_sup in SUPPORT_SIZES:
            if n_total <= n_sup:
                continue

            for split_type in ["random", "scaffold", "size"]:
                if split_type == "scaffold" and not has_scaffold:
                    continue
                if split_type == "size" and not has_size:
                    continue

                repeat_metrics: list[dict] = []

                for _ in range(N_REPEATS):
                    # Build support / query indices (mirrors evaluate_fsmol_test)
                    if split_type == "random":
                        sup_idx = rng.choice(n_total, size=n_sup, replace=False)
                        sup_set = set(sup_idx.tolist())
                        qry_idx = np.array([j for j in range(n_total) if j not in sup_set])

                    elif split_type == "scaffold":
                        sup_key = usable_groups[rng.randint(len(usable_groups))]
                        sup_pool = assay.scaffold_groups[sup_key]
                        qry_idx = np.array([
                            i for k in assay.scaffold_groups if k != sup_key
                            for i in assay.scaffold_groups[k]
                        ])
                        if len(qry_idx) == 0:
                            continue
                        sup_idx = rng.choice(sup_pool, size=n_sup,
                                             replace=len(sup_pool) < n_sup)

                    elif split_type == "size":
                        sup_idx = rng.choice(small_pool, size=n_sup,
                                             replace=len(small_pool) < n_sup)
                        qry_idx = large_pool.copy()

                    sup_fp  = np.stack([assay.fingerprints[j] for j in sup_idx])
                    sup_lbl = assay.labels[sup_idx]
                    qry_fp  = np.stack([assay.fingerprints[j] for j in qry_idx])
                    qry_lbl = assay.labels[qry_idx]
                    bin_lbl = (assay.binary_labels[qry_idx]
                               if assay.binary_labels is not None else None)

                    m = run_one_episode(
                        sup_fp, sup_lbl, qry_fp, qry_lbl,
                        bin_lbl, model, device, full_metrics=True,
                    )
                    repeat_metrics.append(m)

                if not repeat_metrics:
                    continue

                mean_m = {k: float(np.nanmean([m[k] for m in repeat_metrics]))
                          for k in repeat_metrics[0]}
                rows.append({
                    "assay_id":     assay.assay_id,
                    "split_type":   split_type,
                    "support_size": n_sup,
                    "n_total":      n_total,
                    "n_query":      len(qry_idx),
                    **mean_m,
                })

        if (file_idx + 1) % 20 == 0:
            print(f"  {file_idx + 1}/{len(test_files)} assays...", end="\r")

    print(f"\n  Done. {len(test_files) - n_skipped} assays evaluated, {n_skipped} skipped.\n")
    df = pd.DataFrame(rows)

    # ── Summary report ─────────────────────────────────────────────────────────
    methods = {
        "mean":                "Mean-label",
        "knn_k1":              "kNN (k=1)",
        "knn_k3":              "kNN (k=3)",
        "knn_k5":              "kNN (k=5)",
        "kr_tanimoto_a0.01":   "KR-Tanimoto (α=0.01)",
        "kr_tanimoto_a0.1":    "KR-Tanimoto (α=0.10)",
        "kr_tanimoto_a1.0":    "KR-Tanimoto (α=1.00)",
        "ptn":                 "PTN",
    }

    for split_type in ["random", "scaffold", "size"]:
        sub = df[df["split_type"] == split_type]
        if sub.empty:
            continue
        print(f"\n{'='*80}")
        print(f"Split: {split_type}  ({sub['assay_id'].nunique()} assays)")
        print(f"{'='*80}")
        print(f"{'Method':<26} {'Size':>6} {'ΔAUPRC':>9} {'Spearman':>10} {'RMSE':>8}")
        print("-" * 65)
        for n_sup in SUPPORT_SIZES:
            ssub = sub[sub["support_size"] == n_sup]
            if ssub.empty:
                continue
            for mkey, mlabel in methods.items():
                da_col  = f"{mkey}_delta_auprc" if mkey != "mean" else None
                sp_col  = f"{mkey}_spearman"    if mkey != "mean" else None
                mse_col = f"{mkey}_mse"         if mkey != "mean" else mkey

                # mean-label only has MSE (no spearman/delta_auprc computed)
                da   = ssub[da_col].mean()  if da_col  and da_col  in ssub.columns else float("nan")
                sp   = ssub[sp_col].mean()  if sp_col  and sp_col  in ssub.columns else float("nan")
                rmse = float(np.sqrt(ssub[mse_col].mean())) if mse_col in ssub.columns else float("nan")

                da_str = f"{da:+.4f}" if not np.isnan(da) else "   nan  "
                sp_str = f"{sp:+.4f}" if not np.isnan(sp) else "   nan  "
                print(f"  {mlabel:<24} {n_sup:>6} {da_str:>9} {sp_str:>10} {rmse:>8.4f}")
            print()

    out_path = os.path.join(RESULTS_DIR, "diagnostic_baseline_test.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results → {out_path}")
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "test", "both"], default="train",
        help="train: sanity check on 100 train assays. "
             "test: full comparison on all 154 test assays (directly comparable to main results). "
             "both: run train then test."
    )
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        print("\n" + "=" * 70)
        print("TRAIN MODE — sanity check on sampled train assays")
        print("=" * 70 + "\n")
        run_train_diagnostics()

    if args.mode in ("test", "both"):
        print("\n" + "=" * 70)
        print("TEST MODE — all test assays, directly comparable to main results")
        print("=" * 70 + "\n")
        run_test_diagnostics()
