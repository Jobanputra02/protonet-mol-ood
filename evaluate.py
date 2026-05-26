"""
Evaluation: DrugOOD OOD Benchmark
===================================
Protocol (CHOSEN: zero-shot):
    - Load pretrained model, freeze weights (no gradient updates)
    - Context set sampled from DrugOOD train (in-distribution support)
    - Query set = ood_test (OOD) or iid_test (IID)
    - Test multiple context sizes: 64, 128, 256, 512
    - Primary metric: Delta AUPRC (AUPRC(model) - fraction_actives)
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
from model import (PrototypicalNetworkRegression, PrototypicalNetworkClassification,
                   ECFPEncoder, PNAGNNEncoder)
from data import DrugOODEvalDataset, AssayDataset

from config import RESULTS_DIR


# =============================================================================
# METRICS
# =============================================================================

def delta_auprc(predictions_continuous: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    ΔAUPRC = AUPRC(model) − fraction_of_actives_in_query.
    The subtracted term is the random-classifier baseline (FS-Mol paper, eq. 1).
    """
    auprc_model     = average_precision_score(binary_labels, predictions_continuous)
    random_baseline = float(binary_labels.mean())
    return float(auprc_model - random_baseline)


def spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(predictions) < 3:
        return float("nan")
    result = stats.spearmanr(predictions, targets)
    return float(result.statistic)  # type: ignore[union-attr]


def _unwrap(model):
    """Return the underlying module, unwrapping torch.compile's OptimizedModule if present."""
    return getattr(model, '_orig_mod', model)


def _is_gnn(model) -> bool:
    return isinstance(_unwrap(model).encoder, PNAGNNEncoder)


def _mol_input(assay: AssayDataset, indices, device: torch.device, gnn: bool):
    """
    Build the encoder input for `indices` from `assay`.
    ECFP: returns Tensor(n, 2048).
    GNN:  builds PyG graphs from SMILES and returns a PyG Batch on `device`.
    """
    if gnn:
        from featurize import smiles_to_graph
        from torch_geometric.data import Batch as PyGBatch, Data
        import torch as _torch
        from featurize import NODE_FEAT_DIM, EDGE_FEAT_DIM
        _dummy = Data(
            x=_torch.zeros((1, NODE_FEAT_DIM), dtype=_torch.float),
            edge_index=_torch.zeros((2, 0), dtype=_torch.long),
            edge_attr=_torch.zeros((0, EDGE_FEAT_DIM), dtype=_torch.float),
        )
        graphs = []
        for i in indices:
            smi = assay.scaffolds[i]          # scaffolds field stores SMILES
            g   = smiles_to_graph(smi)
            graphs.append(g if g is not None else _dummy)
        return PyGBatch.from_data_list(graphs).to(device)
    else:
        return torch.tensor(
            np.stack([assay.fingerprints[i] for i in indices])
        ).to(device)


def _compute_metrics(preds_np, targets_np, binary_labels, is_classification: bool = False):
    """Compute all metrics for one (predictions, targets, binary_labels) triple.

    is_classification=True suppresses RMSE/MAE/Spearman: preds are probabilities [0,1]
    while targets are continuous IC50 values — those distance metrics are meaningless.
    """
    if (binary_labels is not None
            and binary_labels.sum() > 0
            and binary_labels.sum() < len(binary_labels)):
        d_auprc = delta_auprc(preds_np, binary_labels)
    else:
        d_auprc = float("nan")

    if is_classification:
        return {
            "delta_auprc": d_auprc,
            "rmse":        float("nan"),
            "mae":         float("nan"),
            "spearman":    float("nan"),
        }

    return {
        "delta_auprc": d_auprc,
        "rmse":        float(np.sqrt(np.mean((preds_np - targets_np) ** 2))),
        "mae":         float(np.mean(np.abs(preds_np - targets_np))),
        "spearman":    spearman_correlation(preds_np, targets_np),
    }


# =============================================================================
# MULTI-SCALE EVALUATION (Phase 3 — main evaluation function)
# =============================================================================

def _smiles_to_batch(smiles_list: list[str], device: torch.device):
    """Convert a list of SMILES to a PyG Batch on device. Used for GNN DrugOOD eval."""
    from featurize import smiles_to_graph, NODE_FEAT_DIM, EDGE_FEAT_DIM
    from torch_geometric.data import Batch as PyGBatch, Data
    dummy = Data(
        x=torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float),
    )
    graphs = [smiles_to_graph(smi) or dummy for smi in smiles_list]
    return PyGBatch.from_data_list(graphs).to(device)


def _gnn_predict_chunked(
    model, ctx_batch, ctx_lbl: torch.Tensor,
    query_smiles: list[str], device: torch.device,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Run model.forward over query_smiles in chunks to avoid GPU OOM.
    Support (ctx_batch + ctx_lbl) is re-used for every chunk — it is small
    (≤512 molecules) so re-encoding it is cheap compared to holding a 19k-graph
    batch in GPU memory.
    """
    all_preds: list[np.ndarray] = []
    for start in range(0, len(query_smiles), chunk_size):
        chunk_smiles = query_smiles[start : start + chunk_size]
        chunk_batch  = _smiles_to_batch(chunk_smiles, device)
        preds        = model.forward(ctx_batch, ctx_lbl, chunk_batch)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds) if all_preds else np.array([])


def evaluate_drugood_multiscale(
    model: torch.nn.Module,
    eval_datasets: list[DrugOODEvalDataset],
    device: torch.device,
    context_sizes: list[int] = [64, 128, 256, 512],
    seeds: list[int] = [42, 123, 456],
) -> pd.DataFrame:
    """
    Evaluate across all context sizes and both query sets (ood_test + iid_test).
    Each (split × context_size × query_set) combination is run over multiple context
    seeds and averaged — reported as mean ± std across seeds.

    Returns a long-form DataFrame — one row per (split_type × context_size × query_set):
        index | split_type | query_set | context_set_size | actual_context_n | n_test
              | delta_auprc | delta_auprc_std | rmse | rmse_std | mae | spearman
    """
    model.eval()
    gnn             = _is_gnn(model)
    is_classif      = isinstance(_unwrap(model), PrototypicalNetworkClassification)
    rows = []

    with torch.no_grad():
        for eval_dataset in eval_datasets:
            for context_size in context_sizes:
                for query_set in ["ood_test", "iid_test"]:
                    seed_metrics: list[dict] = []

                    for seed in seeds:
                        ctx_fp, ctx_labels, test_fp, test_labels, test_binary = \
                            eval_dataset.get_episode(context_size, query_set, seed)

                        if len(test_fp) == 0:
                            continue

                        if gnn:
                            rng   = np.random.RandomState(seed)
                            n_ctx = min(context_size, len(eval_dataset.context_smiles))
                            idx   = rng.choice(len(eval_dataset.context_smiles), size=n_ctx, replace=False)
                            ctx_batch = _smiles_to_batch(
                                [eval_dataset.context_smiles[i] for i in idx], device
                            )
                            ctx_lbl     = ctx_labels.to(device)
                            query_smiles = (eval_dataset.ood_test_smiles
                                            if query_set == "ood_test"
                                            else eval_dataset.iid_test_smiles)
                            preds_np = _gnn_predict_chunked(
                                model, ctx_batch, ctx_lbl, query_smiles, device, chunk_size=1024
                            )
                        else:
                            ctx_lbl  = ctx_labels.to(device)
                            preds    = model.forward(ctx_fp.to(device), ctx_lbl, test_fp.to(device))
                            preds_np = preds.cpu().numpy()

                        targets_np = test_labels.numpy()
                        seed_metrics.append(_compute_metrics(preds_np, targets_np, test_binary, is_classif))

                    if not seed_metrics:
                        continue

                    # Average metrics across seeds; report std for key metrics
                    mean_m = {k: float(np.nanmean([m[k] for m in seed_metrics])) for k in seed_metrics[0]}
                    std_m  = {k: float(np.std( [m[k] for m in seed_metrics])) for k in seed_metrics[0]}

                    row = {
                        "split_type":        eval_dataset.split_type,
                        "query_set":         query_set,
                        "context_set_size":  context_size,
                        "actual_context_n":  int(len(ctx_labels)),
                        "n_test":            int(len(test_labels)),
                        "n_seeds":           len(seed_metrics),
                        "delta_auprc":       mean_m["delta_auprc"],
                        "delta_auprc_std":   std_m["delta_auprc"],
                        "rmse":              mean_m["rmse"],
                        "rmse_std":          std_m["rmse"],
                        "mae":               mean_m["mae"],
                        "spearman":          mean_m["spearman"],
                        "spearman_std":      std_m["spearman"],
                    }
                    rows.append(row)

                    print(
                        f"{eval_dataset.split_type:35s} | ctx={context_size:4d} | "
                        f"{query_set:8s} | ΔAUPRC: {mean_m['delta_auprc']:+.4f}±{std_m['delta_auprc']:.4f} | "
                        f"RMSE: {mean_m['rmse']:.4f} | Spearman: {mean_m['spearman']:.4f}"
                    )

    df = pd.DataFrame(rows).reset_index(drop=True)
    df.index.name = "index"
    return df


# =============================================================================
# CHECKPOINT LOADER
# =============================================================================

def _build_encoder_from_config(config: dict, device: torch.device) -> torch.nn.Module:
    """Reconstruct the encoder from the config dict stored in a checkpoint."""
    encoder_type  = config.get("encoder_type", "ecfp")
    embedding_dim = config.get("embedding_dim", 256)

    if encoder_type == "gnn":
        from featurize import NODE_FEAT_DIM, EDGE_FEAT_DIM
        deg_list = config.get("deg", None)
        deg = torch.tensor(deg_list, dtype=torch.long) if deg_list is not None else None
        encoder = PNAGNNEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            hidden_channels=config.get("hidden_channels", 128),
            num_layers=config.get("num_layers", 6),
            embedding_dim=embedding_dim,
            deg=deg,
        )
    else:  # ecfp (default — handles old checkpoints without encoder_type key)
        encoder = ECFPEncoder(
            input_dim=2048,
            hidden_dim=config.get("hidden_dim", 512),
            embedding_dim=embedding_dim,
        )
    return encoder.to(device)


def _load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[PrototypicalNetworkRegression | PrototypicalNetworkClassification, dict]:
    """Load a regression or classification model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint["config"]
    model_type = config.get("model_type", "regression")

    encoder = _build_encoder_from_config(config, device)
    cls     = PrototypicalNetworkClassification if model_type == "classification" \
              else PrototypicalNetworkRegression
    model   = cls(encoder).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if model_type == "classification":
        val_metric = checkpoint.get("val_delta_auprc", None)
        val_str    = f"ΔAUPRC={val_metric:+.4f}" if val_metric is not None else "?"
    else:
        val_metric = checkpoint.get("val_rmse", checkpoint.get("val_loss", None))
        val_str    = f"RMSE={val_metric:.4f}" if val_metric is not None else "?"
    enc_type = config.get("encoder_type", "ecfp")
    print(f"Loaded {enc_type}/{model_type} model from epoch {checkpoint['epoch']} (Val {val_str})")
    return model, config


def load_and_evaluate(
    checkpoint_path: str,
    eval_datasets: list[DrugOODEvalDataset],
    context_sizes: list[int] = [64, 128, 256, 512],
    seeds: list[int] = [42, 123, 456],
) -> pd.DataFrame:
    """
    Load pretrained model from checkpoint and run multi-scale DrugOOD evaluation.
    Returns long-form DataFrame. Caller (main.py) is responsible for saving with run_tag.
    Supports both regression and classification checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model_from_checkpoint(checkpoint_path, device)
    return evaluate_drugood_multiscale(model, eval_datasets, device, context_sizes, seeds)


# =============================================================================
# INSIDE-TASK OOD EVALUATION (FS-Mol test assays)
# =============================================================================

def evaluate_inside_task_ood(
    model: torch.nn.Module,
    test_assays: list[AssayDataset],
    device: torch.device,
    n_support: int = 16,
    n_episodes_per_assay: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Inside-task OOD evaluation on FS-Mol test assays.

    For each assay that has >= 2 scaffold groups:
      - Support set: n_support molecules sampled from scaffold group A
      - Query set:   n_query molecules sampled from scaffold group B (different scaffold)
    This tests whether the model can predict activity across scaffold families
    within a single bioactivity task — the within-task OOD scenario.

    Assays with only 1 scaffold group are skipped (no scaffold split possible).

    Returns a DataFrame with one row per assay:
        assay_id | n_scaffold_groups | n_molecules | spearman | rmse | mae | delta_auprc
    """
    model.eval()
    is_classif = isinstance(_unwrap(model), PrototypicalNetworkClassification)
    rng  = np.random.RandomState(seed)
    rows = []

    n_skipped = 0
    with torch.no_grad():
        for assay in test_assays:
            groups = list(assay.scaffold_groups.keys())
            if len(groups) < 2:
                n_skipped += 1
                continue

            episode_metrics: list[dict] = []
            # Only consider scaffold groups with >= 3 molecules as query candidates.
            # Groups of 1-2 have no rank variance → Spearman undefined.
            valid_query_groups = [g for g in groups if len(assay.scaffold_groups[g]) >= 3]
            if len(valid_query_groups) == 0:
                n_skipped += 1
                continue

            for _ in range(n_episodes_per_assay):
                # Support: any scaffold group; query: only groups with ≥3 molecules
                sup_key = groups[rng.randint(len(groups))]
                qry_key = valid_query_groups[rng.randint(len(valid_query_groups))]
                # Ensure support and query come from different groups
                if sup_key == qry_key and len(valid_query_groups) > 1:
                    remaining = [g for g in valid_query_groups if g != sup_key]
                    qry_key = remaining[rng.randint(len(remaining))]

                sup_pool = assay.scaffold_groups[sup_key]
                qry_pool = assay.scaffold_groups[qry_key]

                sup_idx = rng.choice(sup_pool, size=n_support, replace=len(sup_pool) < n_support)
                qry_idx = np.array(qry_pool)

                sup_fp  = _mol_input(assay, sup_idx, device, _is_gnn(model))
                sup_lbl = torch.tensor(assay.labels[sup_idx]).to(device)
                qry_fp  = _mol_input(assay, qry_idx, device, _is_gnn(model))
                qry_lbl = assay.labels[qry_idx]

                preds    = model.forward(sup_fp, sup_lbl, qry_fp).cpu().numpy()
                bin_lbl  = assay.binary_labels[qry_idx] if assay.binary_labels is not None else None
                episode_metrics.append(_compute_metrics(preds, qry_lbl, bin_lbl, is_classif))

            mean_m = {k: float(np.nanmean([m[k] for m in episode_metrics])) for k in episode_metrics[0]}
            rows.append({
                "assay_id":          assay.assay_id,
                "n_scaffold_groups": len(groups),
                "n_molecules":       len(assay),
                **mean_m,
            })

    df = pd.DataFrame(rows).reset_index(drop=True)
    n_valid_sp = int(df["spearman"].notna().sum())
    print(f"\nInside-task OOD: {len(df)} assays evaluated, {n_skipped} skipped (no scaffold group with ≥3 molecules).")
    print(f"  Mean Spearman : {df['spearman'].mean(skipna=True):.4f}  ({n_valid_sp}/{len(df)} assays non-nan)")
    print(f"  Mean RMSE     : {df['rmse'].mean(skipna=True):.4f}")
    print(f"  Mean ΔAUPRC   : {df['delta_auprc'].mean(skipna=True):.4f}")
    return df


# =============================================================================
# FS-MOL TEST EVALUATION  —  3-curve line plot (Fig 2a / Fig 2b)
# =============================================================================

def _get_mol_sizes(assay: AssayDataset) -> np.ndarray:
    """
    Heavy atom count per molecule.
    assay.scaffolds stores the original SMILES when loaded via _load_assay_file.
    """
    from rdkit import Chem  # type: ignore
    sizes = []
    for smi in assay.scaffolds:
        mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
        sizes.append(mol.GetNumHeavyAtoms() if mol is not None else 0)
    return np.array(sizes, dtype=np.int32)


def evaluate_fsmol_test(
    model: torch.nn.Module,
    test_assay_files: list[str],
    device: torch.device,
    support_sizes: list[int] = [16, 32, 128, 256],
    n_repeats: int = 5,
    seed: int = 42,
    split_type: str = "random",   # "random" | "scaffold" | "size"
    save_preds_path: str | None = None,
) -> pd.DataFrame:
    """
    FS-Mol test evaluation with support-size sweep — produces one curve in Fig 2a.
    Call three times (split_type = "random", "scaffold", "size") for the full figure.

    Split types:
        "random"   — IID baseline: support and query both drawn randomly from assay.
        "scaffold" — inside-task scaffold OOD: support drawn from one Murcko scaffold
                     group; query = all molecules in every other scaffold group.
        "size"     — size shift: support drawn from small molecules (bottom 50% by
                     heavy atom count); query = all large molecules (top 50%).

    For each (assay, support_size), n_repeats episodes are run with different random
    support draws; metrics are nanmean'd across repeats.

    Returns one row per (assay × support_size):
        assay_id | split_type | support_size | n_total | n_query
                 | delta_auprc | rmse | mae | spearman

    MIN_TASK_SIZE=32 is applied consistently: assays with fewer exact-measurement
    molecules are skipped regardless of support_size.
    """
    from data import _load_assay_file

    MIN_TASK_SIZE = 32   # must match main.py — skip assays too small after exact-filtering
    min_grp = max(2, min(support_sizes) // 4)  # scaffold split: min molecules per group

    model.eval()
    is_classif = isinstance(_unwrap(model), PrototypicalNetworkClassification)
    rng       = np.random.RandomState(seed)
    rows:      list[dict] = []
    pred_rows: list[dict] = []
    save_preds = save_preds_path is not None

    for file_idx, fpath in enumerate(test_assay_files):
        assay   = _load_assay_file(fpath)
        n_total = len(assay)

        # Fix 3: apply MIN_TASK_SIZE consistently across all support sizes
        if n_total < MIN_TASK_SIZE:
            continue

        # Pre-compute per-assay structures for non-random splits (done once per assay)
        usable_groups: list[str] = []
        small_pool = large_pool = np.array([], dtype=np.int64)

        if split_type == "scaffold":
            usable_groups = [k for k, v in assay.scaffold_groups.items()
                             if len(v) >= min_grp]
            if len(usable_groups) < 2:
                continue  # not enough scaffold diversity — skip assay entirely

        elif split_type == "size":
            mol_sizes  = _get_mol_sizes(assay)
            sorted_idx = np.argsort(mol_sizes)
            mid        = n_total // 2
            small_pool = sorted_idx[:mid]
            large_pool = sorted_idx[mid:]
            if len(small_pool) < 1 or len(large_pool) == 0:
                continue

        for n_sup in support_sizes:
            if n_total <= n_sup:
                continue

            repeat_metrics: list[dict] = []
            last_qry_idx = np.array([], dtype=np.int64)

            for rep_i in range(n_repeats):

                # ── Build support / query indices ────────────────────────────
                if split_type == "random":
                    sup_idx = rng.choice(n_total, size=n_sup, replace=False)
                    sup_set = set(sup_idx.tolist())
                    qry_idx = np.array([j for j in range(n_total) if j not in sup_set])

                elif split_type == "scaffold":
                    sup_key  = usable_groups[rng.randint(len(usable_groups))]
                    sup_pool = assay.scaffold_groups[sup_key]
                    qry_idx  = np.array([
                        i for k in assay.scaffold_groups if k != sup_key
                        for i in assay.scaffold_groups[k]
                    ])
                    if len(qry_idx) == 0:
                        continue
                    sup_idx = rng.choice(sup_pool, size=n_sup,
                                         replace=len(sup_pool) < n_sup)

                elif split_type == "size":
                    # Support: n_sup from small pool (replace only if pool too small)
                    # Query:   all large molecules (fixed across repeats)
                    sup_idx = rng.choice(small_pool, size=n_sup,
                                         replace=len(small_pool) < n_sup)
                    qry_idx = large_pool.copy()

                # ── Forward pass ─────────────────────────────────────────────
                sup_fp  = _mol_input(assay, sup_idx, device, _is_gnn(model))
                sup_lbl = torch.tensor(assay.labels[sup_idx]).to(device)
                qry_fp  = _mol_input(assay, qry_idx, device, _is_gnn(model))
                qry_lbl = assay.labels[qry_idx]
                bin_lbl = (assay.binary_labels[qry_idx]
                           if assay.binary_labels is not None else None)

                with torch.no_grad():
                    preds = model.forward(sup_fp, sup_lbl, qry_fp).cpu().numpy()

                repeat_metrics.append(_compute_metrics(preds, qry_lbl, bin_lbl, is_classif))
                last_qry_idx = qry_idx

                if save_preds:
                    for mol_i, (p, t) in enumerate(zip(preds, qry_lbl)):
                        pred_rows.append({
                            "assay_id":     assay.assay_id,
                            "split_type":   split_type,
                            "support_size": n_sup,
                            "repeat":       rep_i,
                            "mol_idx":      int(qry_idx[mol_i]),
                            "pred":         float(p),
                            "target":       float(t),
                            "binary_label": int(bin_lbl[mol_i]) if bin_lbl is not None else -1,
                        })

            if not repeat_metrics:
                continue

            mean_m = {k: float(np.nanmean([m[k] for m in repeat_metrics]))
                      for k in repeat_metrics[0]}
            rows.append({
                "assay_id":     assay.assay_id,
                "split_type":   split_type,
                "support_size": n_sup,
                "n_total":      n_total,
                "n_query":      int(len(last_qry_idx)),
                **mean_m,
            })

        if (file_idx + 1) % 20 == 0:
            print(f"  [{split_type}] {file_idx + 1}/{len(test_assay_files)} assays...",
                  end="\r")

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n  No rows for split_type='{split_type}' — check assay sizes / scaffold diversity.")
        return df

    n_assays = df["assay_id"].nunique()
    print(f"\n  Done [{split_type}]. {n_assays} assays × {len(support_sizes)} support sizes.")
    print(f"\n=== FS-Mol Test [{split_type}]: mean metrics per support size ===")
    print(df.groupby("support_size")[["delta_auprc", "spearman", "rmse"]].mean().round(4))

    if save_preds and pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(save_preds_path, index=False)
        print(f"  Predictions saved → {save_preds_path}  ({len(pred_df):,} rows)")

    return df
