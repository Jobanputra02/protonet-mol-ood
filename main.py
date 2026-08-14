#!/usr/bin/env python
"""
Prototypical networks for few-shot molecular property prediction under distribution shift.

Single entry point. Edit the CONFIG block, then: python main.py

What runs is decided by SKIP_TRAINING and the two head lists:

    SKIP_TRAINING | ECFP_HEADS | EMB_HEADS | action
    --------------+------------+-----------+----------------------------------
      False       |  (any)     |  (any)    | train, save model, then evaluate heads
      False       | commented  | commented | train only (no evaluation)
      True        |  present   |  present  | evaluate an existing model
      True        |  present   | commented | model-free baselines only (from data)

ECFP heads use raw fingerprints and need no model; EMB heads use the trained encoder's
embedding. Re-running retrains and overwrites existing outputs.
"""
import os

import torch

from config import (FSMOL_TRAIN, FSMOL_VAL, FSMOL_TEST, DRUGOOD_DIR,
                    BASELINE_CSV_DIR, run_ckpt_dir, run_csv_dir, make_run_tag)
from data import load_fsmol_split, load_drugood_split
from model import ECFPEncoder, FSMolGNNEncoder
from featurize import FSMOL_NODE_FEAT_DIM, compute_fsmol_degree_histogram
from train import pretrain_classification, pretrain_regression
from evaluate import (evaluate_fsmol_test_grid, evaluate_drugood_multiscale,
                      _load_model_from_checkpoint)


# =============================================================================
# CONFIG
# =============================================================================
MODEL_HEAD      = "classification"   # "regression" | "classification"
ENCODER         = "gnn"              # "ecfp" | "gnn"
TRAINING_SPLIT  = "random"           # "random" | "scaffold" | "similarity"  (episode construction)
TRAIN_DISTANCE  = "euclidean"        # "euclidean" | "mahalanobis" (FS-Mol paper default — test both)
N_SUPPORT       = [16, 32, 64, 128, 256]  # int OR list[int] — list = random per episode
SEED            = 0                  # any but 0 | 1 | 2 suggested
SKIP_TRAINING   = False              # False = train ; True = use an existing model

# Model-free baseline heads on raw ECFP fingerprints (comment all to skip):
ECFP_HEADS = [
    # "ecfp_proto_euclid",
    # "ecfp_proto_tanimoto",
    # "ecfp_gp_tanimoto",
    # "ecfp_logreg",
    # "ecfp_rf",
]
# Trained-encoder heads on the learned embedding (comment all to skip model evaluation):
EMB_HEADS = [
    "emb_proto_euclid",
    "emb_proto_mahalanobis",
    "emb_logreg",
    "emb_knn",
]

BENCHMARKS    = ("fsmol_test", "drugood")   # any subset; drugood uses the model (EMB)
SPLITS        = ("random", "scaffold", "similarity", "size")
SUPPORT_SIZES = (16, 32, 64, 128, 256, 512)
N_REPEATS     = 5
# Scaffold-OOD grouping is Butina@0.70 — fixed design choice (see config.py).
# =============================================================================


def _assay_files(split_dir: str) -> list[str]:
    return sorted(os.path.join(split_dir, f)
                  for f in os.listdir(split_dir) if f.endswith(".jsonl.gz"))


def _print_fsmol_summary(df, label: str) -> None:
    """Print mean ΔAUPRC + AUROC per split_type × support_size (and N assays)."""
    print(f"\n=== FS-Mol test summary [{label}] ===")
    for split in sorted(df["split_type"].unique()):
        sub = df[df["split_type"] == split]
        tab = sub.groupby(["head", "support_size"]).agg(
            dAUPRC=("delta_auprc", "mean"), AUROC=("auroc", "mean"),
            n_assays=("assay_id", "nunique")).round(3)
        print(f"\n-- split = {split} --")
        print(tab.to_string())


def _build_encoder(train_files: list[str]):
    """Construct the encoder. The GNN needs the training-set degree histogram for its PNA scalers."""
    if ENCODER == "gnn":
        deg = compute_fsmol_degree_histogram(train_files, n_sample=500)
        return FSMolGNNEncoder(node_feat_dim=FSMOL_NODE_FEAT_DIM, hidden_channels=128,
                               num_layers=10, embedding_dim=512, deg=deg, dropout=0.0)
    return ECFPEncoder(input_dim=2048, hidden_dim=512, embedding_dim=256)


if __name__ == "__main__":
    ecfp_heads = list(ECFP_HEADS)
    emb_heads  = list(EMB_HEADS)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tag    = make_run_tag(ENCODER, MODEL_HEAD, TRAINING_SPLIT, TRAIN_DISTANCE, N_SUPPORT, SEED)
    ckpt_dir   = run_ckpt_dir(run_tag)
    save_path  = os.path.join(ckpt_dir, "model.pt")

    print(f"main.py | {run_tag}")

    # FS-Mol test set is loaded once and reused by both head paths.
    test_assays = None
    if "fsmol_test" in BENCHMARKS and (emb_heads or ecfp_heads):
        test_assays = load_fsmol_split(FSMOL_TEST, max_assays=None)

    # ---- train (or leave the model to be loaded on demand) -----------------
    model = None
    if not SKIP_TRAINING:
        train_files = _assay_files(FSMOL_TRAIN)
        val_files   = _assay_files(FSMOL_VAL)
        encoder     = _build_encoder(train_files)
        # FS-Mol paper LRs: 1e-4 for the GNN, 1e-3 for the fingerprint MLP.
        lr = 1e-4 if ENCODER == "gnn" else 1e-3
        pretrain_fn = pretrain_classification if MODEL_HEAD == "classification" else pretrain_regression
        os.makedirs(ckpt_dir, exist_ok=True)
        model = pretrain_fn(
            encoder, train_assays=train_files, val_assays=val_files,
            n_epochs=100, tasks_per_batch=16, n_support=N_SUPPORT, n_query=256,
            n_episodes_train=1600, n_episodes_val=200, lr=lr,
            save_path=save_path, training_split=TRAINING_SPLIT, seed=SEED,
            train_distance=TRAIN_DISTANCE,
        )
        model.eval()

    # ---- EMB heads (need the trained encoder) ------------------------------
    if emb_heads:
        if model is None:
            model, _ = _load_model_from_checkpoint(save_path, device)
            model.eval()
        csv_dir = run_csv_dir(run_tag)
        os.makedirs(csv_dir, exist_ok=True)

        if "fsmol_test" in BENCHMARKS:
            df = evaluate_fsmol_test_grid(
                model, test_assays, device, splits=SPLITS, support_sizes=SUPPORT_SIZES,
                n_repeats=N_REPEATS, head_names=emb_heads)
            fsmol_csv = os.path.join(csv_dir, "fsmol_test.csv")
            df.to_csv(fsmol_csv, index=False)
            _print_fsmol_summary(df, f"{run_tag} · emb")
            print(f"  saved -> {fsmol_csv}")

        if "drugood" in BENCHMARKS:
            datasets = [
                load_drugood_split(os.path.join(DRUGOOD_DIR, f"lbap_core_ic50_{s}.json"),
                                   split_type=f"lbap_core_ic50_{s}")
                for s in ("scaffold", "size", "assay")
            ]
            ddf = evaluate_drugood_multiscale(
                model, datasets, device,
                context_sizes=list(SUPPORT_SIZES), seeds=list(range(N_REPEATS)))
            ddf.to_csv(os.path.join(csv_dir, "drugood.csv"), index=False)
            print("\n=== DrugOOD summary (ood_test, mean over context sizes) ===")
            print(ddf[ddf["query_set"] == "ood_test"]
                  .groupby("split_type")[["auroc", "delta_auprc"]].mean().round(3).to_string())

    # ---- ECFP heads (model-free, straight from data) -----------------------
    if ecfp_heads and "fsmol_test" in BENCHMARKS:
        bdf = evaluate_fsmol_test_grid(
            None, test_assays, device, splits=SPLITS, support_sizes=SUPPORT_SIZES,
            n_repeats=N_REPEATS, head_names=ecfp_heads)
        os.makedirs(BASELINE_CSV_DIR, exist_ok=True)
        base_csv = os.path.join(BASELINE_CSV_DIR, "fsmol_test.csv")
        bdf.to_csv(base_csv, index=False)
        _print_fsmol_summary(bdf, "baselines")
        print(f"  saved -> {base_csv}")
