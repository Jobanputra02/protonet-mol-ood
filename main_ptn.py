#!/usr/bin/env python
"""
Prototypical Network training and evaluation on FS-Mol / DrugOOD.

Edit the CONFIG block, then: python main_ptn.py

    SKIP_TRAINING | ECFP_HEADS | EMB_HEADS | action
    --------------+------------+-----------+----------------------------------
      False       |  (any)     |  (any)    | train, then evaluate heads
      False       | commented  | commented | train only
      True        |  present   |  present  | evaluate existing checkpoint
      True        |  present   | commented | model-free ECFP baselines only
"""
import os
import torch

from config import (FSMOL_TRAIN, FSMOL_VAL, FSMOL_TEST, DRUGOOD_DIR,
                    BASELINE_CSV_DIR, run_ckpt_dir, run_csv_dir, make_run_tag)
from data import load_fsmol_split, load_drugood_split
from model import ECFPEncoder, FSMolGNNEncoder
from featurize import FSMOL_NODE_FEAT_DIM, compute_fsmol_degree_histogram, compute_mic_degree_histogram
from train import pretrain_classification, pretrain_regression
from evaluate import (evaluate_fsmol_test_grid, evaluate_drugood_multiscale,
                      _load_model_from_checkpoint)


# =============================================================================
# CONFIG
# =============================================================================
MODEL_HEAD     = "classification"   # "classification" | "regression"
ENCODER        = "gnn"              # "ecfp" | "gnn"
TRAINING_SPLIT = "random"           # "random" | "scaffold" | "similarity"
TRAIN_DISTANCE = "euclidean"        # "euclidean" | "mahalanobis"
N_SUPPORT      = 64                 # int  OR  list[int] e.g. [16, 32, 64]
SEED           = 0
SKIP_TRAINING  = False

# ── Dataset ───────────────────────────────────────────────────────────────────
# "fsmol"       : 26,868 train assays from FS-Mol (established benchmark)
# "curated_mic" : curated_mic_assays_balanced.csv  (~1,400 assays, pre-labeled)
# "chembl36_mic": chembl36_curated.csv MIC subset  (~8,400 assays, larger/noisier)
DATASET   = "fsmol"   # "fsmol" | "curated_mic" | "chembl36_mic"
MIC_SPLIT = "A"       # for curated_mic / chembl36_mic: "A"=random "B"=doc "C"=time "D"=organism "E"=gram

# Paths and column configs for MIC datasets (relative to PTN root)
_MIC_CSVS = {
    "curated_mic":  "data/custom/curated_mic_assays_balanced.csv",
    "chembl36_mic": "data/custom/chembl36_curated.csv",
}
_MIC_SPLITS_JSONS = {
    "curated_mic":  "data/custom/mic_splits.json",
    "chembl36_mic": "data/custom/mic_splits_chembl36.json",
}
# Column configs passed to MicAssayStore — keys match MicAssayStore.__init__ kwargs
_MIC_STORE_CONFIGS = {
    "curated_mic": {
        "assay_col": "assay_chembl_id",
        "smiles_col": "canonical_smiles",
        "label_col": "is_potent",
        "label_threshold": None,
        "filter_col": None,
        "filter_val": None,
    },
    "chembl36_mic": {
        "assay_col": "group",
        "smiles_col": "canonical_smiles",
        "label_col": "target_transformed",
        "label_threshold": 6.0,      # pMIC >= 6  →  MIC <= 1 µM (potent)
        "filter_col": "standard_type",
        "filter_val": "MIC",
    },
}

# Model-free ECFP baseline heads (comment all to skip):
ECFP_HEADS = [
    # "ecfp_proto_euclid",
    # "ecfp_proto_tanimoto",
    # "ecfp_gp_tanimoto",
    # "ecfp_logreg",
    # "ecfp_rf",
]

# Learned-embedding heads (comment all to skip model evaluation):
EMB_HEADS = [
    "emb_proto_euclid",
    "emb_proto_mahalanobis",
    "emb_logreg",
    "emb_knn",
]

BENCHMARKS    = ("fsmol_test", "drugood")
SPLITS        = ("random", "scaffold", "similarity", "size")
SUPPORT_SIZES = (16, 32, 64, 128, 256, 512)
N_REPEATS     = 5
# =============================================================================


def _assay_files(split_dir):
    return sorted(os.path.join(split_dir, f)
                  for f in os.listdir(split_dir) if f.endswith(".jsonl.gz"))


def _build_encoder(train_data):
    if ENCODER == "gnn":
        if DATASET == "fsmol":
            deg = compute_fsmol_degree_histogram(train_data, n_sample=500)
        else:
            deg = compute_mic_degree_histogram(train_data.store, n_sample=500)
        return FSMolGNNEncoder(node_feat_dim=FSMOL_NODE_FEAT_DIM, hidden_channels=128,
                               num_layers=10, embedding_dim=512, deg=deg, dropout=0.0)
    return ECFPEncoder(input_dim=2048, hidden_dim=512, embedding_dim=256)


def _get_train_val_data():
    """Return (train_data, val_data) — either file lists (FS-Mol) or MicEpisodeDatsets."""
    if DATASET == "fsmol":
        return _assay_files(FSMOL_TRAIN), _assay_files(FSMOL_VAL)
    # MIC path
    from mic_loader import MicAssayStore, MicEpisodeDataset, MicGraphEpisodeDataset, load_mic_split
    splits_json = _MIC_SPLITS_JSONS[DATASET]
    if not os.path.exists(splits_json):
        raise FileNotFoundError(
            f"Splits file not found: {splits_json}. "
            f"Run data/custom/mic_splits.py first."
        )
    store = MicAssayStore(_MIC_CSVS[DATASET], **_MIC_STORE_CONFIGS[DATASET])
    train_ids, val_ids, _ = load_mic_split(splits_json, MIC_SPLIT)
    DatasetCls = MicGraphEpisodeDataset if ENCODER == "gnn" else MicEpisodeDataset
    train_ds = DatasetCls(store, train_ids, n_episodes_per_epoch=1600,
                          n_support=N_SUPPORT, n_query=256,
                          training_split=TRAINING_SPLIT, use_binary_labels=True)
    val_ds   = DatasetCls(store, val_ids,   n_episodes_per_epoch=200,
                          n_support=N_SUPPORT, n_query=256,
                          training_split=TRAINING_SPLIT, use_binary_labels=True)
    return train_ds, val_ds


def _print_summary(df, label):
    print(f"\n=== FS-Mol test summary [{label}] ===")
    for split in sorted(df["split_type"].unique()):
        sub = df[df["split_type"] == split]
        tab = sub.groupby(["head", "support_size"]).agg(
            dAUPRC=("delta_auprc", "mean"), AUROC=("auroc", "mean"),
            n_assays=("assay_id", "nunique")).round(3)
        print(f"\n-- split = {split} --")
        print(tab.to_string())


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--dataset",        default=DATASET,        choices=["fsmol","curated_mic","chembl36_mic"])
    _p.add_argument("--mic_split",      default=MIC_SPLIT,      choices=["A","B","C","D","E"])
    _p.add_argument("--encoder",        default=ENCODER,        choices=["ecfp","gnn"])
    _p.add_argument("--training_split", default=TRAINING_SPLIT, choices=["random","scaffold","similarity"])
    _p.add_argument("--n_support",      default=N_SUPPORT,      type=int)
    _p.add_argument("--seed",           default=SEED,           type=int)
    _p.add_argument("--skip_training",  action="store_true",    default=SKIP_TRAINING)
    _a = _p.parse_args()
    DATASET        = _a.dataset
    MIC_SPLIT      = _a.mic_split
    ENCODER        = _a.encoder
    TRAINING_SPLIT = _a.training_split
    N_SUPPORT      = _a.n_support
    SEED           = _a.seed
    SKIP_TRAINING  = _a.skip_training

    ecfp_heads = list(ECFP_HEADS)
    emb_heads  = list(EMB_HEADS)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tag    = make_run_tag(ENCODER, MODEL_HEAD, TRAINING_SPLIT, TRAIN_DISTANCE, N_SUPPORT, SEED)
    if DATASET == "fsmol":
        ckpt_dir = run_ckpt_dir(run_tag)
    else:
        ckpt_dir = run_ckpt_dir(f"{DATASET}_split{MIC_SPLIT}/{run_tag}")
    save_path  = os.path.join(ckpt_dir, "model.pt")

    print(f"main_ptn.py | dataset={DATASET}  mic_split={MIC_SPLIT}  {run_tag}")

    test_assays = None
    if DATASET == "fsmol" and "fsmol_test" in BENCHMARKS and (emb_heads or ecfp_heads):
        test_assays = load_fsmol_split(FSMOL_TEST, max_assays=None)

    model = None
    if not SKIP_TRAINING:
        train_data, val_data = _get_train_val_data()
        encoder  = _build_encoder(train_data)
        lr       = 1e-4 if ENCODER == "gnn" else 1e-3
        pretrain_fn = pretrain_classification if MODEL_HEAD == "classification" else pretrain_regression
        os.makedirs(ckpt_dir, exist_ok=True)
        model = pretrain_fn(
            encoder, train_assays=train_data, val_assays=val_data,
            n_epochs=100, tasks_per_batch=16, n_support=N_SUPPORT, n_query=256,
            n_episodes_train=1600, n_episodes_val=200, lr=lr,
            save_path=save_path, training_split=TRAINING_SPLIT, seed=SEED,
            train_distance=TRAIN_DISTANCE,
        )
        model.eval()

    if emb_heads:
        if model is None:
            model, _ = _load_model_from_checkpoint(save_path, device)
            model.eval()
        csv_dir = run_csv_dir(run_tag)
        os.makedirs(csv_dir, exist_ok=True)

        if "fsmol_test" in BENCHMARKS and DATASET == "fsmol":
            df = evaluate_fsmol_test_grid(
                model, test_assays, device, splits=SPLITS, support_sizes=SUPPORT_SIZES,
                n_repeats=N_REPEATS, head_names=emb_heads)
            fsmol_csv = os.path.join(csv_dir, "fsmol_test.csv")
            df.to_csv(fsmol_csv, index=False)
            _print_summary(df, f"{run_tag} · emb")
            print(f"  saved -> {fsmol_csv}")

        if "drugood" in BENCHMARKS and DATASET == "fsmol":
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

    if ecfp_heads and "fsmol_test" in BENCHMARKS:
        bdf = evaluate_fsmol_test_grid(
            None, test_assays, device, splits=SPLITS, support_sizes=SUPPORT_SIZES,
            n_repeats=N_REPEATS, head_names=ecfp_heads)
        os.makedirs(BASELINE_CSV_DIR, exist_ok=True)
        base_csv = os.path.join(BASELINE_CSV_DIR, "fsmol_test.csv")
        bdf.to_csv(base_csv, index=False)
        _print_summary(bdf, "baselines")
        print(f"  saved -> {base_csv}")
