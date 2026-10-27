#!/usr/bin/env python
"""
Cross-Attention In-Context Learning model training and evaluation.

Edit the CONFIG block, then: python main_attention.py

Architecture: label-conditioned support encoding → self-attention (support) →
cross-attention (query attends to support) → PN head.

    SKIP_TRAINING | EMB_HEADS   | action
    --------------+-------------+------------------------------------------
      False       |  present    | train, then evaluate
      False       | commented   | train only
      True        |  present    | evaluate existing checkpoint
"""
import argparse
import os
import torch

from config import (FSMOL_TRAIN, FSMOL_VAL, FSMOL_TEST, DRUGOOD_DIR,
                    run_ckpt_dir, run_csv_dir, make_run_tag)
from data import load_fsmol_split, load_drugood_split
from model import ECFPEncoder, FSMolGNNEncoder
from context_model import CrossAttentionProtoNet
from featurize import FSMOL_NODE_FEAT_DIM, compute_fsmol_degree_histogram, compute_mic_degree_histogram
from train import pretrain_classification
from evaluate import (evaluate_fsmol_test_grid, evaluate_drugood_multiscale,
                      _load_model_from_checkpoint)


# =============================================================================
# CONFIG
# =============================================================================
ENCODER        = "ecfp"    # "ecfp" | "gnn"
TRAINING_SPLIT = "random"  # "random" | "scaffold" | "similarity"
N_SUPPORT      = 64        # int  OR  list[int] e.g. [16, 32, 64]
SEED           = 0
SKIP_TRAINING  = False

# ── Dataset ───────────────────────────────────────────────────────────────────
# "fsmol"       : 26,868 train assays from FS-Mol (established benchmark)
# "curated_mic" : curated_mic_assays_balanced.csv  (~1,400 assays, pre-labeled)
# "chembl36_mic": chembl36_curated.csv MIC subset  (~8,400 assays, larger/noisier)
DATASET   = "fsmol"   # "fsmol" | "curated_mic" | "chembl36_mic"
MIC_SPLIT = "A"       # for curated_mic / chembl36_mic: "A"=random "B"=doc "C"=time "D"=organism "E"=gram

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

# Cross-attention architecture
D_ATTN     = 256   # attention + representation dim; must be divisible by N_HEADS
D_LABEL    = 16    # label embedding dim
N_HEADS    = 4     # attention heads (self- and cross-attention)
N_SELF     = 1     # self-attention layers over labeled support
N_CROSS    = 1     # cross-attention layers (query attends to support)
MLP_HIDDEN = 256   # hidden dim of MLP_F (encoder_out + label → D_ATTN)
DROPOUT    = 0.1
USE_FFN    = False  # add FFN sublayer after each attention layer [DC-5]

# Design-choice ablation flags
QUERY_LABEL_MODE    = "unknown_token"  # [DC-1]  "unknown_token" | "mask"
USE_NESTED_TENSORS  = True             # [DC-2]  True requires PyTorch ≥ 2.1

# Learned-embedding evaluation heads:
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
    """Return (train_data, val_data) — file lists (FS-Mol) or MicEpisodeDatasets."""
    if DATASET == "fsmol":
        return _assay_files(FSMOL_TRAIN), _assay_files(FSMOL_VAL)
    from mic_loader import MicAssayStore, MicEpisodeDataset, MicGraphEpisodeDataset, load_mic_split
    splits_json = _MIC_SPLITS_JSONS[DATASET]
    if not os.path.exists(splits_json):
        raise FileNotFoundError(
            f"Splits file not found: {splits_json}. Run data/custom/mic_splits.py first."
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
    # Command-line args override the CONFIG block above (useful for cluster job arrays).
    _p = argparse.ArgumentParser()
    _p.add_argument("--dataset",        default=DATASET,        choices=["fsmol","curated_mic","chembl36_mic"])
    _p.add_argument("--mic_split",      default=MIC_SPLIT,      choices=["A","B","C","D","E"])
    _p.add_argument("--encoder",        default=ENCODER,        choices=["ecfp", "gnn"])
    _p.add_argument("--training_split", default=TRAINING_SPLIT, choices=["random", "scaffold", "similarity"])
    _p.add_argument("--n_support",      default=N_SUPPORT, type=int)
    _p.add_argument("--seed",           default=SEED,      type=int)
    _p.add_argument("--skip_training",       action="store_true", default=SKIP_TRAINING)
    _p.add_argument("--query_label_mode",    default=QUERY_LABEL_MODE,
                                             choices=["unknown_token", "mask"])
    _p.add_argument("--use_nested_tensors",  action="store_true", default=USE_NESTED_TENSORS)
    _a = _p.parse_args()
    DATASET             = _a.dataset
    MIC_SPLIT           = _a.mic_split
    ENCODER             = _a.encoder
    TRAINING_SPLIT      = _a.training_split
    N_SUPPORT           = _a.n_support
    SEED                = _a.seed
    SKIP_TRAINING       = _a.skip_training
    QUERY_LABEL_MODE    = _a.query_label_mode
    USE_NESTED_TENSORS  = _a.use_nested_tensors

    emb_heads = list(EMB_HEADS)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_dim   = 512 if ENCODER == "gnn" else 256
    run_tag   = make_run_tag(ENCODER, "context_attention", TRAINING_SPLIT, "euclidean", N_SUPPORT, SEED)
    if DATASET == "fsmol":
        ckpt_dir = run_ckpt_dir(run_tag)
    else:
        ckpt_dir = run_ckpt_dir(f"{DATASET}_split{MIC_SPLIT}/{run_tag}")
    save_path = os.path.join(ckpt_dir, "model.pt")

    print(f"main_attention.py | dataset={DATASET}  mic_split={MIC_SPLIT}  {run_tag}")
    print(f"  d_attn={D_ATTN}  n_heads={N_HEADS}  n_self={N_SELF}  n_cross={N_CROSS}"
          f"  mlp_hidden={MLP_HIDDEN}  use_ffn={USE_FFN}"
          f"  query_label_mode={QUERY_LABEL_MODE}  use_nested_tensors={USE_NESTED_TENSORS}")

    test_assays = None
    if DATASET == "fsmol" and "fsmol_test" in BENCHMARKS and emb_heads:
        test_assays = load_fsmol_split(FSMOL_TEST, max_assays=None)

    model = None
    if not SKIP_TRAINING:
        train_data, val_data = _get_train_val_data()
        encoder  = _build_encoder(train_data)
        lr       = 1e-4 if ENCODER == "gnn" else 1e-3

        ctx_model = CrossAttentionProtoNet(
            encoder, encoder_output_dim=enc_dim,
            d_attn=D_ATTN, d_label=D_LABEL, n_heads=N_HEADS,
            n_self_layers=N_SELF, n_cross_layers=N_CROSS,
            mlp_f_hidden=MLP_HIDDEN, dropout=DROPOUT, use_ffn=USE_FFN,
            query_label_mode=QUERY_LABEL_MODE,
            use_nested_tensors=USE_NESTED_TENSORS,
        )
        ctx_cfg = {
            "model_type": "context_attention",
            "encoder_output_dim": enc_dim,
            "d_attn": D_ATTN, "d_label": D_LABEL, "n_heads": N_HEADS,
            "n_self_layers": N_SELF, "n_cross_layers": N_CROSS,
            "mlp_f_hidden": MLP_HIDDEN, "dropout": DROPOUT, "use_ffn": USE_FFN,
            "query_label_mode": QUERY_LABEL_MODE,
            "use_nested_tensors": USE_NESTED_TENSORS,
        }
        os.makedirs(ckpt_dir, exist_ok=True)
        model = pretrain_classification(
            encoder, train_assays=train_data, val_assays=val_data,
            n_epochs=100, tasks_per_batch=16, n_support=N_SUPPORT, n_query=256,
            n_episodes_train=1600, n_episodes_val=200, lr=lr,
            save_path=save_path, training_split=TRAINING_SPLIT, seed=SEED,
            train_distance="euclidean", model=ctx_model, model_config=ctx_cfg,
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
