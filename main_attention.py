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
                    run_ckpt_dir, run_csv_dir, make_run_tag, SUPPORT_SAMPLING)
from data import load_fsmol_split, load_drugood_split
from model import ECFPEncoder, FSMolGNNEncoder
from context_model import (CrossAttentionProtoNet, QUERY_LABEL_MODES, NORM_POSITIONS,
                           ATTN_ORDERS, SOFTMAX_MODES, LOGIT_SCALES)
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

# Design-choice ablation flags — see context_model.py docstring for [DC-n] rationale.
# Every one of these is also a CLI flag, so the ablation grid is driven from the
# submit file without editing this block.
QUERY_LABEL_MODE    = "unknown_token"  # [DC-1]  "unknown_token" | "mask"
USE_NESTED_TENSORS  = True             # [DC-2]  True requires PyTorch >= 2.1
NORM_POSITION       = "pre"            # [DC-3]  "pre" | "post"
SEPARATE_MLP_F      = True             # [DC-4]  separate MLP_S/MLP_Q vs one shared
ATTN_ORDER          = "sequential"     # [DC-6]  "sequential" | "interleaved"
SOFTMAX_MODE        = "standard"       # [DC-10] "standard" | "ssmax" | "qassmax"
SHARE_FINAL_NORM    = True             # [DC-12] one metric space for support+query
LOGIT_SCALE         = "sqrt_d"         # [DC-13] "sqrt_d" | "none" | "learned"
# [DC-11] attention diagnostics are not a training knob — see
# Analysis/model/attention_stats.py, which owns that sweep.

# Optimisation stability (TASKS.md §2.8). WARMUP_STEPS>0 and LR_ATTN are the levers
# for the Attn+ECFP divergence on curated_mic splits C/D.
WARMUP_STEPS = 0        # linear LR warmup over this many optimizer steps
GRAD_CLIP    = 10.0     # max grad-norm; <=0 disables
LR_ATTN      = None     # separate LR for attention/MLP params; None = same as encoder

# Training length (TASKS.md 2.13.4). train.py early-stops on val dAUPRC with
# patience=50 for fsmol+GNN and keeps the best-val checkpoint, so N_EPOCHS is a CAP,
# not a target. The 100-epoch screening grid hit this cap while val was still
# climbing, which made the PTN comparison budget-limited rather than architectural.
N_EPOCHS     = 100      # max epochs; early stopping decides the actual length

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


FSMOL_TASKS = "benchmark"   # "benchmark" (4,938 official) | "all" (26,868 raw pool)


def _assay_files(split_dir, split_name=None):
    """List assay files, restricted to the official FS-Mol task list by default.

    FS-Mol ships 26,868 train tasks but the benchmark defines 4,938 (config.py).
    Using all of them is a documented FS-Mol option, but the extra 21,930 are
    unfiltered and the results are not comparable to published FS-Mol baselines.
    """
    files = sorted(os.path.join(split_dir, f)
                   for f in os.listdir(split_dir) if f.endswith(".jsonl.gz"))
    if FSMOL_TASKS == "benchmark" and split_name:
        from data import filter_to_benchmark
        files = filter_to_benchmark(files, split_name)
    return files


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
        return _assay_files(FSMOL_TRAIN, "train"), _assay_files(FSMOL_VAL, "valid")
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


def _arch_suffix() -> str:
    """Run-tag suffix encoding the architecture design choices.

    Without this, two ablations that differ only in e.g. norm_position would resolve to
    the same `outputs/<run_tag>/` folder and silently overwrite each other's
    checkpoint and CSVs. The MLP mode is always included (not only when non-default)
    so new runs cannot collide with checkpoints written before [DC-4] existed, which
    were all shared-MLP but carried no suffix.

    Only `config.make_run_tag`'s output is left untouched, so the PTN/classification
    run tags that Analysis/model/*.py reconstruct are unaffected.
    """
    parts = ["sepmlp" if SEPARATE_MLP_F else "sharedmlp"]
    if NORM_POSITION != "pre":
        parts.append(NORM_POSITION)
    if ATTN_ORDER != "sequential":
        parts.append(ATTN_ORDER)
    if SOFTMAX_MODE != "standard":
        parts.append(SOFTMAX_MODE)
    if QUERY_LABEL_MODE != "unknown_token":
        parts.append("masklbl")
    if USE_FFN:
        parts.append("ffn")
    if not SHARE_FINAL_NORM:
        parts.append("splitnorm")
    if LOGIT_SCALE != "sqrt_d":
        parts.append(f"ls{LOGIT_SCALE}")
    # benchmark (4,938 tasks) vs the full 26,868 pool is a major experimental factor,
    # so it must not silently share an output folder
    if FSMOL_TASKS != "benchmark":
        parts.append("alltasks")
    # [DC-2] is numerically identical to the dense path (asserted by the test suite),
    # but the M13 measurement compares wall-clock and memory, so the two runs still
    # need separate output folders.
    if not USE_NESTED_TENSORS:
        parts.append("dense")
    # Optimisation settings change the trained weights, so they must be in the tag too.
    # Without this, the 2.8 confirmation runs (--warmup_steps 200 --lr_attn 1e-4) would
    # resolve to the baseline's folder and overwrite its checkpoint and CSVs.
    if WARMUP_STEPS:
        parts.append(f"wu{int(WARMUP_STEPS)}")
    if GRAD_CLIP != 10.0:
        parts.append("noclip" if not GRAD_CLIP or GRAD_CLIP <= 0
                     else f"clip{GRAD_CLIP:g}")
    if LR_ATTN is not None:
        parts.append(f"lra{LR_ATTN:g}")
    # d_attn is the dimension of the metric space the PN head computes distances in.
    # PTN's head works in the encoder's 512-d output, so a d_attn=512 arm is what makes
    # the PTN-vs-attention comparison dimension-matched (TASKS.md 2.13.5). Without this
    # in the tag, that arm overwrites the d_attn=256 baseline's checkpoint and CSVs.
    if D_ATTN != 256:
        parts.append(f"d{D_ATTN}")
    if MLP_HIDDEN != 256:
        parts.append(f"mh{MLP_HIDDEN}")
    # A longer run is a different experiment, not a continuation: the checkpoint holds
    # whichever epoch won on val. Keep it in its own folder (TASKS.md 2.13.4).
    if N_EPOCHS != 100:
        parts.append(f"ep{N_EPOCHS}")
    return "_" + "_".join(parts)


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
    def _n_support(v):
        """int, or a comma list for a deliberate multi-size schedule ("16,32,64").

        With skip-don't-shrink the gate applies to the DRAWN size, so a list varies
        only the schedule - not the task pool (TASKS.md 2.12.6).
        """
        for sep in ("+", ","):          # "+" avoids embedding a comma in a
            if sep in v:               # comma-delimited Condor queue row
                return [int(x) for x in v.split(sep)]
        return int(v)

    _p.add_argument("--n_support",      default=N_SUPPORT, type=_n_support,
                    help='int, or a comma list such as "16,32,64"')
    _p.add_argument("--seed",           default=SEED,      type=int)
    _p.add_argument("--fsmol_tasks",    default=FSMOL_TASKS,
                    choices=["benchmark", "all"],
                    help="benchmark = the official 4,938-task list")
    _p.add_argument("--skip_training",       action="store_true", default=SKIP_TRAINING)
    # Design choices. BooleanOptionalAction gives a real --no_<flag> for every boolean;
    # a bare store_true could not turn off a flag whose default is True.
    _p.add_argument("--query_label_mode",    default=QUERY_LABEL_MODE, choices=QUERY_LABEL_MODES)
    _p.add_argument("--norm_position",       default=NORM_POSITION,    choices=NORM_POSITIONS)
    _p.add_argument("--attn_order",          default=ATTN_ORDER,       choices=ATTN_ORDERS)
    _p.add_argument("--softmax_mode",        default=SOFTMAX_MODE,     choices=SOFTMAX_MODES)
    _p.add_argument("--logit_scale",         default=LOGIT_SCALE,      choices=LOGIT_SCALES)
    _p.add_argument("--share_final_norm",    action=argparse.BooleanOptionalAction,
                                             default=SHARE_FINAL_NORM)
    _p.add_argument("--use_nested_tensors",  action=argparse.BooleanOptionalAction,
                                             default=USE_NESTED_TENSORS)
    _p.add_argument("--separate_mlp_f",      action=argparse.BooleanOptionalAction,
                                             default=SEPARATE_MLP_F)
    _p.add_argument("--use_ffn",             action=argparse.BooleanOptionalAction,
                                             default=USE_FFN)
    # Architecture sizes
    _p.add_argument("--d_attn",         default=D_ATTN,     type=int)
    _p.add_argument("--n_heads",        default=N_HEADS,    type=int)
    _p.add_argument("--n_self",         default=N_SELF,     type=int)
    _p.add_argument("--n_cross",        default=N_CROSS,    type=int)
    # MLP_F is Linear(enc+label -> mlp_f_hidden) -> ReLU -> Linear(mlp_f_hidden -> d_attn).
    # If mlp_f_hidden < d_attn the output is confined to a lower-dimensional affine
    # subspace, so a d_attn=512 arm would not really give a 512-d metric space.
    _p.add_argument("--mlp_f_hidden",   default=MLP_HIDDEN, type=int)
    # Optimisation stability
    _p.add_argument("--warmup_steps",   default=WARMUP_STEPS, type=int)
    _p.add_argument("--grad_clip",      default=GRAD_CLIP,    type=float)
    _p.add_argument("--lr_attn",        default=LR_ATTN,      type=float)
    _p.add_argument("--n_epochs",       default=N_EPOCHS,     type=int,
                    help="max epochs; early stopping (patience 50 on fsmol+GNN) "
                         "decides the real length")
    # Resolve everything, print it, and exit without touching data or the GPU. Use it to
    # validate a Condor grid row before submitting the whole queue.
    _p.add_argument("--dry_run", action="store_true",
                    help="print the resolved configuration and exit")
    _a = _p.parse_args()
    DATASET             = _a.dataset
    MIC_SPLIT           = _a.mic_split
    ENCODER             = _a.encoder
    TRAINING_SPLIT      = _a.training_split
    N_SUPPORT           = _a.n_support
    SEED                = _a.seed
    FSMOL_TASKS         = _a.fsmol_tasks
    SKIP_TRAINING       = _a.skip_training
    QUERY_LABEL_MODE    = _a.query_label_mode
    USE_NESTED_TENSORS  = _a.use_nested_tensors
    NORM_POSITION       = _a.norm_position
    SEPARATE_MLP_F      = _a.separate_mlp_f
    ATTN_ORDER          = _a.attn_order
    SOFTMAX_MODE        = _a.softmax_mode
    LOGIT_SCALE         = _a.logit_scale
    SHARE_FINAL_NORM    = _a.share_final_norm
    USE_FFN             = _a.use_ffn
    D_ATTN, N_HEADS     = _a.d_attn, _a.n_heads
    N_SELF, N_CROSS     = _a.n_self, _a.n_cross
    MLP_HIDDEN          = _a.mlp_f_hidden
    WARMUP_STEPS        = _a.warmup_steps
    GRAD_CLIP           = _a.grad_clip
    LR_ATTN             = _a.lr_attn
    N_EPOCHS            = _a.n_epochs

    emb_heads = list(EMB_HEADS)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_dim   = 512 if ENCODER == "gnn" else 256
    run_tag   = make_run_tag(ENCODER, "context_attention", TRAINING_SPLIT,
                             "euclidean", N_SUPPORT, SEED) + _arch_suffix()
    if DATASET == "fsmol":
        ckpt_dir = run_ckpt_dir(run_tag)
    else:
        ckpt_dir = run_ckpt_dir(f"{DATASET}_split{MIC_SPLIT}/{run_tag}")
    save_path = os.path.join(ckpt_dir, "model.pt")

    print(f"main_attention.py | dataset={DATASET}  mic_split={MIC_SPLIT}  {run_tag}")
    print(f"  d_attn={D_ATTN}  n_heads={N_HEADS}  n_self={N_SELF}  n_cross={N_CROSS}"
          f"  mlp_hidden={MLP_HIDDEN}  use_ffn={USE_FFN}")
    print(f"  [DC-1] query_label_mode={QUERY_LABEL_MODE}   "
          f"[DC-2] use_nested_tensors={USE_NESTED_TENSORS}   "
          f"[DC-3] norm_position={NORM_POSITION}")
    print(f"  [DC-4] separate_mlp_f={SEPARATE_MLP_F}   "
          f"[DC-6] attn_order={ATTN_ORDER}   "
          f"[DC-10] softmax_mode={SOFTMAX_MODE}")
    print(f"  [DC-12] share_final_norm={SHARE_FINAL_NORM}   "
          f"[DC-13] logit_scale={LOGIT_SCALE}")
    print(f"  fsmol_tasks={FSMOL_TASKS}")
    print(f"  warmup_steps={WARMUP_STEPS}  grad_clip={GRAD_CLIP}  lr_attn={LR_ATTN}")
    print(f"  support_sampling={SUPPORT_SAMPLING}  (config.SUPPORT_SAMPLING)")
    print(f"  checkpoint -> {save_path}")

    if _a.dry_run:
        # Construct the model so an invalid architecture combination fails here rather
        # than after the data load, then stop before any training or I/O.
        _enc = ECFPEncoder(input_dim=2048, hidden_dim=512, embedding_dim=256)
        CrossAttentionProtoNet(
            _enc, encoder_output_dim=256,
            d_attn=D_ATTN, d_label=D_LABEL, n_heads=N_HEADS,
            n_self_layers=N_SELF, n_cross_layers=N_CROSS,
            mlp_f_hidden=MLP_HIDDEN, dropout=DROPOUT, use_ffn=USE_FFN,
            query_label_mode=QUERY_LABEL_MODE, use_nested_tensors=USE_NESTED_TENSORS,
            norm_position=NORM_POSITION, separate_mlp_f=SEPARATE_MLP_F,
            attn_order=ATTN_ORDER, softmax_mode=SOFTMAX_MODE,
        )
        print("dry run OK - configuration is valid, nothing was trained")
        raise SystemExit(0)

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
            norm_position=NORM_POSITION,
            separate_mlp_f=SEPARATE_MLP_F,
            attn_order=ATTN_ORDER,
            softmax_mode=SOFTMAX_MODE,
            share_final_norm=SHARE_FINAL_NORM,
            logit_scale=LOGIT_SCALE,
            collect_attn_stats=False,   # never during training — forces the slow path
        )
        # Saved into the checkpoint so evaluate.py rebuilds the exact same architecture.
        # Any flag missing here would silently fall back to its loader default.
        ctx_cfg = {
            "model_type": "context_attention",
            "encoder_output_dim": enc_dim,
            "d_attn": D_ATTN, "d_label": D_LABEL, "n_heads": N_HEADS,
            "n_self_layers": N_SELF, "n_cross_layers": N_CROSS,
            "mlp_f_hidden": MLP_HIDDEN, "dropout": DROPOUT, "use_ffn": USE_FFN,
            "query_label_mode": QUERY_LABEL_MODE,
            "use_nested_tensors": USE_NESTED_TENSORS,
            "norm_position": NORM_POSITION,
            "separate_mlp_f": SEPARATE_MLP_F,
            "attn_order": ATTN_ORDER,
            "softmax_mode": SOFTMAX_MODE,
            "share_final_norm": SHARE_FINAL_NORM,
            "logit_scale": LOGIT_SCALE,
            "fsmol_tasks": FSMOL_TASKS,
        }
        os.makedirs(ckpt_dir, exist_ok=True)
        model = pretrain_classification(
            encoder, train_assays=train_data, val_assays=val_data,
            n_epochs=N_EPOCHS, tasks_per_batch=16, n_support=N_SUPPORT, n_query=256,
            n_episodes_train=1600, n_episodes_val=200, lr=lr,
            save_path=save_path, training_split=TRAINING_SPLIT, seed=SEED,
            train_distance="euclidean", model=ctx_model, model_config=ctx_cfg,
            warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, lr_attn=LR_ATTN,
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
            _print_summary(df, f"{run_tag} | emb")
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
