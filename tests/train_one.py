"""Train one tiny configuration in its own process, then reload + evaluate it.

One process per configuration, which is how the cluster runs them and which keeps
each run's DataLoader workers isolated. Usage:

    python train_one.py <tag> <json-flags> <out.pt> [--encoder gnn] [--dataset mic]
"""
import json
import os
import sys

import numpy as np
import torch

# Run from anywhere: resolve the PTN repo root from this file and work there,
# since the modules under test open data paths relative to the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from config import FSMOL_TRAIN, FSMOL_VAL
from context_model import CrossAttentionProtoNet
from model import ECFPEncoder, FSMolGNNEncoder, PrototypicalNetworkClassification
from train import pretrain_classification

ENC_DIM = 64


def assay_files(d, n):
    fs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl.gz"))
    return fs[:n]


def main():
    tag = sys.argv[1]
    flags = json.loads(sys.argv[2])
    out = sys.argv[3]
    encoder_kind = flags.pop("_encoder", "ecfp")
    dataset = flags.pop("_dataset", "fsmol")
    train_kw = flags.pop("_train_kw", {})
    plain_ptn = flags.pop("_plain_ptn", False)
    # Pop every underscore-prefixed harness key BEFORE `flags` is splatted into the
    # model constructor, or it arrives there as an unexpected kwarg.
    split = flags.pop("_split", "random")
    leftover = [k for k in flags if k.startswith("_")]
    assert not leftover, f"unconsumed harness keys would reach the model: {leftover}"

    if dataset == "fsmol":
        train_data = assay_files(FSMOL_TRAIN, 40)
        val_data = assay_files(FSMOL_VAL, 4)
    else:
        from mic_loader import MicAssayStore, MicEpisodeDataset, load_mic_split
        store = MicAssayStore("data/custom/curated_mic_assays_balanced.csv")
        tr, va, _ = load_mic_split("data/custom/mic_splits.json", "A")
        train_data = MicEpisodeDataset(store, tr[:60], n_episodes_per_epoch=16,
                                       n_support=16, n_query=32, training_split="random")
        val_data = MicEpisodeDataset(store, va[:20], n_episodes_per_epoch=8,
                                     n_support=16, n_query=32, training_split="random")

    if encoder_kind == "gnn":
        from featurize import FSMOL_NODE_FEAT_DIM, compute_fsmol_degree_histogram
        deg = compute_fsmol_degree_histogram(train_data, n_sample=20)
        enc = FSMolGNNEncoder(node_feat_dim=FSMOL_NODE_FEAT_DIM, hidden_channels=32,
                              num_layers=2, embedding_dim=ENC_DIM, deg=deg, dropout=0.0)
        lr = 1e-4
        if dataset != "fsmol":
            from mic_loader import MicGraphEpisodeDataset
            train_data = MicGraphEpisodeDataset(
                train_data.store, train_data.assay_ids, n_episodes_per_epoch=16,
                n_support=16, n_query=32, training_split="random")
            val_data = MicGraphEpisodeDataset(
                val_data.store, val_data.assay_ids, n_episodes_per_epoch=8,
                n_support=16, n_query=32, training_split="random")
    else:
        enc = ECFPEncoder(input_dim=2048, hidden_dim=128, embedding_dim=ENC_DIM)
        lr = 1e-3

    if plain_ptn:
        model = PrototypicalNetworkClassification(enc)
        cfg = {"model_type": "classification"}
    else:
        arch = dict(d_attn=64, d_label=8, n_heads=4, n_self_layers=1,
                    n_cross_layers=1, mlp_f_hidden=64, dropout=0.1)
        arch.update(flags)
        model = CrossAttentionProtoNet(enc, encoder_output_dim=ENC_DIM, **arch)
        cfg = {"model_type": "context_attention", "encoder_output_dim": ENC_DIM, **arch}

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pretrain_classification(
        enc, train_assays=train_data, val_assays=val_data,
        n_epochs=2, tasks_per_batch=4, n_support=16, n_query=32,
        n_episodes_train=16, n_episodes_val=8, lr=lr,
        save_path=out, training_split=split, seed=0,
        train_distance="euclidean", model=model, model_config=cfg, **train_kw)

    assert os.path.exists(out), "no checkpoint written"
    ck = torch.load(out, map_location="cpu")
    assert np.isfinite(ck["val_delta_auprc"]), "val metric not finite"
    print(f"TRAIN_OK {tag} val_dauprc={ck['val_delta_auprc']:+.4f}")


if __name__ == "__main__":
    main()
