"""
Training Loop: FS-Mol Pretraining
==================================
Epoch-based training with gradient accumulation matching the FS-Mol paper protocol:
  - n_epochs outer loop; validates and saves at end of each epoch
  - tasks_per_batch: how many episodes are accumulated per optimizer step (default 16)
  - Streams from ALL training files — no pool size limit
  - Constant LR (FS-Mol paper default; no scheduler)

Both GNN and ECFP use batch_size=1 and accumulate tasks_per_batch forward passes
per optimizer step. Streaming over all 26k assays produces variable query sizes
(small assays may have fewer molecules than n_query), so batching > 1 episode
would fail collation. Gradient accumulation over tasks_per_batch passes produces
the same effective gradient as a single batched forward pass over tasks_per_batch.
"""

import random
from typing import Union
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import (PrototypicalNetworkRegression, PrototypicalNetworkClassification,
                   PNAGNNEncoder, FSMolGNNEncoder)
from data import FSMolEpisodeDataset, FSMolGraphEpisodeDataset, graph_episode_collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _encoder_config(encoder) -> dict:
    """Serialisable config dict for a given encoder — stored inside each checkpoint."""
    if isinstance(encoder, FSMolGNNEncoder):
        deg = encoder.deg
        return {
            "encoder_type":    "fsmol_gnn",
            "hidden_channels": encoder.node_emb.out_features,
            "num_layers":      len(encoder.gnn_layers),
            "embedding_dim":   encoder.fc[-1].out_features,
            "deg":             deg.tolist() if deg is not None else None,
        }
    elif isinstance(encoder, PNAGNNEncoder):
        deg = encoder.deg
        return {
            "encoder_type":    "gnn",
            "hidden_channels": encoder.convs[0].in_channels,
            "num_layers":      len(encoder.convs),
            "embedding_dim":   encoder.output_proj[-1].out_features,
            "deg":             deg.tolist() if deg is not None else None,
        }
    else:  # ECFPEncoder
        return {
            "encoder_type":  "ecfp",
            "hidden_dim":    encoder.network[0].out_features,
            "embedding_dim": encoder.network[-1].out_features,
        }


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a different random seed so episodes don't correlate."""
    seed = int(torch.initial_seed()) % (2 ** 32) + worker_id
    np.random.seed(seed)
    random.seed(seed)


def _make_loaders(encoder, train_files, val_files,
                  n_episodes_train, n_episodes_val,
                  n_support, n_query, shift_aware,
                  use_binary_labels: bool = False):
    """Return (train_loader, val_loader)."""
    is_fsmol_gnn = isinstance(encoder, FSMolGNNEncoder)
    is_gnn       = isinstance(encoder, (PNAGNNEncoder, FSMolGNNEncoder))
    DatasetCls   = FSMolGraphEpisodeDataset if is_gnn else FSMolEpisodeDataset
    collate_fn   = graph_episode_collate if is_gnn else None
    gnn_kwargs   = {"fsmol_style": is_fsmol_gnn} if is_gnn else {}

    # batch_size=1 for all encoders — streaming assays have variable query sizes
    # (small assays may have fewer than n_query remaining molecules after support sampling),
    # so stacking a batch of 16 episodes would fail. Gradient accumulation over
    # tasks_per_batch forward passes produces the same effective gradient.
    train_batch_size = 1

    train_dataset = DatasetCls(
        train_files, n_episodes_train, n_support, n_query,
        shift_aware=shift_aware, use_binary_labels=use_binary_labels, **gnn_kwargs,
    )
    val_dataset = DatasetCls(
        val_files, n_episodes_val, n_support, n_query,
        shift_aware=False, use_binary_labels=use_binary_labels, **gnn_kwargs,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False,
        num_workers=4, pin_memory=not is_gnn, prefetch_factor=4,
        collate_fn=collate_fn, worker_init_fn=_worker_init_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=not is_gnn, prefetch_factor=2,
        collate_fn=collate_fn, worker_init_fn=_worker_init_fn,
        persistent_workers=True,
    )
    return train_loader, val_loader


def _batch_to_device(batch, device):
    """Move (support_input, support_labels, query_input, query_labels) to device."""
    s_in, s_lbl, q_in, q_lbl = batch
    return s_in.to(device), s_lbl.to(device), q_in.to(device), q_lbl.to(device)


# =============================================================================
# CLASSIFICATION TRAINING
# =============================================================================

def pretrain_classification(
    encoder,
    train_assays: list[str],          # .jsonl.gz file paths — ALL training files
    val_assays: list[str],            # .jsonl.gz file paths — all val files
    n_epochs: int = 100,
    tasks_per_batch: int = 16,        # episodes accumulated per optimizer step
    n_support: Union[int, list] = 64,  # fixed int or list[int] drawn per episode
    n_query: int = 256,               # FS-Mol paper episode size
    n_episodes_train: int = 1000,     # episodes per epoch
    n_episodes_val: int = 200,        # val episodes per epoch (~5 per val assay)
    lr: float = 1e-4,                 # FS-Mol paper default (GNN); 1e-3 for ECFP
    save_path: str = "ptn_ecfp_classification_shift_aware.pt",
    shift_aware: bool = True,
    seed: int = 42,
):
    """
    Epoch-based episodic training for PrototypicalNetworkClassification (BCE loss).
    Gradient accumulation over tasks_per_batch=16 episodes per optimizer step.
    Streams from ALL training files each epoch. Validates and saves per epoch.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    is_gnn  = isinstance(encoder, (PNAGNNEncoder, FSMolGNNEncoder))
    n_accum = tasks_per_batch

    train_loader, val_loader = _make_loaders(
        encoder, train_assays, val_assays,
        n_episodes_train, n_episodes_val,
        n_support, n_query, shift_aware,
        use_binary_labels=True,   # classification always trains with pre-binarised ChEMBL labels
    )

    model     = PrototypicalNetworkClassification(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp   = (device.type == "cuda")
    amp_dtype = (torch.bfloat16
                 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    print(f"  AMP dtype: {amp_dtype} (bf16 supported: {torch.cuda.is_bf16_supported() if use_amp else 'N/A'})")

    encoder_cfg         = _encoder_config(encoder)
    best_val_dauprc     = -float("inf")
    no_improve          = 0
    early_stop_patience = 50 if is_gnn else 25

    print(f"  Epochs: {n_epochs}  |  Tasks/step: {tasks_per_batch}  |  LR: {lr}")
    print(f"  Train files: {len(train_assays)}  |  Val files: {len(val_assays)}")
    print(f"  Episodes/epoch: {n_episodes_train}  |  Steps/epoch: ~{n_episodes_train // tasks_per_batch}")

    for epoch in range(1, n_epochs + 1):
        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        accum_count    = 0
        step           = 0
        epoch_loss_buf:   list[float] = []
        epoch_dauprc_buf: list[float] = []

        for batch in train_loader:
            s_in, s_lbl, q_in, q_lbl = _batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, metrics = model.compute_loss_batched(s_in, s_lbl, q_in, q_lbl)

            (loss / n_accum).backward()
            accum_count += 1
            epoch_loss_buf.append(loss.item())
            if not np.isnan(metrics["delta_auprc"]):
                epoch_dauprc_buf.append(metrics["delta_auprc"])

            if accum_count == n_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                step += 1

        # discard any partial accumulated batch at epoch end
        if accum_count > 0:
            optimizer.zero_grad()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_dauprc_buf: list[float] = []
        val_bce_buf:    list[float] = []
        with torch.no_grad():
            for vbatch in val_loader:
                vs_in, vs_lbl, vq_in, vq_lbl = _batch_to_device(vbatch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    vloss, vmet = model.compute_loss_batched(vs_in, vs_lbl, vq_in, vq_lbl)
                val_bce_buf.append(vloss.item())
                if not np.isnan(vmet["delta_auprc"]):
                    val_dauprc_buf.append(vmet["delta_auprc"])

        train_dauprc = float(np.nanmean(epoch_dauprc_buf)) if epoch_dauprc_buf else float("nan")
        train_loss   = float(np.mean(epoch_loss_buf))       if epoch_loss_buf   else float("nan")
        val_dauprc   = float(np.mean(val_dauprc_buf))       if val_dauprc_buf   else float("nan")

        print(
            f"Epoch {epoch:3d}/{n_epochs} | Steps: {step} | "
            f"Train BCE: {train_loss:.4f} | Train ΔAUPRC: {train_dauprc:+.4f} | "
            f"Val ΔAUPRC: {val_dauprc:+.4f}"
        )

        # ------------------------------------------------------------------
        # Checkpoint + early stopping
        # ------------------------------------------------------------------
        if val_dauprc > best_val_dauprc:
            best_val_dauprc = val_dauprc
            no_improve      = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_delta_auprc":     best_val_dauprc,
                "config": {
                    "model_type":  "classification",
                    "n_support":   n_support,
                    "shift_aware": shift_aware,
                    **encoder_cfg,
                },
            }, save_path)
            print(f"  → Saved best model (Val ΔAUPRC: {best_val_dauprc:+.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs).")
                break

    print(f"\nPretraining complete. Best Val ΔAUPRC: {best_val_dauprc:+.4f}")
    return model


# =============================================================================
# REGRESSION TRAINING
# =============================================================================

def pretrain_regression(
    encoder,
    train_assays: list[str],
    val_assays: list[str],
    n_epochs: int = 100,
    tasks_per_batch: int = 16,
    n_support: Union[int, list] = 64,  # fixed int or list[int] drawn per episode
    n_query: int = 256,
    n_episodes_train: int = 1000,
    n_episodes_val: int = 200,
    lr: float = 1e-4,
    save_path: str = "ptn_ecfp_regression_shift_aware.pt",
    shift_aware: bool = True,
    seed: int = 42,
):
    """
    Epoch-based episodic training for PrototypicalNetworkRegression (MSE loss).
    Same protocol as pretrain_classification but minimises RMSE.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    is_gnn  = isinstance(encoder, (PNAGNNEncoder, FSMolGNNEncoder))
    n_accum = tasks_per_batch

    train_loader, val_loader = _make_loaders(
        encoder, train_assays, val_assays,
        n_episodes_train, n_episodes_val,
        n_support, n_query, shift_aware,
    )

    model     = PrototypicalNetworkRegression(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp   = (device.type == "cuda")
    amp_dtype = (torch.bfloat16
                 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    print(f"  AMP dtype: {amp_dtype} (bf16 supported: {torch.cuda.is_bf16_supported() if use_amp else 'N/A'})")

    encoder_cfg         = _encoder_config(encoder)
    best_val_rmse       = float("inf")
    no_improve          = 0
    early_stop_patience = 50 if is_gnn else 25

    print(f"  Epochs: {n_epochs}  |  Tasks/step: {tasks_per_batch}  |  LR: {lr}")
    print(f"  Train files: {len(train_assays)}  |  Val files: {len(val_assays)}")

    for epoch in range(1, n_epochs + 1):
        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        accum_count   = 0
        step          = 0
        epoch_loss_buf: list[float] = []
        epoch_rmse_buf: list[float] = []

        for batch in train_loader:
            s_in, s_lbl, q_in, q_lbl = _batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, metrics = model.compute_loss_batched(s_in, s_lbl, q_in, q_lbl)

            (loss / n_accum).backward()
            accum_count += 1
            epoch_loss_buf.append(loss.item())
            epoch_rmse_buf.append(metrics["rmse"])

            if accum_count == n_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                step += 1

        if accum_count > 0:
            optimizer.zero_grad()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_rmse_buf: list[float] = []
        val_mae_buf:  list[float] = []
        with torch.no_grad():
            for vbatch in val_loader:
                vs_in, vs_lbl, vq_in, vq_lbl = _batch_to_device(vbatch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    _, vmet = model.compute_loss_batched(vs_in, vs_lbl, vq_in, vq_lbl)
                val_rmse_buf.append(vmet["rmse"])
                val_mae_buf.append(vmet["mae"])

        train_rmse = float(np.mean(epoch_rmse_buf)) if epoch_rmse_buf else float("nan")
        train_loss = float(np.mean(epoch_loss_buf)) if epoch_loss_buf else float("nan")
        val_rmse   = float(np.mean(val_rmse_buf))   if val_rmse_buf   else float("nan")
        val_mae    = float(np.mean(val_mae_buf))     if val_mae_buf    else float("nan")

        print(
            f"Epoch {epoch:3d}/{n_epochs} | Steps: {step} | "
            f"Train Loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improve    = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse":            best_val_rmse,
                "config": {
                    "model_type":  "regression",
                    "n_support":   n_support,
                    "shift_aware": shift_aware,
                    **encoder_cfg,
                },
            }, save_path)
            print(f"  → Saved best model (Val RMSE: {best_val_rmse:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs).")
                break

    print(f"\nPretraining complete. Best Val RMSE: {best_val_rmse:.4f}")
    return model
