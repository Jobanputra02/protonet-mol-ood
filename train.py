"""
Training Loop: FS-Mol Pretraining
==================================
Episodic training on FS-Mol assays.
Each step: sample episode → forward pass → MSE loss → backprop → update encoder.
"""

import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import (PrototypicalNetworkRegression, PrototypicalNetworkClassification,
                   PNAGNNEncoder)
from data import FSMolEpisodeDataset, FSMolGraphEpisodeDataset, graph_episode_collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch_regression(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches  = 0
    use_amp    = device.type == "cuda"

    for batch in loader:
        support_fp, support_labels, query_fp, query_labels = batch
        # Move entire batch to GPU in 4 transfers instead of 4*batch_size
        s_fp  = support_fp.to(device)    # (B, n_support, 2048)
        s_lbl = support_labels.to(device)
        q_fp  = query_fp.to(device)
        q_lbl = query_labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, metrics = model.compute_loss_batched(s_fp, s_lbl, q_fp, q_lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rmse += metrics["rmse"]
        n_batches  += 1

    return {
        "loss": total_loss / n_batches,
        "rmse": total_rmse / n_batches,
    }


def validate_regression(model, loader, device):
    model.eval()
    total_rmse = 0.0
    total_mae  = 0.0
    n_batches  = 0
    use_amp    = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            support_fp, support_labels, query_fp, query_labels = batch
            s_fp  = support_fp.to(device)
            s_lbl = support_labels.to(device)
            q_fp  = query_fp.to(device)
            q_lbl = query_labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, metrics = model.compute_loss_batched(s_fp, s_lbl, q_fp, q_lbl)
            total_rmse += metrics["rmse"]
            total_mae  += metrics["mae"]
            n_batches  += 1

    return {
        "rmse": total_rmse / n_batches,
        "mae":  total_mae  / n_batches,
    }



def _encoder_config(encoder) -> dict:
    """Serialisable config dict for a given encoder — stored inside each checkpoint."""
    if isinstance(encoder, PNAGNNEncoder):
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


def _make_loaders(encoder, train_assays, val_assays,
                  n_episodes_train, n_episodes_val, n_support, n_query, shift_aware):
    """Return (train_loader, val_loader) using the right dataset class for the encoder."""
    is_gnn = isinstance(encoder, PNAGNNEncoder)
    DatasetCls = FSMolGraphEpisodeDataset if is_gnn else FSMolEpisodeDataset
    collate_fn = graph_episode_collate if is_gnn else None

    # ECFP fingerprints are compact (2048 float32 per molecule) so a larger pool
    # fits easily in RAM and gives more diversity per epoch.
    # GNN graphs are heavier (variable-size node/edge tensors), so keep pool smaller.
    train_pool_size = 750 if is_gnn else 2000
    # Val has only 40 assays total — load all of them regardless of encoder.
    val_pool_size   = len(val_assays)

    train_dataset = DatasetCls(
        train_assays, n_episodes_train, n_support, n_query,
        shift_aware=shift_aware, pool_size=train_pool_size,
    )
    val_dataset = DatasetCls(
        val_assays, n_episodes_val, n_support, n_query,
        shift_aware=False, pool_size=val_pool_size,
    )
    # num_workers=2: while GPU processes batch N, workers prepare batch N+1.
    # Workers are re-forked each epoch (persistent_workers=False default), so
    # they see the refreshed pool after each refresh_pool() call.
    # Val loader stays single-process — it's only 1 batch total, no benefit.
    train_loader = DataLoader(train_dataset, batch_size=32,  shuffle=False,
                              num_workers=2, pin_memory=not is_gnn,
                              collate_fn=collate_fn, worker_init_fn=_worker_init_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=200, shuffle=False,
                              num_workers=0, pin_memory=not is_gnn,
                              collate_fn=collate_fn)
    return train_loader, val_loader


def pretrain_regression(
    encoder,
    train_assays,
    val_assays,
    n_epochs: int = 50,
    n_support: int = 16,
    n_query: int = 16,
    n_episodes_train: int = 1000,
    n_episodes_val: int = 200,
    lr: float = 1e-3,
    save_path: str = "ptn_ecfp_regression_shift_aware.pt",
    shift_aware: bool = True,
    seed: int = 42,
):
    """
    Episodic training for PrototypicalNetworkRegression (kernel regression, MSE loss).
    Works with any encoder: ECFPEncoder (fingerprint tensors) or PNAGNNEncoder (graphs).
    Validates on RMSE (minimise). Early stopping patience = 25 epochs.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = PrototypicalNetworkRegression(encoder).to(device)

    # CHOSEN: Adam optimizer. Standard choice.
    # ALTERNATIVE: AdamW with weight decay for regularization:
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler: reduce LR when validation RMSE plateaus
    # ALTERNATIVE: CosineAnnealingLR for smoother decay:
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5
        # patience=20: val RMSE is noisy (only 38-40 assays in val pool, random episodes).
        # patience=10 was too aggressive — LR halved 7 times before epoch 100, starving
        # the optimizer of any useful update in the final 30 epochs.
        # min_lr=1e-5: prevent LR from decaying to near-zero before model has converged
    )

    train_loader, val_loader = _make_loaders(
        encoder, train_assays, val_assays,
        n_episodes_train, n_episodes_val, n_support, n_query, shift_aware,
    )

    best_val_rmse  = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 25  # val is noisy but model consistently peaks early

    encoder_cfg = _encoder_config(encoder)

    for epoch in range(1, n_epochs + 1):
        if epoch > 1:
            train_loader.dataset.refresh_pool()   # type: ignore[union-attr]

        train_metrics = train_epoch_regression(model, train_loader, optimizer, device)
        val_metrics   = validate_regression(model, val_loader, device)

        scheduler.step(val_metrics["rmse"])
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train RMSE: {train_metrics['rmse']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": best_val_rmse,
                "config": {
                    "model_type": "regression",
                    "n_support":  n_support,
                    "shift_aware": shift_aware,
                    **encoder_cfg,
                }
            }, save_path)
            print(f"  → Saved new best model (Val RMSE: {best_val_rmse:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs).")
                break

    print(f"\nPretraining complete. Best Val RMSE: {best_val_rmse:.4f}")
    return model


# =============================================================================
# CLASSIFICATION TRAINING — Part A
# =============================================================================

def train_epoch_classification(model, loader, optimizer, device):
    model.train()
    total_loss    = 0.0
    total_dauprc  = []
    n_batches     = 0
    use_amp       = device.type == "cuda"

    for batch in loader:
        support_fp, support_labels, query_fp, query_labels = batch
        s_fp  = support_fp.to(device)
        s_lbl = support_labels.to(device)
        q_fp  = query_fp.to(device)
        q_lbl = query_labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, metrics = model.compute_loss_batched(s_fp, s_lbl, q_fp, q_lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if not np.isnan(metrics["delta_auprc"]):
            total_dauprc.append(metrics["delta_auprc"])
        n_batches += 1

    avg_dauprc = float(np.mean(total_dauprc)) if total_dauprc else float("nan")
    return {"loss": total_loss / n_batches, "delta_auprc": avg_dauprc}


def validate_classification(model, loader, device):
    model.eval()
    total_bce   = 0.0
    all_dauprc  = []
    n_batches   = 0
    use_amp     = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            support_fp, support_labels, query_fp, query_labels = batch
            s_fp  = support_fp.to(device)
            s_lbl = support_labels.to(device)
            q_fp  = query_fp.to(device)
            q_lbl = query_labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, metrics = model.compute_loss_batched(s_fp, s_lbl, q_fp, q_lbl)
            total_bce += loss.item()
            if not np.isnan(metrics["delta_auprc"]):
                all_dauprc.append(metrics["delta_auprc"])
            n_batches += 1

    avg_dauprc = float(np.mean(all_dauprc)) if all_dauprc else float("nan")
    return {"bce": total_bce / n_batches, "delta_auprc": avg_dauprc}


def pretrain_classification(
    encoder,
    train_assays,
    val_assays,
    n_epochs: int = 100,
    n_support: int = 16,
    n_query: int = 16,
    n_episodes_train: int = 1000,
    n_episodes_val: int = 200,
    lr: float = 1e-3,
    save_path: str = "ptn_ecfp_classification_shift_aware.pt",
    shift_aware: bool = True,
    seed: int = 42,
):
    """
    Episodic training for PrototypicalNetworkClassification (true PN, BCE loss).
    Works with any encoder: ECFPEncoder (fingerprint tensors) or PNAGNNEncoder (graphs).
    Validates on ΔAUPRC (maximise). Early stopping patience = 25 epochs.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = PrototypicalNetworkClassification(encoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5
    )

    train_loader, val_loader = _make_loaders(
        encoder, train_assays, val_assays,
        n_episodes_train, n_episodes_val, n_support, n_query, shift_aware,
    )

    best_val_dauprc    = -float("inf")
    epochs_no_improve  = 0
    early_stop_patience = 25
    encoder_cfg = _encoder_config(encoder)

    for epoch in range(1, n_epochs + 1):
        if epoch > 1:
            train_loader.dataset.refresh_pool()   # type: ignore[union-attr]

        train_metrics = train_epoch_classification(model, train_loader, optimizer, device)
        val_metrics   = validate_classification(model, val_loader, device)

        scheduler.step(val_metrics["delta_auprc"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Train BCE: {train_metrics['loss']:.4f} | "
            f"Train ΔAUPRC: {train_metrics['delta_auprc']:+.4f} | "
            f"Val ΔAUPRC: {val_metrics['delta_auprc']:+.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_metrics["delta_auprc"] > best_val_dauprc:
            best_val_dauprc   = val_metrics["delta_auprc"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_delta_auprc": best_val_dauprc,
                "config": {
                    "model_type":  "classification",
                    "n_support":   n_support,
                    "shift_aware": shift_aware,
                    **encoder_cfg,
                    "n_support":     n_support,
                    "shift_aware":   shift_aware,
                },
            }, save_path)
            print(f"  → Saved new best model (Val ΔAUPRC: {best_val_dauprc:+.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs).")
                break

    print(f"\nPretraining complete. Best Val ΔAUPRC: {best_val_dauprc:+.4f}")
    return model