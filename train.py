"""
Training Loop: FS-Mol Pretraining
==================================
Episodic training on FS-Mol assays.
Each step: sample episode → forward pass → MSE loss → backprop → update encoder.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import cast
from model import PrototypicalNetworkRegression
from data import FSMolEpisodeDataset


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches  = 0

    for batch in loader:
        support_fp, support_labels, query_fp, query_labels = batch
        # Move entire batch to GPU in 4 transfers instead of 4*batch_size
        s_fp  = support_fp.to(device)    # (B, n_support, 2048)
        s_lbl = support_labels.to(device)
        q_fp  = query_fp.to(device)
        q_lbl = query_labels.to(device)

        optimizer.zero_grad()
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


def validate(model, loader, device):
    model.eval()
    total_rmse = 0.0
    total_mae  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            support_fp, support_labels, query_fp, query_labels = batch
            s_fp  = support_fp.to(device)
            s_lbl = support_labels.to(device)
            q_fp  = query_fp.to(device)
            q_lbl = query_labels.to(device)

            _, metrics = model.compute_loss_batched(s_fp, s_lbl, q_fp, q_lbl)
            total_rmse += metrics["rmse"]
            total_mae  += metrics["mae"]
            n_batches  += 1

    return {
        "rmse": total_rmse / n_batches,
        "mae":  total_mae  / n_batches,
    }


def collect_val_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, targets) arrays from one pass over the validation loader.
    Used to snapshot the best epoch's predictions for offline analysis."""
    model.eval()
    all_preds:   list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            support_fp, support_labels, query_fp, query_labels = batch
            s_fp  = support_fp.to(device)
            s_lbl = support_labels.to(device)
            q_fp  = query_fp.to(device)
            q_lbl = query_labels.to(device)

            preds = model.forward_batched(s_fp, s_lbl, q_fp)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(q_lbl.cpu().numpy().flatten())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def pretrain(
    train_assays,
    val_assays,
    n_epochs: int = 50,
    n_support: int = 16,
    n_query: int = 16,
    n_episodes_train: int = 1000,
    n_episodes_val: int = 200,
    lr: float = 1e-3,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    save_path: str = "pretrained_model.pt",
    shift_aware: bool = True,   # CHOSEN: shift-aware episodes
    save_val_preds: bool = True,  # save best-epoch val predictions alongside checkpoint
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = PrototypicalNetworkRegression(
        input_dim=2048,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim
    ).to(device)

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

    train_dataset = FSMolEpisodeDataset(
        train_assays, n_episodes_train, n_support, n_query,
        shift_aware=shift_aware, pool_size=750,
    )
    val_dataset = FSMolEpisodeDataset(
        val_assays, n_episodes_val, n_support, n_query,
        shift_aware=False, pool_size=100,
    )

    # num_workers=0: pool is in RAM so no I/O bottleneck — single-process is fine
    # and avoids fork overhead + pool duplication across worker processes.
    # pin_memory=True: faster host→device transfer for the tensors that do go to GPU.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=200, shuffle=False,
                              num_workers=0, pin_memory=True)

    best_val_rmse  = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 40  # 2× LR patience; val is noisy (38-assay pool)

    for epoch in range(1, n_epochs + 1):
        # Rotate in a fresh set of assays each epoch so all ~16k tasks are
        # seen over the course of training (pool_size=1000 → ~6% per epoch).
        if epoch > 1:
            cast(FSMolEpisodeDataset, train_loader.dataset).refresh_pool()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

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

        # Save best model based on validation RMSE
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": best_val_rmse,
                "config": {
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "n_support": n_support,
                    "shift_aware": shift_aware
                }
            }, save_path)
            print(f"  → Saved new best model (Val RMSE: {best_val_rmse:.4f})")

            if save_val_preds:
                preds_np, targets_np = collect_val_predictions(model, val_loader, device)
                preds_path = os.path.splitext(save_path)[0] + "_best_val_preds.npz"
                np.savez(preds_path, preds=preds_np, targets=targets_np, epoch=epoch)
                print(f"  → Saved val predictions ({len(preds_np):,} query points) → {preds_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs).")
                break

    print(f"\nPretraining complete. Best Val RMSE: {best_val_rmse:.4f}")
    return model