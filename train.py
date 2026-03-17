"""
Episodic Training Loop: FS-Mol Pretraining
Each step: sample episode -> forward pass -> MSE loss -> backprop -> update encoder
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import PrototypicalNetworkRegression
from data import FSMolEpisodeDataset

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
    shift_aware: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Training on: {device}")

    model = PrototypicalNetworkRegression(input_dim=2048, hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # adam optimizer; alternative: AdamW with weight decay for regularization

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_dataset = FSMolEpisodeDataset(train_assays, n_episodes_train, n_support, n_query, shift_aware=shift_aware)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    val_dataset = FSMolEpisodeDataset(val_assays, n_episodes_val, n_support, n_query, shift_aware=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    best_val_rmse = float("inf")

    for epoch in range(1, n_epochs + 1):

        # training epochs--
        model.train()
        total_loss = 0.0
        total_rmse = 0.0
        n_episodes = 0

        for batch in train_loader:
            support_fp, support_labels, query_fp, query_labels = batch

            # for batch_size=1
            support_fp = support_fp.squeeze(0).to(device)
            support_labels = support_labels.squeeze(0).to(device)
            query_fp = query_fp.squeeze(0).to(device)
            query_labels = query_labels.squeeze(0).to(device)

            optimizer.zero_grad()
            loss, metrics = model.compute_loss(support_fp, support_labels, query_fp, query_labels)
            loss.backward()

            # clip gradient (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_rmse += metrics["rmse"]
            n_episodes += 1

        train_metrics = {"loss": total_loss / n_episodes, "rmse": total_rmse / n_episodes}
        # -- training epochs

        # validate model --
        model.eval()
        total_rmse = 0.0
        total_mae = 0.0
        n_episodes = 0

        with torch.no_grad():
            for batch in val_loader:
                support_fp, support_labels, query_fp, query_labels = batch
                support_fp = support_fp.squeeze(0).to(device)
                support_labels = support_labels.squeeze(0).to(device)
                query_fp = query_fp.squeeze(0).to(device)
                query_labels = query_labels.squeeze(0).to(device)

                _, metrics = model.compute_loss(support_fp, support_labels, query_fp, query_labels)
                total_rmse += metrics["rmse"]
                total_mae += metrics["mae"]
                n_episodes += 1

        val_metrics = {"rmse": total_rmse / n_episodes,"mae": total_mae / n_episodes}
        # -- validate model

        scheduler.step(val_metrics["rmse"])
        current_lr = optimizer.param_groups[0]['lr']

        # print(
        #     f"Epoch {epoch:3d}/{n_epochs} | "
        #     f"Train Loss: {train_metrics['loss']:.4f} | "
        #     f"Train RMSE: {train_metrics['rmse']:.4f} | "
        #     f"Val RMSE: {val_metrics['rmse']:.4f} | "
        #     f"Val MAE: {val_metrics['mae']:.4f} | "
        #     f"LR: {current_lr:.2e}"
        # )

        # save best model based on validation rmse
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": best_val_rmse,
                "config": {
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "n_support": n_support,
                    "shift_aware": shift_aware
                }
            }, save_path)
            # print(f" -> Saved new best model (Val RMSE: {best_val_rmse:.4f})")

    # print(f"\nPretraining complete. Best Val RMSE: {best_val_rmse:.4f}")
    return model