"""
Evaluation on DrugOOD OOD Benchmark
zero-shot classification : 
- Load pretrained model, freeze weights (no gradient updates)
- At test time, provide a small context set from the test distribution
- Run forward pass: predict test labels using context as support set
- Compute RMSE, MAE, and Spearman correlation
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from model import PrototypicalNetworkRegression
from data import DrugOODEvalDataset


def load_and_evaluate(checkpoint_path: str, eval_datasets: list[DrugOODEvalDataset]) -> dict:
    """
    Loads pretrained model from checkpoint and evaluate on DrugOOD
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = PrototypicalNetworkRegression(
        input_dim=2048,
        hidden_dim=config["hidden_dim"],
        embedding_dim=config["embedding_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # print(f"loaded model from epoch {checkpoint['epoch']} "
    #       f"(Val RMSE: {checkpoint['val_rmse']:.4f})")
    model.eval()
    results = {}

    with torch.no_grad():
        for eval_dataset in eval_datasets:
            context_fp, context_labels, test_fp, test_labels = eval_dataset.get_episode()

            context_fp = context_fp.to(device)
            context_labels = context_labels.to(device)
            test_fp = test_fp.to(device)

            # forward pass
            predictions = model.forward(context_fp, context_labels, test_fp)
            preds_np = predictions.cpu().numpy()
            targets_np = test_labels.numpy()

            rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))
            mae = np.mean(np.abs(preds_np - targets_np))

            if len(preds_np) < 3:
                spearman = float("nan")
            else:
                spearman = float(stats.spearmanr(preds_np, targets_np).statistic)  # type: ignore[union-attr]

            results[eval_dataset.split_type] = {
                "rmse": rmse,
                "mae": mae,
                "spearman": spearman,
                "n_context": len(context_labels),
                "n_test": len(test_labels)
            }

            # print(
            #     f"Split: {eval_dataset.split_type:10s} | "
            #     f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | Spearman: {spearman:.4f} | "
            #     f"Context: {len(context_labels)}, Test: {len(test_labels)}"
            # )
    return results