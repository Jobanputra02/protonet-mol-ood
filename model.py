"""
Prototypical Network for Molecular Property Prediction (Regression)
Design choices:
- Encoder: ECFP fingerprints + MLP (alternative: GNN or ChemBERTa)
- Distance: Euclidean (alternative: learned MLP)
- Prediction: kernel regression
- Temperature: learnable scalar (alternative: fixed temp)
- Loss: MSE (alternative: MAE, Huber loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularEncoder(nn.Module):
    """
    Maps a molecular fingerprint to an embedding vector
    ECFP fingerprint -> MLP -> embedding
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, embedding_dim: int = 256):
        """
        Args:
        - input_dim: Size of ECFP fingerprint
        - hidden_dim: Hidden layer size.
        - embedding_dim: Output embedding size
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # L2 normalization at output which projects embeddings onto unit hypersphere
        # so that Euclidean distance equivalent to cosine distance
        self.normalize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fingerprint tensor of shape (batch_size, input_dim)
        Returns:
            embeddings: shape (batch_size, embedding_dim)
        """
        embeddings = self.network(x)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    m = b.size(0)
    a_sq = (a ** 2).sum(dim=1, keepdim=True).expand(n, m)
    b_sq = (b ** 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = a_sq + b_sq - 2 * torch.mm(a, b.t())
    return dist.clamp(min=0)


# Prototypical Network with Regression
class PrototypicalNetworkRegression(nn.Module):
    """
    1. Encode all support molecules into embedding space via f(x)
    2. Encode query molecule into same space
    3. Compute distances from query to each support embedding
    4. Convert distances to weights via softmax (closer = higher weight)
    5. Predict query label as weighted average of support labels
    This is non-parametric kernel regression in learned embedding space. The network only learns the embedding function f
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, embedding_dim: int = 256):
        super().__init__()
        self.encoder = MolecularEncoder(input_dim, hidden_dim, embedding_dim)

        # learnable temp scaler; contrls sharpness of attention weights over support set
        # initialized to 1 (log_tau=0) and learned during training via backprop
        # large tau: weights become uniform, prediction ~ mean of support labels
        # small tau: weights concentrate on nearest support point
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, support_fingerprints: torch.Tensor, support_labels: torch.Tensor, query_fingerprints: torch.Tensor) -> torch.Tensor:
        """
        given support set, predict query labels.
        Returns:
            query_predictions: (n_query,) predicted activity values
        """
        
        # encode support and query molecules
        support_emb = self.encoder(support_fingerprints)  # (n_support, emb_dim)
        query_emb = self.encoder(query_fingerprints)      # (n_query, emb_dim)

        # compute pairwise distances: query vs support
        distances = euclidean_distance(query_emb, support_emb)

        # convert distances to weights via softmax with temperature
        tau = torch.exp(self.log_tau)        
        # -ve distance: closer support points get higher weight
        weights = F.softmax(-distances / tau, dim=1)  # (n_query, n_support)

        # weighted average of support labels
        predictions = torch.mv(weights, support_labels)  # (n_query,)

        return predictions

    def compute_loss(self, support_fingerprints: torch.Tensor, support_labels: torch.Tensor, query_fingerprints: torch.Tensor, query_labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Full episode forward pass + loss computation
        Returns:
            loss: MSE
            metrics: dict with rmse, mae (for logging)
        """
        predictions = self.forward(support_fingerprints, support_labels, query_fingerprints)
        loss = F.mse_loss(predictions, query_labels)

        with torch.no_grad():
            rmse = torch.sqrt(loss).item()
            mae = F.l1_loss(predictions, query_labels).item()

        return loss, {"rmse": rmse, "mae": mae}
