"""
Prototypical Network for Molecular Property Prediction (Regression)
====================================================================
Design choices made (with alternatives commented):
- Encoder: ECFP fingerprints + MLP  (alternative: GNN like GIN/MPNN, or ChemBERTa)
- Distance: Euclidean               (alternative: learned MLP metric)
- Prediction: kernel regression     (alternative: binning/discretization into classes)
- Temperature: learnable scalar     (alternative: fixed temperature = 1.0)
- Loss: MSE                         (alternative: MAE, Huber loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MOLECULAR ENCODER
# =============================================================================

class MolecularEncoder(nn.Module):
    """
    Maps a molecular fingerprint to an embedding vector.

    CHOSEN: ECFP fingerprint (2048-bit) -> MLP -> embedding
    Reason: Simple, fast, no graph dataloader needed, works well in practice.

    ALTERNATIVE (GNN-based encoder):
    # from torch_geometric.nn import GINConv, global_mean_pool
    # class GNNEncoder(nn.Module):
    #     def __init__(self, embedding_dim):
    #         super().__init__()
    #         self.conv1 = GINConv(nn.Linear(atom_feat_dim, 64))
    #         self.conv2 = GINConv(nn.Linear(64, embedding_dim))
    #     def forward(self, data):
    #         x = F.relu(self.conv1(data.x, data.edge_index))
    #         x = self.conv2(x, data.edge_index)
    #         return global_mean_pool(x, data.batch)

    ALTERNATIVE (ChemBERTa-based encoder):
    # from transformers import AutoModel, AutoTokenizer
    # model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    # Very expressive but slow and heavyweight for a 1-week project.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, embedding_dim: int = 256):
        """
        Args:
            input_dim:     Size of ECFP fingerprint. Default 2048 (ECFP4, nBits=2048).
            hidden_dim:    Hidden layer size.
            embedding_dim: Output embedding size. This is the latent space dimension.
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

        # L2 normalization at output: projects embeddings onto unit hypersphere.
        # This makes Euclidean distance equivalent to cosine distance and
        # stabilizes training. Standard practice in metric learning.
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


# =============================================================================
# DISTANCE FUNCTION
# =============================================================================

def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise squared Euclidean distances between two sets of vectors.

    CHOSEN: Euclidean distance — no extra parameters, standard in ProtoNets.

    ALTERNATIVE (learned distance via MLP):
    # class LearnedDistance(nn.Module):
    #     def __init__(self, embedding_dim):
    #         super().__init__()
    #         self.mlp = nn.Sequential(
    #             nn.Linear(embedding_dim * 2, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 1)
    #         )
    #     def forward(self, a, b):
    #         # a: (n_query, embedding_dim)
    #         # b: (n_support, embedding_dim)
    #         n_q, n_s = a.size(0), b.size(0)
    #         a_exp = a.unsqueeze(1).expand(n_q, n_s, -1)
    #         b_exp = b.unsqueeze(0).expand(n_q, n_s, -1)
    #         pairs = torch.cat([a_exp, b_exp], dim=-1)
    #         return self.mlp(pairs).squeeze(-1)  # (n_query, n_support)

    Args:
        a: (n, d) tensor
        b: (m, d) tensor
    Returns:
        distances: (n, m) tensor of squared distances
    """
    # Efficient computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    n = a.size(0)
    m = b.size(0)
    a_sq = (a ** 2).sum(dim=1, keepdim=True).expand(n, m)   # (n, m)
    b_sq = (b ** 2).sum(dim=1, keepdim=True).expand(m, n).t()  # (n, m)
    dist = a_sq + b_sq - 2 * torch.mm(a, b.t())
    # Clamp to avoid negative values from floating point errors
    return dist.clamp(min=0)


# =============================================================================
# PROTOTYPICAL NETWORK (REGRESSION)
# =============================================================================

class PrototypicalNetworkRegression(nn.Module):
    """
    Prototypical Network adapted for regression via kernel regression.

    HOW IT WORKS:
    1. Encode all support molecules into embedding space via f(x)
    2. Encode query molecule into same space
    3. Compute distances from query to each support embedding
    4. Convert distances to weights via softmax (closer = higher weight)
    5. Predict query label as weighted average of support labels

    This is non-parametric kernel regression in learned embedding space.
    The network only learns the embedding function f; prediction is pure math.

    WHY NOT CLASSIFICATION:
    DrugOOD targets are continuous (IC50, binding affinity).
    Binning them into classes is possible but lossy and arbitrary.
    Kernel regression is principled and preserves label information.

    ALTERNATIVE (binning/classification approach):
    # Discretize labels into N bins, treat as classification.
    # Use standard cross-entropy loss and nearest-prototype classification.
    # Simpler but loses precision and requires choosing bin boundaries.
    # def predict_classification(self, support_embeddings, support_labels, query_embeddings, n_bins=10):
    #     bins = torch.linspace(support_labels.min(), support_labels.max(), n_bins)
    #     binned_labels = torch.bucketize(support_labels, bins)
    #     prototypes = []
    #     for b in range(n_bins):
    #         mask = binned_labels == b
    #         if mask.any():
    #             prototypes.append(support_embeddings[mask].mean(0))
    #     prototypes = torch.stack(prototypes)
    #     dists = euclidean_distance(query_embeddings, prototypes)
    #     return bins[dists.argmin(dim=1)]  # predicted bin center
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, embedding_dim: int = 256):
        super().__init__()
        self.encoder = MolecularEncoder(input_dim, hidden_dim, embedding_dim)

        # CHOSEN: Learnable temperature scalar.
        # Controls sharpness of attention weights over support set.
        # - High temperature (large tau): weights become uniform, prediction ~ mean of support labels
        # - Low temperature (small tau): weights concentrate on nearest support point
        # Initialized to 1.0, learned during training via backprop.
        #
        # ALTERNATIVE (fixed temperature):
        # self.log_tau = None  # and remove from forward(); use raw distances
        self.log_tau = nn.Parameter(torch.zeros(1))  # tau = exp(log_tau), starts at 1.0

    def forward(
        self,
        support_fingerprints: torch.Tensor,
        support_labels: torch.Tensor,
        query_fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """
        Core forward pass: given support set, predict query labels.

        Args:
            support_fingerprints: (n_support, input_dim)  — context molecules
            support_labels:       (n_support,)             — their known activity values
            query_fingerprints:   (n_query, input_dim)    — molecules to predict

        Returns:
            query_predictions:    (n_query,)               — predicted activity values
        """
        # Step 1: Encode support and query molecules
        support_emb = self.encoder(support_fingerprints)  # (n_support, emb_dim)
        query_emb = self.encoder(query_fingerprints)      # (n_query, emb_dim)

        # Step 2: Compute pairwise distances: query vs support
        distances = euclidean_distance(query_emb, support_emb)  # (n_query, n_support)

        # Step 3: Convert distances to weights via softmax with temperature
        # tau = exp(log_tau) ensures tau > 0 always
        tau = torch.exp(self.log_tau)
        # Negative distance: closer support points get higher weight
        weights = F.softmax(-distances / tau, dim=1)  # (n_query, n_support)

        # ALTERNATIVE (fixed temperature = 1.0, no learnable parameter):
        # weights = F.softmax(-distances, dim=1)

        # Step 4: Weighted average of support labels
        # weights: (n_query, n_support), support_labels: (n_support,)
        predictions = torch.mv(weights, support_labels)  # (n_query,)

        return predictions

    def forward_batched(
        self,
        support_fingerprints: torch.Tensor,
        support_labels: torch.Tensor,
        query_fingerprints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched forward pass: processes all episodes in a batch in one GPU call.

        Args:
            support_fingerprints: (B, n_support, input_dim)
            support_labels:       (B, n_support)
            query_fingerprints:   (B, n_query, input_dim)

        Returns:
            predictions: (B, n_query)
        """
        B, n_sup, D = support_fingerprints.shape
        n_qry = query_fingerprints.shape[1]

        # Encode all B*n_sup support and B*n_qry query molecules in one encoder call
        sup_emb = self.encoder(support_fingerprints.reshape(B * n_sup, D)).reshape(B, n_sup, -1)
        qry_emb = self.encoder(query_fingerprints.reshape(B * n_qry, D)).reshape(B, n_qry, -1)

        # Batched pairwise squared Euclidean distances: (B, n_qry, n_sup)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b  — avoids materializing (B,n_qry,n_sup,D)
        sup_sq = (sup_emb ** 2).sum(dim=-1, keepdim=True)           # (B, n_sup, 1)
        qry_sq = (qry_emb ** 2).sum(dim=-1, keepdim=True)           # (B, n_qry, 1)
        dot    = torch.bmm(qry_emb, sup_emb.transpose(1, 2))        # (B, n_qry, n_sup)
        distances = (qry_sq + sup_sq.transpose(1, 2) - 2 * dot).clamp(min=0)

        tau     = torch.exp(self.log_tau)
        weights = F.softmax(-distances / tau, dim=-1)                # (B, n_qry, n_sup)

        # Weighted average: bmm of (B, n_qry, n_sup) × (B, n_sup, 1) → (B, n_qry)
        predictions = torch.bmm(weights, support_labels.unsqueeze(-1)).squeeze(-1)
        return predictions

    def compute_loss_batched(
        self,
        support_fingerprints: torch.Tensor,
        support_labels: torch.Tensor,
        query_fingerprints: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Batched episode loss — replaces the Python for-loop in train_epoch.

        Args:
            support_fingerprints: (B, n_support, input_dim)
            support_labels:       (B, n_support)
            query_fingerprints:   (B, n_query, input_dim)
            query_labels:         (B, n_query)

        Returns:
            loss: scalar MSE averaged over all B*n_query predictions
            metrics: dict with rmse, mae
        """
        predictions = self.forward_batched(support_fingerprints, support_labels, query_fingerprints)
        loss = F.mse_loss(predictions, query_labels)
        with torch.no_grad():
            rmse = torch.sqrt(loss).item()
            mae  = F.l1_loss(predictions, query_labels).item()
        return loss, {"rmse": rmse, "mae": mae}

    def compute_loss(
        self,
        support_fingerprints: torch.Tensor,
        support_labels: torch.Tensor,
        query_fingerprints: torch.Tensor,
        query_labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Full episode forward pass + loss computation.

        Returns:
            loss: scalar tensor (MSE)
            metrics: dict with rmse, mae for logging
        """
        predictions = self.forward(support_fingerprints, support_labels, query_fingerprints)

        # CHOSEN: MSE loss — standard for regression, smooth gradients.
        # ALTERNATIVE (MAE): loss = F.l1_loss(predictions, query_labels)
        # ALTERNATIVE (Huber): loss = F.huber_loss(predictions, query_labels, delta=1.0)
        loss = F.mse_loss(predictions, query_labels)

        with torch.no_grad():
            rmse = torch.sqrt(loss).item()
            mae = F.l1_loss(predictions, query_labels).item()

        return loss, {"rmse": rmse, "mae": mae}