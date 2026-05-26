"""
Prototypical Network for Molecular Property Prediction
=======================================================
Encoders (choose one, pass to the head at construction time):
  ECFPEncoder     — ECFP4 2048-bit fingerprint + 3-layer MLP
  PNAGNNEncoder   — Principal Neighbourhood Aggregation GNN (FS-Mol paper)

Heads (encoder-agnostic, work with any encoder above):
  PrototypicalNetworkRegression    — kernel regression, MSE loss     (Part 0)
  PrototypicalNetworkClassification — true PN binary classification,
                                      BCE loss                        (Part A)

Distance function: squared Euclidean (≡ cosine on L2-normalised unit sphere).

To add a new encoder: implement forward(x) → (n, emb_dim) and pass it to
either head class. No changes to the head code are needed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


# =============================================================================
# ENCODER A: ECFP + MLP
# =============================================================================

class ECFPEncoder(nn.Module):
    """
    Maps a 2048-bit ECFP4 fingerprint to a fixed-size embedding via 3-layer MLP.
    Fast and simple; no graph dataloader required.

    Input:  (n, 2048) float tensor
    Output: (n, embedding_dim) L2-normalised float tensor
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, embedding_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(x), p=2, dim=-1)


# =============================================================================
# ENCODER B: PNA GNN  (FS-Mol paper, Schwartz et al. 2022)
# =============================================================================

class PNAGNNEncoder(nn.Module):
    """
    Principal Neighbourhood Aggregation GNN encoder.
    Matches the architecture used in the FS-Mol prototypical network paper.

    Architecture:
      - Linear node embedding: node_feat_dim → hidden_channels
      - 6 × PNAConv layers with BatchNorm + ReLU
        aggregators : mean, min, max, std
        scalers     : identity, amplification, attenuation
      - Global mean pooling → (n_graphs, hidden_channels)
      - 2-layer MLP projection → embedding_dim
      - L2 normalisation

    Input:  PyTorch Geometric Batch (x, edge_index, edge_attr, batch)
    Output: (n_graphs, embedding_dim) L2-normalised float tensor

    The `deg` tensor (degree histogram over the training set) is required for
    the amplification/attenuation scalers. Compute it once via
    featurize.compute_degree_histogram() and store it in the checkpoint.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        embedding_dim: int = 256,
        deg: torch.Tensor | None = None,
    ):
        super().__init__()
        from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool  # type: ignore

        self.global_mean_pool = global_mean_pool
        self.deg = deg  # stored for checkpoint serialisation via _encoder_config

        aggregators = ["mean", "min", "max", "std"]
        scalers     = ["identity", "amplification", "attenuation"]

        self.node_emb = nn.Linear(node_feat_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(PNAConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=edge_feat_dim,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            ))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, embedding_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch with fields x, edge_index, edge_attr, batch
        Returns:
            embeddings: (n_graphs, embedding_dim) L2-normalised
        """
        x = self.node_emb(batch.x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, batch.edge_index, batch.edge_attr)
            x = bn(x)
            x = F.relu(x)
        x = self.global_mean_pool(x, batch.batch)   # (n_graphs, hidden_channels)
        x = self.output_proj(x)
        return F.normalize(x, p=2, dim=-1)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _encode_many(encoder: nn.Module, inputs, B: int, n: int) -> torch.Tensor:
    """
    Encode B*n molecules and reshape to (B, n, emb_dim).
    Works for both encoder types:
      - ECFPEncoder:   inputs is Tensor(B, n, 2048)  → reshape → encode → reshape
      - PNAGNNEncoder: inputs is PyG Batch(B*n graphs) → encode  → reshape
    """
    if isinstance(inputs, torch.Tensor):
        D = inputs.shape[-1]
        return encoder(inputs.reshape(B * n, D)).reshape(B, n, -1)
    else:
        return encoder(inputs).reshape(B, n, -1)


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
    Kernel regression in learned embedding space (Nadaraya-Watson).
    Encoder-agnostic: pass an ECFPEncoder or PNAGNNEncoder at construction.

    pred(x_q) = Σ_i softmax(-d(f(x_q), f(x_i)) / τ) × y_i
    Loss: MSE.  τ is a learnable scalar (starts at 1.0).
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.log_tau = nn.Parameter(torch.zeros(1))  # tau = exp(log_tau)

    def forward(self, support_input, support_labels: torch.Tensor, query_input) -> torch.Tensor:
        """
        Args:
            support_input:  Tensor(n_sup, D) for ECFP  OR  PyG Batch(n_sup graphs) for GNN
            support_labels: Tensor(n_sup,)
            query_input:    Tensor(n_qry, D) for ECFP  OR  PyG Batch(n_qry graphs) for GNN
        Returns:
            predictions: Tensor(n_qry,)
        """
        sup_emb = self.encoder(support_input)   # (n_sup, emb_dim)
        qry_emb = self.encoder(query_input)     # (n_qry, emb_dim)
        distances = euclidean_distance(qry_emb, sup_emb)   # (n_qry, n_sup)
        tau = torch.exp(self.log_tau)
        weights = F.softmax(-distances / tau, dim=1)       # (n_qry, n_sup)
        return torch.mv(weights, support_labels)            # (n_qry,)

    def forward_batched(self, support_input, support_labels: torch.Tensor, query_input) -> torch.Tensor:
        """
        Args:
            support_input:  Tensor(B, n_sup, D)  OR  PyG Batch(B*n_sup graphs)
            support_labels: Tensor(B, n_sup)
            query_input:    Tensor(B, n_qry, D)  OR  PyG Batch(B*n_qry graphs)
        Returns:
            predictions: Tensor(B, n_qry)
        """
        B, n_sup = support_labels.shape
        n_qry = query_input.shape[1] if isinstance(query_input, torch.Tensor) \
                else query_input.num_graphs // B

        sup_emb = _encode_many(self.encoder, support_input, B, n_sup)   # (B, n_sup, emb_dim)
        qry_emb = _encode_many(self.encoder, query_input,   B, n_qry)   # (B, n_qry, emb_dim)

        sup_sq = (sup_emb ** 2).sum(dim=-1, keepdim=True)           # (B, n_sup, 1)
        qry_sq = (qry_emb ** 2).sum(dim=-1, keepdim=True)           # (B, n_qry, 1)
        dot    = torch.bmm(qry_emb, sup_emb.transpose(1, 2))        # (B, n_qry, n_sup)
        distances = (qry_sq + sup_sq.transpose(1, 2) - 2 * dot).clamp(min=0)

        tau     = torch.exp(self.log_tau)
        weights = F.softmax(-distances / tau, dim=-1)                # (B, n_qry, n_sup)
        return torch.bmm(weights, support_labels.unsqueeze(-1)).squeeze(-1)

    def compute_loss_batched(self, support_input, support_labels, query_input, query_labels):
        predictions = self.forward_batched(support_input, support_labels, query_input)
        loss = F.mse_loss(predictions, query_labels)
        with torch.no_grad():
            rmse = torch.sqrt(loss).item()
            mae  = F.l1_loss(predictions, query_labels).item()
        return loss, {"rmse": rmse, "mae": mae}

    def compute_loss(self, support_input, support_labels, query_input, query_labels):
        predictions = self.forward(support_input, support_labels, query_input)
        loss = F.mse_loss(predictions, query_labels)
        with torch.no_grad():
            rmse = torch.sqrt(loss).item()
            mae = F.l1_loss(predictions, query_labels).item()

        return loss, {"rmse": rmse, "mae": mae}


# =============================================================================
# PROTOTYPICAL NETWORK (BINARY CLASSIFICATION)  — Part A
# =============================================================================

class PrototypicalNetworkClassification(nn.Module):
    """
    True Prototypical Network for active/inactive binary classification.
    Matches the FS-Mol paper (Schwartz et al., 2022) evaluation protocol.

    Per episode:
      1. Binarise support labels: active = label > median(support_labels)
      2. proto_active   = mean(encoder(x_i) for active support molecules)
         proto_inactive = mean(encoder(x_i) for inactive support molecules)
      3. logits_q = [-d(f(x_q), proto_active), -d(f(x_q), proto_inactive)]
      4. P(active | x_q) = softmax(logits_q)[0]
      5. Loss: BCE(P(active), binary_query_labels)

    Median threshold is computed from support only and applied to both support
    and query — same as FS-Mol paper. Guarantees balanced support split.

    Primary metric: ΔAUPRC (same as regression model evaluation).
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_input, support_labels: torch.Tensor, query_input) -> torch.Tensor:
        """
        Args:
            support_input:  Tensor(n_sup, D) or PyG Batch(n_sup graphs)
            support_labels: Tensor(n_sup,) — continuous, binarised internally via support median
            query_input:    Tensor(n_qry, D) or PyG Batch(n_qry graphs)
        Returns:
            p_active: Tensor(n_qry,) — P(active) for each query molecule
        """
        sup_emb = self.encoder(support_input)
        qry_emb = self.encoder(query_input)

        threshold   = support_labels.median()
        active_mask = support_labels > threshold

        if not active_mask.any() or not (~active_mask).any():
            return torch.full((qry_emb.shape[0],), 0.5, device=qry_emb.device)

        proto_active   = sup_emb[active_mask].mean(dim=0)
        proto_inactive = sup_emb[~active_mask].mean(dim=0)
        protos   = torch.stack([proto_active, proto_inactive], dim=0)
        dists    = euclidean_distance(qry_emb, protos)   # (n_qry, 2)
        p_active = F.softmax(-dists, dim=1)[:, 0]
        return p_active

    def forward_batched(
        self, support_input, support_labels: torch.Tensor, query_input,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            support_input:  Tensor(B, n_sup, D) or PyG Batch(B*n_sup graphs)
            support_labels: Tensor(B, n_sup)
            query_input:    Tensor(B, n_qry, D) or PyG Batch(B*n_qry graphs)
        Returns:
            p_active:   Tensor(B, n_qry)
            valid_mask: BoolTensor(B,) — False for degenerate (all-one-class) episodes
        """
        B, n_sup = support_labels.shape
        n_qry = query_input.shape[1] if isinstance(query_input, torch.Tensor) \
                else query_input.num_graphs // B

        sup_emb = _encode_many(self.encoder, support_input, B, n_sup)
        qry_emb = _encode_many(self.encoder, query_input,   B, n_qry)

        # Binarise per episode using support median
        thresholds  = support_labels.median(dim=1).values  # (B,)
        active_mask = support_labels > thresholds.unsqueeze(1)  # (B, n_support) bool

        act_count   = active_mask.float().sum(dim=1)    # (B,)
        inact_count = (~active_mask).float().sum(dim=1) # (B,)
        valid_mask  = (act_count > 0) & (inact_count > 0)  # (B,)

        # Compute class prototypes (clamped counts to avoid 0-division; invalid episodes
        # produce a meaningless prototype that we mask out in the loss)
        act_w   = active_mask.float() / act_count.clamp(min=1).unsqueeze(1)   # (B, n_sup)
        inact_w = (~active_mask).float() / inact_count.clamp(min=1).unsqueeze(1)

        proto_active   = torch.bmm(act_w.unsqueeze(1), sup_emb).squeeze(1)    # (B, emb_dim)
        proto_inactive = torch.bmm(inact_w.unsqueeze(1), sup_emb).squeeze(1)  # (B, emb_dim)

        # Distances from each query to each prototype
        protos   = torch.stack([proto_active, proto_inactive], dim=1)          # (B, 2, emb_dim)
        qry_sq   = (qry_emb ** 2).sum(dim=-1, keepdim=True)                   # (B, n_qry, 1)
        proto_sq = (protos ** 2).sum(dim=-1, keepdim=True)                    # (B, 2, 1)
        dot      = torch.bmm(qry_emb, protos.transpose(1, 2))                 # (B, n_qry, 2)
        dists    = (qry_sq + proto_sq.transpose(1, 2) - 2 * dot).clamp(min=0) # (B, n_qry, 2)

        p_active = F.softmax(-dists, dim=-1)[:, :, 0]  # (B, n_qry)
        return p_active, valid_mask

    # ------------------------------------------------------------------
    # Loss (used by training loop)
    # ------------------------------------------------------------------

    def compute_loss_batched(
        self,
        support_input,
        support_labels: torch.Tensor,
        query_input,
        query_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            support_fingerprints: (B, n_support, input_dim)
            support_labels:       (B, n_support)  — continuous
            query_fingerprints:   (B, n_query, input_dim)
            query_labels:         (B, n_query)    — continuous; binarised here

        Returns:
            loss:    scalar BCE over valid episodes
            metrics: {"delta_auprc": float}
        """
        p_active, valid_mask = self.forward_batched(
            support_input, support_labels, query_input
        )

        # Binarise query using same support-median threshold
        thresholds   = support_labels.median(dim=1).values  # (B,)
        binary_query = (query_labels > thresholds.unsqueeze(1)).float()  # (B, n_query)

        if not valid_mask.any():
            return torch.tensor(0.0, device=p_active.device, requires_grad=True), {"delta_auprc": float("nan")}

        # BCE only over valid (non-degenerate) episodes
        p_valid = p_active[valid_mask]       # (V, n_query)
        b_valid = binary_query[valid_mask]   # (V, n_query)
        loss = F.binary_cross_entropy(p_valid, b_valid)

        with torch.no_grad():
            p_np = p_valid.detach().cpu().numpy()
            b_np = b_valid.detach().cpu().numpy().astype(int)
            dauprc_vals = []
            for i in range(p_np.shape[0]):
                s = b_np[i].sum()
                if 0 < s < len(b_np[i]):
                    try:
                        d = float(average_precision_score(b_np[i], p_np[i])) - float(b_np[i].mean())
                        dauprc_vals.append(d)
                    except Exception:
                        pass
            avg_dauprc = float(np.mean(dauprc_vals)) if dauprc_vals else float("nan")

        return loss, {"delta_auprc": avg_dauprc}