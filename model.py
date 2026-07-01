"""
Prototypical Network for Molecular Property Prediction
=======================================================
Encoders (choose one, pass to the head at construction time):
  ECFPEncoder     - ECFP4 2048-bit fingerprint + 3-layer MLP
  PNAGNNEncoder   - Principal Neighbourhood Aggregation GNN (FS-Mol paper)

Heads (encoder-agnostic, work with any encoder above):
  PrototypicalNetworkRegression    - kernel regression, MSE loss     (Part 0)
  PrototypicalNetworkClassification - true PN binary classification,
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

    CHOSEN: Euclidean distance - no extra parameters, standard in ProtoNets.

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


def _sample_cov(x: torch.Tensor) -> torch.Tensor:
    """Sample covariance of row vectors. x: (n, D) → (D, D) float32."""
    n = x.shape[0]
    if n <= 1:
        return torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=torch.float32)
    x32 = x.float()
    mu  = x32.mean(dim=0, keepdim=True)
    xc  = x32 - mu
    return (xc.T @ xc) / (n - 1)


def _mahalanobis_dists(
    qry_emb: torch.Tensor,
    protos: torch.Tensor,
    sup_emb: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """
    FS-Mol Mahalanobis distance from each query to [proto_active, proto_inactive].

    Shrinkage per class k - exact FS-Mol paper values:
      λ_k = min(n_k / (n_k + 1), 0.1)
      Σ_k = λ_k · cov(class_k) + (1 − λ_k) · cov(task) + 0.1 · I
      d(q, k) = (q − μ_k)ᵀ Σ_k⁻¹ (q − μ_k)

    Used at eval time only. Training uses Euclidean (see forward_batched).

    All covariance ops run in float32 for numerical stability even under AMP.

    Args:
        qry_emb:     (n_qry, D)
        protos:      (2, D)  - [proto_active, proto_inactive]
        sup_emb:     (n_sup, D)
        active_mask: (n_sup,) bool
    Returns:
        dists: (n_qry, 2) Mahalanobis distances to each prototype
    """
    D      = qry_emb.shape[-1]
    device = qry_emb.device
    I      = torch.eye(D, device=device, dtype=torch.float32)
    task_cov = _sample_cov(sup_emb)
    dists = []
    for cls_idx, mask in enumerate([active_mask, ~active_mask]):
        cls_embs = sup_emb[mask]
        n_k  = cls_embs.shape[0]
        lam  = min(n_k / (n_k + 1), 0.1)   # FS-Mol paper: cap at 0.1
        sigma = lam * _sample_cov(cls_embs) + (1.0 - lam) * task_cov + 0.1 * I
        diff  = (qry_emb - protos[cls_idx].unsqueeze(0)).float()   # (n_qry, D)
        try:
            sol = torch.linalg.solve(sigma, diff.T)   # (D, n_qry)
            d   = (diff * sol.T).sum(dim=-1).clamp(min=0)
        except Exception:
            d = (diff ** 2).sum(dim=-1)               # fallback: squared Euclidean
        dists.append(d)
    return torch.stack(dists, dim=-1)   # (n_qry, 2)


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

    def predict_from_embeddings(
        self, sup_emb: torch.Tensor, sup_labels: torch.Tensor, qry_emb: torch.Tensor
    ) -> torch.Tensor:
        """Single-episode prediction given pre-computed embeddings. Avoids re-encoding support."""
        distances = euclidean_distance(qry_emb, sup_emb)
        tau = torch.exp(self.log_tau)
        weights = F.softmax(-distances / tau, dim=1)
        return torch.mv(weights, sup_labels)

    def compute_loss(self, support_input, support_labels, query_input, query_labels):
        predictions = self.forward(support_input, support_labels, query_input)
        loss = F.mse_loss(predictions, query_labels)
        with torch.no_grad():
            rmse = torch.sqrt(loss).item()
            mae = F.l1_loss(predictions, query_labels).item()

        return loss, {"rmse": rmse, "mae": mae}


# =============================================================================
# PROTOTYPICAL NETWORK (BINARY CLASSIFICATION)  - Part A
# =============================================================================

class PrototypicalNetworkClassification(nn.Module):
    """
    True Prototypical Network for active/inactive binary classification.
    Matches the FS-Mol paper (Schwartz et al., 2022) evaluation protocol.

    Per episode:
      1. Binarise support labels: active = label > 0.5  (labels must be pre-binarised 0/1)
      2. proto_active   = mean(encoder(x_i) for active support molecules)
         proto_inactive = mean(encoder(x_i) for inactive support molecules)
      3. logits_q = [-d(f(x_q), proto_active), -d(f(x_q), proto_inactive)]
      4. P(active | x_q) = softmax(logits_q)[0]
      5. Loss: BCE(P(active), binary_query_labels)

    Median threshold is computed from support only and applied to both support
    and query - same as FS-Mol paper. Guarantees balanced support split.

    Primary metric: ΔAUPRC (same as regression model evaluation).
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_input, support_labels: torch.Tensor, query_input) -> torch.Tensor:
        """
        Args:
            support_input:  Tensor(n_sup, D) or PyG Batch(n_sup graphs)
            support_labels: Tensor(n_sup,) - binary (0/1); threshold at 0.5
            query_input:    Tensor(n_qry, D) or PyG Batch(n_qry graphs)
        Returns:
            p_active: Tensor(n_qry,) - P(active) for each query molecule
        """
        sup_emb = self.encoder(support_input)
        qry_emb = self.encoder(query_input)

        active_mask = support_labels > 0.5

        if not active_mask.any() or not (~active_mask).any():
            return torch.full((qry_emb.shape[0],), 0.5, device=qry_emb.device)

        proto_active   = sup_emb[active_mask].mean(dim=0)
        proto_inactive = sup_emb[~active_mask].mean(dim=0)
        protos   = torch.stack([proto_active, proto_inactive], dim=0)
        dists    = _mahalanobis_dists(qry_emb, protos, sup_emb, active_mask)
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
            valid_mask: BoolTensor(B,) - False for degenerate (all-one-class) episodes
        """
        B, n_sup = support_labels.shape
        n_qry = query_input.shape[1] if isinstance(query_input, torch.Tensor) \
                else query_input.num_graphs // B

        sup_emb = _encode_many(self.encoder, support_input, B, n_sup)
        qry_emb = _encode_many(self.encoder, query_input,   B, n_qry)

        # Binarise: support_labels are pre-binarised (0/1) - threshold at 0.5
        active_mask = support_labels > 0.5  # (B, n_support) bool

        act_count   = active_mask.float().sum(dim=1)    # (B,)
        inact_count = (~active_mask).float().sum(dim=1) # (B,)
        valid_mask  = (act_count > 0) & (inact_count > 0)  # (B,)

        # Compute class prototypes (clamped counts to avoid 0-division; invalid episodes
        # produce a meaningless prototype that we mask out in the loss)
        act_w   = active_mask.float() / act_count.clamp(min=1).unsqueeze(1)   # (B, n_sup)
        inact_w = (~active_mask).float() / inact_count.clamp(min=1).unsqueeze(1)

        proto_active   = torch.bmm(act_w.unsqueeze(1), sup_emb).squeeze(1)    # (B, emb_dim)
        proto_inactive = torch.bmm(inact_w.unsqueeze(1), sup_emb).squeeze(1)  # (B, emb_dim)

        # Squared Euclidean to each prototype - fully vectorised, no per-episode loop.
        # Mahalanobis is used only at eval time (predict_from_embeddings) where the GNN
        # is already trained and covariances are meaningful. During training the GNN starts
        # from random weights: embeddings cluster near a random shell on the unit sphere
        # (concentration of measure), so Mahalanobis gives ~uniform distances → BCE stays
        # at ln(2) ≈ 0.693 with zero gradient. Euclidean gives clean signal from day 1.
        diff_a = qry_emb - proto_active.unsqueeze(1)    # (B, n_qry, D)
        diff_i = qry_emb - proto_inactive.unsqueeze(1)  # (B, n_qry, D)
        d_active   = (diff_a ** 2).sum(dim=-1)          # (B, n_qry)
        d_inactive = (diff_i ** 2).sum(dim=-1)          # (B, n_qry)
        dists = torch.stack([d_active, d_inactive], dim=-1)  # (B, n_qry, 2)
        p_active = F.softmax(-dists, dim=-1)[..., 0]    # (B, n_qry)
        # Force degenerate (all-one-class) episodes to 0.5
        p_active = torch.where(valid_mask.unsqueeze(1), p_active,
                               torch.full_like(p_active, 0.5))
        return p_active, valid_mask

    def predict_from_embeddings(
        self, sup_emb: torch.Tensor, sup_labels: torch.Tensor, qry_emb: torch.Tensor,
        distance: str = "mahalanobis",
    ) -> torch.Tensor:
        """Predict given pre-computed embeddings - avoids re-encoding support each query chunk.

        distance="mahalanobis" (default, FS-Mol eval protocol) or "euclidean"
        (the distance the model is actually trained with - use to report the
        train/eval-consistent number alongside the Mahalanobis one).
        """
        active_mask = sup_labels > 0.5   # sup_labels must be binary (0/1)
        if not active_mask.any() or not (~active_mask).any():
            return torch.full((qry_emb.shape[0],), 0.5, device=qry_emb.device)
        proto_active   = sup_emb[active_mask].mean(dim=0)
        proto_inactive = sup_emb[~active_mask].mean(dim=0)
        protos   = torch.stack([proto_active, proto_inactive], dim=0)
        if distance == "euclidean":
            d_active   = ((qry_emb - proto_active.unsqueeze(0)) ** 2).sum(dim=-1)
            d_inactive = ((qry_emb - proto_inactive.unsqueeze(0)) ** 2).sum(dim=-1)
            dists = torch.stack([d_active, d_inactive], dim=-1)
        else:
            dists = _mahalanobis_dists(qry_emb, protos, sup_emb, active_mask)
        return F.softmax(-dists, dim=1)[:, 0]

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
            support_labels:       (B, n_support)  - continuous
            query_fingerprints:   (B, n_query, input_dim)
            query_labels:         (B, n_query)    - continuous; binarised here

        Returns:
            loss:    scalar BCE over valid episodes
            metrics: {"delta_auprc": float}
        """
        p_active, valid_mask = self.forward_batched(
            support_input, support_labels, query_input
        )

        # query_labels are pre-binarised (0/1) - threshold at 0.5
        binary_query = (query_labels > 0.5).float()  # (B, n_query)

        if not valid_mask.any():
            return torch.tensor(0.0, device=p_active.device, requires_grad=True), {"delta_auprc": float("nan")}

        # BCE only over valid (non-degenerate) episodes.
        # F.binary_cross_entropy is blocked by autocast at the C++ level regardless of dtype,
        # so compute it manually - mathematically identical, autocast-safe.
        p_valid = p_active[valid_mask].float()       # (V, n_query)
        b_valid = binary_query[valid_mask].float()   # (V, n_query)
        p_clamped = p_valid.clamp(1e-7, 1 - 1e-7)
        loss = -(b_valid * p_clamped.log() + (1 - b_valid) * (1 - p_clamped).log()).mean()

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


# =============================================================================
# ENCODER C: FS-MOL FAITHFUL GNN  (FSMolGNNEncoder)
# =============================================================================

class BOOMLayer(nn.Module):
    """
    BOOM (Bigger-out-of-Memory) FFN used in the FS-Mol GNN.
    Expands to intermediate_dim with LeakyReLU (NOT ReLU), then projects back.
    """
    def __init__(self, dim: int, intermediate_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class FSMolGNNLayer(nn.Module):
    """
    One FS-Mol-faithful GNN layer:
      1. Pre-Norm LayerNorm → PNA message passing → dropout → ReZero residual
      2. Pre-Norm LayerNorm → BOOM FFN            → dropout → ReZero residual

    alpha_mp and alpha_boom are scalar parameters initialised to 1e-7 (ReZero).
    This allows gradients to flow through the residual path from the start of
    training, so the network can initially behave like an identity mapping.
    """

    def __init__(self, hidden_dim: int, deg: "torch.Tensor", dropout: float = 0.0):
        super().__init__()
        from torch_geometric.nn import PNAConv   # type: ignore
        self.norm_mp   = nn.LayerNorm(hidden_dim)
        self.norm_boom = nn.LayerNorm(hidden_dim)
        self.conv = PNAConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggregators=["sum", "mean", "std", "max"],
            scalers=["identity", "amplification", "attenuation"],
            deg=deg,
            edge_dim=3,          # SINGLE / DOUBLE / TRIPLE one-hot
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
        )
        self.alpha_mp   = nn.Parameter(torch.full((1,), 1e-7))
        self.alpha_boom = nn.Parameter(torch.full((1,), 1e-7))
        self.boom    = BOOMLayer(hidden_dim, intermediate_dim=1024, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Message passing branch
        h = self.conv(self.norm_mp(x), edge_index, edge_attr)
        x = x + self.alpha_mp * self.dropout(h)
        # BOOM FFN branch
        x = x + self.alpha_boom * self.dropout(self.boom(self.norm_boom(x)))
        return x


class CombinedGraphReadout(nn.Module):
    """
    FS-Mol paper graph readout (graph_readout.py CombinedGraphReadout).

    Combines three pooling strategies over ALL intermediate GNN states:
      1. Weighted mean  - per-graph softmax attention weights
      2. Weighted sum   - sigmoid attention weights
      3. Max pooling    - element-wise max over nodes

    Each strategy uses num_heads independent heads of size head_dim. The three
    out_dim-dimensional results are concatenated, passed through ReLU, and
    projected back to out_dim by a bias-free linear layer.

    Default parameters match FS-Mol paper: num_heads=12, head_dim=64, out_dim=512.
    node_dim = (num_gnn_layers + 1) * hidden_channels  (all intermediate states).
    """

    NUM_HEADS = 12   # FS-Mol paper readout default
    HEAD_DIM  = 64   # FS-Mol paper readout default

    def __init__(self, node_dim: int, out_dim: int = 512):
        super().__init__()
        H      = self.NUM_HEADS
        D      = self.HEAD_DIM
        hidden = H * D  # 12 * 64 = 768

        # Weighted mean: softmax attention
        self.score_mean  = nn.Linear(node_dim, H)
        self.values_mean = nn.Linear(node_dim, hidden)
        self.proj_mean   = nn.Linear(hidden, out_dim, bias=False)

        # Weighted sum: sigmoid attention
        self.score_sum   = nn.Linear(node_dim, H)
        self.values_sum  = nn.Linear(node_dim, hidden)
        self.proj_sum    = nn.Linear(hidden, out_dim, bias=False)

        # Max pooling: project then pool
        self.proj_max    = nn.Linear(node_dim, out_dim, bias=False)

        # Combine all three readouts
        self.combine     = nn.Linear(3 * out_dim, out_dim, bias=False)

        self._H = H
        self._D = D

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        wm = self._weighted_mean(x, batch_idx, num_graphs)
        ws = self._weighted_sum(x, batch_idx, num_graphs)
        mx = self._max_pool(x, batch_idx, num_graphs)
        return self.combine(F.relu(torch.cat([wm, ws, mx], dim=-1)))   # (G, out_dim)

    @staticmethod
    def _scatter_add(src: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.zeros(n, src.shape[-1], device=src.device, dtype=src.dtype)
        out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
        return out

    def _weighted_mean(self, x, batch_idx, num_graphs):
        from torch_geometric.utils import softmax   # type: ignore
        H, D    = self._H, self._D
        scores  = self.score_mean(x)                                   # (N, H)
        weights = softmax(scores, batch_idx, num_nodes=num_graphs)     # (N, H) per-graph softmax
        values  = self.values_mean(x).view(-1, H, D)                   # (N, H, D)
        pooled  = self._scatter_add(
            (weights.unsqueeze(-1) * values).reshape(-1, H * D), batch_idx, num_graphs
        )                                                               # (G, H*D)
        return self.proj_mean(pooled)                                   # (G, out_dim)

    def _weighted_sum(self, x, batch_idx, num_graphs):
        H, D    = self._H, self._D
        weights = torch.sigmoid(self.score_sum(x))                     # (N, H)
        values  = self.values_sum(x).view(-1, H, D)                    # (N, H, D)
        pooled  = self._scatter_add(
            (weights.unsqueeze(-1) * values).reshape(-1, H * D), batch_idx, num_graphs
        )                                                               # (G, H*D)
        return self.proj_sum(pooled)                                    # (G, out_dim)

    def _max_pool(self, x, batch_idx, num_graphs):
        from torch_geometric.nn import global_max_pool   # type: ignore
        return global_max_pool(self.proj_max(x), batch_idx, size=num_graphs)  # (G, out_dim)


class FSMolGNNEncoder(nn.Module):
    """
    FS-Mol faithful GNN encoder with feature fusion.

    Architecture (matches FS-Mol paper exactly):
      - Linear node projection: node_feat_dim → hidden_channels (128), NO bias
      - num_layers × FSMolGNNLayer  (PNA + Pre-Norm LN + ReZero + BOOM FFN)
      - CombinedGraphReadout over ALL intermediate node states:
          cat(initial, layer_1, ..., layer_N)  →  (N_nodes, (num_layers+1)*128)
          → weighted_mean + weighted_sum + max  →  512-dim graph embedding
      - Feature fusion: cat(GNN_512, ECFP_2048, desc_42) → Linear(2602→1024) → ReLU → Linear(1024→512)
      - NO L2 normalisation - paper uses raw FC output for Mahalanobis distance

    Edge format: 3-dim one-hot (SINGLE/DOUBLE/TRIPLE) from smiles_to_fsmol_graph().
    Graph-level ecfp (n_graphs, 2048) and descriptors (n_graphs, 42) stored as
    (1, D) per Data object so Batch.from_data_list stacks them correctly.
    """

    ECFP_DIM       = 2048
    DESCRIPTOR_DIM = 42
    READOUT_DIM    = 512   # CombinedGraphReadout output dim

    def __init__(
        self,
        node_feat_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 10,
        embedding_dim: int = 512,
        deg: "torch.Tensor | None" = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.deg = deg   # stored for checkpoint serialisation

        # No bias - matches FS-Mol paper init_node_proj
        self.node_emb = nn.Linear(node_feat_dim, hidden_channels, bias=False)

        self.gnn_layers = nn.ModuleList([
            FSMolGNNLayer(hidden_channels, deg, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Readout concatenates all (num_layers + 1) intermediate states
        readout_node_dim = (num_layers + 1) * hidden_channels   # e.g. 11 * 128 = 1408
        self.readout = CombinedGraphReadout(node_dim=readout_node_dim, out_dim=self.READOUT_DIM)

        fc_in = self.READOUT_DIM + self.ECFP_DIM + self.DESCRIPTOR_DIM  # 512+2048+42 = 2602
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        """
        Returns (n_graphs, embedding_dim) - raw FC output, no L2 normalisation.
        """
        x = self.node_emb(batch.x)          # (N_nodes, 128)

        # Run GNN layers and collect ALL intermediate representations
        all_states = [x]
        for layer in self.gnn_layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
            all_states.append(x)

        # Concatenate all states: (N_nodes, (num_layers+1)*128)
        node_repr = torch.cat(all_states, dim=-1)

        # Graph readout: (n_graphs, 512)
        gnn_emb = self.readout(node_repr, batch.batch, int(batch.num_graphs))

        # Feature fusion: (n_graphs, 2602) → (n_graphs, 512)
        fused = torch.cat([gnn_emb, batch.ecfp, batch.descriptors], dim=-1)
        return self.fc(fused)