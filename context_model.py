"""
Cross-Attention In-Context Learning model for Few-Shot Molecular Property Prediction.

Architecture (Option B, two-step):
  1. Encode all molecules: Encoder → h  (d_enc)
  2. Augment with label context:
       support: label_embed(y_i)  ∈ {active, inactive}  → nn.Embedding(2, d_label)
       query:   see query_label_mode below
     Then: c = MLP_F(concat(h, label_context))  → d_attn
  3. Self-attention over labeled support:  c_sup  →  c_sup'  (context-enriched)
  4. Cross-attention — query attends to enriched support:
       Q = c_qry,  K = V = c_sup'  →  c_qry'
  5. PN head: prototypes from c_sup', Euclidean distance during training,
     Mahalanobis at evaluation (same as PrototypicalNetworkClassification).

All design choices are constructor arguments so nothing is hardcoded.

Compatible interface with PrototypicalNetworkClassification:
  forward_batched(support_input, support_labels, query_input)
  predict_from_embeddings(sup_enc, sup_labels, qry_enc, distance="mahalanobis")
  compute_loss_batched(support_input, support_labels, query_input, query_labels)

═══════════════════════════════════════════════════════════════════════════════════
DESIGN CHOICES — what we decided, why, and what the alternative is
═══════════════════════════════════════════════════════════════════════════════════

[DC-1] QUERY LABEL STRATEGY  (query_label_mode)
  Chosen:      "unknown_token" — a learned nn.Parameter(d_label) represents "label
               unknown". The model learns what "not knowing" should look like in
               label-embedding space. This is the approach in Schimunek et al. 2023.
  Alternative: "mask" — zero-fill the label slot for query molecules. Simpler, no
               extra parameters. Risk: zeros collide with the trained distribution of
               label embeddings (active ≈ positive weights, inactive ≈ negative, so
               zero ≈ "neither" by accident rather than by learning).
  Configurable via: query_label_mode="unknown_token" | "mask"

[DC-2] TENSOR BATCHING STRATEGY  (use_nested_tensors)
  Chosen:      Dense padded tensors (use_nested_tensors=False). All episodes in a
               batch have the same support size K, so padding wastes nothing during
               training. Standard ops, fully debuggable.
  Alternative: Jagged/nested tensors (torch.nested, layout=torch.jagged, PyTorch ≥2.1).
               Lets different episodes have different K in the same batch — useful for
               evaluation where we sweep K=16..512. Attention natively skips padding
               positions. Cross-attention (query→support) falls back to dense even in
               nested mode because jagged K/V with dense Q is not yet stable in PyTorch.
  Configurable via: use_nested_tensors=True | False

[DC-3] NORMALIZATION POSITION  (pre-norm, hardcoded)
  Chosen:      Pre-norm — LayerNorm(x) before Q/K/V projection, residual added after.
               More stable training gradient flow (Xiong et al. 2020 "On Layer
               Normalization in the Transformer Architecture").
  Alternative: Post-norm — LayerNorm after the residual (original "Attention Is All
               You Need"). Can achieve better final performance but requires warm-up
               and careful LR scheduling to avoid early instability.
  To ablate:   Swap the norm position inside _SelfAttnLayer / _CrossAttnLayer.

[DC-4] SHARED MLP_F FOR SUPPORT AND QUERY  (hardcoded)
  Chosen:      The same MLP_F processes both support (with label embed) and query (with
               unknown token). Fewer parameters; the projection space is shared.
  Alternative: Separate MLP_F_sup and MLP_F_qry. Could learn a better projection for
               each role (support has real label signal; query has a dummy token).
               Adds ~2× the MLP_F parameters.

[DC-5] FFN SUBLAYER AFTER ATTENTION  (use_ffn)
  Chosen:      False — skip the feed-forward sublayer after each attention block.
               Reduces parameters; our sequences are short (K ≤ 64).
  Alternative: True — standard transformer block (attention → FFN with 4× expansion).
               Adds expressive capacity per layer, standard in language models.
  Configurable via: use_ffn=True | False

[DC-6] SELF-THEN-CROSS vs INTERLEAVED  (hardcoded as Option B)
  Chosen:      All n_self_layers of self-attention run first over support, then all
               n_cross_layers of cross-attention query→support. Clean two-phase design
               from Schimunek et al. 2023 (their "Option B").
  Alternative: Interleave: (self-attn → cross-attn) × n_layers. Richer information
               exchange but harder to reason about; not evaluated in the paper.

[DC-7] PROTOTYPE AGGREGATION  (hardcoded simple mean)
  Chosen:      Mean of enriched support embeddings per class — same PN head as PTN.
               Euclidean distance during training, Mahalanobis at eval (follows
               FS-Mol evaluation protocol).
  Alternative: Learned aggregation — attend over class-specific support to compute
               a prototype (instead of averaging). Would turn the prototype itself into
               an attention output. More complex, not tested here.

[DC-8] ENCODER SHARING BETWEEN SUPPORT AND QUERY  (hardcoded)
  Chosen:      Same encoder weights for support and query. Standard in ProtoNets.
  Alternative: Separate encoders. Rarely helps; doubles encoder parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from model import _encode_many, _mahalanobis_dists


# =============================================================================
# ATTENTION SUB-LAYERS
# =============================================================================

class _SelfAttnLayer(nn.Module):
    """Pre-norm self-attention block with optional FFN sublayer.  [DC-3, DC-5]"""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1, use_ffn: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ff = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, d) dense or a jagged NestedTensor (B × K_i × d)."""
        normed = self.norm(x)
        h, _ = self.attn(normed, normed, normed)
        x = x + self.drop(h)
        if self.use_ffn:
            x = x + self.drop(self.ff(self.norm_ff(x)))
        return x


class _CrossAttnLayer(nn.Module):
    """Pre-norm cross-attention block: Q attends to KV, with optional FFN sublayer.
    Always uses dense tensors — jagged cross-attention (dense Q, jagged K/V) is not
    stable in current PyTorch.  [DC-2, DC-3, DC-5]"""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1, use_ffn: bool = False):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ff = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)
            )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(h)
        if self.use_ffn:
            q = q + self.drop(self.ff(self.norm_ff(q)))
        return q


# =============================================================================
# CROSS-ATTENTION PROTOTYPICAL NETWORK
# =============================================================================

class CrossAttentionProtoNet(nn.Module):
    """
    In-context learning model for binary few-shot molecular classification.

    Args:
        encoder:             Any encoder module (ECFPEncoder or FSMolGNNEncoder).
        encoder_output_dim:  Output dimension of the encoder (256 ECFP, 512 GNN).
        d_attn:              Attention and representation dimension.  Must be divisible by n_heads.
        d_label:             Dimension of the label embedding.
        n_heads:             Number of attention heads in self- and cross-attention.
        n_self_layers:       Number of self-attention layers over labeled support.
        n_cross_layers:      Number of cross-attention layers (query attends to support).
        mlp_f_hidden:        Hidden dim of MLP_F that projects to d_attn.
        dropout:             Dropout probability in attention and MLP_F.
        use_ffn:             Add a FFN sublayer after each attention layer (Transformer-style).
                             False by default — enable to ablate.  [DC-5]
        query_label_mode:    How to represent the query's unknown label.  [DC-1]
                             "unknown_token" (default) — learned nn.Parameter.
                             "mask"                    — hard zeros (no learned param).
        use_nested_tensors:  If True, pack support molecules into a jagged NestedTensor
                             (torch.nested, requires PyTorch ≥ 2.1) for self-attention,
                             enabling variable-length support sets in the same batch.
                             Falls back gracefully to dense if not available.  [DC-2]
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        d_attn: int = 256,
        d_label: int = 16,
        n_heads: int = 4,
        n_self_layers: int = 1,
        n_cross_layers: int = 1,
        mlp_f_hidden: int = 256,
        dropout: float = 0.1,
        use_ffn: bool = False,
        # Design-choice flags
        query_label_mode: str = "unknown_token",   # [DC-1]  "unknown_token" | "mask"
        use_nested_tensors: bool = False,           # [DC-2]
    ):
        assert d_attn % n_heads == 0, (
            f"d_attn ({d_attn}) must be divisible by n_heads ({n_heads})"
        )
        assert query_label_mode in ("unknown_token", "mask"), (
            f"query_label_mode must be 'unknown_token' or 'mask', got '{query_label_mode}'"
        )
        super().__init__()

        self.encoder = encoder
        self._d_label = d_label
        self.query_label_mode = query_label_mode     # [DC-1]
        self.use_nested_tensors = use_nested_tensors # [DC-2]

        # Label context tokens  [DC-1]
        self.label_embed = nn.Embedding(2, d_label)   # active=1, inactive=0
        # unknown_label always created for checkpoint compatibility; only used when
        # query_label_mode=="unknown_token".
        self.unknown_label = nn.Parameter(torch.zeros(d_label))

        # MLP_F: (d_enc + d_label) → d_attn   [DC-4]
        self.mlp_f = nn.Sequential(
            nn.Linear(encoder_output_dim + d_label, mlp_f_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_f_hidden, d_attn),
        )

        # Step 1 — self-attention over labeled support  [DC-6]
        self.self_attn_layers = nn.ModuleList([
            _SelfAttnLayer(d_attn, n_heads, dropout=dropout, use_ffn=use_ffn)
            for _ in range(n_self_layers)
        ])

        # Step 2 — cross-attention: query attends to enriched support  [DC-6]
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttnLayer(d_attn, n_heads, dropout=dropout, use_ffn=use_ffn)
            for _ in range(n_cross_layers)
        ])

        # Final LayerNorms before computing prototypes / distances
        self.final_norm_sup = nn.LayerNorm(d_attn)
        self.final_norm_qry = nn.LayerNorm(d_attn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment_and_project(
        self,
        sup_enc: torch.Tensor,          # (B, n_sup, d_enc)
        support_labels: torch.Tensor,   # (B, n_sup)  binary float 0/1
        qry_enc: torch.Tensor,          # (B, n_qry, d_enc)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attach label context and project to attention dimension via MLP_F.  [DC-1, DC-4]"""
        B, n_sup, _ = sup_enc.shape
        n_qry = qry_enc.shape[1]

        # Support: embed discrete label
        sup_lbl = support_labels.long().clamp(0, 1)    # (B, n_sup)
        sup_ctx = self.label_embed(sup_lbl)             # (B, n_sup, d_label)

        # Query label context  [DC-1]
        if self.query_label_mode == "unknown_token":
            # Learned "I don't know the label" token
            qry_ctx = (
                self.unknown_label
                .unsqueeze(0).unsqueeze(0)
                .expand(B, n_qry, -1)
            )
        else:
            # "mask": zero-fill — no parameters, but zeros may collide with label
            # embedding weights (alternative worth ablating).
            qry_ctx = torch.zeros(
                B, n_qry, self._d_label,
                device=qry_enc.device, dtype=qry_enc.dtype,
            )

        c_sup = self.mlp_f(torch.cat([sup_enc, sup_ctx], dim=-1))   # (B, n_sup, d_attn)
        c_qry = self.mlp_f(torch.cat([qry_enc, qry_ctx], dim=-1))   # (B, n_qry, d_attn)
        return c_sup, c_qry

    def _attend(
        self,
        c_sup: torch.Tensor,  # (B, n_sup, d_attn)
        c_qry: torch.Tensor,  # (B, n_qry, d_attn)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-attention over support, then cross-attention for query.  [DC-2, DC-6]"""
        if self.use_nested_tensors and len(self.self_attn_layers) > 0:
            c_sup = self._self_attn_nested(c_sup)
        else:
            for layer in self.self_attn_layers:
                c_sup = layer(c_sup)

        # Cross-attention always uses dense tensors  [DC-2]
        for layer in self.cross_attn_layers:
            c_qry = layer(c_qry, c_sup)

        return self.final_norm_sup(c_sup), self.final_norm_qry(c_qry)

    def _self_attn_nested(self, c_sup: torch.Tensor) -> torch.Tensor:
        """Self-attention over support using jagged NestedTensors.  [DC-2]

        Packs each episode's support molecules into a jagged NestedTensor so that
        episodes with different support sizes K_i can share the same batch without
        padding waste. Falls back to dense attention if the PyTorch version does not
        support jagged layout or if an error occurs.

        After attention, unpacks to a padded dense tensor for cross-attention (jagged
        K/V with dense Q is not yet stable in PyTorch — see _CrossAttnLayer docstring).
        """
        try:
            # Unpack batch into per-episode tensors: each (K_i, d_attn)
            sup_list = list(c_sup.unbind(0))
            c_sup_nt = torch.nested.nested_tensor(
                sup_list, layout=torch.jagged, device=c_sup.device, dtype=c_sup.dtype
            )
            for layer in self.self_attn_layers:
                c_sup_nt = layer(c_sup_nt)   # _SelfAttnLayer.forward works on NestedTensors
            # Unpack back to padded dense for cross-attention
            return c_sup_nt.to_padded_tensor(padding=0.0)
        except Exception:
            # Graceful fallback: dense attention (e.g. PyTorch < 2.1 or unsupported op)
            for layer in self.self_attn_layers:
                c_sup = layer(c_sup)
            return c_sup

    def _pn_head_batched(
        self,
        c_sup: torch.Tensor,            # (B, n_sup, d_attn)
        support_labels: torch.Tensor,   # (B, n_sup)
        c_qry: torch.Tensor,            # (B, n_qry, d_attn)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euclidean prototypical head — vectorised over the batch.  [DC-7]"""
        active_mask = support_labels > 0.5  # (B, n_sup)
        act_count   = active_mask.float().sum(dim=1)     # (B,)
        inact_count = (~active_mask).float().sum(dim=1)
        valid_mask  = (act_count > 0) & (inact_count > 0)

        act_w   = active_mask.float() / act_count.clamp(min=1).unsqueeze(1)
        inact_w = (~active_mask).float() / inact_count.clamp(min=1).unsqueeze(1)

        proto_a = torch.bmm(act_w.unsqueeze(1), c_sup).squeeze(1)    # (B, d_attn)
        proto_i = torch.bmm(inact_w.unsqueeze(1), c_sup).squeeze(1)

        d_a = ((c_qry - proto_a.unsqueeze(1)) ** 2).sum(dim=-1)      # (B, n_qry)
        d_i = ((c_qry - proto_i.unsqueeze(1)) ** 2).sum(dim=-1)
        dists   = torch.stack([d_a, d_i], dim=-1)                    # (B, n_qry, 2)
        p_active = F.softmax(-dists, dim=-1)[..., 0]                 # (B, n_qry)

        p_active = torch.where(
            valid_mask.unsqueeze(1), p_active, torch.full_like(p_active, 0.5)
        )
        return p_active, valid_mask

    # ------------------------------------------------------------------
    # Public API (matches PrototypicalNetworkClassification)
    # ------------------------------------------------------------------

    def forward_batched(
        self,
        support_input,
        support_labels: torch.Tensor,   # (B, n_sup)
        query_input,
        train_distance: str = "euclidean",   # kept for API compatibility; always Euclidean
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            support_input:  Tensor(B, n_sup, d_enc)  or  PyG Batch(B*n_sup graphs)
            support_labels: Tensor(B, n_sup)  binary 0/1
            query_input:    Tensor(B, n_qry, d_enc)  or  PyG Batch(B*n_qry graphs)
        Returns:
            p_active:   (B, n_qry)
            valid_mask: (B,) BoolTensor  — False for degenerate (all-one-class) episodes
        """
        B, n_sup = support_labels.shape
        n_qry = (query_input.shape[1] if isinstance(query_input, torch.Tensor)
                 else query_input.num_graphs // B)

        sup_enc = _encode_many(self.encoder, support_input, B, n_sup)
        qry_enc = _encode_many(self.encoder, query_input,   B, n_qry)

        c_sup, c_qry = self._augment_and_project(sup_enc, support_labels, qry_enc)
        c_sup, c_qry = self._attend(c_sup, c_qry)
        return self._pn_head_batched(c_sup, support_labels, c_qry)

    def predict_from_embeddings(
        self,
        sup_enc: torch.Tensor,          # (n_sup, d_enc)  raw encoder output
        sup_labels: torch.Tensor,       # (n_sup,)  binary 0/1
        qry_enc: torch.Tensor,          # (n_qry, d_enc)  raw encoder output
        distance: str = "mahalanobis",
    ) -> torch.Tensor:
        """
        Single-episode prediction from pre-computed encoder outputs.

        Called by evaluate.py to avoid re-encoding support for every query chunk.
        The label embedding + MLP_F + attention must re-run here because the context
        model needs labels at attention time — they cannot be factored out.

        distance: "mahalanobis" (default, FS-Mol eval protocol) or "euclidean".
        """
        c_sup, c_qry = self._augment_and_project(
            sup_enc.unsqueeze(0), sup_labels.float().unsqueeze(0), qry_enc.unsqueeze(0)
        )
        c_sup, c_qry = self._attend(c_sup, c_qry)
        c_sup = c_sup.squeeze(0)   # (n_sup, d_attn)
        c_qry = c_qry.squeeze(0)   # (n_qry, d_attn)

        active_mask = sup_labels > 0.5
        if not active_mask.any() or not (~active_mask).any():
            return torch.full((c_qry.shape[0],), 0.5, device=c_qry.device)

        proto_a = c_sup[active_mask].mean(dim=0)
        proto_i = c_sup[~active_mask].mean(dim=0)
        protos  = torch.stack([proto_a, proto_i], dim=0)

        if distance == "euclidean":
            d_a = ((c_qry - proto_a.unsqueeze(0)) ** 2).sum(dim=-1)
            d_i = ((c_qry - proto_i.unsqueeze(0)) ** 2).sum(dim=-1)
            dists = torch.stack([d_a, d_i], dim=-1)
        else:
            dists = _mahalanobis_dists(c_qry, protos, c_sup, active_mask)

        return F.softmax(-dists, dim=1)[:, 0]

    def get_enriched_embeddings(
        self,
        sup_enc: torch.Tensor,           # (n_sup, d_enc)  raw encoder output
        sup_labels: torch.Tensor,        # (n_sup,)  binary 0/1
        qry_enc: torch.Tensor,           # (n_qry, d_enc)  raw encoder output
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context-enriched (c_sup, c_qry) after MLP_F + attention.

        Pre-compute raw encoder outputs once per assay, then call this per episode
        to get the attention-processed embeddings used by evaluate.py's head registry.
        """
        c_sup, c_qry = self._augment_and_project(
            sup_enc.unsqueeze(0), sup_labels.float().unsqueeze(0), qry_enc.unsqueeze(0)
        )
        c_sup, c_qry = self._attend(c_sup, c_qry)
        return c_sup.squeeze(0), c_qry.squeeze(0)   # (n_sup, d_attn), (n_qry, d_attn)

    def compute_loss_batched(
        self,
        support_input,
        support_labels: torch.Tensor,
        query_input,
        query_labels: torch.Tensor,
        train_distance: str = "euclidean",
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            support_input:  (B, n_sup, d_enc) or PyG Batch
            support_labels: (B, n_sup)  binary 0/1
            query_input:    (B, n_qry, d_enc) or PyG Batch
            query_labels:   (B, n_qry)  binary 0/1
        Returns:
            loss:    scalar BCE over valid episodes
            metrics: {"delta_auprc": float}
        """
        p_active, valid_mask = self.forward_batched(
            support_input, support_labels, query_input
        )
        binary_query = (query_labels > 0.5).float()

        if not valid_mask.any():
            return (
                torch.tensor(0.0, device=p_active.device, requires_grad=True),
                {"delta_auprc": float("nan")},
            )

        p_valid = p_active[valid_mask].float()
        b_valid = binary_query[valid_mask].float()
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
                        d = (
                            float(average_precision_score(b_np[i], p_np[i]))
                            - float(b_np[i].mean())
                        )
                        dauprc_vals.append(d)
                    except Exception:
                        pass
            avg_dauprc = float(np.mean(dauprc_vals)) if dauprc_vals else float("nan")

        return loss, {"delta_auprc": avg_dauprc}
