"""
Cross-Attention In-Context Learning model for Few-Shot Molecular Property Prediction.

Architecture:
  1. Encode all molecules: Encoder → h  (d_enc)
  2. Augment with label context:
       support: label_embed(y_i)  ∈ {active, inactive}  → nn.Embedding(2, d_label)
       query:   see query_label_mode below
     Then: c = MLP_F(concat(h, label_context))  → d_attn
     (MLP_F is shared or split per role — see [DC-4])
  3. Self-attention over labeled support:  c_sup  →  c_sup'  (context-enriched)
  4. Cross-attention — query attends to enriched support:
       Q = c_qry,  K = V = c_sup'  →  c_qry'
     (steps 3/4 run sequentially or interleaved — see [DC-6])
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
               The nested path additionally requires attn_order="sequential",
               softmax_mode="standard" and no attention-stat recording, because those
               paths need explicit attention matrices which jagged layout does not give.
  Configurable via: use_nested_tensors=True | False

[DC-3] NORMALIZATION POSITION  (norm_position)
  Chosen:      "pre" — LayerNorm(x) before Q/K/V projection, residual added after.
               More stable training gradient flow (Xiong et al. 2020 "On Layer
               Normalization in the Transformer Architecture"). This is what MHNfs uses.
  Alternative: "post" — LayerNorm after the residual (original "Attention Is All You
               Need"). Can achieve better final performance but requires warm-up and
               careful LR scheduling to avoid early instability.
  Why it matters here (supervisor, 2026-10-12): norm position controls activation
               magnitude, and activation magnitude entering the attention logits bounds
               how peaked the softmax can get. If magnitudes are too low, attention
               cannot concentrate on the few relevant context molecules. So this is not
               a cosmetic choice — it interacts directly with [DC-10].
  Configurable via: norm_position="pre" | "post"

[DC-4] MLP_F FOR SUPPORT AND QUERY  (separate_mlp_f)
  Chosen:      True — separate MLP_S(x_S | y_S) and MLP_Q(x_Q).
  Alternative: False — one shared MLP_F for both roles. Fewer parameters; the
               projection space is shared.
  Rule (supervisor, 2026-10-12): the MLP split must match the attention split. With
               shared attention weights + masking, share the MLP. With separate
               operators (our [DC-6]: distinct self- and cross-attention weights), the
               initial MLPs should also be separate. Since we use separate operators,
               separate_mlp_f=True is the consistent setting and is therefore the
               default. separate_mlp_f=False exists to ablate that rule.
  Note:        Checkpoints written before this flag existed were trained with a shared
               MLP_F. `_load_model_from_checkpoint` therefore defaults the flag to
               False when a checkpoint config does not carry it.
  Configurable via: separate_mlp_f=True | False

[DC-5] FFN SUBLAYER AFTER ATTENTION  (use_ffn)
  Chosen:      False — skip the feed-forward sublayer after each attention block.
               Reduces parameters; our sequences are short (K ≤ 64 during training).
  Alternative: True — standard transformer block (attention → FFN with 4× expansion).
               Adds expressive capacity per layer, standard in language models.
  Configurable via: use_ffn=True | False

[DC-6] SELF-THEN-CROSS vs INTERLEAVED  (attn_order)
  Chosen:      "sequential" — all n_self_layers of self-attention run first over
               support, then all n_cross_layers of cross-attention query→support.
               Clean two-phase design, as in Schimunek et al. 2023.
  Alternative: "interleaved" — (self-attn → cross-attn) × n_layers. The query sees a
               partially-enriched support and the enrichment continues afterwards, so
               information is exchanged at several depths. Joschka suggested this may
               work better; an encoder-only stack that interleaves the two is the
               standard arrangement in in-context transformers.
               Requires n_self_layers == n_cross_layers.
  Configurable via: attn_order="sequential" | "interleaved"

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

[DC-10] ATTENTION SOFTMAX SCALING  (softmax_mode)
  Chosen:      "standard" — ordinary softmax over logits scaled by 1/sqrt(head_dim).
  Problem:     As the support set grows (we evaluate at K up to 512), a standard
               softmax spreads mass over more keys and cannot stay peaked, so the model
               may be unable to attend to the few compounds that matter. Whether this
               actually happens is an empirical question — measure it with
               `collect_attn_stats` before reaching for a fix.
  Alternatives (Scalable-Softmax, used by TabICL v2 — arXiv 2602.11139):
               "ssmax"   — logits are multiplied by s·log(K), with one learnable s per
                           head. Lets the sharpness grow with context length.
               "qassmax" — query-aware: s additionally depends on the query vector via
                           a linear map, so each query picks its own sharpness.
               Both are initialised so that s·log(64) = 1, i.e. they reproduce standard
               attention at K=64 (our training support size) and only change behaviour
               away from it. That keeps the comparison honest.
  Configurable via: softmax_mode="standard" | "ssmax" | "qassmax"

[DC-11] ATTENTION DIAGNOSTICS  (collect_attn_stats)
  Not a modelling choice — an instrument. When enabled, every attention layer records
  the entropy and the maximum weight of its attention distribution. Normalised entropy
  near 1.0 means attention is uniform (diffuse); near 0.0 means it is concentrated on
  one key. This is measurement M8 in TASKS.md and it is what decides whether [DC-10]
  needs anything other than "standard".
  Enabling it forces the dense attention path and disables PyTorch's fused kernels, so
  it is for analysis runs only, never for training.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from model import _encode_many, _mahalanobis_dists


# Valid values for each configurable design choice — imported by main_attention.py and
# evaluate.py so the CLI, the model and the checkpoint loader cannot drift apart.
QUERY_LABEL_MODES = ("unknown_token", "mask")
NORM_POSITIONS    = ("pre", "post")
ATTN_ORDERS       = ("sequential", "interleaved")
SOFTMAX_MODES     = ("standard", "ssmax", "qassmax")
LOGIT_SCALES      = ("sqrt_d", "none", "learned")

# Reference context length at which scalable softmax reproduces standard attention.
# 64 = our default training support size, so ssmax/qassmax start as a no-op change and
# only diverge from standard attention at other support sizes.
SSMAX_REF_LEN = 64.0


def _ssmax_raw_init() -> float:
    """Raw (pre-softplus) value of s such that softplus(s)·log(SSMAX_REF_LEN) == 1."""
    s_target = 1.0 / math.log(SSMAX_REF_LEN)
    return math.log(math.expm1(s_target))      # inverse of softplus


# =============================================================================
# ATTENTION PRIMITIVE
# =============================================================================

class _MHA(nn.Module):
    """Multi-head attention with optional scalable softmax and stat recording.

    Parameters live inside a real `nn.MultiheadAttention` submodule (`self.attn`) in
    every mode, so a state_dict is interchangeable between softmax modes: switching
    mode adds or drops only the small `s_raw` / `s_proj` tensors. Two consequences:

      * softmax_mode="standard" delegates to `nn.MultiheadAttention`, keeping its fused
        kernels and its native NestedTensor support ([DC-2]).
      * the scalable-softmax modes re-implement the attention math by hand using
        `self.attn`'s projection weights, because the logit scaling has to happen
        between the QK product and the softmax.  [DC-10]
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1,
                 softmax_mode: str = "standard"):
        super().__init__()
        if softmax_mode not in SOFTMAX_MODES:
            raise ValueError(f"softmax_mode must be one of {SOFTMAX_MODES}, got '{softmax_mode}'")
        self.d            = d
        self.n_heads      = n_heads
        self.head_dim     = d // n_heads
        self.dropout_p    = dropout
        self.softmax_mode = softmax_mode

        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

        if softmax_mode != "standard":
            # One learnable sharpness per head, parameterised through softplus so it
            # stays positive. Init: s·log(64) == 1  → identical to standard attention
            # at K=64.
            self.s_raw = nn.Parameter(torch.full((n_heads,), _ssmax_raw_init()))
            if softmax_mode == "qassmax":
                # Query-aware offset. Zero-initialised so training starts exactly at
                # the "ssmax" behaviour and only departs from it if that helps.
                self.s_proj = nn.Linear(d, n_heads)
                nn.init.zeros_(self.s_proj.weight)
                nn.init.zeros_(self.s_proj.bias)

        # Filled by forward() when record is requested; read via pop_stats().
        self._stats: dict[str, float] | None = None

    # -- diagnostics -------------------------------------------------------

    def pop_stats(self) -> dict[str, float] | None:
        """Return and clear the stats recorded by the most recent forward pass."""
        s, self._stats = self._stats, None
        return s

    def _record(self, attn_w: torch.Tensor) -> None:
        """attn_w: (B, H, Lq, Lk) post-softmax weights.  [DC-11]"""
        with torch.no_grad():
            w = attn_w.detach().float()
            n_keys = w.shape[-1]
            ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)   # (B,H,Lq)
            max_w = w.max(-1).values                                          # (B,H,Lq)
            norm = math.log(n_keys) if n_keys > 1 else 1.0
            self._stats = {
                "n_keys":           float(n_keys),
                "entropy":          float(ent.mean()),
                "entropy_norm":     float(ent.mean() / norm),
                "max_weight":       float(max_w.mean()),
                "uniform_weight":   1.0 / n_keys,
            }

    # -- forward -----------------------------------------------------------

    def forward(self, q, k, v, record: bool = False):
        """q: (B, Lq, d) or NestedTensor; k/v: (B, Lk, d) or NestedTensor."""
        if self.softmax_mode == "standard" and not record:
            out, _ = self.attn(q, k, v, need_weights=False)
            return out
        if self.softmax_mode == "standard":
            out, w = self.attn(q, k, v, need_weights=True, average_attn_weights=False)
            self._record(w)
            return out
        return self._scaled_forward(q, k, v, record=record)

    def _scaled_forward(self, q, k, v, record: bool = False):
        """Manual attention with scalable softmax.  Dense tensors only.  [DC-10]"""
        mha = self.attn
        w_in, b_in = mha.in_proj_weight, mha.in_proj_bias
        w_q, w_k, w_v = w_in.chunk(3, dim=0)
        if b_in is not None:
            b_q, b_k, b_v = b_in.chunk(3, dim=0)
        else:
            b_q = b_k = b_v = None

        B, Lq, _ = q.shape
        Lk       = k.shape[1]
        H, hd    = self.n_heads, self.head_dim

        Q = F.linear(q, w_q, b_q).view(B, Lq, H, hd).transpose(1, 2)   # (B,H,Lq,hd)
        K = F.linear(k, w_k, b_k).view(B, Lk, H, hd).transpose(1, 2)   # (B,H,Lk,hd)
        V = F.linear(v, w_v, b_v).view(B, Lk, H, hd).transpose(1, 2)

        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(hd)             # (B,H,Lq,Lk)

        # Scalable softmax: multiply logits by s·log(Lk) so the attainable sharpness
        # grows with context length instead of being washed out by it.
        log_n = math.log(max(float(Lk), 2.0))
        if self.softmax_mode == "ssmax":
            s = F.softplus(self.s_raw).view(1, H, 1, 1)
        else:  # qassmax — sharpness also depends on the query
            s_q = self.s_proj(q).permute(0, 2, 1).unsqueeze(-1)        # (B,H,Lq,1)
            s   = F.softplus(self.s_raw.view(1, H, 1, 1) + s_q)
        logits = logits * (s * log_n)

        attn_w = logits.softmax(dim=-1)
        if record:
            self._record(attn_w)
        attn_w = F.dropout(attn_w, p=self.dropout_p, training=self.training)

        out = (attn_w @ V).transpose(1, 2).reshape(B, Lq, self.d)
        return mha.out_proj(out)


# =============================================================================
# ATTENTION SUB-LAYERS
# =============================================================================

class _SelfAttnLayer(nn.Module):
    """Self-attention block with configurable norm position and optional FFN.

    [DC-3] norm_position:
        "pre"  →  x + Attn(LN(x))          then  x + FF(LN(x))
        "post" →  LN(x + Attn(x))          then  LN(x + FF(x))
    [DC-5] use_ffn, [DC-10] softmax_mode
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1,
                 use_ffn: bool = False, norm_position: str = "pre",
                 softmax_mode: str = "standard"):
        super().__init__()
        if norm_position not in NORM_POSITIONS:
            raise ValueError(f"norm_position must be one of {NORM_POSITIONS}, got '{norm_position}'")
        self.norm_position = norm_position
        self.norm = nn.LayerNorm(d)
        self.attn = _MHA(d, n_heads, dropout=dropout, softmax_mode=softmax_mode)
        self.drop = nn.Dropout(dropout)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ff = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)
            )

    def forward(self, x: torch.Tensor, record: bool = False) -> torch.Tensor:
        """x: (B, K, d) dense or a jagged NestedTensor (B × K_i × d)."""
        if self.norm_position == "pre":
            normed = self.norm(x)
            x = x + self.drop(self.attn(normed, normed, normed, record=record))
            if self.use_ffn:
                x = x + self.drop(self.ff(self.norm_ff(x)))
        else:  # post
            x = self.norm(x + self.drop(self.attn(x, x, x, record=record)))
            if self.use_ffn:
                x = self.norm_ff(x + self.drop(self.ff(x)))
        return x


class _CrossAttnLayer(nn.Module):
    """Cross-attention block: Q attends to KV. Configurable norm position + FFN.

    Always uses dense tensors — jagged cross-attention (dense Q, jagged K/V) is not
    stable in current PyTorch.  [DC-2, DC-3, DC-5, DC-10]
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1,
                 use_ffn: bool = False, norm_position: str = "pre",
                 softmax_mode: str = "standard"):
        super().__init__()
        if norm_position not in NORM_POSITIONS:
            raise ValueError(f"norm_position must be one of {NORM_POSITIONS}, got '{norm_position}'")
        self.norm_position = norm_position
        self.norm_q  = nn.LayerNorm(d)
        # Post-norm normalises *after* the residual, so the K/V stream enters attention
        # raw (whichever layer produced it already normalised it). A norm_kv in post-norm
        # mode would receive no gradient at all, so it is not created. This is why the
        # pre- and post-norm parameter sets differ by exactly these two tensors.
        self.norm_kv = nn.LayerNorm(d) if norm_position == "pre" else None
        self.attn = _MHA(d, n_heads, dropout=dropout, softmax_mode=softmax_mode)
        self.drop = nn.Dropout(dropout)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ff = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)
            )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                record: bool = False) -> torch.Tensor:
        if self.norm_position == "pre":
            kv_n = self.norm_kv(kv)
            q = q + self.drop(self.attn(self.norm_q(q), kv_n, kv_n, record=record))
            if self.use_ffn:
                q = q + self.drop(self.ff(self.norm_ff(q)))
        else:  # post
            q = self.norm_q(q + self.drop(self.attn(q, kv, kv, record=record)))
            if self.use_ffn:
                q = self.norm_ff(q + self.drop(self.ff(q)))
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
                             Must equal n_self_layers when attn_order="interleaved".
        mlp_f_hidden:        Hidden dim of MLP_F that projects to d_attn.
        dropout:             Dropout probability in attention and MLP_F.
        use_ffn:             Add a FFN sublayer after each attention layer.  [DC-5]
        query_label_mode:    How to represent the query's unknown label.  [DC-1]
        use_nested_tensors:  Pack support into a jagged NestedTensor for self-attention.
                             Only active with attn_order="sequential",
                             softmax_mode="standard" and stats off.  [DC-2]
        norm_position:       "pre" | "post" LayerNorm placement.  [DC-3]
        separate_mlp_f:      Separate MLP_S / MLP_Q instead of one shared MLP_F.  [DC-4]
        attn_order:          "sequential" | "interleaved" attention stacking.  [DC-6]
        softmax_mode:        "standard" | "ssmax" | "qassmax".  [DC-10]
        collect_attn_stats:  Record attention entropy / max weight per layer.  [DC-11]
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
        query_label_mode: str = "unknown_token",   # [DC-1]
        use_nested_tensors: bool = False,          # [DC-2]
        norm_position: str = "pre",                # [DC-3]
        separate_mlp_f: bool = True,               # [DC-4]
        attn_order: str = "sequential",            # [DC-6]
        softmax_mode: str = "standard",            # [DC-10]
        collect_attn_stats: bool = False,          # [DC-11]
        share_final_norm: bool = True,             # [DC-12]
        logit_scale: str = "sqrt_d",               # [DC-13]
    ):
        if d_attn % n_heads != 0:
            raise ValueError(f"d_attn ({d_attn}) must be divisible by n_heads ({n_heads})")
        if query_label_mode not in QUERY_LABEL_MODES:
            raise ValueError(
                f"query_label_mode must be one of {QUERY_LABEL_MODES}, got '{query_label_mode}'")
        if norm_position not in NORM_POSITIONS:
            raise ValueError(
                f"norm_position must be one of {NORM_POSITIONS}, got '{norm_position}'")
        if attn_order not in ATTN_ORDERS:
            raise ValueError(f"attn_order must be one of {ATTN_ORDERS}, got '{attn_order}'")
        if softmax_mode not in SOFTMAX_MODES:
            raise ValueError(f"softmax_mode must be one of {SOFTMAX_MODES}, got '{softmax_mode}'")
        if logit_scale not in LOGIT_SCALES:
            raise ValueError(f"logit_scale must be one of {LOGIT_SCALES}, got '{logit_scale}'")
        if attn_order == "interleaved" and n_self_layers != n_cross_layers:
            raise ValueError(
                "attn_order='interleaved' pairs one self-attention layer with one "
                f"cross-attention layer per block, so n_self_layers ({n_self_layers}) "
                f"must equal n_cross_layers ({n_cross_layers})")
        super().__init__()

        self.encoder = encoder
        self._d_label = d_label
        self.query_label_mode   = query_label_mode     # [DC-1]
        self.use_nested_tensors = use_nested_tensors   # [DC-2]
        self.norm_position      = norm_position        # [DC-3]
        self.separate_mlp_f     = separate_mlp_f       # [DC-4]
        self.attn_order         = attn_order           # [DC-6]
        self.softmax_mode       = softmax_mode         # [DC-10]
        self.collect_attn_stats = collect_attn_stats   # [DC-11]

        # Label context tokens  [DC-1]
        self.label_embed = nn.Embedding(2, d_label)   # active=1, inactive=0
        # unknown_label always created for checkpoint compatibility; only used when
        # query_label_mode=="unknown_token".
        self.unknown_label = nn.Parameter(torch.zeros(d_label))

        # MLP_F: (d_enc + d_label) → d_attn   [DC-4]
        def _make_mlp_f():
            return nn.Sequential(
                nn.Linear(encoder_output_dim + d_label, mlp_f_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_f_hidden, d_attn),
            )

        if separate_mlp_f:
            # MLP_S(x_S | y_S) and MLP_Q(x_Q) — consistent with separate attention
            # operators per the supervisor's rule. `mlp_f` is left unset so a shared
            # checkpoint can never be loaded into a separate-MLP model by accident.
            self.mlp_f_sup = _make_mlp_f()
            self.mlp_f_qry = _make_mlp_f()
        else:
            self.mlp_f = _make_mlp_f()

        # Attention stacks  [DC-6]
        self.self_attn_layers = nn.ModuleList([
            _SelfAttnLayer(d_attn, n_heads, dropout=dropout, use_ffn=use_ffn,
                           norm_position=norm_position, softmax_mode=softmax_mode)
            for _ in range(n_self_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttnLayer(d_attn, n_heads, dropout=dropout, use_ffn=use_ffn,
                            norm_position=norm_position, softmax_mode=softmax_mode)
            for _ in range(n_cross_layers)
        ])

        # Final norm before the prototypes / distances are computed.  [DC-12]
        # Shared by default: the PN head measures a distance BETWEEN the query stream
        # and prototypes built from the support stream, so the two must live in one
        # affine frame. Two separate LayerNorms (the original code) give each stream its
        # own learnable gamma/beta, after which "distance" mixes two frames — and the
        # Mahalanobis head is worse still, since it applies the support covariance to
        # query vectors that are no longer in the support's frame.
        self.share_final_norm = share_final_norm
        if share_final_norm:
            self.final_norm = nn.LayerNorm(d_attn)
        else:
            self.final_norm_sup = nn.LayerNorm(d_attn)
            self.final_norm_qry = nn.LayerNorm(d_attn)

        # Logit temperature.  [DC-13]
        # After the final LayerNorm every vector has norm sqrt(d_attn), and the squared
        # Euclidean logit is  2·q·(p_a − p_i) + (|p_i|² − |p_a|²), which therefore scales
        # with d_attn. Measured at initialisation with d_attn=256: mean |logit| is 17.7
        # (pre-norm) and 70.6 (post-norm), i.e. P(active) ≈ 1e-8 to 1e-31 before any
        # training — while the PTN baseline starts at 0.004, a neutral P(active) ≈ 0.5.
        # Comparing the two architectures from such different starting points would
        # confound the comparison, and post-norm sits past the fp16-AMP saturation point.
        # Dividing by sqrt(d_attn) restores a healthy range (0.09 to 4.4).
        self.logit_scale = logit_scale
        if logit_scale == "sqrt_d":
            self._logit_mult = 1.0 / math.sqrt(d_attn)
        elif logit_scale == "none":
            self._logit_mult = 1.0
        else:                                   # "learned"
            self._logit_mult = None
            self.log_temp = nn.Parameter(torch.tensor(math.log(math.sqrt(d_attn))))

    def _scale_logit(self, logit: torch.Tensor) -> torch.Tensor:
        """Temperature on the PN logit.  [DC-13] — see __init__ for why."""
        if self._logit_mult is not None:
            return logit * self._logit_mult
        return logit / self.log_temp.exp().clamp(min=1e-4)

    # ------------------------------------------------------------------
    # Diagnostics  [DC-11]
    # ------------------------------------------------------------------

    def set_collect_attn_stats(self, enabled: bool) -> None:
        """Toggle attention-stat recording. Forces the dense path while enabled."""
        self.collect_attn_stats = enabled

    def pop_attn_stats(self) -> dict[str, dict[str, float]]:
        """Stats from the most recent forward pass, keyed by layer name.

        Returns {} when collection is off or no attention has run yet. Each value has
        `entropy`, `entropy_norm` (entropy / log(n_keys); 1.0 = uniform attention),
        `max_weight`, `uniform_weight` and `n_keys`.
        """
        out: dict[str, dict[str, float]] = {}
        for i, layer in enumerate(self.self_attn_layers):
            s = layer.attn.pop_stats()
            if s is not None:
                out[f"self_{i}"] = s
        for i, layer in enumerate(self.cross_attn_layers):
            s = layer.attn.pop_stats()
            if s is not None:
                out[f"cross_{i}"] = s
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_sup(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_f_sup(x) if self.separate_mlp_f else self.mlp_f(x)

    def _project_qry(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_f_qry(x) if self.separate_mlp_f else self.mlp_f(x)

    def _augment_and_project(
        self,
        sup_enc: torch.Tensor,          # (B, n_sup, d_enc)
        support_labels: torch.Tensor,   # (B, n_sup)  binary float 0/1
        qry_enc: torch.Tensor,          # (B, n_qry, d_enc)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attach label context and project to attention dimension.  [DC-1, DC-4]"""
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

        c_sup = self._project_sup(torch.cat([sup_enc, sup_ctx], dim=-1))   # (B, n_sup, d_attn)
        c_qry = self._project_qry(torch.cat([qry_enc, qry_ctx], dim=-1))   # (B, n_qry, d_attn)
        return c_sup, c_qry

    def _can_use_nested(self) -> bool:
        """Nested support self-attention is only valid on the plain sequential path.

        Interleaving needs a dense c_sup after every block to feed cross-attention,
        scalable softmax needs an explicit logit matrix, and stat recording needs the
        attention weights — none of which jagged layout provides.  [DC-2]
        """
        return (
            self.use_nested_tensors
            and self.attn_order == "sequential"
            and self.softmax_mode == "standard"
            and not self.collect_attn_stats
            and len(self.self_attn_layers) > 0
        )

    def _attend(
        self,
        c_sup: torch.Tensor,  # (B, n_sup, d_attn)
        c_qry: torch.Tensor,  # (B, n_qry, d_attn)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the attention stack.  [DC-2, DC-6, DC-11]"""
        rec = self.collect_attn_stats

        if self.attn_order == "interleaved":
            # (self → cross) per block: the query attends to a support representation
            # that is still being enriched, at every depth.  [DC-6]
            for self_layer, cross_layer in zip(self.self_attn_layers,
                                               self.cross_attn_layers):
                c_sup = self_layer(c_sup, record=rec)
                c_qry = cross_layer(c_qry, c_sup, record=rec)
        else:
            if self._can_use_nested():
                c_sup = self._self_attn_nested(c_sup)
            else:
                for layer in self.self_attn_layers:
                    c_sup = layer(c_sup, record=rec)
            # Cross-attention always uses dense tensors  [DC-2]
            for layer in self.cross_attn_layers:
                c_qry = layer(c_qry, c_sup, record=rec)

        if self.share_final_norm:
            return self.final_norm(c_sup), self.final_norm(c_qry)
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
        """Euclidean prototypical head — vectorised over the batch.  [DC-7]

        Returns the logit (d_inactive - d_active), not a probability. Its sigmoid is the
        softmax over [-d_a, -d_i] that this head used to return, so predictions are
        unchanged — but the un-squashed logit is what keeps the loss differentiable when
        the two distances are far apart. See compute_loss_batched.
        """
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
        logits = self._scale_logit(d_i - d_a)                        # (B, n_qry)  [DC-13]

        # Degenerate episodes get logit 0, i.e. probability 0.5.
        logits = torch.where(valid_mask.unsqueeze(1), logits, torch.zeros_like(logits))
        return logits, valid_mask

    # ------------------------------------------------------------------
    # Public API (matches PrototypicalNetworkClassification)
    # ------------------------------------------------------------------

    def forward(
        self,
        support_input,
        support_labels: torch.Tensor,   # (n_sup,)  binary 0/1
        query_input,
        distance: str = "mahalanobis",
    ) -> torch.Tensor:
        """Single-episode P(active) per query molecule.

        Mirrors `PrototypicalNetworkClassification.forward`, which `evaluate.py` calls
        directly on the non-GNN DrugOOD path (`evaluate.py:241`). Without it, DrugOOD
        evaluation of this model raised "Module is missing the required forward
        function" — the attention model only ever had the batched entry point.

        Args:
            support_input:  Tensor(n_sup, D) or PyG Batch(n_sup graphs)
            support_labels: Tensor(n_sup,)  binary 0/1
            query_input:    Tensor(n_qry, D) or PyG Batch(n_qry graphs)
            distance:       "mahalanobis" (FS-Mol eval protocol) or "euclidean"
        Returns:
            p_active: Tensor(n_qry,)
        """
        sup_enc = self.encoder(support_input)
        qry_enc = self.encoder(query_input)
        return self.predict_from_embeddings(
            sup_enc, support_labels, qry_enc, distance=distance)

    def _forward_logits_batched(
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
            logits:     (B, n_qry)  log-odds of active; 0 for degenerate episodes
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

    def forward_batched(
        self,
        support_input,
        support_labels: torch.Tensor,   # (B, n_sup)
        query_input,
        train_distance: str = "euclidean",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """P(active) per query plus the per-episode validity mask.

        Unchanged in value from before; the probability is now derived from the logit.
        """
        logits, valid_mask = self._forward_logits_batched(
            support_input, support_labels, query_input, train_distance=train_distance)
        return torch.sigmoid(logits), valid_mask

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

        return torch.sigmoid(self._scale_logit(dists[:, 1] - dists[:, 0]))   # [DC-13]

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
        logits, valid_mask = self._forward_logits_batched(
            support_input, support_labels, query_input
        )
        binary_query = (query_labels > 0.5).float()

        if not valid_mask.any():
            return (
                torch.tensor(0.0, device=logits.device, requires_grad=True),
                {"delta_auprc": float("nan")},
            )

        # BCE from logits, not from a clamped probability. The old formulation squashed
        # with a softmax and then took `p.clamp(1e-7, 1-1e-7).log()`; clamp has ZERO
        # gradient outside its bounds, so once the two prototype distances were far
        # apart (a gap of ~20 under fp16 AMP, ~90 in fp32) a *confidently wrong*
        # prediction produced no gradient at all and the model could not recover. That
        # is the mechanism behind the runs stuck at BCE ~ 6.4 with dAUPRC = 0.000
        # (TASKS.md 2.8). Post-norm and interleaved attention both make large distances
        # more likely, so this had to be fixed before they became ablatable options.
        # Mathematically the same loss, autocast-safe, and gradient magnitude 1 when the
        # prediction is confidently wrong.
        l_valid = logits[valid_mask].float()
        b_valid = binary_query[valid_mask].float()
        loss    = F.binary_cross_entropy_with_logits(l_valid, b_valid)
        p_valid = torch.sigmoid(l_valid)

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
