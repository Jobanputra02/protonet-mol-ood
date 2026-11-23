"""
Attention concentration vs support-set size  —  measurement M8 in TASKS.md.
===========================================================================

Answers the question from the 2026-10-12 supervision: **does the model fail to attend
to the few important compounds when the support set is large?**

For each support size it reports, per attention layer:

  entropy        mean Shannon entropy of the attention distribution, in nats
  entropy_norm   entropy / log(n_keys).  1.0 = perfectly uniform (attention cannot
                 discriminate at all); 0.0 = all mass on a single key
  max_weight     mean of the largest attention weight per query
  uniform_weight 1/n_keys, i.e. what max_weight would be under uniform attention

How to read the result:

  entropy_norm staying near 1.0 as n_support grows  →  attention is diffuse and the
      concern is real. That is the evidence for trying softmax_mode="ssmax"/"qassmax"
      ([DC-10]) or norm_position="post" ([DC-3]), which is the cheaper lever.
  entropy_norm falling as n_support grows  →  the model already concentrates; a
      scalable softmax is not needed and we have a measurement that says so.

Run an untrained model to see the architecture's bias at initialisation, and a trained
checkpoint to see what training actually learned. Only the trained numbers belong in
the thesis; the untrained ones are a sanity check that the instrument works.

Usage
-----
    # untrained ECFP model, default architecture
    python Analysis/model/attention_stats.py

    # compare softmax modes at initialisation
    python Analysis/model/attention_stats.py --softmax_mode ssmax

    # a trained checkpoint (architecture is read from the checkpoint config)
    python Analysis/model/attention_stats.py \
        --checkpoint outputs/<run_tag>/checkpoints/model.pt

Output: a CSV under outputs/data_analysis/csvs/ plus a printed table.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import DATA_ANALYSIS_RESULTS_DIR                            # noqa: E402
from context_model import (CrossAttentionProtoNet, NORM_POSITIONS,      # noqa: E402
                           ATTN_ORDERS, SOFTMAX_MODES)
from model import ECFPEncoder                                            # noqa: E402

SUPPORT_SIZES = (16, 32, 64, 128, 256, 512)
N_QUERY       = 64
N_EPISODES    = 20          # episodes averaged per support size


def _build_untrained(args, device):
    enc = ECFPEncoder(input_dim=2048, hidden_dim=512, embedding_dim=256)
    model = CrossAttentionProtoNet(
        enc, encoder_output_dim=256,
        d_attn=args.d_attn, n_heads=args.n_heads,
        n_self_layers=args.n_self, n_cross_layers=args.n_cross,
        norm_position=args.norm_position,
        attn_order=args.attn_order,
        softmax_mode=args.softmax_mode,
        separate_mlp_f=args.separate_mlp_f,
        # Nested tensors are incompatible with stat recording (no explicit attention
        # matrix in jagged layout), so the dense path is forced either way.
        use_nested_tensors=False,
        collect_attn_stats=True,
    ).to(device)
    label = (f"untrained/{args.softmax_mode}/{args.norm_position}/{args.attn_order}")
    return model, label


def _load_checkpoint(path, device):
    from evaluate import _load_model_from_checkpoint
    model, config = _load_model_from_checkpoint(path, device)
    if not isinstance(model, CrossAttentionProtoNet):
        raise SystemExit(
            f"{path} holds a '{config.get('model_type')}' model. "
            "Attention stats only apply to context_attention checkpoints.")
    model.set_collect_attn_stats(True)
    model.use_nested_tensors = False      # stat recording needs the dense path
    label = (f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}"
             f"/{config.get('softmax_mode', 'standard')}")
    return model, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="path to a context_attention model.pt; omit for an untrained model")
    p.add_argument("--softmax_mode", default="standard", choices=SOFTMAX_MODES)
    p.add_argument("--norm_position", default="pre", choices=NORM_POSITIONS)
    p.add_argument("--attn_order", default="sequential", choices=ATTN_ORDERS)
    p.add_argument("--separate_mlp_f", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--d_attn", default=256, type=int)
    p.add_argument("--n_heads", default=4, type=int)
    p.add_argument("--n_self", default=1, type=int)
    p.add_argument("--n_cross", default=1, type=int)
    p.add_argument("--n_episodes", default=N_EPISODES, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--out", default=None, help="output CSV path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.checkpoint:
        model, label = _load_checkpoint(args.checkpoint, device)
    else:
        model, label = _build_untrained(args, device)
    model.eval()
    print(f"Model: {label}   device={device}")

    rows = []
    for n_sup in SUPPORT_SIZES:
        for ep in range(args.n_episodes):
            # Random ECFP-shaped inputs. We are measuring the attention mechanism's
            # capacity to concentrate as a function of n_keys, not chemistry, so the
            # input distribution only has to be realistic in shape and sparsity.
            sup = (torch.rand(n_sup, 2048, device=device) < 0.02).float()
            qry = (torch.rand(N_QUERY, 2048, device=device) < 0.02).float()
            ys  = torch.zeros(n_sup, device=device)
            ys[: n_sup // 2] = 1.0
            ys = ys[torch.randperm(n_sup, device=device)]

            with torch.no_grad():
                sup_enc = model.encoder(sup)
                qry_enc = model.encoder(qry)
                model.get_enriched_embeddings(sup_enc, ys, qry_enc)

            for layer_name, st in model.pop_attn_stats().items():
                rows.append({"model": label, "n_support": n_sup, "episode": ep,
                             "layer": layer_name, **st})

    if not rows:
        raise SystemExit("No attention stats recorded — is collect_attn_stats on?")

    df = pd.DataFrame(rows)
    summary = (df.groupby(["layer", "n_support"])
                 [["n_keys", "entropy", "entropy_norm", "max_weight", "uniform_weight"]]
                 .mean().round(4))
    print("\n=== Attention concentration vs support size ===")
    print("entropy_norm: 1.0 = uniform/diffuse, 0.0 = fully concentrated\n")
    print(summary.to_string())

    out = args.out
    if out is None:
        os.makedirs(DATA_ANALYSIS_RESULTS_DIR, exist_ok=True)
        safe = label.replace("/", "-")
        out = os.path.join(DATA_ANALYSIS_RESULTS_DIR, f"attn_stats_{safe}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
