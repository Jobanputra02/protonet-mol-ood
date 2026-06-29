"""
TTPA Evaluation - Test-Time Prototype Adaptation
=================================================
Runs evaluate_fsmol_test with use_ttpa=True on all four classification models
and compares against the standard (unweighted) baseline.

WHY:
  Our core finding: scaffold-split ΔAUPRC is flat and low (~0.051) regardless
  of support size, while RF at n=128 hits 0.182.  The prototype (mean of support
  embeddings) is biased toward the context scaffold family and fails to represent
  query scaffold chemistry.

  TTPA fix: reweight each support molecule's contribution to the prototype by its
  mean Tanimoto similarity to the query molecules.  Training-free - runs on all
  existing checkpoints without any retraining.

  If TTPA improves scaffold ΔAUPRC, the thesis story is:
    "found the failure mode → fixed it at test time without retraining"

WHAT THIS SCRIPT DOES:
  For each of the 4 classification models (2 encoders × 2 episode types):
    1. Load best checkpoint (seed-averaged: runs seed0, seed1, seed2 separately)
    2. Run scaffold split with standard ProtoNet   → baseline
    3. Run scaffold split with TTPA               → adapted
    4. Also run random split with both            → sanity check (TTPA should not hurt)
    5. Save results to outputs/results/{model}_seed{N}/fsmol_test_results_ttpa.csv

USAGE (from PTN/ root):
    python Analysis/model/run_ttpa.py
    python Analysis/model/run_ttpa.py --model fsmol_gnn_classification_random --seed 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CHECKPOINT_DIR, RESULTS_DIR, FSMOL_TEST
from evaluate import (
    _load_model_from_checkpoint,
    evaluate_fsmol_test,
)
from main import load_fsmol_split


MODELS = [
    "ecfp_classification_random",
    "ecfp_classification_shift_aware",
    "fsmol_gnn_classification_random",
    "fsmol_gnn_classification_shift_aware",
]
SEEDS       = [0, 1, 2]
SPLIT_TYPES = ["scaffold", "random"]   # scaffold = where TTPA should help; random = sanity check
SUPPORT_SIZES = [16, 32, 64, 128, 256, 512]
N_REPEATS   = 5


def run_model_seed(model_name: str, seed: int, test_assays, device: torch.device):
    """Run both standard and TTPA evaluation for one model/seed combination."""
    run_tag  = f"{model_name}_seed{seed}"
    ckpt     = os.path.join(CHECKPOINT_DIR, f"ptn_{run_tag}.pt")
    out_dir  = os.path.join(RESULTS_DIR, run_tag)

    if not os.path.exists(ckpt):
        print(f"  [SKIP] checkpoint not found: {ckpt}")
        return None, None

    print(f"\n{'='*60}")
    print(f"  {run_tag}")
    print(f"{'='*60}")

    model, _ = _load_model_from_checkpoint(ckpt, device)
    model.eval()

    standard_dfs = []
    ttpa_dfs     = []

    for stype in SPLIT_TYPES:
        print(f"\n  -- split_type={stype} --")

        # Standard ProtoNet
        df_std = evaluate_fsmol_test(
            model, test_assays, device,
            support_sizes=SUPPORT_SIZES,
            n_repeats=N_REPEATS,
            split_type=stype,
            use_ttpa=False,
        )
        df_std["method"] = "standard"
        standard_dfs.append(df_std)

        # TTPA
        df_ttpa = evaluate_fsmol_test(
            model, test_assays, device,
            support_sizes=SUPPORT_SIZES,
            n_repeats=N_REPEATS,
            split_type=stype,
            use_ttpa=True,
        )
        df_ttpa["method"] = "ttpa"
        ttpa_dfs.append(df_ttpa)

    std_df  = pd.concat(standard_dfs,  ignore_index=True)
    ttpa_df = pd.concat(ttpa_dfs,      ignore_index=True)
    combined = pd.concat([std_df, ttpa_df], ignore_index=True)

    out_path = os.path.join(out_dir, "fsmol_test_results_ttpa.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    _print_comparison(std_df, ttpa_df, run_tag)
    return std_df, ttpa_df


def _print_comparison(std_df: pd.DataFrame, ttpa_df: pd.DataFrame, tag: str):
    """Print a side-by-side summary table for scaffold and random splits."""
    print(f"\n{'='*60}")
    print(f"  TTPA vs Standard: {tag}")
    print(f"{'='*60}")
    for stype in SPLIT_TYPES:
        s = std_df[std_df["split_type"] == stype].groupby("support_size")["delta_auprc"].mean()
        t = ttpa_df[ttpa_df["split_type"] == stype].groupby("support_size")["delta_auprc"].mean()
        print(f"\n  [{stype}]")
        print(f"  {'n':>6}  {'standard':>10}  {'ttpa':>10}  {'delta':>10}")
        for n in sorted(s.index):
            sv = s.get(n, float("nan"))
            tv = t.get(n, float("nan"))
            diff = tv - sv
            print(f"  {n:>6}  {sv:>+10.4f}  {tv:>+10.4f}  {diff:>+10.4f}")


def aggregate_seeds(model_name: str):
    """Load all seed CSVs and compute 3-seed averages for standard vs TTPA."""
    all_dfs = []
    for seed in SEEDS:
        run_tag = f"{model_name}_seed{seed}"
        path    = os.path.join(RESULTS_DIR, run_tag, "fsmol_test_results_ttpa.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["seed"] = seed
            all_dfs.append(df)

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    agg = (combined
           .groupby(["method", "split_type", "support_size"])["delta_auprc"]
           .agg(["mean", "std"])
           .round(4))

    print(f"\n{'='*60}")
    print(f"  3-seed aggregate: {model_name}")
    print(f"{'='*60}")
    print(agg.to_string())

    out = os.path.join(RESULTS_DIR, f"{model_name}_ttpa_aggregate.csv")
    agg.reset_index().to_csv(out, index=False)
    print(f"\n  Saved -> {out}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=MODELS,
                        help="Run one model only (default: all 4)")
    parser.add_argument("--seed", type=int, default=None, choices=[0, 1, 2],
                        help="Run one seed only (default: all 3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test assays once - shared across all model/seed runs
    print("\nLoading FS-Mol test assays...")
    test_assays = load_fsmol_split(FSMOL_TEST)
    print(f"  {len(test_assays)} test assays loaded.")

    models_to_run = [args.model] if args.model else MODELS
    seeds_to_run  = [args.seed]  if args.seed is not None else SEEDS

    for model_name in models_to_run:
        for seed in seeds_to_run:
            run_model_seed(model_name, seed, test_assays, device)

        if args.seed is None:
            aggregate_seeds(model_name)

    print("\nDone.")
