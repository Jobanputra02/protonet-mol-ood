"""
Run baseline_grid.py for all random-training checkpoints (ECFP seeds 0-2, GNN seeds 1-2).
GNN random seed 0 is already done -> skipped.
Results saved to outputs/results/<run_tag>/baseline_grid.csv

Run overnight:  python run_baseline_grid_all_random.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Analysis.model.baseline_grid import run, DEFAULT_HEADS, summarize
from config import CHECKPOINT_DIR, RESULTS_DIR

RUNS = [
    ("ecfp",      "classification", "random", 0),
    ("ecfp",      "classification", "random", 1),
    ("ecfp",      "classification", "random", 2),
    ("fsmol_gnn", "classification", "random", 1),
    ("fsmol_gnn", "classification", "random", 2),
]

SPLITS        = ["random", "scaffold", "size"]
SUPPORT_SIZES = [16, 32, 64, 128, 256, 512]
N_REPEATS     = 5
BASE_SEED     = 42

print("Loading FS-Mol test assays (shared across all runs) ...")
from main import load_fsmol_split
from config import FSMOL_TEST
test_assays = load_fsmol_split(FSMOL_TEST, max_assays=None)
print(f"  {len(test_assays)} assays loaded.\n")

for encoder, model_head, training_split, seed in RUNS:
    run_tag    = f"{encoder}_{model_head}_{training_split}_seed{seed}"
    checkpoint = os.path.join(CHECKPOINT_DIR, f"ptn_{run_tag}.pt")
    out_dir    = os.path.join(RESULTS_DIR, run_tag)
    out_csv    = os.path.join(out_dir, "baseline_grid.csv")

    if os.path.exists(out_csv):
        print(f"\n[SKIP] {run_tag} - already exists at {out_csv}")
        continue

    print(f"\n{'='*60}")
    print(f"Starting: {run_tag}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  output:     {out_csv}")
    print(f"{'='*60}")
    t0 = time.time()

    os.makedirs(out_dir, exist_ok=True)
    df = run(checkpoint, SPLITS, SUPPORT_SIZES, N_REPEATS, None, BASE_SEED, DEFAULT_HEADS,
             test_assays=test_assays)
    df.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    print(f"\nDone: {run_tag} in {elapsed/60:.1f} min -> {out_csv}")
    summarize(df)

print("\n\nAll runs complete.")
