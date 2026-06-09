"""
config.py — Central path configuration
========================================
Change ENV to switch between local and server environments.
All other scripts import paths from here — never hardcode paths elsewhere.
"""
import os

# ── Environment ────────────────────────────────────────────────────────────────
# Change this one line to switch environments
ENV = "local"   # "local" | "server"

if ENV == "local":
    FSMOL_DIR   = "D:/Thesis/PTN/data/fsmol"
    DRUGOOD_DIR = "D:/Thesis/PTN/data/drugood"
    REPO_ROOT   = "D:/Thesis/PTN"
else:
    FSMOL_DIR   = "/home/chjo00006/PTN/data/fsmol"
    DRUGOOD_DIR = "/home/chjo00006/PTN/data/drugood"
    REPO_ROOT   = "/home/chjo00006/PTN"

# ── Derived paths — do not edit below this line ────────────────────────────────
FSMOL_TRAIN = os.path.join(FSMOL_DIR, "train")
FSMOL_VAL   = os.path.join(FSMOL_DIR, "valid")
FSMOL_TEST  = os.path.join(FSMOL_DIR, "test")

CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")

# Checkpoint file per model combination: ptn_{encoder}_{head}_{split}.pt
# Naming matches the run_tag used for result CSVs and figures.
PTN_ECFP_REGRESSION_SHIFT_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, "ptn_ecfp_regression_shift_aware.pt")
PTN_ECFP_REGRESSION_RANDOM_CHECKPOINT     = os.path.join(CHECKPOINT_DIR, "ptn_ecfp_regression_random.pt")
PTN_ECFP_CLASSIFICATION_SHIFT_CHECKPOINT  = os.path.join(CHECKPOINT_DIR, "ptn_ecfp_classification_shift_aware.pt")
PTN_ECFP_CLASSIFICATION_RANDOM_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ptn_ecfp_classification_random.pt")
PTN_GNN_REGRESSION_SHIFT_CHECKPOINT       = os.path.join(CHECKPOINT_DIR, "ptn_gnn_regression_shift_aware.pt")
PTN_GNN_REGRESSION_RANDOM_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, "ptn_gnn_regression_random.pt")
PTN_GNN_CLASSIFICATION_SHIFT_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "ptn_gnn_classification_shift_aware.pt")
PTN_GNN_CLASSIFICATION_RANDOM_CHECKPOINT  = os.path.join(CHECKPOINT_DIR, "ptn_gnn_classification_random.pt")

# FS-Mol faithful encoder (8-layer GNN + ECFP + descriptors feature fusion)
PTN_FSMOL_GNN_REGRESSION_SHIFT_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, "ptn_fsmol_gnn_regression_shift_aware.pt")
PTN_FSMOL_GNN_REGRESSION_RANDOM_CHECKPOINT     = os.path.join(CHECKPOINT_DIR, "ptn_fsmol_gnn_regression_random.pt")
PTN_FSMOL_GNN_CLASSIFICATION_SHIFT_CHECKPOINT  = os.path.join(CHECKPOINT_DIR, "ptn_fsmol_gnn_classification_shift_aware.pt")
PTN_FSMOL_GNN_CLASSIFICATION_RANDOM_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ptn_fsmol_gnn_classification_random.pt")

OUTPUT_DIR  = os.path.join(REPO_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Per-run subdirectories — one folder per (encoder, head, split) combination.
# Use these instead of RESULTS_DIR / FIGURES_DIR when writing run-specific files.
DATA_ANALYSIS_RESULTS_DIR = os.path.join(RESULTS_DIR, "data_analysis")
DATA_ANALYSIS_FIGURES_DIR = os.path.join(FIGURES_DIR, "data_analysis")
