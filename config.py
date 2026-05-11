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
    FSMOL_DIR   = "/home/chjo00006/data/fsmol"
    DRUGOOD_DIR = "/home/chjo00006/data/drugood"
    REPO_ROOT   = "/home/chjo00006/prototypical_networks"

# ── Derived paths — do not edit below this line ────────────────────────────────
FSMOL_TRAIN = os.path.join(FSMOL_DIR, "train")
FSMOL_VAL   = os.path.join(FSMOL_DIR, "valid")
FSMOL_TEST  = os.path.join(FSMOL_DIR, "test")

CHECKPOINT_DIR  = os.path.join(REPO_ROOT, "checkpoints")
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "pretrained_model.pt")

OUTPUT_DIR  = os.path.join(REPO_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
