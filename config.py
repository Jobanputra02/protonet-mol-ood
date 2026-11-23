"""
Central path configuration. Paths are derived from this file's own location, so the
same code runs on any machine without editing; all other modules import their paths
from here rather than hard-coding them.

Environment overrides (set these in the Condor submit file if needed):
    PTN_ROOT          repo root            (default: the directory holding this file)
    PTN_DATA          data directory       (default: <repo>/data)
    PTN_NUM_WORKERS   DataLoader workers   (see train.py; match request_cpus on Condor)

Run tag format  (use make_run_tag() to build):
    {enc}_{head}_{split}_{distance}_{nsup}_seed{seed}

    enc:      "gnn" | "ecfp"   (use exactly these strings — no "fsmol_gnn")
    head:     "classification" | "regression"
    split:    "random" | "scaffold" | "similarity"
    distance: "euclidean" | "mahalanobis"
    nsup:     "16" | "64" | "163264"  (list[int] → digits joined, e.g. [16,32,64] → "163264")
    seed:     "seed0" | "seed1" | "seed2"

    Example: gnn_classification_similarity_mahalanobis_64_seed1

Output layout (per run_tag):
    outputs/<run_tag>/checkpoints/model.pt   trained model
    outputs/<run_tag>/csvs/fsmol_test.csv    FS-Mol test results
    outputs/<run_tag>/csvs/drugood.csv       DrugOOD results
    outputs/baselines/csvs/fsmol_test.csv    model-free ECFP baselines (shared)
    outputs/data_analysis/figures|csvs/      dataset-audit outputs
"""
import os

# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------
# Resolved from this file's own location, so the same code runs unchanged on any
# machine. This replaced a probe for "/netscratch", which was specific to the old
# Pegasus cluster: on conduit2 that path does not exist, so the probe silently fell
# through to the hardcoded Windows paths and every job failed with a confusing
# "D:/Thesis/..." not-found error.
#
# Override with PTN_ROOT when code and data live in different trees (e.g. code in
# $HOME, data on fast scratch). Condor submit files should set it explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("PTN_ROOT", _HERE)

# Data may sit outside the repo (scratch space); PTN_DATA points at the directory
# that holds fsmol/ and drugood/. Defaults to <repo>/data.
DATA_ROOT = os.environ.get("PTN_DATA", os.path.join(REPO_ROOT, "data"))

FSMOL_DIR   = os.path.join(DATA_ROOT, "fsmol")
# FS-Mol ships a LARGER raw pool than the benchmark uses. Its README:
#   "we specify which tasks are to be used with the file datasets/fsmol-0.1.json,
#    a default list of tasks for each data fold. We note that the complete dataset
#    contains many more tasks."
# Verified 2026-11-18: fsmol-0.1.json = 4,938 train tasks; entire_train_set.json =
# 26,868, which is exactly what data/fsmol/train contains. valid (40) and test (157)
# are identical in both. Training on all 26,868 is a legitimate FS-Mol option but the
# numbers are then NOT comparable to published FS-Mol baselines.
FSMOL_TASK_LIST = os.path.join(DATA_ROOT, "fsmol_meta", "fsmol-0.1.json")
DRUGOOD_DIR = os.path.join(DATA_ROOT, "drugood")
CUSTOM_DIR  = os.path.join(DATA_ROOT, "custom")
FSMOL_TRAIN = os.path.join(FSMOL_DIR, "train")
FSMOL_VAL   = os.path.join(FSMOL_DIR, "valid")
FSMOL_TEST  = os.path.join(FSMOL_DIR, "test")

# Design choice: Butina@0.70 sphere-exclusion (justified by split_ood_characterization —
# NN-Tanimoto ~0.26 with 95.5% assay retention, best OOD/retention trade-off).
# Both training (shift-aware episodes) and evaluation use this cutoff.
SCAFFOLD_OOD_CUTOFF  = 0.70
SCAFFOLD_OOD_VARIANT = "butina_c70"   # filename tag for CSVs

# ---------------------------------------------------------------------------
# Support-set class composition  (TASKS.md 5.2 / 5.3)
# ---------------------------------------------------------------------------
# How the active/inactive composition of a support (context) set is chosen.
#
#   "proportional"  keep the assay's natural active fraction, guaranteeing >=1 of the
#                   minority class. This is what episodic TRAINING does
#                   (StratifiedShuffleSplit in EpisodeSampler), what the FS-Mol paper
#                   does, and what MHNfs inherits from it. It is also the only option
#                   that is realistic at deployment: on a new target you cannot build a
#                   50/50 support set without already knowing the labels you are trying
#                   to predict.
#   "balanced"      aim for 50/50, capped by class availability. Defensible on DrugOOD,
#                   whose train pool is 78-94% active, because a proportional context of
#                   16 leaves the inactive prototype built from ~1-3 molecules and the
#                   ranking metrics then measure prototype noise rather than the
#                   representation. Keep it as a reported ABLATION, not the default.
#
# Why this is a single central knob: DrugOOD evaluation previously hardcoded 50/50 while
# training sampled proportionally, and the code comment claimed the two matched. They did
# not, so every DrugOOD number was produced under a support distribution the model had
# never trained on. Train and eval must read the same setting.
#
# Override per run with PTN_SUPPORT_SAMPLING so the ablation needs no code edit.
SUPPORT_SAMPLING = os.environ.get("PTN_SUPPORT_SAMPLING", "proportional")
if SUPPORT_SAMPLING not in ("proportional", "balanced"):
    raise ValueError(
        f"PTN_SUPPORT_SAMPLING must be 'proportional' or 'balanced', "
        f"got {SUPPORT_SAMPLING!r}")

OUTPUT_DIR       = os.path.join(REPO_ROOT, "outputs")
BASELINE_CSV_DIR = os.path.join(OUTPUT_DIR, "baselines", "csvs")

# Dataset-audit scripts (Analysis/data) write here.
DATA_ANALYSIS_FIGURES_DIR = os.path.join(OUTPUT_DIR, "data_analysis", "figures")
DATA_ANALYSIS_RESULTS_DIR = os.path.join(OUTPUT_DIR, "data_analysis", "csvs")


def run_ckpt_dir(run_tag: str) -> str:
    """outputs/<run_tag>/checkpoints - holds model.pt for this run."""
    return os.path.join(OUTPUT_DIR, run_tag, "checkpoints")


def run_csv_dir(run_tag: str) -> str:
    """outputs/<run_tag>/csvs - holds this run's result CSVs."""
    return os.path.join(OUTPUT_DIR, run_tag, "csvs")


def run_fig_dir(run_tag: str) -> str:
    """outputs/<run_tag>/figures - holds this run's figures."""
    return os.path.join(OUTPUT_DIR, run_tag, "figures")


def make_run_tag(encoder: str, head: str, training_split: str,
                 train_distance: str, n_support, seed: int) -> str:
    """
    Build the canonical run identifier used as the output folder name.
    Format: {enc}_{head}_{split}_{distance}_{nsup}_seed{seed}

    encoder:   "gnn" | "ecfp"
    n_support: int -> str(int); list[int] -> digits joined ("163264" for [16,32,64])
    """
    if isinstance(n_support, (list, tuple)):
        nsup = "".join(str(x) for x in sorted(n_support))
    else:
        nsup = str(int(n_support))
    return f"{encoder}_{head}_{training_split}_{train_distance}_{nsup}_seed{seed}"
