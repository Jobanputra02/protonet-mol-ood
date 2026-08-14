"""
Central path configuration. Edit ENV to switch between local and server; all other
modules import their paths from here rather than hard-coding them.

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

ENV = "local"      # "local" | "server"

if ENV == "local":
    FSMOL_DIR   = "D:/Thesis/PTN/data/fsmol"
    DRUGOOD_DIR = "D:/Thesis/PTN/data/drugood"
    REPO_ROOT   = "D:/Thesis/PTN"
else:
    FSMOL_DIR   = "/home/chjo00006/PTN/data/fsmol"
    DRUGOOD_DIR = "/home/chjo00006/PTN/data/drugood"
    REPO_ROOT   = "/home/chjo00006/PTN"
# FSMOL_DIR   = "/netscratch/jobanputra/PTN/data/fsmol"
# DRUGOOD_DIR = "/netscratch/jobanputra/PTN/data/drugood"
# REPO_ROOT   = "/netscratch/jobanputra/PTN"
FSMOL_TRAIN = os.path.join(FSMOL_DIR, "train")
FSMOL_VAL   = os.path.join(FSMOL_DIR, "valid")
FSMOL_TEST  = os.path.join(FSMOL_DIR, "test")

# Design choice: Butina@0.70 sphere-exclusion (justified by split_ood_characterization —
# NN-Tanimoto ~0.26 with 95.5% assay retention, best OOD/retention trade-off).
# Both training (shift-aware episodes) and evaluation use this cutoff.
SCAFFOLD_OOD_CUTOFF  = 0.70
SCAFFOLD_OOD_VARIANT = "butina_c70"   # filename tag for CSVs

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
