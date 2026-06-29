"""
RF Baseline - Per-Task RandomForest on FS-Mol Test Assays
===========================================================
Trains a RandomForestClassifier on the context set for each FS-Mol test assay
and evaluates on the query set under scaffold-OOD and random splits.

Establishes the per-task supervised learning upper bound:
  what ΔAUPRC is achievable when the classifier is trained fresh on each context set?

RF uses fixed hyperparameters (n_estimators=100, max_depth=10, max_features="sqrt",
min_samples_leaf=2) - good defaults for small support sets. No meta-learning:
the RF is trained fresh on each context set independently.

Protocol matches ProtoNet evaluation exactly:
  - Same 157 FS-Mol test assays, same scaffold/random splits
  - Support sizes: [16, 32, 64, 128, 256, 512], N_REPEATS=5
  - Features: 2048-bit ECFP fingerprints (pre-computed in .jsonl.gz)

Output CSV format matches fsmol_test_results.csv for direct comparison with
ProtoNet results. Comparison figures are saved alongside ProtoNet figures.

Run from PTN/ root:
    python Analysis/model/rf_baseline.py
"""

import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem                            # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold   # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import FSMOL_TEST, RESULTS_DIR, DATA_ANALYSIS_FIGURES_DIR, DATA_ANALYSIS_RESULTS_DIR

os.makedirs(DATA_ANALYSIS_FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_ANALYSIS_RESULTS_DIR, exist_ok=True)

MIN_TASK_SIZE = 32
SUPPORT_SIZES = [16, 32, 64, 128, 256, 512]
N_REPEATS     = 5
SEED          = 42

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=10,
    max_features="sqrt",
    min_samples_leaf=2,
    n_jobs=-1,
)

GNN_SEEDS = [
    os.path.join(RESULTS_DIR, "fsmol_gnn_classification_random_seed0", "fsmol_test_results.csv"),
    os.path.join(RESULTS_DIR, "fsmol_gnn_classification_random_seed1", "fsmol_test_results.csv"),
    os.path.join(RESULTS_DIR, "fsmol_gnn_classification_random_seed2", "fsmol_test_results.csv"),
]
ECFP_SEEDS = [
    os.path.join(RESULTS_DIR, "ecfp_classification_random_seed0", "fsmol_test_results.csv"),
    os.path.join(RESULTS_DIR, "ecfp_classification_random_seed1", "fsmol_test_results.csv"),
    os.path.join(RESULTS_DIR, "ecfp_classification_random_seed2", "fsmol_test_results.csv"),
]


# =============================================================================
# Data loading
# =============================================================================

def _get_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "__none__"
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return sc if sc else "__none__"
    except Exception:
        return "__none__"


def load_assay(filepath: str) -> pd.DataFrame | None:
    rows = []
    with gzip.open(filepath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            mol = json.loads(line)
            if mol.get("Relation", "=") != "=":
                continue
            fp   = mol.get("fingerprints", None)
            smi  = mol.get("SMILES", None)
            prop = mol.get("Property", None)
            if fp is None or smi is None or prop is None or len(fp) != 2048:
                continue
            try:
                label = int(float(prop))
                if label not in (0, 1):
                    continue
            except (ValueError, TypeError):
                continue
            rows.append({"smiles": smi,
                         "fp": np.array(fp, dtype=np.float32),
                         "binary_label": label})

    if len(rows) < MIN_TASK_SIZE:
        return None

    df = pd.DataFrame(rows)
    df.insert(0, "mol_idx", range(len(df)))
    df["scaffold"] = df["smiles"].apply(_get_scaffold)
    return df


# =============================================================================
# Splits
# =============================================================================

def scaffold_split(df: pd.DataFrame, n_support: int,
                   rng: np.random.RandomState) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    groups = df.groupby("scaffold")["mol_idx"].apply(list).to_dict()
    keys   = list(groups.keys())
    rng.shuffle(keys)
    ctx: set[int] = set()
    qry: set[int] = set()
    for k in keys:
        if len(ctx) < n_support:
            ctx.update(groups[k])
        else:
            qry.update(groups[k])
    if len(ctx) < n_support or len(qry) < 8:
        return None
    context = df[df["mol_idx"].isin(ctx)].reset_index(drop=True)
    query   = df[df["mol_idx"].isin(qry)].reset_index(drop=True)
    return context, query


def random_split(df: pd.DataFrame, n_support: int,
                 rng: np.random.RandomState) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = rng.permutation(len(df))
    context = df.iloc[idx[:n_support]].reset_index(drop=True)
    query   = df.iloc[idx[n_support:]].reset_index(drop=True)
    return context, query


# =============================================================================
# RF evaluation
# =============================================================================

def delta_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score) - float(np.mean(y_true))


def rf_eval_assay(df: pd.DataFrame, n_support: int, split_type: str) -> float | None:
    if len(df) < n_support + 8:
        return None

    results = []
    for rep in range(N_REPEATS):
        rng = np.random.RandomState(SEED + rep)
        if split_type == "scaffold":
            out = scaffold_split(df, n_support, rng)
            if out is None:
                continue
            context, query = out
        else:
            context, query = random_split(df, n_support, rng)

        X_c = np.stack(context["fp"].values)
        y_c = context["binary_label"].values
        X_q = np.stack(query["fp"].values)
        y_q = query["binary_label"].values

        if len(np.unique(y_c)) < 2 or len(np.unique(y_q)) < 2:
            continue

        rf = RandomForestClassifier(random_state=SEED + rep, **RF_PARAMS)
        rf.fit(X_c, y_c)
        scores = rf.predict_proba(X_q)[:, 1]
        da = delta_auprc(y_q, scores)
        if not np.isnan(da):
            results.append(da)

    return float(np.mean(results)) if results else None


# =============================================================================
# Main loop
# =============================================================================

def run_rf_baseline() -> pd.DataFrame:
    files = sorted(
        os.path.join(FSMOL_TEST, f)
        for f in os.listdir(FSMOL_TEST)
        if f.endswith(".jsonl.gz")
    )
    total = len(files)
    print(f"Test assays: {total}")
    print(f"Support sizes: {SUPPORT_SIZES}  |  Repeats: {N_REPEATS}  |  RF params: {RF_PARAMS}")

    records = []
    for i, fpath in enumerate(files):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} ...", flush=True)
        df = load_assay(fpath)
        if df is None:
            continue
        assay_id = os.path.basename(fpath).replace(".jsonl.gz", "")
        for stype in ["scaffold", "random"]:
            for n_sup in SUPPORT_SIZES:
                da = rf_eval_assay(df, n_sup, stype)
                if da is None:
                    continue
                records.append({
                    "assay_id":     assay_id,
                    "split_type":   stype,
                    "support_size": n_sup,
                    "n_total":      len(df),
                    "delta_auprc":  round(da, 6),
                    "model":        "RF",
                })

    print(f"\nDone. {len(records)} records.")
    return pd.DataFrame(records)


# =============================================================================
# ProtoNet loader + figures
# =============================================================================

def load_protonet(seed_paths: list[str], split_type: str,
                  model_label: str) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in seed_paths if os.path.exists(p)]
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    sub = (combined[combined["split_type"] == split_type]
           .groupby(["assay_id", "support_size"])["delta_auprc"]
           .mean().reset_index())
    sub["model"] = model_label
    sub["split_type"] = split_type
    return sub


def plot_comparison(rf_df: pd.DataFrame, split_type: str) -> None:
    gnn  = load_protonet(GNN_SEEDS,  split_type, "GNN ProtoNet")
    ecfp = load_protonet(ECFP_SEEDS, split_type, "ECFP ProtoNet")
    rf   = rf_df[rf_df["split_type"] == split_type].copy()
    rf["model"] = "RF baseline"

    all_data = pd.concat([rf, gnn, ecfp], ignore_index=True)
    colors  = {"RF baseline": "gray", "GNN ProtoNet": "steelblue", "ECFP ProtoNet": "tomato"}
    markers = {"RF baseline": "s",    "GNN ProtoNet": "o",          "ECFP ProtoNet": "^"}
    n_per_size = rf.groupby("support_size")["assay_id"].nunique()

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, grp in all_data.groupby("model"):
        agg = grp.groupby("support_size")["delta_auprc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["support_size"], agg["mean"], yerr=agg["std"],
            marker=markers.get(model, "o"), linewidth=2,
            capsize=4, capthick=1.5,
            color=colors.get(model, "black"),
            ecolor=colors.get(model, "black"),
            elinewidth=1, label=model,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    sizes = sorted(all_data["support_size"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}\n(N={n_per_size.get(s,'?')})" for s in sizes], fontsize=8)
    ax.set_xlabel("Support set size", fontsize=11)
    ax.set_ylabel("Mean ΔAUPRC", fontsize=11)
    ax.set_title(f"FS-Mol Test - {split_type.capitalize()} Split: RF vs ProtoNet", fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(DATA_ANALYSIS_FIGURES_DIR, f"fig_rf_vs_protonet_{split_type}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure -> {out}")


def print_summary(rf_df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("RF BASELINE -- Mean dAUPRC per split x support size")
    print("="*60)
    summary = (rf_df.groupby(["split_type", "support_size"])
               .agg(mean_dauprc=("delta_auprc", "mean"),
                    n_assays=("assay_id", "nunique"))
               .round(4))
    print(summary.to_string())

    print("\n" + "="*60)
    print("COMPARISON: RF vs ProtoNet (scaffold split)")
    print("="*60)
    rf_sc = rf_df[rf_df["split_type"] == "scaffold"]
    gnn   = load_protonet(GNN_SEEDS,  "scaffold", "GNN ProtoNet")
    ecfp  = load_protonet(ECFP_SEEDS, "scaffold", "ECFP ProtoNet")

    for model_name, df in [("RF baseline", rf_sc),
                            ("GNN ProtoNet", gnn),
                            ("ECFP ProtoNet", ecfp)]:
        if df.empty:
            continue
        row = df.groupby("support_size")["delta_auprc"].mean().round(4).to_dict()
        vals = "  ".join(f"n={k}: {v:.4f}" for k, v in sorted(row.items()))
        print(f"  {model_name:20s} | {vals}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    rf_df = run_rf_baseline()

    out_csv = os.path.join(DATA_ANALYSIS_RESULTS_DIR, "rf_baseline_results.csv")
    rf_df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    print_summary(rf_df)
    plot_comparison(rf_df, "scaffold")
    plot_comparison(rf_df, "random")

    print("\nDone.")
