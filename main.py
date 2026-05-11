#!/usr/bin/env python
"""
Main Entry Point
================
Full pipeline:
1. Load FS-Mol assays (.jsonl.gz files) → pretrain prototypical network
2. Load DrugOOD JSON splits → evaluate OOD generalization

All paths are centralised in config.py — edit ENV there to switch environments.
"""
import json
import gzip
import os
from typing import Optional
import numpy as np
from config import (FSMOL_TRAIN, FSMOL_VAL, FSMOL_TEST,
                    DRUGOOD_DIR, MODEL_SAVE_PATH, CHECKPOINT_DIR, RESULTS_DIR)
from data import AssayDataset, DrugOODEvalDataset, get_scaffold
from train import pretrain
from evaluate import load_and_evaluate, evaluate_inside_task_ood, evaluate_fsmol_test


# =============================================================================
# FS-MOL LOADER
# =============================================================================
# FS-Mol files are .jsonl.gz: each line is one molecule as a JSON object.
# Confirmed fields from the dataset:
#   "SMILES"                : SMILES string
#   "fingerprints"          : precomputed 2048-bit ECFP list of 0/1 ints
#   "LogRegressionProperty" : log-transformed activity value (used as label)
#   "Relation"              : measurement relation — "=", ">", "<", "~"
#                             Only "=" (exact) measurements are kept.

# Minimum task size after exact-label filtering — supervisor requirement.
MIN_TASK_SIZE = 32


def load_fsmol_assay(filepath: str) -> tuple[AssayDataset, dict]:
    """
    Load one FS-Mol assay file (.jsonl.gz) into an AssayDataset.

    Only keeps compounds where Relation == "=" (exact measurements).
    Inexact compounds (>, <, ~) are dropped per supervisor requirement.

    Returns:
        (dataset, stats) where stats = {
            "n_total":   total molecules in file,
            "n_inexact": dropped due to Relation != "=",
            "n_invalid": dropped due to missing/bad fields,
            "n_kept":    molecules in returned dataset,
        }
    """
    fingerprints   = []
    smiles_list    = []
    labels         = []
    binary_labels  = []
    assay_id       = os.path.basename(filepath).replace(".jsonl.gz", "")

    n_total   = 0
    n_inexact = 0
    n_invalid = 0

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mol = json.loads(line)
            n_total += 1

            # Drop inexact measurements — only keep Relation == "="
            relation = mol.get("Relation", "=")
            if relation != "=":
                n_inexact += 1
                continue

            fp         = mol.get("fingerprints", None)
            smi        = mol.get("SMILES", None)
            label      = mol.get(
                "LogRegressionProperty",
                mol.get("RegressionProperty", mol.get("Property", None))
            )
            # Binary label stored as "Property": "0.0" or "1.0" (string, not bool_label).
            _prop = mol.get("Property", None)
            bool_label = int(float(_prop)) if _prop is not None else None

            if fp is None or label is None or smi is None or len(fp) != 2048:
                n_invalid += 1
                continue
            try:
                fingerprints.append(np.array(fp, dtype=np.float32))
                smiles_list.append(smi)
                labels.append(float(label))
                binary_labels.append(int(bool_label) if bool_label is not None else -1)
            except (ValueError, TypeError):
                n_invalid += 1
                continue

    # Hard-sync lengths — if any conversion failed silently, truncate to minimum
    n = min(len(fingerprints), len(labels), len(smiles_list), len(binary_labels))
    fingerprints  = fingerprints[:n]
    labels        = labels[:n]
    smiles_list   = smiles_list[:n]
    binary_labels = binary_labels[:n]

    # Build AssayDataset bypassing RDKit fingerprint computation
    # by injecting precomputed fingerprints directly after construction.
    dataset = AssayDataset.__new__(AssayDataset)
    dataset.assay_id     = assay_id
    # Keep as list — do NOT call np.array() here (causes OOM across 26k assays)
    dataset.fingerprints = fingerprints
    dataset.labels       = np.array(labels, dtype=np.float32)
    dataset.scaffolds    = smiles_list

    # Scaffold groups built from the SAME filtered list used to populate
    # fingerprints/labels — indices are guaranteed to match.
    dataset.scaffold_groups = {}
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        key = scaffold if scaffold is not None else "__none__"
        dataset.scaffold_groups.setdefault(key, []).append(i)

    bl_arr = np.array(binary_labels, dtype=np.int32)
    dataset.binary_labels = None if (bl_arr == -1).all() else bl_arr

    dataset._validate_scaffold_groups()

    stats = {
        "n_total":   n_total,
        "n_inexact": n_inexact,
        "n_invalid": n_invalid,
        "n_kept":    n,
    }
    return dataset, stats


def load_fsmol_split(split_dir: str, max_assays: Optional[int] = None) -> list[AssayDataset]:
    """
    Load all assays from one FS-Mol split directory (train/ valid/ test/).
    Filters inexact measurements per assay, then drops tasks with <MIN_TASK_SIZE compounds.
    Prints a full data-loss report at the end.

    Args:
        split_dir:  Path to the split folder (contains .jsonl.gz files)
        max_assays: Optional cap — useful for quick tests. None = load all.

    Returns:
        List of AssayDataset, one per assay file (only tasks with >= MIN_TASK_SIZE compounds).
    """
    files = sorted([
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".jsonl.gz")
    ])

    if max_assays is not None:
        files = files[:max_assays]

    assays = []

    # Aggregate data-loss counters across all assay files
    total_mols           = 0
    total_inexact        = 0
    total_invalid        = 0
    mols_in_kept_tasks   = 0   # exact+valid molecules in tasks that pass size filter
    mols_in_dropped_tasks = 0  # exact+valid molecules lost because their task was too small
    n_tasks_raw          = len(files)
    n_tasks_dropped      = 0

    for i, filepath in enumerate(files):
        if i % 500 == 0:
            print(f"  Loading assay {i+1}/{len(files)}...", end="\r")
        dataset, stats = load_fsmol_assay(filepath)

        total_mols    += stats["n_total"]
        total_inexact += stats["n_inexact"]
        total_invalid += stats["n_invalid"]

        if len(dataset) >= MIN_TASK_SIZE:
            assays.append(dataset)
            mols_in_kept_tasks += stats["n_kept"]
        else:
            n_tasks_dropped      += 1
            mols_in_dropped_tasks += stats["n_kept"]

    total_lost = total_inexact + total_invalid + mols_in_dropped_tasks

    print(f"\n{'='*60}")
    print(f"  Data-loss report: {os.path.basename(split_dir)}/")
    print(f"{'='*60}")
    print(f"  Molecules in raw files        : {total_mols:>10,}")
    print(f"  -- Dropped (inexact Relation) : {total_inexact:>10,}  "
          f"({100*total_inexact/max(total_mols,1):.1f}%)")
    print(f"  -- Dropped (bad/missing fields): {total_invalid:>9,}  "
          f"({100*total_invalid/max(total_mols,1):.1f}%)")
    print(f"  -- Dropped (task too small)   : {mols_in_dropped_tasks:>10,}  "
          f"({100*mols_in_dropped_tasks/max(total_mols,1):.1f}%)")
    print(f"  ----------------------------------------")
    print(f"  Total molecules lost          : {total_lost:>10,}  "
          f"({100*total_lost/max(total_mols,1):.1f}%)")
    print(f"  Molecules used for training   : {mols_in_kept_tasks:>10,}  "
          f"({100*mols_in_kept_tasks/max(total_mols,1):.1f}%)")
    print(f"  ---")
    print(f"  Tasks in directory            : {n_tasks_raw:>10,}")
    print(f"  Tasks dropped (<{MIN_TASK_SIZE} molecules)  : {n_tasks_dropped:>10,}  "
          f"({100*n_tasks_dropped/max(n_tasks_raw,1):.1f}%)")
    print(f"  Tasks kept                    : {len(assays):>10,}  "
          f"({100*len(assays)/max(n_tasks_raw,1):.1f}%)")
    print(f"{'='*60}")

    return assays


def get_ram_usage():
    import psutil
    return psutil.Process().memory_info().rss / 1e9


# =============================================================================
# DRUGOOD LOADER
# =============================================================================
# Confirmed JSON structure (from inspection):
#   data["split"]["train"]    — training molecules
#   data["split"]["ood_val"]  — OOD validation
#   data["split"]["ood_test"] — OOD test       → use as query set
#   data["split"]["iid_val"]  — IID validation
#   data["split"]["iid_test"] — IID test
#
# Each entry has:
#   "smiles"    : SMILES string
#   "reg_label" : continuous activity value (pCHEMBL or similar)
#   "cls_label" : binary label (not used — we do regression)
#   "assay_id"  : ChEMBL assay ID
#   "domain_id" : scaffold/size/assay domain identifier

def load_drugood_split(json_path: str, split_type: str) -> DrugOODEvalDataset:
    """
    Load one DrugOOD JSON file into a DrugOODEvalDataset.

    Context set = sampled from train split (in-distribution labeled molecules)
    Query set   = ood_test (OOD molecules to predict)

    Context is sampled from train — not ood_val — so the model sees in-distribution
    support and must generalize to OOD queries. This is the correct zero-shot OOD protocol.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    split = data["split"]

    def extract(entries: list) -> tuple[list[str], list[float], list[int]]:
        smiles, labels, binary = [], [], []
        for entry in entries:
            smi     = entry.get("smiles", None)
            lab     = entry.get("reg_label", None)
            cls_lab = entry.get("cls_label", None)
            if smi and lab is not None:
                try:
                    smiles.append(smi)
                    labels.append(float(lab))
                    binary.append(int(cls_lab) if cls_lab is not None else 0)
                except (ValueError, TypeError):
                    continue
        return smiles, labels, binary

    context_smiles, context_labels, context_binary     = extract(split["train"])
    ood_test_smiles, ood_test_labels, ood_test_binary  = extract(split["ood_test"])
    iid_test_smiles, iid_test_labels, iid_test_binary  = extract(split.get("iid_test", []))

    print(f"  {split_type}: train pool={len(context_labels)}, "
          f"ood_test={len(ood_test_labels)}, iid_test={len(iid_test_labels)}")

    return DrugOODEvalDataset(
        context_smiles=context_smiles,
        context_labels=context_labels,
        ood_test_smiles=ood_test_smiles,
        ood_test_labels=ood_test_labels,
        iid_test_smiles=iid_test_smiles,
        iid_test_labels=iid_test_labels,
        split_type=split_type,
        context_binary_labels=context_binary,
        ood_test_binary_labels=ood_test_binary,
        iid_test_binary_labels=iid_test_binary,
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":

    import psutil
    import pandas as pd

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def ram_gb() -> float:
        return psutil.Process().memory_info().rss / 1e9

    def get_assay_files(split_dir: str) -> list[str]:
        return sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".jsonl.gz")
        ])

    print("=" * 60)
    print("Prototypical Network: Molecular OOD Regression")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Index FS-Mol files (no loading — pretrain streams from disk)
    # ------------------------------------------------------------------
    print("\n[1/4] Indexing FS-Mol files...")
    train_files = get_assay_files(FSMOL_TRAIN)
    val_files   = get_assay_files(FSMOL_VAL)
    test_files  = get_assay_files(FSMOL_TEST)
    print(f"  Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")
    print(f"  RAM: {ram_gb():.1f} GB")

    # ------------------------------------------------------------------
    # STEP 2: Pretrain on FS-Mol (episodic, streaming from disk)
    # ------------------------------------------------------------------
    print("\n[2/4] Pretraining on FS-Mol...")
    model = pretrain(
        train_assays=train_files,
        val_assays=val_files,
        n_epochs=100,
        n_support=16,
        n_query=16,
        n_episodes_train=1000,
        n_episodes_val=200,
        lr=1e-3,
        embedding_dim=256,
        hidden_dim=512,
        save_path=MODEL_SAVE_PATH,
        shift_aware=True,
    )
    print(f"  RAM after pretrain: {ram_gb():.1f} GB")

    device = next(model.parameters()).device
    model.eval()

    # ------------------------------------------------------------------
    # STEP 3: DrugOOD evaluation — zero-shot OOD with context from train
    # Three IC50 shift types: scaffold / size / assay
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating on DrugOOD...")
    eval_datasets = [
        load_drugood_split(
            os.path.join(DRUGOOD_DIR, "lbap_core_ic50_scaffold.json"),
            split_type="lbap_core_ic50_scaffold",
        ),
        load_drugood_split(
            os.path.join(DRUGOOD_DIR, "lbap_core_ic50_size.json"),
            split_type="lbap_core_ic50_size",
        ),
        load_drugood_split(
            os.path.join(DRUGOOD_DIR, "lbap_core_ic50_assay.json"),
            split_type="lbap_core_ic50_assay",
        ),
    ]
    print(f"  RAM after DrugOOD load: {ram_gb():.1f} GB")

    drugood_df = load_and_evaluate(
        MODEL_SAVE_PATH,
        eval_datasets,
        context_sizes=[16, 32, 64, 128, 256, 512],
    )
    print("\n=== DrugOOD results (long-form) ===")
    print(drugood_df.to_string())
    print("\n=== Summary: mean ΔAUPRC per split × query_set ===")
    print(drugood_df.groupby(["split_type", "query_set"])["delta_auprc"].mean().round(4))

    # ------------------------------------------------------------------
    # STEP 4: FS-Mol test evaluation
    #   4a — inside-task OOD at fixed n_support=16 (scaffold split)
    #   4b — support-size sweep, 3 split types → 3 curves for Fig 2a
    # ------------------------------------------------------------------
    print("\n[4/4] FS-Mol test evaluation...")

    # 4a: Inside-task OOD
    print("\n  [4a] Inside-task OOD (scaffold split, n_support=16)...")
    test_assays = load_fsmol_split(FSMOL_TEST, max_assays=None)
    print(f"  Test assays loaded: {len(test_assays)}")

    inside_task_df = evaluate_inside_task_ood(
        model, test_assays, device,
        n_support=16, n_episodes_per_assay=10,
    )
    inside_task_path = os.path.join(RESULTS_DIR, "inside_task_ood_results.csv")
    inside_task_df.to_csv(inside_task_path, index=False)
    print(f"  Saved → {inside_task_path}")

    # 4b: Support-size sweep — 3 split types → 3 curves in Fig 2a
    print("\n  [4b] Support-size sweep (random / scaffold / size splits)...")
    fsmol_dfs = []
    for stype in ["random", "scaffold", "size"]:
        print(f"\n  -- split_type = {stype} --")
        preds_path = os.path.join(RESULTS_DIR, f"fsmol_test_predictions_{stype}.csv")
        df = evaluate_fsmol_test(
            model, test_files, device,
            support_sizes=[16, 32, 64, 128, 256, 512],
            n_repeats=5,
            split_type=stype,
            save_preds_path=preds_path,
        )
        fsmol_dfs.append(df)

    fsmol_test_df = pd.concat(fsmol_dfs, ignore_index=True)
    fsmol_test_path = os.path.join(RESULTS_DIR, "fsmol_test_results.csv")
    fsmol_test_df.to_csv(fsmol_test_path, index=False)
    print(f"\n  All 3 curves saved → {fsmol_test_path}")

    print("\n=== FS-Mol Test: mean ΔAUPRC per split_type × support_size ===")
    print(fsmol_test_df.groupby(["split_type", "support_size"])["delta_auprc"].mean().round(4))
