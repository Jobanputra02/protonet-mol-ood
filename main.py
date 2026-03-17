"""
Entry Point
Full pipeline:
1. Load FS-Mol assays (.jsonl.gz files) -> pretrain prototypical network
2. Load DrugOOD JSON splits -> evaluate OOD generalization

Data paths — update these paths:
FS_MOL_DIR  : folder containing train/ valid/ test/ subfolders of .jsonl.gz files
DRUGOOD_DIR : folder containing the pre-built DrugOOD JSON files to evaluate
"""

import json
import gzip
import os
from typing import Optional
import torch
import numpy as np
from data import AssayDataset, DrugOODEvalDataset
from train import pretrain
from evaluate import load_and_evaluate


# Paths to edit
FS_MOL_DIR  = "D:/Thesis/Prototypical Networks/data/fs-mol"
DRUGOOD_DIR = "D:/Thesis/Prototypical Networks/data/drugood"


def load_fsmol_split(split_dir: str, max_assays: Optional[int] = None) -> list[AssayDataset]:
    """
    Load all assays from one FS-Mol directory (train/ valid/ test/)
    max_assays: None = load all
    Returns: List of AssayDataset, one per assay file. Skips empty assays
    """
    files = sorted([
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".jsonl.gz")
    ])

    if max_assays is not None:
        files = files[:max_assays]

    assays = []
    for i, filepath in enumerate(files):
        # if i % 500 == 0:
        #     print(f" Loading assay {i+1}/{len(files)}")

        # .jsonl.gz: each line is one molecule as a JSON object
        # Each file = one assay
        # here shortcut to avoid heavy recompute: it uses precomputed fingerprints from the file directly (no RDKit calls)
        # field names: "SMILES", "fingerprints", "LogRegressionProperty"
        fingerprints = []
        smiles_list  = []
        labels       = []
        assay_id     = os.path.basename(filepath).replace(".jsonl.gz", "")

        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mol = json.loads(line)

                fp    = mol.get("fingerprints", None)
                smi   = mol.get("SMILES", None)
                label = mol.get("LogRegressionProperty", mol.get("RegressionProperty", mol.get("Property", None)))
                # if LogRegressionProperty not found, try other possible keys (some files have inconsistent naming)

                if fp is None or label is None or smi is None:
                    continue
                if len(fp) != 2048:
                    continue
                try:
                    fingerprints.append(np.array(fp, dtype=np.float32))
                    smiles_list.append(smi)
                    labels.append(float(label))
                except (ValueError, TypeError):
                    continue

        # Hard sync lengths (if any conversion failed, truncate to minimum)
        n = min(len(fingerprints), len(labels), len(smiles_list))
        fingerprints = fingerprints[:n]
        labels       = labels[:n]
        smiles_list  = smiles_list[:n]

        # Build AssayDataset
        dataset = AssayDataset.__new__(AssayDataset)
        dataset.assay_id = assay_id
        dataset.fingerprints = fingerprints
        dataset.labels = np.array(labels, dtype=np.float32)
        dataset.scaffolds = smiles_list
        dataset.scaffold_groups = {}

        for i, smi in enumerate(smiles_list):
            key = smi[:10]
            dataset.scaffold_groups.setdefault(key, []).append(i)

        # skip assays too small to form an episode
        if len(dataset) >= 8:
            assays.append(dataset)

    # print(f" Loaded {len(assays)} assays from {split_dir} ")
    return assays


# DrugOOD Loader
#   data["split"]["train"]  training molecules
#   data["split"]["ood_val"]  OOD validation (used as context (support) set)
#   data["split"]["ood_test"]  OOD test (used as query set)
# Each entry has:
#   "smiles" : SMILES string
#   "reg_label" : continuous activity value (pCHEMBL or similar)
#   "assay_id" : ChEMBL assay ID
#   "domain_id" : scaffold/size/assay domain identifier
def load_drugood_split(json_path: str, split_type: str) -> DrugOODEvalDataset:
    """
    Load one DrugOOD JSON file into a DrugOODEvalDataset
    This is zero-shot OOD evaluation:
    the model sees a few OOD-distribution molecules as context,
    then predicts on the OOD test set (no gradient updates)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    split = data["split"]

    def extract(entries: list) -> tuple[list[str], list[float]]:
        smiles, labels = [], []
        for entry in entries:
            smi = entry.get("smiles", None)
            lab = entry.get("reg_label", None)
            if smi and lab is not None:
                try:
                    smiles.append(smi)
                    labels.append(float(lab))
                except (ValueError, TypeError):
                    continue
        return smiles, labels

    context_smiles, context_labels = extract(split["ood_val"])
    test_smiles, test_labels       = extract(split["ood_test"])

    # Subsample context to 64 molecules
    # DrugOOD's ood_val has ~20k entries but prototypical networks are designed for small support sets (16-64)
    # Using all 20k would be inconsistent with the 16-shot pretraining protocol and a 20k x 20k distance matrix
    # Fixed seed 42 ensures reproducibility across runs.
    MAX_CONTEXT = 64
    if len(context_smiles) > MAX_CONTEXT:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(context_smiles), size=MAX_CONTEXT, replace=False)
        context_smiles = [context_smiles[i] for i in idx]
        context_labels = [context_labels[i] for i in idx]

    # print(f"  {split_type}: context={len(context_labels)}, test={len(test_labels)}")

    return DrugOODEvalDataset(
        test_smiles=test_smiles,
        test_labels=test_labels,
        context_smiles=context_smiles,
        context_labels=context_labels,
        split_type=split_type
    )


if __name__ == "__main__":
    # step 1: Load FS-Mol
    # 26k+ train assays, 40 val assays
    # print("Loading FS-Mol datasets")

    train_assays = load_fsmol_split(os.path.join(FS_MOL_DIR, "train"), max_assays=5000) # None to load all 26k+
    val_assays = load_fsmol_split(os.path.join(FS_MOL_DIR, "valid"), max_assays=None) # bcoz valid are just 40 

    # print(f" Train assays (usable): {len(train_assays)}")
    # print(f" Val assays (usable): {len(val_assays)}")

    # step 2: Pretrain on FS-Mol
    # print("Pretraining on FS-Mol")

    model = pretrain(
        train_assays=train_assays,
        val_assays=val_assays,
        n_epochs=50,
        n_support=16,
        n_query=16,
        n_episodes_train=1000,
        n_episodes_val=200,
        lr=1e-3,
        embedding_dim=256,
        hidden_dim=512,
        save_path="pretrained_model.pt",
        shift_aware=True
    )

    # step 3:Evaluate on DrugOOD
    # print("Evaluating on DrugOOD")

    # evaluating all three shift types for lbap_core_ic50: scaffold (structural shift), size (molecular size shift), assay(measurement shift)
    eval_datasets = [
        load_drugood_split(os.path.join(DRUGOOD_DIR, "lbap_core_ic50_scaffold.json"), split_type="lbap_core_ic50_scaffold"),
        load_drugood_split(os.path.join(DRUGOOD_DIR, "lbap_core_ic50_size.json"), split_type="lbap_core_ic50_size"),
        load_drugood_split(os.path.join(DRUGOOD_DIR, "lbap_core_ic50_assay.json"), split_type="lbap_core_ic50_assay"),
        # we can do more here
    ]
    results = load_and_evaluate("pretrained_model.pt", eval_datasets)

    # for split_type, metrics in results.items():
    #     print(
    #         f"{split_type}: "
    #         f"RMSE={metrics['rmse']:.4f}, "
    #         f"MAE={metrics['mae']:.4f}, "
    #         f"Spearman={metrics['spearman']:.4f}"
    #     )
