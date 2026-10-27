"""
MIC Episode Loader
==================
Episodic few-shot dataset for MIC CSVs (curated_mic or chembl36_mic).
Mirrors FSMolEpisodeDataset so both PTN and CrossAttentionProtoNet
can train on MIC data by swapping one argument.

Usage — curated_mic
-------------------
store = MicAssayStore("data/custom/curated_mic_assays_balanced.csv")
train_ids, val_ids, _ = load_mic_split("data/custom/mic_splits.json", split="A")

Usage — chembl36_mic
--------------------
store = MicAssayStore(
    "data/custom/chembl36_curated.csv",
    assay_col="group", label_col="target_transformed",
    label_threshold=6.0,        # pMIC >= 6  →  potent (MIC <= 1 µM)
    filter_col="standard_type", filter_val="MIC",
)
train_ids, val_ids, _ = load_mic_split("data/custom/mic_splits_chembl36.json", split="A")

# Same DataLoader call as for FS-Mol:
loader = DataLoader(
    MicEpisodeDataset(store, train_ids, n_support=16, n_query=16),
    batch_size=16, num_workers=4, shuffle=True,
)
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

from data import AssayDataset, EpisodeSampler, mol_to_fingerprint, get_scaffold

MIN_ASSAY_SIZE = 32


# =============================================================================
# ASSAY STORE  — pre-loads all balanced assays from CSV into RAM
# =============================================================================

class MicAssayStore:
    """
    Loads a MIC CSV once at startup and builds an AssayDataset per assay.

    Parameters
    ----------
    csv_path        : path to the CSV file
    min_size        : minimum molecules per assay (default 32)
    assay_col       : column containing assay/group identifier
                      curated_mic → "assay_chembl_id"
                      chembl36_mic → "group"
    smiles_col      : column containing SMILES strings
    label_col       : column containing activity labels
                      curated_mic → "is_potent" (already binary 0/1)
                      chembl36_mic → "target_transformed" (pMIC, continuous)
    label_threshold : if set, binarize label as (label >= threshold)
                      use 6.0 for chembl36 pMIC (MIC <= 1 µM → potent)
    filter_col      : optional pre-filter column (e.g. "standard_type")
    filter_val      : value to keep in filter_col (e.g. "MIC")
    """

    def __init__(
        self,
        csv_path: str,
        min_size: int = MIN_ASSAY_SIZE,
        assay_col: str = "assay_chembl_id",
        smiles_col: str = "canonical_smiles",
        label_col: str = "is_potent",
        label_threshold: Optional[float] = None,
        filter_col: Optional[str] = None,
        filter_val=None,
    ):
        print(f"Loading MIC store from {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        if filter_col is not None:
            df = df[df[filter_col] == filter_val].copy()
            print(f"  After filter ({filter_col}={filter_val!r}): {len(df):,} rows")

        self.assay_datasets: dict[str, AssayDataset] = {}
        n_skipped = 0

        for assay_id, grp in df.groupby(assay_col):
            grp = grp.dropna(subset=[smiles_col, label_col])
            if len(grp) < min_size:
                n_skipped += 1
                continue

            smiles_list = grp[smiles_col].tolist()
            raw_labels  = grp[label_col].astype(float).tolist()

            if label_threshold is not None:
                binary_labels = [int(v >= label_threshold) for v in raw_labels]
            else:
                binary_labels = [int(v) for v in raw_labels]
            labels = [float(b) for b in binary_labels]

            ds = self._build_assay(str(assay_id), smiles_list, labels, binary_labels)
            if len(ds) >= min_size:
                self.assay_datasets[str(assay_id)] = ds
            else:
                n_skipped += 1

        print(f"  Loaded {len(self.assay_datasets):,} assays  "
              f"(skipped {n_skipped} with <{min_size} valid molecules)")

    def _build_assay(self, assay_id, smiles_list, labels, binary_labels) -> AssayDataset:
        """Build AssayDataset bypassing the slow SMILES-only constructor path."""
        fps, valid_smi, valid_lab, valid_bin = [], [], [], []
        for smi, lab, bl in zip(smiles_list, labels, binary_labels):
            fp = mol_to_fingerprint(smi)
            if fp is not None:
                fps.append(fp)
                valid_smi.append(smi)
                valid_lab.append(lab)
                valid_bin.append(int(bl))

        ds = AssayDataset.__new__(AssayDataset)
        ds.assay_id     = str(assay_id)
        ds.fingerprints = fps
        ds.labels       = np.array(valid_lab, dtype=np.float32)
        ds.binary_labels = np.array(valid_bin, dtype=np.int32) if valid_bin else None
        ds.scaffolds    = valid_smi  # raw SMILES (same field name as FS-Mol path)

        ds.scaffold_groups = {}
        for i, smi in enumerate(valid_smi):
            sc = get_scaffold(smi)
            key = sc if sc is not None else "__none__"
            ds.scaffold_groups.setdefault(key, []).append(i)

        ds._validate_scaffold_groups()
        return ds

    def get(self, assay_id: str) -> Optional[AssayDataset]:
        return self.assay_datasets.get(assay_id)

    def __len__(self):
        return len(self.assay_datasets)


# =============================================================================
# SPLIT LOADER
# =============================================================================

def load_mic_split(
    splits_json: str,
    split: str = "A",
) -> tuple[list[str], list[str], list[str]]:
    """
    Load train/val/test assay-ID lists from mic_splits.json.

    Args:
        splits_json: path to mic_splits.json produced by mic_splits.py
        split:       one of "A", "B", "C", "D", "E"

    Returns:
        (train_ids, val_ids, test_ids)
    """
    with open(splits_json) as f:
        data = json.load(f)
    if split not in data:
        raise ValueError(f"Split '{split}' not found. Available: {list(data.keys())}")
    s = data[split]
    return s["train"], s["val"], s["test"]


# =============================================================================
# MIC EPISODE DATASET
# =============================================================================

class MicGraphEpisodeDataset(Dataset):
    """
    Episodic GNN dataset for MIC assays.

    Mirrors FSMolGraphEpisodeDataset: returns (sup_graphs, sup_labels, qry_graphs,
    qry_labels) where sup/qry_graphs are list[PyG Data], ready for graph_episode_collate.
    MIC is always binary, so binary_labels are used unconditionally.
    """

    def __init__(
        self,
        store: MicAssayStore,
        assay_ids: list[str],
        n_episodes_per_epoch: int = 4000,
        n_support: int = 16,
        n_query: int = 16,
        training_split: str = "random",
        **_ignored,  # absorb unused kwargs (e.g. use_binary_labels) from callers
    ):
        if training_split not in ("random", "scaffold", "similarity"):
            raise ValueError(f"training_split must be 'random', 'scaffold', or 'similarity'; got '{training_split}'")

        self.assay_ids = [a for a in assay_ids if store.get(a) is not None]
        missing = len(assay_ids) - len(self.assay_ids)
        if missing:
            print(f"  [MicGraphEpisodeDataset] {missing} assay IDs not in store — skipped")

        self.store          = store
        self.n_episodes     = n_episodes_per_epoch
        self.sampler        = EpisodeSampler(n_support, n_query)
        self.training_split = training_split
        self._setup_graph_builder()

        n_sup_str = str(n_support) if isinstance(n_support, int) else f"{min(n_support)}-{max(n_support)}"
        print(f"  MIC GNN dataset: {len(self.assay_ids)} assays, "
              f"n_support={n_sup_str}, training_split={training_split}")

    def _setup_graph_builder(self):
        import torch
        from torch_geometric.data import Data
        from featurize import smiles_to_fsmol_graph, FSMOL_NODE_FEAT_DIM
        self._smiles_to_graph = smiles_to_fsmol_graph
        self._dummy = Data(
            x=torch.zeros((1, FSMOL_NODE_FEAT_DIM), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 3), dtype=torch.float),
            ecfp=torch.zeros((1, 2048), dtype=torch.float),
            descriptors=torch.zeros((1, 42), dtype=torch.float),
        )

    def _build_graphs(self, smiles_list: list) -> list:
        return [self._smiles_to_graph(smi) or self._dummy for smi in smiles_list]

    def refresh_pool(self, verbose: bool = False) -> None:
        pass

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx):
        import torch as _torch
        for _ in range(200):
            aid = self.assay_ids[np.random.randint(len(self.assay_ids))]
            ds  = self.store.get(aid)
            if ds is None or len(ds) < self.sampler.n_support_min:
                continue
            if self.training_split in ("scaffold", "similarity"):
                sup_idx, qry_idx = self.sampler._get_episode_indices_shift_aware(
                    ds, training_split=self.training_split)
            else:
                sup_idx, qry_idx = self.sampler._get_episode_indices_random(ds)
            sup_graphs = self._build_graphs([ds.scaffolds[i] for i in sup_idx])
            qry_graphs = self._build_graphs([ds.scaffolds[i] for i in qry_idx])
            if ds.binary_labels is not None:
                sup_labels = _torch.tensor(ds.binary_labels[sup_idx].astype(np.float32))
                qry_labels = _torch.tensor(ds.binary_labels[qry_idx].astype(np.float32))
            else:
                sup_labels = _torch.tensor(ds.labels[sup_idx])
                qry_labels = _torch.tensor(ds.labels[qry_idx])
            return sup_graphs, sup_labels, qry_graphs, qry_labels
        raise RuntimeError("Could not sample a valid MIC episode after 200 attempts.")


class MicEpisodeDataset(Dataset):
    """
    Episodic few-shot dataset for MIC assays.

    Interface matches FSMolEpisodeDataset so training scripts need only swap
    the dataset class and pass MIC assay IDs instead of file paths.

    Each __getitem__ picks a random assay from the provided id list and samples
    one episode using the same EpisodeSampler / training_split logic as FS-Mol.

    Args:
        store:               MicAssayStore (shared, created once at startup)
        assay_ids:           list of assay_chembl_id strings for this split
        n_episodes_per_epoch: virtual epoch length (episodes sampled with replacement)
        n_support:           int or list[int] — support set size per episode
        n_query:             query set size per episode
        training_split:      "random" | "scaffold" | "similarity"
        use_binary_labels:   always True for MIC (pre-binarised is_potent)
    """

    def __init__(
        self,
        store: MicAssayStore,
        assay_ids: list[str],
        n_episodes_per_epoch: int = 4000,
        n_support: int = 16,
        n_query: int = 16,
        training_split: str = "random",
        use_binary_labels: bool = True,
    ):
        if training_split not in ("random", "scaffold", "similarity"):
            raise ValueError(f"training_split must be 'random', 'scaffold', or 'similarity'; got '{training_split}'")

        # Filter to assays that exist in the store
        self.assay_ids = [a for a in assay_ids if store.get(a) is not None]
        missing = len(assay_ids) - len(self.assay_ids)
        if missing:
            print(f"  [MicEpisodeDataset] {missing} assay IDs not in store (too small or invalid SMILES) — skipped")

        self.store          = store
        self.n_episodes     = n_episodes_per_epoch
        self.sampler        = EpisodeSampler(n_support, n_query)
        self.training_split = training_split
        self.use_binary     = use_binary_labels

        n_sup_str = str(n_support) if isinstance(n_support, int) else f"{min(n_support)}-{max(n_support)}"
        print(f"  MIC dataset: {len(self.assay_ids)} assays, "
              f"n_support={n_sup_str}, training_split={training_split}")

    def refresh_pool(self, verbose: bool = False) -> None:
        pass  # no-op — assays are always in RAM

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx):
        for _ in range(200):
            aid = self.assay_ids[np.random.randint(len(self.assay_ids))]
            ds  = self.store.get(aid)
            if ds is None or len(ds) < self.sampler.n_support_min:
                continue
            return self.sampler.sample_episode(
                ds, training_split=self.training_split, use_binary=self.use_binary
            )
        raise RuntimeError("Could not sample a valid MIC episode after 200 attempts.")
