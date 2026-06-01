"""
Episode Construction for Prototypical Network Training
=======================================================
An "episode" is one few-shot task:
    - Support set: N molecules with known labels (the context)
    - Query set:   M molecules to predict

Two episode construction strategies:
    1. RANDOM:        Support and query sampled randomly from assay.
                      Standard FS-Mol protocol.
    2. SHIFT-AWARE:   Support from one scaffold family, query from another.
                      Forces OOD-robust embedding. USE THIS if prof confirms.

CHOSEN: shift-aware episodes (with random as fallback if scaffold info unavailable).
"""

import gzip
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional

# RDKit has incomplete type stubs — Pylance reports false positives on its
# attributes (MolFromSmiles, GetMorganFingerprintAsBitVect, etc.).
# These are valid at runtime; the "# type: ignore" comments suppress the
# Pylance errors without affecting execution.
from rdkit import Chem, RDLogger  # type: ignore
from rdkit.Chem import AllChem    # type: ignore
RDLogger.DisableLog('rdApp.*')    # suppress valence / sanitization warnings
from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

# RDKit 2022+ exposes a new generator API for fingerprints.
# We use it here to avoid the deprecation warning from GetMorganFingerprintAsBitVect.
# ALTERNATIVE (old API, still works but warns):
# AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore
_MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=2048)  # type: ignore


# =============================================================================
# FINGERPRINT UTILITY
# =============================================================================

def mol_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """
    Convert SMILES string to ECFP4 count fingerprint (2048 bits).

    Returns a COUNT vector (values >= 0, not binary) — each position counts
    how many times a circular substructure of radius ≤ 2 appears.
    This matches the precomputed "fingerprints" field in FS-Mol .jsonl.gz files.
    ECFP4 = radius 2, 2048 bits. Set on the module-level _MORGAN_GENERATOR.

    ALTERNATIVE (MACCS keys, 167 bits):
    # from rdkit.Chem import MACCSkeys  # type: ignore
    # fp = MACCSkeys.GenMACCSKeys(mol)
    # return np.array(fp, dtype=np.float32)

    Returns None if SMILES is invalid (filtered out in AssayDataset).
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    fp = _MORGAN_GENERATOR.GetFingerprintAsNumPy(mol)  # type: ignore[attr-defined]
    return fp.astype(np.float32)


def get_scaffold(smiles: str) -> Optional[str]:
    """
    Extract Bemis-Murcko scaffold from a SMILES string.
    Scaffold = core ring system + linkers, no side chains.
    Used to group molecules by structural family for shift-aware episodes.
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception:
        return None


# =============================================================================
# BASE ASSAY DATASET
# =============================================================================

class AssayDataset:
    """
    Holds all molecules + labels for a single assay.
    Optionally groups molecules by scaffold for shift-aware episode construction.
    """

    def __init__(
        self,
        smiles_list: list[str],
        labels: list[float],
        assay_id: str = "",
        binary_labels: Optional[list[int]] = None
    ):
        self.assay_id = assay_id
        self.fingerprints = []
        self.labels = []
        self.scaffolds = []
        _binary = []

        for i, (smi, lab) in enumerate(zip(smiles_list, labels)):
            fp = mol_to_fingerprint(smi)
            if fp is not None:
                self.fingerprints.append(fp)
                self.labels.append(lab)
                self.scaffolds.append(get_scaffold(smi))
                if binary_labels is not None:
                    _binary.append(binary_labels[i])

        # Keep fingerprints as a plain Python list of 1D arrays.
        # Do NOT convert to a single large numpy array here — with 26k assays
        # loaded simultaneously this causes an OOM error.
        # Conversion to numpy happens lazily in _indices_to_tensors()
        # where we only ever need 16-32 rows at a time.
        self.labels = np.array(self.labels, dtype=np.float32)  # (N,)
        self.binary_labels = np.array(_binary, dtype=np.int32) if _binary else None

        # Group indices by scaffold
        self.scaffold_groups = {}
        for i, sc in enumerate(self.scaffolds):
            key = sc if sc is not None else "__none__"
            self.scaffold_groups.setdefault(key, []).append(i)

        self._validate_scaffold_groups()

    def _validate_scaffold_groups(self):
        """
        Remove any indices from scaffold_groups that are out of bounds.
        Also removes groups that become empty after filtering.
        This guards against index mismatches when datasets are built via
        __new__ bypass (FS-Mol fast loader) where groups are built manually.
        """
        n = len(self)   # uses __len__ which takes min of fingerprints and labels
        cleaned = {}
        for key, indices in self.scaffold_groups.items():
            valid = [i for i in indices if i < n]
            if valid:
                cleaned[key] = valid
        self.scaffold_groups = cleaned

    def __len__(self):
        # Guard against fingerprints and labels being out of sync.
        # Always use the smaller of the two so indices are always valid.
        return min(len(self.fingerprints), len(self.labels))


# =============================================================================
# EPISODE SAMPLER
# =============================================================================

class EpisodeSampler:
    """
    Samples one episode (support + query) from an AssayDataset.

    CHOSEN: shift-aware episodes when possible (at least 2 scaffold groups).
    Falls back to random if the assay has only one scaffold group.

    ALTERNATIVE (always random):
    # Use sample_random_episode() exclusively.
    # Simpler but does not train for OOD robustness.
    """

    def __init__(self, n_support: int = 16, n_query: int = 16):
        """
        Args:
            n_support: Number of molecules in the support (context) set per episode.
            n_query:   Number of molecules in the query set per episode.
        """
        self.n_support = n_support
        self.n_query = n_query

    def sample_episode(self, dataset: AssayDataset, shift_aware: bool = True):
        """
        Sample one episode from a dataset.

        Args:
            dataset:     AssayDataset to sample from
            shift_aware: If True, try to split support/query by scaffold family.
                         Falls back to random if not enough scaffold diversity.

        Returns:
            support_fp:     (n_support, 2048) float tensor
            support_labels: (n_support,) float tensor
            query_fp:       (n_query, 2048) float tensor
            query_labels:   (n_query,) float tensor
        """
        if shift_aware and len(dataset.scaffold_groups) >= 2:
            return self._sample_shift_aware_episode(dataset)
        else:
            return self._sample_random_episode(dataset)

    def _sample_shift_aware_episode(self, dataset: AssayDataset):
        """
        Support and query come from DIFFERENT scaffold families.
        Uses whatever molecules are available — no replacement, capped to pool size.
        Falls back to random if fewer than 2 usable scaffold groups.
        """
        min_group = max(2, self.n_support // 4)
        usable_keys = [k for k, v in dataset.scaffold_groups.items() if len(v) >= min_group]

        if len(usable_keys) < 2:
            return self._sample_random_episode(dataset)

        chosen = np.random.choice(len(usable_keys), size=2, replace=False)
        support_pool = dataset.scaffold_groups[usable_keys[chosen[0]]]
        query_pool   = dataset.scaffold_groups[usable_keys[chosen[1]]]

        # Cap to available pool size — no replacement, no duplication
        n_sup = min(self.n_support, len(support_pool))
        n_qry = min(self.n_query,   len(query_pool))
        sup_idx = np.random.choice(support_pool, size=n_sup, replace=False)
        qry_idx = np.random.choice(query_pool,   size=n_qry, replace=False)

        return self._indices_to_tensors(dataset, sup_idx, qry_idx)

    def _sample_random_episode(self, dataset: AssayDataset):
        """
        Support and query sampled randomly from the whole assay.
        Standard FS-Mol protocol. Used as fallback or for baseline comparison.
        """
        n_total = len(dataset)
        all_idx = np.random.permutation(n_total)

        n_sup = min(self.n_support, n_total // 2)
        n_qry = min(self.n_query, n_total - n_sup)

        sup_idx = all_idx[:n_sup]
        qry_idx = all_idx[n_sup:n_sup + n_qry]

        return self._indices_to_tensors(dataset, sup_idx, qry_idx)

    def _indices_to_tensors(self, dataset, sup_idx, qry_idx):
        # fingerprints is a list of 1D numpy arrays — stack only the rows we need.
        # Must use list comprehension (not array indexing) since fingerprints is a list.
        support_fp     = torch.tensor(np.stack([dataset.fingerprints[i] for i in sup_idx]))
        support_labels = torch.tensor(dataset.labels[sup_idx])
        query_fp       = torch.tensor(np.stack([dataset.fingerprints[i] for i in qry_idx]))
        query_labels   = torch.tensor(dataset.labels[qry_idx])
        return support_fp, support_labels, query_fp, query_labels

    # ------------------------------------------------------------------
    # Index-only variants — used by FSMolGraphEpisodeDataset so graph
    # construction can happen after index selection without duplicating
    # the sampling logic.
    # ------------------------------------------------------------------

    def _get_episode_indices_shift_aware(self, dataset: "AssayDataset"):
        min_group = max(2, self.n_support // 4)
        usable_keys = [k for k, v in dataset.scaffold_groups.items() if len(v) >= min_group]
        if len(usable_keys) < 2:
            return self._get_episode_indices_random(dataset)
        chosen = np.random.choice(len(usable_keys), size=2, replace=False)
        sup_pool = dataset.scaffold_groups[usable_keys[chosen[0]]]
        qry_pool = dataset.scaffold_groups[usable_keys[chosen[1]]]
        # Cap to available size — no replacement, no duplicates
        n_sup = min(self.n_support, len(sup_pool))
        n_qry = min(self.n_query,   len(qry_pool))
        sup_idx = np.random.choice(sup_pool, size=n_sup, replace=False)
        qry_idx = np.random.choice(qry_pool, size=n_qry, replace=False)
        return sup_idx, qry_idx

    def _get_episode_indices_random(self, dataset: "AssayDataset"):
        n_total = len(dataset)
        all_idx = np.random.permutation(n_total)
        n_sup = min(self.n_support, n_total // 2)
        n_qry = min(self.n_query, n_total - n_sup)
        return all_idx[:n_sup], all_idx[n_sup:n_sup + n_qry]


# =============================================================================
# ON-THE-FLY ASSAY FILE LOADER  (used by FSMolEpisodeDataset)
# =============================================================================

def _load_assay_file(filepath: str) -> "AssayDataset":
    """
    Load one FS-Mol .jsonl.gz file into an AssayDataset without tracking stats.
    Used by FSMolEpisodeDataset workers to load assays on demand during training,
    keeping only Relation == "=" (exact measurement) compounds.
    """
    fingerprints  = []
    smiles_list   = []
    labels        = []
    binary_labels = []
    assay_id      = os.path.basename(filepath).replace(".jsonl.gz", "")

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mol = json.loads(line)

            if mol.get("Relation", "=") != "=":
                continue

            fp         = mol.get("fingerprints", None)
            smi        = mol.get("SMILES", None)
            label      = mol.get(
                "LogRegressionProperty",
                mol.get("RegressionProperty", mol.get("Property", None))
            )
            _prop = mol.get("Property", None)
            bool_label = int(float(_prop)) if _prop is not None else None

            if fp is None or label is None or smi is None or len(fp) != 2048:
                continue
            try:
                fingerprints.append(np.array(fp, dtype=np.float32))
                smiles_list.append(smi)
                labels.append(float(label))
                binary_labels.append(int(bool_label) if bool_label is not None else -1)
            except (ValueError, TypeError):
                continue

    n = min(len(fingerprints), len(labels), len(smiles_list))
    fingerprints  = fingerprints[:n]
    labels        = labels[:n]
    smiles_list   = smiles_list[:n]
    binary_labels = binary_labels[:n]

    dataset = AssayDataset.__new__(AssayDataset)
    dataset.assay_id     = assay_id
    dataset.fingerprints = fingerprints
    dataset.labels       = np.array(labels, dtype=np.float32)
    dataset.scaffolds    = smiles_list

    dataset.scaffold_groups = {}
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        key = scaffold if scaffold is not None else "__none__"
        dataset.scaffold_groups.setdefault(key, []).append(i)

    bl_arr = np.array(binary_labels, dtype=np.int32) if binary_labels else np.array([], dtype=np.int32)
    dataset.binary_labels = None if len(bl_arr) == 0 or (bl_arr == -1).all() else bl_arr

    dataset._validate_scaffold_groups()
    return dataset


# =============================================================================
# FS-MOL PRETRAINING DATASET
# =============================================================================

class FSMolEpisodeDataset(Dataset):
    """
    Streaming episodic dataset for FS-Mol pretraining (ECFP encoder).

    Each __getitem__ picks a random file from ALL training assays, loads it from
    disk, and samples one episode. This matches the FS-Mol paper's approach of
    streaming through the full training set rather than caching a fixed pool.

    Requires only n_support molecules per assay (not n_support + n_query).
    Variable episode sizes: small assays contribute shorter episodes rather than
    being excluded.
    """

    def __init__(
        self,
        assay_files: list[str],
        n_episodes_per_epoch: int = 1000,
        n_support: int = 16,
        n_query: int = 16,
        shift_aware: bool = True,
        pool_size: int = 0,   # kept for API compatibility — ignored
    ):
        if not assay_files:
            raise ValueError("No assay files provided.")
        self.all_files   = assay_files
        self.n_episodes  = n_episodes_per_epoch
        self.n_support   = n_support
        self.n_query     = n_query
        self.sampler     = EpisodeSampler(n_support, n_query)
        self.shift_aware = shift_aware
        print(f"  Streaming ECFP dataset: {len(self.all_files)} training files, "
              f"min {n_support} molecules per assay")

    def refresh_pool(self, verbose: bool = False) -> None:
        pass   # no-op — streaming loads fresh from disk each episode

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx):
        for _ in range(200):
            path = self.all_files[np.random.randint(len(self.all_files))]
            ds = _load_assay_file(path)
            if len(ds) >= self.n_support:
                return self.sampler.sample_episode(ds, shift_aware=self.shift_aware)
        raise RuntimeError("Could not find a valid assay after 200 attempts.")


# =============================================================================
# DRUGOOD EVALUATION DATASET
# =============================================================================

# =============================================================================
# GNN GRAPH EPISODE DATASET
# =============================================================================

def _build_graphs_for_assay(dataset: "AssayDataset") -> list:
    """
    Build a PyG Data object for every molecule in `dataset` (uses SMILES from
    dataset.scaffolds).  Returns a list aligned with dataset.fingerprints/labels.
    Molecules whose SMILES yield an invalid graph get a minimal placeholder so
    indices always match — the placeholder has zero edges and a single zero node.
    """
    from featurize import smiles_to_graph, NODE_FEAT_DIM
    import torch
    from torch_geometric.data import Data

    _dummy = Data(
        x=torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 12), dtype=torch.float),
    )

    graphs = []
    for smi in dataset.scaffolds:
        g = smiles_to_graph(smi)
        graphs.append(g if g is not None else _dummy)
    return graphs


def _build_fsmol_graphs_for_assay(dataset: "AssayDataset") -> list:
    """
    Build FS-Mol-faithful PyG Data objects for every molecule in `dataset`.
    Uses smiles_to_fsmol_graph: Kekulized, 3-dim bond-type one-hot edges,
    plus graph-level ECFP (1,2048) and descriptor (1,42) tensors.
    """
    from featurize import smiles_to_fsmol_graph, FSMOL_NODE_FEAT_DIM
    import torch
    from torch_geometric.data import Data

    _dummy = Data(
        x=torch.zeros((1, FSMOL_NODE_FEAT_DIM), dtype=torch.float),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 3), dtype=torch.float),
        ecfp=torch.zeros((1, 2048), dtype=torch.float),
        descriptors=torch.zeros((1, 42), dtype=torch.float),
    )

    graphs = []
    for smi in dataset.scaffolds:
        g = smiles_to_fsmol_graph(smi)
        graphs.append(g if g is not None else _dummy)
    return graphs


def graph_episode_collate(batch):
    """
    Collate function for FSMolGraphEpisodeDataset.

    Input : list of (sup_graphs, sup_labels, qry_graphs, qry_labels)
              where sup_graphs / qry_graphs are list[PyG Data]
    Output: (PyG Batch(B*n_sup), Tensor(B,n_sup),
             PyG Batch(B*n_qry), Tensor(B,n_qry))
    """
    from torch_geometric.data import Batch as PyGBatch

    sup_graphs = [item[0] for item in batch]   # list of list[Data]
    sup_labels = [item[1] for item in batch]   # list of Tensor[n_sup]
    qry_graphs = [item[2] for item in batch]   # list of list[Data]
    qry_labels = [item[3] for item in batch]   # list of Tensor[n_qry]

    sup_batch  = PyGBatch.from_data_list([g for gs in sup_graphs for g in gs])
    qry_batch  = PyGBatch.from_data_list([g for gs in qry_graphs for g in gs])
    sup_labels_t = torch.stack(sup_labels)   # (B, n_sup)
    qry_labels_t = torch.stack(qry_labels)   # (B, n_qry)

    return sup_batch, sup_labels_t, qry_batch, qry_labels_t


class FSMolGraphEpisodeDataset(Dataset):
    """
    Streaming episodic dataset for GNN-based training.

    Each __getitem__ picks a random training file, loads it from disk, samples
    episode indices, then builds PyG graphs ONLY for the selected molecules.
    This matches the FS-Mol paper: all training assays are accessible each epoch,
    and graph featurization happens on-demand rather than pre-caching the pool.

    Key advantages over the old pool approach:
      - Trains on all 26,868 assays (vs 21 with pool_size=750, min_len=320)
      - Builds only ~320 graphs per episode (vs all molecules in pool assays)
      - Variable episode sizes: small assays (< n_support+n_query) are included

    Use with graph_episode_collate and batch_size=1 (variable episode sizes).
    Set num_workers >= 4 to overlap disk I/O with GPU processing.
    """

    def __init__(
        self,
        assay_files: list[str],
        n_episodes_per_epoch: int = 1000,
        n_support: int = 16,
        n_query: int = 16,
        shift_aware: bool = True,
        pool_size: int = 0,   # kept for API compatibility — ignored
        fsmol_style: bool = False,
    ):
        if not assay_files:
            raise ValueError("No assay files provided.")
        self.all_files   = assay_files
        self.n_episodes  = n_episodes_per_epoch
        self.n_support   = n_support
        self.n_query     = n_query
        self.sampler     = EpisodeSampler(n_support, n_query)
        self.shift_aware = shift_aware
        self.fsmol_style = fsmol_style
        self._setup_graph_builder()
        print(f"  Streaming GNN dataset: {len(self.all_files)} training files, "
              f"min {n_support} molecules per assay")

    def _setup_graph_builder(self):
        """Cache featurisation imports and dummy graph to avoid repeated imports."""
        import torch
        from torch_geometric.data import Data
        if self.fsmol_style:
            from featurize import smiles_to_fsmol_graph, FSMOL_NODE_FEAT_DIM
            self._smiles_to_graph = smiles_to_fsmol_graph
            self._dummy = Data(
                x=torch.zeros((1, FSMOL_NODE_FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 3), dtype=torch.float),
                ecfp=torch.zeros((1, 2048), dtype=torch.float),
                descriptors=torch.zeros((1, 42), dtype=torch.float),
            )
        else:
            from featurize import smiles_to_graph, NODE_FEAT_DIM, EDGE_FEAT_DIM
            self._smiles_to_graph = smiles_to_graph
            self._dummy = Data(
                x=torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float),
            )

    def _build_graphs(self, smiles_list: list[str]) -> list:
        """Build graphs for a list of SMILES — only the selected episode molecules."""
        return [self._smiles_to_graph(smi) or self._dummy for smi in smiles_list]

    def refresh_pool(self, verbose: bool = False) -> None:
        pass  # no-op — streaming loads fresh from disk each episode

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx):
        for _ in range(200):
            path = self.all_files[np.random.randint(len(self.all_files))]
            ds = _load_assay_file(path)
            if len(ds) < self.n_support:
                continue
            if self.shift_aware and len(ds.scaffold_groups) >= 2:
                sup_idx, qry_idx = self.sampler._get_episode_indices_shift_aware(ds)
            else:
                sup_idx, qry_idx = self.sampler._get_episode_indices_random(ds)
            sup_graphs = self._build_graphs([ds.scaffolds[i] for i in sup_idx])
            qry_graphs = self._build_graphs([ds.scaffolds[i] for i in qry_idx])
            sup_labels = torch.tensor(ds.labels[sup_idx])
            qry_labels = torch.tensor(ds.labels[qry_idx])
            return sup_graphs, sup_labels, qry_graphs, qry_labels
        raise RuntimeError("Could not find a valid assay after 200 attempts.")


class DrugOODEvalDataset:
    """
    Wraps DrugOOD splits for multi-scale zero-shot evaluation.

    Stores the full train pool (context) plus ood_test and iid_test query sets.
    Context is subsampled at call time to test multiple sizes (64/128/256/512).

    Protocol:
        Context = sampled from train (in-distribution labeled support)
        Query   = ood_test (OOD shift) or iid_test (in-distribution test)
    """

    def __init__(
        self,
        context_smiles: list[str],
        context_labels: list[float],
        ood_test_smiles: list[str],
        ood_test_labels: list[float],
        iid_test_smiles: list[str],
        iid_test_labels: list[float],
        split_type: str = "scaffold",
        context_binary_labels: Optional[list[int]] = None,
        ood_test_binary_labels: Optional[list[int]] = None,
        iid_test_binary_labels: Optional[list[int]] = None,
    ):
        self.split_type = split_type

        ctx_ds     = AssayDataset(context_smiles, context_labels)
        ood_ds     = AssayDataset(ood_test_smiles, ood_test_labels)
        iid_ds     = AssayDataset(iid_test_smiles, iid_test_labels)

        # Keep SMILES for GNN evaluation (fingerprints not used in that path)
        self.context_smiles  = context_smiles
        self.ood_test_smiles = ood_test_smiles
        self.iid_test_smiles = iid_test_smiles

        # Full context pool — subsampled at episode time
        self.context_fp     = torch.tensor(np.stack(ctx_ds.fingerprints))
        self.context_labels = torch.tensor(ctx_ds.labels)
        self.context_binary = np.array(context_binary_labels, dtype=np.int32) \
                              if context_binary_labels is not None else None

        self.ood_test_fp     = torch.tensor(np.stack(ood_ds.fingerprints))
        self.ood_test_labels = torch.tensor(ood_ds.labels)
        self.ood_test_binary = np.array(ood_test_binary_labels, dtype=np.int32) \
                               if ood_test_binary_labels is not None else None

        self.iid_test_fp     = torch.tensor(np.stack(iid_ds.fingerprints)) \
                               if iid_ds.fingerprints else torch.zeros(0, 2048)
        self.iid_test_labels = torch.tensor(iid_ds.labels) \
                               if len(iid_ds.labels) else torch.zeros(0)
        self.iid_test_binary = np.array(iid_test_binary_labels, dtype=np.int32) \
                               if iid_test_binary_labels is not None else None

    def get_episode(
        self,
        context_size: int = 64,
        query_set: str = "ood_test",
        seed: int = 42,
    ):
        """
        Sample context_size molecules from the train pool and return the chosen query set.

        Args:
            context_size: How many context molecules to sample (64 / 128 / 256 / 512)
            query_set:    "ood_test" or "iid_test"
            seed:         Fixed seed for reproducibility across context sizes

        Returns:
            ctx_fp, ctx_labels, test_fp, test_labels, test_binary_labels
        """
        rng   = np.random.RandomState(seed)
        n_ctx = min(context_size, len(self.context_fp))
        idx   = rng.choice(len(self.context_fp), size=n_ctx, replace=False)

        ctx_fp     = self.context_fp[idx]
        ctx_labels = self.context_labels[idx]

        if query_set == "ood_test":
            return ctx_fp, ctx_labels, self.ood_test_fp, self.ood_test_labels, self.ood_test_binary
        else:
            return ctx_fp, ctx_labels, self.iid_test_fp, self.iid_test_labels, self.iid_test_binary