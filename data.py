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
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
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

        Mechanism:
        1. Pick two distinct scaffold groups (each with >= min_group distinct molecules)
        2. Sample support from group A, query from group B
        3. The embedding must generalize across scaffold families to predict well

        This is what makes pretraining OOD-aware.

        IMPORTANT: singleton scaffold groups (1 molecule) are excluded. Sampling 16
        molecules with replacement from a size-1 group produces 16 identical copies →
        uniform softmax weights → prediction = constant → zero gradient. Requiring
        at least n_support//4 distinct molecules per group ensures meaningful episodes.
        Falls back to random episode if fewer than 2 usable groups exist.
        """
        # Require at least 4 distinct molecules per group (for n_support=16).
        # Singleton groups cause degenerate episodes: all support copies are identical,
        # distances are all equal, softmax is uniform → prediction ignores query → no learning.
        min_group = max(2, self.n_support // 4)
        usable_keys = [k for k, v in dataset.scaffold_groups.items() if len(v) >= min_group]

        if len(usable_keys) < 2:
            return self._sample_random_episode(dataset)

        chosen = np.random.choice(len(usable_keys), size=2, replace=False)
        support_scaffold = usable_keys[chosen[0]]
        query_scaffold   = usable_keys[chosen[1]]

        support_pool = dataset.scaffold_groups[support_scaffold]
        query_pool   = dataset.scaffold_groups[query_scaffold]

        # Use replace=False when group is large enough; replace=True only as last resort.
        sup_idx = np.random.choice(support_pool, size=self.n_support,
                                   replace=len(support_pool) < self.n_support)
        qry_idx = np.random.choice(query_pool,   size=self.n_query,
                                   replace=len(query_pool)   < self.n_query)

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
    Pool-based episodic dataset for FS-Mol pretraining.

    At construction (and optionally between epochs), loads `pool_size` assays from
    randomly chosen files into RAM.  Each __getitem__ samples an episode from the
    in-memory pool — no disk I/O during training, so the GPU stays fed.

    Call refresh_pool() at the start of each epoch to rotate in new assays and
    expose the model to the full diversity of the ~16k filtered assays over time.

    Memory footprint: pool_size × ~1.5 MB/assay (fingerprints + labels).
        pool_size=500  →  ~750 MB
        pool_size=1000 →  ~1.5 GB
        pool_size=2000 →  ~3 GB
    """

    def __init__(
        self,
        assay_files: list[str],
        n_episodes_per_epoch: int = 1000,
        n_support: int = 16,
        n_query: int = 16,
        shift_aware: bool = True,
        pool_size: int = 1000,
    ):
        if not assay_files:
            raise ValueError("No assay files provided.")
        self.all_files   = assay_files
        self.n_episodes  = n_episodes_per_epoch
        self.n_support   = n_support
        self.n_query     = n_query
        self.sampler     = EpisodeSampler(n_support, n_query)
        self.shift_aware = shift_aware
        self.pool_size   = pool_size
        self._pool: list = []
        self.refresh_pool(verbose=True)

    def refresh_pool(self, verbose: bool = False) -> None:
        """
        Load a fresh random subset of assay files into memory.
        Call between epochs to ensure the model sees the full dataset over time.
        """
        n       = min(self.pool_size, len(self.all_files))
        files   = np.random.choice(self.all_files, size=n, replace=False)
        pool    = []
        min_len = self.n_support + self.n_query
        for path in files:
            ds = _load_assay_file(path)
            if len(ds) >= min_len:
                pool.append(ds)
        if not pool:
            raise ValueError(
                f"Pool is empty after sampling {n} files — "
                f"no assay has >= {min_len} exact-measurement molecules."
            )
        self._pool = pool
        if verbose:
            print(f"  Pool loaded: {len(pool)} assays in RAM ({n} files sampled)")

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx):
        ds = self._pool[np.random.randint(len(self._pool))]
        return self.sampler.sample_episode(ds, shift_aware=self.shift_aware)


# =============================================================================
# DRUGOOD EVALUATION DATASET
# =============================================================================

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