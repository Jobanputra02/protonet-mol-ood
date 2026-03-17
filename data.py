"""
Episode construction for PTN training
Episode is one few-shot task with a support set (context) and a query set (test)
here two episodes construction strategies: RANDOM (standard FS-Mol) vs SHIFT-AWARE (scaffold-based split).
we choose shift-aware episodes (with random as fallback)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore
_MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=2048)  # type: ignore


# fingerprint utility

def mol_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """
    Convert SMILES string to ECFP4 fingerprint (binary vector, 2048 bits).
    ECFP4 = module-level _MORGAN_GENERATOR.
    Returns None if SMILES is invalid (also filtered out in AssayDataset).
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    fp = _MORGAN_GENERATOR.GetFingerprintAsNumPy(mol)  # type: ignore[attr-defined]
    return fp.astype(np.float32)


def get_scaffold(smiles: str) -> Optional[str]:
    """
    Extract scaffold from a SMILES string. Scaffold (core ring system + linkers, no side chains)
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception:
        return None


# base assay dataset
class AssayDataset:
    """
    Holds all molecules + labels for a single assay.
    also groups molecules by scaffold for shift-aware episode construction
    """

    def __init__(self, smiles_list: list[str], labels: list[float], assay_id: str = ""):
        self.assay_id = assay_id
        self.fingerprints = []
        self.labels = []
        self.scaffolds = []

        for smi, lab in zip(smiles_list, labels):
            fp = mol_to_fingerprint(smi)
            if fp is not None:
                self.fingerprints.append(fp)
                self.labels.append(lab)
                self.scaffolds.append(get_scaffold(smi))

        # Keep fingerprints as a plain Python list of 1D arrays instead of single laerge array
        # conversion is lazily in _indices_to_tensors()
        self.labels = np.array(self.labels, dtype=np.float32)

        # group indices by scaffold
        self.scaffold_groups = {}
        for i, sc in enumerate(self.scaffolds):
            key = sc if sc is not None else "__none__"
            self.scaffold_groups.setdefault(key, []).append(i)

        
        # Remove any indices from scaffold_groups that are out of bounds. Also removes groups that become empty after filtering. 
        # this needed otherwise index mismatches when datasets are built via (FS-Mol fast loader)
        n = len(self.labels)
        cleaned = {}
        for key, indices in self.scaffold_groups.items():
            valid = [i for i in indices if i < n]
            if valid:
                cleaned[key] = valid
        self.scaffold_groups = cleaned

    def __len__(self):
        # Always use the smaller of the two so indices are always valid otherwise error when fingerprints and labels are out of sync
        return min(len(self.fingerprints), len(self.labels))


# episode sampler
class EpisodeSampler:
    """
    Samples one episode (support + query) from an AssayDataset
    here shift-aware episodes when possible (at least 2 scaffold groups) otherwise random episodes
    """

    def __init__(self, n_support: int = 16, n_query: int = 16):
        self.n_support = n_support
        self.n_query = n_query

    def sample_episode(self, dataset: AssayDataset, shift_aware: bool = True):
        """
        shift_aware: True if by scaffold family (if possible) else random.
        Returns:
            support_fp: (n_support, 2048) float tensor
            support_labels: (n_support,) float tensor
            query_fp: (n_query, 2048) float tensor
            query_labels: (n_query,) float tensor
        """
        if shift_aware and len(dataset.scaffold_groups) >= 2:
            # Support and query come from different scaffold families
            # Pick two distinct scaffold groups, Sample support from group A, query from group B
            # The embedding must generalize across scaffold families to predict well
            # (This is what makes pretraining OOD aware)

            scaffold_keys = list(dataset.scaffold_groups.keys())
            # shuffle and split: first half scaffold groups = support, second half = query
            chosen = np.random.choice(len(scaffold_keys), size=2, replace=False)
            support_scaffold = scaffold_keys[chosen[0]]
            query_scaffold = scaffold_keys[chosen[1]]
            support_pool = dataset.scaffold_groups[support_scaffold]
            query_pool = dataset.scaffold_groups[query_scaffold]

            # sample with replacement if pool is smaller than requested size
            n_sup = min(self.n_support, len(support_pool))
            n_qry = min(self.n_query, len(query_pool))

            sup_idx = np.random.choice(support_pool, size=n_sup, replace=len(support_pool) < n_sup)
            qry_idx = np.random.choice(query_pool, size=n_qry, replace=len(query_pool) < n_qry)

        else:
            # support and query sampled randomly from the whole assay.
            # Standard FS-Mol protocol. fallback
            n_total = len(dataset)
            all_idx = np.random.permutation(n_total)

            n_sup = min(self.n_support, n_total // 2)
            n_qry = min(self.n_query, n_total - n_sup)

            sup_idx = all_idx[:n_sup]
            qry_idx = all_idx[n_sup:n_sup + n_qry]

        # indices to tensors
        support_fp     = torch.tensor(np.stack([dataset.fingerprints[i] for i in sup_idx]))
        support_labels = torch.tensor(dataset.labels[sup_idx])
        query_fp       = torch.tensor(np.stack([dataset.fingerprints[i] for i in qry_idx]))
        query_labels   = torch.tensor(dataset.labels[qry_idx])
        return support_fp, support_labels, query_fp, query_labels

# fs-mol pretraining dataset
class FSMolEpisodeDataset(Dataset):
    """
    FS-Mol assays into a PyTorch Dataset where each __getitem__ returns one sampled episode from a randomly chosen assay

    FS-Mol structure:
        26k+ assays from ChEMBL
        assay: list of SMILES, label pairs
        Labels: binary in original fsmol

    usage:
    dataset = FSMolEpisodeDataset(assay_list, n_episodes_per_epoch=10000)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch_size=1 because each episode has variable support/query size. for fixed sizes, batch can be >1
    """

    def __init__(
        self,
        assay_datasets: list[AssayDataset], 
        n_episodes_per_epoch: int = 10000,
        n_support: int = 16,
        n_query: int = 16,
        shift_aware: bool = True):

        self.assay_datasets = [a for a in assay_datasets if len(a) >= n_support + n_query]
        self.n_episodes = n_episodes_per_epoch
        self.sampler = EpisodeSampler(n_support, n_query)
        self.shift_aware = shift_aware

        if len(self.assay_datasets) == 0:
            raise ValueError("No assays with enough molecules for episode sampling.")

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        # Each call samples a new episode from a random assay
        assay = self.assay_datasets[np.random.randint(len(self.assay_datasets))]
        return self.sampler.sample_episode(assay, shift_aware=self.shift_aware)


# drugood evaluation dataset
class DrugOODEvalDataset:
    """
    DrugOOD provides splits defined by distribution shift type:
        - scaffold shift: test molecules have different core scaffold than train
        - size shift: test molecules are larger/smaller than train
        - assay shift: test from different experimental conditions
    we treat the test set as query and use a small context set (support) from the test distribution
    (zero-shot: no gradient updates, just forward pass with test context)
    Alternative to implement: few-shot fine-tuning with gradient steps (MAML-style inner loop).
    """

    def __init__(
        self,
        test_smiles: list[str],
        test_labels: list[float],
        context_smiles: list[str],
        context_labels: list[float],
        split_type: str  # only for logging
    ):
        self.split_type = split_type

        context_dataset = AssayDataset(context_smiles, context_labels)
        test_dataset = AssayDataset(test_smiles, test_labels)

        # fingerprints is now a list of 1D arrays — stack before converting to tensor
        self.context_fp     = torch.tensor(np.stack(context_dataset.fingerprints))
        self.context_labels = torch.tensor(context_dataset.labels)
        self.test_fp        = torch.tensor(np.stack(test_dataset.fingerprints))
        self.test_labels    = torch.tensor(test_dataset.labels)

    def get_episode(self):
        """Returns the full context (support) and test (query) sets."""
        return self.context_fp, self.context_labels, self.test_fp, self.test_labels
