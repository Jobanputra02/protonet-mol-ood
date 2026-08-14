"""
Episode Construction for Prototypical Network Training
=======================================================
An "episode" is one few-shot task:
    - Support set: N molecules with known labels (the context)
    - Query set:   M molecules to predict

Three episode construction strategies (controlled by training_split):
    "random":      Support and query sampled randomly from assay.
                   Standard FS-Mol protocol. No structural separation.
    "scaffold":    Support from one set of Murcko scaffold groups, query from another.
                   Fine-grained structural split — barely OOD (query→support Tanimoto
                   only drops 0.494→0.437 vs random). Exists as a valid option but
                   similarity is preferred.
    "similarity":  Support and query separated by Butina fingerprint clustering
                   (cutoff=SCAFFOLD_OOD_CUTOFF). Coarser than Murcko — collapses
                   similar molecules into clusters, giving a real fingerprint-shift
                   between support and query (0.494→0.345 at n=128).
                   This is the recommended OOD training mode.

Falls back to random when the assay has too few groups to build a valid split.
"""

import gzip
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional

# RDKit has incomplete type stubs - Pylance reports false positives on its
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

    Returns a COUNT vector (values >= 0, not binary) - each position counts
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

    Note: the Murcko scaffold keeps exact ring atoms/bond orders, so it is
    VERY fine-grained: in FS-Mol test the median assay has 157 molecules but 127 Murcko
    scaffolds (~70% singletons). A "scaffold-disjoint" split over Murcko groups therefore
    holds out singletons, not chemotypes -> query->support Tanimoto only drops 0.50->0.44
    vs a random split, i.e. it is barely OOD. For a real scaffold-OOD split use
    get_generic_scaffold() (coarser) as the grouping key instead.
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception:
        return None


def get_generic_scaffold(smiles: str) -> Optional[str]:
    """
    Bemis-Murcko GENERIC (graph-framework) scaffold: the Murcko scaffold with every atom
    made carbon and every bond made single. This is COARSER than get_scaffold(): molecules
    that differ only in heteroatom identity or bond order collapse into ONE group, so a
    scaffold-disjoint split actually holds out whole chemotypes instead of near-unique
    singletons.

    IMPORTANT: this is used ONLY as a GROUPING KEY to decide which molecules go to
    support vs query. The molecule the model actually sees (its real graph / fingerprint /
    label) is never modified - this is not data tampering, it is a coarser split boundary,
    exactly like Murcko scaffolding already is.

    Returns None on invalid SMILES or if generic-scaffold construction fails.
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)          # Murcko scaffold as a Mol
        generic = MurckoScaffold.MakeScaffoldGeneric(core)    # atoms->C, bonds->single
        return Chem.MolToSmiles(generic)                      # type: ignore[attr-defined]
    except Exception:
        return None


def build_scaffold_groups(smiles_list: list[str], mode: str = "murcko") -> dict:
    """
    Group molecule indices by scaffold key. Returns {scaffold_key: [indices]}.

    mode="murcko"  -> get_scaffold         (fine-grained - barely OOD, see note above)
    mode="generic" -> get_generic_scaffold (coarser, but still insufficient here)

    On FS-Mol test, generic scaffolds barely help - singleton fraction
    only 0.70 -> 0.63, query->support Tanimoto only 0.437 -> 0.426 (random is 0.494). The
    ring skeletons themselves differ within an assay, so genericising atoms merges almost
    nothing. For a real (still only mild) shift use build_butina_groups().

    Used by the evaluator to (re)build coarse groups on the fly from an assay's SMILES
    (AssayDataset.scaffolds holds the original SMILES for FS-Mol-loaded assays), without
    changing how the dataset itself was constructed.
    """
    fn = get_generic_scaffold if mode == "generic" else get_scaffold
    groups: dict = {}
    for i, smi in enumerate(smiles_list):
        sc = fn(smi)
        key = sc if sc is not None else "__none__"
        groups.setdefault(key, []).append(i)
    return groups


def build_butina_groups(fingerprints: list, cutoff: float = 0.55) -> dict:
    """
    Group molecule indices by Butina (sphere-exclusion) clustering on ECFP Tanimoto
    distance. Two molecules join the same cluster when their Tanimoto similarity exceeds
    (1 - cutoff). Larger cutoff -> coarser clusters -> stronger held-out shift.

    Returns {cluster_id: [indices]} to feed straight into build_fair_split_indices as the
    grouping key (support = whole clusters, query = held-out clusters). This is the best
    available inside-assay scaffold-OOD split.

    (query->support max-Tanimoto @ n=128, lower = more OOD):
        random 0.494 | murcko 0.437 | generic 0.426 | butina@0.55 0.378 | butina@0.70 0.345.
    So even coarse Butina is only a MILD shift - a single FS-Mol assay is chemically narrow
    (analogs around one target), so no within-assay split is strongly OOD. Genuine scaffold
    shift is a CROSS-dataset property -> that is what DrugOOD measures. Report the achieved
    query->support similarity alongside any inside-task scaffold result.

    fingerprints: list of 1D ECFP count arrays (AssayDataset.fingerprints).
    """
    from rdkit import DataStructs  # type: ignore
    from rdkit.ML.Cluster import Butina  # type: ignore

    bitvects = []
    for f in fingerprints:
        f = np.asarray(f)
        bv = DataStructs.ExplicitBitVect(len(f))
        for b in np.nonzero(f > 0)[0].tolist():
            bv.SetBit(int(b))
        bitvects.append(bv)

    n = len(bitvects)
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return {ci: list(members) for ci, members in enumerate(clusters)}


# Cutoff for shift-aware training episodes — aliased from config so train and eval always
# use the same Butina cutoff (single source of truth: config.SCAFFOLD_OOD_CUTOFF).
from config import SCAFFOLD_OOD_CUTOFF as SHIFT_AWARE_BUTINA_CUTOFF  # noqa: E402


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
        # Do NOT convert to a single large numpy array here - with 26k assays
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

    training_split controls episode construction — see module docstring.
    Falls back to random when the assay cannot yield a valid group-disjoint split.
    """

    def __init__(self, n_support=64, n_query=256):
        """
        Args:
            n_support: Fixed int OR list[int] to sample from per episode.
                       e.g. 64 → always 64; [16, 32, 64, 128, 256] → drawn each episode.
            n_query:   Number of molecules in the query set per episode.
        """
        self.n_support = n_support   # int or list[int]
        self.n_query   = n_query

    @property
    def n_support_min(self) -> int:
        """Smallest possible support size - used for assay size gating."""
        if isinstance(self.n_support, (list, tuple)):
            return int(min(self.n_support))
        return int(self.n_support)

    def _draw_n_support(self) -> int:
        """Sample one support size for the current episode."""
        if isinstance(self.n_support, (list, tuple)):
            return int(np.random.choice(self.n_support))
        return int(self.n_support)

    def sample_episode(self, dataset: AssayDataset, training_split: str = "random",
                       use_binary: bool = False):
        """
        Sample one episode from a dataset.

        Args:
            dataset:        AssayDataset to sample from.
            training_split: "random" | "scaffold" | "similarity" — see module docstring.
            use_binary:     If True and dataset.binary_labels is not None, return
                            pre-binarised (0/1) labels instead of continuous labels.

        Returns:
            support_fp:     (n_support, 2048) float tensor
            support_labels: (n_support,) float tensor  - binary if use_binary=True
            query_fp:       (n_query, 2048) float tensor
            query_labels:   (n_query,) float tensor    - binary if use_binary=True
        """
        if training_split in ("scaffold", "similarity"):
            return self._sample_shift_aware_episode(dataset, training_split=training_split,
                                                    use_binary=use_binary)
        return self._sample_random_episode(dataset, use_binary=use_binary)

    def _sample_shift_aware_episode(self, dataset: AssayDataset,
                                    training_split: str = "similarity",
                                    use_binary: bool = False):
        """Group-disjoint episode (scaffold or similarity), as tensors."""
        sup_idx, qry_idx = self._get_episode_indices_shift_aware(dataset, training_split)
        return self._indices_to_tensors(dataset, sup_idx, qry_idx, use_binary=use_binary)

    def _sample_random_episode(self, dataset: AssayDataset, use_binary: bool = False):
        """
        Support and query sampled from the whole assay using stratified splitting.
        Preserves the natural active/inactive class proportion in the support set,
        matching the FS-Mol paper's StratifiedShuffleSplit approach. Falls back to
        pure random sampling when binary labels are unavailable or all-one-class.
        """
        sup_idx, qry_idx = self._get_episode_indices_random(dataset)
        return self._indices_to_tensors(dataset, sup_idx, qry_idx, use_binary=use_binary)

    def _indices_to_tensors(self, dataset, sup_idx, qry_idx, use_binary: bool = False):
        # fingerprints is a list of 1D numpy arrays - stack only the rows we need.
        # Must use list comprehension (not array indexing) since fingerprints is a list.
        support_fp = torch.tensor(np.stack([dataset.fingerprints[i] for i in sup_idx]))
        query_fp   = torch.tensor(np.stack([dataset.fingerprints[i] for i in qry_idx]))

        if use_binary and dataset.binary_labels is not None:
            support_labels = torch.tensor(dataset.binary_labels[sup_idx].astype(np.float32))
            query_labels   = torch.tensor(dataset.binary_labels[qry_idx].astype(np.float32))
        else:
            support_labels = torch.tensor(dataset.labels[sup_idx])
            query_labels   = torch.tensor(dataset.labels[qry_idx])

        return support_fp, support_labels, query_fp, query_labels

    # ------------------------------------------------------------------
    # Index-only variants - used by FSMolGraphEpisodeDataset so graph
    # construction can happen after index selection without duplicating
    # the sampling logic.
    # ------------------------------------------------------------------

    def _get_episode_indices_shift_aware(self, dataset: "AssayDataset",
                                         training_split: str = "similarity"):
        """
        Group-disjoint support/query split.

        "scaffold"  — uses the Murcko scaffold groups already computed at load time.
                      Fine-grained; rarely separates chemically similar molecules.
        "similarity" — builds Butina clusters (SHIFT_AWARE_BUTINA_CUTOFF) on the fly,
                       giving a real fingerprint-shift between support and query.
                       Butina is O(n^2); a pool cap keeps memory bounded on large assays.

        In both modes, build_fair_split_indices accumulates whole groups into the support
        (no duplication, both classes enforced). Falls back to random when no valid
        group-disjoint split exists for this assay.
        """
        n_sup_target    = self._draw_n_support()
        n_total         = len(dataset)
        adaptive_target = min(n_sup_target, max(2, n_total // 2))

        if training_split == "scaffold":
            groups = dataset.scaffold_groups
            if len(groups) < 2:
                return self._get_episode_indices_random(dataset)
        else:  # "similarity" — Butina fingerprint clustering
            pool_cap = max(4 * (n_sup_target + self.n_query), 1024)
            if n_total > pool_cap:
                pool   = np.random.choice(n_total, size=pool_cap, replace=False)
                groups = {k: [int(pool[i]) for i in v]
                          for k, v in build_butina_groups(
                              [dataset.fingerprints[i] for i in pool],
                              SHIFT_AWARE_BUTINA_CUTOFF).items()}
            else:
                groups = build_butina_groups(dataset.fingerprints, SHIFT_AWARE_BUTINA_CUTOFF)

        sp = build_fair_split_indices(
            n_total, groups,
            getattr(dataset, "binary_labels", None),
            adaptive_target, "scaffold", np.random,
            n_query_cap=self.n_query, require_both_classes=True,
            min_support=min(8, adaptive_target),
        )
        if sp is None:
            return self._get_episode_indices_random(dataset)
        return sp

    def _get_episode_indices_random(self, dataset: "AssayDataset"):
        """
        Stratified support/query split - preserves active/inactive class proportion.
        Matches FS-Mol paper's StratifiedShuffleSplit. Falls back to pure random
        when binary labels are missing or the assay is all one class.
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        n_sup_target = self._draw_n_support()
        n_total      = len(dataset)
        n_sup        = min(n_sup_target, n_total // 2)
        n_qry        = min(self.n_query, n_total - n_sup)

        labels = dataset.binary_labels
        can_stratify = (
            labels is not None
            and len(labels) == n_total
            and len(np.unique(labels)) >= 2   # at least one active and one inactive
            and n_sup >= 2                    # need at least 1 per class
        )

        if can_stratify:
            try:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sup, random_state=None)
                sup_idx, remaining = next(sss.split(np.arange(n_total), labels))
                qry_idx = remaining[:n_qry]
                return sup_idx, qry_idx
            except ValueError:
                pass  # fall through to random if stratification fails (e.g. too few per class)

        # Fallback: pure random
        all_idx = np.random.permutation(n_total)
        return all_idx[:n_sup], all_idx[n_sup:n_sup + n_qry]


# =============================================================================
# FAIR SPLIT CONSTRUCTION  (shared by ProtoNet eval, RF, and head baselines)
# =============================================================================

def build_fair_split_indices(
    n_total: int,
    scaffold_groups: dict,
    binary_labels,
    n_support: int,
    split_type: str,
    rng,
    mol_sizes=None,
    n_query_cap: Optional[int] = None,
    require_both_classes: bool = True,
    max_retries: int = 20,
    min_support: Optional[int] = None,
):
    """
    Build (support_idx, query_idx) with ONE leakage-free protocol used identically
    by ProtoNet, RandomForest, and the embedding+head baselines. The whole point is
    that every model evaluated on a given (assay, support_size, repeat) sees the
    SAME support and query molecules, so differences reflect the model, not the split.

    Protocols:
      - "random":   n_support DISTINCT molecules sampled at random (no replacement);
                    query = all remaining molecules.
      - "scaffold": accumulate WHOLE Murcko scaffold groups (in shuffled order) until
                    at least n_support DISTINCT molecules are collected; query = all
                    molecules from the remaining groups. No molecule is ever duplicated
                    and support spans several scaffolds, exactly like the RF baseline.
                    (This replaces the old single-group + with-replacement protocol.)
      - "size":     support sampled (no replacement) from the smallest 50% of molecules
                    by heavy-atom count; query = the largest 50%.

    Class balance: when binary_labels is available and require_both_classes is True,
    the split is re-drawn (up to max_retries) until the support contains at least one
    active and one inactive molecule. If that is impossible the function returns None,
    and the caller skips that (assay, support_size, repeat) for ALL models equally.

    min_support: evaluation passes None -> the scaffold split must reach EXACTLY
    n_support (fixed support-size buckets for the curves). Training passes a small
    floor -> the scaffold split accepts a SMALLER multi-group support (down to
    min_support) instead of bailing to a random episode, so shift-aware episodes
    actually fire on assays that can't yield a full n_support scaffold-disjoint split.

    Returns:
        (sup_idx, qry_idx) as int numpy arrays, or None if no valid split exists.
        For scaffold splits the support may exceed n_support (whole final group is
        kept); callers should record len(sup_idx) as the actual support size.
    """
    labels = None
    if binary_labels is not None and require_both_classes:
        labels = np.asarray(binary_labels)
        if len(np.unique(labels[np.isin(labels, [0, 1])])) < 2:
            return None  # assay is single-class overall - no balanced split possible

    def _both_classes(idx) -> bool:
        if labels is None:
            return True
        u = np.unique(labels[idx])
        return (0 in u) and (1 in u)

    def _cap_query(qry_idx):
        if n_query_cap is not None and len(qry_idx) > n_query_cap:
            return rng.choice(qry_idx, size=n_query_cap, replace=False)
        return qry_idx

    for _ in range(max_retries):
        if split_type == "random":
            if n_total <= n_support:
                return None
            perm    = rng.permutation(n_total)
            sup_idx = perm[:n_support]
            qry_idx = perm[n_support:]

        elif split_type == "scaffold":
            keys = list(scaffold_groups.keys())
            if len(keys) < 2:
                return None
            rng.shuffle(keys)
            floor = n_support if min_support is None else min_support
            # Accumulate WHOLE groups into support up to n_support, but never consume
            # the final group - it (plus any others past the cut) becomes the query, so
            # support and query scaffolds stay disjoint even on small assays.
            sup_list: list[int] = []
            cut = 0
            for ki in range(len(keys) - 1):
                if len(sup_list) >= n_support:
                    break
                sup_list.extend(scaffold_groups[keys[ki]])
                cut = ki + 1
            sup_idx = np.array(sup_list, dtype=np.int64)
            qry_idx = np.array(
                [i for k in keys[cut:] for i in scaffold_groups[k]], dtype=np.int64
            )
            if len(sup_idx) < floor or len(qry_idx) == 0:
                return None

        elif split_type == "size":
            if mol_sizes is None:
                return None
            order = np.argsort(mol_sizes)
            mid   = n_total // 2
            small, large = order[:mid], order[mid:]
            if len(small) < n_support or len(large) == 0:
                return None
            sup_idx = rng.choice(small, size=n_support, replace=False)
            qry_idx = large

        else:
            raise ValueError(f"Unknown split_type: {split_type}")

        if _both_classes(sup_idx):
            return np.asarray(sup_idx, dtype=np.int64), _cap_query(np.asarray(qry_idx, dtype=np.int64))

    return None


def stratified_context_indices(binary_labels, n_context: int, rng,
                               balanced: bool = False) -> np.ndarray:
    """
    Sample n_context indices for the DrugOOD context (support) set.

    balanced=False (legacy): keep the natural active/inactive ratio, guaranteeing >=1 of
        the minority class. On DrugOOD (78-94% active) this still leaves the inactive
        prototype built from ~1 molecule at small context sizes -> very noisy ranking.

    balanced=True: aim for a 50/50 active/inactive context,
        capped by class availability, spillover to the majority class only if one class is
        too small. This matches training (median 50/50 episodes) and stabilises the inactive
        prototype so the ranking metrics (AUROC / ΔAUPRC) reflect the representation rather
        than a 1-molecule prototype. Choosing the support composition is legitimate - it does
        NOT touch the fixed test set.

    Falls back to uniform sampling when labels are missing or single-class.
    """
    n_pool = len(binary_labels) if binary_labels is not None else 0
    n_context = min(n_context, n_pool)
    if binary_labels is None:
        return rng.choice(n_pool, size=n_context, replace=False)

    labels = np.asarray(binary_labels)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return rng.choice(n_pool, size=n_context, replace=False)

    if balanced:
        # Aim for an even split; if a class is too small, take all of it and fill the
        # remainder from the other class.
        n_pos = min(n_context // 2, len(pos))
        n_neg = min(n_context - n_pos, len(neg))
        n_pos = min(n_context - n_neg, len(pos))   # re-fill if neg was the short class
    else:
        # Allocate proportionally but guarantee >=1 of the minority class.
        frac_pos = len(pos) / n_pool
        n_pos = int(round(frac_pos * n_context))
        n_pos = max(1, min(n_pos, n_context - 1, len(pos)))
        n_neg = n_context - n_pos
        if n_neg > len(neg):
            n_neg = len(neg)
            n_pos = min(n_context - n_neg, len(pos))

    sel = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False),
        rng.choice(neg, size=n_neg, replace=False),
    ])
    rng.shuffle(sel)
    return sel


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
        training_split: str = "random",
        pool_size: int = 0,           # kept for API compatibility - ignored
        use_binary_labels: bool = False,
    ):
        if not assay_files:
            raise ValueError("No assay files provided.")
        if training_split not in ("random", "scaffold", "similarity"):
            raise ValueError(f"training_split must be 'random', 'scaffold', or 'similarity'; got '{training_split}'")
        self.all_files         = assay_files
        self.n_episodes        = n_episodes_per_epoch
        self.n_support         = n_support
        self.n_query           = n_query
        self.sampler           = EpisodeSampler(n_support, n_query)
        self.training_split    = training_split
        self.use_binary_labels = use_binary_labels
        n_sup_str = str(n_support) if isinstance(n_support, int) else f"{min(n_support)}–{max(n_support)} (variable)"
        print(f"  Streaming ECFP dataset: {len(self.all_files)} training files, "
              f"n_support={n_sup_str}, training_split={training_split}")

    def refresh_pool(self, verbose: bool = False) -> None:
        pass   # no-op - streaming loads fresh from disk each episode

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx):
        for _ in range(200):
            path = self.all_files[np.random.randint(len(self.all_files))]
            ds = _load_assay_file(path)
            if len(ds) >= self.sampler.n_support_min:
                return self.sampler.sample_episode(
                    ds, training_split=self.training_split, use_binary=self.use_binary_labels
                )
        raise RuntimeError("Could not find a valid assay after 200 attempts.")


# =============================================================================
# DRUGOOD EVALUATION DATASET
# =============================================================================

# =============================================================================
# GNN GRAPH EPISODE DATASET
# =============================================================================

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
        training_split: str = "random",
        pool_size: int = 0,           # kept for API compatibility - ignored
        fsmol_style: bool = False,
        use_binary_labels: bool = False,
    ):
        if not assay_files:
            raise ValueError("No assay files provided.")
        if training_split not in ("random", "scaffold", "similarity"):
            raise ValueError(f"training_split must be 'random', 'scaffold', or 'similarity'; got '{training_split}'")
        self.all_files         = assay_files
        self.n_episodes        = n_episodes_per_epoch
        self.n_support         = n_support
        self.n_query           = n_query
        self.sampler           = EpisodeSampler(n_support, n_query)
        self.training_split    = training_split
        self.fsmol_style       = fsmol_style
        self.use_binary_labels = use_binary_labels
        self._setup_graph_builder()
        n_sup_str = str(n_support) if isinstance(n_support, int) else f"{min(n_support)}–{max(n_support)} (variable)"
        print(f"  Streaming GNN dataset: {len(self.all_files)} training files, "
              f"n_support={n_sup_str}, training_split={training_split}")

    def _setup_graph_builder(self):
        """Cache featurisation imports and dummy graph to avoid repeated imports."""
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

    def _build_graphs(self, smiles_list: list[str]) -> list:
        """Build graphs for a list of SMILES - only the selected episode molecules."""
        return [self._smiles_to_graph(smi) or self._dummy for smi in smiles_list]

    def refresh_pool(self, verbose: bool = False) -> None:
        pass  # no-op - streaming loads fresh from disk each episode

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx):
        for _ in range(200):
            path = self.all_files[np.random.randint(len(self.all_files))]
            ds = _load_assay_file(path)
            if len(ds) < self.sampler.n_support_min:
                continue
            if self.training_split in ("scaffold", "similarity"):
                sup_idx, qry_idx = self.sampler._get_episode_indices_shift_aware(
                    ds, training_split=self.training_split)
            else:
                sup_idx, qry_idx = self.sampler._get_episode_indices_random(ds)
            sup_graphs = self._build_graphs([ds.scaffolds[i] for i in sup_idx])
            qry_graphs = self._build_graphs([ds.scaffolds[i] for i in qry_idx])
            if self.use_binary_labels and ds.binary_labels is not None:
                sup_labels = torch.tensor(ds.binary_labels[sup_idx].astype(np.float32))
                qry_labels = torch.tensor(ds.binary_labels[qry_idx].astype(np.float32))
            else:
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

        # Full context pool - subsampled at episode time
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
        use_binary: bool = False,
    ):
        """
        Sample context_size molecules from the train pool and return the chosen query set.

        Args:
            context_size: How many context molecules to sample (64 / 128 / 256 / 512)
            query_set:    "ood_test" or "iid_test"
            seed:         Fixed seed for reproducibility across context sizes
            use_binary:   If True and context_binary is not None, return binary (0/1)
                          context labels instead of continuous ones.

        Returns:
            ctx_fp, ctx_labels, test_fp, test_labels, test_binary_labels
        """
        rng   = np.random.RandomState(seed)
        n_ctx = min(context_size, len(self.context_fp))
        # Stratify by binary label when available: the DrugOOD train pool is 78-94%
        # active, so uniform sampling leaves the inactive prototype built from ~0-1
        # molecules at small context sizes. Stratification keeps both classes present.
        if use_binary and self.context_binary is not None:
            # balanced=True: 50/50 active/inactive support so the inactive
            # prototype is stable on DrugOOD's 78-94%-active pool.
            idx = stratified_context_indices(self.context_binary, n_ctx, rng, balanced=True)
        else:
            idx = rng.choice(len(self.context_fp), size=n_ctx, replace=False)

        ctx_fp = self.context_fp[idx]
        if use_binary and self.context_binary is not None:
            ctx_labels = torch.tensor(self.context_binary[idx].astype(np.float32))
        else:
            ctx_labels = self.context_labels[idx]

        if query_set == "ood_test":
            return ctx_fp, ctx_labels, self.ood_test_fp, self.ood_test_labels, self.ood_test_binary
        else:
            return ctx_fp, ctx_labels, self.iid_test_fp, self.iid_test_labels, self.iid_test_binary


# =============================================================================
# DATASET LOADERS
# =============================================================================

# Minimum task size after exact-label filtering (FS-Mol). Was main.MIN_TASK_SIZE.
MIN_TASK_SIZE = 32


def load_fsmol_assay(filepath: str) -> tuple["AssayDataset", dict]:
    """
    Load one FS-Mol assay file (.jsonl.gz) into an AssayDataset.
    Keeps only Relation == "=" (exact) compounds; drops >, <, ~.

    Returns (dataset, stats) where stats = {n_total, n_inexact, n_invalid, n_kept}.
    """
    fingerprints, smiles_list, labels, binary_labels = [], [], [], []
    assay_id = os.path.basename(filepath).replace(".jsonl.gz", "")
    n_total = n_inexact = n_invalid = 0

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mol = json.loads(line)
            n_total += 1
            if mol.get("Relation", "=") != "=":
                n_inexact += 1
                continue
            fp  = mol.get("fingerprints", None)
            smi = mol.get("SMILES", None)
            label = mol.get("LogRegressionProperty",
                            mol.get("RegressionProperty", mol.get("Property", None)))
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

    n = min(len(fingerprints), len(labels), len(smiles_list), len(binary_labels))
    fingerprints, labels = fingerprints[:n], labels[:n]
    smiles_list, binary_labels = smiles_list[:n], binary_labels[:n]

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
    bl_arr = np.array(binary_labels, dtype=np.int32)
    dataset.binary_labels = None if (bl_arr == -1).all() else bl_arr
    dataset._validate_scaffold_groups()

    stats = {"n_total": n_total, "n_inexact": n_inexact,
             "n_invalid": n_invalid, "n_kept": n}
    return dataset, stats


def load_fsmol_split(split_dir: str, max_assays: Optional[int] = None) -> list["AssayDataset"]:
    """
    Load all assays from an FS-Mol split dir (train/valid/test). Filters inexact
    measurements per assay, drops tasks with < MIN_TASK_SIZE compounds, prints a
    data-loss report. Returns a list of AssayDataset.
    """
    files = sorted(os.path.join(split_dir, f)
                   for f in os.listdir(split_dir) if f.endswith(".jsonl.gz"))
    if max_assays is not None:
        files = files[:max_assays]

    assays = []
    total_mols = total_inexact = total_invalid = 0
    mols_in_kept_tasks = mols_in_dropped_tasks = 0
    n_tasks_raw = len(files)
    n_tasks_dropped = 0

    for i, filepath in enumerate(files):
        if i % 500 == 0:
            print(f"  Loading assay {i + 1}/{len(files)}...", end="\r")
        dataset, stats = load_fsmol_assay(filepath)
        total_mols    += stats["n_total"]
        total_inexact += stats["n_inexact"]
        total_invalid += stats["n_invalid"]
        if len(dataset) >= MIN_TASK_SIZE:
            assays.append(dataset)
            mols_in_kept_tasks += stats["n_kept"]
        else:
            n_tasks_dropped += 1
            mols_in_dropped_tasks += stats["n_kept"]

    total_lost = total_inexact + total_invalid + mols_in_dropped_tasks
    print(f"\n{'=' * 60}")
    print(f"  Data-loss report: {os.path.basename(split_dir)}/")
    print(f"{'=' * 60}")
    print(f"  Molecules in raw files        : {total_mols:>10,}")
    print(f"  -- Dropped (inexact Relation) : {total_inexact:>10,}  "
          f"({100 * total_inexact / max(total_mols, 1):.1f}%)")
    print(f"  -- Dropped (bad/missing fields): {total_invalid:>9,}  "
          f"({100 * total_invalid / max(total_mols, 1):.1f}%)")
    print(f"  -- Dropped (task too small)   : {mols_in_dropped_tasks:>10,}  "
          f"({100 * mols_in_dropped_tasks / max(total_mols, 1):.1f}%)")
    print(f"  ----------------------------------------")
    print(f"  Total molecules lost          : {total_lost:>10,}  "
          f"({100 * total_lost / max(total_mols, 1):.1f}%)")
    print(f"  Molecules used                : {mols_in_kept_tasks:>10,}  "
          f"({100 * mols_in_kept_tasks / max(total_mols, 1):.1f}%)")
    print(f"  Tasks kept                    : {len(assays):>10,} / {n_tasks_raw} "
          f"({100 * len(assays) / max(n_tasks_raw, 1):.1f}%)")
    print(f"{'=' * 60}")
    return assays


def load_drugood_split(json_path: str, split_type: str) -> "DrugOODEvalDataset":
    """
    Load one DrugOOD JSON file into a DrugOODEvalDataset.
    Context = train split (in-distribution support); query = ood_test / iid_test.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    split = data["split"]

    def extract(entries: list):
        smiles, labels, binary = [], [], []
        for entry in entries:
            smi = entry.get("smiles", None)
            lab = entry.get("reg_label", None)
            cls_lab = entry.get("cls_label", None)
            if smi and lab is not None:
                try:
                    smiles.append(smi)
                    labels.append(float(lab))
                    binary.append(int(cls_lab) if cls_lab is not None else 0)
                except (ValueError, TypeError):
                    continue
        return smiles, labels, binary

    context_smiles, context_labels, context_binary = extract(split["train"])
    ood_test_smiles, ood_test_labels, ood_test_binary = extract(split["ood_test"])
    iid_test_smiles, iid_test_labels, iid_test_binary = extract(split.get("iid_test", []))
    print(f"  {split_type}: train pool={len(context_labels)}, "
          f"ood_test={len(ood_test_labels)}, iid_test={len(iid_test_labels)}")

    return DrugOODEvalDataset(
        context_smiles=context_smiles, context_labels=context_labels,
        ood_test_smiles=ood_test_smiles, ood_test_labels=ood_test_labels,
        iid_test_smiles=iid_test_smiles, iid_test_labels=iid_test_labels,
        split_type=split_type,
        context_binary_labels=context_binary,
        ood_test_binary_labels=ood_test_binary,
        iid_test_binary_labels=iid_test_binary,
    )


