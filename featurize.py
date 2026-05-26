"""
Molecular Featurization: SMILES → PyTorch Geometric Data
=========================================================
Atom and bond features matching the FS-Mol GNN featurization convention.
Used by PNAGNNEncoder; not needed for ECFP-based models.

Atom features  (NODE_FEAT_DIM = 51):
  Atom type      15  (H,B,C,N,O,F,Si,P,S,Cl,As,Se,Br,I + other)
  Degree         12  (0-10 + other)
  Formal charge   6  (-2,-1,0,1,2 + other)
  Total Hs       10  (0-8 + other)
  Hybridisation   6  (SP,SP2,SP3,SP3D,SP3D2 + other)
  Aromatic        1
  In ring         1

Bond features  (EDGE_FEAT_DIM = 12):
  Bond type       4  (SINGLE,DOUBLE,TRIPLE,AROMATIC)
  Conjugated      1
  In ring         1
  Stereo          6  (NONE,ANY,E,Z,CIS,TRANS)
"""

import numpy as np
from typing import Optional

from rdkit import Chem          # type: ignore
from rdkit.Chem import rdchem   # type: ignore

try:
    from torch_geometric.data import Data
    _PYGEOM_AVAILABLE = True
except ImportError:
    _PYGEOM_AVAILABLE = False
    Data = None  # type: ignore


# ---------------------------------------------------------------------------
# Feature dimension constants — import these everywhere else instead of
# hard-coding numbers, so a single edit here propagates.
# ---------------------------------------------------------------------------

_ATOM_TYPE_LIST   = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]   # H B C N O F Si P S Cl As Se Br I
_DEGREE_LIST      = list(range(11))                                          # 0-10
_FORMAL_CHG_LIST  = [-2, -1, 0, 1, 2]
_TOTAL_HS_LIST    = list(range(9))                                           # 0-8
_HYBRID_LIST      = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

NODE_FEAT_DIM: int = (
    (len(_ATOM_TYPE_LIST) + 1)   # +1 for "other"
    + (len(_DEGREE_LIST) + 1)
    + (len(_FORMAL_CHG_LIST) + 1)
    + (len(_TOTAL_HS_LIST) + 1)
    + (len(_HYBRID_LIST) + 1)
    + 1   # is_aromatic
    + 1   # is_in_ring
)  # = 51

_BOND_STEREO_LIST = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]

EDGE_FEAT_DIM: int = 4 + 1 + 1 + len(_BOND_STEREO_LIST)   # = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(value, choices: list, allow_other: bool = True) -> list[float]:
    """One-hot encode `value` against `choices`; appends an 'other' bucket if allow_other."""
    enc = [0.0] * (len(choices) + int(allow_other))
    try:
        enc[choices.index(value)] = 1.0
    except ValueError:
        if allow_other:
            enc[-1] = 1.0
    return enc


# ---------------------------------------------------------------------------
# Per-atom features
# ---------------------------------------------------------------------------

def atom_features(atom) -> np.ndarray:
    """Return NODE_FEAT_DIM-dimensional float32 feature vector for one RDKit atom."""
    feats: list[float] = []
    feats += _one_hot(atom.GetAtomicNum(),     _ATOM_TYPE_LIST,  allow_other=True)
    feats += _one_hot(atom.GetDegree(),        _DEGREE_LIST,     allow_other=True)
    feats += _one_hot(atom.GetFormalCharge(),  _FORMAL_CHG_LIST, allow_other=True)
    feats += _one_hot(atom.GetTotalNumHs(),    _TOTAL_HS_LIST,   allow_other=True)
    feats += _one_hot(atom.GetHybridization(), _HYBRID_LIST,     allow_other=True)
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-bond features
# ---------------------------------------------------------------------------

_BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE:    [1.0, 0.0, 0.0, 0.0],
    rdchem.BondType.DOUBLE:    [0.0, 1.0, 0.0, 0.0],
    rdchem.BondType.TRIPLE:    [0.0, 0.0, 1.0, 0.0],
    rdchem.BondType.AROMATIC:  [0.0, 0.0, 0.0, 1.0],
}

def bond_features(bond) -> np.ndarray:
    """Return EDGE_FEAT_DIM-dimensional float32 feature vector for one RDKit bond."""
    feats: list[float] = []
    feats += _BOND_TYPE_MAP.get(bond.GetBondType(), [0.0, 0.0, 0.0, 0.0])
    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.IsInRing()))
    feats += _one_hot(bond.GetStereo(), _BOND_STEREO_LIST, allow_other=False)
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# SMILES → PyG Data
# ---------------------------------------------------------------------------

def smiles_to_graph(smiles: str) -> Optional["Data"]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None if the SMILES is invalid or the molecule has no atoms.
    Edges are added in both directions (undirected graph convention for GNNs).
    Isolated atoms (no bonds) produce a valid graph with zero edges.
    """
    if not _PYGEOM_AVAILABLE:
        raise ImportError("torch_geometric is required for GNN encoder. "
                          "Install via: pip install torch-geometric")

    import torch  # local import — featurize.py is imported even for ECFP runs
    from torch_geometric.data import Data as PyGData

    mol = Chem.MolFromSmiles(smiles)   # type: ignore[attr-defined]
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    if not atom_feats:
        return None
    x = torch.tensor(np.stack(atom_feats), dtype=torch.float)   # (n_atoms, NODE_FEAT_DIM)

    # Edge index + edge features (add both directions)
    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        rows += [i, j]
        cols += [j, i]
        edge_feats += [feat, feat]   # same features for both directions

    if edge_feats:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)   # (2, 2*n_bonds)
        edge_attr  = torch.tensor(np.stack(edge_feats), dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)

    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Degree histogram (required by PNAConv scalers)
# ---------------------------------------------------------------------------

def compute_degree_histogram(assay_files: list[str], n_sample: int = 500,
                              max_degree: int = 10) -> "torch.Tensor":
    """
    Compute the node-degree histogram over a sample of training assays.
    Required by PNAConv to normalise its degree scalers.

    Args:
        assay_files: List of .jsonl.gz assay file paths (training set).
        n_sample:    How many assays to sample. 500 is enough for a stable estimate.
        max_degree:  Histogram bins: [0, 1, ..., max_degree].

    Returns:
        deg: LongTensor of shape (max_degree + 1,)
    """
    import torch
    import gzip, json, random
    from torch_geometric.utils import degree as pyg_degree

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    files = random.sample(assay_files, min(n_sample, len(assay_files)))

    n_mols = 0
    for fpath in files:
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mol_json = json.loads(line)
                if mol_json.get("Relation", "=") != "=":
                    continue
                smi = mol_json.get("SMILES", None)
                if smi is None:
                    continue
                data = smiles_to_graph(smi)
                if data is None or data.edge_index.shape[1] == 0:
                    continue
                d = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                d_clamped = d.clamp(max=max_degree)
                deg += torch.bincount(d_clamped, minlength=max_degree + 1)
                n_mols += 1

    print(f"  Degree histogram computed over {n_mols} molecules ({len(files)} assays).")
    return deg
