"""
Molecular Featurization: SMILES → PyTorch Geometric Data (FSMolGNNEncoder)
==========================================================================
FS-Mol-faithful featurization:
  Atom features  (FSMOL_NODE_FEAT_DIM = 40):
    Atom type 19 (18 drug-like types + UNK), Degree 1, Charge 1,
    Radical electrons 1, Isotope 1, Mass 1, Valence 1, Num Hs 1,
    Aromatic 1, Ring membership 13 (any + sizes 3–14)
  Bond features  (FSMOL_EDGE_FEAT_DIM = 3):
    SINGLE / DOUBLE / TRIPLE one-hot (AROMATIC removed by Kekulization)
  Graph-level features stored in PyG Data:
    ecfp        (1, 2048)  ECFP4 fingerprint
    descriptors (1, 42)    RDKit physicochemical descriptors
"""

import numpy as np
from typing import Optional

from rdkit import Chem, RDLogger          # type: ignore
from rdkit.Chem import rdchem             # type: ignore
RDLogger.DisableLog('rdApp.*')            # suppress valence / sanitization warnings

# RDKit 2022+ fingerprint generator API
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore
_MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=2048)     # type: ignore

try:
    from torch_geometric.data import Data
    _PYGEOM_AVAILABLE = True
except ImportError:
    _PYGEOM_AVAILABLE = False
    Data = None  # type: ignore


# =============================================================================
# FS-MOL FAITHFUL FEATURIZATION  (FSMolGNNEncoder)
# =============================================================================

# 18 common drug-like heavy-atom types; atoms not in list → UNK (index 18, last dim)
_FSMOL_ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I',
    'P', 'B', 'Si', 'Se', 'As', 'Na', 'K', 'Ca', 'Mg', 'Fe',
]

FSMOL_NODE_FEAT_DIM: int = (
    len(_FSMOL_ATOM_SYMBOLS) + 1   # 18 types + UNK = 19
    + 7                            # Degree, Charge, RadElec, Isotope, Mass, Valence, NumHs
    + 1                            # IsAromatic
    + 13                           # InAnyRing + ring-size 3..14
)  # = 40

FSMOL_EDGE_FEAT_DIM: int = 3    # SINGLE / DOUBLE / TRIPLE one-hot

DESCRIPTOR_DIM: int = 42         # RDKit physicochemical descriptors

_RDKIT_DESCRIPTOR_LIST: list[str] = [
    # Molecular weight / size  (4)
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumHeavyAtoms",
    # Hydrogen-bonding / heteroatoms  (3)
    "NumHAcceptors", "NumHDonors", "NumRadicalElectrons",
    # Lipophilicity / polarity  (3)
    "MolLogP", "MolMR", "TPSA",
    # Bonds / flexibility  (3)
    "NumRotatableBonds", "NumAmideBonds", "FractionCSP3",
    # Ring counts  (10)
    "RingCount", "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
    "NumAromaticHeterocycles", "NumAromaticCarbocycles",
    "NumSaturatedHeterocycles", "NumSaturatedCarbocycles",
    "NumAliphaticHeterocycles", "NumAliphaticCarbocycles",
    # Atom counts  (3)
    "NHOHCount", "NOCount", "NumValenceElectrons",
    # Topological indices  (8)
    "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
    "Kappa1", "Kappa2", "Kappa3",
    # Graph-theoretic indices  (4)
    "BalabanJ", "BertzCT", "HallKierAlpha", "Ipc",
    # Partial charges  (4)
    "MinAbsPartialCharge", "MaxAbsPartialCharge",
    "MinPartialCharge", "MaxPartialCharge",
]  # total = 4+3+3+3+10+3+8+4+4 = 42


def fsmol_atom_features(atom) -> np.ndarray:
    """40-dim atom feature vector matching FS-Mol's featurisation convention."""
    feats: list[float] = []

    # One-hot atom type (18 known + 1 UNK = 19 dims)
    symbol = atom.GetSymbol()
    oh = [0.0] * (len(_FSMOL_ATOM_SYMBOLS) + 1)
    try:
        oh[_FSMOL_ATOM_SYMBOLS.index(symbol)] = 1.0
    except ValueError:
        oh[-1] = 1.0  # UNK
    feats += oh

    # Scalar features (7 dims)
    feats.append(float(atom.GetDegree()))
    feats.append(float(atom.GetFormalCharge()))
    feats.append(float(atom.GetNumRadicalElectrons()))
    feats.append(float(atom.GetIsotope()))
    feats.append(float(atom.GetMass()))
    feats.append(float(atom.GetTotalValence()))
    feats.append(float(atom.GetTotalNumHs()))

    # Aromaticity (1 dim)
    feats.append(float(atom.GetIsAromatic()))

    # Ring membership (13 dims: in_any_ring + ring_sizes 3–14)
    ri = atom.GetOwningMol().GetRingInfo()
    feats.append(float(atom.IsInRing()))
    for size in range(3, 15):   # 12 ring sizes
        feats.append(float(ri.IsAtomInRingOfSize(atom.GetIdx(), size)))

    return np.array(feats, dtype=np.float32)


def _compute_rdkit_descriptors_from_mol(mol) -> np.ndarray:
    """Compute DESCRIPTOR_DIM-dim descriptor vector from an RDKit mol object."""
    from rdkit.Chem import Descriptors  # type: ignore
    desc: list[float] = []
    for name in _RDKIT_DESCRIPTOR_LIST:
        try:
            fn = getattr(Descriptors, name)
            val = float(fn(mol))
        except Exception:
            val = 0.0
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        # Clip extreme values (e.g. Ipc can be huge for complex molecules)
        val = float(np.clip(val, -1000.0, 1000.0))
        desc.append(val)
    return np.array(desc, dtype=np.float32)


def compute_rdkit_descriptors(smiles: str) -> Optional[np.ndarray]:
    """Compute 42-dim RDKit descriptor vector from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)   # type: ignore[attr-defined]
    if mol is None:
        return None
    return _compute_rdkit_descriptors_from_mol(mol)


# Bond type → index for FS-Mol encoding (AROMATIC mapped to SINGLE as fallback)
_FSMOL_BOND_TYPE_IDX = {
    rdchem.BondType.SINGLE:   0,
    rdchem.BondType.DOUBLE:   1,
    rdchem.BondType.TRIPLE:   2,
    rdchem.BondType.AROMATIC: 0,  # fallback - shouldn't occur after Kekulize
}


def smiles_to_fsmol_graph(smiles: str) -> Optional["Data"]:
    """
    Convert SMILES to PyG Data with FS-Mol-faithful features.

    Key features:
      - Stereochemistry removed (FS-Mol drops it)
      - Kekulization: aromatic bonds → alternating SINGLE/DOUBLE
      - 40-dim node features (see FSMOL_NODE_FEAT_DIM)
      - 3-dim edge features: SINGLE/DOUBLE/TRIPLE one-hot
      - Graph-level ECFP (1,2048) and descriptor (1,42) tensors included
        (shape (1,*) so PyG's Batch.from_data_list stacks correctly to (n_graphs,*))

    Returns None if SMILES is invalid or Kekulization fails.
    """
    if not _PYGEOM_AVAILABLE:
        raise ImportError("torch_geometric is required for FSMolGNNEncoder.")

    import torch
    from torch_geometric.data import Data as PyGData

    mol = Chem.MolFromSmiles(smiles)   # type: ignore[attr-defined]
    if mol is None:
        return None

    # Compute ECFP and descriptors from the original molecule (before Kekulize)
    ecfp_np = _MORGAN_GENERATOR.GetFingerprintAsNumPy(mol).astype(np.float32)  # type: ignore[attr-defined]
    desc_np  = _compute_rdkit_descriptors_from_mol(mol)

    # Remove stereo information (FS-Mol does this before graph construction)
    Chem.RemoveStereochemistry(mol)   # type: ignore[attr-defined]

    # Kekulize: convert aromatic bonds to alternating SINGLE/DOUBLE
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)   # type: ignore[attr-defined]
    except Exception:
        return None

    if mol.GetNumAtoms() == 0:
        return None

    # Node features
    atom_feats = [fsmol_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.stack(atom_feats), dtype=torch.float)   # (n_atoms, 40)

    # Edge index + bond-type one-hot (bidirectional)
    rows: list[int] = []
    cols: list[int] = []
    edge_types: list[int] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _FSMOL_BOND_TYPE_IDX.get(bond.GetBondType(), 0)
        rows += [i, j]; cols += [j, i]; edge_types += [bt, bt]

    if edge_types:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        et          = torch.tensor(edge_types, dtype=torch.long)
        edge_attr   = torch.zeros(len(edge_types), 3, dtype=torch.float)
        edge_attr.scatter_(1, et.unsqueeze(1), 1.0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 3), dtype=torch.float)

    # Graph-level features: shape (1, D) so Batch stacks to (n_graphs, D)
    ecfp_t = torch.tensor(ecfp_np, dtype=torch.float).unsqueeze(0)   # (1, 2048)
    desc_t = torch.tensor(desc_np,  dtype=torch.float).unsqueeze(0)  # (1, 42)

    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   ecfp=ecfp_t, descriptors=desc_t)


def compute_fsmol_degree_histogram(assay_files: list[str], n_sample: int = 500,
                                   max_degree: int = 10) -> "torch.Tensor":
    """
    Degree histogram for FSMolGNNEncoder PNA scalers, computed from FS-Mol-style graphs
    (Kekulized, no aromatic bonds).
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
                data = smiles_to_fsmol_graph(smi)
                if data is None or data.edge_index.shape[1] == 0:
                    continue
                d = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                d_clamped = d.clamp(max=max_degree)
                deg += torch.bincount(d_clamped, minlength=max_degree + 1)
                n_mols += 1

    print(f"  FS-Mol degree histogram over {n_mols} molecules ({len(files)} assays).")
    return deg
