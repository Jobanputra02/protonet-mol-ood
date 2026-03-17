# Prototypical Networks for Molecular OOD Regression

Implementation of Prototypical Networks adapted for regression on molecular property prediction, with out-of-distribution (OOD) generalization evaluation.

- **Pretraining:** FS-Mol (~26k assays from ChEMBL)
- **Evaluation:** DrugOOD benchmark (scaffold, size, and assay shift)
- **Task:** Predict continuous activity values (IC50) for molecules from unseen chemical distributions

## Background

Standard Prototypical Networks (Snell et al., 2017) are designed for few-shot classification. This implementation adapts them for regression via kernel regression in learned embedding space:

$$\hat{y}_q = \sum_{i \in \text{support}} \frac{\exp(-d(f(x_q), f(x_i)))}{\sum_j \exp(-d(f(x_q), f(x_j)))} \cdot y_i$$

The embedding function f is trained episodically on FS-Mol. Episodes are constructed to be shift-aware: support and query molecules come from different scaffold families within the same assay, forcing the embedding to be scaffold-invariant. The pretrained model is then evaluated zero-shot on DrugOOD's OOD test splits.


## Design Choices

| Component | Chosen |
|---|---|
| Encoder | ECFP4 (2048-bit) + MLP |
| Distance | Euclidean |
| Temperature | Learnable scalar |
| Loss | MSE |
| Episodes | Shift-aware (scaffold split) |
| Evaluation | Zero-shot (frozen encoder) |


## Repository Structure

```
.
├── data
    ├── drugood/          # DrugOOD pre-built JSON files (3 files used)
    │   ├── lbap_core_ic50_scaffold.json   # Scaffold shift split
    │   ├── lbap_core_ic50_size.json       # Molecular size shift split
    │   └── lbap_core_ic50_assay.json      # Assay condition shift split
    └── fs-mol/           # FS-Mol assay files (.jsonl.gz, one file per ChEMBL assay)
        ├── train/        # ~26,868 assays used for episodic pretraining
        ├── valid/        # 40 assays used for validation during pretraining
        └── test/         # 157 assays (not used in this work)
├── model.py        # Encoder, distance function, prototypical regression network
├── data.py         # Episode construction, AssayDataset, DrugOODEvalDataset
├── train.py        # Episodic training loop (FS-Mol pretraining)
├── evaluate.py     # Zero-shot OOD evaluation (DrugOOD)
├── main.py         # Full pipeline: data loading, pretraining, evaluation
└── requirements.txt
```

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
rdkit>=2023.3.1
scipy>=1.10.0
```

Install:
```bash
pip install torch numpy scipy rdkit
```

## Data Setup

### FS-Mol

1. Clone the FS-Mol repository:
```bash
git clone https://github.com/microsoft/FS-Mol.git
```

2. Download the dataset from the [FS-Mol GitHub README](https://github.com/microsoft/FS-Mol) (direct download link in the README). Extract to a local directory:
```
data/fs-mol/
├── train/    # ~26,868 .jsonl.gz files
├── valid/    # 40 .jsonl.gz files
└── test/     # 157 .jsonl.gz files
```

Each file is one ChEMBL assay. The loader reads precomputed ECFP fingerprints (`"fingerprints"` field) and log-transformed activity values (`"LogRegressionProperty"`) directly — no RDKit required for FS-Mol loading.

### DrugOOD

Download the pre-built datasets (96 JSON files, based on ChEMBL 30) from:
[https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC](https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC)

This implementation uses three files:
```
data/drugood/
├── lbap_core_ic50_scaffold.json
├── lbap_core_ic50_size.json
└── lbap_core_ic50_assay.json
```

`lbap` = ligand-based activity prediction, `core` = standard benchmark subset, `ic50` = activity type.


## Usage

### 1. Update paths in `main.py`

```python
FS_MOL_DIR  = "path/to/data/fs-mol"
DRUGOOD_DIR = "path/to/data/drugood"
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
1. Load FS-Mol train and validation assays
2. Pretrain the Prototypical Network with shift-aware episodes
3. Evaluate zero-shot on all three DrugOOD IC50 splits

### 3. Quick test

To verify the pipeline before full training, set these in `main.py`:

```python
max_assays=200          # in load_fsmol_split() calls
n_epochs=3              # in pretrain() call
n_episodes_train=100    # in pretrain() call
```



## Results

Results from pretraining on ~5k FS-Mol assays (50 epochs, 1000 episodes/epoch):

| DrugOOD Split | RMSE | MAE | Spearman |
|---|---|---|---|
| IC50 Scaffold | 1.43 | 1.21 | 0.014 |
| IC50 Size | 1.21 | 1.03 | 0.126 |
| IC50 Assay | 1.44 | 1.19 | 0.129 |

**Notes:**
- Scaffold shift is the hardest split (Spearman ≈ 0), consistent with findings in the DrugOOD paper
- Results are from ~18% of available training data due to RAM constraints — full training expected to improve performance
- Context set: 64 molecules sampled from `ood_val` with fixed seed 42 for reproducibility


## References

- Snell et al. (2017) — [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- Stanley et al. (2021) — [FS-Mol: A Few-Shot Learning Dataset of Molecules](https://openreview.net/forum?id=701FtuyLlAd)
- Ji et al. (2022) — [DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery](https://arxiv.org/abs/2201.09637)