# Prototypical Networks for Molecular OOD Property Prediction

Few-shot molecular property prediction with out-of-distribution (OOD) generalisation, evaluated across two benchmarks: FS-Mol and DrugOOD.

- **Pretraining:** FS-Mol (~26k assays from ChEMBL), episodic training
- **Evaluation:** DrugOOD benchmark (scaffold, size, and assay shift) + FS-Mol held-out test set
- **Task:** Predict activity (pIC50) for molecules from unseen chemical distributions using a small support set

---

## Background

Standard Prototypical Networks (Snell et al., 2017) are designed for few-shot classification. This project implements two heads:

**Regression head** — Nadaraya-Watson kernel regression in learned embedding space:

$$\hat{y}_q = \sum_{i \in \text{support}} \frac{\exp(-d(f(x_q), f(x_i)) / \tau)}{\sum_j \exp(-d(f(x_q), f(x_j)) / \tau)} \cdot y_i$$

**Classification head** — Binary active/inactive prototypes with BCE loss. Threshold = support set median.

The embedding function $f$ is trained episodically on FS-Mol with **shift-aware episodes**: support and query molecules come from different Bemis-Murcko scaffold families within the same assay, forcing the embedding to generalise across chemical scaffolds. The pretrained model is then evaluated **zero-shot** on DrugOOD's OOD test splits.

---

## Design Choices

| Component | Implemented | Notes |
|---|---|---|
| Encoders | ECFP4 (2048-bit) MLP; PNA GNN (6-layer) | GNN uses FS-Mol node/edge featurisation (51-dim nodes, 12-dim edges) |
| Heads | Regression (kernel regression, MSE); Classification (binary PN, BCE) | 4 combinations total |
| Training splits | Shift-aware (scaffold OOD episodes); Random | Controlled comparison |
| Distance | Squared Euclidean | FS-Mol paper uses Mahalanobis |
| Temperature | Learnable scalar `log_τ` | Regression head only |
| Primary metric | ΔAUPRC = AUPRC(model) − fraction_actives | Follows FS-Mol paper convention |
| Evaluation | Zero-shot (frozen encoder) on DrugOOD + FS-Mol test | No fine-tuning |

---

## Repository Structure

```
PTN/
├── config.py          # Central path config — edit ENV to switch environments
├── main.py            # Full pipeline entry point
├── model.py           # Encoders + prototypical network heads
├── data.py            # Data loading and episode construction
├── train.py           # Episodic pretraining loop
├── evaluate.py        # Zero-shot evaluation (DrugOOD + FS-Mol test)
├── featurize.py       # GNN atom/bond featurisation and degree histogram
├── requirements.txt
│
├── Analysis/
│   ├── data/          # Dataset audit scripts → see Analysis/data/README.md
│   └── model/         # Result plotting and baselines → see Analysis/model/README.md
│
├── data/
│   ├── fsmol/
│   │   ├── train/     # ~26,868 .jsonl.gz assay files
│   │   ├── valid/     # 40 .jsonl.gz assay files
│   │   └── test/      # 157 .jsonl.gz assay files
│   └── drugood/
│       ├── lbap_core_ic50_scaffold.json
│       ├── lbap_core_ic50_size.json
│       └── lbap_core_ic50_assay.json
│
├── checkpoints/       # One .pt file per (encoder, head, split) combination
└── outputs/
    ├── figures/       # All plots saved here
    └── results/       # All CSV results saved here
```

---

## File Reference

### `config.py`
Central path configuration. **Only file to edit when switching environments.**

```python
ENV = "local"   # change to "server" for HPC runs
```

Exports one checkpoint path per model combination (`PTN_{ENCODER}_{HEAD}_{SPLIT}_CHECKPOINT`) and directory paths for `CHECKPOINT_DIR`, `FIGURES_DIR`, `RESULTS_DIR`.

---

### `model.py`

| Class | Description |
|---|---|
| `ECFPEncoder` | 3-layer MLP: 2048 → 512 → 256. Input is a 2048-bit ECFP4 fingerprint. |
| `PNAGNNEncoder` | 6-layer PNA GNN with 4 towers, aggregators [mean, min, max, std], scalers [identity, amplification, attenuation]. Output: 256-dim embedding via global mean pooling + MLP projection. |
| `PrototypicalNetworkRegression` | Wraps any encoder. Kernel regression with learnable temperature. MSE loss. |
| `PrototypicalNetworkClassification` | Wraps any encoder. Binary active/inactive prototypes. BCE loss. Labels binarised at support set median. |

---

### `data.py`

| Class / Function | Description |
|---|---|
| `AssayDataset` | One FS-Mol assay: fingerprints, labels, SMILES, scaffold groups, binary labels |
| `DrugOODEvalDataset` | One DrugOOD file: context pool, ood_test, iid_test (with SMILES for GNN) |
| `FSMolEpisodeDataset` | Episodic PyTorch Dataset for ECFP training |
| `FSMolGraphEpisodeDataset` | Episodic Dataset for GNN training (returns PyG Batch objects) |
| `get_scaffold(smiles)` | Bemis-Murcko scaffold SMILES — used to group molecules into scaffold families |

---

### `featurize.py`

Atom and bond featurisation for the GNN encoder.

| Export | Description |
|---|---|
| `NODE_FEAT_DIM = 51` | Atom features: type, degree, formal charge, H count, hybridisation, aromaticity, ring membership |
| `EDGE_FEAT_DIM = 12` | Bond features: 4 bond types + conjugated + in-ring + 6 stereo |
| `smiles_to_graph(smiles)` | SMILES → PyG `Data` object |
| `compute_degree_histogram(assay_files, n_sample)` | Samples assays to compute node degree histogram for PNA scalers |

---

### `train.py`

| Function | Description |
|---|---|
| `pretrain_regression(encoder, ...)` | Episodic training, MSE loss, early stopping on Val RMSE (patience=25) |
| `pretrain_classification(encoder, ...)` | Episodic training, BCE loss, early stopping on Val ΔAUPRC (patience=25) |
| `set_seed(seed)` | Seeds Python, NumPy, PyTorch, CUDA for full reproducibility |

Training config: Adam optimiser, `ReduceLROnPlateau` (factor=0.5, patience=20), BF16 mixed precision (A100), `num_workers=2` for train DataLoader.

LR defaults: `1e-4` for GNN (prevents BCE lock at ln(2)), `1e-3` for ECFP.

---

### `evaluate.py`

| Function | Description |
|---|---|
| `evaluate_drugood_multiscale(model, ...)` | Context sizes [16, 32, 64, 128, 256, 512] × 3 seeds × OOD/IID query sets |
| `evaluate_fsmol_test(model, ...)` | Support-size sweep [16, 32, 64, 128, 256, 512] × 5 repeats, split_type ∈ {random, scaffold, size} |
| `evaluate_inside_task_ood(model, ...)` | Within-assay scaffold split at fixed n_support=16 |
| `load_and_evaluate(checkpoint_path, ...)` | Load checkpoint and run DrugOOD evaluation |
| `delta_auprc(preds, binary_labels)` | AUPRC(model) − fraction_positives |

For classification models, RMSE / MAE / Spearman are suppressed (set to NaN) — predictions are probabilities ∈ [0,1] and have no meaningful relationship to continuous IC50 targets.

---

### `main.py`

Full pipeline entry point. Edit the config block at the top to select which model to run:

```python
MODEL_HEAD     = "classification"   # "regression" | "classification"
ENCODER        = "gnn"              # "ecfp" | "gnn"
TRAINING_SPLIT = "shift_aware"      # "shift_aware" | "random"
SEED           = 42
SKIP_TRAINING  = False              # True = load existing checkpoint, skip pretraining
```

Steps run in order:
1. Index FS-Mol assay files
2. Build encoder + pretrain (or load checkpoint if `SKIP_TRAINING=True`)
3. Evaluate zero-shot on DrugOOD (3 shift types × 6 context sizes)
4. Evaluate on FS-Mol test set (3 split types × 6 support sizes + inside-task OOD)

All output files are tagged with `run_tag = f"{ENCODER}_{MODEL_HEAD}_{TRAINING_SPLIT}"`.

---

## Setup

### Requirements

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
rdkit>=2023.3.1
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
psutil
```

```bash
pip install -r requirements.txt
```

### Data

**FS-Mol** — Download from [microsoft/FS-Mol](https://github.com/microsoft/FS-Mol). Extract to `data/fsmol/`. Each file is one ChEMBL assay; the loader reads precomputed ECFP fingerprints (`"fingerprints"`) and log-transformed labels (`"LogRegressionProperty"`) directly.

**DrugOOD** — Three IC50 files from the DrugOOD benchmark:
```
data/drugood/
├── lbap_core_ic50_scaffold.json
├── lbap_core_ic50_size.json
└── lbap_core_ic50_assay.json
```

### Environment

Edit `config.py` — change only `ENV`:
```python
ENV = "local"    # "server" for HPC/server runs
```

---

## Running

```bash
# Full pipeline (train + evaluate) — configure MODEL_HEAD/ENCODER/TRAINING_SPLIT in main.py first
python main.py

# Generate figures from saved CSVs
python Analysis/model/plot_results.py --run_tag gnn_classification_shift_aware

# Run baseline diagnostic (ECFP regression vs kNN / KR-Tanimoto)
python Analysis/model/diagnostic_baseline.py

# Data analysis (no model needed)
python Analysis/data/dataset_overview.py
python Analysis/data/scaffold_analysis.py
python Analysis/data/chemical_diversity.py
```

---

## Results Summary

Eight model combinations: 2 encoders × 2 heads × 2 training splits. Evaluated on 154 FS-Mol test assays and 3 DrugOOD shift types. Primary metric: ΔAUPRC (higher = better; 0 = random classifier).

| Model | FS-Mol random (n=128) | FS-Mol scaffold (n=128) | DrugOOD assay OOD (ctx=256) | Status |
|---|---|---|---|---|
| ECFP / regression / shift_aware | 0.062 | 0.019 | +0.021 | ✓ done |
| GNN / classification / shift_aware | 0.061 | 0.017 | +0.050 | ✓ done |
| ECFP / classification / shift_aware | — | — | — | pending |
| GNN / regression / shift_aware | — | — | — | pending |
| ECFP / regression / random | — | — | — | pending |
| ECFP / classification / random | — | — | — | pending |
| GNN / regression / random | — | — | — | pending |
| GNN / classification / random | — | — | — | pending |

For full per-support-size tables, per-assay distributions, DrugOOD curves by shift type, and baseline comparisons, see [Analysis/model/README.md](Analysis/model/README.md).

For dataset statistics and chemical diversity figures, see [Analysis/data/README.md](Analysis/data/README.md).

---

## References

- Snell et al. (2017) — [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- Stanley et al. (2021) — [FS-Mol: A Few-Shot Learning Dataset of Molecules](https://openreview.net/forum?id=701FtuyLlAd)
- Ji et al. (2022) — [DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery](https://arxiv.org/abs/2201.09637)
- Corso et al. (2020) — [Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718)
