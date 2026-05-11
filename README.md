# Prototypical Networks for Molecular OOD Regression

Implementation of Prototypical Networks adapted for regression on molecular property prediction, with out-of-distribution (OOD) generalization evaluation across two benchmarks: FS-Mol and DrugOOD.

- **Pretraining:** FS-Mol (~26k assays from ChEMBL)
- **Evaluation:** DrugOOD benchmark (scaffold, size, and assay shift) + FS-Mol held-out test set
- **Task:** Predict continuous activity values (pIC50) for molecules from unseen chemical distributions

---

## Background

Standard Prototypical Networks (Snell et al., 2017) are designed for few-shot classification. This implementation adapts them for regression via kernel regression in learned embedding space:

$$\hat{y}_q = \sum_{i \in \text{support}} \frac{\exp(-d(f(x_q), f(x_i)))}{\sum_j \exp(-d(f(x_q), f(x_j)))} \cdot y_i$$

The embedding function $f$ is trained episodically on FS-Mol. Episodes are **shift-aware**: support and query molecules come from different scaffold families within the same assay, forcing the embedding to generalise across chemical scaffolds. The pretrained model is then evaluated **zero-shot** on DrugOOD's OOD test splits.

---

## Design Choices

| Component | Chosen | Alternative |
|---|---|---|
| Encoder | ECFP4 (2048-bit) + MLP | GNN (GIN/MPNN), ChemBERTa |
| Distance | Euclidean | Learned MLP metric |
| Temperature | Learnable scalar | Fixed = 1.0 |
| Loss | MSE | MAE, Huber |
| Episodes | Shift-aware (scaffold split) | Random split |
| Evaluation | Zero-shot (frozen encoder) | Fine-tuned |
| Primary metric | ΔAUPRC | RMSE, Spearman ρ |

---

## Repository Structure

```
PTN/
├── config.py                        # Central path config — edit ENV to switch environments
├── main.py                          # Full pipeline entry point
├── model.py                         # Encoder + prototypical regression network
├── data.py                          # Data loading, episode construction
├── train.py                         # Episodic training loop (FS-Mol pretraining)
├── evaluate.py                      # Zero-shot evaluation (DrugOOD + FS-Mol test)
├── requirements.txt
├── .gitignore
│
├── Analysis/
│   ├── data/
│   │   ├── dataset_overview.py      # Dataset stats + data-loss audit
│   │   ├── scaffold_analysis.py     # Per-task scaffold diversity across splits
│   │   ├── structural_variability.py # Molecule-level feature computation (library)
│   │   └── chemical_diversity.py    # FS-Mol vs DrugOOD comparison (Tanimoto, t-SNE)
│   └── model/
│       ├── diagnostic_baseline.py   # PTN vs kNN / KR-Tanimoto diagnostic
│       └── plot_results.py          # Figures from evaluation CSVs
│
├── data/
│   ├── fsmol/
│   │   ├── train/                   # ~26,868 .jsonl.gz assay files
│   │   ├── valid/                   # 40 .jsonl.gz assay files
│   │   └── test/                    # 157 .jsonl.gz assay files
│   └── drugood/
│       ├── lbap_core_ic50_scaffold.json
│       ├── lbap_core_ic50_size.json
│       └── lbap_core_ic50_assay.json
│
├── checkpoints/
│   └── pretrained_model.pt          # Saved after best validation RMSE
│
└── outputs/
    ├── figures/                     # All plots saved here
    └── results/                     # All CSV results saved here
```

---

## File Reference

### `config.py`
Central path configuration. **This is the only file you need to edit when switching environments.**

```python
ENV = "local"   # change to "server" to use server paths
```

| Export | Description |
|---|---|
| `FSMOL_DIR` | Root of FS-Mol dataset |
| `FSMOL_TRAIN / VAL / TEST` | Derived split directories |
| `DRUGOOD_DIR` | Root of DrugOOD JSON files |
| `MODEL_SAVE_PATH` | Path to `pretrained_model.pt` |
| `CHECKPOINT_DIR` | Directory for model checkpoints |
| `FIGURES_DIR` | Output directory for all plots |
| `RESULTS_DIR` | Output directory for all CSVs |

---

### `model.py`
Defines the encoder and the prototypical regression network.

**Classes:**

| Class | Description |
|---|---|
| `MolecularEncoder` | 3-layer MLP: `input_dim → hidden_dim → hidden_dim → embedding_dim`. Input is a 2048-bit ECFP4 fingerprint. Dropout = 0.2. |
| `PrototypicalNetworkRegression` | Wraps `MolecularEncoder`. Predicts query labels via softmax-weighted kernel regression over support embeddings. Temperature is a learned scalar. |

**Key methods on `PrototypicalNetworkRegression`:**

| Method | Description |
|---|---|
| `forward(support_fp, support_labels, query_fp)` | Returns predicted labels for query molecules. |
| `compute_loss_batched(s_fp, s_lbl, q_fp, q_lbl)` | MSE loss + RMSE metric over a batch of episodes. |

**Default hyperparameters:** `hidden_dim=512`, `embedding_dim=256`

---

### `data.py`
Data loading and episode construction for FS-Mol and DrugOOD.

**Classes:**

| Class | Description |
|---|---|
| `AssayDataset` | Holds one FS-Mol assay: fingerprints, labels, scaffold group index. |
| `DrugOODEvalDataset` | Holds one DrugOOD split file: train pool, iid_test, ood_test. |
| `FSMolEpisodeDataset` | PyTorch Dataset that streams episodic batches from disk. Used by the training loop. |
| `EpisodeSampler` | Samples one (support, query) episode from an `AssayDataset`. Supports random and shift-aware (scaffold) strategies. |

**Key functions:**

| Function | Description |
|---|---|
| `mol_to_fingerprint(smiles)` | SMILES → 2048-bit ECFP4 count vector (float32 numpy array). Returns `None` on invalid SMILES. |
| `get_scaffold(smiles)` | Returns generic Bemis-Murcko scaffold SMILES. Used to group molecules into scaffold families. |
| `_load_assay_file(filepath)` | Load one `.jsonl.gz` file into an `AssayDataset`. Only keeps `Relation == "="` (exact measurements). |

**Episode construction (shift-aware):**
Support is sampled from one scaffold group, query from a different group. This forces the embedding to be scaffold-invariant — the central training signal for OOD generalisation.

**Minimum task size:** `MIN_TASK_SIZE = 32` (assays with fewer exact-measurement molecules are skipped).

---

### `train.py`
Episodic training loop for FS-Mol pretraining.

**Functions:**

| Function | Description |
|---|---|
| `train_epoch(model, loader, optimizer, device)` | One epoch: MSE loss over a batch of episodes with gradient clipping (`max_norm=1.0`). Returns `{"loss", "rmse"}`. |
| `validate(model, loader, device)` | Validation RMSE over one epoch. |
| `pretrain(train_files, val_files, ...)` | Full training loop with LR scheduling and early stopping. Saves best checkpoint to `MODEL_SAVE_PATH`. |

**Training details:**
- Optimiser: Adam, `lr=1e-3`
- LR scheduler: ReduceLROnPlateau (`factor=0.5`, `patience=20`)
- Early stopping: patience = 40 epochs (2× LR patience), triggers on validation RMSE
- Checkpoint: saved whenever validation RMSE improves

---

### `evaluate.py`
Zero-shot evaluation on DrugOOD and FS-Mol test set.

**Metrics:**

| Metric | Description |
|---|---|
| ΔAUPRC | `AUPRC(model) − fraction_actives`. Primary metric (FS-Mol paper convention). Positive = better than random. |
| Spearman ρ | Rank correlation between predicted and actual labels. |
| RMSE | Root mean squared error on log-activity labels. |
| MAE | Mean absolute error. |

**Functions:**

| Function | Description |
|---|---|
| `delta_auprc(preds, binary_labels)` | Computes ΔAUPRC. Binary labels are derived from the median activity threshold. |
| `spearman_correlation(preds, targets)` | Spearman ρ, returns `nan` if fewer than 3 predictions. |
| `evaluate_drugood_multiscale(model, drugood_dir, ...)` | Runs evaluation across all 3 DrugOOD shift types × multiple context sizes × iid/ood_test. Saves `drugood_results.csv`. |
| `evaluate_fsmol_test(model, test_files, ...)` | Evaluates on FS-Mol test assays across random/scaffold/size split types and multiple support sizes. Saves `fsmol_test_results.csv`. |
| `evaluate_inside_task_ood(model, test_files, ...)` | Scaffold-split inside-task OOD evaluation on FS-Mol. Saves `inside_task_ood_results.csv`. |
| `load_and_evaluate(checkpoint_path, ...)` | Load model from checkpoint and run full DrugOOD evaluation. |

**Context / support sizes evaluated:**
- DrugOOD: 16, 32, 64, 128, 256, 512
- FS-Mol test: 16, 32, 64, 128, 256, 512

---

### `main.py`
Full pipeline entry point. Runs all four steps in sequence.

```bash
python main.py
```

**Steps:**

| Step | Description |
|---|---|
| 1 | Index FS-Mol files from `FSMOL_TRAIN`, `FSMOL_VAL`, `FSMOL_TEST` |
| 2 | Pretrain Prototypical Network on FS-Mol train assays |
| 3 | Evaluate zero-shot on all DrugOOD IC50 shifts |
| 4a | Evaluate on FS-Mol test set (random / scaffold / size splits) |
| 4b | Inside-task OOD evaluation on FS-Mol test |

**To skip training and load a saved checkpoint** (e.g. to only re-run evaluation), comment out Step 2 in `main.py`. The model is loaded from `MODEL_SAVE_PATH`.

**Outputs written:**
- `outputs/results/fsmol_test_results.csv`
- `outputs/results/drugood_results.csv`
- `outputs/results/inside_task_ood_results.csv`
- `checkpoints/pretrained_model.pt`

---

## Analysis Scripts

All scripts import paths from `config.py` and write outputs to `FIGURES_DIR` / `RESULTS_DIR`.

### `Analysis/data/dataset_overview.py`
Data audit and dataset statistics. Run this before training to understand the data.

**What it does:**
- Scans every FS-Mol assay file and counts total / inexact / invalid / exact molecules
- Reports what fraction of each assay survives the `Relation == "="` filter
- Computes fraction-active per assay (for ΔAUPRC baseline)
- Reports DrugOOD split sizes (train / iid_test / ood_test per shift type)
- Generates Figures 1a, 1b, 1c (assay size distributions, fraction-active distributions, domain sizes)

**Outputs:**
| File | Description |
|---|---|
| `results/assay_sizes_fsmol.csv` | Per-assay molecule counts for all FS-Mol splits |
| `results/data_loss_per_assay.csv` | Per-assay inexact / invalid drop counts |
| `figures/fig1a_assay_sizes.png` | Assay size distributions (train / valid / test) |
| `figures/fig1b_fraction_actives.png` | Fraction-active distribution across assays |
| `figures/fig1c_domain_sizes.png` | DrugOOD domain size distributions |
| `figures/fig_size_vs_fraction_exact.png` | Assay size vs fraction of exact measurements |

```bash
python Analysis/data/dataset_overview.py
```

---

### `Analysis/data/scaffold_analysis.py`
Per-task scaffold diversity across all FS-Mol splits.

**What it does:**
- For each assay passing `MIN_TASK_SIZE=32`, computes number of unique Bemis-Murcko scaffolds and scaffold diversity ratio (`n_unique_scaffolds / n_molecules`)
- Runs on test + valid (full scan) and train (sampled, default 2000 files)
- Histogram plots for both metrics across splits

**Outputs:**
| File | Description |
|---|---|
| `results/scaffold_diversity_per_task_all_splits.csv` | Per-assay scaffold diversity (all splits) |
| `figures/fig_scaffold_diversity_per_task.png` | Histogram grid: unique scaffolds and diversity ratio |

```bash
python Analysis/data/scaffold_analysis.py
```

---

### `Analysis/data/structural_variability.py`
**Library file — not run directly.** Imported by `chemical_diversity.py`.

Provides molecule-level feature computation using RDKit:

| Function | Description |
|---|---|
| `compute_mol_features(smiles)` | Computes molecular mass, heavy atom count, rotatable bonds, aromatic rings, generic scaffold for one SMILES. |
| `compute_structural_variability(smiles_list)` | Applies `compute_mol_features` to a list and returns a DataFrame. |
| `summarize_variability(df)` | Prints mean/std/range summary of the feature DataFrame. |

---

### `Analysis/data/chemical_diversity.py`
Chemical space comparison between FS-Mol and DrugOOD. Three complementary analyses.

**What it does:**

| Section | Description |
|---|---|
| 1 — Molecular properties | Mean molecular mass, heavy atoms, rotatable bonds, aromatic rings for FS-Mol train/test and DrugOOD train/ood_test. Reports generic scaffold overlap between datasets. |
| 2 — Tanimoto distances | Pairwise Tanimoto distance distributions: FS-Mol internal, DrugOOD internal, cross-dataset. Higher cross-distance = more OOD shift. |
| 3 — t-SNE | 2D projection of 5000 FS-Mol + 5000 DrugOOD molecules using ECFP4 with Jaccard/Tanimoto metric. |

**Outputs:**
| File | Description |
|---|---|
| `results/structural_var_comparison.csv` | Mean molecular properties per dataset |
| `figures/tanimoto_distances.png` | Tanimoto distance distribution histogram |
| `figures/tsne_fsmol_vs_drugood.png` | t-SNE coloured by dataset source |

```bash
python Analysis/data/chemical_diversity.py
```

---

### `Analysis/model/diagnostic_baseline.py`
Supervisor diagnostic: compare PTN vs simple baselines on the same episodes.

**What it does:**
Samples 100 train assays × 10 episodes × 2 split types (random + scaffold) and runs:

| Baseline | Description |
|---|---|
| Mean-label | Predicts mean(support labels) for all queries. Trivial baseline. |
| kNN (k=1,3,5) | sklearn KNeighborsRegressor on raw ECFP fingerprints. |
| KR-Tanimoto (α=0.01/0.1/1.0) | Kernel ridge regression with Tanimoto kernel on raw ECFPs. The key comparison: same kernel regression as PTN but in *raw* fingerprint space instead of *learned* embedding space. |
| PTN | Loaded from `MODEL_SAVE_PATH`. Kernel regression in learned embedding space. |

**Interpretation:**
- `PTN MSE < best-KR-Tanimoto` → learned embedding improves over raw fingerprints ✓
- `PTN MSE ≈ best-KR-Tanimoto` → embedding adds no information over raw fingerprints
- `PTN MSE >> best-KR-Tanimoto` → training is broken
- `All methods ≈ mean-label` → ECFP fingerprints carry no signal for this episode type

**Outputs:**
| File | Description |
|---|---|
| `results/diagnostic_baseline.csv` | Per-episode MSE for all methods |

```bash
python Analysis/model/diagnostic_baseline.py
```

---

### `Analysis/model/plot_results.py`
Generates all result figures from the CSVs produced by `main.py`. Run after `main.py` has completed.

**Figures generated:**

| Figure | Function | Description |
|---|---|---|
| `fig2a_fsmol_line_plot.png` | `plot_fsmol_line` | Three curves (random / scaffold / size split) for ΔAUPRC, Spearman ρ, RMSE vs support size. Error bars = ±1 std across assays. |
| `fig2b_fsmol_boxplot.png` | `plot_fsmol_boxplot` | Per-assay distribution of ΔAUPRC and Spearman ρ across support sizes, grouped by split type. |
| `fig3_drugood_line_plot.png` | `plot_drugood_line` | Spearman ρ and ΔAUPRC vs context size, faceted by shift type (scaffold / size / assay). OOD and IID test as separate lines. |

**Inputs required:**
- `outputs/results/fsmol_test_results.csv`
- `outputs/results/drugood_results.csv`

```bash
python Analysis/model/plot_results.py
```

---

## Setup

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
rdkit>=2023.3.1
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
```

```bash
pip install -r requirements.txt
```

### Data Setup

**FS-Mol** — Download from the [FS-Mol GitHub](https://github.com/microsoft/FS-Mol). Extract to `data/fsmol/`. Each file is one ChEMBL assay; the loader reads precomputed ECFP fingerprints (`"fingerprints"` field) and log-transformed labels (`"LogRegressionProperty"`) directly — no RDKit required for loading.

**DrugOOD** — Download the pre-built JSON datasets from the [DrugOOD Google Drive](https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC). This project uses three files:
```
data/drugood/
├── lbap_core_ic50_scaffold.json   # scaffold-based OOD split
├── lbap_core_ic50_size.json       # molecule-size-based OOD split
└── lbap_core_ic50_assay.json      # assay-condition-based OOD split
```

### Environment Configuration

Edit `config.py` — change only the `ENV` variable:
```python
ENV = "local"    # use "server" for HPC/server runs
```
All paths are derived automatically.

---

## Running the Pipeline

```bash
# 1. Full pipeline (train + evaluate)
python main.py

# 2. Evaluation only (load saved checkpoint, skip training)
#    Comment out Step 2 in main.py, then:
python main.py

# 3. Generate plots from saved CSVs
python Analysis/model/plot_results.py

# 4. Run baseline diagnostic
python Analysis/model/diagnostic_baseline.py

# 5. Data analysis scripts (run independently, no model needed)
python Analysis/data/dataset_overview.py
python Analysis/data/scaffold_analysis.py
python Analysis/data/chemical_diversity.py
```

---

## Results

> Results will be filled in after final evaluation runs.

### FS-Mol Test Set

**ΔAUPRC by split type and support size:**

| Support size | Random | Scaffold | Size |
|---|---|---|---|
| 16 | — | — | — |
| 32 | — | — | — |
| 64 | — | — | — |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |

**Spearman ρ by split type and support size:**

| Support size | Random | Scaffold | Size |
|---|---|---|---|
| 16 | — | — | — |
| 32 | — | — | — |
| 64 | — | — | — |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |

---

### DrugOOD Benchmark

**ΔAUPRC (OOD test):**

| Context size | IC50 Scaffold | IC50 Size | IC50 Assay |
|---|---|---|---|
| 16 | — | — | — |
| 32 | — | — | — |
| 64 | — | — | — |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |

**Spearman ρ (OOD test):**

| Context size | IC50 Scaffold | IC50 Size | IC50 Assay |
|---|---|---|---|
| 16 | — | — | — |
| 32 | — | — | — |
| 64 | — | — | — |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |

---

### Baseline Diagnostic (PTN vs KR-Tanimoto)

> Run `Analysis/model/diagnostic_baseline.py` to generate.

| Method | Random split MSE | Scaffold split MSE |
|---|---|---|
| Mean-label | — | — |
| kNN (k=1) | — | — |
| kNN (k=3) | — | — |
| kNN (k=5) | — | — |
| KR-Tanimoto (best α) | — | — |
| PTN (checkpoint) | — | — |

---

## References

- Snell et al. (2017) — [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- Stanley et al. (2021) — [FS-Mol: A Few-Shot Learning Dataset of Molecules](https://openreview.net/forum?id=701FtuyLlAd)
- Ji et al. (2022) — [DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery](https://arxiv.org/abs/2201.09637)
