# Prototypical Networks for Molecular OOD Property Prediction

Few-shot molecular property prediction with out-of-distribution (OOD) generalisation, evaluated on two benchmarks: FS-Mol and DrugOOD.

- **Pretraining:** FS-Mol (~26k assays from ChEMBL), episodic training
- **Evaluation:** DrugOOD benchmark (scaffold, size, and assay distribution shift) + FS-Mol held-out test set
- **Task:** Predict activity (pIC50) for molecules from unseen chemical distributions using a small support set
- **Primary metric:** ΔAUPRC = AUPRC(model) − fraction_actives (0 = random classifier, higher = better)

---

## Background

Standard Prototypical Networks (Snell et al., 2017) are designed for few-shot classification. This project implements two prediction heads:

**Regression head** - Nadaraya-Watson kernel regression in learned embedding space:

$$\hat{y}_q = \sum_{i \in \text{support}} \frac{\exp(-d(f(x_q), f(x_i)) / \tau)}{\sum_j \exp(-d(f(x_q), f(x_j)) / \tau)} \cdot y_i$$

**Classification head** - Binary active/inactive prototypes (BCE loss). Labels are pre-binarised ChEMBL `Property` field (0/1).

The embedding function $f$ is trained episodically on FS-Mol with configurable episode construction (random, scaffold-disjoint, or fingerprint-similarity-disjoint). The pretrained model is then evaluated **zero-shot** on DrugOOD.

---

## Design Choices

| Component | Implemented | Notes |
|---|---|---|
| Encoders | ECFP4 (2048-bit) MLP (`ecfp`); FS-Mol GNN 10-layer (`gnn`) | GNN uses FS-Mol featurisation |
| Heads | Regression (kernel regression, MSE); Classification (binary PN, BCE) | |
| Training splits | `random`; `scaffold` (Murcko-disjoint episodes); `similarity` (Butina@0.70 cluster-disjoint) | Controls intra-assay OOD hardness during episode construction |
| Training distance | `euclidean` or `mahalanobis` (configurable) | Mahalanobis training requires `sigma.detach()` to avoid gradient explosion |
| Eval split types | `random`, `scaffold` (Murcko), `similarity` (Butina@0.70), `size` | Four distinct eval types; "similarity" = Butina@0.70 sphere-exclusion |
| Butina cutoff | 0.70 (fixed) | Justified by `split_ood_characterization.py`: NN-Tanimoto ~0.26 with 95.5% assay retention — best OOD/retention trade-off. Fixed in `config.py`. |
| Primary metric | ΔAUPRC = AUPRC(model) − fraction_actives | FS-Mol paper convention |
| Run tag format | `{enc}_{head}_{split}_{distance}_{nsup}_seed{seed}` | e.g. `gnn_classification_similarity_mahalanobis_163264_seed0` |
| Evaluation | Zero-shot (frozen encoder) on DrugOOD + FS-Mol test | No fine-tuning |

---

## Repository Structure

```
PTN/
├── config.py          # Central path config + design constants (ENV, scaffold split choice)
├── main.py            # Full pipeline entry point
├── model.py           # Encoders + prototypical network heads
├── data.py            # Data loading and episode construction
├── train.py           # Episodic pretraining loop
├── evaluate.py        # Zero-shot evaluation (DrugOOD + FS-Mol test) + head registry
├── featurize.py       # GNN atom/bond featurisation and degree histogram
├── requirements.txt
│
├── Analysis/
│   ├── data/          # Dataset audit scripts → see Analysis/data/README.md
│   └── model/         # Result plotting → see Analysis/model/README.md
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
└── outputs/
    ├── {run_tag}/                 # e.g. gnn_classification_random_euclidean_163264_seed0
    │   ├── checkpoints/model.pt   # trained model
    │   └── csvs/
    │       ├── fsmol_test.csv     # FS-Mol held-out test results
    │       └── drugood.csv        # DrugOOD results
    ├── baselines/csvs/
    │   └── fsmol_test.csv         # model-free ECFP baselines (shared, encoder-independent)
    └── data_analysis/             # split characterization, dataset audit outputs
```

---

## Results Summary

All values are mean ΔAUPRC on the FS-Mol held-out test set (154 qualifying assays, 5 repeats per assay per support size). Model results report mean ± std over 3 independent seeds (0, 1, 2).

> **N-assay note:** All 154 test assays qualify at n≤64. At n=128 approximately 148 qualify, n=256 ~29, n=512 ~11 (need ≥2n molecules after filtering). The apparent plateau or dip at large n partly reflects assay-selection bias (harder, molecule-rich assays remain), not model failure.

---

## Baseline Grid: Representation × Head

The baseline grid answers one question: **what drives performance - the representation (raw ECFP vs learned embedding) or the inference head?**

Every head receives the **same support/query split** for a given (assay, n, repeat), so differences between heads are purely algorithmic, not due to different data. The frozen checkpoint is used only as an encoder; no retraining happens.

### Prediction Heads

| Head | Abbreviation | Representation | Type | How it works |
|---|---|---|---|---|
| ProtoNet (Euclidean) | PN-E | Embedding | Mean-prototype | Class mean in embedding space; Euclidean distance to classify query |
| ProtoNet (Mahalanobis) | PN-M | Embedding | Mean-prototype | Same prototypes; Mahalanobis distance with shrinkage covariance from support set |
| Logistic Regression | LogReg | Embedding | Adaptive | StandardScaler + L2 LogReg (C=1.0, max_iter=1000) fit fresh on each support set |
| k-Nearest Neighbours | kNN | Embedding | Adaptive | StandardScaler + kNN (k=5) fit fresh on each support set |
| ECFP ProtoNet (Euclidean) | ecfp-PN-E | Raw ECFP | Mean-prototype | Class mean fingerprint; Euclidean distance — ad-hoc geometry for bit vectors |
| ECFP ProtoNet (Tanimoto) | ecfp-PN-T | Raw ECFP | Kernel-prototype | Mean Tanimoto similarity to each class's support molecules — domain-canonical geometry |
| ECFP GP (Tanimoto) | ecfp-GP | Raw ECFP | Kernel machine | GaussianProcessClassifier with Tanimoto kernel; no hyperparameter optimisation; every support molecule is a kernel basis function |
| ECFP Logistic Regression | ecfp-LR | Raw ECFP | Adaptive | LogReg on raw ECFP4 - no encoder |
| ECFP Random Forest | RF | Raw ECFP | Adaptive | RandomForest (100 trees, max_depth=10) on raw ECFP4 - strongest non-meta baseline |

**Why these 9?** The grid has two axes — representation and head — plus a geometry ablation within the raw-ECFP row:

```
                 | Mean-prototype          | Kernel machine | Adaptive (fit per task)
-----------------+-------------------------+----------------+------------------------
Raw ECFP         | ecfp-PN-E → ecfp-PN-T  | ecfp-GP        | ecfp-LR, RF
Learned embedding| PN-E, PN-M              | —              | LogReg, kNN
```

**Why PN-E and PN-M separately?** Training uses Euclidean (numerically stable from random init); eval uses Mahalanobis (the FS-Mol paper's standard). Comparing them tests how much the distance metric matters once the embedding is trained.

**Why the three ECFP prototype heads?** They form a deliberate geometry ablation: `ecfp-PN-E` uses Euclidean distance on bit vectors (no domain motivation); `ecfp-PN-T` uses Tanimoto similarity, the domain-canonical metric for ECFP fingerprints; `ecfp-GP` is the optimal kernel machine using the same Tanimoto kernel, generalising the prototype to all support molecules as basis functions. If `ecfp-PN-T` >> `ecfp-PN-E`, the distance geometry matters. If `emb-PN-M` >> `ecfp-GP`, the learned encoder adds value beyond the optimal fingerprint kernel.

**Why ECFP heads alongside embedding heads?** To isolate the encoder's contribution. If ecfp-LR ≈ emb-LogReg, the MLP encoder adds nothing over the raw fingerprint. If emb-PN-M >> ecfp-GP, the encoder's geometry is what matters — the Tanimoto kernel alone is insufficient.

**ecfp_* heads run once** (independent of checkpoint) and are merged with emb_* heads per checkpoint into one CSV.

---

### Table 1 - ECFP Baselines (no meta-training, encoder-independent)

> Source: `outputs/baselines/csvs/fsmol_test.csv`. Values are deterministic — ECFP fingerprints only, no model involved. 154 qualifying assays; n=256/512 subsets are smaller (≈29/11 assays) and harder.

Mean ΔAUPRC across 154 qualifying assays, 5 repeats per assay. Bold = best head per (split × n).

| n | GP-T | RF | PN-T | LR | PN-E | GP-T | RF | PN-T | LR | PN-E | GP-T | RF | PN-T | LR | PN-E | GP-T | RF | PN-T | LR | PN-E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random** | | | | | **Scaffold** | | | | | **Similarity** | | | | | **Size** | | | | |
| 16  | **0.088** | 0.082 | 0.085 | 0.082 | 0.059 | **0.071** | 0.066 | 0.067 | 0.066 | 0.048 | **0.056** | 0.049 | 0.049 | 0.046 | 0.037 | **0.066** | 0.055 | 0.063 | 0.061 | 0.038 |
| 32  | **0.106** | 0.101 | 0.103 | 0.099 | 0.076 | **0.087** | 0.084 | 0.082 | 0.079 | 0.060 | 0.067 | **0.068** | 0.064 | 0.058 | 0.054 | **0.073** | 0.064 | 0.069 | 0.063 | 0.045 |
| 64  | 0.127 | **0.128** | 0.122 | 0.114 | 0.093 | 0.106 | **0.107** | 0.100 | 0.094 | 0.077 | 0.083 | **0.087** | 0.081 | 0.073 | 0.068 | **0.078** | 0.075 | 0.075 | 0.070 | 0.047 |
| 128 | 0.153 | **0.157** | 0.147 | 0.142 | 0.116 | 0.140 | **0.145** | 0.130 | 0.125 | 0.108 | 0.114 | **0.122** | 0.106 | 0.100 | 0.098 | **0.085** | 0.080 | 0.074 | 0.067 | 0.044 |
| 256 | **0.168** | 0.165 | 0.152 | 0.135 | 0.119 | 0.145 | **0.146** | 0.133 | 0.114 | 0.106 | **0.107** | 0.106 | 0.092 | 0.075 | 0.075 | 0.121 | **0.127** | 0.106 | 0.088 | 0.064 |
| 512 | **0.180** | **0.180** | 0.157 | 0.128 | 0.109 | 0.170 | **0.171** | 0.152 | 0.114 | 0.106 | **0.131** | 0.125 | 0.117 | 0.071 | 0.096 | **0.169** | 0.148 | 0.163 | 0.074 | 0.091 |

**Key takeaways:** GP-Tanimoto dominates at small n and on the size/similarity splits; RF takes over at n≥64 on random and scaffold. OOD severity ordering: random > scaffold > size > similarity — matching query-support Tanimoto similarity (0.42 → 0.37 → 0.35 → 0.28). The similarity split (most OOD) costs roughly 0.030–0.040 ΔAUPRC vs random at each n.

---

### Tables 2–4 — GNN Results (FSMolGNNEncoder, n_support=64, Euclidean training)

> Mean ΔAUPRC averaged over 3 seeds (0, 1, 2) and 5 repeats per assay. Bold = best value in that row across all eval splits × heads. Heads: **PN-E** = emb_proto_euclid, **PN-M** = emb_proto_mahalanobis, **LogReg** = emb_logreg, **kNN** = emb_knn.

#### Table 2 — GNN, random training

| n | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random** | | | | **Scaffold** | | | | **Similarity** | | | | **Size** | | | |
| 16  | **0.140** | 0.131 | 0.128 | 0.086 | **0.125** | 0.118 | 0.112 | 0.077 | **0.122** | 0.113 | 0.103 | 0.075 | **0.118** | 0.106 | 0.102 | 0.067 |
| 32  | **0.168** | 0.158 | 0.144 | 0.108 | **0.154** | 0.146 | 0.132 | 0.097 | **0.150** | 0.141 | 0.124 | 0.096 | **0.139** | 0.126 | 0.111 | 0.081 |
| 64  | **0.193** | 0.189 | 0.160 | 0.133 | **0.181** | 0.176 | 0.146 | 0.116 | **0.175** | 0.169 | 0.138 | 0.114 | **0.156** | 0.146 | 0.117 | 0.095 |
| 128 | 0.210 | **0.217** | 0.183 | 0.159 | 0.207 | **0.212** | 0.173 | 0.146 | 0.202 | **0.204** | 0.165 | 0.139 | **0.067** | 0.064 | 0.047 | 0.039 |
| 256 | 0.142 | **0.157** | 0.130 | 0.113 | 0.125 | **0.143** | 0.111 | 0.104 | 0.109 | **0.114** | 0.090 | 0.072 | 0.094 | **0.102** | 0.071 | 0.060 |
| 512 | 0.127 | **0.156** | 0.134 | 0.114 | 0.130 | **0.155** | 0.120 | 0.092 | 0.118 | **0.133** | 0.093 | 0.072 | 0.119 | **0.132** | 0.088 | 0.077 |

#### Table 3 — GNN, scaffold training

| n | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random** | | | | **Scaffold** | | | | **Similarity** | | | | **Size** | | | |
| 16  | **0.134** | 0.127 | 0.123 | 0.084 | **0.120** | 0.114 | 0.106 | 0.075 | **0.118** | 0.111 | 0.103 | 0.076 | **0.116** | 0.105 | 0.100 | 0.065 |
| 32  | **0.162** | 0.155 | 0.140 | 0.107 | **0.148** | 0.142 | 0.126 | 0.094 | **0.147** | 0.137 | 0.121 | 0.096 | **0.134** | 0.123 | 0.107 | 0.079 |
| 64  | **0.188** | 0.187 | 0.157 | 0.131 | **0.175** | 0.174 | 0.143 | 0.114 | **0.168** | 0.162 | 0.134 | 0.114 | **0.152** | 0.143 | 0.114 | 0.093 |
| 128 | 0.204 | **0.213** | 0.178 | 0.156 | 0.203 | **0.210** | 0.174 | 0.144 | 0.196 | **0.198** | 0.163 | 0.136 | **0.062** | 0.061 | 0.044 | 0.038 |
| 256 | 0.136 | **0.154** | 0.128 | 0.115 | 0.121 | **0.137** | 0.107 | 0.100 | 0.106 | **0.118** | 0.102 | 0.078 | 0.087 | **0.094** | 0.064 | 0.057 |
| 512 | 0.119 | **0.148** | 0.124 | 0.110 | 0.123 | **0.149** | 0.118 | 0.091 | 0.101 | **0.121** | 0.084 | 0.069 | 0.105 | **0.120** | 0.087 | 0.074 |

#### Table 4 — GNN, similarity training

| n | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random** | | | | **Scaffold** | | | | **Similarity** | | | | **Size** | | | |
| 16  | **0.135** | 0.126 | 0.122 | 0.084 | **0.122** | 0.113 | 0.106 | 0.074 | **0.119** | 0.112 | 0.104 | 0.074 | **0.116** | 0.102 | 0.097 | 0.065 |
| 32  | **0.162** | 0.154 | 0.138 | 0.105 | **0.149** | 0.141 | 0.123 | 0.093 | **0.148** | 0.139 | 0.123 | 0.093 | **0.134** | 0.120 | 0.102 | 0.079 |
| 64  | **0.186** | 0.184 | 0.154 | 0.129 | **0.174** | 0.171 | 0.139 | 0.113 | **0.168** | 0.162 | 0.134 | 0.109 | **0.149** | 0.140 | 0.107 | 0.090 |
| 128 | 0.204 | **0.211** | 0.177 | 0.156 | 0.200 | **0.207** | 0.168 | 0.141 | 0.196 | **0.202** | 0.163 | 0.137 | **0.065** | 0.064 | 0.042 | 0.038 |
| 256 | 0.136 | **0.154** | 0.129 | 0.113 | 0.121 | **0.139** | 0.111 | 0.097 | 0.107 | **0.116** | 0.094 | 0.078 | 0.087 | **0.096** | 0.070 | 0.056 |
| 512 | 0.118 | **0.145** | 0.129 | 0.104 | 0.124 | **0.146** | 0.123 | 0.088 | 0.107 | **0.124** | 0.092 | 0.067 | 0.103 | **0.120** | 0.087 | 0.065 |

### Table 5 — ECFP Encoder Results

> Pending training runs.

---

### Key Observations

**Baselines (Table 1):**
- GP-Tanimoto is the strongest single ECFP head across most support sizes and splits; RF is comparable at large n and beats it on the size split at n≥256.
- OOD severity ordering by ΔAUPRC: **random > scaffold > size > similarity** — matches query-support Tanimoto similarity ordering exactly (0.42 → 0.37 → 0.35 → 0.28).
- All heads scale monotonically with n on the random split. The similarity split (Butina@0.70, most OOD) costs roughly 0.030–0.040 ΔAUPRC vs random at each n.
- PN-Euclid is the weakest ECFP head; using Tanimoto geometry (PN-Tanimoto) recovers substantial performance, confirming that the distance metric matters for bit-vector fingerprints.
- Size split is anomalous: GP-T and RF recover to ≈0.169 at n=512 (comparable to scaffold) even though the split is harder than scaffold at small n — possibly due to large-molecule bias in the surviving assays.

**GNN results (Tables 2–4):**
- **PN-E beats PN-M at n≤64; PN-M wins at n≥128.** At small n, the covariance estimate is too noisy to help — Euclidean geometry is cleaner. At n≥128, Mahalanobis consistently adds ~0.005–0.007 ΔAUPRC.
- **Random training > scaffold > similarity** across all eval splits, including the OOD ones. Shift-aware training costs ~0.003–0.006 ΔAUPRC relative to random training — the richer IID gradient signal produces better representations.
- **GNN beats best ECFP baseline (RF/GP-T) at n≤128** on all eval splits. At n=64: GNN PN-E 0.193 vs RF 0.128 (random split). At n=256/512 ECFP recovers, driven by assay-selection bias at large n.
- **Size split crashes at n=128** (0.064–0.067 across all training splits, vs ~0.21 on other splits). Size-stratified splits at large n push very small molecules into the query — a qualitatively different regime. Normal scaling resumes at n=256/512.
- **kNN is consistently the weakest embedding head** — the 5-NN estimate is too local given the embedding dimensionality and support size.
- **Seed variance is very low** (range ~0.002 across seeds at any n), confirming stable training.

**ECFP encoder results:** pending (Table 5).

For full per-assay distributions, per-context-size DrugOOD curves, and baseline comparisons, see [Analysis/model/README.md](Analysis/model/README.md). 

For dataset statistics, see [Analysis/data/README.md](Analysis/data/README.md).

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

**FS-Mol** - Download from [microsoft/FS-Mol](https://github.com/microsoft/FS-Mol). Extract to `data/fsmol/`. Each file is one ChEMBL assay; the loader reads precomputed ECFP fingerprints and log-transformed labels directly.

**DrugOOD** - Three IC50 files from the DrugOOD benchmark:
```
data/drugood/
├── lbap_core_ic50_scaffold.json
├── lbap_core_ic50_size.json
└── lbap_core_ic50_assay.json
```

### Environment

Edit `config.py` - change only `ENV`:
```python
ENV = "local"    # "server" for HPC/server runs
```

---

## Running

```bash
# Full pipeline (train + evaluate)
# Edit the CONFIG block in main.py first:
#   ENCODER        = "gnn"         # "ecfp" | "gnn"
#   TRAINING_SPLIT = "random"      # "random" | "scaffold" | "similarity"
#   TRAIN_DISTANCE = "euclidean"   # "euclidean" | "mahalanobis"
#   N_SUPPORT      = [16, 32, 64]  # int or list[int]
#   SEED           = 0
python main.py

# Line plot: mean ± std ΔAUPRC vs support size (edit CONFIG block in script)
python Analysis/model/plot_line_grid.py

# Boxplot: per-assay ΔAUPRC distribution vs support size (same CONFIG pattern)
python Analysis/model/plot_boxplot_grid.py

# Data analysis (no model needed)
python Analysis/data/dataset_overview.py
python Analysis/data/scaffold_analysis.py
```

---

## References

- Snell et al. (2017) - [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- Stanley et al. (2021) - [FS-Mol: A Few-Shot Learning Dataset of Molecules](https://openreview.net/forum?id=701FtuyLlAd)
- Ji et al. (2022) - [DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery](https://arxiv.org/abs/2201.09637)
- Corso et al. (2020) - [Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718)
- 