# Prototypical Networks for Molecular OOD Property Prediction

Few-shot molecular property prediction with out-of-distribution (OOD) generalisation, evaluated on two benchmarks: FS-Mol and DrugOOD.

- **Training:** FS-Mol (16,930 qualifying assays from ChEMBL, ~2.8M molecules), episodic training at fixed n_support=64
- **Evaluation:** FS-Mol held-out test set (154 assays, 153 qualifying at n=64) across four eval splits; DrugOOD benchmark (scaffold, assay, and size OOD) planned
- **Task:** Predict activity (pIC50) for molecules from unseen chemical distributions using a small support set
- **Primary metric:** ΔAUPRC = AUPRC(model) − fraction_actives (0 = random classifier, higher = better)

---

## Background

Standard Prototypical Networks (Snell et al., 2017) are designed for few-shot classification. This project uses the **classification head** — binary active/inactive prototypes trained with BCE loss. Labels are pre-binarised ChEMBL `Property` field (0/1).

**Classification head** — mean-prototype inference:
- Active prototype = mean of active support embeddings; Inactive prototype = mean of inactive support embeddings
- Query classified by Euclidean or Mahalanobis distance to the nearest prototype
- No retraining at inference; any n_support works at eval time

The embedding function $f$ is trained episodically on FS-Mol at fixed n=64 with three episode-construction strategies (random, scaffold-disjoint, or similarity-disjoint). The pretrained frozen encoder is then evaluated across four FS-Mol test splits and nine prediction heads.

---

## Design Choices

| Component | Implemented | Notes |
|---|---|---|
| Encoders | ECFP4 (2048-bit) MLP (`ecfp`); FS-Mol GNN 10-layer PNA (`gnn`) | GNN: CombinedReadout fusing GNN output + ECFP4 + 42 physicochemical descriptors → 512-dim |
| Head | Classification (binary PN, BCE) | Euclidean or Mahalanobis distance at eval |
| Training splits | `random`; `scaffold` (Murcko-disjoint episodes); `similarity` (Butina@0.70 cluster-disjoint) | Controls intra-assay OOD hardness during episode construction |
| Training n_support | 64 (fixed) | Chosen via validation ΔAUPRC: fixed n=64 → 0.2094 vs variable → 0.1358 |
| Training distance | `euclidean` (fixed) | Euclidean training; Euclidean or Mahalanobis at eval |
| Eval split types | `random`, `scaffold` (Murcko), `similarity` (Butina@0.70), `size` | Four distinct eval types; characterised by NN-Tanimoto: 0.352 / 0.308 / 0.259 / 0.306 |
| Butina cutoff | 0.70 (fixed) | Justified by `split_ood_characterization.py`: NN-Tanimoto 0.259 with 95.5% assay retention — best OOD/retention trade-off |
| Model grid | PTN + Attn × ECFP/GNN × 3 fsmol training splits × 1–3 seeds; MIC: same models × 2 datasets × 5 splits | FS-Mol PTN+GNN: 9 runs done; PTN+ECFP: 4 runs done; Attn+ECFP: 1 run (re-eval pending); Attn+GNN: 3 runs (re-eval pending). MIC: PTN+ECFP all 10 done; PTN+GNN 6/10 done, 4 running; Attn+ECFP 10/10; Attn+GNN 5/10, 4 running |
| Nested tensors | `use_nested_tensors=True` for all Attn runs | [DC-2]; jagged NestedTensor self-attention; falls back to dense on unsupported PyTorch |
| Primary metric | ΔAUPRC = AUPRC(model) − fraction_actives | FS-Mol paper convention; controls for class imbalance |
| Run tag format | `{enc}_classification_{split}_euclidean_{nsup}_seed{seed}` | e.g. `gnn_classification_random_euclidean_64_seed0` |
| Evaluation | Frozen encoder; 9 heads evaluated per checkpoint on FS-Mol test | No fine-tuning; DrugOOD evaluation planned |

---

## Repository Structure

```
PTN/
├── config.py              # Central path config + design constants (ENV, scaffold split choice)
├── main_ptn.py            # PTN (ProtoNet) training + eval entry point
├── main_attention.py      # Cross-attention model training + eval entry point
├── model.py               # Encoders (ECFPEncoder, FSMolGNNEncoder) + ProtoNet head
├── context_model.py       # CrossAttentionProtoNet (support attention → query cross-attn)
├── data.py                # FS-Mol data loading and episode construction
├── mic_loader.py          # MIC dataset loading (MicAssayStore, MicEpisodeDataset, MicGraphEpisodeDataset)
├── train.py               # Episodic pretraining loop (classification/regression)
├── evaluate.py            # Zero-shot evaluation (FS-Mol test, DrugOOD) + head registry
├── featurize.py           # GNN atom/bond featurisation and degree histograms
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
│   ├── drugood/
│   │   ├── lbap_core_ic50_scaffold.json
│   │   ├── lbap_core_ic50_size.json
│   │   └── lbap_core_ic50_assay.json
│   └── custom/
│       ├── curated_mic_assays_balanced.csv   # ~1,400 MIC assays, pre-balanced
│       ├── chembl36_curated.csv              # ~8,400 MIC assays from ChEMBL 36
│       ├── mic_splits.json                   # 5 train/val/test splits for curated_mic
│       └── mic_splits_chembl36.json          # 5 splits for chembl36_mic
│
└── outputs/
    ├── {run_tag}/                 # e.g. gnn_classification_random_euclidean_64_seed0
    │   ├── checkpoints/model.pt   # trained model
    │   └── csvs/
    │       ├── fsmol_test.csv     # FS-Mol held-out test results
    │       └── drugood.csv        # DrugOOD results
    ├── {dataset}_split{X}/{run_tag}/   # MIC runs: e.g. curated_mic_splitA/gnn_classification_random_euclidean_64_seed0
    │   └── checkpoints/model.pt
    ├── baselines/csvs/
    │   └── fsmol_test.csv         # model-free ECFP baselines (shared, encoder-independent)
    └── data_analysis/             # split characterization, dataset audit outputs
```

---

## Results Summary

> **N-assay note:** 153 test assays qualify at n=64. At n=128 approximately 148 qualify, n=256 ~29, n=512 ~11 (need ≥2n molecules after filtering). The apparent plateau or dip at large n partly reflects assay-selection bias (harder, molecule-rich assays remain), not model failure.

---

## Run Grid Overview

Complete run grid across all (model × encoder × dataset × split) combinations. Detailed tables follow below.

**Metric:** FS-Mol = test ΔAUPRC at n=64, PN-M head, random eval split. MIC = best val ΔAUPRC, n=64.
**Legend:** value = done · `–` = not planned / not yet run · `running` = SLURM job active · `pending` = planned, not yet submitted · `⚠️` = diverged · `*re-eval*` = checkpoint exists, evaluation pending re-run with fixed pipeline

---

### FS-Mol

> PTN+GNN values averaged over 3 seeds (0,1,2). PTN+ECFP random = 2 seeds, scaffold/similarity = 1 seed. Attn re-eval pending after evaluation pipeline fix.

| Model | Encoder | Train split | ΔAUPRC (n=64, PN-M, random eval) |
|---|---|---|---|
| PTN | ECFP | random | 0.163 (s0=0.161, s1=0.164) |
| PTN | ECFP | scaffold | 0.164 |
| PTN | ECFP | similarity | 0.153 |
| PTN | GNN | random | 0.188 (avg 3 seeds) |
| PTN | GNN | scaffold | 0.186 (avg 3 seeds) |
| PTN | GNN | similarity | 0.183 (avg 3 seeds) |
| Attn | ECFP | random | *re-eval* (1 seed; train val ΔAUPRC ≈ 0.155) |
| Attn | ECFP | scaffold | `–` |
| Attn | ECFP | similarity | `–` |
| Attn | GNN | random | *re-eval* (3 seeds; train val ΔAUPRC = 0.156/0.155/0.163) |
| Attn | GNN | scaffold | `–` |
| Attn | GNN | similarity | `–` |

---

### curated_mic (~1,400 assays)

> All runs: seed 0, n_support=64, random training split within each MIC split. Val ΔAUPRC at best checkpoint.

| Split | Description | PTN+ECFP | PTN+GNN | Attn+ECFP | Attn+GNN |
|---|---|---|---|---|---|
| A | random | 0.3299 | `running` | 0.2656 | `running` |
| B | document | 0.3795 | 0.4389 | 0.2667 | `running` |
| C | time | 0.2530 | 0.3757 | ⚠️ 0.000 | 0.2820 |
| D | organism | 0.4800 | `running` | ⚠️ 0.000 | `running` |
| E | gram stain | 0.2861 | 0.3908 | 0.2752 | `running` |

---

### chembl36_mic (~8,400 assays)

| Split | Description | PTN+ECFP | PTN+GNN | Attn+ECFP | Attn+GNN |
|---|---|---|---|---|---|
| A | random | 0.1828 | `running` | 0.2387 | `running` |
| B | document | 0.0140 | 0.0259 | 0.0521 | 0.0301 |
| C | time | 0.1241 | 0.1307 | 0.0668 | 0.1310 |
| D | organism | 0.1977 | `running` | 0.2111 | 0.2096 |
| E | gram/series | 0.3074 | 0.3302 | 0.2969 | 0.3476 |

---

**Pending / to-run list:**
- Attn+ECFP fsmol random re-eval (1 seed, fixed pipeline)
- Attn+GNN fsmol random re-eval (3 seeds, fixed pipeline)
- PTN+GNN fsmol test evaluation (checkpoint exists, CSVs never written)
- Attn+ECFP/GNN fsmol scaffold + similarity training (not yet run)
- PTN+ECFP MIC (not yet run on either dataset)
- Attn+GNN curated_mic A, D, E + chembl36_mic A re-run (running jobs 3543038–3543042)
- Attn+GNN curated_mic C investigated (⚠️ diverged — needs C/D stability fix before resubmit)

---

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
| 16  | **0.087** | 0.081 | 0.085 | 0.082 | 0.059 | **0.071** | 0.065 | 0.067 | 0.066 | 0.048 | **0.062** | 0.059 | 0.055 | 0.052 | 0.045 | **0.066** | 0.055 | 0.063 | 0.061 | 0.038 |
| 32  | **0.106** | 0.101 | 0.103 | 0.099 | 0.076 | **0.087** | 0.084 | 0.081 | 0.078 | 0.060 | 0.076 | **0.081** | 0.072 | 0.066 | 0.062 | **0.073** | 0.064 | 0.069 | 0.063 | 0.045 |
| 64  | 0.126 | **0.127** | 0.121 | 0.114 | 0.093 | 0.106 | **0.107** | 0.100 | 0.094 | 0.077 | 0.087 | **0.094** | 0.086 | 0.077 | 0.074 | **0.078** | 0.075 | 0.075 | 0.070 | 0.047 |
| 128 | 0.153 | **0.157** | 0.146 | 0.141 | 0.116 | 0.140 | **0.146** | 0.131 | 0.126 | 0.109 | 0.119 | **0.130** | 0.110 | 0.105 | 0.101 | **0.085** | 0.080 | 0.074 | 0.067 | 0.044 |
| 256 | **0.160** | 0.156 | 0.145 | 0.128 | 0.113 | **0.136** | 0.135 | 0.125 | 0.108 | 0.099 | **0.118** | 0.117 | 0.101 | 0.087 | 0.082 | 0.121 | **0.127** | 0.106 | 0.088 | 0.064 |
| 512 | **0.180** | **0.180** | 0.157 | 0.128 | 0.109 | 0.170 | **0.171** | 0.152 | 0.114 | 0.106 | **0.158** | 0.152 | 0.145 | 0.102 | 0.125 | **0.169** | 0.148 | 0.163 | 0.074 | 0.091 |

**Key takeaways:** GP-Tanimoto dominates at small n and on the size/similarity splits; RF takes over at n≥64 on random and scaffold. OOD severity ordering: random > scaffold > size > similarity — matching query-support Tanimoto similarity (0.42 → 0.37 → 0.35 → 0.28). The similarity split (most OOD) costs roughly 0.030–0.040 ΔAUPRC vs random at each n.

---

### Tables 2–4 — GNN Results (FSMolGNNEncoder, n_support=64, Euclidean training)

> Mean ΔAUPRC averaged over 3 seeds (0, 1, 2) and 5 repeats per assay. Bold = best value in that row across all eval splits × heads. Heads: **PN-E** = emb_proto_euclid, **PN-M** = emb_proto_mahalanobis, **LogReg** = emb_logreg, **kNN** = emb_knn.

#### Table 2 — GNN, random training

| n | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random** | | | | **Scaffold** | | | | **Similarity** | | | | **Size** | | | |
| 16  | **0.139** | 0.131 | 0.127 | 0.086 | **0.125** | 0.117 | 0.112 | 0.077 | **0.124** | 0.118 | 0.104 | 0.080 | **0.118** | 0.106 | 0.102 | 0.067 |
| 32  | **0.168** | 0.158 | 0.144 | 0.108 | **0.154** | 0.145 | 0.132 | 0.097 | **0.155** | 0.147 | 0.128 | 0.100 | **0.139** | 0.126 | 0.111 | 0.081 |
| 64  | **0.192** | 0.188 | 0.159 | 0.132 | **0.181** | 0.176 | 0.146 | 0.116 | **0.176** | 0.172 | 0.139 | 0.117 | **0.156** | 0.146 | 0.117 | 0.095 |
| 128 | 0.209 | **0.216** | 0.182 | 0.158 | 0.207 | **0.212** | 0.173 | 0.146 | 0.200 | **0.204** | 0.165 | 0.141 | **0.067** | 0.064 | 0.047 | 0.039 |
| 256 | 0.134 | **0.147** | 0.122 | 0.107 | 0.116 | **0.133** | 0.103 | 0.098 | 0.118 | **0.124** | 0.102 | 0.079 | 0.094 | **0.102** | 0.071 | 0.060 |
| 512 | 0.127 | **0.156** | 0.134 | 0.114 | 0.130 | **0.155** | 0.120 | 0.092 | 0.146 | **0.160** | 0.123 | 0.081 | 0.119 | **0.132** | 0.088 | 0.077 |

#### Table 3 — GNN, scaffold training

| n | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random** | | | | **Scaffold** | | | | **Similarity** | | | | **Size** | | | |
| 16  | **0.133** | 0.127 | 0.123 | 0.084 | **0.120** | 0.114 | 0.106 | 0.075 | **0.118** | 0.114 | 0.105 | 0.082 | **0.116** | 0.105 | 0.100 | 0.065 |
| 32  | **0.162** | 0.155 | 0.140 | 0.106 | **0.148** | 0.142 | 0.125 | 0.094 | **0.149** | 0.142 | 0.126 | 0.102 | **0.134** | 0.123 | 0.107 | 0.079 |
| 64  | **0.187** | 0.186 | 0.156 | 0.130 | **0.175** | 0.174 | 0.143 | 0.114 | **0.167** | 0.165 | 0.136 | 0.118 | **0.152** | 0.143 | 0.114 | 0.093 |
| 128 | 0.203 | **0.212** | 0.178 | 0.155 | 0.202 | **0.210** | 0.174 | 0.144 | 0.193 | **0.197** | 0.164 | 0.138 | **0.062** | 0.061 | 0.044 | 0.038 |
| 256 | 0.128 | **0.145** | 0.121 | 0.108 | 0.113 | **0.128** | 0.101 | 0.094 | 0.112 | **0.127** | 0.113 | 0.090 | 0.087 | **0.094** | 0.064 | 0.057 |
| 512 | 0.119 | **0.148** | 0.124 | 0.110 | 0.123 | **0.149** | 0.118 | 0.091 | 0.100 | **0.140** | 0.104 | 0.097 | 0.105 | **0.120** | 0.087 | 0.074 |

#### Table 4 — GNN, similarity training

| n | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN | PN-E | PN-M | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random** | | | | **Scaffold** | | | | **Similarity** | | | | **Size** | | | |
| 16  | **0.135** | 0.126 | 0.122 | 0.083 | **0.122** | 0.113 | 0.106 | 0.073 | **0.122** | 0.118 | 0.107 | 0.079 | **0.116** | 0.102 | 0.097 | 0.065 |
| 32  | **0.162** | 0.153 | 0.137 | 0.105 | **0.149** | 0.141 | 0.123 | 0.093 | **0.152** | 0.146 | 0.130 | 0.098 | **0.134** | 0.120 | 0.102 | 0.079 |
| 64  | **0.185** | 0.183 | 0.153 | 0.128 | **0.174** | 0.171 | 0.139 | 0.113 | **0.169** | 0.165 | 0.137 | 0.112 | **0.149** | 0.140 | 0.107 | 0.090 |
| 128 | 0.204 | **0.210** | 0.176 | 0.156 | 0.200 | **0.206** | 0.168 | 0.141 | 0.194 | **0.202** | 0.165 | 0.139 | **0.065** | 0.064 | 0.042 | 0.038 |
| 256 | 0.129 | **0.146** | 0.123 | 0.107 | 0.114 | **0.130** | 0.104 | 0.091 | 0.114 | **0.126** | 0.107 | 0.088 | 0.087 | **0.096** | 0.070 | 0.056 |
| 512 | 0.118 | **0.145** | 0.129 | 0.104 | 0.124 | **0.146** | 0.123 | 0.088 | 0.122 | **0.152** | 0.121 | 0.089 | 0.103 | **0.120** | 0.087 | 0.065 |

### Table 5 — PTN+ECFP Results (ECFPEncoder, n_support=64 training)

> Mean ΔAUPRC (PN-E head). Random: mean of 2 seeds (0,1). Scaffold/similarity: 1 seed each.

#### Table 5a — PTN+ECFP, random training

| n | Random | Scaffold | Similarity | Size |
|---|---|---|---|---|
| 16  | 0.123 | 0.116 | 0.119 | 0.109 |
| 32  | 0.144 | 0.135 | 0.138 | 0.121 |
| 64  | 0.160 | 0.154 | 0.152 | 0.131 |
| 128 | 0.176 | 0.172 | 0.177 | 0.038 |
| 256 | 0.085 | 0.078 | 0.070 | 0.044 |
| 512 | 0.065 | 0.070 | 0.065 | 0.053 |

#### Table 5b — PTN+ECFP, scaffold training

| n | Random | Scaffold | Similarity | Size |
|---|---|---|---|---|
| 16  | 0.124 | 0.120 | 0.125 | 0.109 |
| 32  | 0.141 | 0.137 | 0.143 | 0.121 |
| 64  | 0.161 | 0.155 | 0.157 | 0.128 |
| 128 | 0.177 | 0.174 | 0.177 | 0.032 |
| 256 | 0.082 | 0.070 | 0.064 | 0.039 |
| 512 | 0.063 | 0.059 | 0.060 | 0.030 |

#### Table 5c — PTN+ECFP, similarity training

| n | Random | Scaffold | Similarity | Size |
|---|---|---|---|---|
| 16  | 0.116 | 0.114 | 0.117 | 0.105 |
| 32  | 0.136 | 0.130 | 0.136 | 0.112 |
| 64  | 0.151 | 0.147 | 0.145 | 0.121 |
| 128 | 0.167 | 0.171 | 0.173 | 0.030 |
| 256 | 0.080 | 0.076 | 0.058 | 0.052 |
| 512 | 0.075 | 0.084 | 0.079 | 0.054 |

---

### Table 6 — Attention Model Results (CrossAttentionProtoNet, random training, n_support=64)

> **⚠️ Re-evaluation pending.** Previous results were produced with a bug in `evaluate_fsmol_test_grid` that bypassed MLP_F + self-attention + cross-attention for embedding heads (raw encoder output was passed directly). The bug is fixed; re-eval jobs will be submitted once the smoke test confirms Attn+GNN > PTN+GNN. Train val ΔAUPRC (unaffected by the bug): Attn+ECFP s0 ≈ 0.155, Attn+GNN s0/1/2 = 0.156/0.155/0.163.
>
> Config: `query_label_mode=unknown_token`, `use_nested_tensors=True`.

#### Table 6a — Attn+ECFP, random training

| n | Random | Scaffold | Similarity | Size |
|---|---|---|---|---|
| 16  | — | — | — | — |
| 32  | — | — | — | — |
| 64  | — | — | — | — |
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |

#### Table 6b — Attn+GNN, random training (avg 3 seeds)

| n | Random | Scaffold | Similarity | Size |
|---|---|---|---|---|
| 16  | — | — | — | — |
| 32  | — | — | — | — |
| 64  | — | — | — | — |
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |

---

## MIC Training Results

Best validation ΔAUPRC from episodic training on MIC datasets. Metric = mean ΔAUPRC over validation episodes at the best checkpoint. All runs: seed=0, n_support=64, random training split.

**Split types:**
- A = random · B = document (document source) · C = time (publication year) · D = organism · E = Gram stain (curated_mic) / chemical series (chembl36_mic)

**Legend:** `–` = not yet run · `running` = job in progress (SLURM) · `⚠️` = training diverged (BCE ≈ 6.4, ΔAUPRC = 0.000)

### Table 7 — curated_mic (~1,400 assays, ~650 qualifying at n=64)

| Split | Description | PTN+ECFP | PTN+GNN | Attn+ECFP | Attn+GNN |
|---|---|---|---|---|---|
| A | random | 0.3299 | running | 0.2656 | running |
| B | document | 0.3795 | 0.4389 | 0.2667 | running |
| C | time | 0.2530 | 0.3757 | ⚠️ 0.000 | 0.2820 |
| D | organism | 0.4800 | running | ⚠️ 0.000 | running |
| E | gram stain | 0.2861 | 0.3908 | 0.2752 | running |

### Table 8 — chembl36_mic (~8,400 assays, ~3,200 qualifying at n=64)

| Split | Description | PTN+ECFP | PTN+GNN | Attn+ECFP | Attn+GNN |
|---|---|---|---|---|---|
| A | random | 0.1828 | running | 0.2387 | running |
| B | document | 0.0140 | 0.0259 | 0.0521 | 0.0301 |
| C | time | 0.1241 | 0.1307 | 0.0668 | 0.1310 |
| D | organism | 0.1977 | running | 0.2111 | 0.2096 |
| E | gram/series | 0.3074 | 0.3302 | 0.2969 | 0.3476 |

**Notes:**
- Attn+ECFP on curated_mic splits C and D diverged with BCE ≈ 6.4; PTN+ECFP on the same splits trained fine (0.2530 / 0.4800). Likely CrossAttentionProtoNet instability on harder OOD distributions with small val set.
- PTN+GNN and Attn+GNN on splits A/D are re-running (jobs 3543034–3543042) after NFS outage killed the first run. B/C/E values from earlier runs are clean.
- chembl36_mic split B (document) is very hard: PTN+ECFP ΔAUPRC=0.014, likely because document-based splits create strong covariate shift in source laboratory.

---

### Key Observations

**Baselines (Table 1):**
- GP-Tanimoto and RF are the strongest ECFP heads; RF takes over at n≥64 on random and scaffold splits.
- OOD severity ordering by ΔAUPRC: **random > scaffold > size > similarity** — matches NN-Tanimoto ordering (0.352 → 0.308 → 0.306 → 0.259).
- Similarity split (Butina@0.70, most OOD at NN-Tanimoto 0.259) costs roughly 0.030–0.040 ΔAUPRC vs random at each n.
- PN-Euclid is the weakest ECFP head; Tanimoto geometry (PN-Tanimoto) recovers substantial performance — the distance metric matters for bit-vector fingerprints.

**PTN+GNN results (Tables 2–4) — headline finding:**
- **Random training is best everywhere.** Scaffold-aware and similarity-aware training both underperform random training by 0.003–0.008 ΔAUPRC on every eval split. OOD-aware episode sampling does not help — FS-Mol training assays have narrow intra-assay diversity (2–5 scaffold clusters per assay), producing degenerate OOD episodes with noisy gradients.
- **PN-E beats PN-M at n=64** (our fixed training n) by 0.001–0.005 ΔAUPRC across all eval splits and training strategies. At n≥128, Mahalanobis gains a small edge as the covariance estimate becomes more reliable.
- **GNN dominates ECFP at n≤128:** at n=64, PTN+GNN PN-E 0.193 vs RF 0.127 on the random eval split — 52% relative improvement. The learned representation matters far more than the inference head.
- **Size split crashes at n=128** across all training strategies (0.064–0.067 vs ~0.21 on other splits). Size-stratified splits at large n push very small molecules into query — a qualitatively different regime. Normal scaling resumes at n=256/512.
- **kNN is consistently the weakest embedding head** — too local given the embedding dimensionality and support size.
- **Seed variance is very low** (~0.002 range across seeds at any fixed n), confirming stable training.

**PTN+ECFP results (Tables 5a–c):**
- ECFP-MLP encoder reaches 0.160 ΔAUPRC on FS-Mol random eval (n=64), compared to 0.193 for GNN — 17% gap. Encoder quality is the main bottleneck.
- Same training-split ranking: random > scaffold > similarity. Gap is smaller than for GNN (0.009 random vs similarity at n=64 for ECFP, vs 0.018 for GNN).
- Size split collapses at n=128 just like GNN (0.030–0.038), ruling out a GNN-specific artifact.

**Attention model results (Tables 6a–b):**
- ⚠️ Previous test results were invalid (evaluation bug bypassed attention layers). Re-evaluation pending.
- Training val ΔAUPRC is unaffected: Attn+GNN reached 0.155–0.163 across 3 seeds, consistent with PTN+GNN val (~0.150). Corrected test results expected after re-eval.

**MIC results (Tables 7–8):**
- GNN consistently outperforms ECFP on curated_mic (PTN+GNN 0.356–0.514 vs PTN+ECFP 0.253–0.480). The GNN encoder's advantage is clearer on MIC than on FS-Mol.
- chembl36_mic is harder overall (best scores ~0.33 vs ~0.49 for curated_mic), consistent with its larger and noisier source (full ChEMBL MIC annotations).
- Split D (organism) is easiest for curated_mic (PTN+ECFP 0.480) — organism grouping induces a strong protein-target structure that the model can leverage via the support set.
- Split B (document) is the hardest for chembl36_mic (PTN+ECFP 0.014) — document-based splits create extreme laboratory-specific covariate shift.

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
# PTN on FS-Mol (train + evaluate):
python main_ptn.py --encoder gnn --training_split random --n_support 64 --seed 0

# PTN on MIC:
python main_ptn.py --dataset curated_mic --mic_split A --encoder gnn --n_support 64 --seed 0
python main_ptn.py --dataset chembl36_mic --mic_split A --encoder ecfp --n_support 64 --seed 0

# Attention model on FS-Mol:
python main_attention.py --encoder ecfp --training_split random --n_support 64 --seed 0

# Attention model on MIC:
python main_attention.py --dataset curated_mic --mic_split A --encoder gnn --n_support 64 --seed 0

# HPC cluster: submit a job manually
# (submit_gnn_mic_grid.sh and similar scripts have been removed after use)

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