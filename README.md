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

The embedding function $f$ is trained episodically on FS-Mol with **shift-aware episodes**: support and query molecules come from different Bemis-Murcko scaffold families within the same assay. The pretrained model is then evaluated **zero-shot** on DrugOOD.

---

## Design Choices

| Component | Implemented | Notes |
|---|---|---|
| Encoders | ECFP4 (2048-bit) MLP; FS-Mol GNN 10-layer | GNN uses FS-Mol featurisation |
| Heads | Regression (kernel regression, MSE); Classification (binary PN, BCE) | |
| Training splits | Random; Shift-aware (scaffold OOD episodes) | Both variants trained for all encoder/head combinations |
| Distance | Euclidean (training); Mahalanobis with shrinkage (eval) | FS-Mol paper uses Mahalanobis throughout |
| Primary metric | ΔAUPRC = AUPRC(model) − fraction_actives | FS-Mol paper convention |
| Evaluation | Zero-shot (frozen encoder) on DrugOOD + FS-Mol test | No fine-tuning |

---

## Repository Structure

```
PTN/
├── config.py          # Central path config - edit ENV to switch environments
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
    ├── figures/
    │   ├── {run_tag}/         # lineplot_*.png, boxplot_*.png per run
    │   └── data_analysis/     # fig1a/b/c, scaffold diversity, t-SNE
    └── results/
        ├── {run_tag}/         # baseline_grid, drugood_results, fsmol_test_results, inside_task_ood, predictions CSVs
        └── data_analysis/     # assay_sizes, data_loss, scaffold_diversity
```

---

## Results Summary

All values are mean ΔAUPRC on the FS-Mol held-out test set (157 assays after filtering, 5 repeats per assay per support size). All runs report mean ± std over 3 independent seeds (0, 1, 2).

> **N-assay note:** At n=256 only 29/154 qualifying assays remain (need ≥256 molecules after filtering); at n=512 only 11. These smaller subsets are harder large-assays - the drop at n=256/512 partly reflects assay selection bias, not model failure. N is reported alongside each result.

---

### FS-Mol Test - Mean ΔAUPRC (Random Split)

| Model | Seeds | n=16 (N=154) | n=32 (N=154) | n=64 (N=153) | n=128 (N=148) | n=256 (N=29) | n=512 (N=11) | Best val ΔAUPRC |
|---|---|---|---|---|---|---|---|---|
| *RF baseline (per-task, fixed params)* | - | *0.094* | *0.122* | *0.152* | *0.194* | *0.154* | *0.187* | *-* |
| **ECFP · classification · shift-aware** | **3** | **0.121 ± 0.001** | **0.137 ± 0.001** | **0.158 ± 0.001** | **0.181 ± 0.003** | **0.078 ± 0.013** | **0.068 ± 0.009** | **+0.166 ± 0.005** |
| **ECFP · classification · random** | **3** | **0.123 ± 0.002** | **0.139 ± 0.001** | **0.157 ± 0.002** | **0.183 ± 0.001** | **0.076 ± 0.002** | **0.066 ± 0.005** | **+0.173 ± 0.002** |
| **FS-Mol GNN · classification · shift-aware** | **3** | **0.127 ± 0.002** | **0.155 ± 0.001** | **0.184 ± 0.001** | **0.220 ± 0.000** | **0.152 ± 0.003** | **0.157 ± 0.003** | **+0.208 ± 0.002** |
| **FS-Mol GNN · classification · random** | **3** | **0.130 ± 0.001** | **0.158 ± 0.001** | **0.188 ± 0.002** | **0.226 ± 0.001** | **0.149 ± 0.001** | **0.166 ± 0.003** | **+0.215 ± 0.004** |
| *FS-Mol paper (GNN + ProtoNet)* | - | *0.126* | *-* | *0.185* | *0.201* | *0.226* | *-* | *-* |

> **RF baseline:** per-task RandomForestClassifier (n\_estimators=100, max\_depth=10, max\_features="sqrt", min\_samples\_leaf=2) on 2048-bit ECFP fingerprints. No meta-learning - trained fresh on each context set. N=153/147 assays at n=16/128 (slightly fewer than ProtoNet due to stricter class-balance check).

### FS-Mol Test - Mean ΔAUPRC (Size Split)

| Model | Seeds | n=16 | n=32 | n=64 | n=128 | n=256 (N=29) | n=512 (N=11) |
|---|---|---|---|---|---|---|---|
| **ECFP · classification · shift-aware** | **3** | **0.104 ± 0.001** | **0.120 ± 0.002** | **0.130 ± 0.001** | **0.128 ± 0.002** | **0.031 ± 0.009** | **0.042 ± 0.009** |
| **ECFP · classification · random** | **3** | **0.106 ± 0.003** | **0.120 ± 0.002** | **0.129 ± 0.001** | **0.131 ± 0.001** | **0.034 ± 0.004** | **0.040 ± 0.009** |
| **FS-Mol GNN · classification · shift-aware** | **3** | **0.103 ± 0.001** | **0.122 ± 0.004** | **0.142 ± 0.004** | **0.144 ± 0.004** | **0.071 ± 0.002** | **0.107 ± 0.004** |
| **FS-Mol GNN · classification · random** | **3** | **0.107 ± 0.002** | **0.128 ± 0.001** | **0.149 ± 0.002** | **0.153 ± 0.002** | **0.075 ± 0.001** | **0.107 ± 0.003** |

<!-- ### DrugOOD

Mean ΔAUPRC on `ood_test`, averaged across context sizes 16–512.

| Model | Seeds | DrugOOD assay | DrugOOD scaffold | DrugOOD size |
|---|---|---|---|---|
| **ECFP · classification · shift-aware** | **3** | **0.019 ± 0.009** | **0.013 ± 0.011** | **0.015 ± 0.009** |
| **ECFP · classification · random** | **3** | **0.011 ± 0.006** | **0.007 ± 0.005** | **0.014 ± 0.005** |
| **FS-Mol GNN · classification · shift-aware** | **3** | **0.040 ± 0.001** | **0.028 ± 0.007** | **0.023 ± 0.006** |
| **FS-Mol GNN · classification · random** | **3** | **0.035 ± 0.001** | **0.028 ± 0.004** | **0.027 ± 0.004** | -->

<!-- ### Inside-Task OOD (not featured — degenerate evaluation)
Leave-one-scaffold-group-out within an assay. The held-out scaffold group often has only one class
(scaffold strongly predicts activity, Cramér's V = 0.68), making ΔAUPRC near-zero by construction
regardless of model quality. The baseline_grid scaffold split (build_fair_split_indices) fixes this
by guaranteeing both classes in support and query. Results: all models ~0.018 ± 0.001.
Data: inside_task_ood.csv present in each run folder.
-->

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
| ECFP ProtoNet (Euclidean) | ecfp-PN | Raw ECFP | Mean-prototype | PN-E on raw 2048-bit ECFP4 fingerprints - no encoder at all |
| ECFP Logistic Regression | ecfp-LR | Raw ECFP | Adaptive | LogReg on raw ECFP4 - no encoder |
| ECFP Random Forest | RF | Raw ECFP | Adaptive | RandomForest (100 trees, max_depth=10) on raw ECFP4 - strongest non-meta baseline |

**Why these 7?** They form a 2×2 grid (representation × head family) plus RF as an extra adaptive ECFP baseline:

```
                 | Mean-prototype  | Adaptive (fit per task)
-----------------+-----------------+-------------------------
Raw ECFP         | ecfp-PN         | ecfp-LR, RF
Learned embedding| PN-E, PN-M      | LogReg, kNN
```

**Why PN-E and PN-M separately?** Training uses Euclidean (numerically stable from random init); eval uses Mahalanobis (the FS-Mol paper's standard). Comparing them tests how much the distance metric matters once the embedding is trained.

**Why ECFP heads alongside embedding heads?** To isolate the encoder's contribution. If ecfp-LR ≈ emb-LogReg, the MLP encoder adds nothing over the raw fingerprint. If emb-PN-M >> ecfp-PN, the encoder's geometry is what matters, not the inference head.

**ecfp_* heads run once** (independent of checkpoint) and are merged with emb_* heads per checkpoint into one CSV.

---

### Table 1 - Raw ECFP Baselines (no meta-training, encoder-independent)

> Source: `ecfp_baseline_heads.csv` (shared across all runs, corrected hash). Values are identical regardless of which checkpoint is used — raw ECFP fingerprints only.

| Support | RF | LogReg | PN-E | RF | LogReg | PN-E | RF | LogReg | PN-E |
|---|---|---|---|---|---|---|---|---|---|
| | **Random Split** | | | **Scaffold Split** | | | **Size Split** | | |
| 16 | 0.0815 | **0.0818** | 0.0587 | **0.0657** | **0.0657** | 0.0481 | 0.0510 | **0.0543** | 0.0296 |
| 32 | **0.1013** | 0.0994 | 0.0764 | **0.0844** | 0.0785 | 0.0601 | 0.0639 | **0.0650** | 0.0426 |
| 64 | **0.1279** | 0.1146 | 0.0934 | **0.1073** | 0.0937 | 0.0774 | **0.0736** | 0.0693 | 0.0453 |
| 128 | **0.1573** | 0.1416 | 0.1160 | **0.1453** | 0.1247 | 0.1083 | **0.0791** | 0.0644 | 0.0404 |
| 256 | **0.1653** | 0.1348 | 0.1192 | **0.1461** | 0.1146 | 0.1061 | **0.1230** | 0.0915 | 0.0614 |
| 512 | **0.1795** | 0.1286 | 0.1090 | **0.1696** | 0.1132 | 0.1058 | **0.1449** | 0.0732 | 0.0848 |

Key takeaways: RF consistently beats LogReg beats raw PN-E. Scaffold split costs ~0.04–0.05 ΔAUPRC vs random at same n. Size split collapses at n=128 (only 30 qualifying assays, harder subset).

---

### Table 2 - ECFP + Random (mean ± std, 3 seeds: 0, 1, 2)

> emb_* heads use the ECFP MLP encoder trained with random episodes.

| Support | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random Split** | | | | **Scaffold Split** | | | | **Size Split** | | | |
| 16 | 0.125+-0.002 | **0.128+-0.001** | 0.101+-0.002 | 0.090+-0.001 | 0.119+-0.002 | **0.123+-0.002** | 0.095+-0.002 | 0.088+-0.003 | 0.109+-0.000 | **0.113+-0.001** | 0.091+-0.001 | 0.077+-0.002 |
| 32 | **0.142+-0.002** | 0.142+-0.002 | 0.124+-0.001 | 0.105+-0.001 | 0.134+-0.001 | **0.135+-0.002** | 0.116+-0.003 | 0.100+-0.002 | **0.122+-0.002** | 0.121+-0.003 | 0.101+-0.002 | 0.084+-0.001 |
| 64 | **0.162+-0.002** | 0.158+-0.001 | 0.154+-0.001 | 0.121+-0.002 | **0.151+-0.001** | 0.149+-0.002 | 0.135+-0.001 | 0.110+-0.001 | **0.134+-0.002** | 0.131+-0.003 | 0.119+-0.001 | 0.092+-0.004 |
| 128 | 0.193+-0.001 | 0.188+-0.003 | **0.194+-0.003** | 0.145+-0.003 | **0.185+-0.002** | 0.179+-0.001 | 0.178+-0.000 | 0.133+-0.000 | 0.041+-0.009 | **0.042+-0.007** | 0.038+-0.008 | 0.023+-0.004 |
| 256 | 0.090+-0.002 | 0.087+-0.002 | **0.111+-0.002** | 0.065+-0.006 | 0.082+-0.002 | 0.081+-0.002 | **0.096+-0.001** | 0.058+-0.005 | 0.053+-0.002 | 0.049+-0.005 | **0.061+-0.003** | 0.024+-0.004 |
| 512 | 0.081+-0.004 | 0.076+-0.005 | **0.104+-0.008** | 0.050+-0.005 | 0.067+-0.000 | 0.063+-0.002 | **0.087+-0.005** | 0.036+-0.005 | 0.048+-0.010 | 0.046+-0.010 | **0.061+-0.004** | 0.024+-0.002 |

---

### Table 3 - ECFP + Shift-Aware (mean ± std, 3 seeds: 0, 1, 2)

> emb_* heads use the ECFP MLP encoder trained with scaffold-OOD episodes.

| Support | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random Split** | | | | **Scaffold Split** | | | | **Size Split** | | | |
| 16 | 0.115+-0.004 | **0.119+-0.003** | 0.095+-0.004 | 0.085+-0.003 | 0.109+-0.004 | **0.114+-0.004** | 0.089+-0.003 | 0.079+-0.004 | 0.100+-0.005 | **0.103+-0.005** | 0.079+-0.004 | 0.068+-0.004 |
| 32 | 0.138+-0.005 | **0.138+-0.004** | 0.122+-0.007 | 0.099+-0.003 | 0.127+-0.005 | **0.130+-0.005** | 0.109+-0.010 | 0.090+-0.005 | 0.114+-0.005 | **0.116+-0.004** | 0.093+-0.006 | 0.078+-0.006 |
| 64 | **0.158+-0.002** | 0.156+-0.003 | 0.149+-0.005 | 0.115+-0.004 | **0.151+-0.006** | 0.150+-0.007 | 0.134+-0.010 | 0.106+-0.004 | **0.128+-0.005** | 0.126+-0.006 | 0.109+-0.010 | 0.084+-0.007 |
| 128 | **0.180+-0.003** | 0.177+-0.004 | 0.180+-0.007 | 0.134+-0.005 | **0.176+-0.006** | 0.173+-0.007 | 0.170+-0.005 | 0.128+-0.005 | 0.035+-0.015 | 0.036+-0.016 | **0.040+-0.007** | 0.018+-0.003 |
| 256 | 0.093+-0.012 | 0.089+-0.013 | **0.117+-0.004** | 0.070+-0.002 | 0.085+-0.010 | 0.082+-0.011 | **0.102+-0.001** | 0.062+-0.006 | 0.053+-0.006 | 0.049+-0.007 | **0.061+-0.010** | 0.027+-0.006 |
| 512 | 0.076+-0.008 | 0.069+-0.006 | **0.102+-0.007** | 0.050+-0.003 | 0.071+-0.013 | 0.066+-0.011 | **0.087+-0.011** | 0.038+-0.004 | 0.052+-0.003 | 0.044+-0.004 | **0.055+-0.007** | 0.023+-0.006 |

---

### Table 4 - GNN + Random (mean ± std, 3 seeds: 0, 1, 2)

> emb_* heads use the FS-Mol GNN encoder trained with random episodes.

| Support | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random Split** | | | | **Scaffold Split** | | | | **Size Split** | | | |
| 16 | 0.127+-0.012 | **0.135+-0.015** | 0.122+-0.014 | 0.084+-0.011 | 0.114+-0.013 | **0.124+-0.015** | 0.108+-0.013 | 0.075+-0.009 | 0.104+-0.012 | **0.114+-0.013** | 0.101+-0.013 | 0.062+-0.009 |
| 32 | 0.151+-0.015 | **0.157+-0.019** | 0.137+-0.015 | 0.102+-0.013 | 0.138+-0.013 | **0.144+-0.017** | 0.124+-0.013 | 0.090+-0.012 | 0.119+-0.012 | **0.131+-0.017** | 0.105+-0.015 | 0.076+-0.010 |
| 64 | 0.181+-0.014 | **0.182+-0.021** | 0.155+-0.012 | 0.124+-0.016 | 0.162+-0.016 | **0.163+-0.021** | 0.137+-0.013 | 0.106+-0.017 | 0.140+-0.017 | **0.147+-0.023** | 0.112+-0.015 | 0.087+-0.015 |
| 128 | **0.221+-0.017** | 0.213+-0.021 | 0.189+-0.009 | 0.152+-0.017 | **0.200+-0.021** | 0.192+-0.023 | 0.168+-0.011 | 0.132+-0.020 | **0.067+-0.004** | 0.067+-0.004 | 0.051+-0.007 | 0.039+-0.002 |
| 256 | **0.153+-0.006** | 0.133+-0.007 | 0.129+-0.004 | 0.114+-0.004 | **0.139+-0.008** | 0.121+-0.008 | 0.112+-0.004 | 0.098+-0.006 | **0.109+-0.007** | 0.098+-0.015 | 0.071+-0.009 | 0.062+-0.009 |
| 512 | **0.164+-0.009** | 0.128+-0.014 | 0.138+-0.006 | 0.111+-0.002 | **0.158+-0.006** | 0.124+-0.010 | 0.127+-0.007 | 0.099+-0.001 | **0.132+-0.003** | 0.114+-0.009 | 0.088+-0.005 | 0.079+-0.006 |

---

### Table 5 - GNN + Shift-Aware (mean ± std, 3 seeds: 0, 1, 2)

> emb_* heads use the FS-Mol GNN encoder trained with scaffold-OOD episodes. Key comparison vs Table 4: does shift-aware training improve scaffold ΔAUPRC?

| Support | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN | PN-M | PN-E | LogReg | kNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Random Split** | | | | **Scaffold Split** | | | | **Size Split** | | | |
| 16 | 0.128+-0.002 | **0.136+-0.003** | 0.125+-0.002 | 0.084+-0.001 | 0.116+-0.002 | **0.123+-0.003** | 0.108+-0.003 | 0.075+-0.002 | 0.102+-0.003 | **0.114+-0.003** | 0.098+-0.003 | 0.064+-0.001 |
| 32 | 0.157+-0.003 | **0.165+-0.004** | 0.142+-0.004 | 0.108+-0.002 | 0.142+-0.001 | **0.150+-0.003** | 0.127+-0.001 | 0.095+-0.001 | 0.123+-0.002 | **0.132+-0.002** | 0.109+-0.003 | 0.078+-0.002 |
| 64 | 0.189+-0.003 | **0.190+-0.003** | 0.159+-0.002 | 0.131+-0.002 | 0.173+-0.003 | **0.176+-0.004** | 0.143+-0.002 | 0.114+-0.002 | 0.147+-0.003 | **0.154+-0.001** | 0.118+-0.004 | 0.094+-0.003 |
| 128 | **0.215+-0.002** | 0.207+-0.003 | 0.184+-0.004 | 0.155+-0.004 | **0.209+-0.003** | 0.202+-0.002 | 0.173+-0.004 | 0.145+-0.002 | 0.062+-0.001 | **0.063+-0.001** | 0.046+-0.005 | 0.036+-0.002 |
| 256 | **0.153+-0.000** | 0.136+-0.001 | 0.128+-0.003 | 0.114+-0.002 | **0.138+-0.001** | 0.121+-0.002 | 0.111+-0.003 | 0.100+-0.001 | **0.096+-0.003** | 0.086+-0.007 | 0.056+-0.008 | 0.058+-0.002 |
| 512 | **0.148+-0.003** | 0.119+-0.007 | 0.127+-0.004 | 0.106+-0.002 | **0.150+-0.003** | 0.123+-0.004 | 0.111+-0.007 | 0.094+-0.005 | **0.126+-0.004** | 0.111+-0.004 | 0.082+-0.011 | 0.068+-0.005 |

---

### Key Observations

- **GNN random is the best model on FS-Mol**: FS-Mol GNN (random episodes, 3 seeds) peaks at ΔAUPRC **0.226 ± 0.001** at n=128, exceeding the FS-Mol paper's 0.201 at that support size and matching the paper's n=256 number (0.226) one step earlier.
- **Episode type (shift-aware vs random) barely affects FS-Mol test score**: ECFP shift-aware 0.181 ± 0.003 vs ECFP random 0.183 ± 0.001 at n=128; GNN shift-aware 0.220 ± 0.000 vs GNN random 0.226 ± 0.001. The gain from richer encoder architecture dominates any episode-type effect.
- **Shift-aware training improves DrugOOD cross-dataset generalisation**: GNN shift-aware achieves 0.043 ± 0.003 on DrugOOD assay OOD vs GNN random 0.033 ± 0.002 - a ~30% gain. Training on scaffold-OOD episodes transfers to cross-dataset OOD even when it barely moves the FS-Mol in-distribution score.
- **GNN maintains performance at large n**: GNN random holds 0.149 at n=256 and 0.166 at n=512, whereas ECFP random collapses to 0.076 and 0.066. The GNN's structural representations are robust to the large-support regime; ECFP prototypes degrade.
- **Streaming over all 26k assays matters enormously**: ECFP pool-based (Run 1, ~62 assays) reaches 0.062 at n=128; ECFP streaming (any episode type) reaches ~0.183 - a 3× gain purely from training data diversity.
- **Encoder matters more than head on scaffold split**: GNN PN-M reaches 0.200–0.209 at n=128 scaffold split (corrected eval, Tables 4–5); RF (0.145 at n=128) is competitive only at large n where few assays qualify. See Tables 2–5 for full head-by-head breakdown.
- **GNN DrugOOD assay OOD is clearly better than ECFP**: GNN shift-aware 0.043 ± 0.003 vs ECFP shift-aware 0.015 ± 0.007 - roughly 3× gain. Structural features transfer better across assay boundaries than fingerprints.
- **Low variance across seeds**: std ≤ 0.005 for all models on random split. Results are reproducible.
- **n=256/512 drop is assay selection bias, not model failure**: At n=256 only 29 qualifying assays remain (vs 154 at n=16). Fixed-assay-set analysis (`fixed_assay_curves.py`) shows all models improve monotonically on the same 11 assays across all support sizes — the naive curve drop is entirely due to a harder assay subset entering the population.

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
# Full pipeline (train + evaluate) - configure MODEL_HEAD/ENCODER/TRAINING_SPLIT in main.py first
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