# Prototypical Networks for Molecular OOD Property Prediction

Few-shot molecular property prediction with out-of-distribution (OOD) generalisation, evaluated on two benchmarks: FS-Mol and DrugOOD.

- **Pretraining:** FS-Mol (~26k assays from ChEMBL), episodic training
- **Evaluation:** DrugOOD benchmark (scaffold, size, and assay distribution shift) + FS-Mol held-out test set
- **Task:** Predict activity (pIC50) for molecules from unseen chemical distributions using a small support set
- **Primary metric:** ΔAUPRC = AUPRC(model) − fraction_actives (0 = random classifier, higher = better)

---

## Background

Standard Prototypical Networks (Snell et al., 2017) are designed for few-shot classification. This project implements two prediction heads:

**Regression head** — Nadaraya-Watson kernel regression in learned embedding space:

$$\hat{y}_q = \sum_{i \in \text{support}} \frac{\exp(-d(f(x_q), f(x_i)) / \tau)}{\sum_j \exp(-d(f(x_q), f(x_j)) / \tau)} \cdot y_i$$

**Classification head** — Binary active/inactive prototypes (BCE loss). Threshold = support set median.

The embedding function $f$ is trained episodically on FS-Mol with **shift-aware episodes**: support and query molecules come from different Bemis-Murcko scaffold families within the same assay. The pretrained model is then evaluated **zero-shot** on DrugOOD.

---

## Design Choices

| Component | Implemented | Notes |
|---|---|---|
| Encoders | ECFP4 (2048-bit) MLP; PNA-GNN 6-layer; FS-Mol GNN 10-layer | GNN uses FS-Mol featurisation |
| Heads | Regression (kernel regression, MSE); Classification (binary PN, BCE) | 4 combinations completed |
| Training splits | Shift-aware (scaffold OOD episodes) | Consistent across all 4 runs |
| Distance | Euclidean (training); Mahalanobis with shrinkage (eval) | FS-Mol paper uses Mahalanobis throughout |
| Primary metric | ΔAUPRC = AUPRC(model) − fraction_actives | FS-Mol paper convention |
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

## Training Configuration

Two training regimes were used. This distinction is important when interpreting results.

| Regime | Runs | Training pool | Notes |
|---|---|---|---|
| **Pool-based (old)** | Runs 1, 2, 3 | 62 assays (regression) or 21 assays with ≥320 molecules (GNN) | Fast iteration; severely limited task diversity |
| **Streaming (new)** | Run 4 | All ~16,930 usable FS-Mol train assays, streamed from disk | Matches FS-Mol paper; full task diversity |

The pool-based regime undertrains the model because it sees only a tiny fraction of available tasks per epoch. Run 4 (streaming) is the methodologically correct baseline and should be used for comparison to the FS-Mol paper.

---

## Results Summary

All models trained with shift-aware episodes. Evaluated on 154 FS-Mol test assays and 3 DrugOOD shift types (IC50, 3 seeds, context sizes 16–512).

### FS-Mol Test — Mean ΔAUPRC (Random Split)

| Model | Encoder | Training | n=16 | n=32 | n=64 | n=128 | n=256 | n=512 | Best val |
|---|---|---|---|---|---|---|---|---|---|
| Regression | ECFP | pool-based | 0.028 | 0.033 | 0.041 | 0.062 | 0.065 | 0.056 | RMSE 0.527 (ep15) |
| Classification | ECFP | pool-based | 0.038 | 0.042 | 0.049 | 0.076 | 0.110 | 0.105 | ΔAUPRC +0.162 (ep19) |
| Classification | PNA-GNN 6L | pool-based | 0.028 | 0.035 | 0.041 | 0.061 | 0.071 | 0.058 | ΔAUPRC +0.124 (ep22) |
| Classification | FS-Mol GNN 10L | **streaming** | 0.028 | 0.031 | 0.041 | 0.058 | **0.158** | **0.158** | ΔAUPRC +0.197 (step 3200) |
| *FS-Mol paper* | *GNN + ProtoNet* | *full data* | *0.126* | *—* | *0.185* | *0.201* | *0.226* | *—* |

### FS-Mol Test — Mean ΔAUPRC (Scaffold Split)

| Model | Encoder | n=16 | n=32 | n=64 | n=128 | n=256 | n=512 |
|---|---|---|---|---|---|---|---|
| Regression | ECFP | 0.018 | 0.018 | 0.019 | 0.019 | 0.003 | −0.003 |
| Classification | ECFP | 0.010 | 0.011 | 0.009 | 0.011 | 0.007 | 0.003 |
| Classification | PNA-GNN 6L | 0.015 | 0.014 | 0.014 | 0.017 | 0.001 | 0.014 |
| Classification | FS-Mol GNN 10L | 0.002 | 0.005 | 0.002 | 0.004 | 0.020 | 0.011 |

### Inside-Task OOD and DrugOOD Summary

Inside-task OOD: support and query from different scaffold groups within the same assay (novel evaluation protocol). DrugOOD values are mean ΔAUPRC on ood_test, averaged across context sizes 16–512.

| Model | Encoder | Inside-task ΔAUPRC | DrugOOD assay | DrugOOD scaffold | DrugOOD size |
|---|---|---|---|---|---|
| Regression | ECFP | 0.038 | 0.008 | 0.001 | 0.016 |
| Classification | ECFP | — | 0.021 | 0.019 | 0.021 |
| Classification | PNA-GNN 6L | 0.024 | 0.037 | 0.027 | 0.046 |
| Classification | FS-Mol GNN 10L | 0.034 | 0.042 | 0.031 | 0.029 |

### Key Observations

- **Pool-based vs streaming**: The streaming run (FS-Mol GNN 10L) achieves 0.158 ΔAUPRC at n=256, nearly double the pool-based GNN (0.071) and comparable ECFP classification (0.110), confirming that training data diversity is the dominant factor.
- **Gap to FS-Mol paper**: At n=16–128, all runs are well below the paper (~0.03–0.08 vs 0.126–0.201). The gap closes at n=256 (0.158 vs 0.226). The remaining gap likely reflects custom GNN vs PyG PNAConv implementation differences.
- **Scaffold split is uniformly hard**: All models plateau at 0.001–0.020 ΔAUPRC regardless of encoder or training regime, confirming scaffold OOD as the primary unsolved challenge.
- **GNN improves DrugOOD generalisation**: GNN encoders consistently outperform ECFP on DrugOOD size OOD (0.046 vs 0.016), while ECFP regression performs poorly on scaffold OOD (mean ≈ 0.001, often negative).
- **Classification > regression** on ΔAUPRC at large support sizes (0.110 vs 0.065 at n=256), consistent with FS-Mol paper findings.

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

**FS-Mol** — Download from [microsoft/FS-Mol](https://github.com/microsoft/FS-Mol). Extract to `data/fsmol/`. Each file is one ChEMBL assay; the loader reads precomputed ECFP fingerprints and log-transformed labels directly.

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
python Analysis/model/plot_results.py --run_tag fsmol_gnn_classification_shift_aware

# Data analysis (no model needed)
python Analysis/data/dataset_overview.py
python Analysis/data/scaffold_analysis.py
```

---

## References

- Snell et al. (2017) — [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- Stanley et al. (2021) — [FS-Mol: A Few-Shot Learning Dataset of Molecules](https://openreview.net/forum?id=701FtuyLlAd)
- Ji et al. (2022) — [DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery](https://arxiv.org/abs/2201.09637)
- Corso et al. (2020) — [Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718)
