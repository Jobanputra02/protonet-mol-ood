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
| Heads | Regression (kernel regression, MSE); Classification (binary PN, BCE) | 3 runs completed |
| Training splits | Shift-aware (scaffold OOD episodes) | Consistent across all runs |
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
    │   ├── {run_tag}/         # fig2a, fig2b, fig3 per run
    │   └── data_analysis/     # fig1a/b/c, scaffold diversity, t-SNE
    └── results/
        ├── {run_tag}/         # drugood_results, fsmol_test_results, inside_task_ood, predictions CSVs
        └── data_analysis/     # assay_sizes, data_loss, scaffold_diversity, diagnostic_baseline
```

---

## Training Configuration

Two training regimes were used. This distinction is important when interpreting results.

| Regime | Runs | Training pool | Notes |
|---|---|---|---|
| **Pool-based** | Runs 1–2 (ECFP) | ~62 assays | Fast iteration; severely limited task diversity |
| **Streaming** | Run 3 (FS-Mol GNN) | All ~26,868 FS-Mol train assays, streamed from disk | Matches FS-Mol paper; full task diversity |

The pool-based regime undertrains the model because it sees only a tiny fraction of available tasks per epoch. Run 3 (streaming GNN) is the methodologically correct baseline and should be used for comparison to the FS-Mol paper.

---

## Results Summary

All models trained with shift-aware episodes. Evaluated on 154 FS-Mol test assays and 3 DrugOOD shift types (IC50, 3 seeds, context sizes 16–512).

### FS-Mol Test - Mean ΔAUPRC (Random Split)

| Model | Encoder | Training | n=16 | n=32 | n=64 | n=128 | n=256 | n=512 | Best val |
|---|---|---|---|---|---|---|---|---|---|
| Regression | ECFP | pool-based | 0.028 | 0.033 | 0.041 | 0.062 | 0.065 | 0.056 | RMSE 0.527 (ep15) |
| Classification | ECFP | pool-based | 0.038 | 0.042 | 0.049 | 0.076 | 0.110 | 0.105 | ΔAUPRC +0.162 (ep19) |
| Classification | FS-Mol GNN 10L | **streaming** | **0.129** | 0.156 | 0.183 | **0.222** | 0.151 | 0.153 | ΔAUPRC +0.207 (ep69) |
| *FS-Mol paper* | *GNN + ProtoNet* | *full data* | *0.126* | *-* | *0.185* | *0.201* | *0.226* | *-* |

### FS-Mol Test - Mean ΔAUPRC (Scaffold Split)

| Model | Encoder | n=16 | n=32 | n=64 | n=128 | n=256 | n=512 |
|---|---|---|---|---|---|---|---|
| Regression | ECFP | 0.018 | 0.018 | 0.019 | 0.019 | 0.003 | −0.003 |
| Classification | ECFP | 0.010 | 0.011 | 0.009 | 0.011 | 0.007 | 0.003 |
| Classification | FS-Mol GNN 10L | 0.050 | 0.053 | 0.052 | 0.053 | 0.011 | 0.006 |

### Inside-Task OOD and DrugOOD Summary

Inside-task OOD: support and query from different scaffold groups within the same assay (novel evaluation protocol). DrugOOD values are mean ΔAUPRC on ood_test, averaged across context sizes 16–512.

| Model | Encoder | Inside-task ΔAUPRC | DrugOOD assay | DrugOOD scaffold | DrugOOD size |
|---|---|---|---|---|---|
| Regression | ECFP | 0.041 | 0.008 | 0.000 | 0.016 |
| Classification | ECFP | 0.023 | 0.021 | 0.019 | 0.021 |
| Classification | FS-Mol GNN 10L | 0.017 | 0.036 | 0.026 | 0.025 |

### Key Observations

- **FS-Mol paper parity at n=16**: The GNN run achieves 0.129 at n=16 (paper: 0.126) - the label binarisation fix (using pre-stored ChEMBL 0/1 labels instead of support-median thresholding) was the dominant factor; the old equivalent run had only 0.028 at n=16.
- **Exceeds paper at n=128**: 0.222 vs paper 0.201. At n=256, performance drops to 0.151 (paper: 0.226). This n=256 drop is specific to scaffold-aware training - a pending random-episode run will test whether IID training recovers monotonic improvement.
- **Scaffold split collapse at large n**: GNN scaffold split falls from 0.050 at n=16 to 0.006 at n=512. This is structurally different from random split and is the core OOD finding - prototypical networks become scaffold-specialised at large support sizes.
- **Streaming matters**: The streaming GNN run at n=16 (0.129) vastly outperforms the pool-based ECFP classification (0.038), confirming training data diversity is the dominant factor.
- **Classification > regression** on ΔAUPRC at large support sizes (0.110 vs 0.065 at n=256 for ECFP), consistent with FS-Mol paper findings.
- **GNN improves DrugOOD assay OOD**: GNN assay OOD (0.036) is highest across all runs - full training distribution teaches cross-assay transfer. Scaffold OOD remains harder (0.026) but still positive.

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

# Generate figures from saved CSVs
python Analysis/model/plot_results.py --run_tag fsmol_gnn_classification_shift_aware

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
