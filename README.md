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

| Regime | Runs | Training pool | Notes |
|---|---|---|---|
| **Pool-based** | Runs 1–2 (ECFP shift-aware) | ~62 assays | Legacy runs; severely limited task diversity |
| **Streaming** | Runs 3–4 (ECFP random, GNN shift-aware) | All ~26,868 FS-Mol train assays | Matches FS-Mol paper; full task diversity |

Runs 1–2 are kept as ECFP baselines but are not methodologically comparable to the streaming runs due to the training pool size difference.

---

## Results Summary

Evaluated on 154 FS-Mol test assays and 3 DrugOOD shift types (IC50, 3 seeds, context sizes 16–512).

### FS-Mol Test - Mean ΔAUPRC (Random Split)

| Run | Encoder | Episodes | Training | n=16 | n=32 | n=64 | n=128 | n=256 | n=512 | Best val |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ECFP | shift-aware | pool-based | 0.028 | 0.033 | 0.041 | 0.062 | 0.065 | 0.056 | RMSE 0.527 (ep15) |
| 2 | ECFP | shift-aware | pool-based | 0.038 | 0.042 | 0.049 | 0.076 | 0.110 | 0.105 | ΔAUPRC +0.162 (ep19) |
| 3 | ECFP | **random** | **streaming** | 0.127 | 0.144 | 0.163 | **0.187** | 0.087 | 0.074 | ΔAUPRC +0.186 (ep77) |
| 4 | FS-Mol GNN 10L | shift-aware | **streaming** | **0.129** | 0.156 | 0.183 | **0.222** | 0.151 | 0.153 | ΔAUPRC +0.207 (ep69) |
| *FS-Mol paper* | *GNN + ProtoNet* | *random* | *full data* | *0.126* | *-* | *0.185* | *0.201* | *0.226* | *-* |

### FS-Mol Test - Mean ΔAUPRC (Scaffold Split)

| Run | Encoder | Episodes | n=16 | n=32 | n=64 | n=128 | n=256 | n=512 |
|---|---|---|---|---|---|---|---|---|
| 1 | ECFP | shift-aware | 0.018 | 0.018 | 0.019 | 0.019 | 0.003 | −0.003 |
| 2 | ECFP | shift-aware | 0.010 | 0.011 | 0.009 | 0.011 | 0.007 | 0.003 |
| 3 | ECFP | **random** | 0.064 | 0.059 | 0.057 | 0.064 | 0.007 | 0.009 |
| 4 | FS-Mol GNN 10L | shift-aware | 0.050 | 0.053 | 0.052 | 0.053 | 0.011 | 0.006 |

### Inside-Task OOD and DrugOOD Summary

Inside-task OOD: support and query from different scaffold groups within the same assay (novel evaluation protocol). DrugOOD values are mean ΔAUPRC on ood_test, averaged across context sizes 16–512.

| Run | Encoder | Episodes | Inside-task ΔAUPRC | DrugOOD assay | DrugOOD scaffold | DrugOOD size |
|---|---|---|---|---|---|---|
| 1 | ECFP | shift-aware | 0.041 | 0.008 | 0.000 | 0.016 |
| 2 | ECFP | shift-aware | 0.023 | 0.021 | 0.019 | 0.021 |
| 3 | ECFP | **random** | 0.014 | 0.017 | 0.011 | 0.009 |
| 4 | FS-Mol GNN 10L | shift-aware | 0.017 | 0.036 | 0.026 | 0.025 |

### Key Observations

- **Streaming vs pool-based**: Run 3 (streaming ECFP random) achieves 0.127 at n=16 vs Run 2 (pool-based ECFP shift-aware) 0.038 — a 3× improvement driven entirely by training data diversity (26k assays vs 62).
- **n=256 drop is not unique to scaffold-aware training**: Run 3 (random IID episodes) also drops hard at n=256 (0.187 → 0.087), showing the large-n degradation is an ECFP representation limit, not just a scaffold-aware training artifact. The GNN random run will confirm whether this holds for the GNN encoder.
- **Scaffold split collapse at large n**: All streaming runs collapse at n=256 for scaffold split (Run 3: 0.064 → 0.007, Run 4: 0.050 → 0.011). This is consistent regardless of episode type — scaffold OOD is fundamentally harder.
- **GNN excels at n=128**: Run 4 (GNN shift-aware) peaks at 0.222 at n=128, outperforming paper's 0.201 and all ECFP runs. GNN representations are richer and scale better with support size.
- **GNN improves DrugOOD assay OOD**: GNN assay OOD (0.036) is highest across all runs. ECFP random is weaker on DrugOOD assay (0.017) despite matching GNN on FS-Mol at n=16 — cross-assay transfer benefits from GNN's structural features.
- **FS-Mol paper parity**: Run 4 matches paper at n=16 (0.129 vs 0.126) and exceeds at n=64/128. Remaining gap at n=256 (0.151 vs 0.226) is under investigation with the pending GNN random run.

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
