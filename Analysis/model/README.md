# Model Analysis

Scripts for evaluating trained models and generating result figures. All scripts read paths from `config.py` and write outputs to `FIGURES_DIR` / `RESULTS_DIR`. Run these after `main.py` has produced the evaluation CSVs.

---

## Scripts

### `plot_results.py`

Generates all result figures for a given model run from the CSVs produced by `main.py`.

**Usage:**

```bash
# By full run tag
python Analysis/model/plot_results.py --run_tag gnn_classification_shift_aware

# Or by components
python Analysis/model/plot_results.py --encoder gnn --head classification --split shift_aware
```

The `run_tag` must match a pair of CSVs in `outputs/results/`:
- `fsmol_test_results_{run_tag}.csv`
- `drugood_results_{run_tag}.csv`

**Figures generated:**

| Figure | Description |
|---|---|
| `fig2a_fsmol_line_plot_{run_tag}.png` | ΔAUPRC vs support size, one line per split type (random / scaffold / size). Spearman ρ and RMSE panels included if non-NaN (regression only). Error bars = ±1 std across assays. |
| `fig2b_fsmol_boxplot_{run_tag}.png` | Per-assay ΔAUPRC distribution across support sizes, grouped by split type. Spearman ρ box included for regression only. |
| `fig3_drugood_line_plot_{run_tag}.png` | ΔAUPRC vs context size for each DrugOOD shift type (scaffold / size / assay), OOD and IID as separate lines. Spearman ρ row included for regression only. |

The script automatically detects whether the run is regression or classification and skips panels where all values are NaN (Spearman and RMSE are suppressed for classification models since predictions are probabilities, not continuous values).

---

### `diagnostic_baseline.py`

Compares a pretrained PTN against simple baselines on the same episodes: mean-label predictor, kNN (k=1,3,5), and kernel ridge regression with Tanimoto kernel (KR-Tanimoto, α=0.01/0.1/1.0).

**Usage:**

```bash
python Analysis/model/diagnostic_baseline.py
```

The checkpoint used is determined by `PTN_ECFP_REGRESSION_SHIFT_CHECKPOINT` in `config.py`. Edit that import to test a different model.

**Interpretation guide:**

| Outcome | Meaning |
|---|---|
| `PTN MSE < KR-Tanimoto` | Learned embedding improves over raw ECFP fingerprints ✓ |
| `PTN MSE ≈ KR-Tanimoto` | Embedding adds no information beyond raw fingerprints |
| `All methods ≈ mean-label` | Episode split type is too hard — no method has a signal |
| `KR-Tanimoto catastrophic on scaffold` | Overfits to within-scaffold patterns; expected behaviour |

**Outputs:**

| File | Description |
|---|---|
| `results/diagnostic_baseline.csv` | Per-episode MSE for all methods across split types and support sizes |

---

## Results

### GNN Encoder — Classification Head — Shift-Aware Split

> Checkpoint: epoch 22, Val ΔAUPRC +0.1236. Trained with lr=1e-4, n_support=32, n_query=64, n_episodes=500.
> 154 FS-Mol test assays (3 dropped — fewer than 32 exact-measurement molecules).

#### FS-Mol Test — ΔAUPRC by split type and support size

| Support size | Random | Scaffold | Size |
|---|---|---|---|
| 16 | 0.0277 | 0.0152 | 0.0353 |
| 32 | 0.0350 | 0.0142 | 0.0379 |
| 64 | 0.0413 | 0.0138 | 0.0471 |
| 128 | 0.0607 | 0.0168 | 0.0445 |
| 256 | 0.0714 | 0.0012 | 0.0349 |
| 512 | 0.0580 | 0.0144 | 0.0432 |

Spearman and RMSE are not reported for classification (predictions are probabilities ∈ [0,1]; those metrics against continuous IC50 targets are meaningless).

**Inside-task OOD (scaffold split, n_support=16):** Mean ΔAUPRC = 0.0240 across 153 assays.

#### FS-Mol Test — Figures

**Figure 2(a) — ΔAUPRC vs support size:**

![FS-Mol line plot](../../outputs/figures/fig2a_fsmol_line_plot_gnn_classification_shift_aware.png)

**Figure 2(b) — Per-assay ΔAUPRC distribution:**

![FS-Mol boxplot](../../outputs/figures/fig2b_fsmol_boxplot_gnn_classification_shift_aware.png)

#### DrugOOD — ΔAUPRC by shift type and context size

| Context size | Scaffold OOD | Scaffold IID | Size OOD | Size IID | Assay OOD | Assay IID |
|---|---|---|---|---|---|---|
| 16 | 0.0061 | 0.0309 | 0.0422 | 0.0233 | 0.0189 | 0.0155 |
| 32 | 0.0270 | 0.0425 | 0.0459 | 0.0263 | 0.0174 | 0.0164 |
| 64 | 0.0221 | 0.0275 | 0.0466 | 0.0277 | 0.0413 | 0.0470 |
| 128 | 0.0290 | 0.0364 | 0.0464 | 0.0284 | 0.0482 | 0.0540 |
| 256 | 0.0387 | 0.0448 | 0.0468 | 0.0288 | 0.0497 | 0.0547 |
| 512 | 0.0390 | 0.0442 | 0.0488 | 0.0296 | 0.0481 | 0.0536 |

**Key observations:**
- Size shift: OOD outperforms IID across all context sizes — GNN graph convolutions generalise better to new molecular sizes than new scaffolds
- Assay shift: best overall (OOD 0.048, IID 0.054 at ctx=256) — model handles assay-level shift well
- Scaffold shift: weakest OOD performance; barely improves past ctx=32; variance collapses at large context (std~0.001), suggesting the model ignores support beyond a few molecules for scaffold-shifted queries

**Figure 3 — DrugOOD ΔAUPRC vs context size:**

![DrugOOD line plot](../../outputs/figures/fig3_drugood_line_plot_gnn_classification_shift_aware.png)

---

### ECFP Encoder — Regression Head — Shift-Aware Split

> Checkpoint: epoch 14, Val RMSE 0.5250. Trained with lr=1e-3, n_support=16, n_query=16, n_episodes=1000.

#### FS-Mol Test — ΔAUPRC by split type and support size

| Support size | Random | Scaffold | Size |
|---|---|---|---|
| 16 | 0.0283 | 0.0177 | 0.0305 |
| 32 | 0.0331 | 0.0182 | 0.0325 |
| 64 | 0.0408 | 0.0192 | 0.0298 |
| 128 | 0.0618 | 0.0191 | 0.0340 |
| 256 | 0.0652 | 0.0028 | 0.0340 |
| 512 | 0.0560 | -0.0030 | 0.0379 |

#### FS-Mol Test — Spearman ρ by split type and support size

| Support size | Random | Scaffold | Size |
|---|---|---|---|
| 16 | 0.067 | 0.013 | 0.029 |
| 32 | 0.085 | 0.010 | 0.034 |
| 64 | 0.107 | 0.012 | 0.036 |
| 128 | 0.114 | 0.008 | 0.039 |
| 256 | 0.171 | 0.005 | 0.093 |
| 512 | 0.107 | -0.001 | 0.067 |

#### DrugOOD — ΔAUPRC OOD test

| Context size | IC50 Scaffold | IC50 Size | IC50 Assay |
|---|---|---|---|
| 16 | -0.0166 | +0.0045 | +0.0214 |
| 32 | -0.0010 | -0.0035 | +0.0138 |
| 64 | +0.0187 | +0.0001 | +0.0145 |
| 128 | +0.0114 | +0.0071 | +0.0152 |
| 256 | +0.0136 | +0.0249 | +0.0211 |
| 512 | +0.0211 | +0.0270 | +0.0222 |

---

### Baseline Diagnostic — ECFP Regression (Shift-Aware)

Two modes were run. **Train mode** is a sanity check. **Test mode** uses all 154 held-out test assays with the same protocol as the main FS-Mol evaluation.

#### Train Mode — Sanity Check

> 100 sampled train assays × 10 episodes × 2 split types (random / scaffold). Fixed query size N=16.

| Method | Random MSE | Scaffold MSE |
|---|---|---|
| Mean-label | 0.6304 | 1.0092 |
| kNN (k=1) | 0.7399 | 1.2151 |
| kNN (k=3) | 0.5612 | 1.1262 |
| kNN (k=5) | 0.5501 | 1.0918 |
| KR-Tanimoto (α=0.01) | 0.9180 | 6.226 |
| KR-Tanimoto (α=0.10) | 0.9493 | 6.431 |
| KR-Tanimoto (α=1.00) | 1.7145 | 7.856 |
| **PTN** | **0.5486** | **1.0204** |

PTN marginally best on random; KR-Tanimoto catastrophically overfits on scaffold split (MSE 6–8 vs mean-label 1.0); training is working.

#### Test Mode — All 154 FS-Mol Test Assays

**ΔAUPRC — Random split:**

| Support size | kNN (k=5) | KR-Tanimoto (α=0.01) | PTN |
|---|---|---|---|
| 16 | +0.0434 | +0.0602 | +0.0296 |
| 32 | +0.0515 | +0.0708 | +0.0375 |
| 64 | +0.0620 | +0.0780 | +0.0368 |
| 128 | +0.0842 | +0.1064 | +0.0681 |
| 256 | +0.1408 | +0.1600 | +0.0558 |
| 512 | +0.1410 | +0.1650 | +0.0599 |

**Spearman ρ — Random split:**

| Support size | kNN (k=5) | KR-Tanimoto (α=0.01) | PTN |
|---|---|---|---|
| 16 | +0.138 | +0.176 | +0.068 |
| 32 | +0.197 | +0.215 | +0.088 |
| 64 | +0.245 | +0.255 | +0.104 |
| 128 | +0.282 | +0.274 | +0.107 |
| 256 | +0.418 | +0.471 | +0.171 |
| 512 | +0.293 | +0.323 | +0.116 |

**Scaffold split — ΔAUPRC:**

| Support size | kNN (k=5) | KR-Tanimoto (α=0.01) | PTN |
|---|---|---|---|
| 16 | +0.010 | +0.036 | +0.021 |
| 32 | +0.011 | +0.035 | +0.020 |
| 64 | +0.012 | +0.032 | +0.017 |
| 128 | +0.011 | +0.036 | +0.016 |
| 256 | +0.006 | +0.044 | +0.004 |
| 512 | +0.007 | +0.026 | +0.005 |

**Key findings:**
- kNN-k5 has ~2× better ΔAUPRC and Spearman than PTN on random split — raw ECFP nearest-neighbour retrieval outranks the learned embedding on unseen assays
- KR-Tanimoto has the best ranking metrics on random and size splits, but catastrophic RMSE on scaffold (calibration failure under cross-scaffold extrapolation)
- On scaffold split, all ECFP methods (PTN, kNN, mean-label) cluster together — ECFP carries no cross-scaffold signal regardless of how it is used. This is the representation bottleneck motivating the GNN encoder.
- PTN does not outperform raw ECFP baselines on test assay ranking, reinforcing the need for a structurally-aware encoder
