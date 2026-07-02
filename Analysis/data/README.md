# Data Analysis

Scripts that audit the FS-Mol and DrugOOD datasets **before and independently of any model** - they need only the datasets and `config.py`. Each script answers exactly one question (stated at the top of the file). Outputs go to `DATA_ANALYSIS_FIGURES_DIR` / `DATA_ANALYSIS_RESULTS_DIR`.

| Script | One question it answers |
|---|---|
| `dataset_overview.py` | How much data survives filtering? (assay sizes, fraction-active, per-step data loss, DrugOOD domain sizes) |
| `scaffold_diversity.py` | How scaffold-diverse is each task? (unique Murcko scaffolds, diversity ratio) |
| `scaffold_activity.py` | Does scaffold membership predict the activity label? (Cramér's V) |
| `chemical_space.py` | How chemically different are FS-Mol and DrugOOD? (properties, Tanimoto distance, t-SNE) |
| `shift_aware_episodes.py` | How many shift-aware *training* episodes actually carry usable signal? |

Each script is self-contained (per-molecule RDKit feature helpers are inlined where used - no shared library file) and has a `# CONFIG` block near the top - edit the variables, run with **no arguments** (same mechanism as `main.py`). Figures are written directly by these scripts.

```bash
python Analysis/data/dataset_overview.py     # CONFIG: SPLITS_TO_SCAN (drop "train" for a fast run)
python Analysis/data/scaffold_diversity.py   # CONFIG: TRAIN_SAMPLE
python Analysis/data/scaffold_activity.py    # CONFIG: N_SUPPORT / TRAIN_SAMPLE / SEED
python Analysis/data/chemical_space.py       # CONFIG: N_TSNE / SEED
python Analysis/data/shift_aware_episodes.py # CONFIG: N_EPISODES / N_SUPPORT
```

---

## Key dataset facts

### FS-Mol - data loss after filtering (`Relation == "="`, then drop tasks < 32 molecules)

| Split | Raw molecules | Inexact dropped | Bad/missing | Task-too-small | **Used** | Tasks kept |
|---|---|---|---|---|---|---|
| Train | 5,038,727 | 1,928,837 (38.3%) | 134,751 (2.7%) | 162,246 (3.2%) | **2,812,893 (55.8%)** | 16,930 / 26,868 (63%) |
| Valid | 19,008 | 2,266 (11.9%) | 83 (0.4%) | 36 (0.2%) | **16,623 (87.5%)** | 38 / 40 (95%) |
| Test | 56,220 | 12,429 (22.1%) | 129 (0.2%) | 0 (0.0%) | **43,662 (77.7%)** | 154 / 157 (98%) |

### FS-Mol - assay size (after filtering)

| Split | Tasks | Mean | Median | Min | Max |
|---|---|---|---|---|---|
| Train | 16,930 | 166 | 44 | 32 | 88,353 |
| Valid | 38 | 437 | 157 | 109 | 4,697 |
| Test | 154 | 284 | 157 | 63 | 3,594 |

Test/valid assays are ~3.5× larger than train. **Only very large assays qualify at support sizes 256/512**, so the support-size sweep at n=256/512 is computed on a different (smaller, harder) assay population - a selection effect, not a model effect. The `fixed_assay_curves.py` helper in `../model/` controls for this.

### DrugOOD - label balance (verified, `lbap_core_ic50_scaffold`)

| Split | active | inactive | % active |
|---|---|---|---|
| train (context pool) | 20,799 | 1,226 | **94%** |
| ood_test | 15,144 | 4,336 | 78% |
| iid_test | 28,226 | 3,162 | 90% |

DrugOOD uses a single global pIC50 threshold, so the context pool is heavily active-skewed. Uniform context sampling therefore builds the inactive prototype from ~0-1 molecules at small context sizes, and the ΔAUPRC ceiling on `ood_test` is only `1 - 0.78 = 0.22`. The evaluation now **stratifies** the context sample (`data.stratified_context_indices`) so both classes are represented. Note this differs from FS-Mol training, where `Property` is balanced ~50/50 per assay (verified: 29/29 on a sample assay) - "active" means different things in train vs eval.

### Scaffold-activity degeneracy (Cramér's V, `scaffold_activity.py`)

On 2000 sampled FS-Mol training assays: mean V = 0.68, median 0.72; V > 0.30 in 91% of assays. Scaffold membership strongly predicts the binary label in almost all assays. **Implication for evaluation:** a scaffold split shifts the *label* distribution, not just the structure - so a fair scaffold split must keep both classes present in the support set (see `data.build_fair_split_indices`).

---

## Figures

All generated into `outputs/figures/data_analysis/`:

| Figure | Produced by |
|---|---|
| `fig1a_assay_sizes.png` | `dataset_overview.py` |
| `fig1b_fraction_actives.png` | `dataset_overview.py` |
| `fig1c_domain_sizes.png` | `dataset_overview.py` |
| `fig_size_vs_fraction_exact.png` | `dataset_overview.py` |
| `fig_scaffold_diversity_per_task.png` | `scaffold_diversity.py` |
| `fig_scaffold_activity_corr.png` | `scaffold_activity.py` |
| `tanimoto_distances.png` | `chemical_space.py` |
| `tsne_fsmol_vs_drugood.png` | `chemical_space.py` |
