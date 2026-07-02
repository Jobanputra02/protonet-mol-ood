# Model Analysis

Scripts that evaluate trained checkpoints and produce figures. All read paths from `config.py`. Each script answers one question (stated at the top of the file).

## How the "models" are organised

A model here is a **(representation × head)** pair:

- **Representation** - either raw ECFP fingerprints, or a frozen encoder's embedding. The trainable encoders (`ECFPEncoder`, `PNAGNNEncoder`, `FSMolGNNEncoder`) and the ProtoNet heads live in [`../../model.py`](../../model.py) and are produced by [`../../train.py`](../../train.py).
- **Head** - everything fit *fresh per task* (no meta-training) is defined inside `baseline_grid.py`, one signature `head(Xs, ys, Xq) → p_active`: `proto_euclidean`, `proto_mahalanobis`, `logreg`, `knn`, `random_forest`, `kr_tanimoto`, `mean_label`. (`baseline_grid.py` is to these heads what `main.py` is to `model.py` - the runnable pipeline; the heads are its only user, so they live inline.)

| Script | What it does |
|---|---|
| `baseline_grid.py` | Representation × head on identical splits - writes `baseline_grid.csv` per run |
| `fixed_assay_curves.py` | Recomputes support-size curves on a *fixed* assay set vs the naive varying-population curve - confirms the n=256/512 drop is assay selection bias, not model failure |
| `plot_line_grid.py` | **Line plot** - mean ± std ΔAUPRC vs support size; configurable heads, splits, encoder |
| `plot_boxplot_grid.py` | **Boxplot** - per-assay ΔAUPRC distribution vs support size; same config as line plot |

**Run mechanism:** every script has a `# CONFIG` block at the top - edit the variables, run with **no arguments**. No argparse.

### Plotting scripts config

Both `plot_line_grid.py` and `plot_boxplot_grid.py` share the same config pattern:

```python
ENCODER        = "fsmol_gnn"   # "ecfp" | "fsmol_gnn"
TRAINING_SPLIT = "random"      # "random" | "shift_aware"
SEEDS          = [0, 1, 2]     # which seeds to average over

ECFP_HEADS = ["ecfp_rf", "ecfp_logreg", "ecfp_proto_euclid"]
EMB_HEADS  = ["emb_proto_mahalanobis", "emb_proto_euclid", "emb_logreg", "emb_knn"]
HEADS = EMB_HEADS              # any subset of the 7 heads - comment out to exclude

EVAL_SPLITS = ["random", "scaffold", "size"]  # any subset

# plot_line_grid.py only:
ERROR_BARS = "assays"  # "assays" (std across tasks — standard for benchmarks) | "seeds" (reproducibility check)
```

- **Color** encodes head (7 distinct hues, consistent across both scripts)
- **Line style** encodes eval split in the line plot (solid / dashed / dotted)
- **Output filename** encodes the config: `lineplot_{heads}__{splits}.png` / `boxplot_{heads}__{splits}.png`
- **Output folder**: `outputs/figures/{ENCODER}_classification_{TRAINING_SPLIT}/`

**Line plot** averages per-assay ΔAUPRC across seeds, then plots mean ± std error bars. `ERROR_BARS = "assays"` (default, standard for few-shot benchmarks) shows spread across tasks; `"seeds"` shows reproducibility across the 3 runs.  
**Boxplot** shows per-assay distribution as boxes — always assay variance by construction, no toggle needed.

```bash
# Edit CONFIG block in each script, then:
python Analysis/model/baseline_grid.py       # writes outputs/results/<run_tag>/baseline_grid.csv
python Analysis/model/plot_line_grid.py      # writes outputs/figures/<run_tag>/lineplot_*.png
python Analysis/model/plot_boxplot_grid.py   # writes outputs/figures/<run_tag>/boxplot_*.png
python Analysis/model/fixed_assay_curves.py  # CONFIG: point CSV at a baseline_grid.csv - prints naive vs fixed-assay-set means side by side
```

### Fixed-assay-set finding

The naive support-size curves average over different assay populations at each x-tick (n=16 → 154 assays; n=512 → 11 assays). `fixed_assay_curves.py` restricts to the 11 assays present at every support size and recomputes:

- **Fixed-set curves are monotonically increasing for all models** - the apparent collapse at n=256/512 in the naive curve is entirely due to harder assays entering the population, not model degradation.
- **GNN benefits more from large support than ECFP on the fixed set**: GNN+Random PN-M rises from 0.055 at n=16 to 0.164 at n=512 on the fixed 11 assays; ECFP+Random rises from only 0.031 to 0.081, plateauing earlier.
- The fixed-set means are much lower than the naive means (e.g. GNN n=128: 0.116 fixed vs 0.221 naive), confirming the 11 large assays are a genuinely harder subset.

---

## Headline finding (corrected)

The earlier conclusion - *"scaffold-OOD ΔAUPRC has a structural ceiling ~0.05, intrinsic to the mean-prototype mechanism; RF beats ProtoNet"* - was an **evaluation artifact**. The old scaffold split drew support from a *single* scaffold group, sampled *with replacement* (so "n=128" was often ~10 unique molecules), and frequently produced single-class support → flat 0.5 predictions.

With a leakage-free fair split (multiple whole scaffold groups, no duplication, both classes in support - `data.build_fair_split_indices`), on the same GNN checkpoint (`fsmol_gnn_classification_random` seed 0):

| Scaffold split, ΔAUPRC | n=16 | n=64 | n=128 | n=256 | n=512 |
|---|---|---|---|---|---|
| `emb_proto_euclid` (ProtoNet) | 0.132 | 0.179 | **0.207** | 0.122 | 0.124 |
| `emb_logreg` (adaptive head) | 0.115 | 0.144 | 0.175 | 0.104 | 0.122 |
| `ecfp_rf` (per-task RF) | 0.067 | 0.110 | 0.146 | 0.152 | 0.169 |

What this says (single seed - confirm on seeds 1–2 + ECFP checkpoints, but the effect size is large):

1. **No special scaffold collapse.** Scaffold n=128 (0.207) ≈ random n=128 (0.218); the old "4.5× gap" was the artifact.
2. **Averaging is not the bottleneck** - on the learned embedding the mean-prototype *beats* the adaptive heads at every n.
3. **The learned embedding is the win** - ProtoNet beats per-task RF at all n ≤ 128 (the regime with 146–154 assays). RF only overtakes at n=256/512, which have just 28 and 11 assays (a known assay-selection effect, present equally on the random split).
4. **Euclidean ≈ Mahalanobis** at eval - the train/eval distance mismatch is harmless.

The interventions (TTPA, annealing) were designed to close a gap that turns out to be largely artifactual; read their null results in that light.

For the full per-run paper-reproduction tables and the older (now-superseded) narrative, see the repo git history and [`../../README.md`](../../README.md).
