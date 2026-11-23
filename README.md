# Few-Shot Molecular Property Prediction

Episodic few-shot models for molecular activity prediction, benchmarked on FS-Mol and on
two antimicrobial (MIC) datasets. Two model families are compared under an identical
training and evaluation protocol:

* **PTN** (`model.py`) is a prototypical network. Each molecule is encoded independently,
  the support set is reduced to one prototype per class, and a query is scored by its
  distance to those prototypes.
* **CrossAttentionProtoNet** (`context_model.py`) is an in-context model. Support
  molecules are embedded together with their labels, attend to each other, and queries
  cross-attend to that labelled support set before the same prototype head is applied.
  The motivation is that a support molecule should be interpreted in the context of the
  others, not in isolation.

Encoders are interchangeable: a 10-layer PNA-style GNN with ECFP and descriptor fusion
(`FSMolGNNEncoder`), or an MLP over 2048-bit ECFP4 fingerprints (`ECFPEncoder`).

The primary metric is **delta AUPRC**: AUPRC minus the active ratio of the query set, so
that assays with different class balance can be averaged.

## Results

All numbers below use the GNN encoder, the 4,938-task FS-Mol benchmark training pool, a
random support split, and seed 0. Evaluation is 154 FS-Mol test assays x 6 support sizes
x 5 repeats x 4 heads = 47,348 rows per run. Row counts are identical across runs and the
support and query indices are seeded identically, so every comparison is paired per
(assay, support size, head), n = 2,596. Significance is a Wilcoxon signed-rank test.

### PTN versus attention

Both models were run to convergence with early stopping on validation delta AUPRC
(patience 50). Early stopping fired in every run, so neither model is budget-limited.

| Run | support schedule | d_attn | stopped at epoch | best val | test delta AUPRC | AUROC |
|---|---|---|---|---|---|---|
| PTN | 64 | n/a | 80 | 0.2125 | **0.1335** | 0.6459 |
| PTN | 16+32+64 | n/a | 138 | 0.1776 | 0.1259 | 0.6392 |
| Attention | 64 | 256 | 117 | 0.1688 | 0.1019 | 0.6108 |
| Attention | 16+32+64 | 256 | 208 | 0.1390 | 0.1063 | 0.6154 |
| Attention | 64 | 512 | 166 | 0.1779 | 0.0951 | 0.6013 |

Paired head-to-head, attention minus PTN:

```
attention d256 vs PTN, fixed 64       -0.0340   wins 26.2%   p = 4e-153
attention d512 vs PTN, fixed 64       -0.0408   wins 25.3%   p = 6e-165
attention d256 vs PTN, multi-size     -0.0207   wins 34.0%   p = 4e-77
```

**PTN outperforms the attention model on this benchmark**, by roughly 0.034 delta AUPRC.
Three candidate explanations were tested and rejected:

1. *Training budget.* An earlier 100-epoch grid stopped while attention validation was
   still climbing. Re-running to convergence moved the test metric by +0.0012 for PTN and
   -0.0008 for attention. Longer training does not close the gap.
2. *Metric-space dimension.* PTN computes distances in the encoder's 512-d output while
   the attention model projects to `d_attn=256`. Matching it at 512 (with `mlp_f_hidden`
   raised to 512, otherwise the output is confined to a lower-dimensional subspace) made
   results **worse**, not better.
3. *Procedure asymmetry.* Encoder, task pool, sampler, batch size, query size, episodes
   per epoch, learning rate, seed, split, loss, gradient clipping and evaluation splits
   and heads were verified identical in code. Evaluation does not bypass attention:
   `evaluate.py` routes embeddings through MLP_F and both attention stages before every
   head.

One interaction is worth noting: a multi-size support schedule **helps** the attention
model (+0.0051, p = 7e-07) but **hurts** PTN (-0.0082, p = 1e-25).

### Design-choice screening

One run per design choice, one flag changed, everything else identical. `diff` is variant
minus default. The noise floor at one seed is about 0.005.

| Design choice | Default | Variant | diff | p | Decision |
|---|---|---|---|---|---|
| DC-1 query label | `unknown_token` | `mask` | -0.0040 | 5e-07 | keep default |
| DC-2 nested tensors | `nested` | `dense` | -0.0006 | 0.39 | keep default (tie) |
| DC-3 norm position | `pre` | `post` | -0.0023 | 0.094 | keep default |
| DC-4 MLP_F | `separate` | `shared` | +0.0011 | 0.021 | tie, inside noise |
| DC-5 FFN block | none | `use_ffn` | -0.0040 | 4e-05 | keep default |
| DC-6 attention order | `sequential` | `interleaved` | +0.0005 | 0.033 | tie, inside noise |
| DC-10 softmax | `standard` | `ssmax` | +0.0016 | 2e-05 | variant better |
| DC-10 softmax | `standard` | `qassmax` | +0.0018 | 9e-05 | variant better |
| DC-12 final norm | `shared` | `split` | -0.0041 | 3e-07 | keep default |
| DC-13 logit scale | `sqrt_d` | `none` | -0.0080 | 0.73 | keep default, see below |

Scalable softmax (DC-10) is the only single flag that helps, and only slightly. The best
attention configuration found is **shared MLP_F combined with interleaved attention**
(+0.0060, p = 4e-15, 56.4% of assays), even though each of those flags alone is null.
That is an interaction effect, not an additive gain.

DC-13 needs care. The mean favours `sqrt_d` by 0.0080, but the median favours `none` by
0.0020 and `none` wins 53.4% of assays. Removing the logit temperature is fine on most
assays and catastrophic on a few, so `sqrt_d` is selected for stability rather than for
mean performance.

### Caveats

* One seed. Differences below about 0.005 are not resolvable.
* Mean delta AUPRC per support size is **not** comparable across support sizes: the
  number of qualifying test assays falls from 154 at n=16 to 29 at n=256 and 11 at n=512.
  Comparisons across support sizes must be restricted to a common assay set.
* MIC results from earlier runs are provisional until label binarization is finalised.

## Repository structure

```
model.py              PTN, encoders (ECFPEncoder, FSMolGNNEncoder), prototype heads
context_model.py      CrossAttentionProtoNet and all its configurable design choices
train.py              episodic training loop, early stopping, checkpointing
evaluate.py           FS-Mol test grid and DrugOOD evaluation, head registry
data.py               assay loading, episode sampling, support and query splits
featurize.py          SMILES to graph and fingerprint featurization
mic_loader.py         loaders for the two MIC datasets
config.py             paths and defaults, driven by environment variables
main_ptn.py           entry point for PTN
main_attention.py     entry point for the attention model
Analysis/             figure and measurement scripts (no model required for Analysis/data)
tests/                verification suite, see below
condor/               HTCondor job files for the university GPU cluster
outputs/              generated figures and per-run CSVs (CSVs are not committed)
```

### condor/

Job submission files for the HTCondor scheduler on the university GPU cluster. These are
specific to that cluster and are not needed to run anything locally.

```
setup_env.sub/.sh     builds the virtualenv on the cluster once, before any training
run_job.sh            wrapper every job executes: activates the venv, sets thread
                      environment variables, runs the given python script
convergence.sub       the PTN versus attention convergence runs
gnn_ablation.sub      the 16-arm design-choice screening grid
tests_cpu.sub         verification layers that need no GPU
tests_gpu.sub         verification layers that need a GPU
tests.sub             both, in one job
```

Submit with `condor_submit condor/<file>.sub`. Validate a grid first with
`python tests/validate_condor_grid.py condor/<file>.sub`, which dry-runs every row and
asserts each one resolves to its own output folder. Job logs are written to
`condor/logs/` on the cluster and are not committed.

## Setup

```bash
pip install -r requirements.txt
```

**FS-Mol.** Download from [microsoft/FS-Mol](https://github.com/microsoft/FS-Mol) and
extract to `data/fsmol/`. One file per ChEMBL assay; the loader reads the precomputed
ECFP fingerprints and log-transformed labels directly.

**DrugOOD.** Place the three IC50 files under `data/drugood/`:
`lbap_core_ic50_scaffold.json`, `lbap_core_ic50_size.json`, `lbap_core_ic50_assay.json`.

Paths and a few defaults come from environment variables, so no file needs editing:

| Variable | Default | Purpose |
|---|---|---|
| `PTN_ROOT` | the repository directory | repository root |
| `PTN_DATA` | `$PTN_ROOT/data` | dataset location |
| `PTN_NUM_WORKERS` | auto | DataLoader workers. Set to `0` on Windows, see note below |
| `PTN_SUPPORT_SAMPLING` | `proportional` | `proportional` or `balanced` support class mix |

On Windows, spawn-based DataLoader workers each load torch and the CUDA libraries and can
exhaust the paging file. Set `PTN_NUM_WORKERS=0` there. On a cluster, match it to the
number of CPUs requested.

## Running

```bash
# PTN on FS-Mol, train then evaluate
python main_ptn.py --dataset fsmol --fsmol_tasks benchmark --encoder gnn \
    --training_split random --n_support 64 --seed 0

# attention model, same protocol
python main_attention.py --dataset fsmol --fsmol_tasks benchmark --encoder gnn \
    --training_split random --n_support 64 --seed 0

# run to convergence instead of a fixed 100 epochs (early stopping decides the length)
python main_attention.py --n_epochs 300 ...

# a design-choice variant. Each distinct configuration writes to its own
# outputs/<run_tag>/ folder, so runs cannot overwrite each other
python main_attention.py --softmax_mode ssmax ...
python main_attention.py --no-separate_mlp_f --attn_order interleaved ...
python main_attention.py --d_attn 512 --mlp_f_hidden 512 ...

# multi-size support schedule
python main_attention.py --n_support 16+32+64 ...

# resolve and print a configuration without touching data or the GPU
python main_attention.py --dry_run ...
```

`--n_support` accepts a single integer or several joined with `+`. An assay that cannot
supply the requested support size plus two query molecules, with both classes present in
both folds, is skipped rather than shrunk, so the support size is exact.

## Tests

```bash
python tests/run_all.py              # everything
python tests/run_all.py --fast       # skip the layers that train
python tests/run_all.py --cpu-only   # only layers that never touch CUDA
```

The layers, cheapest first: static source invariants; path and configuration resolution;
an attention mechanism audit that asserts there is no query-to-query information flow, no
label leakage into the query side, no leakage across episodes within a batch, a shared
metric space for support and query, and exact support sizes; Condor grid validation; unit
and numerical tests for every configurable design choice; and an end-to-end run on real
data.

## References

* Stanley et al., *FS-Mol: A Few-Shot Learning Dataset of Molecules*, NeurIPS 2021
  Datasets and Benchmarks.
* Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS 2017.
* Ji et al., *DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for
  AI-aided Drug Discovery*, 2022.
* Nakanishi et al., *Scalable-Softmax Is Superior for Attention*, 2025.
