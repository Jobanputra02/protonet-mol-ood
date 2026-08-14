"""
Evaluation: DrugOOD OOD Benchmark
===================================
Protocol (CHOSEN: zero-shot):
    - Load pretrained model, freeze weights (no gradient updates)
    - Context set sampled from DrugOOD train (in-distribution support)
    - Query set = ood_test (OOD) or iid_test (IID)
    - Test multiple context sizes: 64, 128, 256, 512
    - Primary metric: Delta AUPRC (AUPRC(model) - fraction_actives)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from model import (PrototypicalNetworkRegression, PrototypicalNetworkClassification,
                   ECFPEncoder, FSMolGNNEncoder, _mahalanobis_dists)
from data import DrugOODEvalDataset, AssayDataset


# =============================================================================
# METRICS
# =============================================================================

def delta_auprc(predictions_continuous: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    ΔAUPRC = AUPRC(model) − fraction_of_actives_in_query.
    The subtracted term is the random-classifier baseline (FS-Mol paper, eq. 1).

    CAVEAT: ΔAUPRC's ceiling is (1 − fraction_active). On DrugOOD (74-94% active) that
    ceiling is ~0.22, so this metric is compressed and prevalence-sensitive there. Report
    auroc() as the headline for imbalanced query sets and keep ΔAUPRC as secondary WITH the
    base rate stated.
    """
    auprc_model     = average_precision_score(binary_labels, predictions_continuous)
    random_baseline = float(binary_labels.mean())
    return float(auprc_model - random_baseline)


def auroc(predictions_continuous: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    Area under the ROC curve = P(model scores a random active above a random inactive).
    0.5 = no ranking ability, 1.0 = perfect ordering. Prevalence-INDEPENDENT, so unlike
    ΔAUPRC it is not squeezed by DrugOOD's heavy active-skew - the honest headline metric
    for imbalanced OOD query sets. Returns NaN if the query set is single-class.
    """
    if binary_labels is None or len(np.unique(binary_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(binary_labels, predictions_continuous))


def spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(predictions) < 3:
        return float("nan")
    result = stats.spearmanr(predictions, targets)
    return float(result.statistic)  # type: ignore[union-attr]


def _unwrap(model):
    """Return the underlying module, unwrapping torch.compile's OptimizedModule if present."""
    return getattr(model, '_orig_mod', model)


def _is_gnn(model) -> bool:
    return isinstance(_unwrap(model).encoder, FSMolGNNEncoder)


def _compute_metrics(preds_np, targets_np, binary_labels, is_classification: bool = False):
    """Compute all metrics for one (predictions, targets, binary_labels) triple.

    is_classification=True suppresses RMSE/MAE/Spearman: preds are probabilities [0,1]
    while targets are continuous IC50 values - those distance metrics are meaningless.
    """
    if (binary_labels is not None
            and binary_labels.sum() > 0
            and binary_labels.sum() < len(binary_labels)):
        d_auprc = delta_auprc(preds_np, binary_labels)
        auc     = auroc(preds_np, binary_labels)
    else:
        d_auprc = float("nan")
        auc     = float("nan")

    if is_classification:
        return {
            "delta_auprc": d_auprc,
            "auroc":       auc,
            "rmse":        float("nan"),
            "mae":         float("nan"),
            "spearman":    float("nan"),
        }

    return {
        "delta_auprc": d_auprc,
        "auroc":       auc,
        "rmse":        float(np.sqrt(np.mean((preds_np - targets_np) ** 2))),
        "mae":         float(np.mean(np.abs(preds_np - targets_np))),
        "spearman":    spearman_correlation(preds_np, targets_np),
    }


# =============================================================================
# MULTI-SCALE EVALUATION
# =============================================================================

def _fsmol_dummy_graph():
    """Placeholder graph for molecules whose SMILES fail to parse, keeping indices aligned."""
    from featurize import FSMOL_NODE_FEAT_DIM
    from torch_geometric.data import Data
    return Data(
        x=torch.zeros((1, FSMOL_NODE_FEAT_DIM), dtype=torch.float),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 3), dtype=torch.float),
        ecfp=torch.zeros((1, 2048), dtype=torch.float),
        descriptors=torch.zeros((1, 42), dtype=torch.float),
    )


def _smiles_to_batch(smiles_list: list[str], device: torch.device):
    """Convert a list of SMILES to a PyG Batch on device (FS-Mol featurisation)."""
    from featurize import smiles_to_fsmol_graph
    from torch_geometric.data import Batch as PyGBatch
    dummy = _fsmol_dummy_graph()
    graphs = [smiles_to_fsmol_graph(smi) or dummy for smi in smiles_list]
    return PyGBatch.from_data_list(graphs).to(device)


def _featurize_smiles_list(smiles_list: list[str]) -> list:
    """Featurize SMILES → PyG Data objects (CPU). Pre-compute once per dataset to amortize RDKit cost."""
    from featurize import smiles_to_fsmol_graph
    dummy = _fsmol_dummy_graph()
    return [smiles_to_fsmol_graph(smi) or dummy for smi in smiles_list]


def _encode_smiles_chunked(
    model, smiles_list: list[str], device: torch.device, chunk_size: int = 2048,
) -> torch.Tensor:
    """Encode all SMILES to embeddings in chunks. Returns (n, emb_dim) on device."""
    m = _unwrap(model)
    if not smiles_list:
        return torch.zeros((0,), device=device)
    parts = []
    for start in range(0, len(smiles_list), chunk_size):
        batch = _smiles_to_batch(smiles_list[start:start + chunk_size], device)
        parts.append(m.encoder(batch))
    return torch.cat(parts, dim=0)


def evaluate_drugood_multiscale(
    model: torch.nn.Module,
    eval_datasets: list[DrugOODEvalDataset],
    device: torch.device,
    context_sizes: list[int] = [64, 128, 256, 512],
    seeds: list[int] = [42, 123, 456],
) -> pd.DataFrame:
    """
    Evaluate across all context sizes and both query sets (ood_test + iid_test).
    Each (split × context_size × query_set) combination is run over multiple context
    seeds and averaged - reported as mean ± std across seeds.

    Returns a long-form DataFrame - one row per (split_type × context_size × query_set):
        index | split_type | query_set | context_set_size | actual_context_n | n_test
              | delta_auprc | delta_auprc_std | rmse | rmse_std | mae | spearman
    """
    model.eval()
    gnn             = _is_gnn(model)
    is_classif      = isinstance(_unwrap(model), PrototypicalNetworkClassification)
    rows = []

    with torch.no_grad():
        for eval_dataset in eval_datasets:
            # ------------------------------------------------------------------
            # GNN: pre-featurize all context molecules and pre-encode both query
            # sets ONCE per dataset - amortizes RDKit + GNN cost across all
            # (context_size × query_set × seed) combinations (~24× fewer passes)
            # ------------------------------------------------------------------
            if gnn:
                from torch_geometric.data import Batch as PyGBatch
                ctx_graphs_cache = _featurize_smiles_list(eval_dataset.context_smiles)
                qry_emb_cache = {
                    "ood_test": _encode_smiles_chunked(
                        model, eval_dataset.ood_test_smiles, device),
                    "iid_test": _encode_smiles_chunked(
                        model, eval_dataset.iid_test_smiles, device),
                }

            for context_size in context_sizes:
                for query_set in ["ood_test", "iid_test"]:
                    seed_metrics: list[dict] = []

                    if gnn:
                        pre_qry_emb = qry_emb_cache[query_set]
                        if len(pre_qry_emb) == 0:
                            continue
                        test_labels = (eval_dataset.ood_test_labels if query_set == "ood_test"
                                       else eval_dataset.iid_test_labels)
                        test_binary = (eval_dataset.ood_test_binary if query_set == "ood_test"
                                       else eval_dataset.iid_test_binary)
                        targets_np  = test_labels.numpy()
                        n_ctx       = min(context_size, len(eval_dataset.context_smiles))

                    for seed in seeds:
                        if gnn:
                            rng = np.random.RandomState(seed)
                            # Stratify context by binary label for classification (the
                            # DrugOOD train pool is heavily active-skewed); uniform for
                            # regression or when binary labels are unavailable.
                            if is_classif and eval_dataset.context_binary is not None:
                                from data import stratified_context_indices
                                # balanced=True: 50/50 active/inactive context so the
                                # inactive prototype is not built from ~1 molecule
                                #.
                                idx = stratified_context_indices(
                                    eval_dataset.context_binary, n_ctx, rng, balanced=True)
                            else:
                                idx = rng.choice(
                                    len(eval_dataset.context_smiles), size=n_ctx, replace=False)
                            ctx_batch = PyGBatch.from_data_list(
                                [ctx_graphs_cache[i] for i in idx]
                            ).to(device)
                            if is_classif and eval_dataset.context_binary is not None:
                                ctx_lbl = torch.tensor(
                                    eval_dataset.context_binary[idx].astype(np.float32),
                                    device=device)
                            else:
                                ctx_lbl = eval_dataset.context_labels[idx].to(device)
                            ctx_emb = _unwrap(model).encoder(ctx_batch)
                            preds   = _unwrap(model).predict_from_embeddings(
                                ctx_emb, ctx_lbl, pre_qry_emb
                            )
                            preds_np = preds.cpu().numpy()
                        else:
                            ctx_fp, ctx_labels, test_fp, test_labels, test_binary = \
                                eval_dataset.get_episode(context_size, query_set, seed,
                                                         use_binary=is_classif)
                            if len(test_fp) == 0:
                                continue
                            ctx_lbl    = ctx_labels.to(device)
                            preds      = model.forward(ctx_fp.to(device), ctx_lbl, test_fp.to(device))
                            preds_np   = preds.cpu().numpy()
                            targets_np = test_labels.numpy()

                        seed_metrics.append(_compute_metrics(preds_np, targets_np, test_binary, is_classif))

                    if not seed_metrics:
                        continue

                    # Average metrics across seeds; report std for key metrics
                    with np.errstate(all='ignore'):
                        mean_m = {k: float(np.nanmean([m[k] for m in seed_metrics])) for k in seed_metrics[0]}
                    std_m  = {k: float(np.std([m[k] for m in seed_metrics])) for k in seed_metrics[0]}

                    actual_n = n_ctx if gnn else int(len(ctx_labels))
                    row = {
                        "split_type":        eval_dataset.split_type,
                        "query_set":         query_set,
                        "context_set_size":  context_size,
                        "actual_context_n":  actual_n,
                        "n_test":            int(len(test_labels)),
                        "n_seeds":           len(seed_metrics),
                        "delta_auprc":       mean_m["delta_auprc"],
                        "delta_auprc_std":   std_m["delta_auprc"],
                        "auroc":             mean_m.get("auroc", float("nan")),
                        "auroc_std":         std_m.get("auroc", float("nan")),
                        "rmse":              mean_m["rmse"],
                        "rmse_std":          std_m["rmse"],
                        "mae":               mean_m["mae"],
                        "spearman":          mean_m["spearman"],
                        "spearman_std":      std_m["spearman"],
                    }
                    rows.append(row)

                    print(
                        f"{eval_dataset.split_type:35s} | ctx={context_size:4d} | "
                        f"{query_set:8s} | AUROC: {mean_m.get('auroc', float('nan')):.4f} | "
                        f"dAUPRC: {mean_m['delta_auprc']:+.4f}+/-{std_m['delta_auprc']:.4f} "
                        f"(base rate active shown in n_test) | RMSE: {mean_m['rmse']:.4f}"
                    )

    df = pd.DataFrame(rows).reset_index(drop=True)
    df.index.name = "index"
    return df


# =============================================================================
# CHECKPOINT LOADER
# =============================================================================

def _build_encoder_from_config(config: dict, device: torch.device) -> torch.nn.Module:
    """Reconstruct the encoder from the config dict stored in a checkpoint."""
    encoder_type  = config.get("encoder_type", "ecfp")
    embedding_dim = config.get("embedding_dim", 256)
    deg_list      = config.get("deg", None)
    deg           = torch.tensor(deg_list, dtype=torch.long) if deg_list is not None else None

    if encoder_type == "gnn":
        from featurize import FSMOL_NODE_FEAT_DIM
        encoder = FSMolGNNEncoder(
            node_feat_dim=FSMOL_NODE_FEAT_DIM,
            hidden_channels=config.get("hidden_channels", 128),
            num_layers=config.get("num_layers", 10),
            embedding_dim=embedding_dim,
            deg=deg,
        )
    else:  # ecfp (default)
        encoder = ECFPEncoder(
            input_dim=2048,
            hidden_dim=config.get("hidden_dim", 512),
            embedding_dim=embedding_dim,
        )
    return encoder.to(device)


def _load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[PrototypicalNetworkRegression | PrototypicalNetworkClassification, dict]:
    """Load a regression or classification model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint["config"]
    model_type = config.get("model_type", "regression")

    encoder = _build_encoder_from_config(config, device)
    cls     = PrototypicalNetworkClassification if model_type == "classification" \
              else PrototypicalNetworkRegression
    model   = cls(encoder).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if model_type == "classification":
        val_metric = checkpoint.get("val_delta_auprc", None)
        val_str    = f"dAUPRC={val_metric:+.4f}" if val_metric is not None else "?"
    else:
        val_metric = checkpoint.get("val_rmse", checkpoint.get("val_loss", None))
        val_str    = f"RMSE={val_metric:.4f}" if val_metric is not None else "?"
    enc_type = config.get("encoder_type", "ecfp")
    epoch_num = checkpoint.get("epoch", checkpoint.get("step", "?"))
    print(f"Loaded {enc_type}/{model_type} model from epoch {epoch_num} (Val {val_str})")
    return model, config


def _encode_assay(
    model, assay: AssayDataset, device: torch.device, gnn: bool
) -> torch.Tensor:
    """Pre-encode every molecule in an assay. Returns (n, emb_dim) on device.
    Call once per assay; slice embeddings per episode - eliminates repeated GNN passes."""
    m = _unwrap(model)
    with torch.no_grad():
        if gnn:
            from data import _build_fsmol_graphs_for_assay
            from torch_geometric.data import Batch as PyGBatch
            batch = PyGBatch.from_data_list(_build_fsmol_graphs_for_assay(assay)).to(device)
            return m.encoder(batch)   # (n, emb_dim)
        else:
            fp = torch.tensor(np.stack(assay.fingerprints), dtype=torch.float32).to(device)
            return m.encoder(fp)      # (n, emb_dim)


def _get_mol_sizes(assay: AssayDataset) -> np.ndarray:
    """
    Heavy atom count per molecule.
    assay.scaffolds stores the original SMILES when loaded via _load_assay_file.
    """
    from rdkit import Chem  # type: ignore
    sizes = []
    for smi in assay.scaffolds:
        mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
        sizes.append(mol.GetNumHeavyAtoms() if mol is not None else 0)
    return np.array(sizes, dtype=np.int32)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel as _SKKernel

_RF_PARAMS = dict(n_estimators=100, max_depth=10, max_features="sqrt",
                  min_samples_leaf=2, n_jobs=-1, random_state=0)


class _TanimotoKernel(_SKKernel):
    """
    Tanimoto similarity kernel for binary ECFP fingerprints.

    k(x, y) = |x ∩ y| / |x ∪ y|  (generalised to count vectors via dot-product form)

    No hyperparameters — the kernel encodes the fixed domain-appropriate similarity
    for molecular fingerprints. Passed to GaussianProcessClassifier with optimizer=None
    so sklearn skips log-marginal-likelihood optimisation.
    """

    @property
    def hyperparameters(self):
        return []

    @property
    def theta(self):
        return np.empty(0)

    @theta.setter
    def theta(self, theta):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2))

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=np.float64)
        Y = X if Y is None else np.asarray(Y, dtype=np.float64)
        Xb = (X > 0).astype(np.float64)
        Yb = (Y > 0).astype(np.float64)
        inter = Xb @ Yb.T
        K = inter / np.maximum(Xb.sum(1, keepdims=True) + Yb.sum(1, keepdims=True).T - inter, 1e-8)
        if eval_gradient:
            return K, np.zeros((*K.shape, 0))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False

    def clone_with_theta(self, theta):
        return _TanimotoKernel()

    def get_params(self, deep=True):
        return {}


def _h_proto_euclidean(Xs, ys, Xq):
    """Mean-prototype + squared-Euclidean (the distance ProtoNet trains with)."""
    pa, pi = Xs[ys == 1].mean(0), Xs[ys == 0].mean(0)
    da = ((Xq - pa) ** 2).sum(1)
    di = ((Xq - pi) ** 2).sum(1)
    return 1.0 / (1.0 + np.exp(np.clip(da - di, -30, 30)))


def _h_proto_mahalanobis(Xs, ys, Xq):
    """Mean-prototype + FS-Mol shrinkage Mahalanobis (the paper's eval head)."""
    sup = torch.tensor(np.asarray(Xs), dtype=torch.float32)
    qry = torch.tensor(np.asarray(Xq), dtype=torch.float32)
    active = torch.tensor(ys.astype(bool))
    protos = torch.stack([sup[active].mean(0), sup[~active].mean(0)], dim=0)
    dists = _mahalanobis_dists(qry, protos, sup, active)
    return torch.softmax(-dists, dim=1)[:, 0].numpy()


def _h_sklearn(clf, Xs, ys, Xq):
    clf.fit(Xs, ys)
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.zeros(len(Xq), dtype=np.float32)
    return clf.predict_proba(Xq)[:, classes.index(1)]


def _h_logreg(Xs, ys, Xq):
    return _h_sklearn(make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=1000, C=1.0)), Xs, ys, Xq)


def _h_knn(Xs, ys, Xq, k=5):
    kk = max(1, min(k, len(ys) - 1))
    return _h_sklearn(make_pipeline(StandardScaler(),
                                    KNeighborsClassifier(n_neighbors=kk)), Xs, ys, Xq)


def _h_rf(Xs, ys, Xq):
    return _h_sklearn(RandomForestClassifier(**_RF_PARAMS), Xs, ys, Xq)


def _h_proto_tanimoto(Xs, ys, Xq):
    """
    Tanimoto-kernel prototype classifier.

    For each query molecule q, scores its mean Tanimoto similarity to the active
    support molecules vs the inactive support molecules, then converts to a probability
    via softmax over the two class similarities.

    Motivation: Tanimoto is the canonical distance for ECFP fingerprints; mean-prototype
    with Euclidean distance (_h_proto_euclidean) treats bit vectors as Cartesian points,
    which has no geometric justification. This head uses the domain-appropriate geometry.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    Xq = np.asarray(Xq, dtype=np.float64)
    ys = np.asarray(ys)
    Xsb = (Xs > 0).astype(np.float64)
    Xqb = (Xq > 0).astype(np.float64)
    inter = Xqb @ Xsb.T                                  # (nq, ns)
    T = inter / np.maximum(
        Xqb.sum(1, keepdims=True) + Xsb.sum(1, keepdims=True).T - inter, 1e-8
    )                                                     # (nq, ns) Tanimoto matrix
    sim_a = T[:, ys == 1].mean(1)                        # mean similarity to actives
    sim_i = T[:, ys == 0].mean(1)                        # mean similarity to inactives
    denom  = np.maximum(sim_a + sim_i, 1e-8)
    return sim_a / denom                                  # P(active) in [0, 1]


def _h_gp_tanimoto(Xs, ys, Xq):
    """
    Gaussian process classifier with Tanimoto kernel on ECFP fingerprints.

    Fits a GPC (Laplace approximation) on the support set using the Tanimoto similarity
    kernel — the domain-canonical kernel for molecular fingerprints. No hyperparameter
    optimisation (optimizer=None): the kernel has no free parameters by design.

    Motivation: the Tanimoto-kernel GP is the optimal kernel machine for molecular
    fingerprints. It directly generalises the Tanimoto prototype (_h_proto_tanimoto):
    where the prototype uses a single class mean as the basis, the GP uses every support
    molecule as a kernel basis function. If the GP beats ProtoNet, that validates using
    the right kernel; if ProtoNet beats the GP, that validates the learned representation.
    """
    gpc = GaussianProcessClassifier(kernel=_TanimotoKernel(), optimizer=None,
                                    random_state=0)
    gpc.fit(Xs, ys)
    classes = list(gpc.classes_)
    if 1 not in classes:
        return np.zeros(len(Xq), dtype=np.float32)
    return gpc.predict_proba(Xq)[:, classes.index(1)]


# head -> (callable, representation "embedding"|"ecfp")
FSMOL_HEAD_REGISTRY = {
    # ── Embedding heads (use trained encoder output) ──────────────────────────
    "emb_proto_euclid":      (_h_proto_euclidean,   "embedding"),
    "emb_proto_mahalanobis": (_h_proto_mahalanobis, "embedding"),
    "emb_logreg":            (_h_logreg,            "embedding"),
    "emb_knn":               (_h_knn,               "embedding"),
    # ── ECFP heads (model-free, raw 2048-bit fingerprints) ───────────────────
    # Geometry ablation: Euclidean (ad hoc) → Tanimoto (domain-canonical) → GP
    "ecfp_proto_euclid":     (_h_proto_euclidean,   "ecfp"),
    "ecfp_proto_tanimoto":   (_h_proto_tanimoto,    "ecfp"),
    "ecfp_gp_tanimoto":      (_h_gp_tanimoto,       "ecfp"),
    "ecfp_logreg":           (_h_logreg,            "ecfp"),
    "ecfp_rf":               (_h_rf,                "ecfp"),
}


def _butina_groups_for(assay: AssayDataset) -> dict:
    """Butina@0.70 fingerprint-similarity groups for the 'similarity' eval split.
    Fixed cutoff: config.SCAFFOLD_OOD_CUTOFF = 0.70."""
    from config import SCAFFOLD_OOD_CUTOFF
    from data import build_butina_groups
    return build_butina_groups(assay.fingerprints, SCAFFOLD_OOD_CUTOFF)


def _mean_max_tanimoto(Fs: np.ndarray, Fq: np.ndarray) -> float:
    """Mean over query molecules of the nearest-support Tanimoto (OOD-severity probe)."""
    s = (Fs > 0).astype(np.float32)
    q = (Fq > 0).astype(np.float32)
    inter = q @ s.T
    tani = inter / (q.sum(1, keepdims=True) + s.sum(1, keepdims=True).T - inter + 1e-8)
    return float(tani.max(1).mean())


def evaluate_fsmol_test_grid(
    model: torch.nn.Module,
    test_assays,
    device: torch.device,
    splits=("random", "scaffold", "similarity", "size"),
    support_sizes=(16, 32, 64, 128, 256, 512),
    n_repeats: int = 5,
    base_seed: int = 42,
    head_names=None,
) -> pd.DataFrame:
    """
    Canonical FS-Mol test evaluator: representation × head on shared fair splits.

    Returns one row per (assay, split, support_size, repeat, head):
        assay_id | split_type | support_size | support_actual | n_query |
        repeat | representation | head | delta_auprc | auroc | query_support_sim

    split_type values:
        "random"     — support drawn uniformly at random
        "scaffold"   — Murcko scaffold group-disjoint (assay.scaffold_groups, built at load)
        "similarity" — Butina@0.70 fingerprint-cluster-disjoint (config.SCAFFOLD_OOD_CUTOFF)
        "size"       — support from smallest molecules, query from largest
    """
    from config import SCAFFOLD_OOD_CUTOFF as _EVAL_CUTOFF
    from data import build_fair_split_indices
    if head_names is None:
        head_names = list(FSMOL_HEAD_REGISTRY.keys())
    need_emb = any(FSMOL_HEAD_REGISTRY[h][1] == "embedding" for h in head_names)
    gnn = False
    if need_emb:
        if model is None:
            raise ValueError("emb heads requested but model is None.")
        gnn = _is_gnn(model)
    need_sizes      = "size"       in splits
    need_similarity = "similarity" in splits
    # Seed hash per split type — kept stable so results are reproducible across runs.
    _SI = {"random": 1, "scaffold": 2, "similarity": 3, "size": 4}
    rows = []

    for ai, assay in enumerate(test_assays):
        if assay.binary_labels is None:
            continue
        n_total   = len(assay)
        y_all     = np.asarray(assay.binary_labels)
        E_all     = (_encode_assay(model, assay, device, gnn).detach().cpu().numpy()
                     if need_emb else None)
        F_all     = np.stack(assay.fingerprints).astype(np.float32)
        mol_sizes = _get_mol_sizes(assay) if need_sizes else None
        # Murcko groups built at load time (zero cost); Butina built once per assay if needed.
        murcko_groups  = assay.scaffold_groups
        butina_groups  = _butina_groups_for(assay) if need_similarity else None

        for split in splits:
            # Map split name → (groups_dict, split_type_for_build_fair_split_indices)
            # "scaffold" and "similarity" both use the group-disjoint protocol ("scaffold")
            # in build_fair_split_indices; they differ only in which groups are supplied.
            if split == "scaffold":
                groups, bfsi_split = murcko_groups, "scaffold"
            elif split == "similarity":
                groups, bfsi_split = butina_groups, "scaffold"
            else:
                groups, bfsi_split = murcko_groups, split   # "random" / "size" ignore groups

            for n_sup in support_sizes:
                for rep in range(n_repeats):
                    rng = np.random.RandomState(
                        base_seed + 1009 * rep + 7919 * n_sup + 10000 * _SI.get(split, 0))
                    sp = build_fair_split_indices(
                        n_total, groups, y_all, n_sup, bfsi_split, rng,
                        mol_sizes=mol_sizes, require_both_classes=True)
                    if sp is None:
                        continue
                    s_idx, q_idx = sp
                    if len(np.unique(y_all[q_idx])) < 2:
                        continue
                    ys, yq = y_all[s_idx], y_all[q_idx]
                    qsim = _mean_max_tanimoto(F_all[s_idx], F_all[q_idx])
                    for head in head_names:
                        fn, rep_repr = FSMOL_HEAD_REGISTRY[head]
                        p = (fn(F_all[s_idx], ys, F_all[q_idx]) if rep_repr == "ecfp"
                             else fn(E_all[s_idx], ys, E_all[q_idx]))
                        rows.append({
                            "assay_id": assay.assay_id, "split_type": split,
                            "support_size": n_sup, "support_actual": int(len(s_idx)),
                            "n_query": int(len(q_idx)), "repeat": rep,
                            "representation": rep_repr, "head": head,
                            "delta_auprc": delta_auprc(p, yq), "auroc": auroc(p, yq),
                            "query_support_sim": qsim,
                        })
        if (ai + 1) % 20 == 0:
            print(f"  eval {ai + 1}/{len(test_assays)} assays...", flush=True)
    return pd.DataFrame(rows)
