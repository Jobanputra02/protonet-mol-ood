#!/usr/bin/env python3
"""
Comprehensive EDA — all four datasets
  1. FS-Mol
  2. DrugOOD (lbap_core ic50/ec50/ki/potency, scaffold split)
  3. curated_mic
  4. chembl36_curated (ALL standard_types, no filter)

Analyses per dataset:
  - Total assays / molecules / mean / median / std / min / max
  - Assay size distribution
  - Chemical properties: MW, LogP, HBD, HBA, TPSA, RotBonds,
    ArRings, HAC, fsp3, BertzCT, QED, Rings  (deduplicated SMILES)
  - pActivity / pMIC distribution + binarization threshold
  - Qualification curve (n_support = 8/16/32/64/128/256)
  - Murcko scaffold diversity (unique scaffolds / assay)
  - NN-Tanimoto similarity distribution (structural sparsity)
  - Activity cliff rate (similar structure, opposite label)
  - UMAP chemical space (50K stratified sample)
  - Temporal distribution (year, where available)
  - Document count (where available)
  - Assay type / standard_type breakdown
  - Organism / Gram+/- breakdown (where available)
  - DrugOOD: IID vs OOD property comparison per endpoint

Cross-dataset:
  - Compound overlap heatmap (canonical SMILES)
  - Property violin comparison (MW, LogP, HAC, QED)
  - NN-Tanimoto comparison
  - Combined UMAP (10K per dataset)
"""

import os, sys, json, gzip, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import base64, io

warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 9, "figure.dpi": 100,
})

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed — UMAP skipped", flush=True)

# ── Paths ───────────────────────────────────────────────────────────────────
BASE        = Path("/netscratch/jobanputra/PTN")
FSMOL_DIR   = BASE / "data/fsmol"
DRUGOOD_DIR = BASE / "data/drugood"
MIC_CSV     = BASE / "data/custom/curated_mic_assays_balanced.csv"
CHEMBL_CSV  = BASE / "data/custom/chembl36_curated.csv"
OUT_HTML    = BASE / "Analysis/eda_report.html"
(BASE / "Analysis").mkdir(exist_ok=True)

t0 = time.time()
print("=== EDA started ===", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

PROP_COLS = ["mw","logp","hbd","hba","tpsa","rotbonds","arom","hac","fsp3","bertz","qed","rings"]
PROP_LABELS = {
    "mw":"MW (Da)", "logp":"LogP", "hbd":"HBD", "hba":"HBA",
    "tpsa":"TPSA (Å²)", "rotbonds":"Rot Bonds", "arom":"Arom Rings",
    "hac":"HAC", "fsp3":"fsp3", "bertz":"BertzCT", "qed":"QED", "rings":"Rings",
}

def mol_props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return {
            "mw":       Descriptors.MolWt(m),
            "logp":     Descriptors.MolLogP(m),
            "hbd":      rdMolDescriptors.CalcNumHBD(m),
            "hba":      rdMolDescriptors.CalcNumHBA(m),
            "tpsa":     Descriptors.TPSA(m),
            "rotbonds": rdMolDescriptors.CalcNumRotatableBonds(m),
            "arom":     rdMolDescriptors.CalcNumAromaticRings(m),
            "hac":      m.GetNumHeavyAtoms(),
            "fsp3":     rdMolDescriptors.CalcFractionCSP3(m),
            "bertz":    Descriptors.BertzCT(m),
            "qed":      Chem.QED.qed(m),
            "rings":    rdMolDescriptors.CalcNumRings(m),
        }
    except Exception:
        return None

def ecfp4(smi, nbits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)

def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def murcko_smi(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None

def props_df(smiles_list):
    """Compute properties on deduplicated SMILES list."""
    seen = set(); rows = []
    for s in smiles_list:
        if not s or s in seen:
            continue
        seen.add(s)
        p = mol_props(s)
        if p:
            rows.append(p)
    return pd.DataFrame(rows)

def nn_tanimoto_dist(fps, n=5000):
    """Return list of max NN-Tanimoto for each sampled molecule."""
    if len(fps) == 0:
        return []
    idx = np.random.choice(len(fps), min(n, len(fps)), replace=False)
    sample = [fps[i] for i in idx]
    sims = []
    for i, fp in enumerate(sample):
        rest = sample[:i] + sample[i+1:]
        if rest:
            sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, rest)))
    return sims

def activity_cliff_rate(assay_dict, n_assays=200, thr=0.85):
    """Fraction of similar pairs (Tanimoto>thr) with opposite labels."""
    keys = list(assay_dict.keys())
    if len(keys) > n_assays:
        keys = list(np.random.choice(keys, n_assays, replace=False))
    cliff = total = 0
    for k in keys:
        sml = assay_dict[k]
        fps, lbs = [], []
        for s, lb in sml:
            fp = ecfp4(s)
            if fp is not None:
                fps.append(fp); lbs.append(lb)
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                total += 1
                if DataStructs.TanimotoSimilarity(fps[i], fps[j]) >= thr and lbs[i] != lbs[j]:
                    cliff += 1
    return cliff / total if total else 0.0

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── Plot helpers ─────────────────────────────────────────────────────────────

def plot_prop_grid(df, title, color="#4878CF"):
    cols = PROP_COLS
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        if col in df.columns and df[col].notna().sum() > 0:
            v = df[col].dropna().values
            axes[i].hist(v, bins=50, color=color, edgecolor="none", alpha=0.8)
            axes[i].set_title(PROP_LABELS[col], fontsize=9)
            axes[i].axvline(v.mean(), color="red", lw=1, ls="--", label=f"μ={v.mean():.2f}")
            axes[i].axvline(np.median(v), color="orange", lw=1, ls=":", label=f"med={np.median(v):.2f}")
            axes[i].legend(fontsize=6)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_size_dist(sizes, title, color="#4878CF"):
    sizes = np.array(sizes)
    if len(sizes) == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(sizes, bins=60, color=color, edgecolor="none", alpha=0.8)
    axes[0].axvline(sizes.mean(), color="red", lw=1.2, ls="--", label=f"mean={sizes.mean():.0f}")
    axes[0].axvline(np.median(sizes), color="orange", lw=1.2, ls=":", label=f"med={np.median(sizes):.0f}")
    axes[0].set_xlabel("molecules/assay"); axes[0].set_ylabel("assays"); axes[0].legend(fontsize=8)
    axes[1].hist(np.log10(sizes + 1), bins=60, color=color, edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("log10(size)"); axes[1].set_ylabel("assays")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_qual_curve(pos_neg_list, title, color="#5a9e6f"):
    supports = [8, 16, 32, 64, 128, 256]
    counts = {s: sum(1 for p, n in pos_neg_list if p >= s and n >= s) for s in supports}
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(s) for s in supports], [counts[s] for s in supports], color=color)
    for bar, s in zip(bars, supports):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{counts[s]:,}", ha="center", fontsize=8)
    ax.set_xlabel("n_support (pos=neg=k)"); ax.set_ylabel("qualifying assays")
    ax.set_title(f"Qualification curve — {title}", fontsize=10)
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_nn_tanimoto(sims_dict, title):
    """sims_dict: {label: [sim values]}"""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4878CF", "#e07b54", "#5a9e6f", "#9b5de5"]
    for i, (label, sims) in enumerate(sims_dict.items()):
        if sims:
            ax.hist(sims, bins=50, alpha=0.6, color=colors[i % len(colors)],
                    edgecolor="none", label=f"{label} (n={len(sims):,})")
    ax.set_xlabel("NN-Tanimoto similarity"); ax.set_ylabel("count")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_umap(fp_mat, labels_num, title, cmap="tab10"):
    if not HAS_UMAP or fp_mat is None or len(fp_mat) == 0:
        return None
    print(f"    UMAP {len(fp_mat):,} pts…", flush=True)
    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=8,
                             n_neighbors=30, min_dist=0.1, metric="jaccard")
    emb = reducer.fit_transform(fp_mat)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:,0], emb[:,1], c=labels_num, cmap=cmap,
                    s=2, alpha=0.5, linewidths=0)
    plt.colorbar(sc, ax=ax)
    ax.set_title(title, fontsize=10); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    return fig_to_b64(fig)

def fps_from_smiles(smiles_list, n_umap=50000):
    """Return (fps_list, labels_or_None) after stratified subsample."""
    if len(smiles_list) > n_umap:
        idx = np.random.choice(len(smiles_list), n_umap, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
    fps = [ecfp4(s) for s in smiles_list]
    valid = [fp is not None for fp in fps]
    fps = [fp for fp, v in zip(fps, valid) if v]
    return fps, valid

def fps_to_mat(fps):
    if not fps:
        return None
    return np.array([list(fp.ToBitString()) for fp in fps], dtype=np.uint8)

def fig_hbar(counts, title, color, xlabel="records", top=20):
    counts = counts.head(top)
    fig, ax = plt.subplots(figsize=(9, max(4, len(counts)*0.35)))
    ax.barh(counts.index[::-1], counts.values[::-1], color=color, alpha=0.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel)
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v * 1.005, i, f"{v:,}", va="center", fontsize=7)
    plt.tight_layout()
    return fig_to_b64(fig)

def fig_bar(counts, title, color, xlabel="", ylabel="records"):
    fig, ax = plt.subplots(figsize=(max(5, len(counts)*0.5), 4))
    ax.bar(counts.index.astype(str), counts.values, color=color, alpha=0.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig_to_b64(fig)

def fig_hist(vals, title, xlabel, color, vlines=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=60, color=color, edgecolor="none", alpha=0.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    if vlines:
        for v, lbl, ls in vlines:
            ax.axvline(v, lw=1.2, ls=ls, label=lbl)
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig_to_b64(fig)

def summary_stats(sizes_arr):
    a = np.array(sizes_arr)
    if len(a) == 0:
        return {"count":0,"mean":0,"median":0,"std":0,"min":0,"max":0,"p25":0,"p75":0}
    return {
        "count": int(len(a)), "mean": float(a.mean()),
        "median": float(np.median(a)), "std": float(a.std()),
        "min": int(a.min()), "max": int(a.max()),
        "p25": float(np.percentile(a,25)), "p75": float(np.percentile(a,75)),
    }

# ── HTML helpers ──────────────────────────────────────────────────────────────

def img(b64, alt=""):
    if b64 is None:
        return f"<p style='color:gray;font-style:italic'>[figure unavailable: {alt}]</p>"
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:6px;margin:6px 0">'

def stat_table(d):
    rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{v if not isinstance(v,float) else f'{v:,.4f}'}</td></tr>"
        for k,v in d.items()
    )
    return f"<table class='stat'>{rows}</table>"

def section(title, content, open_=True):
    return f"""
<details {'open' if open_ else ''}>
<summary><h2 style='display:inline'>{title}</h2></summary>
<div class='sec'>{content}</div>
</details>"""

def h3(t):
    return f"<h3>{t}</h3>"


# ════════════════════════════════════════════════════════════════════════════
# 1. FS-MOL
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/4] FS-Mol", flush=True)

fsmol_assays = {}   # aid -> {smiles, labels, assay_type, regression, split}
fsmol_total  = 0

for split in ("train", "valid", "test"):
    sp_dir = FSMOL_DIR / split
    files  = sorted(sp_dir.glob("*.jsonl.gz")) if sp_dir.exists() else []
    print(f"  {split}: {len(files)} files", flush=True)
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    r   = json.loads(line)
                    aid = r.get("Assay_ID", f.stem)
                    if aid not in fsmol_assays:
                        fsmol_assays[aid] = {
                            "smiles": [], "labels": [], "assay_type": set(),
                            "regression": [], "split": split,
                        }
                    smi = r.get("SMILES", "")
                    if smi:
                        fsmol_assays[aid]["smiles"].append(smi)
                    # Property can be "1.0" or 1 — handle both
                    fsmol_assays[aid]["labels"].append(int(float(r.get("Property", 0))))
                    fsmol_assays[aid]["assay_type"].add(str(r.get("AssayType", "")))
                    reg = r.get("RegressionProperty")
                    if reg is not None:
                        fsmol_assays[aid]["regression"].append(float(reg))
                    fsmol_total += 1
        except Exception as e:
            print(f"    [warn] {f.name}: {e}", flush=True)

print(f"  Total records: {fsmol_total:,}  assays: {len(fsmol_assays):,}", flush=True)

fsmol_sizes = [len(v["smiles"]) for v in fsmol_assays.values()]
fsmol_pos   = [sum(v["labels"]) for v in fsmol_assays.values()]
fsmol_neg   = [len(v["labels"]) - p for v, p in zip(fsmol_assays.values(), fsmol_pos)]
fsmol_pos_rate = np.mean([p/(p+n) for p,n in zip(fsmol_pos,fsmol_neg) if p+n > 0])

fsmol_stats = {
    "n_assays": len(fsmol_assays),
    "n_records": fsmol_total,
    **{f"size_{k}": v for k,v in summary_stats(fsmol_sizes).items()},
    "pos_rate_mean": float(fsmol_pos_rate),
}

# Assay type
all_atypes = Counter()
for v in fsmol_assays.values():
    all_atypes.update(v["assay_type"])
fsmol_atype_b64 = fig_bar(pd.Series(all_atypes).sort_values(ascending=False),
                           "FS-Mol — Assay types", "#4878CF")

# Size distribution
fsmol_size_b64 = plot_size_dist(fsmol_sizes, "FS-Mol assay sizes")

# Qualification curve
fsmol_qual_b64 = plot_qual_curve(list(zip(fsmol_pos, fsmol_neg)), "FS-Mol")

# pIC50 distribution
all_reg = [r for v in fsmol_assays.values() for r in v["regression"]]
if all_reg:
    fsmol_reg_b64 = fig_hist(
        all_reg, "FS-Mol — RegressionProperty (pIC50) distribution", "pIC50",
        "#4878CF", vlines=[(6.0,"threshold=6","-.")]
    )
else:
    fsmol_reg_b64 = None

# Split distribution
split_counts = Counter(v["split"] for v in fsmol_assays.values())
fsmol_split_b64 = fig_bar(pd.Series(split_counts).sort_values(ascending=False),
                           "FS-Mol — Assays per split", "#4878CF", ylabel="assays")

# Molecular properties (deduped SMILES)
all_fsmol_smiles = [s for v in fsmol_assays.values() for s in v["smiles"]]
print(f"  Properties on {len(all_fsmol_smiles):,} SMILES (deduped)…", flush=True)
fsmol_prop_df = props_df(all_fsmol_smiles)
print(f"  → {len(fsmol_prop_df):,} unique valid molecules", flush=True)
fsmol_prop_b64 = plot_prop_grid(fsmol_prop_df, "FS-Mol — Molecular Properties")
fsmol_stats["n_unique_mols"] = len(fsmol_prop_df)

# Murcko scaffold diversity
print("  Murcko diversity…", flush=True)
scaff_divs = []
for v in fsmol_assays.values():
    sc = [murcko_smi(s) for s in v["smiles"][:200] if s]
    sc = [x for x in sc if x]
    if sc:
        scaff_divs.append(len(set(sc)) / len(sc))
fsmol_stats["scaffold_diversity_mean"] = float(np.mean(scaff_divs)) if scaff_divs else 0

# NN-Tanimoto
print("  NN-Tanimoto…", flush=True)
fsmol_fps_nn = [ecfp4(s) for s in np.random.choice(all_fsmol_smiles, min(8000, len(all_fsmol_smiles)), replace=False)]
fsmol_fps_nn = [fp for fp in fsmol_fps_nn if fp is not None]
fsmol_nn_sims = nn_tanimoto_dist(fsmol_fps_nn, n=5000)
fsmol_nn_b64  = plot_nn_tanimoto({"FS-Mol": fsmol_nn_sims}, "FS-Mol — NN-Tanimoto distribution")
fsmol_stats["nn_tanimoto_mean"] = float(np.mean(fsmol_nn_sims)) if fsmol_nn_sims else 0

# Activity cliffs
print("  Activity cliffs…", flush=True)
fsmol_cliff_dict = {
    aid: [(s, lb) for s, lb in zip(v["smiles"][:80], v["labels"][:80])]
    for aid, v in fsmol_assays.items()
}
fsmol_cliff = activity_cliff_rate(fsmol_cliff_dict, n_assays=150)
fsmol_stats["activity_cliff_rate"] = float(fsmol_cliff)

# UMAP
print("  UMAP…", flush=True)
all_lbs_fsmol = [lb for v in fsmol_assays.values() for lb in v["labels"]]
umap_smi_fs = all_fsmol_smiles[:]
umap_lbs_fs = all_lbs_fsmol[:]
if len(umap_smi_fs) > 50000:
    idx = np.random.choice(len(umap_smi_fs), 50000, replace=False)
    umap_smi_fs = [umap_smi_fs[i] for i in idx]
    umap_lbs_fs = [umap_lbs_fs[i] for i in idx]
fps_fs = [ecfp4(s) for s in umap_smi_fs]
valid  = [fp is not None for fp in fps_fs]
fps_fs = [fp for fp, v in zip(fps_fs, valid) if v]
lbs_fs = [lb for lb, v in zip(umap_lbs_fs, valid) if v]
fsmol_umap_b64 = plot_umap(fps_to_mat(fps_fs), lbs_fs, "FS-Mol UMAP (50K, label)")

print(f"  FS-Mol done. {time.time()-t0:.0f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# 2. DrugOOD — all 4 endpoints
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/4] DrugOOD", flush=True)

DO_EPS = ["lbap_core_ic50_scaffold", "lbap_core_ec50_scaffold",
          "lbap_core_ki_scaffold",   "lbap_core_potency_scaffold"]
DO_COLORS = {"lbap_core_ic50_scaffold":"#e07b54",
             "lbap_core_ec50_scaffold":"#e0b454",
             "lbap_core_ki_scaffold":"#54b4e0",
             "lbap_core_potency_scaffold":"#7b54e0"}

drugood = {}
for ep in DO_EPS:
    fp = DRUGOOD_DIR / f"{ep}.json"
    if not fp.exists():
        print(f"  [warn] {ep}.json not found", flush=True)
        continue
    with open(fp) as fh:
        raw = json.load(fh)
    records = []
    for sname in ("train","ood_val","ood_test","iid_val","iid_test"):
        for r in raw.get("split",{}).get(sname,[]):
            r["_split"] = sname
            records.append(r)
    drugood[ep] = pd.DataFrame(records)
    print(f"  {ep}: {len(drugood[ep]):,} records", flush=True)

drugood_ep_stats  = {}
drugood_prop_b64s = {}
drugood_umap_b64s = {}
drugood_nn_b64s   = {}
drugood_pic50_b64s = {}
drugood_split_b64s = {}
drugood_ood_b64s   = {}  # IID vs OOD property comparison
drugood_all_fps   = []   # for combined UMAP later

for ep, df in drugood.items():
    short = ep.replace("lbap_core_","").replace("_scaffold","")
    color = DO_COLORS[ep]
    print(f"  [{short}]", flush=True)

    n_assays  = df["assay_id"].nunique()  if "assay_id"  in df.columns else 0
    n_domains = df["domain_id"].nunique() if "domain_id" in df.columns else 0
    reg_vals  = df["reg_label"].dropna().values if "reg_label" in df.columns else np.array([])
    cls_vals  = df["cls_label"].dropna().values if "cls_label" in df.columns else np.array([])
    sizes     = df.groupby("assay_id").size().values if "assay_id" in df.columns else np.array([1]*len(df))

    drugood_ep_stats[short] = {
        "n_records": len(df),
        "n_assays": int(n_assays),
        "n_domains": int(n_domains),
        "pos_rate": float(cls_vals.mean()) if len(cls_vals) else 0,
        "reg_mean": float(reg_vals.mean()) if len(reg_vals) else 0,
        "reg_median": float(np.median(reg_vals)) if len(reg_vals) else 0,
        **{f"size_{k}": v for k,v in summary_stats(sizes).items()},
    }

    # Split distribution
    sc = df["_split"].value_counts() if "_split" in df.columns else pd.Series()
    drugood_split_b64s[short] = fig_bar(sc, f"DrugOOD {short} — split counts", color, ylabel="records")

    # pIC50 distribution
    if len(reg_vals):
        vls = [(reg_vals.mean(), f"μ={reg_vals.mean():.2f}", "--"),
               (np.median(reg_vals), f"med={np.median(reg_vals):.2f}", ":")]
        drugood_pic50_b64s[short] = fig_hist(reg_vals, f"DrugOOD {short} — pIC50", "pIC50", color, vlines=vls)

    # Molecular properties (deduped)
    smiles = df["smiles"].dropna().tolist() if "smiles" in df.columns else []
    print(f"    properties {len(smiles):,}…", flush=True)
    pdf = props_df(smiles)
    drugood_prop_b64s[short] = plot_prop_grid(pdf, f"DrugOOD {short} — Properties", color)
    drugood_ep_stats[short]["n_unique_mols"] = len(pdf)

    # NN-Tanimoto
    fps_nn = [ecfp4(s) for s in np.random.choice(smiles, min(6000,len(smiles)), replace=False)]
    fps_nn = [fp for fp in fps_nn if fp is not None]
    nn_sims = nn_tanimoto_dist(fps_nn, 4000)
    drugood_nn_b64s[short] = plot_nn_tanimoto({short: nn_sims}, f"DrugOOD {short} — NN-Tanimoto")
    drugood_ep_stats[short]["nn_tanimoto_mean"] = float(np.mean(nn_sims)) if nn_sims else 0

    # UMAP colored by label
    umap_smi = smiles[:]
    umap_lbs = df["cls_label"].tolist() if "cls_label" in df.columns else [0]*len(smiles)
    if len(umap_smi) > 50000:
        ix = np.random.choice(len(umap_smi), 50000, replace=False)
        umap_smi = [umap_smi[i] for i in ix]
        umap_lbs = [umap_lbs[i] for i in ix]
    fps_u = [ecfp4(s) for s in umap_smi]
    valid  = [fp is not None for fp in fps_u]
    fps_u  = [fp for fp, v in zip(fps_u, valid) if v]
    ulbs   = [lb for lb, v in zip(umap_lbs, valid) if v]
    drugood_umap_b64s[short] = plot_umap(fps_to_mat(fps_u), ulbs,
                                          f"DrugOOD {short} UMAP (label)", "RdYlGn")
    drugood_all_fps.extend(fps_u[:2500])

    # IID vs OOD property comparison (MW and LogP)
    if "_split" in df.columns and "smiles" in df.columns and len(pdf) > 0:
        groups = {
            sp: df.loc[df["_split"]==sp, "smiles"].dropna().tolist()
            for sp in ["train","iid_test","ood_test"]
        }

        fig_ood, axes_ood = plt.subplots(1, 2, figsize=(10, 4))
        grp_colors = {"train":"#aaaaaa","iid_test":"#4878CF","ood_test":"#e07b54"}
        for prop_idx, prop in enumerate(["mw","logp"]):
            ax = axes_ood[prop_idx]
            for grp, smi_list in groups.items():
                if not smi_list:
                    continue
                sample_smi = np.random.choice(smi_list, min(3000,len(smi_list)), replace=False)
                vals = [p[prop] for s in sample_smi for p in [mol_props(s)] if p is not None]
                if vals:
                    ax.hist(vals, bins=40, alpha=0.5, color=grp_colors[grp],
                            label=f"{grp} (n={len(vals):,})", edgecolor="none", density=True)
            ax.set_title(f"{PROP_LABELS[prop]}", fontsize=9)
            ax.set_ylabel("density"); ax.legend(fontsize=7)
        fig_ood.suptitle(f"DrugOOD {short} — IID vs OOD property shift", fontsize=10, fontweight="bold")
        plt.tight_layout()
        drugood_ood_b64s[short] = fig_to_b64(fig_ood)

print(f"  DrugOOD done. {time.time()-t0:.0f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# 3. curated_mic
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/4] curated_mic", flush=True)

mic_df = pd.read_csv(MIC_CSV)
print(f"  Loaded: {len(mic_df):,} rows, cols: {list(mic_df.columns)}", flush=True)

# Clean is_potent (may have NaN)
if "is_potent" in mic_df.columns:
    mic_df["is_potent"] = mic_df["is_potent"].fillna(0).astype(int)

mic_smiles = mic_df["canonical_smiles"].dropna().tolist() if "canonical_smiles" in mic_df.columns else []
mic_sizes  = mic_df.groupby("assay_chembl_id").size().values if "assay_chembl_id" in mic_df.columns else []
mic_g      = mic_df.groupby("assay_chembl_id") if "assay_chembl_id" in mic_df.columns else None

mic_stats = {
    "n_records": len(mic_df),
    "n_assays": int(mic_df["assay_chembl_id"].nunique()) if "assay_chembl_id" in mic_df.columns else 0,
    "n_compounds": int(mic_df["compound_chembl_id"].nunique()) if "compound_chembl_id" in mic_df.columns else 0,
    "n_documents": int(mic_df["document_chembl_id"].nunique()) if "document_chembl_id" in mic_df.columns else 0,
    "n_targets": int(mic_df["target_chembl_id"].nunique()) if "target_chembl_id" in mic_df.columns else 0,
    "n_organisms": int(mic_df["organism_canonical"].nunique()) if "organism_canonical" in mic_df.columns else 0,
    "pos_rate": float(mic_df["is_potent"].mean()) if "is_potent" in mic_df.columns else 0,
    **{f"size_{k}": v for k,v in summary_stats(mic_sizes).items()},
}

# Size distribution
mic_size_b64 = plot_size_dist(mic_sizes, "curated_mic assay sizes", "#5a9e6f")

# pMIC distribution
if "pmic" in mic_df.columns:
    pmic_vals = mic_df["pmic"].dropna().values
    mic_pmic_b64 = fig_hist(pmic_vals, "curated_mic — pMIC distribution", "pMIC", "#5a9e6f",
                             vlines=[(pmic_vals.mean(),f"μ={pmic_vals.mean():.2f}","--"),
                                     (np.median(pmic_vals),f"med={np.median(pmic_vals):.2f}",":"),
                                     (6.0,"threshold=6","-.")])
else:
    mic_pmic_b64 = None

# Qualification curve
if mic_g is not None and "is_potent" in mic_df.columns:
    mic_pn = [(int(grp["is_potent"].sum()), int((~grp["is_potent"].astype(bool)).sum()))
              for _, grp in mic_g]
    mic_qual_b64 = plot_qual_curve(mic_pn, "curated_mic")
else:
    mic_qual_b64 = None

# Gram +/-
mic_gram_b64 = (fig_bar(mic_df["l2"].value_counts(), "curated_mic — Gram stain (l2)", "#5a9e6f")
                if "l2" in mic_df.columns else None)

# Top organisms
mic_org_b64 = (fig_hbar(mic_df["organism_canonical"].value_counts(), "curated_mic — Top 20 organisms", "#5a9e6f")
               if "organism_canonical" in mic_df.columns else None)

# Top strains
mic_strain_b64 = (fig_hbar(mic_df["strain_canonical"].value_counts(), "curated_mic — Top 20 strains", "#5a9e6f")
                  if "strain_canonical" in mic_df.columns else None)

# Year
if "year" in mic_df.columns:
    yr = mic_df["year"].dropna()
    mic_year_b64 = fig_hist(yr.values, "curated_mic — Year distribution", "Year", "#5a9e6f")
    mic_stats["year_min"] = int(yr.min()); mic_stats["year_max"] = int(yr.max())
else:
    mic_year_b64 = None

# Document distribution (top 20 by record count)
mic_doc_b64 = (fig_hbar(mic_df["document_chembl_id"].value_counts(), "curated_mic — Top 20 documents", "#5a9e6f")
               if "document_chembl_id" in mic_df.columns else None)

# Properties (deduped)
print(f"  Properties {len(mic_smiles):,}…", flush=True)
mic_prop_df  = props_df(mic_smiles)
mic_prop_b64 = plot_prop_grid(mic_prop_df, "curated_mic — Molecular Properties", "#5a9e6f")
mic_stats["n_unique_mols"] = len(mic_prop_df)

# NN-Tanimoto
fps_nn_mic = [ecfp4(s) for s in np.random.choice(mic_smiles, min(6000,len(mic_smiles)), replace=False)]
fps_nn_mic = [fp for fp in fps_nn_mic if fp is not None]
mic_nn_sims = nn_tanimoto_dist(fps_nn_mic, 4000)
mic_nn_b64  = plot_nn_tanimoto({"curated_mic": mic_nn_sims}, "curated_mic — NN-Tanimoto")
mic_stats["nn_tanimoto_mean"] = float(np.mean(mic_nn_sims)) if mic_nn_sims else 0

# UMAP
print("  UMAP curated_mic…", flush=True)
mic_lbs_all = mic_df["is_potent"].tolist() if "is_potent" in mic_df.columns else [0]*len(mic_smiles)
smi_u = mic_smiles[:]; lbs_u = mic_lbs_all[:]
if len(smi_u) > 50000:
    ix = np.random.choice(len(smi_u), 50000, replace=False)
    smi_u = [smi_u[i] for i in ix]; lbs_u = [lbs_u[i] for i in ix]
fps_u  = [ecfp4(s) for s in smi_u]
valid  = [fp is not None for fp in fps_u]
fps_u  = [fp for fp,v in zip(fps_u, valid) if v]
lbs_u  = [lb for lb,v in zip(lbs_u, valid) if v]
mic_umap_b64 = plot_umap(fps_to_mat(fps_u), lbs_u, "curated_mic UMAP (is_potent)", "RdYlGn")

# Activity cliffs
print("  Activity cliffs…", flush=True)
if mic_g is not None:
    mic_cliff_dict = {
        aid: [(row["canonical_smiles"], row["is_potent"])
              for _, row in grp.head(60).iterrows()
              if pd.notna(row.get("canonical_smiles",""))]
        for aid, grp in mic_g
    }
    mic_cliff = activity_cliff_rate(mic_cliff_dict, n_assays=150)
    mic_stats["activity_cliff_rate"] = float(mic_cliff)

print(f"  curated_mic done. {time.time()-t0:.0f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# 4. chembl36_curated — ALL standard_types
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/4] chembl36_curated (all types)", flush=True)

ch_df = pd.read_csv(CHEMBL_CSV)
print(f"  Total rows: {len(ch_df):,}", flush=True)

if "pchembl_value" in ch_df.columns:
    ch_df["is_potent"] = (ch_df["pchembl_value"] >= 6.0).fillna(False).astype(int)

ch_smiles = ch_df["canonical_smiles"].dropna().tolist() if "canonical_smiles" in ch_df.columns else []
ch_sizes  = ch_df.groupby("assay_chembl_id").size().values if "assay_chembl_id" in ch_df.columns else []
ch_g      = ch_df.groupby("assay_chembl_id") if "assay_chembl_id" in ch_df.columns else None

ch_stats = {
    "n_records": len(ch_df),
    "n_assays": int(ch_df["assay_chembl_id"].nunique()) if "assay_chembl_id" in ch_df.columns else 0,
    "n_compounds": int(ch_df["compound_chembl_id"].nunique()) if "compound_chembl_id" in ch_df.columns else 0,
    "n_documents": int(ch_df["activity_doc_id"].nunique()) if "activity_doc_id" in ch_df.columns
                   else (int(ch_df["doc_id"].nunique()) if "doc_id" in ch_df.columns else 0),
    "n_targets": int(ch_df["target_chembl_id"].nunique()) if "target_chembl_id" in ch_df.columns else 0,
    "pos_rate": float(ch_df["is_potent"].mean()) if "is_potent" in ch_df.columns else 0,
    **{f"size_{k}": v for k,v in summary_stats(ch_sizes).items()},
}

# standard_type breakdown
ch_stype_b64 = (fig_hbar(ch_df["standard_type"].value_counts(), "chembl36 — standard_type (top 20)", "#9b5de5")
                if "standard_type" in ch_df.columns else None)

# Assay type (B/F/etc)
ch_atype_b64 = (fig_bar(ch_df["assay_type"].value_counts(), "chembl36 — assay_type (B/F/etc)", "#9b5de5")
                if "assay_type" in ch_df.columns else None)

# Size distribution
ch_size_b64 = plot_size_dist(ch_sizes, "chembl36 assay sizes", "#9b5de5")

# pChEMBL distribution
if "pchembl_value" in ch_df.columns:
    pc_vals = ch_df["pchembl_value"].dropna().values
    ch_pchembl_b64 = fig_hist(pc_vals, "chembl36 — pChEMBL distribution (all types)", "pChEMBL",
                               "#9b5de5", vlines=[(pc_vals.mean(), f"μ={pc_vals.mean():.2f}", "--"),
                                                   (np.median(pc_vals), f"med={np.median(pc_vals):.2f}", ":"),
                                                   (6.0, "threshold=6", "-.")])
    ch_stats["pchembl_mean"] = float(pc_vals.mean()); ch_stats["pchembl_median"] = float(np.median(pc_vals))
else:
    ch_pchembl_b64 = None

# Year distribution
if "year" in ch_df.columns:
    yr_ch = ch_df["year"].dropna()
    ch_year_b64 = fig_hist(yr_ch.values, "chembl36 — Year distribution", "Year", "#9b5de5")
    ch_stats["year_min"] = int(yr_ch.min()); ch_stats["year_max"] = int(yr_ch.max())
else:
    ch_year_b64 = None

# Organism breakdown
org_col = "assay_organism" if "assay_organism" in ch_df.columns else None
if org_col:
    ch_org_b64  = fig_hbar(ch_df[org_col].value_counts(), "chembl36 — Top 20 organisms", "#9b5de5")
    ch_stats["n_organisms"] = int(ch_df[org_col].nunique())
else:
    ch_org_b64 = None

# Qualification curve (all types, pchembl>=6)
if ch_g is not None and "is_potent" in ch_df.columns:
    ch_pn = [(int(grp["is_potent"].sum()), int((~grp["is_potent"].astype(bool)).sum()))
             for _, grp in ch_g]
    ch_qual_b64 = plot_qual_curve(ch_pn, "chembl36 (all types)")
else:
    ch_qual_b64 = None

# Properties (deduped)
print(f"  Properties {len(ch_smiles):,} (deduped)…", flush=True)
ch_prop_df  = props_df(ch_smiles)
ch_prop_b64 = plot_prop_grid(ch_prop_df, "chembl36 — Molecular Properties", "#9b5de5")
ch_stats["n_unique_mols"] = len(ch_prop_df)
print(f"  → {len(ch_prop_df):,} unique valid molecules", flush=True)

# NN-Tanimoto
fps_nn_ch = [ecfp4(s) for s in np.random.choice(ch_smiles, min(6000,len(ch_smiles)), replace=False)]
fps_nn_ch = [fp for fp in fps_nn_ch if fp is not None]
ch_nn_sims = nn_tanimoto_dist(fps_nn_ch, 4000)
ch_nn_b64  = plot_nn_tanimoto({"chembl36": ch_nn_sims}, "chembl36 — NN-Tanimoto")
ch_stats["nn_tanimoto_mean"] = float(np.mean(ch_nn_sims)) if ch_nn_sims else 0

# UMAP
print("  UMAP chembl36…", flush=True)
ch_lbs_all = ch_df["is_potent"].tolist() if "is_potent" in ch_df.columns else [0]*len(ch_smiles)
smi_uc = ch_smiles[:]; lbs_uc = ch_lbs_all[:]
if len(smi_uc) > 50000:
    ix = np.random.choice(len(smi_uc), 50000, replace=False)
    smi_uc = [smi_uc[i] for i in ix]; lbs_uc = [lbs_uc[i] for i in ix]
fps_uc = [ecfp4(s) for s in smi_uc]
valid  = [fp is not None for fp in fps_uc]
fps_uc = [fp for fp,v in zip(fps_uc, valid) if v]
lbs_uc = [lb for lb,v in zip(lbs_uc, valid) if v]
ch_umap_b64 = plot_umap(fps_to_mat(fps_uc), lbs_uc, "chembl36 UMAP (is_potent)", "RdYlGn")

print(f"  chembl36 done. {time.time()-t0:.0f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# 5. CROSS-DATASET
# ════════════════════════════════════════════════════════════════════════════
print("\n[5] Cross-dataset", flush=True)

# Compound overlap (canonical SMILES sets)
print("  Building canonical SMILES sets…", flush=True)
def can_set(smi_list, cap=300000):
    out = set()
    for s in smi_list[:cap]:
        cs = canonical(s)
        if cs:
            out.add(cs)
    return out

do_all_smi = [r for df in drugood.values() for r in df["smiles"].dropna().tolist()] if drugood else []
fs_can  = can_set(all_fsmol_smiles)
do_can  = can_set(do_all_smi)
mic_can = can_set(mic_smiles)
ch_can  = can_set(ch_smiles)

DS = ["FS-Mol","DrugOOD","curated_mic","chembl36"]
ov = np.array([
    [len(fs_can),          len(fs_can&do_can),  len(fs_can&mic_can),  len(fs_can&ch_can)],
    [len(do_can&fs_can),   len(do_can),         len(do_can&mic_can),  len(do_can&ch_can)],
    [len(mic_can&fs_can),  len(mic_can&do_can), len(mic_can),         len(mic_can&ch_can)],
    [len(ch_can&fs_can),   len(ch_can&do_can),  len(ch_can&mic_can),  len(ch_can)],
])
fig_ov, ax_ov = plt.subplots(figsize=(7,5))
im = ax_ov.imshow(ov, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax_ov)
for i in range(4):
    for j in range(4):
        ax_ov.text(j, i, f"{ov[i,j]:,}", ha="center", va="center", fontsize=8,
                   color="white" if ov[i,j] > ov.max()*0.6 else "black")
ax_ov.set_xticks(range(4)); ax_ov.set_xticklabels(DS, fontsize=9)
ax_ov.set_yticks(range(4)); ax_ov.set_yticklabels(DS, fontsize=9)
ax_ov.set_title("Cross-dataset compound overlap (canonical SMILES)", fontsize=10)
plt.tight_layout()
cross_overlap_b64 = fig_to_b64(fig_ov)

# Property comparison
print("  Property comparison…", flush=True)
do_prop_cross = props_df(do_all_smi[:200000]).assign(dataset="DrugOOD") if do_all_smi else pd.DataFrame()
_frames = [
    fsmol_prop_df.assign(dataset="FS-Mol"),
    do_prop_cross,
    mic_prop_df.assign(dataset="curated_mic"),
    ch_prop_df.assign(dataset="chembl36"),
]
combined = pd.concat([f for f in _frames if len(f)], ignore_index=True)

fig_vio, axes_vio = plt.subplots(2, 2, figsize=(12, 8))
axes_vio = axes_vio.flatten()
palette  = {"FS-Mol":"#4878CF","DrugOOD":"#e07b54","curated_mic":"#5a9e6f","chembl36":"#9b5de5"}
for i, col in enumerate(["mw","logp","hac","qed"]):
    ax = axes_vio[i]
    for di, (dname, grp) in enumerate(combined.groupby("dataset", sort=False)):
        vals = grp[col].dropna().values
        if len(vals) < 2:
            continue
        try:
            from scipy.stats import gaussian_kde as _kde
            kde = _kde(vals, bw_method=0.3)
            xs  = np.linspace(np.percentile(vals,1), np.percentile(vals,99), 200)
            ys  = kde(xs); ys = ys / ys.max() * 0.4
            ax.fill_betweenx(xs, di-ys, di+ys, alpha=0.55,
                             color=list(palette.values())[di % 4], label=dname)
        except Exception:
            ax.boxplot(vals, positions=[di], widths=0.5)
    ax.set_xticks(range(len(combined["dataset"].unique())))
    ax.set_xticklabels(sorted(combined["dataset"].unique()), fontsize=8, rotation=20)
    ax.set_title(PROP_LABELS.get(col, col), fontsize=9)
axes_vio[0].legend(fontsize=7)
fig_vio.suptitle("Cross-dataset property comparison", fontsize=12, fontweight="bold")
plt.tight_layout()
cross_prop_b64 = fig_to_b64(fig_vio)

# NN-Tanimoto comparison (all datasets on one plot)
nn_compare_b64 = plot_nn_tanimoto({
    "FS-Mol":      fsmol_nn_sims,
    "DrugOOD":     nn_tanimoto_dist([fp for fp in drugood_all_fps if fp is not None], 4000),
    "curated_mic": mic_nn_sims,
    "chembl36":    ch_nn_sims,
}, "Cross-dataset NN-Tanimoto comparison")

# Combined UMAP (10K per dataset)
print("  Combined UMAP…", flush=True)
combo_fps = []; combo_ds = []
for di, (name, smi_pool) in enumerate([
    ("FS-Mol",      all_fsmol_smiles),
    ("DrugOOD",     do_all_smi),
    ("curated_mic", mic_smiles),
    ("chembl36",    ch_smiles),
]):
    sample = np.random.choice(smi_pool, min(10000, len(smi_pool)), replace=False).tolist() if smi_pool else []
    for s in sample:
        fp = ecfp4(s)
        if fp is not None:
            combo_fps.append(fp); combo_ds.append(di)

combo_umap_b64 = plot_umap(fps_to_mat(combo_fps), combo_ds,
                            "Combined UMAP — 10K per dataset (color=dataset)", "tab10")

print(f"  Cross-dataset done. {time.time()-t0:.0f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n[Rendering HTML]", flush=True)

HTML_HEAD = """<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Dataset EDA</title>
<style>
:root{--bg:#f7f8fa;--fg:#1a1d21;--acc:#4361be;--card:#ffffff;--bdr:#dde1e9;--muted:#666}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171b;--fg:#dde1e9;--card:#1d2127;--bdr:#2d333b;--muted:#999;color-scheme:dark}}
:root[data-theme=dark]{--bg:#14171b;--fg:#dde1e9;--card:#1d2127;--bdr:#2d333b;--muted:#999;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:'Segoe UI',system-ui,sans-serif;
     margin:0;padding:24px 32px;max-width:1100px;margin-inline:auto}
h1{font-size:1.7rem;color:var(--acc);margin-bottom:2px}
h2{font-size:1.1rem;color:var(--acc);margin:0}
h3{font-size:.95rem;color:var(--fg);border-bottom:1px solid var(--bdr);padding-bottom:4px;margin-top:16px}
details{background:var(--card);border:1px solid var(--bdr);border-radius:8px;
        padding:16px;margin-bottom:14px}
summary{cursor:pointer;user-select:none;padding:2px 0}
.sec{margin-top:10px}
table.stat{border-collapse:collapse;font-size:.82rem;margin:8px 0;width:100%}
table.stat td{padding:4px 10px;border:1px solid var(--bdr)}
table.stat tr:nth-child(even) td{background:rgba(128,128,128,.06)}
table.ep{border-collapse:collapse;font-size:.82rem;margin:8px 0;width:100%;overflow-x:auto}
table.ep th,table.ep td{padding:5px 10px;border:1px solid var(--bdr);white-space:nowrap}
table.ep th{background:rgba(128,128,128,.1)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:700px){.grid2{grid-template-columns:1fr}}
.muted{color:var(--muted);font-size:.8rem}
img{display:block}
</style>
</head>
<body>
<h1>Dataset EDA Report</h1>
<p class='muted'>Four datasets: FS-Mol · DrugOOD · curated_mic · chembl36_curated (all standard_types)</p>
"""

parts = [HTML_HEAD]

# ── FS-Mol ──────────────────────────────────────────────────────────────────
c = f"""
{stat_table(fsmol_stats)}
<div class='grid2'>
{h3("Assay size distribution")}{img(fsmol_size_b64)}
{h3("Split distribution")}{img(fsmol_split_b64)}
</div>
{h3("Assay type breakdown")}{img(fsmol_atype_b64)}
{h3("pIC50 / RegressionProperty distribution")}{img(fsmol_reg_b64,"no regression data")}
{h3("Qualification curve (balanced support)")}{img(fsmol_qual_b64)}
{h3("Molecular properties (full unique molecules)")}{img(fsmol_prop_b64)}
{h3("NN-Tanimoto similarity distribution")}{img(fsmol_nn_b64)}
{h3("UMAP (50K, colored by label)")}{img(fsmol_umap_b64,"UMAP unavailable")}
"""
parts.append(section("1. FS-Mol", c))

# ── DrugOOD ─────────────────────────────────────────────────────────────────
ep_table_rows = "".join(
    "<tr><td><b>{}</b></td>".format(ep) +
    "".join("<td>{}</td>".format(v if not isinstance(v,float) else f"{v:.4f}") for v in s.values()) +
    "</tr>"
    for ep, s in drugood_ep_stats.items()
)
ep_keys = list(list(drugood_ep_stats.values())[0].keys()) if drugood_ep_stats else []
ep_table = (
    "<div style='overflow-x:auto'><table class='ep'><tr><th>endpoint</th>"
    + "".join(f"<th>{k}</th>" for k in ep_keys) + "</tr>"
    + ep_table_rows + "</table></div>"
) if drugood_ep_stats else "<p class='muted'>No DrugOOD data</p>"

do_sections = ""
for short in drugood_ep_stats:
    do_sections += f"""
<details>
<summary><b>{short}</b></summary>
<div class='sec'>
{h3("Split counts")}{img(drugood_split_b64s.get(short))}
{h3("pIC50 distribution")}{img(drugood_pic50_b64s.get(short))}
{h3("IID vs OOD property shift (MW, LogP)")}{img(drugood_ood_b64s.get(short),"no OOD comparison")}
{h3("Molecular properties (full unique molecules)")}{img(drugood_prop_b64s.get(short))}
{h3("NN-Tanimoto")}{img(drugood_nn_b64s.get(short))}
{h3("UMAP (50K, label)")}{img(drugood_umap_b64s.get(short),"UMAP unavailable")}
</div>
</details>"""

c = ep_table + do_sections
parts.append(section("2. DrugOOD (4 endpoints — lbap_core scaffold split)", c))

# ── curated_mic ─────────────────────────────────────────────────────────────
c = f"""
{stat_table(mic_stats)}
<div class='grid2'>
{h3("Assay size distribution")}{img(mic_size_b64)}
{h3("Qualification curve")}{img(mic_qual_b64,"no qual data")}
</div>
{h3("pMIC distribution")}{img(mic_pmic_b64,"no pMIC data")}
{h3("Gram stain (l2)")}{img(mic_gram_b64,"no l2 data")}
<div class='grid2'>
{h3("Top 20 organisms")}{img(mic_org_b64,"no organism data")}
{h3("Top 20 strains")}{img(mic_strain_b64,"no strain data")}
</div>
{h3("Year distribution")}{img(mic_year_b64,"no year data")}
{h3("Document distribution (top 20)")}{img(mic_doc_b64,"no document data")}
{h3("Molecular properties (full unique molecules)")}{img(mic_prop_b64)}
{h3("NN-Tanimoto")}{img(mic_nn_b64)}
{h3("UMAP (50K, is_potent)")}{img(mic_umap_b64,"UMAP unavailable")}
"""
parts.append(section("3. curated_mic", c))

# ── chembl36 ────────────────────────────────────────────────────────────────
c = f"""
{stat_table(ch_stats)}
{h3("standard_type breakdown (top 20)")}{img(ch_stype_b64,"no standard_type data")}
{h3("Assay type (B/F/etc)")}{img(ch_atype_b64,"no assay_type data")}
{h3("Assay size distribution")}{img(ch_size_b64)}
{h3("pChEMBL distribution (potent = pChEMBL ≥ 6)")}{img(ch_pchembl_b64,"no pchembl data")}
{h3("Year distribution")}{img(ch_year_b64,"no year data")}
{h3("Organism breakdown (top 20)")}{img(ch_org_b64,"no organism data")}
{h3("Qualification curve (balanced, pChEMBL ≥ 6 = potent)")}{img(ch_qual_b64,"no qual data")}
{h3("Molecular properties (full unique molecules)")}{img(ch_prop_b64)}
{h3("NN-Tanimoto")}{img(ch_nn_b64)}
{h3("UMAP (50K, is_potent)")}{img(ch_umap_b64,"UMAP unavailable")}
"""
parts.append(section("4. chembl36_curated (all standard_types)", c))

# ── Cross-dataset ────────────────────────────────────────────────────────────
c = f"""
{h3("Compound overlap (canonical SMILES exact match)")}{img(cross_overlap_b64)}
{h3("Property comparison (violin, all datasets)")}{img(cross_prop_b64)}
{h3("NN-Tanimoto comparison (structural diversity)")}{img(nn_compare_b64)}
{h3("Combined UMAP (10K per dataset, color=dataset)")}{img(combo_umap_b64,"UMAP unavailable")}
"""
parts.append(section("5. Cross-dataset comparison", c))

parts.append(f"""
<p class='muted' style='text-align:center;margin-top:24px'>
  Generated in {time.time()-t0:.0f}s &bull; {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</p>
</body></html>""")

OUT_HTML.write_text("".join(parts), encoding="utf-8")
print(f"\n=== Done! {OUT_HTML}  ({time.time()-t0:.0f}s) ===", flush=True)
