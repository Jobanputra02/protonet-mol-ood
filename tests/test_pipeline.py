"""End-to-end pipeline test on REAL FS-Mol / MIC data.

Each training runs in its own subprocess (one process per configuration, as on the
cluster), then the checkpoint is reloaded and evaluated in this process.

Run from the PTN directory with PYTHONIOENCODING unset, to prove the ASCII-only
logging fix: any non-ASCII print would crash the subprocess.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch

# Run from anywhere: resolve the PTN repo root from this file and work there,
# since the modules under test open data paths relative to the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

SCRATCH = os.path.dirname(os.path.abspath(__file__))
TRAIN_ONE = os.path.join(SCRATCH, "train_one.py")
TMP = tempfile.mkdtemp(prefix="ptn_pipe_")
PASS, FAIL, CKPTS = [], [], {}


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}", flush=True)
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
        tb = traceback.format_exc()
        print("        " + tb.strip().replace("\n", "\n        ")[:1500], flush=True)


def run_train(tag, flags, timeout=1800):
    out = os.path.join(TMP, f"{tag}.pt")
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)          # force the default locale encoding
    r = subprocess.run([sys.executable, TRAIN_ONE, tag, json.dumps(flags), out],
                       capture_output=True, text=True, timeout=timeout,
                       env=env, errors="replace")
    if r.returncode != 0 or "TRAIN_OK" not in r.stdout:
        raise AssertionError(f"train subprocess failed (rc={r.returncode}):\n"
                             f"--- stdout tail ---\n{r.stdout[-1500:]}\n"
                             f"--- stderr tail ---\n{r.stderr[-2500:]}")
    CKPTS[tag] = out
    line = [l for l in r.stdout.splitlines() if l.startswith("TRAIN_OK")][0]
    print(f"        {line}", flush=True)


def main():
    print("=" * 72)
    print("PIPELINE TEST on real data (subprocess per training config)")
    print("=" * 72)

    # ------------------------------------------------------------ A. data
    print("\n[A] Real data loads")

    def t_fsmol():
        from config import FSMOL_TEST
        from data import load_fsmol_split, load_fsmol_assay
        fs = sorted(os.path.join(FSMOL_TEST, f) for f in os.listdir(FSMOL_TEST)
                    if f.endswith(".jsonl.gz"))[:3]
        assert fs, "no FS-Mol test files"
        for f in fs:
            ds, stats = load_fsmol_assay(f)
            assert len(ds) > 0 and ds.binary_labels is not None
            assert len(ds.fingerprints) == len(ds)
        a = load_fsmol_split(FSMOL_TEST, max_assays=3)
        assert len(a) == 3
        print(f"        3 test assays, sizes {[len(x) for x in a]}")

    def t_drugood():
        from config import DRUGOOD_DIR
        from data import load_drugood_split
        f = os.path.join(DRUGOOD_DIR, "lbap_core_ic50_scaffold.json")
        if not os.path.exists(f):
            print("        (skipped: file missing)")
            return
        d = load_drugood_split(f, split_type="lbap_core_ic50_scaffold")
        assert len(d.context_fp) > 0 and len(d.ood_test_fp) > 0
        print(f"        context={len(d.context_fp)} ood_test={len(d.ood_test_fp)}")

    def t_mic_store():
        if not os.path.exists("data/custom/curated_mic_assays_balanced.csv"):
            print("        (skipped)")
            return
        from mic_loader import MicAssayStore, load_mic_split
        st = MicAssayStore("data/custom/curated_mic_assays_balanced.csv")
        tr, va, te = load_mic_split("data/custom/mic_splits.json", "A")
        assert len(st) > 0 and tr and va and te
        print(f"        {len(st)} MIC assays; split A = {len(tr)}/{len(va)}/{len(te)}")

    check("FS-Mol assays load with binary labels", t_fsmol)
    check("DrugOOD split loads", t_drugood)
    check("MIC store and splits load", t_mic_store)

    # ------------------------------------------------------------ B. training
    print("\n[B] Episodic training, one subprocess per config")

    configs = [
        ("ptn_ecfp",        {"_plain_ptn": True}),
        ("ctx_default",     {}),
        ("ctx_sharedmlp",   {"separate_mlp_f": False}),
        ("ctx_postnorm",    {"norm_position": "post"}),
        ("ctx_interleaved", {"attn_order": "interleaved"}),
        ("ctx_ssmax",       {"softmax_mode": "ssmax"}),
        ("ctx_qassmax",     {"softmax_mode": "qassmax"}),
        ("ctx_masklbl",     {"query_label_mode": "mask"}),
        ("ctx_nested",      {"use_nested_tensors": True}),
        ("ctx_ffn",         {"use_ffn": True}),
        ("ctx_all",         {"norm_position": "post", "attn_order": "interleaved",
                             "softmax_mode": "qassmax", "separate_mlp_f": False,
                             "use_ffn": True, "query_label_mode": "mask"}),
        ("ctx_28knobs",     {"norm_position": "post",
                             "_train_kw": {"warmup_steps": 4, "grad_clip": 1.0,
                                           "lr_attn": 1e-4}}),
        ("ctx_noclip",      {"_train_kw": {"grad_clip": 0.0}}),
        ("ctx_shiftaware",  {"softmax_mode": "ssmax", "_split": "similarity"}),
        ("ctx_gnn",         {"_encoder": "gnn", "attn_order": "interleaved",
                             "softmax_mode": "ssmax"}),
        ("ctx_mic",         {"_dataset": "mic", "norm_position": "post",
                             "softmax_mode": "qassmax", "attn_order": "interleaved",
                             "_train_kw": {"warmup_steps": 4, "lr_attn": 1e-4}}),
    ]
    for tag, flags in configs:
        check(f"train [{tag}]", (lambda t=tag, f=flags: run_train(t, dict(f))))

    # ------------------------------------------------------------ C. reload + eval
    print("\n[C] Reload checkpoints and evaluate on FS-Mol test")

    def _reload_eval(tag):
        def run():
            from config import FSMOL_TEST
            from data import load_fsmol_split
            from evaluate import _load_model_from_checkpoint, evaluate_fsmol_test_grid
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, cfg = _load_model_from_checkpoint(CKPTS[tag], dev)
            model.eval()
            # every design choice must survive the round trip
            for k in ("norm_position", "separate_mlp_f", "attn_order", "softmax_mode",
                      "query_label_mode"):
                if k in cfg:
                    assert getattr(model, k) == cfg[k], (
                        f"{k}: model {getattr(model, k)!r} != checkpoint {cfg[k]!r}")
            assays = load_fsmol_split(FSMOL_TEST, max_assays=3)
            df = evaluate_fsmol_test_grid(
                model, assays, dev, splits=("random", "scaffold"),
                support_sizes=(16, 32), n_repeats=2,
                head_names=["emb_proto_euclid", "emb_proto_mahalanobis", "emb_logreg"])
            assert len(df) > 0, "no evaluation rows"
            assert df["delta_auprc"].notna().all(), "NaN delta_auprc"
            assert df["auroc"].between(0, 1).all(), "AUROC out of range"
            assert df["delta_auprc"].between(-1, 1).all(), "delta_auprc out of range"
        return run

    for tag in [t for t, _ in configs if t in CKPTS]:
        check(f"reload + eval [{tag}]", _reload_eval(tag))

    def t_attention_used():
        """Guard for TASKS.md 2.7: eval must route through MLP_F + attention and must
        use the support labels."""
        from config import FSMOL_TEST
        from data import load_fsmol_split
        from evaluate import _load_model_from_checkpoint
        model, _ = _load_model_from_checkpoint(CKPTS["ctx_default"], torch.device("cpu"))
        model.eval()
        a = load_fsmol_split(FSMOL_TEST, max_assays=1)[0]
        E = torch.tensor(np.stack(a.fingerprints[:40]).astype(np.float32))
        ys = torch.tensor(a.binary_labels[:20].astype(np.float32))
        with torch.no_grad():
            enc = model.encoder(E)
            c_sup, c_qry = model.get_enriched_embeddings(enc[:20], ys, enc[20:])
            c_sup_flip, _ = model.get_enriched_embeddings(enc[:20], 1 - ys, enc[20:])
        assert c_sup.shape[0] == 20 and c_qry.shape[0] == 20
        assert not torch.allclose(c_sup, enc[:20, :c_sup.shape[1]]), \
            "enriched == raw encoder output: attention was bypassed"
        assert not torch.allclose(c_sup, c_sup_flip), \
            "flipping support labels changed nothing: label context unused"

    def t_drugood_eval():
        from config import DRUGOOD_DIR
        from data import load_drugood_split
        from evaluate import _load_model_from_checkpoint, evaluate_drugood_multiscale
        f = os.path.join(DRUGOOD_DIR, "lbap_core_ic50_scaffold.json")
        if not os.path.exists(f) or "ctx_ssmax" not in CKPTS:
            print("        (skipped)")
            return
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = _load_model_from_checkpoint(CKPTS["ctx_ssmax"], dev)
        model.eval()
        d = load_drugood_split(f, split_type="lbap_core_ic50_scaffold")
        df = evaluate_drugood_multiscale(model, [d], dev, context_sizes=[16, 32],
                                         seeds=[0])
        assert len(df) > 0 and df["auroc"].between(0, 1).all()
        print(f"        {len(df)} DrugOOD rows")

    check("eval path uses attention layers and support labels", t_attention_used)
    check("DrugOOD evaluation runs", t_drugood_eval)

    # ------------------------------------------------------------ D. entry points
    print("\n[D] Entry points and analysis script")

    def t_help():
        # main.py is configured by its CONFIG block, not by flags. It must still
        # *reject* arguments rather than ignore them and launch a training run.
        for sc in ("main_attention.py", "main_ptn.py"):
            r = subprocess.run([sys.executable, sc, "--help"], capture_output=True,
                               text=True, timeout=600, errors="replace")
            assert r.returncode == 0, f"{sc} --help failed:\n{r.stderr[-1500:]}"
        r = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True,
                           text=True, timeout=120, errors="replace")
        assert r.returncode == 0, f"main.py --help failed:\n{r.stderr[-1500:]}"
        assert "main_ptn.py" in r.stdout, \
            "main.py --help should point at the flag-driven entry points"
        r = subprocess.run([sys.executable, "main.py", "--encoder", "gnn"],
                           capture_output=True, text=True, timeout=120,
                           errors="replace")
        assert r.returncode != 0, \
            "main.py accepted an unknown flag - it would silently start training"
        r = subprocess.run([sys.executable, "main_attention.py", "--help"],
                           capture_output=True, text=True, timeout=600,
                           errors="replace")
        for flag in ("--norm_position", "--attn_order", "--softmax_mode",
                     "--separate_mlp_f", "--no-separate_mlp_f",
                     "--use_nested_tensors", "--no-use_nested_tensors",
                     "--use_ffn", "--no-use_ffn", "--query_label_mode",
                     "--warmup_steps", "--grad_clip", "--lr_attn",
                     "--d_attn", "--n_heads", "--n_self", "--n_cross"):
            assert flag in r.stdout, f"{flag} missing from --help"

    def t_bad_cli_rejected():
        r = subprocess.run([sys.executable, "main_attention.py",
                            "--softmax_mode", "nonsense"],
                           capture_output=True, text=True, timeout=600,
                           errors="replace")
        assert r.returncode != 0, "invalid --softmax_mode was accepted"
        assert "invalid choice" in (r.stderr + r.stdout)

    def t_interleaved_mismatch_rejected():
        """n_self != n_cross with interleaved must fail fast, before any training."""
        r = subprocess.run([sys.executable, "main_attention.py",
                            "--attn_order", "interleaved",
                            "--n_self", "1", "--n_cross", "3"],
                           capture_output=True, text=True, timeout=900,
                           errors="replace")
        assert r.returncode != 0, "mismatched layer counts were accepted"
        assert "must equal" in (r.stderr + r.stdout), \
            f"wrong error:\n{(r.stderr + r.stdout)[-1200:]}"

    def t_attn_stats_untrained():
        out = os.path.join(TMP, "attn.csv")
        r = subprocess.run([sys.executable, "Analysis/model/attention_stats.py",
                            "--n_episodes", "2", "--out", out,
                            "--softmax_mode", "ssmax"],
                           capture_output=True, text=True, timeout=1800,
                           errors="replace")
        assert r.returncode == 0, f"failed:\n{r.stderr[-2500:]}"
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) > 0 and df["entropy_norm"].between(0, 1.001).all()
        assert set(df["n_support"]) >= {16, 512}
        print(f"        {len(df)} rows, entropy_norm "
              f"[{df['entropy_norm'].min():.3f}, {df['entropy_norm'].max():.3f}]")

    def t_attn_stats_checkpoint():
        if "ctx_qassmax" not in CKPTS:
            print("        (skipped)")
            return
        out = os.path.join(TMP, "attn_ck.csv")
        r = subprocess.run([sys.executable, "Analysis/model/attention_stats.py",
                            "--checkpoint", CKPTS["ctx_qassmax"],
                            "--n_episodes", "2", "--out", out],
                           capture_output=True, text=True, timeout=1800,
                           errors="replace")
        assert r.returncode == 0, f"failed:\n{r.stderr[-2500:]}"
        assert os.path.exists(out)

    def t_run_tag_unique():
        import importlib.util
        import itertools
        from context_model import (NORM_POSITIONS, ATTN_ORDERS, SOFTMAX_MODES,
                                   QUERY_LABEL_MODES)
        spec = importlib.util.spec_from_file_location("ma", "main_attention.py")
        ma = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ma)
        seen = {}
        for sep, np_, od, sm, ql, ff in itertools.product(
                (True, False), NORM_POSITIONS, ATTN_ORDERS, SOFTMAX_MODES,
                QUERY_LABEL_MODES, (True, False)):
            ma.SEPARATE_MLP_F, ma.NORM_POSITION, ma.ATTN_ORDER = sep, np_, od
            ma.SOFTMAX_MODE, ma.QUERY_LABEL_MODE, ma.USE_FFN = sm, ql, ff
            suf = ma._arch_suffix()
            assert suf not in seen, f"collision {suf}"
            seen[suf] = 1
        print(f"        {len(seen)} unique run-tag suffixes")

    check("main_*.py --help work and expose every flag", t_help)
    check("invalid --softmax_mode is rejected", t_bad_cli_rejected)
    check("interleaved with n_self != n_cross fails fast",
          t_interleaved_mismatch_rejected)
    check("attention_stats.py runs (untrained)", t_attn_stats_untrained)
    check("attention_stats.py runs (checkpoint)", t_attn_stats_checkpoint)
    check("run-tag suffixes are unique", t_run_tag_unique)

    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 72)
    print(f"PIPELINE: PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        for n, e in FAIL:
            print(f"  - {n}: {type(e).__name__}: {e}")
        sys.exit(1)
    print("Full pipeline OK.")


if __name__ == "__main__":
    main()
