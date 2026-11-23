"""Verification suite for the configurable design choices (TASKS.md 2.3-2.8).

Run from the PTN directory:  python <this file>
"""
import itertools
import math
import os
import sys
import tempfile
import traceback

import numpy as np
import torch
import torch.nn as nn

# Run from anywhere: resolve the PTN repo root from this file and work there,
# since the modules under test open data paths relative to the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from context_model import (CrossAttentionProtoNet, _MHA, QUERY_LABEL_MODES,
                           NORM_POSITIONS, ATTN_ORDERS, SOFTMAX_MODES,
                           SSMAX_REF_LEN, _ssmax_raw_init)
from model import ECFPEncoder, PrototypicalNetworkClassification

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def mk(**kw):
    enc = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
    kw.setdefault("d_attn", 32)
    kw.setdefault("n_heads", 4)
    kw.setdefault("mlp_f_hidden", 32)
    kw.setdefault("d_label", 8)
    return CrossAttentionProtoNet(enc, encoder_output_dim=32, **kw)


def ep(B=3, ns=8, nq=6, d=2048):
    s = (torch.rand(B, ns, d) < 0.02).float()
    q = (torch.rand(B, nq, d) < 0.02).float()
    y = torch.zeros(B, ns); y[:, : ns // 2] = 1.0
    yq = (torch.rand(B, nq) < 0.4).float()
    return s, y, q, yq


# ---------------------------------------------------------------- 1. construction
print("\n[1] Every design-choice combination constructs, runs forward, and backprops")

combos = list(itertools.product(QUERY_LABEL_MODES, NORM_POSITIONS, ATTN_ORDERS,
                                SOFTMAX_MODES, (True, False), (True, False)))
print(f"  ({len(combos)} combinations)")


def _one_combo(qlm, npos, order, smode, sep, ffn):
    def run():
        m = mk(query_label_mode=qlm, norm_position=npos, attn_order=order,
               softmax_mode=smode, separate_mlp_f=sep, use_ffn=ffn,
               n_self_layers=2, n_cross_layers=2)
        s, y, q, yq = ep()
        loss, met = m.compute_loss_batched(s, y, q, yq)
        assert torch.isfinite(loss), f"non-finite loss {loss}"
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
        assert grads, "no gradients at all"
        assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
        tot = sum(g.abs().sum().item() for g in grads)
        assert tot > 0, "all gradients zero"
        # every *used* parameter must receive a gradient
        missing = [n for n, p in m.named_parameters()
                   if p.requires_grad and p.grad is None]
        # the unknown_label token is genuinely unused when query_label_mode == "mask"
        allowed = {"unknown_label"} if qlm == "mask" else set()
        unexpected = [n for n in missing if n not in allowed]
        assert not unexpected, f"params without grad: {unexpected}"
    return run


for c in combos:
    check(f"combo qlm={c[0]} norm={c[1]} order={c[2]} softmax={c[3]} sep={c[4]} ffn={c[5]}",
          _one_combo(*c))


# ---------------------------------------------------------------- 2. validation
print("\n[2] Invalid configurations are rejected")


def _raises(fn, msg_part=""):
    def run():
        try:
            fn()
        except ValueError as e:
            assert msg_part in str(e), f"wrong message: {e}"
            return
        raise AssertionError("expected ValueError, none raised")
    return run


check("bad query_label_mode", _raises(lambda: mk(query_label_mode="nope"), "query_label_mode"))
check("bad norm_position", _raises(lambda: mk(norm_position="middle"), "norm_position"))
check("bad attn_order", _raises(lambda: mk(attn_order="spiral"), "attn_order"))
check("bad softmax_mode", _raises(lambda: mk(softmax_mode="softmaxx"), "softmax_mode"))
check("d_attn not divisible by n_heads",
      _raises(lambda: mk(d_attn=30, n_heads=4), "divisible"))
check("interleaved with n_self != n_cross",
      _raises(lambda: mk(attn_order="interleaved", n_self_layers=1, n_cross_layers=2),
              "must equal"))
check("interleaved with n_self == n_cross is allowed",
      lambda: mk(attn_order="interleaved", n_self_layers=2, n_cross_layers=2))


# ---------------------------------------------------------------- 3. DC-4 separate MLP
print("\n[3] DC-4: separate vs shared MLP_F")


def t_sep_params():
    sep = mk(separate_mlp_f=True)
    sh = mk(separate_mlp_f=False)
    sep_names = {n for n, _ in sep.named_parameters()}
    sh_names = {n for n, _ in sh.named_parameters()}
    assert any(n.startswith("mlp_f_sup") for n in sep_names), "no mlp_f_sup"
    assert any(n.startswith("mlp_f_qry") for n in sep_names), "no mlp_f_qry"
    assert not any(n.startswith("mlp_f.") for n in sep_names), "shared mlp_f leaked in"
    assert any(n.startswith("mlp_f.") for n in sh_names), "no shared mlp_f"
    assert not any(n.startswith("mlp_f_sup") for n in sh_names), "separate leaked in"
    n_sep = sum(p.numel() for p in sep.parameters())
    n_sh = sum(p.numel() for p in sh.parameters())
    assert n_sep > n_sh, f"separate ({n_sep}) should have more params than shared ({n_sh})"


def t_sep_distinct():
    """Separate MLPs must actually differ in effect, not just in name."""
    m = mk(separate_mlp_f=True)
    m.eval()
    x = torch.randn(2, 5, 32 + 8)
    with torch.no_grad():
        a, b = m.mlp_f_sup(x), m.mlp_f_qry(x)
    assert not torch.allclose(a, b), "mlp_f_sup and mlp_f_qry produce identical output"


def t_shared_is_shared():
    m = mk(separate_mlp_f=False)
    assert m._project_sup is not None
    x = torch.randn(2, 5, 32 + 8)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m._project_sup(x), m._project_qry(x)), \
            "shared mode should route both roles through one MLP"


check("separate_mlp_f creates mlp_f_sup/mlp_f_qry and more params", t_sep_params)
check("separate MLPs are independently initialised", t_sep_distinct)
check("shared mode routes both roles through one MLP", t_shared_is_shared)


# ---------------------------------------------------------------- 4. DC-3 norm position
print("\n[4] DC-3: norm position changes computation")


def t_norm_differs():
    torch.manual_seed(0); pre = mk(norm_position="pre")
    torch.manual_seed(0); post = mk(norm_position="post")
    # identical weights, different arrangement
    sd_pre, sd_post = pre.state_dict(), post.state_dict()
    # Post-norm has no norm_kv (it would be a dead parameter there), so the sets differ
    # by exactly those tensors and nothing else.
    only_pre = set(sd_pre) - set(sd_post)
    assert only_pre == {"cross_attn_layers.0.norm_kv.weight",
                        "cross_attn_layers.0.norm_kv.bias"}, f"unexpected diff {only_pre}"
    assert not set(sd_post) - set(sd_pre), "post-norm gained unexpected params"
    for k in set(sd_pre) & set(sd_post):
        assert torch.allclose(sd_pre[k], sd_post[k]), f"seeding mismatch at {k}"
    pre.eval(); post.eval()
    s, y, q, _ = ep()
    with torch.no_grad():
        a, _ = pre.forward_batched(s, y, q)
        b, _ = post.forward_batched(s, y, q)
    assert not torch.allclose(a, b), "pre and post norm gave identical output"
    for layer in pre.self_attn_layers:
        assert layer.norm_position == "pre"
    for layer in post.self_attn_layers:
        assert layer.norm_position == "post"


check("pre vs post norm produce different outputs with identical weights", t_norm_differs)


# ---------------------------------------------------------------- 5. DC-6 attn order
print("\n[5] DC-6: sequential vs interleaved")


def t_order_differs():
    torch.manual_seed(0); seq = mk(attn_order="sequential", n_self_layers=2, n_cross_layers=2)
    torch.manual_seed(0); itl = mk(attn_order="interleaved", n_self_layers=2, n_cross_layers=2)
    # Same parameter set => a checkpoint is loadable either way, only the wiring differs.
    assert set(seq.state_dict()) == set(itl.state_dict()), "param sets differ"
    # (both are pre-norm here, so norm_kv exists in both)
    seq.eval(); itl.eval()
    s, y, q, _ = ep()
    with torch.no_grad():
        a, _ = seq.forward_batched(s, y, q)
        b, _ = itl.forward_batched(s, y, q)
    assert not torch.allclose(a, b), "sequential and interleaved gave identical output"


def t_order_single_layer_support_path():
    """With 1 self + 1 cross layer, interleaving must still equal sequential, because
    there is only one possible ordering of a single pair."""
    torch.manual_seed(0); seq = mk(attn_order="sequential", n_self_layers=1, n_cross_layers=1)
    torch.manual_seed(0); itl = mk(attn_order="interleaved", n_self_layers=1, n_cross_layers=1)
    seq.eval(); itl.eval()
    s, y, q, _ = ep()
    with torch.no_grad():
        a, _ = seq.forward_batched(s, y, q)
        b, _ = itl.forward_batched(s, y, q)
    assert torch.allclose(a, b, atol=1e-5), "1+1 layers should be order-invariant"


check("sequential vs interleaved differ at 2+2 layers", t_order_differs)
check("sequential == interleaved at 1+1 layers (sanity)", t_order_single_layer_support_path)


# ---------------------------------------------------------------- 6. DC-10 softmax
print("\n[6] DC-10: scalable softmax")


def t_ssmax_init_matches_standard():
    """Documented claim: ssmax is initialised so it reproduces standard attention at
    K = SSMAX_REF_LEN. Verify on the attention primitive directly, dropout off."""
    d, H, K = 32, 4, int(SSMAX_REF_LEN)
    torch.manual_seed(0); std = _MHA(d, H, dropout=0.0, softmax_mode="standard")
    torch.manual_seed(0); ss = _MHA(d, H, dropout=0.0, softmax_mode="ssmax")
    ss.attn.load_state_dict(std.attn.state_dict())
    std.eval(); ss.eval()
    x = torch.randn(2, K, d)
    with torch.no_grad():
        a = std(x, x, x)
        b = ss(x, x, x)
    assert torch.allclose(a, b, atol=1e-4), \
        f"ssmax at K={K} should match standard; max diff {(a - b).abs().max():.2e}"


def t_ssmax_differs_away_from_ref():
    d, H = 32, 4
    torch.manual_seed(0); std = _MHA(d, H, dropout=0.0, softmax_mode="standard")
    torch.manual_seed(0); ss = _MHA(d, H, dropout=0.0, softmax_mode="ssmax")
    ss.attn.load_state_dict(std.attn.state_dict())
    std.eval(); ss.eval()
    x = torch.randn(2, 512, d)
    with torch.no_grad():
        a, b = std(x, x, x), ss(x, x, x)
    assert not torch.allclose(a, b, atol=1e-4), "ssmax should differ from standard at K=512"


def t_ssmax_sharpens_with_length():
    """The whole point of ssmax: attention must be able to stay peaked as K grows."""
    d, H = 32, 4
    ent = {}
    for mode in ("standard", "ssmax"):
        torch.manual_seed(0)
        a = _MHA(d, H, dropout=0.0, softmax_mode=mode)
        a.eval()
        for K in (64, 512):
            torch.manual_seed(1)
            x = torch.randn(2, K, d)
            with torch.no_grad():
                a(x, x, x, record=True)
            ent[(mode, K)] = a.pop_stats()["entropy_norm"]
    # At K=512 ssmax must be strictly more concentrated than standard.
    assert ent[("ssmax", 512)] < ent[("standard", 512)], (
        f"ssmax not sharper at K=512: ssmax={ent[('ssmax',512)]:.4f} "
        f"standard={ent[('standard',512)]:.4f}")
    print(f"        entropy_norm  standard K=512: {ent[('standard',512)]:.4f}  "
          f"ssmax K=512: {ent[('ssmax',512)]:.4f}")


def t_qassmax_init_equals_ssmax():
    d, H, K = 32, 4, int(SSMAX_REF_LEN)
    torch.manual_seed(0); ss = _MHA(d, H, dropout=0.0, softmax_mode="ssmax")
    torch.manual_seed(0); qa = _MHA(d, H, dropout=0.0, softmax_mode="qassmax")
    qa.attn.load_state_dict(ss.attn.state_dict())
    ss.eval(); qa.eval()
    x = torch.randn(2, K, d)
    with torch.no_grad():
        a, b = ss(x, x, x), qa(x, x, x)
    assert torch.allclose(a, b, atol=1e-5), "qassmax should start identical to ssmax"


def t_qassmax_query_dependent():
    """After perturbing s_proj, different queries must get different sharpness."""
    d, H = 32, 4
    qa = _MHA(d, H, dropout=0.0, softmax_mode="qassmax")
    nn.init.normal_(qa.s_proj.weight, std=0.5)
    qa.eval()
    x = torch.randn(1, 64, d)
    with torch.no_grad():
        qa(x, x, x, record=True)
    st = qa.pop_stats()
    assert st is not None and 0 < st["entropy_norm"] <= 1.0001


def t_ssmax_param_names():
    """State dicts must stay interchangeable across softmax modes."""
    d, H = 32, 4
    std = set(_MHA(d, H, softmax_mode="standard").state_dict())
    ss = set(_MHA(d, H, softmax_mode="ssmax").state_dict())
    qa = set(_MHA(d, H, softmax_mode="qassmax").state_dict())
    assert std < ss, f"ssmax must be a superset of standard; extra={ss - std}"
    assert ss < qa, f"qassmax must be a superset of ssmax; extra={qa - ss}"
    assert ss - std == {"s_raw"}, f"unexpected ssmax extras: {ss - std}"
    assert qa - ss == {"s_proj.weight", "s_proj.bias"}, f"unexpected: {qa - ss}"


def t_ssmax_raw_init_math():
    s = torch.nn.functional.softplus(torch.tensor(_ssmax_raw_init())).item()
    got = s * math.log(SSMAX_REF_LEN)
    assert abs(got - 1.0) < 1e-5, f"s*log(ref) = {got}, expected 1.0"


check("ssmax init reproduces standard attention at K=64", t_ssmax_init_matches_standard)
check("ssmax differs from standard at K=512", t_ssmax_differs_away_from_ref)
check("ssmax concentrates attention more than standard at K=512", t_ssmax_sharpens_with_length)
check("qassmax init equals ssmax", t_qassmax_init_equals_ssmax)
check("qassmax is query-dependent once trained", t_qassmax_query_dependent)
check("softmax modes have nested, interchangeable state dicts", t_ssmax_param_names)
check("s init solves s*log(64)==1", t_ssmax_raw_init_math)


# ---------------------------------------------------------------- 7. DC-11 stats
print("\n[7] DC-11: attention statistics")


def t_stats_off_by_default():
    m = mk()
    s, y, q, _ = ep()
    m.eval()
    with torch.no_grad():
        m.forward_batched(s, y, q)
    assert m.pop_attn_stats() == {}, "stats recorded while collection is off"


def t_stats_on():
    m = mk(n_self_layers=2, n_cross_layers=1, collect_attn_stats=True)
    m.eval()
    s, y, q, _ = ep(B=2, ns=16, nq=8)
    with torch.no_grad():
        m.forward_batched(s, y, q)
    st = m.pop_attn_stats()
    assert set(st) == {"self_0", "self_1", "cross_0"}, f"got layers {set(st)}"
    for k, v in st.items():
        assert 0.0 <= v["entropy_norm"] <= 1.0001, f"{k} entropy_norm out of range: {v}"
        assert 0.0 < v["max_weight"] <= 1.0001, f"{k} max_weight out of range: {v}"
        assert v["uniform_weight"] <= v["max_weight"] + 1e-6, \
            f"{k} max_weight below uniform: {v}"
    assert st["self_0"]["n_keys"] == 16, "self-attention should see n_support keys"
    assert st["cross_0"]["n_keys"] == 16, "cross-attention keys should be the support set"
    # popping twice must clear
    assert m.pop_attn_stats() == {}, "stats not cleared after pop"


def t_stats_uniform_case():
    """Zero logits => exactly uniform attention => entropy_norm == 1."""
    d, H, K = 32, 4, 32
    a = _MHA(d, H, dropout=0.0, softmax_mode="standard")
    with torch.no_grad():
        a.attn.in_proj_weight.zero_()
        a.attn.in_proj_bias.zero_()
    a.eval()
    x = torch.randn(1, K, d)
    with torch.no_grad():
        a(x, x, x, record=True)
    st = a.pop_stats()
    assert abs(st["entropy_norm"] - 1.0) < 1e-4, f"expected uniform, got {st}"
    assert abs(st["max_weight"] - 1.0 / K) < 1e-5, f"expected 1/K, got {st}"


def t_stats_toggle():
    m = mk()
    m.set_collect_attn_stats(True)
    assert m.collect_attn_stats is True
    m.set_collect_attn_stats(False)
    assert m.collect_attn_stats is False


check("stats are not recorded by default", t_stats_off_by_default)
check("stats recorded per layer with sane ranges", t_stats_on)
check("uniform attention measures entropy_norm == 1.0", t_stats_uniform_case)
check("set_collect_attn_stats toggles", t_stats_toggle)


# ---------------------------------------------------------------- 8. DC-2 nested
print("\n[8] DC-2: nested tensors and the gating rules")


def t_nested_gating():
    cases = [
        (dict(use_nested_tensors=False), False, "off"),
        (dict(use_nested_tensors=True), True, "on, all compatible"),
        (dict(use_nested_tensors=True, attn_order="interleaved",
              n_self_layers=1, n_cross_layers=1), False, "interleaved"),
        (dict(use_nested_tensors=True, softmax_mode="ssmax"), False, "ssmax"),
        (dict(use_nested_tensors=True, collect_attn_stats=True), False, "stats on"),
        (dict(use_nested_tensors=True, n_self_layers=0), False, "no self layers"),
    ]
    for kw, expected, why in cases:
        m = mk(**kw)
        got = m._can_use_nested()
        assert got == expected, f"_can_use_nested()={got}, expected {expected} ({why})"


def t_nested_runs_and_matches_dense():
    """The nested path must produce the same numbers as dense for equal-length supports."""
    torch.manual_seed(0); dense = mk(use_nested_tensors=False, dropout=0.0)
    torch.manual_seed(0); nest = mk(use_nested_tensors=True, dropout=0.0)
    nest.load_state_dict(dense.state_dict())
    assert nest._can_use_nested(), "nested path should be active here"
    dense.eval(); nest.eval()
    s, y, q, _ = ep(B=3, ns=8, nq=6)
    with torch.no_grad():
        a, _ = dense.forward_batched(s, y, q)
        b, _ = nest.forward_batched(s, y, q)
    assert torch.allclose(a, b, atol=1e-4), \
        f"nested != dense, max diff {(a - b).abs().max():.2e}"


def t_nested_backprops():
    m = mk(use_nested_tensors=True)
    s, y, q, yq = ep()
    loss, _ = m.compute_loss_batched(s, y, q, yq)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters()), "nested path produced no usable gradients"


check("_can_use_nested gating is correct for all incompatible combinations", t_nested_gating)
check("nested path matches dense numerically", t_nested_runs_and_matches_dense)
check("nested path backpropagates", t_nested_backprops)


# ---------------------------------------------------------------- 9. checkpoints
print("\n[9] Checkpoint round-trip for every design choice")


def _roundtrip(**kw):
    def run():
        from evaluate import _load_model_from_checkpoint
        dev = torch.device("cpu")
        enc = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
        cfg = dict(d_attn=32, n_heads=4, mlp_f_hidden=32, d_label=8,
                   n_self_layers=1, n_cross_layers=1, dropout=0.0)
        cfg.update(kw)
        m = CrossAttentionProtoNet(enc, encoder_output_dim=32, **cfg)
        m.eval()
        ckpt_cfg = {
            "model_type": "context_attention", "encoder_output_dim": 32,
            "encoder_type": "ecfp", "hidden_dim": 64, "embedding_dim": 32,
            # mirror main_attention.py, which writes every design choice
            "separate_mlp_f": m.separate_mlp_f, "norm_position": m.norm_position,
            "attn_order": m.attn_order, "softmax_mode": m.softmax_mode,
            "query_label_mode": m.query_label_mode,
            "share_final_norm": m.share_final_norm, "logit_scale": m.logit_scale,
            **{k: v for k, v in cfg.items() if k != "collect_attn_stats"},
        }
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model.pt")
            torch.save({"epoch": 1, "model_state_dict": m.state_dict(),
                        "val_delta_auprc": 0.1, "config": ckpt_cfg}, p)
            loaded, lcfg = _load_model_from_checkpoint(p, dev)
        loaded.eval()
        for f in ("norm_position", "separate_mlp_f", "attn_order", "softmax_mode",
                  "query_label_mode", "use_nested_tensors"):
            if f in cfg:
                assert getattr(loaded, f) == cfg[f], \
                    f"{f}: loaded {getattr(loaded, f)!r} != saved {cfg[f]!r}"
        s, y, q, _ = ep(B=2, ns=8, nq=5)
        with torch.no_grad():
            a, _ = m.forward_batched(s, y, q)
            b, _ = loaded.forward_batched(s, y, q)
        assert torch.allclose(a, b, atol=1e-5), \
            f"reloaded model diverges, max diff {(a - b).abs().max():.2e}"
    return run


for _kw in [
    dict(),
    dict(separate_mlp_f=False),
    dict(norm_position="post"),
    dict(attn_order="interleaved"),
    dict(softmax_mode="ssmax"),
    dict(softmax_mode="qassmax"),
    dict(query_label_mode="mask"),
    dict(use_ffn=True),
    dict(norm_position="post", attn_order="interleaved",
         softmax_mode="qassmax", separate_mlp_f=False, use_ffn=True),
]:
    check(f"checkpoint round-trip {_kw or '(defaults)'}", _roundtrip(**_kw))


def t_config_without_sep_flag_infers_from_weights():
    """A config missing separate_mlp_f must still load: the loader reads the weights."""
    from evaluate import _load_model_from_checkpoint
    for sep in (True, False):
        enc = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
        m = CrossAttentionProtoNet(enc, encoder_output_dim=32, d_attn=32, n_heads=4,
                                   mlp_f_hidden=32, d_label=8, dropout=0.0,
                                   separate_mlp_f=sep)
        cfg = {"model_type": "context_attention", "encoder_output_dim": 32,
               "encoder_type": "ecfp", "hidden_dim": 64, "embedding_dim": 32,
               "d_attn": 32, "d_label": 8, "n_heads": 4, "mlp_f_hidden": 32,
               "dropout": 0.0, "logit_scale": m.logit_scale}  # no separate_mlp_f
        with tempfile.TemporaryDirectory() as td:
            pth = os.path.join(td, "model.pt")
            torch.save({"epoch": 1, "model_state_dict": m.state_dict(),
                        "val_delta_auprc": 0.1, "config": cfg}, pth)
            loaded, _ = _load_model_from_checkpoint(pth, torch.device("cpu"))
        assert loaded.separate_mlp_f == sep, \
            f"inferred separate_mlp_f={loaded.separate_mlp_f}, expected {sep}"
        m.eval(); loaded.eval()
        s, y, q, _ = ep(B=2, ns=8, nq=5)
        with torch.no_grad():
            a, _ = m.forward_batched(s, y, q)
            b, _ = loaded.forward_batched(s, y, q)
        assert torch.allclose(a, b, atol=1e-6)


check("config without separate_mlp_f infers the MLP mode from the weights",
      t_config_without_sep_flag_infers_from_weights)


def t_legacy_checkpoint():
    """A checkpoint written before these flags existed must still load, and must come
    back as a shared-MLP / pre-norm / sequential / standard-softmax model."""
    from evaluate import _load_model_from_checkpoint
    enc = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
    old = CrossAttentionProtoNet(
        enc, encoder_output_dim=32, d_attn=32, n_heads=4, mlp_f_hidden=32, d_label=8,
        dropout=0.0,
        # the pre-flag behaviour: shared MLP_F, SEPARATE final norms, unscaled logit
        separate_mlp_f=False, share_final_norm=False, logit_scale="none")
    legacy_cfg = {   # exactly the keys main_attention.py used to write
        "model_type": "context_attention", "encoder_output_dim": 32,
        "encoder_type": "ecfp", "hidden_dim": 64, "embedding_dim": 32,
        "d_attn": 32, "d_label": 8, "n_heads": 4,
        "n_self_layers": 1, "n_cross_layers": 1,
        "mlp_f_hidden": 32, "dropout": 0.0, "use_ffn": False,
        "query_label_mode": "unknown_token", "use_nested_tensors": False,
    }
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.pt")
        torch.save({"epoch": 7, "model_state_dict": old.state_dict(),
                    "val_delta_auprc": 0.15, "config": legacy_cfg}, p)
        loaded, _ = _load_model_from_checkpoint(p, torch.device("cpu"))
    assert loaded.separate_mlp_f is False, "legacy checkpoint must load as shared MLP"
    assert loaded.share_final_norm is False, "legacy checkpoint must keep split norms"
    assert loaded.logit_scale == "none", "legacy checkpoint must keep the unscaled logit"
    assert loaded.norm_position == "pre"
    assert loaded.attn_order == "sequential"
    assert loaded.softmax_mode == "standard"
    old.eval(); loaded.eval()
    s, y, q, _ = ep(B=2, ns=8, nq=5)
    with torch.no_grad():
        a, _ = old.forward_batched(s, y, q)
        b, _ = loaded.forward_batched(s, y, q)
    assert torch.allclose(a, b, atol=1e-5), "legacy reload changed the predictions"


def t_mismatched_checkpoint_is_loud():
    """Loading a shared-MLP checkpoint into a separate-MLP model must fail loudly,
    not silently produce a randomly-initialised MLP."""
    enc = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
    shared = CrossAttentionProtoNet(enc, encoder_output_dim=32, d_attn=32, n_heads=4,
                                    mlp_f_hidden=32, d_label=8, separate_mlp_f=False)
    enc2 = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
    sep = CrossAttentionProtoNet(enc2, encoder_output_dim=32, d_attn=32, n_heads=4,
                                 mlp_f_hidden=32, d_label=8, separate_mlp_f=True)
    try:
        sep.load_state_dict(shared.state_dict())
    except RuntimeError:
        return
    raise AssertionError("shared->separate load_state_dict should have raised")


check("legacy (pre-flag) checkpoint still loads with old semantics", t_legacy_checkpoint)
check("shared/separate MLP checkpoint mismatch raises", t_mismatched_checkpoint_is_loud)


# ---------------------------------------------------------------- 10. public API
print("\n[10] Public API surface on every softmax mode")


def _api(mode, order, npos):
    def run():
        m = mk(softmax_mode=mode, attn_order=order, norm_position=npos, dropout=0.0)
        m.eval()
        se = torch.randn(10, 32); qe = torch.randn(7, 32)
        ys = torch.tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
        with torch.no_grad():
            for dist in ("euclidean", "mahalanobis"):
                p = m.predict_from_embeddings(se, ys, qe, distance=dist)
                assert p.shape == (7,), f"{dist}: bad shape {p.shape}"
                assert torch.isfinite(p).all(), f"{dist}: non-finite"
                assert ((p >= 0) & (p <= 1)).all(), f"{dist}: out of [0,1]"
            cs, cq = m.get_enriched_embeddings(se, ys, qe)
            assert cs.shape == (10, 32) and cq.shape == (7, 32)
            assert torch.isfinite(cs).all() and torch.isfinite(cq).all()
            # degenerate single-class support must return 0.5, not NaN
            p = m.predict_from_embeddings(se, torch.ones(10), qe)
            assert torch.allclose(p, torch.full((7,), 0.5)), f"degenerate gave {p[:3]}"
    return run


for mode in SOFTMAX_MODES:
    for order in ATTN_ORDERS:
        for npos in NORM_POSITIONS:
            check(f"API softmax={mode} order={order} norm={npos}", _api(mode, order, npos))


def t_degenerate_batched():
    m = mk()
    s, y, q, yq = ep(B=2, ns=8, nq=5)
    y[:] = 1.0                      # every episode single-class
    p, valid = m.forward_batched(s, y, q)
    assert not valid.any(), "single-class episodes should be invalid"
    assert torch.allclose(p, torch.full_like(p, 0.5)), "invalid episodes should give 0.5"
    loss, met = m.compute_loss_batched(s, y, q, yq)
    assert torch.isfinite(loss) and loss.item() == 0.0
    assert math.isnan(met["delta_auprc"])


check("all-degenerate batch yields zero loss and NaN metric", t_degenerate_batched)


# ---------------------------------------------------------------- 11. GNN encoder
print("\n[11] GNN encoder path")


def t_gnn():
    from featurize import FSMOL_NODE_FEAT_DIM, smiles_to_fsmol_graph
    from model import FSMolGNNEncoder
    from torch_geometric.data import Batch
    smis = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "C1CCCCC1", "CCOCC",
            "c1ccncc1", "CC(C)C", "CCCl", "CCBr", "COC", "CCCC"]
    graphs = [smiles_to_fsmol_graph(s) for s in smis]
    assert all(g is not None for g in graphs)
    deg = torch.tensor([0, 10, 20, 10, 2], dtype=torch.long)
    enc = FSMolGNNEncoder(node_feat_dim=FSMOL_NODE_FEAT_DIM, hidden_channels=16,
                          num_layers=2, embedding_dim=32, deg=deg, dropout=0.0)
    for order, mode in (("sequential", "standard"), ("interleaved", "ssmax")):
        m = CrossAttentionProtoNet(enc, encoder_output_dim=32, d_attn=32, n_heads=4,
                                   mlp_f_hidden=32, d_label=8, dropout=0.0,
                                   attn_order=order, softmax_mode=mode)
        B, ns, nq = 2, 4, 2
        sup = Batch.from_data_list([graphs[i % len(graphs)] for i in range(B * ns)])
        qry = Batch.from_data_list([graphs[(i + 3) % len(graphs)] for i in range(B * nq)])
        y = torch.tensor([[1., 1., 0., 0.], [1., 0., 1., 0.]])
        yq = torch.tensor([[1., 0.], [0., 1.]])
        loss, _ = m.compute_loss_batched(sup, y, qry, yq)
        assert torch.isfinite(loss), f"{order}/{mode}: non-finite GNN loss"
        loss.backward()


check("GNN encoder works with new flags", t_gnn)


# ---------------------------------------------------------------- 12. no regression
print("\n[12] Non-attention PTN model is unaffected")


def t_ptn_regression():
    enc = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
    m = PrototypicalNetworkClassification(enc)
    s, y, q, yq = ep()
    loss, met = m.compute_loss_batched(s, y, q, yq, train_distance="euclidean")
    assert torch.isfinite(loss)
    loss.backward()
    m.eval()
    with torch.no_grad():
        p = m.predict_from_embeddings(torch.randn(10, 32),
                                      torch.tensor([1.] * 5 + [0.] * 5),
                                      torch.randn(4, 32))
    assert p.shape == (4,) and torch.isfinite(p).all()


check("PrototypicalNetworkClassification still trains and predicts", t_ptn_regression)


# ---------------------------------------------------------------- 13. train.py
print("\n[13] train.py optimisation knobs (2.8)")


def t_param_groups_and_warmup():
    import inspect
    from train import pretrain_classification
    sig = inspect.signature(pretrain_classification)
    for p in ("warmup_steps", "grad_clip", "lr_attn"):
        assert p in sig.parameters, f"{p} missing from pretrain_classification"
    assert sig.parameters["warmup_steps"].default == 0
    assert sig.parameters["grad_clip"].default == 10.0
    assert sig.parameters["lr_attn"].default is None


def t_param_group_split_logic():
    """The encoder/head split must put every non-encoder parameter in group 2."""
    import torch.optim as optim
    m = mk()
    enc_ids = {id(p) for p in m.encoder.parameters()}
    enc_params = [p for p in m.parameters() if p.requires_grad and id(p) in enc_ids]
    head_params = [p for p in m.parameters() if p.requires_grad and id(p) not in enc_ids]
    assert enc_params and head_params, "one group came out empty"
    assert len(enc_params) + len(head_params) == len(
        [p for p in m.parameters() if p.requires_grad]), "params lost in the split"
    opt = optim.Adam([{"params": enc_params, "lr": 1e-4},
                      {"params": head_params, "lr": 1e-5}])
    sched = optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    lrs = []
    for _ in range(12):
        opt.step(); sched.step()
        lrs.append([g["lr"] for g in opt.param_groups])
    assert lrs[0][0] < lrs[-1][0], "warmup did not increase the encoder LR"
    assert abs(lrs[-1][0] - 1e-4) < 1e-12, f"encoder LR did not reach base: {lrs[-1][0]}"
    assert abs(lrs[-1][1] - 1e-5) < 1e-12, f"head LR did not reach base: {lrs[-1][1]}"
    # both groups must warm up proportionally
    assert abs(lrs[0][0] / 1e-4 - lrs[0][1] / 1e-5) < 1e-9, "groups warmed unevenly"


def t_grad_clip_disable():
    m = mk()
    s, y, q, yq = ep()
    loss, _ = m.compute_loss_batched(s, y, q, yq)
    loss.backward()
    for gc in (0.0, -1.0, None):
        assert not (gc and gc > 0), f"grad_clip={gc} should be treated as disabled"
    assert (10.0 and 10.0 > 0), "grad_clip=10.0 should be treated as enabled"


check("pretrain_classification exposes warmup/clip/lr_attn with safe defaults",
      t_param_groups_and_warmup)
check("param-group split and warmup schedule behave", t_param_group_split_logic)
check("grad_clip<=0 disables clipping", t_grad_clip_disable)


# ---------------------------------------------------------------- 14. run tags
print("\n[14] Run-tag suffixes keep ablations in separate folders")


def t_run_tags():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ma_mod", "main_attention.py")
    ma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ma)   # module-level only; __main__ block does not run

    seen = {}
    # Every setting that changes the trained weights (or that an ablation measures)
    # must appear in the suffix, or two runs share an outputs/ folder and silently
    # overwrite each other's checkpoint. use_nested_tensors and the three optimisation
    # knobs were originally missing, which would have made the 2.8 confirmation runs
    # clobber the baseline.
    grid = itertools.product((True, False), NORM_POSITIONS, ATTN_ORDERS,
                             SOFTMAX_MODES, QUERY_LABEL_MODES, (True, False),
                             (True, False), (0, 200), (10.0, 1.0, 0.0),
                             (None, 1e-4))
    for sep, npos, order, smode, qlm, ffn, nest, wu, clip, lra in grid:
        ma.SEPARATE_MLP_F, ma.NORM_POSITION = sep, npos
        ma.ATTN_ORDER, ma.SOFTMAX_MODE = order, smode
        ma.QUERY_LABEL_MODE, ma.USE_FFN = qlm, ffn
        ma.USE_NESTED_TENSORS = nest
        ma.WARMUP_STEPS, ma.GRAD_CLIP, ma.LR_ATTN = wu, clip, lra
        suf = ma._arch_suffix()
        key = (sep, npos, order, smode, qlm, ffn, nest, wu, clip, lra)
        assert suf not in seen, f"suffix collision {suf!r}: {key} vs {seen[suf]}"
        seen[suf] = key
        assert suf.startswith("_") and " " not in suf and "/" not in suf, f"bad suffix {suf!r}"
        assert all(c.isalnum() or c in "._-" for c in suf), f"unsafe path chars: {suf!r}"
    # defaults must still be distinguishable from pre-flag runs
    ma.SEPARATE_MLP_F, ma.NORM_POSITION = True, "pre"
    ma.ATTN_ORDER, ma.SOFTMAX_MODE = "sequential", "standard"
    ma.QUERY_LABEL_MODE, ma.USE_FFN = "unknown_token", False
    ma.USE_NESTED_TENSORS, ma.WARMUP_STEPS = True, 0
    ma.GRAD_CLIP, ma.LR_ATTN = 10.0, None
    assert ma._arch_suffix() == "_sepmlp", f"default suffix is {ma._arch_suffix()!r}"
    print(f"        {len(seen)} configurations -> {len(seen)} distinct suffixes")


check("every architecture combination maps to a unique run-tag suffix", t_run_tags)


# ---------------------------------------------------------------- 15. saturation
print("\n[15] Loss stays differentiable when the model is confidently wrong")


def t_no_dead_params():
    """Every parameter of every configuration must be reachable by the gradient."""
    for npos in NORM_POSITIONS:
        for order in ATTN_ORDERS:
            m = mk(norm_position=npos, attn_order=order, use_ffn=True,
                   n_self_layers=2, n_cross_layers=2)
            s, y, q, yq = ep()
            loss, _ = m.compute_loss_batched(s, y, q, yq)
            loss.backward()
            dead = [n for n, pp in m.named_parameters()
                    if pp.requires_grad and pp.grad is None]
            assert not dead, f"norm={npos} order={order}: dead params {dead}"


def t_ptn_confidently_wrong_has_gradient():
    """The regression behind BCE 6.4 / dAUPRC 0.000: a saturated but WRONG prediction
    must still produce a gradient. With the old clamp formulation it produced none."""
    from model import PrototypicalNetworkClassification

    class Spread(nn.Module):
        """Encoder that pushes the two class prototypes `scale` apart."""
        def __init__(self, scale):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))
            self.scale = scale

        def forward(self, x):
            v = (x[:, :1] * 2 - 1) * self.scale * self.w
            return torch.cat([v, torch.zeros(x.shape[0], 3, device=x.device)], dim=1)

    results = []
    for scale in (1.0, 10.0, 50.0):
        m = PrototypicalNetworkClassification(Spread(scale))
        B, ns, nq = 2, 8, 6
        s = torch.zeros(B, ns, 4); s[:, : ns // 2, 0] = 1.0
        y = torch.zeros(B, ns);    y[:, : ns // 2] = 1.0
        q = torch.zeros(B, nq, 4); q[:, : nq // 2, 0] = 1.0
        yq = torch.zeros(B, nq);   yq[:, : nq // 2] = 0.0   # inverted => always wrong
        yq[:, nq // 2:] = 1.0
        loss, _ = m.compute_loss_batched(s, y, q, yq, train_distance="euclidean")
        loss.backward()
        g = sum(pp.grad.abs().sum().item() for pp in m.parameters()
                if pp.grad is not None)
        assert torch.isfinite(loss), f"scale={scale}: loss not finite"
        assert g > 0, (f"scale={scale}: confidently-wrong prediction gave zero gradient "
                       f"(loss={loss.item():.4f}) - the clamp bug is back")
        results.append((scale, loss.item(), g))
    for sc, l, g in results:
        print(f"        scale={sc:5.1f}  loss={l:9.4f}  grad={g:.3e}")


def t_ctx_confidently_wrong_has_gradient():
    m = mk(dropout=0.0)
    B, ns, nq = 2, 8, 6
    s = (torch.rand(B, ns, 2048) < 0.02).float()
    q = (torch.rand(B, nq, 2048) < 0.02).float()
    y = torch.zeros(B, ns); y[:, : ns // 2] = 1.0
    with torch.no_grad():                 # force a huge distance gap
        m.final_norm.weight.mul_(60.0)
    logits, valid = m._forward_logits_batched(s, y, q)
    assert valid.all()
    gap = logits.abs().max().item()
    yq = (logits.detach() < 0).float()    # label = opposite of the prediction
    loss, _ = m.compute_loss_batched(s, y, q, yq)
    loss.backward()
    g = sum(pp.grad.abs().sum().item() for pp in m.parameters() if pp.grad is not None)
    assert g > 0, f"zero gradient at logit gap {gap:.1f}"
    print(f"        logit gap {gap:.1f}, loss {loss.item():.3f}, total grad {g:.3e}")


def t_ctx_probability_unchanged():
    """forward_batched must return exactly sigmoid(logit) = the old softmax value."""
    m = mk(dropout=0.0)
    m.eval()
    s, y, q, _ = ep()
    with torch.no_grad():
        logits, _ = m._forward_logits_batched(s, y, q)
        p_new, _ = m.forward_batched(s, y, q)
    ref = 1.0 / (1.0 + torch.exp(-logits.double()))
    assert torch.allclose(p_new.double(), ref, atol=1e-9), \
        f"probability changed, max diff {(p_new.double() - ref).abs().max():.2e}"
    assert ((p_new >= 0) & (p_new <= 1)).all()


def t_ptn_probability_matches_softmax():
    """PTN head: sigmoid(logit) must equal the old softmax(-dists)[0]."""
    from model import PrototypicalNetworkClassification
    enc = ECFPEncoder(input_dim=2048, hidden_dim=64, embedding_dim=32)
    m = PrototypicalNetworkClassification(enc)
    m.eval()
    s, y, q, _ = ep()
    for dist in ("euclidean", "mahalanobis"):
        with torch.no_grad():
            logits, _ = m._forward_logits_batched(s, y, q, train_distance=dist)
            pr, _ = m.forward_batched(s, y, q, train_distance=dist)
            ref = torch.softmax(
                torch.stack([torch.zeros_like(logits), -logits], -1), -1)[..., 0]
        assert torch.allclose(pr, ref, atol=1e-6), f"{dist}: PTN probability changed"
        assert ((pr >= 0) & (pr <= 1)).all()


def t_degenerate_logit_is_half():
    m = mk()
    s, y, q, _ = ep()
    y[:] = 1.0
    logits, valid = m._forward_logits_batched(s, y, q)
    assert not valid.any()
    assert torch.allclose(logits, torch.zeros_like(logits)), "degenerate logit != 0"
    pr, _ = m.forward_batched(s, y, q)
    assert torch.allclose(pr, torch.full_like(pr, 0.5)), "degenerate probability != 0.5"


def t_amp_fp16_gradient_survives():
    """Under fp16 autocast the old code lost gradients at a logit gap of only ~20."""
    if not torch.cuda.is_available():
        print("        (skipped: no CUDA)")
        return
    dev = torch.device("cuda")
    m = mk(dropout=0.0).to(dev)
    B, ns, nq = 2, 8, 6
    s = (torch.rand(B, ns, 2048, device=dev) < 0.02).float()
    q = (torch.rand(B, nq, 2048, device=dev) < 0.02).float()
    y = torch.zeros(B, ns, device=dev); y[:, : ns // 2] = 1.0
    with torch.no_grad():
        m.final_norm.weight.mul_(60.0)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = m._forward_logits_batched(s, y, q)
        yq = (logits.detach() < 0).float()
        loss, _ = m.compute_loss_batched(s, y, q, yq)
    loss.backward()
    g = sum(pp.grad.abs().sum().item() for pp in m.parameters() if pp.grad is not None)
    assert g > 0, "fp16 autocast lost the gradient"
    print(f"        fp16 loss={loss.item():.3f} total grad={g:.3e}")


check("no configuration has dead (gradient-less) parameters", t_no_dead_params)
check("PTN: confidently-wrong prediction still has gradient",
      t_ptn_confidently_wrong_has_gradient)
check("context model: confidently-wrong prediction still has gradient",
      t_ctx_confidently_wrong_has_gradient)
check("context model probabilities unchanged by the logits refactor",
      t_ctx_probability_unchanged)
check("PTN probabilities unchanged by the logits refactor",
      t_ptn_probability_matches_softmax)
check("degenerate episode gives logit 0 / probability 0.5", t_degenerate_logit_is_half)
check("fp16 autocast keeps the gradient", t_amp_fp16_gradient_survives)


# ---------------------------------------------------------------- 16. interface parity
print("\n[16] CrossAttentionProtoNet satisfies the same interface as the PTN head")


def t_forward_exists():
    """Regression guard: evaluate.py calls model.forward(sup, lbl, qry) directly on the
    non-GNN DrugOOD path. The attention model had no forward(), so DrugOOD evaluation
    of it raised NotImplementedError."""
    from model import PrototypicalNetworkClassification
    ptn_api = {n for n in dir(PrototypicalNetworkClassification)
               if not n.startswith("__")}
    ctx_api = {n for n in dir(CrossAttentionProtoNet) if not n.startswith("__")}
    missing = {"forward", "forward_batched", "predict_from_embeddings",
               "compute_loss_batched"} - ctx_api
    assert not missing, f"CrossAttentionProtoNet is missing {missing}"
    # anything public the PTN head offers and this one does not is a parity gap
    gap = {n for n in ptn_api - ctx_api
           if not n.startswith("_") and callable(getattr(
               PrototypicalNetworkClassification, n, None))}
    assert not gap, f"interface parity gap vs the PTN head: {gap}"


def t_forward_matches_predict_from_embeddings():
    """forward() must be exactly encoder + predict_from_embeddings, for both distances
    and every softmax mode."""
    for mode in SOFTMAX_MODES:
        m = mk(softmax_mode=mode, dropout=0.0)
        m.eval()
        sup = (torch.rand(10, 2048) < 0.02).float()
        qry = (torch.rand(7, 2048) < 0.02).float()
        ys = torch.tensor([1.] * 5 + [0.] * 5)
        for dist in ("mahalanobis", "euclidean"):
            with torch.no_grad():
                a = m.forward(sup, ys, qry, distance=dist)
                b = m.predict_from_embeddings(m.encoder(sup), ys, m.encoder(qry),
                                              distance=dist)
            assert a.shape == (7,), f"{mode}/{dist}: shape {a.shape}"
            assert torch.allclose(a, b, atol=1e-6), f"{mode}/{dist}: forward != predict"
            assert torch.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()


def t_forward_default_is_mahalanobis():
    """evaluate.py calls forward() with no distance argument and expects the FS-Mol
    protocol's Mahalanobis, matching the PTN head's default."""
    m = mk(dropout=0.0)
    m.eval()
    sup = (torch.rand(12, 2048) < 0.02).float()
    qry = (torch.rand(6, 2048) < 0.02).float()
    ys = torch.tensor([1.] * 6 + [0.] * 6)
    with torch.no_grad():
        default = m.forward(sup, ys, qry)
        maha = m.forward(sup, ys, qry, distance="mahalanobis")
    assert torch.allclose(default, maha), "forward() default is not mahalanobis"


def t_forward_degenerate():
    m = mk(dropout=0.0)
    m.eval()
    sup = (torch.rand(8, 2048) < 0.02).float()
    qry = (torch.rand(5, 2048) < 0.02).float()
    with torch.no_grad():
        p = m.forward(sup, torch.ones(8), qry)
    assert torch.allclose(p, torch.full((5,), 0.5)), f"degenerate gave {p}"


def t_forward_gnn():
    from featurize import FSMOL_NODE_FEAT_DIM, smiles_to_fsmol_graph
    from model import FSMolGNNEncoder
    from torch_geometric.data import Batch
    smis = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "C1CCCCC1", "CCOCC",
            "c1ccncc1", "CC(C)C", "CCCl", "CCBr"]
    graphs = [smiles_to_fsmol_graph(x) for x in smis]
    deg = torch.tensor([0, 10, 20, 10, 2], dtype=torch.long)
    enc = FSMolGNNEncoder(node_feat_dim=FSMOL_NODE_FEAT_DIM, hidden_channels=16,
                          num_layers=2, embedding_dim=32, deg=deg, dropout=0.0)
    m = CrossAttentionProtoNet(enc, encoder_output_dim=32, d_attn=32, n_heads=4,
                               mlp_f_hidden=32, d_label=8, dropout=0.0)
    m.eval()
    sup = Batch.from_data_list(graphs[:8])
    qry = Batch.from_data_list(graphs[8:])
    ys = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
    with torch.no_grad():
        p = m.forward(sup, ys, qry)
    assert p.shape == (2,) and torch.isfinite(p).all()


check("CrossAttentionProtoNet has forward() and full PTN interface parity",
      t_forward_exists)
check("forward() == encoder + predict_from_embeddings, all modes and distances",
      t_forward_matches_predict_from_embeddings)
check("forward() defaults to mahalanobis like the PTN head",
      t_forward_default_is_mahalanobis)
check("forward() returns 0.5 for a single-class support set", t_forward_degenerate)
check("forward() works with the GNN encoder", t_forward_gnn)


# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("\nFailures:")
    for n, e in FAIL:
        print(f"  - {n}: {type(e).__name__}: {e}")
    sys.exit(1)
print("All design-choice checks passed.")
