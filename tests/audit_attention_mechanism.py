"""
Mechanism audit for CrossAttentionProtoNet.
===========================================

Verifies the INFORMATION FLOW and the MATH, not just that the code runs. Everything
here is an invariant the architecture must satisfy by construction; a failure is a bug,
not a tuning issue.

Run before committing GPU hours:
    python tests/audit_attention_mechanism.py

The invariants, and why each matters
------------------------------------
A1  Query molecules never interact.    The design is "support attends to support; query
                                       attends to support". If changing query j moves
                                       query i's prediction, there is query-query
                                       attention and the model is transductive - it
                                       would see the whole query set at once, which
                                       inflates every reported number.
A2  Episodes never interact.           Episode b in a batch must not influence episode
                                       b'. A reshape bug here silently shares
                                       information across tasks.
A3  Query labels never enter.          A label-leak would make the benchmark meaningless.
A4  Support order does not matter.     Attention is permutation-equivariant and the
                                       prototype is a mean, so the episode's prediction
                                       must be invariant to support ordering.
A5  Query order does not matter.       Predictions must permute with the queries.
A6  Support labels DO matter.          Flipping them must change predictions, or the
                                       label channel is dead.
A7  Support molecules DO matter.       Replacing support molecules while keeping labels
                                       must change predictions. If not, the model is
                                       shortcutting through the label embedding and the
                                       "context" is decorative.
A8  Train path == eval path.           forward_batched (training) and
                                       get_enriched_embeddings + PN head (evaluation)
                                       must produce the same numbers, or we train one
                                       model and evaluate a different one.
A9  Eval is deterministic.             Dropout must be off under .eval().
A10 Metric space is shared.            A prototypical head measures distance between
                                       queries and class centroids. Those must live in
                                       ONE space. Separate LayerNorms on the support and
                                       query streams put them in two affine frames.
A11 Support and query are disjoint.    The episode sampler must not leak a support
                                       molecule into the query set.
"""

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from context_model import (CrossAttentionProtoNet, ATTN_ORDERS,       # noqa: E402
                           NORM_POSITIONS, SOFTMAX_MODES)
from model import ECFPEncoder                                          # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}\n        {e}")
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")


def mk(seed=0, **kw):
    torch.manual_seed(seed)
    enc = ECFPEncoder(input_dim=2048, hidden_dim=128, embedding_dim=64)
    kw.setdefault("d_attn", 64)
    kw.setdefault("n_heads", 4)
    kw.setdefault("mlp_f_hidden", 64)
    kw.setdefault("d_label", 8)
    kw.setdefault("dropout", 0.0)
    m = CrossAttentionProtoNet(enc, encoder_output_dim=64, **kw)
    m.eval()
    return m


def episode(B=2, ns=12, nq=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(B, ns, 2048, generator=g) < 0.02).float()
    q = (torch.rand(B, nq, 2048, generator=g) < 0.02).float()
    y = torch.zeros(B, ns)
    y[:, : ns // 2] = 1.0
    return s, y, q


def variants():
    """A representative spread of configurations, not all 2304."""
    out = [("default", {})]
    for npos in NORM_POSITIONS[1:]:
        out.append((f"norm={npos}", {"norm_position": npos}))
    for order in ATTN_ORDERS[1:]:
        out.append((f"order={order}", {"attn_order": order}))
    for sm in SOFTMAX_MODES[1:]:
        out.append((f"softmax={sm}", {"softmax_mode": sm}))
    out.append(("sharedmlp", {"separate_mlp_f": False}))
    out.append(("masklbl", {"query_label_mode": "mask"}))
    out.append(("ffn", {"use_ffn": True}))
    out.append(("2+2 layers", {"n_self_layers": 2, "n_cross_layers": 2}))
    return out


print("=" * 78)
print("ATTENTION MECHANISM AUDIT")
print("=" * 78)

# ---------------------------------------------------------------- A1
print("\n[A1] Query molecules must not interact with each other")


def _a1(kw):
    def run():
        m = mk(**kw)
        s, y, q = episode()
        with torch.no_grad():
            p0, _ = m.forward_batched(s, y, q)
            q2 = q.clone()
            q2[:, -1] = (torch.rand(q.shape[0], 2048) < 0.02).float()   # change LAST query
            p1, _ = m.forward_batched(s, y, q2)
        kept = p0[:, :-1]
        moved = p1[:, :-1]
        d = (kept - moved).abs().max().item()
        assert d < 1e-6, (f"changing one query moved the OTHER queries by {d:.3e} "
                          f"-> query-query information flow exists")
    return run


for name, kw in variants():
    check(f"A1 no query-query flow [{name}]", _a1(kw))

# ---------------------------------------------------------------- A2
print("\n[A2] Episodes in a batch must not interact")


def _a2(kw):
    def run():
        m = mk(**kw)
        s, y, q = episode(B=3)
        with torch.no_grad():
            p_batch, _ = m.forward_batched(s, y, q)
            # same episodes, one at a time
            p_solo = torch.stack([m.forward_batched(s[i:i+1], y[i:i+1], q[i:i+1])[0][0]
                                  for i in range(s.shape[0])])
        d = (p_batch - p_solo).abs().max().item()
        assert d < 1e-5, (f"batched and per-episode predictions differ by {d:.3e} "
                          f"-> episodes leak into each other")
    return run


for name, kw in variants():
    check(f"A2 episode independence [{name}]", _a2(kw))

# ---------------------------------------------------------------- A3
print("\n[A3] Query labels must never reach the model")


def a3_signature():
    import inspect
    for meth in ("forward_batched", "_forward_logits_batched", "get_enriched_embeddings",
                 "predict_from_embeddings", "forward"):
        sig = inspect.signature(getattr(CrossAttentionProtoNet, meth))
        bad = [p for p in sig.parameters
               if "quer" in p and "label" in p]
        assert not bad, f"{meth} takes a query-label argument: {bad}"
    # compute_loss_batched is the only method that may see query labels, and it must
    # not pass them into the forward call — they may only build the BCE target.
    src = inspect.getsource(CrossAttentionProtoNet.compute_loss_batched)
    i = src.index("_forward_logits_batched")
    call = src[i:src.index(")", i) + 1]
    assert "query_label" not in call, f"query labels enter the forward call: {call}"
    for line in src.splitlines():
        t = line.strip()
        if "query_labels" in t and not t.startswith("#"):
            ok = ("binary_query" in t) or t.startswith("query_labels:")
            assert ok, f"query_labels used outside target construction: {t}"


check("A3 no query-label parameter anywhere in the forward path", a3_signature)

# ---------------------------------------------------------------- A4 / A5
print("\n[A4/A5] Permutation behaviour")


def _a4(kw):
    def run():
        m = mk(**kw)
        s, y, q = episode()
        perm = torch.randperm(s.shape[1])
        with torch.no_grad():
            l0, _ = m._forward_logits_batched(s, y, q)
            l1, _ = m._forward_logits_batched(s[:, perm], y[:, perm], q)
        d = (l0 - l1).abs().max().item()
        assert d < 1e-5, f"support order changed predictions by {d:.3e}"
    return run


def _a5(kw):
    def run():
        m = mk(**kw)
        s, y, q = episode()
        perm = torch.randperm(q.shape[1])
        with torch.no_grad():
            l0, _ = m._forward_logits_batched(s, y, q)
            l1, _ = m._forward_logits_batched(s, y, q[:, perm])
        d = (l0[:, perm] - l1).abs().max().item()
        assert d < 1e-5, f"query predictions did not permute with the queries ({d:.3e})"
    return run


for name, kw in variants():
    check(f"A4 support-order invariance [{name}]", _a4(kw))
for name, kw in variants():
    check(f"A5 query-order equivariance [{name}]", _a5(kw))

# ---------------------------------------------------------------- A6 / A7
print("\n[A6/A7] The context must actually be used")


def _a6(kw):
    def run():
        m = mk(**kw)
        s, y, q = episode()
        with torch.no_grad():
            # Compare LOGITS, not probabilities. A saturated sigmoid underflows and
            # makes a live model look dead - this test did exactly that before.
            l0, _ = m._forward_logits_batched(s, y, q)
            l1, _ = m._forward_logits_batched(s, 1 - y, q)
        d = (l0 - l1).abs().max().item()
        assert d > 1e-4, f"flipping support LABELS changed nothing ({d:.3e})"
    return run


def _a7(kw):
    def run():
        m = mk(**kw)
        s, y, q = episode()
        s2 = (torch.rand_like(s) < 0.02).float()      # different molecules, same labels
        with torch.no_grad():
            l0, _ = m._forward_logits_batched(s, y, q)
            l1, _ = m._forward_logits_batched(s2, y, q)
        d = (l0 - l1).abs().max().item()
        assert d > 1e-4, (f"replacing support MOLECULES changed nothing ({d:.3e}) "
                          f"-> the model shortcuts through the label embedding")
    return run


for name, kw in variants():
    check(f"A6 support labels matter [{name}]", _a6(kw))
for name, kw in variants():
    check(f"A7 support molecules matter [{name}]", _a7(kw))

# ---------------------------------------------------------------- A8
print("\n[A8] The training path and the evaluation path must agree")


def _a8(kw):
    def run():
        m = mk(**kw)
        s, y, q = episode(B=1, ns=12, nq=8)
        with torch.no_grad():
            p_train, _ = m.forward_batched(s, y, q)           # training path
            se = m.encoder(s[0])
            qe = m.encoder(q[0])
            c_s, c_q = m.get_enriched_embeddings(se, y[0], qe)   # eval path
            am = y[0] > 0.5
            pa, pi = c_s[am].mean(0), c_s[~am].mean(0)
            d_a = ((c_q - pa) ** 2).sum(-1)
            d_i = ((c_q - pi) ** 2).sum(-1)
            p_eval = torch.sigmoid(m._scale_logit(d_i - d_a))   # [DC-13] temperature
        d = (p_train[0] - p_eval).abs().max().item()
        assert d < 1e-5, (f"training and evaluation paths disagree by {d:.3e} "
                          f"-> we would train one model and evaluate another")
    return run


for name, kw in variants():
    check(f"A8 train path == eval path [{name}]", _a8(kw))


def a8b():
    """predict_from_embeddings(euclidean) must equal the same manual computation."""
    m = mk()
    s, y, q = episode(B=1)
    with torch.no_grad():
        se, qe = m.encoder(s[0]), m.encoder(q[0])
        p1 = m.predict_from_embeddings(se, y[0], qe, distance="euclidean")
        p2, _ = m.forward_batched(s, y, q)
    d = (p1 - p2[0]).abs().max().item()
    assert d < 1e-5, f"predict_from_embeddings != forward_batched ({d:.3e})"


check("A8b predict_from_embeddings == forward_batched (euclidean)", a8b)

# ---------------------------------------------------------------- A9
print("\n[A9] Evaluation must be deterministic")


def a9():
    m = mk(dropout=0.3)
    m.eval()
    s, y, q = episode()
    with torch.no_grad():
        a, _ = m.forward_batched(s, y, q)
        b, _ = m.forward_batched(s, y, q)
    assert torch.equal(a, b), "eval() is not deterministic - dropout still active"
    m.train()
    with torch.no_grad():
        c, _ = m.forward_batched(s, y, q)
        d_, _ = m.forward_batched(s, y, q)
    assert not torch.equal(c, d_), "train() is deterministic - dropout is not applied"


check("A9 dropout off under eval(), on under train()", a9)

# ---------------------------------------------------------------- A10
print("\n[A10] Support and query must share one metric space")


def a10_shared_norm():
    m = mk()
    assert m.share_final_norm and hasattr(m, "final_norm"), (
        "support and query go through SEPARATE LayerNorms with independent learnable "
        "gamma/beta, but the PN head computes distances BETWEEN the two streams. They "
        "must share one norm (share_final_norm=True) or the prototypes and the queries "
        "sit in different affine frames and the 'distance' is not a distance in one "
        "space. The Mahalanobis head is worse: it applies the support covariance to "
        "query vectors that are no longer in the support's frame.")


def a10_same_sphere():
    """A shared LayerNorm puts both streams on the sqrt(d_attn) sphere."""
    m = mk()
    s_, y, q = episode()
    with torch.no_grad():
        se, qe = m.encoder(s_[0]), m.encoder(q[0])
        c_s, c_q = m.get_enriched_embeddings(se, y[0], qe)
    d_attn = c_s.shape[-1]
    for nm, c in (("support", c_s), ("query", c_q)):
        got = c.norm(dim=-1)
        assert torch.allclose(got, torch.full_like(got, d_attn ** 0.5), atol=1e-3),             f"{nm} vectors are not on the sqrt(d_attn) sphere: {got[:3].tolist()}"


check("A10 support and query share the final LayerNorm", a10_shared_norm)
check("A10b both streams land on the same sphere", a10_same_sphere)

# ---------------------------------------------------------------- A12
print("\n[A12] Logits must not be saturated at initialisation")


def _a12(npos, d_attn, enc_dim):
    def run():
        torch.manual_seed(0)
        enc = ECFPEncoder(input_dim=2048, hidden_dim=512, embedding_dim=enc_dim)
        m = CrossAttentionProtoNet(enc, encoder_output_dim=enc_dim, d_attn=d_attn,
                                   n_heads=4, mlp_f_hidden=256, d_label=16,
                                   dropout=0.0, norm_position=npos)
        m.eval()
        g = torch.Generator().manual_seed(0)
        s_ = (torch.rand(1, 64, 2048, generator=g) < 0.02).float()
        q_ = (torch.rand(1, 64, 2048, generator=g) < 0.02).float()
        y_ = torch.zeros(1, 64); y_[:, :32] = 1.0
        with torch.no_grad():
            lg, _ = m._forward_logits_batched(s_, y_, q_)
        mx = lg.abs().max().item()
        assert mx < 15, (
            f"|logit| reaches {mx:.1f} at init (d_attn={d_attn}, enc={enc_dim}, "
            f"{npos}-norm) -> P(active) is saturated before any training, while the "
            f"PTN baseline starts at ~0.004. Comparing the two architectures from "
            f"such different starting points is a confound. See [DC-13] logit_scale.")
    return run


for _d, _e in ((256, 512), (256, 256)):
    for _np in NORM_POSITIONS:
        check(f"A12 no saturation at init [d_attn={_d} enc={_e} {_np}]",
              _a12(_np, _d, _e))

# ---------------------------------------------------------------- A11
print("\n[A11] Episode sampler must keep support and query disjoint")


def a11():
    from data import build_fair_split_indices
    rng = np.random.RandomState(0)
    bad = 0
    for trial in range(200):
        n_total = int(rng.randint(40, 300))
        y = rng.randint(0, 2, n_total)
        groups = {f"g{i}": [j for j in range(n_total) if j % 7 == i] for i in range(7)}
        for split in ("random", "scaffold", "size"):
            sp = build_fair_split_indices(
                n_total, groups, y, 16, split, rng,
                mol_sizes=rng.rand(n_total) if split == "size" else None,
                require_both_classes=True)
            if sp is None:
                continue
            si, qi = sp
            if set(si.tolist()) & set(qi.tolist()):
                bad += 1
            if len(set(si.tolist())) != len(si):
                bad += 1
    assert bad == 0, f"{bad} episodes had support/query overlap or duplicate support"


check("A11 support and query are disjoint, support has no duplicates", a11)

# ---------------------------------------------------------------- A13
print("\n[A13] The support size must be EXACT - skip, never shrink")


def _synthetic_assay(n_mol, frac_pos=0.5):
    """Minimal AssayDataset without the slow SMILES constructor."""
    from data import AssayDataset
    ds = AssayDataset.__new__(AssayDataset)
    ds.assay_id = f"synthetic_{n_mol}"
    ds.fingerprints = [(np.random.rand(2048) < 0.02).astype(np.float32)
                       for _ in range(n_mol)]
    n_pos = max(1, int(n_mol * frac_pos))
    b = np.array([1] * n_pos + [0] * (n_mol - n_pos), dtype=np.int32)
    np.random.shuffle(b)
    ds.binary_labels = b
    ds.labels = b.astype(np.float32)
    ds.scaffolds = [f"C{i}" for i in range(n_mol)]
    ds.scaffold_groups = {f"g{i % 5}": [j for j in range(n_mol) if j % 5 == i]
                          for i in range(5)}
    return ds


def a13_exact_or_none():
    """An assay smaller than n_support + MIN_QUERY must be SKIPPED (None), and one
    that is large enough must yield EXACTLY n_support - never a reduced size."""
    from data import EpisodeSampler, MIN_QUERY
    np.random.seed(0)
    for n_sup in (16, 32, 64):
        samp = EpisodeSampler(n_support=n_sup, n_query=64)
        # too small -> must return None, not a shrunken episode
        for n_mol in (n_sup - 1, n_sup, n_sup + MIN_QUERY - 1):
            ds = _synthetic_assay(max(4, n_mol))
            sp = samp._get_episode_indices_random(ds)
            assert sp is None, (
                f"assay with {n_mol} molecules produced an episode at n_support="
                f"{n_sup}; it must be skipped, not shrunk "
                f"(got support of {len(sp[0])})")
        # large enough -> must be exactly n_sup
        for n_mol in (n_sup + MIN_QUERY, n_sup * 2, n_sup * 4):
            ds = _synthetic_assay(n_mol)
            sp = samp._get_episode_indices_random(ds)
            assert sp is not None, f"{n_mol} molecules should serve n_support={n_sup}"
            assert len(sp[0]) == n_sup, (
                f"support is {len(sp[0])}, expected exactly {n_sup} "
                f"(assay had {n_mol} molecules) - the old min(n, total//2) shrink")
            assert not (set(sp[0].tolist()) & set(sp[1].tolist())), "support/query overlap"


def a13_multi_size_gates_per_draw():
    """With a list-valued n_support the gate must apply to the DRAWN size, not to
    min(n_support) - otherwise a 64-molecule assay is admitted into a 64-support
    episode and then silently shrunk."""
    from data import EpisodeSampler
    np.random.seed(0)
    samp = EpisodeSampler(n_support=[16, 32, 64], n_query=64)
    ds = _synthetic_assay(40)      # can serve 16 and 32, never 64
    sizes = set()
    for _ in range(300):
        sp = samp._get_episode_indices_random(ds)
        if sp is not None:
            sizes.add(len(sp[0]))
    assert sizes, "no episode drawn at all from a 40-molecule assay"
    assert sizes <= {16, 32}, f"drew support sizes {sizes} from a 40-molecule assay"
    assert 64 not in sizes, "a 64-support episode was served by a 40-molecule assay"


def a13_both_classes_in_both_folds():
    from data import EpisodeSampler
    np.random.seed(0)
    samp = EpisodeSampler(n_support=16, n_query=64)
    # heavily skewed assay: only 2 positives, cannot put one in each fold reliably
    ds = _synthetic_assay(200, frac_pos=0.01)
    for _ in range(50):
        sp = samp._get_episode_indices_random(ds)
        if sp is None:
            continue
        y = ds.binary_labels
        assert len(np.unique(y[sp[0]])) >= 2, "support fold is single-class"
        assert len(np.unique(y[sp[1]])) >= 2, "query fold is single-class"


check("A13 support is exact, or the assay is skipped", a13_exact_or_none)
check("A13b multi-size gates on the drawn size", a13_multi_size_gates_per_draw)
check("A13c both folds always carry both classes", a13_both_classes_in_both_folds)


# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
print(f"MECHANISM AUDIT: PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("\nFAILURES:")
    for n, e in FAIL:
        print(f"\n  * {n}\n    {e}")
    sys.exit(1)
print("All mechanism invariants hold.")
