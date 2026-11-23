"""Static (AST + text) audit of the 2.3-2.8 changes. Needs no torch.

Checks the structural invariants that the runtime suite also covers, so that there is
some verification available even when the ML stack cannot load.
"""
import ast
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

OK, BAD = [], []


def check(name, cond, detail=""):
    if cond:
        OK.append(name)
        print(f"  PASS  {name}")
    else:
        BAD.append(name)
        print(f"  FAIL  {name}  {detail}")


def src(f):
    return open(f, encoding='utf-8-sig').read()


def tree(f):
    return ast.parse(src(f))


def find_class(t, name):
    for n in ast.walk(t):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def find_func(node, name):
    for n in node.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


print("=" * 70)
print("STATIC AUDIT of context_model.py / model.py / train.py / main_attention.py")
print("=" * 70)

cm_src, cm = src('context_model.py'), tree('context_model.py')
m_src, m = src('model.py'), tree('model.py')
t_src, t = src('train.py'), tree('train.py')
ma_src = src('main_attention.py')
ev_src = src('evaluate.py')

# ---------------------------------------------------------------- 1. constructor args
print("\n[1] CrossAttentionProtoNet exposes every design choice")
cls = find_class(cm, 'CrossAttentionProtoNet')
check("class CrossAttentionProtoNet exists", cls is not None)
init = find_func(cls, '__init__')
args = [a.arg for a in init.args.args] + [a.arg for a in init.args.kwonlyargs]
for a in ("query_label_mode", "use_nested_tensors", "norm_position",
          "separate_mlp_f", "attn_order", "softmax_mode", "collect_attn_stats",
          "use_ffn", "d_attn", "d_label", "n_heads", "n_self_layers",
          "n_cross_layers", "mlp_f_hidden", "dropout"):
    check(f"__init__ takes {a}", a in args, f"args={args}")

# defaults
defaults = {}
pos = init.args.args[-len(init.args.defaults):] if init.args.defaults else []
for a, d in zip(pos, init.args.defaults):
    try:
        defaults[a.arg] = ast.literal_eval(d)
    except Exception:
        defaults[a.arg] = '<expr>'
check("separate_mlp_f defaults to True (matches DC-6 separate operators)",
      defaults.get("separate_mlp_f") is True, defaults.get("separate_mlp_f"))
check("norm_position defaults to 'pre'", defaults.get("norm_position") == "pre")
check("attn_order defaults to 'sequential'", defaults.get("attn_order") == "sequential")
check("softmax_mode defaults to 'standard'", defaults.get("softmax_mode") == "standard")
check("collect_attn_stats defaults to False (never on during training)",
      defaults.get("collect_attn_stats") is False)
check("use_nested_tensors defaults to False", defaults.get("use_nested_tensors") is False)

# ---------------------------------------------------------------- 2. validation
print("\n[2] Invalid values raise before any compute")
init_src = ast.get_source_segment(cm_src, init)
for tok, msg in [("QUERY_LABEL_MODES", "query_label_mode"),
                 ("NORM_POSITIONS", "norm_position"),
                 ("ATTN_ORDERS", "attn_order"),
                 ("SOFTMAX_MODES", "softmax_mode")]:
    check(f"__init__ validates {msg}", tok in init_src)
check("__init__ validates d_attn % n_heads", "d_attn % n_heads" in init_src)
check("__init__ rejects interleaved with n_self != n_cross",
      "interleaved" in init_src and "must equal" in init_src)
nraise = sum(1 for n in ast.walk(init) if isinstance(n, ast.Raise))
check("validation raises (>=6 raise statements)", nraise >= 6, f"found {nraise}")

# ---------------------------------------------------------------- 3. logits refactor
print("\n[3] Loss computed from logits in BOTH heads (the 2.8 fix)")
for f, s, label in (("context_model.py", cm_src, "context"), ("model.py", m_src, "PTN")):
    check(f"{label}: has _forward_logits_batched", "_forward_logits_batched" in s)
    check(f"{label}: uses binary_cross_entropy_with_logits",
          "binary_cross_entropy_with_logits" in s)
    check(f"{label}: no p.clamp(1e-7, 1 - 1e-7) left",
          "clamp(1e-7, 1 - 1e-7)" not in s.replace("# ", "ZZ"),
          "the saturating clamp is still present")
    check(f"{label}: no softmax(-dists) left",
          "F.softmax(-dists" not in s and "softmax(-dists_b" not in s)
    check(f"{label}: forward_batched delegates to the logits variant",
          re.search(r"def forward_batched\([^)]*\)[^:]*:(?:.|\n){0,900}?"
                    r"_forward_logits_batched", s) is not None)
    check(f"{label}: compute_loss_batched calls the logits variant",
          re.search(r"def compute_loss_batched\((?:.|\n){0,2500}?"
                    r"_forward_logits_batched", s) is not None)

# Degenerate episodes must map to logit 0 (= probability 0.5), not to 0.5 directly.
# In context_model.py the PN head owns that zeroing; in model.py it is inline.
def _code_only(text):
    """Strip docstrings and comments so a match means real code, not prose."""
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return "\n".join(l.split("#")[0] for l in text.split("\n"))


def _func(text, name):
    return _code_only(re.search(
        r"\n    def " + name + r"\((?:.|\n)*?(?=\n    def )", text).group(0))


for _f, _s, _label, _owner in (("context_model.py", cm_src, "context", "_pn_head_batched"),
                               ("model.py", m_src, "PTN", "_forward_logits_batched")):
    _fn = _func(_s, _owner)
    check(f"{_label}: degenerate episodes get logit 0 (in {_owner})",
          "zeros_like(logits)" in _fn,
          "degenerate episodes may still be hardcoded to 0.5")
    _lg = _func(_s, "_forward_logits_batched")
    check(f"{_label}: logits variant returns an un-squashed logit",
          "sigmoid" not in _lg and "softmax" not in _lg,
          "the logits path squashes, defeating the fix")
    # Scope by class: model.py also has an unrelated forward_batched on the
    # regression head, which must stay untouched.
    _cls_name = ("CrossAttentionProtoNet" if _label == "context"
                 else "PrototypicalNetworkClassification")
    _t = cm if _label == "context" else m
    _wrap = _code_only(ast.get_source_segment(
        _s, find_func(find_class(_t, _cls_name), "forward_batched")))
    check(f"{_label}: forward_batched is the single place that squashes",
          "torch.sigmoid(logits)" in _wrap)

# ---------------------------------------------------------------- 4. norm_kv
print("\n[4] DC-3: no dead parameter under post-norm")
ca = find_class(cm, '_CrossAttnLayer')
ca_src = ast.get_source_segment(cm_src, ca)
check("norm_kv is conditional on pre-norm",
      'nn.LayerNorm(d) if norm_position == "pre" else None' in ca_src)
fwd = ast.get_source_segment(cm_src, find_func(ca, 'forward'))
pre_part, _, post_part = fwd.partition("else:  # post")
check("norm_kv used only in the pre-norm branch",
      "norm_kv" in pre_part and "norm_kv" not in post_part)
check("post-norm branch normalises after the residual",
      "self.norm_q(q + self.drop" in post_part)

sa = find_class(cm, '_SelfAttnLayer')
sa_fwd = ast.get_source_segment(cm_src, find_func(sa, 'forward'))
check("self-attn pre-norm: residual added after attention",
      "x = x + self.drop(self.attn(normed" in sa_fwd)
check("self-attn post-norm: norm applied to the residual sum",
      "self.norm(x + self.drop(self.attn(x, x, x" in sa_fwd)

# ---------------------------------------------------------------- 5. DC-4 wiring
print("\n[5] DC-4: separate vs shared MLP_F wiring")
check("builds mlp_f_sup / mlp_f_qry when separate", "self.mlp_f_sup" in cm_src
      and "self.mlp_f_qry" in cm_src)
check("builds a single mlp_f when shared", "self.mlp_f = _make_mlp_f()" in cm_src)
check("_project_sup routes by the flag",
      "self.mlp_f_sup(x) if self.separate_mlp_f else self.mlp_f(x)" in cm_src)
check("_project_qry routes by the flag",
      "self.mlp_f_qry(x) if self.separate_mlp_f else self.mlp_f(x)" in cm_src)
check("_augment_and_project uses the role-specific projections",
      "_project_sup(torch.cat" in cm_src and "_project_qry(torch.cat" in cm_src)

# ---------------------------------------------------------------- 6. DC-6 wiring
print("\n[6] DC-6: attention order")
at = ast.get_source_segment(cm_src, find_func(cls, '_attend'))
check("interleaved zips self and cross layers",
      "zip(self.self_attn_layers" in at and "self.cross_attn_layers)" in at)
check("interleaved feeds the evolving support into cross-attention",
      re.search(r"c_sup = self_layer\(c_sup.*?\)\s*\n\s*c_qry = cross_layer\(c_qry, c_sup",
                at, re.S) is not None)
check("sequential runs all self layers then all cross layers",
      "for layer in self.self_attn_layers" in at and
      "for layer in self.cross_attn_layers" in at)

# ---------------------------------------------------------------- 7. DC-2 gating
print("\n[7] DC-2: nested-tensor gating")
cu = ast.get_source_segment(cm_src, find_func(cls, '_can_use_nested'))
for cond, why in [("self.use_nested_tensors", "flag"),
                  ('self.attn_order == "sequential"', "interleaving needs dense c_sup"),
                  ('self.softmax_mode == "standard"', "ssmax needs an explicit logit matrix"),
                  ("not self.collect_attn_stats", "stats need attention weights"),
                  ("len(self.self_attn_layers) > 0", "nothing to do with 0 layers")]:
    check(f"nested gated on {cond} ({why})", cond in cu)

# ---------------------------------------------------------------- 8. DC-10
print("\n[8] DC-10: scalable softmax")
mha = find_class(cm, '_MHA')
check("_MHA exists", mha is not None)
mha_src = ast.get_source_segment(cm_src, mha)
check("parameters live in a real nn.MultiheadAttention (state-dict compatibility)",
      "self.attn = nn.MultiheadAttention" in mha_src)
check("standard mode delegates to nn.MultiheadAttention (keeps fused kernels + nested)",
      'if self.softmax_mode == "standard" and not record' in mha_src)
check("ssmax/qassmax add a learnable per-head s_raw",
      "self.s_raw = nn.Parameter(torch.full((n_heads,)" in mha_src)
check("qassmax adds a query-dependent projection",
      "self.s_proj = nn.Linear(d, n_heads)" in mha_src)
check("qassmax s_proj is zero-initialised (starts equal to ssmax)",
      "nn.init.zeros_(self.s_proj.weight)" in mha_src
      and "nn.init.zeros_(self.s_proj.bias)" in mha_src)
check("logits scaled by s * log(n_keys)", "logits * (s * log_n)" in mha_src)
check("log_n taken from the number of KEYS, not queries",
      "log_n = math.log(max(float(Lk), 2.0))" in mha_src)
check("s kept positive via softplus", "F.softplus(self.s_raw" in mha_src)
check("attention dropout applied in the manual path",
      "F.dropout(attn_w" in mha_src and "training=self.training" in mha_src)
check("1/sqrt(head_dim) scaling retained", "math.sqrt(hd)" in mha_src)
check("init solves s*log(SSMAX_REF_LEN)==1",
      "def _ssmax_raw_init" in cm_src and "math.log(math.expm1(s_target))" in cm_src)
check("SSMAX_REF_LEN is 64 (the training support size)", "SSMAX_REF_LEN = 64.0" in cm_src)

# ---------------------------------------------------------------- 9. DC-11
print("\n[9] DC-11: attention diagnostics")
check("_record computes normalised entropy", "entropy_norm" in cm_src
      and "ent.mean() / norm" in cm_src)
check("entropy normalised by log(n_keys)", "math.log(n_keys)" in cm_src)
check("stats recorded under no_grad", re.search(
    r"def _record(?:.|\n){0,200}?with torch\.no_grad\(\)", cm_src) is not None)
check("pop_attn_stats clears after reading", "s, self._stats = self._stats, None" in cm_src)
check("set_collect_attn_stats exists", "def set_collect_attn_stats" in cm_src)
check("standard mode requests unaveraged per-head weights when recording",
      "average_attn_weights=False" in cm_src)

# ---------------------------------------------------------------- 10. train.py (2.8)
print("\n[10] 2.8 optimisation knobs in train.py")
pc = find_func(t, 'pretrain_classification')
targs = [a.arg for a in pc.args.args]
tdef = {}
if pc.args.defaults:
    for a, d in zip(pc.args.args[-len(pc.args.defaults):], pc.args.defaults):
        try:
            tdef[a.arg] = ast.literal_eval(d)
        except Exception:
            tdef[a.arg] = '<expr>'
for a in ("warmup_steps", "grad_clip", "lr_attn"):
    check(f"pretrain_classification takes {a}", a in targs)
check("warmup_steps defaults to 0 (previous behaviour)", tdef.get("warmup_steps") == 0)
check("grad_clip defaults to 10.0 (previous behaviour)", tdef.get("grad_clip") == 10.0)
check("lr_attn defaults to None (previous behaviour)", tdef.get("lr_attn") is None)
pc_src = ast.get_source_segment(t_src, pc)
check("encoder/head param groups built when lr_attn is set",
      'if lr_attn is not None and hasattr(model, "encoder")' in pc_src)
check("param split uses id() identity, not name matching",
      "enc_ids     = {id(p) for p in model.encoder.parameters()}" in pc_src)
check("warmup uses LambdaLR with a linear ramp",
      "LambdaLR" in pc_src and "min(1.0, float(s + 1) / float(warmup_steps))" in pc_src)
check("scheduler stepped after optimizer.step()",
      re.search(r"optimizer\.step\(\)\s*\n\s*if scheduler is not None:\s*\n\s*scheduler\.step\(\)",
                pc_src) is not None)
check("grad_clip <= 0 disables clipping",
      "if grad_clip and grad_clip > 0:" in pc_src)
check("clip uses the parameter, not a hardcoded 10.0",
      "clip_grad_norm_(model.parameters(), grad_clip)" in pc_src)
check("regression trainer left untouched (still hardcodes 10.0)",
      t_src.count("clip_grad_norm_(model.parameters(), 10.0)") == 1)
check("PTN_NUM_WORKERS override exists", "def _num_workers" in t_src)
check("prebuilt datasets pinned to 0 workers",
      re.search(r"def _num_workers(?:.|\n){0,900}?if default == 0:\s*\n\s*return 0", t_src)
      is not None)

# ---------------------------------------------------------------- 11. checkpoints
print("\n[11] Checkpoint round-trip plumbing")
for k in ("norm_position", "separate_mlp_f", "attn_order", "softmax_mode"):
    check(f"evaluate.py rebuilds {k}", f'{k}=' in ev_src)
    check(f"main_attention.py saves {k} in the checkpoint config",
          f'"{k}": ' in ma_src)
check("separate_mlp_f inferred from the weights, not trusted from the config",
      'inferred_sep = any(k.startswith("mlp_f_sup.") for k in sd)' in ev_src)
check("config/weights disagreement warns and follows the weights",
      "trusting the weights" in ev_src)
check("evaluate.py passes the inferred value", "separate_mlp_f=inferred_sep" in ev_src)

# ---------------------------------------------------------------- 12. CLI
print("\n[12] CLI exposes every choice, and booleans can be turned off")
for flag in ("--norm_position", "--attn_order", "--softmax_mode", "--query_label_mode",
             "--d_attn", "--n_heads", "--n_self", "--n_cross",
             "--warmup_steps", "--grad_clip", "--lr_attn"):
    check(f"{flag} is a CLI flag", f'"{flag}"' in ma_src)
for flag in ("separate_mlp_f", "use_nested_tensors", "use_ffn"):
    check(f"--{flag} uses BooleanOptionalAction (so --no-{flag} works)",
          re.search(rf'"--{flag}",\s*\n?\s*action=argparse\.BooleanOptionalAction',
                    ma_src) is not None)
check("no store_true boolean with default True remains",
      not re.search(r'action="store_true",\s*default=(True|USE_)', ma_src),
      "a store_true flag defaulting True cannot be disabled")
check("choices come from the shared constants, not duplicated literals",
      "choices=NORM_POSITIONS" in ma_src and "choices=ATTN_ORDERS" in ma_src
      and "choices=SOFTMAX_MODES" in ma_src and "choices=QUERY_LABEL_MODES" in ma_src)
check("run tag gets an architecture suffix", "_arch_suffix()" in ma_src)
check("suffix always encodes the MLP mode (no collision with pre-DC-4 runs)",
      '"sepmlp" if SEPARATE_MLP_F else "sharedmlp"' in ma_src)
check("training never enables attention stats",
      "collect_attn_stats=False" in ma_src)
check("2.8 knobs forwarded to the trainer",
      "warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, lr_attn=LR_ATTN" in ma_src)

# ---------------------------------------------------------------- 13. ASCII logs
print("\n[13] stdout stays ASCII (Condor / cp1252 safety)")
import glob
non_ascii = []
for f in sorted(glob.glob('*.py')) + sorted(glob.glob('Analysis/**/*.py', recursive=True)):
    lines = src(f).split('\n')
    depth = 0
    for i, l in enumerate(lines, 1):
        o = l.count('(') - l.count(')')
        if (('print(' in l) or depth > 0) and any(ord(c) > 127 for c in l):
            non_ascii.append(f"{f}:{i}")
        depth = max(0, o) if 'print(' in l else (max(0, depth + o) if depth > 0 else 0)
check("no non-ASCII inside any print()", not non_ascii, str(non_ascii[:5]))
lit = []
for f in sorted(glob.glob('*.py')):
    for i, l in enumerate(src(f).split('\n'), 1):
        st = l.strip()
        if st.startswith('#') or st.startswith('"""') or st.startswith("'''"):
            continue
        if re.search(r'f?["\'][^"\']*[^\x00-\x7f][^"\']*["\']', l) and 'f"' in l:
            lit.append(f"{f}:{i}")
check("no non-ASCII f-string literals that could reach stdout via a variable",
      not lit, str(lit[:5]))

# ---------------------------------------------------------------- 14. analysis script
print("\n[14] attention_stats.py (the M8 tool)")
as_src = src('Analysis/model/attention_stats.py')
check("forces the dense path (stats need attention matrices)",
      "use_nested_tensors=False" in as_src and "model.use_nested_tensors = False" in as_src)
check("enables stat collection", "collect_attn_stats=True" in as_src
      and "set_collect_attn_stats(True)" in as_src)
check("sweeps support sizes up to 512", "SUPPORT_SIZES = (16, 32, 64, 128, 256, 512)" in as_src)
check("can load a trained checkpoint", "_load_model_from_checkpoint" in as_src)
check("rejects non-attention checkpoints",
      "Attention stats only apply to context_attention checkpoints" in as_src)
check("writes a CSV", "to_csv(out, index=False)" in as_src)

# ---------------------------------------------------------------- 15. interface parity
print("\n[15] Interface parity with the PTN head")
check("CrossAttentionProtoNet defines forward() (evaluate.py calls it directly)",
      find_func(cls, "forward") is not None,
      "DrugOOD evaluation calls model.forward and would raise NotImplementedError")
_f = ast.get_source_segment(cm_src, find_func(cls, "forward"))
check("forward() delegates to predict_from_embeddings",
      "self.predict_from_embeddings(" in _f)
check("forward() defaults to the FS-Mol mahalanobis protocol",
      'distance: str = "mahalanobis"' in _f)
_CRLF = b"\x0d\x0a"
_SH = sorted(glob.glob("condor/*.sh") + glob.glob("tests/*.sh"))
check("shell scripts have LF line endings (Condor refuses CRLF executables)",
      all(_CRLF not in open(f, "rb").read() for f in _SH),
      "a .sh written from Python on Windows gets CRLF and Condor rejects it as an "
      "executable: 'Executable file ... is a script with CRLF line endings'. "
      "Write with newline='\\n' or convert before transferring. Offenders: "
      + str([f for f in _SH if _CRLF in open(f, "rb").read()]))

check("DC-12 final norm is shared by default (one metric space)",
      'share_final_norm: bool = True' in cm_src and 'self.final_norm = nn.LayerNorm' in cm_src,
      "support and query would sit in different affine frames")
check("DC-12 _attend routes both streams through the shared norm",
      'return self.final_norm(c_sup), self.final_norm(c_qry)' in cm_src)
check("DC-13 logit temperature defaults to sqrt_d",
      'logit_scale: str = "sqrt_d"' in cm_src)
check("DC-13 temperature applied in the batched head",
      'self._scale_logit(d_i - d_a)' in cm_src)
check("DC-13 temperature applied in predict_from_embeddings",
      'self._scale_logit(dists[:, 1] - dists[:, 0])' in cm_src)
check("DC-12/13 round-trip through the checkpoint loader",
      'share_final_norm=inferred_share' in ev_src and 'logit_scale=config.get' in ev_src)
check("DC-12/13 written into the checkpoint config",
      '"share_final_norm": SHARE_FINAL_NORM' in ma_src and '"logit_scale": LOGIT_SCALE' in ma_src)
check("DC-12/13 appear in the run-tag suffix",
      'splitnorm' in ma_src and 'ls{LOGIT_SCALE}' in ma_src)

check("main.py rejects arguments instead of ignoring them",
      "ArgumentParser(" in src("main.py") and "parse_args()" in src("main.py"),
      "an unknown flag would silently launch a full training run")

print("\n" + "=" * 70)
print(f"STATIC AUDIT: PASSED {len(OK)}   FAILED {len(BAD)}")
if BAD:
    for n in BAD:
        print(f"  - {n}")
    sys.exit(1)
print("All structural invariants hold.")
