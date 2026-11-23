import importlib
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

print("=== FIX 1: config paths resolve from this file's location ===")
import config
root = config.REPO_ROOT.replace(os.sep, '/')
# The invariant is that REPO_ROOT resolves to the directory holding config.py - NOT
# that it equals any particular path. Asserting a literal like "Thesis/PTN" is exactly
# the machine-specific assumption this test exists to rule out, and it failed the first
# time the suite ran on the cluster, where the root is /home/chjo00006/PTN.
expect = os.path.dirname(os.path.abspath(config.__file__)).replace(os.sep, '/')
assert root == expect, f"REPO_ROOT {root!r} != the directory holding config.py {expect!r}"
print(f"  REPO_ROOT    {root}   (resolved from config.py's own location)")
for n in ("FSMOL_TRAIN", "FSMOL_VAL", "FSMOL_TEST", "DRUGOOD_DIR", "CUSTOM_DIR"):
    p = getattr(config, n)
    assert os.path.isdir(p), f"{n} -> {p} missing"
    print(f"  {n:<12} {p}")
cfg_src = open('config.py', encoding='utf-8').read()
assert "os.path.exists(\"/netscratch\")" not in cfg_src, "Pegasus probe still present"
code_lines = [l.split('#')[0] for l in cfg_src.splitlines()]
assert not any('D:/Thesis' in l for l in code_lines), \
    "hardcoded Windows path still present in code (comments are fine)"
print("  OK: no /netscratch probe, no hardcoded absolute paths")

print("\n=== FIX 1b: env overrides (how Condor will drive it) ===")
for env, expect_root, expect_data in (
        ({"PTN_ROOT": "/home/chjo00006/PTN"}, "/home/chjo00006/PTN", "/home/chjo00006/PTN/data"),
        ({"PTN_ROOT": "/home/chjo00006/PTN", "PTN_DATA": "/scratch/d"},
         "/home/chjo00006/PTN", "/scratch/d")):
    for k, v in env.items():
        os.environ[k] = v
    importlib.reload(config)
    got_root = config.REPO_ROOT.replace(os.sep, '/')
    got_data = config.DATA_ROOT.replace(os.sep, '/')
    assert got_root == expect_root, f"{got_root} != {expect_root}"
    assert got_data == expect_data, f"{got_data} != {expect_data}"
    print(f"  {env} -> root={got_root} data={got_data}")
    for k in env:
        os.environ.pop(k)
importlib.reload(config)

print("\n=== FIX 3: SUPPORT_SAMPLING is honoured, not hardcoded ===")
from data import stratified_context_indices
labels = np.array([1] * 85 + [0] * 15)          # DrugOOD-like: 85% active
for mode, lo, hi in (("proportional", 0.75, 0.95), ("balanced", 0.40, 0.60)):
    os.environ["PTN_SUPPORT_SAMPLING"] = mode
    importlib.reload(config)
    assert config.SUPPORT_SAMPLING == mode
    fracs = []
    for seed in range(20):
        idx = stratified_context_indices(
            labels, 16, np.random.RandomState(seed),
            balanced=(config.SUPPORT_SAMPLING == "balanced"))
        fracs.append(labels[idx].mean())
    m = float(np.mean(fracs))
    print(f"  {mode:<13} support active fraction = {m:.3f}  (expected {lo}-{hi})")
    assert lo <= m <= hi, f"{mode}: {m}"
os.environ.pop("PTN_SUPPORT_SAMPLING")
importlib.reload(config)
assert config.SUPPORT_SAMPLING == "proportional"
print("  OK: default proportional, which is what training uses")

print("\n=== FIX 3b: no hardcoded balanced=True remains at a call site ===")
for f in ("data.py", "evaluate.py"):
    src = open(f, encoding='utf-8').read()
    bad = [l.strip() for l in src.split('\n')
           if 'stratified_context_indices(' in l or 'balanced=True)' in l]
    for l in bad:
        assert 'balanced=True)' not in l, f"{f}: {l}"
    print(f"  {f}: call sites read config.SUPPORT_SAMPLING")

print("\n=== FIX 3c: invalid value rejected at import ===")
os.environ["PTN_SUPPORT_SAMPLING"] = "fifty-fifty"
try:
    importlib.reload(config)
    raise AssertionError("garbage value was accepted")
except ValueError as e:
    print(f"  OK: {e}")
finally:
    os.environ.pop("PTN_SUPPORT_SAMPLING")
    importlib.reload(config)

print("\n=== FIX 3d: training still samples proportionally (unchanged) ===")
from data import load_fsmol_assay, EpisodeSampler
from config import FSMOL_TEST
f = sorted(os.path.join(FSMOL_TEST, x) for x in os.listdir(FSMOL_TEST)
           if x.endswith('.jsonl.gz'))[0]
ds, _ = load_fsmol_assay(f)
pop = float(np.mean(ds.binary_labels))
samp = EpisodeSampler(n_support=16, n_query=32)
fr = []
for _ in range(40):
    si, _qi = samp._get_episode_indices_random(ds)
    fr.append(float(np.mean(ds.binary_labels[si])))
print(f"  assay active fraction {pop:.3f}; mean support active fraction {np.mean(fr):.3f}")
assert abs(np.mean(fr) - pop) < 0.12, "training support no longer tracks the assay ratio"
print("  OK: train and eval now use the same rule")

print("\nALL THREE FIXES VERIFIED")
