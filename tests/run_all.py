#!/usr/bin/env python
"""Run the whole verification suite.

    python tests/run_all.py            # everything
    python tests/run_all.py --fast     # skip the slow layers (no training)
    python tests/run_all.py --cpu-only # only layers that never touch CUDA
    python tests/run_all.py --gpu-only # only the layers that need a GPU

Three layers, cheapest first:

  0. verify_fixes.py        config path resolution + support-sampling consistency.
  1. audit_static.py        AST/text invariants. Needs no torch, so it still works when
                            the ML stack cannot load. ~1 second.
  2a audit_attention_mechanism.py  Information-flow and math invariants of the
                            attention model (no query-query flow, no episode
                            leakage, no label leak, shared metric space,
                            no logit saturation). Run before any GPU grid.
  2. test_design_choices.py Unit + numerical tests for every configurable design choice,
                            the loss-gradient fix, checkpoint round-trips, the GNN path
                            and the run-tag scheme. Needs torch. ~2 minutes.
  2b validate_condor_grid.py  Dry-runs every row of every condor/*.sub, active and
                            commented, and asserts each maps to its OWN output
                            folder. ~13s per row. Run before ANY submission.
  3. test_pipeline.py       End-to-end on real FS-Mol / MIC data: episodic training in a
                            subprocess per config, then reload + evaluate. ~20+ minutes.

On Windows, layer 3 needs PTN_NUM_WORKERS=0 (see TASKS.md section 9.5); this runner sets
it automatically on win32. Do NOT set it on the cluster - there, match request_cpus.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (label, script, slow, needs_gpu)
# needs_gpu is about cluster citizenship, not correctness: a layer that never touches
# CUDA should not hold a GPU slot while other jobs queue. Only the end-to-end layer
# trains; design-choices has a single fp16-autocast check.
SUITES = [
    ("static invariants",   "audit_static.py",              False, False),
    ("infra fixes",         "verify_fixes.py",              False, False),
    ("attention mechanism", "audit_attention_mechanism.py", False, False),
    ("condor grid",         "validate_condor_grid.py",      True,  False),
    ("design choices",      "test_design_choices.py",       False, True),
    ("end-to-end pipeline", "test_pipeline.py",             True,  True),
]


def main():
    fast = "--fast" in sys.argv
    cpu_only = "--cpu-only" in sys.argv
    gpu_only = "--gpu-only" in sys.argv
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.platform == "win32":
        # Spawn-based DataLoader workers each load torch + CUDA DLLs and exhaust the
        # Windows paging file. See TASKS.md section 9.5.
        env["PTN_NUM_WORKERS"] = "0"

    results = []
    for label, script, slow, needs_gpu in SUITES:
        skip = None
        if fast and slow:
            skip = "--fast"
        elif cpu_only and needs_gpu:
            skip = "--cpu-only"
        elif gpu_only and not needs_gpu:
            skip = "--gpu-only"
        if skip:
            print(f"\n### SKIP {label} ({skip})")
            results.append((label, None, 0.0))
            continue
        print(f"\n{'=' * 72}\n### {label}  ({script})\n{'=' * 72}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           cwd=ROOT, env=env)
        dt = time.time() - t0
        results.append((label, r.returncode, dt))
        if r.returncode != 0:
            print(f"\n### {label} FAILED after {dt:.0f}s - stopping here.", flush=True)
            break

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    worst = 0
    for label, rc, dt in results:
        state = "SKIPPED" if rc is None else ("PASS" if rc == 0 else f"FAIL (rc={rc})")
        print(f"  {state:<14} {label:<24} {dt:6.0f}s")
        if rc:
            worst = rc
    sys.exit(worst)


if __name__ == "__main__":
    main()
