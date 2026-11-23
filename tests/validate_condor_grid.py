"""
Validate a Condor submit grid before spending GPU hours.

For every row — active AND commented-out — this:
  1. parses it against the `queue ... from (...)` header (arity check),
  2. runs the real entry point with --dry_run, which resolves the whole config and
     constructs the model, so an invalid architecture fails in seconds,
  3. collects the resolved checkpoint path and asserts every row maps to a DISTINCT
     one, because two arms sharing an outputs/ folder silently overwrite each other.

Commented rows are checked too: they are the stages you will uncomment later, and a
typo there fails the submission after stage 1 has already run.

    python tests/validate_condor_grid.py                      # all .sub files
    python tests/validate_condor_grid.py condor/gnn_ablation.sub
"""

import csv
import glob
import os
import re
import shlex
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)


def parse(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"^queue\s+(.*?)\s+from\s*\($", txt, re.M)
    if not m:
        return None, []
    cols = [c.strip() for c in m.group(1).split(",")]
    body = txt.split("from (", 1)[1].rsplit(")", 1)[0]
    rows = []
    for raw in body.split("\n"):
        t = raw.strip()
        if not t:
            continue
        commented = t.startswith("#")
        inner = t.lstrip("#").strip() if commented else t
        # A real grid row begins with the script to run. Counting commas is not enough:
        # a prose comment inside the block can easily contain three of them.
        if not re.match(r"^[\w./-]+\.py\s*,", inner):
            continue
        vals = next(csv.reader([inner], skipinitialspace=True))
        if len(vals) != len(cols):
            raise AssertionError(
                f"{path}: row has {len(vals)} fields, header has {len(cols)}: {inner}")
        rows.append((commented, dict(zip(cols, vals))))
    return cols, rows


def build_args(txt, row):
    """Reconstruct the command from the submit file's `arguments =` line."""
    # Read the `arguments =` line and follow backslash continuations. A regex with a
    # lookahead does not work here: the block is followed by `#` comments, not by a
    # word character, so the lookahead never fires and it swallows the whole file.
    lines = txt.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if re.match(r"^\s*arguments\s*=", l))
    buf = []
    for l in lines[start:]:
        stripped = l.rstrip()
        cont = stripped.endswith("\\")
        buf.append(stripped[:-1] if cont else stripped)
        if not cont:
            break
    raw = " ".join(" ".join(buf).split())
    raw = raw.split("=", 1)[1].strip()
    for k, v in row.items():
        raw = raw.replace(f"$({k})", v)
    left = re.findall(r"\$\(([A-Za-z_]+)\)", raw)
    assert not left, f"unresolved macros {left} in: {raw}"
    parts = shlex.split(raw)
    # Two idioms are in use:
    #   vanilla universe: `arguments = python <script.py> ...`
    #   docker universe:  `executable = condor/run_job.sh`, so arguments start with
    #                     the script and run_job.sh supplies the interpreter.
    if parts and parts[0] == "python":
        parts = parts[1:]
    assert parts and parts[0].endswith(".py"), \
        f"arguments must start with a .py script, got: {parts[:2]}"
    return parts


def main():
    targets = sys.argv[1:] or sorted(glob.glob("condor/*.sub"))
    overall_bad = 0
    for path in targets:
        txt = open(path, encoding="utf-8").read()
        cols, rows = parse(path)
        if not rows:
            print(f"\n{path}: no queue grid found, skipped")
            continue
        print(f"\n{'=' * 100}\n{path}   ({len(cols)} columns, {len(rows)} rows)\n{'=' * 100}")
        seen, bad = {}, 0
        for commented, row in rows:
            args = build_args(txt, row) + ["--dry_run"]
            r = subprocess.run([sys.executable] + args, capture_output=True,
                               text=True, timeout=900, errors="replace")
            tag = "commented" if commented else "ACTIVE   "
            label = f"{row.get('arm', '?'):<16} seed={row.get('seed', '?')}"
            if r.returncode != 0 or "dry run OK" not in r.stdout:
                bad += 1
                print(f"  FAIL [{tag}] {label}\n{(r.stderr or r.stdout)[-700:]}")
                continue
            ck = [l for l in r.stdout.splitlines() if "checkpoint ->" in l]
            ck = ck[0].split("->")[1].strip() if ck else "<none>"
            if ck in seen:
                bad += 1
                print(f"  COLLISION [{tag}] {label}\n"
                      f"      shares a checkpoint path with {seen[ck]}\n      {ck}")
            else:
                seen[ck] = label
                print(f"  OK   [{tag}] {label}  -> {os.path.basename(os.path.dirname(os.path.dirname(ck)))}")
        print(f"\n  {len(rows)} rows, {len(seen)} distinct checkpoint paths, {bad} problems")
        overall_bad += bad

    print(f"\n{'=' * 100}")
    if overall_bad:
        print(f"GRID VALIDATION FAILED: {overall_bad} problems")
        sys.exit(1)
    print("GRID VALIDATION PASSED - every row resolves and every row has its own folder")


if __name__ == "__main__":
    main()
