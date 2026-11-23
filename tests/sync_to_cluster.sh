#!/bin/bash
# Checksum-verified code sync to the cluster.
#
#   bash tests/sync_to_cluster.sh            # report differences only
#   bash tests/sync_to_cluster.sh --push     # transfer the differing files, then re-verify
#
# Why this exists: a piecemeal `scp a.py b.py dir/*.py` silently misses files that sit
# in a directory you did not name - which is how the cluster ended up running an old
# Analysis/eda_comprehensive.py while the local suite passed. This compares md5 of every
# tracked source file on both sides and reports MISSING / DIFFERS / OK, so "the cluster
# matches local" becomes something verified rather than assumed.
#
# Deliberately NOT synced: data/ (7.6 GB, already on the cluster), outputs/ (results are
# produced there), venv/ (built by condor/setup_env.sub), __pycache__.
set -u

REMOTE=chjo00006@conduit2.hpc.uni-saarland.de
RDIR=/home/chjo00006/PTN
PUSH=${1:-}

cd "$(dirname "$0")/.." || exit 1

# The file set: source and docs, never data or build artefacts.
mapfile -t FILES < <(
  find . -type f \( -name '*.py' -o -name '*.sub' -o -name '*.sh' -o -name '*.md' \
                    -o -name '*.txt' -o -name '*.json' -o -name '*.html' \) \
    -not -path './data/*' -not -path './outputs/*' -not -path './venv/*' \
    -not -path '*/__pycache__/*' -not -path './.git/*' \
    | sed 's|^\./||' | sort
)
# A few small GENERATED artefacts live under data/ but are code-like, tiny, and
# required at run time. data/ is otherwise excluded (7.6 GB, already on the cluster).
for extra in data/fsmol_meta/fsmol-0.1.json              data/custom/mic_splits.json              data/custom/mic_splits_chembl36.json; do
    [ -f "$extra" ] && FILES+=("$extra")
done

echo "comparing ${#FILES[@]} files against $REMOTE:$RDIR"

# one round trip for every remote checksum
REMOTE_SUMS=$(printf '%s\n' "${FILES[@]}" \
  | ssh -o BatchMode=yes "$REMOTE" "cd $RDIR && xargs -I{} sh -c 'if [ -f \"{}\" ]; then md5sum \"{}\"; else echo \"MISSING  {}\"; fi' 2>/dev/null")

declare -a NEED=()      # must be initialised: `set -u` trips on an empty array
ok=0; diff=0; miss=0
for f in "${FILES[@]}"; do
    lsum=$(md5sum "$f" | awk '{print $1}')
    rline=$(printf '%s\n' "$REMOTE_SUMS" | grep -F "  $f" | head -1)
    if [ -z "$rline" ] || [[ "$rline" == MISSING* ]]; then
        echo "  MISSING  $f"; NEED+=("$f"); miss=$((miss+1))
    else
        rsum=$(echo "$rline" | awk '{print $1}')
        if [ "$lsum" != "$rsum" ]; then
            echo "  DIFFERS  $f"; NEED+=("$f"); diff=$((diff+1))
        else
            ok=$((ok+1))
        fi
    fi
done

echo
echo "  $ok identical, $diff differ, $miss missing"

if [ ${#NEED[@]} -eq 0 ]; then
    echo "cluster matches local."
    exit 0
fi

if [ "$PUSH" != "--push" ]; then
    echo "re-run with --push to transfer the ${#NEED[@]} file(s) above."
    exit 1
fi

echo
echo "pushing ${#NEED[@]} file(s)..."
# create any needed directories first, then copy each file to its own path
printf '%s\n' "${NEED[@]}" | xargs -n1 dirname | sort -u \
  | sed "s|^|mkdir -p $RDIR/|" | ssh -o BatchMode=yes "$REMOTE" 'bash -s'
for f in "${NEED[@]}"; do
    scp -q -o BatchMode=yes "$f" "$REMOTE:$RDIR/$f" && echo "  sent $f" || echo "  FAILED $f"
done

echo
echo "re-verifying..."
# plain re-run (no --push) so a persistent mismatch reports instead of looping
exec bash "$0"
