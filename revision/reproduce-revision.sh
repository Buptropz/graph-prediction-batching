#!/usr/bin/env bash
# Additional experiments of the revised paper.  Run from the extracted archive root or from anywhere
# inside it:  bash vldb2027/revision/reproduce-revision.sh <target> [output directory]
#   verify   regenerate every table, figure and number from stored results and compare the numbers
#   smoke    rerun GraphAny on one Cora panel (CPU, about ten seconds) and compare with stored rows
#   cpu      rerun every answer-only experiment on CPU (default configuration, adjacency sweep,
#            vectorized function, trained networks, Ephris and GraphAny lane plans)
#   gpu      rerun the timing, scale and arrival experiments on the device in CUDA_VISIBLE_DEVICES
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$root/vldb2027/upgrade/experiment:$root/vldb2027/revision/experiment${PYTHONPATH:+:$PYTHONPATH}"
py=${PYTHON:-python3}
exp=vldb2027/revision/experiment
res=vldb2027/revision/results
action=${1:-verify}
out=${2:-$(mktemp -d)}
mkdir -p "$out"
out=$(cd "$out" && pwd)
# Live targets read the upstream corpus files, which the archive restores from pinned URLs
# (about 17 MB, checked byte for byte; a no-op once the files are present).
if [ "$action" != verify ]; then "$py" vldb2027/reproduction/fetch_raw.py; fi
case "$action" in
 verify)
   cp "$res/headlines-rev.json" "$out/headlines-stored.json"
   "$py" "$exp/report.py" > /dev/null
   cmp "$out/headlines-stored.json" "$res/headlines-rev.json"
   echo "Numbers of the paper regenerated from stored results and matched."
   ;;
 smoke)
   "$py" "$exp/anygraph.py" --tasks cora --panels 0 --no-validate --out "$out/graphany.json" --vectors "$out/graphany.npz"
   "$py" - "$out/graphany.json" "$res/graphany.json" <<'PY'
import json, sys
fresh, stored = (json.load(open(p))["rows"] for p in sys.argv[1:3])
key = lambda r: (r["task"], r["panel"], r["features"], r["layout"], r["topology"], r["recipe"], r["fan"])
old = {key(r): r for r in stored}
for r in fresh:
    for plan in ("union", "axis"):
        for field in ("flips", "threshold", "changed_lists"):
            assert r[plan][field] == old[key(r)][plan][field], (key(r), plan, field)
print("Verified", len(fresh), "GraphAny conditions of Cora panel 0 against the stored counts.")
PY
   ;;
 cpu)
   "$py" "$exp/accuracy.py" --out "$out/accuracy.json"
   "$py" "$exp/default.py" --device cpu --out "$out/default"
   "$py" "$exp/dose.py" --device cpu --out "$out/dose.json"
   "$py" "$exp/chunks.py" --device cpu --out "$out/chunks.json"
   "$py" "$exp/gnn.py" --device cpu --out "$out/gnn.json"
   "$py" "$exp/ephris_axis.py" --device cpu --out "$out/ephris_axis.json"
   "$py" "$exp/anygraph.py" --out "$out/graphany.json" --vectors "$out/graphany.npz"
   ;;
 gpu)
   "$py" "$exp/timing.py" --out "$out/timing"
   "$py" "$exp/scale.py" --out "$out/scale.json"
   "$py" "$exp/arrivals.py" --out "$out/arrivals.json"
   ;;
 *) echo "unknown target: $action" >&2; exit 2 ;;
esac
echo "Output directory: $out"
