#!/bin/bash
# Build the showcase tower from several piles in parallel and keep the tallest (most stones, then
# height) as results/showcase. Usage: scripts/build_showcase_best.sh [n_piles] [candidates]
set -u
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
N=${1:-8}
C=${2:-24}
for k in $(seq 0 $((N - 1))); do
  python3 build_showcase.py --profile study --stone-class blocky_quarry --max-stones 9 --pile 14 --candidates $C --tower $k --showcase-dir showcase_pile$k > ../results/logs/showcase_pile$k.log 2>&1 &
done
wait
python3 - <<'PY'
import json, shutil
from pathlib import Path
R = Path("../results")
best = None
for d in sorted(R.glob("showcase_pile*")):
    f = d / "summary.json"
    if not f.exists():
        continue
    s = json.load(open(f))
    key = (s["n_stones"], s["height_m"])
    print(d.name, key)
    if best is None or key > best[0]:
        best = (key, d)
if best is not None:
    dst = R / "showcase"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(best[1], dst)
    print("kept", best[1].name, best[0])
PY
