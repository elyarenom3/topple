"""Replay experiment towers in MuJoCo and Drake (topple/validation/replay.py).

Writes one JSON file per tower to results/engine_truth/<experiment>/tower_XX.json and skips towers
already written, so an interrupted run resumes.

    python run_engine_truth.py --exp E1all --workers 3
    python run_engine_truth.py --exp E2all --workers 3 --towers 0-11
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"


def _worker(payload):
    exp_dir, tower, engines, out = payload
    import logging
    import traceback

    logging.getLogger("drake").setLevel(logging.ERROR)
    from topple.validation.replay import replay_tower

    t0 = time.time()
    try:
        rows = replay_tower(exp_dir, tower, engines=engines, log=lambda s: print(s, flush=True))
        json.dump({"exp": os.path.basename(exp_dir), "tower": tower, "engines": list(engines), "seconds": time.time() - t0, "rows": rows}, open(out, "w"), indent=1)
        return tower, len(rows), None
    except Exception:
        return tower, 0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True, help="experiment name prefix, e.g. E1all")
    ap.add_argument("--towers", default=None, help="range a-b or comma list (default: all in the records)")
    ap.add_argument("--engines", default="mujoco,drake")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    exp_dir = sorted(glob.glob(str(ROOT / "results" / "experiments" / f"{args.exp}_*")))[0]
    recs = json.load(open(Path(exp_dir) / "records.json"))
    towers = sorted({r["tower"] for r in recs})
    if args.towers:
        if "-" in args.towers:
            a, b = map(int, args.towers.split("-"))
            towers = [t for t in towers if a <= t <= b]
        else:
            towers = [int(x) for x in args.towers.split(",")]
    out_dir = ROOT / "results" / "engine_truth" / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    engines = tuple(args.engines.split(","))
    jobs = [(exp_dir, t, engines, str(out_dir / f"tower_{t:02d}.json")) for t in towers if not (out_dir / f"tower_{t:02d}.json").exists()]
    print(f"{args.exp}: {len(jobs)} towers to replay in {engines}", flush=True)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        for tower, n, err in ex.map(_worker, jobs):
            print(f"tower {tower} done: {n} records" if err is None else f"tower {tower} failed:\n{err}", flush=True)
    print(f"{args.exp} finished {time.ctime()}", flush=True)


if __name__ == "__main__":
    main()
