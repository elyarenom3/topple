#!/usr/bin/env python3
"""Run one experimental condition (Section 11.3) and write records plus a results table row.

Examples
--------
python scripts/run_experiment.py --condition E1 --profile quick --towers 2
python scripts/run_experiment.py --condition E2 --propagation unscented --towers 10
python scripts/run_experiment.py --condition E4 --kappa 0.5 --placement arm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _worker(payload):
    """Run one tower in a worker process (numba threads limited to one per worker)."""
    cfg, cond_kwargs, tower, verbose = payload
    from topple.pipeline import Condition, Experiment

    exp = Experiment(cfg, Condition(**cond_kwargs), verbose=verbose)
    try:
        return tower, exp.run_tower(tower), None
    except Exception as exc:  # a failing tower is reported, the others continue
        import traceback

        return tower, [], traceback.format_exc()

from topple.core.config import load_config  # noqa: E402
from topple.core.provenance import record as provenance_record  # noqa: E402
from topple.eval.report import condition_row, write_table  # noqa: E402
from topple.pipeline import Condition, Experiment, records_to_rows, save_records  # noqa: E402

# Conditions (Section 11.3). Shared placements: E1 and E7 evaluate all margins and cone
# discretisations on the oracle placements; E2 evaluates all propagation methods and the simulation
# score (E8) on the same perceived placements, so their tests are run once.
CONDITIONS = {
    "E1all": dict(oracle=True, propagation="none", margin="tilt", extra={"eval_margins": ["push", "fc", "support", "rocking"], "eval_cones": [4, 16, 32]}),
    "E2all": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["none", "unscented", "first_order"], "eval_simulation": True, "eval_margins": ["rocking"]}),
    # E2 with a looser chance constraint (run with --delta 0.3) so that towers grow under perception
    "E2grow": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["none"], "eval_margins": ["rocking"]}),
    # E2 planning with the learned recalibration in the chance constraint (run with --delta 0.5)
    "E2plan": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["none"], "eval_margins": ["rocking"], "recalibrate_planner": True}),
    # diagnostics for tower height under perception (E2plan settings plus the overrides in
    # CONFIG_OVERRIDES): more candidate poses per stone, and more and sharper depth views
    "E2cand": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["none"], "eval_margins": ["rocking"], "recalibrate_planner": True}),
    "E2view": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["none"], "eval_margins": ["rocking"], "recalibrate_planner": True}),
    # E2plan with the screening changes: sequential screen, acceptance on a fresh full evaluation,
    # duplicate settled poses skipped, flat top ranking (settings in CONFIG_OVERRIDES)
    "E2plan2": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["none"], "eval_margins": ["rocking"], "recalibrate_planner": True}),
    # E2plan2 with a frozen recalibrator (scripts/fig_calibration_check.py --fit), run on fresh
    # stone sets with --tower-start 24 as the held out calibration test
    "E2fresh": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["none"], "eval_margins": ["rocking"], "recalibrate_planner": True, "recalibrator_path": str(Path(__file__).resolve().parents[1] / "results" / "recalibration" / "frozen.pkl")}),
    "E2ut": dict(oracle=False, propagation="mc", margin="tilt", extra={"eval_methods": ["unscented"]}),
    "E1": dict(oracle=True, propagation="none", margin="tilt"),
    "E1_push": dict(oracle=True, propagation="none", margin="push"),
    "E1_fc": dict(oracle=True, propagation="none", margin="fc"),
    "E2": dict(oracle=False, propagation="mc", margin="tilt"),
    "E2_point": dict(oracle=False, propagation="none", margin="tilt"),
    "E2_ut": dict(oracle=False, propagation="unscented", margin="tilt"),
    "E2_fo": dict(oracle=False, propagation="first_order", margin="tilt"),
    "E4": dict(oracle=False, propagation="mc", margin="tilt"),
    "E5": dict(oracle=False, propagation="mc", margin="tilt"),
    "E6": dict(oracle=False, propagation="mc", margin="tilt"),
    "E7": dict(oracle=True, propagation="none", margin="tilt"),
    "E8": dict(oracle=False, propagation="none", margin="simulation"),
}

# configuration changes that belong to a condition (dotted keys), applied before --set
CONFIG_OVERRIDES = {
    "E2cand": {"planner.n_candidates": 48},
    "E2plan2": {"planner.n_candidates": 48, "planner.sequential_screen": True, "planner.accept_full": True, "planner.accept_tries": 3, "planner.dedupe_mm": 2.0, "planner.build_weight": 3.0},
    "E2fresh": {"planner.n_candidates": 48, "planner.sequential_screen": True, "planner.accept_full": True, "planner.accept_tries": 3, "planner.dedupe_mm": 2.0, "planner.build_weight": 3.0},
    "E2view": {"sensor.width": 640, "sensor.height": 480, "sensor.in_hand_views": 8, "sensor.in_hand_elevations": [-30.0, -70.0]},
}


def set_dotted(cfg: dict, key: str, value) -> None:
    """Set cfg["a"]["b"] = value for key "a.b", creating sections as needed."""
    *path, last = key.split(".")
    d = cfg
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="E2")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--towers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--propagation", default=None)
    ap.add_argument("--margin", default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--placement", default=None)
    ap.add_argument("--cone-sides", type=int, default=None)
    ap.add_argument("--stone-class", default=None)
    ap.add_argument("--noise-scale", type=float, default=None)
    ap.add_argument("--friction-log-sigma", type=float, default=None)
    ap.add_argument("--tests", default="tilt,fixed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="towers run in parallel processes")
    ap.add_argument("--max-stones", type=int, default=None)
    ap.add_argument("--probes", type=int, default=None, help="probe placements per step")
    ap.add_argument("--tag", default="", help="suffix for the output directory name")
    ap.add_argument("--delta", type=float, default=None, help="chance constraint p_hat >= 1 - delta")
    ap.add_argument("--tower-start", type=int, default=0, help="first tower index (fresh stone sets for held out tests)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, e.g. planner.n_candidates=48 (value parsed as YAML)")
    args = ap.parse_args(argv)

    cfg = load_config(args.profile)
    if args.towers is not None:
        cfg["experiment"]["n_towers"] = args.towers
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.max_stones is not None:
        cfg["experiment"]["max_stones"] = args.max_stones
    if args.probes is not None:
        cfg["experiment"]["probes_per_step"] = args.probes
    if args.delta is not None:
        cfg["planner"]["delta"] = args.delta
    import yaml

    for key, value in CONFIG_OVERRIDES.get(args.condition, {}).items():
        set_dotted(cfg, key, value)
    for item in args.set:
        key, _, value = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    if args.placement == "arm":
        # the pile within the reach of the arm, and fingers that open wide enough for the stones
        cfg["stones"]["pile_layout"] = "arc"
        cfg["arm"]["finger_max"] = max(float(cfg["arm"].get("finger_max", 0.08)), 0.12)
    base = dict(CONDITIONS.get(args.condition, CONDITIONS["E2"]))
    for k, v in [("propagation", args.propagation), ("margin", args.margin), ("kappa", args.kappa), ("placement", args.placement), ("cone_sides", args.cone_sides), ("stone_class", args.stone_class), ("noise_scale", args.noise_scale), ("friction_log_sigma", args.friction_log_sigma)]:
        if v is not None:
            base[k] = v
    base.setdefault("kappa", cfg["planner"]["kappa"])
    base.setdefault("cone_sides", cfg["world"]["cone_sides"])
    base.setdefault("friction_log_sigma", cfg["friction"]["prior_log_sigma"])
    base["tests"] = tuple(args.tests.split(","))
    name = args.condition
    suffix = "_".join(f"{k}{v}" for k, v in [("prop", base["propagation"]), ("m", base["margin"]), ("k", base["kappa"]), ("N", base["cone_sides"]), ("ns", base["noise_scale"] if "noise_scale" in base else 1.0), ("fs", base["friction_log_sigma"]), ("d", cfg["planner"]["delta"]), ("cls", base.get("stone_class") or "all")])
    cond = Condition(name=name, **base)
    t0 = time.time()
    towers = list(range(args.tower_start, args.tower_start + int(cfg["experiment"]["n_towers"])))
    errors = {}
    if args.workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # one thread per worker: numba, OpenBLAS and OpenMP pools otherwise oversubscribe the cores
        for var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "1"
        ctx = mp.get_context("spawn")
        cond_kwargs = {k: v for k, v in cond.__dict__.items()}
        records = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            for tower, recs, err in ex.map(_worker, [(cfg, cond_kwargs, tw, not args.quiet) for tw in towers]):
                records += recs
                if err:
                    errors[tower] = err
                    print(f"tower {tower} failed:\n{err}", flush=True)
                else:
                    print(f"tower {tower} done: {len(recs)} records", flush=True)
    else:
        exp = Experiment(cfg, cond, verbose=not args.quiet)
        records = []
        for tw in towers:
            try:
                records += exp.run_tower(tw)
            except Exception:
                import traceback

                errors[tw] = traceback.format_exc()
                print(f"tower {tw} failed:\n{errors[tw]}", flush=True)
    records.sort(key=lambda r: (r.tower, r.placement, bool(r.meta.get("probe", False))))
    elapsed = time.time() - t0
    out_dir = Path(args.out) / "experiments" / f"{name}_{suffix}{('_' + args.tag) if args.tag else ''}_{cfg['_config_hash']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_records(records, out_dir / "records.json")
    row = condition_row(name, cond.margin, cond.propagation, records_to_rows(records), n_bins=int(cfg["eval"]["bins"]))
    row["elapsed_s"] = elapsed
    write_table([row], out_dir / "table.csv")
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance_record(cfg, {"condition": cond.__dict__, "elapsed_s": elapsed, "tower_errors": errors}), f, indent=1, default=str)
    print(json.dumps({k: row.get(k) for k in ("condition", "mean_height", "mean_stones_before_failure", "collapse_rate", "rmse_tilt_deg", "coverage_95", "brier", "ece", "ms_per_decision", "n_placements", "n_probes")}, indent=1, default=float))
    print("wrote", out_dir)
    return out_dir


if __name__ == "__main__":
    main()
