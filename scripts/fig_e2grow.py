"""E2 with a looser chance constraint (delta 0.3, condition E2grow) and with the learned
recalibration in the planner's chance constraint (delta 0.5, condition E2plan, when present),
compared with E2all (delta 0.05).

Reports how far the towers grow and the calibration of each stated survival probability, on all
placements and split by level (on the ground, placement 0; stacked, placement 1 or higher).
The learned recalibration is trained on E4, E5 (off centre) and E6 only, with tower indices held
out. Intervals are 95% bootstrap intervals over towers.

    python fig_e2grow.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topple.eval.metrics import expected_calibration_error  # noqa: E402
from topple.eval.recalibration import leave_tower_out, settings_from_name, training_records  # noqa: E402

EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "e2grow"
N_BOOT = 1000
SKIP_TRAIN = ("E1all", "E2all", "E2grow", "E2plan", "E2cand", "E2view", "E2fresh", "E5_propmc_mtilt_k1.0_N8_ns1.0_fs0.25_")
PLANNER_RECAL = ("E2plan", "E2cand", "E2view", "E2plan2")
EXPERIMENTS = ("E2all", "E2grow") + PLANNER_RECAL


def load(prefix: str):
    p = sorted(glob.glob(str(EXP / f"{prefix}_*" / "records.json")))[0]
    s = settings_from_name(Path(p).parent.name)
    recs = json.load(open(p))
    for r in recs:
        r["settings"] = s
    return recs


def train_set():
    out = []
    for p in sorted(glob.glob(str(EXP / "*" / "records.json"))):
        name = Path(p).parent.name
        if name.startswith(SKIP_TRAIN):
            continue
        s = settings_from_name(name)
        for r in json.load(open(p)):
            r["settings"] = s
            out.append(r)
    return out


def predictions(recs, train, exp):
    P = {"Monte Carlo": np.array([r["p_hat"] for r in recs])}
    for k, name in (("method:none", "point estimate"), ("margin:rocking", "rocking margin, Monte Carlo")):
        P[name] = np.array([r["meta"]["preds"][k]["p_hat"] for r in recs])
    P["Monte Carlo with learned recalibration"] = leave_tower_out(recs, train, "boosted", context=True)
    if exp in PLANNER_RECAL:
        # the recalibrator the planner used: earlier perception experiments, this tower index held out
        P["recalibration used by the planner"] = leave_tower_out(recs, training_records(EXP), "boosted", context=True)
    return P


def _auc(p, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def score(p, y, towers, rng):
    def stats(ii):
        return float(np.mean((p[ii] - y[ii]) ** 2)), float(expected_calibration_error(p[ii], y[ii], n_bins=10))

    ut = np.unique(towers)
    idx = {t: np.where(towers == t)[0] for t in ut}
    boots = np.array([stats(np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])) for _ in range(N_BOOT)])
    b, e = stats(np.arange(len(y)))
    return {"brier": b, "brier_ci": np.percentile(boots[:, 0], [2.5, 97.5]).tolist(), "ece": e,
            "ece_ci": np.percentile(boots[:, 1], [2.5, 97.5]).tolist(), "auc": _auc(p, y), "mean_p": float(p.mean()), "rate": float(y.mean())}


def growth(recs):
    build = [r for r in recs if not r["meta"].get("probe")]
    towers = sorted({r["tower"] for r in recs})
    stones = [sum(1 for r in build if r["tower"] == t) for t in towers]
    ended = sum(1 for r in recs if r["meta"].get("tag") == "final_infeasible")
    heights = [max([r["height"] for r in build if r["tower"] == t] or [0.0]) for t in towers]
    return {"towers": len(towers), "stones_mean": float(np.mean(stones)), "stones_max": int(max(stones)),
            "height_mean_cm": 100 * float(np.mean(heights)), "height_max_cm": 100 * float(max(heights)),
            "collapse_rate": float(np.mean([r["collapsed"] for r in build])), "n": len(recs),
            "n_stacked": sum(r["placement"] >= 1 for r in recs), "n_build": len(build),
            "steps_accepted": len(build) / max(len(build) + ended, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = train_set()
    res = {}
    exps = [e for e in EXPERIMENTS if glob.glob(str(EXP / f"{e}_*" / "records.json"))]
    for exp in exps:
        recs = load(exp)
        y = np.array([r["y"] for r in recs], dtype=float)
        towers = np.array([r["tower"] for r in recs])
        level = np.array([r["placement"] for r in recs])
        P = predictions(recs, train, exp)
        res[exp] = {"growth": growth(recs), "scores": {}}
        for split, m in (("all", level >= 0), ("ground", level == 0), ("stacked", level >= 1)):
            res[exp]["scores"][split] = {k: score(p[m], y[m], towers[m], np.random.default_rng(0)) for k, p in P.items()}
    json.dump(res, open(OUT / "e2grow.json", "w"), indent=1)

    def ci(v, k):
        return f"{v[k]:.3f} [{v[k + '_ci'][0]:.3f}, {v[k + '_ci'][1]:.3f}]"

    L = ["# E2 with a looser chance constraint", "",
         "Written by `scripts/fig_e2grow.py`. E2all uses delta 0.05 and E2grow delta 0.3. E2plan uses delta 0.5 with the learned recalibration in the chance constraint; E2cand adds 48 candidate poses per stone, E2view 8 in hand views at two elevations and 640 x 480 depth images, and E2plan2 the sequential screen with acceptance on a fresh full evaluation. All use the perceived stones and the same stones per tower index. Steps accepted: building placements over steps that ended a tower for want of a candidate meeting the chance constraint plus building placements. Intervals: 95% bootstrap over towers.", "",
         "| Experiment | Towers | Stones per tower (mean, max) | Height cm (mean, max) | Collapse rate | Steps accepted | Placements | Stacked placements |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for exp, v in res.items():
        g = v["growth"]
        L.append(f"| {exp} | {g['towers']} | {g['stones_mean']:.2f}, {g['stones_max']} | {g['height_mean_cm']:.1f}, {g['height_max_cm']:.1f} | {g['collapse_rate']:.2f} | {g['steps_accepted']:.2f} | {g['n']} | {g['n_stacked']} |")
    for split in ("all", "ground", "stacked"):
        L += ["", f"## Placements: {split}", "", "| Experiment | Prediction | n | Survival rate | Mean stated p | Brier | ECE | AUC |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for exp, v in res.items():
            for k, s in v["scores"][split].items():
                nn = v["growth"]["n_stacked"] if split == "stacked" else (v["growth"]["n"] - v["growth"]["n_stacked"] if split == "ground" else v["growth"]["n"])
                L.append(f"| {exp} | {k} | {nn} | {s['rate']:.2f} | {s['mean_p']:.2f} | {ci(s, 'brier')} | {ci(s, 'ece')} | {s['auc']:.2f} |")
    (OUT / "e2grow.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
