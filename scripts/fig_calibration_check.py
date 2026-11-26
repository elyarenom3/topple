#!/usr/bin/env python3
"""Calibration check: choose, freeze and test the recalibrator on fresh towers.

    python fig_calibration_check.py --fit    # compare recalibrators, freeze the better one
    python fig_calibration_check.py --test   # score the frozen recalibrator on E2fresh

--fit pools every perceived stone experiment on towers 0 to 23 (E2all, E2grow, E4, E5, E6,
E2plan, E2cand, E2view, E2plan2), compares the gradient boosted recalibrator with a few parameter
logistic one (logit of the Monte Carlo probability, a stacked indicator, their product and the
depth noise scale) under leave one tower out, freezes the one with the lower Brier score (fitted
on all pooled records) to results/recalibration/frozen.pkl, and writes the comparison.

--test scores the frozen recalibrator on E2fresh (the E2plan2 planner with the frozen
recalibrator in its chance constraint, on fresh stone sets: towers 24 to 47), on all, stacked and
building placements, and compares observed whole tower survival with the product of the stated
probabilities of each tower's building placements.

Scores: Brier, ECE (10 equal width bins), ECE (7 equal mass bins), debiased L2 calibration error
(Kumar et al. 2019) and AUC, with 95% bootstrap intervals over tower indices. "Chance level" is the
mean ECE of a perfectly calibrated predictor with the same stated probabilities and sample size.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topple.eval.metrics import debiased_calibration_error, expected_calibration_error  # noqa: E402
from topple.eval.recalibration import Recalibrator, leave_tower_out, settings_from_name  # noqa: E402

EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "calibration_check"
FROZEN = ROOT / "results" / "recalibration" / "frozen.pkl"
POOL = ("E2all", "E2grow", "E4", "E5", "E6", "E2plan", "E2cand", "E2view", "E2plan2")
KINDS = {"boosted": "gradient boosted (current)", "logit_level": "logistic on logit p, stacked terms"}
N_BOOT = 1000


def load(prefix: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(EXP / f"{prefix}_*" / "records.json"))):
        name = Path(p).parent.name
        s = settings_from_name(name)
        for r in json.load(open(p)):
            r["settings"] = s
            r["exp"] = name
            out.append(r)
    return out


def _auc(p, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def _stats(p, y):
    return (float(np.mean((p - y) ** 2)), expected_calibration_error(p, y, 10), expected_calibration_error(p, y, 7, "equal_mass"), debiased_calibration_error(p, y, 7))


def chance_ece(p, rng, n=500):
    """Mean ECE (10 equal width bins) of a perfectly calibrated predictor stating p."""
    return float(np.mean([expected_calibration_error(p, (rng.random(len(p)) < p).astype(float), 10) for _ in range(n)]))


def score(p, y, towers, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    ut = np.unique(towers)
    idx = {t: np.where(towers == t)[0] for t in ut}
    boots = np.array([_stats(p[ii], y[ii]) for ii in (np.concatenate([idx[t] for t in rng.choice(ut, len(ut))]) for _ in range(N_BOOT))])
    pt = _stats(p, y)
    out = {"n": int(len(y)), "rate": float(y.mean()), "mean_p": float(p.mean()), "auc": _auc(p, y), "chance_ece": chance_ece(p, rng)}
    for j, k in enumerate(("brier", "ece", "ece_mass", "dce")):
        out[k] = [pt[j], float(np.nanpercentile(boots[:, j], 2.5)), float(np.nanpercentile(boots[:, j], 97.5))]
    return out


def fmt(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def table(title, rows):
    L = [f"## {title}", "", "| Placements | Prediction | n | Survival rate | Mean stated p | Brier | ECE | ECE, equal mass | Debiased L2 error | ECE at chance level | AUC |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for split, name, s in rows:
        L.append(f"| {split} | {name} | {s['n']} | {s['rate']:.2f} | {s['mean_p']:.2f} | {fmt(s['brier'])} | {fmt(s['ece'])} | {fmt(s['ece_mass'])} | {fmt(s['dce'])} | {s['chance_ece']:.3f} | {s['auc']:.2f} |")
    return L


def fit():
    recs = [r for pre in POOL for r in load(pre)]
    y = np.array([r["y"] for r in recs], dtype=float)
    towers = np.array([r["tower"] for r in recs])
    stacked = np.array([r["placement"] >= 1 for r in recs])
    P = {"Monte Carlo as stated": np.array([r["p_hat"] for r in recs])}
    for kind, label in KINDS.items():
        P[label] = leave_tower_out(recs, recs, kind)
    res, rows = {}, []
    for split, m in (("all", np.ones(len(y), bool)), ("stacked", stacked)):
        res[split] = {}
        for name, p in P.items():
            s = score(p[m], y[m], towers[m])
            res[split][name] = s
            rows.append((split, name, s))
    briers = {kind: res["all"][label]["brier"][0] for kind, label in KINDS.items()}
    # the simpler model unless the boosted one has a clearly lower Brier score (by 0.002)
    choice = "boosted" if briers["boosted"] < briers["logit_level"] - 0.002 else "logit_level"
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    Recalibrator(choice, context=True).fit(recs).save(FROZEN)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"pool": POOL, "n": int(len(y)), "choice": choice, "scores": res}, open(OUT / "model_choice.json", "w"), indent=1)
    exps = sorted({r["exp"].split("_")[0] for r in recs})
    L = ["# Recalibrator choice", "", f"Written by `scripts/fig_calibration_check.py --fit`. {len(y)} placements from {', '.join(exps)} (towers 0 to 23), leave one tower out. Intervals: 95% bootstrap over tower indices.", ""]
    L += table("Leave one tower out", rows)
    L += ["", f"Frozen: **{KINDS[choice]}** (`{choice}`), fitted on all {len(y)} placements, saved to `results/recalibration/frozen.pkl`. The simpler model is kept unless the boosted one lowers the Brier score by more than 0.002."]
    (OUT / "model_choice.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def test():
    recs = load("E2fresh")
    if not recs:
        sys.exit("no E2fresh records")
    frozen = Recalibrator.load(FROZEN)
    y = np.array([r["y"] for r in recs], dtype=float)
    towers = np.array([r["tower"] for r in recs])
    stacked = np.array([r["placement"] >= 1 for r in recs])
    build = np.array([not r["meta"].get("probe") for r in recs])
    P = {"Monte Carlo as stated": np.array([r["p_hat"] for r in recs]), "frozen recalibrator": frozen.predict(recs)}
    rows, res = [], {}
    for split, m in (("all", np.ones(len(y), bool)), ("stacked", stacked), ("building", build)):
        res[split] = {}
        for name, p in P.items():
            s = score(p[m], y[m], towers[m])
            res[split][name] = s
            rows.append((split, name, s))
    # planner's acceptance probability (fresh full evaluation, frozen recalibrator) on building placements
    acc = np.array([r["meta"].get("accept_p") if r["meta"].get("accept_p") is not None else np.nan for r in recs], dtype=float)
    ma = build & np.isfinite(acc)
    if ma.sum() > 1:
        s = score(acc[ma], y[ma], towers[ma])
        res["building"]["planner acceptance probability"] = s
        rows.append(("building", "planner acceptance probability", s))
    # whole tower: observed survival of every building placement compared with the product of
    # their stated probabilities
    tw = []
    for t in np.unique(towers):
        mb = build & (towers == t)
        if not mb.any():
            continue
        p_t = acc[mb] if np.isfinite(acc[mb]).all() else P["frozen recalibrator"][mb]
        tw.append((float(np.prod(p_t)), float(np.all(y[mb] == 1)), int(mb.sum())))
    tw = np.array(tw)
    rng = np.random.default_rng(0)
    bt = np.array([tw[rng.integers(len(tw), size=len(tw))].mean(axis=0)[:2] for _ in range(N_BOOT)])
    tower = {"towers": int(len(tw)), "mean_predicted": float(tw[:, 0].mean()), "observed": float(tw[:, 1].mean()),
             "predicted_ci": np.percentile(bt[:, 0], [2.5, 97.5]).tolist(), "observed_ci": np.percentile(bt[:, 1], [2.5, 97.5]).tolist(),
             "brier": float(np.mean((tw[:, 0] - tw[:, 1]) ** 2)), "stones_mean": float(tw[:, 2].mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"scores": res, "tower": tower, "frozen": frozen.kind}, open(OUT / "fresh_test.json", "w"), indent=1)
    L = ["# Frozen recalibrator on fresh towers (E2fresh)", "",
         f"Written by `scripts/fig_calibration_check.py --test`. E2fresh: the E2plan2 planner with the frozen recalibrator (`{frozen.kind}`) in its chance constraint, on fresh stone sets (towers 24 to 47). Intervals: 95% bootstrap over towers.", ""]
    L += table("Placements", rows)
    L += ["", "## Whole towers", "", "| Towers | Building placements per tower | Mean of the product of stated probabilities | Observed share of towers where every placement survived | Brier |", "| --- | --- | --- | --- | --- |",
          f"| {tower['towers']} | {tower['stones_mean']:.2f} | {tower['mean_predicted']:.2f} [{tower['predicted_ci'][0]:.2f}, {tower['predicted_ci'][1]:.2f}] | {tower['observed']:.2f} [{tower['observed_ci'][0]:.2f}, {tower['observed_ci'][1]:.2f}] | {tower['brier']:.3f} |"]
    (OUT / "fresh_test.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.fit:
        fit()
    if a.test:
        test()
