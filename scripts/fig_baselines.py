"""Baselines for the stated survival probability, scored on the placements the system scored.

Evaluation set: every E2all placement (chosen and probe, as in the report). Each placement
already holds the Monte Carlo survival probability (the main method) and the point estimate,
unscented, first order and rocking predictions for the same placement.

Baseline 1, curve fits: a logistic or isotonic map from one margin value to survival, fitted on
the training towers. Applied to the point estimate tilt margin and to the Monte Carlo mean
margin, and as isotonic recalibration of the Monte Carlo probability itself. On E1all (exact
shapes) the same fits compare the support polygon margin with the tilt, push, force closure and
rocking margins.

Baseline 2, learned classifiers: gradient boosted trees and logistic regression on features in
the records (margins, Monte Carlo summaries, level in the tower, tower height, number of contacts,
stone class, and the experiment's noise, friction, risk and delta settings). Variants: point
estimate features on E2all only; with Monte Carlo features on E2all only; with Monte Carlo
features trained on E2all, E4, E5 and E6 together.

Validation: leave one tower out. The model for tower t never sees tower t of any experiment,
since experiments share stones by tower index and the E5 centre point repeats E2all placements.
Metrics: ECE (10 equal width bins), Brier score and ROC AUC, with 95% intervals from a bootstrap
over towers; differences from the Monte Carlo probability use the same (paired) resamples.

    python fig_baselines.py            # writes results/baselines/
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topple.eval.metrics import expected_calibration_error, reliability_diagram  # noqa: E402

EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "baselines"
CLASSES = ["flat_river", "blocky_quarry", "rounded_cobble"]
N_BOOT = 2000


# ---------------------------------------------------------------------------- data
def _settings(name: str) -> dict:
    def num(tag, default):
        m = re.search(rf"_{tag}([0-9.]+?)_", name)
        return float(m.group(1)) if m else default

    return {"kappa": num("k", 1.0), "ns": num("ns", 1.0), "fs": num("fs", 0.25), "delta": num("d", 0.05)}


def load(pattern: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(EXP / pattern / "records.json"))):
        name = os.path.basename(os.path.dirname(p))
        s = _settings(name)
        for r in json.load(open(p)):
            r = dict(r)
            r["exp"] = name
            r.update(s)
            out.append(r)
    return out


def mc_features(r: dict) -> list[float]:
    smp = np.asarray(r["samples"], dtype=float)
    return [r["m_hat"], r["m_lo"], r["m_hi"], float(smp.std()) if smp.size else 0.0, r["p_hat"]]


def base_features(r: dict) -> list[float]:
    cls = r["meta"].get("stone_class", "")
    return [float(r["placement"]), float(r["height"]), float(r["meta"].get("n_contacts", np.nan))] + [float(cls == c) for c in CLASSES]


def setting_features(r: dict) -> list[float]:
    return [r["ns"], r["fs"], r["kappa"], r["delta"]]


def point_features(r: dict) -> list[float]:
    pr = r["meta"]["preds"]
    return [pr["method:none"]["mean"]]


# ---------------------------------------------------------------------------- models
def _logistic():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _boosted():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, l2_regularization=1.0, random_state=0)


def _isotonic():
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")


def loto(test_rows, train_pool, feat, model_fn, kind="classifier"):
    """Leave one tower out predictions for ``test_rows``; training uses ``train_pool`` without
    any record of the held out tower index."""
    p = np.full(len(test_rows), np.nan)
    towers = np.array([r["tower"] for r in test_rows])
    tr_towers = np.array([r["tower"] for r in train_pool])
    X_all = np.array([feat(r) for r in train_pool], dtype=float)
    y_all = np.array([r["y"] for r in train_pool], dtype=float)
    X_te_all = np.array([feat(r) for r in test_rows], dtype=float)
    for t in np.unique(towers):
        te = towers == t
        tr = tr_towers != t
        m = model_fn()
        if kind == "isotonic":
            m.fit(X_all[tr, 0], y_all[tr])
            p[te] = m.predict(X_te_all[te, 0])
        else:
            m.fit(X_all[tr], y_all[tr].astype(int))
            p[te] = m.predict_proba(X_te_all[te])[:, 1]
    return p


# ---------------------------------------------------------------------------- metrics
def _auc(p, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def metrics(p, y):
    return {"ece": float(expected_calibration_error(p, y, 10)), "brier": float(np.mean((p - y) ** 2)), "auc": _auc(p, y)}


def bootstrap(preds: dict, y, towers, ref: str, rng):
    """Point values, 95% intervals and paired differences from ``ref`` (bootstrap over towers)."""
    ut = np.unique(towers)
    idx_of = {t: np.where(towers == t)[0] for t in ut}
    boots = {k: {m: [] for m in ("ece", "brier", "auc")} for k in preds}
    diffs = {k: {m: [] for m in ("ece", "brier", "auc")} for k in preds}
    for _ in range(N_BOOT):
        pick = rng.choice(ut, size=len(ut), replace=True)
        ii = np.concatenate([idx_of[t] for t in pick])
        mref = metrics(preds[ref][ii], y[ii])
        for k, p in preds.items():
            mk = metrics(p[ii], y[ii])
            for m in mk:
                boots[k][m].append(mk[m])
                diffs[k][m].append(mk[m] - mref[m])
    out = {}
    for k, p in preds.items():
        pt = metrics(p, y)
        row = {}
        for m in pt:
            b = np.array(boots[k][m])
            d = np.array(diffs[k][m])
            row[m] = [pt[m], float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
            row["d_" + m] = [pt[m] - metrics(preds[ref], y)[m], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        out[k] = row
    return out


# ---------------------------------------------------------------------------- report
def table(res: dict, groups: dict, ref: str, title: str, n: int, n_towers: int) -> str:
    def f(v):
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    def fd(v):
        return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"

    lines = [f"### {title}", "", f"{n} placements, {n_towers} towers. Intervals: 95%, bootstrap over towers. Differences are paired, relative to {ref}.", "",
             "| Method | Group | ECE | Brier | AUC | ECE difference | Brier difference |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, row in res.items():
        dece = "" if k == ref else fd(row["d_ece"])
        dbr = "" if k == ref else fd(row["d_brier"])
        lines.append(f"| {k} | {groups[k]} | {f(row['ece'])} | {f(row['brier'])} | {f(row['auc'])} | {dece} | {dbr} |")
    return "\n".join(lines) + "\n"


def figure(preds: dict, y, towers, keys, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        d = reliability_diagram(preds[k], y, n_bins=10, n_boot=200, rng=np.random.default_rng(0), groups=towers)
        conf = np.asarray(d["pbar"], dtype=float)
        acc = np.asarray(d["ybar"], dtype=float)
        cnt = np.asarray(d["counts"], dtype=float)
        ok = (cnt > 0) & np.isfinite(conf) & np.isfinite(acc)
        ax.plot([0, 1], [0, 1], color="0.6", lw=1)
        ax.scatter(conf[ok], acc[ok], s=12 + 60 * cnt[ok] / max(cnt.max(), 1), color="#3b6ea8", zorder=3)
        ax.plot(conf[ok], acc[ok], color="#3b6ea8", lw=1)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("stated survival probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("observed survival rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # ------------------------------------------------ E2all: same placements for every method
    e2 = load("E2all_*")
    pool = e2 + load("E4_*") + load("E5_*") + load("E6_*")
    y = np.array([r["y"] for r in e2], dtype=float)
    towers = np.array([r["tower"] for r in e2])
    pr = lambda key: np.array([r["meta"]["preds"][key]["p_hat"] for r in e2], dtype=float)  # noqa: E731

    preds, groups = {}, {}

    def add(name, group, p):
        preds[name] = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        groups[name] = group

    add("Monte Carlo (main method)", "stated by the system", [r["p_hat"] for r in e2])
    add("point estimate", "stated by the system", pr("method:none"))
    add("unscented", "stated by the system", pr("method:unscented"))
    add("first order", "stated by the system", pr("method:first_order"))
    add("rocking margin", "stated by the system", pr("margin:rocking"))

    pe = lambda r: point_features(r)  # noqa: E731
    mcm = lambda r: [r["m_hat"]]  # noqa: E731
    mcp = lambda r: [r["p_hat"]]  # noqa: E731
    add("logistic on point estimate margin", "curve fit", loto(e2, e2, pe, _logistic))
    add("isotonic on point estimate margin", "curve fit", loto(e2, e2, pe, _isotonic, "isotonic"))
    add("logistic on Monte Carlo mean margin", "curve fit", loto(e2, e2, mcm, _logistic))
    add("Monte Carlo, isotonic recalibration", "curve fit", loto(e2, e2, mcp, _isotonic, "isotonic"))

    f_point = lambda r: point_features(r) + base_features(r)  # noqa: E731
    f_mc_e2 = lambda r: point_features(r) + mc_features(r) + base_features(r)  # noqa: E731
    f_mc_all = lambda r: mc_features(r) + base_features(r) + setting_features(r)  # noqa: E731
    add("boosted trees, point estimate features", "learned (E2all)", loto(e2, e2, f_point, _boosted))
    add("logistic, point estimate features", "learned (E2all)", loto(e2, e2, f_point, _logistic))
    add("boosted trees, with Monte Carlo features", "learned (E2all)", loto(e2, e2, f_mc_e2, _boosted))
    add("logistic, with Monte Carlo features", "learned (E2all)", loto(e2, e2, f_mc_e2, _logistic))
    add("boosted trees, all experiments", f"learned ({len(pool)} placements)", loto(e2, pool, f_mc_all, _boosted))
    add("logistic, all experiments", f"learned ({len(pool)} placements)", loto(e2, pool, f_mc_all, _logistic))

    ref = "Monte Carlo (main method)"
    res = bootstrap(preds, y, towers, ref, rng)

    # ------------------------------------------------ E1all: exact shapes, margin types
    e1 = load("E1all_*")
    y1 = np.array([r["y"] for r in e1], dtype=float)
    t1 = np.array([r["tower"] for r in e1])
    preds1, groups1 = {}, {}
    for kind, getter in [("tilt", lambda r: [r["m_hat"]])] + [(k, (lambda k: lambda r: [r["meta"]["preds"]["margin:" + k]["mean"]])(k)) for k in ("support", "push", "fc", "rocking")]:
        label = {"fc": "force closure", "support": "support polygon"}.get(kind, kind)
        preds1[f"logistic on {label} margin"] = loto(e1, e1, getter, _logistic)
        groups1[f"logistic on {label} margin"] = "curve fit"
        preds1[f"isotonic on {label} margin"] = np.clip(loto(e1, e1, getter, _isotonic, "isotonic"), 0, 1)
        groups1[f"isotonic on {label} margin"] = "curve fit"
    preds1["tilt margin as stated (exact shapes)"] = np.array([r["p_hat"] for r in e1], dtype=float)
    groups1["tilt margin as stated (exact shapes)"] = "stated by the system"
    ref1 = "logistic on tilt margin"
    res1 = bootstrap(preds1, y1, t1, ref1, np.random.default_rng(1))

    # ------------------------------------------------ write
    json.dump({"E2all": {"n": len(e2), "towers": int(len(np.unique(towers))), "pool": len(pool), "results": res, "groups": groups},
               "E1all": {"n": len(e1), "towers": int(len(np.unique(t1))), "results": res1, "groups": groups1}},
              open(OUT / "baselines.json", "w"), indent=1)
    md = ["# Baselines for the survival probability", "", "Written by `scripts/fig_baselines.py` from the saved experiment records; leave one tower out validation.", "",
          table(res, groups, ref, "E2all: estimated shapes, every method on the same placements", len(e2), len(np.unique(towers))),
          table(res1, groups1, ref1, "E1all: exact shapes, curve fits per margin type", len(e1), len(np.unique(t1)))]
    (OUT / "baselines.md").write_text("\n".join(md))
    best_fit = min((k for k in preds if groups[k] == "curve fit" and "Monte Carlo" not in k), key=lambda k: res[k]["brier"][0])
    best_learned = min((k for k in preds if groups[k].startswith("learned")), key=lambda k: res[k]["brier"][0])
    figure(preds, y, towers, [ref, "point estimate", best_fit, best_learned], OUT / "baselines_reliability.png")
    print((OUT / "baselines.md").read_text())


if __name__ == "__main__":
    main()
