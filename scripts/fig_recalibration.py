"""Final method: the Monte Carlo survival probability with learned recalibration.

The recalibrator (``topple/eval/recalibration.py``) is tested on each experiment group in turn,
trained only on the other groups: E2all, E4 (four risk weights, arm placement, delta 0.3), E5 (the
six points off the centre of the cross design) and E6 (three stone classes). Within that, the
model for tower t never sees tower index t, since experiments share stones by tower index. The
E5 centre point repeats E2all placements, so it is left out of training when E2all is tested.

Methods: the Monte Carlo probability as stated; isotonic recalibration of it; logistic
regression on the Monte Carlo outputs; gradient boosted trees on the Monte Carlo outputs and the
placement context (the final method, chosen beforehand from the E2all baselines). Metrics: Brier
score, ECE (10 bins) and ROC AUC, with 95% intervals from a bootstrap over tower indices and
paired differences from the stated probability. The final calibrator, trained on every group, is
saved to results/recalibration/calibrator.pkl.

    python fig_recalibration.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topple.eval.metrics import expected_calibration_error, reliability_diagram  # noqa: E402
from topple.eval.recalibration import Recalibrator, leave_tower_out, settings_from_name  # noqa: E402

EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "recalibration"
N_BOOT = 2000
CENTRE = "E5_propmc_mtilt_k1.0_N8_ns1.0_fs0.25_"


def load(pattern: str, exclude: str | None = None) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(EXP / pattern / "records.json"))):
        name = os.path.basename(os.path.dirname(p))
        if exclude and name.startswith(exclude):
            continue
        s = settings_from_name(name)
        for r in json.load(open(p)):
            r["settings"] = s
            r["exp"] = name
            out.append(r)
    return out


def isotonic_loto(test, train):
    from sklearn.isotonic import IsotonicRegression

    p = np.full(len(test), np.nan)
    towers = np.array([r["tower"] for r in test])
    for t in np.unique(towers):
        tr = [r for r in train if r["tower"] != t]
        m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit([r["p_hat"] for r in tr], [r["y"] for r in tr])
        idx = np.where(towers == t)[0]
        p[idx] = m.predict([test[i]["p_hat"] for i in idx])
    return p


def metrics(p, y):
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(expected_calibration_error(p, y, 10)), "auc": auc}


def bootstrap(preds: dict, y, towers, ref, rng):
    ut = np.unique(towers)
    idx_of = {t: np.where(towers == t)[0] for t in ut}
    keys = ("brier", "ece", "auc")
    boots = {k: {m: [] for m in keys} for k in preds}
    diffs = {k: {m: [] for m in keys} for k in preds}
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_of[t] for t in rng.choice(ut, size=len(ut), replace=True)])
        mref = metrics(preds[ref][ii], y[ii])
        for k, p in preds.items():
            mk = metrics(p[ii], y[ii])
            for m in keys:
                boots[k][m].append(mk[m])
                diffs[k][m].append(mk[m] - mref[m])
    out = {}
    ref_pt = metrics(preds[ref], y)
    for k, p in preds.items():
        pt = metrics(p, y)
        out[k] = {}
        for m in keys:
            out[k][m] = [pt[m], float(np.nanpercentile(boots[k][m], 2.5)), float(np.nanpercentile(boots[k][m], 97.5))]
            out[k]["d_" + m] = [pt[m] - ref_pt[m], float(np.nanpercentile(diffs[k][m], 2.5)), float(np.nanpercentile(diffs[k][m], 97.5))]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {
        "E2all": (load("E2all_*"), lambda r: r["exp"].startswith("E2all") or r["exp"].startswith(CENTRE)),
        "E4": (load("E4_*"), lambda r: r["exp"].startswith("E4")),
        "E5": (load("E5_*", exclude=CENTRE), lambda r: r["exp"].startswith("E5")),
        "E6": (load("E6_*"), lambda r: r["exp"].startswith("E6")),
    }
    everything = load("E2all_*") + load("E4_*") + load("E5_*") + load("E6_*")
    ref = "Monte Carlo as stated"
    final = "Monte Carlo with learned recalibration (final)"
    results, pooled = {}, {"y": [], "towers": [], "preds": {}}
    for g, (test, in_group) in groups.items():
        train = [r for r in everything if not in_group(r)]
        y = np.array([r["y"] for r in test], dtype=float)
        towers = np.array([r["tower"] for r in test])
        preds = {
            ref: np.array([r["p_hat"] for r in test], dtype=float),
            "isotonic recalibration": isotonic_loto(test, train),
            "logistic on Monte Carlo outputs": leave_tower_out(test, train, "logistic", context=False),
            final: leave_tower_out(test, train, "boosted", context=True),
        }
        results[g] = {"n": len(test), "n_train": len(train), "towers": int(len(np.unique(towers))), "res": bootstrap(preds, y, towers, ref, np.random.default_rng(0))}
        pooled["y"].append(y)
        pooled["towers"].append(np.array([f"{g}:{t}" for t in towers]))
        for k, p in preds.items():
            pooled["preds"].setdefault(k, []).append(p)
        print(g, len(test), "train", len(train), {k: round(v["brier"][0], 3) for k, v in results[g]["res"].items()})
    y_all = np.concatenate(pooled["y"])
    t_all = np.concatenate(pooled["towers"])
    p_all = {k: np.concatenate(v) for k, v in pooled["preds"].items()}
    results["all groups"] = {"n": len(y_all), "towers": int(len(np.unique(t_all))), "res": bootstrap(p_all, y_all, t_all, ref, np.random.default_rng(1))}

    # final calibrator on every group
    Recalibrator("boosted", context=True).fit(everything).save(OUT / "calibrator.pkl")

    def f(v):
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    def fd(v):
        return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"

    md = ["# Monte Carlo with learned recalibration", "",
          "Written by `scripts/fig_recalibration.py`. Each group is predicted by models trained on the other groups only, with tower indices held out; intervals are 95% bootstrap intervals over towers and differences are paired with the stated Monte Carlo probability.", ""]
    for g, d in results.items():
        md += [f"## {g} ({d['n']} placements, {d['towers']} towers)", "", "| Method | Brier | ECE | AUC | Brier difference | ECE difference |", "| --- | --- | --- | --- | --- | --- |"]
        for k, row in d["res"].items():
            md.append(f"| {k} | {f(row['brier'])} | {f(row['ece'])} | {f(row['auc'])} | {'' if k == ref else fd(row['d_brier'])} | {'' if k == ref else fd(row['d_ece'])} |")
        md.append("")
    (OUT / "recalibration.md").write_text("\n".join(md))
    json.dump(results, open(OUT / "recalibration.json", "w"), indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(groups) + ["all groups"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.3), sharey=True)
    for ax, g in zip(axes, names):
        if g == "all groups":
            y, tw, pp = y_all, t_all, p_all
        else:
            i = names.index(g)
            y, tw = pooled["y"][i], pooled["towers"][i]
            pp = {k: v[i] for k, v in pooled["preds"].items()}
        ax.plot([0, 1], [0, 1], color="0.6", lw=1)
        for k, col in ((ref, "#9aa5b1"), (final, "#3b6ea8")):
            d = reliability_diagram(pp[k], y, n_bins=10, n_boot=50, rng=np.random.default_rng(0), groups=tw)
            c = np.asarray(d["counts"], dtype=float)
            ok = (c > 0) & np.isfinite(d["pbar"]) & np.isfinite(d["ybar"])
            ax.plot(np.asarray(d["pbar"])[ok], np.asarray(d["ybar"])[ok], "o-", ms=4, color=col, lw=1, label="stated" if k == ref else "recalibrated")
        ax.set_title(g, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("survival probability")
    axes[0].set_ylabel("observed survival rate")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "recalibration_reliability.png", dpi=150)
    print((OUT / "recalibration.md").read_text())


if __name__ == "__main__":
    main()
