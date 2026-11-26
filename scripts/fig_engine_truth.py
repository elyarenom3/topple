"""Score the stated predictions against outcomes measured in MuJoCo and Drake.

Reads the replays written by run_engine_truth.py (results/engine_truth/<experiment>/tower_XX.json)
and the experiment records. For each engine it reports how often the outcome agrees with the
topple's engine, and the calibration of the stated survival probability (Brier, ECE, AUC) and
the error of the stated margin (bias, RMSE, Spearman) with that engine's outcomes as the truth.
Intervals are 95% bootstrap intervals over towers. Uses whatever towers have been replayed.

    python fig_engine_truth.py
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

OUT = ROOT / "results" / "engine_truth"
ENGINES = ("topple", "mujoco", "drake")
N_BOOT = 1000


def key(r):
    return (r["tower"], r["placement"], r["stone"], bool(r.get("probe", r.get("meta", {}).get("probe", False))))


def load(exp: str):
    """Rows with the stated predictions and the outcome in each engine."""
    rec_path = sorted(glob.glob(str(ROOT / "results" / "experiments" / f"{exp}_*" / "records.json")))[0]
    recs = {}
    for r in json.load(open(rec_path)):
        recs.setdefault(key(r), []).append(r)
    rows = []
    for f in sorted(glob.glob(str(OUT / exp / "tower_*.json"))):
        for row in json.load(open(f))["rows"]:
            cands = recs.get(key(row), [])
            if not cands:
                continue
            # the same stone can be probed twice at one step: match on the recorded prediction and outcome
            rec = row.get("recorded", {})
            r = min(cands, key=lambda c: abs(c["p_hat"] - rec.get("p_hat", c["p_hat"])) + abs(c["m_true"] - rec.get("m_true", c["m_true"])) + abs(c["m_hat"] - rec.get("m_hat", c["m_hat"])))
            cands.remove(r)
            out = {"tower": row["tower"], "record": r, "y": {"topple": float(r["y"])}, "tilt": {"topple": float(r["m_true"])}, "replay": row.get("topple_replay")}
            ok = True
            for e, v in row["engines"].items():
                if "error" in v:
                    ok = False
                    continue
                out["y"][e] = float(v["fixed"])
                out["tilt"][e] = float(v["tilt"])
            out["complete"] = ok and all(e in out["y"] for e in ENGINES)
            rows.append(out)
    return rows


def predictions(exp: str, rows) -> dict:
    """Stated survival probabilities (and margins) of each method on the replayed rows."""
    P, M = {}, {}
    recs = [x["record"] for x in rows]
    if exp == "E2all":
        P["Monte Carlo"] = np.array([r["p_hat"] for r in recs])
        M["Monte Carlo"] = np.array([r["m_hat"] for r in recs])
        for k, name in (("method:none", "point estimate"), ("method:unscented", "unscented"), ("margin:rocking", "rocking margin, Monte Carlo")):
            P[name] = np.array([r["meta"]["preds"][k]["p_hat"] for r in recs])
            M[name] = np.array([r["meta"]["preds"][k]["mean"] for r in recs])
        P["Monte Carlo with learned recalibration"] = _recalibrated(recs)
    else:
        P["tilt margin"] = np.array([r["p_hat"] for r in recs])
        M["tilt margin"] = np.array([r["m_hat"] for r in recs])
        P["rocking margin"] = np.array([r["meta"]["preds"]["margin:rocking"]["p_hat"] for r in recs])
        M["rocking margin"] = np.array([r["meta"]["preds"]["margin:rocking"]["mean"] for r in recs])
    return P, M


def _recalibrated(test):
    """Final method on E2all: trained on the other experiment groups (fig_recalibration.py)."""
    from topple.eval.recalibration import leave_tower_out, settings_from_name

    train = []
    for p in sorted(glob.glob(str(ROOT / "results" / "experiments" / "*" / "records.json"))):
        name = Path(p).parent.name
        if name.startswith(("E1all", "E2all", "E2grow", "E2plan", "E5_propmc_mtilt_k1.0_N8_ns1.0_fs0.25_")):
            continue
        s = settings_from_name(name)
        for r in json.load(open(p)):
            r["settings"] = s
            train.append(r)
    for r in test:
        r.setdefault("settings", settings_from_name("E2all_propmc_mtilt_k1.0_N8_ns1.0_fs0.25_d0.05_"))
    return leave_tower_out(test, train, "boosted", context=True)


def _auc(p, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def _boot(fn, towers, rng):
    ut = np.unique(towers)
    idx = {t: np.where(towers == t)[0] for t in ut}
    vals = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[t] for t in rng.choice(ut, size=len(ut), replace=True)])
        vals.append(fn(ii))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def analyse(exp: str) -> dict:
    rows = [r for r in load(exp) if r["complete"]]
    if not rows:
        return {}
    towers = np.array([r["tower"] for r in rows])
    rng = np.random.default_rng(0)
    res = {"n": len(rows), "towers": int(len(np.unique(towers)))}
    rep = [r for r in rows if r["replay"] is not None]
    res["replay_check"] = {"n": len(rep), "same_y": int(sum(r["replay"]["fixed"] == r["y"]["topple"] for r in rep)),
                           "max_tilt_diff_deg": float(max([abs(r["replay"]["tilt"] - r["tilt"]["topple"]) for r in rep] or [0.0]))}
    Y = {e: np.array([r["y"][e] for r in rows]) for e in ENGINES}
    T = {e: np.array([r["tilt"][e] for r in rows]) for e in ENGINES}
    from scipy.stats import spearmanr

    res["agreement"] = {}
    for e in ENGINES[1:]:
        res["agreement"][e] = {"same_y": float(np.mean(Y[e] == Y["topple"])), "tilt_mad_deg": float(np.mean(np.abs(T[e] - T["topple"]))),
                               "tilt_spearman": float(spearmanr(T[e], T["topple"])[0]), "survival_rate": float(Y[e].mean())}
    res["agreement"]["topple"] = {"survival_rate": float(Y["topple"].mean())}
    P, M = predictions(exp, rows)
    res["scores"] = {}
    for name, p in P.items():
        res["scores"][name] = {}
        for e in ENGINES:
            y = Y[e]
            d = {"brier": float(np.mean((p - y) ** 2)), "ece": float(expected_calibration_error(p, y, 10)), "auc": _auc(p, y)}
            d["brier_ci"] = _boot(lambda ii: np.mean((p[ii] - y[ii]) ** 2), towers, rng)
            d["ece_ci"] = _boot(lambda ii: expected_calibration_error(p[ii], y[ii], 10), towers, rng)
            if name in M:
                m, t = M[name], T[e]
                ok = np.isfinite(m) & np.isfinite(t)
                d["margin_bias"] = float(np.mean(m[ok] - t[ok]))
                d["margin_rmse"] = float(np.sqrt(np.mean((m[ok] - t[ok]) ** 2)))
                d["margin_spearman"] = float(spearmanr(m[ok], t[ok])[0])
            res["scores"][name][e] = d
    return res


def main():
    results = {exp: analyse(exp) for exp in ("E1all", "E2all")}
    json.dump(results, open(OUT / "engine_truth.json", "w"), indent=1)
    md = ["# Predictions scored against MuJoCo and Drake", "", "Written by `scripts/fig_engine_truth.py` from the replays of `scripts/run_engine_truth.py`. Intervals: 95% bootstrap over towers.", ""]
    for exp, res in results.items():
        if not res:
            continue
        rc = res["replay_check"]
        md += [f"## {exp} ({res['n']} placements, {res['towers']} towers)", "",
               f"Replay check in the topple's engine: {rc['same_y']} of {rc['n']} building placements give the recorded outcome; largest critical tilt difference {rc['max_tilt_diff_deg']:.2f} degrees.", "",
               "| Engine | Survival rate | Same outcome as topple's engine | Critical tilt, mean difference (deg) | Spearman |", "| --- | --- | --- | --- | --- |"]
        for e in ENGINES:
            a = res["agreement"][e]
            md.append(f"| {e} | {a['survival_rate']:.2f} | {'' if e == 'topple' else f'{a['same_y']:.2f}'} | {'' if e == 'topple' else f'{a['tilt_mad_deg']:.2f}'} | {'' if e == 'topple' else f'{a['tilt_spearman']:.2f}'} |")
        md += ["", "| Prediction | Truth from | Brier | ECE | AUC | Margin bias (deg) | Margin RMSE (deg) | Margin Spearman |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, by_e in res["scores"].items():
            for e, d in by_e.items():
                ci_b, ci_e = d["brier_ci"], d["ece_ci"]
                mb = f"{d['margin_bias']:+.2f} | {d['margin_rmse']:.2f} | {d['margin_spearman']:.2f}" if "margin_bias" in d else " | | "
                md.append(f"| {name} | {e} | {d['brier']:.3f} [{ci_b[0]:.3f}, {ci_b[1]:.3f}] | {d['ece']:.3f} [{ci_e[0]:.3f}, {ci_e[1]:.3f}] | {d['auc']:.2f} | {mb} |")
        md.append("")
    (OUT / "engine_truth.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
