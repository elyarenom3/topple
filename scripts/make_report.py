#!/usr/bin/env python3
"""Results table (Section 11.4) and the written report, from every experiment on disk.

Rows: E1 (exact inputs, three margins), E2 (perception, four propagation methods on the same
placements), E8 (simulation score on the same placements), E4 (risk weight), E5 (sensitivity),
E6 (stone class) and E7 (cone sides). Confidence intervals come from a bootstrap over towers.

python scripts/make_report.py
"""

from __future__ import annotations

import json

import numpy as np

from _common import FIGURES, RESULTS, STILLS, VIDEOS, find_experiments, load_experiment

from topple.eval.metrics import brier_score, expected_calibration_error, interval_coverage, margin_errors, murphy_decomposition, spearman
from topple.eval.report import COLUMNS, bootstrap_ci, condition_row, write_table

TAU = 3.0


def rows_with(records, key=None, rng=None):
    """Records with p_hat, m_hat and samples taken from meta['preds'][key] (None: base prediction).
    Methods without samples (unscented, first order) get normal samples from their mean and std."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for r in records:
        r2 = dict(r)
        r2["probe"] = bool(r.get("probe", r.get("meta", {}).get("probe", False)))
        if key is not None:
            d = r["meta"].get("preds", {}).get(key)
            if d is None or "mean" not in d:
                continue
            r2["m_hat"] = d["mean"]
            if key == "margin:simulation":
                r2["p_hat"] = float(d["mean"] > TAU)
                r2["samples"] = np.array([d["mean"]])
            else:
                r2["p_hat"] = d["p_hat"]
                if key.startswith("method:") and key.split(":")[1] in ("unscented", "first_order"):
                    r2["samples"] = np.clip(d["mean"] + d["std"] * rng.standard_normal(256), 0.0, None)
                else:
                    r2["samples"] = np.asarray(d["samples"], dtype=float) if d.get("samples") else np.array([d["mean"]])
        r2["samples"] = np.asarray(r2["samples"], dtype=float)
        if r2["samples"].size == 0:
            r2["samples"] = np.array([r2["m_hat"]])
        out.append(r2)
    return out


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


def ci_of(records, metric):
    if not records:
        return (np.nan, np.nan, np.nan)
    g = np.array([r["tower"] for r in records])
    if metric == "ece":
        p = np.array([r["p_hat"] for r in records])
        y = np.array([r["y"] for r in records], dtype=float)
        idx = np.arange(len(records))
        return bootstrap_ci(idx, g, stat=lambda ii: expected_calibration_error(p[ii.astype(int)], y[ii.astype(int)], 10) if len(ii) else np.nan, n_boot=200)
    if metric == "brier":
        v = np.array([(r["p_hat"] - r["y"]) ** 2 for r in records])
        return bootstrap_ci(v, g, n_boot=200)
    raise ValueError(metric)


def main():
    table_rows = []
    extra = {}
    # ------------------------------------------------------------------ E1 and E7
    recs1, _ = (None, None)
    p1 = find_experiments("E1all")
    if p1:
        recs1, _ = load_experiment(p1[-1])
        table_rows.append(condition_row("E1", "tilt", "none", rows_with(recs1)))
        ok = [r for r in recs1 if np.isfinite(r["m_true"]) and not r["collapsed"]]
        mt = np.array([r["m_true"] for r in ok])
        cone = {}
        for N in (4, 8, 16, 32):
            m = np.array([r["m_hat"] for r in ok]) if N == 8 else np.array([r["meta"]["preds"].get(f"cone:{N}", {}).get("mean", np.nan) for r in ok])
            cone[N] = margin_errors(m, mt)
        extra["E7"] = cone
        pt = np.array([r.get("push_true", np.nan) for r in ok], dtype=float)
        push = np.array([r["meta"]["preds"].get("margin:push", {}).get("mean", np.nan) for r in ok])
        fc = np.array([r["meta"]["preds"].get("margin:fc", {}).get("mean", np.nan) for r in ok])
        sup = np.array([r["meta"]["preds"].get("margin:support", {}).get("mean", np.nan) for r in ok])
        tilt = np.array([r["m_hat"] for r in ok])
        rock = np.array([r["meta"]["preds"].get("margin:rocking", {}).get("mean", np.nan) for r in ok])
        rr_rock = rows_with(recs1, "margin:rocking")
        if rr_rock:
            table_rows.append(condition_row("E1", "rocking (Margin 4)", "none", rr_rock))

        def sp(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            return spearman(a[m], b[m]), int(m.sum())

        extra["E1"] = {
            "tilt_vs_tilt_test": margin_errors(tilt, mt),
            "push_vs_push_test": margin_errors(push, pt),
            "rocking_vs_tilt_test": margin_errors(rock[np.isfinite(rock)], mt[np.isfinite(rock)]) if np.isfinite(rock).any() else None,
            "spearman": {"tilt/tilt": sp(tilt, mt), "push/push": sp(push, pt), "fc/tilt": sp(fc, mt), "fc/push": sp(fc, pt), "support/tilt": sp(sup, mt), "rocking/tilt": sp(rock, mt)},
            "n_single_stone": int(sum(1 for r in ok if r["placement"] == 0)),
            "single_stone": margin_errors(np.array([r["m_hat"] for r in ok if r["placement"] == 0]), np.array([r["m_true"] for r in ok if r["placement"] == 0])),
            "stacked": margin_errors(np.array([r["m_hat"] for r in ok if r["placement"] > 0]), np.array([r["m_true"] for r in ok if r["placement"] > 0])),
        }
        for lab, sel in (("single_stone", lambda r: r["placement"] == 0), ("stacked", lambda r: r["placement"] > 0)):
            a = np.array([rock[i] for i, r in enumerate(ok) if sel(r)])
            b = np.array([mt[i] for i, r in enumerate(ok) if sel(r)])
            m = np.isfinite(a)
            extra["E1"]["rocking_" + lab] = margin_errors(a[m], b[m]) if m.any() else None
    # ------------------------------------------------------------------ E2 and E8
    p2 = find_experiments("E2all")
    if p2:
        recs2, _ = load_experiment(p2[-1])
        # unscented predictions from the rerun on the same placements (towers 0 to 11)
        put = find_experiments("E2ut")
        has_ut = any("method:unscented" in r["meta"].get("preds", {}) and "error" not in r["meta"]["preds"]["method:unscented"] for r in recs2)
        if put and not has_ut:
            recs_ut, _ = load_experiment(put[-1])

            def key(r):
                T = np.asarray(r["meta"]["pose"])
                return (r["tower"], r["placement"], r["stone"], bool(r["meta"].get("probe")), tuple(np.round(T[:3, 3], 4)))

            ut = {key(r): r["meta"].get("preds", {}).get("method:unscented") for r in recs_ut}
            matched = 0
            for r in recs2:
                r["meta"].setdefault("preds", {}).pop("method:unscented", None)
                k = key(r)
                if k in ut and ut[k] is not None:
                    r["meta"]["preds"]["method:unscented"] = ut[k]
                    matched += 1
            extra["E2_unscented_matched"] = {"matched": matched, "rerun_records": len(recs_ut), "E2_records": len(recs2)}
            # the same subset for every method, for a like for like comparison
            sub = [r for r in recs2 if "method:unscented" in r["meta"]["preds"]]
            for k2, prop in ((None, "MC (unscented subset)"), ("method:unscented", "unscented")):
                rr = rows_with(sub, k2)
                table_rows.append(condition_row("E2 subset", "tilt", prop, rr))
        for key, prop in ((None, "MC"), ("method:unscented", "unscented"), ("method:first_order", "first order"), ("method:none", "none (point)"), ("margin:rocking", "MC, rocking margin")):
            if key is not None and not any(key in r["meta"].get("preds", {}) for r in recs2):
                continue
            rr = rows_with(recs2, key)
            row = condition_row("E2", "tilt", prop, rr)
            ci = ci_of(rr, "ece")
            row["ece_ci"] = [ci[1], ci[2]]
            table_rows.append(row)
        rr = rows_with(recs2, "margin:simulation")
        row = condition_row("E8", "simulation score", "none", rr)
        table_rows.append(row)
        # E6 by stone class on the same placements
        by_class = {}
        for cls in ("flat_river", "blocky_quarry", "rounded_cobble"):
            rc = [r for r in rows_with(recs2) if r["meta"].get("stone_class") == cls]
            if len(rc) >= 3:
                by_class[cls] = condition_row("E6 " + cls, "tilt", "MC", rc)
        extra["E6_by_placed_class"] = by_class
        extra["E2_murphy"] = murphy_decomposition(np.array([r["p_hat"] for r in recs2]), np.array([r["y"] for r in recs2], dtype=float), 5)
    # ------------------------------------------------------------------ E4, E5, E6 runs
    for prefix in ("E4_", "E5_", "E6_"):
        for path in sorted(find_experiments(prefix), key=lambda p: p.name):
            recs, table = load_experiment(path)
            nm = path.name
            if prefix == "E4_":
                label = "E4 kappa " + nm.split("_k")[1].split("_")[0]
            elif prefix == "E5_":
                label = "E5 noise " + nm.split("_ns")[1].split("_")[0] + ", friction sigma " + nm.split("_fs")[1].split("_")[0]
            else:
                label = "E6 " + nm.split("_cls")[1].rsplit("_", 1)[0].replace("_", " ")
            row = condition_row(label, "tilt", "MC", rows_with(recs))
            if not any(np.isfinite(r["m_true"]) and not r["collapsed"] for r in recs):
                # runs with the fixed perturbation test only: no measured tilt margin
                row["rmse_tilt_deg"] = None
                row["coverage_95"] = None
            row["run"] = nm
            table_rows.append(row)
    write_table(table_rows, RESULTS / "tables" / "results_table.csv")
    with open(RESULTS / "tables" / "extra_metrics.json", "w") as f:
        json.dump(extra, f, indent=1, default=float)
    print("wrote", RESULTS / "tables" / "results_table.md")
    # ------------------------------------------------------------------ inventory
    inv = {
        "figures": sorted(p.name for p in FIGURES.glob("*.png")) if FIGURES.exists() else [],
        "stills": sorted(p.name for p in STILLS.glob("*.png")) if STILLS.exists() else [],
        "videos": sorted(p.name for p in VIDEOS.glob("*.mp4")) if VIDEOS.exists() else [],
    }
    with open(RESULTS / "tables" / "inventory.json", "w") as f:
        json.dump(inv, f, indent=1)
    print(json.dumps({k: len(v) for k, v in inv.items()}))


if __name__ == "__main__":
    main()
