#!/usr/bin/env python3
"""Result figures from experiment records (Section 11.5), one command per figure.

F7   predicted compared with measured tilt margin (oracle and perception runs)
F8   reliability diagrams of the survival probability for every propagation method (E2) and the
     simulation score (E8) on the same placements, with cluster bootstrap bands over towers
F10  tower height and collapse rate against the risk weight kappa (E4)
F11  calibration error over depth noise scale and friction prior width (E5)
F12a margin error against friction cone sides N (E7)
F12b the three margin definitions against their dynamic tests (E1)

python scripts/fig_results.py F8
"""

from __future__ import annotations

import numpy as np

from _common import FIGURES, find_experiments, load_experiment, out_dir, parser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from topple.eval.figures import COLUMN_WIDTH_IN, heatmap, lines, margin_scatter, palette, reliability_figure, save  # noqa: E402
from topple.eval.metrics import expected_calibration_error, margin_errors, reliability_diagram  # noqa: E402

TAU = 3.0


def latest(prefix):
    paths = find_experiments(prefix)
    return load_experiment(paths[-1]) if paths else (None, None)


def pred_arrays(rows, key=None):
    """(p_hat, m_hat, lo, hi) arrays for the base prediction or for meta['preds'][key]."""
    p, m, lo, hi = [], [], [], []
    for r in rows:
        if key is None:
            p.append(r["p_hat"])
            m.append(r["m_hat"])
            s = np.asarray(r["samples"])
            lo.append(np.quantile(s, 0.025))
            hi.append(np.quantile(s, 0.975))
        else:
            d = r["meta"].get("preds", {}).get(key)
            if d is None or "mean" not in d:
                p.append(np.nan)
                m.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            m.append(d["mean"])
            lo.append(d["lo"])
            hi.append(d["hi"])
            p.append(d["p_hat"] if not key.startswith("margin:simulation") else float(d["mean"] > TAU))
    return np.array(p, dtype=float), np.array(m, dtype=float), np.array(lo, dtype=float), np.array(hi, dtype=float)


def f7(args):
    for prefix, label in (("E2all", "perception, Monte Carlo"), ("E1all", "exact inputs")):
        rows, _ = latest(prefix)
        if not rows:
            continue
        ok = [r for r in rows if np.isfinite(r["m_true"]) and not r["collapsed"]]
        p, m, lo, hi = pred_arrays(ok)
        mt = np.array([r["m_true"] for r in ok])
        e = margin_errors(m, mt)
        path = margin_scatter(m, mt, out_dir(args, FIGURES) / f"F7_margin_scatter_{prefix}", lo, hi, title=f"{label}: RMSE {e['rmse']:.1f} deg, Spearman {e['spearman']:.2f}")
        print("wrote", path)
        # the rocking aware margin on the same placements
        p4, m4, lo4, hi4 = pred_arrays(ok, "margin:rocking")
        k = np.isfinite(m4)
        if k.sum() >= 5:
            e4 = margin_errors(m4[k], mt[k])
            path = margin_scatter(m4[k], mt[k], out_dir(args, FIGURES) / f"F7_margin_scatter_{prefix}_rocking", lo4[k], hi4[k], title=f"{label}, rocking margin: RMSE {e4['rmse']:.1f} deg, Spearman {e4['spearman']:.2f}")
            print("wrote", path)


def f8(args):
    rows, _ = latest("E2all")
    if not rows:
        print("no E2all records")
        return
    ut_rows, _ = latest("E2ut")
    has_ut = any("mean" in r["meta"].get("preds", {}).get("method:unscented", {}) for r in rows)
    if ut_rows and not has_ut:
        key = lambda r: (r["tower"], r["placement"], r["stone"], bool(r["meta"].get("probe")), tuple(np.round(np.asarray(r["meta"]["pose"])[:3, 3], 4)))
        ut = {key(r): r["meta"].get("preds", {}).get("method:unscented") for r in ut_rows}
        for r in rows:
            r["meta"].setdefault("preds", {}).pop("method:unscented", None)
            if ut.get(key(r)) is not None:
                r["meta"]["preds"]["method:unscented"] = ut[key(r)]
    y = np.array([r["y"] for r in rows], dtype=float)
    g = np.array([r["tower"] for r in rows])
    diagrams = {}
    for key, label in ((None, "Monte Carlo"), ("method:unscented", "unscented"), ("method:first_order", "first order"), ("method:none", "point estimate"), ("margin:simulation", "simulation score"), ("margin:rocking", "rocking margin (MC)")):
        p, *_ = pred_arrays(rows, key)
        ok = np.isfinite(p)
        if ok.sum() < 5:
            continue
        diagrams[label] = reliability_diagram(p[ok], y[ok], n_bins=5, groups=g[ok], n_boot=200)
    rows1, _ = latest("E1all")
    if rows1:
        p1 = np.array([r["p_hat"] for r in rows1])
        y1 = np.array([r["y"] for r in rows1], dtype=float)
        diagrams["exact inputs"] = reliability_diagram(p1, y1, n_bins=5, groups=np.array([r["tower"] for r in rows1]), n_boot=200)
    path = reliability_figure(diagrams, out_dir(args, FIGURES) / "F8_reliability", title="Reliability of the predicted survival probability")
    print("wrote", path)
    # probability integral transform of the measured margin under the Monte Carlo samples
    from topple.eval.metrics import pit_values

    ok = [r for r in rows if np.isfinite(r["m_true"])]
    S = np.array([np.asarray(r["samples"]) for r in ok], dtype=object)
    pit = np.array([np.mean(np.asarray(s) < r["m_true"]) + 0.5 * np.mean(np.asarray(s) == r["m_true"]) for s, r in zip(S, ok)])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.7 * COLUMN_WIDTH_IN))
    ax.hist(pit, bins=10, range=(0, 1), color=palette(1)[0], edgecolor="white")
    ax.axhline(len(pit) / 10, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("PIT of the measured tilt margin")
    ax.set_ylabel("placements")
    ax.set_title("A flat histogram means calibrated margin samples")
    print("wrote", save(fig, out_dir(args, FIGURES) / "F8b_pit"))


def f10(args):
    pts = []
    for path in find_experiments("E4_"):
        rows, table = load_experiment(path)
        if not table:
            continue
        k = float(path.name.split("_k")[1].split("_")[0])
        pts.append((k, table["mean_height"], table["collapse_rate"], table.get("mean_stones_before_failure", np.nan)))
    if not pts:
        print("no E4 records")
        return
    pts.sort()
    xs = np.array([p[0] for p in pts])
    fig, ax1 = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.75 * COLUMN_WIDTH_IN))
    c = palette(2)
    ax1.plot(xs, [100 * p[1] for p in pts], "o-", color=c[0], ms=3, label="mean height (cm)")
    ax1.set_xlabel("risk weight kappa")
    ax1.set_ylabel("mean tower height (cm)", color=c[0])
    ax1.set_ylim(0, None)
    ax2 = ax1.twinx()
    ax2.plot(xs, [p[2] for p in pts], "s--", color=c[1], ms=3, label="collapse rate")
    ax2.set_ylabel("collapse rate", color=c[1])
    ax2.set_ylim(0, 1)
    ax1.set_title("Height and collapse rate against kappa")
    print("wrote", save(fig, out_dir(args, FIGURES) / "F10_kappa"))


def f11(args):
    pts = {}
    for path in find_experiments("E5_"):
        rows, table = load_experiment(path)
        if not table:
            continue
        ns = float(path.name.split("_ns")[1].split("_")[0])
        fs = float(path.name.split("_fs")[1].split("_")[0])
        pts[(ns, fs)] = (table["ece"], table["brier"])
    base_rows, base_table = latest("E2all")
    if base_table:
        pts.setdefault((1.0, 0.25), (base_table["ece"], base_table["brier"]))
    if not pts:
        print("no E5 records")
        return
    xs = sorted({k[0] for k in pts})
    ys = sorted({k[1] for k in pts})
    if len(xs) > 1 and len(ys) > 1 and len(pts) == len(xs) * len(ys):
        Z = np.full((len(ys), len(xs)), np.nan)
        for (ns, fs), v in pts.items():
            Z[ys.index(fs), xs.index(ns)] = v[0]
        print("wrote", heatmap(Z, xs, ys, out_dir(args, FIGURES) / "F11_sensitivity", xlabel="depth noise scale", ylabel="friction prior log sigma", title="ECE over noise and friction prior width", cbar_label="ECE"))
        return
    # cross design: two panels through the base point
    fig, axes = plt.subplots(1, 2, figsize=(2 * COLUMN_WIDTH_IN, 0.75 * COLUMN_WIDTH_IN))
    c = palette(2)
    a = sorted([(ns, v) for (ns, fs), v in pts.items() if abs(fs - 0.25) < 1e-9])
    b = sorted([(fs, v) for (ns, fs), v in pts.items() if abs(ns - 1.0) < 1e-9])
    for ax, data, xl in ((axes[0], a, "depth noise scale"), (axes[1], b, "friction prior log sigma")):
        if data:
            ax.plot([d[0] for d in data], [d[1][0] for d in data], "o-", color=c[0], ms=3, label="ECE")
            ax.plot([d[0] for d in data], [d[1][1] for d in data], "s--", color=c[1], ms=3, label="Brier")
        ax.set_xlabel(xl)
        ax.set_ylim(0, None)
        ax.legend(frameon=False)
    axes[0].set_ylabel("score")
    fig.suptitle("Calibration against perception noise and friction prior width", fontsize=9)
    fig.tight_layout()
    print("wrote", save(fig, out_dir(args, FIGURES) / "F11_sensitivity"))


def f12(args):
    rows, _ = latest("E1all")
    if not rows:
        print("no E1all records")
        return
    ok = [r for r in rows if np.isfinite(r["m_true"]) and not r["collapsed"]]
    mt = np.array([r["m_true"] for r in ok])
    # E7: cone sides
    Ns, rm, bias = [], [], []
    for N in (4, 8, 16, 32):
        if N == 8:
            m = np.array([r["m_hat"] for r in ok])
        else:
            _, m, _, _ = pred_arrays(ok, f"cone:{N}")
        e = margin_errors(m, mt)
        Ns.append(N)
        rm.append(e["rmse"])
        bias.append(e["bias"])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.75 * COLUMN_WIDTH_IN))
    c = palette(2)
    ax.plot(Ns, rm, "o-", color=c[0], ms=3, label="RMSE")
    ax.plot(Ns, bias, "s--", color=c[1], ms=3, label="bias")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("friction cone sides N")
    ax.set_ylabel("tilt margin error (deg)")
    ax.set_title("Static margin error against the cone discretisation")
    ax.legend(frameon=False)
    print("wrote", save(fig, out_dir(args, FIGURES) / "F12a_cone_sides"))
    # E1: margin definitions against their tests
    fig, axes = plt.subplots(1, 3, figsize=(2.2 * COLUMN_WIDTH_IN, 0.75 * COLUMN_WIDTH_IN))
    tilt = np.array([r["m_hat"] for r in ok])
    pt = np.array([r.get("push_true", np.nan) for r in ok], dtype=float)
    _, push, _, _ = pred_arrays(ok, "margin:push")
    _, fc, _, _ = pred_arrays(ok, "margin:fc")
    from topple.eval.metrics import spearman

    for ax, x, yv, xl, yl in ((axes[0], tilt, mt, "tilt margin (deg)", "critical tilt (deg)"), (axes[1], push, pt, "push margin (x weight)", "push test (x weight)"), (axes[2], fc, mt, "support quality radius", "critical tilt (deg)")):
        m = np.isfinite(x) & np.isfinite(yv)
        ax.scatter(x[m], yv[m], s=8, color=c[0])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"Spearman {spearman(x[m], yv[m]):.2f}")
    fig.tight_layout()
    print("wrote", save(fig, out_dir(args, FIGURES) / "F12b_margin_definitions"))


if __name__ == "__main__":
    ap = parser("result figures")
    ap.add_argument("figure", choices=["F7", "F8", "F10", "F11", "F12", "all"])
    args = ap.parse_args()
    fns = {"F7": f7, "F8": f8, "F10": f10, "F11": f11, "F12": f12}
    for k in (fns if args.figure == "all" else [args.figure]):
        fns[k](args)
