import numpy as np

from topple.eval.metrics import brier_score, expected_calibration_error, interval_coverage, margin_errors, murphy_decomposition, pit_values, reliability_diagram
from topple.eval.report import bootstrap_ci, condition_row, write_table
from topple.eval.sobol import saltelli_design, sobol_indices


def test_calibrated_forecasts_have_small_ece():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.03
    over = np.clip(p * 1.4, 0, 1)
    assert expected_calibration_error(over, y) > 0.1
    d = murphy_decomposition(p, y)
    assert abs(d["reliability"] - d["resolution"] + d["uncertainty"] - d["brier"]) < 0.01


def test_brier_bounds():
    assert brier_score([1, 0], [1, 0]) == 0
    assert brier_score([0, 1], [1, 0]) == 1


def test_coverage_and_pit():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(1000, 300))
    t = rng.normal(size=1000)
    cov = interval_coverage(s, t)
    assert abs(cov[0.5] - 0.5) < 0.05 and abs(cov[0.95] - 0.95) < 0.03
    pit = pit_values(s, t)
    h, _ = np.histogram(pit, bins=5, range=(0, 1))
    assert h.min() > 150


def test_reliability_diagram_bootstrap_bands():
    rng = np.random.default_rng(2)
    p = rng.random(400)
    y = (rng.random(400) < p).astype(float)
    groups = np.repeat(np.arange(40), 10)
    d = reliability_diagram(p, y, n_bins=5, groups=groups, n_boot=50)
    m = np.isfinite(d["ybar"])
    assert np.all(d["lo"][m] <= d["ybar"][m] + 1e-9) and np.all(d["hi"][m] >= d["ybar"][m] - 1e-9)


def test_sobol_recovers_linear_model():
    rng = np.random.default_rng(3)
    d = saltelli_design(4000, 3, rng)
    f = lambda X: 2 * X[:, 0] + X[:, 1]
    r = sobol_indices(f(d["A"]), f(d["B"]), np.array([f(d["AB"][i]) for i in range(3)]), n_boot=20)
    assert abs(r["S"][0] - 0.8) < 0.05 and abs(r["S"][1] - 0.2) < 0.05 and abs(r["S"][2]) < 0.05


def test_report_row_and_table(tmp_path):
    rng = np.random.default_rng(4)
    recs = []
    for t in range(5):
        for k in range(4):
            p = rng.random()
            recs.append({"tower": t, "p_hat": p, "y": float(rng.random() < p), "m_hat": 10 + rng.normal(), "m_true": 10 + rng.normal(), "samples": 10 + rng.normal(size=20), "height": 0.1 * (k + 1), "collapsed": k == 3 and t % 2 == 0, "ms_decision": 5.0})
    row = condition_row("E2", "tilt", "mc", recs)
    assert row["n_placements"] == 20 and 0 <= row["ece"] <= 1 and row["mean_height"] > 0
    write_table([row], tmp_path / "table.csv")
    assert (tmp_path / "table.md").exists()
    m, lo, hi = bootstrap_ci(np.array([r["m_hat"] for r in recs]), np.array([r["tower"] for r in recs]), n_boot=50)
    assert lo <= m <= hi
