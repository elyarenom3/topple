"""Calibration metrics and scoring rules (Section 11.2).

* Brier score (Brier 1950) with Murphy's (1973) decomposition into reliability,
  resolution and uncertainty.
* Expected and maximum calibration error with equal width and equal mass bins
  (Guo, Pleiss, Sun and Weinberger 2017).
* Reliability diagrams (Niculescu-Mizil and Caruana 2005) with bootstrap bands.
* Interval coverage and the probability integral transform histogram.
* Margin error bias, RMSE and Spearman correlation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def _bin_edges(p: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme == "equal_mass":
        qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        qs[0], qs[-1] = 0.0, 1.0
        return np.unique(qs)
    raise ValueError(scheme)


def _assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    b = np.searchsorted(edges, p, side="right") - 1
    return np.clip(b, 0, len(edges) - 2)


def murphy_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10, scheme: str = "equal_width") -> dict:
    """Brier = reliability - resolution + uncertainty (Murphy 1973), with binned forecasts."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    edges = _bin_edges(p, n_bins, scheme)
    b = _assign_bins(p, edges)
    ybar = y.mean()
    rel = 0.0
    res = 0.0
    for k in range(len(edges) - 1):
        m = b == k
        nk = int(m.sum())
        if nk == 0:
            continue
        pk = p[m].mean()
        yk = y[m].mean()
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - ybar) ** 2
    unc = ybar * (1 - ybar)
    return {"brier": brier_score(p, y), "reliability": rel / n, "resolution": res / n, "uncertainty": float(unc), "bins": len(edges) - 1}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10, scheme: str = "equal_width") -> float:
    """ECE = sum_m |B_m| / n * |ybar(B_m) - pbar(B_m)|."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")
    edges = _bin_edges(p, n_bins, scheme)
    b = _assign_bins(p, edges)
    ece = 0.0
    for k in range(len(edges) - 1):
        m = b == k
        nk = int(m.sum())
        if nk == 0:
            continue
        ece += nk / n * abs(y[m].mean() - p[m].mean())
    return float(ece)


def maximum_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10, scheme: str = "equal_width") -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = _bin_edges(p, n_bins, scheme)
    b = _assign_bins(p, edges)
    mce = 0.0
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() == 0:
            continue
        mce = max(mce, abs(y[m].mean() - p[m].mean()))
    return float(mce)


def reliability_diagram(p: np.ndarray, y: np.ndarray, n_bins: int = 10, scheme: str = "equal_width", n_boot: int = 200, rng=None, groups=None) -> dict:
    """Binned observed frequency against predicted probability with bootstrap bands.

    ``groups`` (tower index per placement) selects a cluster bootstrap over towers, since
    placements within a tower are correlated (Section 11.3).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = _bin_edges(p, n_bins, scheme)
    b = _assign_bins(p, edges)
    nb = len(edges) - 1
    counts = np.array([int((b == k).sum()) for k in range(nb)])
    pbar = np.array([p[b == k].mean() if counts[k] else np.nan for k in range(nb)])
    ybar = np.array([y[b == k].mean() if counts[k] else np.nan for k in range(nb)])
    boots = np.full((n_boot, nb), np.nan)
    idx_sets = _bootstrap_indices(len(p), n_boot, rng, groups)
    for i, idx in enumerate(idx_sets):
        bb = b[idx]
        yy = y[idx]
        for k in range(nb):
            m = bb == k
            if m.any():
                boots[i, k] = yy[m].mean()
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return {"edges": edges, "counts": counts, "pbar": pbar, "ybar": ybar, "lo": lo, "hi": hi, "ece": expected_calibration_error(p, y, n_bins, scheme)}


def _bootstrap_indices(n: int, n_boot: int, rng, groups=None):
    out = []
    if groups is None:
        for _ in range(n_boot):
            out.append(rng.integers(0, n, size=n))
        return out
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        out.append(np.concatenate([members[g] for g in pick]))
    return out


def interval_coverage(samples: np.ndarray, truth: np.ndarray, levels=(0.5, 0.8, 0.95)) -> dict:
    """Fraction of truths inside the central posterior interval at each level.

    ``samples`` has shape (n_cases, n_samples).
    """
    s = np.asarray(samples, dtype=float)
    t = np.asarray(truth, dtype=float)
    out = {}
    for lv in levels:
        lo = np.quantile(s, 0.5 - lv / 2, axis=1)
        hi = np.quantile(s, 0.5 + lv / 2, axis=1)
        out[lv] = float(np.mean((t >= lo) & (t <= hi)))
    return out


def pit_values(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Probability integral transform: rank of the truth among the posterior samples in [0, 1]."""
    s = np.asarray(samples, dtype=float)
    t = np.asarray(truth, dtype=float)[:, None]
    # randomised tie breaking is unnecessary for continuous margins; midrank for ties
    below = (s < t).mean(axis=1)
    equal = (s == t).mean(axis=1)
    return below + 0.5 * equal


def pit_uniformity(pit: np.ndarray) -> dict:
    """Kolmogorov Smirnov statistic of the PIT values against U(0, 1)."""
    pit = np.asarray(pit, dtype=float)
    ks = stats.kstest(pit, "uniform")
    return {"ks_stat": float(ks.statistic), "p_value": float(ks.pvalue)}


def margin_errors(m_hat: np.ndarray, m_true: np.ndarray) -> dict:
    m_hat = np.asarray(m_hat, dtype=float)
    m_true = np.asarray(m_true, dtype=float)
    ok = np.isfinite(m_hat) & np.isfinite(m_true)
    e = m_hat[ok] - m_true[ok]
    return {"bias": float(e.mean()) if ok.any() else float("nan"), "rmse": float(np.sqrt((e ** 2).mean())) if ok.any() else float("nan"), "mae": float(np.abs(e).mean()) if ok.any() else float("nan"), "spearman": spearman(m_hat[ok], m_true[ok]), "n": int(ok.sum())}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def summary(p: np.ndarray, y: np.ndarray, m_hat=None, m_true=None, samples=None, n_bins: int = 10) -> dict:
    """All scalar metrics of Section 11.2 in one dict."""
    out = {
        "n": int(len(p)),
        "brier": brier_score(p, y),
        "ece_width": expected_calibration_error(p, y, n_bins, "equal_width"),
        "ece_mass": expected_calibration_error(p, y, n_bins, "equal_mass"),
        "mce": maximum_calibration_error(p, y, n_bins, "equal_width"),
    }
    out.update({f"murphy_{k}": v for k, v in murphy_decomposition(p, y, n_bins).items() if k != "brier"})
    if m_hat is not None and m_true is not None:
        out.update({f"margin_{k}": v for k, v in margin_errors(m_hat, m_true).items()})
    if samples is not None and m_true is not None:
        cov = interval_coverage(samples, m_true)
        out.update({f"coverage_{int(k * 100)}": v for k, v in cov.items()})
        out.update({f"pit_{k}": v for k, v in pit_uniformity(pit_values(samples, m_true)).items()})
    return out
