"""Uncertainty propagation (Section 8.8): Monte Carlo, unscented transform, first order."""

from __future__ import annotations

import numpy as np


def monte_carlo(fn, samples) -> np.ndarray:
    """Evaluate fn on each sample; returns an array of outputs."""
    return np.asarray([fn(s) for s in samples], dtype=np.float64)


def bootstrap_se(values: np.ndarray, rng: np.random.Generator, n_boot: int = 200) -> float:
    """Bootstrap standard error of the mean, used to check Monte Carlo convergence."""
    v = np.asarray(values, dtype=np.float64)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.std(means))


def unscented(fn, mean, cov, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
    """Unscented transform (Julier and Uhlmann 1997) with 2n + 1 sigma points.

    Returns (mean, variance, sigma_points, values) for a scalar valued fn.
    """
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    P = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    n = m.size
    lam = alpha * alpha * (n + kappa) - n
    try:
        L = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh((n + lam) * P)
        L = U @ np.diag(np.sqrt(np.maximum(w, 0)))
    pts = [m]
    for i in range(n):
        pts.append(m + L[:, i])
        pts.append(m - L[:, i])
    pts = np.array(pts)
    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha * alpha + beta)
    vals = np.array([float(fn(p)) for p in pts])
    mu = float(Wm @ vals)
    var = float(Wc @ (vals - mu) ** 2)
    return mu, var, pts, vals


def first_order(fn, grad_fn, mean, cov):
    """First order propagation: mean fn(mean), variance J Sigma J^T."""
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    P = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    J = np.asarray(grad_fn(m), dtype=np.float64).reshape(1, -1)
    var = float(J @ P @ J.T)
    return float(fn(m)), var


def finite_difference_gradient(fn, x, h: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (fn(x + e) - fn(x - e)) / (2 * h)
    return g
