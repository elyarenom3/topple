"""Saltelli design and first order Sobol indices (Section 11.3, condition E3).

The design follows Saltelli (2002): two independent sample matrices A and B of size
(n, d) and the d matrices AB_i that take column i from B. First order indices use the
Jansen (1999) estimator, total order indices the Jansen estimator too. Bootstrap over
rows gives confidence intervals.
"""

from __future__ import annotations

import numpy as np


def saltelli_design(n: int, d: int, rng: np.random.Generator, sampler=None) -> dict:
    """Return dict(A, B, AB) with AB of shape (d, n, d). Samples are uniform on [0, 1)."""
    if sampler is None:
        A = rng.random((n, d))
        B = rng.random((n, d))
    else:
        A = sampler(n, d, rng)
        B = sampler(n, d, rng)
    AB = np.empty((d, n, d))
    for i in range(d):
        AB[i] = A
        AB[i][:, i] = B[:, i]
    return {"A": A, "B": B, "AB": AB}


def sobol_indices(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray, n_boot: int = 200, rng=None) -> dict:
    """First order S_i and total order ST_i from model outputs on the design.

    fA, fB : (n,) outputs on A and B; fAB : (d, n) outputs on AB_i.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    fA = np.asarray(fA, dtype=float)
    fB = np.asarray(fB, dtype=float)
    fAB = np.asarray(fAB, dtype=float)
    d, n = fAB.shape

    def est(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        var = np.var(np.concatenate([a, b]))
        if var <= 0:
            return np.zeros(d), np.zeros(d)
        S = np.array([(var - 0.5 * np.mean((b - ab[i]) ** 2)) / var for i in range(d)])
        ST = np.array([0.5 * np.mean((a - ab[i]) ** 2) / var for i in range(d)])
        return S, ST

    S, ST = est(np.arange(n))
    Sb = np.empty((n_boot, d))
    STb = np.empty((n_boot, d))
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Sb[k], STb[k] = est(idx)
    return {
        "S": S,
        "ST": ST,
        "S_ci": np.percentile(Sb, [2.5, 97.5], axis=0),
        "ST_ci": np.percentile(STb, [2.5, 97.5], axis=0),
    }


def one_at_a_time(fn, baseline: dict, variations: dict, rng: np.random.Generator, n: int = 64) -> dict:
    """Variance of fn when one uncertainty source at a time is sampled and the rest held at
    their means (the "one source at a time" part of E3). ``variations[name]`` is a callable
    rng -> value for that input."""
    out = {}
    for name, sampler in variations.items():
        vals = []
        for _ in range(n):
            kw = dict(baseline)
            kw[name] = sampler(rng)
            vals.append(fn(**kw))
        out[name] = float(np.var(vals))
    return out
