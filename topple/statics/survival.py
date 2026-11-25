"""Probability of survival from margin samples (Section 7.8)."""

from __future__ import annotations

import numpy as np


def survival_probability(margins, tau_deg: float = 3.0) -> float:
    """p_hat = fraction of margin samples above the test perturbation tau."""
    m = np.asarray(margins, dtype=np.float64).reshape(-1)
    if m.size == 0:
        return 0.0
    return float(np.mean(m > tau_deg))


def margin_summary(margins, levels=(0.5, 0.8, 0.95)) -> dict:
    """Mean, standard deviation and central intervals of the margin samples."""
    m = np.asarray(margins, dtype=np.float64).reshape(-1)
    out = {"mean": float(np.mean(m)), "std": float(np.std(m, ddof=1)) if m.size > 1 else 0.0, "n": int(m.size), "median": float(np.median(m))}
    for lv in levels:
        lo, hi = np.quantile(m, [(1 - lv) / 2, 1 - (1 - lv) / 2])
        out[f"q{int(round(lv * 100))}"] = (float(lo), float(hi))
    return out
