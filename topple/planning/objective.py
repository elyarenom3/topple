"""Placement objective and chance constraint (Section 9.2)."""

from __future__ import annotations

import numpy as np


def objective(margin_samples, dh: float, kappa: float = 1.0, lambda_h: float = 2.0) -> float:
    """J = E[M] - kappa Std[M] + lambda_h dh."""
    m = np.asarray(margin_samples, dtype=np.float64).reshape(-1)
    if m.size == 0:
        return -np.inf
    std = float(np.std(m, ddof=1)) if m.size > 1 else 0.0
    return float(np.mean(m) - kappa * std + lambda_h * dh)


def chance_ok(p_hat: float, delta: float = 0.05) -> bool:
    """Chance constraint p_hat >= 1 - delta."""
    return bool(p_hat >= 1.0 - delta - 1e-12)
