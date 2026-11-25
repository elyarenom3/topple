"""Friction coefficient prior and grid posterior (Section 8.7).

Prior: log normal with median 0.6 and log standard deviation 0.25 per stone pair. Optional
measurement: tilt the stone on a plate, record the slip angle, update a grid posterior with the
likelihood of the observed angle given mu equal to tan of the slip angle plus noise.
"""

from __future__ import annotations

import numpy as np


class FrictionPrior:
    def __init__(self, median: float = 0.6, log_sigma: float = 0.25):
        self.median = float(median)
        self.log_sigma = float(log_sigma)

    def sample(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray | float:
        return np.exp(np.log(self.median) + self.log_sigma * rng.normal(size=n))

    def logpdf(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        z = (np.log(mu) - np.log(self.median)) / self.log_sigma
        return -0.5 * z * z - np.log(mu * self.log_sigma * np.sqrt(2 * np.pi))

    @property
    def mean(self) -> float:
        return self.median * np.exp(0.5 * self.log_sigma ** 2)


class GridPosterior:
    """Posterior over mu on a grid, updated from slip angle observations."""

    def __init__(self, prior: FrictionPrior, mu_min: float = 0.1, mu_max: float = 2.0, n: int = 400):
        self.grid = np.linspace(mu_min, mu_max, n)
        self.logp = prior.logpdf(self.grid)
        self.logp -= np.max(self.logp)
        self.n_obs = 0

    def update(self, slip_angle_obs: float, noise_deg: float = 1.0) -> None:
        """Observed slip angle alpha_obs ~ N(atan(mu), noise)."""
        sig = np.deg2rad(noise_deg)
        pred = np.arctan(self.grid)
        self.logp += -0.5 * ((slip_angle_obs - pred) / sig) ** 2
        self.logp -= np.max(self.logp)
        self.n_obs += 1

    def pdf(self) -> np.ndarray:
        p = np.exp(self.logp)
        return p / np.trapezoid(p, self.grid)

    def mean(self) -> float:
        return float(np.trapezoid(self.grid * self.pdf(), self.grid))

    def std(self) -> float:
        m = self.mean()
        return float(np.sqrt(np.trapezoid((self.grid - m) ** 2 * self.pdf(), self.grid)))

    def map(self) -> float:
        return float(self.grid[int(np.argmax(self.logp))])

    def sample(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray | float:
        p = self.pdf()
        c = np.cumsum(p)
        c /= c[-1]
        u = rng.random(size=n)
        return np.interp(u, c, self.grid)


def simulate_slip_angle(mu_true: float, rng: np.random.Generator, noise_deg: float = 1.0) -> float:
    """Tilt plate experiment in closed form: slip at tan(alpha) = mu, plus angle noise."""
    return float(np.arctan(mu_true) + np.deg2rad(noise_deg) * rng.normal())
