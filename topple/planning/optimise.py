"""Derivative free and gradient optimisers for pose refinement (Section 9.2).

``cma_es`` follows the (mu/mu_w, lambda) CMA-ES of Hansen (2016), with the rank one and rank mu
covariance updates and cumulative step size adaptation. ``nelder_mead`` is the simplex method of
Nelder and Mead (1965). All three maximise ``f``.
"""

from __future__ import annotations

import numpy as np


def cma_es(f, x0, sigma0: float = 0.5, iters: int = 100, rng=None, popsize: int | None = None, bounds=None, tol: float = 1e-10):
    """Maximise f(x) by CMA-ES. Returns (x_best, f_best, history)."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x0, dtype=np.float64).copy()
    n = x.size
    lam = popsize or 4 + int(3 * np.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    pc = np.zeros(n)
    ps = np.zeros(n)
    C = np.eye(n)
    sigma = float(sigma0)
    B, D = np.eye(n), np.ones(n)
    x_best, f_best = x.copy(), -np.inf
    history = []
    for it in range(iters):
        z = rng.normal(size=(lam, n))
        y = (z * D) @ B.T
        X = x + sigma * y
        if bounds is not None:
            lo, hi = bounds
            X = np.clip(X, lo, hi)
            y = (X - x) / sigma
        vals = np.array([f(xi) for xi in X])
        order = np.argsort(-vals)
        if vals[order[0]] > f_best:
            f_best, x_best = float(vals[order[0]]), X[order[0]].copy()
        history.append(f_best)
        ysel = y[order[:mu]]
        ymean = w @ ysel
        x = x + sigma * ymean
        Cinv_sqrt_y = B @ ((B.T @ ymean) / D)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt_y
        hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) / chiN < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
        rank_mu = sum(w[i] * np.outer(ysel[i], ysel[i]) for i in range(mu))
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * rank_mu
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        C = 0.5 * (C + C.T)
        evals, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals, 1e-20))
        if sigma * D.max() < tol:
            break
    return x_best, f_best, history


def nelder_mead(f, x0, step: float = 0.1, iters: int = 200, tol: float = 1e-8, bounds=None):
    """Maximise f(x) by the Nelder Mead simplex method. Returns (x_best, f_best, history)."""
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.size
    step = np.broadcast_to(np.asarray(step, dtype=np.float64), (n,))

    def clip(x):
        if bounds is None:
            return x
        return np.clip(x, bounds[0], bounds[1])

    simplex = [clip(x0.copy())]
    for i in range(n):
        p = x0.copy()
        p[i] += step[i]
        simplex.append(clip(p))
    vals = [-f(p) for p in simplex]  # minimise -f
    history = []
    alpha, gamma, rho, sig = 1.0, 2.0, 0.5, 0.5
    for _ in range(iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        history.append(-vals[0])
        if abs(vals[-1] - vals[0]) < tol and np.max([np.linalg.norm(p - simplex[0]) for p in simplex]) < tol:
            break
        c = np.mean(simplex[:-1], axis=0)
        xr = clip(c + alpha * (c - simplex[-1]))
        fr = -f(xr)
        if fr < vals[0]:
            xe = clip(c + gamma * (xr - c))
            fe = -f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc = clip(c + rho * (xr - c))
            else:
                xc = clip(c + rho * (simplex[-1] - c))
            fc = -f(xc)
            if fc < min(fr, vals[-1]):
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = clip(simplex[0] + sig * (simplex[i] - simplex[0]))
                    vals[i] = -f(simplex[i])
    k = int(np.argmin(vals))
    return simplex[k], -vals[k], history


def gradient_ascent(f, grad, x0, step: float = 0.1, iters: int = 100, tol: float = 1e-8, bounds=None):
    """Maximise f by gradient ascent with backtracking. Returns (x_best, f_best, history)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    fx = f(x)
    history = [fx]
    for _ in range(iters):
        g = np.asarray(grad(x), dtype=np.float64)
        if np.linalg.norm(g) < tol:
            break
        s = step
        while s > 1e-12:
            xn = x + s * g
            if bounds is not None:
                xn = np.clip(xn, bounds[0], bounds[1])
            fn = f(xn)
            if fn > fx:
                x, fx = xn, fn
                break
            s *= 0.5
        else:
            break
        history.append(fx)
    return x, fx, history
