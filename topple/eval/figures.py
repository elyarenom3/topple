"""2D figure helpers following the figure standards of Section 10.10.

All plots share the colour map of the 3D overlays (``topple.render.color.colormap`` when
available, else a matplotlib fallback), vector PDF output, and font sizes legible at
column width (Rougier, Droettboom and Bourne 2014).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMN_WIDTH_IN = 3.4
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def _encoded_table(name: str = "sequential", n: int = 256) -> np.ndarray:
    """The overlay colour map of the renderer (linear sRGB) encoded for display."""
    from ..render.color import colormap, srgb_encode

    lin = colormap(np.linspace(0.0, 1.0, n), name, 0.0, 1.0)
    return np.clip(srgb_encode(lin), 0.0, 1.0)


def shared_cmap(name: str = "sequential"):
    """Matplotlib colour map built from the renderer's table, so 2D plots and 3D overlays match."""
    from matplotlib.colors import ListedColormap

    try:
        return ListedColormap(_encoded_table(name), name=f"topple_{name}")
    except Exception:
        return plt.get_cmap("viridis")


def palette(n: int):
    """n categorical colours sampled from the shared sequential map (avoiding its darkest end)."""
    try:
        tab = _encoded_table("sequential", 256)
        idx = np.linspace(40, 215, max(n, 1)).astype(int)
        return [tuple(tab[i]) for i in idx]
    except Exception:
        cmap = plt.get_cmap("viridis")
        return [cmap((i + 0.5) / n) for i in range(n)]


def save(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def reliability_figure(diagrams: dict[str, dict], path, title: str = ""):
    """diagrams: name -> output of eval.metrics.reliability_diagram."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN))
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--", label="perfect calibration")
    cols = palette(max(len(diagrams), 1))
    for (name, d), c in zip(diagrams.items(), cols):
        m = np.isfinite(d["ybar"])
        ax.fill_between(d["pbar"][m], d["lo"][m], d["hi"][m], color=c, alpha=0.2, lw=0)
        ax.plot(d["pbar"][m], d["ybar"][m], "o-", color=c, ms=3, lw=1, label=f"{name} (ECE {d['ece']:.3f})")
    ax.set_xlabel("predicted probability of survival")
    ax.set_ylabel("observed survival frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    return save(fig, path)


def margin_scatter(m_hat, m_true, path, lo=None, hi=None, xlabel="predicted tilt margin (deg)", ylabel="measured critical tilt (deg)", title=""):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN))
    m_hat = np.asarray(m_hat)
    m_true = np.asarray(m_true)
    c = palette(3)
    if lo is not None and hi is not None:
        ax.errorbar(m_hat, m_true, xerr=[np.maximum(m_hat - lo, 0), np.maximum(hi - m_hat, 0)], fmt="none", ecolor=c[1], alpha=0.4, lw=0.6)
    ax.scatter(m_hat, m_true, s=8, color=c[0], alpha=0.8)
    lim = [0, max(np.nanmax(m_hat) if len(m_hat) else 1, np.nanmax(m_true) if len(m_true) else 1) * 1.05]
    ax.plot(lim, lim, color="0.5", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return save(fig, path)


def sobol_bars(names, S, ST, path, S_ci=None, ST_ci=None, title="First order and total Sobol indices"):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.7 * COLUMN_WIDTH_IN))
    x = np.arange(len(names))
    c = palette(2)
    err_s = None if S_ci is None else np.abs(np.asarray(S_ci) - np.asarray(S))
    err_t = None if ST_ci is None else np.abs(np.asarray(ST_ci) - np.asarray(ST))
    ax.bar(x - 0.2, S, 0.4, color=c[0], label="first order", yerr=err_s, capsize=2)
    ax.bar(x + 0.2, ST, 0.4, color=c[1], label="total", yerr=err_t, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Sobol index")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False)
    return save(fig, path)


def lines(xs, series: dict[str, np.ndarray], path, xlabel="", ylabel="", title="", bands: dict | None = None):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.75 * COLUMN_WIDTH_IN))
    cols = palette(max(len(series), 1))
    for (name, ys), c in zip(series.items(), cols):
        ax.plot(xs, ys, "o-", ms=3, lw=1, color=c, label=name)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(xs, lo, hi, color=c, alpha=0.2, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(frameon=False)
    return save(fig, path)


def histogram(values, path, bins=10, xlabel="", title="", range_=(0, 1)):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.7 * COLUMN_WIDTH_IN))
    ax.hist(values, bins=bins, range=range_, color=palette(1)[0], edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    return save(fig, path)


def heatmap(Z, xs, ys, path, xlabel="", ylabel="", title="", cbar_label=""):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.8 * COLUMN_WIDTH_IN))
    cmap = shared_cmap("sequential")
    im = ax.imshow(np.asarray(Z), origin="lower", aspect="auto", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap=cmap)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return save(fig, path)
