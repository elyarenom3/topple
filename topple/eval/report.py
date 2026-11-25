"""Result tables and bootstrap confidence intervals over towers (Section 11.4)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .metrics import summary

COLUMNS = ["condition", "margin", "propagation", "mean_height", "collapse_rate", "rmse_tilt_deg", "coverage_95", "brier", "ece", "ms_per_decision"]


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, stat=np.mean, n_boot: int = 200, rng=None) -> tuple[float, float, float]:
    """Cluster bootstrap over towers: resample towers with replacement, recompute the statistic."""
    rng = np.random.default_rng(0) if rng is None else rng
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: values[groups == g] for g in uniq}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots.append(stat(np.concatenate([members[g] for g in pick])))
    boots = np.array(boots)
    return float(stat(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def condition_row(name: str, margin: str, propagation: str, records: list[dict], n_bins: int = 10) -> dict:
    """Aggregate placement records of one condition into a results table row.

    Each record has keys: tower, p_hat, y, m_hat, m_true, samples (optional), height,
    collapsed, ms_decision.
    """
    if not records:
        return {c: None for c in COLUMNS} | {"condition": name, "margin": margin, "propagation": propagation}
    p = np.array([r["p_hat"] for r in records], dtype=float)
    y = np.array([r["y"] for r in records], dtype=float)
    m_hat = np.array([r.get("m_hat", np.nan) for r in records], dtype=float)
    m_true = np.array([r.get("m_true", np.nan) for r in records], dtype=float)
    has_samples = all(r.get("samples") is not None for r in records)
    samples = np.array([r["samples"] for r in records]) if has_samples else None
    s = summary(p, y, m_hat, m_true, samples, n_bins)
    towers = sorted({r["tower"] for r in records})
    heights = []
    collapses = []
    stones = []
    for t in towers:
        # tower metrics use the placements that built the tower (probes are excluded)
        rt = [r for r in records if r["tower"] == t and not r.get("probe", False)]
        h = [r.get("height", 0.0) for r in rt if np.isfinite(r.get("height", np.nan))]
        heights.append(max(h) if h else 0.0)
        collapses.append(any(r.get("collapsed", False) for r in rt))
        n_ok = 0
        for r in sorted(rt, key=lambda r: r.get("placement", 0)):
            if r.get("collapsed", False):
                break
            n_ok += 1
        stones.append(n_ok)
    ms = [r.get("ms_decision", np.nan) for r in records]
    return {
        "condition": name,
        "margin": margin,
        "propagation": propagation,
        "mean_height": float(np.mean(heights)),
        "mean_stones_before_failure": float(np.mean(stones)),
        "collapse_rate": float(np.mean(collapses)),
        "rmse_tilt_deg": s.get("margin_rmse", np.nan),
        "coverage_95": s.get("coverage_95", np.nan),
        "brier": s["brier"],
        "ece": s["ece_width"],
        "ms_per_decision": float(np.nanmean(ms)),
        "n_placements": len(records),
        "n_probes": int(sum(1 for r in records if r.get("probe", False))),
        "n_towers": len(towers),
        **{k: v for k, v in s.items() if k not in ("brier",)},
    }


def write_table(rows: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = COLUMNS + sorted({k for r in rows for k in r if k not in COLUMNS})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    with open(path.with_suffix(".md"), "w") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "---|" * len(COLUMNS) + "\n")
        for r in rows:
            cells = []
            for c in COLUMNS:
                v = r.get(c)
                cells.append(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else ("" if v is None else str(v)))
            f.write("| " + " | ".join(cells) + " |\n")
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
