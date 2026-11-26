"""Learned recalibration of the Monte Carlo survival probability.

Monte Carlo propagation states a survival probability from the posterior alone. The recalibrator
maps its outputs, with the context of the placement, to a corrected probability learned from the
labelled outcomes of earlier experiments:

    p_final = g(p_mc, margin mean, lo, hi, sd, level, height, contacts, stone class, settings)

``g`` is a small gradient boosted tree classifier (the final method) or a logistic regression on
the Monte Carlo outputs alone (the simpler variant). Records are the dictionaries written by the
experiment runner, with the experiment settings under ``record["settings"]``.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

CLASSES = ("flat_river", "blocky_quarry", "rounded_cobble")
MC_FEATURES = ("p_hat", "m_hat", "m_lo", "m_hi", "m_sd")
CONTEXT_FEATURES = ("level", "height", "n_contacts") + tuple(f"class_{c}" for c in CLASSES) + ("noise_scale", "friction_log_sigma", "kappa", "delta")
DEFAULT_SETTINGS = {"kappa": 1.0, "noise_scale": 1.0, "friction_log_sigma": 0.25, "delta": 0.05}


def settings_from_name(name: str) -> dict:
    """Experiment settings from a results directory name such as
    ``E5_propmc_mtilt_k1.0_N8_ns2.0_fs0.25_d0.05_clsall_<hash>``."""

    def num(tag, default):
        m = re.search(rf"_{tag}([0-9.]+?)_", name)
        return float(m.group(1)) if m else default

    return {"kappa": num("k", 1.0), "noise_scale": num("ns", 1.0), "friction_log_sigma": num("fs", 0.25), "delta": num("d", 0.05)}


def features(record: dict, context: bool = True) -> list[float]:
    smp = np.asarray(record.get("samples", []), dtype=float)
    f = [float(record["p_hat"]), float(record["m_hat"]), float(record["m_lo"]), float(record["m_hi"]), float(smp.std()) if smp.size else 0.0]
    if context:
        meta = record.get("meta", {})
        s = {**DEFAULT_SETTINGS, **record.get("settings", {})}
        cls = meta.get("stone_class", "")
        f += [float(record["placement"]), float(record["height"]), float(meta.get("n_contacts", np.nan))]
        f += [float(cls == c) for c in CLASSES]
        f += [s["noise_scale"], s["friction_log_sigma"], s["kappa"], s["delta"]]
    return f


def _model(kind: str):
    if kind == "boosted":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, l2_regularization=1.0, random_state=0)
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if kind == "logit_level":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000))
    raise ValueError(kind)


def logit_level_features(record: dict) -> list[float]:
    """Few parameter recalibration: logit of the Monte Carlo probability, a stacked indicator, their
    product (a separate slope and intercept for stacked placements) and the depth noise scale."""
    p = float(np.clip(record["p_hat"], 0.01, 0.99))
    z = float(np.log(p / (1 - p)))
    st = float(record["placement"] >= 1)
    s = {**DEFAULT_SETTINGS, **record.get("settings", {})}
    return [z, st, z * st, float(s["noise_scale"])]


class Recalibrator:
    """``Recalibrator("boosted", context=True)`` is the final method; ``("logistic", False)`` uses
    the Monte Carlo outputs alone."""

    def __init__(self, kind: str = "boosted", context: bool = True):
        self.kind = kind
        self.context = context
        self.model = None

    def _X(self, records) -> np.ndarray:
        if self.kind == "logit_level":
            return np.array([logit_level_features(r) for r in records], dtype=float)
        return np.array([features(r, self.context) for r in records], dtype=float)

    def fit(self, records) -> "Recalibrator":
        y = np.array([r["y"] for r in records], dtype=float).astype(int)
        self.model = _model(self.kind).fit(self._X(records), y)
        return self

    def predict(self, records) -> np.ndarray:
        return self.model.predict_proba(self._X(records))[:, 1]

    def save(self, path) -> None:
        Path(path).write_bytes(pickle.dumps({"kind": self.kind, "context": self.context, "model": self.model}))

    @staticmethod
    def load(path) -> "Recalibrator":
        d = pickle.loads(Path(path).read_bytes())
        r = Recalibrator(d["kind"], d["context"])
        r.model = d["model"]
        return r


TRAIN_PREFIXES = ("E2all", "E2grow", "E4", "E5", "E6")


def training_records(exp_dir, prefixes=TRAIN_PREFIXES, exclude_tower: int | None = None) -> list[dict]:
    """Labelled records of earlier experiments (perceived stones) for fitting a recalibrator,
    with their settings; ``exclude_tower`` drops every record of that tower index, since
    experiments share stones by tower index."""
    import glob
    import json

    out = []
    for pre in prefixes:
        for p in sorted(glob.glob(str(Path(exp_dir) / f"{pre}_*" / "records.json"))):
            name = Path(p).parent.name
            s = settings_from_name(name)
            for r in json.load(open(p)):
                if exclude_tower is not None and r["tower"] == exclude_tower:
                    continue
                r["settings"] = s
                out.append(r)
    return out


def candidate_record(p_hat: float, samples, level: int, height: float, n_contacts: int, stone_class, settings: dict) -> dict:
    """A record in the experiment format for one screened candidate, for ``Recalibrator.predict``."""
    smp = np.asarray(samples, dtype=float).reshape(-1)
    lo, hi = (np.quantile(smp, [0.025, 0.975]) if smp.size else (np.nan, np.nan))
    return {"p_hat": float(p_hat), "m_hat": float(smp.mean()) if smp.size else np.nan, "m_lo": float(lo), "m_hi": float(hi), "samples": smp,
            "placement": int(level), "height": float(height), "meta": {"n_contacts": int(n_contacts), "stone_class": stone_class or ""}, "settings": dict(settings)}


def leave_tower_out(test_records, train_records, kind: str = "boosted", context: bool = True) -> np.ndarray:
    """Predictions for ``test_records`` where the model for tower t is trained on
    ``train_records`` without any record of tower index t (experiments share stones by tower)."""
    p = np.full(len(test_records), np.nan)
    towers = np.array([r["tower"] for r in test_records])
    tr_towers = np.array([r["tower"] for r in train_records])
    for t in np.unique(towers):
        train = [r for r, tt in zip(train_records, tr_towers) if tt != t]
        idx = np.where(towers == t)[0]
        rc = Recalibrator(kind, context).fit(train)
        p[idx] = rc.predict([test_records[i] for i in idx])
    return p
