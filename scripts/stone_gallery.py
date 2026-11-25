"""Stone gallery: three stones per class saved as OBJ with a descriptor table (figure F2 input)."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topple.core.config import load_config  # noqa: E402
from topple.core.rng import Stage, make_rng  # noqa: E402
from topple.stonegen import StoneGenerator  # noqa: E402


def main(out_dir: str = "results/stones", per_class: int = 3, seed: int = 0, config: str = "default"):
    cfg = load_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = StoneGenerator(cfg)
    rows = []
    for cls in g.classes:
        rng = make_rng(seed, 0, Stage.STONEGEN, sub=hash(cls) & 0xFFFF)
        for i in range(per_class):
            t0 = time.time()
            s = g.sample(rng, cls, name=f"{cls}_{i}")
            dt = time.time() - t0
            s.mesh.save(out / f"{cls}_{i}.obj")
            row = s.summary()
            row["seconds"] = round(dt, 3)
            row["valid"] = all(s.mesh.is_valid().values())
            rows.append(row)
            print(f"{cls}_{i}: family {s.family}, {s.mesh.n_faces} faces, {dt:.2f} s, sphericity {s.descriptors['sphericity']:.3f}")
    keys = list(rows[0].keys())
    with open(out / "descriptors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / "descriptors.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
