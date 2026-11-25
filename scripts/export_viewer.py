#!/usr/bin/env python3
"""Interactive WebGL viewer of the showcase tower (extension 9 of Section 14).

Exports the final tower from results/showcase (the same geometry as the renders) into one HTML
file with three.js: orbit, zoom, and toggles for the overlays O1 (contact forces coloured by
friction utilisation, friction cones), O2 (thrust line), O4 (centres of mass of the upper
subassemblies) and O7 (margin tint).

python scripts/export_viewer.py [--showcase results/showcase]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import RESULTS, parser

from topple.core.transforms import tangent_basis
from topple.render.color import colormap, diverging_colormap, srgb_encode
from topple.render.showcase import Showcase

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stone tower viewer</title>
<style>
:root { --bg: #f4f3ef; --fg: #1d1d1b; --panel: rgba(255,255,255,0.88); }
@media (prefers-color-scheme: dark) { :root { --bg: #151514; --fg: #ecebe6; --panel: rgba(30,30,28,0.88); } }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; }
#c { width: 100%; height: 100%; display: block; }
#panel { position: absolute; top: 12px; left: 12px; background: var(--panel); padding: 10px 12px; border-radius: 8px; max-width: 300px; }
#panel h1 { font-size: 15px; margin: 0 0 6px; }
#panel label { display: block; margin: 3px 0; cursor: pointer; }
#panel p { margin: 6px 0 0; font-size: 12px; opacity: 0.8; }
</style></head><body>
<canvas id="c"></canvas>
<div id="panel"><h1>Stone tower</h1>
<label><input type="checkbox" id="forces" checked> Contact forces (O1)</label>
<label><input type="checkbox" id="cones"> Friction cones (O1)</label>
<label><input type="checkbox" id="thrust" checked> Thrust line (O2)</label>
<label><input type="checkbox" id="com"> Centres of mass (O4)</label>
<label><input type="checkbox" id="tint"> Margin tint (O7)</label>
<p id="info"></p></div>
<script type="importmap">{"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const DATA = __DATA__;
const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
scene.background = new THREE.Color(dark ? 0x151514 : 0xf4f3ef);
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 20);
const c = DATA.center;
camera.up.set(0, 0, 1);
camera.position.set(c[0] + DATA.extent * 3.2, c[1] - DATA.extent * 3.2, c[2] + DATA.extent * 1.2);
const controls = new OrbitControls(camera, canvas);
controls.target.set(c[0], c[1], c[2]);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(1, -1, 2); scene.add(key);
const ground = new THREE.Mesh(new THREE.CircleGeometry(DATA.extent * 4, 64), new THREE.MeshStandardMaterial({ color: dark ? 0x2a2a28 : 0xd9d7d0, roughness: 0.9 }));
scene.add(ground);
const stones = [];
for (const s of DATA.stones) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(s.V, 3));
  g.setIndex(s.F); g.computeVertexNormals();
  const m = new THREE.MeshStandardMaterial({ color: new THREE.Color(...s.color), roughness: 0.75 });
  const mesh = new THREE.Mesh(g, m); mesh.userData = s; scene.add(mesh); stones.push(mesh);
}
function group(name) { const g = new THREE.Group(); g.name = name; scene.add(g); return g; }
const gF = group("forces"), gC = group("cones"), gT = group("thrust"), gM = group("com");
for (const a of DATA.arrows) {
  const dir = new THREE.Vector3(...a.d); const len = dir.length(); dir.normalize();
  gF.add(new THREE.ArrowHelper(dir, new THREE.Vector3(...a.p), len, new THREE.Color(...a.color).getHex(), Math.min(0.006, len * 0.4), 0.003));
  const cone = new THREE.Mesh(new THREE.ConeGeometry(Math.tan(a.half) * 0.02, 0.02, 24, 1, true), new THREE.MeshBasicMaterial({ color: 0xaad4ff, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
  cone.position.set(...a.p); const n = new THREE.Vector3(...a.n);
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), n); cone.position.addScaledVector(n, 0.01); gC.add(cone);
}
if (DATA.thrust.length > 1) {
  const curve = new THREE.CatmullRomCurve3(DATA.thrust.map(p => new THREE.Vector3(...p)));
  gT.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 64, 0.0022, 8), new THREE.MeshBasicMaterial({ color: 0xff8c26 })));
}
for (const m of DATA.coms) {
  const s = new THREE.Mesh(new THREE.SphereGeometry(0.0045, 16, 12), new THREE.MeshBasicMaterial({ color: 0xf2e94e }));
  s.position.set(...m.c); gM.add(s);
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...m.c), new THREE.Vector3(m.c[0], m.c[1], m.z)]);
  gM.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xf2e94e })));
}
function sync() {
  gF.visible = document.getElementById("forces").checked;
  gC.visible = document.getElementById("cones").checked;
  gT.visible = document.getElementById("thrust").checked;
  gM.visible = document.getElementById("com").checked;
  const tint = document.getElementById("tint").checked;
  for (const s of stones) s.material.color.setRGB(...(tint ? s.userData.tint : s.userData.color));
}
for (const id of ["forces", "cones", "thrust", "com", "tint"]) document.getElementById(id).addEventListener("change", sync);
sync();
document.getElementById("info").textContent = DATA.info;
function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener("resize", resize); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


def main(args):
    show = Showcase(Path(args.showcase))
    tint = show.margin_tint()
    stones = []
    base_cols = {"river_basalt": (0.18, 0.18, 0.19), "granite": (0.55, 0.53, 0.5), "sandstone": (0.66, 0.53, 0.38), "slate": (0.22, 0.24, 0.27), "limestone": (0.7, 0.68, 0.62), "quartzite": (0.78, 0.75, 0.72)}
    for i in show.tower_ids:
        m = show.world_mesh(i, show.pose(i))
        s = show.stones_by_id(i)
        stones.append({
            "name": s["name"], "V": np.round(m.vertices, 5).ravel().tolist(), "F": m.faces.ravel().tolist(),
            "color": list(base_cols.get(s["material"], (0.6, 0.6, 0.6))),
            "tint": [float(x) for x in np.clip(srgb_encode(np.asarray(tint.get(i, (0.6, 0.6, 0.6)))), 0, 1)],
        })
    st = show.statics_for(None)
    arrows = []
    if st["forces"] is not None:
        F = np.asarray(st["forces"])
        fw = []
        for cc, f in zip(st["contacts"], F):
            n = np.asarray(cc["n"])
            t1, t2 = tangent_basis(n)
            fw.append(f[0] * t1 + f[1] * t2 + f[2] * n)
        fw = np.array(fw)
        mags = np.linalg.norm(fw, axis=1)
        fmax = max(mags.max(), 1e-9)
        for cc, f, mag in zip(st["contacts"], fw, mags):
            if mag <= 1e-9:
                continue
            n = np.asarray(cc["n"])
            fn = float(f @ n)
            util = float(np.linalg.norm(f - fn * n) / max(cc["mu"] * fn, 1e-12)) if fn > 0 else 1.0
            col = srgb_encode(colormap(np.array([np.clip(util, 0, 1)]), "sequential", 0, 1))[0]
            arrows.append({"p": cc["p"], "d": (0.03 * f / fmax).tolist(), "n": n.tolist(), "half": float(np.arctan(cc["mu"])), "color": [float(x) for x in col]})
    coms = []
    ids = show.tower_ids
    for j in range(1, len(ids)):
        upper = ids[j:]
        w = np.array([show.stones_by_id(i)["mass"] for i in upper])
        c = (w[:, None] * np.array([show.pose(i).t for i in upper])).sum(axis=0) / w.sum()
        z = float(show.world_mesh(ids[j - 1], show.pose(ids[j - 1])).vertices[:, 2].max())
        coms.append({"c": c.tolist(), "z": z})
    lo, hi = show.tower_bounds()
    recs = [r for r in show.data["records"] if not r["meta"].get("probe")]
    info = f"{len(ids)} stones, {100 * hi[2]:.1f} cm. Final placement: predicted tilt margin {recs[-1]['m_hat']:.1f} deg, measured {recs[-1]['m_true']:.1f} deg." if recs else ""
    data = {"stones": stones, "arrows": arrows, "thrust": show.thrust_line(None).tolist(), "coms": coms, "center": (0.5 * (lo + hi)).tolist(), "extent": float(max(hi - lo)), "info": info}
    out = RESULTS / "viewer"
    out.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    (out / "tower_viewer.html").write_text(html)
    print("wrote", out / "tower_viewer.html", f"({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = parser("WebGL tower viewer")
    ap.add_argument("--showcase", default=str(RESULTS / "showcase"))
    main(ap.parse_args())
