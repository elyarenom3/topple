"""Scene assembly from the recorded showcase tower (Sections 10.5 to 10.8).

Shared by the still and video scripts: loads results/showcase, places the stones with their
material presets, adds the ground and lighting presets, and builds every overlay O1 to O9 as
scene geometry from the recorded simulation and the static analysis.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..core.mesh import TriMesh, box
from ..core.transforms import SE3
from ..physics.recorder import StateRecorder
from . import overlays as ov
from .color import colormap, diverging_colormap
from .presets import camera_preset, ground_preset, lighting_preset, material_preset
from .scene import Camera, Scene


class Showcase:
    def __init__(self, directory: str | Path):
        d = Path(directory)
        with open(d / "showcase.pkl", "rb") as fh:
            self.data = pickle.load(fh)
        self.dir = d
        self.build = StateRecorder.load(d / "build_record.npz")
        self.clips = {}
        for k in ("tilt", "collapse", "shake"):
            p = d / f"{k}_record.npz"
            if p.exists():
                self.clips[k] = StateRecorder.load(p)
        self.stones = self.data["stones"]
        self.meshes = {s["id"]: TriMesh(s["V"], s["F"], s["name"]) for s in self.stones}
        self.rng = np.random.default_rng(7)
        self.materials = {s["id"]: material_preset(s["material"], np.random.default_rng(1000 + s["id"]), seed=1000 + s["id"]) for s in self.stones}
        self.tower_ids = list(self.data["tower_ids"])
        self.final_q = self.data["final_state"]["q"]
        self.final_x = self.data["final_state"]["x"]
        self.pile_q = self.data["pile_state"]["q"]
        self.pile_x = self.data["pile_state"]["x"]
        # subset map: clip recorders index ground plus tower bodies only
        self.subset_map = self.data["subset_map"]
        if any(st.get("forces") is None for st in self.data["statics"]):
            self.data["statics"] = self.hull_statics()

    def hull_statics(self, band: float = 5e-4, cone_sides: int = 8) -> list:
        """Contact forces of every partial tower from the hull contact model of the margin analysis
        (one normal per contact point, from the closest face of the supporting hull) and the
        minimum norm force program."""
        from ..statics.analysis import build_tower
        from ..statics.equilibrium import min_norm_forces

        out = []
        ids = self.tower_ids
        for j in range(1, len(ids) + 1):
            realised = []
            for i in ids[:j]:
                T = self.pose(i)
                realised.append({"mesh": self.meshes[i].transformed(T), "mass": self.stones_by_id(i)["mass"], "com": T.t.copy(), "mu": 0.6, "name": self.stones_by_id(i)["name"]})
            tower = build_tower(realised, band, 8, geometry="hull", settle=False)
            try:
                F = np.asarray(min_norm_forces(tower, None, cone_sides)).tolist()
            except Exception:
                F = None
            body_of = {0: self.data["ground_id"]}
            body_of.update({k + 1: i for k, i in enumerate(ids[:j])})
            out.append({
                "n": j,
                "contacts": [{"a": body_of[c.body_a], "b": body_of[c.body_b], "p": np.asarray(c.point).tolist(), "n": np.asarray(c.normal).tolist(), "mu": float(c.mu)} for c in tower.contacts],
                "forces": F,
            })
        return out

    # ----------------------------------------------------------------- geometry
    def pose(self, body_id: int, state="final") -> SE3:
        if state == "final":
            return SE3(self.final_q[body_id], self.final_x[body_id])
        if state == "pile":
            return SE3(self.pile_q[body_id], self.pile_x[body_id])
        raise ValueError(state)

    def tower_bounds(self, ids=None):
        ids = self.tower_ids if ids is None else ids
        V = np.concatenate([self.meshes[i].transformed(self.pose(i)).vertices for i in ids])
        lo, hi = V.min(axis=0), V.max(axis=0)
        lo[2] = 0.0
        return lo, hi

    def world_mesh(self, body_id, T: SE3) -> TriMesh:
        return self.meshes[body_id].transformed(T)

    # ----------------------------------------------------------------- scenes
    def base_scene(self, ground: str = "studio_slab", lighting: str = "studio", center=None, size: float = 0.6, sun_azimuth_deg: float = 210.0) -> Scene:
        sc = Scene()
        for m, mat, name in ground_preset(ground, size=6.0, rng=np.random.default_rng(3)):
            if ground == "riverbank":
                # put the flat centre of the bank at z = 0 under the tower
                if name == "ground":
                    z0 = float(np.median(m.vertices[np.linalg.norm(m.vertices[:, :2], axis=1) < 0.25, 2])) if np.any(np.linalg.norm(m.vertices[:, :2], axis=1) < 0.25) else 0.0
                    self._bank_shift = -z0
                m = m.translated([0.0, 0.0, getattr(self, "_bank_shift", 0.0)])
            sc.add(m, mat, name=name)
        c = np.zeros(3) if center is None else np.asarray(center)
        lights, sky, bg = lighting_preset(lighting, center=c, size=size, sun_azimuth_deg=sun_azimuth_deg)
        for l in lights:
            sc.add_light(l)
        if sky is not None:
            sc.set_sky(sky)
        sc.background = bg
        return sc

    def add_stones(self, sc: Scene, ids, poses: dict, poses_end: dict | None = None, tint: dict | None = None):
        for i in ids:
            mat = self.materials[i]
            if tint and i in tint:
                import copy

                mat = copy.copy(mat)
                mat.base_color = tuple(np.clip(np.asarray(tint[i]) * 0.8 + 0.2 * np.asarray(mat.base_color), 0, 1))
            sc.add(self.meshes[i], mat, poses[i], None if poses_end is None else poses_end.get(i), object_id=int(i), name=self.stones_by_id(i)["name"])

    def stones_by_id(self, i):
        for s in self.stones:
            if s["id"] == i:
                return s
        raise KeyError(i)

    # ----------------------------------------------------------------- overlays
    def statics_for(self, n: int | None = None):
        st = self.data["statics"]
        return st[-1] if n is None else st[n - 1]

    def add_force_overlays(self, sc: Scene, n: int | None = None, scale: float = 0.03, cones: bool = True, plane=None):
        """O1: arrows sized by force magnitude and coloured by utilisation, with friction cones."""
        st = self.statics_for(n)
        if st["forces"] is None:
            return
        F_local = np.asarray(st["forces"])
        from ..core.transforms import tangent_basis

        class _C:
            pass

        cs, fw = [], []
        for c, f in zip(st["contacts"], F_local):
            n_ = np.asarray(c["n"])
            t1, t2 = tangent_basis(n_)
            fw.append(f[0] * t1 + f[1] * t2 + f[2] * n_)
            cc = _C()
            cc.point = np.asarray(c["p"])
            cc.normal = n_
            if plane is not None:
                cc.point = self._project(cc.point[None, :], plane)[0]
                nn = np.asarray(plane[1], dtype=float) / np.linalg.norm(plane[1])
                fw[-1] = fw[-1] - (fw[-1] @ nn) * nn
            cs.append(cc)
        mu = float(np.mean([c["mu"] for c in st["contacts"]])) if st["contacts"] else 0.6
        for mesh, col, alpha, glass in ov.contact_force_overlays(cs, np.array(fw), mu, scale=scale, cone_height=0.02):
            if glass and not cones:
                continue
            sc.add_overlay(mesh, col, alpha=alpha, glass=glass)

    def thrust_line(self, n: int | None = None) -> np.ndarray:
        """O2: line of action of the resultant force across each interface, bottom to top."""
        st = self.statics_for(n)
        if st["forces"] is None:
            return np.zeros((0, 3))
        from ..core.transforms import tangent_basis

        pts = []
        by_pair = {}
        for c, f in zip(st["contacts"], np.asarray(st["forces"])):
            n_ = np.asarray(c["n"])
            t1, t2 = tangent_basis(n_)
            fw = f[0] * t1 + f[1] * t2 + f[2] * n_
            by_pair.setdefault((c["a"], c["b"]), []).append((np.asarray(c["p"]), fw))
        for key in sorted(by_pair, key=lambda k: np.mean([p[2] for p, _ in by_pair[k]])):
            P = np.array([p for p, _ in by_pair[key]])
            F = np.array([f for _, f in by_pair[key]])
            Fz = F[:, 2].clip(min=1e-12)
            # centre of pressure: force weighted mean point on the interface
            pts.append((P * Fz[:, None]).sum(axis=0) / Fz.sum())
        if len(pts) < 2:
            return np.array(pts)
        # extend above the top interface along the top stone's weight line
        top = self.tower_ids[-1] if n is None else self.tower_ids[n - 1]
        com_top = self.pose(top).t
        pts.append(np.array([pts[-1][0] * 0.5 + com_top[0] * 0.5, pts[-1][1] * 0.5 + com_top[1] * 0.5, com_top[2]]))
        return np.array(pts)

    @staticmethod
    def _project(P: np.ndarray, plane) -> np.ndarray:
        """Project points onto the section plane (point, normal towards the camera) and lift them
        just in front of it, so section overlays are drawn on the cut."""
        p0, nrm, off = plane
        nrm = np.asarray(nrm, dtype=float) / np.linalg.norm(nrm)
        return P - ((P - np.asarray(p0)) @ nrm)[:, None] * nrm[None, :] + off * nrm[None, :]

    def add_thrust_line(self, sc: Scene, n: int | None = None, radius: float = 0.0025, plane=None):
        P = self.thrust_line(n)
        if plane is not None and len(P):
            P = self._project(P, plane)
        if len(P) >= 2:
            sc.add_overlay(ov.tube(P, radius), (1.0, 0.55, 0.15), alpha=0.95, blend="add")

    def add_com_overlays(self, sc: Scene, radius: float = 0.004):
        """O4: centre of mass of each upper subassembly with a plumb line to the interface below."""
        ids = self.tower_ids
        for j in range(1, len(ids)):
            upper = ids[j:]
            masses = np.array([self.stones_by_id(i)["mass"] for i in upper])
            coms = np.array([self.pose(i).t for i in upper])
            c = (masses[:, None] * coms).sum(axis=0) / masses.sum()
            below = self.world_mesh(ids[j - 1], self.pose(ids[j - 1]))
            z_int = float(below.vertices[:, 2].max())
            sc.add_overlay(ov.sphere(c, radius, 2), (0.95, 0.95, 0.3), alpha=1.0)
            sc.add_overlay(ov.bar(c, np.array([c[0], c[1], z_int]), 0.0012), (0.95, 0.95, 0.3), alpha=0.9)

    def margin_tint(self) -> dict:
        """O7: stone colour from the margin of the cut below it (diverging map centred at 0)."""
        recs = [r for r in self.data["records"] if not r["meta"].get("probe")]
        vals = {}
        for r, i in zip(recs, self.tower_ids):
            vals[i] = r["m_true"] if np.isfinite(r["m_true"]) else r["m_hat"]
        v = np.array(list(vals.values()), dtype=float)
        cols = diverging_colormap(v, center=3.0, span=max(10.0, float(np.nanmax(np.abs(v - 3.0)))))
        return {i: tuple(c) for i, c in zip(vals.keys(), cols)}

    def add_posterior_shells(self, sc: Scene, T_world: SE3, n: int = 32, alpha: float = 0.06):
        """O5: posterior shape samples as translucent shells with additive compositing, and the 95%
        centre of mass ellipsoid."""
        post = self.data["posterior"]
        scan = SE3.from_matrix(post["scan_pose"])
        T = T_world @ scan.inverse()
        for V, F in post["samples"][:n]:
            sc.add_overlay(TriMesh(V, F).transformed(T), (0.55, 0.75, 1.0), alpha=alpha, blend="add", outline=0.0)
        C = np.asarray(post["coms"])
        if len(C) > 3:
            mu = T.apply(C.mean(axis=0))
            R = T.R
            S = R @ np.cov(C.T) @ R.T + 1e-10 * np.eye(3)
            sc.add_overlay(ov.ellipsoid(mu, S), (1.0, 0.8, 0.3), alpha=0.35)

    def camera(self, name: str, width: int, height: int, fill: float = 0.7, **over) -> Camera:
        """Preset camera (Section 10.5) with its distance scaled so the tower fills ``fill`` of the
        frame height; the preset direction, lens and aperture are kept."""
        lo, hi = self.tower_bounds()
        cam = camera_preset(name, (lo, hi), width, height)
        for k, v in over.items():
            setattr(cam, k, v)
        if not cam.ortho:
            sensor_h = cam.sensor_mm[1] * 1e-3
            if width / height < cam.sensor_mm[0] / cam.sensor_mm[1]:
                sensor_h = cam.sensor_mm[0] * 1e-3 * height / width
            fov_v = 2.0 * np.arctan(0.5 * sensor_h / (cam.focal_mm * 1e-3))
            extent = max(float(hi[2] - lo[2]), 0.6 * float(np.max(hi[:2] - lo[:2])), 0.05)
            d_fit = extent / fill / (2.0 * np.tan(0.5 * fov_v))
            direction = np.asarray(cam.position, dtype=float) - np.asarray(cam.look_at, dtype=float)
            direction /= max(np.linalg.norm(direction), 1e-12)
            cam.position = np.asarray(cam.look_at, dtype=float) + d_fit * direction
            cam.focus_distance = float(d_fit)
        if hasattr(cam, "__post_init__"):
            cam.__post_init__()
        return cam

    def meter_point(self):
        """Exposure meter point: open ground beside the tower, facing up (18% grey card under the key)."""
        lo, hi = self.tower_bounds()
        c = 0.5 * (lo + hi)
        r = 0.6 * float(np.max(hi[:2] - lo[:2])) + 0.05
        return np.array([c[0] + r, c[1] - r, 0.002])
