"""Scene description, camera model and packing for the kernels (Section 10.1, 10.2).

A ``Scene`` holds instances (mesh, material, transforms at shutter open and close), overlay
instances (unlit emissive geometry rendered in a second pass), lights, the sky and an optional
homogeneous medium. ``Scene.build`` packs everything into flat arrays: one BVH per unique mesh
(from ``topple.core.raycast``), a top level BVH over instance motion bounds, a material table and
a light table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.mesh import TriMesh
from ..core.transforms import SE3, quat_to_rot
from .lights import LIGHT_COLS, AreaLight, Sky, SunLight
from .materials import M_LIGHT, M_NCOLS, Material
from .textures import vertex_attributes


@dataclass
class Instance:
    mesh: TriMesh
    material: Material
    T0: SE3 = field(default_factory=SE3)
    T1: Optional[SE3] = None
    object_id: int = 0
    overlay: bool = False
    vertex_colors: Optional[np.ndarray] = None
    name: str = ""


@dataclass
class Medium:
    sigma_t: float = 0.5
    albedo: float = 0.9
    g: float = 0.6
    z_max: float = 0.4  # slab 0 <= z <= z_max

    def row(self) -> np.ndarray:
        return np.array([1.0, self.sigma_t, self.albedo, self.g, self.z_max])


@dataclass
class Camera:
    """Thin lens camera with a polygonal aperture, or a traced lens system (``lens``). Camera
    frame: x right, y down, z forward."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, -2.0, 0.6]))
    look_at: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    focal_mm: float = 50.0
    f_number: float = 8.0
    focus_distance: Optional[float] = None
    sensor_mm: tuple = (36.0, 24.0)
    width: int = 160
    height: int = 107
    shutter: tuple = (0.0, 1.0 / 48.0)
    aperture_blades: int = 9
    ortho: bool = False
    ortho_width: float = 1.0
    circle_of_confusion_mm: float = 0.03
    exposure: float = 1.0
    name: str = "camera"
    lens: Optional[str] = None  # None: thin lens; "dgauss50": traced double Gauss (Kolb et al. 1995)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.look_at = np.asarray(self.look_at, dtype=np.float64)
        self.up = np.asarray(self.up, dtype=np.float64)
        if self.focus_distance is None:
            self.focus_distance = float(np.linalg.norm(self.look_at - self.position))

    def frame(self) -> np.ndarray:
        fwd = self.look_at - self.position
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, self.up)
        if np.linalg.norm(right) < 1e-9:
            right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd], axis=1)  # columns

    def sensor_size_m(self) -> tuple:
        """Sensor width fixed at 36 mm, height follows the image aspect (sensor crop)."""
        w = self.sensor_mm[0] / 1000.0
        h = w * self.height / self.width
        return w, h

    def dof_limits(self) -> tuple:
        """Hyperfocal distance, near and far limits of acceptable sharpness (metres)."""
        f = self.focal_mm / 1000.0
        c = self.circle_of_confusion_mm / 1000.0
        N = self.f_number
        s = self.focus_distance
        H = f * f / (N * c) + f
        near = s * H / (H + (s - f))
        far = s * H / (H - (s - f)) if H > (s - f) else np.inf
        return H, near, far

    def to_array(self) -> np.ndarray:
        R = self.frame()
        w, h = self.sensor_size_m()
        p = np.zeros(24)
        p[0:3] = self.position
        p[3:12] = R.reshape(-1)
        p[12] = self.focal_mm / 1000.0
        p[13] = w
        p[14] = h
        p[15] = 0.0 if self.ortho else (self.focal_mm / 1000.0) / (2.0 * self.f_number)
        p[16] = self.focus_distance
        p[17] = self.aperture_blades
        p[18] = 1.0 if self.ortho else 0.0
        p[19] = self.ortho_width
        p[20] = self.ortho_width * self.height / self.width
        if self.lens and not self.ortho:
            if self.lens != "dgauss50":
                raise ValueError(f"unknown lens {self.lens}")
            from .lens import pack_camera_lens

            p = pack_camera_lens(p, self.focal_mm, self.f_number, float(self.focus_distance), w, h)
        return p

    def pixel_ray(self, px: float, py: float):
        """Central (pinhole) ray through pixel (px, py) for tests and overlays."""
        p = self.to_array()
        R = p[3:12].reshape(3, 3)
        if self.ortho:
            x = (px / self.width - 0.5) * p[19]
            y = (py / self.height - 0.5) * p[20]
            o = p[0:3] + R @ np.array([x, y, 0.0])
            return o, R @ np.array([0.0, 0.0, 1.0])
        x = (px / self.width - 0.5) * p[13]
        y = (py / self.height - 0.5) * p[14]
        d = np.array([x, y, p[12]])
        d /= np.linalg.norm(d)
        return p[0:3].copy(), R @ d


class Scene:
    def __init__(self):
        self.instances: list[Instance] = []
        self.overlays: list[Instance] = []
        self.lights: list = []  # AreaLight, SunLight
        self.sky: Optional[Sky] = None
        self.medium: Optional[Medium] = None
        self.background: np.ndarray = np.zeros(3)  # radiance seen by escaped rays when no sky
        self._next_object_id = 1

    # --------------------------------------------------------------- assembly
    def add(self, mesh: TriMesh, material: Material, T0: SE3 | None = None, T1: SE3 | None = None, object_id: int | None = None, vertex_colors=None, name: str = "") -> Instance:
        if object_id is None:
            object_id = self._next_object_id
            self._next_object_id += 1
        inst = Instance(mesh, material, SE3() if T0 is None else T0, T1, object_id, False, vertex_colors, name)
        if len(mesh.faces):  # an empty mesh (e.g. a stone wholly on the cut side of a section) adds nothing
            self.instances.append(inst)
        return inst

    def add_overlay(self, mesh: TriMesh, color=(1.0, 1.0, 1.0), alpha: float = 1.0, blend: str = "over", glass: bool = False, vertex_colors=None, outline: float = 0.25, T0: SE3 | None = None, T1: SE3 | None = None, name: str = "overlay") -> Instance:
        """Overlay geometry: unlit emissive, composited alpha over (or additively) in a second pass."""
        mat = Material(name=name, emission=tuple(np.asarray(color, dtype=np.float64)[:3]), alpha=alpha, blend=1 if blend == "add" else 0, unlit=not glass, glass_shell=glass, outline=outline, use_vertex_colors=vertex_colors is not None)
        inst = Instance(mesh, mat, SE3() if T0 is None else T0, T1, 0, True, vertex_colors, name)
        if len(mesh.faces):
            self.overlays.append(inst)
        return inst

    def add_light(self, light) -> None:
        self.lights.append(light)

    def set_sky(self, sky: Sky) -> None:
        self.sky = sky

    def set_medium(self, medium: Medium | None) -> None:
        self.medium = medium

    def bounds(self, include_ground: bool = False) -> tuple:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for inst in self.instances:
            if not include_ground and inst.name == "ground":
                continue
            for T in (inst.T0, inst.T1 if inst.T1 is not None else inst.T0):
                V = T.apply(inst.mesh.vertices)
                lo = np.minimum(lo, V.min(axis=0))
                hi = np.maximum(hi, V.max(axis=0))
        if not np.all(np.isfinite(lo)):
            return np.zeros(3), np.zeros(3)
        return lo, hi

    # ----------------------------------------------------------------- packing
    def build(self) -> "ScenePack":
        return ScenePack(self)


def _motion_bounds(mesh: TriMesh, T0: SE3, T1: SE3 | None, n_steps: int = 4):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    V = mesh.vertices
    if T1 is None:
        W = T0.apply(V)
        return W.min(axis=0), W.max(axis=0)
    for k in range(n_steps + 1):
        T = T0.interpolate(T1, k / n_steps)
        W = T.apply(V)
        lo = np.minimum(lo, W.min(axis=0))
        hi = np.maximum(hi, W.max(axis=0))
    # pad for the arc between interpolation samples
    pad = 0.02 * (hi - lo).max()
    return lo - pad, hi + pad


def _build_top_bvh(bmin: np.ndarray, bmax: np.ndarray, leaf_size: int = 2):
    """Median split BVH over instance boxes. Returns flattened arrays."""
    n = bmin.shape[0]
    nodes = []  # (bmin, bmax, left, right, start, count)
    order = np.arange(n, dtype=np.int64)
    if n == 0:
        return (np.zeros((1, 3)), np.zeros((1, 3)), np.array([-1]), np.array([-1]), np.array([0]), np.array([0]), order)
    cent = 0.5 * (bmin + bmax)
    stack = [(0, n, -1, False)]
    out = {"bmin": [], "bmax": [], "left": [], "right": [], "start": [], "count": []}
    while stack:
        lo, hi, parent, is_left = stack.pop()
        idx = order[lo:hi]
        nid = len(out["bmin"])
        out["bmin"].append(bmin[idx].min(axis=0))
        out["bmax"].append(bmax[idx].max(axis=0))
        out["left"].append(-1)
        out["right"].append(-1)
        out["start"].append(lo)
        out["count"].append(hi - lo)
        if parent >= 0:
            out["left" if is_left else "right"][parent] = nid
        if hi - lo <= leaf_size:
            continue
        ext = out["bmax"][nid] - out["bmin"][nid]
        axis = int(np.argmax(ext))
        perm = np.argsort(cent[idx, axis], kind="stable")
        order[lo:hi] = idx[perm]
        mid = lo + (hi - lo) // 2
        out["count"][nid] = 0
        stack.append((mid, hi, nid, False))
        stack.append((lo, mid, nid, True))
    return (
        np.ascontiguousarray(np.array(out["bmin"])),
        np.ascontiguousarray(np.array(out["bmax"])),
        np.array(out["left"], dtype=np.int64),
        np.array(out["right"], dtype=np.int64),
        np.array(out["start"], dtype=np.int64),
        np.array(out["count"], dtype=np.int64),
        order,
    )


class GeometryPack:
    """Concatenated per mesh BVHs and vertex data for a list of instances."""

    def __init__(self, instances: list[Instance], mats: dict):
        # unique mesh records keyed by (mesh identity, vertex colour identity)
        records = {}
        self.inst_mesh = np.zeros(len(instances), dtype=np.int64)
        for k, inst in enumerate(instances):
            key = (id(inst.mesh), id(inst.vertex_colors) if inst.vertex_colors is not None else 0, bool(inst.material.weathering_amp > 0))
            if key not in records:
                records[key] = (len(records), inst)
            self.inst_mesh[k] = records[key][0]
        meshes = [rec[1] for rec in sorted(records.values(), key=lambda r: r[0])]
        node_off, tri_off, vert_off = [0], [0], [0]
        bmin, bmax, left, right, start, count = [], [], [], [], [], []
        tv0, te1, te2, order, faces, vnorm, vattr, vcol = [], [], [], [], [], [], [], []
        for inst in meshes:
            m = inst.mesh
            b = m.bvh()
            bmin.append(b.bmin)
            bmax.append(b.bmax)
            left.append(b.left)
            right.append(b.right)
            start.append(b.start)
            count.append(b.count)
            tv0.append(b.tri_v0)
            te1.append(b.tri_e1)
            te2.append(b.tri_e2)
            order.append(b.order)
            faces.append(m.faces)
            vnorm.append(m.vertex_normals())
            if inst.material.weathering_amp > 0:
                vattr.append(vertex_attributes(m))
            else:
                va = np.zeros((m.n_vertices, 2))
                va[:, 1] = 1.0
                vattr.append(va)
            if inst.vertex_colors is not None:
                vcol.append(np.asarray(inst.vertex_colors, dtype=np.float64).reshape(-1, 3)[: m.n_vertices])
            else:
                vcol.append(np.ones((m.n_vertices, 3)))
            node_off.append(node_off[-1] + len(b.bmin))
            tri_off.append(tri_off[-1] + len(b.order))
            vert_off.append(vert_off[-1] + m.n_vertices)
        if not meshes:
            # empty pack placeholder
            self.node_off = np.array([0, 1], dtype=np.int64)
            self.tri_off = np.array([0, 1], dtype=np.int64)
            self.vert_off = np.array([0, 3], dtype=np.int64)
            self.bmin = np.zeros((1, 3))
            self.bmax = np.zeros((1, 3))
            self.left = np.array([-1], dtype=np.int64)
            self.right = np.array([-1], dtype=np.int64)
            self.start = np.array([0], dtype=np.int64)
            self.count = np.array([0], dtype=np.int64)
            self.tv0 = np.zeros((1, 3))
            self.te1 = np.zeros((1, 3))
            self.te2 = np.zeros((1, 3))
            self.order = np.zeros(1, dtype=np.int64)
            self.faces = np.zeros((1, 3), dtype=np.int64)
            self.vnorm = np.zeros((3, 3))
            self.vattr = np.zeros((3, 2))
            self.vcol = np.ones((3, 3))
        else:
            self.node_off = np.array(node_off, dtype=np.int64)
            self.tri_off = np.array(tri_off, dtype=np.int64)
            self.vert_off = np.array(vert_off, dtype=np.int64)
            self.bmin = np.ascontiguousarray(np.concatenate(bmin))
            self.bmax = np.ascontiguousarray(np.concatenate(bmax))
            self.left = np.concatenate(left)
            self.right = np.concatenate(right)
            self.start = np.concatenate(start)
            self.count = np.concatenate(count)
            self.tv0 = np.ascontiguousarray(np.concatenate(tv0))
            self.te1 = np.ascontiguousarray(np.concatenate(te1))
            self.te2 = np.ascontiguousarray(np.concatenate(te2))
            self.order = np.concatenate(order)
            self.faces = np.ascontiguousarray(np.concatenate(faces))
            self.vnorm = np.ascontiguousarray(np.concatenate(vnorm))
            self.vattr = np.ascontiguousarray(np.concatenate(vattr))
            self.vcol = np.ascontiguousarray(np.concatenate(vcol))
        # instances
        n = len(instances)
        self.inst_mat = np.zeros(n, dtype=np.int64)
        self.inst_obj = np.zeros(n, dtype=np.int64)
        self.inst_q0 = np.zeros((n, 4))
        self.inst_t0 = np.zeros((n, 3))
        self.inst_q1 = np.zeros((n, 4))
        self.inst_t1 = np.zeros((n, 3))
        self.inst_moving = np.zeros(n, dtype=np.int64)
        self.inst_light = np.full(n, -1, dtype=np.int64)
        ibmin = np.zeros((max(n, 1), 3))
        ibmax = np.zeros((max(n, 1), 3))
        for k, inst in enumerate(instances):
            self.inst_mat[k] = mats[id(inst.material)]
            self.inst_obj[k] = inst.object_id
            self.inst_q0[k] = inst.T0.q
            self.inst_t0[k] = inst.T0.t
            if inst.T1 is not None:
                self.inst_q1[k] = inst.T1.q
                self.inst_t1[k] = inst.T1.t
                self.inst_moving[k] = 1
            else:
                self.inst_q1[k] = inst.T0.q
                self.inst_t1[k] = inst.T0.t
            lo, hi = _motion_bounds(inst.mesh, inst.T0, inst.T1)
            ibmin[k] = lo
            ibmax[k] = hi
        if n == 0:
            self.inst_mat = np.zeros(1, dtype=np.int64)
            self.inst_obj = np.zeros(1, dtype=np.int64)
            self.inst_q0 = np.array([[1.0, 0, 0, 0]])
            self.inst_t0 = np.zeros((1, 3))
            self.inst_q1 = self.inst_q0.copy()
            self.inst_t1 = np.zeros((1, 3))
            self.inst_moving = np.zeros(1, dtype=np.int64)
            self.inst_light = np.full(1, -1, dtype=np.int64)
            self.inst_mesh = np.zeros(1, dtype=np.int64)
            top = _build_top_bvh(np.zeros((0, 3)), np.zeros((0, 3)))
        else:
            top = _build_top_bvh(ibmin, ibmax)
        self.top_bmin, self.top_bmax, self.top_left, self.top_right, self.top_start, self.top_count, self.top_order = top
        self.empty = n == 0

    def tuple(self):
        return (
            self.node_off, self.tri_off, self.vert_off, self.bmin, self.bmax, self.left, self.right, self.start, self.count,
            self.tv0, self.te1, self.te2, self.order, self.faces, self.vnorm, self.vattr, self.vcol,
            self.inst_mesh, self.inst_mat, self.inst_obj, self.inst_q0, self.inst_t0, self.inst_q1, self.inst_t1, self.inst_moving, self.inst_light,
            self.top_bmin, self.top_bmax, self.top_left, self.top_right, self.top_start, self.top_count, self.top_order,
        )


class ScenePack:
    def __init__(self, scene: Scene):
        self.scene = scene
        # materials
        mats: dict = {}
        rows = []
        light_rows = []
        light_instances = []
        for inst in scene.instances + scene.overlays:
            if id(inst.material) not in mats:
                mats[id(inst.material)] = len(rows)
                rows.append(inst.material.row())
        # lights: quads get geometry instances with emitter materials
        for li, light in enumerate(scene.lights):
            light_rows.append(light.row())
            if isinstance(light, AreaLight):
                m = Material(name=light.name, emission=tuple(light.radiance), emitter=True)
                r = m.row()
                r[M_LIGHT] = li
                mats[id(m)] = len(rows)
                rows.append(r)
                inst = Instance(light.mesh(), m, SE3(), None, 0, False, None, light.name)
                light_instances.append((inst, li))
        if scene.sky is not None:
            light_rows.append(scene.sky.row())
            self.sky_index = len(light_rows) - 1
        else:
            self.sky_index = -1
        self.mats = np.ascontiguousarray(np.array(rows if rows else [Material().row()]))
        self.lights = np.ascontiguousarray(np.array(light_rows)) if light_rows else np.zeros((0, LIGHT_COLS))
        all_instances = list(scene.instances) + [li[0] for li in light_instances]
        self.main = GeometryPack(all_instances, mats)
        for k, (inst, li) in enumerate(light_instances):
            self.main.inst_light[len(scene.instances) + k] = li
        self.overlay = GeometryPack(scene.overlays, mats)
        # sky
        if scene.sky is not None and scene.sky.params is not None:
            self.sky_p = np.ascontiguousarray(scene.sky.params)
        else:
            self.sky_p = np.zeros(32)
        if scene.sky is not None and scene.sky.image is not None:
            self.env_img, self.env_marg, self.env_cond = scene.sky.env_tables()
        else:
            self.env_img, self.env_marg, self.env_cond = np.zeros((1, 1, 3)), np.ones(1), np.ones((1, 1))
        self.medium = scene.medium.row() if scene.medium is not None else np.zeros(5)
        self.background = np.asarray(scene.background, dtype=np.float64)
