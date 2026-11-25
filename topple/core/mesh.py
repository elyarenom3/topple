"""Triangle meshes, mass properties, quality checks and signed distance fields.

Mass properties follow Mirtich (1996) and Eberly's "Polyhedral mass properties
(revisited)" (Section 8.6). Quality checks follow Section 5.4: watertight,
manifold, consistently oriented, free of self intersections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

from .transforms import SE3, quat_rotate, rot_to_quat


class TriMesh:
    """Indexed triangle mesh with float64 vertices and int64 faces."""

    def __init__(self, vertices, faces, name: str = "mesh"):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)
        self.name = name
        self._bvh = None
        self._cache: dict = {}

    # ------------------------------------------------------------------ basics
    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    def copy(self) -> "TriMesh":
        return TriMesh(self.vertices.copy(), self.faces.copy(), self.name)

    def invalidate(self):
        self._bvh = None
        self._cache = {}

    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    def face_normals(self, normalized: bool = True) -> np.ndarray:
        t = self.triangles()
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        if normalized:
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        return n

    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.face_normals(normalized=False), axis=1)

    def vertex_normals(self) -> np.ndarray:
        fn = self.face_normals(normalized=False)  # area weighted
        vn = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)

    def surface_area(self) -> float:
        return float(self.face_areas().sum())

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def extents(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def bounding_radius(self, center=None) -> float:
        c = self.centroid() if center is None else np.asarray(center)
        return float(np.linalg.norm(self.vertices - c, axis=1).max())

    # ----------------------------------------------------------- mass properties
    def volume(self) -> float:
        t = self.triangles()
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def centroid(self) -> np.ndarray:
        """Volume centroid, Section 8.6 formula."""
        t = self.triangles()
        w = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))
        V = w.sum() / 6.0
        return (w[:, None] * (t[:, 0] + t[:, 1] + t[:, 2])).sum(axis=0) / (24.0 * V)

    def mass_properties(self, density: float = 1.0) -> dict:
        """Eberly's polyhedral mass properties.

        Returns dict(mass, volume, com, inertia_com (3x3 about the centre of mass, world
        axes), inertia_origin).
        """
        t = self.triangles()
        p0, p1, p2 = t[:, 0], t[:, 1], t[:, 2]
        d = np.cross(p1 - p0, p2 - p0)  # unnormalised normal (twice area)

        def f1(w0, w1, w2):
            return w0 + w1 + w2

        def f2(w0, w1, w2):
            return w0 * w0 + w0 * w1 + w1 * w1 + w2 * f1(w0, w1, w2)

        def f3(w0, w1, w2):
            return w0 ** 3 + w0 * w0 * w1 + w0 * w1 * w1 + w1 ** 3 + w2 * f2(w0, w1, w2)

        def g(w0, w1, w2, wi):
            return f2(w0, w1, w2) + wi * (f1(w0, w1, w2) + wi)

        x0, y0, z0 = p0[:, 0], p0[:, 1], p0[:, 2]
        x1, y1, z1 = p1[:, 0], p1[:, 1], p1[:, 2]
        x2, y2, z2 = p2[:, 0], p2[:, 1], p2[:, 2]
        intg = np.zeros(10)
        intg[0] = (d[:, 0] * f1(x0, x1, x2)).sum()
        intg[1] = (d[:, 0] * f2(x0, x1, x2)).sum()
        intg[2] = (d[:, 1] * f2(y0, y1, y2)).sum()
        intg[3] = (d[:, 2] * f2(z0, z1, z2)).sum()
        intg[4] = (d[:, 0] * f3(x0, x1, x2)).sum()
        intg[5] = (d[:, 1] * f3(y0, y1, y2)).sum()
        intg[6] = (d[:, 2] * f3(z0, z1, z2)).sum()
        intg[7] = (d[:, 0] * (y0 * g(x0, x1, x2, x0) + y1 * g(x0, x1, x2, x1) + y2 * g(x0, x1, x2, x2))).sum()
        intg[8] = (d[:, 1] * (z0 * g(y0, y1, y2, y0) + z1 * g(y0, y1, y2, y1) + z2 * g(y0, y1, y2, y2))).sum()
        intg[9] = (d[:, 2] * (x0 * g(z0, z1, z2, z0) + x1 * g(z0, z1, z2, z1) + x2 * g(z0, z1, z2, z2))).sum()
        intg *= np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
        vol = intg[0]
        mass = density * vol
        com = intg[1:4] / vol
        cx, cy, cz = com
        Ixx = intg[5] + intg[6]
        Iyy = intg[4] + intg[6]
        Izz = intg[4] + intg[5]
        Ixy = -intg[7]
        Iyz = -intg[8]
        Ixz = -intg[9]
        I_origin = density * np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        # shift to centre of mass
        I_com = I_origin.copy()
        I_com[0, 0] -= mass * (cy * cy + cz * cz)
        I_com[1, 1] -= mass * (cz * cz + cx * cx)
        I_com[2, 2] -= mass * (cx * cx + cy * cy)
        I_com[0, 1] += mass * cx * cy
        I_com[1, 0] = I_com[0, 1]
        I_com[1, 2] += mass * cy * cz
        I_com[2, 1] = I_com[1, 2]
        I_com[0, 2] += mass * cz * cx
        I_com[2, 0] = I_com[0, 2]
        return {"mass": float(mass), "volume": float(vol), "com": com, "inertia_com": I_com, "inertia_origin": I_origin}

    def principal_frame(self, density: float = 1.0) -> SE3:
        """SE3 of the body frame (origin at centre of mass, axes along principal axes of inertia)
        expressed in the mesh's current frame, so that mesh_in_body = frame.inverse().apply(vertices)."""
        mp = self.mass_properties(density)
        w, U = np.linalg.eigh(mp["inertia_com"])
        # sort by descending moment so axis 0 is the long axis (smallest moment last)
        order = np.argsort(w)[::-1]
        U = U[:, order]
        if np.linalg.det(U) < 0:
            U[:, 2] *= -1
        return SE3(rot_to_quat(U), mp["com"])

    def to_principal_frame(self, density: float = 1.0) -> tuple["TriMesh", SE3]:
        fr = self.principal_frame(density)
        m = self.transformed(fr.inverse())
        return m, fr

    # -------------------------------------------------------------- transforms
    def transformed(self, T: SE3) -> "TriMesh":
        return TriMesh(T.apply(self.vertices), self.faces.copy(), self.name)

    def transform_(self, T: SE3) -> "TriMesh":
        self.vertices = T.apply(self.vertices)
        self.invalidate()
        return self

    def translated(self, t) -> "TriMesh":
        return TriMesh(self.vertices + np.asarray(t, dtype=np.float64), self.faces.copy(), self.name)

    def scaled(self, s) -> "TriMesh":
        return TriMesh(self.vertices * np.asarray(s, dtype=np.float64), self.faces.copy(), self.name)

    # ------------------------------------------------------------------ topology
    def edges(self, unique: bool = True) -> np.ndarray:
        e = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        if not unique:
            return e
        return np.unique(np.sort(e, axis=1), axis=0)

    def _edge_counts(self):
        e = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        key = np.sort(e, axis=1)
        uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return e, uniq, inv, counts

    def is_watertight(self) -> bool:
        """Every edge is shared by exactly two faces."""
        _, _, _, counts = self._edge_counts()
        return bool(np.all(counts == 2))

    def is_manifold(self) -> bool:
        """Edge manifold (at most two faces per edge) and vertex manifold (one fan per vertex)."""
        _, _, _, counts = self._edge_counts()
        if np.any(counts > 2):
            return False
        # vertex manifold: faces around each vertex form one connected fan
        from collections import defaultdict

        vert_faces = defaultdict(list)
        for fi, f in enumerate(self.faces):
            for v in f:
                vert_faces[int(v)].append(fi)
        F = self.faces
        for v, fl in vert_faces.items():
            if len(fl) <= 1:
                continue
            # union find on faces sharing an edge incident to v
            parent = {f: f for f in fl}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            edge_to_face = {}
            for f in fl:
                tri = F[f]
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    if a == v or b == v:
                        other = b if a == v else a
                        if other in edge_to_face:
                            ra, rb = find(f), find(edge_to_face[other])
                            parent[ra] = rb
                        else:
                            edge_to_face[other] = f
            roots = {find(f) for f in fl}
            if len(roots) > 1:
                return False
        return True

    def is_consistently_oriented(self) -> bool:
        """Each shared edge is traversed in opposite directions by its two faces."""
        e = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        key = e[:, 0] * (self.n_vertices + 1) + e[:, 1]
        rev = e[:, 1] * (self.n_vertices + 1) + e[:, 0]
        s = set(key.tolist())
        return all(r in s for r in rev.tolist()) and len(s) == len(key)

    def fix_orientation(self) -> "TriMesh":
        """Flip all faces if the signed volume is negative (outward normals)."""
        if self.volume() < 0:
            self.faces = self.faces[:, [0, 2, 1]].copy()
            self.invalidate()
        return self

    def self_intersects(self) -> bool:
        """Edge against non adjacent triangle test using the BVH (free of self intersections if False)."""
        bvh = self.bvh()
        t = self.triangles()
        F = self.faces
        o = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
        dvec = np.concatenate([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]])
        L = np.linalg.norm(dvec, axis=1)
        d = dvec / np.maximum(L[:, None], 1e-300)
        from .raycast import _batch_intersect_excluding

        face_of_edge = np.concatenate([np.arange(self.n_faces)] * 3)
        ok = _batch_intersect_excluding(o, d, L * (1 - 1e-6), F, face_of_edge, *bvh.arrays(), bvh.order)
        return bool(ok.any())

    def is_valid(self) -> dict:
        return {
            "watertight": self.is_watertight(),
            "manifold": self.is_manifold(),
            "oriented": self.is_consistently_oriented() and self.volume() > 0,
            "self_intersection_free": not self.self_intersects(),
        }

    # ------------------------------------------------------------------ geometry
    def convex_hull(self) -> "TriMesh":
        hull = ConvexHull(self.vertices)
        V = self.vertices[hull.vertices]
        remap = -np.ones(self.n_vertices, dtype=np.int64)
        remap[hull.vertices] = np.arange(len(hull.vertices))
        F = remap[hull.simplices]
        m = TriMesh(V, F, self.name + "_hull")
        # orient outward: each face normal should point away from the centroid
        c = V.mean(axis=0)
        n = m.face_normals()
        tc = m.triangles().mean(axis=1)
        flip = np.einsum("ij,ij->i", n, tc - c) < 0
        F[flip] = F[flip][:, [0, 2, 1]]
        m.faces = F
        m.invalidate()
        return m

    def convexity(self) -> float:
        return self.volume() / self.convex_hull().volume()

    def sphericity(self) -> float:
        V = self.volume()
        A = self.surface_area()
        return float(np.pi ** (1 / 3) * (6 * V) ** (2 / 3) / A)

    def bvh(self):
        if self._bvh is None:
            from .raycast import BVH

            self._bvh = BVH(self.vertices, self.faces)
        return self._bvh

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Inside test by ray parity (watertight meshes)."""
        p = np.atleast_2d(points)
        d = np.tile(np.array([0.577350269, 0.57735, 0.577351]), (len(p), 1))
        n = self.bvh().count_hits(p, d)
        return (n % 2) == 1

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(points)
        dist, _, _ = self.bvh().closest_points(p)
        inside = self.contains(p)
        return np.where(inside, -dist, dist)

    def sample_surface(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Uniform surface samples with face normals."""
        a = self.face_areas()
        p = a / a.sum()
        fi = rng.choice(self.n_faces, size=n, p=p)
        r1 = np.sqrt(rng.random(n))
        r2 = rng.random(n)
        t = self.triangles()[fi]
        pts = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
        return pts, self.face_normals()[fi]

    # ------------------------------------------------------- local operators
    def vertex_adjacency(self) -> list[np.ndarray]:
        from collections import defaultdict

        adj = defaultdict(set)
        for a, b in self.edges():
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        return [np.array(sorted(adj[i]), dtype=np.int64) for i in range(self.n_vertices)]

    def laplacian_umbrella(self) -> np.ndarray:
        """Uniform (umbrella) Laplacian displacement for each vertex."""
        e = self.edges()
        disp = np.zeros_like(self.vertices)
        cnt = np.zeros(self.n_vertices)
        np.add.at(disp, e[:, 0], self.vertices[e[:, 1]] - self.vertices[e[:, 0]])
        np.add.at(disp, e[:, 1], self.vertices[e[:, 0]] - self.vertices[e[:, 1]])
        np.add.at(cnt, e[:, 0], 1)
        np.add.at(cnt, e[:, 1], 1)
        return disp / np.maximum(cnt, 1)[:, None]

    def taubin_smooth(self, iterations: int = 10, lam: float = 0.5, mu: float = -0.53) -> "TriMesh":
        """Taubin lambda/mu smoothing (volume preserving in the mean)."""
        m = self.copy()
        for _ in range(iterations):
            m.vertices = m.vertices + lam * m.laplacian_umbrella()
            m.vertices = m.vertices + mu * m.laplacian_umbrella()
        m.invalidate()
        return m

    def mean_curvature(self) -> np.ndarray:
        """Discrete mean curvature H at vertices from the cotangent Laplacian (Meyer et al. 2003).

        Positive for convex regions when normals point outward.
        """
        V, F = self.vertices, self.faces
        n = self.n_vertices
        Lsum = np.zeros((n, 3))
        area = np.zeros(n)
        for k in range(3):
            i = F[:, k]
            j = F[:, (k + 1) % 3]
            l = F[:, (k + 2) % 3]
            # angle at l opposite edge (i, j)
            a = V[i] - V[l]
            b = V[j] - V[l]
            cos = np.einsum("ij,ij->i", a, b)
            sin = np.linalg.norm(np.cross(a, b), axis=1)
            cot = cos / np.maximum(sin, 1e-12)
            w = 0.5 * cot
            np.add.at(Lsum, i, w[:, None] * (V[j] - V[i]))
            np.add.at(Lsum, j, w[:, None] * (V[i] - V[j]))
        fa = self.face_areas()
        for k in range(3):
            np.add.at(area, F[:, k], fa / 3.0)
        Hn = -Lsum / (2.0 * np.maximum(area, 1e-18)[:, None])  # mean curvature normal, outward positive for convex
        vn = self.vertex_normals()
        return np.einsum("ij,ij->i", Hn, vn)

    def subdivide(self) -> "TriMesh":
        """Midpoint subdivision (each face into four)."""
        V, F = self.vertices, self.faces
        e, uniq, inv, _ = self._edge_counts()
        mid = 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])
        nv = V.shape[0]
        m01 = inv[: F.shape[0]] + nv
        m12 = inv[F.shape[0] : 2 * F.shape[0]] + nv
        m20 = inv[2 * F.shape[0] :] + nv
        newF = np.concatenate(
            [
                np.stack([F[:, 0], m01, m20], 1),
                np.stack([F[:, 1], m12, m01], 1),
                np.stack([F[:, 2], m20, m12], 1),
                np.stack([m01, m12, m20], 1),
            ]
        )
        return TriMesh(np.concatenate([V, mid]), newF, self.name)

    def merge_close_vertices(self, tol: float = 1e-9) -> "TriMesh":
        q = np.round(self.vertices / tol).astype(np.int64)
        _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        V = self.vertices[idx]
        F = inv.reshape(-1)[self.faces]
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        return TriMesh(V, F[keep], self.name)

    # ---------------------------------------------------------------------- io
    def save(self, path: str | os.PathLike):
        path = str(path)
        if path.endswith(".npz"):
            np.savez_compressed(path, vertices=self.vertices, faces=self.faces, name=self.name)
        elif path.endswith(".obj"):
            with open(path, "w") as f:
                f.write(f"o {self.name}\n")
                for v in self.vertices:
                    f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
                for t in self.faces + 1:
                    f.write(f"f {t[0]} {t[1]} {t[2]}\n")
        else:
            raise ValueError("unsupported mesh format: " + path)

    @staticmethod
    def load(path: str | os.PathLike) -> "TriMesh":
        path = str(path)
        if path.endswith(".npz"):
            d = np.load(path)
            return TriMesh(d["vertices"], d["faces"], str(d["name"]) if "name" in d else "mesh")
        if path.endswith(".obj"):
            V, F = [], []
            with open(path) as f:
                for line in f:
                    if line.startswith("v "):
                        V.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith("f "):
                        idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                        for k in range(1, len(idx) - 1):
                            F.append([idx[0], idx[k], idx[k + 1]])
            return TriMesh(np.array(V), np.array(F))
        try:
            import trimesh

            tm = trimesh.load(path, force="mesh")
            return TriMesh(np.asarray(tm.vertices), np.asarray(tm.faces))
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"cannot load {path}: {exc}")

    def to_trimesh(self):
        import trimesh

        return trimesh.Trimesh(self.vertices.copy(), self.faces.copy(), process=False)

    def __repr__(self) -> str:
        return f"TriMesh({self.name}, V={self.n_vertices}, F={self.n_faces})"


# -------------------------------------------------------------------- primitives


def box(extents=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)) -> TriMesh:
    e = np.asarray(extents, dtype=np.float64) / 2
    c = np.asarray(center, dtype=np.float64)
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64) * e + c
    # vertex index = 4*x + 2*y + z with (0/1)
    F = np.array(
        [
            [0, 1, 3], [0, 3, 2],  # x = -1 face (normal -x)
            [4, 7, 5], [4, 6, 7],  # x = +1
            [0, 4, 5], [0, 5, 1],  # y = -1
            [2, 3, 7], [2, 7, 6],  # y = +1
            [0, 2, 6], [0, 6, 4],  # z = -1
            [1, 5, 7], [1, 7, 3],  # z = +1
        ],
        dtype=np.int64,
    )
    m = TriMesh(V, F, "box")
    return m.fix_orientation()


def icosphere(subdivisions: int = 3, radius: float = 1.0) -> TriMesh:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    m = TriMesh(V / np.linalg.norm(V, axis=1, keepdims=True), F, "icosphere")
    for _ in range(subdivisions):
        m = m.subdivide()
        m.vertices /= np.linalg.norm(m.vertices, axis=1, keepdims=True)
    m.vertices *= radius
    m.invalidate()
    return m.fix_orientation()


def tetrahedron(scale: float = 1.0) -> TriMesh:
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64) * scale
    F = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    return TriMesh(V, F, "tetrahedron").fix_orientation()


def cylinder(radius: float = 1.0, height: float = 1.0, n: int = 32) -> TriMesh:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring_b = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n, -height / 2)], 1)
    ring_t = ring_b + np.array([0, 0, height])
    V = np.concatenate([ring_b, ring_t, [[0, 0, -height / 2]], [[0, 0, height / 2]]])
    cb, ct = 2 * n, 2 * n + 1
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [cb, j, i], [ct, n + i, n + j]]
    return TriMesh(V, np.array(F), "cylinder").fix_orientation()


def plane(size: float = 10.0, z: float = 0.0) -> TriMesh:
    s = size / 2
    V = np.array([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]])
    F = np.array([[0, 1, 2], [0, 2, 3]])
    return TriMesh(V, F, "plane")


def concatenate(meshes: list[TriMesh]) -> TriMesh:
    Vs, Fs, off = [], [], 0
    for m in meshes:
        Vs.append(m.vertices)
        Fs.append(m.faces + off)
        off += m.n_vertices
    return TriMesh(np.concatenate(Vs), np.concatenate(Fs), "concat")


# -------------------------------------------------------------------------- SDF


@dataclass
class SDF:
    """Signed distance field on a regular grid (negative inside)."""

    origin: np.ndarray
    spacing: float
    values: np.ndarray  # (nx, ny, nz)

    @property
    def shape(self):
        return self.values.shape

    @staticmethod
    def from_mesh(mesh: TriMesh, spacing: float = 0.001, padding: float = 0.01) -> "SDF":
        lo, hi = mesh.bounds()
        lo = lo - padding
        hi = hi + padding
        n = np.ceil((hi - lo) / spacing).astype(int) + 1
        xs = lo[0] + spacing * np.arange(n[0])
        ys = lo[1] + spacing * np.arange(n[1])
        zs = lo[2] + spacing * np.arange(n[2])
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        d = mesh.signed_distance(P)
        return SDF(lo, spacing, d.reshape(n[0], n[1], n[2]))

    def query(self, points: np.ndarray) -> np.ndarray:
        """Trilinear interpolation; outside the grid returns distance to the grid box plus edge value."""
        p = np.atleast_2d(points)
        g = (p - self.origin) / self.spacing
        n = np.array(self.shape)
        gc = np.clip(g, 0, n - 1 - 1e-9)
        i0 = np.floor(gc).astype(int)
        f = gc - i0
        i1 = np.minimum(i0 + 1, n - 1)
        v = self.values
        c00 = v[i0[:, 0], i0[:, 1], i0[:, 2]] * (1 - f[:, 0]) + v[i1[:, 0], i0[:, 1], i0[:, 2]] * f[:, 0]
        c01 = v[i0[:, 0], i0[:, 1], i1[:, 2]] * (1 - f[:, 0]) + v[i1[:, 0], i0[:, 1], i1[:, 2]] * f[:, 0]
        c10 = v[i0[:, 0], i1[:, 1], i0[:, 2]] * (1 - f[:, 0]) + v[i1[:, 0], i1[:, 1], i0[:, 2]] * f[:, 0]
        c11 = v[i0[:, 0], i1[:, 1], i1[:, 2]] * (1 - f[:, 0]) + v[i1[:, 0], i1[:, 1], i1[:, 2]] * f[:, 0]
        c0 = c00 * (1 - f[:, 1]) + c10 * f[:, 1]
        c1 = c01 * (1 - f[:, 1]) + c11 * f[:, 1]
        val = c0 * (1 - f[:, 2]) + c1 * f[:, 2]
        # outside correction
        outside = np.linalg.norm((g - gc) * self.spacing, axis=1)
        return val + outside

    def gradient(self, points: np.ndarray, h: float | None = None) -> np.ndarray:
        h = self.spacing if h is None else h
        p = np.atleast_2d(points)
        g = np.zeros_like(p)
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            g[:, k] = (self.query(p + e) - self.query(p - e)) / (2 * h)
        return g

    def save(self, path):
        np.savez_compressed(path, origin=self.origin, spacing=self.spacing, values=self.values)

    @staticmethod
    def load(path) -> "SDF":
        d = np.load(path)
        return SDF(d["origin"], float(d["spacing"]), d["values"])


# ------------------------------------------------------------ convex decomposition


def convex_pieces(mesh: TriMesh, convexity_threshold: float = 0.97, max_pieces: int = 8) -> list[TriMesh]:
    """Convex decomposition for collision (Section 5.4).

    Stones are nearly convex; if V / V_hull exceeds the threshold the hull is the single piece.
    Otherwise VHACD through trimesh is used when available, with the hull as the fallback.
    """
    hull = mesh.convex_hull()
    if mesh.volume() / hull.volume() >= convexity_threshold:
        return [hull]
    try:
        import trimesh

        tm = mesh.to_trimesh()
        parts = trimesh.decomposition.convex_decomposition(tm, maxConvexHulls=max_pieces)
        out = []
        for p in parts:
            if isinstance(p, dict):
                m = TriMesh(p["vertices"], p["faces"])
            else:
                m = TriMesh(np.asarray(p.vertices), np.asarray(p.faces))
            if m.n_faces >= 4:
                out.append(m.convex_hull())
        if out:
            return out
    except Exception:
        pass
    return [hull]
