"""Static tower description: point masses with contacts (Section 7.1).

A ``Tower`` holds the bodies (mass and centre of mass) and the contacts between them. The
ground is a body with ``is_ground=True``; it contributes no equilibrium equations. Contact
normals point from ``body_a`` towards ``body_b`` and the contact force acts on ``body_b``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import TriMesh, box


@dataclass
class StaticBody:
    id: int
    mass: float
    com: np.ndarray
    is_ground: bool = False
    name: str = ""

    def __post_init__(self):
        self.com = np.asarray(self.com, dtype=np.float64).reshape(3)
        self.mass = float(self.mass)


@dataclass
class StaticContact:
    body_a: int
    body_b: int
    point: np.ndarray
    normal: np.ndarray
    mu: float = 0.6

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=np.float64).reshape(3)
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        self.normal = n / max(np.linalg.norm(n), 1e-300)
        self.mu = float(self.mu)


def _farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    """Indices of ``k`` points chosen by farthest point sampling, starting from the extreme point."""
    n = len(points)
    if n <= k:
        return np.arange(n)
    c = points.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points - c, axis=1)))
    chosen = [first]
    d = np.linalg.norm(points - points[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def contacts_from_meshes(upper: TriMesh, lower: TriMesh, margin: float = 1.5e-3, max_points: int = 8, n_samples: int = 1500, rng=None):
    """Proximity contacts between two world frame meshes.

    Surface samples of ``upper`` (plus its vertices) whose signed distance to ``lower`` is below
    ``margin`` are contact candidates. The contact point is the closest point on ``lower`` and
    the normal is the outward normal of ``lower`` there (pointing into ``upper``). The set is
    reduced to ``max_points`` by farthest point sampling. Returns (points (k, 3), normals (k, 3)).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    pts = upper.vertices
    if n_samples > 0:
        s, _ = upper.sample_surface(n_samples, rng)
        pts = np.concatenate([pts, s])
    # cheap bounding box reject
    lo_u, hi_u = pts.min(axis=0), pts.max(axis=0)
    lo_l, hi_l = lower.bounds()
    if np.any(lo_u > hi_l + margin) or np.any(hi_u < lo_l - margin):
        return np.zeros((0, 3)), np.zeros((0, 3))
    bvh = lower.bvh()
    dist, cp, tid = bvh.closest_points(pts)
    inside = lower.contains(pts)
    sd = np.where(inside, -dist, dist)
    near = sd < margin
    if not np.any(near):
        return np.zeros((0, 3)), np.zeros((0, 3))
    cpts = cp[near]
    fn = lower.face_normals()[tid[near]]
    # merge duplicates on a 0.5 mm grid before reduction
    key = np.round(cpts / 5e-4).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    cpts, fn = cpts[idx], fn[idx]
    sel = _farthest_point_sample(cpts, max_points)
    return cpts[sel], fn[sel]


class Tower:
    """Bodies and contacts for the static analysis."""

    def __init__(self, bodies: list[StaticBody], contacts: list[StaticContact], gravity=(0.0, 0.0, -9.81)):
        self.bodies = list(bodies)
        self.contacts = list(contacts)
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3)
        self._index = {b.id: i for i, b in enumerate(self.bodies)}

    # ------------------------------------------------------------ properties
    @property
    def n_bodies(self) -> int:
        return len(self.bodies)

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)

    def body(self, body_id: int) -> StaticBody:
        return self.bodies[self._index[body_id]]

    def free_bodies(self) -> list[StaticBody]:
        return [b for b in self.bodies if not b.is_ground]

    def ground_ids(self) -> set:
        return {b.id for b in self.bodies if b.is_ground}

    def down(self) -> np.ndarray:
        g = self.gravity
        return g / max(np.linalg.norm(g), 1e-300)

    def height(self, p: np.ndarray) -> float:
        return float(-np.dot(np.asarray(p), self.down()))

    def ordered_free_bodies(self) -> list[StaticBody]:
        """Free bodies ordered by centre of mass height, bottom to top."""
        return sorted(self.free_bodies(), key=lambda b: self.height(b.com))

    def top_body(self) -> StaticBody:
        return self.ordered_free_bodies()[-1]

    def total_weight(self) -> float:
        return float(sum(b.mass for b in self.free_bodies()) * np.linalg.norm(self.gravity))

    def copy(self) -> "Tower":
        return Tower(
            [StaticBody(b.id, b.mass, b.com.copy(), b.is_ground, b.name) for b in self.bodies],
            [StaticContact(c.body_a, c.body_b, c.point.copy(), c.normal.copy(), c.mu) for c in self.contacts],
            self.gravity.copy(),
        )

    # ------------------------------------------------------------ constructors
    @staticmethod
    def from_world(world, margin: float | None = None, mu_override: dict | None = None) -> "Tower":
        """Tower from a physics ``World`` through ``topple.physics.collision.find_contacts``.

        ``mu_override`` maps (id_a, id_b) pairs or single body ids to friction coefficients.
        """
        from ..physics.collision import find_contacts

        bodies = []
        for b in world.bodies:
            is_ground = bool(b.static) or not np.isfinite(b.mass)
            bodies.append(StaticBody(b.id, 0.0 if is_ground else b.mass, np.array(b.x, dtype=np.float64), is_ground, b.name))
        contacts = []
        mu_override = mu_override or {}
        for c in find_contacts(world, margin):
            key = (c.body_a, c.body_b)
            if key in mu_override:
                mu = mu_override[key]
            elif (c.body_b, c.body_a) in mu_override:
                mu = mu_override[(c.body_b, c.body_a)]
            elif c.body_a in mu_override or c.body_b in mu_override:
                mu = np.sqrt(mu_override.get(c.body_a, c.mu) * mu_override.get(c.body_b, c.mu))
            else:
                mu = c.mu
            contacts.append(StaticContact(c.body_a, c.body_b, np.array(c.point), np.array(c.normal), mu))
        return Tower(bodies, contacts, np.array(world.gravity))

    @staticmethod
    def from_estimates(bodies: list, contacts: list[StaticContact], gravity=(0.0, 0.0, -9.81)) -> "Tower":
        """Tower from body dicts (keys id, mass, com, is_ground, name) or ``StaticBody`` objects."""
        out = []
        for b in bodies:
            if isinstance(b, StaticBody):
                out.append(b)
            else:
                out.append(StaticBody(int(b["id"]), float(b.get("mass", 0.0)), np.asarray(b["com"], dtype=np.float64), bool(b.get("is_ground", False)), str(b.get("name", ""))))
        return Tower(out, list(contacts), gravity)

    @staticmethod
    def from_meshes(
        meshes: list[TriMesh],
        masses: list[float],
        coms: list | None = None,
        mus: list[float] | None = None,
        ground_mu: float = 0.6,
        ground: bool = True,
        ground_size: float = 4.0,
        margin: float = 1.5e-3,
        max_points: int = 8,
        n_samples: int = 1500,
        rng=None,
        names: list[str] | None = None,
        gravity=(0.0, 0.0, -9.81),
        pair_mu: dict | None = None,
    ) -> "Tower":
        """Tower from world frame meshes with proximity contacts (``contacts_from_meshes``).

        Body ids: ground is 0, mesh k is k + 1. Pair friction is sqrt(mu_a mu_b) unless given in
        ``pair_mu`` keyed by (id_a, id_b).
        """
        rng = np.random.default_rng(0) if rng is None else rng
        n = len(meshes)
        coms = [m.centroid() for m in meshes] if coms is None else coms
        mus = [0.6] * n if mus is None else list(mus)
        names = [f"stone{k}" for k in range(n)] if names is None else names
        bodies = []
        world_meshes = []
        mu_of = {}
        if ground:
            bodies.append(StaticBody(0, 0.0, np.zeros(3), True, "ground"))
            world_meshes.append((0, box((ground_size, ground_size, 0.2), center=(0.0, 0.0, -0.1))))
            mu_of[0] = ground_mu
        for k, m in enumerate(meshes):
            bid = k + 1
            bodies.append(StaticBody(bid, masses[k], np.asarray(coms[k]), False, names[k]))
            world_meshes.append((bid, m))
            mu_of[bid] = mus[k]
        tower = Tower(bodies, [], gravity)
        hz = {bid: tower.height(np.asarray(m.centroid()) if bid != 0 else np.zeros(3)) for bid, m in world_meshes}
        pair_mu = pair_mu or {}
        contacts = []
        for i in range(len(world_meshes)):
            for j in range(i + 1, len(world_meshes)):
                ia, ma = world_meshes[i]
                ib, mb = world_meshes[j]
                # lower body is a, upper is b (normal from lower into upper)
                if hz[ia] > hz[ib]:
                    ia, ma, ib, mb = ib, mb, ia, ma
                pts, nrm = contacts_from_meshes(mb, ma, margin, max_points, n_samples, rng)
                if len(pts) == 0:
                    continue
                mu = pair_mu.get((ia, ib), pair_mu.get((ib, ia), float(np.sqrt(mu_of[ia] * mu_of[ib]))))
                for p, nn in zip(pts, nrm):
                    contacts.append(StaticContact(ia, ib, p, nn, mu))
        tower.contacts = contacts
        return tower

    # ------------------------------------------------------------ structure
    def interfaces(self) -> list[list[StaticContact]]:
        """Contacts grouped by body pair, ordered by mean contact height, bottom to top."""
        groups: dict = {}
        for c in self.contacts:
            key = (min(c.body_a, c.body_b), max(c.body_a, c.body_b))
            groups.setdefault(key, []).append(c)
        out = list(groups.values())
        out.sort(key=lambda cs: np.mean([self.height(c.point) for c in cs]))
        return out

    def upper_subassembly(self, cut_index: int) -> "Tower":
        """Subassembly above cut ``cut_index`` (Section 7.6).

        Free bodies are ordered by centre of mass height; cut k lies below the k-th free body.
        Every body below the cut becomes ground, contacts among lower bodies are dropped.
        """
        ordered = self.ordered_free_bodies()
        if not 0 <= cut_index < len(ordered):
            raise IndexError("cut index out of range")
        upper_ids = {b.id for b in ordered[cut_index:]}
        bodies = []
        for b in self.bodies:
            if b.id in upper_ids:
                bodies.append(StaticBody(b.id, b.mass, b.com.copy(), False, b.name))
            else:
                bodies.append(StaticBody(b.id, 0.0, b.com.copy(), True, b.name))
        contacts = [c for c in self.contacts if (c.body_a in upper_ids) or (c.body_b in upper_ids)]
        return Tower(bodies, contacts, self.gravity.copy())

    def n_cuts(self) -> int:
        return len(self.free_bodies())

    # ------------------------------------------------------------ test helpers
    def rotated_about_vertical(self, angle: float) -> "Tower":
        """Rotate the whole tower about the gravity axis through the origin (invariance tests)."""
        from ..core.transforms import quat_from_axis_angle, quat_rotate

        q = quat_from_axis_angle(-self.down(), angle)
        t = self.copy()
        for b in t.bodies:
            b.com = quat_rotate(q, b.com)
        for c in t.contacts:
            c.point = quat_rotate(q, c.point)
            c.normal = quat_rotate(q, c.normal)
        return t

    def scaled_mass(self, factor: float) -> "Tower":
        t = self.copy()
        for b in t.bodies:
            b.mass *= factor
        return t

    def with_mu(self, mu: float) -> "Tower":
        t = self.copy()
        for c in t.contacts:
            c.mu = mu
        return t

    def __repr__(self) -> str:
        return f"Tower(bodies={self.n_bodies}, free={len(self.free_bodies())}, contacts={self.n_contacts})"
