"""Motion planning in joint space (Section 9.5): RRT-Connect (Kuffner and LaValle 2000),
RRT* (Karaman and Frazzoli 2011), shortcut smoothing and time parameterisation under velocity
limits. Collision checking uses link capsules against obstacle meshes inflated by the pose and
shape uncertainty."""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh


def _interpolate(a, b, resolution):
    n = max(int(np.ceil(np.linalg.norm(b - a) / resolution)), 1)
    return [a + (b - a) * (k / n) for k in range(1, n + 1)]


def edge_collision_free(a, b, collision_fn, resolution: float = 0.05) -> bool:
    return all(not collision_fn(q) for q in _interpolate(a, b, resolution))


def path_collision_free(path, collision_fn, resolution: float = 0.05) -> bool:
    if collision_fn(path[0]):
        return False
    return all(edge_collision_free(path[i], path[i + 1], collision_fn, resolution) for i in range(len(path) - 1))


class _Tree:
    def __init__(self, root):
        self.nodes = [np.asarray(root, dtype=np.float64)]
        self.parent = [-1]

    def nearest(self, q):
        d = np.linalg.norm(np.array(self.nodes) - q, axis=1)
        return int(np.argmin(d))

    def add(self, q, parent):
        self.nodes.append(np.asarray(q, dtype=np.float64))
        self.parent.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, i):
        out = []
        while i >= 0:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out


class RRTConnect:
    """Bidirectional RRT with greedy connect heuristic."""

    def __init__(self, sample_fn, collision_fn, q_min, q_max, step: float = 0.15, max_iters: int = 5000, rng=None, check_resolution: float = 0.05):
        self.sample_fn = sample_fn
        self.collision_fn = collision_fn
        self.q_min, self.q_max = np.asarray(q_min, dtype=np.float64), np.asarray(q_max, dtype=np.float64)
        self.step = step
        self.max_iters = max_iters
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.res = check_resolution
        self.iterations = 0

    def _extend(self, tree: _Tree, q):
        i = tree.nearest(q)
        qn = tree.nodes[i]
        d = np.linalg.norm(q - qn)
        if d < 1e-12:
            return "reached", i
        qnew = q if d <= self.step else qn + (q - qn) * (self.step / d)
        if not edge_collision_free(qn, qnew, self.collision_fn, self.res):
            return "trapped", i
        j = tree.add(qnew, i)
        return ("reached" if d <= self.step else "advanced"), j

    def _connect(self, tree: _Tree, q):
        status = "advanced"
        j = -1
        while status == "advanced":
            status, j = self._extend(tree, q)
        return status, j

    def plan(self, q0, q1):
        q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
        if self.collision_fn(q0) or self.collision_fn(q1):
            return None
        if edge_collision_free(q0, q1, self.collision_fn, self.res):
            return [q0, q1]
        ta, tb = _Tree(q0), _Tree(q1)
        a_is_start = True
        for it in range(self.max_iters):
            self.iterations = it + 1
            qr = np.asarray(self.sample_fn(self.rng), dtype=np.float64)
            status, i = self._extend(ta, qr)
            if status != "trapped":
                status2, j = self._connect(tb, ta.nodes[i])
                if status2 == "reached":
                    pa = ta.path_to_root(i)[::-1]
                    pb = tb.path_to_root(j)
                    path = pa + pb if a_is_start else pb[::-1] + pa[::-1]
                    return [np.asarray(p) for p in path]
            ta, tb = tb, ta
            a_is_start = not a_is_start
        return None


class RRTStar:
    """RRT* with k nearest rewiring; returns the best path found within ``max_iters``."""

    def __init__(self, sample_fn, collision_fn, q_min, q_max, step: float = 0.15, max_iters: int = 2000, rng=None, check_resolution: float = 0.05, goal_bias: float = 0.1, goal_tol: float = 0.05):
        self.sample_fn = sample_fn
        self.collision_fn = collision_fn
        self.q_min, self.q_max = np.asarray(q_min, dtype=np.float64), np.asarray(q_max, dtype=np.float64)
        self.step, self.max_iters = step, max_iters
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.res, self.goal_bias, self.goal_tol = check_resolution, goal_bias, goal_tol

    def plan(self, q0, q1):
        q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
        if self.collision_fn(q0) or self.collision_fn(q1):
            return None
        nodes = [q0]
        parent = [-1]
        cost = [0.0]
        goal_nodes = []
        n = q0.size
        for it in range(self.max_iters):
            qr = q1 if self.rng.random() < self.goal_bias else np.asarray(self.sample_fn(self.rng), dtype=np.float64)
            arr = np.array(nodes)
            d = np.linalg.norm(arr - qr, axis=1)
            i = int(np.argmin(d))
            qn = nodes[i]
            dist = d[i]
            if dist < 1e-12:
                continue
            qnew = qr if dist <= self.step else qn + (qr - qn) * (self.step / dist)
            if self.collision_fn(qnew):
                continue
            # neighbours
            card = len(nodes)
            r = min(self.step * 3.0, (np.log(card + 1) / (card + 1)) ** (1.0 / n) * 4.0 * self.step + self.step)
            dn = np.linalg.norm(arr - qnew, axis=1)
            near = np.nonzero(dn <= r)[0]
            best_parent, best_cost = i, cost[i] + dist if dist <= self.step else cost[i] + self.step
            if not edge_collision_free(qn, qnew, self.collision_fn, self.res):
                continue
            for j in near:
                c = cost[j] + dn[j]
                if c < best_cost and edge_collision_free(nodes[j], qnew, self.collision_fn, self.res):
                    best_parent, best_cost = int(j), c
            nodes.append(qnew)
            parent.append(best_parent)
            cost.append(best_cost)
            k = len(nodes) - 1
            # rewire
            for j in near:
                c = best_cost + dn[j]
                if c < cost[j] and edge_collision_free(qnew, nodes[j], self.collision_fn, self.res):
                    parent[j] = k
                    cost[j] = c
            if np.linalg.norm(qnew - q1) <= self.goal_tol and edge_collision_free(qnew, q1, self.collision_fn, self.res):
                goal_nodes.append((best_cost + np.linalg.norm(qnew - q1), k))
        if not goal_nodes:
            return None
        goal_nodes.sort()
        k = goal_nodes[0][1]
        path = []
        while k >= 0:
            path.append(nodes[k])
            k = parent[k]
        path = path[::-1]
        if np.linalg.norm(path[-1] - q1) > 1e-12:
            path.append(q1)
        return path


def shortcut(path, collision_fn, rng=None, iters: int = 100, resolution: float = 0.05):
    """Random shortcutting that keeps the path collision free."""
    rng = np.random.default_rng(0) if rng is None else rng
    path = [np.asarray(p, dtype=np.float64) for p in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if edge_collision_free(path[i], path[j], collision_fn, resolution):
            path = path[: i + 1] + path[j:]
    return path


def time_parameterise(path, dq_max, ddq_max=None):
    """Piecewise linear joint trajectory limited by velocity (and optionally acceleration)
    through trapezoidal segment durations. Returns (knot times, q(t))."""
    path = [np.asarray(p, dtype=np.float64) for p in path]
    dq_max = np.asarray(dq_max, dtype=np.float64)
    times = [0.0]
    for a, b in zip(path[:-1], path[1:]):
        delta = np.abs(b - a)
        T = float(np.max(delta / dq_max)) if np.any(delta > 0) else 0.0
        if ddq_max is not None:
            T = max(T, float(np.max(2 * np.sqrt(delta / np.asarray(ddq_max)))))
        times.append(times[-1] + max(T, 1e-9))
    times = np.array(times)
    P = np.array(path)

    def q_of_t(t):
        t = float(np.clip(t, times[0], times[-1]))
        k = int(np.searchsorted(times, t, side="right") - 1)
        k = min(max(k, 0), len(times) - 2)
        u = (t - times[k]) / max(times[k + 1] - times[k], 1e-12)
        # smooth (minimum jerk style) blending inside each segment keeps velocities continuous at rest
        s = u * u * (3 - 2 * u)
        return P[k] + s * (P[k + 1] - P[k])

    return times, q_of_t


def make_arm_collision_fn(arm, obstacles: list[TriMesh], ground_z: float = 0.0, inflate: float = 0.0, n_axis_samples: int = 5, check_self: bool = False, attached_points=None):
    """Collision test of the arm's link capsules against obstacle meshes and the ground.

    Points along each capsule axis are tested with ``TriMesh.signed_distance``; the link is in
    collision if any distance is below radius + inflate, or below the ground. ``attached_points``
    (points of a held object in the tool frame) are tested too, against the obstacles with the
    inflation and against the ground.
    """
    offs = np.linspace(0, 1, n_axis_samples)
    held = None if attached_points is None else np.asarray(attached_points, dtype=float)

    def collides(q) -> bool:
        if held is not None:
            P = arm.fk(q).apply(held)
            if np.any(P[:, 2] < ground_z + 0.002):
                return True
            for m in obstacles:
                lo, hi = m.bounds()
                if np.any(P.max(axis=0) < lo - inflate) or np.any(P.min(axis=0) > hi + inflate):
                    continue
                if np.any(m.signed_distance(P) < inflate):
                    return True
        caps = arm.world_capsules(q)
        pts, rads = [], []
        for p0, p1, r in caps:
            for u in offs:
                pts.append(p0 + u * (p1 - p0))
                rads.append(r)
        pts = np.array(pts)
        rads = np.array(rads)
        # skip the base link (first capsule) for the ground test
        n0 = n_axis_samples
        if np.any(pts[n0:, 2] - rads[n0:] < ground_z):
            return True
        for m in obstacles:
            lo, hi = m.bounds()
            if np.any(pts.max(axis=0) < lo - 0.2) or np.any(pts.min(axis=0) > hi + 0.2):
                continue
            d = m.signed_distance(pts)
            if np.any(d < rads + inflate):
                return True
        return False

    return collides
