"""Bounding volume hierarchy, ray casting and closest point queries (numba kernels).

Shared by the simulated depth sensor (Section 8.1), the signed distance fields
(Section 5.4), narrow phase verification (Section 6.4) and the path tracer
(Section 10.1). The BVH is built top down with a binned surface area heuristic.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

INF = np.inf
MAX_SAH_DEPTH = 40  # beyond this depth the build uses median splits (kernel stacks hold 128 entries)
STACK = 128


# ----------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------


class BVH:
    """Flattened BVH over a triangle soup.

    Arrays
    ------
    bmin, bmax : (n_nodes, 3) node bounds
    left, right : (n_nodes,) child indices, -1 for leaves
    start, count : (n_nodes,) triangle range in ``order`` for leaves (count = 0 for interior nodes)
    order : (n_tris,) permutation of triangle ids
    tri_v0, tri_e1, tri_e2 : (n_tris, 3) precomputed triangle data in ``order``
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, leaf_size: int = 4):
        V = np.ascontiguousarray(vertices, dtype=np.float64)
        F = np.ascontiguousarray(faces, dtype=np.int64)
        self.vertices = V
        self.faces = F
        n = F.shape[0]
        if n == 0:
            raise ValueError("BVH needs at least one triangle")
        tri = V[F]
        tmin = tri.min(axis=1)
        tmax = tri.max(axis=1)
        cent = tri.mean(axis=1)
        order = np.arange(n, dtype=np.int64)
        nodes_bmin, nodes_bmax, left, right, start, count = [], [], [], [], [], []
        stack = [(0, n, -1, False, 0)]
        while stack:
            lo, hi, parent, is_left, depth = stack.pop()
            idx = order[lo:hi]
            bmin = tmin[idx].min(axis=0)
            bmax = tmax[idx].max(axis=0)
            node_id = len(nodes_bmin)
            nodes_bmin.append(bmin)
            nodes_bmax.append(bmax)
            left.append(-1)
            right.append(-1)
            start.append(lo)
            count.append(hi - lo)
            if parent >= 0:
                if is_left:
                    left[parent] = node_id
                else:
                    right[parent] = node_id
            if hi - lo <= leaf_size:
                continue
            if depth >= MAX_SAH_DEPTH:
                # deep subtree (clustered centroids): force balanced median splits so that the
                # traversal stack of the kernels is never exceeded
                axis = int(np.argmax(bmax - bmin))
                split = _partition_by_order(idx, cent[idx, axis], (hi - lo) // 2)
            else:
                split = self._sah_split(idx, cent, tmin, tmax, bmin, bmax)
            if split is None or split <= 0 or split >= hi - lo:
                continue
            count[node_id] = 0
            mid = lo + int(split)
            stack.append((mid, hi, node_id, False, depth + 1))
            stack.append((lo, mid, node_id, True, depth + 1))
        self.bmin = np.ascontiguousarray(np.array(nodes_bmin, dtype=np.float64))
        self.bmax = np.ascontiguousarray(np.array(nodes_bmax, dtype=np.float64))
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.start = np.array(start, dtype=np.int64)
        self.count = np.array(count, dtype=np.int64)
        self.order = order
        t = tri[order]
        self.tri_v0 = np.ascontiguousarray(t[:, 0])
        self.tri_e1 = np.ascontiguousarray(t[:, 1] - t[:, 0])
        self.tri_e2 = np.ascontiguousarray(t[:, 2] - t[:, 0])

    @staticmethod
    def _sah_split(idx, cent, tmin, tmax, bmin, bmax, n_bins: int = 12):
        """Partition ``idx`` in place and return the local split offset (or None)."""
        ext = bmax - bmin
        axis = int(np.argmax(ext))
        n = len(idx)
        if ext[axis] <= 0 or n < 2:
            return None
        c = cent[idx, axis]
        cmin, cmax = c.min(), c.max()
        if cmax - cmin <= 1e-12:
            return _partition_by_order(idx, c, n // 2)
        bins = np.minimum(((c - cmin) / (cmax - cmin) * n_bins).astype(np.int64), n_bins - 1)
        counts = np.bincount(bins, minlength=n_bins)
        bb_min = np.full((n_bins, 3), INF)
        bb_max = np.full((n_bins, 3), -INF)
        for b in range(n_bins):
            m = bins == b
            if counts[b] > 0:
                bb_min[b] = tmin[idx[m]].min(axis=0)
                bb_max[b] = tmax[idx[m]].max(axis=0)
        best_cost, best_b = INF, -1
        for b in range(1, n_bins):
            nl = counts[:b].sum()
            nr = counts[b:].sum()
            if nl == 0 or nr == 0:
                continue
            cost = nl * _area(bb_min[:b].min(axis=0), bb_max[:b].max(axis=0)) + nr * _area(bb_min[b:].min(axis=0), bb_max[b:].max(axis=0))
            if cost < best_cost:
                best_cost, best_b = cost, b
        if best_b < 0 or best_cost >= n * _area(bmin, bmax):
            if n <= 4:
                return None
            return _partition_by_order(idx, c, n // 2)
        left_mask = bins < best_b
        nl = int(left_mask.sum())
        idx[:] = np.concatenate([idx[left_mask], idx[~left_mask]])
        return nl

    def arrays(self):
        return (self.bmin, self.bmax, self.left, self.right, self.start, self.count, self.tri_v0, self.tri_e1, self.tri_e2)

    # convenience wrappers ---------------------------------------------------------
    def intersect(self, origins: np.ndarray, dirs: np.ndarray, tmin: float = 1e-6, tmax: float = INF):
        """Closest hit per ray. Returns (t, tri_id in original face indexing, u, v); tri_id = -1 for miss."""
        o = np.ascontiguousarray(np.atleast_2d(origins), dtype=np.float64)
        d = np.ascontiguousarray(np.atleast_2d(dirs), dtype=np.float64)
        t, tid, u, v = _batch_intersect(o, d, tmin, tmax, *self.arrays())
        hit = tid >= 0
        tid_orig = np.where(hit, self.order[np.maximum(tid, 0)], -1)
        return t, tid_orig, u, v

    def occluded(self, origins: np.ndarray, dirs: np.ndarray, tmin: float = 1e-6, tmax: float = INF) -> np.ndarray:
        o = np.ascontiguousarray(np.atleast_2d(origins), dtype=np.float64)
        d = np.ascontiguousarray(np.atleast_2d(dirs), dtype=np.float64)
        return _batch_any(o, d, tmin, tmax, *self.arrays())

    def count_hits(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Number of crossings along each ray (parity gives inside/outside for watertight meshes)."""
        o = np.ascontiguousarray(np.atleast_2d(origins), dtype=np.float64)
        d = np.ascontiguousarray(np.atleast_2d(dirs), dtype=np.float64)
        return _batch_count(o, d, *self.arrays())

    def closest_points(self, points: np.ndarray):
        """Unsigned distance, closest point and original triangle id for each query point."""
        p = np.ascontiguousarray(np.atleast_2d(points), dtype=np.float64)
        dist, cp, tid = _batch_closest(p, *self.arrays())
        return dist, cp, self.order[tid]


def _partition_by_order(idx, c, mid):
    perm = np.argsort(c, kind="stable")
    idx[:] = idx[perm]
    return mid


def _area(lo, hi):
    e = np.maximum(hi - lo, 0.0)
    return 2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])


# ----------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def ray_aabb(ox, oy, oz, idx_, idy, idz, bmin, bmax, tmin, tmax):
    t1 = (bmin[0] - ox) * idx_
    t2 = (bmax[0] - ox) * idx_
    lo = min(t1, t2)
    hi = max(t1, t2)
    t1 = (bmin[1] - oy) * idy
    t2 = (bmax[1] - oy) * idy
    lo = max(lo, min(t1, t2))
    hi = min(hi, max(t1, t2))
    t1 = (bmin[2] - oz) * idz
    t2 = (bmax[2] - oz) * idz
    lo = max(lo, min(t1, t2))
    hi = min(hi, max(t1, t2))
    return hi >= max(lo, tmin) and lo <= tmax


@njit(cache=True)
def ray_tri(o, d, v0, e1, e2, tmin, tmax):
    """Moller Trumbore. Returns (t, u, v) or (-1, 0, 0)."""
    px = d[1] * e2[2] - d[2] * e2[1]
    py = d[2] * e2[0] - d[0] * e2[2]
    pz = d[0] * e2[1] - d[1] * e2[0]
    det = e1[0] * px + e1[1] * py + e1[2] * pz
    if abs(det) < 1e-14:
        return -1.0, 0.0, 0.0
    inv = 1.0 / det
    sx = o[0] - v0[0]
    sy = o[1] - v0[1]
    sz = o[2] - v0[2]
    u = (sx * px + sy * py + sz * pz) * inv
    if u < 0.0 or u > 1.0:
        return -1.0, 0.0, 0.0
    qx = sy * e1[2] - sz * e1[1]
    qy = sz * e1[0] - sx * e1[2]
    qz = sx * e1[1] - sy * e1[0]
    v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return -1.0, 0.0, 0.0
    t = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv
    if t < tmin or t > tmax:
        return -1.0, 0.0, 0.0
    return t, u, v


@njit(cache=True)
def _safe_inv(x):
    if abs(x) < 1e-300:
        return 1e300 if x >= 0 else -1e300
    return 1.0 / x


@njit(cache=True)
def intersect_closest(o, d, tmin, tmax, bmin, bmax, left, right, start, count, tv0, te1, te2):
    """Closest hit. Returns (t, local_tri, u, v); local_tri = -1 on miss."""
    stack = np.empty(STACK, dtype=np.int64)
    sp = 0
    stack[sp] = 0
    sp += 1
    best_t = tmax
    best_i = -1
    best_u = 0.0
    best_v = 0.0
    idx_ = _safe_inv(d[0])
    idy = _safe_inv(d[1])
    idz = _safe_inv(d[2])
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not ray_aabb(o[0], o[1], o[2], idx_, idy, idz, bmin[node], bmax[node], tmin, best_t):
            continue
        if count[node] > 0:
            s = start[node]
            for k in range(s, s + count[node]):
                t, u, v = ray_tri(o, d, tv0[k], te1[k], te2[k], tmin, best_t)
                if t >= 0.0 and t < best_t:
                    best_t = t
                    best_i = k
                    best_u = u
                    best_v = v
        else:
            stack[sp] = left[node]
            sp += 1
            stack[sp] = right[node]
            sp += 1
    if best_i < 0:
        return -1.0, -1, 0.0, 0.0
    return best_t, best_i, best_u, best_v


@njit(cache=True)
def intersect_any(o, d, tmin, tmax, bmin, bmax, left, right, start, count, tv0, te1, te2):
    stack = np.empty(STACK, dtype=np.int64)
    sp = 0
    stack[sp] = 0
    sp += 1
    idx_ = _safe_inv(d[0])
    idy = _safe_inv(d[1])
    idz = _safe_inv(d[2])
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not ray_aabb(o[0], o[1], o[2], idx_, idy, idz, bmin[node], bmax[node], tmin, tmax):
            continue
        if count[node] > 0:
            s = start[node]
            for k in range(s, s + count[node]):
                t, u, v = ray_tri(o, d, tv0[k], te1[k], te2[k], tmin, tmax)
                if t >= 0.0:
                    return True
        else:
            stack[sp] = left[node]
            sp += 1
            stack[sp] = right[node]
            sp += 1
    return False


@njit(cache=True)
def intersect_count(o, d, bmin, bmax, left, right, start, count, tv0, te1, te2):
    stack = np.empty(STACK, dtype=np.int64)
    sp = 0
    stack[sp] = 0
    sp += 1
    idx_ = _safe_inv(d[0])
    idy = _safe_inv(d[1])
    idz = _safe_inv(d[2])
    n = 0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not ray_aabb(o[0], o[1], o[2], idx_, idy, idz, bmin[node], bmax[node], 0.0, 1e300):
            continue
        if count[node] > 0:
            s = start[node]
            for k in range(s, s + count[node]):
                t, u, v = ray_tri(o, d, tv0[k], te1[k], te2[k], 0.0, 1e300)
                if t >= 0.0:
                    n += 1
        else:
            stack[sp] = left[node]
            sp += 1
            stack[sp] = right[node]
            sp += 1
    return n


@njit(cache=True, parallel=True)
def _batch_intersect(o, d, tmin, tmax, bmin, bmax, left, right, start, count, tv0, te1, te2):
    n = o.shape[0]
    t = np.empty(n)
    tid = np.empty(n, dtype=np.int64)
    uu = np.empty(n)
    vv = np.empty(n)
    for i in prange(n):
        a, b, c, e = intersect_closest(o[i], d[i], tmin, tmax, bmin, bmax, left, right, start, count, tv0, te1, te2)
        t[i] = a
        tid[i] = b
        uu[i] = c
        vv[i] = e
    return t, tid, uu, vv


@njit(cache=True, parallel=True)
def _batch_any(o, d, tmin, tmax, bmin, bmax, left, right, start, count, tv0, te1, te2):
    n = o.shape[0]
    out = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        out[i] = intersect_any(o[i], d[i], tmin, tmax, bmin, bmax, left, right, start, count, tv0, te1, te2)
    return out


@njit(cache=True, parallel=True)
def _batch_count(o, d, bmin, bmax, left, right, start, count, tv0, te1, te2):
    n = o.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in prange(n):
        out[i] = intersect_count(o[i], d[i], bmin, bmax, left, right, start, count, tv0, te1, te2)
    return out


@njit(cache=True)
def _point_seg_closest(p, x0, y0, z0, x1, y1, z1):
    ex, ey, ez = x1 - x0, y1 - y0, z1 - z0
    ee = ex * ex + ey * ey + ez * ez
    if ee < 1e-300:
        tpar = 0.0
    else:
        tpar = ((p[0] - x0) * ex + (p[1] - y0) * ey + (p[2] - z0) * ez) / ee
        tpar = min(max(tpar, 0.0), 1.0)
    rx, ry, rz = x0 + tpar * ex, y0 + tpar * ey, z0 + tpar * ez
    return (p[0] - rx) ** 2 + (p[1] - ry) ** 2 + (p[2] - rz) ** 2, rx, ry, rz


@njit(cache=True)
def point_tri_closest(p, v0, e1, e2):
    """Closest point on triangle (Ericson, Real Time Collision Detection 5.1.5). Returns (dist2, cx, cy, cz).

    Degenerate (zero area) triangles fall back to the closest point on their edges.
    """
    ax, ay, az = v0[0], v0[1], v0[2]
    bx, by, bz = v0[0] + e1[0], v0[1] + e1[1], v0[2] + e1[2]
    cx, cy, cz = v0[0] + e2[0], v0[1] + e2[1], v0[2] + e2[2]
    nx = e1[1] * e2[2] - e1[2] * e2[1]
    ny = e1[2] * e2[0] - e1[0] * e2[2]
    nz = e1[0] * e2[1] - e1[1] * e2[0]
    area2 = nx * nx + ny * ny + nz * nz
    l1 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]
    l2 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]
    if area2 <= 1e-24 * max(l1 * l2, 1e-300):
        best, qx, qy, qz = _point_seg_closest(p, ax, ay, az, bx, by, bz)
        d2, rx, ry, rz = _point_seg_closest(p, bx, by, bz, cx, cy, cz)
        if d2 < best:
            best, qx, qy, qz = d2, rx, ry, rz
        d2, rx, ry, rz = _point_seg_closest(p, cx, cy, cz, ax, ay, az)
        if d2 < best:
            best, qx, qy, qz = d2, rx, ry, rz
        return best, qx, qy, qz
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = p[0] - ax, p[1] - ay, p[2] - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return (apx * apx + apy * apy + apz * apz), ax, ay, az
    bpx, bpy, bpz = p[0] - bx, p[1] - by, p[2] - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return (bpx * bpx + bpy * bpy + bpz * bpz), bx, by, bz
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        den = d1 - d3
        v = d1 / den if abs(den) > 1e-300 else 0.0
        qx, qy, qz = ax + v * abx, ay + v * aby, az + v * abz
        return ((p[0] - qx) ** 2 + (p[1] - qy) ** 2 + (p[2] - qz) ** 2), qx, qy, qz
    cpx, cpy, cpz = p[0] - cx, p[1] - cy, p[2] - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return (cpx * cpx + cpy * cpy + cpz * cpz), cx, cy, cz
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        den = d2 - d6
        w = d2 / den if abs(den) > 1e-300 else 0.0
        qx, qy, qz = ax + w * acx, ay + w * acy, az + w * acz
        return ((p[0] - qx) ** 2 + (p[1] - qy) ** 2 + (p[2] - qz) ** 2), qx, qy, qz
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        den = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / den if abs(den) > 1e-300 else 0.0
        qx, qy, qz = bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz)
        return ((p[0] - qx) ** 2 + (p[1] - qy) ** 2 + (p[2] - qz) ** 2), qx, qy, qz
    s = va + vb + vc
    if abs(s) < 1e-300:
        return _point_seg_closest(p, ax, ay, az, bx, by, bz)
    denom = 1.0 / s
    v = vb * denom
    w = vc * denom
    qx, qy, qz = ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w
    return ((p[0] - qx) ** 2 + (p[1] - qy) ** 2 + (p[2] - qz) ** 2), qx, qy, qz


@njit(cache=True)
def _aabb_dist2(p, lo, hi):
    d = 0.0
    for k in range(3):
        if p[k] < lo[k]:
            d += (lo[k] - p[k]) ** 2
        elif p[k] > hi[k]:
            d += (p[k] - hi[k]) ** 2
    return d


@njit(cache=True)
def closest_point(p, bmin, bmax, left, right, start, count, tv0, te1, te2):
    stack = np.empty(STACK, dtype=np.int64)
    sp = 0
    stack[sp] = 0
    sp += 1
    best = 1e300
    bi = -1
    bx = 0.0
    by = 0.0
    bz = 0.0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if _aabb_dist2(p, bmin[node], bmax[node]) > best:
            continue
        if count[node] > 0:
            s = start[node]
            for k in range(s, s + count[node]):
                d2, qx, qy, qz = point_tri_closest(p, tv0[k], te1[k], te2[k])
                if d2 < best:
                    best = d2
                    bi = k
                    bx, by, bz = qx, qy, qz
        else:
            l = left[node]
            r = right[node]
            dl = _aabb_dist2(p, bmin[l], bmax[l])
            dr = _aabb_dist2(p, bmin[r], bmax[r])
            # push farther first so nearer is popped first
            if dl < dr:
                stack[sp] = r
                sp += 1
                stack[sp] = l
                sp += 1
            else:
                stack[sp] = l
                sp += 1
                stack[sp] = r
                sp += 1
    return np.sqrt(best), bi, bx, by, bz


@njit(cache=True, parallel=True)
def _batch_closest(p, bmin, bmax, left, right, start, count, tv0, te1, te2):
    n = p.shape[0]
    dist = np.empty(n)
    cp = np.empty((n, 3))
    tid = np.empty(n, dtype=np.int64)
    for i in prange(n):
        d, k, x, y, z = closest_point(p[i], bmin, bmax, left, right, start, count, tv0, te1, te2)
        dist[i] = d
        tid[i] = k
        cp[i, 0] = x
        cp[i, 1] = y
        cp[i, 2] = z
    return dist, cp, tid


def brute_force_intersect(vertices, faces, origins, dirs, tmin=1e-6, tmax=INF):
    """Reference closest hit without acceleration, for BVH tests (Section 12.5)."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    tri = V[F]
    v0 = np.ascontiguousarray(tri[:, 0])
    e1 = np.ascontiguousarray(tri[:, 1] - tri[:, 0])
    e2 = np.ascontiguousarray(tri[:, 2] - tri[:, 0])
    o = np.atleast_2d(origins).astype(np.float64)
    d = np.atleast_2d(dirs).astype(np.float64)
    return _brute(o, d, tmin, tmax, v0, e1, e2)


@njit(cache=True)
def _brute(o, d, tmin, tmax, v0, e1, e2):
    n = o.shape[0]
    t_out = np.full(n, -1.0)
    id_out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        best = tmax
        bi = -1
        for k in range(v0.shape[0]):
            t, u, v = ray_tri(o[i], d[i], v0[k], e1[k], e2[k], tmin, best)
            if t >= 0.0 and t < best:
                best = t
                bi = k
        if bi >= 0:
            t_out[i] = best
            id_out[i] = bi
    return t_out, id_out


@njit(cache=True, parallel=True)
def _batch_intersect_excluding(o, d, tmax, faces, face_of_edge, bmin, bmax, left, right, start, count, tv0, te1, te2, order):
    """Self intersection helper: does edge i (from o[i] along d[i] for tmax[i]) cross a triangle that
    shares no vertex with the edge's own face?"""
    n = o.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        f = face_of_edge[i]
        a0, a1, a2 = faces[f, 0], faces[f, 1], faces[f, 2]
        stack = np.empty(STACK, dtype=np.int64)
        sp = 0
        stack[sp] = 0
        sp += 1
        idx_ = _safe_inv(d[i, 0])
        idy = _safe_inv(d[i, 1])
        idz = _safe_inv(d[i, 2])
        hit = False
        while sp > 0 and not hit:
            sp -= 1
            node = stack[sp]
            if not ray_aabb(o[i, 0], o[i, 1], o[i, 2], idx_, idy, idz, bmin[node], bmax[node], 1e-9, tmax[i]):
                continue
            if count[node] > 0:
                s = start[node]
                for k in range(s, s + count[node]):
                    g = order[k]
                    b0, b1, b2 = faces[g, 0], faces[g, 1], faces[g, 2]
                    shares = (b0 == a0) or (b0 == a1) or (b0 == a2) or (b1 == a0) or (b1 == a1) or (b1 == a2) or (b2 == a0) or (b2 == a1) or (b2 == a2)
                    if shares:
                        continue
                    t, u, v = ray_tri(o[i], d[i], tv0[k], te1[k], te2[k], 1e-9, tmax[i])
                    if t >= 0.0:
                        hit = True
                        break
            else:
                stack[sp] = left[node]
                sp += 1
                stack[sp] = right[node]
                sp += 1
        out[i] = hit
    return out
