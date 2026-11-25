"""Quadric error edge collapse decimation (Garland and Heckbert 1997).

Used to bring marching cubes output (several triangles per surface voxel) down to the
500 to 3000 face range of the stone meshes. Collapses that would break the manifold
(link condition) or flip a face normal are rejected, so a valid closed input stays valid.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit

from ..core.mesh import TriMesh


@njit(cache=True)
def _quad_cost(Qs, x):
    c = 0.0
    h = np.empty(4)
    h[0] = x[0]
    h[1] = x[1]
    h[2] = x[2]
    h[3] = 1.0
    for i in range(4):
        for j in range(4):
            c += h[i] * Qs[i, j] * h[j]
    return c


@njit(cache=True)
def _optimal_jit(Qa, Qb, pa, pb, use_opt):
    Qs = Qa + Qb
    a00, a01, a02 = Qs[0, 0], Qs[0, 1], Qs[0, 2]
    a10, a11, a12 = Qs[1, 0], Qs[1, 1], Qs[1, 2]
    a20, a21, a22 = Qs[2, 0], Qs[2, 1], Qs[2, 2]
    b0, b1, b2 = -Qs[0, 3], -Qs[1, 3], -Qs[2, 3]
    det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    scale = abs(a00 * a11 * a22) + 1e-300
    best = 1e300
    bx = 0.5 * (pa + pb)
    if use_opt and abs(det) > 1e-9 * scale:
        x = np.empty(3)
        x[0] = (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / det
        x[1] = (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / det
        x[2] = (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / det
        # keep the optimum close to the edge (guards against ill conditioned quadrics)
        L = np.sqrt(((pa - pb) ** 2).sum()) + 1e-12
        if np.sqrt(((x - bx) ** 2).sum()) < 2.0 * L:
            c = _quad_cost(Qs, x)
            if c < best:
                best = c
                bx = x
    for k in range(3):
        if k == 0:
            cand = pa
        elif k == 1:
            cand = pb
        else:
            cand = 0.5 * (pa + pb)
        c = _quad_cost(Qs, cand)
        if c < best:
            best = c
            bx = cand.copy()
    return best, bx


def _face_quadrics(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2.0 * area[:, None], 1e-300)
    d = -np.einsum("ij,ij->i", n, t[:, 0])
    p = np.concatenate([n, d[:, None]], axis=1)  # (m, 4)
    Q = p[:, :, None] * p[:, None, :] * np.maximum(area, 1e-18)[:, None, None]
    return Q


def decimate(mesh: TriMesh, target_faces: int, max_iter: int | None = None, placement: str = "subset") -> TriMesh:
    """Return a decimated copy with about ``target_faces`` faces.

    ``placement`` is "optimal" (quadric minimiser, may poke through nearby sheets of thin
    objects) or "subset" (best of the two endpoints and the midpoint, safe by construction).
    """
    V = mesh.vertices.copy()
    F = mesh.faces.copy()
    nf = F.shape[0]
    if nf <= target_faces:
        return mesh.copy()
    Qf = _face_quadrics(V, F)
    nv = V.shape[0]
    Q = np.zeros((nv, 4, 4))
    for k in range(3):
        np.add.at(Q, F[:, k], Qf)
    vf = [set() for _ in range(nv)]
    for fi, f in enumerate(F):
        for v in f:
            vf[int(v)].add(fi)
    alive_f = np.ones(nf, dtype=bool)
    alive_v = np.ones(nv, dtype=bool)
    version = np.zeros(nv, dtype=np.int64)

    def neighbours(v):
        s = set()
        for fi in vf[v]:
            s.update(int(x) for x in F[fi])
        s.discard(v)
        return s

    use_opt = placement == "optimal"

    def optimal(a, b):
        c, x = _optimal_jit(Q[a], Q[b], V[a], V[b], use_opt)
        return float(c), x

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    heap = []
    edges = set()
    for f in F:
        for i in range(3):
            edges.add(edge_key(int(f[i]), int(f[(i + 1) % 3])))
    for a, b in edges:
        c, _ = optimal(a, b)
        heap.append((c, a, b, 0, 0))
    heapq.heapify(heap)
    n_alive = nf
    iters = 0
    max_iter = max_iter or 10 * nf
    while n_alive > target_faces and heap and iters < max_iter:
        iters += 1
        c, a, b, va, vb = heapq.heappop(heap)
        if not (alive_v[a] and alive_v[b]) or version[a] != va or version[b] != vb:
            continue
        shared = vf[a] & vf[b]
        if len(shared) != 2:
            continue  # boundary or non manifold edge
        # link condition
        na, nb = neighbours(a), neighbours(b)
        opp = set()
        for fi in shared:
            for v in F[fi]:
                if v != a and v != b:
                    opp.add(int(v))
        if (na & nb) != opp:
            continue
        _, x = optimal(a, b)
        # flip test on the faces that survive
        ok = True
        for v in (a, b):
            for fi in vf[v]:
                if fi in shared:
                    continue
                f = F[fi]
                p = [V[int(u)] for u in f]
                n0 = np.cross(p[1] - p[0], p[2] - p[0])
                pn = [x if int(u) in (a, b) else V[int(u)] for u in f]
                n1 = np.cross(pn[1] - pn[0], pn[2] - pn[0])
                if np.dot(n0, n1) <= 1e-12 * max(np.dot(n0, n0), 1e-300):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # collapse b into a
        V[a] = x
        Q[a] = Q[a] + Q[b]
        for fi in shared:
            alive_f[fi] = False
            for u in F[fi]:
                vf[int(u)].discard(fi)
            n_alive -= 1
        for fi in list(vf[b]):
            F[fi][F[fi] == b] = a
            vf[a].add(fi)
        vf[b] = set()
        alive_v[b] = False
        version[a] += 1
        for u in neighbours(a):
            cst, _ = optimal(a, u)
            k = edge_key(a, u)
            heapq.heappush(heap, (cst, k[0], k[1], int(version[k[0]]), int(version[k[1]])))
    Fk = F[alive_f]
    used, inv = np.unique(Fk.ravel(), return_inverse=True)
    out = TriMesh(V[used], inv.reshape(-1, 3), mesh.name)
    return out
