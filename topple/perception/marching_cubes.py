"""Marching cubes with a 256 case table generated from the Kuhn tetrahedral decomposition.

Each cube is split into six tetrahedra around its main diagonal (corners 0 to 7). Every one
of the 256 corner sign configurations is triangulated by marching tetrahedra at import time,
with triangle orientation fixed so that normals point toward positive values (outside for a
signed distance field that is negative inside). Because adjacent cubes share the same face
diagonals, the extracted surface is closed and consistent across cubes; the price compared
with the classic table is a larger triangle count. Vertices on shared edges are merged, so
the output is an indexed watertight mesh.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh

CORNERS = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], dtype=np.int64)
TETS = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def _build_tables():
    pairs = set()
    for t in TETS:
        for i in range(4):
            for j in range(i + 1, 4):
                pairs.add((min(t[i], t[j]), max(t[i], t[j])))
    edges = sorted(pairs)
    eidx = {e: k for k, e in enumerate(edges)}
    tri_table = np.full((256, 12, 3), -1, dtype=np.int64)
    ntri = np.zeros(256, dtype=np.int64)
    P = CORNERS.astype(np.float64)
    for case in range(256):
        inside = [(case >> c) & 1 for c in range(8)]
        vals = np.array([-1.0 if inside[c] else 1.0 for c in range(8)])
        tris = []
        for tet in TETS:
            ins = [c for c in tet if inside[c]]
            outs = [c for c in tet if not inside[c]]
            if len(ins) == 0 or len(ins) == 4:
                continue
            if len(ins) == 1:
                a = ins[0]
                tri_list = [[(a, outs[0]), (a, outs[1]), (a, outs[2])]]
            elif len(ins) == 3:
                a = outs[0]
                tri_list = [[(a, ins[0]), (a, ins[1]), (a, ins[2])]]
            else:
                i0, i1 = ins
                o0, o1 = outs
                tri_list = [[(i0, o0), (i0, o1), (i1, o1)], [(i0, o0), (i1, o1), (i1, o0)]]
            # gradient of the linear interpolant inside the tetrahedron
            M = np.stack([P[tet[k]] - P[tet[0]] for k in range(1, 4)])
            rhs = np.array([vals[tet[k]] - vals[tet[0]] for k in range(1, 4)])
            g = np.linalg.solve(M, rhs)
            for tri in tri_list:
                pts = [0.5 * (P[e[0]] + P[e[1]]) for e in tri]
                n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                if np.dot(n, g) < 0:
                    tri = [tri[0], tri[2], tri[1]]
                tris.append([eidx[(min(e), max(e))] for e in tri])
        ntri[case] = len(tris)
        for k, t in enumerate(tris):
            tri_table[case, k] = t
    edge_arr = np.array(edges, dtype=np.int64)
    return edge_arr, tri_table, ntri


EDGES, TRI_TABLE, NTRI = _build_tables()


def marching_cubes(values: np.ndarray, origin=(0.0, 0.0, 0.0), spacing: float = 1.0, iso: float = 0.0, mask: np.ndarray | None = None) -> TriMesh:
    """Extract the iso surface of a scalar grid. NaN voxels (or voxels where ``mask`` is False) are skipped."""
    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = np.where(mask, v, np.nan)
    nx, ny, nz = v.shape
    if min(nx, ny, nz) < 2:
        return TriMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), "mc")
    # corner values at the iso level are nudged outside so that no crossing lands exactly on a
    # corner; coincident crossings would otherwise have to be merged, which can create
    # non manifold edges where two sheets touch
    eps = 1e-6 * spacing
    v = np.where(np.abs(v - iso) < eps, iso + eps, v)
    cv = [v[dx : nx - 1 + dx, dy : ny - 1 + dy, dz : nz - 1 + dz] for dx, dy, dz in CORNERS]
    finite = np.ones((nx - 1, ny - 1, nz - 1), dtype=bool)
    case = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.int64)
    for c in range(8):
        finite &= np.isfinite(cv[c])
        case |= (cv[c] < iso).astype(np.int64) << c
    sel = finite & (case != 0) & (case != 255)
    ci, cj, ck = np.nonzero(sel)
    if ci.size == 0:
        return TriMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), "mc")
    cases = case[ci, cj, ck]
    tris = TRI_TABLE[cases]  # (m, 12, 3)
    tmask = tris[:, :, 0] >= 0
    cube_of_tri = np.repeat(np.arange(ci.size), tmask.sum(axis=1))
    tri_edges = tris[tmask]  # (n_tri, 3) edge ids
    # global vertex ids of edge endpoints
    ca = EDGES[tri_edges, 0]  # (n_tri, 3) corner a
    cb = EDGES[tri_edges, 1]
    base = np.stack([ci, cj, ck], 1)[cube_of_tri]  # (n_tri, 3)

    def gid(corner):
        pos = base[:, None, :] + CORNERS[corner]  # (n_tri, 3, 3)
        return (pos[..., 0] * ny + pos[..., 1]) * nz + pos[..., 2]

    ga = gid(ca)
    gb = gid(cb)
    lo = np.minimum(ga, gb)
    hi = np.maximum(ga, gb)
    key = lo.astype(np.int64) * (nx * ny * nz) + hi
    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    faces = inv.reshape(-1, 3)
    ulo = uniq // (nx * ny * nz)
    uhi = uniq % (nx * ny * nz)
    vflat = v.ravel()
    va = vflat[ulo]
    vb = vflat[uhi]
    t = (iso - va) / (vb - va)
    t = np.clip(t, 0.0, 1.0)

    def pos_of(g):
        i = g // (ny * nz)
        j = (g // nz) % ny
        k = g % nz
        return np.stack([i, j, k], 1).astype(np.float64)

    pa = pos_of(ulo)
    pb = pos_of(uhi)
    V = (pa + t[:, None] * (pb - pa)) * spacing + np.asarray(origin, dtype=np.float64)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return TriMesh(V, faces[keep], "mc")


def connected_components(mesh: TriMesh) -> np.ndarray:
    """Component label per face (union find over shared vertices)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as cc

    F = mesh.faces
    n = mesh.n_vertices
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    A = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    _, labels = cc(A, directed=False)
    return labels[F[:, 0]]


def largest_component(mesh: TriMesh) -> TriMesh:
    if mesh.n_faces == 0:
        return mesh
    lab = connected_components(mesh)
    counts = np.bincount(lab)
    best = int(np.argmax(counts))
    F = mesh.faces[lab == best]
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return TriMesh(mesh.vertices[used], inv.reshape(-1, 3), mesh.name)
