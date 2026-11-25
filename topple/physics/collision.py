"""Collision detection (Section 6.4).

Broad phase: a dynamic AABB tree with fat boxes. Narrow phase: GJK for separation and
EPA for penetration on convex pieces, a half space special case for the ground, and a
contact manifold built from the support patch of both pieces (vertices within the
contact margin of the contact plane), with exact per point separation from ray casts
through both convex pieces. Manifolds are reduced to at most ``max_manifold_points``
by farthest point sampling that always keeps the deepest point. Persistent feature ids
support warm starting. ``verify_contacts_sdf`` cross checks contact points against the
signed distance fields of the bodies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit

from ..core.transforms import quat_rotate, quat_to_rot, tangent_basis
from .body import Contact

# ----------------------------------------------------------------------------------
# Broad phase: dynamic AABB tree
# ----------------------------------------------------------------------------------


class AABBTree:
    """Dynamic AABB tree with fat boxes (incremental insert, remove and update)."""

    def __init__(self, fat_margin: float = 0.01):
        self.fat_margin = fat_margin
        self.lo: List[np.ndarray] = []
        self.hi: List[np.ndarray] = []
        self.parent: List[int] = []
        self.child1: List[int] = []
        self.child2: List[int] = []
        self.item: List[int] = []  # body id for leaves, -1 for internal
        self.root = -1
        self.free: List[int] = []
        self.leaf_of: Dict[int, int] = {}

    # -- node pool --------------------------------------------------------------
    def _alloc(self, lo, hi, item=-1):
        if self.free:
            n = self.free.pop()
            self.lo[n], self.hi[n], self.parent[n], self.child1[n], self.child2[n], self.item[n] = lo, hi, -1, -1, -1, item
            return n
        self.lo.append(lo)
        self.hi.append(hi)
        self.parent.append(-1)
        self.child1.append(-1)
        self.child2.append(-1)
        self.item.append(item)
        return len(self.lo) - 1

    def _is_leaf(self, n):
        return self.child1[n] == -1

    @staticmethod
    def _area(lo, hi):
        e = hi - lo
        return 2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])

    # -- operations -------------------------------------------------------------
    def insert(self, item: int, lo: np.ndarray, hi: np.ndarray) -> int:
        m = self.fat_margin
        leaf = self._alloc(np.asarray(lo, dtype=float) - m, np.asarray(hi, dtype=float) + m, item)
        self.leaf_of[item] = leaf
        if self.root == -1:
            self.root = leaf
            return leaf
        # find best sibling by the insertion cost heuristic
        node = self.root
        llo, lhi = self.lo[leaf], self.hi[leaf]
        while not self._is_leaf(node):
            c1, c2 = self.child1[node], self.child2[node]
            area = self._area(self.lo[node], self.hi[node])
            comb_lo = np.minimum(self.lo[node], llo)
            comb_hi = np.maximum(self.hi[node], lhi)
            comb_area = self._area(comb_lo, comb_hi)
            cost = 2.0 * comb_area
            inherit = 2.0 * (comb_area - area)

            def child_cost(c):
                nlo = np.minimum(self.lo[c], llo)
                nhi = np.maximum(self.hi[c], lhi)
                if self._is_leaf(c):
                    return self._area(nlo, nhi) + inherit
                return (self._area(nlo, nhi) - self._area(self.lo[c], self.hi[c])) + inherit

            cost1, cost2 = child_cost(c1), child_cost(c2)
            if cost < cost1 and cost < cost2:
                break
            node = c1 if cost1 < cost2 else c2
        sibling = node
        old_parent = self.parent[sibling]
        new_parent = self._alloc(np.minimum(self.lo[sibling], llo), np.maximum(self.hi[sibling], lhi))
        self.parent[new_parent] = old_parent
        self.child1[new_parent] = sibling
        self.child2[new_parent] = leaf
        self.parent[sibling] = new_parent
        self.parent[leaf] = new_parent
        if old_parent == -1:
            self.root = new_parent
        else:
            if self.child1[old_parent] == sibling:
                self.child1[old_parent] = new_parent
            else:
                self.child2[old_parent] = new_parent
        self._refit(self.parent[leaf])
        return leaf

    def _refit(self, node):
        while node != -1:
            c1, c2 = self.child1[node], self.child2[node]
            self.lo[node] = np.minimum(self.lo[c1], self.lo[c2])
            self.hi[node] = np.maximum(self.hi[c1], self.hi[c2])
            node = self.parent[node]

    def remove(self, item: int):
        leaf = self.leaf_of.pop(item)
        if leaf == self.root:
            self.root = -1
            self.free.append(leaf)
            return
        parent = self.parent[leaf]
        grand = self.parent[parent]
        sibling = self.child2[parent] if self.child1[parent] == leaf else self.child1[parent]
        if grand == -1:
            self.root = sibling
            self.parent[sibling] = -1
        else:
            if self.child1[grand] == parent:
                self.child1[grand] = sibling
            else:
                self.child2[grand] = sibling
            self.parent[sibling] = grand
            self._refit(grand)
        self.free.append(parent)
        self.free.append(leaf)

    def update(self, item: int, lo: np.ndarray, hi: np.ndarray) -> bool:
        """Reinsert only when the tight box leaves its fat box. Returns True when moved."""
        if item not in self.leaf_of:
            self.insert(item, lo, hi)
            return True
        leaf = self.leaf_of[item]
        if np.all(lo >= self.lo[leaf]) and np.all(hi <= self.hi[leaf]):
            return False
        self.remove(item)
        self.insert(item, lo, hi)
        return True

    def query_pairs(self) -> List[Tuple[int, int]]:
        """All leaf pairs whose fat boxes overlap."""
        pairs: List[Tuple[int, int]] = []
        if self.root == -1 or self._is_leaf(self.root):
            return pairs
        stack = [(self.child1[self.root], self.child2[self.root])]
        # also need pairs within each subtree
        stack_single = [self.root]
        while stack_single:
            n = stack_single.pop()
            if self._is_leaf(n):
                continue
            c1, c2 = self.child1[n], self.child2[n]
            stack_single.append(c1)
            stack_single.append(c2)
            if n != self.root:
                stack.append((c1, c2))
        while stack:
            a, b = stack.pop()
            if np.any(self.hi[a] < self.lo[b]) or np.any(self.hi[b] < self.lo[a]):
                continue
            la, lb = self._is_leaf(a), self._is_leaf(b)
            if la and lb:
                ia, ib = self.item[a], self.item[b]
                pairs.append((ia, ib) if ia < ib else (ib, ia))
            elif la:
                stack.append((a, self.child1[b]))
                stack.append((a, self.child2[b]))
            elif lb:
                stack.append((self.child1[a], b))
                stack.append((self.child2[a], b))
            else:
                stack.append((self.child1[a], self.child1[b]))
                stack.append((self.child1[a], self.child2[b]))
                stack.append((self.child2[a], self.child1[b]))
                stack.append((self.child2[a], self.child2[b]))
        return sorted(set(pairs))

    def query_box(self, lo, hi) -> List[int]:
        out = []
        if self.root == -1:
            return out
        stack = [self.root]
        while stack:
            n = stack.pop()
            if np.any(self.hi[n] < lo) or np.any(hi < self.lo[n]):
                continue
            if self._is_leaf(n):
                out.append(self.item[n])
            else:
                stack.append(self.child1[n])
                stack.append(self.child2[n])
        return out


# ----------------------------------------------------------------------------------
# GJK and EPA (numba)
# ----------------------------------------------------------------------------------


@njit(cache=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _cross(a, b):
    out = np.empty(3)
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
    return out


@njit(cache=True)
def _support_idx(V, d):
    best = -1e300
    bi = 0
    for i in range(V.shape[0]):
        s = V[i, 0] * d[0] + V[i, 1] * d[1] + V[i, 2] * d[2]
        if s > best:
            best = s
            bi = i
    return bi


@njit(cache=True)
def _closest_segment(a, b):
    ab = b - a
    den = _dot(ab, ab)
    if den < 1e-300:
        return a.copy(), 1.0, 0.0
    t = -_dot(a, ab) / den
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return a + t * ab, 1.0 - t, t


@njit(cache=True)
def _closest_triangle(a, b, c):
    """Closest point of triangle (a, b, c) to the origin with barycentric coordinates."""
    ab = b - a
    ac = c - a
    ap = -a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), 1.0, 0.0, 0.0
    bp = -b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, 1.0 - v, v, 0.0
    cp = -c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, 1.0 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), 0.0, 1.0 - w, w
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w, 1.0 - v - w, v, w


@njit(cache=True)
def _origin_in_tetra(a, b, c, d):
    """True when the origin is inside tetrahedron abcd (robust to orientation)."""
    n1 = _cross(b - a, c - a)
    s1 = _dot(n1, -a) * _dot(n1, d - a)
    n2 = _cross(c - b, d - b)
    s2 = _dot(n2, -b) * _dot(n2, a - b)
    n3 = _cross(d - c, a - c)
    s3 = _dot(n3, -c) * _dot(n3, b - c)
    n4 = _cross(a - d, b - d)
    s4 = _dot(n4, -d) * _dot(n4, c - d)
    return s1 >= 0.0 and s2 >= 0.0 and s3 >= 0.0 and s4 >= 0.0


@njit(cache=True)
def gjk_core(VA, VB, max_iter=96, tol=1e-10):
    """GJK distance between convex vertex sets.

    Returns (status, dist, S, IA, IB, k, lam) where status 0 = separated (dist valid),
    1 = origin inside or on the simplex (penetrating or touching). S holds the final
    simplex (Minkowski points), IA/IB the support indices, lam the barycentric weights.
    """
    S = np.zeros((4, 3))
    IA = np.zeros(4, dtype=np.int64)
    IB = np.zeros(4, dtype=np.int64)
    lam = np.zeros(4)
    d = np.empty(3)
    cA = np.zeros(3)
    cB = np.zeros(3)
    for i in range(VA.shape[0]):
        cA += VA[i]
    cA /= VA.shape[0]
    for i in range(VB.shape[0]):
        cB += VB[i]
    cB /= VB.shape[0]
    d[0] = cA[0] - cB[0]
    d[1] = cA[1] - cB[1]
    d[2] = cA[2] - cB[2]
    if _dot(d, d) < 1e-30:
        d[0] = 1.0
    ia = _support_idx(VA, d)
    ib = _support_idx(VB, -d)
    S[0] = VA[ia] - VB[ib]
    IA[0] = ia
    IB[0] = ib
    k = 1
    v = S[0].copy()
    lam[0] = 1.0
    for it in range(max_iter):
        # closest point on simplex and reduction
        if k == 1:
            v = S[0].copy()
            lam[0] = 1.0
        elif k == 2:
            v, l0, l1 = _closest_segment(S[0], S[1])
            lam[0] = l0
            lam[1] = l1
        elif k == 3:
            v, l0, l1, l2 = _closest_triangle(S[0], S[1], S[2])
            lam[0] = l0
            lam[1] = l1
            lam[2] = l2
        else:
            if _origin_in_tetra(S[0], S[1], S[2], S[3]):
                return 1, 0.0, S, IA, IB, 4, lam
            best = 1e300
            bv = np.zeros(3)
            bl = np.zeros(4)
            # faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            for f in range(4):
                if f == 0:
                    i0, i1, i2 = 0, 1, 2
                elif f == 1:
                    i0, i1, i2 = 0, 1, 3
                elif f == 2:
                    i0, i1, i2 = 0, 2, 3
                else:
                    i0, i1, i2 = 1, 2, 3
                p, l0, l1, l2 = _closest_triangle(S[i0], S[i1], S[i2])
                dd = _dot(p, p)
                if dd < best:
                    best = dd
                    bv = p
                    bl[:] = 0.0
                    bl[i0] = l0
                    bl[i1] = l1
                    bl[i2] = l2
            v = bv
            lam = bl
        # reduce simplex to vertices with positive weight
        kk = 0
        for i in range(k):
            if lam[i] > 0.0:
                S[kk] = S[i]
                IA[kk] = IA[i]
                IB[kk] = IB[i]
                lam[kk] = lam[i]
                kk += 1
        k = kk
        vv = _dot(v, v)
        if vv < 1e-22:
            return 1, 0.0, S, IA, IB, k, lam
        d = -v
        ia = _support_idx(VA, d)
        ib = _support_idx(VB, -d)
        w = VA[ia] - VB[ib]
        # no progress: w is not closer to origin along d than v
        if vv - _dot(v, w) <= tol * vv + 1e-24:
            return 0, math.sqrt(vv), S, IA, IB, k, lam
        dup = False
        for i in range(k):
            if IA[i] == ia and IB[i] == ib:
                dup = True
        if dup:
            return 0, math.sqrt(vv), S, IA, IB, k, lam
        if k < 4:
            S[k] = w
            IA[k] = ia
            IB[k] = ib
            k += 1
        else:
            return 0, math.sqrt(vv), S, IA, IB, k, lam
    return 0, math.sqrt(_dot(v, v)), S, IA, IB, k, lam


@njit(cache=True)
def _blow_up(VA, VB, S, IA, IB, k):
    """Extend a degenerate simplex to a tetrahedron (touching cases)."""
    dirs = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    cand = np.zeros((10, 3))
    nc = 0
    if k == 2:
        e = S[1] - S[0]
        ax = np.array([1.0, 0.0, 0.0]) if abs(e[0]) < 0.9 * math.sqrt(_dot(e, e)) else np.array([0.0, 1.0, 0.0])
        p = _cross(e, ax)
        cand[nc] = p
        nc += 1
        cand[nc] = -p
        nc += 1
        q = _cross(e, p)
        cand[nc] = q
        nc += 1
        cand[nc] = -q
        nc += 1
    elif k == 3:
        nrm = _cross(S[1] - S[0], S[2] - S[0])
        cand[nc] = nrm
        nc += 1
        cand[nc] = -nrm
        nc += 1
    for i in range(6):
        cand[nc] = dirs[i]
        nc += 1
    for ci in range(nc):
        if k >= 4:
            break
        d = cand[ci]
        if _dot(d, d) < 1e-30:
            continue
        ia = _support_idx(VA, d)
        ib = _support_idx(VB, -d)
        w = VA[ia] - VB[ib]
        ok = True
        for i in range(k):
            if IA[i] == ia and IB[i] == ib:
                ok = False
        if not ok:
            continue
        # check non degeneracy
        if k == 1:
            if _dot(w - S[0], w - S[0]) < 1e-24:
                ok = False
        elif k == 2:
            c = _cross(S[1] - S[0], w - S[0])
            if _dot(c, c) < 1e-24:
                ok = False
        elif k == 3:
            nrm = _cross(S[1] - S[0], S[2] - S[0])
            if abs(_dot(nrm, w - S[0])) < 1e-18:
                ok = False
        if not ok:
            continue
        S[k] = w
        IA[k] = ia
        IB[k] = ib
        k += 1
    return S, IA, IB, k


@njit(cache=True)
def epa_core(VA, VB, S, IA, IB, k, max_iter=128, tol=1e-9):
    """Expanding polytope algorithm. Returns (ok, depth, normal, pa, pb).

    The normal points from A towards B (direction to translate B so the pieces separate).
    """
    if k < 4:
        S, IA, IB, k = _blow_up(VA, VB, S, IA, IB, k)
    if k < 4:
        return False, 0.0, np.zeros(3), np.zeros(3), np.zeros(3)
    maxv = max_iter + 8
    maxf = 4 * maxv
    P = np.zeros((maxv, 3))
    PA = np.zeros(maxv, dtype=np.int64)
    PB = np.zeros(maxv, dtype=np.int64)
    for i in range(4):
        P[i] = S[i]
        PA[i] = IA[i]
        PB[i] = IB[i]
    nv = 4
    F = np.zeros((maxf, 3), dtype=np.int64)
    FN = np.zeros((maxf, 3))
    FD = np.zeros(maxf)
    alive = np.zeros(maxf, dtype=np.bool_)
    nf = 0
    # centroid for orientation
    cen = (P[0] + P[1] + P[2] + P[3]) * 0.25
    init = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    for f in range(4):
        a, b, c = init[f, 0], init[f, 1], init[f, 2]
        n = _cross(P[b] - P[a], P[c] - P[a])
        ln = math.sqrt(_dot(n, n))
        if ln < 1e-30:
            continue
        n /= ln
        if _dot(n, P[a] - cen) < 0.0:
            b, c = c, b
            n = -n
        F[nf, 0] = a
        F[nf, 1] = b
        F[nf, 2] = c
        FN[nf] = n
        FD[nf] = _dot(n, P[a])
        alive[nf] = True
        nf += 1
    if nf < 4:
        return False, 0.0, np.zeros(3), np.zeros(3), np.zeros(3)
    best_f = -1
    for it in range(max_iter):
        # closest face
        best = 1e300
        best_f = -1
        for f in range(nf):
            if alive[f] and FD[f] < best:
                best = FD[f]
                best_f = f
        if best_f < 0:
            return False, 0.0, np.zeros(3), np.zeros(3), np.zeros(3)
        n = FN[best_f]
        ia = _support_idx(VA, n)
        ib = _support_idx(VB, -n)
        w = VA[ia] - VB[ib]
        if _dot(n, w) - best < tol or nv >= maxv:
            break
        # duplicate vertex check
        dup = False
        for i in range(nv):
            if PA[i] == ia and PB[i] == ib:
                dup = True
        if dup:
            break
        P[nv] = w
        PA[nv] = ia
        PB[nv] = ib
        wi = nv
        nv += 1
        # remove visible faces, collect horizon edges
        edges = np.zeros((maxf, 2), dtype=np.int64)
        ne = 0
        for f in range(nf):
            if not alive[f]:
                continue
            if _dot(FN[f], w - P[F[f, 0]]) > 1e-12:
                alive[f] = False
                for e in range(3):
                    a = F[f, e]
                    b = F[f, (e + 1) % 3]
                    # if reversed edge exists in list, remove it (shared with another removed face)
                    found = -1
                    for j in range(ne):
                        if edges[j, 0] == b and edges[j, 1] == a:
                            found = j
                            break
                    if found >= 0:
                        edges[found] = edges[ne - 1]
                        ne -= 1
                    else:
                        edges[ne, 0] = a
                        edges[ne, 1] = b
                        ne += 1
        if ne == 0:
            break
        for j in range(ne):
            if nf >= maxf:
                break
            a = edges[j, 0]
            b = edges[j, 1]
            nn = _cross(P[b] - P[a], P[wi] - P[a])
            ln = math.sqrt(_dot(nn, nn))
            if ln < 1e-30:
                continue
            nn /= ln
            if _dot(nn, P[a] - cen) < 0.0:
                nn = -nn
                F[nf, 0] = b
                F[nf, 1] = a
            else:
                F[nf, 0] = a
                F[nf, 1] = b
            F[nf, 2] = wi
            FN[nf] = nn
            FD[nf] = _dot(nn, P[a])
            alive[nf] = True
            nf += 1
        # compact face list occasionally
        if nf > maxf - 16:
            m = 0
            for f in range(nf):
                if alive[f]:
                    F[m] = F[f]
                    FN[m] = FN[f]
                    FD[m] = FD[f]
                    alive[m] = True
                    m += 1
            for f in range(m, nf):
                alive[f] = False
            nf = m
    if best_f < 0:
        return False, 0.0, np.zeros(3), np.zeros(3), np.zeros(3)
    a, b, c = F[best_f, 0], F[best_f, 1], F[best_f, 2]
    p, l0, l1, l2 = _closest_triangle(P[a], P[b], P[c])
    pa = l0 * VA[PA[a]] + l1 * VA[PA[b]] + l2 * VA[PA[c]]
    pb = l0 * VB[PB[a]] + l1 * VB[PB[b]] + l2 * VB[PB[c]]
    return True, FD[best_f], FN[best_f].copy(), pa, pb


@njit(cache=True)
def convex_query(VA, VB):
    """Separation (positive) or penetration (negative) between convex pieces.

    Returns (ok, sep, normal from A to B, pa, pb).
    """
    status, dist, S, IA, IB, k, lam = gjk_core(VA, VB)
    if status == 0 and dist > 1e-7:
        pa = np.zeros(3)
        pb = np.zeros(3)
        for i in range(k):
            pa += lam[i] * VA[IA[i]]
            pb += lam[i] * VB[IB[i]]
        n = pb - pa
        ln = math.sqrt(_dot(n, n))
        if ln < 1e-300:
            return False, 0.0, np.zeros(3), pa, pb
        return True, dist, n / ln, pa, pb
    # Touching or shallow overlap. EPA is ill conditioned when the pieces touch (its start
    # simplex is nearly flat and it can return a large depth with a wrong witness), so the
    # query is repeated with B moved apart along the centroid direction d, which has a
    # positive component along any separating normal of two convex pieces. The separation
    # along the normal n of the moved pair, minus the shift along n, is the separation of the
    # original pair along n. EPA is used for overlaps deeper than the shift.
    cA = np.zeros(3)
    for i in range(VA.shape[0]):
        cA += VA[i]
    cA /= VA.shape[0]
    cB = np.zeros(3)
    for i in range(VB.shape[0]):
        cB += VB[i]
    cB /= VB.shape[0]
    d = cB - cA
    ld = math.sqrt(_dot(d, d))
    if ld > 1e-300:
        d = d / ld
        for delta in (1e-4, 3e-3):
            VB2 = VB + delta * d
            st2, dist2, S2, IA2, IB2, k2, lam2 = gjk_core(VA, VB2)
            if st2 != 0 or dist2 <= 1e-7:
                continue
            pa = np.zeros(3)
            pb = np.zeros(3)
            for i in range(k2):
                pa += lam2[i] * VA[IA2[i]]
                pb += lam2[i] * VB2[IB2[i]]
            n = pb - pa
            ln = math.sqrt(_dot(n, n))
            if ln < 1e-300:
                continue
            n = n / ln
            if k2 == 3 and lam2[0] > 1e-9 and lam2[1] > 1e-9 and lam2[2] > 1e-9:
                # closest point inside a simplex triangle: the triangle lies in the support plane,
                # and its cross product normal is far more accurate than the witness difference
                m = _cross(S2[1] - S2[0], S2[2] - S2[0])
                lm = math.sqrt(_dot(m, m))
                if lm > 1e-300:
                    m = m / lm
                    if _dot(m, n) < 0.0:
                        m = -m
                    n = m
                    dist2 = -_dot(S2[0], n)
            dn = delta * _dot(d, n)
            return True, dist2 - dn, n, pa, pb - delta * d
    ok, depth, n, pa, pb = epa_core(VA, VB, S, IA, IB, k)
    if not ok:
        return False, 0.0, np.zeros(3), np.zeros(3), np.zeros(3)
    return True, -depth, n, pa, pb


def gjk_distance(A_vertices: np.ndarray, B_vertices: np.ndarray):
    """Distance between two convex vertex sets and the witness points (0 when overlapping)."""
    VA = np.ascontiguousarray(A_vertices, dtype=np.float64)
    VB = np.ascontiguousarray(B_vertices, dtype=np.float64)
    status, dist, S, IA, IB, k, lam = gjk_core(VA, VB)
    if status == 1:
        return 0.0, None, None
    pa = sum(lam[i] * VA[IA[i]] for i in range(k))
    pb = sum(lam[i] * VB[IB[i]] for i in range(k))
    return float(dist), pa, pb


def epa_penetration(A_vertices: np.ndarray, B_vertices: np.ndarray):
    """Penetration depth, normal (A to B) and witness points for overlapping convex sets."""
    VA = np.ascontiguousarray(A_vertices, dtype=np.float64)
    VB = np.ascontiguousarray(B_vertices, dtype=np.float64)
    ok, sep, n, pa, pb = convex_query(VA, VB)
    if not ok:
        return 0.0, np.zeros(3), None, None
    return float(max(-sep, 0.0)), n, pa, pb


# ----------------------------------------------------------------------------------
# Manifold construction (numba)
# ----------------------------------------------------------------------------------


@njit(cache=True)
def _ray_convex_tolerant(o, d, V, F, want_max):
    """Entry (min t) or exit (max t) of a ray through a closed convex triangle mesh."""
    best = -1.0 if want_max else 1e300
    hit = False
    for f in range(F.shape[0]):
        v0 = V[F[f, 0]]
        e1 = V[F[f, 1]] - v0
        e2 = V[F[f, 2]] - v0
        px = d[1] * e2[2] - d[2] * e2[1]
        py = d[2] * e2[0] - d[0] * e2[2]
        pz = d[0] * e2[1] - d[1] * e2[0]
        det = e1[0] * px + e1[1] * py + e1[2] * pz
        if abs(det) < 1e-18:
            continue
        inv = 1.0 / det
        sx = o[0] - v0[0]
        sy = o[1] - v0[1]
        sz = o[2] - v0[2]
        u = (sx * px + sy * py + sz * pz) * inv
        if u < -1e-5 or u > 1.0 + 1e-5:
            continue
        qx = sy * e1[2] - sz * e1[1]
        qy = sz * e1[0] - sx * e1[2]
        qz = sx * e1[1] - sy * e1[0]
        v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
        if v < -1e-5 or u + v > 1.0 + 1e-5:
            continue
        t = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv
        if want_max:
            if t > best:
                best = t
                hit = True
        else:
            if t < best:
                best = t
                hit = True
    return hit, best


@njit(cache=True)
def _farthest_point_reduce(P2, depth, max_points, keep_first):
    """Indices of up to max_points points chosen by farthest point sampling in 2D.

    ``keep_first`` is the index that must be kept (the deepest point).
    """
    n = P2.shape[0]
    chosen = np.empty(max_points, dtype=np.int64)
    if n <= max_points:
        for i in range(n):
            chosen[i] = i
        return chosen[:n]
    chosen[0] = keep_first
    nc = 1
    mind = np.full(n, 1e300)
    for i in range(n):
        dx = P2[i, 0] - P2[keep_first, 0]
        dy = P2[i, 1] - P2[keep_first, 1]
        mind[i] = dx * dx + dy * dy
    while nc < max_points:
        best = -1.0
        bi = -1
        for i in range(n):
            if mind[i] > best:
                best = mind[i]
                bi = i
        if bi < 0 or best <= 0.0:
            break
        chosen[nc] = bi
        nc += 1
        for i in range(n):
            dx = P2[i, 0] - P2[bi, 0]
            dy = P2[i, 1] - P2[bi, 1]
            dd = dx * dx + dy * dy
            if dd < mind[i]:
                mind[i] = dd
    return chosen[:nc]


@njit(cache=True)
def _hull2d(P, n):
    """Andrew monotone chain. Returns (indices ccw, count)."""
    order = np.argsort(P[:n, 0] * 1e6 + P[:n, 1])
    # stable lexicographic sort
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = order[i]
    # insertion sort refinement for exact lexicographic order
    for i in range(1, n):
        j = i
        while j > 0:
            a = idx[j - 1]
            b = idx[j]
            if P[a, 0] > P[b, 0] or (P[a, 0] == P[b, 0] and P[a, 1] > P[b, 1]):
                idx[j - 1], idx[j] = b, a
                j -= 1
            else:
                break
    H = np.empty(2 * n + 2, dtype=np.int64)
    k = 0
    for ii in range(n):
        i = idx[ii]
        while k >= 2:
            a = H[k - 2]
            b = H[k - 1]
            cr = (P[b, 0] - P[a, 0]) * (P[i, 1] - P[a, 1]) - (P[b, 1] - P[a, 1]) * (P[i, 0] - P[a, 0])
            if cr <= 1e-18:
                k -= 1
            else:
                break
        H[k] = i
        k += 1
    t = k + 1
    for ii in range(n - 2, -1, -1):
        i = idx[ii]
        while k >= t:
            a = H[k - 2]
            b = H[k - 1]
            cr = (P[b, 0] - P[a, 0]) * (P[i, 1] - P[a, 1]) - (P[b, 1] - P[a, 1]) * (P[i, 0] - P[a, 0])
            if cr <= 1e-18:
                k -= 1
            else:
                break
        H[k] = i
        k += 1
    k -= 1  # last point equals first
    if k < 1:
        k = 1
    return H[:k], k


@njit(cache=True)
def _clip_polygon(S, ns, C, ncl):
    """Sutherland Hodgman: clip subject polygon S (ns points) by convex ccw polygon C."""
    maxp = 128
    out = np.empty((maxp, 2))
    cur = np.empty((maxp, 2))
    nc = ns
    for i in range(ns):
        cur[i] = S[i]
    for e in range(ncl):
        ax = C[e, 0]
        ay = C[e, 1]
        bx = C[(e + 1) % ncl, 0]
        by = C[(e + 1) % ncl, 1]
        no = 0
        if nc == 0:
            break
        for i in range(nc):
            px = cur[i, 0]
            py = cur[i, 1]
            qx = cur[(i + 1) % nc, 0]
            qy = cur[(i + 1) % nc, 1]
            sp = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            sq = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
            pin = sp >= -1e-15
            qin = sq >= -1e-15
            if pin:
                if no < maxp:
                    out[no, 0] = px
                    out[no, 1] = py
                    no += 1
            if pin != qin:
                den = sp - sq
                if abs(den) > 1e-300:
                    t = sp / den
                    if no < maxp:
                        out[no, 0] = px + t * (qx - px)
                        out[no, 1] = py + t * (qy - py)
                        no += 1
        nc = no
        for i in range(nc):
            cur[i] = out[i]
    return cur[:nc].copy(), nc


@njit(cache=True)
def _tiered_reduce(P2, sep, max_points, band):
    """Farthest point sampling that fills the manifold with load bearing points first.

    The deepest point is kept, then points whose separation is within ``band`` of the deepest
    are chosen by farthest point sampling, and any remaining slots are filled by farthest point
    sampling over the other (speculative) points. A purely spatial sampling can drop the
    vertices of the facet a stone rests on in favour of distant points that are later rejected.
    """
    n = P2.shape[0]
    if n <= max_points:
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = i
        return out
    # the first point within 5 um of the deepest, so that near ties keep a stable order from step
    # to step (persistent features keep their warm start)
    smin = sep[0]
    for i in range(n):
        if sep[i] < smin:
            smin = sep[i]
    deep = 0
    for i in range(n):
        if sep[i] <= smin + 5e-6:
            deep = i
            break
    chosen = np.empty(max_points, dtype=np.int64)
    taken = np.zeros(n, dtype=np.bool_)
    chosen[0] = deep
    taken[deep] = True
    nc = 1
    mind = np.empty(n)
    for i in range(n):
        dx = P2[i, 0] - P2[deep, 0]
        dy = P2[i, 1] - P2[deep, 1]
        mind[i] = dx * dx + dy * dy
    lim = sep[deep] + band
    for tier in range(2):
        while nc < max_points:
            best = -1.0
            bi = -1
            for i in range(n):
                if taken[i] or (tier == 0 and sep[i] > lim):
                    continue
                if mind[i] > best:
                    best = mind[i]
                    bi = i
            if bi < 0 or best <= 0.0:
                break
            chosen[nc] = bi
            taken[bi] = True
            nc += 1
            for i in range(n):
                dx = P2[i, 0] - P2[bi, 0]
                dy = P2[i, 1] - P2[bi, 1]
                dd = dx * dx + dy * dy
                if dd < mind[i]:
                    mind[i] = dd
    return chosen[:nc]


@njit(cache=True)
def manifold_convex(VA, FA, VB, FB, n, pa, pb, sep, margin, max_points, t1, t2, L):
    """Contact manifold for a convex pair with normal n (A to B) and witness points.

    Candidates are the deepest point, the support patch vertices of both pieces and the
    corners of the intersection of the two projected patch polygons. Each candidate gets its
    exact separation from ray casts through both pieces. Returns (points (m,3),
    separations (m,), source (m,), vertex id (m,)).
    """
    p0 = 0.5 * (pa + pb)
    hA = -1e300
    for i in range(VA.shape[0]):
        s = _dot(VA[i], n)
        if s > hA:
            hA = s
    hB = 1e300
    for i in range(VB.shape[0]):
        s = _dot(VB[i], n)
        if s < hB:
            hB = s
    tolp = margin + max(0.0, -sep) + 1e-9
    maxc = VA.shape[0] + VB.shape[0] + 130
    cand2 = np.zeros((maxc, 2))
    cand_sep = np.zeros(maxc)
    cand_src = np.zeros(maxc, dtype=np.int64)
    cand_vid = np.zeros(maxc, dtype=np.int64)
    nc = 0
    cand_sep[nc] = sep
    cand_src[nc] = 2
    nc += 1
    a_start = nc
    for i in range(VA.shape[0]):
        s = _dot(VA[i], n)
        if s >= hA - tolp:
            r = VA[i] - p0
            cand2[nc, 0] = _dot(r, t1)
            cand2[nc, 1] = _dot(r, t2)
            cand_sep[nc] = hB - s
            cand_src[nc] = 0
            cand_vid[nc] = i
            nc += 1
    a_end = nc
    for i in range(VB.shape[0]):
        s = _dot(VB[i], n)
        if s <= hB + tolp:
            r = VB[i] - p0
            cand2[nc, 0] = _dot(r, t1)
            cand2[nc, 1] = _dot(r, t2)
            cand_sep[nc] = s - hA
            cand_src[nc] = 1
            cand_vid[nc] = i
            nc += 1
    b_end = nc
    # intersection polygon of the two projected patches
    na = a_end - a_start
    nb = b_end - a_end
    if na >= 2 and nb >= 2:
        PA2 = cand2[a_start:a_end].copy()
        PB2 = cand2[a_end:b_end].copy()
        ha, ka = _hull2d(PA2, na)
        hb, kb = _hull2d(PB2, nb)
        if ka >= 2 and kb >= 3:
            S = np.empty((ka, 2))
            for i in range(ka):
                S[i] = PA2[ha[i]]
            C = np.empty((kb, 2))
            for i in range(kb):
                C[i] = PB2[hb[i]]
            R, nr = _clip_polygon(S, ka, C, kb)
        elif kb >= 2 and ka >= 3:
            S = np.empty((kb, 2))
            for i in range(kb):
                S[i] = PB2[hb[i]]
            C = np.empty((ka, 2))
            for i in range(ka):
                C[i] = PA2[ha[i]]
            R, nr = _clip_polygon(S, kb, C, ka)
        else:
            R = np.zeros((0, 2))
            nr = 0
        if nr > 0:
            cx = 0.0
            cy = 0.0
            for i in range(nr):
                cx += R[i, 0]
                cy += R[i, 1]
            cx /= nr
            cy /= nr
            for i in range(nr):
                if nc >= maxc:
                    break
                # shrink slightly toward the centroid so rays stay inside both silhouettes
                cand2[nc, 0] = cx + (R[i, 0] - cx) * (1.0 - 1e-6)
                cand2[nc, 1] = cy + (R[i, 1] - cy) * (1.0 - 1e-6)
                cand_sep[nc] = sep
                cand_src[nc] = 4
                cand_vid[nc] = i
                nc += 1
    # faces near the patch (any vertex within 4 tolp of the support plane) for the ray casts
    selA = np.zeros(FA.shape[0], dtype=np.int64)
    nA = 0
    for f in range(FA.shape[0]):
        for e in range(3):
            if _dot(VA[FA[f, e]], n) >= hA - 2.0 * tolp:
                selA[nA] = f
                nA += 1
                break
    selB = np.zeros(FB.shape[0], dtype=np.int64)
    nB = 0
    for f in range(FB.shape[0]):
        for e in range(3):
            if _dot(VB[FB[f, e]], n) <= hB + 2.0 * tolp:
                selB[nB] = f
                nB += 1
                break
    FAs = np.empty((nA, 3), dtype=np.int64)
    for f in range(nA):
        FAs[f] = FA[selA[f]]
    FBs = np.empty((nB, 3), dtype=np.int64)
    for f in range(nB):
        FBs[f] = FB[selB[f]]
    # preselect by farthest point sampling (2 x max_points)
    pre = _tiered_reduce(cand2[:nc], cand_sep[:nc], 2 * max_points, 3e-4)
    m = pre.shape[0]
    pts = np.zeros((m, 3))
    seps = np.zeros(m)
    src = np.zeros(m, dtype=np.int64)
    vid = np.zeros(m, dtype=np.int64)
    P2 = np.zeros((m, 2))
    k = 0
    for j in range(m):
        i = pre[j]
        lat = p0 + cand2[i, 0] * t1 + cand2[i, 1] * t2
        if cand_src[i] == 2:
            pts[k] = lat
            seps[k] = sep
            src[k] = 2
            vid[k] = 0
            P2[k] = cand2[i]
            k += 1
            continue
        o = lat - n * L
        hitA, tA = _ray_convex_tolerant(o, n, VA, FAs, True)
        hitB, tB = _ray_convex_tolerant(o, n, VB, FBs, False)
        if not (hitA and hitB):
            continue
        s = tB - tA
        if s > margin:
            continue
        pts[k] = o + n * (0.5 * (tA + tB))
        seps[k] = s
        src[k] = cand_src[i]
        vid[k] = cand_vid[i]
        P2[k] = cand2[i]
        k += 1
    sel = _tiered_reduce(P2[:k], seps[:k], max_points, 1e-4)
    out_p = np.zeros((sel.shape[0], 3))
    out_s = np.zeros(sel.shape[0])
    out_src = np.zeros(sel.shape[0], dtype=np.int64)
    out_vid = np.zeros(sel.shape[0], dtype=np.int64)
    m_out = 0
    for j in range(sel.shape[0]):
        q = pts[sel[j]]
        dup = False
        for i in range(m_out):
            d = q - out_p[i]
            if _dot(d, d) < 1e-10:
                dup = True
                break
        if dup:
            continue
        out_p[m_out] = q
        out_s[m_out] = seps[sel[j]]
        out_src[m_out] = src[sel[j]]
        out_vid[m_out] = vid[sel[j]]
        m_out += 1
    return out_p[:m_out], out_s[:m_out], out_src[:m_out], out_vid[:m_out]


@njit(cache=True)
def manifold_halfspace(Vg, margin, max_points):
    """Contacts of a vertex set (in the ground frame, plane z = 0) with the half space z <= 0."""
    n = Vg.shape[0]
    idx = np.empty(n, dtype=np.int64)
    P2 = np.empty((n, 2))
    sep = np.empty(n)
    m = 0
    deepest = 0
    dmin = 1e300
    for i in range(n):
        z = Vg[i, 2]
        if z <= margin:
            idx[m] = i
            P2[m, 0] = Vg[i, 0]
            P2[m, 1] = Vg[i, 1]
            sep[m] = z
            if z < dmin:
                dmin = z
                deepest = m
            m += 1
    if m == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)
    sel = _tiered_reduce(P2[:m], sep[:m], max_points, 1e-4)
    out_p = np.zeros((sel.shape[0], 3))
    out_s = np.zeros(sel.shape[0])
    out_v = np.zeros(sel.shape[0], dtype=np.int64)
    for j in range(sel.shape[0]):
        i = sel[j]
        out_p[j, 0] = P2[i, 0]
        out_p[j, 1] = P2[i, 1]
        out_p[j, 2] = 0.5 * sep[i]
        out_s[j] = sep[i]
        out_v[j] = idx[i]
    return out_p, out_s, out_v


@njit(cache=True)
def _tangent_basis_nb(n):
    t1 = np.empty(3)
    t2 = np.empty(3)
    if n[2] < -0.9999999:
        t1[0] = 0.0
        t1[1] = -1.0
        t1[2] = 0.0
        t2[0] = -1.0
        t2[1] = 0.0
        t2[2] = 0.0
        return t1, t2
    a = 1.0 / (1.0 + n[2])
    b = -n[0] * n[1] * a
    t1[0] = 1.0 - n[0] * n[0] * a
    t1[1] = b
    t1[2] = -n[0]
    t2[0] = b
    t2[1] = 1.0 - n[1] * n[1] * a
    t2[2] = -n[1]
    return t1, t2


# ----------------------------------------------------------------------------------
# Contact generation for a World
# ----------------------------------------------------------------------------------


def _feature_id(a: int, b: int, source: int, vid: int) -> int:
    return ((a * 4096 + b) * 4 + source) * 1_000_003 + vid


class CollisionCache:
    """Per world cache: AABB tree, piece face arrays and a packed vertex layout so that all
    piece vertices are transformed to world space by one numba call per step."""

    def __init__(self, fat_margin: float = 0.01):
        self.tree = AABBTree(fat_margin)
        self.piece_faces: Dict[int, List[np.ndarray]] = {}
        self.piece_verts: Dict[int, List[np.ndarray]] = {}
        self._layout_key = None
        self._Vall = None
        self._offsets = None
        self._piece_body = None
        self._body_pieces: List[List[int]] = []

    def faces_for(self, body) -> List[np.ndarray]:
        key = id(body.shape)
        if key not in self.piece_faces:
            self.piece_faces[key] = [np.ascontiguousarray(p.faces, dtype=np.int64) for p in body.shape.pieces]
            self.piece_verts[key] = [np.ascontiguousarray(p.vertices, dtype=np.float64) for p in body.shape.pieces]
        return self.piece_faces[key]

    def body_verts(self, body) -> List[np.ndarray]:
        key = id(body.shape)
        if key not in self.piece_verts:
            self.faces_for(body)
        return self.piece_verts[key]

    def layout(self, bodies):
        key = tuple((b.id, id(b.shape)) for b in bodies)
        if key != self._layout_key:
            Vs, offs, pb, bp = [], [0], [], []
            for b in bodies:
                lst = []
                if b.shape.kind != "halfspace":
                    for V in self.body_verts(b):
                        lst.append(len(pb))
                        Vs.append(V)
                        offs.append(offs[-1] + V.shape[0])
                        pb.append(b.id)
                bp.append(lst)
            self._Vall = np.ascontiguousarray(np.concatenate(Vs)) if Vs else np.zeros((0, 3))
            self._offsets = np.array(offs, dtype=np.int64)
            self._piece_body = np.array(pb, dtype=np.int64)
            self._body_pieces = bp
            self._layout_key = key
        return self._Vall, self._offsets, self._piece_body, self._body_pieces


@njit(cache=True)
def _transform_all(Vall, offsets, piece_body, q, x):
    """World vertices of every piece plus per piece AABBs."""
    npc = offsets.shape[0] - 1
    W = np.empty_like(Vall)
    lo = np.empty((npc, 3))
    hi = np.empty((npc, 3))
    for p in range(npc):
        b = piece_body[p]
        qw, qx, qy, qz = q[b, 0], q[b, 1], q[b, 2], q[b, 3]
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qz * qw)
        r02 = 2 * (qx * qz + qy * qw)
        r10 = 2 * (qx * qy + qz * qw)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qx * qw)
        r20 = 2 * (qx * qz - qy * qw)
        r21 = 2 * (qy * qz + qx * qw)
        r22 = 1 - 2 * (qx * qx + qy * qy)
        l0 = l1 = l2 = 1e300
        h0 = h1 = h2 = -1e300
        for i in range(offsets[p], offsets[p + 1]):
            vx, vy, vz = Vall[i, 0], Vall[i, 1], Vall[i, 2]
            wx = r00 * vx + r01 * vy + r02 * vz + x[b, 0]
            wy = r10 * vx + r11 * vy + r12 * vz + x[b, 1]
            wz = r20 * vx + r21 * vy + r22 * vz + x[b, 2]
            W[i, 0] = wx
            W[i, 1] = wy
            W[i, 2] = wz
            if wx < l0:
                l0 = wx
            if wx > h0:
                h0 = wx
            if wy < l1:
                l1 = wy
            if wy > h1:
                h1 = wy
            if wz < l2:
                l2 = wz
            if wz > h2:
                h2 = wz
        lo[p, 0], lo[p, 1], lo[p, 2] = l0, l1, l2
        hi[p, 0], hi[p, 1], hi[p, 2] = h0, h1, h2
    return W, lo, hi


def world_pieces(body, cache: CollisionCache) -> List[np.ndarray]:
    """World space vertex arrays of all convex pieces of a body."""
    R = quat_to_rot(body.q)
    x = body.x
    return [np.ascontiguousarray(V @ R.T + x) for V in cache.body_verts(body)]


def find_contacts(world, margin: Optional[float] = None) -> List[Contact]:
    """All contacts of ``world`` whose separation is below ``margin`` (depth = -separation)."""
    margin = world.contact_margin if margin is None else float(margin)
    cache = world._collision_cache
    bodies = world.bodies
    nb = len(bodies)
    if nb == 0:
        return []
    # world vertices and AABBs (one numba call for all pieces)
    Vall, offsets, piece_body, body_pieces = cache.layout(bodies)
    q = np.empty((nb, 4))
    xs = np.empty((nb, 3))
    for i, b in enumerate(bodies):
        q[i] = b.q
        xs[i] = b.x
    W, plo, phi = _transform_all(Vall, offsets, piece_body, q, xs)
    wverts: List[List[np.ndarray]] = [None] * nb
    for b in bodies:
        if b.shape.kind == "halfspace":
            wverts[b.id] = []
            R = quat_to_rot(b.q)
            size = b.shape.bounding_radius
            corners = np.array([[sx * size, sy * size, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1.0, 0.0)])
            cw = corners @ R.T + b.x
            lo, hi = cw.min(axis=0), cw.max(axis=0)
        else:
            pcs = body_pieces[b.id]
            wverts[b.id] = [W[offsets[p] : offsets[p + 1]] for p in pcs]
            lo = plo[pcs].min(axis=0) if len(pcs) > 1 else plo[pcs[0]]
            hi = phi[pcs].max(axis=0) if len(pcs) > 1 else phi[pcs[0]]
        cache.tree.update(b.id, lo - margin, hi + margin)
    for item in list(cache.tree.leaf_of.keys()):
        if item >= nb or bodies[item].id != item:
            cache.tree.remove(item)
    pairs = cache.tree.query_pairs()
    contacts: List[Contact] = []
    max_pts = world.max_manifold_points
    for ia_, ib_ in pairs:
        A = bodies[ia_]
        B = bodies[ib_]
        if (A.static or A.kinematic) and (B.static or B.kinematic):
            continue
        if A.group >= 0 and A.group == B.group and world._skip_internal:
            # merged group: skip internal contacts (they are kept by the merge manager)
            pass
        if A.shape.kind == "halfspace" or B.shape.kind == "halfspace":
            if A.shape.kind == "halfspace":
                g, s, gi, si = A, B, ia_, ib_
                flip = False
            else:
                g, s, gi, si = B, A, ib_, ia_
                flip = True
            contacts.extend(_halfspace_contacts(world, g, s, wverts[si], margin, max_pts))
            continue
        mu = world.pair_friction(ia_, ib_)
        FA_list = cache.faces_for(A)
        FB_list = cache.faces_for(B)
        VA_list = wverts[ia_]
        VB_list = wverts[ib_]
        L = A.shape.bounding_radius + B.shape.bounding_radius + margin
        pair_pts = []
        single = len(VA_list) == 1 and len(VB_list) == 1
        for pa_i, VA in enumerate(VA_list):
            if not single:
                loA, hiA = VA.min(axis=0) - margin, VA.max(axis=0) + margin
            for pb_i, VB in enumerate(VB_list):
                if not single:
                    if np.any(VB.max(axis=0) < loA) or np.any(VB.min(axis=0) > hiA):
                        continue
                ok, sep, n, pa, pb = convex_query(VA, VB)
                if not ok or sep > margin:
                    continue
                t1, t2 = _tangent_basis_nb(n)
                pts, seps, src, vid = manifold_convex(VA, FA_list[pa_i], VB, FB_list[pb_i], n, pa, pb, sep, margin, max_pts, t1, t2, L)
                for j in range(pts.shape[0]):
                    pair_pts.append((pts[j], seps[j], n, t1, t2, int(src[j]), int(vid[j]) + 100003 * (pa_i * 64 + pb_i)))
        if not pair_pts:
            continue
        if len(pair_pts) > max_pts:
            P2 = np.array([[p[0] @ p[3], p[0] @ p[4]] for p in pair_pts])
            seps = np.array([p[1] for p in pair_pts])
            keep = _tiered_reduce(P2, seps, max_pts, 1e-4)
            pair_pts = [pair_pts[i] for i in keep]
        for p, sep, n, t1, t2, src, vid in pair_pts:
            contacts.append(Contact(ia_, ib_, p, n, -float(sep), t1, t2, mu, feature=_feature_id(ia_, ib_, src, vid)))
    return contacts


def _halfspace_contacts(world, ground, body, wv: List[np.ndarray], margin: float, max_pts: int) -> List[Contact]:
    """Contacts between a body and a half space ground (plane z = 0 in the ground frame)."""
    Rg = quat_to_rot(ground.q)
    n_world = Rg[:, 2]
    mu = world.pair_friction(ground.id, body.id)
    out = []
    t1, t2 = tangent_basis(n_world)
    all_pts = []
    for pi, V in enumerate(wv):
        Vg = np.ascontiguousarray((V - ground.x) @ Rg)
        pts, seps, vids = manifold_halfspace(Vg, margin, max_pts)
        for j in range(pts.shape[0]):
            pw = pts[j] @ Rg.T + ground.x
            all_pts.append((pw, float(seps[j]), int(vids[j]) + 100003 * pi))
    if len(all_pts) > max_pts:
        P2 = np.array([[p[0] @ t1, p[0] @ t2] for p in all_pts])
        seps = np.array([p[1] for p in all_pts])
        keep = _tiered_reduce(P2, seps, max_pts, 1e-4)
        all_pts = [all_pts[i] for i in keep]
    for pw, sep, vid in all_pts:
        out.append(Contact(ground.id, body.id, pw, n_world.copy(), -sep, t1, t2, mu, feature=_feature_id(ground.id, body.id, 3, vid)))
    return out


def verify_contacts_sdf(world, contacts: List[Contact], tol: float = 1e-3) -> dict:
    """Cross check contact points against the signed distance fields of the bodies.

    For each contact the point is mapped into both body frames; the sum of the two signed
    distances should equal the separation (-depth) within ``tol``. Bodies without an SDF
    (ground half spaces use the analytic plane) are evaluated analytically.
    """
    errs = []
    checked = 0
    for c in contacts:
        vals = []
        for bid in (c.body_a, c.body_b):
            b = world.bodies[bid]
            pl = b.pose.inverse().apply(c.point)
            if b.shape.kind == "halfspace":
                vals.append(float(pl[2]))
            elif b.shape.sdf is not None:
                vals.append(float(b.shape.sdf.query(pl[None, :])[0]))
            else:
                vals.append(None)
        if None in vals:
            continue
        checked += 1
        errs.append(abs(vals[0] + vals[1] - (-c.depth)))
    return {"checked": checked, "max_discrepancy": float(max(errs)) if errs else 0.0, "ok": (max(errs) if errs else 0.0) <= tol}
