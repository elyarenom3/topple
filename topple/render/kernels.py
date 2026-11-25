"""Numba kernels of the path tracer (Section 10.1).

Unidirectional path tracing with next event estimation and multiple importance sampling
(Kajiya 1986, Veach 1997, Pharr, Jakob and Humphreys 2023) over area lights, the sun disc and
the sky. Russian roulette from a configurable depth, separate depth limits for diffuse and
specular chains, thin lens camera with a polygonal aperture, motion blur by time sampling with
per instance transform interpolation, an optional homogeneous slab medium with single scattering
by distance sampling, random walk subsurface scattering, and overlay compositing in the same
camera sample so overlays share depth of field, motion blur and occlusion.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..core.raycast import ray_aabb, ray_tri, _safe_inv
from .materials import (
    M_ABSORB, M_CAUCHY_B, M_IOR,
    M_ALPHA, M_BASE, M_BLEND, M_EMIT, M_LIGHT, M_OUTLINE, M_TYPE, M_USE_VCOL, M_ANISO_DIR,
    S_BASE, S_COAT, S_COAT_IOR, S_EMIT, S_FLAKE_ON, S_METAL, S_N, S_NCOLS, S_ROUGH, S_SSS_MFP, S_SSS_W, S_SPEC, S_AX, S_AY, S_DIFFUSE_MODEL, S_FLAKE_INT,
    bsdf_eval, bsdf_sample, fresnel_dielectric, luminance3, shade_material,
)
from .sampling import SOBOL_V, hash_u32, polygon_aperture_sample, rng_u01, sobol_owen
from .lens import lens_sample
from .spectral import S_MNEE, S_SPECTRAL, cauchy_ior, to_rgb, upsample, upsample_positive
from .sky import sky_radiance

PI = np.pi
INF = 1e30
EPS_RAY = 1e-5

# ------------------------------------------------------------------ quaternions


@njit(cache=True, inline="always")
def q_rot(qw, qx, qy, qz, vx, vy, vz):
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


@njit(cache=True, inline="always")
def q_rot_inv(qw, qx, qy, qz, vx, vy, vz):
    return q_rot(qw, -qx, -qy, -qz, vx, vy, vz)


@njit(cache=True)
def inst_xform(i_q0, i_t0, i_q1, i_t1, i_moving, i, u):
    """Transform of instance i at shutter fraction u: (qw, qx, qy, qz, tx, ty, tz)."""
    if i_moving[i] == 0 or u <= 0.0:
        return i_q0[i, 0], i_q0[i, 1], i_q0[i, 2], i_q0[i, 3], i_t0[i, 0], i_t0[i, 1], i_t0[i, 2]
    a0, a1, a2, a3 = i_q0[i, 0], i_q0[i, 1], i_q0[i, 2], i_q0[i, 3]
    b0, b1, b2, b3 = i_q1[i, 0], i_q1[i, 1], i_q1[i, 2], i_q1[i, 3]
    d = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
    if d < 0.0:
        b0, b1, b2, b3 = -b0, -b1, -b2, -b3
        d = -d
    if d > 0.9995:
        w0 = 1.0 - u
        w1 = u
    else:
        th = np.arccos(min(d, 1.0))
        s = np.sin(th)
        w0 = np.sin((1.0 - u) * th) / s
        w1 = np.sin(u * th) / s
    qw = w0 * a0 + w1 * b0
    qx = w0 * a1 + w1 * b1
    qy = w0 * a2 + w1 * b2
    qz = w0 * a3 + w1 * b3
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw /= n
    qx /= n
    qy /= n
    qz /= n
    tx = (1.0 - u) * i_t0[i, 0] + u * i_t1[i, 0]
    ty = (1.0 - u) * i_t0[i, 1] + u * i_t1[i, 1]
    tz = (1.0 - u) * i_t0[i, 2] + u * i_t1[i, 2]
    return qw, qx, qy, qz, tx, ty, tz


# ------------------------------------------------------------------ traversal


@njit(cache=True)
def mesh_hit(o, d, tmin, tmax, m, node_off, tri_off, bmin, bmax, left, right, start, count, tv0, te1, te2):
    """Closest hit against mesh m of a geometry pack. Returns (t, global tri index, u, v)."""
    stack = np.empty(64, dtype=np.int64)
    sp = 0
    stack[sp] = node_off[m]
    sp += 1
    noff = node_off[m]
    toff = tri_off[m]
    best_t = tmax
    best_i = -1
    bu = 0.0
    bv = 0.0
    idx_ = _safe_inv(d[0])
    idy = _safe_inv(d[1])
    idz = _safe_inv(d[2])
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not ray_aabb(o[0], o[1], o[2], idx_, idy, idz, bmin[node], bmax[node], tmin, best_t):
            continue
        if count[node] > 0:
            s = toff + start[node]
            for k in range(s, s + count[node]):
                t, u, v = ray_tri(o, d, tv0[k], te1[k], te2[k], tmin, best_t)
                if t >= 0.0 and t < best_t:
                    best_t = t
                    best_i = k
                    bu = u
                    bv = v
        else:
            stack[sp] = noff + left[node]
            sp += 1
            stack[sp] = noff + right[node]
            sp += 1
    return best_t, best_i, bu, bv


@njit(cache=True)
def mesh_any(o, d, tmin, tmax, m, node_off, tri_off, bmin, bmax, left, right, start, count, tv0, te1, te2):
    stack = np.empty(64, dtype=np.int64)
    sp = 0
    stack[sp] = node_off[m]
    sp += 1
    noff = node_off[m]
    toff = tri_off[m]
    idx_ = _safe_inv(d[0])
    idy = _safe_inv(d[1])
    idz = _safe_inv(d[2])
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not ray_aabb(o[0], o[1], o[2], idx_, idy, idz, bmin[node], bmax[node], tmin, tmax):
            continue
        if count[node] > 0:
            s = toff + start[node]
            for k in range(s, s + count[node]):
                t, u, v = ray_tri(o, d, tv0[k], te1[k], te2[k], tmin, tmax)
                if t >= 0.0:
                    return True
        else:
            stack[sp] = noff + left[node]
            sp += 1
            stack[sp] = noff + right[node]
            sp += 1
    return False


@njit(cache=True)
def scene_hit(o, d, ut, tmin, tmax, G):
    """Closest hit over all instances of pack G at shutter fraction ut.

    Returns (t, instance, global tri, u, v); instance = -1 on miss.
    """
    (node_off, tri_off, vert_off, bmin, bmax, left, right, start, count, tv0, te1, te2, order, faces, vnorm, vattr, vcol,
     i_mesh, i_mat, i_obj, i_q0, i_t0, i_q1, i_t1, i_moving, i_light,
     t_bmin, t_bmax, t_left, t_right, t_start, t_count, t_order) = G
    best_t = tmax
    best_inst = -1
    best_tri = -1
    bu = 0.0
    bv = 0.0
    if t_count[0] == 0 and t_left[0] < 0:
        return best_t, best_inst, best_tri, bu, bv
    stack = np.empty(64, dtype=np.int64)
    sp = 0
    stack[sp] = 0
    sp += 1
    idx_ = _safe_inv(d[0])
    idy = _safe_inv(d[1])
    idz = _safe_inv(d[2])
    ol = np.empty(3)
    dl = np.empty(3)
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not ray_aabb(o[0], o[1], o[2], idx_, idy, idz, t_bmin[node], t_bmax[node], tmin, best_t):
            continue
        if t_count[node] > 0:
            for k in range(t_start[node], t_start[node] + t_count[node]):
                i = t_order[k]
                qw, qx, qy, qz, tx, ty, tz = inst_xform(i_q0, i_t0, i_q1, i_t1, i_moving, i, ut)
                ox, oy, oz = q_rot_inv(qw, qx, qy, qz, o[0] - tx, o[1] - ty, o[2] - tz)
                dx, dy, dz = q_rot_inv(qw, qx, qy, qz, d[0], d[1], d[2])
                ol[0] = ox
                ol[1] = oy
                ol[2] = oz
                dl[0] = dx
                dl[1] = dy
                dl[2] = dz
                t, tri, u, v = mesh_hit(ol, dl, tmin, best_t, i_mesh[i], node_off, tri_off, bmin, bmax, left, right, start, count, tv0, te1, te2)
                if tri >= 0 and t < best_t:
                    best_t = t
                    best_inst = i
                    best_tri = tri
                    bu = u
                    bv = v
        else:
            stack[sp] = t_left[node]
            sp += 1
            stack[sp] = t_right[node]
            sp += 1
    return best_t, best_inst, best_tri, bu, bv


@njit(cache=True)
def scene_occluded(o, d, ut, tmin, tmax, G):
    (node_off, tri_off, vert_off, bmin, bmax, left, right, start, count, tv0, te1, te2, order, faces, vnorm, vattr, vcol,
     i_mesh, i_mat, i_obj, i_q0, i_t0, i_q1, i_t1, i_moving, i_light,
     t_bmin, t_bmax, t_left, t_right, t_start, t_count, t_order) = G
    if t_count[0] == 0 and t_left[0] < 0:
        return False
    stack = np.empty(64, dtype=np.int64)
    sp = 0
    stack[sp] = 0
    sp += 1
    idx_ = _safe_inv(d[0])
    idy = _safe_inv(d[1])
    idz = _safe_inv(d[2])
    ol = np.empty(3)
    dl = np.empty(3)
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not ray_aabb(o[0], o[1], o[2], idx_, idy, idz, t_bmin[node], t_bmax[node], tmin, tmax):
            continue
        if t_count[node] > 0:
            for k in range(t_start[node], t_start[node] + t_count[node]):
                i = t_order[k]
                qw, qx, qy, qz, tx, ty, tz = inst_xform(i_q0, i_t0, i_q1, i_t1, i_moving, i, ut)
                ox, oy, oz = q_rot_inv(qw, qx, qy, qz, o[0] - tx, o[1] - ty, o[2] - tz)
                dx, dy, dz = q_rot_inv(qw, qx, qy, qz, d[0], d[1], d[2])
                ol[0] = ox
                ol[1] = oy
                ol[2] = oz
                dl[0] = dx
                dl[1] = dy
                dl[2] = dz
                if mesh_any(ol, dl, tmin, tmax, i_mesh[i], node_off, tri_off, bmin, bmax, left, right, start, count, tv0, te1, te2):
                    return True
        else:
            stack[sp] = t_left[node]
            sp += 1
            stack[sp] = t_right[node]
            sp += 1
    return False


@njit(cache=True)
def hit_data(G, inst, tri, u, v, ut, o, d, t, out):
    """Fill ``out`` (24): world p 0:3, geometric n 3:6, shading n 6:9, local p 9:12,
    local shading n 12:15, curvature 15, occlusion 16, vertex colour 17:20, local geometric n 20:23.
    Returns the instance quaternion components and translation."""
    (node_off, tri_off, vert_off, bmin, bmax, left, right, start, count, tv0, te1, te2, order, faces, vnorm, vattr, vcol,
     i_mesh, i_mat, i_obj, i_q0, i_t0, i_q1, i_t1, i_moving, i_light,
     t_bmin, t_bmax, t_left, t_right, t_start, t_count, t_order) = G
    m = i_mesh[inst]
    qw, qx, qy, qz, tx, ty, tz = inst_xform(i_q0, i_t0, i_q1, i_t1, i_moving, inst, ut)
    # local ray
    ox, oy, oz = q_rot_inv(qw, qx, qy, qz, o[0] - tx, o[1] - ty, o[2] - tz)
    dx, dy, dz = q_rot_inv(qw, qx, qy, qz, d[0], d[1], d[2])
    plx = ox + t * dx
    ply = oy + t * dy
    plz = oz + t * dz
    # geometric normal
    e1 = te1[tri]
    e2 = te2[tri]
    gx = e1[1] * e2[2] - e1[2] * e2[1]
    gy = e1[2] * e2[0] - e1[0] * e2[2]
    gz = e1[0] * e2[1] - e1[1] * e2[0]
    gn = np.sqrt(gx * gx + gy * gy + gz * gz)
    if gn > 0:
        gx /= gn
        gy /= gn
        gz /= gn
    # shading normal from vertex normals
    f = order[tri]  # local face index within mesh m (tri is global; order is concatenated per mesh)
    fo = tri_off[m]
    face = faces[fo + f]
    vo = vert_off[m]
    w0 = 1.0 - u - v
    i0 = vo + face[0]
    i1 = vo + face[1]
    i2 = vo + face[2]
    sx = w0 * vnorm[i0, 0] + u * vnorm[i1, 0] + v * vnorm[i2, 0]
    sy = w0 * vnorm[i0, 1] + u * vnorm[i1, 1] + v * vnorm[i2, 1]
    sz = w0 * vnorm[i0, 2] + u * vnorm[i1, 2] + v * vnorm[i2, 2]
    sn = np.sqrt(sx * sx + sy * sy + sz * sz)
    if sn > 1e-12:
        sx /= sn
        sy /= sn
        sz /= sn
    else:
        sx, sy, sz = gx, gy, gz
    # keep shading normal on the geometric side
    if sx * gx + sy * gy + sz * gz < 0.0:
        sx, sy, sz = -sx, -sy, -sz
    curv = w0 * vattr[i0, 0] + u * vattr[i1, 0] + v * vattr[i2, 0]
    occ = w0 * vattr[i0, 1] + u * vattr[i1, 1] + v * vattr[i2, 1]
    out[0], out[1], out[2] = o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]
    wgx, wgy, wgz = q_rot(qw, qx, qy, qz, gx, gy, gz)
    out[3], out[4], out[5] = wgx, wgy, wgz
    wsx, wsy, wsz = q_rot(qw, qx, qy, qz, sx, sy, sz)
    out[6], out[7], out[8] = wsx, wsy, wsz
    out[9], out[10], out[11] = plx, ply, plz
    out[12], out[13], out[14] = sx, sy, sz
    out[15] = curv
    out[16] = occ
    for k in range(3):
        out[17 + k] = w0 * vcol[i0, k] + u * vcol[i1, k] + v * vcol[i2, k]
    out[20], out[21], out[22] = gx, gy, gz
    return qw, qx, qy, qz, tx, ty, tz


# ------------------------------------------------------------------- lights


@njit(cache=True)
def env_lookup(dx, dy, dz, env_img, rot):
    H = env_img.shape[0]
    W = env_img.shape[1]
    th = np.arccos(min(max(dz, -1.0), 1.0))
    ph = np.arctan2(dy, dx) - rot
    u = (ph / (2.0 * PI)) % 1.0
    v = th / PI
    i = min(int(v * H), H - 1)
    j = min(int(u * W), W - 1)
    return env_img[i, j, 0], env_img[i, j, 1], env_img[i, j, 2], i, j


@njit(cache=True)
def env_pdf_dir(dz, i, j, env_marg, env_cond):
    H = env_marg.shape[0]
    W = env_cond.shape[1]
    p_row = env_marg[i] - (env_marg[i - 1] if i > 0 else 0.0)
    p_col = env_cond[i, j] - (env_cond[i, j - 1] if j > 0 else 0.0)
    sin_th = np.sqrt(max(1.0 - dz * dz, 1e-9))
    return p_row * p_col * H * W / (2.0 * PI * PI * sin_th)


@njit(cache=True)
def env_radiance(dx, dy, dz, lrow, sky_p, M, env_img):
    """Radiance of the sky light (type 2 analytic, 3 hdr) in direction d."""
    if lrow[0] == 2:
        r, g, b = sky_radiance(dx, dy, dz, sky_p, M)
        s = lrow[13]
        return r * s, g * s, b * s
    r, g, b, i, j = env_lookup(dx, dy, dz, env_img, lrow[14])
    s = lrow[13]
    return r * s, g * s, b * s


@njit(cache=True)
def sample_light(lights, l, px, py, pz, nx, ny, nz, u1, u2, sky_p, M, env_img, env_marg, env_cond, out):
    """Sample light l from point p with shading normal n.

    out: 0:3 direction, 3 distance, 4:7 radiance. Returns pdf (solid angle), 0 if invalid.
    """
    lr = lights[l]
    typ = int(lr[0])
    if typ == 0:
        xx = lr[1] + u1 * lr[4] + u2 * lr[7]
        xy = lr[2] + u1 * lr[5] + u2 * lr[8]
        xz = lr[3] + u1 * lr[6] + u2 * lr[9]
        wx, wy, wz = xx - px, xy - py, xz - pz
        dist = np.sqrt(wx * wx + wy * wy + wz * wz)
        if dist < 1e-9:
            return 0.0
        wx /= dist
        wy /= dist
        wz /= dist
        cos_l = -(wx * lr[14] + wy * lr[15] + wz * lr[16])
        if cos_l <= 1e-6:
            return 0.0
        out[0], out[1], out[2] = wx, wy, wz
        out[3] = dist
        out[4], out[5], out[6] = lr[10], lr[11], lr[12]
        return dist * dist / (lr[13] * cos_l)
    if typ == 1:
        cos_half = lr[13]
        ct = 1.0 - u1 * (1.0 - cos_half)
        st = np.sqrt(max(0.0, 1.0 - ct * ct))
        ph = 2.0 * PI * u2
        sx, sy, sz = lr[1], lr[2], lr[3]
        if abs(sx) < 0.9:
            tx, ty, tz = 0.0, -sz, sy
        else:
            tx, ty, tz = sz, 0.0, -sx
        tn = np.sqrt(tx * tx + ty * ty + tz * tz)
        tx /= tn
        ty /= tn
        tz /= tn
        bx = sy * tz - sz * ty
        by = sz * tx - sx * tz
        bz = sx * ty - sy * tx
        out[0] = st * np.cos(ph) * tx + st * np.sin(ph) * bx + ct * sx
        out[1] = st * np.cos(ph) * ty + st * np.sin(ph) * by + ct * sy
        out[2] = st * np.cos(ph) * tz + st * np.sin(ph) * bz + ct * sz
        out[3] = INF
        out[4], out[5], out[6] = lr[10], lr[11], lr[12]
        return 1.0 / (2.0 * PI * (1.0 - cos_half))
    if typ == 2:
        # cosine weighted hemisphere about n
        r = np.sqrt(u1)
        ph = 2.0 * PI * u2
        lx = r * np.cos(ph)
        ly = r * np.sin(ph)
        lz = np.sqrt(max(0.0, 1.0 - u1))
        if abs(nx) < 0.9:
            tx, ty, tz = 0.0, -nz, ny
        else:
            tx, ty, tz = nz, 0.0, -nx
        tn = np.sqrt(tx * tx + ty * ty + tz * tz)
        tx /= tn
        ty /= tn
        tz /= tn
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx
        wx = lx * tx + ly * bx + lz * nx
        wy = lx * ty + ly * by + lz * ny
        wz = lx * tz + ly * bz + lz * nz
        out[0], out[1], out[2] = wx, wy, wz
        out[3] = INF
        r_, g_, b_ = env_radiance(wx, wy, wz, lr, sky_p, M, env_img)
        out[4], out[5], out[6] = r_, g_, b_
        return lz / PI
    # hdr map importance sampling
    H = env_marg.shape[0]
    W = env_cond.shape[1]
    # binary search rows
    lo = 0
    hi = H - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if env_marg[mid] < u1:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    lo = 0
    hi = W - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if env_cond[i, mid] < u2:
            lo = mid + 1
        else:
            hi = mid
    j = lo
    th = PI * (i + 0.5) / H
    ph = 2.0 * PI * (j + 0.5) / W + lr[14]
    wx = np.sin(th) * np.cos(ph)
    wy = np.sin(th) * np.sin(ph)
    wz = np.cos(th)
    out[0], out[1], out[2] = wx, wy, wz
    out[3] = INF
    s = lr[13]
    out[4], out[5], out[6] = env_img[i, j, 0] * s, env_img[i, j, 1] * s, env_img[i, j, 2] * s
    return env_pdf_dir(wz, i, j, env_marg, env_cond)


@njit(cache=True)
def escaped_radiance(dx, dy, dz, lights, nL, sky_index, sky_p, M, env_img, env_marg, env_cond, background, pdf_bsdf, nprev_x, nprev_y, nprev_z, use_mis, out, skip_sun=False):
    """Radiance for a ray leaving the scene (sun disc and sky with MIS weights). ``skip_sun``
    drops the sun disc for paths that manifold next event estimation already counts."""
    out[0] = background[0]
    out[1] = background[1]
    out[2] = background[2]
    for l in range(nL):
        lr = lights[l]
        typ = int(lr[0])
        if typ == 1:
            c = dx * lr[1] + dy * lr[2] + dz * lr[3]
            if c >= lr[13] and not skip_sun:
                w = 1.0
                if use_mis:
                    pl = (1.0 / nL) / (2.0 * PI * (1.0 - lr[13]))
                    w = pdf_bsdf * pdf_bsdf / (pdf_bsdf * pdf_bsdf + pl * pl)
                out[0] += lr[10] * w
                out[1] += lr[11] * w
                out[2] += lr[12] * w
        elif typ == 2 or typ == 3:
            r, g, b = env_radiance(dx, dy, dz, lr, sky_p, M, env_img)
            w = 1.0
            if use_mis:
                if typ == 2:
                    c = max(dx * nprev_x + dy * nprev_y + dz * nprev_z, 0.0)
                    pl = (1.0 / nL) * c / PI
                else:
                    rr, gg, bb, i, j = env_lookup(dx, dy, dz, env_img, lr[14])
                    pl = (1.0 / nL) * env_pdf_dir(dz, i, j, env_marg, env_cond)
                w = pdf_bsdf * pdf_bsdf / (pdf_bsdf * pdf_bsdf + pl * pl)
            out[0] += r * w
            out[1] += g * w
            out[2] += b * w


# ------------------------------------------------------------------- medium


@njit(cache=True)
def slab_interval(oz, dz, tmax, zmax):
    """Parametric interval [ta, tb] of a ray inside the slab 0 <= z <= zmax."""
    if abs(dz) < 1e-12:
        if 0.0 <= oz <= zmax:
            return 0.0, tmax
        return 0.0, 0.0
    t0 = (0.0 - oz) / dz
    t1 = (zmax - oz) / dz
    ta = max(min(t0, t1), 0.0)
    tb = min(max(t0, t1), tmax)
    if tb <= ta:
        return 0.0, 0.0
    return ta, tb


@njit(cache=True)
def hg_phase(cos_theta, g):
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * PI * denom * np.sqrt(denom))


# --------------------------------------------------------------------- frames


@njit(cache=True)
def make_frame(nx, ny, nz, ax, ay, az, out_t, out_b):
    """Tangent frame with t along the projection of a (anisotropy direction) onto the plane."""
    d = ax * nx + ay * ny + az * nz
    tx = ax - d * nx
    ty = ay - d * ny
    tz = az - d * nz
    n = np.sqrt(tx * tx + ty * ty + tz * tz)
    if n < 1e-6:
        if abs(nx) < 0.9:
            tx, ty, tz = 0.0, -nz, ny
        else:
            tx, ty, tz = nz, 0.0, -nx
        n = np.sqrt(tx * tx + ty * ty + tz * tz)
    out_t[0], out_t[1], out_t[2] = tx / n, ty / n, tz / n
    out_b[0] = ny * out_t[2] - nz * out_t[1]
    out_b[1] = nz * out_t[0] - nx * out_t[2]
    out_b[2] = nx * out_t[1] - ny * out_t[0]


@njit(cache=True, inline="always")
def to_local(vx, vy, vz, t, b, nx, ny, nz, out):
    out[0] = vx * t[0] + vy * t[1] + vz * t[2]
    out[1] = vx * b[0] + vy * b[1] + vz * b[2]
    out[2] = vx * nx + vy * ny + vz * nz


@njit(cache=True, inline="always")
def to_world(l, t, b, nx, ny, nz):
    return (
        l[0] * t[0] + l[1] * b[0] + l[2] * nx,
        l[0] * t[1] + l[1] * b[1] + l[2] * ny,
        l[0] * t[2] + l[1] * b[2] + l[2] * nz,
    )


# -------------------------------------------------------------------- sampler


@njit(cache=True, inline="always")
def get_u(dim, sidx, gseed, pseed, px, py, mask, rng_state):
    if dim < 4:
        u = sobol_owen(sidx, dim, gseed, SOBOL_V)
        n = mask.shape[0]
        u = u + mask[(py + 7 * dim) % n, (px + 13 * dim) % n]
        if u >= 1.0:
            u -= 1.0
        return u
    if dim < 64:
        return sobol_owen(sidx, dim, pseed, SOBOL_V)
    return rng_u01(rng_state)


# ----------------------------------------------------------------- subsurface


@njit(cache=True)
def random_walk_sss(G, inst, qw, qx, qy, qz, tx, ty, tz, pl, nl, sp, rng_state, thr, out_p, out_n):
    """Random walk inside the mesh of ``inst`` starting at local point pl with local outward
    normal nl. Updates throughput thr; writes exit point (world) and outward normal (world).
    Returns True on exit, False if the walk was terminated."""
    (node_off, tri_off, vert_off, bmin, bmax, left, right, start, count, tv0, te1, te2, order, faces, vnorm, vattr, vcol,
     i_mesh, i_mat, i_obj, i_q0, i_t0, i_q1, i_t1, i_moving, i_light,
     t_bmin, t_bmax, t_left, t_right, t_start, t_count, t_order) = G
    m = i_mesh[inst]
    mfp = (sp[S_SSS_MFP] + sp[S_SSS_MFP + 1] + sp[S_SSS_MFP + 2]) / 3.0
    if mfp <= 0.0:
        return False
    o = np.empty(3)
    d = np.empty(3)
    # cosine sampled direction into the surface
    u1 = rng_u01(rng_state)
    u2 = rng_u01(rng_state)
    r = np.sqrt(u1)
    ph = 2.0 * PI * u2
    lx = r * np.cos(ph)
    ly = r * np.sin(ph)
    lz = np.sqrt(max(0.0, 1.0 - u1))
    nx, ny, nz = -nl[0], -nl[1], -nl[2]
    if abs(nx) < 0.9:
        ax, ay, az = 0.0, -nz, ny
    else:
        ax, ay, az = nz, 0.0, -nx
    an = np.sqrt(ax * ax + ay * ay + az * az)
    ax /= an
    ay /= an
    az /= an
    bx = ny * az - nz * ay
    by = nz * ax - nx * az
    bz = nx * ay - ny * ax
    d[0] = lx * ax + ly * bx + lz * nx
    d[1] = lx * ay + ly * by + lz * ny
    d[2] = lx * az + ly * bz + lz * nz
    o[0] = pl[0] - 1e-5 * nl[0]
    o[1] = pl[1] - 1e-5 * nl[1]
    o[2] = pl[2] - 1e-5 * nl[2]
    for step in range(48):
        s = -np.log(max(rng_u01(rng_state), 1e-12)) * mfp
        t, tri, u, v = mesh_hit(o, d, 1e-6, s, m, node_off, tri_off, bmin, bmax, left, right, start, count, tv0, te1, te2)
        if tri >= 0:
            # exit through the surface
            e1 = te1[tri]
            e2 = te2[tri]
            gx = e1[1] * e2[2] - e1[2] * e2[1]
            gy = e1[2] * e2[0] - e1[0] * e2[2]
            gz = e1[0] * e2[1] - e1[1] * e2[0]
            gn = np.sqrt(gx * gx + gy * gy + gz * gz)
            gx /= gn
            gy /= gn
            gz /= gn
            if gx * d[0] + gy * d[1] + gz * d[2] < 0.0:
                gx, gy, gz = -gx, -gy, -gz
            ex = o[0] + t * d[0]
            ey = o[1] + t * d[1]
            ez = o[2] + t * d[2]
            wx, wy, wz = q_rot(qw, qx, qy, qz, ex, ey, ez)
            out_p[0], out_p[1], out_p[2] = wx + tx, wy + ty, wz + tz
            out_n[0], out_n[1], out_n[2] = q_rot(qw, qx, qy, qz, gx, gy, gz)
            return True
        o[0] += s * d[0]
        o[1] += s * d[1]
        o[2] += s * d[2]
        for k in range(3):
            thr[k] *= sp[S_BASE + k]
        # isotropic scattering
        z = 1.0 - 2.0 * rng_u01(rng_state)
        rr = np.sqrt(max(0.0, 1.0 - z * z))
        ph = 2.0 * PI * rng_u01(rng_state)
        d[0] = rr * np.cos(ph)
        d[1] = rr * np.sin(ph)
        d[2] = z
    return False


# ------------------------------------------------------------------- overlay


@njit(cache=True)
def composite_overlays(o, d, ut, t_main, OV, mats, out_rgba):
    """Front to back compositing of overlay hits closer than t_main into out_rgba (premultiplied)."""
    (node_off, tri_off, vert_off, bmin, bmax, left, right, start, count, tv0, te1, te2, order, faces, vnorm, vattr, vcol,
     i_mesh, i_mat, i_obj, i_q0, i_t0, i_q1, i_t1, i_moving, i_light,
     t_bmin, t_bmax, t_left, t_right, t_start, t_count, t_order) = OV
    acc_a = 0.0
    r_acc = 0.0
    g_acc = 0.0
    b_acc = 0.0
    tcur = EPS_RAY
    hd = np.empty(24)
    oo = np.empty(3)
    oo[0], oo[1], oo[2] = o[0], o[1], o[2]
    for layer in range(12):
        t, inst, tri, u, v = scene_hit(oo, d, ut, tcur, t_main, OV)
        if inst < 0:
            break
        hit_data(OV, inst, tri, u, v, ut, oo, d, t, hd)
        mat = mats[i_mat[inst]]
        if hd[3] * d[0] + hd[4] * d[1] + hd[5] * d[2] > 0.0:
            # back face of a closed overlay solid: skip, so each element composites once
            tcur = t + 1e-6
            continue
        nx, ny, nz = hd[6], hd[7], hd[8]
        ndv = abs(nx * d[0] + ny * d[1] + nz * d[2])
        cr = mat[M_EMIT]
        cg = mat[M_EMIT + 1]
        cb = mat[M_EMIT + 2]
        if mat[M_USE_VCOL] > 0.5:
            cr *= hd[17]
            cg *= hd[18]
            cb *= hd[19]
        a = mat[M_ALPHA]
        if mat[M_TYPE] == 2:
            # glass shell: Fresnel rim
            rim = (1.0 - ndv) ** 3
            a = a + (1.0 - a) * rim
            cr = cr * (0.4 + 0.6 * rim) + 0.3 * rim
            cg = cg * (0.4 + 0.6 * rim) + 0.3 * rim
            cb = cb * (0.4 + 0.6 * rim) + 0.3 * rim
        elif mat[M_OUTLINE] > 0.0 and ndv < mat[M_OUTLINE]:
            cr, cg, cb = 0.02, 0.02, 0.02
            a = 1.0
        if mat[M_BLEND] > 0.5:
            r_acc += (1.0 - acc_a) * a * cr
            g_acc += (1.0 - acc_a) * a * cg
            b_acc += (1.0 - acc_a) * a * cb
        else:
            r_acc += (1.0 - acc_a) * a * cr
            g_acc += (1.0 - acc_a) * a * cg
            b_acc += (1.0 - acc_a) * a * cb
            acc_a += (1.0 - acc_a) * a
        tcur = t + 1e-6
        if acc_a > 0.995:
            break
    out_rgba[0] += r_acc
    out_rgba[1] += g_acc
    out_rgba[2] += b_acc
    out_rgba[3] += acc_a


# --------------------------------------------------------------------- camera


@njit(cache=True)
def camera_ray(cam, px, py, u0, u1, u2, u3, u4, o, d):
    """Generate a world ray for pixel (px, py) with jitter (u0, u1), lens (u2, u3), triangle pick u4.
    Returns the sample weight (1 for the thin lens; the traced lens weight, 0 when blocked)."""
    W = cam[21]
    H = cam[22]
    R00, R01, R02 = cam[3], cam[4], cam[5]
    R10, R11, R12 = cam[6], cam[7], cam[8]
    R20, R21, R22 = cam[9], cam[10], cam[11]
    if cam[23] > 0.5 and cam[18] < 0.5:
        # traced lens system (Kolb et al. 1995): the image is inverted on the film
        fx = -((px + u0) / W - 0.5) * cam[13]
        fy = -((py + u1) / H - 0.5) * cam[14]
        lo = np.empty(6)
        wgt = lens_sample(cam, fx, fy, u2, u3, lo)
        if wgt <= 0.0:
            return 0.0
        o[0] = cam[0] + R00 * lo[0] + R01 * lo[1] + R02 * lo[2]
        o[1] = cam[1] + R10 * lo[0] + R11 * lo[1] + R12 * lo[2]
        o[2] = cam[2] + R20 * lo[0] + R21 * lo[1] + R22 * lo[2]
        d[0] = R00 * lo[3] + R01 * lo[4] + R02 * lo[5]
        d[1] = R10 * lo[3] + R11 * lo[4] + R12 * lo[5]
        d[2] = R20 * lo[3] + R21 * lo[4] + R22 * lo[5]
        return wgt
    if cam[18] > 0.5:
        x = ((px + u0) / W - 0.5) * cam[19]
        y = ((py + u1) / H - 0.5) * cam[20]
        o[0] = cam[0] + R00 * x + R01 * y
        o[1] = cam[1] + R10 * x + R11 * y
        o[2] = cam[2] + R20 * x + R21 * y
        d[0], d[1], d[2] = R02, R12, R22
        return 1.0
    sx = ((px + u0) / W - 0.5) * cam[13]
    sy = ((py + u1) / H - 0.5) * cam[14]
    f = cam[12]
    # point on the plane of focus (camera space)
    s = cam[16]
    fx = sx * s / f
    fy = sy * s / f
    fz = s
    ap = cam[15]
    if ap > 0.0:
        lx, ly = polygon_aperture_sample(u4, u2, u3, int(cam[17]))
        lx *= ap
        ly *= ap
    else:
        lx = 0.0
        ly = 0.0
    dx = fx - lx
    dy = fy - ly
    dz = fz
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= n
    dy /= n
    dz /= n
    o[0] = cam[0] + R00 * lx + R01 * ly
    o[1] = cam[1] + R10 * lx + R11 * ly
    o[2] = cam[2] + R20 * lx + R21 * ly
    d[0] = R00 * dx + R01 * dy + R02 * dz
    d[1] = R10 * dx + R11 * dy + R12 * dz
    d[2] = R20 * dx + R21 * dy + R22 * dz
    return 1.0


# ----------------------------------------------------------------- manifold next event estimation


@njit(cache=True)
def _refract(dx, dy, dz, nx, ny, nz, e):
    """Refract travel direction d at a surface with normal n facing against d, relative index
    e = n_i / n_t. Returns (ok, direction)."""
    cosi = -(dx * nx + dy * ny + dz * nz)
    sin2t = e * e * max(0.0, 1.0 - cosi * cosi)
    if sin2t >= 1.0:
        return False, 0.0, 0.0, 0.0
    cost = np.sqrt(1.0 - sin2t)
    rx = e * dx + (e * cosi - cost) * nx
    ry = e * dy + (e * cosi - cost) * ny
    rz = e * dz + (e * cosi - cost) * nz
    ln = np.sqrt(rx * rx + ry * ry + rz * rz)
    return True, rx / ln, ry / ln, rz / ln


@njit(cache=True)
def _dielectric_normal(G, mats, inst_w, ut, yx, yy, yz, gwx, gwy, gwz, hd, sp2):
    """Shading normal (world, on the side of gw) of dielectric instance inst_w at the surface point
    found by casting along -gw from just above (yx, yy, yz). Returns (ok, n, surface point)."""
    o = np.empty(3)
    d = np.empty(3)
    o[0], o[1], o[2] = yx + gwx * 2e-3, yy + gwy * 2e-3, yz + gwz * 2e-3
    d[0], d[1], d[2] = -gwx, -gwy, -gwz
    t, inst, tri, u, v = scene_hit(o, d, ut, 0.0, 4e-3, G)
    if inst != inst_w:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    qw, qx, qy, qz, tx, ty, tz = hit_data(G, inst, tri, u, v, ut, o, d, t, hd)
    mat = mats[G[18][inst]]
    shade_material(mat, hd[9], hd[10], hd[11], hd[12], hd[13], hd[14], hd[20], hd[21], hd[22], hd[15], hd[16], hd[17], hd[18], hd[19], sp2)
    nx, ny, nz = q_rot(qw, qx, qy, qz, sp2[S_N], sp2[S_N + 1], sp2[S_N + 2])
    if nx * gwx + ny * gwy + nz * gwz < 0.0:
        nx, ny, nz = -nx, -ny, -nz
    return True, nx, ny, nz, hd[0], hd[1], hd[2]


@njit(cache=True)
def _mnee_constraint(G, mats, inst_w, ut, y0, e1, e2, a, b, gw, s, eta_rel, x, hd, sp2):
    """Tangential mismatch between the refracted sun direction at y(a, b) and the direction from
    y to the receiver x. Returns (ok, c1, c2, y, refracted direction)."""
    yx = y0[0] + a * e1[0] + b * e2[0]
    yy = y0[1] + a * e1[1] + b * e2[1]
    yz = y0[2] + a * e1[2] + b * e2[2]
    ok, nx, ny, nz, px, py, pz = _dielectric_normal(G, mats, inst_w, ut, yx, yy, yz, gw[0], gw[1], gw[2], hd, sp2)
    if not ok:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ok, rx, ry, rz = _refract(-s[0], -s[1], -s[2], nx, ny, nz, eta_rel)
    if not ok:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dx, dy, dz = x[0] - px, x[1] - py, x[2] - pz
    ln = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= ln
    dy /= ln
    dz /= ln
    cx, cy, cz = dx - rx, dy - ry, dz - rz
    return True, cx * e1[0] + cy * e1[1] + cz * e1[2], cx * e2[0] + cy * e2[1] + cz * e2[2], px, py, pz, rx, ry, rz


@njit(cache=True)
def mnee_sun(G, mats, lr, ut, x, gxr, sp, wo_l, tvec, bvec, nsx, nsy, nsz, E, Eavg, settings, spectral, lam, collapsed, out):
    """Sun light reaching receiver x through one smooth dielectric interface (water with ripples):
    a Newton solve on the specular constraint finds the surface point y whose refracted sun ray
    passes through x (Hanika et al. 2015; directional light form). The irradiance at x is the sun
    irradiance through the interface times the beam compression dA_y / dA_x, from finite
    differences of the map from the surface to the receiver plane. ``out`` gets f * E (per
    channel, without the path throughput). Returns True when a contribution was found."""
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    s = np.empty(3)
    s[0], s[1], s[2] = lr[1], lr[2], lr[3]
    o = np.empty(3)
    o[0], o[1], o[2] = x[0], x[1], x[2]
    t, inst, tri, u, v = scene_hit(o, s, ut, EPS_RAY, INF, G)
    if inst < 0:
        return False
    mat = mats[G[18][inst]]
    if int(mat[M_TYPE]) != 4:
        return False
    hd = np.empty(24)
    sp2 = np.empty(S_NCOLS)
    hit_data(G, inst, tri, u, v, ut, o, s, t, hd)
    gw = np.empty(3)
    gw[0], gw[1], gw[2] = hd[3], hd[4], hd[5]
    if gw[0] * s[0] + gw[1] * s[1] + gw[2] * s[2] < 0.0:
        gw[0], gw[1], gw[2] = -gw[0], -gw[1], -gw[2]
    ior = mat[M_IOR]
    dispersive = spectral and mat[M_CAUCHY_B] > 0.0
    if dispersive:
        ior = cauchy_ior(ior, mat[M_CAUCHY_B], lam[0])
    eta_rel = 1.0 / ior  # light travels from air into the dielectric
    ok, r0x, r0y, r0z = _refract(-s[0], -s[1], -s[2], gw[0], gw[1], gw[2], eta_rel)
    if not ok:
        return False
    rg = r0x * gw[0] + r0y * gw[1] + r0z * gw[2]
    if rg > -1e-6:
        return False
    sig = ((x[0] - hd[0]) * gw[0] + (x[1] - hd[1]) * gw[1] + (x[2] - hd[2]) * gw[2]) / rg
    y0 = np.empty(3)
    y0[0], y0[1], y0[2] = x[0] - sig * r0x, x[1] - sig * r0y, x[2] - sig * r0z
    e1 = np.empty(3)
    e2 = np.empty(3)
    make_frame(gw[0], gw[1], gw[2], 1.0, 0.0, 0.0, e1, e2)
    a = 0.0
    b = 0.0
    h = 1e-5
    converged = False
    for it in range(24):
        ok, c1, c2, py_, py2, py3, rx, ry, rz = _mnee_constraint(G, mats, inst, ut, y0, e1, e2, a, b, gw, s, eta_rel, x, hd, sp2)
        if not ok:
            return False
        if c1 * c1 + c2 * c2 < 1e-14:
            converged = True
            break
        ok1, a1, a2, _, _, _, _, _, _ = _mnee_constraint(G, mats, inst, ut, y0, e1, e2, a + h, b, gw, s, eta_rel, x, hd, sp2)
        ok2, b1, b2, _, _, _, _, _, _ = _mnee_constraint(G, mats, inst, ut, y0, e1, e2, a, b + h, gw, s, eta_rel, x, hd, sp2)
        if not (ok1 and ok2):
            return False
        J00 = (a1 - c1) / h
        J10 = (a2 - c2) / h
        J01 = (b1 - c1) / h
        J11 = (b2 - c2) / h
        det = J00 * J11 - J01 * J10
        if abs(det) < 1e-12:
            return False
        da = (J11 * c1 - J01 * c2) / det
        db = (-J10 * c1 + J00 * c2) / det
        st = np.sqrt(da * da + db * db)
        if st > 5e-3:
            da *= 5e-3 / st
            db *= 5e-3 / st
        a -= da
        b -= db
    if not converged:
        return False
    yp = np.empty(3)
    yp[0], yp[1], yp[2] = py_, py2, py3
    # receiver side: x to y unoccluded, and y to the sun unoccluded
    dxy = np.empty(3)
    dxy[0], dxy[1], dxy[2] = yp[0] - x[0], yp[1] - x[1], yp[2] - x[2]
    dist = np.sqrt(dxy[0] ** 2 + dxy[1] ** 2 + dxy[2] ** 2)
    dxy /= dist
    if dxy[0] * gxr[0] + dxy[1] * gxr[1] + dxy[2] * gxr[2] <= 0.0:
        return False
    if scene_occluded(o, dxy, ut, EPS_RAY, dist * (1.0 - 1e-3), G):
        return False
    oy = np.empty(3)
    oy[0], oy[1], oy[2] = yp[0] + gw[0] * 1e-5, yp[1] + gw[1] * 1e-5, yp[2] + gw[2] * 1e-5
    if scene_occluded(oy, s, ut, EPS_RAY, INF, G):
        return False
    # Fresnel transmittance of the sun ray at y
    okn, nx, ny, nz, _, _, _ = _dielectric_normal(G, mats, inst, ut, yp[0], yp[1], yp[2], gw[0], gw[1], gw[2], hd, sp2)
    if not okn:
        return False
    T = 1.0 - fresnel_dielectric(s[0] * nx + s[1] * ny + s[2] * nz, ior)
    # beam compression: map (a, b) -> receiver plane through x with normal gxr
    X = np.empty((3, 3))
    for k in range(3):
        aa = a + (h if k == 1 else 0.0)
        bb = b + (h if k == 2 else 0.0)
        ok, c1, c2, qx_, qy_, qz_, rx, ry, rz = _mnee_constraint(G, mats, inst, ut, y0, e1, e2, aa, bb, gw, s, eta_rel, x, hd, sp2)
        if not ok:
            return False
        den = rx * gxr[0] + ry * gxr[1] + rz * gxr[2]
        if abs(den) < 1e-9:
            return False
        tau = ((x[0] - qx_) * gxr[0] + (x[1] - qy_) * gxr[1] + (x[2] - qz_) * gxr[2]) / den
        X[k, 0], X[k, 1], X[k, 2] = qx_ + tau * rx, qy_ + tau * ry, qz_ + tau * rz
    ux, uy, uz = X[1, 0] - X[0, 0], X[1, 1] - X[0, 1], X[1, 2] - X[0, 2]
    vx, vy, vz = X[2, 0] - X[0, 0], X[2, 1] - X[0, 1], X[2, 2] - X[0, 2]
    cx_ = uy * vz - uz * vy
    cy_ = uz * vx - ux * vz
    cz_ = ux * vy - uy * vx
    dAx = np.sqrt(cx_ * cx_ + cy_ * cy_ + cz_ * cz_) / (h * h)
    if dAx < 1e-3:
        dAx = 1e-3  # focus singularity: clamp the compression at 1000
    cos_g = abs(gw[0] * s[0] + gw[1] * s[1] + gw[2] * s[2])
    omega = 2.0 * PI * (1.0 - lr[13])
    wi = np.empty(3)
    to_local(dxy[0], dxy[1], dxy[2], tvec, bvec, nsx, nsy, nsz, wi)
    if wi[2] <= 0.0:
        return False
    f = np.empty(3)
    bsdf_eval(sp, wo_l, wi, E, Eavg, f)
    for k in range(3):
        Ls = lr[10 + k]
        if spectral:
            Ls = upsample(settings, lr[10], lr[11], lr[12], lam[k], True)
        sa = mat[M_ABSORB + k]
        if spectral:
            sa = upsample_positive(settings, mat[M_ABSORB], mat[M_ABSORB + 1], mat[M_ABSORB + 2], lam[k])
        out[k] = f[k] * Ls * omega * cos_g * T / dAx * np.exp(-sa * dist)
    if dispersive:
        # hero wavelength: the refraction is exact for lam[0] only
        if not collapsed:
            out[0] *= 3.0
        out[1] = 0.0
        out[2] = 0.0
    return True


# ----------------------------------------------------------------- path trace


@njit(cache=True)
def trace_sample(px, py, sidx, gseed, pseed, mask, rng_state, cam, G, OV, mats, lights, nL, sky_index, sky_p, M, env_img, env_marg, env_cond, background, medium, E, Eavg, settings, acc, counters):
    """Trace one camera sample for pixel (px, py) and accumulate into acc (14 floats):
    beauty 0:3, albedo 3:6, normal 6:9, depth 9, objid 10, overlay 11:15 (so acc has 15).
    counters[0] += rays."""
    max_diff = int(settings[0])
    max_spec = int(settings[1])
    rr_start = int(settings[2])
    o = np.empty(3)
    d = np.empty(3)
    u0 = get_u(0, sidx, gseed, pseed, px, py, mask, rng_state)
    u1 = get_u(1, sidx, gseed, pseed, px, py, mask, rng_state)
    u2 = get_u(2, sidx, gseed, pseed, px, py, mask, rng_state)
    u3 = get_u(3, sidx, gseed, pseed, px, py, mask, rng_state)
    u4 = get_u(4, sidx, gseed, pseed, px, py, mask, rng_state)
    ut = get_u(5, sidx, gseed, pseed, px, py, mask, rng_state)
    wcam = camera_ray(cam, px, py, u0, u1, u2, u3, u4, o, d)
    if wcam <= 0.0:
        # blocked inside the traced lens: the sample is black
        return
    thr = np.ones(3) * wcam
    L = np.zeros(3)
    spectral = settings[S_SPECTRAL] > 0.5
    mnee_on = settings[S_MNEE] > 0.5
    lam = np.empty(3)
    if spectral:
        ulam = rng_u01(rng_state)
        for k in range(3):
            fk = ulam + k / 3.0
            lam[k] = 380.0 + 350.0 * (fk - np.floor(fk))
    else:
        lam[0] = 550.0
        lam[1] = 550.0
        lam[2] = 550.0
    collapsed = False
    inside = False  # below a dielectric surface (water)
    sig_a = np.zeros(3)
    prev_delta = False
    had_diffuse = False
    n_refr = 0
    n_refl_d = 0
    mn = np.empty(3)
    xr = np.empty(3)
    gr = np.empty(3)
    hd = np.empty(24)
    sp = np.empty(S_NCOLS)
    tvec = np.empty(3)
    bvec = np.empty(3)
    wo_l = np.empty(3)
    wi_l = np.empty(3)
    f = np.empty(3)
    ls = np.empty(7)
    esc = np.empty(3)
    exit_p = np.empty(3)
    exit_n = np.empty(3)
    pl = np.empty(3)
    nl = np.empty(3)
    n_diff = 0
    n_spec = 0
    pdf_prev = 0.0
    nprev_x, nprev_y, nprev_z = 0.0, 0.0, 1.0
    dim = 6
    med_on = medium[0] > 0.5
    sigma_t = medium[1]
    first = True
    for bounce in range(max_spec + 1):
        t, inst, tri, u, v = scene_hit(o, d, ut, EPS_RAY, INF, G)
        counters[0] += 1
        if first:
            t_main = t if inst >= 0 else INF
            composite_overlays(o, d, ut, t_main, OV, mats, acc[11:15])
            counters[0] += 1
        # medium transmittance and single scattering (camera rays)
        if med_on:
            tmax_seg = t if inst >= 0 else 1e3
            ta, tb = slab_interval(o[2], d[2], tmax_seg, medium[4])
            if tb > ta:
                if first:
                    um = get_u(dim, sidx, gseed, pseed, px, py, mask, rng_state)
                    dim += 1
                    ts = ta - np.log(max(um, 1e-12)) / sigma_t
                    if ts < tb:
                        # scatter event: single scattering towards a light, then stop
                        if nL > 0:
                            ul = get_u(dim, sidx, gseed, pseed, px, py, mask, rng_state)
                            ua = get_u(dim + 1, sidx, gseed, pseed, px, py, mask, rng_state)
                            ub = get_u(dim + 2, sidx, gseed, pseed, px, py, mask, rng_state)
                            dim += 3
                            l = min(int(ul * nL), nL - 1)
                            sx, sy, sz = o[0] + ts * d[0], o[1] + ts * d[1], o[2] + ts * d[2]
                            pdf_l = sample_light(lights, l, sx, sy, sz, 0.0, 0.0, 1.0, ua, ub, sky_p, M, env_img, env_marg, env_cond, ls)
                            if pdf_l > 0.0:
                                spt = np.empty(3)
                                spt[0], spt[1], spt[2] = sx, sy, sz
                                ld = np.empty(3)
                                ld[0], ld[1], ld[2] = ls[0], ls[1], ls[2]
                                counters[0] += 1
                                if not scene_occluded(spt, ld, ut, EPS_RAY, ls[3] * (1.0 - 1e-4), G):
                                    la, lb = slab_interval(sz, ld[2], min(ls[3], 1e3), medium[4])
                                    tr = np.exp(-sigma_t * max(lb - la, 0.0))
                                    cos_th = -(d[0] * ld[0] + d[1] * ld[1] + d[2] * ld[2])
                                    ph = hg_phase(cos_th, medium[3]) * medium[2]
                                    w = ph * tr / (pdf_l / nL)
                                    if spectral:
                                        l0, l1, l2 = ls[4], ls[5], ls[6]
                                        for k in range(3):
                                            ls[4 + k] = upsample(settings, l0, l1, l2, lam[k], True)
                                    for k in range(3):
                                        L[k] += thr[k] * ls[4 + k] * w
                        # the surface contribution is handled by the no-scatter branch (unbiased)
                        break
                    # no scatter before the surface: weight 1 (delta tracking)
                else:
                    tr = np.exp(-sigma_t * (tb - ta))
                    for k in range(3):
                        thr[k] *= tr
        if inst < 0:
            skip_sun = mnee_on and had_diffuse and n_refr == 1 and n_refl_d == 0
            escaped_radiance(d[0], d[1], d[2], lights, nL, sky_index, sky_p, M, env_img, env_marg, env_cond, background, pdf_prev, nprev_x, nprev_y, nprev_z, (not first) and (not prev_delta), esc, skip_sun)
            if spectral:
                e0, e1_, e2_ = esc[0], esc[1], esc[2]
                for k in range(3):
                    esc[k] = upsample(settings, e0, e1_, e2_, lam[k], True)
            for k in range(3):
                L[k] += thr[k] * esc[k]
            break
        qw, qx, qy, qz, tx, ty, tz = hit_data(G, inst, tri, u, v, ut, o, d, t, hd)
        mat = mats[G[18][inst]]
        mtype = int(mat[M_TYPE])
        if inside:
            for k in range(3):
                thr[k] *= np.exp(-sig_a[k] * t)
        if mtype == 4:
            # smooth dielectric: Fresnel choice between reflection and refraction
            shade_material(mat, hd[9], hd[10], hd[11], hd[12], hd[13], hd[14], hd[20], hd[21], hd[22], hd[15], hd[16], hd[17], hd[18], hd[19], sp)
            nx, ny, nz = q_rot(qw, qx, qy, qz, sp[S_N], sp[S_N + 1], sp[S_N + 2])
            gx, gy, gz = hd[3], hd[4], hd[5]
            entering = (d[0] * gx + d[1] * gy + d[2] * gz) < 0.0
            if not entering:
                gx, gy, gz = -gx, -gy, -gz
            if nx * gx + ny * gy + nz * gz < 0.0:
                nx, ny, nz = -nx, -ny, -nz
            if -(d[0] * nx + d[1] * ny + d[2] * nz) <= 0.0:
                nx, ny, nz = gx, gy, gz
            ior = mat[M_IOR]
            if spectral and mat[M_CAUCHY_B] > 0.0:
                ior = cauchy_ior(ior, mat[M_CAUCHY_B], lam[0])
                if not collapsed:
                    thr[0] *= 3.0
                    thr[1] = 0.0
                    thr[2] = 0.0
                    collapsed = True
            eta_t = ior if entering else 1.0 / ior  # n_t / n_i
            cosi = -(d[0] * nx + d[1] * ny + d[2] * nz)
            Fr = fresnel_dielectric(cosi, eta_t)
            if first:
                acc[3] += 1.0
                acc[4] += 1.0
                acc[5] += 1.0
                acc[6] += nx
                acc[7] += ny
                acc[8] += nz
                acc[9] += t
                acc[10] = G[19][inst]
            px_, py_, pz_ = hd[0], hd[1], hd[2]
            if rng_u01(rng_state) < Fr:
                d[0] += 2.0 * cosi * nx
                d[1] += 2.0 * cosi * ny
                d[2] += 2.0 * cosi * nz
                ln = np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                d[0] /= ln
                d[1] /= ln
                d[2] /= ln
                o[0], o[1], o[2] = px_ + gx * 1e-5, py_ + gy * 1e-5, pz_ + gz * 1e-5
                n_refl_d += 1
            else:
                ok_r, rx, ry, rz = _refract(d[0], d[1], d[2], nx, ny, nz, 1.0 / eta_t)
                if not ok_r:
                    break
                d[0], d[1], d[2] = rx, ry, rz
                # radiance scales with (n_i / n_t)^2 across the interface
                for k in range(3):
                    thr[k] /= eta_t * eta_t
                o[0], o[1], o[2] = px_ - gx * 1e-5, py_ - gy * 1e-5, pz_ - gz * 1e-5
                inside = entering
                if inside:
                    for k in range(3):
                        sig_a[k] = mat[M_ABSORB + k]
                    if spectral:
                        for k in range(3):
                            sig_a[k] = upsample_positive(settings, mat[M_ABSORB], mat[M_ABSORB + 1], mat[M_ABSORB + 2], lam[k])
                n_refr += 1
            prev_delta = True
            pdf_prev = 0.0
            n_spec += 1
            first = False
            if (n_diff + n_spec) >= max_spec:
                break
            continue
        if mtype == 3:
            # emitter geometry
            li = G[25][inst]
            cos_l = -(d[0] * hd[3] + d[1] * hd[4] + d[2] * hd[5])
            if cos_l > 0.0:
                w = 1.0
                if not first and li >= 0:
                    lr = lights[li]
                    pl_ = (1.0 / nL) * t * t / (lr[13] * cos_l)
                    w = pdf_prev * pdf_prev / (pdf_prev * pdf_prev + pl_ * pl_)
                for k in range(3):
                    em = mat[M_EMIT + k] if not spectral else upsample(settings, mat[M_EMIT], mat[M_EMIT + 1], mat[M_EMIT + 2], lam[k], True)
                    L[k] += thr[k] * em * w
            if first:
                for k in range(3):
                    acc[3 + k] += mat[M_EMIT + k]
                acc[9] += t
                acc[10] = G[19][inst]
            break
        if mtype == 1 or mtype == 2:
            for k in range(3):
                em = mat[M_EMIT + k] if not spectral else upsample(settings, mat[M_EMIT], mat[M_EMIT + 1], mat[M_EMIT + 2], lam[k], True)
                L[k] += thr[k] * em
            if first:
                acc[9] += t
                acc[10] = G[19][inst]
            break
        shade_material(mat, hd[9], hd[10], hd[11], hd[12], hd[13], hd[14], hd[20], hd[21], hd[22], hd[15], hd[16], hd[17], hd[18], hd[19], sp)
        # world shading normal (perturbed)
        nx, ny, nz = q_rot(qw, qx, qy, qz, sp[S_N], sp[S_N + 1], sp[S_N + 2])
        gx, gy, gz = hd[3], hd[4], hd[5]
        wox, woy, woz = -d[0], -d[1], -d[2]
        if wox * gx + woy * gy + woz * gz < 0.0:
            # back face: flip both normals (two sided shading)
            nx, ny, nz = -nx, -ny, -nz
            gx, gy, gz = -gx, -gy, -gz
        if nx * wox + ny * woy + nz * woz < 0.0:
            # perturbed normal facing away: fall back to the geometric normal
            nx, ny, nz = gx, gy, gz
        if first:
            for k in range(3):
                acc[3 + k] += sp[S_BASE + k]
            acc[6] += nx
            acc[7] += ny
            acc[8] += nz
            acc[9] += t
            acc[10] = G[19][inst]
        if spectral:
            b0, b1, b2 = sp[S_BASE], sp[S_BASE + 1], sp[S_BASE + 2]
            e0, e1_, e2_ = sp[S_EMIT], sp[S_EMIT + 1], sp[S_EMIT + 2]
            m0, m1, m2 = sp[S_SSS_MFP], sp[S_SSS_MFP + 1], sp[S_SSS_MFP + 2]
            for k in range(3):
                sp[S_BASE + k] = upsample(settings, b0, b1, b2, lam[k], False)
                sp[S_EMIT + k] = upsample(settings, e0, e1_, e2_, lam[k], True)
                sp[S_SSS_MFP + k] = upsample_positive(settings, m0, m1, m2, lam[k])
        had_diffuse = True
        n_refr = 0
        n_refl_d = 0
        prev_delta = False
        for k in range(3):
            L[k] += thr[k] * sp[S_EMIT + k]
        ppx, ppy, ppz = hd[0], hd[1], hd[2]
        # subsurface random walk
        if sp[S_SSS_W] > 0.0:
            usss = rng_u01(rng_state)
            fo = fresnel_dielectric(max(nx * wox + ny * woy + nz * woz, 0.0), 1.4)
            if usss < sp[S_SSS_W] * (1.0 - fo):
                pl[0], pl[1], pl[2] = hd[9], hd[10], hd[11]
                nl[0], nl[1], nl[2] = hd[20], hd[21], hd[22]
                if hd[20] * hd[12] + hd[21] * hd[13] + hd[22] * hd[14] < 0.0:
                    pass
                ok = random_walk_sss(G, inst, qw, qx, qy, qz, tx, ty, tz, pl, nl, sp, rng_state, thr, exit_p, exit_n)
                if not ok:
                    break
                ppx, ppy, ppz = exit_p[0], exit_p[1], exit_p[2]
                nx, ny, nz = exit_n[0], exit_n[1], exit_n[2]
                gx, gy, gz = nx, ny, nz
                # exit lobe: white Lambert
                sp[S_BASE] = 1.0
                sp[S_BASE + 1] = 1.0
                sp[S_BASE + 2] = 1.0
                sp[S_SSS_W] = 0.0
                sp[S_COAT] = 0.0
                sp[S_METAL] = 0.0
                sp[S_SPEC] = 0.0
                sp[S_ROUGH] = 1.0
                sp[S_AX] = 1.0
                sp[S_AY] = 1.0
                sp[S_FLAKE_ON] = 0.0
                sp[S_DIFFUSE_MODEL] = 0.0
                wox, woy, woz = nx, ny, nz
        # anisotropy direction in world
        adx, ady, adz = q_rot(qw, qx, qy, qz, mat[M_ANISO_DIR], mat[M_ANISO_DIR + 1], mat[M_ANISO_DIR + 2])
        make_frame(nx, ny, nz, adx, ady, adz, tvec, bvec)
        to_local(wox, woy, woz, tvec, bvec, nx, ny, nz, wo_l)
        if wo_l[2] <= 0.0:
            wo_l[2] = 1e-4
        # offset origin along the geometric normal
        ox_ = ppx + gx * 1e-5
        oy_ = ppy + gy * 1e-5
        oz_ = ppz + gz * 1e-5
        # next event estimation
        if nL > 0:
            ul = get_u(dim, sidx, gseed, pseed, px, py, mask, rng_state)
            ua = get_u(dim + 1, sidx, gseed, pseed, px, py, mask, rng_state)
            ub = get_u(dim + 2, sidx, gseed, pseed, px, py, mask, rng_state)
            dim += 3
            l = min(int(ul * nL), nL - 1)
            pdf_l = sample_light(lights, l, ox_, oy_, oz_, nx, ny, nz, ua, ub, sky_p, M, env_img, env_marg, env_cond, ls)
            if pdf_l > 0.0 and spectral:
                l0, l1, l2 = ls[4], ls[5], ls[6]
                for k in range(3):
                    ls[4 + k] = upsample(settings, l0, l1, l2, lam[k], True)
            if pdf_l > 0.0:
                cos_i = ls[0] * nx + ls[1] * ny + ls[2] * nz
                if cos_i > 0.0 and (ls[0] * gx + ls[1] * gy + ls[2] * gz) > 0.0:
                    to_local(ls[0], ls[1], ls[2], tvec, bvec, nx, ny, nz, wi_l)
                    pdf_b = bsdf_eval(sp, wo_l, wi_l, E, Eavg, f)
                    if f[0] + f[1] + f[2] > 0.0:
                        so = np.empty(3)
                        so[0], so[1], so[2] = ox_, oy_, oz_
                        sd = np.empty(3)
                        sd[0], sd[1], sd[2] = ls[0], ls[1], ls[2]
                        counters[0] += 1
                        if not scene_occluded(so, sd, ut, EPS_RAY, ls[3] * (1.0 - 1e-4), G):
                            pl_ = pdf_l / nL
                            w = pl_ * pl_ / (pl_ * pl_ + pdf_b * pdf_b)
                            tr = 1.0
                            if med_on:
                                la, lb = slab_interval(oz_, sd[2], min(ls[3], 1e3), medium[4])
                                tr = np.exp(-sigma_t * max(lb - la, 0.0))
                            for k in range(3):
                                L[k] += thr[k] * f[k] * ls[4 + k] * cos_i * w * tr / pl_
        # manifold next event estimation: the sun through a dielectric surface (caustics)
        if mnee_on and nL > 0:
            xr[0], xr[1], xr[2] = ox_, oy_, oz_
            gr[0], gr[1], gr[2] = gx, gy, gz
            for l2 in range(nL):
                if int(lights[l2][0]) != 1:
                    continue
                counters[0] += 8
                if mnee_sun(G, mats, lights[l2], ut, xr, gr, sp, wo_l, tvec, bvec, nx, ny, nz, E, Eavg, settings, spectral, lam, collapsed, mn):
                    for k in range(3):
                        L[k] += thr[k] * mn[k]
        # BSDF sampling
        ua = get_u(dim, sidx, gseed, pseed, px, py, mask, rng_state)
        ub = get_u(dim + 1, sidx, gseed, pseed, px, py, mask, rng_state)
        uc = get_u(dim + 2, sidx, gseed, pseed, px, py, mask, rng_state)
        dim += 3
        pdf_b, is_spec = bsdf_sample(sp, wo_l, ua, ub, uc, E, Eavg, wi_l, f)
        if pdf_b <= 0.0:
            break
        wx, wy, wz = to_world(wi_l, tvec, bvec, nx, ny, nz)
        if wx * gx + wy * gy + wz * gz <= 0.0:
            break
        cos_i = wi_l[2]
        for k in range(3):
            thr[k] *= f[k] * cos_i / pdf_b
        if is_spec:
            n_spec += 1
        else:
            n_diff += 1
        if n_diff >= max_diff or (n_diff + n_spec) >= max_spec:
            break
        if bounce >= rr_start:
            q = min(max(thr[0], max(thr[1], thr[2])), 0.95)
            ur = get_u(dim, sidx, gseed, pseed, px, py, mask, rng_state)
            dim += 1
            if ur > q:
                break
            for k in range(3):
                thr[k] /= q
        o[0], o[1], o[2] = ox_, oy_, oz_
        d[0], d[1], d[2] = wx, wy, wz
        pdf_prev = pdf_b
        nprev_x, nprev_y, nprev_z = nx, ny, nz
        first = False
    if spectral:
        Lrgb = np.empty(3)
        to_rgb(settings, L, lam, Lrgb)
        for k in range(3):
            L[k] = Lrgb[k]
    for k in range(3):
        if not np.isfinite(L[k]):
            L[k] = 0.0
        acc[k] += min(L[k], settings[3])


@njit(cache=True, parallel=True)
def render_tiles(tiles, tile_size, W, H, s0, s1, gseed, mask, cam, G, OV, mats, lights, nL, sky_index, sky_p, M, env_img, env_marg, env_cond, background, medium, E, Eavg, settings, acc_img, counters):
    """Render samples [s0, s1) for every tile in ``tiles`` (n, 2) of (x0, y0)."""
    ntiles = tiles.shape[0]
    for ti in prange(ntiles):
        x0 = tiles[ti, 0]
        y0 = tiles[ti, 1]
        rng_state = np.empty(1, dtype=np.uint64)
        acc = np.zeros(15)
        for yy in range(y0, min(y0 + tile_size, H)):
            for xx in range(x0, min(x0 + tile_size, W)):
                pseed = hash_u32(hash_u32(np.uint64(xx), np.uint64(yy)), np.uint64(gseed))
                for s in range(s0, s1):
                    for k in range(15):
                        acc[k] = 0.0
                    rng_state[0] = (np.uint64(0x9E3779B97F4A7C15) ^ (pseed * np.uint64(0x2545F4914F6CDD1D)) ^ (np.uint64(s) * np.uint64(0xBF58476D1CE4E5B9))) | np.uint64(1)
                    trace_sample(xx, yy, s, gseed, pseed, mask, rng_state, cam, G, OV, mats, lights, nL, sky_index, sky_p, M, env_img, env_marg, env_cond, background, medium, E, Eavg, settings, acc, counters[ti : ti + 1])
                    for k in range(10):
                        acc_img[yy, xx, k] += acc[k]
                    if s == 0:
                        acc_img[yy, xx, 10] = acc[10]
                    for k in range(11, 15):
                        acc_img[yy, xx, k] += acc[k]
