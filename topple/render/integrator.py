"""Render driver (Section 10.1, 10.9): tiles, checkpoints, passes, denoising and statistics."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .kernels import render_tiles
from .materials import kulla_conty_tables
from .sampling import blue_noise_mask
from .scene import Camera, Scene, ScenePack
from .sky import _M

PASSES = ("beauty", "albedo", "normal", "depth", "object_id", "overlay")


class RenderStats:
    def __init__(self):
        self.rays = 0
        self.seconds = 0.0
        self.samples = 0

    @property
    def rays_per_second(self) -> float:
        return self.rays / max(self.seconds, 1e-9)

    def __repr__(self):
        return f"RenderStats(rays={self.rays}, seconds={self.seconds:.2f}, rays_per_second={self.rays_per_second:.3g})"


def render(
    scene: Scene | ScenePack,
    camera: Camera,
    spp: int = 16,
    max_depth_diffuse: int = 12,
    max_depth_specular: int = 64,
    rr_start_depth: int = 3,
    seed: int = 0,
    tile: int = 64,
    checkpoint: str | os.PathLike | None = None,
    denoise: bool = False,
    clamp: float = 1e6,
    samples_per_pass: int | None = None,
    stats: RenderStats | None = None,
    progress: bool = False,
    spectral: bool = False,
    mnee: bool = False,
) -> dict:
    """Render ``scene`` through ``camera``.

    ``spectral`` traces three stratified wavelengths per sample (hero wavelength, Cauchy
    dispersion in dielectrics); ``mnee`` adds manifold next event estimation of the sun through
    smooth dielectric surfaces (caustics under water).

    Returns a dict with passes: beauty (H, W, 3) linear radiance, albedo, normal, depth (H, W),
    object_id (H, W), overlay (H, W, 4) premultiplied RGBA, composite (H, W, 3) = beauty with the
    overlay alpha composited over it, and optionally beauty_denoised.
    """
    pack = scene if isinstance(scene, ScenePack) else scene.build()
    W, H = camera.width, camera.height
    cam = camera.to_array()
    cam[21] = W
    cam[22] = H
    E, Eavg = kulla_conty_tables()
    mask = blue_noise_mask(64)
    from .spectral import pack_settings

    settings = pack_settings(np.array([max_depth_diffuse, max_depth_specular, rr_start_depth, clamp], dtype=np.float64), spectral, mnee)
    tiles = np.array([(x, y) for y in range(0, H, tile) for x in range(0, W, tile)], dtype=np.int64)
    acc = np.zeros((H, W, 15))
    done = np.zeros(len(tiles), dtype=bool)
    counters = np.zeros(len(tiles), dtype=np.int64)
    s_done = 0
    if checkpoint is not None and Path(checkpoint).exists():
        ck = np.load(checkpoint)
        if ck["acc"].shape == acc.shape and int(ck["spp"]) == spp:
            acc = ck["acc"]
            done = ck["done"]
            counters = ck["counters"]
    st = stats if stats is not None else RenderStats()
    G = pack.main.tuple()
    OV = pack.overlay.tuple()
    nL = pack.lights.shape[0]
    t0 = time.time()
    rays0 = counters.sum()
    # samples are rendered in passes so that a checkpoint is written between them
    spp_chunk = spp if samples_per_pass is None else max(1, samples_per_pass)
    todo = np.where(~done)[0]
    if len(todo) > 0:
        for s0 in range(0, spp, spp_chunk):
            s1 = min(spp, s0 + spp_chunk)
            sub = np.zeros(len(todo), dtype=np.int64)
            render_tiles(
                tiles[todo], tile, W, H, s0, s1, np.uint64(seed), mask, cam, G, OV, pack.mats, pack.lights, nL, pack.sky_index,
                pack.sky_p, _M, pack.env_img, pack.env_marg, pack.env_cond, pack.background, pack.medium, E, Eavg, settings, acc, sub,
            )
            counters[todo] += sub
            if checkpoint is not None:
                np.savez(checkpoint, acc=acc, done=done, counters=counters, spp=spp, s_done=s1)
            if progress:
                print(f"  samples {s1}/{spp}, {time.time() - t0:.1f} s")
        done[:] = True
        if checkpoint is not None:
            np.savez(checkpoint, acc=acc, done=done, counters=counters, spp=spp, s_done=spp)
    st.seconds += time.time() - t0
    st.rays += int(counters.sum() - rays0)
    st.samples += spp * W * H
    inv = 1.0 / spp
    out = {
        "beauty": acc[:, :, 0:3] * inv,
        "albedo": acc[:, :, 3:6] * inv,
        "normal": acc[:, :, 6:9] * inv,
        "depth": acc[:, :, 9] * inv,
        "object_id": acc[:, :, 10].astype(np.int64),
        "overlay": acc[:, :, 11:15] * inv,
        "stats": st,
    }
    ov = out["overlay"]
    out["composite"] = out["beauty"] * (1.0 - ov[:, :, 3:4]) + ov[:, :, 0:3]
    if denoise:
        from .denoise import denoise as _dn

        out["beauty_denoised"] = _dn(out["beauty"], out["albedo"], out["normal"])
        out["composite_denoised"] = out["beauty_denoised"] * (1.0 - ov[:, :, 3:4]) + ov[:, :, 0:3]
    return out


def estimate_irradiance(scene: Scene | ScenePack, point, normal, n: int = 2048, seed: int = 0) -> float:
    """Monte Carlo estimate of the irradiance (W m^-2, luminance weighted) at a point with
    the given normal from the scene lights, with shadow rays. Used for the grey card exposure."""
    from .kernels import sample_light, scene_occluded
    from .materials import luminance3
    from .sampling import rng_u01

    pack = scene if isinstance(scene, ScenePack) else scene.build()
    G = pack.main.tuple()
    nL = pack.lights.shape[0]
    if nL == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    p = np.asarray(point, dtype=np.float64)
    nrm = np.asarray(normal, dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    ls = np.zeros(7)
    total = 0.0
    for k in range(n):
        l = k % nL
        o = p + 1e-4 * nrm
        pdf = sample_light(pack.lights, l, o[0], o[1], o[2], nrm[0], nrm[1], nrm[2], rng.random(), rng.random(), pack.sky_p, _M, pack.env_img, pack.env_marg, pack.env_cond, ls)
        if pdf <= 0:
            continue
        cos = ls[0] * nrm[0] + ls[1] * nrm[1] + ls[2] * nrm[2]
        if cos <= 0:
            continue
        if scene_occluded(o, ls[0:3].copy(), 0.0, 1e-5, ls[3] * (1.0 - 1e-4), G):
            continue
        total += luminance3(ls[4], ls[5], ls[6]) * cos / pdf
    return total / n * nL  # each sample estimates one light's irradiance; lights are summed


def key_light_exposure(scene: Scene | ScenePack, point, normal=(0, 0, 1), n: int = 2048) -> float:
    """Exposure that maps an 18 percent grey card at ``point`` facing ``normal`` to 0.18."""
    from .color import exposure_from_grey_card

    return exposure_from_grey_card(estimate_irradiance(scene, point, normal, n))
