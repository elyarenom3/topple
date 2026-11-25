"""Video output (Section 10.8) and camera paths (Section 10.5).

Frames are written as PNG (and optionally EXR) sequences; ``encode`` produces an H.265 or H.264
mp4 through imageio's ffmpeg plugin when it is available and otherwise leaves the PNG
sequence in place and reports that. Camera moves follow minimum jerk profiles between
keyframes (Flash and Hogan 1985): the path through the keyframes is a Catmull Rom spline and the
progress along each segment follows the quintic 10 s^3 - 15 s^4 + 6 s^5, so acceleration is
continuous and motion blur stays smooth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .color import tonemap
from .io import write_exr, write_png


def min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _catmull_rom(P: np.ndarray, seg: int, t: float) -> np.ndarray:
    n = len(P)
    p0 = P[max(seg - 1, 0)]
    p1 = P[seg]
    p2 = P[min(seg + 1, n - 1)]
    p3 = P[min(seg + 2, n - 1)]
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def camera_path_min_jerk(keyframes: list, times: list, n_frames: int, fps: float = 24.0):
    """Interpolate camera keyframes (dicts with position, look_at and optional focus_distance,
    focal_mm, f_number) at frame times. Returns a list of dicts, one per frame."""
    times = np.asarray(times, dtype=np.float64)
    pos = np.array([np.asarray(k["position"], dtype=np.float64) for k in keyframes])
    look = np.array([np.asarray(k["look_at"], dtype=np.float64) for k in keyframes])
    scal = {}
    for key in ("focus_distance", "focal_mm", "f_number"):
        if all(key in k and k[key] is not None for k in keyframes):
            scal[key] = np.array([float(k[key]) for k in keyframes])
    out = []
    for i in range(n_frames):
        t = i / fps
        seg = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
        s = (t - times[seg]) / max(times[seg + 1] - times[seg], 1e-9)
        u = float(min_jerk(np.array([s]))[0])
        if len(keyframes) == 2:
            p = pos[0] + (pos[1] - pos[0]) * u
            l = look[0] + (look[1] - look[0]) * u
        else:
            p = _catmull_rom(pos, seg, u)
            l = _catmull_rom(look, seg, u)
        fr = {"position": p, "look_at": l}
        for key, arr in scal.items():
            fr[key] = float(arr[seg] + (arr[seg + 1] - arr[seg]) * u)
        out.append(fr)
    return out


def orbit_keyframes(center, radius: float, height: float, n: int = 8, start_deg: float = 0.0, sweep_deg: float = 360.0):
    """Keyframes on a circular dolly of ``radius`` around ``center`` at ``height``."""
    c = np.asarray(center, dtype=np.float64)
    out = []
    for k in range(n + 1):
        a = np.radians(start_deg + sweep_deg * k / n)
        out.append({"position": c + np.array([radius * np.cos(a), radius * np.sin(a), height - c[2]]), "look_at": c})
    return out


class FrameWriter:
    """Writes numbered PNG frames (tone mapped) and optional EXR passes to a directory."""

    def __init__(self, directory, exposure: float = 1.0, exr: bool = False, src_white=None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.exposure = exposure
        self.exr = exr
        self.src_white = src_white
        self.count = 0

    def write(self, passes: dict, key: str = "composite") -> Path:
        img = passes.get(key, passes["beauty"])
        png = self.dir / f"frame_{self.count:05d}.png"
        write_png(png, tonemap(img, self.exposure, self.src_white))
        if self.exr:
            write_exr(self.dir / f"frame_{self.count:05d}.exr", {k: v for k, v in passes.items() if isinstance(v, np.ndarray) and k not in ("object_id",)})
        self.count += 1
        return png


def encode(frames_dir, out_path, fps: float = 24.0, codec: str = "libx265", quality: int = 8) -> str:
    """Encode frame_*.png in ``frames_dir`` into ``out_path`` (mp4). Returns a status string."""
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        return "no frames"
    try:
        import imageio.v2 as iio

        try:
            writer = iio.get_writer(str(out_path), fps=fps, codec=codec, quality=quality, pixelformat="yuv420p10le" if codec == "libx265" else "yuv420p")
        except Exception:
            writer = iio.get_writer(str(out_path), fps=fps, codec="libx264", quality=quality)
        for f in frames:
            writer.append_data(iio.imread(str(f)))
        writer.close()
        return f"encoded {len(frames)} frames to {out_path}"
    except Exception as exc:
        return f"ffmpeg unavailable ({exc.__class__.__name__}); {len(frames)} PNG frames kept in {frames_dir}"
