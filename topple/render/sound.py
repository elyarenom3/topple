"""Modal impact sound for the collapse video (optional item of Section 10.6).

Each stone gets vibration modes from a mass spring network on its (decimated) surface: lumped
vertex masses from the vertex areas and the stone mass, and an axial spring on every edge. The
network gives the mode shapes and the spacing of the frequencies for the stone's own shape; its
stiffness is scaled so that the lowest elastic mode matches the fundamental spheroidal mode of a
homogeneous sphere of the same volume, omega R / c_s = 2.64 (Lamb 1882), with the shear wave
speed of granite. Damping is a constant loss factor per mode.

Impacts are read from a recorded clip: the impulse on a stone between two records is its momentum
change minus the impulse of gravity. An impulse above a threshold excites every mode at the
vertex that faces the impulse (the contact side) with modal velocity phi_k . J; the radiated
pressure is taken proportional to the modal acceleration. A short raised cosine pulse of the
contact force is added, since stone modes lie mostly in the upper audible band. Frequencies
above the Nyquist limit are dropped. Event times follow the video time base (slow motion keeps
the real pitch).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

GRANITE = {"E": 5.0e10, "nu": 0.25, "density": 2650.0, "loss": 0.006}
LAMB_S2 = 2.64  # omega R / c_s of the fundamental spheroidal mode of a free sphere


def _edges(F: np.ndarray) -> np.ndarray:
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def surface_modes(mesh, mass: float, n_modes: int = 24, max_faces: int = 400, material: dict = GRANITE) -> dict:
    """Modal frequencies (Hz), mass normalised mode shapes (n_modes, nV, 3) at the vertices of the
    decimated surface, and those vertices."""
    from ..perception.decimate import decimate

    m = decimate(mesh, max_faces) if mesh.n_faces > max_faces else mesh
    V = np.asarray(m.vertices, dtype=float)
    F = np.asarray(m.faces, dtype=np.int64)
    nV = len(V)
    # lumped masses from vertex areas
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    va = np.zeros(nV)
    for k in range(3):
        np.add.at(va, F[:, k], a / 3.0)
    mv = mass * va / va.sum()
    M = np.repeat(mv, 3)
    # stiffness of axial edge springs (unit spring constant, scaled below)
    K = np.zeros((3 * nV, 3 * nV))
    for i, j in _edges(F):
        d = V[j] - V[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        u = d / L
        k = np.outer(u, u) / L
        for p, q, s in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
            K[3 * p: 3 * p + 3, 3 * q: 3 * q + 3] += s * k
    w2, phi = eigh(K, np.diag(M))
    elastic = np.where(w2 > 1e-6 * max(w2.max(), 1e-300))[0]
    if len(elastic) == 0:
        return {"freq": np.zeros(0), "shapes": np.zeros((0, nV, 3)), "vertices": V, "vertex_mass": mv}
    # scale: lowest elastic mode at the Lamb frequency of the equivalent sphere
    E, nu, rho = material["E"], material["nu"], material["density"]
    cs = np.sqrt(E / (2.0 * (1.0 + nu)) / rho)
    R = (3.0 * abs(mesh.volume()) / (4.0 * np.pi)) ** (1.0 / 3.0)
    w0 = LAMB_S2 * cs / R
    scale = w0 / np.sqrt(w2[elastic[0]])
    sel = elastic[:n_modes]
    freq = np.sqrt(w2[sel]) * scale / (2.0 * np.pi)
    shapes = phi[:, sel].T.reshape(len(sel), nV, 3)  # phi^T M phi = 1
    return {"freq": freq, "shapes": shapes, "vertices": V, "vertex_mass": mv}


def impacts_from_clip(rec, masses: dict, gravity=(0.0, 0.0, -9.81), threshold: float = 2e-3) -> list:
    """(time, body id, impulse vector) for every record interval whose non gravitational momentum
    change exceeds ``threshold`` (N s)."""
    g = np.asarray(gravity, dtype=float)
    out = []
    for a in range(len(rec.states) - 1):
        s0, s1 = rec.states[a], rec.states[a + 1]
        dt = rec.times[a + 1] - rec.times[a]
        for b, m in masses.items():
            J = m * (s1.v[b] - s0.v[b]) - m * g * dt
            if np.linalg.norm(J) > threshold:
                out.append((float(rec.times[a + 1]), int(b), J))
    return out


def synthesize(events: list, modes: dict, poses: dict, duration: float, sr: int = 48000, time_map=None, loss: float = GRANITE["loss"], pulse_ms: float = 0.6, pulse_gain: float = 0.3) -> np.ndarray:
    """Mono signal (float, peak 0.9) for the impact ``events``. ``modes[b]`` from
    ``surface_modes``; ``poses[b](t)`` gives the stone's pose (SE3) at time t, used to find the
    contact vertex; ``time_map`` maps simulation time to signal time."""
    n = int(duration * sr) + 1
    y = np.zeros(n)
    tmap = time_map or (lambda t: t)
    nyq = 0.45 * sr
    pulse_n = max(2, int(pulse_ms * 1e-3 * sr))
    pulse = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(pulse_n) / (pulse_n - 1))
    for t, b, J in events:
        md = modes.get(b)
        if md is None or len(md["freq"]) == 0:
            continue
        ts = tmap(t)
        i0 = int(round(ts * sr))
        if i0 < 0 or i0 >= n:
            continue
        T = poses[b](t)
        Jb = T.inverse().apply_vector(J) if hasattr(T, "apply_vector") else _rot_inv(T.q, J)
        V = md["vertices"]
        # the contact is on the side the impulse pushes from
        jn = Jb / max(np.linalg.norm(Jb), 1e-12)
        vi = int(np.argmin(V @ jn))
        tt = np.arange(n - i0) / sr
        for f, shape in zip(md["freq"], md["shapes"]):
            if f >= nyq:
                continue
            w = 2.0 * np.pi * f
            qdot = float(shape[vi] @ Jb)
            env = np.exp(-0.5 * loss * w * tt)
            cut = np.searchsorted(-env, -1e-4)  # stop once the mode has decayed
            y[i0: i0 + cut] += qdot * w * env[:cut] * np.sin(w * tt[:cut])
        j = min(pulse_n, n - i0)
        y[i0: i0 + j] += pulse_gain * np.linalg.norm(J) * pulse[:j] * (2.0 * np.pi * 2000.0)
    peak = float(np.abs(y).max())
    return 0.9 * y / peak if peak > 0 else y


def _rot_inv(q, v):
    from ..core.transforms import quat_conj, quat_rotate

    return quat_rotate(quat_conj(q), v)


def write_wav(path, y: np.ndarray, sr: int = 48000) -> None:
    from scipy.io import wavfile

    wavfile.write(str(path), sr, (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16))


def mux_audio(video_path, wav_path, out_path) -> int:
    """Copy the video stream and add the WAV as AAC audio (ffmpeg from imageio-ffmpeg)."""
    import subprocess

    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(video_path), "-i", str(wav_path), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)]
    return subprocess.run(cmd).returncode
