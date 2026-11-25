"""Demo render: icospheres and a box on the studio preset (Section 10.9 smoke test)."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topple.core.mesh import box, icosphere
from topple.core.transforms import SE3
from topple.render import Camera, Material, Scene, render, tonemap, write_png, write_exr, key_light_exposure
from topple.render.presets import ground_preset, lighting_preset, material_preset

out_dir = Path(__file__).resolve().parents[1] / "results"
out_dir.mkdir(exist_ok=True)
sc = Scene()
for m, mat, name in ground_preset("studio_slab"):
    sc.add(m, mat, name=name)
rng = np.random.default_rng(0)
sc.add(icosphere(4, 0.08), material_preset("granite", rng), SE3(t=[0.0, 0.0, 0.08]))
sc.add(icosphere(4, 0.06), material_preset("river_basalt", rng), SE3(t=[0.0, 0.0, 0.22]))
sc.add(icosphere(3, 0.05), material_preset("quartzite", rng), SE3(t=[0.2, 0.12, 0.05]))
sc.add(box((0.12, 0.09, 0.07)), material_preset("sandstone", rng), SE3.from_axis_angle([0, 0, 1], 0.4, [-0.22, 0.1, 0.035]))
lights, sky, bg = lighting_preset("studio", center=(0, 0, 0.15))
for l in lights:
    sc.add_light(l)
sc.background = bg
cam = Camera(position=[0.75, -0.6, 0.28], look_at=[0.0, 0.0, 0.14], focal_mm=50, f_number=4.0, width=200, height=133)
exposure = key_light_exposure(sc, [0, 0, 0.3], [0, 0, 1])
t0 = time.time()
out = render(sc, cam, spp=32, seed=0, denoise=True)
dt = time.time() - t0
st = out["stats"]
write_png(out_dir / "demo_render.png", tonemap(out["beauty"], exposure))
write_png(out_dir / "demo_render_denoised.png", tonemap(out["beauty_denoised"], exposure))
write_exr(out_dir / "demo_render.exr", {k: out[k] for k in ("beauty", "albedo", "normal", "depth", "overlay")})
print(f"rendered {cam.width}x{cam.height} at 32 spp in {dt:.2f} s, {st.rays} rays, {st.rays_per_second/1e6:.2f} M rays/s, exposure {exposure:.3g}")
print("wrote", out_dir / "demo_render.png")
