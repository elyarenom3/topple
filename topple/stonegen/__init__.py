"""Stone generation (Section 5): four procedural families, erosion, descriptors and rejection sampling."""

from .noise import perlin3, fbm3, worley3  # noqa: F401
from .families import family_a_hull, family_b_ellipsoid, family_c_superquadric, family_d_voronoi, radial_remesh_convex  # noqa: F401
from .erosion import erode  # noqa: F401
from .descriptors import descriptors  # noqa: F401
from .generator import StoneSpec, StoneGenerator, stone_to_body  # noqa: F401
