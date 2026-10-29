"""Inputs of the voxel-map example in doc/microstructure.md: a synthetic voxel map of an additively
built block, written as an ExaConstit microstructure file, and a round tensile bar meshed with
hexahedra.

  python3 make_voxel_example.py OUT_DIR

writes OUT_DIR/am_block.h5 and OUT_DIR/round_bar.mesh. Needs numpy, scipy and h5py.

The block stands in for measured data (a 3D EBSD or diffraction volume) of a part built by laser
powder-bed fusion: a 1 x 1 x 2 block of 160 x 160 x 320 voxels, built along z, with laser tracks
along x spaced 0.25 apart in y. Grains grow as columns along the local growth direction, which
tilts from the build direction toward the centre line of each track (up to 35 degrees), so the
columns form the chevron pattern seen in sections across the tracks. Each grain has a random
orientation.

The bar (radius 0.35, length 1.6, axis along z through the block centre) is a structured
hexahedral "O-grid": a square core of hexahedra surrounded by four blocks that blend from the core
edges to the circle.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "scripts", "microstructure"))
import exaconstit_microstructure_io as msio  # noqa: E402

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(11)

# ----- voxel map of the additively built block -----
nx, ny, nz = 160, 160, 320
h = 1.0 / 160.0
num_grains = 1500
elongation = 6.0          # columns are this many times longer than wide
track_spacing = 0.25      # laser tracks along x, side by side in y
max_tilt = np.radians(35.0)
seeds = rng.random((num_grains, 3)) * np.array([1.0, 1.0, 2.0])


def growth_direction(y):
    """Unit growth direction per point: tilted in the y-z plane toward the nearest track centre."""
    u = np.mod(y, track_spacing) / track_spacing - 0.5          # -0.5 .. 0.5 across a track
    tilt = max_tilt * np.clip(-u / 0.2, -1.0, 1.0)              # toward the centre line u = 0
    return np.stack([np.zeros_like(y), np.sin(tilt), np.cos(tilt)], axis=1)


# Candidate seeds come from a search in coordinates squeezed along z (columns are mostly along
# z); the winner is the smallest column distance |r|^2 - (1 - 1/k^2) (g . r)^2 among them
squeeze = np.array([1.0, 1.0, 1.0 / 4.0])
tree = cKDTree(seeds * squeeze)
x = (np.arange(nx) + 0.5) * h
y = (np.arange(ny) + 0.5) * h
z = (np.arange(nz) + 0.5) * h
Z, Y, X = np.meshgrid(z, y, x, indexing="ij")  # (nz, ny, nx): x varies fastest
points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
del X, Y, Z
labels = np.empty(len(points), dtype=np.int64)
chunk = 200_000  # bounds the memory of the candidate arrays
for start in range(0, len(points), chunk):
    p = points[start:start + chunk]
    _, candidates = tree.query(p * squeeze, k=32)
    r = p[:, None, :] - seeds[candidates]                       # (n, 32, 3)
    g = growth_direction(p[:, 1])[:, None, :]
    along = np.sum(r * g, axis=2)
    d2 = np.sum(r * r, axis=2) - (1.0 - 1.0 / elongation**2) * along**2
    labels[start:start + chunk] = candidates[np.arange(len(p)), np.argmin(d2, axis=1)]
grain_ids = (labels + 1).reshape(nz, ny, nx)
del points, labels

ids = np.unique(grain_ids)
quats = rng.normal(size=(ids.size, 4))  # uniformly random rotations
quats /= np.linalg.norm(quats, axis=1, keepdims=True)
quats[quats[:, 0] < 0] *= -1.0
msio.write_voxel_map(os.path.join(out_dir, "am_block.h5"), grain_ids, origin=(0.0, 0.0, 0.0),
                     spacing=(h, h, h), feature_ids=ids, feature_quats=quats,
                     feature_phases=np.ones(ids.size, dtype=np.int32),
                     attributes={"source": "synthetic additively built block, "
                                           "make_voxel_example.py"})

# ----- round bar: hexahedral O-grid, written as an MFEM mesh -----
radius, length, z0 = 0.35, 1.6, 0.2
center = np.array([0.5, 0.5])
core = 0.5 * radius        # half-width of the square core
n_core, n_ring, n_axial = 56, 22, 256


def ring_point(side, t, rho):
    """Point of the outer block on core side `side` (0..3) at position t along the side and
    rho from the core edge (0) to the circle (1)."""
    angle = np.pi / 4 + side * np.pi / 2 + t * np.pi / 2
    corner_a = core * np.sqrt(2.0) * np.array([np.cos(angle - t * np.pi / 2),
                                               np.sin(angle - t * np.pi / 2)])
    corner_b = core * np.sqrt(2.0) * np.array([np.cos(angle - t * np.pi / 2 + np.pi / 2),
                                               np.sin(angle - t * np.pi / 2 + np.pi / 2)])
    inner = (1.0 - t) * corner_a + t * corner_b
    outer = radius * np.array([np.cos(angle), np.sin(angle)])
    return (1.0 - rho) * inner + rho * outer


# 2D quads (4 points each, counter-clockwise) and which of them lie on the outer circle
quads, outer_edges = [], []
s = np.linspace(-core, core, n_core + 1)
for j in range(n_core):
    for i in range(n_core):
        quads.append([(s[i], s[j]), (s[i + 1], s[j]), (s[i + 1], s[j + 1]), (s[i], s[j + 1])])
for side in range(4):
    for i in range(n_core):
        t0, t1 = i / n_core, (i + 1) / n_core
        for k in range(n_ring):
            r0, r1 = k / n_ring, (k + 1) / n_ring
            quads.append([ring_point(side, t0, r0), ring_point(side, t1, r0),
                          ring_point(side, t1, r1), ring_point(side, t0, r1)])
            if k == n_ring - 1:
                outer_edges.append((ring_point(side, t1, 1.0), ring_point(side, t0, 1.0)))
quads = np.array(quads, dtype=float) + center
outer_edges = np.array(outer_edges, dtype=float) + center
# counter-clockwise seen from +z, so the hexahedra have positive volume
e1, e2 = quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0]
area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
quads[area < 0] = quads[area < 0][:, ::-1]

# 2D vertices without duplicates, then the layers along z
key = np.round(quads.reshape(-1, 2) / 1e-9).astype(np.int64)
uniq, inverse = np.unique(key, axis=0, return_inverse=True)
xy = uniq * 1e-9
quad_v = inverse.reshape(-1, 4)
edge_key = np.round(outer_edges.reshape(-1, 2) / 1e-9).astype(np.int64)
lookup = {tuple(k): i for i, k in enumerate(uniq)}
edge_v = np.array([lookup[tuple(k)] for k in edge_key]).reshape(-1, 2)
nv2 = len(xy)
zs = z0 + length * np.arange(n_axial + 1) / n_axial
vertices = np.concatenate([np.column_stack([xy, np.full(nv2, zk)]) for zk in zs])
hexes = np.concatenate([np.hstack([quad_v + k * nv2, quad_v + (k + 1) * nv2])
                        for k in range(n_axial)])
bottom = quad_v[:, ::-1]                         # outward normal -z
top = quad_v + n_axial * nv2                     # outward normal +z
side_faces = np.concatenate([np.column_stack([edge_v[:, 1] + k * nv2, edge_v[:, 0] + k * nv2,
                                              edge_v[:, 0] + (k + 1) * nv2,
                                              edge_v[:, 1] + (k + 1) * nv2])
                             for k in range(n_axial)])
with open(os.path.join(out_dir, "round_bar.mesh"), "w") as f:
    f.write("MFEM mesh v1.0\n\ndimension\n3\n\nelements\n%d\n" % len(hexes))
    f.writelines("1 5 %d %d %d %d %d %d %d %d\n" % tuple(e) for e in hexes)
    f.write("\nboundary\n%d\n" % (len(side_faces) + 2 * len(bottom)))
    f.writelines("1 3 %d %d %d %d\n" % tuple(q) for q in side_faces)   # 1 = side
    f.writelines("2 3 %d %d %d %d\n" % tuple(q) for q in bottom)       # 2 = bottom end
    f.writelines("3 3 %d %d %d %d\n" % tuple(q) for q in top)          # 3 = top end
    f.write("\nvertices\n%d\n3\n" % len(vertices))
    f.writelines("%.15g %.15g %.15g\n" % tuple(v) for v in vertices)
print("grains in the voxel map:", ids.size, " hexahedra:", len(hexes),
      " vertices:", len(vertices))
