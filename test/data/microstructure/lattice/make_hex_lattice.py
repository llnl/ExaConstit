"""All-hex mesh of a strut lattice by voxels and a pillow layer (as voxel meshers such as Sculpt
build them), for the conformal-fit validation tests (test_microstructure_hook_e2e_lattice_*).

Struts are capsules (cylinders with round ends) of radius r: "sc" (simple cubic) struts run along
the axes through the cell centers and end on the box faces; "bcc" struts go from each cell center
to its corners. The box is [0, n]^3.
  1. Core: the grid cubes (m per cell and axis) whose center lies deeper than `core` voxel sizes
     inside the struts, so every core node is inside.
  2. Pillows: one hex on every outward face of the core that is not on a box face, from the face
     to the points where rays from its nodes meet the strut surface (the struts intersected with
     the box). A ray follows the outward normal of the core surface at the node, averaged over
     its surface neighbors `smooth` times; near a box face (within `near` voxel sizes) it stays
     parallel to that face; a ray longer than 2.5 voxel sizes (nearly tangent to the surface) is
     replaced by the closest point.
  3. MFEM mesh v1.0: element attribute 1; boundary attributes 1 z = 0, 2 x = 0, 3 y = 0,
     4 z = n, 5 x = n, 6 y = n, 7 the strut surfaces. The smallest corner scaled Jacobian is
     printed; the "bcc" kind folds pillows where the diagonal struts meet the box faces, "sc" with
     the defaults does not.

The committed lattice_sc_hex.mesh: python3 make_hex_lattice.py lattice_sc_hex.mesh
(n = 1, r = 0.2, m = 16, sc).
"""
import argparse
import itertools

import numpy as np

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("out")
parser.add_argument("--kind", choices=("sc", "bcc"), default="sc")
parser.add_argument("--cells", type=int, default=1)
parser.add_argument("--radius", type=float, default=0.2)
parser.add_argument("--voxels", type=int, default=16, help="voxels per cell and axis")
parser.add_argument("--core", type=float, default=1.3)
parser.add_argument("--smooth", type=int, default=4)
parser.add_argument("--near", type=float, default=0.5)
args = parser.parse_args()
n, r, m, out, kind = args.cells, args.radius, args.voxels, args.out, args.kind
core, smooth_normals, near = args.core, args.smooth, args.near
ray_limit = 2.5
L = float(n)
N = n * m  # voxels per side

segs = []
if kind == "bcc":
    for i, j, k in itertools.product(range(n), repeat=3):
        c = np.array([i + 0.5, j + 0.5, k + 0.5])
        for d in itertools.product((-0.5, 0.5), repeat=3):
            segs.append((c, c + np.array(d)))
else:  # simple cubic: struts along the axes through the cell centers, ends on the box faces
    for a in range(3):
        for i, j in itertools.product(range(n), repeat=2):
            p = np.zeros(3)
            q = np.zeros(3)
            others = [b for b in range(3) if b != a]
            p[others[0]] = q[others[0]] = i + 0.5
            p[others[1]] = q[others[1]] = j + 0.5
            q[a] = n
            segs.append((p, q))
A = np.array([s[0] for s in segs])
B = np.array([s[1] for s in segs])
AB = B - A
ABn = (AB * AB).sum(1)


def phi_grad(p):
    """Signed distance to the capsule union (approximate inside, exact outside) and its gradient."""
    p = np.atleast_2d(p)
    ap = p[:, None, :] - A[None, :, :]
    t = np.clip((ap * AB[None]).sum(2) / ABn[None], 0.0, 1.0)
    q = A[None] + t[..., None] * AB[None]
    dv = p[:, None, :] - q
    dist = np.sqrt((dv * dv).sum(2))
    idx = dist.argmin(1)
    dmin = dist[np.arange(len(p)), idx]
    g = dv[np.arange(len(p)), idx] / np.maximum(dmin, 1e-30)[:, None]
    return dmin - r, g


h = L / N
centers = (np.indices((N, N, N)).reshape(3, -1).T + 0.5) * h
# Core: voxels well inside, so every core node is inside the struts by about 0.4 h or more
inside = phi_grad(centers)[0] < -core * h
cells = np.argwhere(inside.reshape(N, N, N))  # (i, j, k) of kept voxels (indices order x, y, z)

corner = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
node_id = {}
hexes = []
for c in cells:
    ids = []
    for o in corner:
        key = (c[0] + o[0], c[1] + o[1], c[2] + o[2])
        if key not in node_id:
            node_id[key] = len(node_id)
        ids.append(node_id[key])
    hexes.append(ids)
X = [np.array(key) * h for key in sorted(node_id, key=node_id.get)]

# Pillows: the outward faces of the core that are not on a box plane (oriented outward, so the
# normal points into the pillow: the bottom face of an MFEM hex)
faces_local = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
nbr = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
kept = set(map(tuple, cells))
tol = 1e-12 * L
core_hexes = len(hexes)
pillow_faces = []
for e, c in enumerate(cells):
    for f, d in zip(faces_local, nbr):
        if (c[0] + d[0], c[1] + d[1], c[2] + d[2]) in kept:
            continue
        q = [hexes[e][k] for k in f]
        pts = np.array([X[v] for v in q])
        axis = int(np.nonzero(d)[0][0])
        if abs(pts[0, axis]) < tol or abs(pts[0, axis] - L) < tol:
            continue  # the face lies on a box plane
        pillow_faces.append((q, np.array(d, float)))
# Node normals: the mean outward normal of the pillowed faces at the node, then averaged over
# the surface neighbors (smooths the staircase), with the components normal to box planes the
# node is on removed
normal = {}
surf_nbrs = {}
for q, d in pillow_faces:
    for i, v in enumerate(q):
        normal[v] = normal.get(v, 0.0) + d
        surf_nbrs.setdefault(v, set()).update((q[(i + 1) % 4], q[(i + 3) % 4]))
for it in range(smooth_normals):
    normal = {v: n + sum(normal[u] for u in surf_nbrs[v]) for v, n in normal.items()}
prime = {}
for v, nv in normal.items():
    p0 = X[v]
    nv = nv.copy()
    onbox = (np.abs(p0) < tol) | (np.abs(p0 - L) < tol)
    nearbox = (np.abs(p0) < near * h) | (np.abs(p0 - L) < near * h)
    nv[nearbox] = 0.0  # rays near a box plane stay parallel to it
    if np.linalg.norm(nv) < 1e-9:  # the normal is all box-plane directions (a box edge)
        nv = -phi_grad(p0)[1][0]
        nv[onbox] = 0.0
        nv = -nv if np.linalg.norm(nv) > 0 else nv
    nv = nv / max(np.linalg.norm(nv), 1e-300)
    # phi(p0 + t nv) = 0 by bisection on [0, tmax] (phi(p0) < 0)
    lo, hi = 0.0, 4.0 * h
    free = ~onbox

    def phi_dom(p):  # struts intersected with the box planes the node is not on
        box_term = np.max(np.concatenate([-p[free], p[free] - L])) if free.any() else -np.inf
        return max(phi_grad(p)[0][0], float(box_term))
    while phi_dom(p0 + hi * nv) < 0.0:
        hi *= 2.0
    for it in range(60):
        mid = 0.5 * (lo + hi)
        if phi_dom(p0 + mid * nv) < 0.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    p = p0 + t * nv
    if t > ray_limit * h:  # a ray nearly tangent to the surface: closest point instead
        p = p0.copy()
        for it in range(4):
            ph, g = phi_grad(p)
            g = g[0].copy()
            g[onbox] = 0.0
            p = p - ph[0] / max(g @ g, 1e-30) * g
        p = np.clip(p, 0.0, L)
    prime[v] = len(X)
    X.append(p)
for q, d in pillow_faces:
    hexes.append(q + [prime[v] for v in q])
X = np.array(X)
hexes = np.array(hexes)
print(f"{core_hexes} core hexes + {len(hexes) - core_hexes} pillows")

# Boundary faces: faces that belong to one hex only (outward from it)
count = {}
for e, hx in enumerate(hexes):
    for f in faces_local:
        q = tuple(hx[k] for k in f)
        key = tuple(sorted(q))
        count.setdefault(key, []).append(q)
bfaces = np.array([v[0] for v in count.values() if len(v) == 1])


def corner_sj(Xh):
    """Smallest corner scaled Jacobian of each hex (MFEM corner order)."""
    nbrs = {0: (1, 3, 4), 1: (2, 0, 5), 2: (3, 1, 6), 3: (0, 2, 7), 4: (7, 5, 0), 5: (4, 6, 1),
            6: (5, 7, 2), 7: (6, 4, 3)}
    out = np.full(len(Xh), np.inf)
    for c, (a, b, d) in nbrs.items():
        e1 = Xh[:, a] - Xh[:, c]
        e2 = Xh[:, b] - Xh[:, c]
        e3 = Xh[:, d] - Xh[:, c]
        det = (np.cross(e1, e2) * e3).sum(1)
        nrm = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)
        out = np.minimum(out, det / nrm)
    return out


sj = corner_sj(X[hexes])
print(f"corner scaled Jacobian: min {sj.min():.3f}, share < 0.2: {(sj < 0.2).mean():.4f}, "
      f"inverted {(sj <= 0).sum()}")
with open(out, "w") as f:
    f.write("MFEM mesh v1.0\n\ndimension\n3\n\nelements\n%d\n" % len(hexes))
    for hx in hexes:
        f.write("1 5 " + " ".join(map(str, hx)) + "\n")
    f.write("\nboundary\n%d\n" % len(bfaces))
    for q in bfaces:
        pts = X[list(q)]
        attr = 7
        for a, d, val in ((1, 2, 0.0), (2, 0, 0.0), (3, 1, 0.0), (4, 2, L), (5, 0, L), (6, 1, L)):
            if np.all(np.abs(pts[:, d] - val) < 1e-9):
                attr = a
                break
        f.write("%d 3 %s\n" % (attr, " ".join(map(str, q))))
    f.write("\nvertices\n%d\n3\n" % len(X))
    for p in X:
        f.write("%.17g %.17g %.17g\n" % tuple(p))
