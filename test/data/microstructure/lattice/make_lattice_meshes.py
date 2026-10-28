"""Tet and hex meshes of a BCC strut lattice with filleted junctions, for the conformal-fit
lattice tests (test_microstructure_hook_e2e_lattice_*).

Geometry: struts of radius r from each cell center to its corners, in the infinite lattice cut by
the box [0, n]^3 (so the corner junctions are filleted like the center ones). Junctions are
filleted at radius rho by a morphological closing of the strut union (dilate by rho, erode by
rho), on a grid of 160 points per unit; the level set phi is the signed distance to that solid,
smoothed over about one grid spacing.

  tet: marching cubes of phi (80 points per unit) to an STL, gmsh classifies it and meshes the
       volume (Delaunay, Netgen and Relocate3D optimization), then the nodes are optimized.
  hex: grid cubes (`voxels` per unit) whose center is deeper than 1.3 cube sizes inside form
       the core; one pillow hex on every outward core face not on a box face goes from the face
       to the surface along the smoothed core-surface normals; then the nodes are optimized.

Node optimization (both): minimize the sum over element corners of mu^p, with mu = |T|_F^2 /
(3 det(T)^(2/3)) the shape measure of the corner matrix T (relative to the regular tet for
tets; 1 for a cube corner or a regular tet), det regularized so inverted corners untangle, plus
a penalty holding the strut-surface nodes on phi = 0 (raised in steps, then an exact projection);
box-plane coordinates stay fixed. First p = 1 (untangling), then p = 4 (the worst corners
weigh most, which leaves the fit room under its quality limits). JAX gives the gradients,
scipy's L-BFGS minimizes.

Output: MFEM mesh v1.0, element attribute 1; boundary attributes 1 z = 0, 2 x = 0, 3 y = 0,
4 z = n, 5 x = n, 6 y = n, 7 the strut surfaces.

Needs numpy, scipy, scikit-image and jax; tet also the gmsh Python module (4.15).
The committed meshes:
  python3 make_lattice_meshes.py tet lattice_bcc_tet.mesh
  python3 make_lattice_meshes.py hex lattice_bcc_hex.mesh
"""
import argparse
import itertools
import math
import os
import tempfile

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("kind", choices=("tet", "hex"))
parser.add_argument("out")
parser.add_argument("--cells", type=int, default=1)
parser.add_argument("--radius", type=float, default=0.12)
parser.add_argument("--fillet", type=float, default=0.15)
parser.add_argument("--voxels", type=int, default=32, help="hex: grid cubes per unit")
parser.add_argument("--size", type=float, default=0.08, help="tet: gmsh element size")
args = parser.parse_args()
n, r, rho = args.cells, args.radius, args.fillet
L = float(n)


# ----- level set -----
def lattice_phi(m, pad=0.35):
    h = 1.0 / m
    npad = int(math.ceil(pad / h))
    ax = np.arange(-npad, n * m + npad + 1) * h
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
    du = np.full(G.shape[:3], np.inf)
    for i, j, l in itertools.product(range(-1, n + 1), repeat=3):
        c = np.array([i + 0.5, j + 0.5, l + 0.5])
        for d in itertools.product((-0.5, 0.5), repeat=3):
            a, b = c, c + np.array(d)
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            if np.any(hi < -pad - r) or np.any(lo > n + pad + r):
                continue
            ab = b - a
            t = np.clip(((G - a) @ ab) / (ab @ ab), 0, 1)
            du = np.minimum(du, np.linalg.norm(G - (a + t[..., None] * ab), axis=-1))
    if rho <= 0:
        return ax, h, du - r
    closed = gaussian_filter(rho - distance_transform_edt(du < r + rho) * h, 1.2) < 0
    phi = (distance_transform_edt(~closed) - distance_transform_edt(closed)) * h
    return ax, h, gaussian_filter(phi, 1.0)


ax, hp, phi = lattice_phi(160)
x0g, ng = ax[0], len(ax)
PG = np.stack([phi, *np.gradient(phi, hp)], -1)


def phi_grad(p):
    """Trilinear phi and unit gradient at the points p (k, 3)."""
    p = np.atleast_2d(p)
    u = np.clip((p - x0g) / hp, 0, ng - 1.000001)
    i = u.astype(int)
    f = u - i
    out = 0.0
    for c in range(8):
        o = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1])
        w = np.prod(np.where(o, f, 1 - f), axis=1)
        idx = i + o
        out = out + w[:, None] * PG[idx[:, 0], idx[:, 1], idx[:, 2]]
    g = out[:, 1:]
    return out[:, 0], g / np.maximum(np.linalg.norm(g, axis=1), 1e-30)[:, None]


def on_box(p):
    return (np.abs(p) < 1e-9 * L) | (np.abs(p - L) < 1e-9 * L)


# ----- initial meshes -----
HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
TET_FACES = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]


def tet_mesh():
    import gmsh
    from skimage.measure import marching_cubes
    m = 80
    ax80, h80, phi80 = lattice_phi(m)
    G = np.meshgrid(ax80, ax80, ax80, indexing="ij")
    box = np.max(np.stack([v for g in G for v in (-g, g - L)]), 0)
    V, T, _, _ = marching_cubes(np.maximum(phi80, box), 0.0, spacing=(h80,) * 3)
    V = np.clip(V + ax80[0], 0, L)
    key = np.round(V / 1e-9).astype(np.int64)
    _, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V, T = V[uid], inv.ravel()[T]
    P = V[T]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    lmax = np.max([np.linalg.norm(P[:, a] - P[:, b], axis=1)
                   for a, b in ((0, 1), (1, 2), (2, 0))], 0)
    T = T[area > 1e-3 * lmax ** 2]
    with tempfile.TemporaryDirectory() as tmp:
        stl = os.path.join(tmp, "lattice.stl")
        with open(stl, "w") as o:
            o.write("solid lattice\n")
            for t in T:
                o.write("facet normal 0 0 0\n outer loop\n")
                for q in V[t]:
                    o.write(f"  vertex {q[0]:.9e} {q[1]:.9e} {q[2]:.9e}\n")
                o.write(" endloop\nendfacet\n")
            o.write("endsolid lattice\n")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(stl)
        gmsh.model.mesh.classifySurfaces(40 * math.pi / 180, True, True, math.pi)
        gmsh.model.mesh.createGeometry()
        loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in gmsh.model.getEntities(2)])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        for key_, val in (("Mesh.MeshSizeMin", args.size), ("Mesh.MeshSizeMax", args.size),
                          ("Mesh.Algorithm", 6), ("Mesh.Algorithm3D", 1),
                          ("Mesh.Optimize", 1), ("Mesh.OptimizeNetgen", 1)):
            gmsh.option.setNumber(key_, val)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize("Relocate3D")
        tags, coords, _ = gmsh.model.mesh.getNodes()
        _, _, conn = gmsh.model.mesh.getElements(3)
        gmsh.finalize()
    X = np.asarray(coords).reshape(-1, 3)
    where = np.full(int(np.max(tags)) + 1, -1)
    where[np.asarray(tags, int)] = np.arange(len(tags))
    E = where[np.asarray(conn[0], int).reshape(-1, 4)]
    used = np.unique(E)
    remap = np.full(len(X), -1)
    remap[used] = np.arange(len(used))
    X, E = X[used], remap[E]
    vol = np.einsum("ij,ij->i", np.cross(X[E[:, 1]] - X[E[:, 0]], X[E[:, 2]] - X[E[:, 0]]),
                    X[E[:, 3]] - X[E[:, 0]])
    E[vol < 0] = E[vol < 0][:, [0, 2, 1, 3]]
    return X, E


def hex_mesh():
    m = args.voxels
    N, h = n * m, L / (n * m)
    centers = (np.indices((N, N, N)).reshape(3, -1).T + 0.5) * h
    cells = np.argwhere((phi_grad(centers)[0] < -1.3 * h).reshape(N, N, N))
    corner = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1),
              (0, 1, 1)]
    node_id, hexes = {}, []
    for c in cells:
        ids = []
        for o in corner:
            key = (c[0] + o[0], c[1] + o[1], c[2] + o[2])
            ids.append(node_id.setdefault(key, len(node_id)))
        hexes.append(ids)
    X = [np.array(key) * h for key in sorted(node_id, key=node_id.get)]
    nbr = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
    kept = set(map(tuple, cells))
    pillow_faces = []
    for e, c in enumerate(cells):
        for f, d in zip(HEX_FACES, nbr):
            if (c[0] + d[0], c[1] + d[1], c[2] + d[2]) in kept:
                continue
            q = [hexes[e][k] for k in f]
            a = int(np.nonzero(d)[0][0])
            x = X[q[0]][a]
            if abs(x) < 1e-12 * L or abs(x - L) < 1e-12 * L:
                continue  # on a box plane
            pillow_faces.append((q, np.array(d, float)))
    normal, surf_nbrs = {}, {}
    for q, d in pillow_faces:
        for i, v in enumerate(q):
            normal[v] = normal.get(v, 0.0) + d
            surf_nbrs.setdefault(v, set()).update((q[(i + 1) % 4], q[(i + 3) % 4]))
    for _ in range(4):
        normal = {v: nv + sum(normal[u] for u in surf_nbrs[v]) for v, nv in normal.items()}
    prime = {}
    for v, nv in normal.items():
        p0 = X[v]
        nv = nv.copy()
        onbox = on_box(p0)
        nv[(np.abs(p0) < 0.5 * h) | (np.abs(p0 - L) < 0.5 * h)] = 0.0
        if np.linalg.norm(nv) < 1e-9:
            nv = phi_grad(p0)[1][0]
            nv[onbox] = 0.0
        nv = nv / max(np.linalg.norm(nv), 1e-300)
        free = ~onbox

        def phi_dom(p):  # struts intersected with the box planes the node is not on
            bt = np.max(np.concatenate([-p[free], p[free] - L])) if free.any() else -np.inf
            return max(float(phi_grad(p)[0][0]), float(bt))
        lo, hi = 0.0, 4.0 * h
        while phi_dom(p0 + hi * nv) < 0.0 and hi < 40 * h:
            hi *= 2.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if phi_dom(p0 + mid * nv) < 0.0 else (lo, mid)
        prime[v] = len(X)
        X.append(np.clip(p0 + 0.5 * (lo + hi) * nv, 0.0, L))
    for q, _ in pillow_faces:
        hexes.append(q + [prime[v] for v in q])
    return np.array(X), np.array(hexes)


def boundary(E, X, faces):
    count = {}
    for e in E:
        for f in faces:
            q = tuple(int(e[k]) for k in f)
            count.setdefault(tuple(sorted(q)), []).append(q)
    out = []
    for qs in count.values():
        if len(qs) != 1:
            continue
        pts = X[list(qs[0])]
        attr = 7
        for a, d, val in ((1, 2, 0.0), (2, 0, 0.0), (3, 1, 0.0), (4, 2, L), (5, 0, L), (6, 1, L)):
            if np.all(np.abs(pts[:, d] - val) < 1e-9 * L):
                attr = a
                break
        out.append((attr, qs[0]))
    return out


# ----- node optimization -----
def optimize(X0, E, bdr):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    tet = E.shape[1] == 4
    if tet:
        corners = np.array([(0, 1, 2, 3), (1, 2, 0, 3), (2, 0, 1, 3), (3, 0, 2, 1)])
        w_reg = np.array([[1.0, 0.5, 0.5], [0.0, math.sqrt(3) / 2, math.sqrt(3) / 6],
                          [0.0, 0.0, math.sqrt(2.0 / 3.0)]])
        w_inv = jnp.asarray(np.linalg.inv(w_reg))
    else:
        corners = np.array([(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
                            (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)])
    S = np.unique([v for attr, q in bdr if attr == 7 for v in q])
    free = ~on_box(X0)
    h = np.median(np.linalg.norm(X0[E[:, 1]] - X0[E[:, 0]], axis=1))
    PHI, Ej = jnp.asarray(phi), jnp.asarray(E)

    def trilinear(p):
        u = jnp.clip((p - x0g) / hp, 0, ng - 1.000001)
        i0 = jnp.floor(u).astype(jnp.int32)
        f = u - i0
        out = 0.0
        for c in range(8):
            o = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1])
            idx = i0 + o
            out = out + jnp.prod(jnp.where(o, f, 1 - f), axis=1) * PHI[idx[:, 0], idx[:, 1],
                                                                        idx[:, 2]]
        return out

    def energy(X, w, delta, power):
        P = X[Ej]
        c, a, b, d = corners.T
        M = jnp.stack([P[:, a] - P[:, c], P[:, b] - P[:, c], P[:, d] - P[:, c]], -1)
        if tet:
            M = M @ w_inv
        det = jnp.linalg.det(M)
        dreg = 0.5 * (det + jnp.sqrt(det * det + delta * delta))
        mu = jnp.sum(M * M, axis=(-2, -1)) / (3.0 * dreg ** (2.0 / 3.0))
        return (jnp.mean(mu ** power)
                + w * jnp.sum(trilinear(X[S]) ** 2) / (h * h * max(len(S), 1)))

    value_grad = jax.jit(jax.value_and_grad(energy))
    X = X0.copy()

    def run(w, delta, power, iters):
        def f(z):
            Y = X.copy()
            Y[free] = z
            v, g = value_grad(jnp.asarray(Y), w, delta, power)
            return float(v), np.asarray(g)[free]
        X[free] = minimize(f, X[free], jac=True, method="L-BFGS-B",
                           options={"maxiter": iters, "maxcor": 30}).x

    for power in (1.0, 4.0):
        for w, delta in ((10.0, 1e-2), (100.0, 1e-4), (1e3, 0.0), (1e4, 0.0)):
            run(w, delta * h ** 3, power, 300)
        # exact projection of the strut-surface nodes (box-plane coordinates kept), then a polish
        # with them fixed
        for _ in range(10):
            f, g = phi_grad(X[S])
            g[~free[S]] = 0.0
            X[S] = np.clip(X[S] - (f / np.maximum((g * g).sum(1), 1e-30))[:, None] * g, 0, L)
        keep = free[S].copy()
        free[S] = False
        run(0.0, 0.0, power, 200)
        free[S] = keep
    return X


def quality(X, E):
    if E.shape[1] == 4:
        e = [X[E[:, k]] - X[E[:, 0]] for k in (1, 2, 3)]
        vol = np.einsum("ij,ij->i", np.cross(e[0], e[1]), e[2]) / 6
        l2 = sum(((X[E[:, a]] - X[E[:, b]]) ** 2).sum(1)
                 for a, b in itertools.combinations(range(4), 2))
        mr = 12 * (3 * np.abs(vol)) ** (2 / 3) / l2 * np.sign(vol)
        return f"{len(E)} tets, smallest mean ratio {mr.min():.3f}, inverted {(vol <= 0).sum()}"
    P = X[E]
    sj = np.inf
    for c, a, b, d in ((0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7), (4, 7, 5, 0),
                       (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)):
        e1, e2, e3 = P[:, a] - P[:, c], P[:, b] - P[:, c], P[:, d] - P[:, c]
        det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
        sj = np.minimum(sj, det / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
                                   * np.linalg.norm(e3, axis=1)))
    return f"{len(E)} hexes, smallest scaled Jacobian {sj.min():.3f}, inverted {(sj <= 0).sum()}"


X, E = tet_mesh() if args.kind == "tet" else hex_mesh()
faces = TET_FACES if args.kind == "tet" else HEX_FACES
print("initial:", quality(X, E))
X = optimize(X, E, boundary(E, X, faces))
print("optimized:", quality(X, E))
bdr = boundary(E, X, faces)
geom, bgeom = (4, 2) if args.kind == "tet" else (5, 3)
with open(args.out, "w") as f:
    f.write(f"MFEM mesh v1.0\n\ndimension\n3\n\nelements\n{len(E)}\n")
    for e in E:
        f.write(f"1 {geom} " + " ".join(map(str, e)) + "\n")
    f.write(f"\nboundary\n{len(bdr)}\n")
    for attr, q in bdr:
        f.write(f"{attr} {bgeom} " + " ".join(map(str, q)) + "\n")
    f.write(f"\nvertices\n{len(X)}\n3\n")
    for p in X:
        f.write("%.10g %.10g %.10g\n" % tuple(p))
