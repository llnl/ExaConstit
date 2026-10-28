"""Converts a gmsh 2.2 ASCII tet mesh of a lattice in the box [0, L]^3 to an MFEM v1.0 mesh
(make_bcc_geo.py shows the whole chain).

Element attribute 1. Boundary triangles get attributes by where they lie: 1 z = 0, 2 x = 0,
3 y = 0, 4 z = L, 5 x = L, 6 y = L, 7 elsewhere (the strut surfaces).
usage: python3 msh_to_mfem.py in.msh L out.mesh
"""
import sys

import numpy as np

src, L, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
lines = open(src).read().split("\n")
i = lines.index("$Nodes")
nn = int(lines[i + 1])
X = np.array([[float(v) for v in lines[i + 2 + k].split()[1:4]] for k in range(nn)])
ids = {int(lines[i + 2 + k].split()[0]): k for k in range(nn)}
j = lines.index("$Elements")
ne = int(lines[j + 1])
tets, tris = [], []
for k in range(ne):
    v = [int(t) for t in lines[j + 2 + k].split()]
    kind, ntags = v[1], v[2]
    nodes = [ids[t] for t in v[3 + ntags:]]
    if kind == 4:
        tets.append(nodes)
    elif kind == 2:
        tris.append(nodes)


def bdr_attr(pts, tol=1e-9):
    for attr, d, val in ((1, 2, 0.0), (2, 0, 0.0), (3, 1, 0.0), (4, 2, L), (5, 0, L), (6, 1, L)):
        if np.all(np.abs(pts[:, d] - val) < tol):
            return attr
    return 7


with open(out, "w") as f:
    f.write("MFEM mesh v1.0\n\ndimension\n3\n\nelements\n%d\n" % len(tets))
    for t in tets:
        f.write("1 4 " + " ".join(map(str, t)) + "\n")
    f.write("\nboundary\n%d\n" % len(tris))
    for t in tris:
        f.write("%d 2 %s\n" % (bdr_attr(X[t]), " ".join(map(str, t))))
    f.write("\nvertices\n%d\n3\n" % len(X))
    for p in X:
        f.write("%.17g %.17g %.17g\n" % tuple(p))
print(len(tets), "tets", len(tris), "boundary triangles")
