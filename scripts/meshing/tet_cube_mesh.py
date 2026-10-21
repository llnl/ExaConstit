#!/usr/bin/env python3
"""Write an MFEM tetrahedral mesh of the unit cube for ExaConstit.

The cube is divided into n x n x n hex cells and each cell is split into 6 tets
(Kuhn/Freudenthal split, which is conforming across cells). Every tet takes the grain
ID of its hex cell as its element attribute, read from a grain map with one ID per
cell ordered x fastest, then y, then z (the order of ExaConstit's auto-generated
mesh). Boundary attributes follow ExaConstit's auto mesh:
1 = z = 0, 2 = x = 0, 3 = y = 0, 4 = z = 1, 5 = x = 1, 6 = y = 1.

Usage: tet_cube_mesh.py n grain_map_file output.mesh
"""
import sys

import numpy as np


def main():
    n = int(sys.argv[1])
    grains = np.loadtxt(sys.argv[2], dtype=int).ravel()
    out = sys.argv[3]
    if grains.size != n**3:
        sys.exit(f"Grain map has {grains.size} entries; {n**3} are needed")

    def vid(i, j, k):
        return i + (n + 1) * (j + (n + 1) * k)

    verts = np.array([[i / n, j / n, k / n] for k in range(n + 1)
                      for j in range(n + 1) for i in range(n + 1)])
    # Cell corners are numbered by bits (x, y, z) = (1, 2, 4); all 6 tets share the
    # main diagonal 0-7
    kuhn = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7),
            (0, 5, 1, 7)]
    elems = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                corners = [vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1))
                           for b in range(8)]
                attr = grains[i + n * (j + n * k)]
                for tet in kuhn:
                    v = [corners[a] for a in tet]
                    x = verts[v]
                    # Positive orientation
                    if np.linalg.det(np.array([x[1] - x[0], x[2] - x[0], x[3] - x[0]])) < 0:
                        v[2], v[3] = v[3], v[2]
                    elems.append((attr, v))

    def boundary_attr(tri):
        x = verts[list(tri)]
        # (attribute on the min face, attribute on the max face) for x, y, z
        for d, (lo, hi) in enumerate([(2, 5), (3, 6), (1, 4)]):
            if np.all(np.abs(x[:, d]) < 1e-12):
                return lo
            if np.all(np.abs(x[:, d] - 1.0) < 1e-12):
                return hi
        return 0

    bdr = []
    for _, v in elems:
        for face in [(v[1], v[2], v[3]), (v[0], v[3], v[2]), (v[0], v[1], v[3]),
                     (v[0], v[2], v[1])]:
            attr = boundary_attr(face)
            if attr:
                bdr.append((attr, face))

    with open(out, "w") as fh:
        fh.write("MFEM mesh v1.0\n\ndimension\n3\n\nelements\n%d\n" % len(elems))
        for attr, v in elems:
            fh.write("%d 4 %s\n" % (attr, " ".join(map(str, v))))
        fh.write("\nboundary\n%d\n" % len(bdr))
        for attr, face in bdr:
            fh.write("%d 2 %s\n" % (attr, " ".join(map(str, face))))
        fh.write("\nvertices\n%d\n3\n" % len(verts))
        for x in verts:
            fh.write("%.16g %.16g %.16g\n" % tuple(x))
    print(len(elems), "tets,", len(bdr), "boundary triangles")


if __name__ == "__main__":
    main()
