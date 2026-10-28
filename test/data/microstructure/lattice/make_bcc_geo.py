"""gmsh .geo (OpenCASCADE) of an n^3 BCC strut lattice in the box [0, n]^3, for the conformal-fit
validation tests (test_microstructure_hook_e2e_lattice_*).

Struts are cylinders of radius r from each cell center to its corners, with spheres of radius r
at the cell centers and corners; their union is cut by the box. Every face gets one physical
group; msh_to_mfem.py assigns the boundary attributes by position.

The committed lattice_bcc_tet.mesh (gmsh 3.0.6):
  python3 make_bcc_geo.py 1 0.12 0.08 > bcc.geo
  gmsh bcc.geo -3 -format msh2 -o bcc.msh
  python3 msh_to_mfem.py bcc.msh 1.0 lattice_bcc_tet.mesh

usage: python3 make_bcc_geo.py n r mesh_size
"""
import itertools
import sys

n, r, h = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
out = ['SetFactory("OpenCASCADE");', f'Mesh.CharacteristicLengthMin = {h};',
       f'Mesh.CharacteristicLengthMax = {h};']
k = 0
for i, j, l in itertools.product(range(n), repeat=3):
    c = (i + 0.5, j + 0.5, l + 0.5)
    for dx, dy, dz in itertools.product((-0.5, 0.5), repeat=3):
        k += 1
        out.append(f'Cylinder({k}) = {{{c[0]}, {c[1]}, {c[2]}, {dx}, {dy}, {dz}, {r}}};')
# Spheres at the nodes round the junctions
nodes = set(itertools.product(range(n + 1), repeat=3))
nodes |= {(i + 0.5, j + 0.5, l + 0.5) for i, j, l in itertools.product(range(n), repeat=3)}
for p in sorted(nodes):
    k += 1
    out.append(f'Sphere({k}) = {{{p[0]}, {p[1]}, {p[2]}, {r}}};')
out.append(f'u() = BooleanUnion{{ Volume{{1}}; Delete; }}{{ Volume{{2:{k}}}; Delete; }};')
out.append(f'Box({k + 1}) = {{0, 0, 0, {n}, {n}, {n}}};')
out.append('v() = BooleanIntersection{ Volume{u()}; Delete; }'
           f'{{ Volume{{{k + 1}}}; Delete; }};')
out.append('s() = Boundary{ Volume{v()}; };')
out.append('Physical Volume(1) = v();')
out.append('Physical Surface(1) = s();')
print("\n".join(out))
