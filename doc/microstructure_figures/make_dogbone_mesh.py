"""Hexahedral mesh of a flat tensile specimen (dog-bone) for the guide's passthrough example.

  python3 make_dogbone_mesh.py OUT.msh [--size H] [--thickness T]

Needs the gmsh Python module (pip install gmsh). The shape is the 6 mm wide subsize sheet-type
specimen of ASTM E8/E8M (Fig. 1), in millimeters: width W = 6 of the reduced parallel section,
which is A = 32 long; fillet radius R = 6; grip sections C = 10 wide and B = 30 long, so the
overall length is 100.9 (the standard asks for at least 100). The thickness T is the sheet
thickness (at most 6 in the standard; 2 here by default).

The length runs along x, the width along y, the thickness along z, centered on the origin. The
reduced section with its two fillets is element attribute 1 and the grips attribute 2 (Gmsh
physical volumes), the element sets the generator's active_attributes and passthrough_attributes
refer to. The mesh is structured: H is the element size in the reduced section, the elements grow
toward the grip ends.
"""
import argparse
import math

import gmsh

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("out")
parser.add_argument("--size", type=float, default=0.064, help="element size in the gauge (mm)")
parser.add_argument("--thickness", type=float, default=2.0, help="sheet thickness T (mm)")
args = parser.parse_args()

W, A, R, B, C = 6.0, 32.0, 6.0, 30.0, 10.0
h, T = args.size, args.thickness
# The fillet is tangent to the reduced section and ends on the grip width: its length along x
fillet = math.sqrt(R**2 - (R - (C - W) / 2) ** 2)
x1 = A / 2               # end of the reduced parallel section
x2 = x1 + fillet         # start of the grip section
x3 = x2 + B              # end of the specimen

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # the version MFEM reads with attributes
gmsh.model.add("dogbone")
geo = gmsh.model.geo


def point(x, y):
    return geo.addPoint(x, y, -T / 2)


# Planform split into five four-sided patches along x: grip, fillet, reduced section, fillet, grip
xs = [-x3, -x2, -x1, x1, x2, x3]
half = [C / 2, C / 2, W / 2, W / 2, C / 2, C / 2]
bottom = [point(x, -w) for x, w in zip(xs, half)]
top = [point(x, w) for x, w in zip(xs, half)]
# Fillet centers sit R away from the reduced-section edge, outside the specimen
centers = {(i, s): point(xc, s * (W / 2 + R)) for i, xc in ((1, -x1), (3, x1)) for s in (-1, 1)}

n_gauge = round(A / h)
n_fillet = round(fillet / h)
n_width = round(W / h)
# Grip elements grow by `grow` per element from h at the fillet toward the grip ends
grow = 1.03
n_grip = math.ceil(math.log(1 + B * (grow - 1) / h) / math.log(grow))
sides = [geo.addLine(b, t) for b, t in zip(bottom, top)]
patches = []
for i in range(5):
    if i in (1, 3):
        # fillet arcs: from the grip corner to the reduced section (i = 1), or back (i = 3)
        cb, ct = centers[(i, -1)], centers[(i, 1)]
        low = geo.addCircleArc(bottom[i], cb, bottom[i + 1])
        up = geo.addCircleArc(top[i], ct, top[i + 1])
        n, prog = n_fillet, 1.0
    else:
        low = geo.addLine(bottom[i], bottom[i + 1])
        up = geo.addLine(top[i], top[i + 1])
        n = n_gauge if i == 2 else n_grip
        # curves run along +x: the left grip shrinks toward the fillet, the right one grows
        prog = {0: 1.0 / grow, 2: 1.0, 4: grow}[i]
    for curve in (low, up):
        geo.mesh.setTransfiniteCurve(curve, n + 1, "Progression", prog)
    loop = geo.addCurveLoop([low, sides[i + 1], -up, -sides[i]])
    patches.append(geo.addPlaneSurface([loop]))
for s in sides:
    geo.mesh.setTransfiniteCurve(s, n_width + 1)
for p in patches:
    geo.mesh.setTransfiniteSurface(p)
    geo.mesh.setRecombine(2, p)

n_thick = max(1, round(T / h))
volumes = []
for p in patches:
    out = geo.extrude([(2, p)], 0, 0, T, numElements=[n_thick], recombine=True)
    volumes.append(next(tag for dim, tag in out if dim == 3))
geo.synchronize()
gmsh.model.addPhysicalGroup(3, volumes[1:4], 1)
gmsh.model.setPhysicalName(3, 1, "reduced_section")
gmsh.model.addPhysicalGroup(3, [volumes[0], volumes[4]], 2)
gmsh.model.setPhysicalName(3, 2, "grips")
# Boundary sets for a simulation: the two grip end faces (1 and 2) and the rest (3)
ends = []
for x in (-x3, x3):
    ends.append([tag for dim, tag in gmsh.model.getEntitiesInBoundingBox(
        x - 1e-6, -C, -T, x + 1e-6, C, T, 2)])
gmsh.model.addPhysicalGroup(2, ends[0], 1)
gmsh.model.setPhysicalName(2, 1, "grip_end_minus_x")
gmsh.model.addPhysicalGroup(2, ends[1], 2)
gmsh.model.setPhysicalName(2, 2, "grip_end_plus_x")
outer = gmsh.model.getBoundary([(3, v) for v in volumes], combined=True, oriented=False)
rest = [tag for dim, tag in outer if tag not in ends[0] + ends[1]]
gmsh.model.addPhysicalGroup(2, rest, 3)
gmsh.model.setPhysicalName(2, 3, "sides")
gmsh.model.mesh.generate(3)
gmsh.write(args.out)
num = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
print("wrote", args.out, "with", num, "hexahedra")
gmsh.finalize()
