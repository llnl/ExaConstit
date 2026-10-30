"""Renders a generated microstructure (the generator's ParaView output) to a PNG with ParaView.

Run with ParaView's pvbatch (ParaView 5.11 or newer):

    pvbatch --force-offscreen-rendering render_grains.py RESULT.pvd OUT.png [options]

Each grain gets a fixed random color from its grain id, so the same grain has the same color in
every picture of a run. With --phase-csv, every phase has one flat color instead (phase 1 blue,
phase 2 orange-red, then purple and yellow); --gray-above draws passthrough blocks (the grain ids
after the last grain) gray. The part is drawn as its outer surface; --cutaway removes one corner
octant of the bounding box so the inside shows.
"""
import argparse
import math

from paraview.simple import (  # pvbatch provides paraview.simple
    CleantoGrid,
    Clip,
    ExtractSurface,
    GetActiveViewOrCreate,
    PointDatatoCellData,
    MergeBlocks,
    OpenDataFile,
    ProgrammableFilter,
    Render,
    SaveScreenshot,
    Show,
    Threshold,
    UpdatePipeline,
)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("pvd")
parser.add_argument("out")
parser.add_argument("--grain-field", default="grain_id",
                    help="field holding the grain id (point or cell data)")
parser.add_argument("--phase-csv", default="",
                    help="the run's <basename>_grains.csv: color by phase families")
parser.add_argument("--gray-above", type=int, default=0,
                    help="draw grain ids above this number (passthrough blocks) light gray")
parser.add_argument("--cutaway", action="store_true", help="remove the +x +y +z corner octant")
parser.add_argument("--cut-exact", action="store_true",
                    help="cut the corner with flat planes through the elements (instead of "
                         "removing whole elements); right for meshes whose nodes were moved")
parser.add_argument("--cut-fraction", type=float, default=0.5,
                    help="size of the removed corner as a fraction of each box edge")
parser.add_argument("--edges", action="store_true", help="draw the element edges")
parser.add_argument("--azimuth", type=float, default=35.0)
parser.add_argument("--elevation", type=float, default=25.0)
parser.add_argument("--zoom", type=float, default=1.12)
parser.add_argument("--focus", type=float, nargs=3, default=None,
                    help="look at this point instead of the center")
parser.add_argument("--view-up", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help="screen-up direction; give e.g. 0 1 0 with --elevation 90 (top view)")
parser.add_argument("--parallel", action="store_true",
                    help="parallel projection (no perspective), e.g. for a top view")
parser.add_argument("--size", type=int, nargs=2, default=[1600, 1200])
parser.add_argument("--seed", type=int, default=7, help="color permutation")
parser.add_argument("--raster", action="store_true",
                    help="plain OpenGL shading instead of OSPRay ray tracing (faster)")
parser.add_argument("--samples", type=int, default=16, help="ray-tracing samples per pixel")
args = parser.parse_args()

# grain id -> phase, from the grain table the generator writes next to the mesh
phase_of = {}
if args.phase_csv:
    import csv

    with open(args.phase_csv) as table:
        for row in csv.DictReader(line for line in table if not line.startswith("#")):
            phase_of[int(row["grain_id"])] = int(row["phase"])

reader = MergeBlocks(Input=OpenDataFile(args.pvd))  # one dataset from the per-rank pieces
UpdatePipeline()
# The generator writes grain_id as an element-wise constant on per-element points; as cell data
# every element carries its grain id exactly
if reader.PointData.GetArray(args.grain_field) is not None:
    reader = PointDatatoCellData(Input=reader)
    reader.ProcessAllArrays = 1
# Every element has its own copies of its corner points in this output, so without merging them
# every element face would count as outer surface (memory grows with the element count)
reader = CleantoGrid(Input=reader)
reader.UpdatePipeline()
source = reader
if args.cutaway:
    # The +x +y +z corner box of the part, cut_fraction of each bounding-box edge
    b = reader.GetDataInformation().GetBounds()
    f = args.cut_fraction
    lo = [b[0] + (1 - f) * (b[1] - b[0]), b[2] + (1 - f) * (b[3] - b[2]),
          b[4] + (1 - f) * (b[5] - b[4])]
if args.cutaway and args.cut_exact:
    # Clip away the corner box exactly: the cut faces are flat planes through the elements
    hi = [b[1] + 1.0, b[3] + 1.0, b[5] + 1.0]  # past the part, so only the three inner faces cut
    source = Clip(Input=reader)
    source.ClipType = "Box"
    source.ClipType.Position = lo
    source.ClipType.Length = [h - l for h, l in zip(hi, lo)]
    source.Exact = 1
    source.Invert = 0  # keep what lies outside the box
elif args.cutaway:
    # Remove the elements whose center lies in the corner box (whole elements, so the cut faces
    # show the element staircase), by flagging them and thresholding the flag away
    flag = ProgrammableFilter(Input=reader)
    flag.Script = f"""
import numpy as np
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtk.numpy_interface import dataset_adapter as dsa
centers = vtkCellCenters()
centers.SetInputData(self.GetInput())
centers.Update()
c = np.asarray(dsa.WrapDataObject(centers.GetOutput()).Points)
lo = np.array({lo!r})
out = dsa.WrapDataObject(self.GetOutput())
out.ShallowCopy(self.GetInput())
out.CellData.append(np.all(c > lo, axis=1).astype(np.float64), 'cut')
"""
    source = Threshold(Input=flag, Scalars=["CELLS", "cut"], LowerThreshold=-0.5,
                       UpperThreshold=0.5)

surface = ExtractSurface(Input=source)
# Color per cell from the grain id (and phase), as unsigned char RGB used directly
colors = ProgrammableFilter(Input=surface)
colors.Script = f"""
import colorsys
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa
inp = dsa.WrapDataObject(self.GetInput())
out = dsa.WrapDataObject(self.GetOutput())
out.ShallowCopy(self.GetInput())
g = np.asarray(inp.CellData['{args.grain_field}']).astype(np.int64).ravel()
phase_of = {phase_of!r}
p = np.array([phase_of.get(k, 1) for k in g.tolist()], dtype=np.int64)
def u(k, salt):
    # fixed pseudo-random number in [0, 1) per (grain, salt), splitmix64-style
    z = (k * 0x9E3779B97F4A7C15 + salt * 0xBF58476D1CE4E5B9 + {args.seed}) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return ((z ^ (z >> 31)) >> 11) / float(1 << 53)
table = {{}}
rgb = np.empty((g.size, 3), dtype=np.uint8)
# One flat color per phase (blue, orange-red, purple, yellow): the picture shows the phases only
phase_hue = [(0.58, 0.65), (0.04, 0.75), (0.80, 0.45), (0.14, 0.70)]
for i, (k, ph) in enumerate(zip(g.tolist(), p.tolist())):
    key = (k, ph)
    if key not in table:
        if 0 < {args.gray_above} < k:
            h, s, v = 0.0, 0.0, 0.72
        elif phase_of:
            h, s = phase_hue[(ph - 1) % len(phase_hue)]
            v = 0.9
        else:
            h = u(k, 1)
            s = 0.22 + 0.33 * u(k, 2)
            v = 0.80 + 0.18 * u(k, 3)
        table[key] = [int(255 * c) for c in colorsys.hsv_to_rgb(h, s, v)]
    rgb[i] = table[key]
out.CellData.append(rgb, 'rgb')
"""

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = args.size
view.Background = [1.0, 1.0, 1.0]
try:
    view.BackgroundColorMode = "Single Color"
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.OrientationAxesVisibility = 0
view.UseFXAA = 1
if not args.raster:
    # OSPRay ray caster with ambient occlusion: creases and the cut-away read in 3D
    view.EnableRayTracing = 1
    view.BackEnd = "OSPRay raycaster"
    view.SamplesPerPixel = args.samples
    view.AmbientSamples = 4
    view.Shadows = 0  # shadows of the cut-away edges streak across the inner walls
    view.LightScale = 1.0

display = Show(colors, view)
display.SetRepresentationType("Surface With Edges" if args.edges else "Surface")
display.ColorArrayName = ["CELLS", "rgb"]
display.MapScalars = 0
display.Specular = 0.15
if args.raster:
    # the plain OpenGL lighting is darker than the ray caster's: lift the flat shading
    display.Ambient = 0.35
    display.Diffuse = 0.75
display.EdgeColor = [0.15, 0.15, 0.15]
try:
    display.LineWidth = 1.0
except AttributeError:
    pass

b = colors.GetDataInformation().GetBounds()
center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
if args.focus:
    center = list(args.focus)
diag = ((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2) ** 0.5
az = math.radians(args.azimuth)
el = math.radians(args.elevation)
dist = 2.2 * diag / args.zoom
Render(view)  # the first render resets the camera; place it afterwards
view.CameraFocalPoint = center
view.CameraPosition = [center[0] + dist * math.cos(el) * math.cos(az),
                       center[1] + dist * math.cos(el) * math.sin(az),
                       center[2] + dist * math.sin(el)]
view.CameraViewUp = list(args.view_up)
view.CameraViewAngle = 30.0
view.CameraParallelProjection = 1 if args.parallel else 0
view.CameraParallelScale = dist * math.tan(math.radians(15.0))  # same framing as the 30 deg view
Render(view)
SaveScreenshot(args.out, view, ImageResolution=args.size, TransparentBackground=0)
print("wrote", args.out)
