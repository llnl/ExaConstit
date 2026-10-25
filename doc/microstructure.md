# ExaConstit microstructure generator: theory notes and file formats

This document holds the background and exact file specifications behind
`src/microstructure_options.toml`, the user reference for every generator option. Read the options
file first; come here when you need the derivation behind a number, or need to write or read the
files with your own scripts.

Contents:
1. Johnson-Mehl grain count
2. Geodesic distance and the metrics
3. File formats: inputs (seed table CSV, ExaConstit microstructure file, phase-to-region file) and
   outputs (mesh, grain file, orientation file, region map, grain table, adjacency table,
   microstructure file)
4. Conformal grain boundaries (`[Conformal]`)

---

## 1. Johnson-Mehl grain count

Johnson-Mehl placement models nucleation and growth: nuclei appear at random places and times at a
constant rate I (nuclei per unit volume per unit time), each grows as a sphere at constant speed v,
and a nucleus that would appear inside an already grown grain never forms.

**Untransformed fraction.** For a constant rate in an infinite body, the fraction of material not
yet reached by any grain at time t is the classical Johnson-Mehl-Avrami-Kolmogorov (JMAK) result

    1 - X(t) = exp(-(pi/3) I v^3 t^4).

**Grain count.** New grains can only appear in untransformed material, so nuclei are accepted at the
rate I (1 - X(t)) per unit volume. Integrating over all time:

    N / V = integral_0^inf I exp(-(pi/3) I v^3 t^4) dt
          = I Gamma(5/4) ((pi/3) I v^3)^(-1/4)
          = Gamma(5/4) (3/pi)^(1/4) (I/v)^(3/4)
          = 0.8960 (I/v)^(3/4),

using integral_0^inf exp(-a t^4) dt = Gamma(5/4) a^(-1/4). Gamma(5/4) = 0.9064 and
(3/pi)^(1/4) = 0.9885, whose product is the 0.896 in the options file. Turned around, about N grains
in a volume V need I/v = (N / (0.896 V))^(4/3).

**Finite parts.** The formula is for an infinite body. Grains at a free surface can only grow
inward, which leaves room for more nuclei, so finite parts get somewhat more grains; [Cleanup] then
merges the smallest ones. Example 04 of `microstructure_examples` (unit cube, I/v = 500): the
formula predicts 95, 99 nuclei formed, 89 grains remained after cleanup with
`min_grain_elements = 8`.

**Completion time.** The untransformed fraction drops below 5% at
t = (ln 20 / ((pi/3) I v^3))^(1/4) = 1.30 (I v^3)^(-1/4).

The same formula is implemented as `JohnsonMehlGrainDensity` in
`src/microstructure/seeding/johnson_mehl.hpp`.

References:
- W. A. Johnson, R. F. Mehl, "Reaction kinetics in processes of nucleation and growth", Trans.
  AIME 135 (1939) 416-442.
- M. Avrami, "Kinetics of Phase Change. I General Theory", J. Chem. Phys. 7 (1939) 1103-1112,
  doi:10.1063/1.1750380.
- A. N. Kolmogorov, Izv. Akad. Nauk SSSR, Ser. Mat. 1 (1937) 355-359.

---

## 2. Geodesic distance and the metrics

`distance = "geodesic"` measures distance along paths that stay inside the material. The solver
(`labeling/geodesic_fim.hpp`, `labeling/geodesic_graph.hpp`) grows **one** front outward from all
seeds at once; every mesh vertex keeps the first arrival and the seed it came from. The front
starts at seed i with the value -a_i (its head start), so the result is

    T(x) = min_i ( d_path(x, s_i) - a_i ).

This is exact when "closest" means *path length plus a fixed offset per seed*, because the minimum
over seeds of (path length + constant) is itself the solution of one front problem (an eikonal
equation |grad T| = 1 with those starting values). That covers:
- voronoi (all a_i = 0),
- additive (a_i from the seed weights, e.g. Johnson-Mehl birth times).

**Quadric with one tensor A** (anisotropy kind "none" or "tensor", no seed weights). Paths are
measured with the A-length sqrt(r^T A r) instead of the plain length. Write A = L L^T (Cholesky)
and map the mesh with y = L^T x: the A-length of any path is the plain length of its image, a
linear map keeps elements as elements and linear interpolation linear, so the same single front
runs unchanged on the mapped coordinates (`labeling/geodesic_metric.hpp`). On a convex part the
labels equal the straight-line quadric labels.

**Columnar and lamellar.** Their tensors I - a a^T (columns along a) and n n^T (layers with normal
n) give zero length along the column axis (within the layer plane). A path-length solver cannot
use them: the map y = L^T x would collapse the mesh onto a plane or a line. The geodesic mode
therefore uses a regularized tensor in which columns are k times longer than wide,

    columnar:  |r|^2 - (1 - 1/k^2) (a . r)^2,      lamellar:  (n . r)^2 + |r - (n . r) n|^2 / k^2,

with k = `geodesic.elongation`. With finite k a column can end: the boundary between neighboring
columns i and j, at a point whose height differs from seed i's by dz, moves by about
(dz / k)^2 / (2 d_ij) toward i, with d_ij the distance between the columns. With dz at most the
part length L along the axis and d_ij about the column width w, the shift is at most about
w (L / (k w))^2 / 2. The automatic value k = 10 L / w keeps it below w / 200. Here w is
sqrt(Lu Lv / N), the column width if the N seeds shared the bounding-box cross-section Lu x Lv;
for layers, the in-plane diameter over the thickness Ln / N is used. k is rounded up to two
significant digits, at least 10 and at most 1e4. Beyond 1e4 the mapped elements become slivers
that the solver's triangle update starts to discard as degenerate (near k ~ 1e6). The extents
come from exact min/max reductions, so k, and with it the labels, do not depend on the number of
ranks.

**Laguerre and weighted quadric** (closest = d^2 - w_i, `labeling/geodesic_weighted.hpp`). A heavy
seed can own points that another seed's front reaches first, so one front per mesh is not enough:
every mesh vertex keeps a short list of (seed, path length) pairs, and the eikonal update runs per
seed. A seed j *dominates* seed i at a vertex when it is both closer (d_j <= d_i) and has the lower
value (d_j^2 - w_j <= d_i^2 - w_i). Then j wins at every point reached from seed i through that
vertex: with L the remaining path length,

    (d_j + L)^2 - w_j - (d_i + L)^2 + w_i = (d_j^2 - w_j) - (d_i^2 - w_i) + 2 L (d_j - d_i) <= 0,

so i is dropped there. This is exact; the lists stay short (the winner plus a few neighbors near
grain boundaries; with equal weights only the winner). A cap (`geodesic.max_seeds_per_node`,
default 16) bounds the memory; if more seeds remain, the worst are cut and the log reports it. Only
then can the result depend on the number of ranks. Weighted quadric runs the same solver in the
mapped coordinates y = L^T x.

**Per-seed tensors** (`anisotropy = { kind = "per_seed" }`). Every seed measures paths with its own
tensor A_i, through its own map y = L_i^T x inside the same solver. The drop rule above needs the
remaining path to be no longer for seed j than for seed i, which holds when A_j <= A_i (A_i - A_j
positive semidefinite): then j may drop i as before. Otherwise a seed that is behind at a vertex
can catch up along its long direction, and it must be kept. Seeds with really different shapes are
rarely ordered, so the lists fill up to the cap; the log reports how often the cap cut a seed. With
at most `max_seeds_per_node` seeds in total nothing can be cut and the result is exact. Every
tensor must be positive definite here (finite grain sizes along every direction).

---

## 3. File formats

All lengths are in the units of the file; `units_scale` and `translate` convert them to mesh units
(x_mesh = units_scale * x_file + translate). The converters in `scripts/microstructure/` have their
own `--units-scale`, applied when they write; use the converter's or the options file's, not both,
or the data is scaled twice.

Orientations are always unit quaternions `w x y z` in ExaConstit's convention: a passive rotation
from the crystal frame to the sample frame. Concretely, ExaCMech builds C = R(q) with

    C = [[w^2+x^2-y^2-z^2, 2(xy - wz),      2(xz + wy)     ],
         [2(xy + wz),      w^2-x^2+y^2-z^2, 2(yz - wx)     ],
         [2(xz - wy),      2(yz + wx),      w^2-x^2-y^2+z^2]]

and maps tensors from crystal to sample components as A_sample = C A_crystal C^T
(ExaCMech `quat_to_tensor`). The Bunge matrix g (sample to crystal, rows = crystal axes in sample
coordinates) is C^T, so a quaternion computed from g must be conjugated.

### 3.1 Seed table as CSV text ("ExaConstit seed table v1", optional)

The CSV form is for short grain lists written by hand or in a spreadsheet. It is never required:
a microstructure file with only a grain table (section 3.2) says the same and is what the
converters and the generator write. Both are read by `[Seeds] source = "file"`; the reader tells
them apart by the HDF5 signature at the start of the file.

Grammar (reader: `src/microstructure/io/seed_table_io.hpp`):
- UTF-8; a byte-order mark and Windows (CRLF) line endings are accepted.
- Blank lines and lines whose first non-blank character is `#` are comments. A first comment line
  `# ExaConstit seed table v1` is customary (the converters write it) but not required.
- The first non-comment line is the header: comma-separated column names, case-insensitive,
  surrounding spaces ignored, any order.
- Every following line is one seed, with exactly as many comma-separated fields as the header.
  Spaces around fields are allowed. Numbers are C-locale decimals (point as decimal separator);
  scientific notation (1.5e-3) is fine; inf, nan and hexadecimal floats are refused.

Columns:

| Column | Required | Meaning |
|---|---|---|
| `x, y, z` | yes | seed position |
| `id` | no | integer, unique, >= 1 (default: the row number, counting from 1) |
| `qw, qx, qy, qz` | no (all four or none) | orientation; a row with norm off 1 by more than 1e-3 is an error; written normalized |
| `phase` | no | integer >= 1 |
| `weight` | no | laguerre / quadric w or additive a, in mesh units (not scaled) |
| `volume` or `radius` or `diameter` | no (at most one) | grain size > 0; volume scales by s^3, radius and diameter by s |
| `m11, m22, m33, m23, m13, m12` | no (all six or none) | grain shape tensor M (symmetric positive definite, length^2), scaled by s^2; used as A = M^-1 |

Refused columns: Euler-like names (`phi1`, `phi`, `phi2`, `euler*`, `bunge*`, `rodrigues*`,
`expmap*`), rotation-matrix entries (`r11` .. `r33`), and ambiguous quaternion names (`q0` .. `q3`,
`quat*`). Other unknown columns are ignored with one warning. Duplicate ids are an error;
duplicate positions only warn.

The quaternion sign does not matter physically (q and -q are the same rotation); it is kept as
written, apart from normalization.

### 3.2 ExaConstit microstructure file (HDF5)

One HDF5 file holds a grain table and, optionally, a voxel map (a grain id per cell of a regular
grid). With a voxel map it is read by `[Source] type = "voxel_map"`, `[VoxelMap] format =
"exaconstit"`; with only a grain table (with positions) by `[Seeds] source = "file"`. The
generator writes the grain table of a run with `[Output] microstructure_file = "grains"`.

Readers: `src/microstructure/io/microstructure_file_io.hpp` (header, grain table, grid
description; through Conduit relay) and `voxel_map_io.hpp` (the per-cell ids, read in tiles).
Writer for scripts: `write_microstructure_file` (and its grid shortcut `write_voxel_map`) in
`scripts/microstructure/exaconstit_microstructure_io.py`, which needs only h5py.

**Every value is an HDF5 dataset, never an HDF5 attribute.** Conduit's HDF5 reader reads groups
and datasets only, so a datasets-only file loads into a `conduit::Node` with one call, and the
voxel part is a valid Conduit Blueprint mesh (`conduit::blueprint::mesh::verify` accepts the
`mesh` group; tested for files written by the C++ test fixtures and by the Python writer).

| Path | Type | Values | Meaning |
|---|---|---|---|
| `format` | string | | `"ExaConstit microstructure"` |
| `format_version` | integer | 1 | checked: a newer version is refused |
| `grains/id` | integer | G | grain ids, >= 1, < 2^31, unique (required) |
| `grains/quat` | float | 4 per grain | w x y z, crystal-to-sample (passive), norm within 1e-3 of 1 |
| `grains/phase` | integer | G | phase per grain, >= 1 |
| `grains/position` | float | 3 per grain | seed point x y z (required to use the file as seeds) |
| `grains/volume` | float | G | target grain volume, > 0 (scaled by s^3) |
| `grains/weight` | float | G | laguerre / additive weight, in mesh units (not scaled) |
| `grains/shape` | float | 6 per grain | shape tensor M = m11 m22 m33 m23 m13 m12 (SPD, length^2, scaled by s^2) |
| `mesh/coordsets/coords/type` | string | | `"uniform"` |
| `mesh/coordsets/coords/dims/i`, `j`, `k` | integer | 1 each | grid POINTS per axis = cells + 1 (Blueprint convention) |
| `mesh/coordsets/coords/origin/x`, `y`, `z` | float | 1 each | lower corner of cell (0, 0, 0) |
| `mesh/coordsets/coords/spacing/dx`, `dy`, `dz` | float | 1 each | cell edge lengths, > 0 |
| `mesh/topologies/topo/type`, `coordset` | string | | `"uniform"`, `"coords"` |
| `mesh/fields/grain_id/association`, `topology` | string | | `"element"`, `"topo"` |
| `mesh/fields/grain_id/values` | integer | nx ny nz | grain id per cell, <= 0 = unassigned |

Rules:
- **Optional means "not given".** Every `grains/` dataset except `id` is optional, and a missing
  dataset never stands for a default value.
- **Row layout.** Per-grain datasets hold one row per grain: shape (c G) flat (as Conduit writes
  it) or (G, c) (as h5py writes a list of rows), with c values per grain. A transposed (c, G)
  array is refused instead of being misread.
- **Memory order of the grid:** x varies fastest, then y, then z. Cell (i, j, k), 0-based with i
  along x, is entry i + nx (j + ny k) of the flattened array, i.e. `values[k, j, i]` in numpy.
  `values` may be stored flat (nx ny nz) or as (nz, ny, nx), optionally (nz, ny, nx, 1); any other
  shape, including (nx, ny, nz) with nx != nz, is refused. Data in another order must be
  transposed before writing.
- Cell (i, j, k) is the box from origin + (i, j, k) * spacing to origin + (i+1, j+1, k+1) * spacing.
  Origin and spacing are required, although Blueprint itself would default them.
- **Types.** Any HDF5 integer or float type is converted on reading (int32, uint16, float32, ...).
  Strings may be fixed or variable length (h5py) or an array of 1-byte characters ending in a NUL
  (Conduit, so also the files the generator writes). Single numbers may be HDF5 scalars (h5py) or
  1-element arrays (Conduit).
- **Grain table and grid.** `grains/id` must list every id the mesh uses when orientations or
  phases are read; extra rows are allowed and ids need not be contiguous. The Python writer, given
  no per-grain data, writes the positive ids found in the grid.
- **Growing the format.** Readers ignore groups and datasets they do not know (e.g. the
  `provenance/` strings the converters write). Adding an optional dataset never changes
  `format_version`; the version only changes when the meaning of existing data changes, and a
  reader refuses versions newer than it knows.
- **Compression.** Chunked datasets are fine; ranks read cubes of `tile_size`^3 cells, so chunks
  of a similar size (e.g. 32^3) read efficiently. Compressed datasets need an HDF5 library built
  with the matching filter: gzip needs HDF5 built with zlib, which ExaConstit's own HDF5 build is
  not, so the Python writer does not compress unless asked (`compression="gzip"`). The reader names
  a missing filter in its error; `h5repack -f NONE in.h5 out.h5` removes compression.
- The serial HDF5 library is enough: ranks open the file read-only, each reading only its cubes.
- ExaConstit's older auto-mesh grain file plus orientation file is not read here.

DREAM.3D files are read directly (`format = "dream3d"`): `FeatureIds` per voxel, per-feature
`AvgQuats` and `Phases` (row = feature id, row 0 unused). Files written by DREAM.3D's EbsdLib store
quaternions scalar-last and as passive sample-to-crystal rotations, so use `quat_layout = "xyzw"`
and `conjugate = true` (verified against EbsdLib's Euler-to-quaternion conversion,
`Source/EbsdLib/Core/OrientationTransformation.hpp`, with `epsijk = 1` from
`Source/EbsdLib/Orientation/OrientationFwd.hpp`).

**Checking that the mesh lies inside the data.** The tool's summary prints the voxel grid
(dimensions, origin and spacing in mesh units) and the number of sample points that fell outside
the grid. Outside points take a neighbor's grain, so a large count means the mesh extends past the
data; compare the mesh bounding box with origin .. origin + dims * spacing.

### 3.2a Phase-to-region file (`[Regions] phase_to_region_file`)

Plain text, read while the options are parsed: one `phase region` pair of integers per line,
separated by spaces or tabs, the layout of ExaConstit's `region_mapping_file`. Blank lines and
lines whose first non-blank character is `#` are skipped. A phase listed twice with the same region
is accepted, with two different regions it is an error, and so is any line that is not exactly two
integers. The values follow the inline `phase_to_region` rules (phase ids >= 1, the regions used
must be exactly 1..M). It replaces the inline map; giving both is an error.

### 3.3 Output files (written by the tool; `<b>` = `<directory>/<basename>`)

| File | Format |
|---|---|
| `<b>.mesh` / `<b>.mesh.gz` | MFEM serial mesh ("MFEM mesh v1.0"), element attribute = grain id 1..N, boundary attributes kept from the input mesh, gzip when compression > 0, coordinates at 17 digits |
| `<b>.mesh.000000`, ... | MFEM parallel mesh ("MFEM mesh v1.2"), one file per part, gzip-compressed when compression > 0 (names unchanged; MFEM detects gzip from the first bytes) |
| `<b>.grains` | text, one grain id per line, for auto box meshes (with `grain_file = "auto"` not after a `[Conformal]` fit, whose moved nodes only the mesh file carries): line e = element e in ExaConstit's auto-mesh order (x fastest, then y, then z); read by ExaConstit's `grain_file` |
| `<b>.ori` | text, one `w x y z` line per grain (line g = grain g), 17 significant digits, no header; read by `orientation_file` |
| `<b>.regions` | text, one `grain region` line per grain, no header (only written with 2+ regions); read by `region_mapping_file` |
| `<b>_grains.csv` | CSV with header: grain_id, qw, qx, qy, qz, then region, source_id, num_elements, volume, cx, cy, cz, num_neighbors, target_volume, phase, filler, feature_id (optional columns only when they carry information) |
| `<b>_adjacency.csv` | CSV with header `grain_a,grain_b,shared_area,num_faces`, one row per pair of grains sharing element faces (grain_a < grain_b) |
| `<b>.h5` | microstructure file (section 3.2), only with `microstructure_file = "grains"`: `grains/id` = 1..N (passthrough grains left out, filler grains kept), `quat` (final orientations), `phase` (when every grain has one), and when every grain grew from a seed `position`, `weight`, `shape` (M = A^-1, only when every per-seed tensor is invertible) and `volume` (the seed's size target, when targets were used). A grain split off by the cleanup repeats its parent's seed. |
| `<b>_paraview/` | ParaView collection (`.pvd`), cell field `grain_id` |

Passthrough parts ([Mesh] passthrough_attributes) become the last grain ids, N+1, N+2, ..., so in
ParaView "grain_id > N" selects them.

---

## 4. Conformal grain boundaries (`[Conformal]`)

Code: `src/microstructure/conformal/`. The step runs after labeling (driver step 14) and moves
the mesh nodes; elements, connectivity and element type stay the same.

**Node labels.** Every node gets the sorted set of grains of the elements around it (1 label:
grain interior; 2: on a grain boundary; 3: on a triple line; 4: at a quad point), and the outer
boundary surfaces it lies on: boundary faces are grouped into surfaces by their normals (faces
whose normals differ by less than 30 degrees are one surface), and each surface records whether it
is flat (one exact normal) or curved. A node with more than three constraints in total (grain
labels minus one, plus up to three surfaces) is over-constrained and is not fitted.

**Targets.** For a node with labels a, b (and c, d) the analytic grain boundary is the set where
the metric values agree, f_a(y) = f_b(y) (= f_c(y) ...), with f the metric of the tessellation
(Voronoi, Laguerre, Additive or Quadric; for Voronoi this is the bisector plane). The node's target
is the closest point y of that set (plus the planes of its flat boundary surfaces) to the node
position x, found by a constrained Gauss-Newton projection:

    y <- x - J^T (J J^T)^-1 (J (x - y) + phi(y)),

where phi stacks the constraint values and J their gradients at y. On a curved boundary surface the
projection alternates with the closest point on the node's ring of boundary triangles. A target is
kept only if the constraints are met to a tolerance, it is at most about one element size h from
the node, and no other seed is closer there (otherwise the node is HELD where it is).

**Smoothed staircase targets.** Grains read from a voxel map, grown along paths (geodesic
distance), cut back by the connected mode, or moved by the fragment cleanup have no analytic
boundary. Their grain boundary nodes get the position of a smoothed copy of the staircase
(`conformal/smoothed_targets.hpp`): only nodes with 2 or 3 labels move; a node averages over its
neighbors along element edges that carry all of its labels (a face node over its boundary and the
triple lines around it, a triple line node along its line) and lie on all of its outer surfaces;
steps alternate the weights 0.5 and -0.53 (Taubin's lambda|mu smoothing: the negative step undoes
the shrinking of plain averaging but keeps the smoothing of short wavelengths such as stair
steps), 20 pairs, with the node put back on its outer surfaces after every step and the final
move limited to 2 element sizes. Quad points and over-constrained nodes do not move and anchor the
network. Every copy of a shared node merges the neighbor lists of all ranks, sorted by the
neighbors' input positions, so the result is the same bits for any partition. On an inclined
Voronoi bicrystal (12 elements per unit) the smoothed staircase lies on average 0.086 h from the
true bisector, against 0.34 h for the staircase nodes.

Which target a node gets: tessellations in the euclidean or connected mode use the analytic
boundary where it is valid (the checks above) and the smoothed staircase for every other grain
boundary node; voxel maps and geodesic labels use the smoothed staircase only (their grains are not
cells of the metric). The fitter treats both kinds the same way.

**Passthrough parts.** The grains of passthrough attributes are *fixed labels*. The faces between a
fixed-label element and any other grain are treated like the outer boundary: their normals join
the boundary surfaces of their nodes, their triangles the curved-surface patches, so nodes slide
along them and never cross them; the fixed label leaves the label set of every node that has
another label. Label repair never changes the grain of a fixed-label element, nor gives its label
to another element.

**Label repair.** Stair-step corners give nodes whose labels do not match any analytic feature
(e.g. a node with three grains where the tessellation has only a boundary between two) and single
elements sticking out of a grain. A few such elements are given to a face-neighboring grain, chosen
by how much they reduce the unfit nodes and protruding elements (previewed with the target rules),
at most one per neighborhood (two element rings) per pass, and kept only if that count really
drops; a grain never loses its last element (or goes below Cleanup.min_grain_elements).

**Fit: spring-balanced pseudo-elastic morph.** The mesh is treated as a linear elastic solid with
Young's modulus 1 (times a per-element stiffening factor) and Poisson ratio 0.3, assembled on the
input geometry. Every fitted node v is pulled toward its target t_v by a spring of stiffness
k_v = c h_v m_v (a tenth of that on triple lines and quad points). The displacement u minimizes

    1/2 u^T K u + 1/2 sum_v k_v |x_v + u_v - t_v|^2,

one linear system (K + P) u = P (t - x) solved by conjugate gradients with an algebraic multigrid
preconditioner (hypre BoomerAMG, systems mode). The springs let the elastic energy and the pull of
the targets balance: an element at a stair-step corner cannot become exactly smooth (its two faces
on the boundary would become one plane), and the balance leaves it a bounded kink instead of a
collapsed element. The rounds double c; after each solve every element's scaled Jacobian (at its
corners; times sqrt(2) for tets, so a regular tet scores 1) is checked against the floor, 0.3 for
hexes and 0.2 for tets, and its aspect ratio (the ratio of the extreme singular values of the
corner edge matrix, relative to the regular tet for tets) against the ceiling 4; an element whose
input value is already beyond a limit is held to its input value instead. If an element breaks a
limit, the springs of the nodes of the elements near the floor or the ceiling are halved and the
round is solved again; elements below 0.6 are stiffened by
(0.6 / sj)^2 (capped at 1000, spread to their neighbors), so distorted elements pass further
distortion on to their neighbors. The rounds stop when the mean distance of the fitted nodes from
their grain boundaries improves by less than 10 % (from the fourth kept round), when a round would
leave more than 1 % of the elements below floor + 0.2 beyond those of the input, or after 10
rounds; the last round that passed is kept. If already the first round fails that 1 % guard it is
retried with a quarter of the spring, up to three times, so meshes whose elements start close to
the guard still get a partial fit.

**Outer surfaces.** On a box face (normal along a coordinate axis) the normal coordinate is fixed,
so nodes slide along the face and keep that coordinate bit for bit. On a tilted flat surface, or
on a single curved surface, a node slides along it: a stiff normal spring (10^4 h) in the solve,
then an exact projection back onto the plane(s) or onto the closest point of its ring of input
boundary triangles. Nodes where a curved surface meets another surface do not move.

**Measured behavior** (test meshes, 20-50 grains): the mean distance of the grain boundary nodes
from their targets drops from about 0.4 h to about 0.21-0.24 h on hex meshes with a smallest scaled
Jacobian of about 0.4, for analytic targets (Voronoi) and smoothed ones (a noisy voxel map, geodesic
grains) alike, and to about 0.3 h on tet meshes (tets made by cutting cubes start at a scaled
Jacobian of 0.58, so the quality guard stops them sooner).

