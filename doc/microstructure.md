# ExaConstit microstructure generator: theory notes and file formats

This document holds the background and exact file specifications behind
`src/microstructure_options.toml`, the user reference for every generator option. Read the options
file first; come here when you need the derivation behind a number, or need to write or read the
files with your own scripts.

Contents:
1. Johnson-Mehl grain count
2. Why geodesic distance supports only some metrics
3. File formats: inputs (seed table, voxel map) and outputs (mesh, grain file, orientation file,
   region map, grain table, adjacency table)

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

## 2. Why geodesic distance supports only some metrics

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

It does not cover:
- **laguerre**, closest = d^2 - w_i. A heavy seed can own points that another seed's front reaches
  first; the single front discards it there, and its grain comes out too small or empty. Exact
  results need one distance field per seed (or several candidate seeds carried per vertex), so the
  cost grows with the number of seeds.
- **quadric**, closest = (x - s)^T A (x - s) - w_i. Distance depends on direction, so the front
  must move with a direction-dependent speed (an anisotropic eikonal solver). With one global
  tensor and no weights, a single anisotropic front would suffice; per-seed tensors need per-seed
  fields as for laguerre. Columnar and lamellar tensors are degenerate (zero speed limit along a
  direction) and need separate treatment.

Both extensions are Stage 2 action items (`agent_microstructure/PLAN.md`).

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

### 3.1 Seed table ("ExaConstit seed table v1", CSV text)

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
| `id` | no | integer, unique, >= 0 (default: row index, 0-based) |
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

### 3.2 Voxel map ("ExaConstit voxel map v2", HDF5)

Reader: `src/microstructure/io/voxel_map_io.hpp`. Writer for scripts:
`write_voxel_map` in `scripts/microstructure/exaconstit_microstructure_io.py`. ("v2" because
ExaConstit's auto-mesh grain file plus orientation file counts as v1.) This format is scheduled to
be replaced in Stage 2 by one format covering seeds, regular grids and arbitrary point sets
(`agent_microstructure/DATA_FORMAT_PROPOSAL.md`).

| Path | Kind | Stored type | Shape | Meaning |
|---|---|---|---|---|
| `/grain_ids` | dataset | any HDF5 integer type (read as int64) | (nz, ny, nx) or (nz, ny, nx, 1) | grain id of every voxel; <= 0 = unassigned |
| `/origin` | dataset (not an attribute) | any float type (read as double) | (3,) | lower corner of voxel (0,0,0), x y z |
| `/spacing` | dataset | any float type | (3,) | voxel edge lengths dx dy dz, > 0 |
| `/features/id` | dataset, optional | any integer type | (nf,) | grain ids of the per-grain table, unique, >= 1 |
| `/features/quat` | dataset, optional | any float type | (nf, 4) | w x y z per grain, crystal-to-sample |
| `/features/phase` | dataset, optional | any integer type | (nf,) | phase per grain, >= 1 |

Rules:
- **Memory order:** x varies fastest, then y, then z. Voxel (i, j, k), 0-based with i along x, is
  entry i + nx (j + ny k) of the flattened array, i.e. `grain_ids[k, j, i]` in numpy. Data in
  another order must be transposed before writing.
- Voxel (i, j, k) is the box from origin + (i, j, k) * spacing to origin + (i+1, j+1, k+1) * spacing.
- HDF5 converts stored types to the reader's types, so int32/int64 and float32/float64 all work.
- Every grain id that the mesh actually uses must have a row in the feature table when
  orientations or phases are read; unused rows are allowed. Ids need not be contiguous.
- Chunked and compressed datasets are fine (HDF5 handles them; gzip is built into HDF5, other
  filters need the matching HDF5 plugin). Ranks read cubes of `tile_size`^3 voxels, so chunks of a
  similar size (e.g. 32^3) read efficiently. The serial HDF5 library is enough: ranks open the
  file read-only, each reading only its own cubes.
- A `format` attribute is written by the Python writer but not checked by the reader.

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

### 3.3 Output files (written by the tool; `<b>` = `<directory>/<basename>`)

| File | Format |
|---|---|
| `<b>.mesh` / `<b>.mesh.gz` | MFEM serial mesh ("MFEM mesh v1.0"), element attribute = grain id 1..N, boundary attributes kept from the input mesh, gzip when compression > 0, coordinates at 17 digits |
| `<b>.mesh.000000`, ... | MFEM parallel mesh ("MFEM mesh v1.2"), one file per part, gzip-compressed when compression > 0 (names unchanged; MFEM detects gzip from the first bytes) |
| `<b>.grains` | text, one grain id per line, for auto box meshes: line e = element e in ExaConstit's auto-mesh order (x fastest, then y, then z); read by ExaConstit's `grain_file` |
| `<b>.ori` | text, one `w x y z` line per grain (line g = grain g), 17 significant digits, no header; read by `orientation_file` |
| `<b>.regions` | text, one `grain region` line per grain, no header (only written with 2+ regions); read by `region_mapping_file` |
| `<b>_grains.csv` | CSV with header: grain_id, qw, qx, qy, qz, then region, source_id, num_elements, volume, cx, cy, cz, num_neighbors, target_volume, phase, filler, feature_id (optional columns only when they carry information) |
| `<b>_adjacency.csv` | CSV with header `grain_a,grain_b,shared_area,num_faces`, one row per pair of grains sharing element faces (grain_a < grain_b) |
| `<b>_paraview/` | ParaView collection (`.pvd`), cell field `grain_id` |

Passthrough parts ([Mesh] passthrough_attributes) become the last grain ids, N+1, N+2, ..., so in
ParaView "grain_id > N" selects them.
