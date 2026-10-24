# Microstructure converters

Small Python 3 scripts that turn the output of other microstructure tools into the two input
formats of the ExaConstit microstructure generator. They need numpy; the voxel-map writers also
need h5py. Run any script with `--help` for its options. Each script's module docstring
describes the source format and cites the source code or documentation it follows.

| Script | Input | Output |
|---|---|---|
| `hexrd_grains_to_seeds.py` | HEXRD `grains.out` (far-field HEDM) | seed table CSV: position `t_vec_c`, quaternion of `exp_map_c` |
| `midas_grains_to_seeds.py` | MIDAS `Grains.csv` (far-field HEDM) | seed table CSV: position `X,Y,Z`, quaternion of `O11..O33`, `radius`, `phase` |
| `spparks_to_voxels.py` | SPPARKS text site dump | microstructure file with a voxel map: spin = grain id, no orientations |
| `exaca_to_voxels.py` | ExaCA VTK `GrainID` field (+ orientation file) | microstructure file with a voxel map, optional per-grain quaternions |
| `exaconstit_microstructure_io.py` | (shared module) | microstructure file and seed-table writers/readers, quaternion helpers |

Formats written:

- **ExaConstit microstructure file** (HDF5, `format_version` 1): a grain table (`grains/id`,
  optional `quat`, `phase`, `position`, `volume`, `weight`, `shape`) and, for the grid
  converters, a voxel map stored as a Conduit Blueprint uniform mesh under `mesh/`. Used with
  `[Source] type = "voxel_map"`, `format = "exaconstit"` (with a voxel map) or `[Seeds] source =
  "file"` (grain table with positions). Written with h5py only (`write_microstructure_file`,
  `write_voxel_map`); the Conduit Python module is not needed. Specification:
  `doc/microstructure.md`, section 3.2.
- **ExaConstit seed table v1** (CSV, `[Seeds] source = "file"`, optional): `#` comments, a header
  of column names, one seed per row; real numbers with 17 significant digits. Ids start at 1:
  `hexrd_grains_to_seeds.py` writes HEXRD grain ID + 1 (HEXRD counts from 0), and
  `midas_grains_to_seeds.py` has `--id-offset` for files that count from 0.

Orientations: ExaConstit uses crystal->sample passive quaternions `w x y z` and passes them
through unchanged, apart from normalizing them to unit length. The converters only change the
parametrization (matrix or exponential map to quaternion, w >= 0). Every converter that reads
orientations requires `--conjugate {yes,no}`: you state whether the source rotation must be
inverted to match ExaConstit's convention.

Lengths: `--units-scale` multiplies every length written. The generator has its own
`units_scale`; use one or the other.

Tests: `test/microstructure/python/test_converters.py` (extensive test suite, registered as
`test_microstructure_converters` when `EXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON`).
