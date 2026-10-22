# Microstructure converters

Small Python 3 scripts that turn the output of other microstructure tools into the two input
formats of the ExaConstit microstructure generator. They need numpy; the voxel-map writers also
need h5py. Run any script with `--help` for its options. Each script's module docstring
describes the source format and cites the source code or documentation it follows.

| Script | Input | Output |
|---|---|---|
| `hexrd_grains_to_seeds.py` | HEXRD `grains.out` (far-field HEDM) | seed table CSV: position `t_vec_c`, quaternion of `exp_map_c` |
| `midas_grains_to_seeds.py` | MIDAS `Grains.csv` (far-field HEDM) | seed table CSV: position `X,Y,Z`, quaternion of `O11..O33`, `radius`, `phase` |
| `spparks_to_voxels.py` | SPPARKS text site dump | voxel map HDF5: spin = grain id, no orientations |
| `exaca_to_voxels.py` | ExaCA VTK `GrainID` field (+ orientation file) | voxel map HDF5, optional per-grain quaternions |
| `exaconstit_microstructure_io.py` | (shared module) | seed-table and voxel-map writers/readers, quaternion helpers |

Formats written:

- **ExaConstit seed table v1** (CSV, `[Seeds] source = "file"`): `#` comments, a header of
  column names, one seed per row; real numbers with 17 significant digits.
- **ExaConstit voxel map v1** (HDF5, `[Source] type = "voxel_map"`, `format = "exaconstit"`):
  `/grain_ids` (nz, ny, nx), `/origin` (lower corner of voxel (0,0,0)), `/spacing`, optional
  `/features/{id,quat,phase}`.

Orientations: ExaConstit uses crystal->sample passive quaternions `w x y z` and passes them
through unchanged, apart from normalizing them to unit length. The converters only change the
parametrization (matrix or exponential map to quaternion, w >= 0). Every converter that reads
orientations requires `--conjugate {yes,no}`: you state whether the source rotation must be
inverted to match ExaConstit's convention.

Lengths: `--units-scale` multiplies every length written. The generator has its own
`units_scale`; use one or the other.

Tests: `test/microstructure/python/test_converters.py` (extensive test suite, registered as
`test_microstructure_converters` when `EXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON`).
