# xtal_light_up

Fast lattice strain / "light-up" post-processing for ExaConstit results. This
directory contains a Rust library with Python bindings, built with
[PyO3](https://pyo3.rs). It computes the quantities a powder diffraction
("light-up") experiment measures, from ExaConstit's visualization output:
- lattice strains;
- directional stiffness;
- Taylor factors;
- equivalent plastic strain rates;
- in-fiber volumes.

The calculations mirror ExaConstit's in-situ light-up
(`LightUp` in `src/postprocessing/mechanics_lightup.cpp`), so both give the
same results for every crystal system ExaConstit supports. This module is useful
when you didn't turn on the in-situ light-up for a run, or want to try other
HKLs, sample directions or tolerances afterwards.

## Contents

| Path | What it is |
|------|------------|
| `src/lib.rs` | Crate root |
| `src/pyxtal_light_up/mod.rs` | The Python-facing functions (see [API](#api)) |
| `src/pyxtal_light_up/lattice.rs` | Crystal systems: symmetry operators, lattice parameters, reciprocal lattice |
| `src/pyxtal_light_up/math.rs` | Small fixed-size linear algebra helpers |
| `light_up_py/fiber_calcs_rank.py` | Script that processes an ExaConstit ADIOS2 output end to end |
| `pyxtallu.py` | Small example that runs each function and compares the in-fiber check with HEXRD |
| `setup.py`, `xtal_light_up/` | Python packaging (setuptools-rust) |

## Building and installing

Requirements:
- **Rust 1.88 or newer:** for example from <https://rustup.rs>.
- **Python 3.9 or newer, with NumPy.**
- **Build tools:** `pip install setuptools-rust setuptools wheel`.

From this directory:

```bash
pip install .                      # regular install
# or
python setup.py develop --user     # editable install into user site-packages
```

Then:

```python
import xtal_light_up.xtal_light_up as xlup
help(xlup)                         # module and per-function documentation
```

To run `light_up_py/fiber_calcs_rank.py` you'll also need the ADIOS2 Python
bindings (`import adios2`, with `adios2.FileReader` available). `pyxtallu.py`
also needs [HEXRD](https://github.com/HEXRD/hexrd).

## What's needed from ExaConstit

The functions work on per-element data from an ExaConstit run that uses an
ExaCMech crystal plasticity model. `fiber_calcs_rank.py` reads that data from the
ADIOS2 visualization output, which requires:

1. **ADIOS2 output:** set `adios2 = true` under `[Visualizations]` in the options
   file. MFEM must be built with ADIOS2 (`MFEM_USE_ADIOS2`). The `.bp` output is
   written to the visualization output location
   (`[output_directory]/[basename]/[floc]`, see `Visualizations.floc` in
   `src/options.toml`).
2. **The fields below projected:** `[PostProcessing.projections]` does this by
   default with `auto_enable_compatible = true`. Otherwise, list them in
   `enabled_projections`.

   | `enabled_projections` | ADIOS2 variable | Shape per element | Used for |
   |-----------------------|-----------------|-------------------|----------|
   | `"volume"` | `Element Volumes` | scalar | volume weighting |
   | `"xtal_ori"` | `Crystal Orientations` | quaternion `[w, x, y, z]` | which elements are in each fiber |
   | `"elastic_strain"` | `Elastic Strains` | 6 (sample frame) | lattice strains, directional stiffness |
   | `"cauchy"` | `Cauchy Stress` | 6 | directional stiffness |
   | `"shear_rate"` | `Shearing Rate` | number of slip systems | Taylor factor |
   | `"dpeff"` | `Equivalent Plastic Strain Rate` | scalar | Taylor factor, equivalent plastic strain rate |

3. **The number of MPI ranks the run used** (`-nranks` for the script). The
   ADIOS2 output is written as one block per rank, and every block is read.

The equivalent plastic strain rate comes straight from ExaCMech. It isn't
recomputed from the slip rates, so the Taylor factor works for any slip system
set. The elastic strains are expected in the sample frame, which is how
ExaConstit v0.9+ writes them. `strain_lattice2sample` converts lattice frame
strains from older runs.

## Crystal systems

`calc_within_fibers` takes the same crystal system names and lattice parameters
as ExaConstit's in-situ light-up options (`laue_type` and `lattice_params`). Any
angles are in radians.

| `lattice_type` | Symmetry operations | `lattice_params` |
|----------------|---------------------|------------------|
| `cubic` (default) | 24 | a |
| `hexagonal` | 12 | a, c |
| `trigonal` | 6 | a, c |
| `rhombohedral` | 6 | a, alpha |
| `tetragonal` | 8 | a, c |
| `orthorhombic` | 4 | a, b, c |
| `monoclinic` | 2 | a, b, c, beta |
| `triclinic` | 1 | a, b, c, alpha, beta, gamma |

HKLs are given as 3-index Miller indices, also for hexagonal and trigonal
crystals. For example, (10-10) is `[1, 0, 0]`.

## API

All arrays must be C-contiguous NumPy arrays: `float64`, or `bool` for the
in-fiber flags. Arrays are laid out as `(steps, elements, ...)` per step and
element. Symmetric tensors are 6-vectors `[11, 22, 33, 23, 13, 12]` of tensor
(not engineering shear) components. Output arguments are filled in place; the
"added to" outputs accumulate across calls. Call `help()` on a function for the
full details.

| Function | Computes |
|----------|----------|
| `calc_within_fibers(lattice_orientations, sample_dir, hkls, lattice_params, distance_tolerance_rad, in_fibers, lattice_type="cubic")` | `(nhkl, steps, elems)` flags for which elements are in each HKL fiber |
| `calc_lattice_strains(sample_strains, sample_dir, elem_vols, in_fibers, lattice_strains, lattice_volumes, per_rank_update)` | Volume weighted lattice strain and in-fiber volume per HKL and step |
| `calc_directional_stiffness_lattice_fiber(cauchy_stress, sample_strain, directional_stiffness, elem_vols, in_fibers, per_rank_update)` | Volume weighted 33 stress / 33 strain per HKL and step |
| `calc_taylor_factors_lattice_fiber(shear_rates, eq_pl_strain_rate, taylor_factors, lattice_eq_pl_strain_rate, elem_vols, in_fibers, per_rank_update)` | Volume weighted Taylor factor and equivalent plastic strain rate per HKL and step |
| `calc_directional_stiffness(cauchy_stress, sample_strain, directional_stiffness)` | Per-element 33 stress / 33 strain |
| `calc_taylor_factors(shear_rates, eq_pl_strain_rate, taylor_factors)` | Per-element Taylor factor |
| `strain_lattice2sample(lattice_orientations, strains)` | Rotates lattice frame strains to the sample frame in place |

**`per_rank_update`:**
- **`True`:** the fiber functions add raw volume weighted sums to their outputs.
  This lets you accumulate the results from every rank's block, then divide by
  `lattice_volumes` once at the end. `fiber_calcs_rank.py` does this.
- **`False`:** each call adds the volume average of just the elements passed in.

### Quick example

```python
import numpy as np
import xtal_light_up.xtal_light_up as xlup

# quats: (steps, elems, 4), strains: (steps, elems, 6), vols: (steps, elems)
hkls = np.array([[1., 0., 0.], [0., 0., 2.], [1., 0., 1.]])   # Ti (10-10), (0002), (10-11)
s_dir = np.array([0.0, 0.0, 1.0])
in_fibers = np.zeros((hkls.shape[0], steps, elems), dtype=bool)
xlup.calc_within_fibers(quats, s_dir, hkls, [2.95, 4.68], np.deg2rad(5.0), in_fibers,
                        lattice_type="hexagonal")

lattice_strains = np.zeros((hkls.shape[0], steps))
lattice_vols = np.zeros_like(lattice_strains)
xlup.calc_lattice_strains(strains, s_dir, vols, in_fibers, lattice_strains, lattice_vols, True)
lattice_strains /= lattice_vols   # volume average per HKL and step
```

## Development

The Rust code should stay free of `cargo clippy` warnings and be formatted with
`cargo fmt`. On macOS, building or testing outside of setuptools-rust needs the
Python symbols left for runtime lookup:

```bash
export PYO3_PYTHON=$(which python3)
export RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"   # macOS only
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --lib       # symmetry group closure, reciprocal lattice, parameter checks
```

`src/pyxtal_light_up/lattice.rs` has to stay in sync with
`GetSymmetryGroups` and `LatticeTypeGeneral::ComputeLatticeBParam` in ExaConstit's
`src/postprocessing/mechanics_lightup.cpp`. If one changes, change the other.
