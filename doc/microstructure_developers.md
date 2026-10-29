# Microstructure generator: developer guide

This guide is for developers who change or extend ExaConstit's microstructure generator
(`src/microstructure/`) or its integration with the mechanics code. It complements the general
[developer's guide](../developers_guide.md), whose code standards apply here too. What the
generator does for users, with pictures and the options behind each feature, is in the
[user guide](microstructure.md); every option is documented in
[`src/microstructure_options.toml`](../src/microstructure_options.toml).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Building](#building)
3. [Architecture](#architecture)
4. [Source directory structure](#source-directory-structure)
5. [Rules every change follows](#rules-every-change-follows)
6. [Algorithms in more depth](#algorithms-in-more-depth)
7. [Test suite](#test-suite)
8. [Common changes](#common-changes)

## Prerequisites
- **Knowledge**: MFEM's parallel meshes (`ParMesh`, element attributes, `Nodes` for curved
  meshes), MPI collectives, and the basics of Voronoi-type tessellations; the user guide explains
  the latter.
- **Libraries**: MFEM, MPI and Axom (spin BVHs, primal geometry). Reading and writing the HDF5
  microstructure file needs Axom built with Conduit and HDF5; without them every HDF5 reader
  throws a clear error on every rank and the rest of the generator works.
- **Python** (converters and their tests): numpy, and h5py for the microstructure file.

## Building

| CMake option | Default | Effect |
|---|---|---|
| `ENABLE_MICROSTRUCTURE` | `ON` | builds the generator libraries, the tool and the `[Mesh.Microstructure]` hook; needs Axom (turned off with a message when Axom is not found) |
| `MICROSTRUCTURE_STRICT_FP` | `OFF` | compiles the generator kernels with `-ffp-contract=off`, removing one source of CPU/GPU round-off differences in grain labels (not a guarantee of bitwise agreement) |
| `EXACONSTIT_ENABLE_EXTENSIVE_TESTS` | `OFF` | registers the microstructure tests (and the other extensive tests) |

Targets (`src/microstructure/CMakeLists.txt`): one static library per module, so a change in one
module never recompiles another. Each module collects its own sources by glob, so a new file
needs a re-run of CMake but no edit of the CMake file.

| Target | Contents |
|---|---|
| `ms_core` | header-only: the top-level headers (types, exec dispatch, options struct) and `util/` |
| `ms_options` | `microstructure_options.*` |
| `ms_mesh`, `ms_io`, `ms_geometry`, `ms_seeding`, `ms_morphology`, `ms_conformal` | the subdirectories of the same name |
| `ms_labeling` | `labeling/`, `orientation/`, `regions/` |
| `exaconstit_microstructure` | the driver and report (`microstructure_driver.*`, `microstructure_report.*`) |
| `microstructure_generator` | the standalone tool (`tools/microstructure_generator.cpp`) |

The libraries need only MFEM, Axom and MPI; they do not depend on the mechanics library. The
mechanics side links `exaconstit_microstructure` only when the generator is built
(`src/CMakeLists.txt`).

## Architecture

**One entry point.** `GenerateMicrostructure` (`microstructure_driver.hpp`) takes validated
options and a distributed mesh and returns a `MicrostructureResult`: the grain of every local
element, one orientation, phase and region per grain, and a report. Both front ends call it:

- the standalone tool parses an options file and command-line flags
  (`microstructure_options.cpp`), builds or reads the mesh, calls the driver and writes files
  (`io/output_writer.*`, `microstructure_report.*`);
- the mechanics hook (`src/sim_state/microstructure_hook.*`) turns the `[Mesh.Microstructure]`
  table of an ExaConstit deck into the same options (`ParseMicrostructureHook`), runs the driver
  on the simulation's own mesh (`RunMicrostructureHook`) and hands the grains to
  `SimulationState`. `src/options/option_microstructure_hook.hpp` is the narrow interface the
  options parser uses, so the options code does not include the generator.

**The pipeline and where each step lives.**

| Step | What | Code |
|---|---|---|
| 1–2 | refuse unimplemented options; preflight the mesh; build the element mask of the active part; one point locator shared by every step | `GenerateMicrostructure`, `mesh/preflight`, `mesh/attribute_domain`, `mesh/element_mask.hpp`, `seeding/material_point_locator` |
| 3 | seeds: generated, Johnson-Mehl, or a seed table with region of interest and filler | `seeding/seed_generators`, `seeding/johnson_mehl`, `io/seed_table_io`, `io/microstructure_file_io`, `seeding/roi_filler` |
| 4 | size targets per seed | `morphology/size_targets` |
| 5 | Lloyd relaxation and the Laguerre volume fit | `morphology/lloyd`, `morphology/weight_fit`, `labeling/cell_measures` |
| 6–8 | element labels in the chosen distance mode | `labeling/tessellation_labeler`, `labeling/quadrature_vote`, `labeling/connected_labeler`, `labeling/geodesic_*`, `labeling/vertex_field` |
| 3–8 (voxel maps) | voxel labels instead of seeds | `io/voxel_map_io`, `labeling/voxel_labeler` |
| 9 | fragment clean-up and small-grain merging | `labeling/fragment_cleanup`, `mesh/mesh_adjacency` |
| 10 | orientations | `orientation/` |
| 11–12 | phases and regions | `regions/region_assigner` |
| 13 | passthrough parts appended as grains | `mesh/attribute_domain` |
| 14 | conformal grain boundaries | `conformal/` |
| report | grain statistics, tables and log text | `labeling/grain_statistics`, `microstructure_report` |

`LabelFromSeeds` and `LabelFromVoxels` in `microstructure_driver.cpp` carry steps 3–9 for the two
grain sources, and `LabelByDistanceMode` steps 6–8. Each function's comment in the source lists
its numbered steps.

## Source directory structure

### Top level (`src/microstructure/`)
- **`microstructure_driver.hpp/cpp`**: `GenerateMicrostructure` and the step functions above
- **`microstructure_options.hpp/cpp`**: the options structs with their defaults, TOML parsing,
  `validate()` with every combination rule, and command-line overrides
- **`microstructure_report.hpp/cpp`**: record tables and log text of a finished run
- **`microstructure_types.hpp`**: shared data contracts (`SeedSet`, `MicrostructureResult`, ...)
- **`exec_dispatch.hpp`**: maps the `exec` option ("cpu", "openmp", "gpu") to Axom execution
  spaces and MFEM devices

### `seeding/`
- **`seed_generators`**: random seeds inside the material (rejection or element-volume-weighted
  sampling), minimum spacing (random sequential adsorption), surface clearance, regular lattices
- **`min_distance_grid.hpp`**: the hash grid behind the minimum spacing
- **`johnson_mehl`**: nucleation and growth as a space-time Poisson process
- **`roi_filler`**: random filler seeds outside a measured region of interest
- **`material_point_locator`**: "is this point inside the material?" (BVH over element boxes,
  exact point-in-element test)
- **`seed_element_locator`**: partition-independent searches: the element containing a seed, the
  nearest element centroid, the nearest of any point set (`NearestPoints`)
- **`boundary_clearance`**: distance of points to the outer surface

### `geometry/`
- **`metric_functors.hpp`**: the four metrics as plain-data functors usable in GPU kernels
- **`seed_index`**: the exact nearest-seed search (Axom BVH with metric lower bounds)
- **`anisotropy_tensors.hpp`**: columnar and lamellar tensors
- **`periodic_images`**: seed copies for periodic boxes

### `labeling/`
- **`tessellation_labeler`**, **`quadrature_vote`**: Euclidean labels (centroid or vote)
- **`connected_labeler`**, **`connected_limits.hpp`**: the connected distance mode
- **`vertex_field`**: distance and seed label on every field node, shared by the geodesic solvers
- **`geodesic_graph`**: shortest paths along element edges and diagonals
- **`geodesic_fim`**, **`fim_scan.hpp`**, **`eikonal_local_solver.hpp`**: the Fast Iterative
  Method for the eikonal equation
- **`geodesic_weighted`**: multi-seed fronts for Laguerre, weighted quadric and per-seed tensors
- **`geodesic_metric`**, **`geodesic_common.hpp`**, **`geodesic_limits.hpp`**: shared types,
  the path metric and limits
- **`voxel_labeler`**: element labels from a voxel map
- **`fragment_cleanup`**: disconnected grains and small grains
- **`cell_measures`**: per-grain volumes and centroids with reproducible sums
- **`grain_statistics`**: per-grain statistics and grain adjacency for the report

### `morphology/`
- **`size_targets`**: equal or log-normal target volumes
- **`weight_fit`**: Laguerre weights that give every grain its target volume
- **`lloyd`**: Lloyd relaxation, optionally with the volume fit after every step
- **`sample_labels`**: the labeling step shared by both iterations

### `conformal/`
- **`node_labels`**: grains and outer surfaces at every node
- **`node_targets`**, **`manifold_projector.hpp`**: analytic targets and their validation
- **`smoothed_targets`**: smoothed-staircase targets where no analytic boundary exists
- **`label_repair`**: element flips that remove spots the fit cannot handle
- **`adjacency_check`**: mesh grain adjacency compared with the analytic one
- **`boundary_projector`**: curved-surface patches, closest points, plane projection
- **`conformal_fitter`**: the spring-balanced pseudo-elastic morph
- **`quality`**: scaled Jacobian, aspect ratio and mean ratio

### `mesh/`
- **`element_key`**, **`vertex_coordinates`**: partition-independent element identity and
  vertex positions
- **`mesh_adjacency`**: face adjacency including neighbors on other ranks
- **`mesh_sampler`**: element volumes, centroids and quadrature points
- **`element_mask.hpp`**, **`attribute_domain`**: the active part of a mesh
- **`serial_order`**, **`straight_sided`**, **`preflight`**: serial numbering, straight-sided
  detection and mesh checks

### `io/`
- **`microstructure_file_io`**: the ExaConstit microstructure file (HDF5 through Conduit)
- **`voxel_map_io`**: tiled voxel-map reading (ExaConstit and DREAM.3D files)
- **`seed_table_io`**: CSV seed tables
- **`output_writer`**: mesh, grain, orientation and region files for ExaConstit

### `orientation/`, `regions/`, `util/`, `tools/`
- **`orientation_assigner`**, **`random_orientation.hpp`**: orientations per grain
- **`region_assigner`**: phases to regions, synthetic phase fractions
- **`counter_rng.hpp`**, **`random_distributions.hpp`**: Philox counter-based random numbers
- **`fixed_point.hpp`**: reproducible global sums
- **`collective_errors.hpp`**: errors raised on every rank together
- **`microstructure_generator.cpp`**: the standalone tool

### Outside `src/microstructure/`
- **`src/sim_state/microstructure_hook.hpp/cpp`**: the `[Mesh.Microstructure]` hook
- **`src/options/option_microstructure_hook.hpp`**: the options parser's view of the hook
- **`scripts/microstructure/`**: Python converters (HEXRD, MIDAS, ExaCA, SPPARKS) and the
  microstructure-file writer, with their own README
- **`test/data/microstructure/lattice/`**: the strut-lattice test meshes and
  `make_lattice_meshes.py`, which builds them (tet or hex); its node optimization uses the OpenMP
  energy kernel `lattice_energy.cpp` (`--engine native`, the default) or JAX (`--engine jax`)
- **`doc/microstructure_figures/`**: the pictures of the user guide with everything that makes
  them: `sources/` holds the options file of every picture, `make_voxel_example.py` the data and
  mesh of the voxel example, and `render_grains.py` the ParaView script that renders a run
  (`pvbatch --force-offscreen-rendering render_grains.py RUN.pvd OUT.png --cutaway`), and
  `finish_figures.py` trims a render (or joins two side by side) into the JPEG the guide shows,
  and `voxels_to_vti.py` turns the voxel map of a microstructure file into a VTK image for
  rendering. `--cut-exact` cuts with flat planes, which suits meshes whose nodes were moved by
  the conformal fit.
  When an option changes, re-run the affected options files so the guide stays truthful.

## Rules every change follows

**Partition independence.** The microstructure depends only on the options and the mesh, never
on the number of MPI ranks or on the partition. The tests run every MPI-dependent function on
several rank counts and compare the results bit for bit. The building blocks:

- every random number comes from the counter-based generator (`util/counter_rng.hpp`), keyed by
  `rng_seed`, a purpose and the identity of the object it belongs to (seed index, element key),
  never from a stream consumed in rank order;
- elements are identified by `ElementKey` and vertices by `VertexKey`, 64-bit hashes of their
  coordinates, and every tie is broken by the smallest key;
- vertex positions come from `VertexCoordinates` (the owner's value broadcast to every copy),
  never from `Mesh::GetVertex`, which can differ between ranks for meshes with Nodes;
- global sums that decide anything go through `util/fixed_point.hpp`, so they are identical in
  any order;
- seeds and per-seed data are replicated on every rank.

The one documented exception is the geodesic mode with weights or per-seed tensors when the
per-node candidate lists overflow (section 5); the log reports it.

**Collectives and errors.** A function that communicates is collective and says so in its
comment. A check that can fail on some ranks only must fail on all of them:
`util/collective_errors.hpp` (`ThrowIfAnyRank`, `AnyRank`) reduces the condition first, so no
rank is left waiting in a later collective.

**Host and device.** The labeling kernels run through `mfem::forall` and Axom execution spaces
selected by `exec_dispatch.hpp`; the metric functors are plain data and `MFEM_HOST_DEVICE`. Code
marked host only reads data with `HostRead()`.

**Options stay in sync.** A new or changed option touches, in the same commit:
`microstructure_options.hpp` (field, default, documentation), `microstructure_options.cpp`
(parsing, `validate()` rules and messages), `src/microstructure_options.toml` (the user
reference; active keys at their defaults), `test/microstructure/test_options.cpp`
(`OptionsTemplateDocumentsTheDefaults` fails when a default changes or a key is missing), the
user guide when the feature is user-visible, and the `[Mesh.Microstructure]` comments of
`src/options.toml` for hook-side changes.

**Unsupported on purpose.** `[Orientations] source = "list"` is parsed and refused with a message;
`deterministic` is accepted but has no effect (results never depend on the rank count, and
bitwise CPU/GPU agreement is not promised); a time-dependent Johnson-Mehl rate
(`JohnsonMehlParameters::rate_function`) exists in the library but not as an option.

**Combination rules.** Every "cannot be combined with" rule of the options reference lives in
`MicrostructureOptions::validate()` (`microstructure_options.cpp`), which prints every problem it
finds before returning; `test_options.cpp` checks the rules. The mechanics hook adds its own
refusals in `ParseMicrostructureHook`, checked by `test/test_microstructure_hook.cpp`.

**File formats.** The microstructure file is read by `io/microstructure_file_io` and
`io/voxel_map_io` and written by `scripts/microstructure/exaconstit_microstructure_io.py` and
`io/output_writer`; the readers and the Python writer must agree. `test_converters.py` checks the
layout the writer produces, and the data-source end-to-end chain converts a SPPARKS dump with the
Python writer and runs the generator on the result. Adding an optional dataset never
changes `format_version`; change it only when the meaning of existing data changes, and then
update, in one commit, the reader, the writers, the format table of the user guide (section 13),
the converters and their tests.

**Restarts.** A restarted simulation rebuilds the mesh, grain numbers and state variables
(orientations included) from its checkpoint, so `RunMicrostructureHook` is skipped and only the
deck is checked (`src/sim_state/microstructure_hook.cpp`). New generator output that the
simulation keeps after start-up must therefore live in the checkpointed state; see "Keeping New
State Restartable" in the developer's guide.

**GPUs.** The labeling kernels follow `exec` ("gpu" needs a CUDA or HIP build; inside a simulation
they follow `[Solvers] rtmodel`). Labels on a GPU can differ from a CPU run where an element's
grain is decided by round-off; `MICROSTRUCTURE_STRICT_FP` removes the compiler's fused
multiply-adds as one source of such differences. The test suite runs on the host.

**Documentation and formatting.** Every class, function, member and test carries a Doxygen
comment stating units, whether it is collective, where it runs (host, device) and what it
throws. Format with the repository `.clang-format` (developer's guide, "Code Formating"),
covering `src/` and `test/`.

## Algorithms in more depth

The user guide explains each method at the level a user needs. This section records the details
a developer needs to change them.

### Johnson-Mehl nucleation

Candidates are a homogeneous space-time Poisson process of rate $I$ in the sampling box: waiting
times are exponential, positions uniform. A candidate at $(\mathbf{x}, t)$ is discarded when an
earlier grain $j$ already covers it, $\lVert \mathbf{x}-\mathbf{s}_j \rVert < v\,(t - t_j)$.
Generation stops at `end_time`, at `num_nuclei`, or when a guaranteed upper bound of the
completion time is passed. Accepted seeds carry the additive weight $a_i = -v\,t_i$. The grain
density $0.896\,(I/v)^{3/4}$ is `JohnsonMehlGrainDensity` in `seeding/johnson_mehl.hpp`; the
untransformed fraction drops below 5 % at $t = 1.30\,(I v^3)^{-1/4}$.

### Geodesic distance

The solvers grow one front from all seeds at once; every field node (a vertex, or every nodal point
of a curved mesh) keeps the first arrival and its seed. The front starts at seed $i$ with $-a_i$, so
the result is $T(\mathbf{x}) = \min_i \left( d_\text{path}(\mathbf{x}, \mathbf{s}_i) - a_i \right)$.
This is exact when "closest" is a path length plus a constant per seed (Voronoi, additive), because
that minimum solves one eikonal problem.

**One quadric tensor.** Paths are measured with the length
$\sqrt{\mathbf{r}^{\mathsf T}\mathbf{A}\mathbf{r}}$. With
$\mathbf{A} = \mathbf{L}\mathbf{L}^{\mathsf T}$ (Cholesky) and the map
$\mathbf{y} = \mathbf{L}^{\mathsf T}\mathbf{x}$, the $\mathbf{A}$-length of a path is the plain
length of its image, and the linear map keeps elements as elements, so the same solver runs on
mapped coordinates (`labeling/geodesic_metric.hpp`). On a convex part the labels equal the
straight-line quadric labels.

**Columnar and lamellar.** Their tensors ($\mathbf{I} - \mathbf{a}\mathbf{a}^{\mathsf T}$ and
$\mathbf{n}\mathbf{n}^{\mathsf T}$) have a zero direction, which would collapse the mapped mesh.
The geodesic mode uses regularized lengths in which columns are $k$ times longer than wide:

$$
\text{columnar: } \lVert\mathbf{r}\rVert^2 - \left(1 - k^{-2}\right)(\mathbf{a}\cdot\mathbf{r})^2,
\qquad
\text{lamellar: } (\mathbf{n}\cdot\mathbf{r})^2 + k^{-2}\,\lVert\mathbf{r} - (\mathbf{n}\cdot\mathbf{r})\,\mathbf{n}\rVert^2 .
$$

With finite $k$ a column can end: between columns $i$ and $j$ at distance $d_{ij}$, a point whose
height differs from seed $i$'s by $\Delta z$ moves the boundary by about
$(\Delta z / k)^2 / (2 d_{ij})$. With $\Delta z \le L$ (the part length along the axis) and
$d_{ij} \approx w$ (the column width), the shift is at most about $w\,(L / (k w))^2 / 2$; the
automatic $k = 10 L / w$ keeps it below $w/200$. Here $w = \sqrt{L_u L_v / N}$, the width if the
$N$ seeds shared the bounding-box cross-section; for layers, the in-plane diameter over the
thickness $L_n / N$ is used. $k$ is rounded up to two significant digits, at least 10 and at most
$10^4$ (beyond that the mapped elements become slivers the local solver starts to discard). The
extents come from exact min/max reductions, so $k$ does not depend on the rank count.

**Laguerre and weighted quadric** (closest $= d^2 - w_i$, `labeling/geodesic_weighted.hpp`). A
heavy seed can own points another front reaches first, so every node keeps a short list of
(seed, path length) pairs and the eikonal update runs per seed. Seed $j$ dominates seed $i$ at a
node when $d_j \le d_i$ and $d_j^2 - w_j \le d_i^2 - w_i$; then $j$ wins at every point reached
from $i$ through that node, since for any remaining path length $L$

$$
(d_j + L)^2 - w_j - (d_i + L)^2 + w_i = (d_j^2 - w_j) - (d_i^2 - w_i) + 2L\,(d_j - d_i) \le 0 ,
$$

so $i$ is dropped. This is exact. A cap (`geodesic.max_seeds_per_node`, default 16) bounds the
memory; when more candidates remain the worst are cut and the log reports it. Only then can the
result depend on the rank count.

**Per-seed tensors.** Every seed measures paths with its own tensor through its own map. The drop
rule needs the remaining path to be no longer for $j$ than for $i$, which holds when
$\mathbf{A}_j \preceq \mathbf{A}_i$; otherwise both are kept, so the lists fill toward the cap.
With at most `max_seeds_per_node` seeds in total nothing can be cut.

### Conformal grain boundaries

The step runs last (driver step 14) and moves the nodes; elements, connectivity and element type
stay the same.

**Node labels** (`node_labels`). Every node gets the sorted set of grains around it and the outer
surfaces it lies on: boundary faces are grouped into surfaces by their normals (normals within 30°
form one surface), each surface flat (one exact normal) or curved. A node with more than three
constraints (labels minus one, plus surfaces) is over-constrained and not fitted.

**Analytic targets** (`node_targets`, `manifold_projector.hpp`). For labels $a, b$ (and $c, d$),
the boundary is where the metric values agree, $f_a(\mathbf{y}) = f_b(\mathbf{y})$. The target
is the closest such point, together with the planes of flat outer surfaces, from a constrained
Gauss-Newton projection

$$
\mathbf{y} \leftarrow \mathbf{x} - \mathbf{J}^{\mathsf T}\left(\mathbf{J}\mathbf{J}^{\mathsf T}\right)^{-1}\left(\mathbf{J}(\mathbf{x}-\mathbf{y}) + \boldsymbol{\phi}(\mathbf{y})\right),
$$

with $\boldsymbol{\phi}$ the constraint values and $\mathbf{J}$ their gradients. On a curved outer
surface the projection alternates with the closest point on a patch of boundary triangles around
the node (`boundary_projector`), grown ring by ring until it holds every point within the move
limit (at most three rings; beyond the first ring only triangles of the node's own surface). On a
rim where a curved surface meets another surface the node has a patch per surface. A target is
kept only if the constraints are met to a tolerance, it is within about one element size of the
node, and no other seed is closer there; otherwise the node is held.

**Smoothed targets** (`smoothed_targets`). Voxel, geodesic, connected-mode and cleaned-up grains
have no analytic boundary. Nodes with two or three labels average over neighbors along element
edges that carry all their labels and lie on all their outer surfaces, alternating the weights
$0.5$ and $-0.53$ (Taubin's $\lambda|\mu$ smoothing, which removes stair steps without
shrinking), 20 pairs, with the node put back on its outer surfaces after every step and the final
move limited to two element sizes. Quadruple points and over-constrained nodes anchor the
network. Neighbor lists of shared nodes are merged over ranks and sorted by input position, so
the result is bitwise identical on any partition. Tessellation grains in the Euclidean and
connected modes use analytic targets where valid and smoothed ones elsewhere; voxel and geodesic
grains use smoothed targets only.

**Passthrough parts** are fixed labels: faces between them and grains act like outer boundary
surfaces, so nodes slide along them, and label repair never changes them.

**Label repair** (`label_repair`). Stair-step corners create nodes whose labels match no analytic
feature and single elements sticking out of grains. A few such elements per pass are given to a
face neighbor, chosen by how much they reduce the unfit nodes and protruding elements, at most
one per two-ring neighborhood, kept only if the count really drops; a grain never loses its last
element.

**Adjacency check** (`adjacency_check`). The mesh adjacency (two grains share a node) is
compared with the analytic one (some node can be moved onto their boundary within the move limit
with no other seed closer) and logged. Each pair is merged on an owner rank chosen by a hash of
the pair, so the counts do not depend on the partition.

**Fit** (`conformal_fitter`). The mesh is a linear elastic solid (Young's modulus 1 times a
per-element stiffening factor, Poisson ratio 0.3) on the input geometry. Every fitted node $v$ is
pulled toward its target $\mathbf{t}_v$ by a spring $k_v = c\,h_v\,m_v$. The displacement
$\mathbf{u}$ minimizes

$$
\tfrac12\,\mathbf{u}^{\mathsf T}\mathbf{K}\mathbf{u} + \tfrac12 \sum_v k_v \lVert \mathbf{x}_v + \mathbf{u}_v - \mathbf{t}_v \rVert^2 ,
$$

one linear system $(\mathbf{K} + \mathbf{P})\,\mathbf{u} = \mathbf{P}\,(\mathbf{t} - \mathbf{x})$,
solved by conjugate gradients with an $\ell_1$ Gauss-Seidel preconditioner (hypre). Rounds double
$c$. After each solve every element is checked: scaled Jacobian at the corners at least 0.3 for
hexahedra (times $\sqrt2$ for tetrahedra), aspect ratio (ratio of extreme singular values of the
corner edge matrix) at most 4, and for tetrahedra the mean ratio
$3\,\lvert\det \mathbf{S}\rvert^{2/3} / \lVert\mathbf{S}\rVert_F^2$ at least 0.4. An element that
starts beyond or within 10 % of a limit may lose at most 10 % of its input value (aspect ratio:
grow by 30 %, for thin pillow layers). Where a limit breaks, the springs of nearby nodes are
halved and the round repeated; elements below 0.6 are stiffened by $(0.6/\mathrm{sj})^2$ (capped
at 1000). Rounds stop when the mean boundary distance improves by less than 10 % (from the fourth
kept round), when more than 1 % of the elements would end below floor + 0.2 beyond the input, or
after 10 rounds; the last passing round is kept. A first round failing the 1 % guard is retried
with a quarter of the spring, up to three times.

**Outer surfaces.** On an axis-aligned face the normal coordinate is fixed bit for bit. Tilted and
curved surfaces get a stiff normal spring ($10^3 h$) in the solve and an exact projection after
it (`ProjectOntoPlanes`, or the patch closest point); rims alternate between their two surfaces;
corners do not move.

**Second-order meshes.** Every nodal point is a node of the fit. The fit runs in two passes: the
vertices with an order-1 displacement field interpolated to all nodal points, then, with the
vertices fixed, the edge and face nodes toward their boundaries, which curves the element faces.
Orders 3 and higher are refused.

## Test suite

The tests are registered with `-DEXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON` and carry the CTest label
`microstructure`:

```bash
ctest -L microstructure --output-on-failure            # everything
ctest -R test_geodesic_fim --output-on-failure        # one executable, all rank counts
```

`ms_add_gtest` (`test/microstructure/CMakeLists.txt`) builds one gtest executable per file and
registers one CTest entry per rank count (`<name>_np1` ... `_np4`, limited by
`EXACONSTIT_TEST_MAX_MPI_TASKS`). Each test file's `.cmake` lists its rank counts. MPI-dependent
tests compare their results across rank counts or against a serial reference, so every
partition-independence claim is checked.

### Unit and module tests (`test/microstructure/`)

| Test | Covers |
|---|---|
| `test_skeleton` | build smoke test: exec dispatch, contract defaults, the auto mesh |
| `test_options` | TOML parsing, defaults, validation, command-line overrides; the options template against the defaults |
| `test_util` (`test_counter_rng`, `test_random_distributions`, `test_random_orientation`, `test_fixed_point`) | random numbers, distributions, random orientations, reproducible sums |
| `test_mesh` | serial-order bookkeeping, element and quadrature samplers, preflight checks |
| `test_straight_sided` | straight-sided detection and its effect on seeding and labeling |
| `test_curved_geometry`, `test_curved_determinism` | geometry building blocks and partition independence on curved and parallel-refined meshes |
| `test_geometry` | metric functors and the exact nearest-seed search |
| `test_seeding` | seed generation, spacing, clearance, lattices |
| `test_seed_table` | seed tables, region of interest and filler |
| `test_morphology` | cell measures, size targets, weight fit, Lloyd relaxation |
| `test_morphology_jm` | Johnson-Mehl nucleation and the additive metric |
| `test_labeling` | labels, grain compaction, orientations, the pipeline |
| `test_sampling_modes` | quadrature vote and periodic images |
| `test_connected` | the connected mode, exclusion-aware search, the seed element locator |
| `test_geodesic`, `test_geodesic_fim`, `test_geodesic_curved` | vertex field, graph and FIM geodesic solvers, curved meshes |
| `test_fragments` | face adjacency, element keys, fragment clean-up |
| `test_regions` | phases to regions, phase fractions, attribute domains |
| `test_mask_seeding`, `test_mask_labeling` | everything restricted to the active elements of a mask |
| `test_voxel_map` | voxel-map reading and labeling |
| `test_io`, `test_tool_outputs` | output files read back by ExaConstit's own readers; grain statistics and tables |
| `test_node_labels`, `test_node_targets`, `test_label_repair`, `test_adjacency_check`, `test_conformal_fit` | the conformal steps |
| `test_driver_generated_seeds`, `test_driver_data_sources`, `test_driver_distance_modes`, `test_driver_conformal` | `GenerateMicrostructure` end to end per feature group |
| `test_curved_pipeline` | the whole pipeline on curved meshes |
| `python/test_converters.py` (CTest `test_microstructure_converters`, `test_converters.cmake`) | the Python converters in `scripts/microstructure/` and the microstructure-file writer: each converter runs on small inputs and its output is read back |

### End-to-end chains

- `test_tool_data_sources.cmake`, `test_tool_distance_modes.cmake`: the standalone tool on
  hand-written inputs in `test/data/microstructure/e2e/`, then `mechanics` on its output (decks
  in `test/data/microstructure/mechanics/`).
- `test/CMakeLists.txt`, `test_microstructure_hook_*`: the hook unit test
  (`test/test_microstructure_hook.cpp`) and the hook chain through `mechanics`: hook against
  tool, rank counts, two phases, restart, element order 2 and refinement, passthrough parts with
  a parallel mesh, curved meshes, conformal boundaries, and geodesic grains on the strut
  lattices. The comment above the chain lists every step.

Steps of a chain are linked with CTest fixtures, so a failing step skips the ones that depend on
it.

## Common changes

**Adding an option.** Add the field with its default and documentation to
`microstructure_options.hpp`; parse it and add its `validate()` rules in
`microstructure_options.cpp`; document it in `src/microstructure_options.toml`; extend
`OptionsTemplateDocumentsTheDefaults` in `test_options.cpp`; add a driver-level test. If the hook
must refuse or adapt it, change `ParseMicrostructureHook`.

**Adding a metric.** Add a functor to `geometry/metric_functors.hpp` with its value, gradient and
the lower bounds `SeedIndex` needs, extend `DispatchMetric` and the options, and test exactness
against brute force in `test_geometry.cpp`. Decide whether the geodesic solvers can support it
(one front, or the weighted multi-seed solver), whether the conformal targets can use its
boundary, and update the validation rules accordingly.

**Keeping the documents in step.** A structural change (a new directory, module or entry point)
updates this document and the short `microstructure/` entry of `developers_guide.md`; a
user-visible change updates the user guide, including its recipes table.

**Adding an output file.** Write it in `io/output_writer.*` or `microstructure_report.*`, test
that it reads back with the reader ExaConstit uses (`test_io.cpp`), and list it in the user
guide's file table and the `[Output]` section of the options reference.
