#pragma once

#include "microstructure/microstructure_types.hpp"

// toml11's `operator"" _toml` trips -Wdeprecated-literal-operator on recent clang. This header
// is included by every consumer of the options, so keep that third-party warning out of them.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#include "TOML_Reader/toml.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

/**
 * @file microstructure_options.hpp
 * @brief Options of the microstructure generator (tool input and, later, the mechanics hook).
 *
 * One struct per TOML table ([Mesh], [Source], [Seeds], [Tessellation], [VoxelMap], [Cleanup],
 * [Orientations], [Regions], [Output]), collected in MicrostructureOptions. The defaults written
 * here are the "-n N" quick path: Voronoi tessellation of rejection-sampled random seeds, random
 * orientations, one region, gzip-compressed serial mesh output.
 *
 * Fields that select among a few named choices but have no enum in microstructure_types.hpp are
 * stored as std::string; each lists its accepted values. Parsing lives in from_toml() and the
 * per-table readers in microstructure_options.cpp; range and consistency checks live in
 * validate(). Lengths are in the units of the mesh coordinates (the generator never converts
 * units, apart from Seeds.units_scale for seed tables).
 *
 * Typical use (the standalone tool):
 * @code
 *   auto options = MicrostructureOptions::parse_options_file("micro.toml"); // not validated
 *   options.apply_cli_overrides(cli);                                       // -m -n -s -o --parts
 *   if (!options.validate()) { abort on all ranks }
 * @endcode
 */

namespace exaconstit::microstructure {

/**
 * @brief Where the mesh comes from ([Mesh] and its sub-table [Mesh.Auto]), and which parts of it
 * receive grains.
 *
 * TOML keys: `type`, `floc`, `active_attributes`, `passthrough_attributes`, and for type = "auto"
 * the sub-table [Mesh.Auto] with `ncuts` (alias `nxyz`) and `length` (alias `mxyz`). If both a
 * name and its alias are given, `ncuts` / `length` win.
 */
struct MeshInput {
    /**
     * @brief "file" (one serial MFEM mesh file), "auto" (Cartesian box of hexahedra, the same
     * mesh ExaConstit builds for [Mesh] type = "auto", see utilities/auto_mesh.hpp) or "parallel"
     * (a decomposition with one file per part, given by its file prefix).
     */
    std::string type = "file";
    /**
     * @brief Mesh file for "file"; the file prefix for "parallel" (part p is
     * mfem::MakeParFilename(prefix, p), e.g. "mesh." -> "mesh.000000"). Relative paths are taken
     * from the current working directory.
     */
    std::filesystem::path floc;
    /** @brief "auto" only: number of elements along x, y, z (TOML `ncuts`, alias `nxyz`). */
    std::array<int, 3> nxyz = {1, 1, 1};
    /**
     * @brief "auto" only: box edge lengths along x, y, z, in mesh length units; the box is
     * [0, Lx] x [0, Ly] x [0, Lz] (TOML `length`, alias `mxyz`).
     */
    std::array<double, 3> mxyz = {1.0, 1.0, 1.0};
    /**
     * @brief Element attributes (> 0) of the parts of the mesh that receive grains (TOML
     * `active_attributes = [1, 4]`). Empty (default) means every attribute not listed in
     * passthrough_attributes. Seeding, labeling and cleanup then run on the sub-mesh of these
     * elements only (mesh/attribute_domain.hpp).
     */
    std::vector<int> active_attributes;
    /**
     * @brief Parts of the mesh kept as one grain each, without a tessellation: mesh element
     * attribute (> 0) -> ExaConstit region id (>= 1) of that part (TOML
     * `passthrough_attributes = { "2" = 3 }`; TOML keys are strings and are read as integers).
     * Typical use: the build plate under an additively manufactured part, or a grip. Each
     * passthrough attribute becomes one extra grain with the identity orientation in its region;
     * a passthrough region may share its id with a phase region (same material). Empty (default):
     * every element is active.
     */
    std::map<int, int> passthrough_attributes;
};

/**
 * @brief What defines the grains ([Source]).
 */
struct SourceOptions {
    /**
     * @brief "tessellation" (grains grown from seeds: [Seeds] and [Tessellation]) or "voxel_map"
     * (grain ids read from a voxelized microstructure: [VoxelMap]; no seeds, no tessellation).
     */
    std::string type = "tessellation";
};

/**
 * @brief Target grain sizes ([Seeds.Size]).
 *
 * Grain sizes are described by the equivalent diameter d (the diameter of the sphere with the
 * grain's volume, V = pi d^3 / 6). With a size distribution every seed gets a target volume
 * (morphology/size_targets.hpp), the targets are scaled to fill the material, and they are used
 * for (a) polydisperse seed placement (larger grains get a larger exclusion radius), (b) the
 * Laguerre starting weights w_i = r_i^2 (r_i = equivalent-sphere radius), and (c) the volume fit
 * ([Tessellation] fit_volumes).
 */
struct SeedSizeOptions {
    /**
     * @brief "none" (no size targets; a volume fit then aims at equal volumes), "lognormal"
     * (ln d normally distributed with standard deviation sigma) or "from_source" (volumes from the
     * seed table; needs Seeds.source = "file").
     */
    std::string distribution = "none";
    /**
     * @brief Mean equivalent diameter E[d] (mesh length units). When > 0 and Seeds.num_seeds is
     * 0, the number of grains is chosen so the expected grain volumes fill the material. 0 = not
     * given. At most one of mean_diameter and mean_volume may be given.
     */
    double mean_diameter = 0.0;
    /**
     * @brief Mean grain volume E[V] (mesh length units cubed), the alternative to mean_diameter:
     * for the log-normal, E[d] = (6 E[V] / pi)^(1/3) exp(-sigma^2). 0 = not given.
     */
    double mean_volume = 0.0;
    /**
     * @brief Standard deviation of ln d (dimensionless, >= 0; "lognormal" only, ignored with a
     * warning otherwise); 0.2-0.5 is typical.
     */
    double sigma = 0.0;
    /**
     * @brief Polydisperse random sequential adsorption: seeds are placed largest grain first,
     * and two seeds i, j must be at least exclusion_factor (r_i + r_j) apart (r = equivalent-
     * sphere radius of the target volume). Replaces Seeds.min_distance / min_distance_factor.
     * Needs placement "rejection" or "element_weighted" and no Seeds.sampling_box (the targets
     * fill the whole material). With distribution = "none" all radii are equal (monodisperse
     * RSA with distance 2 kappa r), and the targets reported for the grains are equal volumes.
     * With Seeds.allow_fewer_seeds, a stalled placement keeps the seeds placed so far, i.e. the
     * largest slots, and their targets are rescaled to fill the material.
     */
    bool polydisperse_placement = false;
    /**
     * @brief Exclusion radius of a seed as a fraction kappa of its equivalent-sphere radius
     * (0 < kappa; the spheres of radius kappa r_i then fill kappa^3 of the material, which must be
     * <= 0.3, i.e. kappa <= 0.669). Polydisperse placement only.
     */
    double exclusion_factor = 0.5;
};

/**
 * @brief Johnson-Mehl nucleation and growth ([Seeds.JohnsonMehl]), used with placement =
 * "johnson_mehl" (seeding/johnson_mehl.hpp). Nuclei appear at random places and times at a
 * constant rate per unit volume and time and grow as spheres at a constant speed; a nucleus born
 * inside an already grown grain is discarded. The grains are the cells of the Additive metric
 * with weights a_i = -growth_speed t_i (t_i = birth time), so [Tessellation] metric must be
 * "additive". Time units are arbitrary but must be consistent (only rate x time and speed x time
 * matter).
 */
struct JohnsonMehlOptions {
    /** @brief Nucleation rate I (> 0): nuclei per unit volume per unit time. */
    double rate = 0.0;
    /** @brief Growth speed v (> 0): length per unit time. */
    double growth_speed = 1.0;
    /** @brief Stop nucleating at this time (> 0), or 0 = until the material is transformed. */
    double end_time = 0.0;
    /** @brief Stop after this many nuclei (> 0), or 0 = no count limit. */
    std::int64_t num_nuclei = 0;
};

/**
 * @brief Seed generation ([Seeds]). Every grain grows from one seed point.
 */
struct SeedOptions {
    /** @brief "random" (generated) or "file" (read from a seed table, see `file`). */
    std::string source = "random";
    /**
     * @brief Number of seeds (= grains before cleanup) for random sources; must be > 0 for
     * random seeds unless [Seeds.Size] gives a mean grain size (the count then follows from the
     * material volume) or placement = "johnson_mehl" (the nucleation process sets the count).
     * Ignored for "file", where the table length sets the count.
     */
    std::int64_t num_seeds = 0;
    /**
     * @brief Key of the counter-based random number generator (util/counter_rng.hpp): every random
     * value is computed from (rng_seed, purpose, id) rather than drawn from a running sequence,
     * so the microstructure depends only on this key, never on the number of MPI ranks. Must be
     * >= 0 in TOML (TOML integers are signed).
     */
    std::uint64_t rng_seed = 0;
    /**
     * @brief Placement algorithm for generated seeds. If [Seeds] gives no `placement` and
     * source = "file", the parser sets FILE; validate() rejects any other mismatch between
     * `source` and `placement`.
     */
    SeedPlacement placement = SeedPlacement::REJECTION;
    /**
     * @brief Minimum seed spacing for random sequential adsorption (RSA: a candidate seed closer
     * than this to an accepted one is rejected), as a fraction of the mean grain size
     * (V/N)^(1/3), V = material volume, N = num_seeds. 0 disables it.
     */
    double min_distance_factor = 0.0;
    /**
     * @brief Absolute minimum seed spacing d for RSA, in mesh length units (TOML `min_distance`).
     * When > 0 it is used instead of min_distance_factor; 0 (default) means "use the factor".
     */
    double min_distance = 0.0;
    /** @brief Minimum distance of a seed from the domain boundary, in mesh length units; 0
     * disables it. */
    double boundary_clearance = 0.0;
    /**
     * @brief "material" (seeds only inside the mesh) or "bbox" (anywhere in the sampling box: the
     * mesh bounding box, or the user sampling_box when one is given).
     */
    std::string sampling_domain = "material";
    /**
     * @brief True when the user gave a sampling box (TOML `sampling_box = [xlo, ylo, zlo, xhi,
     * yhi, zhi]`, mesh length units). Seeds are then drawn only inside that box (and, for
     * sampling_domain = "material", inside the material), instead of the whole mesh bounding box.
     * Useful to put grains only in a sub-region (e.g. a gauge section).
     */
    bool use_sampling_box = false;
    /** @brief Lower corner (x, y, z) of the sampling box; used when use_sampling_box is true. */
    std::array<double, 3> sampling_box_lo = {0.0, 0.0, 0.0};
    /** @brief Upper corner (x, y, z) of the sampling box; must exceed sampling_box_lo per axis. */
    std::array<double, 3> sampling_box_hi = {0.0, 0.0, 0.0};
    /**
     * @brief If the RSA minimum distance (or a tiny material fraction) makes seed generation
     * stall before num_seeds seeds are placed: false (default) stops with an error; true keeps
     * the seeds placed so far and reports the achieved count (TOML `allow_fewer_seeds`).
     */
    bool allow_fewer_seeds = false;
    /** @brief Lattice for LATTICE placement: "cubic", "bcc", "fcc" or "hcp". */
    std::string lattice_type = "cubic";
    /** @brief Random lattice offset as a fraction of the lattice spacing (>= 0). */
    double lattice_jitter = 0.0;
    /**
     * @brief Seed table for source = "file" (relative to the current working directory), in the
     * "ExaConstit seed table v1" CSV format (io/seed_table_io.hpp).
     */
    std::filesystem::path file;
    /**
     * @brief Scale applied to the table positions before `translate` (> 0): x_mesh = units_scale
     * * x_file + translate. Rotations are not allowed, since they would also have to rotate the
     * orientations.
     */
    double units_scale = 1.0;
    /** @brief Translation (mesh length units) applied to the table positions after scaling. */
    std::array<double, 3> translate = {0.0, 0.0, 0.0};
    /**
     * @brief source = "file" only: a region of interest (ROI) was given (TOML
     * `roi = {lo = [x, y, z], hi = [x, y, z]}`, mesh length units, after units_scale and
     * translate). Table rows whose position lies outside the closed box [roi_lo, roi_hi] are
     * dropped (and counted in the report). Typical use: the measured volume of an HEDM scan.
     * Note the two box tests: rows are kept in the closed box (a row exactly on a face stays),
     * while the ROI material volume V_roi of the filler counts material strictly inside it.
     */
    bool use_roi = false;
    /** @brief Lower corner of the ROI box (mesh units). */
    std::array<double, 3> roi_lo = {0.0, 0.0, 0.0};
    /** @brief Upper corner of the ROI box (mesh units; > roi_lo on every axis). */
    std::array<double, 3> roi_hi = {0.0, 0.0, 0.0};
    /**
     * @brief How the material outside the ROI gets grains: "none" (default; the table grains grow
     * into it) or "random" (random "filler" seeds strictly outside the ROI box,
     * seeding/roi_filler.hpp; needs `roi`). Filler seeds get source ids max table id + 1, + 2, ...
     * (max over the whole table, before the ROI drop), random orientations, and the phase
     * Regions.filler_phase. Their metric weight follows the table so they compete like an
     * average table grain: with table weights and no size targets, the mean table weight; with
     * table weights and size targets (Laguerre), their target weight r^2 plus the mean over the
     * table rows of (table weight - target weight r^2). Weighted metrics only see weight
     * differences, so this keeps a common offset of the table weights from emptying the filler
     * grains.
     */
    std::string fill_outside_roi = "none";
    /**
     * @brief Number of filler seeds (>= 0); 0 (default) = automatic: the same mean grain volume as
     * the ROI, round(N_roi V_outside / V_roi).
     */
    std::int64_t filler_num_seeds = 0;
    /**
     * @brief Minimum spacing of the filler seeds among themselves, as a fraction of
     * (V_outside / N_filler)^(1/3) (>= 0; default 0 = none).
     */
    double filler_min_distance_factor = 0.0;
    /** @brief Target grain sizes ([Seeds.Size]). */
    SeedSizeOptions size;
    /** @brief Johnson-Mehl parameters ([Seeds.JohnsonMehl]); placement = "johnson_mehl" only. */
    JohnsonMehlOptions johnson_mehl;
};

/**
 * @brief Volume fitting ([Tessellation] `fit_volumes = {enabled, tolerance, max_iterations,
 * samples}`): adjust the seed weights so every grain gets its target volume ([Seeds.Size], or
 * equal volumes without a size distribution) with morphology/weight_fit.hpp. Needs a weighted
 * metric (laguerre, quadric or additive).
 */
struct FitVolumesOptions {
    /** @brief Run the fit. */
    bool enabled = false;
    /** @brief Stop when every relative volume error is <= tolerance (> 0). */
    double tolerance = 0.02;
    /** @brief Largest number of solver steps (>= 0). */
    int max_iterations = 200;
    /**
     * @brief Points the volumes are measured with: "quadrature" (every quadrature point; smooth
     * response, recommended) or "element" (element centroids: the volumes ExaConstit will see,
     * but they change in whole-element jumps).
     */
    std::string samples = "quadrature";
};

/**
 * @brief Tessellation metric and element sampling ([Tessellation]).
 */
struct TessellationOptions {
    /** @brief Metric family (TOML `metric`: "voronoi", "laguerre", "quadric", "additive"). */
    MetricType metric = MetricType::VORONOI;
    /**
     * @brief Global anisotropy tensor A of the quadric metric, distance^2 = (x - s)^T A (x - s).
     *
     * A is a symmetric 3x3 matrix
     * @code
     *   A = [[xx, xy, xz],
     *        [xy, yy, yz],
     *        [xz, yz, zz]]
     * @endcode
     * stored as its 6 independent entries in the order [xx, yy, zz, yz, xz, xy] (diagonal first,
     * then the off-diagonals in Voigt order). It acts as a "ruler" that measures distance
     * differently along different directions: a small eigenvalue along a direction makes that
     * direction look short, so grains grow elongated along it. Must be positive definite
     * (checked in validate()). The identity reproduces Voronoi. TOML:
     * `anisotropy = {global_tensor = [...]}` or the shorthand `global_tensor = [...]` (either
     * sets anisotropy_kind = "tensor"). Ignored for the columnar and lamellar kinds.
     */
    std::array<double, 6> global_tensor = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    /**
     * @brief Number of parameter bins of the seed search structure (>= 1; performance only).
     *
     * Seeds are grouped by their weight/shape parameters and each group gets its own BVH
     * (bounding volume hierarchy: a tree of nested bounding boxes that lets the search skip seeds
     * that are provably too far away from an element). More bins give tighter bounds per tree.
     */
    int num_bins = 8;
    /** @brief Element sampling mode (TOML `sampling`: "centroid", "quadrature_vote"). */
    SamplingMode sampling = SamplingMode::CENTROID;
    /**
     * @brief Periodicity of the microstructure along x, y, z: a grain cut by one face continues on
     * the opposite face (implemented with periodic images of the seeds).
     */
    std::array<bool, 3> periodic = {false, false, false};
    /**
     * @brief Periodic axes only: matching tolerance of the periodic face pairing used by the
     * fragment cleanup, as a fraction of the smallest edge of the faces on the periodic planes
     * (PeriodicFaceParameters::relative_tolerance, mesh/mesh_adjacency.hpp). In (0, 0.5);
     * default 1e-6. The mesh must have matching faces on opposite periodic planes, otherwise the
     * run stops with an error.
     */
    double periodic_tolerance = 1.0e-6;
    /**
     * @brief How "closest seed" is measured on parts with holes, notches or lattice struts (TOML
     * `distance`):
     * - "euclidean" (default): the metric value in straight lines, even across a void;
     * - "connected": the straight-line (metric) labels, but every grain keeps only the elements
     *   connected through material to the element holding its seed; cut-off elements take the
     *   next best seed (labeling/connected_labeler.hpp). Every grain boundary is still the
     *   boundary between two seeds of the metric. Works with every metric;
     * - "geodesic": the distance is the length of the shortest path inside the material, so
     *   grains grow around corners and voids (labeling/geodesic_fim.hpp or geodesic_graph.hpp;
     *   solver choice in `geodesic`). Voronoi (path length d) or Additive (d - a_i, e.g.
     *   Johnson-Mehl) metric only, without anisotropy: Laguerre and Quadric values are not path
     *   lengths. Two properties to know:
     *   - paths connect elements through shared vertices, edges and faces (the solvers work on
     *     the mesh vertices), while the connected mode and the fragment cleanup connect elements
     *     through shared faces only. Elements that share nodes are connected in the conforming FE
     *     mesh, so a grain may grow through an edge or vertex contact; the cleanup then treats a
     *     piece attached only that way as a separate fragment ([Cleanup] fragments policy);
     *   - with Johnson-Mehl placement the nucleation step still decides which nuclei survive with
     *     straight-line growth spheres: a candidate is discarded when an earlier grain's sphere
     *     covers it, even across a void. Only the final labels use path lengths (the driver
     *     records a warning).
     * Scope limits of "connected" and "geodesic" (checked in validate() and by the driver): no
     * periodic axes, no fit_volumes, no Lloyd relaxation / centroidal power, no voxel maps. Both
     * run on the active part of the mesh (Mesh.active_attributes / passthrough_attributes), so
     * paths never pass through a passthrough part.
     */
    DistanceMode distance = DistanceMode::EUCLIDEAN;
    /**
     * @brief distance = "connected" only: largest number of relabeling rounds (TOML
     * `connected_max_rounds`, default 16; in 0..CONNECTED_MAX_ROUNDS_LIMIT = 1000 of
     * labeling/connected_limits.hpp, an error in any mode outside that range; 16-64 is plenty
     * for lattices). Each round moves the elements still cut off from their seed to the next best
     * seed; elements left after the last round
     * keep their label and are merged by the fragment cleanup. Memory: one int per cut-off element
     * per round performed.
     */
    int connected_max_rounds = 16;
    /**
     * @brief distance = "geodesic" only: the path-length solver (TOML
     * `geodesic = { solver = "..." }`):
     * - "fim" (default): Fast Iterative Method eikonal solver (labeling/geodesic_fim.hpp): solves
     *   |grad T| = 1 inside the material with linear interpolation over element faces, so the
     *   distances are not tied to edge directions (accurate; the recommended solver);
     * - "graph": shortest paths along the element edges and diagonals (labeling/geodesic_graph.hpp;
     *   fast but approximate: distances are over-estimated by up to ~12% on hex meshes and ~30% on
     *   tet meshes, grain boundaries move much less).
     * An unknown name is an error even when distance is not "geodesic" (validate()).
     */
    std::string geodesic_solver = "fim";
    /**
     * @brief Quadrature-vote sampling only: subdivision levels 0..3 of the voting points, 8^levels
     * points per element (labeling/quadrature_vote.hpp; default 2 = 64 points). Level 1 is not
     * recommended (its 8 points measured worse than centroid labeling).
     */
    int vote_levels = 2;
    /**
     * @brief Kind of anisotropy of the Quadric metric (TOML `anisotropy = {kind = ...}`):
     * - "none": global_tensor as stored (the identity unless set, i.e. Voronoi distances);
     * - "tensor": the user tensor anisotropy.global_tensor (set automatically when a
     *   global_tensor is written);
     * - "columnar": grains unbounded along anisotropy_axis (A = I - a a^T, positive
     *   semidefinite), e.g. columnar grains of additively manufactured or directionally solidified
     *   material;
     * - "lamellar": flat layers perpendicular to anisotropy_normal (A = n n^T);
     * - "per_seed": every seed has its own tensor A_i = M_i^-1 from the shape columns
     *   (m11 .. m12) of a seed table (Seeds.source = "file"); random filler seeds get
     *   A = I / r^2, with r the equivalent-sphere radius of their equal share V_outside / N_filler
     *   of the material outside the ROI, so their f = 1 surface is the sphere of that volume.
     * Columnar, lamellar and per_seed need metric = "quadric". Without "per_seed", shape columns
     * of a seed table are ignored (with a warning).
     */
    std::string anisotropy_kind = "none";
    /** @brief Column axis for anisotropy_kind = "columnar" (normalized; TOML `axis`). */
    std::array<double, 3> anisotropy_axis = {0.0, 0.0, 1.0};
    /** @brief Layer normal for anisotropy_kind = "lamellar" (TOML `normal`). */
    std::array<double, 3> anisotropy_normal = {0.0, 0.0, 1.0};
    /** @brief Volume fitting (TOML `fit_volumes = {...}`). */
    FitVolumesOptions fit_volumes;
    /**
     * @brief Number of Lloyd relaxation steps (>= 0): each moves every seed to the centroid of its
     * grain, giving rounder, more uniform grains (morphology/lloyd.hpp). Needs metric voronoi or
     * laguerre. Uses the fit_volumes.samples points.
     */
    int lloyd_iterations = 0;
    /**
     * @brief Centroidal power mode: every Lloyd step is followed by a volume refit, so grains are
     * both centered on their seeds and of their target volumes. Needs lloyd_iterations > 0 and
     * metric = "laguerre"; uses the fit_volumes tolerance and max_iterations.
     */
    bool centroidal_power = false;
};

/**
 * @brief A voxelized microstructure as the grain source ([VoxelMap]; used with [Source] type =
 * "voxel_map"). The file formats are described in io/voxel_map_io.hpp; the element labeling in
 * labeling/voxel_labeler.hpp.
 */
struct VoxelMapOptions {
    /** @brief "exaconstit" (canonical "ExaConstit voxel map v1" HDF5 layout) or "dream3d". */
    std::string format = "exaconstit";
    /** @brief The HDF5 file (relative to the current working directory). */
    std::filesystem::path file;
    /** @brief HDF5 path of the per-voxel id dataset; empty = the format's default / name search. */
    std::string feature_ids_path;
    /** @brief HDF5 path of the per-grain quaternions; empty = default / name search. */
    std::string quats_path;
    /**
     * @brief HDF5 path of the per-grain phases; empty = default / name search. Phases are read
     * whenever the file has them (they fill MicrostructureResult::grain_phase); a file without
     * phases is an error only for Regions.mode = "from_phase".
     */
    std::string phases_path;
    /**
     * @brief How the file stores a quaternion: "" (not given), "wxyz" or "xyzw". Required for
     * DREAM.3D files when quaternions are read (Orientations.source = "from_source"); no default.
     */
    std::string quat_layout;
    /**
     * @brief Whether the stored quaternions must be conjugated to give ExaConstit's crystal-to-
     * sample passive rotation: -1 not given, 0 false, 1 true (TOML boolean `conjugate`). Required
     * for DREAM.3D files when quaternions are read; no default.
     */
    int conjugate = -1;
    /** @brief `origin` was given: replaces the file's lower corner of voxel (0,0,0). */
    bool has_origin = false;
    /** @brief Lower corner of voxel (0,0,0), file units (TOML `origin = [x, y, z]`). */
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    /** @brief `spacing` was given: replaces the file's voxel edge lengths. */
    bool has_spacing = false;
    /** @brief Voxel edge lengths, file units, > 0 (TOML `spacing = [dx, dy, dz]`). */
    std::array<double, 3> spacing = {0.0, 0.0, 0.0};
    /** @brief File-to-mesh length factor (> 0): x_mesh = units_scale * x_file + translate. */
    double units_scale = 1.0;
    /** @brief Added after scaling (mesh units). */
    std::array<double, 3> translate = {0.0, 0.0, 0.0};
    /**
     * @brief How elements sample the voxels: "centroid" (default; the voxel under the element
     * centroid) or "quadrature_vote" (volume-weighted vote of 8^vote_levels points per element).
     */
    SamplingMode sampling = SamplingMode::CENTROID;
    /** @brief Vote sampling only: subdivision levels 0..3 (default 2). */
    int vote_levels = 2;
    /** @brief Edge of the voxel tiles each rank reads, in voxels (>= 1; performance only). */
    int tile_size = 32;
};

/**
 * @brief Grain cleanup after labeling ([Cleanup]).
 */
struct CleanupOptions {
    /**
     * @brief What to do with a grain whose elements form several disconnected pieces (connected
     * through element faces; with periodic axes also through matching periodic faces):
     * - "auto": for generated seeds (random, lattice, Johnson-Mehl) split large pieces into new
     *   grains (a grain really cut in two, e.g. by a void of a lattice) and merge tiny pieces
     *   (min_fragment_elements / min_fragment_fraction; typically single elements that touch
     *   their grain only along an edge) into the neighbor they share the most faces with; for
     *   data sources such as seed tables merge every extra piece;
     * - "split": every extra piece becomes its own grain;
     * - "merge": every extra piece joins the neighboring grain it shares the most faces with;
     * - "none": count only.
     */
    std::string fragments = "auto";
    /**
     * @brief "auto" only: an extra piece with fewer elements than this is tiny and merged
     * (>= 0; default 8).
     */
    int min_fragment_elements = 8;
    /**
     * @brief "auto" only: an extra piece with less than this fraction of its grain's elements is
     * tiny and merged too (in [0, 1]; default 0.1).
     */
    double min_fragment_fraction = 0.1;
    /**
     * @brief Which piece of a disconnected grain keeps the grain's identity (and orientation):
     * "largest" (most elements) or "seed" (the piece containing the grain's seed, falling back to
     * the largest when the seed lies in no element of the grain). With Tessellation.distance =
     * "connected" or "geodesic" the piece grown from the seed (its anchor / source element) is
     * always kept, whatever this says.
     */
    std::string keep_component = "largest";
    /**
     * @brief Grains with fewer elements than this are merged into the neighbor they share the
     * most faces with. 0 and 1 both mean no merging (every grain has at least one element);
     * negative values are rejected.
     */
    int min_grain_elements = 1;
};

/**
 * @brief Orientation assignment ([Orientations]). Orientations are crystal-to-sample, passive
 * quaternions in ExaConstit's `w x y z` order. Supplied orientations are normalized to unit length
 * and otherwise passed through unchanged (convention, sign and component order kept).
 */
struct OrientationOptions {
    /**
     * @brief "random" (uniformly distributed rotations from the counter RNG), "from_source"
     * (taken from the seed table's qw, qx, qy, qz columns, or from the voxel map's per-grain
     * quaternions; grains without one, such as random filler grains, get random orientations)
     * or "list" (from a separate list; not implemented yet, refused by the driver).
     */
    std::string source = "random";
};

/**
 * @brief Region assignment ([Regions]): how grains are grouped into ExaConstit material regions.
 */
struct RegionOptions {
    /**
     * @brief "single" (one region), "from_phase" (regions from the grain phases of the seed table
     * or the voxel map, regions/region_assigner.hpp) or "phase_fractions" (synthetic phases with
     * prescribed volume fractions, `phases`, then regions from those phases).
     */
    std::string mode = "single";
    /**
     * @brief Phase ids (>= 0) whose grains are not crystals; their regions are flagged
     * non-crystalline and their grains get the identity orientation (TOML
     * `non_crystalline_phases = [2]`).
     */
    std::vector<int> non_crystalline_phases;
    /**
     * @brief Optional explicit phase -> region map (TOML `phase_to_region = { "0" = 1, "3" = 1,
     * "7" = 2 }`; keys are strings read as integers >= 0, values region ids >= 1). Empty: regions
     * 1..M in ascending phase order. Several phases may share a region; the region ids used must
     * be exactly 1..M.
     */
    std::map<int, int> phase_to_region;
    /**
     * @brief phase_fractions only: the phase ids (>= 0, distinct) of TOML
     * `phases = [{id = 1, fraction = 0.7}, {id = 2, fraction = 0.3}]`.
     */
    std::vector<int> phase_ids;
    /** @brief phase_fractions only: volume fraction of phase_ids[k] (> 0, sum 1 within 1e-6). */
    std::vector<double> phase_fractions;
    /**
     * @brief `filler_phase` was given. Required when random ROI filler seeds
     * (Seeds.fill_outside_roi = "random") are combined with mode = "from_phase".
     */
    bool has_filler_phase = false;
    /** @brief Phase id (>= 0) of the filler grains (TOML `filler_phase`). */
    int filler_phase = 0;
};

/**
 * @brief Output files ([Output]). Every file is named directory/basename.<extension>.
 */
struct OutputOptions {
    /** @brief Output directory. */
    std::filesystem::path directory = ".";
    /** @brief Base name of every output file (must not be empty). */
    std::string basename = "microstructure";
    /**
     * @brief Serial file, parallel decomposition (readable by ExaConstit's [Mesh] type =
     * "parallel") or both (TOML `mesh_format`: "serial", "parallel", "both").
     */
    MeshFormat mesh_format = MeshFormat::SERIAL;
    /** @brief Number of parts of the parallel decomposition (>= 0); 0 means the number of ranks. */
    int num_parts = 0;
    /** @brief gzip level 0-9 of the mesh files; 0 writes plain text. */
    int compression = 6;
    /**
     * @brief Write the per-element grain ids to a separate grain file: "auto" (only for auto
     * meshes, which ExaConstit rebuilds itself and which therefore cannot carry the ids in a mesh
     * file), "true" or "false". TOML accepts the string or a boolean (`grain_file = true` is
     * stored as "true").
     */
    std::string grain_file = "auto";
    /** @brief Write a ParaView collection with the grain ids for visualization. */
    bool paraview = true;
    /**
     * @brief Also write the grain adjacency table `<basename>_adjacency.csv`: one row per pair of
     * grains that share at least one element face, with the total shared face area (mesh length
     * units squared) and the number of shared faces. With periodic axes, faces matched across
     * opposite box planes count too. TOML `grain_adjacency` (boolean, default false).
     */
    bool grain_adjacency = false;
    // No ExaConstit input (TOML) files are generated: users write their own ExaConstit options.
    // Provenance (tool version, rng_seed, requested and final grain counts) goes in the mesh file
    // header comments and in the rank-0 log at the end of a run.
};

/**
 * @brief Command-line overrides of the tool (`-m -n -s -o --parts`). Fields left unset
 * (std::nullopt) do not change anything. Applied by MicrostructureOptions::apply_cli_overrides.
 */
struct CliOverrides {
    /** @brief -m FILE: serial mesh file; sets [Mesh] type = "file". */
    std::optional<std::filesystem::path> mesh;
    /**
     * @brief -n N: number of random seeds. Quick path: also switches a seed-table source back to
     * random seeds with REJECTION placement (an explicit random placement is kept).
     */
    std::optional<std::int64_t> num_seeds;
    /** @brief -s SEED: Seeds.rng_seed. */
    std::optional<std::uint64_t> rng_seed;
    /** @brief -o NAME: Output.basename. */
    std::optional<std::string> basename;
    /**
     * @brief --parts N: Output.num_parts. N > 0 also turns a SERIAL mesh_format into BOTH, since
     * asking for parts means asking for a decomposition (the serial file is still written).
     */
    std::optional<int> num_parts;
};

/**
 * @brief All options of the microstructure generator.
 *
 * Every rank parses the same input and holds an identical copy; nothing here communicates
 * between ranks. Warnings and validation errors are printed on rank 0 only (through
 * options/option_util.hpp), so MPI must be initialized before from_toml, parse_options_file or
 * validate are called.
 */
struct MicrostructureOptions {
    /**
     * @brief Where the kernels run: "cpu", "openmp" or "gpu". Selects the mfem::Device backend
     * exactly like ExaConstit's RTModel option ("cpu", "raja-omp", "raja-cuda"/"raja-hip"; see
     * ConfigureMfemDevice in exec_dispatch.hpp). BVH (seed search) work follows the same backend
     * through DispatchAxomExec. validate() rejects unknown names and backends this build lacks
     * (IsExecAvailable). Passing validate() does not make GPU BVH work safe everywhere:
     * DispatchAxomExec can only run GPU kernels from a source file compiled as CUDA/HIP (it
     * throws in a host-compiled file).
     */
    std::string exec = "cpu";
    /**
     * @brief Require results that are independent of rank count and execution space. Not
     * implemented yet (the flag is parsed and stored only). Rank-count independence is the
     * design goal regardless; bitwise CPU/GPU agreement would additionally need
     * MICROSTRUCTURE_STRICT_FP and matching host/device math libraries.
     */
    bool deterministic = true;

    MeshInput mesh;                   ///< [Mesh]
    SourceOptions source;             ///< [Source]
    SeedOptions seeds;                ///< [Seeds]
    TessellationOptions tessellation; ///< [Tessellation]
    VoxelMapOptions voxel_map;        ///< [VoxelMap]
    CleanupOptions cleanup;           ///< [Cleanup]
    OrientationOptions orientations;  ///< [Orientations]
    RegionOptions regions;            ///< [Regions]
    OutputOptions output;             ///< [Output]

    /**
     * @brief Checks the final options (after CLI overrides): accepted string values, positive
     * counts, consistent source/placement, existing input files, a positive definite anisotropy
     * tensor (or a valid columnar/lamellar direction with the quadric metric), output ranges,
     * and the combinations of the size, Johnson-Mehl, volume-fit, Lloyd and periodic options
     * (e.g. Johnson-Mehl needs the additive metric; fitting needs a weighted metric; neither
     * Lloyd nor fitting works with periodic axes).
     *
     * Stage 1d checks: attribute and region ids (> 0 / >= 1, no attribute both active and
     * passthrough; with Regions.mode = "single" the passthrough regions must make the region
     * ids 1..R); passthrough attributes refuse boundary_clearance > 0 and periodic axes; the ROI
     * (seed tables only, hi > lo) and filler keys (fill_outside_roi = "random" needs the ROI);
     * seed tables refuse Lloyd / centroidal power and Johnson-Mehl; anisotropy "per_seed" needs
     * the quadric metric and a seed table; [VoxelMap] parameters (format, existing file, the
     * DREAM.3D quaternion declarations when orientations are read, positive scale and spacing,
     * vote_levels 0-3, tile_size >= 1); voxel maps refuse every option that would shape a
     * tessellation (seed table, Johnson-Mehl, sizes, ROI, metric, anisotropy, periodic,
     * fit_volumes, Lloyd, keep_component = "seed", a distance other than "euclidean") and only
     * warn about harmless seed-placement leftovers; from_source orientations and from_phase
     * regions need a seed table or a voxel map; phase fractions need distinct phase ids >= 0
     * with positive fractions summing to 1; from_phase with filler needs Regions.filler_phase.
     * Whether a table or voxel map actually holds the needed columns or datasets is only known
     * when it is read, so the driver checks that (on every rank together).
     *
     * Stage 1e checks: distance = "connected" or "geodesic" refuses periodic axes, fit_volumes,
     * Lloyd relaxation / centroidal power and voxel maps; "geodesic" also needs metric "voronoi"
     * or "additive" without anisotropy (anisotropy kind "none"). In any mode the solver name
     * must be "fim" or "graph" and connected_max_rounds must be in 0..1000
     * (CONNECTED_MAX_ROUNDS_LIMIT, labeling/connected_limits.hpp); valid values of a mode that is
     * not selected only warn that they are ignored.
     *
     * Every problem is printed (rank 0) before returning, so the user sees all of them at once.
     * Not collective. Every rank holds the same options, but the file-existence checks assume
     * every rank sees the same file system; to stop all ranks together even when that fails, the
     * caller combines the results with MPI_Allreduce(MPI_MIN) before aborting. Requires MPI to be
     * initialized.
     *
     * @param problems optional; each error message is also appended here (all ranks, not only
     *                 rank 0), so a non-root rank whose checks fail alone can print them.
     * @return true if every check passed.
     * @throws nothing for invalid options (they are reported, not thrown); only
     *         std::filesystem::filesystem_error if the file system cannot be queried at all.
     */
    bool validate(std::vector<std::string>* problems = nullptr) const;

    /**
     * @brief Applies the tool's command-line overrides on top of the parsed options (see
     * CliOverrides for the side effects of -n and --parts). Call before validate(). Host only,
     * not collective, never throws.
     */
    void apply_cli_overrides(const CliOverrides& overrides);

    /**
     * @brief Builds the options from a parsed TOML document. Missing keys keep their defaults;
     * unknown keys produce a rank-0 warning. Does NOT validate.
     *
     * Not collective, but needs MPI initialized (rank-0 warnings).
     *
     * @throws std::invalid_argument for values that cannot be stored (unknown enum name such as
     *         metric or placement, wrong array length, negative rng_seed, unknown [Source] type,
     *         a non-integer key of an attribute or phase map); toml11 exceptions
     *         (toml::type_error) for values of the wrong TOML type.
     */
    static MicrostructureOptions from_toml(const toml::value& toml_input);

    /**
     * @brief Reads and parses a TOML options file, then calls from_toml(). Does NOT validate:
     * the caller applies CLI overrides first and then calls validate().
     *
     * Not collective (every rank reads the small file itself), but needs MPI initialized.
     *
     * @throws std::runtime_error if the file does not exist; toml11 syntax errors (with line
     *         numbers) for malformed TOML; anything from_toml() throws.
     */
    static MicrostructureOptions parse_options_file(const std::filesystem::path& path);
};

} // namespace exaconstit::microstructure
