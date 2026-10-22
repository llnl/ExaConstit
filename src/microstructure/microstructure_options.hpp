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
#include <optional>
#include <string>
#include <vector>

/**
 * @file microstructure_options.hpp
 * @brief Options of the microstructure generator (tool input and, later, the mechanics hook).
 *
 * One struct per TOML table ([Mesh], [Seeds], [Tessellation], [Cleanup], [Orientations],
 * [Regions], [Output]), collected in MicrostructureOptions. The defaults written here are the
 * "-n N" quick path: Voronoi tessellation of rejection-sampled random seeds, random orientations,
 * one region, gzip-compressed serial mesh output.
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
 * @brief Where the mesh comes from ([Mesh] and its sub-table [Mesh.Auto]).
 *
 * TOML keys: `type`, `floc`, and for type = "auto" the sub-table [Mesh.Auto] with `ncuts` (alias
 * `nxyz`) and `length` (alias `mxyz`). If both a name and its alias are given, `ncuts` / `length`
 * win.
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
    /** @brief Seed table for source = "file" (relative to the current working directory). */
    std::filesystem::path file;
    /**
     * @brief Scale applied to the table positions before `translate` (> 0): x_mesh = units_scale
     * * x_file + translate. Rotations are not allowed, since they would also have to rotate the
     * orientations.
     */
    double units_scale = 1.0;
    /** @brief Translation (mesh length units) applied to the table positions after scaling. */
    std::array<double, 3> translate = {0.0, 0.0, 0.0};
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
    /** @brief Distance mode (TOML `distance`: "euclidean", "connected", "geodesic"). */
    DistanceMode distance = DistanceMode::EUCLIDEAN;
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
     * - "lamellar": flat layers perpendicular to anisotropy_normal (A = n n^T).
     * Columnar and lamellar need metric = "quadric".
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
     * the largest when the seed lies in no element of the grain).
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
 * quaternions in ExaConstit's `w x y z` order and are passed through verbatim.
 */
struct OrientationOptions {
    /**
     * @brief "random" (uniformly distributed rotations from the counter RNG), "from_source"
     * (taken from the seed table; needs Seeds.source = "file") or "list" (from a separate list).
     * Only "random" is implemented so far; the driver refuses the others before any work.
     */
    std::string source = "random";
};

/**
 * @brief Region assignment ([Regions]): how grains are grouped into ExaConstit material regions.
 */
struct RegionOptions {
    /** @brief "single" (one region), "from_phase" (region = seed phase) or "phase_fractions". */
    std::string mode = "single";
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
    SeedOptions seeds;                ///< [Seeds]
    TessellationOptions tessellation; ///< [Tessellation]
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
     *         metric or placement, wrong array length, negative rng_seed, unsupported [Source]
     *         type); toml11 exceptions (toml::type_error) for values of the wrong TOML type.
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
