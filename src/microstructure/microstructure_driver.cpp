/*
 * Implementation of GenerateMicrostructure (microstructure_driver.hpp): the Stage 1a-1e pipeline
 * support checks -> active part (attribute domain) -> preflight -> grain labels (seeds + size
 * targets -> Lloyd / volume fit -> tessellation labels in the chosen distance mode -> compact ids,
 * or a voxel map) -> fragment cleanup -> orientations -> phases -> regions -> passthrough grains.
 *
 * Every step is either computed from data that is identical on all ranks (the options, the
 * replicated seed set, the seed table and per-grain tables) or combined with MPI reductions inside
 * the called functions, so all ranks make the same decisions and, on failure, throw the same
 * exception together. A function that every rank must call because ranks exchange data inside it
 * is called "collective".
 *
 * Terms used in this file:
 * - Size targets: one target volume per seed (morphology/size_targets.hpp), scaled so they fill
 *   the material; r_i = (3 V_i / (4 pi))^(1/3) is the equivalent-sphere radius of target V_i.
 * - Polydisperse RSA: random sequential adsorption in which every seed slot has its own exclusion
 *   radius (kappa r_i), largest first (seeding/seed_generators.hpp, SeedingParameters::rsa_radius).
 * - Fixed-point sum: a sum in which every term is first rounded to an integer multiple of a tiny
 *   quantum and the integers are added (util/fixed_point.hpp); integer addition does not depend
 *   on the order of the terms, so the result is bitwise the same for every rank count.
 * - Source id: the stable id of a seed (SeedSet::source_id) or of a voxel feature; random
 *   orientations are keyed by it.
 * - Data-driven source: a seed table or a voxel map (measured grains), as opposed to generated
 *   seeds (random, lattice, Johnson-Mehl).
 * - ROI: the region of interest of a seed table, an axis-aligned box; filler seeds are random
 *   seeds placed outside it.
 * - Active mesh: the elements that receive grains. With passthrough attributes it is a sub-mesh
 *   (mfem::ParSubMesh) of the whole mesh, owned by an AttributeDomain that lives only inside
 *   GenerateMicrostructure.
 * - Distance mode (Tessellation.distance): "euclidean" (nearest seed in straight lines),
 *   "connected" (straight-line labels, but a grain keeps only the elements connected through
 *   material to its seed's anchor element) or "geodesic" (nearest seed along paths inside the
 *   material).
 * - Anchor element: the element that contains a seed (for a seed outside the material, the
 *   element nearest to it), named by its element key (mesh/element_key.hpp: a hash of the exact
 *   vertex coordinates, the same for every partition).
 */
#include "microstructure/microstructure_driver.hpp"

#include "microstructure/geometry/periodic_images.hpp"
#include "microstructure/io/seed_table_io.hpp"
#include "microstructure/io/voxel_map_io.hpp"
#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/labeling/geodesic_fim.hpp"
#include "microstructure/labeling/geodesic_graph.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/labeling/voxel_labeler.hpp"
#include "microstructure/mesh/attribute_domain.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/preflight.hpp"
#include "microstructure/morphology/size_targets.hpp"
#include "microstructure/orientation/orientation_assigner.hpp"
#include "microstructure/regions/region_assigner.hpp"
#include "microstructure/seeding/material_point_locator.hpp"
#include "microstructure/seeding/roi_filler.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

namespace {

constexpr double PI = 3.14159265358979323846;

/** True when the grains come from a voxel map instead of a tessellation. */
bool UsesVoxelMap(const MicrostructureOptions& options) {
    return options.source.type == "voxel_map";
}

/** True for data-driven sources (measured grains): a seed table or a voxel map. */
bool IsDataDriven(const MicrostructureOptions& options) {
    return UsesVoxelMap(options) || options.seeds.source == "file";
}

/** True when any axis is periodic. */
bool AnyPeriodic(const TessellationOptions& tess) {
    return tess.periodic[0] || tess.periodic[1] || tess.periodic[2];
}

/*
 * Warn: records a data-dependent warning in the report and prints it once, on rank 0 of `comm`
 * (std::cerr). Used for things that are only known after a seed table or voxel map is read
 * (e.g. ignored table columns), which validate() cannot see. Not collective (every rank records
 * the same text, since the data is replicated).
 */
void Warn(MPI_Comm comm, GenerationReport& report, const std::string& message) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::cerr << "GenerateMicrostructure: warning: " << message << std::endl;
    }
    report.warnings.push_back(message);
}

/*
 * CheckSupportedOptions
 *
 * The options structure describes the whole planned generator, but this pipeline implements the
 * Stage 1a-1e subset. Options that ask for something not implemented yet are refused here,
 * before any work, with a message naming the option, rather than being silently ignored. The
 * tessellation checks (distance, vote levels, anisotropy kind) are owned by the labeler
 * (CheckSupportedTessellation) and called from here, so they also run before seeding. The
 * combination checks of validate() that would otherwise fail deep inside a collective step (or,
 * worse, silently do something else) are repeated, so a caller that skipped validate() still
 * gets a clear message on every rank.
 */
void CheckSupportedOptions(const MicrostructureOptions& options) {
    const auto refuse = [](const std::string& what) {
        throw std::invalid_argument("GenerateMicrostructure: " + what);
    };
    const SeedOptions& seeds = options.seeds;
    const TessellationOptions& tess = options.tessellation;
    const bool voxel = UsesVoxelMap(options);
    const bool table = !voxel && seeds.source == "file";
    const bool passthrough = !options.mesh.passthrough_attributes.empty();

    // Grain source
    if (options.source.type != "tessellation" && !voxel) {
        refuse("unknown Source.type \"" + options.source.type + "\"");
    }
    if (seeds.source != "random" && seeds.source != "file") {
        refuse("unknown Seeds.source \"" + seeds.source + "\"");
    }
    if (seeds.source == "file" && seeds.placement != SeedPlacement::FILE) {
        refuse("Seeds.source = \"file\" requires Seeds.placement = \"file\"");
    }
    if (seeds.source == "random" && seeds.placement == SeedPlacement::FILE && !voxel) {
        refuse("Seeds.placement = \"file\" requires Seeds.source = \"file\"");
    }
    if (table && seeds.file.empty()) {
        refuse("Seeds.source = \"file\" needs 'Seeds.file'");
    }
    if (voxel && options.voxel_map.file.empty()) {
        refuse("Source.type = \"voxel_map\" needs 'VoxelMap.file'");
    }

    // Regions and orientations
    const std::string& mode = options.regions.mode;
    if (mode != "single" && mode != "from_phase" && mode != "phase_fractions") {
        refuse("unknown Regions.mode \"" + mode + "\"");
    }
    if (mode == "from_phase" && !IsDataDriven(options)) {
        refuse("Regions.mode = \"from_phase\" needs a seed table or a voxel map (generated seeds "
               "have no phase)");
    }
    if (options.orientations.source == "list") {
        refuse("Orientations.source = \"list\" is not implemented yet");
    }
    if (options.orientations.source != "random" && options.orientations.source != "from_source") {
        refuse("unknown Orientations.source \"" + options.orientations.source + "\"");
    }
    if (options.orientations.source == "from_source" && !IsDataDriven(options)) {
        refuse("Orientations.source = \"from_source\" needs a seed table or a voxel map "
               "(generated seeds carry no orientation)");
    }

    // Cleanup
    const std::string& fragments = options.cleanup.fragments;
    if (fragments != "auto" && fragments != "split" && fragments != "merge" &&
        fragments != "none") {
        refuse("unknown Cleanup.fragments \"" + fragments + "\"");
    }
    if (options.cleanup.keep_component != "largest" && options.cleanup.keep_component != "seed") {
        refuse("unknown Cleanup.keep_component \"" + options.cleanup.keep_component + "\"");
    }
    if (options.cleanup.min_grain_elements < 0) {
        refuse("Cleanup.min_grain_elements < 0");
    }

    // Passthrough parts: the interface with the active part is a free surface of the sub-mesh
    // (mesh/attribute_domain.hpp), so the boundary clearance and periodic-plane detection would
    // treat it as an outer boundary
    if (passthrough && seeds.boundary_clearance > 0.0 && !voxel) {
        refuse("Seeds.boundary_clearance > 0 cannot be combined with Mesh.passthrough_attributes");
    }
    if (passthrough && AnyPeriodic(tess)) {
        refuse("Tessellation.periodic cannot be combined with Mesh.passthrough_attributes");
    }

    // Seed tables: measured positions must not move, and the ROI options need a table
    if (table && (tess.lloyd_iterations > 0 || tess.centroidal_power)) {
        refuse("Tessellation.lloyd_iterations / centroidal_power cannot be used with "
               "Seeds.source = \"file\" (they would move the measured seed positions)");
    }
    if (table && seeds.size.polydisperse_placement) {
        refuse("Seeds.Size.polydisperse_placement cannot be used with Seeds.source = \"file\"");
    }
    if (seeds.fill_outside_roi != "none" && seeds.fill_outside_roi != "random") {
        refuse("unknown Seeds.fill_outside_roi \"" + seeds.fill_outside_roi + "\"");
    }
    if (seeds.fill_outside_roi == "random" && !seeds.use_roi) {
        refuse("Seeds.fill_outside_roi = \"random\" needs Seeds.roi");
    }
    if ((seeds.use_roi || seeds.fill_outside_roi == "random") && !table) {
        refuse("Seeds.roi and fill_outside_roi need Seeds.source = \"file\"");
    }
    if (seeds.size.distribution == "from_source" && !table) {
        refuse("Seeds.Size.distribution = \"from_source\" needs Seeds.source = \"file\"");
    }
    if (tess.anisotropy_kind == "per_seed" && (!table || tess.metric != MetricType::QUADRIC)) {
        refuse("Tessellation.anisotropy.kind = \"per_seed\" needs Seeds.source = \"file\" and "
               "metric = \"quadric\"");
    }

    // Voxel maps: no seeds and no tessellation, so options that would shape a tessellation are
    // refused rather than ignored
    if (voxel) {
        if (seeds.source == "file" || seeds.placement == SeedPlacement::JOHNSON_MEHL ||
            seeds.size.distribution != "none" || seeds.size.polydisperse_placement ||
            tess.metric != MetricType::VORONOI || tess.anisotropy_kind != "none" ||
            AnyPeriodic(tess) || tess.fit_volumes.enabled || tess.lloyd_iterations > 0 ||
            tess.centroidal_power || options.cleanup.keep_component == "seed" ||
            tess.distance != DistanceMode::EUCLIDEAN) {
            refuse("Source.type = \"voxel_map\" cannot be combined with seed-table, Johnson-Mehl, "
                   "size, metric, anisotropy, periodic, fit_volumes, Lloyd, keep_component = "
                   "\"seed\" or connected / geodesic distance options (no seeds and no "
                   "tessellation)");
        }
        const VoxelMapOptions& map = options.voxel_map;
        if (map.format != "exaconstit" && map.format != "dream3d") {
            refuse("unknown VoxelMap.format \"" + map.format + "\"");
        }
        if (map.quat_layout != "" && map.quat_layout != "wxyz" && map.quat_layout != "xyzw") {
            refuse("unknown VoxelMap.quat_layout \"" + map.quat_layout + "\"");
        }
        return; // the tessellation checks below do not apply
    }

    // Generated-seed combinations
    if (AnyPeriodic(tess) && seeds.placement == SeedPlacement::JOHNSON_MEHL) {
        refuse("placement = \"johnson_mehl\" is not supported with periodic axes");
    }
    if (seeds.size.polydisperse_placement && seeds.use_sampling_box) {
        refuse("polydisperse placement cannot be combined with Seeds.sampling_box");
    }
    if (AnyPeriodic(tess) && (tess.fit_volumes.enabled || tess.lloyd_iterations > 0)) {
        refuse("fit_volumes and lloyd_iterations are not supported with periodic axes");
    }
    if (seeds.placement == SeedPlacement::JOHNSON_MEHL && tess.metric != MetricType::ADDITIVE) {
        refuse("placement = \"johnson_mehl\" needs Tessellation.metric = \"additive\"");
    }

    // Stage 1e distance modes. The labeler check below covers periodic axes and, for geodesic,
    // the metric and anisotropy; the steps that move seeds or weights with straight-line cells
    // (Lloyd, the volume fit) are refused here, as is a solver this build does not have.
    if (tess.distance != DistanceMode::EUCLIDEAN) {
        if (tess.fit_volumes.enabled || tess.lloyd_iterations > 0 || tess.centroidal_power) {
            refuse("Tessellation.distance = \"connected\" / \"geodesic\" cannot be combined "
                   "with fit_volumes, lloyd_iterations or centroidal_power");
        }
    }
    // Values invalid in themselves are refused in any mode, as in validate()
    if (tess.connected_max_rounds < 0 || tess.connected_max_rounds > CONNECTED_MAX_ROUNDS_LIMIT) {
        refuse("Tessellation.connected_max_rounds must be in 0.." +
               std::to_string(CONNECTED_MAX_ROUNDS_LIMIT));
    }
    if (tess.geodesic_solver != "graph" && tess.geodesic_solver != "fim") {
        refuse("unknown Tessellation.geodesic.solver \"" + tess.geodesic_solver + "\"");
    }
    CheckSupportedTessellation(tess);
}

/*
 * MaterialVolume (collective): the volume of the whole (active) mesh, as a reproducible
 * fixed-point sum of the element volumes (CellMeasureAccumulator with every element labeled 0:
 * its single cell is the whole material). Bitwise the same on every rank and for every rank
 * count, so decisions based on it (the seed count, the polydisperse radii) are identical
 * everywhere.
 */
double MaterialVolume(mfem::ParMesh& pmesh, MPI_Comm comm) {
    WeightedSamples samples;
    MakeElementWeightedSamples(pmesh, samples);
    const CellMeasureAccumulator accumulator(comm, samples, 1);
    mfem::Array<int> all_zero(samples.Size());
    all_zero = 0;
    return accumulator.Measure(all_zero).total_volume;
}

/*
 * GrainVolumes (collective): the volume of every grain 1..num_grains as fixed-point sums of its
 * element volumes (CellMeasureAccumulator over one sample per element, cell = grain - 1), so the
 * volumes are bitwise identical for every rank count. Used by AssignPhaseFractions, whose result
 * would otherwise depend on the summation order of a plain floating-point MPI sum.
 */
std::vector<double> GrainVolumes(mfem::ParMesh& pmesh, MPI_Comm comm,
                                 const mfem::Array<int>& element_grain, int num_grains) {
    WeightedSamples samples;
    MakeElementWeightedSamples(pmesh, samples);
    const CellMeasureAccumulator accumulator(comm, samples, num_grains);
    // 0-based cell index of every element (one sample per element, in element order)
    const int ne = element_grain.Size();
    mfem::Array<int> cell(ne);
    const int* grain = element_grain.HostRead();
    for (int e = 0; e < ne; ++e) {
        cell[e] = grain[e] - 1;
    }
    return accumulator.Measure(cell).volume;
}

/** [Seeds.Size] as the parameter struct of morphology/size_targets.hpp. */
SizeTargetParameters MakeSizeParameters(const MicrostructureOptions& options) {
    const SeedSizeOptions& size = options.seeds.size;
    SizeTargetParameters params;
    params.rng_seed = options.seeds.rng_seed;
    params.distribution = SizeDistribution::EQUAL;
    if (size.distribution == "lognormal") {
        params.distribution = SizeDistribution::LOGNORMAL;
        params.sigma = size.sigma;
    }
    // Mean diameter: given directly, or from the mean volume. For a log-normal d with
    // parameters (mu, sigma), E[d^3] = E[d]^3 exp(3 sigma^2), so
    // E[d] = (6 E[V] / pi)^(1/3) exp(-sigma^2); for equal sizes sigma = 0.
    if (size.mean_diameter > 0.0) {
        params.mean_diameter = size.mean_diameter;
    } else if (size.mean_volume > 0.0) {
        params.mean_diameter =
            std::cbrt(6.0 * size.mean_volume / PI) * std::exp(-params.sigma * params.sigma);
    }
    return params;
}

/** True when the run needs per-seed target volumes. */
bool NeedsSizeTargets(const MicrostructureOptions& options) {
    const SeedSizeOptions& size = options.seeds.size;
    return size.distribution != "none" || size.polydisperse_placement ||
           options.tessellation.fit_volumes.enabled || options.tessellation.centroidal_power;
}

/*
 * GenerateJohnsonMehl (collective): the [Seeds.JohnsonMehl] options as JohnsonMehlParameters, the
 * nucleation run, and the parts of the seeding report the tool prints for every placement.
 */
SeedSet GenerateJohnsonMehl(const MicrostructureOptions& options, mfem::ParMesh& pmesh,
                            GenerationReport& report) {
    const JohnsonMehlOptions& jm = options.seeds.johnson_mehl;
    JohnsonMehlParameters params;
    params.nucleation_rate = jm.rate;
    params.growth_speed = jm.growth_speed;
    params.end_time = jm.end_time;
    params.num_nuclei = jm.num_nuclei;
    params.rng_seed = options.seeds.rng_seed;
    params.use_sampling_box = options.seeds.use_sampling_box;
    params.sampling_box_lo = options.seeds.sampling_box_lo;
    params.sampling_box_hi = options.seeds.sampling_box_hi;
    params.allow_stall = options.seeds.allow_fewer_seeds;

    SeedSet seeds = GenerateJohnsonMehlSeeds(pmesh, params, &report.johnson_mehl);
    report.used_johnson_mehl = true;
    report.seeding.requested = jm.num_nuclei;
    report.seeding.achieved = static_cast<std::int64_t>(seeds.Size());
    report.seeding.placement_used = SeedPlacement::JOHNSON_MEHL;
    report.seeding.candidates_generated = report.johnson_mehl.candidates_generated;
    report.seeding.candidates_in_domain = report.johnson_mehl.candidates_in_material;
    report.seeding.sampling_lo = report.johnson_mehl.sampling_lo;
    report.seeding.sampling_hi = report.johnson_mehl.sampling_hi;
    report.seeding.stalled = (report.johnson_mehl.stop == JohnsonMehlStop::STALLED);
    return seeds;
}

/*
 * GeneratePlacedSeeds (collective): rejection / element-weighted / lattice seeds, with the seed
 * count from the mean grain size when num_seeds is 0 and polydisperse RSA when requested.
 *
 * Polydisperse RSA: N slot volumes are drawn from the size distribution keyed by the slot index
 * j = 0..N-1 (the seeds do not exist yet, so their source ids cannot be the key) and normalized
 * to the material volume; sorted from largest to smallest, slot j (the j-th accepted seed) gets
 * the exclusion radius kappa r_(j). The j-th seed's target volume is then the j-th largest
 * volume (returned in `slot_volume`, in seed order), which keeps every seed's target consistent
 * with the exclusion sphere it was placed with.
 */
SeedSet GeneratePlacedSeeds(const MicrostructureOptions& options, mfem::ParMesh& pmesh,
                            MPI_Comm comm, double material_volume,
                            std::vector<double>& slot_volume, GenerationReport& report) {
    SeedingParameters params = MakeSeedingParameters(options);
    const SeedSizeOptions& size = options.seeds.size;

    // Seed count from the mean grain size (identical on every rank: V is a reproducible sum)
    if (params.num_seeds <= 0 && (size.mean_diameter > 0.0 || size.mean_volume > 0.0)) {
        params.num_seeds = EstimateSeedCount(material_volume, MakeSizeParameters(options));
    }

    // Polydisperse RSA: slot volumes, largest first, and their exclusion radii
    slot_volume.clear();
    if (size.polydisperse_placement) {
        std::vector<std::int64_t> slot_id(static_cast<std::size_t>(params.num_seeds));
        std::iota(slot_id.begin(), slot_id.end(), std::int64_t(0));
        const SizeTargets slots =
            MakeSizeTargets(slot_id, material_volume, MakeSizeParameters(options));
        params.rsa_radius.resize(slot_id.size());
        slot_volume.resize(slot_id.size());
        for (std::size_t j = 0; j < slot_id.size(); ++j) {
            const auto slot = static_cast<std::size_t>(slots.placement_order[j]);
            slot_volume[j] = slots.volume[slot];
            params.rsa_radius[j] = size.exclusion_factor * slots.radius[slot];
        }
        params.min_distance = 0.0;
        params.min_distance_factor = 0.0;
        report.used_polydisperse_placement = true;
    }

    SeedSet seeds = GenerateSeeds(pmesh, comm, params, &report.seeding);
    // With allow_fewer_seeds fewer slots may have been filled: keep the targets of those (the
    // largest slots, since they are filled first); MakeSeedTargets rescales them to fill the
    // material
    slot_volume.resize(std::min(slot_volume.size(), seeds.Size()));
    return seeds;
}

/*
 * TableSeeds: the seeds of a seed table plus optional random filler seeds, with the per-seed data
 * the later steps need. The table columns that the tessellation does not read (quaternions,
 * phases) are kept here instead of in the SeedSet, because the filler seeds have none and every
 * per-seed SeedSet array must have one entry per seed (periodic images copy them).
 */
struct TableSeeds {
    SeedSet seeds;                  ///< table rows (kept by the ROI), then the filler seeds
    std::int64_t num_table = 0;     ///< seeds 0..num_table-1 are table rows
    std::vector<double> quat;       ///< 4 per seed (filler rows 0), when the table has quats
    std::vector<char> has_quat;     ///< 1 for table rows with a quaternion, 0 for filler seeds
    std::vector<int> phase;         ///< per seed: table phase, filler_phase or -1 (empty: none)
    std::vector<double> table_volume; ///< table volume of seeds 0..num_table-1 (or empty)
    double roi_volume = 0.0;        ///< V_roi (filler runs only)
    double outside_volume = 0.0;    ///< V_outside (filler runs only)
};

/*
 * ReadTableSeeds (collective): Seeds.source = "file".
 *
 * Steps:
 *  1. ReadSeedTable (rank 0 parses, every rank receives the same table; errors are collective).
 *  2. The filler id offset: max table id + 1 over the whole table (before the ROI drop), so the
 *     filler ids do not change when the ROI is moved and never collide with any table id.
 *  3. ROI: KeepSeedsInBox drops rows outside the closed box (identical on every rank). An empty
 *     result is an error (all ranks, same data).
 *  4. Checks of the data against the options (identical data, so all ranks throw together):
 *     from_source orientations need quaternions, from_phase regions need phases, from_source
 *     sizes need a size column, per_seed anisotropy needs shape columns. Columns the run does not
 *     use are dropped with a warning (weights for Voronoi, shape tensors without per_seed).
 *     Rows are kept in the closed ROI box, while the filler's V_roi counts material strictly
 *     inside it (a row exactly on a box face is kept; material on the face counts as outside).
 *  5. Filler seeds (fill_outside_roi = "random"): FillOutsideRoi with the ROI as exclusion box,
 *     rejection sampler, ids from the offset. They get the mean table weight when the table has
 *     weights (see step 6) and,
 *     with per_seed tensors, A = I / r_f^2 where r_f is the equivalent-sphere radius of the equal
 *     share V_f = V_outside / N_filler of the outside material: then f_i(x) = |x - s|^2 / r_f^2
 *     equals 1 on the sphere of volume V_f, the same normalization as a table seed's
 *     (x - s)^T M^-1 (x - s) = 1 on its measured ellipsoid.
 *  6. The combined SeedSet: table rows first (seed indices 0..num_table-1), then the filler.
 */
TableSeeds ReadTableSeeds(const MicrostructureOptions& options, mfem::ParMesh& pmesh,
                          MPI_Comm comm, GenerationReport& report) {
    const SeedOptions& opts = options.seeds;
    const TessellationOptions& tess = options.tessellation;

    // Step 1: the table, in mesh units
    SeedTableParameters table_params;
    table_params.path = opts.file;
    table_params.units_scale = opts.units_scale;
    table_params.translate = opts.translate;
    SeedTable table = ReadSeedTable(table_params, comm);
    report.used_seed_table = true;
    report.num_table_rows = static_cast<std::int64_t>(table.seeds.Size());
    report.table_has_quat = table.has_quat;
    report.table_has_phase = table.has_phase;
    report.table_has_weight = table.has_weight;
    report.table_has_volume = table.has_volume;
    report.table_has_shape = table.has_shape;
    report.table_ignored_columns = table.ignored_columns;

    // Step 2: filler id offset from the whole table (ReadSeedTable refuses an empty table)
    std::int64_t max_id = 0;
    for (const std::int64_t id : table.seeds.source_id) {
        max_id = std::max(max_id, id);
    }

    // Step 3: region of interest
    if (opts.use_roi) {
        report.used_roi = true;
        report.roi_lo = opts.roi_lo;
        report.roi_hi = opts.roi_hi;
        report.num_table_dropped = KeepSeedsInBox(table, opts.roi_lo, opts.roi_hi);
    }
    const auto num_table = static_cast<std::int64_t>(table.seeds.Size());
    report.num_table_seeds = num_table;
    if (num_table == 0) {
        throw std::runtime_error("GenerateMicrostructure: no row of the seed table '" +
                                 opts.file.string() + "' lies inside Seeds.roi");
    }

    // Step 4: the data the options need, and the columns this run does not use
    if (options.orientations.source == "from_source" && !table.has_quat) {
        throw std::invalid_argument("GenerateMicrostructure: Orientations.source = "
                                    "\"from_source\" needs the qw, qx, qy, qz columns in the seed "
                                    "table '" + opts.file.string() + "'");
    }
    if (options.regions.mode == "from_phase" && !table.has_phase) {
        throw std::invalid_argument("GenerateMicrostructure: Regions.mode = \"from_phase\" needs "
                                    "a phase column in the seed table '" + opts.file.string() +
                                    "'");
    }
    if (opts.size.distribution == "from_source" && !table.has_volume) {
        throw std::invalid_argument("GenerateMicrostructure: Seeds.Size.distribution = "
                                    "\"from_source\" needs a volume, radius or diameter column in "
                                    "the seed table '" + opts.file.string() + "'");
    }
    const bool per_seed = (tess.anisotropy_kind == "per_seed");
    if (per_seed && !table.has_shape) {
        throw std::invalid_argument("GenerateMicrostructure: Tessellation.anisotropy.kind = "
                                    "\"per_seed\" needs the shape columns m11, m22, m33, m23, m13, "
                                    "m12 in the seed table '" + opts.file.string() + "'");
    }
    if (table.has_shape && !per_seed) {
        Warn(comm, report, "the shape columns of the seed table are ignored (they are used only "
                           "with Tessellation.anisotropy.kind = \"per_seed\")");
        table.seeds.shape.clear();
    }
    if (table.has_weight && tess.metric == MetricType::VORONOI) {
        Warn(comm, report, "the weight column of the seed table is ignored by the Voronoi metric");
        table.seeds.weight.clear();
    }
    if (!table.ignored_columns.empty()) {
        std::string list;
        for (const std::string& name : table.ignored_columns) {
            list += (list.empty() ? "" : ", ") + name;
        }
        report.warnings.push_back("ignored seed table columns: " + list);
    }

    TableSeeds out;
    out.num_table = num_table;
    out.table_volume = table.volume;

    // Step 5: filler seeds outside the ROI
    SeedSet filler;
    if (opts.fill_outside_roi == "random") {
        RoiFillerParameters filler_params;
        filler_params.lo = opts.roi_lo;
        filler_params.hi = opts.roi_hi;
        filler_params.num_filler = opts.filler_num_seeds;
        filler_params.rng_seed = opts.rng_seed;
        filler_params.source_id_offset = max_id + 1;
        filler_params.min_distance_factor = opts.filler_min_distance_factor;
        RoiFillerResult filled = FillOutsideRoi(pmesh, comm, filler_params, num_table);
        filler = std::move(filled.filler);
        out.roi_volume = filled.roi_material_volume;
        out.outside_volume = filled.outside_material_volume;
        report.used_filler = true;
        report.num_filler_requested = filled.num_requested;
        report.num_filler_seeds = static_cast<std::int64_t>(filler.Size());
        report.filler_source_id_offset = max_id + 1;
        report.filler_seeding = filled.report;
        report.roi_material_volume = filled.roi_material_volume;
        report.outside_material_volume = filled.outside_material_volume;
    }
    const std::size_t nt = table.seeds.Size();
    const std::size_t nf = filler.Size();

    // Step 6: combined seed set, table rows first
    SeedSet& seeds = out.seeds;
    seeds.x = table.seeds.x;
    seeds.y = table.seeds.y;
    seeds.z = table.seeds.z;
    seeds.source_id = table.seeds.source_id;
    seeds.x.insert(seeds.x.end(), filler.x.begin(), filler.x.end());
    seeds.y.insert(seeds.y.end(), filler.y.begin(), filler.y.end());
    seeds.z.insert(seeds.z.end(), filler.z.begin(), filler.z.end());
    seeds.source_id.insert(seeds.source_id.end(), filler.source_id.begin(),
                           filler.source_id.end());
    // Metric weight: table weights, and for the filler seeds the mean table weight. Weighted
    // metrics only see weight differences, so a filler weight of 0 next to table weights with a
    // large common offset C (e.g. w = r^2 + C) would empty every filler grain; the mean makes a
    // filler seed compete like an average table grain (the automatic filler count also gives it
    // the average table grain volume). With size targets and the Laguerre metric LabelFromSeeds
    // replaces this by the filler seed's target weight plus the mean table offset.
    if (!table.seeds.weight.empty()) {
        double mean_weight = 0.0;
        for (std::size_t i = 0; i < nt; ++i) {
            mean_weight += table.seeds.weight[i];
        }
        mean_weight /= static_cast<double>(nt);
        seeds.weight = table.seeds.weight;
        seeds.weight.resize(nt + nf, mean_weight);
    }
    // Per-seed Quadric tensors: table A = M^-1, filler A = I / r_f^2 (see step 5)
    if (per_seed) {
        seeds.shape = table.seeds.shape;
        if (nf > 0) {
            const double v_f = out.outside_volume / static_cast<double>(nf);
            const double r_f = std::cbrt(3.0 * v_f / (4.0 * PI));
            const double a = 1.0 / (r_f * r_f);
            if (!(a > 0.0) || !std::isfinite(a)) {
                throw std::runtime_error("GenerateMicrostructure: cannot build the filler seeds' "
                                         "Quadric tensors (outside volume " +
                                         std::to_string(out.outside_volume) + ")");
            }
            for (std::size_t j = 0; j < nf; ++j) {
                const double row[6] = {a, a, a, 0.0, 0.0, 0.0};
                seeds.shape.insert(seeds.shape.end(), row, row + 6);
            }
        }
    }
    // Quaternions (table rows as read; normalized when written per grain) and phases, kept
    // outside the SeedSet
    if (table.has_quat) {
        out.quat = table.seeds.quat;
        out.quat.resize(4 * (nt + nf), 0.0);
        out.has_quat.assign(nt + nf, 0);
        std::fill(out.has_quat.begin(), out.has_quat.begin() + static_cast<std::ptrdiff_t>(nt), 1);
    }
    if (table.has_phase || options.regions.has_filler_phase) {
        out.phase.assign(nt + nf, -1);
        for (std::size_t i = 0; i < nt && table.has_phase; ++i) {
            out.phase[i] = table.seeds.phase[i];
        }
        for (std::size_t j = 0; j < nf; ++j) {
            out.phase[nt + j] =
                options.regions.has_filler_phase ? options.regions.filler_phase : -1;
        }
    }

    // The seeding report the tool prints for every run: table seeds plus requested filler seeds
    report.seeding.requested = num_table + report.num_filler_requested;
    report.seeding.achieved = static_cast<std::int64_t>(seeds.Size());
    report.seeding.placement_used = SeedPlacement::FILE;
    return out;
}

/*
 * MakeSeedTargets: the target volume of every seed (seed order), normalized to the material.
 *  - polydisperse placement: the slot volumes the seeds were placed with;
 *  - distribution "from_source" (seed tables): the table volumes. With filler seeds the table
 *    targets are scaled to sum to the material volume inside the ROI and every filler seed gets
 *    V_outside / N_filler; without filler the table grains fill the whole material, so their
 *    volumes are scaled to it;
 *  - otherwise: drawn from the distribution keyed by each seed's source id (EQUAL without a
 *    distribution, for a plain volume fit).
 * The result is finally normalized to `material_volume` (MakeSizeTargetsFromVolumes). Without
 * filler that normalization does the whole scaling of the table volumes; with filler it only
 * removes the tiny difference between the quadrature-point volumes of the ROI split and the
 * element volume sum. Host only; replicated inputs give identical targets on every rank.
 */
SizeTargets MakeSeedTargets(const MicrostructureOptions& options, const SeedSet& seeds,
                            double material_volume, const std::vector<double>& slot_volume,
                            const TableSeeds* table) {
    if (options.seeds.size.polydisperse_placement) {
        return MakeSizeTargetsFromVolumes(slot_volume, material_volume);
    }
    if (options.seeds.size.distribution == "from_source" && table != nullptr) {
        const auto nt = static_cast<std::size_t>(table->num_table);
        const std::size_t nf = seeds.Size() - nt;
        std::vector<double> relative(table->table_volume.begin(),
                                     table->table_volume.begin() + static_cast<std::ptrdiff_t>(nt));
        if (options.seeds.fill_outside_roi == "random") {
            double table_sum = 0.0;
            for (const double v : relative) {
                table_sum += v;
            }
            for (double& v : relative) {
                v *= table->roi_volume / table_sum;
            }
            for (std::size_t j = 0; j < nf; ++j) {
                relative.push_back(table->outside_volume / static_cast<double>(nf));
            }
        }
        return MakeSizeTargetsFromVolumes(relative, material_volume);
    }
    return MakeSizeTargets(seeds.source_id, material_volume, MakeSizeParameters(options));
}

/*
 * RelaxAndFit (collective): Lloyd relaxation and/or the volume fit, on the same sample points
 * (quadrature points or element centroids, Tessellation.fit_volumes.samples).
 *  - Lloyd: iterations with the user's metric (Voronoi or Laguerre with the current weights); in
 *    centroidal power mode every step is paired with a Laguerre refit to the targets.
 *  - Volume fit (not in centroidal power mode, which already refits): FitWeights from the
 *    current weights (table weights or the Laguerre starting weights w = r^2 when present); the
 *    fitted weights replace seeds.weight.
 */
void RelaxAndFit(const MicrostructureOptions& options, mfem::ParMesh& pmesh, MPI_Comm comm,
                 const std::vector<double>& target_volume, SeedSet& seeds,
                 GenerationReport& report) {
    const TessellationOptions& tess = options.tessellation;
    const bool lloyd = tess.lloyd_iterations > 0;
    const bool fit = tess.fit_volumes.enabled && !tess.centroidal_power;
    if (!lloyd && !fit) {
        return;
    }

    // Sample points of the material (shared by Lloyd and the fit)
    WeightedSamples samples;
    if (tess.fit_volumes.samples == "element") {
        MakeElementWeightedSamples(pmesh, samples);
    } else {
        MakeQuadratureWeightedSamples(pmesh, -1, samples);
    }

    // Solver controls of the fit (also used by the refits of centroidal power mode)
    WeightFitParameters fit_params;
    fit_params.metric = (tess.metric == MetricType::VORONOI) ? MetricType::LAGUERRE : tess.metric;
    fit_params.setup.global_tensor = EffectiveGlobalTensor(tess);
    fit_params.num_bins = tess.num_bins;
    fit_params.tolerance = tess.fit_volumes.tolerance;
    fit_params.max_iterations = tess.fit_volumes.max_iterations;

    if (lloyd) {
        LloydParameters lloyd_params;
        lloyd_params.iterations = tess.lloyd_iterations;
        lloyd_params.metric = tess.metric;
        lloyd_params.num_bins = tess.num_bins;
        lloyd_params.centroidal_power = tess.centroidal_power;
        lloyd_params.weight_fit = fit_params;
        lloyd_params.weight_fit.metric = MetricType::LAGUERRE;
        seeds = RunLloyd(pmesh, comm, samples, seeds, lloyd_params, target_volume, &report.lloyd);
        report.ran_lloyd = true;
    }
    if (fit) {
        const WeightFitResult result =
            FitWeights(comm, samples, seeds, target_volume, seeds.weight, fit_params);
        seeds.weight = result.weight;
        report.weight_fit = result.report;
        report.ran_weight_fit = true;
    }
}

/*
 * SeedAnchorKeys (collective): for KeepRule::SEED_ELEMENT, the element key of the element that
 * contains each grain's seed and carries that grain's label.
 *
 * Each rank tests, for every local element e labeled g, whether grain g's seed lies in e (element
 * box first, then the exact point-in-element test of MaterialPointLocator). The smallest key
 * found per grain is reduced with MPI_Allreduce(MIN) on unsigned 64-bit integers, so every rank
 * gets the same anchors; a grain whose seed is in no element of its own label keeps the value
 * UINT64_MAX, which matches no element in practice, so the cleanup falls back to the largest
 * piece for it. The cost is one box test per local element (plus exact tests for the few
 * elements whose box contains their grain's seed).
 */
std::vector<std::uint64_t> SeedAnchorKeys(mfem::ParMesh& pmesh, MPI_Comm comm,
                                          const SeedSet& seeds, const TessellationLabels& labels) {
    const std::uint64_t none = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> anchor(static_cast<std::size_t>(labels.num_grains), none);
    const int ne = pmesh.GetNE();
    if (ne > 0) {
        const MaterialPointLocator locator(pmesh);
        const int* grain = labels.element_grain.HostRead();
        for (int e = 0; e < ne; ++e) {
            const auto g = static_cast<std::size_t>(grain[e] - 1);
            const auto s = static_cast<std::size_t>(labels.grain_seed[g]);
            const std::array<double, 3> p = {seeds.x[s], seeds.y[s], seeds.z[s]};
            const auto& box = locator.ElementBox(e);
            bool in_box = true;
            for (int d = 0; d < 3; ++d) {
                in_box = in_box && p[static_cast<std::size_t>(d)] >= box.getMin()[d] &&
                         p[static_cast<std::size_t>(d)] <= box.getMax()[d];
            }
            if (in_box && locator.PointInElement(e, p)) {
                anchor[g] = std::min(anchor[g], ElementKey(pmesh, e));
            }
        }
    }
    if (!anchor.empty()) {
        MPI_Allreduce(MPI_IN_PLACE, anchor.data(), static_cast<int>(anchor.size()), MPI_UINT64_T,
                      MPI_MIN, comm);
    }
    return anchor;
}

/*
 * CountCoincidentSeeds: number of seeds whose exact position (x, y, z) equals that of a
 * lower-index seed, the quantity the seed search reports for the straight-line labels
 * (TessellationLabels::num_coincident_seeds). Used for geodesic runs, which do not build a seed
 * search. Host only; replicated seeds give the same count on every rank.
 */
int CountCoincidentSeeds(const SeedSet& seeds) {
    std::set<std::array<double, 3>> seen;
    int count = 0;
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        // insert(...).second is false when the same position was inserted before
        if (!seen.insert({seeds.x[i], seeds.y[i], seeds.z[i]}).second) {
            ++count;
        }
    }
    return count;
}

/*
 * StraightLineLabels (collective): LabelTessellation with distance = EUCLIDEAN, i.e. the plain
 * metric labels (nearest seed in straight lines, centroid or quadrature vote). The labeler refuses
 * the other distance modes, so a copy of the options with the mode reset is passed. Used as the
 * starting labels of the connected mode and as the fallback for unreached geodesic elements.
 */
TessellationLabels StraightLineLabels(mfem::ParMesh& pmesh, MPI_Comm comm, const SeedSet& seeds,
                                      const TessellationOptions& tess) {
    TessellationOptions euclidean = tess;
    euclidean.distance = DistanceMode::EUCLIDEAN;
    return LabelTessellation(pmesh, comm, seeds, euclidean);
}

/*
 * LabelGeodesic (collective): the geodesic solver selected by Tessellation.geodesic_solver:
 * "fim" -> LabelGeodesicFim (labeling/geodesic_fim.hpp, Fast Iterative Method eikonal solver, the
 * default), "graph" -> LabelGeodesicGraph (labeling/geodesic_graph.hpp, shortest paths along the
 * element edges and diagonals). Both take the same arguments and return the same GeodesicLabels
 * (labeling/geodesic_common.hpp). Additive offsets a_i (the seed weights, e.g. -v t_i for
 * Johnson-Mehl nuclei) are used when the metric is "additive"; with "voronoi" every a_i = 0.
 * Sampling (centroid or vote) follows Tessellation.sampling / vote_levels, as for the
 * straight-line labels.
 */
GeodesicLabels LabelGeodesic(const TessellationOptions& tess, mfem::ParMesh& pmesh, MPI_Comm comm,
                             const SeedSet& seeds) {
    GeodesicParameters params;
    params.sampling = tess.sampling;
    params.vote_levels = tess.vote_levels;
    const bool additive = (tess.metric == MetricType::ADDITIVE);
    if (tess.geodesic_solver == "graph") {
        return LabelGeodesicGraph(pmesh, comm, seeds, additive, params);
    }
    if (tess.geodesic_solver == "fim") {
        return LabelGeodesicFim(pmesh, comm, seeds, additive, params);
    }
    throw std::invalid_argument("GenerateMicrostructure: Tessellation.geodesic.solver = \"" +
                                tess.geodesic_solver + "\" is not available in this build");
}

/*
 * DistanceModeLabels: the result of steps 6-8 for any distance mode.
 */
struct DistanceModeLabels {
    /** Compact labels over the original seeds (TessellationLabels as from LabelTessellation). */
    TessellationLabels labels;
    /**
     * Connected / geodesic modes: per seed, the element key of the element the mode grew the
     * seed's grain from (connected: the anchor element; geodesic: the source element). Empty for
     * the Euclidean mode (which uses SeedAnchorKeys for keep_component = "seed").
     */
    std::vector<std::uint64_t> seed_anchor_key;
    /** Elements cut by a grain boundary in the vote, counted on this rank (summed later). */
    long long local_split_elements = 0;
};

/*
 * LabelByDistanceMode (collective): steps 6-8 by Tessellation.distance.
 *
 *  - "euclidean": LabelTessellation as before (periodic images, centroid or vote, compaction).
 *  - "connected":
 *     1. the straight-line labels (StraightLineLabels; validated: no periodic axes, so no images);
 *     2. ConstrainToConnectedCells with connected_max_rounds and the same sampling: orphans
 *        (elements cut off from their seed's anchor element through material) move to the next
 *        best seed, round by round (labeling/connected_labeler.hpp);
 *     3. CompactSeedLabels: seeds that lost every element disappear from the grain list.
 *     Orphans left after the last round and bodies without any anchor keep their labels; the
 *     fragment cleanup (step 9) merges or splits them. The vote-split count is that of the
 *     straight-line vote (step 1).
 *  - "geodesic":
 *     1. LabelGeodesic (path lengths inside the active material);
 *     2. elements left with element_seed = -1 take their straight-line nearest-seed label. Two
 *        kinds, counted separately: unreached elements (no seed reaches them: a body without a
 *        seed; the cleanup then splits each such body off as a grain of its own, since an
 *        isolated body has no face neighbor to merge into) and unlabeled elements (reached, but
 *        their winning vertex has no label: only on a degenerate mesh). The decision to compute
 *        the straight-line labels uses the global sum of both counts, identical on every rank,
 *        so either every rank or none calls the collective LabelTessellation;
 *     3. CompactSeedLabels.
 *     The vote-split count of the solver is already a global sum, so it is stored on rank 0
 *     only and the caller's global sum restores it exactly.
 *
 * Every step is partition independent (see the called functions), so the labels are the same for
 * every rank count.
 */
DistanceModeLabels LabelByDistanceMode(const MicrostructureOptions& options, mfem::ParMesh& pmesh,
                                       MPI_Comm comm, const SeedSet& seeds,
                                       const MeshAdjacency& adjacency, GenerationReport& report) {
    const TessellationOptions& tess = options.tessellation;
    const int num_seeds = static_cast<int>(seeds.Size());
    report.distance = tess.distance;
    DistanceModeLabels out;

    // Plain straight-line labels: unchanged Stage 1a-1d path
    if (tess.distance == DistanceMode::EUCLIDEAN) {
        out.labels = LabelTessellation(pmesh, comm, seeds, tess);
        out.local_split_elements = out.labels.num_split_elements;
        return out;
    }

    if (tess.distance == DistanceMode::CONNECTED) {
        // Step 1: straight-line labels as the starting point (0-based seed index per element)
        const TessellationLabels straight = StraightLineLabels(pmesh, comm, seeds, tess);
        mfem::Array<int> element_seed;
        element_seed = straight.element_seed; // deep copy (mfem::Array::operator=, array.hpp:121)

        // Step 2: keep only what is connected to each seed's anchor element through material
        ConnectedLabelParameters params;
        params.max_rounds = tess.connected_max_rounds;
        params.sampling = tess.sampling;
        params.vote_levels = tess.vote_levels;
        report.connected =
            ConstrainToConnectedCells(pmesh, comm, adjacency, seeds, tess, element_seed, params);
        report.used_connected = true;

        // Step 3: compact grain ids over the seeds that still own elements
        CompactSeedLabels(comm, element_seed, num_seeds, out.labels);
        out.labels.num_coincident_seeds = straight.num_coincident_seeds;
        out.labels.num_split_elements = straight.num_split_elements;
        out.local_split_elements = straight.num_split_elements;

        // Anchors for the cleanup's keep rule: the anchor elements the rounds used
        // (ConnectedAnchorKeys of the straight-line input labels: the element containing the seed
        // if it carries the seed's label, else the nearest element centroid with that label;
        // UINT64_MAX for a seed that labeled no element), returned in the report
        out.seed_anchor_key = report.connected.anchor_key;
        return out;
    }

    // Geodesic. Johnson-Mehl: the nucleation step (seeding/johnson_mehl.cpp) decided which
    // nuclei survive with straight-line growth spheres, so a nucleus may have been discarded
    // because an earlier grain's sphere covered it across a void, although that grain would reach
    // it later (or never) along material paths. Only the labels below use path lengths; say so.
    if (options.seeds.placement == SeedPlacement::JOHNSON_MEHL) {
        Warn(comm, report, "Johnson-Mehl with Tessellation.distance = \"geodesic\": nucleation "
                           "still discards a candidate covered by an earlier grain's "
                           "straight-line growth sphere, even across a void; only the final "
                           "labels use path lengths inside the material");
    }

    // Step 1: path-length labels (the VertexField inside `geodesic` refers to pmesh and dies at
    // the end of this function). Paths connect elements through shared vertices, edges and faces;
    // the face-based cleanup (step 9) handles pieces attached only through an edge or vertex.
    GeodesicLabels geodesic = LabelGeodesic(tess, pmesh, comm, seeds);
    report.used_geodesic = true;
    report.geodesic_solver = tess.geodesic_solver;
    report.geodesic = geodesic.report;
    report.num_unreached_elements = geodesic.report.unreached_elements;
    report.num_unlabeled_elements = geodesic.report.unlabeled_elements;
    mfem::Array<int> element_seed;
    element_seed = geodesic.element_seed;

    // Step 2: unreached and unlabeled elements (element_seed = -1) take their straight-line
    // label (global counts: every rank takes the same branch, so the collective labeling below is
    // called by all or none)
    int num_coincident = 0;
    if (geodesic.report.unreached_elements + geodesic.report.unlabeled_elements > 0) {
        const TessellationLabels straight = StraightLineLabels(pmesh, comm, seeds, tess);
        num_coincident = straight.num_coincident_seeds;
        const int ne = element_seed.Size();
        const int* fallback = straight.element_seed.HostRead();
        int* label = element_seed.HostReadWrite();
        for (int e = 0; e < ne; ++e) {
            if (label[e] < 0) {
                label[e] = fallback[e];
            }
        }
    } else {
        num_coincident = CountCoincidentSeeds(seeds);
    }

    // Step 3: compact grain ids
    CompactSeedLabels(comm, element_seed, num_seeds, out.labels);
    out.labels.num_coincident_seeds = num_coincident;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    out.local_split_elements =
        (rank == 0) ? static_cast<long long>(geodesic.report.split_elements) : 0;
    out.labels.num_split_elements = static_cast<int>(out.local_split_elements);
    out.seed_anchor_key = geodesic.sources.source_element_key;
    return out;
}

/*
 * CleanupParameters: [Cleanup] as FragmentCleanupParameters. "auto" splits large fragments and
 * merges tiny ones (SPLIT_LARGE_MERGE_TINY) for grains grown from generated seeds (random,
 * lattice and Johnson-Mehl), and merges every fragment of data-driven grains (seed tables and
 * voxel maps), whose grain list is fixed by the measurement.
 */
FragmentCleanupParameters CleanupParameters(const MicrostructureOptions& options) {
    FragmentCleanupParameters params;
    const std::string& fragments = options.cleanup.fragments;
    if (fragments == "split") {
        params.policy = FragmentPolicy::SPLIT;
    } else if (fragments == "merge") {
        params.policy = FragmentPolicy::MERGE;
    } else if (fragments == "none") {
        params.policy = FragmentPolicy::NONE;
    } else if (!IsDataDriven(options)) {
        params.policy = FragmentPolicy::SPLIT_LARGE_MERGE_TINY; // auto, generated seeds
    } else {
        params.policy = FragmentPolicy::MERGE; // auto, data-driven grains
    }
    params.keep = (options.cleanup.keep_component == "seed") ? KeepRule::SEED_ELEMENT
                                                             : KeepRule::LARGEST;
    params.min_grain_elements = options.cleanup.min_grain_elements;
    params.min_fragment_elements = options.cleanup.min_fragment_elements;
    params.min_fragment_fraction = options.cleanup.min_fragment_fraction;
    return params;
}

/*
 * ActiveGrains: the grains of the active mesh after cleanup, with everything the orientation,
 * phase and region steps need per grain. Per-grain vectors have num_grains entries (index g-1);
 * element_grain describes the local elements of the active mesh.
 */
struct ActiveGrains {
    mfem::Array<int> element_grain;            ///< local active elements, grain 1..num_grains
    int num_grains = 0;
    std::vector<std::int64_t> source_id;       ///< stable id (seed, feature or derived)
    std::vector<double> source_quat;           ///< 4 per grain: measured orientation (or 0)
    std::vector<char> has_source_quat;         ///< 1 when source_quat holds a measured one
    std::vector<int> phase;                    ///< phase or -1
    std::vector<char> filler;                  ///< 1 for filler grains
    std::vector<int> seed;                     ///< parent seed index, -1 for voxel grains
    std::vector<std::int64_t> feature;         ///< voxel feature id, -1 for seed grains
    std::shared_ptr<MeshAdjacency> adjacency;  ///< face adjacency of the active mesh
    /** Quadrature vote: local elements cut by a grain boundary (summed by the caller). */
    long long local_split_elements = 0;
};

/*
 * LabelFromSeeds (collective): steps 3-9 for a tessellation on the active mesh `pmesh`.
 *
 *  3. Material volume when needed, then the seeds: a seed table (+ ROI + filler), Johnson-Mehl
 *     nucleation, or generated seeds; then the per-seed size targets. Laguerre seeds start from
 *     w_i = r_i^2, except when a seed table supplies weights (the table weights are explicit data
 *     and are kept; a warning says the starting weights are not used); the filler seeds then get
 *     their own r_j^2 plus the mean over the table rows of (table weight - r_i^2).
 *  5. Lloyd relaxation and/or the volume fit.
 *  6-8. LabelByDistanceMode: Euclidean labels (LabelTessellation: periodic images, centroid or
 *     vote labels, compaction), or the connected / geodesic labels built from them.
 *  9. Fragment cleanup on the (periodic-aware) face adjacency, which is built before step 6
 *     because the connected mode needs it too.
 * Finally the per-grain tables: parent seed, source id, measured orientation and phase of the
 * parent seed (seed tables), and the filler flag (parent seed index >= number of table rows).
 */
ActiveGrains LabelFromSeeds(const MicrostructureOptions& options, mfem::ParMesh& pmesh,
                            MPI_Comm comm, GenerationReport& report) {
    const TessellationOptions& tess = options.tessellation;
    const bool table_source = (options.seeds.source == "file");

    // Step 3a: the material volume, when the seed count, the size targets or the report need it
    const bool johnson_mehl = (options.seeds.placement == SeedPlacement::JOHNSON_MEHL);
    const bool size_targets = NeedsSizeTargets(options);
    const bool count_from_size = !table_source && options.seeds.num_seeds <= 0 &&
                                 (options.seeds.size.mean_diameter > 0.0 ||
                                  options.seeds.size.mean_volume > 0.0);
    double material_volume = 0.0;
    if (size_targets || count_from_size || johnson_mehl) {
        material_volume = MaterialVolume(pmesh, comm);
    }
    report.material_volume = material_volume;

    // Step 3b: seeds, identical on every rank
    SeedSet seeds;
    std::vector<double> slot_volume;
    std::unique_ptr<TableSeeds> table;
    if (table_source) {
        table = std::make_unique<TableSeeds>(ReadTableSeeds(options, pmesh, comm, report));
        seeds = table->seeds;
    } else if (johnson_mehl) {
        seeds = GenerateJohnsonMehl(options, pmesh, report);
        report.seeding.material_volume = material_volume;
    } else {
        seeds = GeneratePlacedSeeds(options, pmesh, comm, material_volume, slot_volume, report);
    }
    if (seeds.Size() == 0) {
        // Only possible when stalls are allowed (or a nucleation run ends before its first
        // nucleus); decided from the replicated seed set, so every rank throws together.
        throw std::runtime_error(
            "GenerateMicrostructure: no seed could be placed in the material\n" +
            report.seeding.Summary());
    }

    // Step 3c: size targets per seed; Laguerre seeds start from w_i = r_i^2 unless the seed
    // table supplied weights
    std::vector<double> target_volume;
    if (size_targets) {
        const SizeTargets targets =
            MakeSeedTargets(options, seeds, material_volume, slot_volume, table.get());
        target_volume = targets.volume;
        if (tess.metric == MetricType::LAGUERRE) {
            if (seeds.weight.empty()) {
                seeds.weight = targets.initial_weight;
            } else {
                Warn(comm, report, "the seed table weights are used as the Laguerre weights; the "
                                   "size targets' starting weights w = r^2 are not used for the "
                                   "table seeds");
                // Filler seeds: their own target weight shifted by the mean offset of the table
                // weights from the table targets' weights, so a common offset C of the table
                // weights is carried over (only weight differences matter)
                if (table && seeds.Size() > static_cast<std::size_t>(table->num_table)) {
                    const auto nt = static_cast<std::size_t>(table->num_table);
                    double offset = 0.0;
                    for (std::size_t i = 0; i < nt; ++i) {
                        offset += seeds.weight[i] - targets.initial_weight[i];
                    }
                    offset /= static_cast<double>(nt);
                    for (std::size_t j = nt; j < seeds.Size(); ++j) {
                        seeds.weight[j] = targets.initial_weight[j] + offset;
                    }
                }
            }
        }
    }

    // Step 5: Lloyd relaxation and/or the volume fit (validated: not with periodic axes)
    RelaxAndFit(options, pmesh, comm, target_volume, seeds, report);

    // Face adjacency of the active mesh, used by the connected mode (step 7) and the fragment
    // cleanup (step 9); with periodic axes it pairs matching faces on opposite planes of the
    // mesh box, so wrapped grains stay one grain (periodic axes are refused with the connected
    // and geodesic modes)
    ActiveGrains out;
    if (AnyPeriodic(tess)) {
        const PeriodicBox box = PeriodicBoxFromMesh(pmesh, comm, tess.periodic);
        PeriodicFaceParameters pairing;
        pairing.periodic = box.periodic;
        pairing.lo = box.lo;
        pairing.hi = box.hi;
        pairing.relative_tolerance = tess.periodic_tolerance;
        out.adjacency = std::make_shared<MeshAdjacency>(pmesh, pairing);
        report.periodic_faces = out.adjacency->PeriodicReport();
    } else {
        out.adjacency = std::make_shared<MeshAdjacency>(pmesh);
    }

    // Steps 6-8: labels and compact grain ids (collective), by distance mode. With periodic axes
    // the labeler builds the periodic images itself and maps their labels back to the original
    // seeds.
    const DistanceModeLabels mode =
        LabelByDistanceMode(options, pmesh, comm, seeds, *out.adjacency, report);
    const TessellationLabels& labels = mode.labels;
    report.periodic_layers = labels.periodic_layers;
    report.num_periodic_images = labels.num_periodic_images;
    report.num_labeled_grains = labels.num_grains;
    report.num_coincident_seeds = labels.num_coincident_seeds;
    report.num_seeds = static_cast<std::int64_t>(seeds.Size());
    report.num_empty_seeds = report.num_seeds - labels.num_grains;
    report.seed_target_volume = target_volume;
    report.seeds = seeds;
    out.local_split_elements = mode.local_split_elements;

    // Step 9: fragment cleanup over the face adjacency.
    // Keep rule. The connected and geodesic modes always keep, for each grain, the piece that
    // holds the element the mode grew the seed's grain from (connected: the anchor element;
    // geodesic: the source element), whatever Cleanup.keep_component says: in these modes that
    // piece *is* the seed's grain, and the other pieces are leftovers (unresolved orphans, or a
    // body without a seed that took a straight-line label). With "largest", a larger leftover
    // body would take over the seed's identity and its real piece would be merged away or
    // renamed. The Euclidean mode keeps Cleanup.keep_component; for "seed" it uses
    // SeedAnchorKeys (the element of the grain that contains the seed), unchanged from Stage 1b
    // so earlier results stay bitwise the same. (The two anchors differ only for a seed exactly
    // on a face between grains and for seeds outside the material.) SeedAnchorKeys is kept for the
    // Euclidean mode rather than replaced by ConnectedAnchorKeys: for a seed whose containing
    // element went to a neighbor, SeedAnchorKeys has no anchor (largest piece kept) while
    // ConnectedAnchorKeys would pick the nearest same-label element, which can change which piece
    // keeps a grain's identity in existing Stage 1b-1d runs. A grain whose anchor element carries
    // another label at the end (a seed without a cell) falls back to the largest piece inside
    // CleanupFragments.
    FragmentCleanupParameters cleanup_params = CleanupParameters(options);
    if (!mode.seed_anchor_key.empty()) {
        cleanup_params.keep = KeepRule::SEED_ELEMENT;
    }
    if (cleanup_params.keep == KeepRule::SEED_ELEMENT) {
        if (!mode.seed_anchor_key.empty()) {
            cleanup_params.grain_anchor_key.resize(labels.grain_seed.size());
            for (std::size_t g = 0; g < labels.grain_seed.size(); ++g) {
                const auto s = static_cast<std::size_t>(labels.grain_seed[g]);
                cleanup_params.grain_anchor_key[g] = mode.seed_anchor_key[s];
            }
        } else {
            cleanup_params.grain_anchor_key = SeedAnchorKeys(pmesh, comm, seeds, labels);
        }
    }
    std::vector<std::int64_t> labeled_source_id(labels.grain_seed.size());
    for (std::size_t g = 0; g < labels.grain_seed.size(); ++g) {
        labeled_source_id[g] = seeds.source_id[static_cast<std::size_t>(labels.grain_seed[g])];
    }
    FragmentCleanupResult cleaned = CleanupFragments(
        *out.adjacency, labels.element_grain, labels.num_grains, labeled_source_id, cleanup_params);
    report.cleanup = cleaned.report;

    // Per-grain tables. Grain g descends from input grain grain_parent[g-1], whose seed is
    // labels.grain_seed[parent - 1]; its measured data (orientation, phase) are the seed's.
    const auto num_grains = static_cast<std::size_t>(cleaned.num_grains);
    out.element_grain = cleaned.element_grain;
    out.num_grains = cleaned.num_grains;
    out.source_id = cleaned.grain_source_id;
    out.seed.resize(num_grains);
    out.feature.assign(num_grains, -1);
    out.source_quat.assign(4 * num_grains, 0.0);
    out.has_source_quat.assign(num_grains, 0);
    out.phase.assign(num_grains, -1);
    out.filler.assign(num_grains, 0);
    for (std::size_t g = 0; g < num_grains; ++g) {
        const int parent = cleaned.grain_parent[g];
        const int s = labels.grain_seed[static_cast<std::size_t>(parent - 1)];
        const auto si = static_cast<std::size_t>(s);
        out.seed[g] = s;
        if (table) {
            if (!table->has_quat.empty() && table->has_quat[si] != 0) {
                out.has_source_quat[g] = 1;
                for (std::size_t k = 0; k < 4; ++k) {
                    out.source_quat[4 * g + k] = table->quat[4 * si + k];
                }
            }
            if (!table->phase.empty()) {
                out.phase[g] = table->phase[si];
            }
            out.filler[g] = (s >= table->num_table) ? 1 : 0;
        }
    }
    return out;
}

/** [VoxelMap] options as the reader's parameter struct (io/voxel_map_io.hpp). */
VoxelMapParameters MakeVoxelMapParameters(const MicrostructureOptions& options) {
    const VoxelMapOptions& map = options.voxel_map;
    VoxelMapParameters params;
    params.format = (map.format == "dream3d") ? VoxelFormat::DREAM3D : VoxelFormat::EXACONSTIT;
    params.file = map.file;
    params.feature_ids_path = map.feature_ids_path;
    params.quats_path = map.quats_path;
    params.phases_path = map.phases_path;
    // Quaternions only when used; phases are required only for from_phase regions (otherwise
    // LabelFromVoxels reads them when the file has them, as information)
    params.read_quats = (options.orientations.source == "from_source");
    params.read_phases = (options.regions.mode == "from_phase");
    params.quat_layout = QuatLayout::UNSET;
    if (map.quat_layout == "wxyz") {
        params.quat_layout = QuatLayout::WXYZ;
    } else if (map.quat_layout == "xyzw") {
        params.quat_layout = QuatLayout::XYZW;
    }
    params.conjugate = map.conjugate;
    params.has_origin = map.has_origin;
    params.origin = map.origin;
    params.has_spacing = map.has_spacing;
    params.spacing = map.spacing;
    params.units_scale = map.units_scale;
    params.translate = map.translate;
    return params;
}

/*
 * LabelFromVoxels (collective): the voxel-map replacement of steps 3-9 on the active mesh.
 *
 * Steps: ReadVoxelGridInfo (metadata, read once); LabelElementsFromVoxels (each rank reads only
 * the voxel tiles under its own sample points; elements without a vote are filled from their
 * neighbors); CompactFeatureLabels (grains 1..N in ascending feature id); the fragment cleanup
 * with the feature ids as source ids (fragments = "auto" merges: the feature list is measured);
 * ReadVoxelFeatures for the features the final grains descend from: quaternions when
 * orientations come from the source, phases whenever the file has them (required only for
 * from_phase regions); then the per-grain tables of the final grains from their parent features
 * (a piece split off a feature keeps the feature's orientation and phase).
 */
ActiveGrains LabelFromVoxels(const MicrostructureOptions& options, mfem::ParMesh& pmesh,
                             MPI_Comm comm, GenerationReport& report) {
    const VoxelMapParameters map = MakeVoxelMapParameters(options);
    VoxelLabelParameters label_params;
    label_params.sampling = options.voxel_map.sampling;
    label_params.vote_levels = options.voxel_map.vote_levels;
    label_params.tile_size = options.voxel_map.tile_size;

    // Labels from the voxels, then compact grain ids in ascending feature id
    const VoxelGridInfo grid = ReadVoxelGridInfo(map, comm);
    const VoxelLabels labels = LabelElementsFromVoxels(pmesh, comm, map, grid, label_params);
    mfem::Array<int> element_grain;
    std::vector<std::int64_t> grain_feature;
    CompactFeatureLabels(comm, labels.element_feature, element_grain, grain_feature);
    const int num_labeled = static_cast<int>(grain_feature.size());

    // Report: grid and fill statistics; the read volume is per rank, so also give global sums
    report.used_voxel_map = true;
    report.voxel_grid = grid;
    report.voxel_num_filled = labels.num_filled;
    report.voxel_fill_rounds = labels.fill_rounds;
    report.voxel_num_outside_points = labels.num_outside_points;
    report.voxel_tiles_read = labels.num_tiles_read;
    report.voxel_voxels_read = labels.num_voxels_read;
    long long local_read[2] = {labels.num_tiles_read, labels.num_voxels_read};
    long long total_read[2] = {0, 0};
    MPI_Allreduce(local_read, total_read, 2, MPI_LONG_LONG, MPI_SUM, comm);
    report.voxel_total_tiles_read = static_cast<std::int64_t>(total_read[0]);
    report.voxel_total_voxels_read = static_cast<std::int64_t>(total_read[1]);
    report.num_labeled_grains = num_labeled;

    // Fragment cleanup (the keep rule is "largest": voxel grains have no seed)
    ActiveGrains out;
    out.adjacency = std::make_shared<MeshAdjacency>(pmesh);
    const FragmentCleanupParameters cleanup_params = CleanupParameters(options);
    FragmentCleanupResult cleaned = CleanupFragments(*out.adjacency, element_grain, num_labeled,
                                                     grain_feature, cleanup_params);
    report.cleanup = cleaned.report;

    // The features the final grains descend from (a grain merged away entirely by the cleanup
    // no longer needs its row), strictly increasing as ReadVoxelFeatures requires
    std::set<std::int64_t> used_set;
    for (const int parent : cleaned.grain_parent) {
        used_set.insert(grain_feature[static_cast<std::size_t>(parent - 1)]);
    }
    const std::vector<std::int64_t> used_features(used_set.begin(), used_set.end());

    // Per-grain data of those features. Phases are informative (MicrostructureResult::
    // grain_phase) like the phase column of a seed table, so they are read whenever the file has
    // them; they are required only for Regions.mode = "from_phase" (map.read_phases). The reader
    // throws the same error on every rank, so trying with phases and falling back without them
    // is collective-safe; a failure of the fallback is a real error and propagates.
    VoxelFeatureTable features;
    if (map.read_phases) {
        features = ReadVoxelFeatures(map, grid, used_features, comm);
    } else {
        VoxelMapParameters with_phases = map;
        with_phases.read_phases = true;
        try {
            features = ReadVoxelFeatures(with_phases, grid, used_features, comm);
        } catch (const std::runtime_error& e) {
            report.warnings.push_back(std::string("voxel map phases not read (optional unless "
                                                  "Regions.mode = \"from_phase\"): ") +
                                      e.what());
            if (map.read_quats) {
                features = ReadVoxelFeatures(map, grid, used_features, comm);
            }
        }
    }

    // Per-grain tables of the final grains from their parent features
    const auto num_grains = static_cast<std::size_t>(cleaned.num_grains);
    out.element_grain = cleaned.element_grain;
    out.num_grains = cleaned.num_grains;
    out.source_id = cleaned.grain_source_id;
    out.seed.assign(num_grains, -1);
    out.feature.resize(num_grains);
    out.source_quat.assign(4 * num_grains, 0.0);
    out.has_source_quat.assign(num_grains, 0);
    out.phase.assign(num_grains, -1);
    out.filler.assign(num_grains, 0);
    for (std::size_t g = 0; g < num_grains; ++g) {
        const int parent = cleaned.grain_parent[g];
        const std::int64_t feature = grain_feature[static_cast<std::size_t>(parent - 1)];
        out.feature[g] = feature;
        if (features.id.empty()) {
            continue;
        }
        // Row of the feature in the sorted table (ReadVoxelFeatures checked it exists)
        const auto it = std::lower_bound(features.id.begin(), features.id.end(), feature);
        const auto row = static_cast<std::size_t>(it - features.id.begin());
        if (!features.quat.empty()) {
            out.has_source_quat[g] = 1;
            for (std::size_t k = 0; k < 4; ++k) {
                out.source_quat[4 * g + k] = features.quat[4 * row + k];
            }
        }
        if (!features.phase.empty()) {
            out.phase[g] = features.phase[row];
        }
    }
    return out;
}

/*
 * AssignRegions: steps 11-12 (phases and regions) on the active grains, filling
 * result.grain_phase, grain_region, num_regions, region_phases and region_crystalline (all of
 * size num_active_grains / num_regions), and setting the identity quaternion for grains of
 * non-crystalline regions.
 *
 *  - "phase_fractions": synthetic phases (AssignPhaseFractions) from the fixed-point element
 *    volumes of the grains and their source ids, then regions from those phases;
 *  - "from_phase": the phases of the data (every grain must have one: filler grains take
 *    Regions.filler_phase), then AssignRegionsFromPhases;
 *  - "single": region 1 for every grain; grain_phase keeps the data's phases (or -1), and
 *    region 1 lists the distinct phases present.
 * Host only except GrainVolumes (collective) for phase_fractions.
 */
void AssignRegions(const MicrostructureOptions& options, mfem::ParMesh& pmesh, MPI_Comm comm,
                   const ActiveGrains& active, MicrostructureResult& result,
                   GenerationReport& report) {
    const RegionOptions& regions = options.regions;
    const auto num_grains = static_cast<std::size_t>(active.num_grains);
    result.grain_phase = active.phase;

    if (regions.mode == "single") {
        result.num_regions = 1;
        result.grain_region.assign(num_grains, 1);
        std::set<int> present;
        for (const int phase : result.grain_phase) {
            if (phase >= 0) {
                present.insert(phase);
            }
        }
        result.region_phases.assign(1, std::vector<int>(present.begin(), present.end()));
        result.region_crystalline.assign(1, 1);
        return;
    }

    // Synthetic phases: greedy by deficit over the grains, visited in a random order keyed by
    // their source ids (regions/region_assigner.hpp)
    if (regions.mode == "phase_fractions") {
        const std::vector<double> volume =
            GrainVolumes(pmesh, comm, active.element_grain, active.num_grains);
        PhaseFractionParameters params;
        params.phase_id = regions.phase_ids;
        params.fraction = regions.phase_fractions;
        params.rng_seed = options.seeds.rng_seed;
        const PhaseFractionResult phases = AssignPhaseFractions(volume, active.source_id, params);
        result.grain_phase = phases.grain_phase;
        report.phase_fraction_ids = regions.phase_ids;
        report.phase_fraction_achieved = phases.achieved_fraction;
    }

    // Every grain needs a phase now (from the data, or the synthetic ones)
    for (std::size_t g = 0; g < num_grains; ++g) {
        if (result.grain_phase[g] < 0) {
            throw std::invalid_argument(
                "GenerateMicrostructure: Regions.mode = \"" + regions.mode + "\" but grain " +
                std::to_string(g + 1) + " has no phase" +
                (active.filler[g] != 0 ? " (a filler grain: set Regions.filler_phase)" : ""));
        }
    }
    PhaseRegionParameters region_params;
    region_params.non_crystalline_phases = regions.non_crystalline_phases;
    region_params.phase_to_region = regions.phase_to_region;
    const RegionAssignment assignment = AssignRegionsFromPhases(result.grain_phase, region_params);
    result.num_regions = assignment.num_regions;
    result.grain_region = assignment.grain_region;
    result.region_phases = assignment.region_phases;
    result.region_crystalline = assignment.region_crystalline;

    // Grains of non-crystalline regions carry no orientation: identity quaternion (w = 1)
    for (std::size_t g = 0; g < num_grains; ++g) {
        const auto r = static_cast<std::size_t>(result.grain_region[g] - 1);
        if (result.region_crystalline[r] == 0) {
            result.grain_quat[4 * g + 0] = 1.0;
            result.grain_quat[4 * g + 1] = 0.0;
            result.grain_quat[4 * g + 2] = 0.0;
            result.grain_quat[4 * g + 3] = 0.0;
        }
    }
}

/*
 * AppendPassthroughGrains: step 13. One grain per passthrough attribute k (ascending attribute
 * order, the order AttributeDomain::ToParent numbers them: grain N + 1 + k) with the identity
 * quaternion, the region given in [Mesh] passthrough_attributes, phase -1, source id -attribute
 * and no filler flag. Then the check that the phase regions and the passthrough regions together
 * are exactly 1..R (a passthrough region may equal a phase region: the same material), and the
 * per-region tables are extended: a region holding only passthrough grains has no phases and is
 * not crystalline. Host only; identical data on every rank, so every rank throws together.
 */
void AppendPassthroughGrains(const MicrostructureOptions& options,
                             const std::vector<int>& passthrough_attributes,
                             MicrostructureResult& result, std::vector<int>& grain_seed,
                             std::vector<std::int64_t>& grain_feature) {
    for (const int attribute : passthrough_attributes) {
        const int region = options.mesh.passthrough_attributes.at(attribute);
        const double identity[4] = {1.0, 0.0, 0.0, 0.0};
        result.grain_quat.insert(result.grain_quat.end(), identity, identity + 4);
        result.grain_region.push_back(region);
        result.grain_phase.push_back(-1);
        result.grain_source_id.push_back(-static_cast<std::int64_t>(attribute));
        result.grain_filler.push_back(0);
        grain_seed.push_back(-1);
        grain_feature.push_back(-1);
    }
    result.num_grains += static_cast<int>(passthrough_attributes.size());

    // Region ids used by any grain must be exactly 1..R
    std::set<int> used(result.grain_region.begin(), result.grain_region.end());
    const int num_regions = *used.rbegin();
    if (*used.begin() < 1 || num_regions != static_cast<int>(used.size())) {
        std::string missing;
        for (int r = 1; r <= num_regions; ++r) {
            if (used.count(r) == 0) {
                missing += (missing.empty() ? "" : ", ") + std::to_string(r);
            }
        }
        throw std::invalid_argument(
            "GenerateMicrostructure: the phase regions and Mesh.passthrough_attributes regions "
            "together must be exactly 1.." + std::to_string(num_regions) + "; unused region ids: " +
            missing);
    }
    // New regions (passthrough only): no phases, not crystalline
    result.region_phases.resize(static_cast<std::size_t>(num_regions));
    result.region_crystalline.resize(static_cast<std::size_t>(num_regions), 0);
    result.num_regions = num_regions;
}

/*
 * CheckUnitQuaternions: final guard that every row of grain_quat is a unit quaternion. The rows
 * come from AssignMixedGrainOrientations (every row, supplied or random, divided by its norm on
 * the host in double precision) or are exact identity rows (non-crystalline regions, passthrough
 * grains), so |norm - 1| is a few units of round-off (2.2e-16 measured); 1e-15 leaves room for
 * that while catching any path that would bypass the normalization. Host only; the table is
 * replicated, so every rank throws together.
 */
void CheckUnitQuaternions(const MicrostructureResult& result) {
    for (std::size_t g = 0; 4 * g < result.grain_quat.size(); ++g) {
        const double* q = &result.grain_quat[4 * g];
        const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (!(std::fabs(norm - 1.0) <= 1.0e-15)) {
            throw std::logic_error("GenerateMicrostructure: the orientation of grain " +
                                   std::to_string(g + 1) + " is not a unit quaternion (norm " +
                                   std::to_string(norm) + ")");
        }
    }
}

} // namespace

/*
 * GenerateMicrostructure
 *
 * Steps (see the header for the physical meaning of each):
 *  1. Refuse unimplemented options (identical on all ranks).
 *  2. Preflight the whole mesh (before the active part, so passthrough parts are checked too):
 *     CheckMesh reduces every check over the communicator, so ThrowOnPreflightFailure throws on
 *     all ranks or none.
 *  2b. Active part: an AttributeDomain when [Mesh] lists active or passthrough attributes
 *     (collective; its sub-mesh lives only inside this function).
 *  3-9. Grains of the active mesh: LabelFromSeeds (seeds, size targets, Lloyd / fit, labels in
 *     the chosen distance mode, cleanup) or LabelFromVoxels (voxel labels, compaction, cleanup).
 * 10. Orientations: measured ones normalized to unit length (from_source), random ones keyed by
 *     source id otherwise (AssignMixedGrainOrientations).
 * 11-12. Phases and regions (AssignRegions); identity quaternions for non-crystalline regions.
 * 13. Passthrough: ToParent and one grain per passthrough attribute (AppendPassthroughGrains);
 *     the report's adjacency is rebuilt on the whole mesh when a sub-mesh was used. Then
 *     CheckUnitQuaternions confirms every orientation row has norm 1.
 * Then the report (the global reductions always run, so ranks that differ in passing a report
 * cannot deadlock).
 */
MicrostructureResult GenerateMicrostructure(const MicrostructureOptions& options,
                                            mfem::ParMesh& pmesh,
                                            MPI_Comm comm,
                                            GenerationReport* report_out) {
    GenerationReport report;

    // Step 1: unimplemented options
    CheckSupportedOptions(options);

    // Step 2 (before the active part is built): mesh preflight of the whole mesh (collective;
    // global verdict), so inverted or unsupported elements are caught in passthrough parts too.
    // CheckMesh reduces over the ParMesh's own communicator, which must be `comm`. The active
    // elements are a subset of these elements (a sub-mesh copies them unchanged,
    // mesh/attribute_domain.hpp), so a second preflight of the active mesh would add nothing.
    const PreflightReport preflight = CheckMesh(pmesh, comm);
    ThrowOnPreflightFailure(preflight);

    // Step 2b: the active part of the mesh. Declared before everything that refers to the active
    // mesh, so those objects (adjacency, labels) are destroyed first.
    std::unique_ptr<AttributeDomain> domain;
    std::vector<int> passthrough_attributes;
    if (!options.mesh.active_attributes.empty() || !options.mesh.passthrough_attributes.empty()) {
        for (const auto& entry : options.mesh.passthrough_attributes) {
            passthrough_attributes.push_back(entry.first); // std::map: ascending
        }
        domain = std::make_unique<AttributeDomain>(pmesh, options.mesh.active_attributes,
                                                   passthrough_attributes);
        report.used_attribute_domain = true;
        report.used_submesh = domain->UsesSubMesh();
        report.active_attributes = domain->ActiveAttributes();
        report.passthrough_attributes = domain->PassthroughAttributes();
        report.passthrough_element_counts = domain->PassthroughElementCounts();
        for (const int attribute : report.passthrough_attributes) {
            report.passthrough_regions.push_back(options.mesh.passthrough_attributes.at(attribute));
        }
    }
    mfem::ParMesh& active_mesh = domain ? domain->ActiveMesh() : pmesh;

    // Steps 3-9: grains of the active mesh
    ActiveGrains active = UsesVoxelMap(options)
                              ? LabelFromVoxels(options, active_mesh, comm, report)
                              : LabelFromSeeds(options, active_mesh, comm, report);

    // Step 10: orientations, one per active grain; measured ones only when asked for
    MicrostructureResult result;
    std::vector<char> use_source_quat = active.has_source_quat;
    if (options.orientations.source != "from_source") {
        std::fill(use_source_quat.begin(), use_source_quat.end(), 0);
    }
    AssignMixedGrainOrientations(options.seeds.rng_seed, active.source_id, active.source_quat,
                                 use_source_quat, result);
    result.grain_filler = active.filler;

    // Steps 11-12: phases and regions of the active grains
    AssignRegions(options, active_mesh, comm, active, result, report);

    // Step 13: back to the whole mesh, plus one grain per passthrough attribute
    std::vector<int> grain_seed = active.seed;
    std::vector<std::int64_t> grain_feature = active.feature;
    std::shared_ptr<const MeshAdjacency> adjacency = active.adjacency;
    long long local_counts[3] = {pmesh.GetNE(), active_mesh.GetNE(), active.local_split_elements};
    if (domain) {
        result.element_grain = domain->ToParent(active.element_grain, active.num_grains);
        AppendPassthroughGrains(options, passthrough_attributes, result, grain_seed,
                                grain_feature);
        if (domain->UsesSubMesh()) {
            // The cleanup adjacency belongs to the sub-mesh, which dies with `domain`: give the
            // caller an adjacency of the whole mesh instead (collective on every rank, whether or
            // not it passed a report). Periodic axes are refused with passthrough attributes.
            adjacency = std::make_shared<MeshAdjacency>(pmesh);
        }
    } else {
        result.element_grain = active.element_grain;
    }

    // Every orientation written is a unit quaternion
    CheckUnitQuaternions(result);

    // Report. The reductions run on every rank whether or not it passed a report, so callers
    // that differ in that respect cannot deadlock.
    long long global_counts[3] = {0, 0, 0};
    MPI_Allreduce(local_counts, global_counts, 3, MPI_LONG_LONG, MPI_SUM, comm);
    if (report_out != nullptr) {
        report.num_grains = result.num_grains;
        report.grain_seed = grain_seed;
        report.grain_feature = grain_feature;
        report.num_elements = static_cast<std::int64_t>(global_counts[0]);
        report.num_active_elements = static_cast<std::int64_t>(global_counts[1]);
        report.num_split_elements = static_cast<std::int64_t>(global_counts[2]);
        report.num_passthrough_grains = static_cast<int>(passthrough_attributes.size());
        report.num_filler_grains = static_cast<int>(
            std::count(result.grain_filler.begin(), result.grain_filler.end(), char(1)));
        report.adjacency = adjacency; // for reuse by the caller (see GenerationReport)
        *report_out = std::move(report);
    }
    return result;
}

} // namespace exaconstit::microstructure
