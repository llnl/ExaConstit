/*
 * Implementation of GenerateMicrostructure (microstructure_driver.hpp): the Stage 1a-1c pipeline
 * support checks -> preflight -> seeds (+ size targets) -> Lloyd / volume fit -> labels (centroid
 * or vote, periodic images) -> compact ids -> fragment cleanup -> orientations -> regions.
 *
 * Every step is either computed from data that is identical on all ranks (the options, the
 * replicated seed set and per-seed tables) or combined with MPI reductions inside the called
 * functions, so all ranks make the same decisions and, on failure, throw the same exception
 * together. A function that every rank must call because ranks exchange data inside it is called
 * "collective".
 *
 * Terms used in this file:
 * - Size targets: one target volume per seed (morphology/size_targets.hpp), scaled so they fill
 *   the material; r_i = (3 V_i / (4 pi))^(1/3) is the equivalent-sphere radius of target V_i.
 * - Polydisperse RSA: random sequential adsorption in which every seed slot has its own exclusion
 *   radius (kappa r_i), largest first (seeding/seed_generators.hpp, SeedingParameters::rsa_radius).
 * - Source id: the stable id of a seed (SeedSet::source_id); random orientations are keyed by it.
 */
#include "microstructure/microstructure_driver.hpp"

#include "microstructure/geometry/periodic_images.hpp"
#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/preflight.hpp"
#include "microstructure/morphology/size_targets.hpp"
#include "microstructure/orientation/orientation_assigner.hpp"
#include "microstructure/seeding/material_point_locator.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

namespace {

constexpr double PI = 3.14159265358979323846;

/*
 * CheckSupportedOptions
 *
 * The options structure describes the whole planned generator, but this pipeline implements the
 * Stage 1a-1c subset. Options that ask for something not implemented yet are refused here,
 * before any work, with a message naming the option, rather than being silently ignored. The
 * tessellation checks (distance, vote levels, anisotropy kind) are owned by the labeler
 * (CheckSupportedTessellation) and called from here, so they also run before seeding. The
 * combination checks of validate() that would otherwise fail deep inside a collective step are
 * repeated, so a caller that skipped validate() still gets a clear message on every rank.
 */
void CheckSupportedOptions(const MicrostructureOptions& options) {
    if (options.seeds.source != "random") {
        throw std::invalid_argument(
            "GenerateMicrostructure: Seeds.source = \"" + options.seeds.source +
            "\" is not implemented yet (only generated \"random\" seeds)");
    }
    if (options.regions.mode != "single") {
        throw std::invalid_argument("GenerateMicrostructure: Regions.mode = \"" +
                                    options.regions.mode +
                                    "\" is not implemented yet (only \"single\")");
    }
    const std::string& fragments = options.cleanup.fragments;
    if (fragments != "auto" && fragments != "split" && fragments != "merge" &&
        fragments != "none") {
        throw std::invalid_argument("GenerateMicrostructure: unknown Cleanup.fragments \"" +
                                    fragments + "\"");
    }
    if (options.cleanup.keep_component != "largest" && options.cleanup.keep_component != "seed") {
        throw std::invalid_argument("GenerateMicrostructure: unknown Cleanup.keep_component \"" +
                                    options.cleanup.keep_component + "\"");
    }
    if (options.cleanup.min_grain_elements < 0) {
        throw std::invalid_argument("GenerateMicrostructure: Cleanup.min_grain_elements < 0");
    }
    if (options.orientations.source != "random") {
        throw std::invalid_argument("GenerateMicrostructure: Orientations.source = \"" +
                                    options.orientations.source +
                                    "\" is not implemented yet (only \"random\")");
    }
    const TessellationOptions& tess = options.tessellation;
    const bool periodic = tess.periodic[0] || tess.periodic[1] || tess.periodic[2];
    if (periodic && options.seeds.placement == SeedPlacement::JOHNSON_MEHL) {
        throw std::invalid_argument("GenerateMicrostructure: placement = \"johnson_mehl\" is not "
                                    "supported with periodic axes");
    }
    if (options.seeds.size.polydisperse_placement && options.seeds.use_sampling_box) {
        throw std::invalid_argument("GenerateMicrostructure: polydisperse placement cannot be "
                                    "combined with Seeds.sampling_box");
    }
    if (periodic && (tess.fit_volumes.enabled || tess.lloyd_iterations > 0)) {
        throw std::invalid_argument("GenerateMicrostructure: fit_volumes and lloyd_iterations are "
                                    "not supported with periodic axes");
    }
    if (options.seeds.placement == SeedPlacement::JOHNSON_MEHL &&
        tess.metric != MetricType::ADDITIVE) {
        throw std::invalid_argument("GenerateMicrostructure: placement = \"johnson_mehl\" needs "
                                    "Tessellation.metric = \"additive\"");
    }
    if (options.seeds.size.distribution == "from_source") {
        throw std::invalid_argument("GenerateMicrostructure: Seeds.Size.distribution = "
                                    "\"from_source\" needs a seed table (not implemented yet)");
    }
    CheckSupportedTessellation(tess);
}

/*
 * MaterialVolume (collective): the volume of the whole mesh, as a reproducible fixed-point sum of
 * the element volumes (CellMeasureAccumulator with every element labeled 0: its single cell is
 * the whole material). Bitwise the same on every rank and for every rank count, so decisions
 * based on it (the seed count, the polydisperse radii) are identical everywhere.
 */
double MaterialVolume(mfem::ParMesh& pmesh, MPI_Comm comm) {
    WeightedSamples samples;
    MakeElementWeightedSamples(pmesh, samples);
    const CellMeasureAccumulator accumulator(comm, samples, 1);
    mfem::Array<int> all_zero(samples.Size());
    all_zero = 0;
    return accumulator.Measure(all_zero).total_volume;
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
 * MakeSeedTargets: the target volume of every seed (seed order), normalized to the material.
 *  - polydisperse placement: the slot volumes the seeds were placed with;
 *  - otherwise: drawn from the distribution keyed by each seed's source id (EQUAL without a
 *    distribution, for a plain volume fit).
 * Host only; replicated inputs give identical targets on every rank.
 */
SizeTargets MakeSeedTargets(const MicrostructureOptions& options, const SeedSet& seeds,
                            double material_volume, const std::vector<double>& slot_volume) {
    if (options.seeds.size.polydisperse_placement) {
        return MakeSizeTargetsFromVolumes(slot_volume, material_volume);
    }
    return MakeSizeTargets(seeds.source_id, material_volume, MakeSizeParameters(options));
}

/*
 * RelaxAndFit (collective): Lloyd relaxation and/or the volume fit, on the same sample points
 * (quadrature points or element centroids, Tessellation.fit_volumes.samples).
 *  - Lloyd: iterations with the user's metric (Voronoi or Laguerre with the current weights); in
 *    centroidal power mode every step is paired with a Laguerre refit to the targets.
 *  - Volume fit (not in centroidal power mode, which already refits): FitWeights from the
 *    current weights; the fitted weights replace seeds.weight.
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
 * CleanupParameters: [Cleanup] as FragmentCleanupParameters. "auto" splits large fragments and
 * merges tiny ones (SPLIT_LARGE_MERGE_TINY) for grains grown from generated seeds (random,
 * lattice and Johnson-Mehl are all Seeds.source = "random"), and merges every fragment of
 * data-driven grains, whose grain list is fixed by the measurement.
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
    } else if (options.seeds.source == "random") {
        params.policy = FragmentPolicy::SPLIT_LARGE_MERGE_TINY; // auto, generated seeds
    } else {
        params.policy = FragmentPolicy::MERGE; // auto, data-driven seeds
    }
    params.keep = (options.cleanup.keep_component == "seed") ? KeepRule::SEED_ELEMENT
                                                             : KeepRule::LARGEST;
    params.min_grain_elements = options.cleanup.min_grain_elements;
    params.min_fragment_elements = options.cleanup.min_fragment_elements;
    params.min_fragment_fraction = options.cleanup.min_fragment_fraction;
    return params;
}

} // namespace

/*
 * GenerateMicrostructure
 *
 * Steps (see the header for the physical meaning of each):
 *  1. Refuse unimplemented options (identical on all ranks).
 *  2. Preflight the mesh: CheckMesh reduces every check over the communicator, so
 *     ThrowOnPreflightFailure throws on all ranks or none.
 *  3. Seeds (collective; the same SeedSet on every rank), the material volume when a stage needs
 *     it, and the per-seed size targets (Laguerre starting weights w = r^2).
 *  5. Lloyd relaxation / volume fit (collective).
 *  6-8. LabelTessellation: periodic images, centroid or vote labels, map back, compaction.
 *  9. Fragment cleanup on the (periodic-aware) face adjacency.
 * 10. Orientations from the final grain -> (parent seed, source id) tables (host, replicated).
 * 11. Regions: every grain in region 1, and the report (the global reductions always run, so
 *     ranks that differ in passing a report cannot deadlock).
 */
MicrostructureResult GenerateMicrostructure(const MicrostructureOptions& options,
                                            mfem::ParMesh& pmesh,
                                            MPI_Comm comm,
                                            GenerationReport* report_out) {
    GenerationReport report;

    // Step 1: unimplemented options
    CheckSupportedOptions(options);

    // Step 2: mesh preflight (collective; global verdict). CheckMesh(pmesh) reduces over the
    // ParMesh's own communicator, which must be `comm`.
    const PreflightReport preflight = CheckMesh(pmesh, comm);
    ThrowOnPreflightFailure(preflight);

    // Step 3a: the material volume, when the seed count, the size targets or the report need it
    const bool johnson_mehl = (options.seeds.placement == SeedPlacement::JOHNSON_MEHL);
    const bool size_targets = NeedsSizeTargets(options);
    const bool count_from_size = options.seeds.num_seeds <= 0 &&
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
    if (johnson_mehl) {
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

    // Step 3c: size targets per seed; Laguerre seeds start from w_i = r_i^2
    std::vector<double> target_volume;
    if (size_targets) {
        const SizeTargets targets = MakeSeedTargets(options, seeds, material_volume, slot_volume);
        target_volume = targets.volume;
        if (options.tessellation.metric == MetricType::LAGUERRE) {
            seeds.weight = targets.initial_weight;
        }
    }

    // Step 5: Lloyd relaxation and/or the volume fit (validated: not with periodic axes)
    RelaxAndFit(options, pmesh, comm, target_volume, seeds, report);

    // Steps 6-8: labels and compact grain ids (collective). With periodic axes the labeler builds
    // the periodic images itself and maps their labels back to the original seeds.
    const TessellationLabels labels = LabelTessellation(pmesh, comm, seeds, options.tessellation);
    report.periodic_layers = labels.periodic_layers;
    report.num_periodic_images = labels.num_periodic_images;

    // Step 9: fragment cleanup over the face adjacency; with periodic axes the adjacency pairs
    // matching faces on opposite planes of the mesh box, so wrapped grains stay one grain
    const TessellationOptions& tess = options.tessellation;
    const bool periodic = tess.periodic[0] || tess.periodic[1] || tess.periodic[2];
    std::shared_ptr<MeshAdjacency> adjacency;
    if (periodic) {
        const PeriodicBox box = PeriodicBoxFromMesh(pmesh, comm, tess.periodic);
        PeriodicFaceParameters pairing;
        pairing.periodic = box.periodic;
        pairing.lo = box.lo;
        pairing.hi = box.hi;
        pairing.relative_tolerance = tess.periodic_tolerance;
        adjacency = std::make_shared<MeshAdjacency>(pmesh, pairing);
        report.periodic_faces = adjacency->PeriodicReport();
    } else {
        adjacency = std::make_shared<MeshAdjacency>(pmesh);
    }
    FragmentCleanupParameters cleanup_params = CleanupParameters(options);
    if (cleanup_params.keep == KeepRule::SEED_ELEMENT) {
        cleanup_params.grain_anchor_key = SeedAnchorKeys(pmesh, comm, seeds, labels);
    }
    std::vector<std::int64_t> labeled_source_id(labels.grain_seed.size());
    for (std::size_t g = 0; g < labels.grain_seed.size(); ++g) {
        labeled_source_id[g] = seeds.source_id[static_cast<std::size_t>(labels.grain_seed[g])];
    }
    FragmentCleanupResult cleaned = CleanupFragments(
        *adjacency, labels.element_grain, labels.num_grains, labeled_source_id, cleanup_params);

    // Step 10: orientations, one per final grain. Grain g descends from input grain
    // grain_parent[g-1], whose seed is labels.grain_seed[parent - 1].
    std::vector<int> grain_seed(static_cast<std::size_t>(cleaned.num_grains));
    for (std::size_t g = 0; g < grain_seed.size(); ++g) {
        const int parent = cleaned.grain_parent[g];
        grain_seed[g] = labels.grain_seed[static_cast<std::size_t>(parent - 1)];
    }
    MicrostructureResult result;
    AssignGrainOrientations(options.orientations, options.seeds.rng_seed, seeds, grain_seed,
                            cleaned.grain_source_id, result);
    result.element_grain = cleaned.element_grain;

    // Step 11: a single material region, then the report. The reductions run on every rank
    // whether or not it passed a report, so callers that differ in that respect cannot deadlock.
    result.num_regions = 1;
    result.grain_region.assign(static_cast<std::size_t>(result.num_grains), 1);

    long long local_counts[2] = {pmesh.GetNE(), labels.num_split_elements};
    long long global_counts[2] = {0, 0};
    MPI_Allreduce(local_counts, global_counts, 2, MPI_LONG_LONG, MPI_SUM, comm);
    if (report_out != nullptr) {
        report.num_seeds = static_cast<std::int64_t>(seeds.Size());
        report.seed_target_volume = target_volume;
        report.num_labeled_grains = labels.num_grains;
        report.cleanup = cleaned.report;
        report.num_grains = result.num_grains;
        report.grain_seed = grain_seed;
        report.num_empty_seeds = report.num_seeds - labels.num_grains;
        report.num_coincident_seeds = labels.num_coincident_seeds;
        report.num_elements = static_cast<std::int64_t>(global_counts[0]);
        report.num_split_elements = static_cast<std::int64_t>(global_counts[1]);
        report.adjacency = adjacency; // for reuse by the caller (see GenerationReport)
        *report_out = std::move(report);
    }
    return result;
}

} // namespace exaconstit::microstructure
