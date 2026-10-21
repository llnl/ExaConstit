/*
 * Implementation of GenerateMicrostructure (microstructure_driver.hpp): the Stage 1a pipeline
 * preflight -> seeds -> labels -> orientations -> regions.
 *
 * Every step is either computed from data that is identical on all ranks (the options, the
 * replicated seed set) or combined with MPI reductions inside the called functions, so all ranks
 * make the same decisions and, on failure, throw the same exception together. A function that
 * every rank must call because ranks exchange data inside it is called "collective".
 */
#include "microstructure/microstructure_driver.hpp"

#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/preflight.hpp"
#include "microstructure/orientation/orientation_assigner.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <mpi.h>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/*
 * CheckSupportedOptions
 *
 * The options structure describes the whole planned generator, but this pipeline implements the
 * Stage 1a subset. Options that ask for something not implemented yet are refused here, before
 * any work, with a message naming the option, rather than being silently ignored. The
 * tessellation sampling/distance/periodic checks are owned by the labeler
 * (CheckSupportedTessellation) and called from here, so they also run before seeding; the seed
 * placement checks live in the seed generator (its parameter check runs before any
 * communication).
 *
 * Accepted without effect: Cleanup.fragments = "auto" / "none" and min_grain_elements 0 or 1,
 * since the default "auto" must keep the "-n N" quick path working. They will take effect once
 * fragment cleanup exists.
 */
void CheckSupportedOptions(const MicrostructureOptions& options) {
    if (options.seeds.source != "random") {
        throw std::invalid_argument(
            "GenerateMicrostructure: Seeds.source = \"" + options.seeds.source +
            "\" is not implemented yet (only generated \"random\" seeds)");
    }
    if (options.cleanup.fragments == "split" || options.cleanup.fragments == "merge") {
        throw std::invalid_argument("GenerateMicrostructure: Cleanup.fragments = \"" +
                                    options.cleanup.fragments +
                                    "\" is not implemented yet (use \"auto\" or \"none\")");
    }
    if (options.cleanup.min_grain_elements > 1) {
        throw std::invalid_argument(
            "GenerateMicrostructure: Cleanup.min_grain_elements > 1 (merging small grains) is "
            "not implemented yet");
    }
    if (options.regions.mode != "single") {
        throw std::invalid_argument("GenerateMicrostructure: Regions.mode = \"" +
                                    options.regions.mode +
                                    "\" is not implemented yet (only \"single\")");
    }
    CheckSupportedTessellation(options.tessellation);
}

} // namespace

/*
 * GenerateMicrostructure
 *
 * Steps (see the header for the physical meaning of each):
 *  1. Refuse unimplemented options (identical on all ranks).
 *  2. Preflight the mesh: CheckMesh reduces every check over the communicator, so
 *     ThrowOnPreflightFailure throws on all ranks or none.
 *  3. Seeds: GenerateSeeds over `comm` (collective; the same SeedSet on every rank).
 *  4. Labels: LabelTessellation (local nearest-seed search, then a collective compaction to
 *     grain ids 1..N in seed order).
 *  5. Orientations from the grain -> seed table (host, replicated).
 *  6. Regions: every grain in region 1.
 *  7. Report: global counts for the caller's log (the element count needs one MPI_Allreduce,
 *     done unconditionally).
 */
MicrostructureResult GenerateMicrostructure(const MicrostructureOptions& options,
                                            mfem::ParMesh& pmesh,
                                            MPI_Comm comm,
                                            GenerationReport* report) {
    // Step 1: unimplemented options
    CheckSupportedOptions(options);

    // Step 2: mesh preflight (collective; global verdict). CheckMesh(pmesh) reduces over the
    // ParMesh's own communicator, which must be `comm`.
    const PreflightReport preflight = CheckMesh(pmesh, comm);
    ThrowOnPreflightFailure(preflight);

    // Step 3: seeds, identical on every rank
    const SeedingParameters seeding_params = MakeSeedingParameters(options);
    SeedingReport seeding_report;
    const SeedSet seeds = GenerateSeeds(pmesh, comm, seeding_params, &seeding_report);
    if (seeds.Size() == 0) {
        // Only possible with allow_fewer_seeds when not a single seed could be placed; the
        // decision is made from the replicated seed set, so every rank throws together.
        throw std::runtime_error(
            "GenerateMicrostructure: no seed could be placed in the material\n" +
            seeding_report.Summary());
    }

    // Step 4: element labels and compact grain ids (collective)
    TessellationLabels labels = LabelTessellation(pmesh, comm, seeds, options.tessellation);

    // Step 5: orientations, one per grain, keyed by the seed's source id
    MicrostructureResult result;
    AssignOrientations(options.orientations, options.seeds.rng_seed, seeds, labels.grain_seed,
                       result);
    result.element_grain = labels.element_grain;

    // Step 6: a single material region
    result.num_regions = 1;
    result.grain_region.assign(static_cast<std::size_t>(result.num_grains), 1);

    // Step 7: global counts for the log. The reduction runs on every rank whether or not it
    // passed a report, so callers that differ in that respect cannot deadlock.
    long long local_elements = pmesh.GetNE();
    long long global_elements = 0;
    MPI_Allreduce(&local_elements, &global_elements, 1, MPI_LONG_LONG, MPI_SUM, comm);
    if (report != nullptr) {
        report->seeding = seeding_report;
        report->num_seeds = static_cast<std::int64_t>(seeds.Size());
        report->num_grains = result.num_grains;
        report->num_empty_seeds = report->num_seeds - result.num_grains;
        report->num_coincident_seeds = labels.num_coincident_seeds;
        report->num_elements = static_cast<std::int64_t>(global_elements);
    }
    return result;
}

} // namespace exaconstit::microstructure
