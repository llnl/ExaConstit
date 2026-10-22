/*
 * Implementation of the tessellation labeler (tessellation_labeler.hpp): element centroids ->
 * exact nearest seed -> compact grain ids.
 *
 * Terms used in this file:
 * - BVH (bounding volume hierarchy): a tree of nested bounding boxes that lets us skip seeds that
 *   are provably too far away; SeedIndex (geometry/seed_index.hpp) builds one and searches it.
 * - Execution space: the Axom tag type (axom::SEQ_EXEC, axom::CUDA_EXEC<256>, ...) that says
 *   where the BVH kernels run. DispatchAxomExec (exec_dispatch.hpp) picks the one that matches
 *   the configured mfem::Device, so the seed search runs where all other kernels run.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 * - Bitmask: an array of 64-bit integers used as a list of yes/no flags, one bit per seed:
 *   bit (i mod 64) of word (i / 64) says whether seed i is used.
 */
#include "microstructure/labeling/tessellation_labeler.hpp"

#include "microstructure/exec_dispatch.hpp"
#include "microstructure/geometry/anisotropy_tensors.hpp"
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/geometry/periodic_images.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

/*
 * CheckSupportedTessellation
 *
 * The options structure already has room for later distance modes (connected / geodesic
 * distance). This labeler implements the plain (Euclidean) metric distance with centroid or
 * quadrature-vote sampling, with or without periodic images, so anything else is refused with a
 * clear message instead of being silently ignored. The vote level range (0..3) and the
 * anisotropy kind are checked here too, so a direct caller gets the same messages as the driver.
 * Every rank holds the same options, so every rank throws together. Public so the driver can call
 * it before the collective seeding; the labeler calls it again for direct callers.
 */
void CheckSupportedTessellation(const TessellationOptions& tessellation) {
    if (tessellation.distance != DistanceMode::EUCLIDEAN) {
        throw std::invalid_argument(
            "Tessellation labeler: only distance = \"euclidean\" is implemented so far");
    }
    if (tessellation.vote_levels < 0 || tessellation.vote_levels > 3) {
        throw std::invalid_argument("Tessellation labeler: vote_levels must be in 0..3");
    }
    // Resolving the tensor validates the anisotropy kind and its direction
    EffectiveGlobalTensor(tessellation);
}

/*
 * EffectiveGlobalTensor: "none"/"tensor" return the stored tensor unchanged (so options that
 * never mention anisotropy behave exactly as before the kinds existed); "columnar"/"lamellar"
 * build the semidefinite tensors with the helpers of geometry/metric_functors.hpp, which
 * normalize the direction and throw for a zero or non-finite one.
 */
std::array<double, 6> EffectiveGlobalTensor(const TessellationOptions& tessellation) {
    const std::string& kind = tessellation.anisotropy_kind;
    if (kind == "none" || kind == "tensor") {
        return tessellation.global_tensor;
    }
    if (kind == "columnar") {
        return MakeColumnarTensor(tessellation.anisotropy_axis);
    }
    if (kind == "lamellar") {
        return MakeLamellarTensor(tessellation.anisotropy_normal);
    }
    throw std::invalid_argument("Tessellation labeler: unknown anisotropy kind \"" + kind +
                                "\" (expected none, tensor, columnar or lamellar)");
}

/*
 * ThrowIfAnyElementUnlabeled
 *
 * Every rank looks for its first element with seed < 0; LowestRankWith (MPI_Allreduce MIN over
 * "my rank if I have one") picks the lowest such rank, which broadcasts its local element id, so
 * all ranks throw the same message together instead of failing later with a misleading error
 * (e.g. the compaction's index check) or hanging in a collective call.
 */
void ThrowIfAnyElementUnlabeled(MPI_Comm comm,
                                const mfem::Array<int>& element_seed,
                                const std::string& what,
                                const std::string& hint) {
    // First local element without a seed, if any
    const int num_elements = element_seed.Size();
    const int* seed_host = (num_elements > 0) ? element_seed.HostRead() : nullptr;
    int first_bad = -1;
    for (int e = 0; e < num_elements; ++e) {
        if (seed_host[e] < 0) {
            first_bad = e;
            break;
        }
    }

    // Lowest rank with such an element shares its element id; everyone throws the same message
    const int bad_rank = LowestRankWith(first_bad >= 0, comm);
    if (bad_rank < 0) {
        return;
    }
    MPI_Bcast(&first_bad, 1, MPI_INT, bad_rank, comm);
    throw std::runtime_error(what + " (local element " + std::to_string(first_bad) + " on rank " +
                             std::to_string(bad_rank) + ")" + hint);
}

/*
 * LabelElementsWithSeeds
 *
 * Algorithm:
 *  1. Volume centroid of every local element (ComputeElementSamples: an mfem::forall kernel over
 *     mfem::GeometricFactors, sum_q w_q detJ_q x_q / sum_q w_q detJ_q). The true volume centroid
 *     is used, not the vertex average, so curved and distorted elements are sampled at their
 *     center of mass.
 *  2. Pick the metric functor type from tessellation.metric (DispatchMetric) and the Axom
 *     execution space from the mfem::Device (DispatchAxomExec). Both dispatchers call the lambda
 *     with a value whose *type* selects the template instantiation of SeedIndex; nested lambdas
 *     are the documented way to combine them (metric_functors.hpp, DispatchMetric).
 *  3. Build the SeedIndex (one BVH per parameter bin over the seed points) and query it with the
 *     centroids: FindBest returns, per centroid, the seed with the smallest f_i, ties to the
 *     lowest seed index. Only the best seed is needed here, so the second-best search is off.
 *  4. Copy the winning seed index into element_seed with an mfem::forall kernel. FindBest wrote
 *     its result where the execution space runs (device memory on a GPU), and mfem::forall runs
 *     on the matching mfem::Device backend, so Read()/Write() hand out pointers valid there.
 *
 * Why the result is partition independent: the centroid of an element depends only on that
 * element's nodes, and the search result depends only on the centroid and the replicated seeds.
 */
void LabelElementsWithSeeds(mfem::Mesh& local_mesh,
                            const SeedSet& seeds,
                            const TessellationOptions& tessellation,
                            mfem::Array<int>& element_seed,
                            int* num_coincident_seeds) {
    CheckSupportedTessellation(tessellation);
    const std::size_t num_seeds = seeds.Size();
    if (num_seeds == 0) {
        throw std::invalid_argument("Tessellation labeler: the seed set is empty");
    }

    // Step 1: element volume centroids (negative quadrature order = exact default order)
    ElementSamples samples;
    ComputeElementSamples(local_mesh, -1, samples);
    const int num_elements = local_mesh.GetNE();
    element_seed.SetSize(num_elements);

    // Quadric parameters: the global anisotropy tensor (columnar/lamellar resolved), or per-seed
    // tensors when the seed set carries them (SeedSet::shape, 6 entries per seed)
    MetricSetup setup;
    setup.global_tensor = EffectiveGlobalTensor(tessellation);
    setup.per_seed_shape = !seeds.shape.empty();

    int coincident = 0;
    mfem::Array<BestPair> best;

    // Steps 2-3: choose metric type and execution space, build, search. The dispatch lambdas
    // are generic (auto parameters); the mfem::forall of step 4 stays outside them because CUDA
    // does not allow device lambdas to be defined inside a generic lambda.
    DispatchMetric(tessellation.metric, [&](auto metric_tag) {
        using Metric = decltype(metric_tag);
        DispatchAxomExec([&](auto exec_tag) {
            using Exec = decltype(exec_tag);

            // Step 3a: seed search structure. It is rebuilt on every rank from the replicated
            // seeds; a rank without elements still builds it so its coincident-seed count
            // matches the other ranks.
            SeedIndex<Exec, Metric> index;
            index.Build(seeds, setup, tessellation.num_bins, AxomAllocatorIdForCurrentDevice());
            coincident = index.GetReport().num_coincident;

            // Step 3b: nearest seed of every centroid (exact, lowest index on ties)
            index.FindBest(samples.cx, samples.cy, samples.cz, best, false);
        });
    });

    // Step 4: keep only the winning seed index (id1) per element. FindBest reports id1 = -1 for
    // a query point that is not finite (e.g. a 0/0 centroid of a zero-volume element); the -1 is
    // passed through and turned into a collective error by LabelTessellation.
    if (num_elements > 0) {
        const BestPair* best_data = best.Read();
        int* seed_out = element_seed.Write();
        mfem::forall(num_elements, [=] MFEM_HOST_DEVICE(int e) { seed_out[e] = best_data[e].id1; });
    }

    if (num_coincident_seeds != nullptr) {
        *num_coincident_seeds = coincident;
    }
}

/*
 * CompactSeedLabels
 *
 * Algorithm:
 *  1. Check every local seed index (host loop over HostRead()). A bad index on any rank is
 *     reported with one MPI_Allreduce(MAX) of an error flag, so every rank throws together and
 *     no rank is left waiting in the next collective call.
 *  2. Mark the used seeds in a local bitmask: bit (i mod 64) of word i / 64 is set when some
 *     local element was won by seed i.
 *  3. MPI_Allreduce with MPI_BOR (bitwise OR) over 64-bit unsigned words: afterwards bit i is
 *     set on every rank exactly when seed i won an element on *some* rank. The OR does not
 *     depend on the order in which ranks contribute, so the mask is the same for any rank count.
 *  4. Walk the seeds in increasing index and give each used seed the next grain id 1, 2, 3, ...
 *     (seed_grain, grain_seed). Every rank does this with the same mask, so all ranks agree.
 *  5. element_grain[e] = seed_grain[element_seed[e]] in an mfem::forall kernel (the table is
 *     copied to an mfem::Array so the kernel can read it on the device).
 */
void CompactSeedLabels(MPI_Comm comm,
                       const mfem::Array<int>& element_seed,
                       int num_seeds,
                       TessellationLabels& labels) {
    if (num_seeds < 1) {
        throw std::invalid_argument("CompactSeedLabels: num_seeds must be >= 1");
    }
    const int num_elements = element_seed.Size();

    // Step 1: validate the local indices, then agree on failure across all ranks
    const int* seed_host = (num_elements > 0) ? element_seed.HostRead() : nullptr;
    int local_bad = 0;
    for (int e = 0; e < num_elements; ++e) {
        if (seed_host[e] < 0 || seed_host[e] >= num_seeds) {
            local_bad = 1;
            break;
        }
    }
    int any_bad = 0;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
    if (any_bad != 0) {
        throw std::invalid_argument(
            "CompactSeedLabels: a seed index is outside [0, num_seeds) on at least one rank");
    }

    // Step 2: local used-seed bitmask, ceil(num_seeds / 64) words
    const std::size_t num_words = (static_cast<std::size_t>(num_seeds) + 63) / 64;
    std::vector<std::uint64_t> used(num_words, 0);
    for (int e = 0; e < num_elements; ++e) {
        const auto seed = static_cast<std::size_t>(seed_host[e]);
        used[seed / 64] |= std::uint64_t(1) << (seed % 64);
    }

    // Step 3: global OR of the masks (word count is the same on every rank: num_seeds is)
    MPI_Allreduce(MPI_IN_PLACE, used.data(), static_cast<int>(num_words), MPI_UINT64_T, MPI_BOR,
                  comm);

    // Step 4: compact grain ids in increasing seed index
    labels.seed_grain.assign(static_cast<std::size_t>(num_seeds), 0);
    labels.grain_seed.clear();
    int next_grain = 1;
    for (int seed = 0; seed < num_seeds; ++seed) {
        const auto s = static_cast<std::size_t>(seed);
        const bool is_used = ((used[s / 64] >> (s % 64)) & std::uint64_t(1)) != 0;
        if (is_used) {
            labels.seed_grain[s] = next_grain;
            labels.grain_seed.push_back(seed);
            ++next_grain;
        }
    }
    labels.num_grains = next_grain - 1;

    // Step 5: per-element grain ids. The seed-to-grain table goes into an mfem::Array so the
    // kernel reads it through Read() (device memory on a GPU backend).
    labels.element_seed = element_seed;
    labels.element_grain.SetSize(num_elements);
    if (num_elements == 0) {
        return;
    }
    mfem::Array<int> seed_grain_table(num_seeds);
    for (int seed = 0; seed < num_seeds; ++seed) {
        seed_grain_table[seed] = labels.seed_grain[static_cast<std::size_t>(seed)];
    }
    const int* table = seed_grain_table.Read();
    const int* seed_of_element = element_seed.Read();
    int* grain_of_element = labels.element_grain.Write();
    mfem::forall(num_elements, [=] MFEM_HOST_DEVICE(int e) {
        grain_of_element[e] = table[seed_of_element[e]];
    });
}

/*
 * LabelTessellation
 *
 * Steps (see the header):
 *  1. Checks (identical on every rank). The seed count must fit in an int (the search and the
 *     grain tables use int indices; SeedIndex::Build enforces the same limit).
 *  2. Periodic images when any axis is periodic: box from the mesh (collective), provably
 *     sufficient layers for the metric, image set with the image -> original map.
 *  3. Nearest seed per local element (no communication): centroid search with the (image) seeds
 *     and MapImageLabelsToOriginal, or the quadrature vote with the image -> original map as the
 *     seed owners (votes of a seed and its images count for the same grain).
 *  4. Collective check for elements without a seed.
 *  5. Compaction to grain ids over the original seeds (collective).
 */
TessellationLabels LabelTessellation(mfem::Mesh& local_mesh,
                                     MPI_Comm comm,
                                     const SeedSet& seeds,
                                     const TessellationOptions& tessellation) {
    // Step 1: checks
    CheckSupportedTessellation(tessellation);
    if (seeds.Size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Tessellation labeler: more than 2^31 - 1 seeds");
    }
    if (seeds.Size() == 0) {
        throw std::invalid_argument("Tessellation labeler: the seed set is empty");
    }
    const int num_seeds = static_cast<int>(seeds.Size());

    // Step 2: periodic images (the extended seed list; originals first)
    bool periodic = false;
    for (int d = 0; d < 3; ++d) {
        periodic = periodic || tessellation.periodic[d];
    }
    PeriodicImages images;
    const SeedSet* search_seeds = &seeds;
    int layers = 0;
    if (periodic) {
        const PeriodicBox box = PeriodicBoxFromMesh(local_mesh, comm, tessellation.periodic);
        PeriodicImageParameters image_params;
        layers = RequiredImageLayers(box, tessellation.metric, EffectiveGlobalTensor(tessellation),
                                     seeds);
        image_params.layers = layers;
        images = BuildPeriodicImages(seeds, box, image_params);
        search_seeds = &images.seeds;
    }

    // Step 3: nearest seed per local element, as an index into the input seeds
    mfem::Array<int> element_seed;
    int num_coincident = 0;
    int num_split = 0;
    if (tessellation.sampling == SamplingMode::QUADRATURE_VOTE) {
        QuadratureVoteParameters vote;
        vote.metric = tessellation.metric;
        vote.global_tensor = EffectiveGlobalTensor(tessellation);
        vote.num_bins = tessellation.num_bins;
        vote.levels = tessellation.vote_levels;
        QuadratureVoteReport vote_report;
        LabelElementsByQuadratureVote(local_mesh, *search_seeds, vote, element_seed,
                                      images.original, &vote_report);
        num_coincident = vote_report.num_coincident_seeds;
        num_split = vote_report.num_split_elements;
    } else {
        LabelElementsWithSeeds(local_mesh, *search_seeds, tessellation, element_seed,
                               &num_coincident);
        if (periodic) {
            MapImageLabelsToOriginal(images, element_seed);
        }
    }

    // Step 4: elements without a seed (id1 = -1: non-finite centroid or voting points)
    if (tessellation.sampling == SamplingMode::QUADRATURE_VOTE) {
        ThrowIfAnyElementUnlabeled(comm, element_seed,
                                   "Tessellation labeler: no voting point of an element has a "
                                   "finite position",
                                   "; the element has non-finite node coordinates");
    } else {
        ThrowIfAnyElementUnlabeled(comm, element_seed,
                                   "Tessellation labeler: non-finite element centroid",
                                   "; the element is degenerate (zero volume) or has non-finite "
                                   "node coordinates");
    }

    // Step 5: compact grain ids over the original seeds (collective)
    TessellationLabels labels;
    CompactSeedLabels(comm, element_seed, num_seeds, labels);
    labels.num_coincident_seeds = num_coincident;
    labels.periodic_layers = layers;
    labels.num_periodic_images = images.num_images;
    labels.num_split_elements = num_split;
    return labels;
}

} // namespace exaconstit::microstructure
