/*
 * Implementation of the per-grain orientation assignment (orientation_assigner.hpp).
 *
 * The tables built here have one row per grain (thousands, rarely millions of rows) and are
 * replicated on every rank, so they are filled with plain host loops; RandomQuaternion is the
 * same host/device function the GPU code paths use, so a later device version gives identical
 * bits.
 */
#include "microstructure/orientation/orientation_assigner.hpp"

#include "microstructure/orientation/random_orientation.hpp"
#include "microstructure/util/counter_rng.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

/*
 * AssignOrientations: the per-grain source id is the source id of the grain's seed; everything
 * else is AssignGrainOrientations. Checks that need the seed set (one source id per seed, seed
 * indices in range) run first, so the error messages are the same as before.
 */
void AssignOrientations(const OrientationOptions& orientations,
                        std::uint64_t rng_seed,
                        const SeedSet& seeds,
                        const std::vector<int>& grain_seed,
                        MicrostructureResult& result) {
    const std::size_t num_seeds = seeds.Size();
    if (seeds.source_id.size() != num_seeds) {
        throw std::invalid_argument(
            "AssignOrientations: SeedSet::source_id must have one entry per seed");
    }
    std::vector<std::int64_t> grain_source_id(grain_seed.size(), 0);
    for (std::size_t g = 0; g < grain_seed.size(); ++g) {
        const int seed = grain_seed[g];
        if (seed >= 0 && static_cast<std::size_t>(seed) < num_seeds) {
            grain_source_id[g] = seeds.source_id[static_cast<std::size_t>(seed)];
        }
        // An out-of-range index is reported by AssignGrainOrientations below
    }
    AssignGrainOrientations(orientations, rng_seed, seeds, grain_seed, grain_source_id, result);
}

/*
 * AssignGrainOrientations
 *
 * Steps:
 *  1. Check the inputs: a supported source, one source id per seed, per-seed quaternions for
 *     "from_source", matching per-grain array sizes, and every grain's seed index inside the seed
 *     set.
 *  2. For each grain g (row g-1): store its source id, then either draw the random quaternion of
 *     stream (rng_seed, ORIENTATION, source id) or copy the parent seed's 4 quaternion values.
 *     The int64 source id is converted to the RNG's uint64 id with static_cast, which C++ defines
 *     as the value modulo 2^64: non-negative ids keep their value, and a negative id (seed tables,
 *     or the derived ids of split grains) maps to a distinct large value. The map is one-to-one,
 *     so distinct source ids always get distinct random streams.
 */
void AssignGrainOrientations(const OrientationOptions& orientations,
                             std::uint64_t rng_seed,
                             const SeedSet& seeds,
                             const std::vector<int>& grain_seed,
                             const std::vector<std::int64_t>& grain_source_id,
                             MicrostructureResult& result) {
    // Step 1: input checks (identical on every rank, so all ranks throw together)
    const bool random = (orientations.source == "random");
    const bool from_source = (orientations.source == "from_source");
    if (orientations.source == "list") {
        throw std::invalid_argument(
            "AssignOrientations: Orientations.source = \"list\" is not implemented yet");
    }
    if (!random && !from_source) {
        throw std::invalid_argument("AssignOrientations: unknown Orientations.source \"" +
                                    orientations.source + "\"");
    }
    const std::size_t num_seeds = seeds.Size();
    if (seeds.source_id.size() != num_seeds) {
        throw std::invalid_argument(
            "AssignOrientations: SeedSet::source_id must have one entry per seed");
    }
    if (from_source && seeds.quat.size() != 4 * num_seeds) {
        throw std::invalid_argument(
            "AssignOrientations: Orientations.source = \"from_source\" needs 4 quaternion values "
            "per seed (SeedSet::quat), but the seed source provides none");
    }
    if (grain_source_id.size() != grain_seed.size()) {
        throw std::invalid_argument(
            "AssignOrientations: grain_seed and grain_source_id must have the same size");
    }
    for (const int seed : grain_seed) {
        if (seed < 0 || static_cast<std::size_t>(seed) >= num_seeds) {
            throw std::invalid_argument("AssignOrientations: grain seed index out of range");
        }
    }

    // Step 2: one row per grain
    const std::size_t num_grains = grain_seed.size();
    result.num_grains = static_cast<int>(num_grains);
    result.grain_quat.assign(4 * num_grains, 0.0);
    result.grain_source_id = grain_source_id;
    for (std::size_t g = 0; g < num_grains; ++g) {
        double* q = &result.grain_quat[4 * g];
        if (random) {
            // Stream of this grain's source id: independent of grain numbering, rank count and
            // call order
            const CounterRng rng(rng_seed, RngPurpose::ORIENTATION,
                                 static_cast<std::uint64_t>(grain_source_id[g]));
            RandomQuaternion(rng, 0u, q);
        } else {
            // Verbatim pass-through of the parent seed's quaternion (w x y z as supplied)
            const auto seed = static_cast<std::size_t>(grain_seed[g]);
            for (std::size_t k = 0; k < 4; ++k) {
                q[k] = seeds.quat[4 * seed + k];
            }
        }
    }
}

} // namespace exaconstit::microstructure
