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
 * AssignOrientations
 *
 * Steps:
 *  1. Check the inputs: a supported source, one source id per seed, per-seed quaternions for
 *     "from_source", and every grain's seed index inside the seed set.
 *  2. For each grain g (row g-1): look up its seed i, store the seed's source id, then either
 *     draw the random quaternion of stream (rng_seed, ORIENTATION, source_id) or copy the seed's
 *     4 quaternion values. The int64 source id is converted to the RNG's uint64 id with
 *     static_cast, which C++ defines as the value modulo 2^64: non-negative ids keep their value,
 *     and a negative id (possible for seed tables) maps to a distinct large value. The map is
 *     one-to-one, so distinct source ids always get distinct random streams.
 */
void AssignOrientations(const OrientationOptions& orientations,
                        std::uint64_t rng_seed,
                        const SeedSet& seeds,
                        const std::vector<int>& grain_seed,
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
    for (const int seed : grain_seed) {
        if (seed < 0 || static_cast<std::size_t>(seed) >= num_seeds) {
            throw std::invalid_argument("AssignOrientations: grain seed index out of range");
        }
    }

    // Step 2: one row per grain
    const std::size_t num_grains = grain_seed.size();
    result.num_grains = static_cast<int>(num_grains);
    result.grain_quat.assign(4 * num_grains, 0.0);
    result.grain_source_id.assign(num_grains, 0);
    for (std::size_t g = 0; g < num_grains; ++g) {
        const auto seed = static_cast<std::size_t>(grain_seed[g]);
        const std::int64_t source_id = seeds.source_id[seed];
        result.grain_source_id[g] = source_id;

        double* q = &result.grain_quat[4 * g];
        if (random) {
            // Stream of this seed: independent of grain numbering, rank count and call order
            const CounterRng rng(rng_seed, RngPurpose::ORIENTATION,
                                 static_cast<std::uint64_t>(source_id));
            RandomQuaternion(rng, 0u, q);
        } else {
            // Verbatim pass-through (w x y z as supplied)
            for (std::size_t k = 0; k < 4; ++k) {
                q[k] = seeds.quat[4 * seed + k];
            }
        }
    }
}

} // namespace exaconstit::microstructure
