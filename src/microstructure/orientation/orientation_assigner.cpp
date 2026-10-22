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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/*
 * CopyNormalizedQuaternion: writes in / |in| to out, |in| = sqrt(w^2 + x^2 + y^2 + z^2) in double
 * precision. This is the single place where every per-grain orientation (supplied or random) is
 * made a unit quaternion. For supplied (measured) orientations the convention, the sign (e.g. a
 * w >= 0 choice of the source) and the component order are kept, only the length changes.
 * Measured quaternions typically have norms within 1e-3 of 1 (the readers refuse anything
 * further off), so the direction is not affected beyond round-off. Random quaternions are
 * normalized too, so their unit norm does not depend on the accuracy of the math functions they
 * were drawn with. A zero or non-finite quaternion has no direction and is an error, never a silent
 * identity. Host only (the per-grain tables are host arrays).
 */
void CopyNormalizedQuaternion(const double* in, double* out, std::size_t grain) {
    const double norm = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2] + in[3] * in[3]);
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        throw std::invalid_argument("AssignOrientations: the supplied quaternion of grain " +
                                    std::to_string(grain + 1) +
                                    " is zero or not finite, so it cannot be normalized");
    }
    for (std::size_t k = 0; k < 4; ++k) {
        out[k] = in[k] / norm;
    }
}

} // namespace

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
            // Drawn, then normalized on the host in double precision like every other row, so
            // the unit norm does not depend on the accuracy of sin/cos/sqrt where it was drawn
            double drawn[4];
            RandomQuaternion(rng, 0u, drawn);
            CopyNormalizedQuaternion(drawn, q, g);
        } else {
            // The parent seed's quaternion (w x y z as supplied), normalized to unit length
            const auto seed = static_cast<std::size_t>(grain_seed[g]);
            CopyNormalizedQuaternion(&seeds.quat[4 * seed], q, g);
        }
    }
}

/*
 * AssignMixedGrainOrientations
 *
 * Steps: check that the three per-grain arrays agree in size; then, grain by grain, copy the
 * supplied quaternion normalized to unit length (CopyNormalizedQuaternion) or draw the random one
 * of the grain's source-id stream (the same
 * stream and int64 -> uint64 conversion as AssignGrainOrientations, so a random grain gets the
 * same orientation whichever function is used).
 */
void AssignMixedGrainOrientations(std::uint64_t rng_seed,
                                  const std::vector<std::int64_t>& grain_source_id,
                                  const std::vector<double>& source_quat,
                                  const std::vector<char>& has_source_quat,
                                  MicrostructureResult& result) {
    // Step 1: sizes (identical on every rank, so all ranks throw together)
    const std::size_t num_grains = grain_source_id.size();
    if (has_source_quat.size() != num_grains || source_quat.size() != 4 * num_grains) {
        throw std::invalid_argument("AssignMixedGrainOrientations: need one flag and 4 quaternion "
                                    "values per grain");
    }

    // Step 2: one row per grain
    result.num_grains = static_cast<int>(num_grains);
    result.grain_quat.assign(4 * num_grains, 0.0);
    result.grain_source_id = grain_source_id;
    for (std::size_t g = 0; g < num_grains; ++g) {
        double* q = &result.grain_quat[4 * g];
        if (has_source_quat[g] != 0) {
            // Supplied (measured) orientation, normalized to unit length, otherwise unchanged
            CopyNormalizedQuaternion(&source_quat[4 * g], q, g);
        } else {
            // Generated grain: random orientation of its own source-id stream
            const CounterRng rng(rng_seed, RngPurpose::ORIENTATION,
                                 static_cast<std::uint64_t>(grain_source_id[g]));
            // Drawn, then normalized on the host in double precision like every other row, so
            // the unit norm does not depend on the accuracy of sin/cos/sqrt where it was drawn
            double drawn[4];
            RandomQuaternion(rng, 0u, drawn);
            CopyNormalizedQuaternion(drawn, q, g);
        }
    }
}

} // namespace exaconstit::microstructure
