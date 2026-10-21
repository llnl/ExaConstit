#pragma once

#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"

#include <cstdint>
#include <vector>

/**
 * @file orientation_assigner.hpp
 * @brief One crystal orientation per grain: uniformly random, or passed through from the seed
 * source.
 *
 * Orientations are unit quaternions in ExaConstit's `w x y z` order, crystal-to-sample and
 * passive (the convention of ExaConstit's orientation file). They are never converted: random
 * orientations are uniform, so the convention does not change their distribution, and
 * orientations supplied by the user are copied verbatim.
 *
 * **Random orientations.** Grain g grew from seed i = grain_seed[g-1]; its orientation is
 * RandomQuaternion (orientation/random_orientation.hpp) of the counter-based random stream keyed
 * by (rng_seed, RngPurpose::ORIENTATION, SeedSet::source_id[i]) (the int64 id converted to the
 * RNG's uint64 id, a one-to-one conversion, so negative table ids are allowed). A
 * counter-based random number generator computes each number from (seed, purpose, id) instead of
 * drawing "the next number", so the result does not depend on call order or rank count. Keying
 * by the seed's source id (not by the grain id g) means a grain keeps its orientation when other
 * seeds are rejected or win no element, which would renumber the grains.
 */

namespace exaconstit::microstructure {

/**
 * @brief Fills the per-grain orientation and source-id tables of `result`.
 *
 * - orientations.source = "random": result.grain_quat[4(g-1) .. 4(g-1)+3] =
 *   RandomQuaternion(CounterRng(rng_seed, ORIENTATION, seeds.source_id[grain_seed[g-1]]),
 *   sample 0), canonical sign w >= 0;
 * - orientations.source = "from_source": the 4 values seeds.quat[4i .. 4i+3] of the grain's seed
 *   i, copied verbatim (no normalization or sign change).
 *
 * result.grain_source_id[g-1] = seeds.source_id[grain_seed[g-1]] in both cases, and
 * result.num_grains is set to grain_seed.size(). Regions are not touched.
 *
 * Host only (the per-grain tables are small, replicated host arrays); not collective, but every
 * rank gets bitwise identical tables because the inputs are replicated.
 *
 * @param orientations  [Orientations] options; source must be "random" or "from_source".
 * @param rng_seed      Seeds.rng_seed, the key of the counter-based RNG.
 * @param seeds         replicated seed set; source_id must have one entry per seed, and quat 4
 *                      entries per seed for "from_source".
 * @param grain_seed    0-based seed index of grain g at index g-1 (from the labeler).
 * @param result        output: num_grains, grain_quat (size 4 num_grains) and grain_source_id
 *                      (size num_grains) are overwritten.
 * @throws std::invalid_argument for source = "list" (not implemented yet) or any other value,
 *         for "from_source" without per-seed quaternions, a source_id array of the wrong size,
 *         or a seed index out of range.
 */
void AssignOrientations(const OrientationOptions& orientations,
                        std::uint64_t rng_seed,
                        const SeedSet& seeds,
                        const std::vector<int>& grain_seed,
                        MicrostructureResult& result);

} // namespace exaconstit::microstructure
