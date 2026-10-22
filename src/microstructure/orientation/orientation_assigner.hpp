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
 * orientations supplied by the user are normalized to unit length (q / |q|, double precision)
 * and otherwise passed through unchanged (convention, sign and component order kept). Random
 * quaternions are normalized the same way (RandomQuaternion's output divided by its norm on the
 * host in double precision), so the unit norm does not depend on the accuracy of sin, cos and
 * sqrt where they were drawn; this may change their last bit relative to RandomQuaternion. Every
 * row written here therefore has norm 1 to machine precision.
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
 *   sample 0) divided by its norm, canonical sign w >= 0;
 * - orientations.source = "from_source": the 4 values seeds.quat[4i .. 4i+3] of the grain's seed
 *   i, normalized to unit length (no sign change or reordering).
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
 *         for "from_source" without per-seed quaternions or with a zero / non-finite one, a
 *         source_id array of the wrong size, or a seed index out of range.
 */
void AssignOrientations(const OrientationOptions& orientations,
                        std::uint64_t rng_seed,
                        const SeedSet& seeds,
                        const std::vector<int>& grain_seed,
                        MicrostructureResult& result);

/**
 * @brief AssignOrientations with an explicit source id per grain, for grains that do not map
 * one-to-one onto seeds (after fragment cleanup split a grain into several).
 *
 * Grain g (row g-1) has the parent seed grain_seed[g-1] and the source id
 * grain_source_id[g-1]:
 * - "random": the orientation is RandomQuaternion of the stream (rng_seed, ORIENTATION,
 *   grain_source_id[g-1]), normalized. A grain split off another one carries a derived source id
 *   (DeriveFragmentSourceId, labeling/fragment_cleanup.hpp), so it gets its own reproducible
 *   random orientation; a grain that kept its seed's source id gets exactly the orientation
 *   AssignOrientations would give it.
 * - "from_source": the 4 values of the parent seed's quaternion, normalized to unit length, so
 *   every piece of a measured grain keeps the measured orientation.
 *
 * result.grain_source_id = grain_source_id and result.num_grains = grain_seed.size().
 * Host only, not collective (replicated inputs give identical tables on every rank).
 *
 * @param grain_seed      0-based parent seed index of every grain (size num_grains).
 * @param grain_source_id source id of every grain (size num_grains).
 * @throws std::invalid_argument as AssignOrientations, and if the two per-grain arrays differ in
 *         size.
 */
void AssignGrainOrientations(const OrientationOptions& orientations,
                             std::uint64_t rng_seed,
                             const SeedSet& seeds,
                             const std::vector<int>& grain_seed,
                             const std::vector<std::int64_t>& grain_source_id,
                             MicrostructureResult& result);

/**
 * @brief Orientations when only some grains have a supplied orientation: a mix of measured and
 * generated grains (seed-table grains and random filler grains, or voxel-map features).
 *
 * Grain g (row g-1):
 * - has_source_quat[g-1] != 0: the 4 values source_quat[4(g-1) .. 4(g-1)+3] (w x y z) divided by
 *   their norm (no sign change or reordering), e.g. the measured orientation of its seed or
 *   voxel feature (a piece split off a measured grain also copies its parent's orientation);
 * - otherwise: the random quaternion of the stream (rng_seed, ORIENTATION, grain_source_id[g-1]),
 *   normalized: exactly what AssignGrainOrientations gives a random grain with that source id.
 *
 * result.num_grains, grain_quat and grain_source_id are overwritten; nothing else is touched.
 * Host only, not collective (replicated inputs give identical tables on every rank).
 *
 * @param rng_seed        Seeds.rng_seed, the key of the counter-based RNG.
 * @param grain_source_id source id of every grain (size N = number of grains).
 * @param source_quat     4 N values; rows of grains without a supplied orientation are ignored.
 * @param has_source_quat N flags (1 = copy the supplied row, 0 = random).
 * @param result          output.
 * @throws std::invalid_argument if the sizes do not match, or a supplied row is zero or not
 *         finite (it cannot be normalized).
 */
void AssignMixedGrainOrientations(std::uint64_t rng_seed,
                                  const std::vector<std::int64_t>& grain_source_id,
                                  const std::vector<double>& source_quat,
                                  const std::vector<char>& has_source_quat,
                                  MicrostructureResult& result);

} // namespace exaconstit::microstructure
