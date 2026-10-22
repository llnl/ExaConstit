#pragma once

#include <cstdint>
#include <vector>

/**
 * @file size_targets.hpp
 * @brief Target grain volumes: equal, or drawn from a log-normal grain size distribution, scaled
 * so they fill the material exactly; plus the derived quantities that start the size fitting
 * (equivalent-sphere radii, a largest-first placement order, Laguerre initial weights).
 *
 * **Grain size distribution.** Grain sizes are commonly described by the *equivalent diameter*
 * d, the diameter of the sphere with the grain's volume, V = (pi / 6) d^3. Measured grain size
 * distributions are close to log-normal: ln d is normally distributed with mean mu and standard
 * deviation sigma. Users give the mean diameter E[d] and sigma; since E[d] = exp(mu + sigma^2/2)
 * for a log-normal, mu = ln(E[d]) - sigma^2 / 2. sigma is dimensionless (0 = all grains equal;
 * 0.2-0.5 is typical of recrystallized metals).
 *
 * **Normalization.** The tessellation fills the material completely, so the targets must add up
 * to the material volume. Drawn volumes are therefore rescaled: V_i^target = V_i x V_material /
 * sum_j V_j. Only the *relative* sizes survive this scaling; the mean diameter matters only for
 * choosing the number of grains (EstimateSeedCount).
 *
 * **Determinism.** The size of grain i is drawn from the counter-based RNG (util/counter_rng.hpp:
 * random numbers computed from (rng_seed, purpose, id), so they don't depend on call order or
 * rank count) keyed by (rng_seed, RngPurpose::SEED_WEIGHT, source_id of seed i). A seed keeps its
 * size when other seeds are added or removed, and every rank computes the same targets.
 *
 * **Starting the fit.** For a Laguerre (power) tessellation, f_i(x) = |x - s_i|^2 - w_i: a seed
 * with weight w_i = r_i^2 "claims" a ball of radius r_i around itself when its neighbors have no
 * weight. With the equivalent-sphere radius r_i = (3 V_i / (4 pi))^(1/3), w_i = r_i^2 is
 * therefore a good first guess for the weight fit (morphology/weight_fit.hpp). Polydisperse
 * random sequential addition (placing seeds with a minimum spacing that depends on their size)
 * works best when the largest grains are placed first, so a largest-first order is provided too.
 *
 * All functions here are host only and not collective; they work on small per-seed vectors that
 * are replicated (identical) on every rank.
 */

namespace exaconstit::microstructure {

/**
 * @brief Largest accepted log-normal sigma. Volumes scale like exp(3 sigma z) with |z| < 8.6 for
 * the 53-bit normal draws, so sigma <= 3 keeps every volume ratio within exp(+-78), far from
 * double overflow/underflow; measured grain size distributions have sigma well below 1.
 */
inline constexpr double MAX_LOGNORMAL_SIGMA = 3.0;

/** @brief Shape of the target size distribution. */
enum class SizeDistribution {
    EQUAL,    ///< every grain gets V_material / N
    LOGNORMAL ///< ln(equivalent diameter) ~ normal(mu, sigma), mu from the mean diameter
};

/** @brief Parameters of the target size distribution. Host data. */
struct SizeTargetParameters {
    /** @brief Distribution shape. */
    SizeDistribution distribution = SizeDistribution::EQUAL;
    /**
     * @brief Mean equivalent diameter E[d] (mesh length units, > 0). Used only by
     * EstimateSeedCount; the normalized targets do not depend on it.
     */
    double mean_diameter = 1.0;
    /** @brief Standard deviation of ln d (dimensionless, in [0, MAX_LOGNORMAL_SIGMA]). LOGNORMAL
     * only. */
    double sigma = 0.0;
    /** @brief Key of the counter-based RNG (the run's rng_seed). */
    std::uint64_t rng_seed = 0;
};

/**
 * @brief Target volumes and the derived starting data of the size fit, one entry per seed
 * (0-based seed index, the order of the SeedSet). Host data, identical on every rank.
 */
struct SizeTargets {
    /** @brief V_i^target (length^3), > 0, summing to material_volume. */
    std::vector<double> volume;
    /** @brief Equivalent-sphere radius r_i = (3 V_i / (4 pi))^(1/3) (length). */
    std::vector<double> radius;
    /** @brief Laguerre initial weight w_i = r_i^2 (length^2). */
    std::vector<double> initial_weight;
    /**
     * @brief Seed indices from the largest to the smallest target volume; equal volumes are
     * ordered by increasing seed index. A permutation of 0..N-1.
     */
    std::vector<int> placement_order;
    /** @brief The material volume the targets were scaled to (length^3). */
    double material_volume = 0.0;
};

/**
 * @brief Checks the parameters.
 * @throws std::invalid_argument if mean_diameter is not positive and finite or sigma is outside
 *         [0, MAX_LOGNORMAL_SIGMA] (LOGNORMAL only).
 */
void ValidateSizeTargetParameters(const SizeTargetParameters& params);

/**
 * @brief Draws one target volume per seed from the distribution and normalizes them to the
 * material volume.
 *
 * @param source_id       stable id of every seed (SeedSet::source_id), size N >= 1; the RNG key.
 * @param material_volume total material volume (length^3, > 0), e.g. CellMeasures::total_volume
 *                        or the sum of the element volumes.
 * @param params          distribution parameters.
 * @return targets for the N seeds (see SizeTargets).
 * @throws std::invalid_argument for N == 0, a non-positive material volume or invalid params.
 */
SizeTargets MakeSizeTargets(const std::vector<std::int64_t>& source_id, double material_volume,
                            const SizeTargetParameters& params);

/**
 * @brief Targets from given relative volumes (e.g. grain volumes measured by HEDM), normalized
 * to the material volume.
 *
 * @param relative_volume one positive, finite value per seed (any units; only ratios matter).
 * @param material_volume total material volume (length^3, > 0).
 * @throws std::invalid_argument for an empty input, a non-positive or non-finite value or a
 *         non-positive material volume.
 */
SizeTargets MakeSizeTargetsFromVolumes(const std::vector<double>& relative_volume,
                                       double material_volume);

/**
 * @brief Number of grains whose expected total volume equals the material volume:
 * round(V_material / E[V]) with E[V] = (pi / 6) E[d^3] and, for the log-normal,
 * E[d^3] = exp(3 mu + 9 sigma^2 / 2). At least 1.
 *
 * @param material_volume total material volume (length^3, > 0).
 * @param params          distribution parameters (mean_diameter in mesh length units).
 * @throws std::invalid_argument for invalid parameters or a count above 2^31 - 1.
 */
std::int64_t EstimateSeedCount(double material_volume, const SizeTargetParameters& params);

} // namespace exaconstit::microstructure
