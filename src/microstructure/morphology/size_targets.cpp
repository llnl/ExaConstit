/*
 * Implementation of the target grain volumes (size_targets.hpp). Everything is host code over
 * small per-seed vectors; the log-normal draws use the MFEM_HOST_DEVICE distribution helpers of
 * util/random_distributions.hpp, called here on the host.
 */
#include "microstructure/morphology/size_targets.hpp"

#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/random_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

constexpr double PI = 3.14159265358979323846;

/**
 * @brief Fills radius, initial weight and placement order from the (already normalized) target
 * volumes.
 *
 * Radius: the sphere of volume V has radius (3 V / (4 pi))^(1/3). Weight: r^2 (see the file
 * comment of size_targets.hpp). Order: indices sorted by decreasing volume; std::stable_sort
 * keeps equal volumes in increasing index, so the order is fully determined by the volumes.
 */
void FillDerived(SizeTargets& targets) {
    const std::size_t n = targets.volume.size();
    targets.radius.resize(n);
    targets.initial_weight.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        targets.radius[i] = std::cbrt(3.0 * targets.volume[i] / (4.0 * PI));
        targets.initial_weight[i] = targets.radius[i] * targets.radius[i];
    }
    targets.placement_order.resize(n);
    std::iota(targets.placement_order.begin(), targets.placement_order.end(), 0);
    std::stable_sort(targets.placement_order.begin(), targets.placement_order.end(),
                     [&](int a, int b) {
                         return targets.volume[static_cast<std::size_t>(a)] >
                                targets.volume[static_cast<std::size_t>(b)];
                     });
}

/** @brief Throws unless the material volume is positive and finite. */
void CheckMaterialVolume(double material_volume) {
    if (!(material_volume > 0.0) || !std::isfinite(material_volume)) {
        throw std::invalid_argument("Size targets: the material volume must be positive and "
                                    "finite");
    }
}

/**
 * @brief Scales `raw` (positive values) so they sum to material_volume. The sum is taken in
 * increasing seed index, the same on every rank.
 */
void Normalize(std::vector<double>& raw, double material_volume) {
    double sum = 0.0;
    for (double v : raw) {
        sum += v;
    }
    const double factor = material_volume / sum;
    for (double& v : raw) {
        v *= factor;
    }
}

} // namespace

/*
 * ValidateSizeTargetParameters: the mean diameter enters a logarithm and sigma a normal draw, so
 * both must be finite, and sigma is bounded by MAX_LOGNORMAL_SIGMA so that volume ratios stay
 * representable (see size_targets.hpp); written as !(x > 0) so NaN fails too.
 */
void ValidateSizeTargetParameters(const SizeTargetParameters& params) {
    if (!(params.mean_diameter > 0.0) || !std::isfinite(params.mean_diameter)) {
        throw std::invalid_argument("Size targets: mean_diameter must be positive and finite");
    }
    if (params.distribution == SizeDistribution::LOGNORMAL &&
        (!(params.sigma >= 0.0) || !(params.sigma <= MAX_LOGNORMAL_SIGMA))) {
        throw std::invalid_argument("Size targets: sigma must be in [0, " +
                                    std::to_string(MAX_LOGNORMAL_SIGMA) + "]");
    }
}

/*
 * MakeSizeTargets
 *
 * Steps:
 *  1. Draw a raw volume per seed. EQUAL: 1 for every seed. LOGNORMAL: d_i = exp(mu + sigma z_i)
 *     with z_i standard normal from CounterRng(rng_seed, SEED_WEIGHT, source_id_i), sample 0,
 *     and V_i = (pi / 6) d_i^3, which is proportional to exp(3 sigma z_i); only that factor is
 *     formed, because the normalization removes the common part (and forming d_i^3 directly
 *     could overflow or underflow for extreme mean diameters). The source id (not the seed
 *     index) is the key, so a seed keeps its size when other candidate seeds are rejected.
 *  2. Normalize to the material volume.
 *  3. Radii, initial weights and the largest-first order.
 */
SizeTargets MakeSizeTargets(const std::vector<std::int64_t>& source_id, double material_volume,
                            const SizeTargetParameters& params) {
    if (source_id.empty()) {
        throw std::invalid_argument("Size targets: no seeds");
    }
    CheckMaterialVolume(material_volume);
    ValidateSizeTargetParameters(params);

    // Step 1: raw volumes
    std::vector<double> raw(source_id.size(), 1.0);
    if (params.distribution == SizeDistribution::LOGNORMAL) {
        for (std::size_t i = 0; i < source_id.size(); ++i) {
            // int64 -> uint64 is the defined modulo-2^64 conversion (one-to-one)
            const CounterRng rng(params.rng_seed, RngPurpose::SEED_WEIGHT,
                                 static_cast<std::uint64_t>(source_id[i]));
            // V_i = (pi/6) exp(3 mu + 3 sigma z_i); the factor (pi/6) exp(3 mu) is the same for
            // every seed and cancels in the normalization, so only exp(3 sigma z_i) is formed.
            // With |z| < 8.6 (53-bit uniforms) and sigma <= 3 it lies within exp(+-78): finite.
            const double z = StandardNormal(rng, 0);
            raw[i] = std::exp(3.0 * params.sigma * z);
            if (!(raw[i] > 0.0) || !std::isfinite(raw[i])) {
                throw std::invalid_argument("Size targets: a log-normal volume ratio is not "
                                            "representable; reduce sigma");
            }
        }
    }

    // Steps 2-3
    SizeTargets targets;
    targets.material_volume = material_volume;
    Normalize(raw, material_volume);
    targets.volume = std::move(raw);
    FillDerived(targets);
    return targets;
}

/*
 * MakeSizeTargetsFromVolumes: validate, normalize, derive (steps 2-3 of MakeSizeTargets).
 */
SizeTargets MakeSizeTargetsFromVolumes(const std::vector<double>& relative_volume,
                                       double material_volume) {
    if (relative_volume.empty()) {
        throw std::invalid_argument("Size targets: no seeds");
    }
    CheckMaterialVolume(material_volume);
    for (double v : relative_volume) {
        if (!(v > 0.0) || !std::isfinite(v)) {
            throw std::invalid_argument("Size targets: every relative volume must be positive "
                                        "and finite");
        }
    }
    SizeTargets targets;
    targets.material_volume = material_volume;
    targets.volume = relative_volume;
    Normalize(targets.volume, material_volume);
    FillDerived(targets);
    return targets;
}

/*
 * EstimateSeedCount: N = V_material / E[V]. For d log-normal(mu, sigma), d^3 is log-normal(3 mu,
 * 3 sigma), whose mean is exp(3 mu + 9 sigma^2 / 2); for EQUAL, E[d^3] = mean_diameter^3.
 */
std::int64_t EstimateSeedCount(double material_volume, const SizeTargetParameters& params) {
    CheckMaterialVolume(material_volume);
    ValidateSizeTargetParameters(params);
    double mean_d3 = params.mean_diameter * params.mean_diameter * params.mean_diameter;
    if (params.distribution == SizeDistribution::LOGNORMAL) {
        const double mu = std::log(params.mean_diameter) - 0.5 * params.sigma * params.sigma;
        mean_d3 = std::exp(3.0 * mu + 4.5 * params.sigma * params.sigma);
    }
    const double mean_volume = PI / 6.0 * mean_d3;
    if (!(mean_volume > 0.0) || !std::isfinite(mean_volume)) {
        throw std::invalid_argument("Size targets: the mean grain volume is not representable; "
                                    "check mean_diameter");
    }
    const double count = std::round(material_volume / mean_volume);
    if (!(count <= static_cast<double>(std::numeric_limits<int>::max()))) {
        throw std::invalid_argument("Size targets: the estimated grain count exceeds 2^31 - 1; "
                                    "increase mean_diameter");
    }
    return std::max<std::int64_t>(1, static_cast<std::int64_t>(count));
}

} // namespace exaconstit::microstructure
