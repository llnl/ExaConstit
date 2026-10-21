#pragma once

#include "microstructure/util/counter_rng.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <math.h>

/**
 * @file random_distributions.hpp
 * @brief Statistical distributions (uniform on [a, b], uniform in the unit cube, normal,
 * log-normal) sampled from the counter-based RNG of util/counter_rng.hpp.
 *
 * Each function takes a CounterRng (the stream of one entity, keyed by (rng_seed, purpose, id))
 * and a sample index, and returns the same value whenever it is called with the same arguments,
 * on any rank, thread or device. Header-only, stateless and `MFEM_HOST_DEVICE`, so usable inside
 * mfem::forall kernels. Math uses the C math library functions from <math.h> (`sqrt`, `log`,
 * `cos`, `exp`, double precision), which CUDA and HIP provide in device code; the C++ `std::`
 * versions are not used because they are not reliably available inside GPU kernels. Unlike the
 * raw RNG bits, these functions may round differently in the host and device math libraries.
 *
 * Bit usage (which blocks of the stream each function reads) is documented per function, so
 * callers can tell when two calls share bits.
 */

namespace exaconstit::microstructure {

namespace detail {
constexpr double TWO_PI = 6.283185307179586476925286766559;
} // namespace detail

/**
 * @brief Uniform double in [a, b] (a < b); same bits as rng.UniformDouble(sample), i.e. block
 * (sample, 0), words 0-1. The unit value is in [0, 1), but a + (b - a) * u can round up to exactly
 * b, so b itself is possible (rarely): treat the range as [a, b) only up to round-off.
 */
MFEM_HOST_DEVICE inline double UniformIn(const CounterRng& rng, double a, double b,
                                         std::uint32_t sample) {
    return a + (b - a) * rng.UniformDouble(sample);
}

/**
 * @brief Three independent uniforms in [0, 1), e.g. a point in the unit cube that is then mapped
 * to a bounding box. Needs 192 bits, so it reads blocks (sample, 0) (words 0-3) and (sample, 1)
 * (words 0-1).
 * @param out the three values, written to out[0..2].
 */
MFEM_HOST_DEVICE inline void UniformUnitCube(const CounterRng& rng, std::uint32_t sample,
                                             double out[3]) {
    const PhiloxBlock b0 = rng.Block(sample, 0);
    const PhiloxBlock b1 = rng.Block(sample, 1);
    out[0] = BitsToUnitDouble(b0.v[0], b0.v[1]);
    out[1] = BitsToUnitDouble(b0.v[2], b0.v[3]);
    out[2] = BitsToUnitDouble(b1.v[0], b1.v[1]);
}

/**
 * @brief Standard normal N(0, 1) by the Box-Muller transform: for independent uniforms u1 in
 * (0, 1] and u2 in [0, 1), sqrt(-2 ln u1) cos(2 pi u2) is exactly normally distributed.
 *
 * Box-Muller produces a pair of normals (cos and sin); only the cosine one is used, so one
 * sample index gives one normal and nothing is cached between calls. Reads block (sample, 0).
 */
MFEM_HOST_DEVICE inline double StandardNormal(const CounterRng& rng, std::uint32_t sample) {
    const PhiloxBlock b = rng.Block(sample);
    // 1 - u lies in (0, 1], so the log is finite
    const double u1 = 1.0 - BitsToUnitDouble(b.v[0], b.v[1]);
    const double u2 = BitsToUnitDouble(b.v[2], b.v[3]);
    const double radius = sqrt(-2.0 * log(u1));
    return radius * cos(detail::TWO_PI * u2);
}

/**
 * @brief Normal with mean `mu` and standard deviation `sigma` (same bits as
 * StandardNormal(rng, sample)).
 */
MFEM_HOST_DEVICE inline double Normal(const CounterRng& rng, double mu, double sigma,
                                      std::uint32_t sample) {
    return mu + sigma * StandardNormal(rng, sample);
}

/**
 * @brief Log-normal: exp(N(mu, sigma)), e.g. for grain-size-like weights. `mu` and `sigma` are
 * the mean and standard deviation of the underlying normal (of the log), not of the result.
 * Same bits as StandardNormal(rng, sample).
 */
MFEM_HOST_DEVICE inline double LogNormal(const CounterRng& rng, double mu, double sigma,
                                         std::uint32_t sample) {
    return exp(Normal(rng, mu, sigma, sample));
}

} // namespace exaconstit::microstructure
