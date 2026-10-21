#pragma once

#include "mfem.hpp"

#include <cmath>   // std::frexp, std::ldexp, std::isfinite in the host-only ForRange
#include <cstdint>
#include <math.h>  // C math library floor() in Encode, which also runs in GPU kernels
#include <stdexcept>

/**
 * @file fixed_point.hpp
 * @brief Reproducible global sums: an int64 fixed-point encoding whose sums are bitwise
 * identical for any summation order, thread schedule or MPI rank count.
 *
 * This header holds exactly one concept, the FixedPointScale encoding used to make sums
 * reproducible. It has no dependence on the RNG or on any mesh type.
 *
 * **The problem.** Floating-point addition is not associative: (a + b) + c can differ from
 * a + (b + c) in the last bits. A global sum (total volume, a grain centroid) accumulated with
 * atomics, an OpenMP reduction or `MPI_Allreduce(MPI_SUM)` on doubles therefore can change with
 * the thread schedule or the number of ranks, and a grain label that depends on such a sum could
 * flip between runs.
 *
 * **Fixed-point sums.** Each value is rounded once to the nearest multiple of a small fixed
 * *quantum* (the resolution, e.g. 2^-40) and stored as the integer count of quanta in an int64.
 * Integer addition is exact and associative, so the integer sum is the same for every order and
 * every partition of the terms; decoding it gives a bit-identical double everywhere. Usage:
 * encode each term once, add the integers in any order (atomics, `MPI_SUM` on `MPI_INT64_T`), and
 * decode the total. The price is a fixed absolute resolution: each term carries an error of at
 * most quantum / 2.
 *
 * Overflow bounds (the caller's responsibility; nothing is checked inside kernels):
 * - one term: |x - origin| * inv_quantum must stay below 2^63 (~9.22e18);
 * - a sum of N terms: N * max|x - origin| * inv_quantum must stay below 2^63.
 *
 * `FixedPointScale::ForRange` picks the finest power-of-two quantum that keeps N terms of a given
 * magnitude within 2^62, which leaves a factor of two of headroom for rounding and sign. The
 * resulting quantum is at most 2 * N * max_abs / 2^62 (worst case of the power-of-two rounding).
 * E.g. for 1e9 terms of magnitude 1 that bound is ~4.3e-10, and ForRange actually returns
 * 2^-32 ~ 2.3e-10 per term (absolute). The decoded sum is within N * quantum / 2 of the exact sum.
 */

namespace exaconstit::microstructure {

/**
 * @brief Affine map between doubles and int64: q = round((x - origin) * inv_quantum).
 *
 * Use origin = 0 for non-negative quantities (volumes, weights) and a domain-centered origin for
 * coordinates, so the encoded magnitudes, and with them the overflow risk, stay small.
 * Plain data (two doubles), so it can be captured by value in kernels; all methods except
 * ForRange are host and device and never fail (overflow is not checked, see the file header).
 */
struct FixedPointScale {
    double origin = 0.0;      ///< value encoded as 0 (same units as x)
    double inv_quantum = 1.0; ///< 1 / resolution; a power of two when built with ForRange

    /**
     * @brief Encodes one value, rounding to the nearest quantum (halves round up, toward
     * +infinity, via floor(t + 0.5)).
     */
    MFEM_HOST_DEVICE std::int64_t Encode(double x) const {
        const double quanta = (x - origin) * inv_quantum;
        return static_cast<std::int64_t>(floor(quanta + 0.5));
    }

    /** @brief Decodes one encoded value: origin + q * quantum. */
    MFEM_HOST_DEVICE double Decode(std::int64_t q) const {
        return origin + static_cast<double>(q) / inv_quantum;
    }

    /**
     * @brief Decodes the integer sum of `count` encoded values. Every term was encoded relative
     * to `origin`, so the sum carries `count * origin`, which is added back here.
     * @param sum   integer sum of the encoded terms (e.g. after MPI_Allreduce).
     * @param count global number of terms in the sum (not the local count of one rank).
     */
    MFEM_HOST_DEVICE double DecodeSum(std::int64_t sum, std::int64_t count) const {
        return static_cast<double>(count) * origin + static_cast<double>(sum) / inv_quantum;
    }

    /**
     * @brief Decodes the mean of `count` (> 0) encoded values whose integer sum is `sum`
     * (e.g. a grain centroid coordinate from summed element centroids).
     */
    MFEM_HOST_DEVICE double DecodeMean(std::int64_t sum, std::int64_t count) const {
        return origin + static_cast<double>(sum) / (static_cast<double>(count) * inv_quantum);
    }

    /** @brief The resolution of one term (1 / inv_quantum). */
    MFEM_HOST_DEVICE double Quantum() const {
        return 1.0 / inv_quantum;
    }

    /**
     * @brief Finest power-of-two scale for which a sum of `max_terms` values with
     * |x - origin| <= `max_abs` stays within 2^62.
     *
     * Steps: the largest allowed inv_quantum is limit = 2^62 / (max_terms * max_abs); round it
     * down to a power of two. A power-of-two `inv_quantum` makes the multiplication in Encode and
     * the division in Decode exact (they only shift the binary exponent), so apart from the
     * round-off of x - origin itself the only encode error is the rounding to the nearest
     * quantum.
     *
     * Host only (setup code; uses exceptions). Not collective, but every rank must call it with
     * the same arguments (e.g. global bounds after an MPI reduction) so all ranks use the same
     * scale.
     *
     * @param origin    value encoded as 0.
     * @param max_abs   bound on |x - origin| of any term (> 0, finite).
     * @param max_terms bound on the global number of terms summed (>= 1).
     * @throws std::invalid_argument if `max_abs` is not positive and finite or `max_terms` < 1
     */
    static FixedPointScale ForRange(double origin, double max_abs, std::int64_t max_terms) {
        // Written as !(max_abs > 0) so NaN is rejected too
        if (!(max_abs > 0.0) || !std::isfinite(max_abs) || max_terms < 1) {
            throw std::invalid_argument(
                "FixedPointScale::ForRange: max_abs must be positive and finite, max_terms >= 1");
        }

        // Largest inv_quantum that keeps max_terms * max_abs * inv_quantum <= 2^62
        const double limit = std::ldexp(1.0, 62) / (static_cast<double>(max_terms) * max_abs);

        // Round down to a power of two: frexp writes limit = m * 2^exponent with m in [0.5, 1),
        // so 2^(exponent - 1) is the largest power of two that is <= limit
        int exponent = 0;
        std::frexp(limit, &exponent);

        FixedPointScale scale;
        scale.origin = origin;
        scale.inv_quantum = std::ldexp(1.0, exponent - 1);
        return scale;
    }
};

} // namespace exaconstit::microstructure
