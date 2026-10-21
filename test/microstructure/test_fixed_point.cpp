/*
 * Tests of util/fixed_point.hpp, reproducible global sums. Each double is rounded once to an
 * integer multiple of a small "quantum" and stored as an int64; integer addition gives the same
 * total in any order, so global sums (volumes, centroids) are bitwise reproducible across thread
 * schedules and MPI rank counts. The last test uses MPI and is run at np = 1..4.
 */
#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/fixed_point.hpp"
#include "microstructure/util/random_distributions.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

/**
 * @brief `n` test values with magnitudes spread over six decades (|x| in [1e-6, 1]) and both
 * signs. Adding such values in double precision loses different low-order bits depending on the
 * order, which is exactly the situation fixed-point sums are meant to fix.
 */
std::vector<double> MixedMagnitudeValues(int n) {
    std::vector<double> values(n);
    for (int i = 0; i < n; ++i) {
        const ms::CounterRng rng(1u, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        const double sign_and_mantissa = ms::UniformIn(rng, -1.0, 1.0, 0u);
        const double decade = ms::UniformIn(rng, -6.0, 0.0, 1u);
        values[i] = sign_and_mantissa * std::pow(10.0, decade);
    }
    return values;
}

/**
 * @brief Reorders `values` with a Fisher-Yates shuffle driven by the counter RNG. Unlike
 * std::shuffle (whose algorithm differs between standard libraries) this gives the same order on
 * every platform, so the test is reproducible everywhere.
 */
std::vector<double> CounterShuffle(const std::vector<double>& values, std::uint64_t key) {
    std::vector<double> out = values;
    const ms::CounterRng rng(key, ms::RngPurpose::GENERIC, 0u);
    for (std::size_t i = out.size() - 1; i > 0; --i) {
        // Pick j uniformly in [0, i]
        const double u = rng.UniformDouble(static_cast<std::uint32_t>(i));
        const auto j = static_cast<std::size_t>(u * static_cast<double>(i + 1));
        std::swap(out[i], out[j]);
    }
    return out;
}

/** @brief Plain left-to-right double-precision sum (what MPI_SUM on doubles does per rank). */
double DoubleSum(const std::vector<double>& values) {
    double sum = 0.0;
    for (double x : values) {
        sum += x;
    }
    return sum;
}

/** @brief Fixed-point sum: encode every term, add the integers, decode the total. */
double FixedPointSum(const std::vector<double>& values, const ms::FixedPointScale& scale) {
    std::int64_t sum = 0;
    for (double x : values) {
        sum += scale.Encode(x);
    }
    return scale.DecodeSum(sum, static_cast<std::int64_t>(values.size()));
}

/**
 * @brief Accurate reference sum, used only to bound the error of the fixed-point sums.
 *
 * Neumaier's compensated summation: alongside the running double sum it accumulates, in
 * `compensation`, the low-order bits each addition rounds away, and adds them back at the end.
 * The result is accurate to about one rounding of the final sum (~1e-16 relative), independent
 * of the number of terms, far below the N * quantum / 2 bounds tested here. (long double is not
 * used: on arm64 macOS it is the same 8-byte type as double, so it would add no precision.)
 */
double ExactSum(const std::vector<double>& values) {
    double sum = 0.0;
    double compensation = 0.0;
    for (double x : values) {
        const double t = sum + x;
        // Recover the part of the smaller operand that was rounded away in t
        if (std::abs(sum) >= std::abs(x)) {
            compensation += (sum - t) + x;
        } else {
            compensation += (x - t) + sum;
        }
        sum = t;
    }
    return sum + compensation;
}

} // namespace

// ForRange picks the finest power-of-two scale whose worst-case sum fits in the 2^62 budget:
// - inv_quantum is a power of two (frexp mantissa exactly 0.5), so scaling by it is exact;
// - N * max_abs * inv_quantum <= 2^62 (fits), and doubling inv_quantum would exceed it (finest);
// - N terms all at the extreme value +max_abs really sum without int64 overflow.
// Encode/Decode round-trip error is at most half a quantum, because Encode rounds to the nearest
// integer and the power-of-two scaling is exact; the only other errors are the double roundings of
// x - origin and origin + q * quantum, a few ulps of x, allowed for in the tolerance. Invalid
// ranges are rejected.
TEST(FixedPoint, ForRangeBoundsAndRoundTrip) {
    const double origin = 0.5;
    const double max_abs = 2.0;
    const std::int64_t max_terms = 1000000;
    const auto scale = ms::FixedPointScale::ForRange(origin, max_abs, max_terms);

    // Power of two, within the 2^62 budget, and no coarser than the budget requires
    int exponent = 0;
    EXPECT_EQ(std::frexp(scale.inv_quantum, &exponent), 0.5);
    const double budget = static_cast<double>(max_terms) * max_abs * scale.inv_quantum;
    EXPECT_LE(budget, std::ldexp(1.0, 62));
    EXPECT_GT(2.0 * budget, std::ldexp(1.0, 62));

    // Worst case: every term at origin + max_abs. Each encodes to max_abs * inv_quantum, so
    // max_terms of them must not exceed 2^62 (checked by division, which cannot overflow).
    const std::int64_t extreme = scale.Encode(origin + max_abs);
    EXPECT_LE(extreme, (std::int64_t{1} << 62) / max_terms);

    // The origin encodes to 0; values symmetric about the origin encode to opposite integers
    EXPECT_EQ(scale.Encode(origin), 0);
    EXPECT_EQ(scale.Encode(2.5), -scale.Encode(-1.5));

    // Round trip of many values spread over the whole allowed range [origin - max_abs,
    // origin + max_abs]: error <= quantum / 2 (+ a few ulps of x for the double arithmetic)
    const double half_quantum = 0.5 * scale.Quantum();
    for (int i = 0; i < 10000; ++i) {
        const ms::CounterRng rng(3u, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        const double x = ms::UniformIn(rng, origin - max_abs, origin + max_abs, 0u);
        const double ulps = 4.0 * std::numeric_limits<double>::epsilon() * std::abs(x);
        EXPECT_LE(std::abs(scale.Decode(scale.Encode(x)) - x), half_quantum + ulps) << "x = " << x;
    }

    EXPECT_THROW(ms::FixedPointScale::ForRange(0.0, 0.0, 10), std::invalid_argument);
    EXPECT_THROW(ms::FixedPointScale::ForRange(0.0, 1.0, 0), std::invalid_argument);
}

// Why fixed-point sums exist, shown on data: adding the same mixed-magnitude doubles in different
// orders (as different thread schedules or rank counts would) gives double results that differ
// in the last bits, while the fixed-point sum decodes to the bitwise identical double in every
// order. The fixed-point result is also accurate: each term is rounded by at most quantum/2, so
// the decoded sum is within N * quantum / 2 of the exact sum (plus the single rounding of the
// final decode to double, ~1 ulp of the sum).
TEST(FixedPoint, SumIsOrderIndependentWhereDoubleSumIsNot) {
    const int n = 100000;
    const std::vector<double> values = MixedMagnitudeValues(n);
    const auto scale = ms::FixedPointScale::ForRange(0.0, 1.0, n);

    // Several orders: as generated, reversed, sorted ascending, and three shuffles
    std::vector<std::vector<double>> orders;
    orders.push_back(values);
    orders.emplace_back(values.rbegin(), values.rend());
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    orders.push_back(sorted);
    for (std::uint64_t key = 10; key < 13; ++key) {
        orders.push_back(CounterShuffle(values, key));
    }

    // Double sums: at least two orders must disagree, otherwise the test data does not exercise
    // the problem and the fixed-point check below would prove nothing
    const double double_reference = DoubleSum(orders[0]);
    bool double_sums_differ = false;
    for (const auto& ordered : orders) {
        if (DoubleSum(ordered) != double_reference) {
            double_sums_differ = true;
        }
    }
    EXPECT_TRUE(double_sums_differ)
        << "test data too benign: double sums agree in every order";

    // Fixed-point sums: bitwise identical decoded doubles in every order
    const double fixed_reference = FixedPointSum(orders[0], scale);
    for (std::size_t k = 0; k < orders.size(); ++k) {
        EXPECT_EQ(FixedPointSum(orders[k], scale), fixed_reference) << "order " << k;
    }

    // Accuracy against the compensated reference sum
    const double exact = ExactSum(values);
    const double bound = n * 0.5 * scale.Quantum() +
                         4.0 * std::numeric_limits<double>::epsilon() * std::abs(fixed_reference);
    EXPECT_LE(std::abs(fixed_reference - exact), bound);
}

// With a nonzero origin (how coordinates are encoded, to keep the integers small) every term
// carries the offset, so a sum of N terms decodes with N * origin and the mean with origin.
// Values are 10 + v with |v| <= 1, so the expected sum is 10 N + sum(v) and the mean
// 10 + sum(v) / N; the tolerances are N * quantum / 2 plus double round-off of the ~1e6 result.
TEST(FixedPoint, NonzeroOriginSumAndMean) {
    const int n = 100000;
    const std::vector<double> values = MixedMagnitudeValues(n);
    const auto shifted = ms::FixedPointScale::ForRange(10.0, 1.0, n);

    std::int64_t sum = 0;
    for (double v : values) {
        sum += shifted.Encode(10.0 + v);
    }
    const double exact = ExactSum(values);
    EXPECT_NEAR(shifted.DecodeSum(sum, n), 10.0 * n + exact, 1e-6);
    EXPECT_NEAR(shifted.DecodeMean(sum, n), 10.0 + exact / n, 1e-9);
}

// The MPI usage pattern: each rank encodes the terms it owns, the int64 partial sums are combined
// with MPI_Allreduce(MPI_INT64_T, MPI_SUM), and the total is decoded with the global term count.
// The decoded double must be bitwise equal to the serial fixed-point sum at every rank count, and
// within N * quantum / 2 of the exact sum. This catches a wrong MPI datatype/operation, terms
// owned twice or not at all (the global count check), and decoding with a local instead of a
// global count. Ownership is deliberately uneven and interleaved, like a real mesh partition.
TEST(FixedPoint, MpiSumIsRankCountIndependent) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 50001;
    const auto scale = ms::FixedPointScale::ForRange(0.0, 6.0, n);

    // Every rank generates all terms (the counter RNG makes that cheap and identical), keeps the
    // serial reference, and adds only its own terms to the local partial sums
    std::vector<double> all_values(n);
    std::int64_t local_sum = 0;
    std::int64_t local_count = 0;
    for (int i = 0; i < n; ++i) {
        const ms::CounterRng rng(5u, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        // Clamp to the range the scale was built for (|x| <= 6, i.e. 6 sigma)
        all_values[i] = std::clamp(ms::Normal(rng, 0.0, 1.0, 0u), -6.0, 6.0);
        const bool owned_here = ((i * 7 + i / 13) % size) == rank;
        if (owned_here) {
            local_sum += scale.Encode(all_values[i]);
            local_count += 1;
        }
    }

    // Collective: combine integer partial sums and term counts across all ranks
    std::int64_t global_sum = 0;
    std::int64_t global_count = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(global_count, n);

    // Bitwise equal to the serial fixed-point sum, and accurate
    const double decoded = scale.DecodeSum(global_sum, global_count);
    EXPECT_EQ(decoded, FixedPointSum(all_values, scale));
    const double exact = ExactSum(all_values);
    EXPECT_LE(std::abs(decoded - exact),
              n * 0.5 * scale.Quantum() + 1.0e-12);
}
