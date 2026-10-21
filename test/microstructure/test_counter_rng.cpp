/*
 * Tests of util/counter_rng.hpp, the counter-based random number generator (RNG). "Counter-based"
 * means each random number is computed directly from (rng_seed, purpose, id, sample) by a fixed
 * scrambling function (Philox4x32-10), instead of being the next output of a generator with
 * internal state. Seed positions and orientations therefore do not depend on call order, thread
 * schedule or MPI rank count. These tests are serial (they run identically on every rank).
 */
#include "microstructure/util/counter_rng.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

/** @brief Builds a Philox counter/output block from four 32-bit words. */
ms::PhiloxBlock MakeBlock(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) {
    ms::PhiloxBlock blk;
    blk.v[0] = a;
    blk.v[1] = b;
    blk.v[2] = c;
    blk.v[3] = d;
    return blk;
}

/** @brief Expects two Philox blocks to be equal word by word. */
void ExpectBlock(const ms::PhiloxBlock& got, const ms::PhiloxBlock& want) {
    for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(got.v[k], want.v[k]) << "word " << k;
    }
}

} // namespace

// The Philox implementation is bit-compatible with Random123, the reference implementation by the
// algorithm's authors. The three counter/key/output triples are copied from Random123's
// known-answer file (examples/kat_vectors, "philox4x32 10"); any change to the round constants,
// the round count or the word shuffling breaks them.
TEST(CounterRng, PhiloxKnownAnswer) {
    ExpectBlock(ms::Philox4x32_10(MakeBlock(0, 0, 0, 0), 0, 0),
                MakeBlock(0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u));
    ExpectBlock(ms::Philox4x32_10(MakeBlock(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
                                  0xffffffffu,
                                  0xffffffffu),
                MakeBlock(0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu));
    ExpectBlock(ms::Philox4x32_10(MakeBlock(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u),
                                  0xa4093822u,
                                  0x299f31d0u),
                MakeBlock(0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u));
}

// UniformDouble lies in [0, 1) and has the moments of a uniform distribution: mean 1/2 and
// variance 1/12. Tolerances are ~6 standard errors for n = 200000, so a correct generator fails
// with negligible probability (and the streams are fixed, so the outcome is in fact
// deterministic).
TEST(CounterRng, UniformDoubleMomentsAndRange) {
    const ms::CounterRng rng(12345u, ms::RngPurpose::GENERIC, 7u);
    const int n = 200000;

    // UniformDouble: range, mean and variance
    double sum = 0.0;
    double sum_sq = 0.0;
    double lo = 1.0;
    double hi = 0.0;
    for (int i = 0; i < n; ++i) {
        const double u = rng.UniformDouble(static_cast<std::uint32_t>(i));
        sum += u;
        sum_sq += u * u;
        lo = std::min(lo, u);
        hi = std::max(hi, u);
    }
    const double mean = sum / n;
    const double var = sum_sq / n - mean * mean;
    EXPECT_GE(lo, 0.0);
    EXPECT_LT(hi, 1.0);
    // Standard error of the mean ~ 0.29 / sqrt(n) ~ 6.5e-4; allow ~6 sigma
    EXPECT_NEAR(mean, 0.5, 4e-3);
    EXPECT_NEAR(var, 1.0 / 12.0, 2e-3);
}

// The defining property of a counter-based RNG: the value for (seed, purpose, id, sample) is the
// same no matter in which order ids are visited or what other draws happen in between (which is
// what makes results independent of MPI rank count). Also checks that every component of the key
// (seed, including its high 32 bits; purpose; id, including its high 32 bits; sample index)
// actually changes the stream, so no input is silently ignored.
TEST(CounterRng, StreamsAreIndependentOfCallOrder) {
    const std::uint64_t seed = 0x0123456789abcdefull;
    const int n = 1000;

    // Reference: ids visited in increasing order
    std::vector<double> forward(n);
    for (int i = 0; i < n; ++i) {
        const auto id = static_cast<std::uint64_t>(i);
        const ms::CounterRng rng(seed, ms::RngPurpose::SEED_POSITION, id);
        forward[i] = rng.UniformDouble(3u);
    }

    // Same keys visited in a shuffled order, with unrelated draws interleaved
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(42u));
    for (int i : order) {
        const auto id = static_cast<std::uint64_t>(i);
        const ms::CounterRng other(seed, ms::RngPurpose::ORIENTATION, id);
        (void)other.UniformDouble(3u);
        const ms::CounterRng rng(seed, ms::RngPurpose::SEED_POSITION, id);
        EXPECT_EQ(rng.UniformDouble(3u), forward[i]);
    }

    // Seed, purpose, high id bits and sample index all change the stream
    const ms::CounterRng base(seed, ms::RngPurpose::SEED_POSITION, 5u);
    EXPECT_NE(base.UniformDouble(0u),
              ms::CounterRng(seed + 1, ms::RngPurpose::SEED_POSITION, 5u).UniformDouble(0u));
    EXPECT_NE(
        base.UniformDouble(0u),
        ms::CounterRng(seed ^ (1ull << 40), ms::RngPurpose::SEED_POSITION, 5u).UniformDouble(0u));
    EXPECT_NE(base.UniformDouble(0u),
              ms::CounterRng(seed, ms::RngPurpose::LATTICE_JITTER, 5u).UniformDouble(0u));
    // Element-weighted seeding uses its own purpose with a full 64-bit centroid hash as id, so
    // its streams differ from the rejection candidates' (SEED_POSITION, k) streams for the same
    // id, including ids with the top bit set
    const std::uint64_t top_bit_id = 5u | (1ull << 63);
    EXPECT_NE(
        base.UniformDouble(0u),
        ms::CounterRng(seed, ms::RngPurpose::ELEMENT_WEIGHTED_POSITION, 5u).UniformDouble(0u));
    EXPECT_NE(
        ms::CounterRng(seed, ms::RngPurpose::SEED_POSITION, top_bit_id).UniformDouble(0u),
        ms::CounterRng(seed, ms::RngPurpose::ELEMENT_WEIGHTED_POSITION, top_bit_id)
            .UniformDouble(0u));
    EXPECT_NE(
        base.UniformDouble(0u),
        ms::CounterRng(seed, ms::RngPurpose::SEED_POSITION, 5u + (1ull << 32)).UniformDouble(0u));
    EXPECT_NE(base.UniformDouble(0u), base.UniformDouble(1u));
}

// UniformDouble and the raw Philox blocks compile and run inside an mfem::forall kernel
// (MFEM_HOST_DEVICE, no host-only calls) and give bitwise the same values as host calls. Note:
// the local MFEM build has no GPU backend, so this runs on the CPU; it checks the
// kernel-compatible structure. The RNG uses only integer arithmetic, so device results are
// bitwise identical by construction.
TEST(CounterRng, KernelMatchesHost) {
    const int n = 257;
    mfem::Vector out(5 * n);
    const std::uint64_t seed = 77u;

    // Kernel: per id, one uniform double and the four raw words of block (sample 2, sub-block 1)
    // stored as doubles (exact, they are < 2^32). Layout out[5*i + k].
    double* d_out = out.Write();
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int i) {
        const auto id = static_cast<std::uint64_t>(i);
        const ms::CounterRng rng(seed, ms::RngPurpose::SEED_POSITION, id);
        d_out[5 * i] = rng.UniformDouble(0u);
        const ms::PhiloxBlock b = rng.Block(2u, 1u);
        for (int k = 0; k < 4; ++k) {
            d_out[5 * i + 1 + k] = static_cast<double>(b.v[k]);
        }
    });

    // Host: the same ids, visited backwards to also exercise order independence
    const double* h_out = out.HostRead();
    for (int i = n - 1; i >= 0; --i) {
        const auto id = static_cast<std::uint64_t>(i);
        const ms::CounterRng rng(seed, ms::RngPurpose::SEED_POSITION, id);
        EXPECT_EQ(h_out[5 * i], rng.UniformDouble(0u));
        const ms::PhiloxBlock b = rng.Block(2u, 1u);
        for (int k = 0; k < 4; ++k) {
            EXPECT_EQ(h_out[5 * i + 1 + k], static_cast<double>(b.v[k]));
        }
    }
}
