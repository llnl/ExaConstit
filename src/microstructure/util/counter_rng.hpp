#pragma once

#include "mfem.hpp"

#include <cstdint>

/**
 * @file counter_rng.hpp
 * @brief Counter-based random number generator (Philox4x32-10): raw random bits and uniform
 * doubles in [0, 1), independent of rank count and call order.
 *
 * This header is only the generator. Distributions built on it (uniform on [a, b], normal,
 * log-normal) are in util/random_distributions.hpp; random orientations are in
 * orientation/random_orientation.hpp.
 *
 * **What a counter-based RNG is.** A usual random number generator (std::mt19937, rand) has an
 * internal state and returns "the next number"; which number a seed receives then depends on how
 * many numbers were drawn before it, i.e. on loop order, thread schedule and how the seeds are
 * split across MPI ranks. A counter-based RNG instead computes each random number directly from
 * its address: random numbers are computed from (rng_seed, purpose, id, sample), so they don't
 * depend on call order or rank count. Seed 17 gets the same position on 1 rank or 1000 ranks, on
 * the CPU or a GPU.
 *
 * **Philox4x32-10** (Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as easy as 1, 2, 3",
 * SC11) is the scrambling function used here. It maps a 128-bit counter (four 32-bit words) and a
 * 64-bit key to 128 random-looking bits by 10 rounds of integer multiply / xor / word swaps. It is
 * a *bijection* (a one-to-one map, so distinct counters never collide), passes the standard
 * statistical test batteries (TestU01 BigCrush), and is the same generator cuRAND/rocRAND offer.
 *
 * Everything here is header-only, `MFEM_HOST_DEVICE` (callable from host code and from inside
 * GPU kernels) and stateless, so it can be used inside `mfem::forall` loops. Only integer
 * arithmetic and one integer-to-double conversion are used, so results are bitwise identical on
 * every platform.
 *
 * **Bit layout of one Philox call** (the "address" of a block of 128 random bits):
 * - key   = rng_seed (low 32 bits, high 32 bits);
 * - ctr0  = id low 32 bits, ctr1 = id high 32 bits;
 * - ctr2  = (purpose << 16) | sub_block, where sub_block numbers the extra blocks a distribution
 *           needs when one block (128 bits) is not enough;
 * - ctr3  = sample index.
 *
 * Each distribution consumes the blocks `(sample, sub_block 0..k)` of its stream, so different
 * sample indices never overlap. Calling two different distributions with the same
 * `(purpose, id, sample)` reuses the same bits; give independent quantities distinct purposes or
 * sample indices.
 */

namespace exaconstit::microstructure {

/**
 * @brief What a random stream is used for. Part of the Philox counter, so streams of different
 * purposes are independent even for the same seed and id. Values are stable: append new purposes
 * at the end, never renumber (renumbering changes every generated microstructure). Values must
 * fit in 16 bits (they are shifted into the top half of ctr2).
 */
enum class RngPurpose : std::uint32_t {
    SEED_POSITION = 1,  ///< seed coordinates of rejection-sampling candidates (id = candidate k)
    ORIENTATION = 2,    ///< random grain orientations
    LATTICE_JITTER = 3, ///< perturbation of lattice seed positions
    JOHNSON_MEHL = 4,   ///< nucleation times
    SEED_WEIGHT = 5,    ///< Laguerre / additive weights
    SEED_SHAPE = 6,     ///< per-seed quadric shape tensors
    PHASE = 7,          ///< phase assignment
    /**
     * Element-volume-weighted seed sampling: arrival times and in-element positions of the
     * points an element emits (id = 64-bit hash of the element centroid, all 64 bits used).
     * A purpose of its own keeps these streams disjoint from the SEED_POSITION candidate streams
     * for every possible hash value.
     */
    ELEMENT_WEIGHTED_POSITION = 8,
    GENERIC = 0xFFFF ///< tests and ad-hoc use
};

/**
 * @brief One Philox4x32 block: four 32-bit words (128 bits), used both as counter input and as
 * random output. Plain data, so it can live in GPU registers.
 */
struct PhiloxBlock {
    std::uint32_t v[4]; ///< the four words, v[0] first
};

namespace detail {

// Philox4x32 round multipliers and key increments (Salmon et al. 2011; identical to
// Random123's PHILOX_M4x32_0/1 and PHILOX_W32_0/1). Changing any of them changes every stream.
constexpr std::uint32_t PHILOX_M0 = 0xD2511F53u;
constexpr std::uint32_t PHILOX_M1 = 0xCD9E8D57u;
constexpr std::uint32_t PHILOX_W0 = 0x9E3779B9u; // golden ratio
constexpr std::uint32_t PHILOX_W1 = 0xBB67AE85u; // sqrt(3) - 1
constexpr double INV_2_POW_53 = 1.0 / 9007199254740992.0; // 2^-53

} // namespace detail

/**
 * @brief Maps 64 random bits to a double in [0, 1) with 53 bits of resolution. Public so the
 * distributions can turn any two words of a block into a uniform.
 *
 * A double has a 53-bit significand, so the top 53 of the 64 bits are kept (shift right by 11)
 * and scaled by 2^-53. Every result k * 2^-53 (k = 0 .. 2^53 - 1) is exactly representable and
 * equally likely; 1.0 itself is never returned. Host and device.
 *
 * @param hi upper 32 random bits.
 * @param lo lower 32 random bits.
 */
MFEM_HOST_DEVICE inline double BitsToUnitDouble(std::uint32_t hi, std::uint32_t lo) {
    const std::uint64_t bits = (static_cast<std::uint64_t>(hi) << 32) | lo;
    const std::uint64_t top_53_bits = bits >> 11;
    return static_cast<double>(top_53_bits) * detail::INV_2_POW_53;
}

/**
 * @brief The raw Philox4x32-10 bijection, bit-compatible with Random123's
 * `philox4x32_R(10, ctr, key)` (checked against Random123's known-answer vectors in
 * test/microstructure/test_util.cpp).
 *
 * Each of the 10 rounds multiplies two of the words by fixed odd constants as 32x32 -> 64-bit
 * products, then mixes the high halves with the other two words and the round key by xor, and
 * reorders the words. The key is bumped by fixed constants between rounds ("key schedule").
 *
 * Host and device; no state, no allocation.
 *
 * @param ctr  128-bit counter (the address of the block).
 * @param key0 low 32 bits of the key.
 * @param key1 high 32 bits of the key.
 * @return 128 random bits.
 */
MFEM_HOST_DEVICE inline PhiloxBlock Philox4x32_10(PhiloxBlock ctr, std::uint32_t key0,
                                                  std::uint32_t key1) {
    for (int round = 0; round < 10; ++round) {
        // Key schedule: round 0 uses the user key, every later round adds the Weyl constants
        if (round > 0) {
            key0 += detail::PHILOX_W0;
            key1 += detail::PHILOX_W1;
        }

        // Two 32x32 -> 64-bit products, split into high and low 32-bit halves
        const std::uint64_t product0 = static_cast<std::uint64_t>(detail::PHILOX_M0) * ctr.v[0];
        const std::uint64_t product1 = static_cast<std::uint64_t>(detail::PHILOX_M1) * ctr.v[2];
        const std::uint32_t hi0 = static_cast<std::uint32_t>(product0 >> 32);
        const std::uint32_t lo0 = static_cast<std::uint32_t>(product0);
        const std::uint32_t hi1 = static_cast<std::uint32_t>(product1 >> 32);
        const std::uint32_t lo1 = static_cast<std::uint32_t>(product1);

        // Mix and permute the words (the Philox "S-box" step); exactly Random123's round
        ctr.v[0] = hi1 ^ ctr.v[1] ^ key0;
        ctr.v[1] = lo1;
        ctr.v[2] = hi0 ^ ctr.v[3] ^ key1;
        ctr.v[3] = lo0;
    }
    return ctr;
}

/**
 * @brief A random stream keyed by `(rng_seed, purpose, id)`; samples are addressed by index.
 *
 * Usage inside a kernel over seeds:
 * @code
 *   mfem::forall(num_seeds, [=] MFEM_HOST_DEVICE (int i) {
 *       const CounterRng rng(rng_seed, RngPurpose::SEED_POSITION, i);
 *       const double u = rng.UniformDouble(attempt); // attempt = 0, 1, 2, ...
 *       ...
 *   });
 * @endcode
 * Trivially copyable (two 64-bit words and an enum), so it can be captured by value in kernels.
 * Constructing one costs nothing; build it inside the kernel from the global id. All methods are
 * const, host and device, and never fail.
 */
class CounterRng {
public:
    /**
     * @brief Creates the stream for one entity.
     * @param rng_seed user seed (`seeds.rng_seed`); all 64 bits are used.
     * @param purpose  what the numbers are for (independent streams per purpose).
     * @param id       stable *global* id of the entity (seed index, candidate index, grain
     *                 id, ...), never a rank-local index, otherwise results would change with the
     *                 partition. All 64 bits are used.
     */
    MFEM_HOST_DEVICE CounterRng(std::uint64_t rng_seed, RngPurpose purpose, std::uint64_t id)
        : m_seed(rng_seed), m_id(id), m_purpose(purpose) {}

    /**
     * @brief Raw 128 random bits of block `(sample, sub_block)` of this stream.
     * @param sample    sample index (any 32-bit value).
     * @param sub_block extra block number for distributions that need more than 128 bits;
     *                  only the low 16 bits are used.
     */
    MFEM_HOST_DEVICE PhiloxBlock Block(std::uint32_t sample, std::uint32_t sub_block = 0) const {
        // Assemble the counter as documented in the file header
        PhiloxBlock ctr;
        ctr.v[0] = static_cast<std::uint32_t>(m_id);       // id, low word
        ctr.v[1] = static_cast<std::uint32_t>(m_id >> 32); // id, high word
        ctr.v[2] = (static_cast<std::uint32_t>(m_purpose) << 16) | (sub_block & 0xFFFFu);
        ctr.v[3] = sample;

        // Key = rng_seed split into two 32-bit words
        const auto key_lo = static_cast<std::uint32_t>(m_seed);
        const auto key_hi = static_cast<std::uint32_t>(m_seed >> 32);
        return Philox4x32_10(ctr, key_lo, key_hi);
    }

    /** @brief Uniform double in [0, 1) with 53 random bits (block (sample, 0), words 0-1). */
    MFEM_HOST_DEVICE double UniformDouble(std::uint32_t sample) const {
        const PhiloxBlock b = Block(sample);
        return BitsToUnitDouble(b.v[0], b.v[1]);
    }

private:
    std::uint64_t m_seed; ///< Philox key (user rng_seed)
    std::uint64_t m_id;   ///< global entity id, counter words 0-1
    RngPurpose m_purpose; ///< stream purpose, top half of counter word 2
};

} // namespace exaconstit::microstructure
