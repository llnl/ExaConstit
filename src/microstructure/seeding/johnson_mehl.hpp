#pragma once

#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file johnson_mehl.hpp
 * @brief Johnson-Mehl seeds: grains that nucleate at random places *and* random times, then grow
 * at a constant speed until they impinge on each other.
 *
 * **Johnson-Mehl nucleation and growth in plain words.** Think of a solid-state transformation
 * or a solidifying melt. New grains (nuclei) appear at random positions in the untransformed
 * material, at a rate of I nuclei per unit volume and unit time, and every grain grows outward
 * as a sphere at the same speed v. A grain stops growing where it meets another one. Two rules
 * follow:
 * - A nucleus can only appear in material that is still untransformed. A candidate nucleus born
 *   at position x and time t is discarded if an earlier grain j (born at s_j, t_j) has already
 *   reached x: |x - s_j| < v (t - t_j). (Such discarded candidates are sometimes called
 *   "phantom" nuclei; they never grow.)
 * - A point x ends up in the grain that reaches it first. Grain i arrives at time
 *   t_i + |x - s_i| / v, so x belongs to the grain with the smallest
 *
 *       |x - s_i| - a_i,   with a_i = -v t_i,
 *
 *   which is exactly the Additive metric of geometry/metric_functors.hpp with weight a_i (a
 *   grain that started earlier has a larger head start a_i). Boundaries between two grains are
 *   sheets of hyperboloids that bulge around the later grain; grains born at the same time meet
 *   on flat bisector planes, as in a Voronoi tessellation.
 *
 * This file produces the seeds of that process (position s_i, time t_i, weight a_i); labeling
 * the mesh with MetricType::ADDITIVE then gives the Johnson-Mehl microstructure. With a high rate
 * and short time window almost all nuclei are born together and the result approaches a Voronoi
 * tessellation; with a low rate the sizes spread out, because early grains grow large before
 * late ones appear.
 *
 * **How the random process is generated (space-time Poisson process).** Candidate nuclei are a
 * Poisson process in the sampling box times the time axis: candidate k (k = 0, 1, 2, ...) is
 * born at
 *
 *     t_k = t_{k-1} + E_k / (I_max V_box),   t_{-1} = 0,
 *     c_k = lo + U_k (hi - lo)   (component-wise),
 *
 * where V_box is the volume of the sampling box [lo, hi], E_k = -ln(1 - u) is an exponential
 * random number and U_k a uniform point of the unit cube. Waiting times between events of a
 * Poisson process are exponential, so the candidates are exactly a homogeneous space-time Poisson
 * process of rate I_max, listed in time order. Then three filters are applied in order k:
 *  1. **In the material:** c_k must lie inside an element (rejection sampling with the same exact
 *     point-in-element test as seed_generators.hpp, MaterialPointLocator), which turns the box
 *     process into a process of rate I_max in the material.
 *  2. **Time-dependent rate (optional):** if a rate function I(t) <= I_max is given, the
 *     candidate is kept with probability I(t_k) / I_max ("thinning", which turns a rate-I_max
 *     Poisson process into a rate-I(t) one).
 *  3. **Untransformed:** the candidate is discarded if an already accepted nucleus covers it,
 *     |c_k - s_j| < v (t_k - t_j). Accepted nuclei are kept in a uniform grid of cells (a
 *     "hash grid": space cut into cubes, every nucleus filed under its cube), so only nuclei in
 *     cubes within reach v t_k of c_k are checked.
 *
 * All random numbers come from the counter-based RNG (random numbers computed from (rng_seed,
 * purpose, id), so they don't depend on call order or rank count) with key (rng_seed,
 * RngPurpose::JOHNSON_MEHL, k): sample 0 gives U_k, sample 1 gives E_k, sample 2 the thinning
 * number.
 *
 * **Parallel, identical answer everywhere.** Every rank generates the same candidates. The
 * in-material test of a batch of candidates is split over the mesh partition: each rank marks the
 * candidates inside its own elements in a bitmask (one bit per candidate), and one
 * MPI_Allreduce with bitwise OR gives every rank the global mask. Filters 2 and 3 then run on
 * every rank on identical data, in candidate order, so all ranks accept the same nuclei, and the
 * result does not depend on the rank count or the partition.
 *
 * **When generation stops.** At the first of:
 * - the next candidate is born after end_time (if end_time > 0);
 * - num_nuclei nuclei have been accepted (if num_nuclei > 0);
 * - the material is fully transformed: every point of the material has been reached by some
 *   grain, so every later candidate would be discarded anyway. This is detected with a
 *   guaranteed upper bound of the transformation-completion time (see GenerateJohnsonMehlSeeds),
 *   and it never changes which nuclei are accepted; it only avoids generating candidates that
 *   would all be rejected. With end_time = 0 and num_nuclei = 0 the process simply runs until the
 *   material is fully transformed, which is the classical Johnson-Mehl microstructure.
 * - the stall cap max_candidates is reached (an error unless allow_stall is set).
 *
 * **Collective** (used below): a function that every rank of the communicator must call, in the
 * same order, because it exchanges data between ranks (here MPI_Allreduce).
 *
 * Host only: the seeding loop is sequential by nature, like RSA in seed_generators.hpp.
 */

namespace exaconstit::microstructure {

/**
 * @brief Parameters of Johnson-Mehl nucleation and growth. Units: length = mesh length units,
 * time = any consistent unit (only products v t and I t matter).
 */
struct JohnsonMehlParameters {
    /**
     * @brief Nucleation rate I in nuclei per unit volume per unit time (> 0). With a rate
     * function this is the upper bound I_max used to generate candidates.
     */
    double nucleation_rate = 0.0;
    /**
     * @brief Optional time-dependent rate I(t) (nuclei / (volume time)); empty = constant rate.
     * Must satisfy 0 <= I(t) <= nucleation_rate for every t at which it is called, and must
     * return the same value on every rank for the same t (a deterministic host function).
     * Called once per in-material candidate, in candidate order, on every rank.
     */
    std::function<double(double)> rate_function;
    /** @brief Growth speed v (> 0), length per unit time; grains grow as spheres of radius v t. */
    double growth_speed = 1.0;
    /** @brief Stop at this time (> 0), or 0 = no time limit (see "When generation stops"). */
    double end_time = 0.0;
    /**
     * @brief Stop after this many accepted nuclei (> 0), or 0 = no count limit. This is an upper
     * limit: if the material is fully transformed (or end_time passes) first, fewer nuclei are
     * returned and JohnsonMehlReport::stop says why. To aim for about N grains, choose the rate
     * with JohnsonMehlGrainDensity.
     */
    std::int64_t num_nuclei = 0;
    /** @brief Key of the counter-based RNG. */
    std::uint64_t rng_seed = 0;
    /**
     * @brief Draw candidates from [sampling_box_lo, sampling_box_hi] (intersected with the
     * material by the inside test) instead of the rounded-outward mesh bounding box. The
     * nucleation rate is per unit volume, so a smaller box only restricts where nuclei appear.
     */
    bool use_sampling_box = false;
    std::array<double, 3> sampling_box_lo = {0.0, 0.0, 0.0}; ///< user box lower corner
    std::array<double, 3> sampling_box_hi = {0.0, 0.0, 0.0}; ///< user box upper corner
    /**
     * @brief Stall cap: largest number of candidates generated (in the box, before any filter).
     * 0 selects 2^26. Reaching it throws, unless allow_stall is true.
     */
    std::int64_t max_candidates = 0;
    /** @brief If true, reaching max_candidates returns the nuclei accepted so far (noted in the
     * report) instead of throwing. */
    bool allow_stall = false;
    /**
     * @brief Testing aid: record every candidate discarded by the "untransformed" filter (its
     * position, time and the nucleus that covered it) in JohnsonMehlReport. Memory grows with
     * the number of discarded candidates; leave false in production.
     */
    bool record_rejected = false;
};

/**
 * @brief Checks a parameter set; called by GenerateJohnsonMehlSeeds. Host only, not collective.
 * @throws std::invalid_argument naming the first invalid field (non-positive or non-finite rate
 *         or speed, negative end_time / num_nuclei / max_candidates, empty user box).
 */
void ValidateJohnsonMehlParameters(const JohnsonMehlParameters& params);

/**
 * @brief Why GenerateJohnsonMehlSeeds stopped.
 */
enum class JohnsonMehlStop {
    END_TIME,          ///< the next candidate was born after end_time
    NUM_NUCLEI,        ///< num_nuclei nuclei were accepted
    FULLY_TRANSFORMED, ///< every material point was reached by a grain
    STALLED            ///< max_candidates was reached (allow_stall)
};

/**
 * @brief What GenerateJohnsonMehlSeeds did. Identical on every rank; counts are global.
 */
struct JohnsonMehlReport {
    std::int64_t candidates_generated = 0; ///< candidates drawn in the sampling box
    std::int64_t candidates_in_material = 0; ///< of those, inside the material
    std::int64_t rejected_by_rate = 0;     ///< in material, removed by the rate-function thinning
    std::int64_t rejected_covered = 0;     ///< in material, already covered by a grain
    std::int64_t accepted = 0;             ///< nuclei returned
    JohnsonMehlStop stop = JohnsonMehlStop::END_TIME; ///< stopping reason
    /** @brief Time of the last candidate examined (time units). */
    double last_candidate_time = 0.0;
    /**
     * @brief Smallest proven upper bound of the time at which the material is fully transformed
     * (+infinity if never computed). Every material point is reached by some grain no later than
     * this.
     */
    double completion_time_bound = 0.0;
    std::array<double, 3> sampling_lo = {0, 0, 0}; ///< box the candidates were drawn from
    std::array<double, 3> sampling_hi = {0, 0, 0};
    /**
     * @brief Only with record_rejected: the discarded ("covered") candidates, 3 coordinates each
     * in rejected_xyz[3 r + d], their birth times, and the 0-based index (into the returned
     * SeedSet) of the first accepted nucleus found to cover each of them.
     */
    std::vector<double> rejected_xyz;
    std::vector<double> rejected_time;
    std::vector<std::int64_t> rejected_by;

    /** @brief Multi-line summary for the rank-0 log. Host only. */
    std::string Summary() const;
};

/**
 * @brief Generates Johnson-Mehl nuclei in the material of the distributed mesh.
 *
 * **Collective** over `comm`: every rank passes its own partition (an empty partition is allowed)
 * and identical parameters, and every rank returns the identical SeedSet. Host only.
 *
 * Algorithm (details in the file comment):
 *  1. Element boxes and the exact inside test of the local elements (MaterialPointLocator).
 *  2. Global mesh box by MPI_Allreduce (MIN/MAX), rounded outward (RoundBoxOutward) unless a
 *     user sampling box is given.
 *  3. Candidates in batches of growing size: positions and times on every rank, in-material
 *     mask by bitwise-OR reduction, then the rate and coverage filters in candidate order.
 *  4. After any batch in which most in-material candidates were already covered, a proven upper
 *     bound T_done of the transformation-completion time is computed (collective, MPI_MAX):
 *     T_done = max over elements e of [T(c_e) + r_e / v], where T(x) = min_j (t_j + |x - s_j| / v)
 *     is the arrival time of the first grain at x, c_e is the center of the bounding box of
 *     element e and r_e (slightly enlarged) is half its diagonal. T changes by at most
 *     |x - y| / v between two points x and y (a minimum of such functions does too), and every
 *     point of element e lies within r_e of c_e, so every material point is reached by T_done.
 *     A later candidate born after T_done is therefore covered, and generation stops. Adding
 *     nuclei can only lower T, so the bound stays valid; it is the same on every rank because the
 *     maximum is exact and every term depends only on replicated data and element geometry.
 *
 * @param local_mesh this rank's part of the 3D hex or tet mesh (a serial mfem::Mesh only with a
 *        single-rank communicator, or with a different piece on each rank). Not modified.
 * @param comm   communicator over which the mesh is distributed.
 * @param params parameters (validated here).
 * @param report optional; filled on every rank.
 * @return the accepted nuclei in birth order: x, y, z (mesh length units), nucleation_time t_i,
 *         weight a_i = -v t_i (Additive metric, length units), and source_id = candidate index k
 *         (strictly increasing, with gaps). The other SeedSet arrays are empty.
 * @throws std::invalid_argument for invalid parameters, or if the rate function returns a value
 *         outside [0, nucleation_rate] (identical on every rank); std::runtime_error for a mesh
 *         that is empty, not 3D or has a zero-volume sampling box, for a local setup failure on
 *         any rank (broadcast so all ranks throw together), and for a stall unless allow_stall.
 */
SeedSet GenerateJohnsonMehlSeeds(const mfem::Mesh& local_mesh, MPI_Comm comm,
                                 const JohnsonMehlParameters& params,
                                 JohnsonMehlReport* report = nullptr);

/** @brief GenerateJohnsonMehlSeeds over the ParMesh's own communicator. Collective. */
SeedSet GenerateJohnsonMehlSeeds(const mfem::ParMesh& pmesh, const JohnsonMehlParameters& params,
                                 JohnsonMehlReport* report = nullptr);

/**
 * @brief Classical prediction of the number of grains per unit volume of a fully transformed
 * Johnson-Mehl microstructure with constant rate I and speed v in infinite space:
 * N_V = Gamma(5/4) (3 / pi)^(1/4) (I / v)^(3/4) ~= 0.8960 (I / v)^(3/4).
 *
 * Derivation: the untransformed fraction at time t is 1 - X(t) = exp(-(pi/3) I v^3 t^4)
 * (Johnson-Mehl-Avrami-Kolmogorov), nuclei are accepted at the rate I (1 - X(t)) per unit
 * volume, and the integral of that over all t is the expression above. Useful as a sanity check
 * and to choose I for a target grain count. Finite parts have somewhat more grains, because
 * grains near the surface can only grow inward. Host only.
 *
 * @param nucleation_rate I (> 0), nuclei / (volume time).
 * @param growth_speed    v (> 0), length / time.
 * @return expected grains per unit volume (1 / length^3).
 */
double JohnsonMehlGrainDensity(double nucleation_rate, double growth_speed);

} // namespace exaconstit::microstructure
