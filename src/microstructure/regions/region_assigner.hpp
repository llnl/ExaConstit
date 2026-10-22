#pragma once

#include <cstdint>
#include <map>
#include <vector>

/**
 * @file region_assigner.hpp
 * @brief Turns grain phases into ExaConstit material regions, and gives grains a phase when a
 *        synthetic multiphase microstructure with prescribed volume fractions is requested.
 *
 * **Regions in ExaConstit.** Every `[[Materials]]` block of an ExaConstit input is one *region*;
 * region ids are 1, 2, 3, ... in the order of the blocks. The region mapping file lists one
 * `grain region` line per grain, so every grain belongs to exactly one material model. A
 * *phase* (e.g. alpha and beta titanium, or a crystalline matrix and an amorphous inclusion) is
 * what the input data calls a material; this file maps phases to region ids.
 *
 * Two independent tools live here:
 * - AssignRegionsFromPhases: each grain already has a phase (from a seed table or a voxel map);
 *   every phase present becomes one region.
 * - AssignPhaseFractions: grains have no phase yet; each one is given a phase so that the phase
 *   volume fractions come as close as possible to the requested ones.
 *
 * Both are host-only, not collective (no MPI calls) and deterministic: their inputs are the
 * replicated per-grain tables, identical on every rank, so every rank computes the same answer.
 */

namespace exaconstit::microstructure {

/**
 * @brief How phases become regions (input of AssignRegionsFromPhases).
 */
struct PhaseRegionParameters {
    /**
     * @brief Phase ids (>= 0) whose grains are not crystals (amorphous material, pores filled
     * with a binder, a build plate described as a phase, ...). Their regions are flagged
     * non-crystalline, and the driver writes the identity orientation for their grains.
     * Phases listed here that no grain has are ignored. Order and duplicates do not matter.
     */
    std::vector<int> non_crystalline_phases;
    /**
     * @brief Optional explicit phase -> region map (key: phase id >= 0, value: region id >= 1).
     *
     * Empty (default): the regions are numbered 1..M in ascending order of the phases present
     * (the "compact ascending" numbering), e.g. phases {0, 3, 7} -> regions {1, 2, 3}.
     *
     * Non-empty: every entry must have a key >= 0 and a region id >= 1 (checked for every
     * entry, including phases no grain has). Every phase present must be a key. Several phases
     * may share one region (many-to-one, e.g. two crystallographic variants modeled by the same
     * material), but the set of region ids used by the phases present must be exactly 1..M
     * (M = number of distinct region ids used), so that the region ids line up with M
     * `[[Materials]]` blocks. Entries of phases that no grain has are otherwise ignored.
     *
     * One region is one material model, so a region may not mix phases listed in
     * non_crystalline_phases with crystalline phases.
     */
    std::map<int, int> phase_to_region;
};

/**
 * @brief Result of AssignRegionsFromPhases. Every array is replicated on every rank.
 */
struct RegionAssignment {
    /**
     * @brief Size N (number of grains): grain_region[g - 1] = region id (1..num_regions) of
     * grain g. This is the content of the ExaConstit region mapping file.
     */
    std::vector<int> grain_region;
    /**
     * @brief Number of regions M >= 1: the number of distinct phases present (compact
     * numbering), or the number of distinct region ids used by the phases present (explicit
     * map).
     */
    int num_regions = 1;
    /**
     * @brief Size M: region_phases[r - 1] = the phase ids (present in the grains) that region r
     * holds, ascending. One phase per region with the compact numbering; possibly several with a
     * many-to-one explicit map.
     */
    std::vector<std::vector<int>> region_phases;
    /**
     * @brief Size M: region_crystalline[r - 1] = 1 when region r is crystalline (it uses the
     * grain orientations), 0 when its phases are listed in non_crystalline_phases (a region never
     * mixes the two). Stored as char rather than bool so the vector is a plain array of bytes.
     */
    std::vector<char> region_crystalline;
};

/**
 * @brief Maps every grain to a region through its phase.
 *
 * Steps: collect the distinct phases present (ascending); number them 1..M, or look each one up
 * in params.phase_to_region; then give each grain the region of its phase, list the phases of
 * every region and flag the non-crystalline regions.
 *
 * Host only; not collective; deterministic.
 *
 * @param grain_phase size N >= 1: grain_phase[g - 1] = phase id (>= 0) of grain g.
 * @param params      non-crystalline phases and the optional explicit phase -> region map.
 * @return per-grain region ids, the phases of every region and its crystalline flag.
 * @throws std::invalid_argument when:
 *         - grain_phase is empty, a phase id is negative, or a listed non-crystalline phase is
 *           negative;
 *         - (explicit map) an entry has a negative key or a region id < 1; a phase present has
 *           no entry; or the region ids used by the phases present are not exactly 1..M (the
 *           message lists the unused ids);
 *         - a region mixes crystalline and non-crystalline phases (the message names the region
 *           and its phases).
 */
RegionAssignment AssignRegionsFromPhases(const std::vector<int>& grain_phase,
                                         const PhaseRegionParameters& params);

/**
 * @brief Requested phase volume fractions for a synthetic multiphase microstructure (input of
 * AssignPhaseFractions).
 */
struct PhaseFractionParameters {
    /** @brief Phase ids, size K >= 1, each >= 0, all distinct. */
    std::vector<int> phase_id;
    /**
     * @brief Target volume fraction f_k of phase_id[k], size K; every f_k > 0 and the sum is 1
     * within 1e-6. The fractions are then divided by their sum, so the targets used (and the
     * accuracy bound below) refer to fractions that sum to 1 up to round-off.
     */
    std::vector<double> fraction;
    /** @brief Key of the counter-based RNG that sets the order in which grains are visited. */
    std::uint64_t rng_seed = 0;
};

/**
 * @brief Result of AssignPhaseFractions. Every array is replicated on every rank.
 */
struct PhaseFractionResult {
    /** @brief Size N: grain_phase[g - 1] = phase id (one of params.phase_id) of grain g. */
    std::vector<int> grain_phase;
    /**
     * @brief Size K: achieved volume fraction of phase_id[k], i.e. (summed volume of its grains)
     * / (total grain volume V).
     */
    std::vector<double> achieved_fraction;
};

/**
 * @brief Gives every grain a phase so that the phase volume fractions approach the targets.
 *
 * **Algorithm (greedy by deficit).**
 *  1. Visit order: grain g gets the random 64-bit key computed by the counter-based RNG from
 *     (rng_seed, RngPurpose::PHASE, grain_source_id[g - 1]) (util/counter_rng.hpp: random
 *     numbers computed from (seed, purpose, id), so they do not depend on call order or rank
 *     count). Grains are visited in increasing key, ties going to the lower source id. The order
 *     is therefore random but fixed by the source ids alone: it does not depend on the order in
 *     which the grains are listed, nor on the rank count.
 *  2. The targets are normalized, f_k <- f_k / (sum of all f), so they sum to 1 (up to
 *     round-off) and the deficits below add up to exactly the unassigned volume.
 *  3. V = total grain volume, summed in visit order (so even its rounding does not depend on the
 *     input order).
 *  4. Each visited grain goes to the phase with the largest remaining deficit
 *     f_k V - assigned_k (assigned_k = volume already given to phase k), ties going to the
 *     lower index k.
 *
 * **Accuracy bound.** For every phase, |achieved_k - f_k| <= v_max / V, where f_k is the
 * normalized target and v_max is the largest grain volume. Why (exact arithmetic):
 *  - Over-filling: while grains remain, the deficits sum to V - (assigned volume) > 0, so the
 *    largest deficit is > 0. A phase therefore only receives a grain when its deficit is
 *    positive, and afterwards its deficit is >= -v >= -v_max. Deficits only decrease when a
 *    grain is received, so no deficit ever goes below -v_max: assigned_k <= f_k V + v_max.
 *  - Under-filling: suppose phase k ends with deficit d_k > v_max. Every other phase j that
 *    received a grain had, at its last grain (volume v), a deficit >= the deficit of k at that
 *    moment >= d_k (k's deficit never grows), so j ends with deficit >= d_k - v > 0; phases that
 *    received nothing keep deficit f_j V > 0. Then every final deficit is positive, contradicting
 *    that they sum to V - V = 0. Hence d_k <= v_max.
 *  Dividing by V gives the bound. Many small grains therefore give fractions close to the target.
 *
 * Host only; not collective; deterministic.
 *
 * @param grain_volume    size N >= 1: volume of grain g at index g - 1 (length^3, each > 0 and
 *                        finite). Must itself be rank-count independent (bitwise identical on
 *                        every rank count), e.g. the fixed-point grain volumes of
 *                        labeling/grain_statistics; a plain floating-point MPI sum of element
 *                        volumes depends on the summation order, and any difference can change
 *                        which phase a grain receives.
 * @param grain_source_id size N: stable source id of grain g at index g - 1 (e.g.
 *                        MicrostructureResult::grain_source_id), all distinct.
 * @param params          target phases, fractions and the RNG seed.
 * @return the phase of every grain and the achieved fractions.
 * @throws std::invalid_argument for empty or mismatched sizes, a non-positive or non-finite
 *         volume, repeated source ids, an empty phase list, a negative or repeated phase id, a
 *         non-positive or non-finite fraction, or fractions that do not sum to 1 within 1e-6.
 */
PhaseFractionResult AssignPhaseFractions(const std::vector<double>& grain_volume,
                                         const std::vector<std::int64_t>& grain_source_id,
                                         const PhaseFractionParameters& params);

} // namespace exaconstit::microstructure
