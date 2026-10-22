#include "microstructure/regions/region_assigner.hpp"

#include "microstructure/util/counter_rng.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/*
 * JoinInts: "a, b, c" for an error message.
 */
template <typename Container>
std::string JoinInts(const Container& values) {
    std::ostringstream out;
    bool first = true;
    for (const auto v : values) {
        out << (first ? "" : ", ") << v;
        first = false;
    }
    return out.str();
}

} // namespace

/*
 * AssignRegionsFromPhases
 *
 * Steps:
 *  1. Check the phase ids and collect the distinct phases present in an ordered set (std::set
 *     keeps its entries sorted, so iterating it gives the phases in ascending order).
 *  2. Region of each present phase: its 1-based position in that ascending list, or, with an
 *     explicit map, the map entry. The explicit map is checked so that the region ids line up
 *     with the [[Materials]] blocks: every entry well formed (key >= 0, region >= 1), every
 *     present phase listed, and the set of region ids used by the present phases exactly 1..M.
 *     Several phases may share a region (many-to-one).
 *  3. Per-region tables: the phases of every region (ascending, because the phase -> region map
 *     is visited in ascending phase order) and the crystalline flag; a region mixing
 *     crystalline and non-crystalline phases is rejected (one region = one material model).
 *  4. Per-grain regions.
 *
 * Every check depends only on the replicated inputs, so on a parallel run all ranks throw the
 * same error together.
 */
RegionAssignment AssignRegionsFromPhases(const std::vector<int>& grain_phase,
                                         const PhaseRegionParameters& params) {
    // Step 1: validate the phase ids and list the phases present, ascending
    if (grain_phase.empty()) {
        throw std::invalid_argument("AssignRegionsFromPhases: there are no grains");
    }
    std::set<int> present;
    std::set<int> negative;
    for (const int phase : grain_phase) {
        if (phase < 0) {
            negative.insert(phase);
        } else {
            present.insert(phase);
        }
    }
    if (!negative.empty()) {
        throw std::invalid_argument("AssignRegionsFromPhases: phase ids must be >= 0; found " +
                                    JoinInts(negative));
    }
    std::set<int> non_crystalline;
    for (const int phase : params.non_crystalline_phases) {
        if (phase < 0) {
            throw std::invalid_argument(
                "AssignRegionsFromPhases: non-crystalline phase ids must be >= 0; found " +
                std::to_string(phase));
        }
        non_crystalline.insert(phase);
    }

    // Step 2: region id of every present phase, and the number of regions M
    std::map<int, int> region_of_phase;
    int num_regions = 0;
    if (params.phase_to_region.empty()) {
        // Compact ascending numbering: the k-th smallest present phase is region k
        int next_region = 1;
        for (const int phase : present) {
            region_of_phase[phase] = next_region++;
        }
        num_regions = static_cast<int>(present.size());
    } else {
        // Every entry must be well formed, also those of phases no grain has
        for (const auto& entry : params.phase_to_region) {
            if (entry.first < 0 || entry.second < 1) {
                throw std::invalid_argument(
                    "AssignRegionsFromPhases: phase_to_region entry " +
                    std::to_string(entry.first) + " -> " + std::to_string(entry.second) +
                    " is invalid (phase ids must be >= 0 and region ids >= 1)");
            }
        }
        // Every present phase must be listed
        std::vector<int> missing;
        for (const int phase : present) {
            const auto it = params.phase_to_region.find(phase);
            if (it == params.phase_to_region.end()) {
                missing.push_back(phase);
            } else {
                region_of_phase[phase] = it->second;
            }
        }
        if (!missing.empty()) {
            throw std::invalid_argument(
                "AssignRegionsFromPhases: phase_to_region has no entry for phase(s) " +
                JoinInts(missing) + " (every phase present in the grains must be listed)");
        }
        // The region ids used must be exactly 1..M: with all ids >= 1 (checked above) and
        // M = number of distinct ids, that holds exactly when the largest id equals M
        std::set<int> used;
        for (const auto& entry : region_of_phase) {
            used.insert(entry.second);
        }
        num_regions = static_cast<int>(used.size());
        const int largest = *used.rbegin();
        if (largest != num_regions) {
            std::vector<int> unused;
            for (int r = 1; r <= largest; r++) {
                if (used.count(r) == 0) {
                    unused.push_back(r);
                }
            }
            throw std::invalid_argument(
                "AssignRegionsFromPhases: the region ids used by the phases present (" +
                JoinInts(used) + ") must be exactly 1.." + std::to_string(num_regions) +
                " (one region per [[Materials]] block); region id(s) " + JoinInts(unused) +
                " are not used");
        }
    }

    // Step 3: phases of every region (ascending) and the crystalline flag
    RegionAssignment result;
    result.num_regions = num_regions;
    result.region_phases.assign(static_cast<std::size_t>(num_regions), std::vector<int>());
    for (const auto& entry : region_of_phase) {
        result.region_phases[static_cast<std::size_t>(entry.second - 1)].push_back(entry.first);
    }
    result.region_crystalline.assign(static_cast<std::size_t>(num_regions), 1);
    for (int r = 1; r <= num_regions; r++) {
        const std::vector<int>& phases = result.region_phases[static_cast<std::size_t>(r - 1)];
        int num_non_crystalline = 0;
        for (const int phase : phases) {
            num_non_crystalline += non_crystalline.count(phase) ? 1 : 0;
        }
        if (num_non_crystalline > 0 && num_non_crystalline < static_cast<int>(phases.size())) {
            throw std::invalid_argument(
                "AssignRegionsFromPhases: region " + std::to_string(r) + " holds phases " +
                JoinInts(phases) +
                ", which mix crystalline and non-crystalline phases; one region is one material "
                "model, so map them to different regions");
        }
        result.region_crystalline[static_cast<std::size_t>(r - 1)] =
            (num_non_crystalline > 0) ? 0 : 1;
    }

    // Step 4: per-grain regions
    result.grain_region.resize(grain_phase.size());
    for (std::size_t g = 0; g < grain_phase.size(); g++) {
        result.grain_region[g] = region_of_phase.at(grain_phase[g]);
    }
    return result;
}

/*
 * AssignPhaseFractions
 *
 * Steps (see the header for the algorithm and the proof of the accuracy bound):
 *  1. Validate the inputs. Distinct source ids are required because they define the visit
 *     order; two grains with the same id would make the order depend on the input order.
 *     Then normalize the targets, f_k <- f_k / (sum of f), so the deficits sum to exactly the
 *     unassigned volume (up to round-off) as the accuracy proof in the header assumes.
 *  2. Visit order: sort the grain indices by (random key, source id). The key is the first 64
 *     bits of the counter-based RNG block for (rng_seed, PHASE, source id), sample 0.
 *  3. Total volume V, summed in visit order.
 *  4. Greedy pass: each grain goes to the phase with the largest deficit f_k V - assigned_k;
 *     the strict ">" comparison keeps the lower index on ties.
 *  5. Achieved fractions assigned_k / V.
 */
PhaseFractionResult AssignPhaseFractions(const std::vector<double>& grain_volume,
                                         const std::vector<std::int64_t>& grain_source_id,
                                         const PhaseFractionParameters& params) {
    // Step 1: validate grains ...
    const std::size_t num_grains = grain_volume.size();
    if (num_grains == 0) {
        throw std::invalid_argument("AssignPhaseFractions: there are no grains");
    }
    if (grain_source_id.size() != num_grains) {
        throw std::invalid_argument("AssignPhaseFractions: grain_volume has " +
                                    std::to_string(num_grains) + " entries but grain_source_id " +
                                    std::to_string(grain_source_id.size()));
    }
    for (std::size_t g = 0; g < num_grains; g++) {
        if (!(grain_volume[g] > 0.0) || !std::isfinite(grain_volume[g])) {
            throw std::invalid_argument("AssignPhaseFractions: grain " + std::to_string(g + 1) +
                                        " has a non-positive or non-finite volume");
        }
    }
    {
        std::vector<std::int64_t> sorted_ids(grain_source_id);
        std::sort(sorted_ids.begin(), sorted_ids.end());
        const auto dup = std::adjacent_find(sorted_ids.begin(), sorted_ids.end());
        if (dup != sorted_ids.end()) {
            throw std::invalid_argument("AssignPhaseFractions: source id " +
                                        std::to_string(*dup) + " is used by more than one grain");
        }
    }
    // ... and phases
    const std::size_t num_phases = params.phase_id.size();
    if (num_phases == 0) {
        throw std::invalid_argument("AssignPhaseFractions: no phases given");
    }
    if (params.fraction.size() != num_phases) {
        throw std::invalid_argument("AssignPhaseFractions: " + std::to_string(num_phases) +
                                    " phase ids but " + std::to_string(params.fraction.size()) +
                                    " fractions");
    }
    std::set<int> seen_phases;
    double fraction_sum = 0.0;
    for (std::size_t k = 0; k < num_phases; k++) {
        const int phase = params.phase_id[k];
        if (phase < 0 || !seen_phases.insert(phase).second) {
            throw std::invalid_argument("AssignPhaseFractions: phase id " +
                                        std::to_string(phase) + " is negative or repeated");
        }
        const double f = params.fraction[k];
        if (!(f > 0.0) || !std::isfinite(f)) {
            throw std::invalid_argument("AssignPhaseFractions: the fraction of phase " +
                                        std::to_string(phase) + " must be > 0 and finite");
        }
        fraction_sum += f;
    }
    if (std::fabs(fraction_sum - 1.0) > 1.0e-6) {
        std::ostringstream msg;
        msg.precision(17);
        msg << "AssignPhaseFractions: the phase fractions sum to " << fraction_sum
            << " instead of 1 (tolerance 1e-6)";
        throw std::invalid_argument(msg.str());
    }
    std::vector<double> target(num_phases);
    for (std::size_t k = 0; k < num_phases; k++) {
        target[k] = params.fraction[k] / fraction_sum;
    }

    // Step 2: random but input-order-independent visit order
    std::vector<std::uint64_t> key(num_grains);
    for (std::size_t g = 0; g < num_grains; g++) {
        const CounterRng rng(params.rng_seed, RngPurpose::PHASE,
                             static_cast<std::uint64_t>(grain_source_id[g]));
        const PhiloxBlock bits = rng.Block(0);
        key[g] = (static_cast<std::uint64_t>(bits.v[0]) << 32) | bits.v[1];
    }
    std::vector<std::size_t> order(num_grains);
    for (std::size_t g = 0; g < num_grains; g++) {
        order[g] = g;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (key[a] != key[b]) {
            return key[a] < key[b];
        }
        return grain_source_id[a] < grain_source_id[b];
    });

    // Step 3: total volume in visit order
    double total_volume = 0.0;
    for (const std::size_t g : order) {
        total_volume += grain_volume[g];
    }

    // Step 4: greedy assignment to the phase with the largest remaining deficit
    PhaseFractionResult result;
    result.grain_phase.assign(num_grains, -1);
    std::vector<double> assigned(num_phases, 0.0);
    for (const std::size_t g : order) {
        std::size_t best = 0;
        double best_deficit = target[0] * total_volume - assigned[0];
        for (std::size_t k = 1; k < num_phases; k++) {
            const double deficit = target[k] * total_volume - assigned[k];
            if (deficit > best_deficit) { // strict: ties keep the lower index
                best = k;
                best_deficit = deficit;
            }
        }
        result.grain_phase[g] = params.phase_id[best];
        assigned[best] += grain_volume[g];
    }

    // Step 5: achieved fractions
    result.achieved_fraction.resize(num_phases);
    for (std::size_t k = 0; k < num_phases; k++) {
        result.achieved_fraction[k] = assigned[k] / total_volume;
    }
    return result;
}

} // namespace exaconstit::microstructure
