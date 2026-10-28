#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "microstructure/conformal/node_targets.hpp"

/**
 * @file adjacency_check.hpp
 * @brief Stage 2c, second part: the grain adjacency of the mesh compared with the analytic one.
 *
 * **What it tells.** Two grains are *adjacent in the mesh* when some node carries both labels
 * (they share a face, an edge or a node). Two seeds are *analytically adjacent* when their analytic
 * boundary passes close to the mesh: some node can be moved onto it within the move limit
 * (NodeTargetOptions::max_move element sizes, together with the boundary surfaces at the node)
 * with no other seed closer there (the rules of node_targets.hpp, without the smoothed fallback).
 * The comparison is made between seeds: the grains of a pair map to their seeds
 * (grain_seed), and pairs with a grain without seed (voxel, geodesic and passthrough grains) or
 * with two grains of the same seed (pieces of one cell) are not compared.
 *   - *fake* adjacency: grains that touch in the mesh although their analytic boundary does not
 *     come near any node that carries both (label repair left a staircase artefact, or the grains
 *     touch only at a node or an edge);
 *   - *missing* adjacency: seeds whose analytic boundary passes near the mesh although no node
 *     carries both grains (a face smaller than the mesh resolves, e.g. a short edge of the
 *     Voronoi diagram that the staircase cuts off).
 *
 * **How.** Discrete pairs: every pair of labels of every owned node. A discrete pair is confirmed
 * analytic at a node that carries it when the pair's own query at the node is FIT
 * (NodeTargets::EvaluateAnalytic). Analytic pairs elsewhere: at every owned node the best seed and
 * the next ones whose boundary with it is estimated within twice the move limit
 * (NodeTargets::NearbySeeds, up to ADJACENCY_NEARBY_SEEDS seeds), each pair (best, other) queried
 * at the node; a seed pair is queried with the first grain of each seed. So pairs are found from
 * any node near their boundary, not only from the nodes of the two grains.
 *
 * **Determinism.** Every pair goes to an owner rank (a hash of the pair, MPI_Alltoallv), which
 * merges the flags of all copies; the counts are summed and the examples are the smallest pairs
 * of the merged lists. Node queries depend only on replicated data, so every number is the same
 * for any partition.
 *
 * **Collective** over the mesh communicator of `targets`. Host.
 */

namespace exaconstit::microstructure {

/** @brief Seeds per node in the search for analytic pairs (the best and up to 3 others). */
inline constexpr int ADJACENCY_NEARBY_SEEDS = 4;
/** @brief Fake and missing pairs listed in AdjacencyComparison (the smallest ones). */
inline constexpr int ADJACENCY_EXAMPLES = 10;

/** @brief Global result of CompareAdjacency (the same on every rank). */
struct AdjacencyComparison {
    /** @brief Seed pairs adjacent in the mesh (grains with two distinct seeds). */
    std::int64_t discrete = 0;
    /** @brief Seed pairs whose analytic boundary passes near the mesh. */
    std::int64_t analytic = 0;
    /** @brief Discrete pairs that are not analytic. */
    std::int64_t fake = 0;
    /** @brief Analytic pairs that are not discrete. */
    std::int64_t missing = 0;
    /** @brief Grain pairs adjacent in the mesh that are not compared (a grain without seed). */
    std::int64_t unchecked = 0;
    /** @brief The smallest fake seed pairs (0-based seeds, first < second). */
    std::vector<std::array<int, 2>> fake_examples;
    /** @brief The smallest missing seed pairs (0-based seeds, first < second). */
    std::vector<std::array<int, 2>> missing_examples;
};

/**
 * @brief Compares the grain adjacency of the mesh with the analytic one (see the file comment).
 *
 * **Collective.** With NodeTargetOptions::analytic = false nothing is compared: every discrete
 * grain pair is unchecked.
 * @param targets    targets of the current labels (the labels after repair).
 * @param grain_seed grain g (1-based) grew from seed grain_seed[g - 1], -1 = none; the vector
 *                   given to NodeTargets.
 */
AdjacencyComparison CompareAdjacency(const NodeTargets& targets,
                                     const std::vector<int>& grain_seed);

} // namespace exaconstit::microstructure
