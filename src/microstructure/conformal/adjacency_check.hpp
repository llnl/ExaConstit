#pragma once

#include "microstructure/conformal/node_targets.hpp"

#include <array>
#include <cstdint>
#include <vector>

/**
 * @file adjacency_check.hpp
 * @brief Checks whether the grains that touch in the mesh are the grains that should touch: the
 * grain adjacency of the labeled mesh compared with the exact (analytic) tessellation.
 *
 * **Why it matters.** A tessellation such as Voronoi defines exactly which grains share a
 * boundary. On a mesh, every element gets one grain, so the grain boundaries follow element
 * faces (a staircase). The staircase can create contacts that the exact tessellation does not
 * have, and it can lose small faces that it does have. Which grains touch decides which grain
 * boundaries carry load and misorientation in a crystal plasticity model, so the generator
 * reports both kinds of disagreement in its log after the conformal fit.
 *
 * **Words used below.**
 *   - Two grains are *adjacent in the mesh* (a *discrete* pair) when some mesh node belongs to
 *     elements of both grains: they share a face, an edge or only a node.
 *   - Two seeds are *analytically adjacent* when their exact boundary passes close to the mesh:
 *     some node could be moved onto that boundary within the move limit of the conformal fit
 *     (NodeTargetOptions::max_move element sizes, keeping the node on the outer surfaces it lies
 *     on) and no third seed is closer there. These are the rules NodeTargets uses to place nodes
 *     on grain boundaries, without its fallback to the smoothed staircase.
 *   - A *fake* adjacency is a discrete pair that is not analytic: the grains touch in the mesh
 *     although their exact boundary comes near no node that both of them own. Typical causes are
 *     a staircase step left after label repair, or grains that touch only at one node or edge.
 *   - A *missing* adjacency is an analytic pair that is not discrete: the exact boundary passes
 *     near the mesh but no node belongs to both grains. Typical cause: a face of the Voronoi
 *     diagram smaller than an element, which the staircase cuts off.
 *
 * **What is compared.** Grains are compared through the seeds they grew from (grain_seed). A
 * pair that involves a grain without a seed (grains read from a voxel map, grown along paths with
 * distance = "geodesic", or passthrough parts) has no exact boundary and is counted as
 * *unchecked*. Two grains of the same seed (pieces of one cell kept as separate grains) are not a
 * pair at all.
 *
 * **How pairs are found.** Discrete pairs come from every pair of labels on every node this rank
 * owns. Such a pair is confirmed analytic when NodeTargets::EvaluateAnalytic finds a target for
 * exactly those two grains at that node. Analytic pairs that are not discrete cannot be found
 * from the labels, so at every owned node the closest seed and the next seeds whose boundary with
 * it is estimated to lie within twice the move limit (NodeTargets::NearbySeeds, at most
 * ADJACENCY_NEARBY_SEEDS seeds) are paired and queried as well. A pair is therefore found from any
 * node near its boundary, not only from nodes of its own two grains.
 *
 * **Same answer on any number of ranks.** A pair can be seen on several ranks, and with
 * different flags. Every pair is sent to one *owner rank* chosen from a hash of the pair (an
 * all-to-all exchange, MPI_Alltoallv: every rank sends a list to every other rank), the owner
 * combines the flags of all copies, and the counts are summed over the owners. The node queries
 * use only data every rank holds in full (the seeds), so the counts and the example lists are
 * identical for every partition of the mesh.
 *
 * Host only. CompareAdjacency is collective over the mesh communicator of `targets` (every rank
 * must call it).
 */

namespace exaconstit::microstructure {

/** @brief Seeds paired per node when searching for analytic pairs: the closest seed and up to
 * three more. */
inline constexpr int ADJACENCY_NEARBY_SEEDS = 4;
/** @brief Number of fake and of missing pairs listed as examples in AdjacencyComparison (the
 * pairs with the smallest seed indices). */
inline constexpr int ADJACENCY_EXAMPLES = 10;

/**
 * @brief Result of CompareAdjacency: counts of grain pairs that touch in the mesh and in the
 * exact tessellation, and of the disagreements. The same on every rank.
 */
struct AdjacencyComparison {
    /** @brief Seed pairs adjacent in the mesh (pairs of grains grown from two different seeds). */
    std::int64_t discrete = 0;
    /** @brief Seed pairs whose exact boundary passes close to the mesh. */
    std::int64_t analytic = 0;
    /** @brief Seed pairs adjacent in the mesh but not in the exact tessellation. */
    std::int64_t fake = 0;
    /** @brief Seed pairs adjacent in the exact tessellation but not in the mesh. */
    std::int64_t missing = 0;
    /** @brief Grain pairs adjacent in the mesh that cannot be compared because one of the grains
     * has no seed. */
    std::int64_t unchecked = 0;
    /** @brief Up to ADJACENCY_EXAMPLES fake pairs as (seed a, seed b), 0-based seed indices with
     * a < b, smallest pairs first. */
    std::vector<std::array<int, 2>> fake_examples;
    /** @brief Up to ADJACENCY_EXAMPLES missing pairs, in the same form as fake_examples. */
    std::vector<std::array<int, 2>> missing_examples;
};

/**
 * @brief Compares the grain adjacency of the labeled mesh with that of the exact tessellation
 * (see the file comment for the definitions).
 *
 * With NodeTargetOptions::analytic = false there is no exact boundary to compare with: every
 * pair adjacent in the mesh is counted as unchecked and the other counts are 0.
 *
 * Host only. Collective over the mesh communicator of `targets`.
 * @param targets    node targets built from the current element labels (after label repair).
 * @param grain_seed seed of every grain: grain g (1-based) grew from seed grain_seed[g - 1]
 *                   (0-based), or -1 when it has no seed. The same vector given to NodeTargets.
 * @return the global counts and examples, identical on every rank.
 */
AdjacencyComparison CompareAdjacency(const NodeTargets& targets,
                                     const std::vector<int>& grain_seed);

} // namespace exaconstit::microstructure
