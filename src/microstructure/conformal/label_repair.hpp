#pragma once

#include "microstructure/conformal/node_labels.hpp"
#include "microstructure/conformal/node_targets.hpp"
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <memory>
#include <vector>

/**
 * @file label_repair.hpp
 * @brief Small changes to the grain labels of individual elements that remove the spots the
 * conformal fit cannot handle, made before any node moves.
 *
 * **Why labels need repair.** The grain of an element comes from sampling the tessellation at its
 * center (or from a vote of its quadrature points), so a grain boundary in the mesh follows
 * element faces like a staircase. NodeTargets (node_targets.hpp) finds, for most nodes on such a
 * staircase, a point on the exact grain boundary to move to. Two kinds of spots have no good
 * target:
 *   - a *peninsula*: an element that shares at most one of its faces with its own grain and all
 *     the others with one other grain b. Fitting would squeeze it into a sliver; it belongs to b.
 *     (This generalizes to many grains the rule of MFEM's mesh-fitting miniapp,
 *     ModifyAttributeForMarkingDOFS in miniapps/meshing/mesh-fitting.hpp, where an element
 *     switches material when all but one of its faces see the other material.)
 *   - a *bad node*: a node with status FAKE_JUNCTION (the grains on the node have no common exact
 *     boundary near it, or a grain touches itself there only at a node or an edge, a "pinch") or
 *     OVER_CONSTRAINED (more grains and outer surfaces meet at the node than a point can satisfy,
 *     such as five grains, or four grains on the outer boundary).
 * An element at a bad node can often take the grain of one of its face neighbors that the node
 * already carries; the node then loses one grain and becomes fittable.
 *
 * **Changing an element's grain ("a flip").** The repair runs in passes. Each pass:
 *   1. builds the node labels and targets of the current element labels, and finds the
 *      peninsulas;
 *   2. lets every element that is a peninsula or touches a bad node *preview* switching to each
 *      grain b of its face neighbors (only grains that may change: not the fixed grains of
 *      passthrough parts, NodeLabelOptions::fixed_labels). The preview recomputes the grains its
 *      nodes would carry (a grain leaves a node when this element was the node's only element of
 *      that grain, NodeLabels::LabelCounts), evaluates those nodes with the target rules
 *      (NodeTargets::Evaluate), and recomputes whether the element itself would still be a
 *      peninsula. Each element proposes the switch that lowers its *local potential* (below) the
 *      most, if any; on equal gains the smaller grain id wins, and a switch that was undone once is
 *      never proposed again. (The face neighbors keep their peninsula flags in the preview: a
 *      neighbor on another rank cannot be recounted here, and recounting only the local ones
 *      would make the result depend on the partition.);
 *   3. chooses switches that are far apart, so they cannot interfere: a switch is chosen when its
 *      priority is the largest within two element rings (the elements that share a node with an
 *      element that shares a node with it). The priority is the gain, then the element key, a
 *      number that identifies the element the same way on any partition (mesh/element_key.hpp);
 *   4. drops the chosen switches out of a grain that would be left with fewer than
 *      min_grain_elements elements (all switches out of that grain in this pass);
 *   5. applies the switches; undoes every switch into or out of a grain that now consists of
 *      more separate pieces than before (a switch may never split a grain; pieces are counted with
 *      LabelComponents, labeling/fragment_cleanup.hpp, the same on any partition); rebuilds the
 *      node labels and targets; and undoes every switch after which the element's local potential
 *      did not strictly go down (the preview cannot see a neighbor turning into a peninsula; this
 *      check can).
 * The local potential of an element is 8 x (bad nodes among its nodes) + (peninsulas among the
 * element and its face neighbors). An element and its face neighbors are at most 7 elements, so
 * removing one bad node always outweighs any change in peninsulas. Every kept switch lowers the
 * total, so the repair cannot cycle. The passes stop when no switch is proposed or kept, or after
 * max_passes.
 *
 * **Targets during and after the passes.** During the passes the targets use the exact-boundary
 * rules only: no fallback to the smoothed staircase, and the bad nodes of grains too thin for the
 * mesh and the pinched nodes count as bad, so the repair still tries to remove them. The targets
 * returned at the end are rebuilt with the options as given (with the smoothed fallback, thin
 * grains and remaining pinches held in place). Without exact boundaries
 * (NodeTargetOptions::analytic = false: grains read from voxel maps or grown along paths) only
 * over-constrained nodes count as bad, so the repair removes peninsulas and over-constrained
 * nodes. Nodes that are still bad at the end keep their status; the fit leaves them in place.
 * They are reported, not errors.
 *
 * **Same result on any partition.** Proposals use the element's grain, its face neighbors' grains
 * (neighbors on other ranks come as ghost copies, MeshAdjacency) and node statuses, all the same
 * on every rank; priorities use element keys; the undo decisions use the same data. The repaired
 * labels therefore do not depend on the number of ranks.
 *
 * How the repaired grain adjacency compares with the exact one is checked in adjacency_check.hpp.
 *
 * Host only. Both functions are collective over the mesh communicator.
 */

namespace exaconstit::microstructure {

/** @brief Settings of RepairLabels. */
struct LabelRepairOptions {
    /** @brief Most passes (>= 0); the repair stops earlier when no switch is proposed or kept. */
    int max_passes = 50;
    /** @brief A switch may not leave a grain with fewer elements than this (>= 1). */
    int min_grain_elements = 1;
    /** @brief Settings of the node labels (feature angle, which nodes, fixed grains). */
    NodeLabelOptions labels;
    /** @brief Settings of the targets (move limit, tolerances, smoothed fallback, holds). */
    NodeTargetOptions targets;
};

/** @brief What the repair did. Global counts (summed over all ranks), identical on every rank. */
struct LabelRepairReport {
    int passes = 0;                     ///< passes that had at least one proposal
    std::int64_t flips = 0;             ///< element switches kept
    std::int64_t undone = 0;            ///< switches applied and undone (potential did not drop)
    std::int64_t blocked = 0;           ///< switches dropped by min_grain_elements
    std::int64_t split_undone = 0;      ///< switches undone because they split a grain
    std::int64_t bad_nodes_before = 0;  ///< FAKE_JUNCTION + OVER_CONSTRAINED nodes before
    std::int64_t bad_nodes_after = 0;   ///< the same with the final targets (nodes left unfit)
    std::int64_t peninsulas_before = 0; ///< peninsula elements before
    std::int64_t peninsulas_after = 0;  ///< peninsula elements after
};

/** @brief Result of RepairLabels: the report, and the node labels and targets of the repaired
 * element labels (ready for the fit; both refer to the mesh given to RepairLabels). */
struct LabelRepairResult {
    /** @brief What the repair did. */
    LabelRepairReport report;
    /** @brief Node labels of the repaired element labels. */
    std::unique_ptr<NodeLabels> labels;
    /** @brief Node targets of the repaired labels, built with the options as given. */
    std::unique_ptr<NodeTargets> targets;
};

/**
 * @brief Repairs the element labels in place (see the file comment).
 *
 * Host only. Collective over the mesh communicator.
 * @param pmesh         conforming 3D ParMesh (the requirements of NodeLabels).
 * @param element_label in: grain per local element (1-based); out: the repaired grains.
 * @param metric        the tessellation metric (Voronoi, Laguerre, ...), for the exact boundaries.
 * @param seeds         the seeds of the tessellation (every rank holds all seeds).
 * @param setup         metric parameters of the seeds (weights, anisotropy).
 * @param grain_seed    seed of every grain: grain g grew from seed grain_seed[g - 1] (0-based), or
 *                      -1 for a grain without a seed (no exact boundary).
 * @param options       passes, smallest grain size, node label and target settings.
 * @return the report and the node labels and targets of the repaired labels.
 * @throws std::invalid_argument on every rank for invalid options or input (see NodeLabels and
 *         NodeTargets).
 */
LabelRepairResult RepairLabels(mfem::ParMesh& pmesh,
                               mfem::Array<int>& element_label,
                               MetricType metric,
                               const SeedSet& seeds,
                               const MetricSetup& setup,
                               const std::vector<int>& grain_seed,
                               const LabelRepairOptions& options = {});

} // namespace exaconstit::microstructure
