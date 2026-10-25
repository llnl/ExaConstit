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
 * @brief Stage 2c of conformal fitting: small changes to the element labels that remove nodes the
 * fitter cannot place, before any node moves.
 *
 * **Why labels need repair.** The element labels come from sampling the grains at element
 * centers (or votes), so grain boundaries in the mesh are staircases. Most staircase nodes have a
 * valid target (node_targets.hpp), but some configurations cannot be fitted:
 *   - a *peninsula*: an element that touches its own grain through one face only and the other
 *     grain b through all its other faces. Fitting would squeeze it to a sliver; it belongs to b.
 *     (The rule of MFEM's mesh-fitting miniapp, ModifyAttributeForMarkingDOFS in
 *     miniapps/meshing/mesh-fitting.hpp, for N grains: there the element switches when all but
 *     one of its faces see another material.)
 *   - a *bad node*: a FAKE_JUNCTION (the grains of its label set have no common analytic boundary
 *     near it) or OVER_CONSTRAINED node (5 or more grains, a quad point on the outer boundary,
 *     ...).
 *     An element at such a node can often take the label of a face neighbor that the node already
 *     has, which removes the extra label from the node.
 *
 * **Passes.** Each pass:
 *   1. computes the node labels and targets of the current labels, and the peninsulas;
 *   2. proposes at most one flip per element (owner rank). Elements of analytic grains
 *      (grain_seed >= 0) that touch a bad node or are peninsulas preview the flip to each
 *      analytic face-neighbor label b: the label sets their nodes would have (a label leaves a
 *      node when the element was the node's only element with it; NodeLabels::LabelCounts) are
 *      evaluated with the target rules (NodeTargets::Evaluate), and the element's own peninsula
 *      flag is recomputed (its face neighbors keep theirs in the preview: a neighbor on another
 *      rank cannot be recounted, and recounting only local ones would make the result depend on
 *      the partition). The flip with the largest gain in local potential (below) is proposed if the
 *      gain is positive (ties: lowest label); a flip that was undone before is not proposed again;
 *   3. selects flips that are far apart: a flip is chosen when its priority is the largest within
 *      two element rings (elements sharing a node with an element sharing a node with it).
 *      Priority: the previewed gain, then the partition-independent element key
 *      (mesh/element_key.hpp). Within two rings no two chosen flips touch the same node or the
 *      same neighbor, so their effects are independent;
 *   4. drops the chosen flips out of a grain that would leave it with fewer than
 *      min_grain_elements elements (all flips out of that grain in this pass);
 *   5. applies the flips, recomputes, and undoes every flip whose local potential did not strictly
 *      decrease (the preview does not see a neighbor becoming a peninsula; this check does).
 *      Potential = 8 x (bad nodes among the element's nodes) + (peninsulas among the element and
 *      its face neighbors), so removing a bad node outweighs any change of peninsulas (at most 7
 *      elements). Undone flips are remembered and never proposed again.
 * The passes stop when no flip is proposed or kept, or after max_passes. The total potential
 * decreases strictly with every kept flip, so the repair cannot cycle. Nodes still bad at the end
 * stay FAKE_JUNCTION / OVER_CONSTRAINED ("unfit"): the fitter leaves them in place; they are
 * reported, not errors.
 *
 * **Determinism.** Proposals use the element's labels, its face neighbors' labels (ghosts from
 * MeshAdjacency) and node statuses, all identical on every rank; priorities use element keys; the
 * undo decisions use the same data. The repaired labels therefore do not depend on the partition.
 *
 * **Not covered yet** (planned): pinches (a grain touching itself at a single node or edge),
 * grains split by a flip (flips take labels from face neighbors, so a grain grows only into faces
 * it touches, but a flip can cut the grain it leaves), under-resolved grains, and a comparison of
 * the discrete with the analytic grain adjacency. Elements that would collapse when their nodes
 * move are kept above the quality floor by the fitter (conformal_fitter.hpp).
 *
 * Host and MPI (Stage 2 is CPU-only).
 */

namespace exaconstit::microstructure {

/** @brief Settings of RepairLabels. */
struct LabelRepairOptions {
    /** @brief Largest number of passes (the repair stops earlier when no flip has a gain). */
    int max_passes = 50;
    /** @brief A flip may not leave a grain with fewer elements than this (>= 1). */
    int min_grain_elements = 1;
    /** @brief Node label settings (feature angle, node choice). */
    NodeLabelOptions labels;
    /** @brief Target settings (max_move, tolerance, ...). */
    NodeTargetOptions targets;
};

/** @brief What the repair did (global counts, identical on every rank). */
struct LabelRepairReport {
    int passes = 0;                     ///< passes run (with proposals)
    std::int64_t flips = 0;             ///< flips kept
    std::int64_t undone = 0;            ///< flips applied and undone (no decrease)
    std::int64_t blocked = 0;           ///< flips dropped by min_grain_elements
    std::int64_t bad_nodes_before = 0;  ///< FAKE_JUNCTION + OVER_CONSTRAINED nodes, before
    std::int64_t bad_nodes_after = 0;   ///< ... after (unfit nodes)
    std::int64_t peninsulas_before = 0; ///< peninsula elements before
    std::int64_t peninsulas_after = 0;  ///< ... after
};

/** @brief The repaired state: the report plus the node labels and targets of the final labels. */
struct LabelRepairResult {
    LabelRepairReport report;
    std::unique_ptr<NodeLabels> labels;
    std::unique_ptr<NodeTargets> targets;
};

/**
 * @brief Repairs the element labels in place (see the file comment).
 *
 * **Collective** over the mesh communicator.
 * @param pmesh         conforming 3D ParMesh (as NodeLabels).
 * @param element_label in: grain id per local element (>= 1); out: repaired.
 * @param metric, seeds, setup, grain_seed the analytic grain boundaries (as NodeTargets).
 * @param options       passes, grain size guard, label and target settings.
 * @return the report and the node labels / targets of the repaired labels (they refer to
 *         `pmesh`).
 * @throws std::invalid_argument on every rank for invalid options or input (see NodeLabels and
 *         NodeTargets).
 */
LabelRepairResult RepairLabels(mfem::ParMesh& pmesh, mfem::Array<int>& element_label,
                               MetricType metric, const SeedSet& seeds, const MetricSetup& setup,
                               const std::vector<int>& grain_seed,
                               const LabelRepairOptions& options = {});

/**
 * @brief Flips elements whose *current* center (the mesh may already be fitted) lies clearly in
 * another grain: after a fit the grain boundaries have moved, and an element that ended up on the
 * far side of a neighboring grain's analytic boundary belongs to that grain.
 *
 * For every element of an analytic grain that has not flipped before (`flipped`): the seed with
 * the smallest metric value at the element center (exact seed search); if it is not the
 * element's own seed, the distance of the center past the boundary between the two seeds,
 * (f_own - f_best) / |grad f_own - grad f_best|, is compared with `hysteresis` times the element
 * size (Mesh::GetElementSize(e, 0)). Beyond it, the element takes the label of a face neighbor
 * that grew from the best seed (if there is none, it does not flip). Flipped elements are marked
 * in `flipped`, so an element flips at most once over repeated calls.
 *
 * Partition independent: element centers come from the element's own nodes and the seeds are
 * replicated. **Collective** (ghost labels, global count).
 * @param flipped per local element, in/out (resized to NE if empty).
 * @return the global number of flips.
 */
std::int64_t FlipToCurrentOwner(mfem::ParMesh& pmesh, mfem::Array<int>& element_label,
                                MetricType metric, const SeedSet& seeds, const MetricSetup& setup,
                                const std::vector<int>& grain_seed, double hysteresis,
                                std::vector<char>& flipped);

} // namespace exaconstit::microstructure
