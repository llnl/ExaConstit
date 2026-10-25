#pragma once

#include "microstructure/conformal/node_labels.hpp"

#include "mfem.hpp"

#include <vector>

/**
 * @file smoothed_targets.hpp
 * @brief Targets for grain boundaries that have no analytic description: the staircase itself,
 * smoothed (Stage 2 of conformal fitting).
 *
 * **What it is for.** Analytic targets (node_targets.hpp) need grains that are cells of a metric
 * of the seeds. Grains read from a voxel map, grown along paths (geodesic distance), cut by the
 * connected mode, or merged by the fragment cleanup have boundaries that are known only through
 * the element labels: the staircase between the elements. Their nodes get the position of a
 * smoothed copy of that staircase instead. The fitter (conformal_fitter.hpp) treats both kinds
 * of target the same way.
 *
 * **The smoothing.** Only grain boundary nodes move: nodes with 2 labels (on a boundary face) or 3
 * labels (on a triple line) that are not over-constrained and not on a curved feature edge or
 * corner of the outer boundary. Node u is a *neighbor* of such a node v when they are next to each
 * other on the node lattice of an element they share (an element edge for order-1 Nodes; the edges
 * of the sub-elements of the uniform lattice for order 2), u has every label of v (so a face node
 * averages over its own boundary and the triple lines around it, and a triple line node only along
 * its line), and u lies on every boundary surface of v (normals within the feature angle). Every
 * step moves each node to
 *
 *     p_v <- p_v + w (mean of p_u over its neighbors - p_v),
 *
 * alternating w = 0.5 and w = -0.53 (Taubin's lambda|mu smoothing, "A signal processing approach
 * to fair surface design", SIGGRAPH 1995: plain averaging shrinks a curved boundary, the negative
 * step undoes the shrinking but keeps the smoothing of short wavelengths such as stair steps).
 * For order-p Nodes the nodal spacing is the element size / p, so the steps are multiplied by p^2
 * and the pass band 1/lambda + 1/mu divided by p^2 (mu = -0.507 for p = 2): the smoothing then
 * reaches as far, and keeps the same wavelengths, in element sizes.
 * Nodes that do not move (quad points, over-constrained nodes) anchor the network. After every
 * step a node on boundary surfaces is put back on them: onto its planes (box coordinates kept
 * bit for bit), or onto its patch of a curved surface (boundary_projector.hpp). The result is
 * limited to `max_move` element sizes (h times the order of the Nodes) from the input position
 * (scaled back along the move).
 *
 * **Determinism.** A node's neighbors can live on other ranks. Each step, every rank sends for
 * each grain boundary node the (input position, current position) of the neighbors it knows
 * (VertexField::ExchangeSharedLists); every copy merges its own and the received lists, removes
 * duplicates by input position and sorts by it, then averages in that order. So every copy of a
 * node computes the same bits, for any partition.
 *
 * Host and MPI.
 */

namespace exaconstit::microstructure {

class SurfacePatches;

/** @brief Settings of SmoothGrainBoundaries. */
struct SmoothingParameters {
    /** @brief Pairs of (lambda, mu) steps for order-1 Nodes (times order^2 for higher orders, so
     * the smoothing reaches as far in element sizes). */
    int steps = 20;
    /** @brief Largest distance of a smoothed position from the input, in element sizes h. */
    double max_move = 2.0;
};

/**
 * @brief Smoothed positions of the grain boundary nodes (see the file comment).
 *
 * **Collective** over the mesh communicator.
 * @param labels   node labels (label sets, surfaces).
 * @param patches  patches of the nodes on exactly one curved surface.
 * @param h        mesh size per node.
 * @param params   steps and move limit.
 * @param smoothed out: 3 per node, the smoothed position (the input position where `moved` is 0).
 * @param moved    out: per node, 1 when the node takes part in the smoothing (a grain boundary node
 *                 as defined in the file comment), else 0.
 */
void SmoothGrainBoundaries(const NodeLabels& labels, const SurfacePatches& patches,
                           const mfem::Vector& h, const SmoothingParameters& params,
                           mfem::Vector& smoothed, std::vector<char>& moved);

} // namespace exaconstit::microstructure
