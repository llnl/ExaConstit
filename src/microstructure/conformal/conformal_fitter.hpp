#pragma once

#include "mfem.hpp"

#include <cstdint>

#include "microstructure/conformal/node_targets.hpp"
#include "microstructure/conformal/quality.hpp"

/**
 * @file conformal_fitter.hpp
 * @brief Moves the mesh nodes so that the stair-stepped grain boundaries of a labeled mesh become
 * smooth surfaces close to the true grain boundaries, while every element keeps a usable shape.
 *
 * **The problem.** After labeling, every element belongs to one grain, so a grain boundary runs
 * along element faces and looks like a staircase. NodeTargets (node_targets.hpp) gives every node
 * on a grain boundary a target: the nearest point on the exact boundary between its seeds, or on
 * a smoothed copy of the staircase where no exact boundary exists. Moving each node straight to
 * its target does not work: at every step of the staircase one element has two neighboring faces
 * on the boundary, and making the boundary smooth there would flatten that element. The fitter
 * therefore treats element quality as a hard limit and the target as a goal: boundaries end up
 * smooth to within a fraction of an element of their targets (a softened staircase) and no
 * element falls below the quality limits.
 *
 * **Method: an elastic solid pulled by springs.** The mesh is treated as a linear elastic body
 * (Poisson ratio `poisson_ratio`, Young's modulus 1 times a per-element stiffness factor, see
 * below). Every fitted node v is pulled toward its target t_v by a spring of stiffness
 * k_v = c h_v m_v, with h_v the local element size, c the spring factor of the current round and
 * m_v a per-node multiplier that starts at 1. The node displacements u minimize the elastic
 * energy plus the spring energy,
 *
 *     1/2 u^T K u + 1/2 sum_v k_v |x_v + u_v - t_v|^2,
 *
 * so u solves one linear system (K + P) u = P (t - x), P the diagonal matrix of the springs. The
 * system is solved by the conjugate gradient method (CG) with an l1 Gauss-Seidel preconditioner
 * from hypre. Because the targets are pulled by springs rather than imposed, the elastic energy
 * and the pull balance: at a staircase step the element keeps a bounded kink instead of being
 * flattened. K is always assembled on the input mesh, so each solve starts from the same body.
 *
 * **Rounds.** Round k uses the spring factor c = `initial_spring` * 2^k, from weak to strong.
 * After each solve every element is checked against its quality limits (quality.hpp, measured at
 * its corners):
 *   - hexes: scaled Jacobian at least MESH_QUALITY_FLOOR_HEX and aspect ratio at most
 *     MESH_ASPECT_RATIO_MAX;
 *   - tets: mean ratio at least MESH_MEAN_RATIO_FLOOR_TET.
 * An element that is already beyond a limit in the input mesh, or within 10 % of it, may instead
 * lose at most 10 % of its input value; for a hex whose aspect ratio is already above the
 * ceiling, the aspect ratio may grow by 30 % (such elements are thin, like a layer of flat hexes
 * along a curved surface, and their aspect ratio changes with every small move across them).
 * This way graded or unstructured meshes, whose worst elements often start below the limits,
 * can still be fitted.
 *
 * **Backing off where elements suffer.** When an element breaks a limit, the multipliers m_v of
 * the fitted nodes of every element close to a limit (within 0.05 of a floor, or 0.5 of the
 * aspect ratio ceiling) are halved, and the round is solved again, at most `max_attempts` times.
 * A node shared by several ranks takes the smallest multiplier of its copies, so all copies
 * agree. Only the nodes that cause trouble are pulled less; the rest of the mesh keeps fitting.
 * An element close to a limit with no fitted node of its own is distorted through its neighbors,
 * so the fitted nodes of the elements around it back off instead.
 *
 * **When the rounds stop.** A round whose final solve respects every limit is kept. The rounds
 * stop when
 *   - a round from the fourth on improves the mean deviation by less than `min_improvement`
 *     (relative; the first rounds have weak springs and always improve little);
 *   - a round cannot be brought within the limits in `max_attempts` solves;
 *   - a round would leave more than 1 % of the elements *mediocre* (hexes with a scaled Jacobian
 *     below the floor + 0.2, tets with a mean ratio below 0.5), not counting those already
 *     mediocre in the input. This bulk quality guard puts quality first: a little more fit is not
 *     worth many mediocre elements. If the very first round fails it, the round is retried with a
 *     quarter of the spring (at most 3 times), so a mesh whose elements start close to the guard
 *     still gets a partial fit;
 *   - `max_rounds` rounds are done.
 * The nodes of the last kept round are the result (the input mesh if no round is kept).
 *
 * **Stiffening distorted elements.** After each solve, an element with a scaled Jacobian sj below
 * 0.6 gets the stiffness factor (0.6 / sj)^`stiffening_exponent` for the following solves (capped
 * at 1000 and never lowered again). The factor is then spread to the neighboring elements
 * `stiffness_smoothing` times (each node takes the largest factor of its elements, then each
 * element the mean of its nodes, kept when larger). Distorted elements thus resist further
 * distortion and pass it on to their neighbors, which leaves far fewer poor elements along the
 * grain boundaries. A poor input element that breaks its limit also doubles its factor.
 *
 * **Which nodes move.** Every node may move, with these restrictions:
 *   - on a flat outer face whose normal is a coordinate axis (the faces of a box), the coordinate
 *     along that axis is fixed: nodes slide in the face, edge nodes keep two coordinates, corner
 *     nodes all three, and the fixed values are kept exactly, bit for bit;
 *   - on one or two other flat surfaces (tilted planes), the node slides in the plane, or along
 *     the line where the two planes meet;
 *   - on one curved surface, or on a sharp edge where a curved surface meets one other surface,
 *     the node slides on the input boundary faces around it (patches, boundary_projector.hpp);
 *   - on three or more surfaces (a corner), the node does not move.
 * During the solve, a sliding node is held near its surfaces by a stiff spring along their
 * normals (1000 h; the plane normals, or for a curved surface the area-weighted normal of its
 * patch). After every solve it is put back on its surfaces exactly (projection onto the planes,
 * or the closest point on its patches, box-face coordinates restored exactly).
 *
 * **Quad points.** A point where four grains meet is a single point, but several nodes can carry
 * the same four grains. Only the node closest to that point is fitted; the decision is made over
 * all ranks, so every copy of a node agrees.
 *
 * **Deviation.** The deviation of a fitted node is the distance from its final position to its
 * grain boundary (the closest point from there, NodeTargets::Evaluate with a position), divided
 * by the local element size h. The report gives the largest and the mean deviation over the
 * fitted nodes, before and after the fit.
 *
 * **Higher-order meshes.** On a mesh whose Nodes are of order p > 1 every nodal point (vertices
 * and the extra points on edges, faces and inside elements) is fitted, in two passes. First the
 * vertices, with an order-1 displacement field interpolated to every nodal point, so the
 * elements move as straight-sided elements would. Then, with the vertices fixed, the other nodal
 * points are pulled toward the closest point on their grain boundary from where they are, which
 * curves the element faces (FitConformal in conformal_fitter.cpp explains why). Quality is
 * checked at every nodal point (quality.hpp), and the elasticity uses a Gauss rule of order 2 p.
 *
 * **Requirements.** The mesh must have Nodes (a continuous H1 ParGridFunction; for a mesh without
 * Nodes call ParMesh::SetCurvature(1) before building the node labels), and the node labels must
 * be built on those Nodes (FieldNodes::MESH_NODES), so that node v is scalar dof v of the Nodes
 * space. Host only (the fit runs on the CPU), MPI parallel.
 *
 * **Partition independence.** Everything up to the linear solves gives the same result on any
 * number of ranks. The solves go through hypre, whose results differ between partitions at the
 * level of the solver tolerance; a quality check that lands right at a limit can then decide
 * differently on a different partition.
 */

namespace exaconstit::microstructure {

/** @brief Smallest scaled Jacobian (quality.hpp) a fitted hex may have, unless its input value
 * was already lower (see the file comment). */
inline constexpr double MESH_QUALITY_FLOOR_HEX = 0.3;
/**
 * @brief Smallest mean ratio (quality.hpp; 1 for a regular tet) a fitted tet may have, unless its
 * input value was already lower.
 *
 * Tets are judged by the mean ratio instead of the scaled Jacobian floor and the aspect ratio
 * ceiling. The corner scaled Jacobian is a poor shape measure for tets: a perfect Kuhn tet (one of
 * the six tets a cube is cut into) scores only 0.58 (mean ratio 0.76), and the thin "sliver" tets
 * of unstructured meshes jump in scaled Jacobian with tiny node moves. With the scaled Jacobian
 * rules the fit of tet meshes stopped early; with the mean ratio floor it reaches about the
 * quality of hex fits while the worst dihedral angles stay those of the input mesh.
 */
inline constexpr double MESH_MEAN_RATIO_FLOOR_TET = 0.4;
/**
 * @brief Largest corner aspect ratio (quality.hpp) a fitted hex may have, unless its input value
 * was already higher.
 *
 * Without this ceiling the fit of a 50-grain hex polycrystal left aspect ratios up to about 6
 * (0.4 % of the elements above 4). With it the largest stays below 4 and the smallest scaled
 * Jacobian is higher, for a slightly larger mean deviation (about 0.02 h).
 */
inline constexpr double MESH_ASPECT_RATIO_MAX = 4.0;

/** @brief Settings of FitConformal. These are library defaults, not user options. */
struct ConformalFitOptions {
    /** @brief Spring factor c of the first round (> 0); it doubles every round. */
    double initial_spring = 1.0;
    /** @brief Most rounds (>= 1). */
    int max_rounds = 10;
    /** @brief Most solves per round, each after a back-off (>= 1). */
    int max_attempts = 12;
    /** @brief The rounds stop (from the fourth round on) when one improves the mean deviation by
     * less than this fraction (>= 0). */
    double min_improvement = 0.1;
    /** @brief Exponent of the stiffening factor (0.6 / sj)^exponent of distorted elements
     * (>= 0; 0 = no stiffening). */
    double stiffening_exponent = 2.0;
    /** @brief Times the stiffening factor is spread to neighboring elements (>= 0). */
    int stiffness_smoothing = 2;
    /** @brief Poisson ratio of the elastic body, in [0, 0.5). */
    double poisson_ratio = 0.3;
};

/** @brief What FitConformal did. Every value is global (summed or reduced over all ranks) and
 * identical on every rank. */
struct ConformalFitReport {
    /** @brief Nodes pulled toward a target (each node counted once, by its owner rank). */
    std::int64_t fit_nodes = 0;
    /** @brief Nodes with all three coordinates fixed (box corners, corners of the outer surface).
     * On a higher-order mesh this is the count of the second pass, in which every vertex is
     * fixed. */
    std::int64_t fixed_nodes = 0;
    /** @brief Rounds kept (on a higher-order mesh, of both passes). */
    int rounds = 0;
    /** @brief Linear elastic solves done, over all rounds and attempts. */
    int solves = 0;
    /** @brief Spring factor c of the last kept round (0 when none was kept). */
    double final_spring = 0.0;
    /** @brief Mean deviation of the fitted nodes before the fit, in element sizes h. */
    double initial_deviation = 0.0;
    /** @brief Largest deviation of a fitted node after the fit, in element sizes h. */
    double max_deviation = 0.0;
    /** @brief Mean deviation of the fitted nodes after the fit, in element sizes h. */
    double mean_deviation = 0.0;
    /** @brief Quality of the input mesh. */
    MeshQuality quality_before;
    /** @brief Quality of the fitted mesh. */
    MeshQuality quality_after;
};

/**
 * @brief Moves the mesh nodes so that the grain boundaries approach their targets within the
 * element quality limits (see the file comment). Changes the Nodes of `pmesh` in place.
 *
 * Host only. Collective over the mesh communicator.
 * @param pmesh   the mesh the targets were computed on, in the same state (with Nodes; see
 *                Requirements in the file comment).
 * @param targets node targets of the current element labels.
 * @param options library settings (the defaults are the tested ones).
 * @return what the fit did, identical on every rank.
 * @throws std::invalid_argument on every rank if the mesh has no Nodes, the node labels were not
 *         built on them, or an option is out of range.
 */
ConformalFitReport FitConformal(mfem::ParMesh& pmesh,
                                const NodeTargets& targets,
                                const ConformalFitOptions& options = {});

} // namespace exaconstit::microstructure
