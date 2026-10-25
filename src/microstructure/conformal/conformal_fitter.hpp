#pragma once

#include "microstructure/conformal/node_targets.hpp"
#include "microstructure/conformal/quality.hpp"

#include "mfem.hpp"

#include <cstdint>

/**
 * @file conformal_fitter.hpp
 * @brief Stage 2d of conformal fitting: moves the mesh nodes so that the grain boundaries become
 * smooth surfaces close to their targets (node_targets.hpp: the analytic boundaries, or the
 * smoothed staircase for boundaries without an analytic description), while every element keeps
 * a usable shape.
 *
 * **Goal and trade-off.** A staircase grain boundary cannot be made exactly smooth with the same
 * elements: at every step of the staircase one element has two adjacent faces on the boundary,
 * and those faces would become coplanar (a flat, unusable element). The fitter therefore treats
 * mesh quality as a hard constraint and the analytic surface as a goal: grain boundaries end up
 * smooth and wavy (softened staircases) within a fraction of an element of the analytic surface,
 * and no element falls below a fixed quality floor.
 *
 * **Method: spring-balanced pseudo-elastic morph.** The mesh is treated as a linear elastic
 * solid (Poisson ratio `poisson_ratio`), and every fitted node (status FIT) is pulled toward its
 * target by a spring of stiffness k_v = c h_v m_v (h_v the node's mesh size, m_v a per-node
 * multiplier; nodes on triple lines and quad points get a tenth). The displacement u from the
 * input mesh minimizes
 *
 *     1/2 u^T K u + 1/2 sum_v k_v |x_v + u_v - t_v|^2,
 *
 * one linear system (K + P) u = P (t - x), solved by CG with a BoomerAMG preconditioner in
 * elasticity mode. Unlike prescribing the targets, the springs let the elastic energy and the
 * pull of the targets balance, so the corner elements of the staircase keep a bounded kink
 * instead of collapsing. K is always assembled on the input geometry.
 *
 * **Rounds and local back-off.** Round k uses c = `initial_spring` * 2^k. After each solve every
 * element is checked against its quality limits (at the corners, quality.hpp): for hexes the floor
 * MESH_QUALITY_FLOOR_HEX on the scaled Jacobian and the ceiling MESH_ASPECT_RATIO_MAX on the aspect
 * ratio; for tets the floor MESH_MEAN_RATIO_FLOOR_TET on the mean ratio. An element of the input
 * beyond a limit, or within 10 % of it, may lose at most 10 % of its input value instead, so
 * graded or unstructured input meshes (whose worst elements are often below the floors) can still
 * be fitted. If an element breaks a limit, the multipliers m_v of the fitted nodes of every
 * element within 0.05 of the floor or 0.5 of the ceiling are halved and the round is solved again
 * (at most `max_attempts` times); the halving is agreed over the copies of a shared node (minimum).
 * So only the nodes that block the fit are pulled less, and the rest of the mesh keeps fitting. A
 * round that ends above the floor is kept; the rounds stop when a round from the fourth on (the
 * weak early springs improve little) improves the mean deviation by less than `min_improvement`
 * (relative), when a round cannot be brought above the floor, when a round would leave more than
 * 1 % of the elements mediocre (hexes below floor + 0.2, tets below a mean ratio of 0.5) beyond
 * those already mediocre in the input mesh (the bulk
 * quality guard: quality first, a little more fit is not worth many mediocre elements), or after
 * `max_rounds`. If the very first round fails the bulk guard, it is retried with a quarter of the
 * spring (at most 3 times), so a mesh whose elements start close to the guard still gets a partial
 * fit. The last kept state is the result (the input mesh if none passes).
 *
 * **Quality stiffening.** After each solve, elements below a scaled Jacobian of 0.6 get a stiffness
 * factor (0.6 / sj)^`stiffening_exponent` (capped at 1e3, never lowered again) for the next solve,
 * and the stiffness field is spread to neighboring elements `stiffness_smoothing` times (node
 * maximum, then element mean, kept if larger). Distorted elements resist further distortion and
 * pass it to their neighbors, which leaves far fewer poor elements along the grain boundaries.
 *
 * **Which nodes move.** Every node is free except: on a flat boundary surface whose normal is a
 * coordinate axis (box faces) that coordinate is fixed (nodes slide along the face, edges keep two
 * coordinates, corners all three; the value is kept bit for bit). Free surface nodes stay on their
 * surfaces:
 *  - on one or two flat surfaces with at least one tilted (normal not a coordinate axis), the node
 *    slides along them (along their intersection for two);
 *  - on exactly one curved surface, it slides along the input boundary faces around it (its
 *    one-ring patch, boundary_projector.hpp; it moves well under an element per fit);
 *  - on a curved surface and another one, or on three or more surfaces, it is fixed.
 * A sliding node gets a stiff normal spring (1e4 h, along the tilted planes' normals or the
 * area-weighted normal of its patch) in the elastic solve, and after every solve it is put back
 * on its surfaces exactly (projection onto the planes or the closest point on its patch; box face
 * coordinates restored bit for bit). Where several nodes share one 4-label set (a quad point),
 * only the one closest to the quad point is fitted (decided over all ranks).
 *
 * **Deviation.** The deviation of a fitted node is its distance from its grain boundary at its
 * final position (the closest point from there, NodeTargets::Evaluate with a position), in units
 * of h; the report gives the max and the mean over the fitted nodes.
 *
 * **Higher-order Nodes (order 2).** Every nodal point is fitted, in two passes: the vertices with
 * an order-1 displacement field interpolated to every nodal point, then, with the vertices fixed,
 * the edge and face nodes toward the closest point of their grain boundary from their current
 * position, which curves the element faces (FitConformal in conformal_fitter.cpp says why). Quality
 * is checked at every nodal point (quality.hpp); the elasticity uses a Gauss rule of order 2 p.
 *
 * **Requirements.** The mesh must have Nodes (a continuous H1 ParGridFunction; for a mesh without
 * Nodes call ParMesh::SetCurvature(1) before building the node labels), and the node labels must
 * use those Nodes (FieldNodes::MESH_NODES), so node v is scalar dof v of the Nodes space. Host
 * and MPI (Stage 2 is CPU-only).
 *
 * **Determinism.** Everything up to the linear solves is partition independent. The solves go
 * through hypre, so positions differ between partitions at the level of the solver tolerance; a
 * quality check that lands right at the floor can then decide differently.
 */

namespace exaconstit::microstructure {

/** @brief Quality floor (smallest scaled Jacobian) of a fitted linear hex mesh. */
inline constexpr double MESH_QUALITY_FLOOR_HEX = 0.3;
/**
 * @brief Quality floor of a fitted tet mesh: the smallest mean ratio (quality.hpp, 1 for a regular
 * tet). It replaces the scaled Jacobian floor and the aspect ratio ceiling for tets: a perfect
 * Kuhn tet (a cube cut into six) has a corner scaled Jacobian of only 0.58 and a mean ratio of
 * 0.76. Measured: on an 18^3 x 6 Kuhn mesh with 50 grains the fit reaches 0.23 h (0.28 with the
 * scaled Jacobian rules) with the worst dihedral angle at 16.5 degrees, and an unstructured gmsh
 * cube (whose own worst tet has 10 degrees) is fitted from 0.34 to 0.26 h (it hardly moved
 * before).
 */
inline constexpr double MESH_MEAN_RATIO_FLOOR_TET = 0.4;
/**
 * @brief Largest corner aspect ratio of a fitted element (quality.hpp; hex and tet). Measured on a
 * 32^3 hex / 50-grain polycrystal: without it the fit leaves aspect ratios up to about 6 (0.4 %
 * of the elements above 4); with it the largest is below 4, the smallest scaled Jacobian rises
 * from 0.41 to 0.48 and the mean distance to the targets grows from 0.22 to 0.24 h.
 */
inline constexpr double MESH_ASPECT_RATIO_MAX = 4.0;

/** @brief Settings of FitConformal (library defaults, not user options). */
struct ConformalFitOptions {
    double initial_spring = 1.0;      ///< spring factor c of the first round (> 0)
    int max_rounds = 10;              ///< rounds (c doubles each round)
    int max_attempts = 12;            ///< solves per round with local back-off
    double min_improvement = 0.1;     ///< stop when a round improves the mean deviation less
    double stiffening_exponent = 2.0; ///< quality stiffening (0.6 / sj)^exponent (0 = off)
    int stiffness_smoothing = 2;      ///< spreading passes of the stiffness field
    double poisson_ratio = 0.3;       ///< of the pseudo-elastic solid, in [0, 0.5)
};

/** @brief What the fit did (global values, identical on every rank). */
struct ConformalFitReport {
    std::int64_t fit_nodes = 0;     ///< fitted nodes (pulled toward their targets)
    std::int64_t fixed_nodes = 0;   ///< nodes with every component fixed
    int rounds = 0;                 ///< rounds kept
    int solves = 0;                 ///< elastic solves (all rounds and attempts)
    double final_spring = 0.0;      ///< spring factor c of the last kept round
    double initial_deviation = 0.0; ///< mean deviation of the fitted nodes before the fit / h
    double max_deviation = 0.0;     ///< largest distance of a fitted node from its GB / h
    double mean_deviation = 0.0;    ///< mean of the same
    MeshQuality quality_before;     ///< input mesh
    MeshQuality quality_after;      ///< fitted mesh
};

/**
 * @brief Fits the mesh nodes (see the file comment). Changes the mesh Nodes.
 *
 * **Collective** over the mesh communicator.
 * @param pmesh   the mesh of `targets` (with Nodes; see Requirements).
 * @param targets node targets computed on this mesh, in its current state.
 * @param options library settings.
 * @throws std::invalid_argument on every rank if the mesh has no Nodes, the node labels do not use
 *         them, or the options are invalid.
 */
ConformalFitReport FitConformal(mfem::ParMesh& pmesh, const NodeTargets& targets,
                                const ConformalFitOptions& options = {});

} // namespace exaconstit::microstructure
