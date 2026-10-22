#pragma once

#include "microstructure/labeling/geodesic_common.hpp"
#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <memory>
#include <mpi.h>

/**
 * @file geodesic_graph.hpp
 * @brief Geodesic grain labels from shortest paths along the edges and diagonals of the mesh
 * elements (the "graph geodesic").
 *
 * **What it computes.** The distance from a seed to a point is measured along paths that stay
 * inside the material, so a grain grows around corners and voids instead of across them (a lattice
 * strut, a U-shaped bracket). The paths are approximated on a *graph*: its nodes are the mesh
 * vertices and its edges connect every pair of vertices of the same element (a hex: 28 pairs = 12
 * edges + 12 face diagonals + 4 body diagonals; a tet: its 6 edges), with straight-line lengths.
 * Since an element is (nearly) convex, each such edge lies inside the material, so every graph path
 * is a path inside the material. T(v) = the length of the shortest graph path from any seed to
 * vertex v, label(v) = the seed it comes from (VertexField). Each element then goes to the seed
 * with the smallest arrival distance at its centroid, T(v) + |p - v| minimized over the element's
 * vertices v (or by a volume-weighted vote of subcell points).
 *
 * **Accuracy.** Graph paths can only turn at vertices and only move along the directions of the
 * element edges and diagonals, so T over-estimates the true geodesic distance. On a regular hex
 * grid (26 directions) the over-estimate is zero along the axes and the face and body diagonals
 * and largest in between: at most 8.2% for directions in a coordinate plane and at most 12.8% in
 * 3D (direction near (0.89, 0.37, 0.28)). A Kuhn tet grid is a hex grid with every cube split
 * into 6 tets that all share the cube's diagonal from its (0,0,0) to its (1,1,1) corner; its edges
 * offer only that one body diagonal and one diagonal per face, so it has fewer directions and errs
 * much more in some of them. Measured in test_geodesic
 * (vertices more than 2 units from a point source on a 1 x 1 x 10 strut, h = 1/8): hex max 11.8%,
 * mean 3.7%; tet max 28.9%, mean 6.7%; around the inner corner of an L-shaped strut: hex max
 * 10.1%, tet max 30.5%. The error does not shrink with mesh refinement; it is a property of the
 * edge directions. Two seeds reaching a boundary from similar directions see similar errors, so
 * grain boundaries move much less than the distances themselves (the Additive test locates the
 * boundary exactly on a strut). The Fast Iterative Method solver (labeling/geodesic_fim.hpp) on
 * the same VertexField removes this direction bias and should be preferred where accuracy
 * matters.
 *
 * **Algorithm.**
 * 1. VertexField::InitializeSources: the vertices of each seed's source element start at
 *    T0 = |v - s_i| - a_i.
 * 2. Distances (phase 1). Each rank runs Dijkstra's algorithm on its own elements: vertices are
 *    taken in order of increasing T from a binary heap (a priority queue: the smallest entry comes
 *    out first in O(log n)), and each taken vertex v offers T(v) + |u - v| to every vertex u
 *    sharing an element with it ("relaxation"; kept when smaller). Then
 *    VertexField::ReduceSharedDistance gives every copy of a partition-boundary vertex the
 *    smallest T over the ranks, and Dijkstra re-runs from the vertices whose T dropped. Rounds
 *    repeat until no T changes on any rank.
 * 3. Labels (phase 2). A vertex u with T(u) < +infinity is reached through a "tight" neighbor: a
 *    vertex v of a common element with T(v) < T(u) and T(v) + |u - v| == T(u) exactly (in
 *    floating point), or it is a source vertex with T0(u) == T(u). (T(v) == T(u) is accepted too
 *    when |u - v| > 0 is below the rounding error of T, a degenerate mesh; see below.)
 *    label(u) = the smallest seed index among its source seed and the labels of its tight
 *    neighbors. Each rank fills the labels
 *    in order of increasing T (tight neighbors come first), VertexField::ReduceSharedLabels takes
 *    the minimum over the copies, and the pass repeats until no label changes.
 * 4. Element labels: VertexField::LabelElementCentroids or LabelElementsByVote.
 *
 * **Why the result does not depend on the partition (bitwise).** Write w(v,u) for the computed
 * edge length (the same bits on every rank: same coordinates, same arithmetic, correctly rounded
 * sqrt) and fl(t + w) for the rounded sum.
 * - Every value ever assigned to T(u) is a path sum built outward from a source,
 *   fl(...fl(fl(T0(s) + w1) + w2)... + wk) along some path s -> u. So T(u) >= T*(u), the smallest
 *   such path sum over all paths.
 * - When nothing changes any more, no relaxation improves any vertex, on any rank (the last
 *   exchange made all copies equal). Since fl(t + w) is non-decreasing in t, induction along the
 *   path that attains T*(u) gives T(u) <= T*(u).
 * - Hence T(u) = T*(u), a quantity defined by the mesh alone: the order in which vertices are
 *   processed, the partition and the number of rounds do not matter.
 * Labels: once T is fixed, the tight relation is fixed and only points from smaller to larger T
 * (T(v) < T(u) is required), so the label equation label(u) = min(source label, labels of tight
 * neighbors) has exactly one solution, found by induction in increasing T. Labels start at a value
 * larger than any seed index and only decrease toward that solution, so the rounds reach it on any
 * partition. If degenerate edges make two tight neighbors share the same T, the relation can
 * contain a cycle; the rounds (local passes repeated until stable, then exchanges) still converge
 * to one well-defined answer, the largest solution of the label equation, reached from above
 * whatever the order of the updates, so it is partition independent as well (such vertices may
 * stay unlabeled and are counted in unlabeled_vertices). (Labels are not carried along in phase 1
 * because there a lexicographic (T, label) relaxation can keep a stale label: a neighbor's T may
 * drop by a rounding error without changing the rounded sum, and its new label would then be
 * ignored.)
 * Element labels use only the vertex values of the element, its own seeds (the direct term in a
 * source element, VertexField::LabelPoints) and its centroid or voting points, which are
 * element-local, so they are partition independent too. Vertex coordinates are identical on all
 * copies by construction (VertexField broadcasts the owner's coordinates).
 *
 * **Termination.** Every round of phase 1 strictly lowers at least one T (otherwise it stops), and
 * T takes finitely many floating-point values, so it ends; in practice the number of rounds is one
 * more than the largest number of rank-boundary crossings along a shortest path. Phase 2 lowers at
 * least one label per round and labels are bounded below. GeodesicParameters::max_rounds is only a
 * safety net.
 *
 * **Scope (Stage 1e).** Voronoi (a_i = 0) and Additive (distance - a_i) metrics only; no periodic
 * axes; the caller passes the active material mesh (AttributeDomain::ActiveMesh) so paths stay
 * inside it. A body of the mesh with no seed keeps T = +infinity and label -1, and its elements get
 * element_seed = -1 (the driver decides what to do with them).
 */

namespace exaconstit::microstructure {

// GeodesicParameters, GeodesicReport and GeodesicLabels are shared with the FIM solver and live
// in labeling/geodesic_common.hpp.

/**
 * @brief Graph-geodesic labels of the local elements (see the file comment).
 *
 * **Collective** over `comm`, which must be the mesh communicator (pmesh.GetComm()). Host
 * (Dijkstra and the exchanges), plus device kernels for the element labeling.
 *
 * @param pmesh    conforming 3D ParMesh of the active material (see VertexField). A rank may own
 *                 zero elements.
 * @param comm     the mesh communicator.
 * @param seeds    replicated seeds, N >= 1, finite positions.
 * @param additive true: Additive metric, the source offset is a_i = seeds.weight[i] (empty weight
 *                 array = all zero); false: Voronoi, a_i = 0.
 * @param params   sampling and safety options.
 * @return element_seed, the converged VertexField, the source report and statistics.
 * @throws std::invalid_argument on every rank for invalid parameters or seeds (replicated inputs);
 *         std::runtime_error on every rank for a mesh that VertexField rejects, a mesh without
 *         elements, or max_rounds exceeded.
 */
GeodesicLabels LabelGeodesicGraph(mfem::ParMesh& pmesh,
                                  MPI_Comm comm,
                                  const SeedSet& seeds,
                                  bool additive,
                                  const GeodesicParameters& params);

} // namespace exaconstit::microstructure
