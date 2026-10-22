#pragma once

#include "microstructure/labeling/geodesic_common.hpp"
#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <mpi.h>

/**
 * @file geodesic_fim.hpp
 * @brief Geodesic grain labels from the eikonal equation |grad T| = 1, solved on the mesh vertices
 * with the Fast Iterative Method (FIM). This is the default, accurate geodesic solver; the graph
 * geodesic (labeling/geodesic_graph.hpp) is the fast, approximate alternative.
 *
 * **What it computes.** T(x) = the length of the shortest path from the nearest seed to x that
 * stays inside the material (minus the seed's additive weight a_i for the Additive metric). It is
 * the solution of the eikonal equation |grad T| = 1 (the front T = t moves with unit speed
 * through the material) with T = |x - s_i| - a_i near seed i. Grains then grow around corners and
 * voids instead of reaching across them (lattice struts, a U-shaped bracket). T and the seed index
 * are stored on every mesh vertex (VertexField), and each element goes to the seed with the
 * smallest arrival distance at its centroid (or by a volume-weighted vote of subcell points).
 *
 * **Why not the graph.** A graph path can only move along element edges and diagonals, which
 * over-estimates distances by up to 12.8% on a hex grid and ~30% on a tet grid whatever the mesh
 * size. The eikonal update below lets a path cross an element face at any point (the arrival time
 * on the face is interpolated linearly), so the direction bias disappears and the error shrinks
 * with the mesh size (first order, plus the usual extra error next to a point source).
 *
 * **Local update** (labeling/eikonal_local_solver.hpp). For a vertex v and a triangle (a, b, c) of
 * other vertices of one of v's elements, the candidate value is
 *     min over points p of the triangle of  T_lin(p) + |v - p|,
 * where T_lin interpolates T(a), T(b), T(c) linearly over the triangle: the front reaches p at
 * time T_lin(p) and then travels in a straight line to v. The minimum is computed in closed form
 * (Fu, Kirby & Whitaker, SIAM J. Sci. Comput. 35(5), 2013), with the triangle's edges and vertices
 * as fallbacks when the minimizer lies outside the triangle.
 *   - Tets: the triangle is the face of each adjacent tet opposite v (the standard tet FIM).
 *   - Hexes ("virtual simplices"): every triangle of the other 7 vertices of each adjacent hex (35
 *     per hex), so a path may enter the hex anywhere on the faces, diagonals or interior planes
 *     spanned by its vertices. All of these triangles lie inside a straight-sided hex with planar
 *     faces (which is convex), so a path never leaves the element and never cuts across a void.
 *     A trilinear hex with warped (non-planar) faces is not convex: a path may leave it by up to
 *     the face warp. For a curved hex only the vertices are used (the element is replaced by the
 *     convex hull of its corners), the same approximation as VertexField's element labeling.
 *   - Paths go from element to element through every shared vertex, edge or face. Two elements
 *     touching only along an edge or at a vertex are therefore connected for the geodesic solvers
 *     (a front passes through the shared edge), whereas the face-based fragment cleanup
 *     (labeling/fragment_cleanup.hpp) treats them as separate pieces.
 *   - Any other element type (wedge, pyramid) uses the same rule: every triangle of the other
 *     vertices of the element.
 * In practice the element's candidates are evaluated as: every other vertex alone (T(a) + |v - a|),
 * every pair (the minimum over the open segment), and every triple (the interior minimizer); the
 * smallest of these equals the smallest full-triangle minimum, since every edge and vertex of a
 * triangle appears as a pair or a vertex of its own.
 *
 * **Jacobi rounds (phase 1, distances).** Starting from VertexField::InitializeSources (the
 * vertices of each seed's source element hold T0 = |v - s_i| - a_i, all others +infinity):
 *  1. every *active* vertex v computes T_new(v) = min(T(v), min over its candidates), using only
 *     the previous round's T values ("synchronous" or Jacobi update: no vertex sees a value written
 *     in the same round, so the order of the vertices does not matter); one mfem::forall iteration
 *     per vertex, reading the vertex -> element and element -> vertex tables, no atomics;
 *  2. VertexField::ReduceSharedDistance gives every copy of a partition-boundary vertex the
 *     smallest T over the ranks;
 *  3. the next round's active vertices are those sharing an element with a vertex whose T changed
 *     in step 1 or 2 (round 1: those sharing an element with a reached vertex);
 *  4. repeat until no T changes on any rank.
 *
 * **Labels (phase 2).** Seed labels are not carried along in phase 1 (a label riding on a value
 * that later drops by a rounding error could stay stale). Once T has converged:
 *  1. For every reached vertex v, m(v) = the smallest stored candidate value of the final T (the
 *     values of the last distance sweeps, not recomputed), also counting the source value T0(v)
 *     (minimum over the ranks, so all copies agree).
 *  2. The *upwind* vertices of v are the vertices of every candidate attaining m(v) exactly (the
 *     support of the minimizer: 3 vertices for a triangle interior, 2 for a segment, 1 for a
 *     vertex), restricted to vertices u with T(u) < T(v). v is a *source* vertex when T0(v) = m(v).
 *  3. label(v) = the smallest seed index among v's source seed (if any) and the labels of its
 *     upwind vertices. Computed by Jacobi rounds with VertexField::ReduceSharedLabels: labels start
 *     at "unknown" (larger than any seed index) except at source vertices and only decrease.
 * Ties go to the lowest seed index.
 *
 * **Why the result does not depend on the partition (bitwise).**
 *  - Every candidate value is computed by the same code (one kernel, CandidateSweep, in one
 *    binary on every rank) from the same inputs: the vertex coordinates are bitwise equal on all
 *    copies, and the local solver sorts the triangle or segment vertices by their coordinates
 *    before computing, so the arithmetic does not depend on the local vertex numbering or on the
 *    element's vertex order. This holds with or without MICROSTRUCTURE_STRICT_FP (off by
 *    default); it would *not* hold if a value were recomputed in a second kernel, where the
 *    compiler may fuse multiply-adds differently. The label pass therefore reads the candidate
 *    values and supports stored by the distance sweeps (per vertex-element pair) and never
 *    recomputes them.
 *  - Consistency of the stored values with T: a vertex's stored candidates are recomputed whenever
 *    a vertex of its elements changes, so after the last round they are the candidates of the
 *    converged T, and T(v) is no larger than any of them or than T0(v): T(v) <= m(v). Normally
 *    T(v) = m(v); if a candidate went up by a rounding bit when an input went down (the closed
 *    form is not exactly monotone), m(v) > T(v), and the upwind set is still taken from the
 *    candidates attaining m(v), filtered by T(u) < T(v).
 *  - In round k, a full synchronous sweep over the whole mesh would give
 *        T^k(v) = min( T^{k-1}(v), min over all candidates of v computed from T^{k-1} ).
 *    Each rank computes the minimum over the candidates of its own elements; step 2 takes the
 *    minimum over the ranks. A minimum of doubles is exact and does not depend on the order in
 *    which it is taken, so the result equals the full sweep, provided all copies agree at the start
 *    of the round (they do, after the previous step 2; at round 1 after InitializeSources).
 *  - Skipping inactive vertices changes nothing: a vertex none of whose neighbors changed would
 *    recompute exactly the candidates it computed when it was last active, and its T is already no
 *    larger than those.
 *  Hence the sequence T^0, T^1, ... is the one of the full sweep on a single rank: the converged T,
 *  the number of distance rounds, the upwind sets (computed from final T and the global m), the
 *  labels and the number of label rounds are all identical on any partition and rank count.
 *
 * **Termination.** T only decreases (min with the old value) and a vertex value can take only
 * finitely many double values, and a round without any change ends the loop; so phase 1 ends. In
 * practice the number of rounds is close to the number of elements along the longest shortest
 * path plus a few rounds of small corrections. In phase 2 every upwind vertex has a strictly
 * smaller T, so the label equation has exactly one solution (by induction in increasing T); labels
 * start above it, only decrease, and reach it after at most "longest upwind chain" rounds.
 * GeodesicParameters::max_rounds is only a safety net.
 *
 * **Where it runs.** The update and label sweeps are mfem::forall kernels (MFEM's parallel loop:
 * GPU kernels when the mfem::Device has a CUDA or HIP backend, host loops otherwise); the rank
 * exchanges and source placement run on the host.
 *
 * **Scope (Stage 1e).** Voronoi (a_i = 0) and Additive (distance - a_i) metrics only; no periodic
 * axes; the caller passes the active material mesh (AttributeDomain::ActiveMesh) so paths stay
 * inside it. A body of the mesh with no seed keeps T = +infinity and label -1, and its elements get
 * element_seed = -1 (the driver decides what to do with them). Elements must have at most 8
 * vertices (all linear 3D MFEM elements).
 */

namespace exaconstit::microstructure {

/**
 * @brief Geodesic labels of the local elements from the FIM eikonal solver (see the file comment).
 *
 * **Collective** over `comm`, which must be the mesh communicator (pmesh.GetComm()). Device kernels
 * (mfem::forall) for the update and label sweeps and the element labeling, host for the exchanges.
 *
 * @param pmesh    conforming 3D ParMesh of the active material (see VertexField); elements with at
 *                 most 8 vertices. A rank may own zero elements.
 * @param comm     the mesh communicator.
 * @param seeds    replicated seeds, N >= 1, finite positions (mesh length units).
 * @param additive true: Additive metric, the source offset is a_i = seeds.weight[i] (empty weight
 *                 array = all zero); false: Voronoi, a_i = 0.
 * @param params   sampling and safety options (same struct as the graph solver).
 * @return element_seed (0-based seed index per local element, -1 if unreached), the converged
 *         VertexField (T and label per vertex), the source report and the statistics (the FIM
 *         round counts are partition independent).
 * @throws std::invalid_argument on every rank for invalid parameters or seeds (replicated inputs);
 *         std::runtime_error on every rank for a mesh that VertexField rejects, a mesh without
 *         elements, an element with more than 8 vertices, or max_rounds exceeded.
 */
GeodesicLabels LabelGeodesicFim(mfem::ParMesh& pmesh,
                                MPI_Comm comm,
                                const SeedSet& seeds,
                                bool additive,
                                const GeodesicParameters& params);

} // namespace exaconstit::microstructure
