#pragma once

#include "mfem.hpp"

#include <mpi.h>
#include <vector>

#include "microstructure/labeling/geodesic_common.hpp"
#include "microstructure/labeling/geodesic_limits.hpp"
#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/microstructure_types.hpp"

/**
 * @file geodesic_weighted.hpp
 * @brief Geodesic grain labels for metrics whose value is not a path length plus a constant:
 * Laguerre, f_i = d_i^2 - w_i, Quadric with weights (d_i measured with the tensor,
 * GeodesicParameters::stretch) and Quadric with one tensor per seed (seed_stretch). d_i is the
 * length of the shortest path from seed i inside the material (the eikonal distance of
 * labeling/geodesic_fim.hpp, seed by seed).
 *
 * **Why one front is not enough.** The single-front solver keeps per node only the first arrival
 * and its seed, which is right when every seed's value is its path length plus a constant
 * (Voronoi, Additive). With f_i = d_i^2 - w_i a heavy seed far away can own a point that a light
 * seed's front reaches first, so a node has to remember several seeds.
 *
 * **Seed lists.** Every node keeps up to K = GeodesicParameters::max_seeds_per_node entries
 * (seed i, T_i), T_i approximating d_i at the node. One Jacobi round (all nodes from the previous
 * round's lists):
 *  1. for every element of a node v with a node that changed in the previous round, and every
 *     seed i in the lists of the element's other corners (of each cell around v): the FIM local
 *     update of v from those corners' T_i (a corner without i counts as unreached), through the
 *     same per-cell scan as the single-front solver (ScanCell, labeling/fim_scan.hpp);
 *  2. the node's new list: its old entries and the new values merged (per seed the smaller T, a
 *     smaller T taken only when it improves by more than the stopping tolerance, as in the
 *     single-front solver), then pruned and cut to K (below);
 *  3. every shared node's lists are exchanged (VertexField::ExchangeSharedPayload) and each rank
 *     merges all copies with the same rule.
 * The rounds stop when no list changes on any rank.
 *
 * **Pruning (exact).** Seed j dominates seed i at v when T_j <= T_i and f_j <= f_i (one of them
 * strict). Then j beats i at every point reached from seed i through v: with L the remaining path
 * length, (T_j + L)^2 - w_j - (T_i + L)^2 + w_i = (f_j - f_i) + 2 L (T_j - T_i) <= 0. So i is
 * dropped from v's list. Along the path to any point it still owns a seed is never dominated, so
 * the values it needs upstream are kept. With equal weights only one seed survives per node and
 * the solver reduces to the single front. Removing dominated entries does not depend on the order
 * in which entries are merged, so the result of a merge is a function of the set of entries.
 *
 * **Capacity.** If more than K entries remain, the K with the smallest (f, seed) are kept and the
 * node counts as overflowing (GeodesicReport::overflow_nodes). The cut can drop a seed that would
 * win further on, and since ranks cut their partial lists before the exchange, a cut can also
 * make the result depend on the partition. With no overflow the labels and distances are exact
 * (for this discretization) and bitwise partition independent, as for the single-front solver.
 *
 * **Per-seed tensors** (anisotropy kind "per_seed", GeodesicParameters::seed_stretch). Seed i
 * measures paths with its own tensor A_i = S_i^T S_i: its local updates map the cell corners with
 * S_i, and its source values are |S_i (v - s_i)|. The pruning rule above needs the remaining path
 * to be no longer for j than for i, so j may drop i only where in addition A_j <= A_i (A_i - A_j
 * positive semidefinite, checked conservatively; always true for identical tensors). Seeds with
 * really different shapes are rarely ordered that way and stay in the lists until the capacity
 * cut, so overflows are expected when more than max_seeds_per_node seeds compete; with at most
 * max_seeds_per_node seeds in total no list can overflow and the result is exact.
 *
 * **Labels.** label(v) = the entry with the smallest (f, seed); element labels from the lists
 * (VertexField::SetSeedLists: d_i through the corners of the point's cell, or directly from a seed
 * in that cell). The VertexField's T and label hold the winner's T_i and seed for output.
 *
 * **Cost.** Per element scan, one local-update scan per seed of the corners' lists (usually one
 * or two: dominance leaves the winner plus a few neighbors near boundaries); memory 12 K bytes per
 * node for each of the two list copies.
 *
 * **Where it runs.** The sweeps are mfem::forall kernels; sources, exchanges and merges of shared
 * nodes run on the host.
 */

namespace exaconstit::microstructure {

/**
 * @brief Weighted geodesic labels (see the file comment).
 *
 * **Collective** over `comm` (the mesh communicator).
 * @param pmesh   conforming 3D ParMesh of the active material (elements with at most 8 vertices).
 * @param comm    the mesh communicator.
 * @param seeds   replicated seeds, N >= 1.
 * @param weights w_i per seed (length^2 in the metric), size N; empty = all zero.
 * @param params  sampling, rounds, tolerance, metric stretch and max_seeds_per_node.
 * @return element_seed, the VertexField (winner's T and seed per node, plus the seed lists used
 *         for the element labels), the source report and the statistics (distance_rounds; no
 *         label rounds; overflow_nodes and max_list_length).
 * @throws std::invalid_argument on every rank for invalid parameters or weights;
 *         std::runtime_error as LabelGeodesicFim.
 */
GeodesicLabels LabelGeodesicWeighted(mfem::ParMesh& pmesh,
                                     MPI_Comm comm,
                                     const SeedSet& seeds,
                                     const std::vector<double>& weights,
                                     const GeodesicParameters& params);

} // namespace exaconstit::microstructure
