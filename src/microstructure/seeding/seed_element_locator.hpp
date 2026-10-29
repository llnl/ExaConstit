#pragma once

#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/microstructure_types.hpp"
#include "microstructure/seeding/material_point_locator.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <mpi.h>
#include <vector>

/**
 * @file seed_element_locator.hpp
 * @brief Finds, for seed points, the mesh element that contains them, the nearest element
 * centroid, or the nearest of any set of points, the same way on every rank count and partition.
 *
 * **Why.** The connected and geodesic distance modes start each grain from one element of the
 * mesh that "belongs" to its seed: the element containing the seed, or for a seed outside the
 * material (in a void of a lattice, or just outside a curved surface) an element close to it.
 * The answer must not depend on how the mesh is split over the MPI ranks, so every choice is
 * named by a partition-independent key and every tie is broken by the smallest key:
 * - element key: a 64-bit hash of the element's corner positions rounded to a fine grid
 *   (mesh/element_key.hpp);
 * - vertex key: the splitmix64 hash of a single point's exact coordinates (VertexKey below). For
 *   mesh vertices or field nodes the coordinates are the owner-broadcast ones
 *   (mesh/vertex_coordinates.hpp), so every copy of a shared point has the same key on every
 *   rank.
 *
 * **How.** Each rank answers for its own local elements or vertices, and the answers are combined
 * with MPI_Allreduce(MIN):
 * - containment: smallest key of a local element containing the point, then MIN over ranks;
 * - nearest: smallest squared distance d^2 on each rank, MIN over ranks, then the smallest key
 *   among the local candidates at exactly that d^2 (other ranks contribute UINT64_MAX), MIN over
 *   ranks. Distances of a given element centroid or point are computed with the same formula from
 *   the same numbers on every rank, so "exactly equal" means the same thing everywhere.
 *
 * The local searches use an Axom BVH (bounding volume hierarchy: a tree of nested bounding boxes
 * that lets us skip elements or vertices that are provably too far away), so each query costs
 * about log(number of local elements) instead of a loop over all of them:
 * - containment: element boxes of MaterialPointLocator, spin::BVH::findPoints, then the exact
 *   point-in-element test MaterialPointLocator::PointInElement;
 * - nearest: zero-size boxes at the centroids (or points), walked with the BVH traverser; a box
 *   is opened only if its distance to the point is <= the best d^2 found so far (`<=` so exact
 *   ties are all visited), and the leaf keeps the lexicographically smallest (d^2, key).
 *
 * **Which helper each mode uses.** The containment search serves both modes (and the geodesic
 * source elements, labeling/vertex_field.hpp). The connected mode anchors a seed outside the
 * material to the nearest element *centroid* (of the elements carrying the seed's label,
 * NearestElementCentroidKeys, labeling/connected_labeler.hpp). The geodesic mode snaps it to the
 * elements around the nearest *field node*: NearestPoints over the nodes that carry its distance
 * field (NearestFieldNodes in labeling/vertex_field.cpp).
 *
 * All functions are host only (MFEM's point-in-element test runs on the host) and collective.
 */

namespace exaconstit::microstructure {

/**
 * @brief Partition-independent key of a mesh vertex: the splitmix64 hash of its three exact
 * coordinate bit patterns, with the same start value and mixing as ElementKey (element_key.hpp)
 * applied to a single vertex. Host only.
 */
std::uint64_t VertexKey(double x, double y, double z);

/**
 * @brief For every seed, the smallest element key of a local element (on any rank) that contains
 * the seed point, or UINT64_MAX if no element does.
 *
 * A point on a face or edge shared by several elements counts as inside all of them (the
 * tolerances of MaterialPointLocator are slightly positive), so the smallest key decides.
 *
 * **Collective** over `comm` (one MPI_Allreduce of N 64-bit words). Host only.
 *
 * @param pmesh       this rank's part of the 3D mesh (may have no local elements).
 * @param comm        communicator of the mesh.
 * @param seeds       replicated seed positions, N >= 0.
 * @param element_key optional size NE: ComputeElementKeys(pmesh); computed here when null.
 * @param context     optional active-element mask and shared locator (MaterialContext): only
 *                    active elements can contain a seed. nullptr = every element, a locator
 *                    built here.
 * @return size N, identical on every rank.
 * @throws std::invalid_argument on every rank if element_key has the wrong size on any rank.
 */
std::vector<std::uint64_t>
ContainingElementKeys(mfem::ParMesh& pmesh,
                      MPI_Comm comm,
                      const SeedSet& seeds,
                      const std::vector<std::uint64_t>* element_key = nullptr,
                      const MaterialContext* context = nullptr);

/** @brief Result of a nearest-element or nearest-vertex search for a list of seeds. */
struct NearestKeys {
    /** @brief Per listed seed: the key of the nearest element (or vertex); UINT64_MAX when the
     * whole mesh has no element (or vertex). */
    std::vector<std::uint64_t> key;
    /** @brief Per listed seed: the squared distance to it (length^2); DBL_MAX when none. */
    std::vector<double> distance_squared;
};

/** @brief NearestKeys plus where the winner lies on this rank (result of NearestPoints). */
struct NearestPointResult : NearestKeys {
    /**
     * @brief Per query: the index, in this rank's point arrays, of this rank's copy of the
     * winning point, or -1 if this rank holds no copy. A point shared by several ranks (same
     * coordinates and key on each) is found on every rank that holds it.
     */
    std::vector<int> local_point;
};

/**
 * @brief For the seeds seed_ids[k], the element whose volume centroid is nearest (smallest
 * squared distance; ties: smallest element key), over all ranks.
 *
 * **Collective** over `comm` (two MPI_Allreduce calls of seed_ids.size() entries). Host only.
 *
 * @param pmesh       this rank's part of the 3D mesh (may have no local elements).
 * @param comm        communicator of the mesh.
 * @param seeds       replicated seed positions.
 * @param seed_ids    0-based indices into `seeds`, the same list on every rank (any order).
 * @param centroids   optional: ComputeElementSamples(pmesh, -1, ...) of this mesh (computed here
 *                    when null); its cx/cy/cz are read with HostRead().
 * @param element_key optional size NE: ComputeElementKeys(pmesh); computed here when null.
 * @param mask        optional active elements: only their centroids are candidates.
 * @return per listed seed (index k), identical on every rank.
 * @throws std::invalid_argument on every rank for a wrong input size or a seed id outside
 *         [0, N) on any rank.
 */
NearestKeys NearestElementCentroidKeys(mfem::ParMesh& pmesh,
                                       MPI_Comm comm,
                                       const SeedSet& seeds,
                                       const std::vector<int>& seed_ids,
                                       const ElementSamples* centroids = nullptr,
                                       const std::vector<std::uint64_t>* element_key = nullptr,
                                       const ElementMask* mask = nullptr);

/**
 * @brief For each query point, the nearest of the points held by all ranks (smallest squared
 * distance; ties: smallest key), and on every rank that holds a copy of it, that copy's index.
 *
 * Each rank passes its own points; a point held by several ranks must have the same coordinates
 * and key on each of them (for example owner-broadcast node coordinates with VertexKey). A rank
 * holds a copy of the winner exactly when its own local best has the global (d^2, key): its
 * copies carry the same d^2 and key, and nothing on the rank can beat the global minimum.
 *
 * **Collective** over `comm` (two MPI_Allreduce calls of the number of queries). Host only.
 *
 * @param comm       communicator over which the points are spread.
 * @param px, py, pz this rank's points (mesh length units; may be empty).
 * @param keys       key of each point (same size as px).
 * @param qx, qy, qz the queries, the same list on every rank.
 * @return per query: key and squared distance of the winner (identical on every rank) and the
 *         local index of this rank's copy (rank local).
 */
NearestPointResult NearestPoints(MPI_Comm comm,
                                 const std::vector<double>& px,
                                 const std::vector<double>& py,
                                 const std::vector<double>& pz,
                                 const std::vector<std::uint64_t>& keys,
                                 const std::vector<double>& qx,
                                 const std::vector<double>& qy,
                                 const std::vector<double>& qz);

} // namespace exaconstit::microstructure
