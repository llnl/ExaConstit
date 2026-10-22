#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <vector>

/**
 * @file element_key.hpp
 * @brief A 64-bit identity of a mesh element that is the same on every rank count and every
 * mesh partition.
 *
 * **Why it is needed.** Several steps of the generator must break ties or order elements in a way
 * that does not depend on how the mesh was split over the MPI ranks: element-weighted seed
 * sampling, and fragment cleanup, where a disconnected piece of a grain (a *fragment*) is named
 * by its smallest element key. Local element numbers (0..NE_local-1) and ParMesh global numbers
 * (rank 0's elements first, then rank 1's, ...) both change with the partition, so neither can be
 * used.
 *
 * **What the key is.** A hash (a fixed scrambling of bits into a number) of the element's exact
 * vertex coordinates: the coordinates are sorted lexicographically (by x, then y, then z), so the
 * local vertex numbering of the element does not matter, and their raw IEEE-754 bit patterns are
 * folded one after another through the splitmix64 mixing function. Vertex coordinates are copied
 * bit for bit from the serial mesh into every partition, so every rank that holds the element
 * (as an owned element or as a face neighbor) computes the same key. Computed quantities such as
 * a centroid are deliberately not used: CPU and GPU round-off can differ in the last bit.
 *
 * **Uniqueness.** In a conforming mesh no two elements have the same vertex set, so two distinct
 * elements get the same key only through a 64-bit hash collision, with probability about
 * NE^2 / 2^65 (about 3e-10 for NE = 10^5 elements). All vertices are used, not only the smallest
 * one: the 6 tets of a Kuhn-split hex share their smallest vertex.
 *
 * **Failure modes.** A collision (two elements with the same key) is not detected. It can only
 * happen by a hash collision (probability above) or when two elements have exactly the same
 * vertex coordinates (a duplicated element in a broken mesh). Users that name groups of elements
 * by their smallest key (fragment cleanup) would then treat two such groups of the same grain as
 * one; see labeling/fragment_cleanup.hpp.
 *
 * **Limits.** The key depends on the exact vertex bits. Two meshes whose vertices differ in the
 * last bit (for example a mesh refined in parallel, where a new vertex can be an average summed
 * in a different order) give different keys. Curved elements are keyed by their vertices only
 * (the high-order Nodes are not used), which still identifies the element.
 *
 * The algorithm is bit-identical to the element key that seeding/seed_generators.cpp uses for
 * element-weighted sampling, so that code can switch to this shared version without changing any
 * seed.
 */

namespace exaconstit::microstructure {

/**
 * @brief Partition-independent 64-bit key of element `e` (see the file comment).
 *
 * Host only, not collective.
 *
 * @param mesh the mesh holding the element (a serial mfem::Mesh or this rank's ParMesh). Vertex
 *             coordinates are read with Mesh::GetVertex; spaceDim must be 3 (2D meshes are hashed
 *             with their third coordinate, which MFEM stores as 0).
 * @param e    0-based local element index, 0 <= e < mesh.GetNE().
 * @return the key; equal for the same element on every rank and partition.
 */
std::uint64_t ElementKey(const mfem::Mesh& mesh, int e);

/**
 * @brief Keys of all local elements: keys[e] = ElementKey(mesh, e) for e = 0..NE-1.
 *
 * Host only, not collective.
 *
 * @param mesh the mesh (serial, or this rank's partition).
 * @return a vector of size mesh.GetNE(), indexed by 0-based local element id.
 */
std::vector<std::uint64_t> ComputeElementKeys(const mfem::Mesh& mesh);

} // namespace exaconstit::microstructure
