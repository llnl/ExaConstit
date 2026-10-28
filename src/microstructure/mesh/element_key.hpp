#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <vector>

/**
 * @file element_key.hpp
 * @brief A 64-bit identity of a mesh element that is the same on every rank count and every
 * mesh partition, also for curved meshes and meshes refined in parallel.
 *
 * **Why it is needed.** Several steps of the generator must break ties or order elements in a way
 * that does not depend on how the mesh was split over the MPI ranks: element-weighted seed
 * sampling (each element draws its random points from a stream named by its key), and fragment
 * cleanup, where a disconnected piece of a grain (a *fragment*) is named by its smallest element
 * key. Local element numbers (0..NE_local-1) and ParMesh global numbers (rank 0's elements first,
 * then rank 1's, ...) both change with the partition, so neither can be used.
 *
 * **What the key is.** A hash (a fixed scrambling of bits into a number) of the element's corner
 * positions, rounded to a grid (about 2^-20 of the element size):
 *  1. Corner positions: the Nodes values at the element's vertex dofs when the mesh has
 *     continuous (H1) Nodes, else the vertex array (VertexPoint, mesh/straight_sided.hpp). The
 *     Nodes are the true geometry; MFEM's vertex array of a ParMesh with Nodes is an average that
 *     depends on the partition (mesh/vertex_coordinates.hpp), so it is never used when Nodes
 *     exist.
 *  2. Grid spacing (the *quantum*) q: a power of two near 2^-20 times a length scale s of the
 *     element; precisely q = 2^(floor(log2(c s)) - 20) with c = 0.618... (see
 *     ELEMENT_KEY_SIZE_FACTOR) and s = max(h, min(M, 1024 h)), where h is the element's size (the
 *     largest edge of the box around its corners) and M the largest |corner coordinate|. The
 *     magnitude counts because the round-off between two copies of a coordinate grows with |x|,
 *     not with h; the cap of 1024 h keeps q below about 6e-4 h however far the element is from
 *     the origin. For a unit element near the origin q is about 5e-7.
 *  3. Each coordinate is rounded to the nearest multiple of q (an exact operation for a power of
 *     two q), so two copies of a corner that differ only in the last bits (as after a parallel
 *     refinement, where the copies are interpolated by different elements in an order that
 *     depends on the partition, or between a serial and a parallel run) round to the same value.
 *  4. The rounded corners are sorted lexicographically (by x, then y, then z), so the local vertex
 *     numbering of the element does not matter, and their IEEE-754 bit patterns are folded one
 *     after another through the splitmix64 mixing function.
 * Computed quantities such as a centroid are deliberately not used.
 *
 * **Who computes a key.** Each element's key is computed only by the rank that owns the element,
 * from that rank's own data; ghost elements receive their owner's key through the adjacency
 * exchange (MeshAdjacency::ExchangeKeys). So within one run every element has exactly one key
 * and the function needs no communication (it is not collective).
 *
 * **Uniqueness.** In a conforming mesh no two elements have the same vertex set. Rounding moves a
 * corner by at most q/2 <= 3e-4 h, so two distinct corners of an element stay distinct unless
 * they are closer than q, i.e. unless the element is a sliver whose shortest edge is below about
 * 6e-4 of its box size (aspect ratio above about 1600). Two distinct elements therefore get the
 * same key only through a 64-bit hash collision, with probability about NE^2 / 2^65 (about 3e-10
 * for NE = 10^5 elements), or when they are duplicated or degenerate (all corners within q of
 * each other). All corners are used, not only the smallest one: the 6 tets of a Kuhn-split hex
 * share their smallest vertex.
 *
 * **Residual risk (stated honestly).** Exact geometric keys are impossible after MFEM's parallel
 * UniformRefinement: the copies of a shared node are computed by different elements, in an order
 * that depends on the partition, and differ by a few units in the last place (ulp) of |x|. This
 * holds for meshes with Nodes (the refined Nodes are interpolated per element) and for hex meshes
 * without Nodes too (a new face-center vertex is an average of four vertices whose summation order
 * differs between the ranks of the face; mesh/vertex_coordinates.hpp). Rounding only helps when
 * both copies of a coordinate fall on the same side of a rounding boundary (a half-multiple of
 * q). With a copy difference delta of 1e-15 to 5e-15 max(|x|, h) (the measured range, about 5 to
 * 20 ulp) and q >= 2.9e-7 s, the probability that one such coordinate rounds differently is
 * about delta / q:
 *  - |x| up to 1000 h (s grows with |x|): about 3.4e-9 to 1.7e-8 per coordinate;
 *  - |x| = 1e4 h (s capped at 1024 h): about 3.4e-8 to 1.7e-7 per coordinate;
 *  - |x| = 1e6 h: about 3.4e-6 to 1.7e-5 per coordinate.
 * A hex has 24 corner coordinates, so per million elements whose corner copies all differ this
 * is about 0.08 to 0.4 elements with a different key (|x| <= 1000 h), 0.8 to 4 elements
 * (|x| = 1e4 h) or 80 to 400 elements (|x| = 1e6 h). Coordinates whose copies are bitwise equal
 * (any mesh distributed from a serial mesh and not refined in parallel) never mismatch. The
 * quantum itself changes only if c s lies within round-off of a power of two; c is irrational so
 * that common sizes (powers of two, 1/3, 1/10, ...) are far from such a boundary. A mismatched
 * element gets a different random stream in element-weighted seeding (its points change), or a
 * different winner of an exact tie (see below); results stay valid, they are just not identical
 * between the two runs. Coordinates whose magnitude exceeds 2^53 q are not rounded (a double
 * cannot represent the grid there); their exact bits are used.
 *
 * **What depends on keys.** Only element-weighted seed placement (the element's random stream and
 * the grid its sampling box is rounded to, ElementKeyQuantum) and naming or exact-tie breaking:
 * fragment cleanup names a fragment by its smallest key; the driver's seed anchors; connected
 * mode names components and anchors by keys (labeling/connected_labeler.cpp); the geodesic modes
 * give a seed the containing element with the smallest key (labeling/vertex_field.cpp).
 * Rejection sampling, the default placement, does not use keys at all.
 *
 * **Change from the earlier key (2026-10-22).** The earlier key hashed the exact vertex-array
 * bits; a first rounded key used a quantum of 2^-40 h, which proved too fine for coordinates far
 * from the origin and was coarsened to the one above. Keys are therefore different numbers now:
 * element-weighted seed streams (and so element-weighted seeds) and the smallest-key tie-breaks
 * of fragment cleanup changed once for every mesh. They are again the same for every rank count,
 * and now also for meshes with Nodes whose vertex array is stale or partition dependent.
 *
 * Failure modes: a collision (two elements with the same key) is not detected. It can only happen
 * by a hash collision (probability above) or for duplicated or degenerate elements (corners
 * closer than q). Users that name groups of elements by their smallest key (fragment cleanup)
 * would then treat two such groups of the same grain as one; see labeling/fragment_cleanup.hpp.
 */

namespace exaconstit::microstructure {

/** @brief The quantum is 2^-ELEMENT_KEY_QUANTUM_BITS times the element's length scale. */
constexpr int ELEMENT_KEY_QUANTUM_BITS = 20;

/**
 * @brief Cap on the coordinate magnitude in the length scale, in element sizes:
 * s = max(h, min(M, ELEMENT_KEY_MAX_MAGNITUDE_RATIO h)). Keeps q <= about 6e-4 h.
 */
constexpr double ELEMENT_KEY_MAX_MAGNITUDE_RATIO = 1024.0;

/**
 * @brief Factor c applied to the element size before taking its power of two (the golden-ratio
 * conjugate (sqrt(5) - 1) / 2). Irrational, so the quantum does not jump for typical element
 * sizes such as 2^-k or 1/n.
 */
constexpr double ELEMENT_KEY_SIZE_FACTOR = 0.6180339887498949;

/**
 * @brief Partition-independent 64-bit key of element `e` (see the file comment).
 *
 * Host only, not collective.
 *
 * @param mesh the mesh holding the element (a serial mfem::Mesh or this rank's ParMesh). Corner
 *             coordinates come from the continuous Nodes (vertex dofs) when present, else from
 *             Mesh::GetVertex; spaceDim <= 3 (missing coordinates count as 0).
 * @param e    0-based local element index, 0 <= e < mesh.GetNE().
 * @return the key; equal for the same element on every rank and partition (up to the residual
 *         risk in the file comment).
 */
std::uint64_t ElementKey(const mfem::Mesh& mesh, int e);

/**
 * @brief The quantum q of element `e` (file comment, step 2), from this rank's own corner
 * positions; 0 for a degenerate element (zero or non-finite length scale), meaning "no rounding".
 *
 * Host only, not collective. Used by ElementKey and by element-weighted seeding, which rounds the
 * element's sampling box to this grid so its sampled points are the same on every rank count.
 *
 * @param mesh the mesh holding the element (serial, or this rank's partition).
 * @param e    0-based local element index.
 * @return q, an exact power of two (same length unit as the mesh coordinates), or 0.
 */
double ElementKeyQuantum(const mfem::Mesh& mesh, int e);

/**
 * @brief x rounded to the nearest multiple of q (ties to even); q <= 0 returns x. -0.0 becomes
 * +0.0, so both signs of zero give the same bits. Host only.
 *
 * @param x a coordinate (mesh length unit).
 * @param q a power-of-two quantum from ElementKeyQuantum, or 0.
 */
double RoundToKeyQuantum(double x, double q);

/**
 * @brief Widens [lo, hi] to whole multiples of q that still contain it: lo becomes
 * (nearbyint(lo / q) - 1) q and hi becomes (nearbyint(hi / q) + 1) q. Rounding to the nearest
 * multiple first means two copies of an endpoint that differ in the last bits give the same
 * result unless they straddle a half-multiple of q (the same residual risk as the key). q <= 0,
 * or a result that is not finite, leaves both unchanged. Host only.
 *
 * @param q  a power-of-two quantum from ElementKeyQuantum, or 0.
 * @param lo lower end of the interval (in/out).
 * @param hi upper end of the interval (in/out), hi >= lo.
 */
void RoundIntervalOutward(double q, double& lo, double& hi);

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
