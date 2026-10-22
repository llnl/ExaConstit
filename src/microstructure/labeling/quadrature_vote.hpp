#pragma once

#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <array>
#include <mpi.h>
#include <vector>

/**
 * @file quadrature_vote.hpp
 * @brief Labels each element by a volume-weighted majority vote of points inside it, instead of
 * by its centroid alone.
 *
 * **Why vote.** The centroid labeler (tessellation_labeler.hpp) gives an element to the grain
 * that owns the element's centroid. When a grain boundary cuts an element, the centroid may lie
 * on the side that holds the *smaller* part of the element's volume: for a tetrahedron cut by a
 * plane parallel to a face, the centroid side can hold as little as 27/64 = 42% of the volume,
 * and for a curved boundary (Additive, per-seed Quadric) the same happens on hexes. The vote
 * instead estimates how much of the element lies in each grain and gives the element to the
 * grain with the largest share, which is the label that minimizes the misassigned volume.
 *
 * **How.** Each element gets a set of voting points with weights w_q detJ_q (a quadrature rule:
 * the weights of an element sum to its volume). Every point is labeled with its own nearest seed
 * (the exact search SeedIndex::FindBest); each label collects the weights of the points it won;
 * the label with the largest total wins. The total of a label is then a quadrature estimate of
 * the volume of the element that lies in that grain.
 *
 * **Which points (subcell centers).** The share estimate is only as good as the point set. We
 * split the reference element `levels` times into 8 children of equal volume (the same pattern as
 * uniform refinement: a hex into 2 x 2 x 2 sub-hexes, a tet into 4 corner tets and 4 tets of the
 * middle octahedron) and use the center of each child, weighted by the child's volume (times
 * detJ at the center). That is the composite midpoint rule: 8^levels points (64 for the default
 * levels = 2), all with equal reference weights, spread evenly over the element, so a grain that
 * covers a fraction s of the element wins about s of the weight, to within the fraction of
 * children cut by the boundary.
 * MFEM's standard quadrature rules (mfem::IntRules) are deliberately not used: they are built to
 * integrate smooth polynomials, not to measure volume fractions of a discontinuous grain
 * indicator. (Before MFEM v4.10 several tetrahedron rules even had a negative weight, which
 * would make a point vote *against* the grain that wins it; ExaConstit now requires v4.10, whose
 * simplex rules are all positive.) With positive tet rules (4 to 24 points, very unequal weights)
 * the vote was measured to assign volumes *worse* than the centroid on a coarse tet mesh; the
 * 64-point subcell rule, with equal weights, does better (test_sampling_modes,
 * VolumesCloserToAnalyticThanCentroid).
 *
 * **Ties.** The label with the larger total weight wins; when two totals are exactly equal (a
 * symmetric cut), the lowest seed index wins. The totals are summed in increasing point order
 * inside each element, which depends only on the element, so the result is identical for every
 * partition and rank count.
 *
 * **Memory.** Points are evaluated and searched in chunks of whole elements
 * (QuadratureVoteParameters::max_points_per_chunk), so the per-point data - coordinates and
 * weight (32 bytes) and the search result (24 bytes) - never exceeds about 56 bytes per point of
 * one chunk (default 2^21 points: about 117 MB). What is kept for the whole local mesh does not
 * depend on `levels`: the element-wise node coordinates of the shared sampler (3 x nodes per
 * element doubles: 192 bytes per linear hex, 96 per linear tet; mesh/mesh_sampler.hpp,
 * QuadraturePointSampler) plus two ints per element (the result and a split flag). So at level 3
 * (512 points) a rank with 1M linear hexes needs about 0.2 GB for the node data plus one chunk,
 * not the 16 GB that evaluating all points at once would take.
 *
 * **Periodic images.** With periodic seed images (geometry/periodic_images.hpp) the points of one
 * element can be won by a seed and by an image of the same seed; both must count for the same
 * grain. Pass the image-to-original map as `seed_owner`: votes are then collected per original
 * seed and the output holds original seed indices.
 *
 * **Where it runs.** Point coordinates and detJ come from the shared QuadraturePointSampler
 * (shape-function interpolation of the element nodes, an mfem::forall kernel per chunk) and the
 * vote is an mfem::forall kernel, both on the backend of the
 * configured mfem::Device; the seed search runs on the matching Axom execution space
 * (DispatchAxomExec, exec_dispatch.hpp).
 */

namespace exaconstit::microstructure {

/**
 * @brief Parameters of the vote (plain struct; the integration code fills it from the user
 * options).
 */
struct QuadratureVoteParameters {
    /** @brief Metric family of the tessellation. */
    MetricType metric = MetricType::VORONOI;
    /**
     * @brief Global Quadric tensor [Axx, Ayy, Azz, Ayz, Axz, Axy] (ignored when seeds.shape is
     * filled: per-seed tensors win), as in TessellationOptions::global_tensor.
     */
    std::array<double, 6> global_tensor = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    /** @brief Parameter bins of the seed search (>= 1; performance only). */
    int num_bins = 8;
    /**
     * @brief Subdivision levels of the voting points, 0..3: 8^levels points per element (1, 8,
     * 64, 512). Level 0 is the single reference-element center (centroid labeling for affine
     * elements). Each level halves the width of the band around a grain boundary in which the
     * share estimate can be wrong. Memory per element kept for the whole mesh does not grow
     * with the level (see "Memory" in the file comment); the work per element grows as 8^levels,
     * which is why 3 (512 points) is the maximum offered. Level 1 is not recommended: its 8
     * points give share steps of 1/8, coarser than the centroid's own error window, and it
     * measured worse than centroid labeling.
     */
    int levels = 2;
    /**
     * @brief Upper bound on the voting points processed at once (>= 1). Chunks always hold
     * whole elements, at least one. The default gives one chunk for up to ~2 million points per
     * rank.
     */
    int max_points_per_chunk = 1 << 21;
};

/**
 * @brief Statistics of one vote, for logging and tests. Local to the rank (not reduced).
 */
struct QuadratureVoteReport {
    /** @brief Voting points per element (0 on a rank without elements). */
    int num_points_per_element = 0;
    /** @brief Number of element chunks processed. */
    int num_chunks = 0;
    /**
     * @brief Local elements whose points did not all vote for the same seed (after the
     * seed_owner map), i.e. elements cut by a grain boundary.
     */
    int num_split_elements = 0;
    /**
     * @brief Seeds that share their exact position with a lower-index seed (from the seed
     * search; identical on every rank).
     */
    int num_coincident_seeds = 0;
};

/**
 * @brief The voting rule on the reference element: subcell centers after `levels` splits into 8
 * equal-volume children, each with weight (reference volume) / 8^levels.
 *
 * Reference elements are MFEM's: the unit cube [0,1]^3 (volume 1) and the tet with vertices
 * (0,0,0), (1,0,0), (0,1,0), (0,0,1) (volume 1/6). The hex rule is a tensor product of the 1D
 * composite midpoint rule (points (i + 1/2) / 2^levels, x fastest), which is the point order
 * MFEM's tensor-product evaluation expects (fe_base.cpp, GetTensorDofToQuad).
 *
 * The rules are built once per (geometry, levels) and kept in static storage for the lifetime of
 * the program: MFEM caches interpolation data per rule *address* (FiniteElementSpace::
 * GetQuadratureInterpolator, FiniteElement::GetDofToQuad), so a rule must never be freed and its
 * address reused while those caches live. Host only; not thread-safe on first construction.
 *
 * @param geometry mfem::Geometry::CUBE or mfem::Geometry::TETRAHEDRON.
 * @param levels   0..3.
 * @throws std::invalid_argument for another geometry or levels outside 0..3.
 */
const mfem::IntegrationRule& VoteRule(mfem::Geometry::Type geometry, int levels);

/**
 * @brief Physical coordinates and weights w_q detJ_q of the voting points (VoteRule) of the local
 * element range [elem_begin, elem_end).
 *
 * Same output layout as ComputeQuadraturePointSamples (mesh_sampler.hpp): entry
 * (e - elem_begin) * num_qp + q holds point q of local element e; the weights of an element sum
 * to its volume (exactly, up to round-off, for affine elements). Computed by the shared sampler
 * ComputeQuadraturePointSamples(mesh, VoteRule(...), ...) of mesh/mesh_sampler.hpp, which
 * evaluates the requested range only (the mesh is not modified, see the cache caveat there).
 *
 * Not collective. Host function launching device kernels.
 *
 * @param mesh       3D mesh with one local element geometry (CUBE or TETRAHEDRON). Non-const
 *                   because MFEM's FE space constructor needs it; not modified.
 * @param levels     subdivision levels, 0..3.
 * @param elem_begin first 0-based local element.
 * @param elem_end   one past the last; 0 <= elem_begin <= elem_end <= NE_local.
 * @param out        overwritten (num_qp = 8^levels, or 0 on a rank without elements).
 * @throws std::invalid_argument for an unsupported geometry or levels; std::runtime_error for an
 *         invalid range, a non-3D or mixed-geometry mesh, or more points than fit in an int.
 */
void ComputeVotePoints(mfem::Mesh& mesh,
                       int levels,
                       int elem_begin,
                       int elem_end,
                       QuadraturePointSamples& out);

/**
 * @brief For every local element, the seed that wins the volume-weighted vote of the element's
 * voting points (see the file comment). Same output shape as LabelElementsWithSeeds.
 *
 * Not collective (each rank labels its own elements). Host function launching device kernels.
 *
 * @param local_mesh   this rank's elements (3D, one element geometry, hex or tet; a ParMesh
 *                     binds here). Non-const only for the point sampler; not modified.
 * @param seeds        replicated seed set searched at every voting point, N >= 1 (with periodic
 *                     images: PeriodicImages::seeds).
 * @param params       metric, tensor, bins, levels, chunk size.
 * @param element_seed resized to the local element count. element_seed[e] = the winning seed
 *                     index, mapped through seed_owner when given (so an index in
 *                     [0, number of owners)), or -1 if no voting point of the element has a
 *                     finite position. Written where the mfem::Device runs kernels (HostRead()
 *                     for the host).
 * @param seed_owner   optional, size N: seed_owner[i] = the seed whose grain seed i stands for
 *                     (PeriodicImages::original); votes are collected per owner and ties go to
 *                     the lowest owner index. Empty = every seed is its own owner.
 * @param report       optional statistics.
 * @throws std::invalid_argument for an empty seed set, num_bins < 1, max_points_per_chunk < 1,
 *         levels outside 0..3, an element geometry other than hex/tet, a seed_owner of the wrong
 *         size or with negative entries, or invalid seed data (see SeedIndex::Build);
 *         std::runtime_error for a mixed-geometry or non-3D mesh.
 */
void LabelElementsByQuadratureVote(mfem::Mesh& local_mesh,
                                   const SeedSet& seeds,
                                   const QuadratureVoteParameters& params,
                                   mfem::Array<int>& element_seed,
                                   const std::vector<int>& seed_owner = {},
                                   QuadratureVoteReport* report = nullptr);

/**
 * @brief The vote followed by compaction to grain ids 1..N (CompactSeedLabels), i.e. the
 * vote counterpart of LabelTessellation.
 *
 * **Collective** over `comm`. Elements without a finite voting point (seed -1) are detected
 * collectively before the compaction: all ranks throw std::runtime_error naming the lowest such
 * rank and its first such local element, as LabelTessellation does.
 *
 * Periodic use:
 * @code
 *   PeriodicImages images = BuildPeriodicImages(seeds, box, PeriodicImageParameters{});
 *   TessellationLabels labels = LabelTessellationByQuadratureVote(
 *       mesh, comm, images.seeds, params, images.original, images.num_original);
 *   // labels.element_seed and labels.grain_seed hold original seed indices
 * @endcode
 *
 * @param local_mesh      this rank's part of the mesh.
 * @param comm            communicator over which the mesh is distributed.
 * @param seeds           seed set searched at the voting points (N >= 1, N < 2^31).
 * @param params          vote parameters.
 * @param seed_owner      optional owner map (see LabelElementsByQuadratureVote).
 * @param num_owner_seeds number of owner seeds, i.e. the size of the seed_grain table (with
 *                        periodic images: PeriodicImages::num_original). Negative means N when
 *                        seed_owner is empty, else 1 + the largest owner index.
 * @return element_seed (owner indices), element_grain, num_grains, seed_grain (size
 *         num_owner_seeds), grain_seed and num_coincident_seeds.
 * @throws as LabelElementsByQuadratureVote and CompactSeedLabels. Parameter errors are identical
 *         on every rank (same parameters everywhere), and the later checks are reduced, so all
 *         ranks throw together.
 */
TessellationLabels LabelTessellationByQuadratureVote(mfem::Mesh& local_mesh,
                                                     MPI_Comm comm,
                                                     const SeedSet& seeds,
                                                     const QuadratureVoteParameters& params,
                                                     const std::vector<int>& seed_owner = {},
                                                     int num_owner_seeds = -1);

} // namespace exaconstit::microstructure
