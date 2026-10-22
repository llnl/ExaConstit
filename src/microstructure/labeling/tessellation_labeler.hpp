#pragma once

#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file tessellation_labeler.hpp
 * @brief Gives every element of the mesh to a grain: the grain whose seed is "closest" to the
 * element centroid under the tessellation metric (Voronoi, Laguerre, Quadric or Additive), then
 * renumbers the grains that actually received elements as compact ids 1..N.
 *
 * **Two steps.**
 * 1. *Nearest seed per element* (LabelElementsWithSeeds, not collective). The volume centroid
 *    of every local element is computed from the element geometry (ComputeElementSamples,
 *    mesh/mesh_sampler.hpp: sum over quadrature points of w detJ x divided by the element
 *    volume), and the seed with the smallest metric value f_i(centroid) is found with the exact
 *    seed search (SeedIndex::FindBest, geometry/seed_index.hpp). Ties go to the lowest seed
 *    index. The answer for an element depends only on its own geometry and on the replicated
 *    seed set, so it is identical for every mesh partition and rank count.
 * 2. *Compaction to grain ids* (CompactSeedLabels, collective). Some seeds may win no element at
 *    all (a seed in a corner of a coarse mesh, a Voronoi seed sharing its position with a
 *    lower-index seed, an empty Laguerre cell). ExaConstit needs grain ids 1..N without gaps (one
 *    orientation row per grain), so every rank marks the seeds its elements use in a bitmask
 *    (an array of 64-bit words, one bit per seed), the masks are combined over all ranks with a
 *    bitwise OR (MPI_Allreduce with MPI_BOR), and the used seeds are numbered 1, 2, 3, ... in
 *    increasing seed index. Every rank sees the same combined mask, so every rank computes the
 *    same seed-to-grain table, again independent of the rank count.
 *
 * **Other sampling modes and periodicity.** LabelTessellation (the complete labeling) can
 * replace step 1 by a volume-weighted vote of points inside each element (sampling =
 * "quadrature_vote", labeling/quadrature_vote.hpp), and for periodic axes searches periodic seed
 * images (geometry/periodic_images.hpp) whose labels are mapped back to their original seeds
 * before step 2.
 *
 * Grain pieces that are cut off from the rest of their grain (possible across holes or notches
 * of the material, or for some metrics) are not split or merged here; that is a separate
 * cleanup step (labeling/fragment_cleanup.hpp).
 *
 * **Where it runs.** The element sampler and the id mapping are mfem::forall kernels, so they
 * run on the backend of the configured mfem::Device. The seed search runs on the matching Axom
 * execution space (DispatchAxomExec, exec_dispatch.hpp). The bitmask and the seed-to-grain table
 * are small host arrays.
 */

namespace exaconstit::microstructure {

/**
 * @brief Refuses tessellation options this labeler does not implement: distance other than
 * EUCLIDEAN (connected and geodesic distance are a later stage), vote_levels outside 0..3 and an
 * unknown anisotropy kind. Centroid and quadrature-vote sampling and periodic axes are supported
 * (by LabelTessellation).
 *
 * Host only, not collective; every rank holds the same options, so every rank throws together.
 * The driver calls it before seeding, so unsupported options are refused before any collective
 * work.
 *
 * @throws std::invalid_argument naming the unsupported option.
 */
void CheckSupportedTessellation(const TessellationOptions& tessellation);

/**
 * @brief The global Quadric tensor the labeler uses, [Axx, Ayy, Azz, Ayz, Axz, Axy].
 *
 * - anisotropy_kind "none" or "tensor": tessellation.global_tensor as stored (the identity by
 *   default, which reproduces Voronoi distances);
 * - "columnar": MakeColumnarTensor(anisotropy_axis) = I - a a^T (geometry/anisotropy_tensors.hpp):
 *   grains unbounded along the unit axis a;
 * - "lamellar": MakeLamellarTensor(anisotropy_normal) = n n^T: layers perpendicular to n.
 *
 * Only the Quadric metric reads the tensor. Host only, not collective.
 *
 * @throws std::invalid_argument for an unknown kind or a zero / non-finite axis or normal.
 */
std::array<double, 6> EffectiveGlobalTensor(const TessellationOptions& tessellation);

/**
 * @brief Collective check that every local element received a seed (no entry is -1).
 *
 * If any rank has an element with seed index < 0, MPI_Allreduce(MIN) finds the lowest such rank
 * (LowestRankWith, util/collective_errors.hpp), which broadcasts the local id of its first such
 * element, and every rank throws the same std::runtime_error:
 * "<what> (local element e on rank r)<hint>".
 *
 * **Collective** over `comm`. Host only.
 *
 * @param comm         communicator over which the elements are distributed.
 * @param element_seed this rank's per-element seed indices (HostRead).
 * @param what         start of the message (names the caller and the problem).
 * @param hint         end of the message (likely cause), may be empty.
 */
void ThrowIfAnyElementUnlabeled(MPI_Comm comm,
                                const mfem::Array<int>& element_seed,
                                const std::string& what,
                                const std::string& hint = "");

/**
 * @brief Element labels of one labeling run.
 *
 * Per-element arrays describe this rank's local elements (0-based local element index); per-seed
 * and per-grain arrays are host data replicated (identical) on every rank.
 */
struct TessellationLabels {
    /**
     * @brief Size: local NE. element_seed[e] = 0-based index (into the SeedSet) of the seed that
     * won local element e. May be device-resident; read on the host with HostRead().
     */
    mfem::Array<int> element_seed;
    /**
     * @brief Size: local NE. element_grain[e] = compact grain id (1..num_grains) of local
     * element e. May be device-resident; read on the host with HostRead().
     */
    mfem::Array<int> element_grain;
    /** @brief Global number of grains: the number of seeds that won at least one element. */
    int num_grains = 0;
    /**
     * @brief Size: number of seeds N. seed_grain[i] = grain id (1-based) of seed i, or 0 if seed
     * i won no element anywhere. Increasing over the used seeds: seed order is grain order.
     */
    std::vector<int> seed_grain;
    /**
     * @brief Size: num_grains. grain_seed[g - 1] = 0-based seed index grain g grew from. Strictly
     * increasing (grains are numbered in seed order).
     */
    std::vector<int> grain_seed;
    /**
     * @brief Number of seeds that share their exact position with a lower-index seed (reported
     * by the seed search; identical on every rank). For Voronoi such a seed never wins (equal f,
     * the lower index wins ties). For weighted metrics it can: a coincident seed with a larger
     * Laguerre/Additive weight, or a different per-seed quadric tensor, wins where its f is
     * smaller.
     */
    int num_coincident_seeds = 0;
    /**
     * @brief Periodic runs (LabelTessellation with a periodic axis): number of image layers used
     * (RequiredImageLayers) and number of seed images searched; 0 otherwise. Identical on every
     * rank.
     */
    int periodic_layers = 0;
    int num_periodic_images = 0;
    /**
     * @brief Quadrature-vote runs: local elements whose voting points did not all vote for the
     * same (original) seed, i.e. elements cut by a grain boundary. Local to this rank; 0 for
     * centroid sampling.
     */
    int num_split_elements = 0;
};

/**
 * @brief For every local element, the index of the seed with the smallest metric value at the
 * element's volume centroid (step 1 in the file comment).
 *
 * Builds a SeedIndex for the metric `tessellation.metric` (DispatchMetric) on the Axom execution
 * space that matches the mfem::Device (DispatchAxomExec), with `tessellation.num_bins` parameter
 * bins and the global anisotropy tensor EffectiveGlobalTensor(tessellation) for the Quadric
 * metric (per-seed tensors are used instead when `seeds.shape` is filled). This is the plain
 * centroid search with exactly the given seeds: `tessellation.sampling` and
 * `tessellation.periodic` are not looked at here (LabelTessellation handles them, e.g. by
 * passing periodic images as `seeds`). Seed weights come from
 * `seeds.weight` (empty means all zero, so Laguerre and Additive then reduce to Voronoi-like
 * cells). The result is exact: it equals a loop over all seeds with the same metric, ties going
 * to the lowest index.
 *
 * Not collective (each rank labels its own elements). Host function launching device kernels.
 *
 * @param local_mesh   this rank's elements (3D, one element geometry; a ParMesh binds here).
 *                     Non-const only because the element sampler needs it; not modified.
 * @param seeds        replicated seed set, N >= 1 seeds (positions, and weight/shape as used by
 *                     the metric).
 * @param tessellation metric, num_bins and the anisotropy (EffectiveGlobalTensor) are used (the
 *                     tensor is ignored when seeds.shape is non-empty: per-seed tensors win).
 *                     A distance other than Euclidean is rejected.
 * @param element_seed resized to the local element count; element_seed[e] = 0-based seed index,
 *                     or -1 when the element centroid is not finite (FindBest's answer for a
 *                     non-finite query, e.g. 0/0 for a zero-volume element; LabelTessellation
 *                     turns that into an error). Written where the mfem::Device runs kernels
 *                     (HostRead() for the host).
 * @param num_coincident_seeds optional; receives the number of seeds that coincide exactly with
 *                     a lower-index seed (local count; identical on every rank since the seed
 *                     set is replicated).
 * @throws std::invalid_argument for an empty seed set, an unsupported distance or anisotropy, or
 *         invalid seed data (see SeedIndex::Build); std::runtime_error from the element sampler
 *         for a mixed-geometry or non-3D mesh.
 */
void LabelElementsWithSeeds(mfem::Mesh& local_mesh,
                            const SeedSet& seeds,
                            const TessellationOptions& tessellation,
                            mfem::Array<int>& element_seed,
                            int* num_coincident_seeds = nullptr);

/**
 * @brief Turns per-element seed indices into compact grain ids 1..N (step 2 in the file
 * comment).
 *
 * Grain ids are assigned in increasing seed index over the seeds that won at least one element on
 * any rank, so the numbering is identical on every rank and for every rank count.
 *
 * **Collective** over `comm` (one MPI_Allreduce with MPI_BOR on ceil(num_seeds / 64) 64-bit
 * words). Host function; the final element-id mapping is an mfem::forall kernel.
 *
 * @param comm          communicator over which the elements are distributed.
 * @param element_seed  this rank's per-element seed indices, each in [0, num_seeds).
 * @param num_seeds     number of seeds N (same on every rank).
 * @param labels        output: element_grain, num_grains, seed_grain and grain_seed are filled;
 *                      element_seed is copied in.
 * @throws std::invalid_argument on every rank if num_seeds < 1 or if any rank has a seed index
 *         outside [0, num_seeds) (the check is reduced over `comm`, so all ranks throw together).
 */
void CompactSeedLabels(MPI_Comm comm,
                       const mfem::Array<int>& element_seed,
                       int num_seeds,
                       TessellationLabels& labels);

/**
 * @brief The complete labeling: nearest seed per element (centroid or quadrature vote), periodic
 * seed images when requested, then compact grain ids.
 *
 * Steps:
 *  1. Periodic axes (tessellation.periodic): the periodic box is the mesh bounding box
 *     (PeriodicBoxFromMesh, collective), the number of image layers is RequiredImageLayers for
 *     the metric and tensor, and the seeds are replaced by BuildPeriodicImages(seeds) (wrapped
 *     originals first, then images; geometry/periodic_images.hpp).
 *  2. Per-element seed: centroid sampling (LabelElementsWithSeeds, then image indices mapped back
 *     to their originals with MapImageLabelsToOriginal) or quadrature voting
 *     (LabelElementsByQuadratureVote with tessellation.vote_levels and the image-to-original map
 *     as seed owners, labeling/quadrature_vote.hpp). Either way the result holds indices into
 *     the input `seeds`.
 *  3. Elements without a seed (-1: non-finite centroid or voting points) are detected
 *     collectively (ThrowIfAnyElementUnlabeled).
 *  4. Compaction to grain ids 1..N in seed order (CompactSeedLabels).
 *
 * **Collective** over `comm`. Every exception is thrown on all ranks together (the parameter
 * checks are identical on every rank, and the later checks are reduced). The labels do not
 * depend on the rank count or partition.
 *
 * @param local_mesh   this rank's part of the mesh (a serial mfem::Mesh with MPI_COMM_SELF works
 *                     too).
 * @param comm         communicator over which the mesh is distributed.
 * @param seeds        replicated seed set (N >= 1, N < 2^31; with periodic images the image
 *                     count must fit too).
 * @param tessellation tessellation options (metric, anisotropy, num_bins, sampling, vote_levels,
 *                     periodic, distance).
 * @return the labels of the local elements (seed indices into `seeds`) and the replicated
 *         seed/grain tables, plus the periodic and vote statistics.
 * @throws std::invalid_argument for unsupported options, an empty seed set or invalid seed data;
 *         std::invalid_argument from RequiredImageLayers for a sheared semidefinite Quadric
 *         tensor with a periodic axis; std::runtime_error for elements without a seed.
 */
TessellationLabels LabelTessellation(mfem::Mesh& local_mesh,
                                     MPI_Comm comm,
                                     const SeedSet& seeds,
                                     const TessellationOptions& tessellation);

} // namespace exaconstit::microstructure
