#pragma once

#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <array>
#include <mpi.h>
#include <vector>

/**
 * @file periodic_images.hpp
 * @brief Periodic seed images: copies of the seeds shifted by whole box lengths, so that a
 * tessellation of a box-shaped domain wraps around periodic faces (a grain cut by the face x = lo
 * continues on the face x = hi).
 *
 * **What a periodic microstructure is.** Think of the box as one cell of an infinite crystal
 * that repeats the box in every periodic direction (like the unit cell of a lattice). Every seed
 * s_i then exists at all positions s_i + n L, where L = (L_x, L_y, L_z) are the box lengths and
 * n = (n_x, n_y, n_z) is any integer vector (n_d = 0 on non-periodic axes). A point x of the box
 * belongs to the grain of the seed image with the smallest metric value f. Because every copy of
 * the box sees the same seeds, the grain structure repeats, and the grains on opposite faces
 * match up: an RVE with periodic boundary conditions needs exactly this.
 *
 * **Periodic images.** The labeler only knows a finite seed list, so we hand it the seeds plus a
 * finite number of shifted copies ("images"): every seed shifted by n_d in {-k, ..., k} box
 * lengths on each periodic axis (k = "layers", default 1). One periodic axis gives 3 copies of
 * each seed (the seed and its images left and right), two axes give 3 x 3 = 9, three axes give
 * 3 x 3 x 3 = 27 (the seed and its 26 neighbors). After labeling, the index of an image is mapped
 * back to the index of the seed it was copied from (MapImageLabelsToOriginal), so an element won
 * by an image belongs to the original seed's grain. That is what makes a grain that crosses a
 * periodic face one grain instead of two.
 *
 * **Why one layer (a margin of one full box length) is exact.** First the seeds are wrapped into
 * the box: on each periodic axis a seed coordinate is moved by whole box lengths into
 * [lo, hi). Now take any point x of the box and any seed s. On a periodic axis d both x_d and s_d
 * lie in [lo_d, hi_d], so their difference is less than one box length: |x_d - s_d| <= L_d.
 * Among all copies s_d + n_d L_d, the closest one to x_d is therefore one of n_d = -1, 0, +1
 * (moving one more box length away can only increase the gap). For the metrics whose value grows
 * with the size of each coordinate difference separately - Voronoi |x - s|^2, Laguerre
 * |x - s|^2 - w, Additive / Johnson-Mehl |x - s| - a, and a Quadric with a *diagonal* tensor
 * (axis-aligned anisotropy) sum_d A_dd (x_d - s_d)^2 - w - the image of s with the smallest f is
 * thus among the 27 copies with n_d in {-1, 0, 1}. The seed weight w is the same for all copies
 * of one seed, so it does not change which copy is best. Hence, for every seed, its best copy is
 * in the image set, and the minimum over the image set equals the minimum over the infinite
 * periodic array: the labels are exactly those of the infinite periodic tessellation. No
 * assumption on grain size, seed spacing or seed count is needed, which is why the full box
 * length is the default.
 *
 * **Assumption: query points lie in the box.** Both this argument and the bound of
 * RequiredImageLayers take the points being labeled (element centroids or voting points) inside
 * [lo, hi] on the periodic axes, as they are for a box mesh. A point outside the box by up to
 * half a box length is still labeled exactly with one layer for the separable metrics above
 * (|x_d - s_d| <= 1.5 L_d, so the closest copy is still among n_d = -1, 0, 1). For the Quadric
 * bound an overshoot delta adds delta / L_min layers (see RequiredImageLayers). Curved elements
 * whose faces bulge past the box planes violate the assumption; periodic boxes should have planar
 * faces.
 *
 * **When one layer is not enough.** A Quadric with off-diagonal terms (anisotropy not aligned
 * with the box axes) mixes the coordinates: a strongly sheared ruler can make a copy two or more
 * box lengths away "closer" than any of the 27 neighbors. RequiredImageLayers computes a number
 * of layers that is provably sufficient in that case (see its documentation); use it to set
 * PeriodicImageParameters::layers.
 *
 * **A smaller margin.** Copies far from the box rarely win anything, and each one costs search
 * time, so PeriodicImageParameters::margin can keep only the images within a given distance of
 * the box. This is exact only if no point of the box is won by an image farther away than the
 * margin; for Voronoi that holds when the margin is at least the largest distance from any point
 * of the box to its nearest seed copy (the radius of the largest seed-free ball). That is
 * typically a few mean seed spacings for well-spread seeds, but it is not guaranteed in general,
 * so use it as a performance option only.
 *
 * **Tie-break.** The extended seed list has the wrapped originals first (indices 0..N-1, same
 * order as the input) and then the images. The labeler breaks exact ties by the lowest index of
 * the extended list, which is the same on every rank, so labels stay independent of the rank
 * count. (A point exactly equidistant from a seed and another seed's image may therefore go to a
 * different seed than a non-periodic run would pick, but always the same one.)
 *
 * Steps that use face adjacency see the two halves of a wrapped grain as touching only through
 * the periodic face, which the mesh itself does not connect. The fragment cleanup of the driver
 * therefore builds its adjacency with periodic face pairing (MeshAdjacency with
 * PeriodicFaceParameters, mesh/mesh_adjacency.hpp), which joins matching faces on opposite
 * planes of the box, so a wrapped grain stays one grain.
 */

namespace exaconstit::microstructure {

/**
 * @brief The periodic box: which axes are periodic and the box extent on every axis.
 *
 * Host data, identical on every rank. Lengths are in mesh units. On a periodic axis d the box
 * length is L_d = hi[d] - lo[d] (> 0). lo/hi are also filled for non-periodic axes (they are
 * used only by RequiredImageLayers).
 */
struct PeriodicBox {
    /** @brief periodic[d] is true when axis d (0 = x, 1 = y, 2 = z) is periodic. */
    std::array<bool, 3> periodic = {false, false, false};
    /** @brief Lower box corner (mesh length units). */
    std::array<double, 3> lo = {0.0, 0.0, 0.0};
    /** @brief Upper box corner (mesh length units). */
    std::array<double, 3> hi = {1.0, 1.0, 1.0};
};

/**
 * @brief How many images to make.
 *
 * Plain parameter struct (the integration code fills it from the user options).
 */
struct PeriodicImageParameters {
    /**
     * @brief Number of image layers k >= 1: images are shifted by n_d in {-k, ..., k} box lengths
     * on every periodic axis, (2k + 1)^p copies of each seed for p periodic axes (before the
     * margin filter). 1 is exact for Voronoi, Laguerre, Additive and axis-aligned Quadric metrics;
     * see RequiredImageLayers for general Quadric tensors.
     */
    int layers = 1;
    /**
     * @brief Keep only images that lie within this distance (mesh length units) of the box on
     * every periodic axis: lo_d - margin <= coordinate <= hi_d + margin. Negative (the default)
     * means "layers full box lengths" on each periodic axis, i.e. keep every image (exact). A
     * value >= 0 is a performance option, exact only under the condition in the file comment.
     */
    double margin = -1.0;
};

/**
 * @brief A seed set extended by periodic images, with the map back to the original seeds.
 *
 * Host data, identical on every rank (built from the replicated SeedSet).
 */
struct PeriodicImages {
    /**
     * @brief The extended seed set: entries 0..num_original-1 are the input seeds in input order
     * with their positions wrapped into the box on the periodic axes (bitwise unchanged when they
     * already lie in [lo, lo + L), with L = hi - lo); entries num_original.. are the images,
     * ordered by shift vector n (lexicographic, n_x slowest, skipping n = 0) and then by
     * original seed index. Every non-empty
     * per-seed array of the input (weight, shape, quat, source_id, phase, parent_id,
     * nucleation_time) is copied to the images, so an image carries the same weight, tensor,
     * orientation and source id as its original.
     */
    SeedSet seeds;
    /**
     * @brief Size seeds.Size(): original[k] = 0-based index in the input SeedSet of the seed that
     * entry k is a copy of (original[k] = k for k < num_original).
     */
    std::vector<int> original;
    /** @brief Number of input seeds N. */
    int num_original = 0;
    /** @brief Number of images kept (seeds.Size() - num_original). */
    int num_images = 0;
};

/**
 * @brief The bounding box of the whole distributed mesh, used as the periodic box.
 *
 * Computed from the mesh node coordinates: the Nodes GridFunction when the mesh has one (curved,
 * high-order, or topologically periodic meshes), else the vertices. Per-rank minimum and maximum,
 * then MPI_Allreduce with MIN and MAX. A topologically periodic MFEM mesh (Mesh::MakePeriodic,
 * discontinuous Nodes) is handled: its vertices on opposite faces are merged and would give a box
 * one element short, but its Nodes keep every element's corners. For a periodic microstructure
 * the mesh must fill a box whose faces are planes perpendicular to the periodic axes.
 *
 * **Collective** over `comm` (two MPI_Allreduce calls). Host only. A rank without elements
 * contributes nothing.
 *
 * @param local_mesh this rank's part of the mesh (a serial mesh with MPI_COMM_SELF works too).
 * @param comm       communicator over which the mesh is distributed.
 * @param periodic   periodic flags to copy into the result.
 * @return the box (identical on every rank).
 * @throws std::invalid_argument on every rank if the mesh is not 3D, the global mesh has no nodes,
 *         or a periodic axis has zero (or non-finite) extent.
 */
PeriodicBox PeriodicBoxFromMesh(const mfem::Mesh& local_mesh,
                                MPI_Comm comm,
                                const std::array<bool, 3>& periodic);

/**
 * @brief Wraps the seeds into the box and appends their periodic images.
 *
 * Host only; not collective (every rank builds the same result from the replicated seeds). With
 * no periodic axis the result is a copy of the input with no images.
 *
 * @param seeds  input seeds, N >= 1, finite positions. Seeds outside the box on a periodic axis
 *               are moved into it by whole box lengths (that is the same physical microstructure).
 * @param box    the periodic box (PeriodicBoxFromMesh); L_d > 0 on every periodic axis.
 * @param params layers and margin.
 * @return the extended seed set and the image-to-original map.
 * @throws std::invalid_argument if N == 0, a position is not finite, a periodic box length is not
 *         positive and finite, layers < 1, or the extended seed count would exceed 2^31 - 1.
 */
PeriodicImages BuildPeriodicImages(const SeedSet& seeds,
                                   const PeriodicBox& box,
                                   const PeriodicImageParameters& params);

/**
 * @brief Replaces image indices by original seed indices in a per-element seed array, so that
 * every element won by an image is given to the grain of the seed the image was copied from.
 *
 * Runs as an mfem::forall kernel where the mfem::Device runs kernels (host or GPU). Not
 * collective. After this call the indices are in [0, num_original) and can go straight into
 * CompactSeedLabels (labeling/tessellation_labeler.hpp) with num_seeds = images.num_original.
 *
 * @param images       the image set the labels were computed with.
 * @param element_seed per local element: an index into images.seeds (as produced by
 *                     LabelElementsWithSeeds or LabelElementsByQuadratureVote with images.seeds),
 *                     or -1 (no seed: passed through unchanged). Overwritten in place.
 * @throws std::invalid_argument if an index is >= images.seeds.Size() (checked on the host).
 */
void MapImageLabelsToOriginal(const PeriodicImages& images, mfem::Array<int>& element_seed);

/**
 * @brief A number of image layers that makes the periodic labeling provably exact for the given
 * metric.
 *
 * - Voronoi, Laguerre, Additive: 1 (see the file comment).
 * - Quadric with diagonal tensors (global, or every per-seed tensor): 1, for the same reason.
 * - Quadric with off-diagonal terms: for each tensor A with eigenvalues mu_min > 0 and mu_max,
 *   k = max(1, ceil( sqrt(mu_max / mu_min) * R / L_min )), maximized over the tensors, where
 *   L_min is the smallest periodic box length and R^2 = sum over periodic axes (L_d / 2)^2 + sum
 *   over non-periodic axes E_d^2, with E_d the extent of the box and the seeds together on axis d.
 *   Proof: with wrapped seeds the copy with n_d in {-1, 0, 1} chosen per axis to minimize each
 *   |x_d - s_d - n_d L_d| has displacement d* with |d*|^2 <= R^2, so its f is at most
 *   mu_max R^2 - w. A copy with |n_d| >= k + 1 on some periodic axis has |displacement_d| >=
 *   (k + 1) L_d - L_d >= k L_min, so its f is at least mu_min k^2 L_min^2 - w >= mu_max R^2 - w.
 *   The weight w is the same for all copies of one seed, so no copy outside k layers can beat
 *   the best copy inside. The proof assumes the query point x lies in the box on the periodic
 *   axes (|x_d - s_d| <= L_d); if points can lie outside it by up to delta, use
 *   k >= (sqrt(mu_max / mu_min) R + delta) / L_min instead (|displacement_d| >= k L_min - delta).
 *
 * Host only; not collective.
 *
 * @param box           the periodic box.
 * @param metric        metric family.
 * @param global_tensor Quadric global tensor [Axx, Ayy, Azz, Ayz, Axz, Axy] (used when
 *                      seeds.shape is empty).
 * @param seeds         the seeds (per-seed tensors in seeds.shape, and positions for E_d).
 * @return k >= 1 (1 when no axis is periodic).
 * @throws std::invalid_argument for a Quadric tensor with off-diagonal terms and mu_min = 0
 *         (semidefinite and not axis-aligned: no finite number of layers is guaranteed exact),
 *         or an invalid tensor (see ComputeQuadricFactor).
 */
int RequiredImageLayers(const PeriodicBox& box,
                        MetricType metric,
                        const std::array<double, 6>& global_tensor,
                        const SeedSet& seeds);

} // namespace exaconstit::microstructure
