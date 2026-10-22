#pragma once

#include "microstructure/mesh/mesh_adjacency.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <vector>

/**
 * @file grain_statistics.hpp
 * @brief Per-grain statistics of a finished element labeling: element count, volume, volume
 * centroid, number of neighboring grains, and the shared face area of every pair of touching
 * grains; plus the comparison of grain volumes with size-control target volumes.
 *
 * These are the numbers the generator tool writes into its grain table and grain adjacency CSV
 * files and prints in its end-of-run summary. They describe the grains exactly as ExaConstit will
 * see them: whole elements, with the final grain ids (after fragment cleanup).
 *
 * **Same numbers for any rank count.** A floating-point sum depends on the order of its terms,
 * and that order depends on how the mesh is partitioned. All global sums here therefore use
 * fixed-point integers (util/fixed_point.hpp): every term is rounded once to a multiple of a tiny
 * quantum and stored as a 64-bit integer count of quanta; integer sums are exact, so they do not
 * depend on the order of the additions. Volumes and centroids come from CellMeasureAccumulator
 * (labeling/cell_measures.hpp), which works this way; face areas are summed the same way here.
 * The individual terms are themselves partition independent: element volumes and centroids are
 * computed from the element's own vertices, and each face area is computed from the face's
 * corners taken in a canonical order (see ComputeGrainStatistics), not in the order in which a
 * particular rank happens to store the face. The result is bitwise identical for every number
 * of ranks (for meshes of order > 1, see the note in ComputeGrainStatistics).
 */

namespace exaconstit::microstructure {

/**
 * @brief Statistics of grains 1..num_grains. Host data.
 *
 * Per-grain vectors have num_grains entries; entry g-1 (0-based) belongs to grain g (1-based).
 * Everything except the pair table is identical on every rank; the pair table (pair_a, pair_b,
 * pair_area, pair_faces) is filled on the `root` rank of ComputeGrainStatistics only and empty
 * on the other ranks, because it is only written to a file and can be large (about 7 pairs per
 * grain).
 */
struct GrainStatistics {
    /** @brief Number of grains N. */
    int num_grains = 0;
    /** @brief Global number of elements of each grain. */
    std::vector<std::int64_t> num_elements;
    /** @brief Volume of each grain: sum of its element volumes (mesh length units cubed). */
    std::vector<double> volume;
    /** @brief Volume centroid of each grain, 3 per grain interleaved: x y z of grain g at
     * [3(g-1) .. 3(g-1)+2] (mesh length units). Plain volume-weighted average of the element
     * centroids: for a grain that wraps around a periodic box it is not unwrapped and may lie
     * outside the grain. */
    std::vector<double> centroid;
    /** @brief Number of distinct other grains that share at least one element face with each
     * grain (including faces paired across periodic box planes, when the adjacency has them).
     * Identical on every rank. */
    std::vector<int> num_neighbors;
    /**
     * @brief Pairs of grains that share element faces, sorted by (pair_a, pair_b) with
     * pair_a < pair_b (1-based grain ids). The four pair vectors have the same length. Root rank
     * only (empty elsewhere).
     */
    std::vector<int> pair_a, pair_b;
    /** @brief Total shared face area of each pair (mesh length units squared). */
    std::vector<double> pair_area;
    /** @brief Number of shared element faces of each pair. */
    std::vector<std::int64_t> pair_faces;
    /** @brief Sum of all grain volumes: the material volume (fixed-point sum). */
    double total_volume = 0.0;
};

/**
 * @brief Computes the GrainStatistics of a labeled distributed mesh.
 *
 * Steps:
 *  1. Element count, volume and volume centroid of every grain: one sample per element (its
 *     volume centroid, weighted by its volume; MakeElementWeightedSamples) measured with
 *     CellMeasureAccumulator, whose sums are fixed-point and therefore rank-count independent.
 *  2. Grain ids of the ghost elements (elements of other ranks across shared or periodic faces):
 *     MeshAdjacency::ExchangeLabels.
 *  3. Every face between two different grains is counted exactly once over all ranks: from the
 *     side of the element with the *smaller* grain id. (Each face appears in the adjacency of
 *     both of its elements, which may live on two ranks; exactly one of the two sides has the
 *     smaller grain id, so no rank needs to know what the other counted.) Its area is computed
 *     from the face's corner points in a canonical order: start at the lexicographically
 *     smallest corner (compare x, then y, then z) and walk around the face toward the smaller of
 *     its two neighbors. A triangle's area is half the cross-product norm; a quadrilateral's is
 *     the integral of the bilinear surface's area element with MFEM's order-3 square rule
 *     (2 x 2 Gauss points; for linear meshes this is the order MeshAdjacency uses). The
 *     canonical order makes the floating-point operations, and so the bits of the area,
 *     independent of how a rank stores the face.
 *  4. Areas are summed per grain pair in fixed point (quantum from the global largest face area
 *     and face count, MPI_Allreduce MAX / SUM). The per-rank pair records are gathered on `root`
 *     only (MPI_Gatherv of 64-bit integers) and merged there by exact integer addition.
 *  5. The neighbor count of a grain is the number of pairs it appears in (computed on root and
 *     broadcast, one int per grain).
 *
 * Limits: root holds one 32-byte record per (rank, grain pair) touched by that rank, i.e. about
 * 7 x num_grains records plus duplicates for pairs that span ranks (~250 MB for 10^6 grains),
 * and the gather is indexed with int, which caps the total at INT_MAX / 4 (~5.4e8) records
 * (std::runtime_error on every rank beyond that).
 *
 * Mesh order: meshes without Nodes or with order-1 Nodes (common for meshes read with a nodes
 * section) use the canonical corner formula of step 3 and are bitwise rank-count independent.
 * For Nodes of order > 1 (curved faces) the face area is taken from MeshAdjacency::FaceAreas()
 * (quadrature on the curved face) instead. That value may differ in the last bits between
 * partitions that store the face with a different orientation, so the shared areas of curved
 * meshes are rank-count independent only to round-off; everything else stays bitwise.
 *
 * **Collective** over the mesh communicator (every rank must call it, also a rank without
 * elements). Host function; the element samples and the cell-measure accumulation run as
 * mfem::forall kernels on the configured mfem::Device, the face loop on the host.
 *
 * @param adjacency     face adjacency of this rank's mesh partition (with periodic pairing when
 *                      grains should count as neighbors across periodic box faces). The mesh is
 *                      adjacency.GetParMesh(); MFEM caches geometric factors in it, otherwise it
 *                      is not modified.
 * @param element_grain size = local element count; grain id (1-based, 1..num_grains) of every
 *                      local element. Read on the host (HostRead).
 * @param num_grains    number of grains N >= 1 (same on every rank).
 * @param root          rank (of the mesh communicator) that receives the pair table.
 * @return the statistics; the pair table on `root` only, everything else on every rank.
 * @throws std::invalid_argument on every rank if num_grains < 1, root is not a rank, the label
 *         array has the wrong size or a grain id is outside 1..num_grains on any rank (from
 *         CellMeasureAccumulator); std::runtime_error on every rank if the pair records exceed
 *         the gather limit above.
 */
GrainStatistics ComputeGrainStatistics(const MeshAdjacency& adjacency,
                                       const mfem::Array<int>& element_grain,
                                       int num_grains,
                                       int root = 0);

/**
 * @brief Element-level volumes of the seeds compared with their size-control targets.
 *
 * Size control (weight fitting, polydisperse placement) gives every seed a target volume, but
 * the weight fit measures volumes on quadrature points while the final grains are whole elements.
 * This comparison shows how close the *final* element grains come to the targets. Host data.
 */
struct TargetVolumeComparison {
    /** @brief Number of seeds with a target (size of the target list). */
    int num_seeds = 0;
    /** @brief Seeds whose grains hold no element at all (relative error 1). */
    int num_empty_seeds = 0;
    /** @brief Element-level volume of every seed: the sum of the volumes of all final grains that
     * descend from it (a seed can have several grains after fragment splitting, or none). */
    std::vector<double> seed_volume;
    /** @brief max over seeds of |V_s - V_s^target| / V_s^target. */
    double max_relative_error = 0.0;
    /** @brief Mean over seeds of the same relative error. */
    double mean_relative_error = 0.0;
    /** @brief 0-based index of the seed with the largest relative error (-1 without seeds). */
    int worst_seed = -1;
};

/**
 * @brief Compares the element-level seed volumes with the target volumes.
 *
 * Each grain's volume is added to the seed it descends from (in grain order, so the result is
 * the same on every rank), then the relative errors are taken seed by seed. Host only; not
 * collective (all inputs are replicated).
 *
 * @param grain_volume        volume of grain g at index g-1 (e.g. GrainStatistics::volume).
 * @param grain_seed          0-based seed index of grain g at index g-1 (GenerationReport::
 *                            grain_seed); same size as grain_volume.
 * @param seed_target_volume  target volume of every seed (> 0), seed order.
 * @return the comparison.
 * @throws std::invalid_argument if the grain vectors differ in size, a seed index is outside the
 *         target list, or a target is not positive and finite.
 */
TargetVolumeComparison CompareWithTargetVolumes(const std::vector<double>& grain_volume,
                                                const std::vector<int>& grain_seed,
                                                const std::vector<double>& seed_target_volume);

} // namespace exaconstit::microstructure
