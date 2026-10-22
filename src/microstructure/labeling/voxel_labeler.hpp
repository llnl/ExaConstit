#pragma once

#include "microstructure/io/voxel_map_io.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <mpi.h>
#include <vector>

/**
 * @file voxel_labeler.hpp
 * @brief Gives every element of a distributed mesh the grain of the voxels under it, from a
 * voxel map (io/voxel_map_io.hpp), then fills elements that no voxel covers from their face
 * neighbors and renumbers the grains used as compact ids 1..N.
 *
 * **Steps** (LabelElementsFromVoxels):
 * 1. *Which voxels this rank needs (tiles).* The voxel grid is split into fixed cubic *tiles*
 *    of tile_size^3 voxels (smaller at the far edges of the grid). A first pass over this rank's
 *    actual sample points (centroids, or the voting points chunk by chunk) marks every tile
 *    holding the voxel of a point; each point also marks the tile of the voxel one round-off
 *    step (10^-6 voxel) to either side, so a point lying on a voxel face is always covered. The
 *    sample points themselves decide, not the element vertices or nodes: a curved (quadratic)
 *    element can bulge past its nodes.
 * 2. *Reading.* The rank reads only the marked tiles (one HDF5 hyperslab read per tile,
 *    ReadVoxelBoxes), checks that every id read is < 2^31 (labels are int; a failure on any rank
 *    stops all ranks), stores ids <= 0 as 0, and keeps the tiles back to back in an
 *    mfem::Array<int> with a small dense table over the box of marked tiles giving the start of
 *    each tile's ids (-1 = tile not read). Memory therefore grows with the number of touched
 *    tiles, not with the bounding box of the rank's elements (a mesh partition can consist of
 *    pieces spread over the whole grid).
 * 3. *Lookup and vote.* A second pass over the same sample points (recomputed, so memory stays
 *    bounded by one chunk) looks up each point's voxel in two steps, tile then voxel within the
 *    tile, in an mfem::forall kernel. The voxel of a point x is floor((x - origin) / spacing)
 *    per axis (mesh units). A point outside the grid, or on a voxel with id <= 0, does not vote.
 *    Each element goes to the feature id with the largest total weight w*detJ of its points
 *    (ties: lowest id). Centroid sampling uses the element volume centroid (one point decides);
 *    vote sampling uses the subcell-center voting points of labeling/quadrature_vote.hpp
 *    (8^vote_levels points per element).
 * 4. *Fill.* Elements with no vote (outside the grid, or entirely over unassigned voxels) are
 *    filled in synchronous rounds: in each round, every still-unassigned element takes the
 *    feature id that the largest number of its face neighbors carried at the end of the previous
 *    round (each shared face counts once; ties: lowest id), using the neighbor exchange of
 *    mesh/mesh_adjacency.hpp so neighbors on other ranks count too. Rounds repeat until one
 *    changes nothing. Elements still unassigned then (a piece of the mesh that touches no voxel
 *    at all) are a collective error with the global count.
 * 5. *Compaction* (CompactFeatureLabels): the feature ids used anywhere become grains 1..N in
 *    ascending feature id.
 *
 * **Rank-count independence.** An element's sample points depend only on its own nodes, the
 * voxel lookup only on the point and the (global) grid, the vote sums weights in a fixed order
 * inside the element, and a fill round reads only the previous round's labels; so the labels are
 * identical for every partition and rank count.
 *
 * **Where it runs.** Point sampling, the voxel lookup and the vote are mfem::forall kernels on
 * the backend of the configured mfem::Device; the tile marking (it reads the point coordinates
 * on the host), the tile reads, the fill rounds (small loops over face neighbors) and the
 * compaction tables are host code.
 *
 * **Typical use** (the caller reads the grid description once and passes it in):
 * @code
 *   const VoxelGridInfo grid = ReadVoxelGridInfo(map, comm);
 *   VoxelLabels labels = LabelElementsFromVoxels(pmesh, comm, map, grid, params);
 *   mfem::Array<int> element_grain;
 *   std::vector<std::int64_t> grain_feature;
 *   CompactFeatureLabels(comm, labels.element_feature, element_grain, grain_feature);
 *   VoxelFeatureTable features = ReadVoxelFeatures(map, grid, grain_feature, comm);
 * @endcode
 */

namespace exaconstit::microstructure {

/**
 * @brief How elements sample the voxel map (plain struct; the integration code fills it from the
 * user options).
 */
struct VoxelLabelParameters {
    /** @brief CENTROID (one point per element) or QUADRATURE_VOTE (subcell voting points). */
    SamplingMode sampling = SamplingMode::CENTROID;
    /**
     * @brief Vote sampling only: subdivision levels of the voting points, 0..3 (8^levels points
     * per element), as QuadratureVoteParameters::levels.
     */
    int vote_levels = 2;
    /**
     * @brief Vote sampling only: upper bound on the voting points evaluated at once (>= 1); chunks
     * hold whole elements. Bounds memory, does not change the result.
     */
    int max_points_per_chunk = 1 << 21;
    /**
     * @brief Edge length of a voxel tile, in voxels (>= 1). Ranks read whole tiles; smaller
     * tiles read fewer unneeded voxels but need more HDF5 reads. Does not change the result.
     */
    int tile_size = 32;
};

/**
 * @brief Result of LabelElementsFromVoxels.
 */
struct VoxelLabels {
    /**
     * @brief Size: local NE. Feature id (>= 1, as stored in the voxel map) of each local element
     * after the fill. Host-valid (HostRead()).
     */
    mfem::Array<int> element_feature;
    /** @brief Elements filled from neighbors (global count over all ranks). */
    std::int64_t num_filled = 0;
    /** @brief Fill rounds that assigned at least one element (identical on every rank). */
    int fill_rounds = 0;
    /**
     * @brief Sample points that fell outside the voxel grid (global count over all ranks; one
     * point per element for centroid sampling).
     */
    std::int64_t num_outside_points = 0;
    /** @brief Voxel tiles read by this rank (local count). */
    std::int64_t num_tiles_read = 0;
    /** @brief Voxels read by this rank (local count; the memory the rank spent on voxel ids). */
    std::int64_t num_voxels_read = 0;
};

/**
 * @brief Labels every local element with a feature id from the voxel map (steps 1-4 of the file
 * comment).
 *
 * Reads this rank's voxel tiles (ReadVoxelBoxes); the grid description is read once by the
 * caller (ReadVoxelGridInfo) and passed in, so it can be reused for ReadVoxelFeatures. Does not
 * read the per-grain table.
 *
 * **Collective** over `comm`: error agreement, the fill's neighbor exchanges and reductions.
 * Every exception is thrown on all ranks together. Host function launching device kernels.
 *
 * @param pmesh  the distributed mesh (3D, conforming, one element geometry: hex or tet for vote
 *               sampling). Non-const for MFEM's sampler and face-neighbor exchange; element data
 *               are not changed.
 * @param comm   the mesh communicator.
 * @param map    voxel map parameters (file, format, transforms).
 * @param grid   ReadVoxelGridInfo(map, comm) (identical on every rank).
 * @param params sampling mode, vote parameters and tile size.
 * @return per-element feature ids and fill statistics.
 * @throws std::invalid_argument on every rank for bad parameters (sampling mode, vote_levels,
 *         max_points_per_chunk, tile_size, a grid with non-positive dims or spacing);
 *         std::runtime_error on every rank for read errors, an id >= 2^31 in a tile some rank
 *         read, more than 2^31 - 1 voxels in one rank's tiles (use more ranks: each rank reads
 *         only the tiles under its own elements), or elements that cannot be filled (message
 *         with the global count).
 */
VoxelLabels LabelElementsFromVoxels(mfem::ParMesh& pmesh,
                                    MPI_Comm comm,
                                    const VoxelMapParameters& map,
                                    const VoxelGridInfo& grid,
                                    const VoxelLabelParameters& params);

/**
 * @brief Renumbers feature ids as compact grain ids 1..N in ascending feature id, over the ids
 * used on any rank (step 4 of the file comment).
 *
 * Each rank sorts its distinct feature ids; the lists are gathered on every rank
 * (MPI_Allgather of the counts, then MPI_Allgatherv of the ids) and merged, so every rank builds
 * the same table. element_grain[e] is then found by a binary search of the table (a binary
 * search finds a value in a sorted list by repeatedly halving the range that can hold it), in
 * an mfem::forall kernel.
 *
 * **Collective** over `comm`. Memory: the gathered lists hold at most (number of ranks) x
 * (distinct ids per rank) ints.
 *
 * @param comm            communicator over which the elements are distributed.
 * @param element_feature size local NE; every entry >= 1.
 * @param element_grain   resized to local NE; grain id 1..N of each element (written where the
 *                        mfem::Device runs kernels; HostRead() for the host).
 * @param grain_feature   size N on every rank: grain_feature[g - 1] = feature id of grain g
 *                        (strictly increasing).
 * @throws std::invalid_argument on every rank if any rank has a feature id < 1.
 */
void CompactFeatureLabels(MPI_Comm comm,
                          const mfem::Array<int>& element_feature,
                          mfem::Array<int>& element_grain,
                          std::vector<std::int64_t>& grain_feature);

} // namespace exaconstit::microstructure
