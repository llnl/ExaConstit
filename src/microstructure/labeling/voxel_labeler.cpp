/*
 * Implementation of the voxel labeler (voxel_labeler.hpp): sample points -> touched voxel tiles
 * -> tile reads -> voxel lookup and weighted vote per element -> fill of unvoted elements from
 * face neighbors -> compaction of feature ids to grain ids.
 *
 * Terms used in this file:
 * - mfem::forall: MFEM's parallel loop; a GPU kernel when the mfem::Device has a CUDA or HIP
 *   backend, a plain host loop otherwise. MFEM_HOST_DEVICE marks code callable inside it.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it. Local failures are turned into collective ones with the helpers of
 *   util/collective_errors.hpp, so a failing rank never leaves the others waiting.
 * - Voxel tile: a fixed cube of tile_size^3 voxels of the grid (smaller at the far edges). A
 *   rank reads only the tiles that hold voxels under its own sample points.
 * - Binary search: finds a value in a sorted list by repeatedly halving the range that can hold
 *   it.
 * - Synchronous fill round: every unassigned element decides from the labels its neighbors had
 *   at the end of the previous round, so the order in which elements (or ranks) are processed
 *   does not matter.
 */
#include "microstructure/labeling/voxel_labeler.hpp"

#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/**
 * @brief Point label codes of the lookup (positive values are feature ids).
 *
 * k_outside_block means a point inside the grid whose tile this rank did not read. Tiles are
 * chosen from the very same sample points (with a round-off margin), so this cannot happen
 * unless the two passes over the points disagree; it is kept as a consistency check that stops
 * all ranks with an "internal error" message instead of silently mislabeling an element.
 */
constexpr int k_no_vote = 0;        ///< inside the grid on a voxel with id <= 0
constexpr int k_outside_grid = -1;  ///< outside the voxel grid
constexpr int k_outside_block = -2; ///< inside the grid, tile not read (consistency check)

/** @brief Round-off margin of the tile marking, as a fraction of a voxel. */
constexpr double k_tile_margin = 1.0e-6;

/**
 * @brief Device-callable two-level voxel lookup: tile first, then the voxel within the tile.
 *
 * Plain data (copied into kernels by value). The tiles this rank read are stored back to back in
 * `ids` (each tile x fastest over its own extents, ids <= 0 stored as 0); `offsets` is a dense
 * table over the box of tile indices [t0, t0 + range) (x fastest) giving the start of each tile
 * in `ids`, or -1 when the tile was not read. Both pointers address memory where kernels run
 * (mfem::Array<int>::Read()).
 */
struct TileLookup {
    double origin[3];     ///< grid lower corner (mesh units)
    double spacing[3];    ///< voxel edge lengths (mesh units)
    double dims_d[3];     ///< voxel counts of the grid, as doubles for the range test
    long long dims[3];    ///< voxel counts of the grid
    long long tile;       ///< tile edge length in voxels
    long long t0[3];      ///< first tile index of the offset table per axis
    long long range[3];   ///< tile counts of the offset table per axis
    const int* offsets;   ///< range[0]*range[1]*range[2] entries
    const int* ids;       ///< tile data

    /**
     * @brief Label of the point (x, y, z): the feature id (> 0), k_no_vote, k_outside_grid or
     * k_outside_block. The voxel index along axis d is floor((x_d - origin_d) / spacing_d); the
     * range test is done on the double value before converting to an integer, so a far-away
     * point cannot overflow the conversion, and a NaN fails every comparison and is reported as
     * outside the grid. Runs on host or device.
     */
    MFEM_HOST_DEVICE int operator()(double x, double y, double z) const {
        const double p[3] = {x, y, z};
        long long v[3];  // voxel index
        long long t[3];  // tile index
        long long e[3];  // extents of that tile (edge tiles are smaller)
        // Grid test on every axis first (a point outside the grid never votes)
        for (int d = 0; d < 3; ++d) {
            const double f = floor((p[d] - origin[d]) / spacing[d]);
            if (!(f >= 0.0 && f < dims_d[d])) {
                return k_outside_grid;
            }
            v[d] = static_cast<long long>(f);
        }
        // Tile index, relative to the offset table
        long long rel[3];
        for (int d = 0; d < 3; ++d) {
            t[d] = v[d] / tile;
            rel[d] = t[d] - t0[d];
            if (rel[d] < 0 || rel[d] >= range[d]) {
                return k_outside_block;
            }
            const long long remaining = dims[d] - t[d] * tile;
            e[d] = (remaining < tile) ? remaining : tile;
        }
        const int offset = offsets[(rel[2] * range[1] + rel[1]) * range[0] + rel[0]];
        if (offset < 0) {
            return k_outside_block;
        }
        // Voxel within the tile, x fastest
        const long long local = ((v[2] - t[2] * tile) * e[1] + (v[1] - t[1] * tile)) * e[0] +
                                (v[0] - t[0] * tile);
        return ids[offset + local];
    }
};

/**
 * @brief Labels the points of one chunk with the lookup (one mfem::forall iteration per point).
 * `label` is resized to the number of points.
 *
 * A named function (not a lambda inside a lambda) so the device lambda is defined at function
 * scope, which CUDA requires.
 */
void LabelPoints(const TileLookup& lookup, const mfem::Vector& x, const mfem::Vector& y,
                 const mfem::Vector& z, mfem::Array<int>& label) {
    const int n = x.Size();
    label.SetSize(n);
    if (n == 0) {
        return;
    }
    const double* px = x.Read();
    const double* py = y.Read();
    const double* pz = z.Read();
    int* out = label.Write();
    const TileLookup lk = lookup;
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int i) { out[i] = lk(px[i], py[i], pz[i]); });
}

/**
 * @brief The weighted vote of one chunk of elements (one mfem::forall iteration per element).
 *
 * Element e of the chunk owns points base .. base + nq - 1, base = (e - chunk_begin) * nq.
 * Points with label <= 0 do not vote. For every label that first appears at point q, its total
 * weight is summed over points q..nq-1 in increasing order (no scratch memory, works the same on
 * a GPU, and the sum order depends only on the element); the larger total wins and an exactly
 * equal total goes to the lower feature id. This is the same scheme as VoteChunk in
 * labeling/quadrature_vote.cpp, with feature ids as labels.
 *
 * Also counts, per element, the points outside the grid and flags points on tiles that were not read.
 *
 * @param element_feature output at chunk_begin + e: winning feature id, or 0 without a vote.
 * @param element_outside output: points of the element outside the grid.
 * @param element_bad     output: 1 if a point fell on a tile that was not read.
 */
void VoteChunk(int chunk_begin, int chunk_elems, int nq, const mfem::Array<int>& label,
               const mfem::Vector& weight, mfem::Array<int>& element_feature,
               mfem::Array<int>& element_outside, mfem::Array<int>& element_bad) {
    const int* point_label = label.Read();
    const double* point_weight = weight.Read();
    // ReadWrite: the arrays cover all local elements; only this chunk's entries are written
    int* feature_out = element_feature.ReadWrite();
    int* outside_out = element_outside.ReadWrite();
    int* bad_out = element_bad.ReadWrite();

    mfem::forall(chunk_elems, [=] MFEM_HOST_DEVICE(int local) {
        const int base = local * nq; // first point of this element in the chunk
        int winner = 0;              // 0 = no vote yet
        double winner_weight = 0.0;
        int outside = 0;
        int bad = 0;
        for (int q = 0; q < nq; ++q) {
            const int lab = point_label[base + q];
            if (lab == k_outside_grid) {
                ++outside;
            }
            if (lab == k_outside_block) {
                bad = 1;
            }
            if (lab <= 0) {
                continue;
            }
            // Count each label once: skip it if an earlier point already carried it
            bool seen_before = false;
            for (int p = 0; p < q; ++p) {
                if (point_label[base + p] == lab) {
                    seen_before = true;
                    break;
                }
            }
            if (seen_before) {
                continue;
            }
            // Total weight of this label, summed in increasing point order
            double total = 0.0;
            for (int p = q; p < nq; ++p) {
                if (point_label[base + p] == lab) {
                    total += point_weight[base + p];
                }
            }
            // Larger total wins; exact tie -> lower feature id
            if (winner == 0 || total > winner_weight ||
                (total == winner_weight && lab < winner)) {
                winner = lab;
                winner_weight = total;
            }
        }
        feature_out[chunk_begin + local] = winner;
        outside_out[chunk_begin + local] = outside;
        bad_out[chunk_begin + local] = bad;
    });
}

/**
 * @brief The sample points of this rank's elements, delivered chunk by chunk: one chunk with the
 * element centroids (weights = element volumes) for centroid sampling, or the subcell voting
 * points of labeling/quadrature_vote.hpp in chunks of whole elements for vote sampling.
 *
 * ForEachChunk can be called several times and always delivers bitwise identical points in the
 * same chunks (the centroids are computed once and kept; the voting points are recomputed by the
 * same sampler kernel), which is what lets the tile marking pass and the lookup pass agree.
 * Host object; not collective.
 */
class SamplePoints {
public:
    SamplePoints(mfem::ParMesh& pmesh, const VoxelLabelParameters& params)
        : m_num_elements(pmesh.GetNE()) {
        if (m_num_elements == 0) {
            return;
        }
        if (params.sampling == SamplingMode::CENTROID) {
            // Volume centroids (mesh/mesh_sampler.hpp); the weight is the element volume
            ComputeElementSamples(pmesh, -1, m_centroids);
            m_nq = 1;
            m_chunk_elems = m_num_elements;
        } else {
            if (pmesh.GetNumGeometries(3) > 1) {
                throw std::runtime_error("Voxel labeler: vote sampling needs one element "
                                         "geometry");
            }
            m_sampler = std::make_unique<QuadraturePointSampler>(
                pmesh, VoteRule(pmesh.GetElementGeometry(0), params.vote_levels));
            m_nq = m_sampler->NumPointsPerElement();
            m_chunk_elems = std::min(m_num_elements,
                                     std::max(1, params.max_points_per_chunk / m_nq));
        }
    }

    /** @brief Points per element. */
    int PointsPerElement() const {
        return m_nq;
    }

    /**
     * @brief Calls visit(begin, end, x, y, z, weight) for every chunk of elements [begin, end),
     * with the QuadraturePointSamples layout (point q of element e at (e - begin) * nq + q).
     */
    template <typename Visitor>
    void ForEachChunk(Visitor&& visit) const {
        if (m_num_elements == 0) {
            return;
        }
        if (!m_sampler) {
            visit(0, m_num_elements, m_centroids.cx, m_centroids.cy, m_centroids.cz,
                  m_centroids.volume);
            return;
        }
        QuadraturePointSamples points;
        for (int begin = 0; begin < m_num_elements; begin += m_chunk_elems) {
            const int end = std::min(m_num_elements, begin + m_chunk_elems);
            m_sampler->Sample(begin, end, points);
            visit(begin, end, points.x, points.y, points.z, points.weight);
        }
    }

private:
    int m_num_elements = 0;
    int m_nq = 0;
    int m_chunk_elems = 0;
    ElementSamples m_centroids;
    std::unique_ptr<QuadraturePointSampler> m_sampler;
};

/** @brief Tile counts of the grid and the tile edge length (host). */
struct TileGrid {
    std::int64_t tile = 32;
    std::array<std::int64_t, 3> count = {0, 0, 0}; ///< tiles per axis, ceil(dims / tile)
};

/**
 * @brief Pass 1: the sorted global ids ((tz * nty + ty) * ntx + tx) of every tile holding the
 * voxel of a sample point, or of the voxel 10^-6 voxel to either side of it (round-off margin).
 * Points outside the grid (by more than the margin) and non-finite points mark nothing.
 *
 * Host loop over the point coordinates (HostRead). The marks of each chunk are sorted and made
 * unique, then merged into the running list, so memory grows with the touched tiles plus one
 * chunk.
 */
std::vector<std::int64_t> MarkTiles(const SamplePoints& points, const VoxelGridInfo& grid,
                                    const TileGrid& tiles) {
    std::vector<std::int64_t> touched;
    std::vector<std::int64_t> chunk_marks;
    std::vector<std::int64_t> merged;
    points.ForEachChunk([&](int, int, const mfem::Vector& x, const mfem::Vector& y,
                            const mfem::Vector& z, const mfem::Vector&) {
        const int n = x.Size();
        const double* coords[3] = {x.HostRead(), y.HostRead(), z.HostRead()};
        chunk_marks.clear();
        for (int i = 0; i < n; ++i) {
            // Per axis: tile range of the voxels within the margin of the point
            std::int64_t t_lo[3];
            std::int64_t t_hi[3];
            bool inside = true;
            for (int d = 0; d < 3 && inside; ++d) {
                const double u = (coords[d][i] - grid.origin[d]) / grid.spacing[d];
                const double n_d = static_cast<double>(grid.dims[d]);
                // Also rejects NaN (every comparison with NaN is false)
                if (!(u + k_tile_margin >= 0.0 && u - k_tile_margin < n_d)) {
                    inside = false;
                    break;
                }
                const double lo = std::min(std::max(std::floor(u - k_tile_margin), 0.0), n_d - 1);
                const double hi = std::min(std::max(std::floor(u + k_tile_margin), 0.0), n_d - 1);
                t_lo[d] = static_cast<std::int64_t>(lo) / tiles.tile;
                t_hi[d] = static_cast<std::int64_t>(hi) / tiles.tile;
            }
            if (!inside) {
                continue;
            }
            for (std::int64_t tz = t_lo[2]; tz <= t_hi[2]; ++tz) {
                for (std::int64_t ty = t_lo[1]; ty <= t_hi[1]; ++ty) {
                    for (std::int64_t tx = t_lo[0]; tx <= t_hi[0]; ++tx) {
                        chunk_marks.push_back((tz * tiles.count[1] + ty) * tiles.count[0] + tx);
                    }
                }
            }
        }
        std::sort(chunk_marks.begin(), chunk_marks.end());
        chunk_marks.erase(std::unique(chunk_marks.begin(), chunk_marks.end()), chunk_marks.end());
        merged.clear();
        std::set_union(touched.begin(), touched.end(), chunk_marks.begin(), chunk_marks.end(),
                       std::back_inserter(merged));
        touched.swap(merged);
    });
    return touched;
}

/** @brief The tiles this rank read, ready for the device lookup. */
struct TileStore {
    mfem::Array<int> ids;     ///< tile data back to back
    mfem::Array<int> offsets; ///< dense offset table over the touched tile box
    std::array<std::int64_t, 3> t0 = {0, 0, 0};
    std::array<std::int64_t, 3> range = {0, 0, 0};
    std::int64_t num_tiles = 0;
    std::int64_t num_voxels = 0;
};

/**
 * @brief Pass 1 result -> data: builds the offset table over the box of touched tiles, reads
 * every touched tile with one hyperslab each (ReadVoxelBoxes, one file open), checks the ids
 * (< 2^31) and stores them (ids <= 0 as 0). Throws on a local failure (the caller makes it
 * collective).
 */
TileStore ReadTiles(const VoxelMapParameters& map, const VoxelGridInfo& grid,
                    const TileGrid& tiles, const std::vector<std::int64_t>& touched) {
    TileStore store;
    if (touched.empty()) {
        return store;
    }
    // Step 1: tile coordinates, the touched tile box, and each tile's voxel box
    const std::int64_t inf = std::numeric_limits<std::int64_t>::max();
    std::array<std::int64_t, 3> t_min = {inf, inf, inf};
    std::array<std::int64_t, 3> t_max = {-1, -1, -1};
    std::vector<VoxelBox> boxes;
    boxes.reserve(touched.size());
    std::int64_t total = 0;
    for (const std::int64_t id : touched) {
        const std::array<std::int64_t, 3> t = {id % tiles.count[0],
                                               (id / tiles.count[0]) % tiles.count[1],
                                               id / (tiles.count[0] * tiles.count[1])};
        VoxelBox box;
        std::int64_t count = 1;
        for (int d = 0; d < 3; ++d) {
            t_min[d] = std::min(t_min[d], t[d]);
            t_max[d] = std::max(t_max[d], t[d]);
            box.lo[d] = t[d] * tiles.tile;
            box.hi[d] = std::min(grid.dims[d], box.lo[d] + tiles.tile);
            count *= box.hi[d] - box.lo[d];
        }
        boxes.push_back(box);
        total += count;
    }
    const std::int64_t int_max = std::numeric_limits<int>::max();
    if (total > int_max) {
        throw std::runtime_error("Voxel labeler: the voxel tiles under this rank's elements hold " +
                                 std::to_string(total) +
                                 " voxels, more than an int index allows; use more ranks (each "
                                 "rank reads only the tiles under its own elements)");
    }
    std::int64_t table_size = 1;
    for (int d = 0; d < 3; ++d) {
        store.t0[d] = t_min[d];
        store.range[d] = t_max[d] - t_min[d] + 1;
        table_size *= store.range[d];
    }
    if (table_size > int_max) {
        throw std::runtime_error("Voxel labeler: the tile offset table of this rank would need " +
                                 std::to_string(table_size) +
                                 " entries; use a larger tile_size or more ranks");
    }

    // Step 2: one hyperslab per tile, then the offset table and the int copy
    const std::vector<std::int64_t> raw = ReadVoxelBoxes(map, grid, boxes);
    store.offsets.SetSize(static_cast<int>(table_size));
    int* offset_host = store.offsets.HostWrite();
    for (std::int64_t k = 0; k < table_size; ++k) {
        offset_host[k] = -1;
    }
    store.ids.SetSize(static_cast<int>(total));
    int* id_host = store.ids.HostWrite();
    std::int64_t offset = 0;
    for (const VoxelBox& box : boxes) {
        std::array<std::int64_t, 3> rel{};
        std::int64_t count = 1;
        for (int d = 0; d < 3; ++d) {
            rel[d] = box.lo[d] / tiles.tile - store.t0[d];
            count *= box.hi[d] - box.lo[d];
        }
        offset_host[(rel[2] * store.range[1] + rel[1]) * store.range[0] + rel[0]] =
            static_cast<int>(offset);
        for (std::int64_t v = offset; v < offset + count; ++v) {
            const std::int64_t value = raw[static_cast<std::size_t>(v)];
            if (value > int_max) {
                throw std::runtime_error("Voxel labeler: the voxel map holds the grain id " +
                                         std::to_string(value) +
                                         " (>= 2^31); ids must fit a 32-bit signed integer");
            }
            id_host[v] = (value > 0) ? static_cast<int>(value) : k_no_vote;
        }
        offset += count;
    }
    store.num_tiles = static_cast<std::int64_t>(boxes.size());
    store.num_voxels = total;
    return store;
}

/** @brief Sum of an int64 over all ranks (collective). */
std::int64_t GlobalSum(std::int64_t value, MPI_Comm comm) {
    std::int64_t total = 0;
    MPI_Allreduce(&value, &total, 1, MPI_INT64_T, MPI_SUM, comm);
    return total;
}

/**
 * @brief Steps 1-3 on this rank (no communication): mark tiles from the sample points, read
 * them, then look up and vote. Throws on any local failure; the caller turns that into a
 * collective error.
 *
 * @param element_feature resized to NE: voted feature id or 0.
 * @param result          receives num_tiles_read / num_voxels_read (local).
 * @param num_outside     points outside the grid (local).
 * @param any_bad         true if a point fell on a tile that was not read (consistency check).
 */
void VoteLocally(mfem::ParMesh& pmesh, const VoxelMapParameters& map,
                 const VoxelLabelParameters& params, const VoxelGridInfo& grid,
                 mfem::Array<int>& element_feature, VoxelLabels& result,
                 std::int64_t& num_outside, bool& any_bad) {
    const int ne = pmesh.GetNE();
    element_feature.SetSize(ne);
    num_outside = 0;
    any_bad = false;

    // Step 1: tiles touched by the actual sample points
    const SamplePoints points(pmesh, params);
    TileGrid tiles;
    tiles.tile = params.tile_size;
    for (int d = 0; d < 3; ++d) {
        tiles.count[d] = (grid.dims[d] + tiles.tile - 1) / tiles.tile;
    }
    const std::vector<std::int64_t> touched = MarkTiles(points, grid, tiles);

    // Step 2: read them (nothing is read on a rank without elements or entirely off the grid)
    const TileStore store = ReadTiles(map, grid, tiles, touched);
    result.num_tiles_read = store.num_tiles;
    result.num_voxels_read = store.num_voxels;
    if (ne == 0) {
        return;
    }
    TileLookup lookup{};
    for (int d = 0; d < 3; ++d) {
        lookup.origin[d] = grid.origin[d];
        lookup.spacing[d] = grid.spacing[d];
        lookup.dims_d[d] = static_cast<double>(grid.dims[d]);
        lookup.dims[d] = static_cast<long long>(grid.dims[d]);
        lookup.t0[d] = static_cast<long long>(store.t0[d]);
        lookup.range[d] = static_cast<long long>(store.range[d]);
    }
    lookup.tile = static_cast<long long>(tiles.tile);
    // With no tile read every range is 0, so the lookup returns before touching the (null)
    // pointers.
    lookup.offsets = (store.offsets.Size() > 0) ? store.offsets.Read() : nullptr;
    lookup.ids = (store.ids.Size() > 0) ? store.ids.Read() : nullptr;

    // Step 3: second pass over the same points: lookup and vote, chunk by chunk
    mfem::Array<int> element_outside(ne);
    mfem::Array<int> element_bad(ne);
    mfem::Array<int> point_label;
    const int nq = points.PointsPerElement();
    points.ForEachChunk([&](int begin, int end, const mfem::Vector& x, const mfem::Vector& y,
                            const mfem::Vector& z, const mfem::Vector& weight) {
        LabelPoints(lookup, x, y, z, point_label);
        VoteChunk(begin, end - begin, nq, point_label, weight, element_feature, element_outside,
                  element_bad);
    });

    // Local statistics (host)
    const int* outside = element_outside.HostRead();
    const int* bad = element_bad.HostRead();
    for (int e = 0; e < ne; ++e) {
        num_outside += outside[e];
        any_bad = any_bad || (bad[e] != 0);
    }
}

/**
 * @brief Step 4: fills unvoted elements (feature 0) in synchronous rounds from their face
 * neighbors (see the header). Collective.
 *
 * Each round: exchange the current labels with the neighboring ranks (MeshAdjacency::
 * ExchangeLabels); for every unassigned element count, per feature id, the entries of its
 * neighbor list (one entry per shared face) whose neighbor carries that id; take the id with the
 * most faces (ties: lowest id); write into a copy, so all decisions of a round use the previous
 * round's labels; sum the newly assigned elements over all ranks; stop when that sum is 0.
 * The adjacency is built only if some rank has an unvoted element.
 */
void FillFromNeighbors(mfem::ParMesh& pmesh, MPI_Comm comm, mfem::Array<int>& element_feature,
                       VoxelLabels& result) {
    const int ne = element_feature.Size();
    int* current = element_feature.HostReadWrite();

    // Global count of unvoted elements; nothing to do (and no adjacency to build) if zero
    std::int64_t local_unassigned = 0;
    for (int e = 0; e < ne; ++e) {
        if (current[e] <= 0) {
            ++local_unassigned;
        }
    }
    std::int64_t unassigned = GlobalSum(local_unassigned, comm);
    if (unassigned == 0) {
        return;
    }

    const MeshAdjacency adjacency(pmesh);
    const int* offsets = adjacency.Offsets().HostRead();
    const int* neighbors = adjacency.Neighbors().HostRead();
    mfem::Array<int> ghost;
    std::vector<int> next(static_cast<std::size_t>(ne));

    while (unassigned > 0) {
        // Labels of the ghost elements (neighbors owned by other ranks), from the last round
        adjacency.ExchangeLabels(element_feature, ghost);
        current = element_feature.HostReadWrite();
        const int* ghost_label = ghost.HostRead();

        // Decide every unassigned element from the previous round's labels only
        std::int64_t local_assigned = 0;
        for (int e = 0; e < ne; ++e) {
            next[static_cast<std::size_t>(e)] = current[e];
            if (current[e] > 0) {
                continue;
            }
            int best = 0;
            int best_faces = 0;
            for (int k = offsets[e]; k < offsets[e + 1]; ++k) {
                const int n = neighbors[k];
                const int label = (n < ne) ? current[n] : ghost_label[n - ne];
                if (label <= 0) {
                    continue;
                }
                // Number of shared faces with neighbors of this label (a handful of entries)
                int faces = 0;
                for (int m = offsets[e]; m < offsets[e + 1]; ++m) {
                    const int nm = neighbors[m];
                    const int label_m = (nm < ne) ? current[nm] : ghost_label[nm - ne];
                    if (label_m == label) {
                        ++faces;
                    }
                }
                if (faces > best_faces || (faces == best_faces && label < best)) {
                    best = label;
                    best_faces = faces;
                }
            }
            if (best > 0) {
                next[static_cast<std::size_t>(e)] = best;
                ++local_assigned;
            }
        }

        // Did this round change anything anywhere?
        const std::int64_t assigned = GlobalSum(local_assigned, comm);
        if (assigned == 0) {
            break;
        }
        for (int e = 0; e < ne; ++e) {
            current[e] = next[static_cast<std::size_t>(e)];
        }
        result.num_filled += assigned;
        result.fill_rounds += 1;
        unassigned -= assigned;
    }

    // Anything left touches no voxel, directly or through neighbors: same error on every rank
    if (unassigned > 0) {
        throw std::runtime_error(
            "Voxel labeler: " + std::to_string(unassigned) +
            " element(s) could not be given a grain: they lie outside the voxel grid or over "
            "unassigned voxels (id <= 0) only, and are not connected through faces to any "
            "element that received a grain (check origin, spacing, units_scale and translate)");
    }
}

} // namespace

/*
 * LabelElementsFromVoxels
 *
 * Steps (see the header):
 *  0. Parameter and grid checks, identical on every rank (so every rank throws together).
 *  1-3. Tile marking from the sample points, tile reads, lookup and vote (VoteLocally); any
 *     local exception (read error, id >= 2^31, too many voxels) becomes one collective error
 *     (ThrowIfAnyRankFailed), as does a point on a tile that was not read.
 *  4. Global count of points outside the grid, then the fill rounds (FillFromNeighbors).
 */
VoxelLabels LabelElementsFromVoxels(mfem::ParMesh& pmesh,
                                    MPI_Comm comm,
                                    const VoxelMapParameters& map,
                                    const VoxelGridInfo& grid,
                                    const VoxelLabelParameters& params) {
    // Step 0: parameters and grid
    if (params.sampling != SamplingMode::CENTROID &&
        params.sampling != SamplingMode::QUADRATURE_VOTE) {
        throw std::invalid_argument("Voxel labeler: unknown sampling mode");
    }
    if (params.sampling == SamplingMode::QUADRATURE_VOTE &&
        (params.vote_levels < 0 || params.vote_levels > 3)) {
        throw std::invalid_argument("Voxel labeler: vote_levels must be in 0..3");
    }
    if (params.max_points_per_chunk < 1) {
        throw std::invalid_argument("Voxel labeler: max_points_per_chunk must be >= 1");
    }
    if (params.tile_size < 1) {
        throw std::invalid_argument("Voxel labeler: tile_size must be >= 1");
    }
    for (int d = 0; d < 3; ++d) {
        if (grid.dims[d] < 1 || !(grid.spacing[d] > 0.0) || !std::isfinite(grid.spacing[d]) ||
            !std::isfinite(grid.origin[d])) {
            throw std::invalid_argument("Voxel labeler: invalid voxel grid (use the result of "
                                        "ReadVoxelGridInfo)");
        }
    }

    // Steps 1-3: local work, errors made collective
    VoxelLabels result;
    std::int64_t local_outside = 0;
    bool any_bad = false;
    std::string error;
    try {
        VoteLocally(pmesh, map, params, grid, result.element_feature, result, local_outside,
                    any_bad);
    } catch (const std::exception& e) {
        error = e.what();
        if (error.empty()) {
            error = "Voxel labeler: unknown error";
        }
    }
    ThrowIfAnyRankFailed(comm, error);
    if (any_bad) {
        error = "Voxel labeler: internal error, a sample point lies on a voxel tile that this "
                "rank did not read (the tile marking and lookup passes disagree)";
    }
    ThrowIfAnyRankFailed(comm, error);

    // Step 4: statistics and fill
    result.num_outside_points = GlobalSum(local_outside, comm);
    FillFromNeighbors(pmesh, comm, result.element_feature, result);
    return result;
}

/*
 * CompactFeatureLabels
 *
 * Steps (see the header):
 *  1. Every local id must be >= 1 (checked collectively).
 *  2. Sorted distinct local ids.
 *  3. MPI_Allgather of the list lengths, exclusive prefix sum of the lengths for the receive
 *     offsets (each rank's list starts after the lists of the lower ranks), MPI_Allgatherv of the
 *     lists.
 *  4. Sort and remove duplicates: the global ascending list of used ids = grain_feature, the same
 *     on every rank whatever the rank count.
 *  5. element_grain[e] = 1 + position of element_feature[e] in that list, by binary search in an
 *     mfem::forall kernel.
 */
void CompactFeatureLabels(MPI_Comm comm,
                          const mfem::Array<int>& element_feature,
                          mfem::Array<int>& element_grain,
                          std::vector<std::int64_t>& grain_feature) {
    const int ne = element_feature.Size();
    const int* feature = (ne > 0) ? element_feature.HostRead() : nullptr;

    // Step 1: validity
    bool local_bad = false;
    for (int e = 0; e < ne; ++e) {
        local_bad = local_bad || feature[e] < 1;
    }
    ThrowIfAnyRank<std::invalid_argument>(comm, local_bad,
                                          "CompactFeatureLabels: a feature id < 1 on at least "
                                          "one rank (every element must have a grain)");

    // Step 2: sorted distinct local ids
    std::vector<int> local_ids(feature, feature + ne);
    std::sort(local_ids.begin(), local_ids.end());
    local_ids.erase(std::unique(local_ids.begin(), local_ids.end()), local_ids.end());

    // Step 3: gather all lists on every rank
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    int local_count = static_cast<int>(local_ids.size());
    std::vector<int> counts(static_cast<std::size_t>(nranks), 0);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    std::int64_t total = 0;
    for (int r = 0; r < nranks; ++r) {
        displs[static_cast<std::size_t>(r)] = static_cast<int>(total);
        total += counts[static_cast<std::size_t>(r)];
    }
    if (total > std::numeric_limits<int>::max()) {
        throw std::runtime_error("CompactFeatureLabels: too many distinct ids to gather");
    }
    std::vector<int> all_ids(static_cast<std::size_t>(total), 0);
    MPI_Allgatherv(local_ids.data(), local_count, MPI_INT, all_ids.data(), counts.data(),
                   displs.data(), MPI_INT, comm);

    // Step 4: global ascending distinct list
    std::sort(all_ids.begin(), all_ids.end());
    all_ids.erase(std::unique(all_ids.begin(), all_ids.end()), all_ids.end());
    grain_feature.assign(all_ids.begin(), all_ids.end());

    // Step 5: grain id of every element by binary search of the table
    element_grain.SetSize(ne);
    if (ne == 0) {
        return;
    }
    const int num_grains = static_cast<int>(all_ids.size());
    mfem::Array<int> table(num_grains);
    for (int g = 0; g < num_grains; ++g) {
        table[g] = all_ids[static_cast<std::size_t>(g)];
    }
    const int* sorted = table.Read();
    const int* in = element_feature.Read();
    int* out = element_grain.Write();
    mfem::forall(ne, [=] MFEM_HOST_DEVICE(int e) {
        // The id is in the table (it came from this rank's own list): find its position
        int low = 0;
        int high = num_grains - 1;
        const int id = in[e];
        while (low < high) {
            const int mid = low + (high - low) / 2;
            if (sorted[mid] < id) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        out[e] = low + 1;
    });
}

} // namespace exaconstit::microstructure
