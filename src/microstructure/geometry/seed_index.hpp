#pragma once

#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include "axom/primal/geometry/BoundingBox.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/spin/BVH.hpp"

#include <memory>
#include <vector>

/**
 * @file seed_index.hpp
 * @brief Exact nearest-seed search: for each query point (an element centroid, a quadrature
 * point, ...) find the seed with the smallest metric value f_i(x), i.e. the grain the point
 * belongs to, and optionally the second smallest.
 *
 * **Why a search structure.** Comparing every point with every seed costs (number of points) x
 * (number of seeds) metric evaluations, which is too slow for millions of elements and
 * thousands of grains. SeedIndex stores the seeds in a BVH (bounding volume hierarchy: a tree of
 * nested bounding boxes that lets us skip seeds that are provably too far away). For a query
 * point x the tree is walked from the root; a box is opened only if the metric lower bound
 * (metric_functors.hpp, LowerBound) for the seeds inside it is not larger than the best f found
 * so far. Because the bound is a true lower bound, no skipped seed could have won: the answer is
 * *exactly* the one a brute-force loop over all seeds would give, including the tie-break by
 * lowest seed index, at a cost of roughly log(number of seeds) per query.
 *
 * **Bins for weighted metrics.** The lower bound of a box needs the largest weight (and, for the
 * Quadric metric, the smallest tensor eigenvalue) of the seeds in that box. Axom's BVH cannot
 * store such per-node data on the device, so seeds are instead split into up to
 * SEED_INDEX_MAX_BINS groups ("bins") by their weight/eigenvalue, with the same number of seeds
 * in each (quantile bins), and every bin gets its own BVH with a tight BinBound. A query visits
 * the bins in order of the lower bound of their root box, carrying the best value found so far
 * from bin to bin, and stops as soon as the next bin's root bound exceeds it. Voronoi seeds
 * (no weights) always use one bin.
 *
 * **Box-aware bound for one global quadric tensor.** The traversal passes each box's corners,
 * not only its distance to the query, to the metric's LowerBound. The Quadric metric with one
 * global tensor A uses them for its projected-interval bound (QuadricMetric in
 * metric_functors.hpp): the box is projected onto each row of the factor M of A, and the
 * distance of the query to those projected intervals bounds f. This keeps the search fast for
 * semidefinite A (columnar grains A = I - a a^T, lamellae A = n n^T), where the smallest
 * eigenvalue is 0 and the distance-only bound would make every query visit every seed.
 * The optional leaf_visits output of FindBest measures this pruning.
 *
 * **Coincident seeds.** Seeds with exactly the same coordinates are stored as one BVH leaf: the
 * lowest index is the representative and the others are chained behind it (ascending index).
 * Whenever the traversal reaches that leaf, the whole chain is evaluated together. This gives one
 * leaf per distinct position and makes every member of the chain a candidate, so the result
 * (including the exact second best, which for equal weights is the duplicate itself at equal f)
 * is exactly the brute-force answer for every metric, including per-seed quadric tensors where
 * neither of two coincident seeds wins everywhere. Their count is reported in SeedIndexReport.
 * (Tree depth is not the reason: Axom's 30-bit Morton codes break ties by sorted position
 * (build_radix_tree.hpp:262-276), so the depth is at most about 30 + log2(N) <= 62 with or
 * without duplicates, within the fixed 64-entry traversal stack of bvh_traverse.hpp.)
 *
 * **Execution.** SeedIndex is templated on an Axom execution space (axom::SEQ_EXEC, OMP_EXEC,
 * CUDA_EXEC<256>, HIP_EXEC<256>), normally chosen with DispatchAxomExec (exec_dispatch.hpp) to
 * match the configured mfem::Device, and on a metric functor type (VoronoiMetric, ...), normally
 * chosen with DispatchMetric. Explicit instantiations exist in seed_index.cpp for SEQ_EXEC (and
 * OMP/CUDA/HIP when Axom and the compiler provide them) with each of the four metrics.
 *
 * **Device precondition.** Seed copies and query/result arrays are mfem::Vector / mfem::Array
 * data accessed with Read(on_device)/Write(on_device). MFEM returns *device* pointers only when
 * the configured mfem::Device has a GPU backend; otherwise it returns host pointers. A GPU
 * execution space (CUDA_EXEC/HIP_EXEC) therefore requires an mfem::Device with a CUDA or HIP
 * backend (mfem::Backend::CUDA_MASK | HIP_MASK; the "debug" device does not count). Build,
 * FindBest and FindWithin check this and throw std::invalid_argument otherwise. DispatchAxomExec
 * picks matching spaces automatically.
 */

namespace exaconstit::microstructure {

/** @brief Largest number of parameter bins (one BVH each). Kept small because the per-bin
 * traversal data is passed by value into GPU kernels. */
inline constexpr int SEED_INDEX_MAX_BINS = 8;

/**
 * @brief Statistics of a SeedIndex build, for logging and tests. Host data.
 */
struct SeedIndexReport {
    /** @brief Number of seeds N in the SeedSet. */
    int num_seeds = 0;
    /** @brief Number of BVH leaves, i.e. distinct seed positions (N - num_coincident). */
    int num_indexed = 0;
    /**
     * @brief Number of seeds that share their exact coordinates with a lower-index seed. They are
     * not dropped: they are evaluated together with their representative.
     */
    int num_coincident = 0;
    /** @brief Number of parameter bins (BVHs) actually used, 1..SEED_INDEX_MAX_BINS. */
    int num_bins = 0;
};

/**
 * @brief Exact nearest / second-nearest seed search for one metric, on one execution space.
 *
 * Typical use:
 * @code
 *   SeedIndex<axom::SEQ_EXEC, LaguerreMetric> index;
 *   index.Build(seeds, MetricSetup{}, 8, AxomAllocatorIdForCurrentDevice());
 *   mfem::Array<BestPair> best;
 *   index.FindBest(samples.cx, samples.cy, samples.cz, best, false);
 *   // best[e].id1 = 0-based seed index of the grain of sample e
 * @endcode
 *
 * Not collective: every rank builds its own copy from the (replicated) SeedSet and queries its
 * own points, and the results do not depend on the rank count. Not copyable (owns the BVHs).
 *
 * @tparam ExecSpace Axom execution space of the BVH build and query kernels.
 * @tparam Metric    one of VoronoiMetric, LaguerreMetric, QuadricMetric, AdditiveMetric.
 */
template <typename ExecSpace, typename Metric>
class SeedIndex {
public:
    /** @brief The Axom BVH type, one per bin. */
    using BVHType = axom::spin::BVH<3, ExecSpace, double>;

    /** @brief Creates an empty index; call Build before any query. Host only. */
    SeedIndex();
    /** @brief Releases the seed copies and the BVHs (host and device memory). Host only. */
    ~SeedIndex();
    SeedIndex(const SeedIndex&) = delete;
    SeedIndex& operator=(const SeedIndex&) = delete;

    /**
     * @brief Copies the seeds into memory owned by the index and builds the per-bin BVHs.
     *
     * Host function that launches kernels on ExecSpace. Not collective. Can be called again to
     * rebuild with new seeds (previous data is released).
     *
     * @param seeds        seed positions (x, y, z; size N >= 1) and, depending on the metric,
     *                     weight (size N, or empty = all zero; ignored by Voronoi) and shape
     *                     (Quadric with setup.per_seed_shape: 6 entries per seed, see
     *                     SeedSet::shape). Other SeedSet fields are not used.
     * @param setup        metric parameters not stored per seed (Quadric global tensor).
     * @param num_bins     requested number of parameter bins, >= 1; capped at
     *                     SEED_INDEX_MAX_BINS and at the number of distinct positions; Voronoi,
     *                     and any metric whose bin parameter is the same for all seeds, uses 1.
     *                     Performance only: the result does not depend on it.
     * @param allocator_id Umpire/Axom allocator for the BVH arrays; must be accessible from
     *                     ExecSpace (use AxomAllocatorIdForCurrentDevice()).
     * @throws std::invalid_argument if N == 0, N > 2^31 - 1, array sizes are inconsistent, a
     *         coordinate or weight is not finite, num_bins < 1, or a quadric tensor is invalid
     *         (see ComputeQuadricFactor), or if ExecSpace runs on a GPU but the mfem::Device has
     *         no CUDA/HIP backend (see "Device precondition" in the file comment).
     */
    void Build(const SeedSet& seeds, const MetricSetup& setup, int num_bins, int allocator_id);

    /**
     * @brief For every query point q, the seed with the smallest f (ties: lowest index) and,
     * if requested, the second smallest.
     *
     * Exact: the result equals a brute-force loop over all seeds with the same Metric::Eval.
     * Host function launching one kernel on ExecSpace; not collective.
     *
     * @param qx, qy, qz    query coordinates, same size nq (mesh length units). Read with
     *                      Read(on_device) where on_device matches ExecSpace.
     * @param out           resized to nq; out[q] = {id1, id2, f1, f2} with 0-based seed
     *                      indices of the SeedSet given to Build. Written on the device when
     *                      ExecSpace runs on the device (use HostRead() to inspect on the host).
     * @param exact_second  if true, (id2, f2) is the exact second-best pair (ties ordered by
     *                      index; id2 = -1 and f2 = DBL_MAX when N == 1). If false, id2 = -1 and
     *                      f2 holds the running second value of the pruned search, which is not
     *                      the true second best and must not be used; the search is then somewhat
     *                      faster.
     *
     * id1 = -1 (and f1 = DBL_MAX) when no f is comparable, i.e. for a non-finite query
     * coordinate (every comparison with NaN is false, so no seed is ever accepted). Callers
     * should check for it.
     *
     * @param leaf_visits   optional diagnostic (default nullptr = not computed): resized to nq,
     *                      (*leaf_visits)[q] = number of BVH leaves (distinct seed positions)
     *                      the pruned search of query q reached. Equals the number of distinct
     *                      seed positions when nothing is pruned. Written like `out`.
     *
     * @pre Build was called; qx, qy, qz have equal sizes.
     * @throws std::invalid_argument if ExecSpace runs on a GPU but the mfem::Device has no
     *         CUDA/HIP backend.
     */
    void FindBest(const mfem::Vector& qx, const mfem::Vector& qy, const mfem::Vector& qz,
                  mfem::Array<BestPair>& out, bool exact_second,
                  mfem::Array<int>* leaf_visits = nullptr) const;

    /**
     * @brief For every query point q, the seed with the smallest f (ties: lowest index) among
     * the seeds that are *not* on q's exclusion list.
     *
     * Used by the connected-cell labeler (labeling/connected_labeler.hpp): an element whose
     * nearest seed's cell does not reach it through material puts that seed on its list and asks
     * for the next best one.
     *
     * The exclusion lists are stored in CSR form (compressed rows: all lists back to back in one
     * array, with an offset array saying where each list starts): the list of query q is
     * ids[offsets[q] .. offsets[q+1] - 1]. Lists are short (a few entries), in any order, and may
     * repeat an entry or be empty (then the result equals FindBest's id1/f1).
     *
     * Exact, with the same tie-break as FindBest: the result equals a brute-force loop over the
     * non-excluded seeds with the same Metric::Eval. Excluded seeds are skipped only where seeds
     * are evaluated (the BVH leaves); the box lower bounds that decide which branches to open are
     * unchanged and are compared with the best value among non-excluded seeds found so far, so no
     * branch holding a better non-excluded seed is ever skipped. A coincident seed (same position
     * as a lower-index seed) is excluded only if its own index is on the list.
     *
     * Host function launching one kernel on ExecSpace (the list is scanned with a plain loop
     * inside the kernel); not collective.
     *
     * @param qx, qy, qz query coordinates, size nq each (mesh length units). Read with
     *                   Read(on_device) where on_device matches ExecSpace.
     * @param offsets    size nq + 1, offsets[0] = 0, non-decreasing. Read on ExecSpace (its last
     *                   entry is also read on the host for the size check).
     * @param ids        size >= offsets[nq]: 0-based seed indices of the lists. Read on
     *                   ExecSpace.
     * @param out        resized to nq; out[q] = {id1, -1, f1, f2}: id1 is the best non-excluded
     *                   seed (-1 with f1 = DBL_MAX if every seed is excluded or the query point is
     *                   not finite); id2 is always -1 and f2 is a running value of the pruned
     *                   search that must not be used. Written on ExecSpace.
     * @pre Build was called.
     * @throws std::invalid_argument if offsets/ids have inconsistent sizes, or if ExecSpace runs on
     *         a GPU but the mfem::Device has no CUDA/HIP backend.
     */
    void FindBestExcluding(const mfem::Vector& qx, const mfem::Vector& qy, const mfem::Vector& qz,
                           const mfem::Array<int>& offsets, const mfem::Array<int>& ids,
                           mfem::Array<BestPair>& out) const;

    /**
     * @brief Range query: for every query point q, all seeds with f_i(x_q) <= f_max.
     *
     * Uses the same pruned traversal in two kernels (count, then fill), with the prefix sum and
     * a per-query sort on the host, so it is meant for moderate output sizes (validation of
     * boundary projections, tests). Not collective.
     *
     * @param qx, qy, qz query coordinates, size nq each.
     * @param f_max      threshold on the metric value (metric units).
     * @param offsets    resized to nq + 1; the seeds of query q are ids[offsets[q] ..
     *                   offsets[q+1] - 1]. Host data on return.
     * @param ids        0-based seed indices, ascending within each query. Host data on return.
     * @pre Build was called.
     * @throws std::invalid_argument if ExecSpace runs on a GPU but the mfem::Device has no
     *         CUDA/HIP backend.
     */
    void FindWithin(const mfem::Vector& qx, const mfem::Vector& qy, const mfem::Vector& qz,
                    double f_max, mfem::Array<int>& offsets, mfem::Array<int>& ids) const;

    /**
     * @brief The metric functor, pointing at the index's own seed copies. Its pointers are
     * valid in ExecSpace's memory space (host pointers for SEQ/OMP) for the lifetime of the
     * index or until the next Build. Useful for evaluating f or grad f of a known seed.
     *
     * Warning: for GPU execution spaces the pointers are device pointers, so GetMetric().Eval /
     * Grad must only be called inside kernels on that device, never directly on the host.
     */
    const Metric& GetMetric() const {
        return m_metric;
    }

    /** @brief Build statistics. Host. */
    const SeedIndexReport& GetReport() const {
        return m_report;
    }

    /**
     * @brief Host array of size N: representative[i] is the lowest seed index with exactly the
     * same coordinates as seed i (i itself if seed i is not coincident with a lower index).
     */
    const mfem::Array<int>& GetRepresentative() const {
        return m_representative;
    }

    /** @brief Parameter summary of bin b (0 <= b < GetReport().num_bins). Host. */
    const BinBound& GetBinBound(int b) const {
        return m_bin_bound[b];
    }

    /** @brief True after a successful Build. */
    bool IsBuilt() const {
        return m_built;
    }

private:
    /** @brief Seed copies (all N seeds, original order); read through m_metric in kernels. */
    mfem::Vector m_x, m_y, m_z, m_weight, m_factor;
    /** @brief Representatives (BVH leaves) grouped by bin: bin b holds
     * m_indexed_ids[m_bin_begin[b] .. m_bin_begin[b+1] - 1], ascending seed index. */
    mfem::Array<int> m_indexed_ids;
    /** @brief Size N: next seed with the same coordinates (ascending index), or -1. */
    mfem::Array<int> m_coincident_next;
    /** @brief Size N, host: see GetRepresentative(). */
    mfem::Array<int> m_representative;
    /** @brief One BVH per bin (unique_ptr because the BVH is neither copyable nor needed by
     * value). */
    std::vector<std::unique_ptr<BVHType>> m_bvh;
    /** @brief Per-bin parameter summaries and root boxes (host copies; passed into kernels by
     * value). */
    BinBound m_bin_bound[SEED_INDEX_MAX_BINS];
    axom::primal::BoundingBox<double, 3> m_bin_root[SEED_INDEX_MAX_BINS];
    int m_bin_begin[SEED_INDEX_MAX_BINS + 1] = {};
    /** @brief Device functor over the seed copies. */
    Metric m_metric{};
    SeedIndexReport m_report{};
    int m_allocator_id = 0;
    bool m_built = false;
};

} // namespace exaconstit::microstructure
