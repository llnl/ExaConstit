/*
 * Implementation of the exact nearest-seed search (seed_index.hpp).
 *
 * Terms used in this file:
 * - BVH (bounding volume hierarchy): a tree of nested bounding boxes that lets us skip seeds
 *   that are provably too far away. Axom's linear BVH (axom::spin::BVH) is built from one box
 *   per seed; a seed is a point, so its box has zero size.
 * - Traversal: walking the tree from the root. Axom's traverse_tree(point, leaf_action,
 *   predicate) (spin/policy/LinearBVH.hpp:82-103) opens a child box only if predicate(point,
 *   box) is true, visits the child whose box centroid is closer first, and calls
 *   leaf_action(leaf, leaf_nodes) for every leaf it reaches; leaf_nodes[leaf] is the index of
 *   the original box passed to the BVH build (bvh_traverse.hpp:186-191). The predicate is
 *   re-evaluated at every node, so it can compare against a "best value so far" that the leaf
 *   action keeps lowering, which prunes more and more of the tree as the search proceeds. The
 *   same pattern is used by Axom's distributed closest-point query
 *   (quest/detail/DistributedClosestPointImpl.hpp:1005-1041).
 * - Leaf actions are deferred: Axom keeps one found leaf and continues to the next before calling
 *   the action (bvh_traverse.hpp:160-192). The predicate therefore sometimes sees a best value
 *   that is not yet updated; that only opens a few extra boxes, it never skips a needed one.
 * - Execution space: the Axom tag type (SEQ_EXEC, CUDA_EXEC<256>, ...) that says where
 *   axom::for_all loops run. Kernels are written with AXOM_LAMBDA ([=] plus host/device
 *   annotations), capture only plain data by value, and contain no std:: containers,
 *   exceptions or virtual calls, so they compile for GPUs.
 */
#include "microstructure/geometry/seed_index.hpp"

#include "axom/core/Array.hpp"
#include "axom/core/execution/for_all.hpp"
#include "axom/primal/operators/squared_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <float.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace exaconstit::microstructure {

namespace detail {

using SeedPoint = axom::primal::Point<double, 3>;
using SeedBox = axom::primal::BoundingBox<double, 3>;

/**
 * @brief Everything a query kernel needs, as one plain-data struct passed by value into the
 * kernel: the metric functor, one Axom traverser per bin (the traverser only holds non-owning
 * views of the BVH arrays), the bin root boxes and bounds, and the seed-index arrays.
 *
 * Device-copyable; all pointers/views live in the execution space's memory.
 */
template <typename Metric, typename Traverser>
struct SearchView {
    Metric metric;                           ///< f_i(x) and its lower bounds
    int num_bins;                            ///< bins in use, 1..SEED_INDEX_MAX_BINS
    Traverser trav[SEED_INDEX_MAX_BINS];     ///< traverser of the BVH of bin b
    SeedBox root[SEED_INDEX_MAX_BINS];       ///< bounding box of all seeds of bin b
    BinBound bound[SEED_INDEX_MAX_BINS];     ///< weight/eigenvalue summary of bin b
    int bin_begin[SEED_INDEX_MAX_BINS + 1];  ///< bin b = indexed_ids[bin_begin[b]..[b+1])
    const int* indexed_ids;                  ///< BVH box j of bin b -> seed indexed_ids[begin+j]
    const int* coincident_next;              ///< next seed at the same position, or -1
};

/**
 * @brief Builds a SearchView from the index data. Every member is set in one aggregate
 * initialization: the per-bin arrays are expanded element by element with std::index_sequence
 * (I = 0 .. SEED_INDEX_MAX_BINS - 1), which is needed because Axom's traverser type has no
 * default constructor. Slots of unused bins get a traverser over empty views (never traversed)
 * and default boxes/bounds. Host only.
 *
 * @param bin_root, bin_bound arrays of SEED_INDEX_MAX_BINS entries; bin_begin has one more.
 */
template <typename Metric, typename BVHType, std::size_t... I>
SearchView<Metric, typename BVHType::TraverserType>
MakeSearchView(const Metric& metric, const std::vector<std::unique_ptr<BVHType>>& bvh,
               const SeedBox* bin_root, const BinBound* bin_bound, const int* bin_begin,
               const int* indexed_ids, const int* coincident_next, std::index_sequence<I...>) {
    using Traverser = typename BVHType::TraverserType;
    const int num_bins = static_cast<int>(bvh.size());
    const Traverser empty{axom::ArrayView<const SeedBox>{},
                          axom::ArrayView<const std::int32_t>{},
                          axom::ArrayView<const std::int32_t>{}};
    SearchView<Metric, Traverser> view{
        metric,
        num_bins,
        {(static_cast<int>(I) < num_bins ? bvh[I]->getTraverser() : empty)...},
        {bin_root[I]...},
        {bin_bound[I]...},
        {bin_begin[I]..., bin_begin[SEED_INDEX_MAX_BINS]},
        indexed_ids,
        coincident_next};
    return view;
}

/**
 * @brief Running best and second-best (f, seed) pairs of one query point, ordered by f and then
 * by seed index (the tie-break). Plain data, lives in registers inside the query kernel.
 */
struct RunningBest {
    double f1 = DBL_MAX;
    double f2 = DBL_MAX;
    int id1 = INT32_MAX; ///< INT32_MAX = "none yet", so any real seed index sorts before it
    int id2 = INT32_MAX;

    /** @brief True if (fa, ia) comes before (fb, ib): smaller f, or equal f and lower index. */
    MFEM_HOST_DEVICE static bool Before(double fa, int ia, double fb, int ib) {
        return (fa < fb) || (fa == fb && ia < ib);
    }

    /** @brief Inserts the candidate (f, id) into the best two. */
    MFEM_HOST_DEVICE void Offer(double f, int id) {
        if (Before(f, id, f1, id1)) {
            f2 = f1;
            id2 = id1;
            f1 = f;
            id1 = id;
        } else if (Before(f, id, f2, id2)) {
            f2 = f;
            id2 = id;
        }
    }
};

/**
 * @brief Pruned traversal of the BVH of bin b for the query point p.
 *
 * A box is opened when the metric lower bound for its seeds is <= `threshold`; `threshold` is
 * read through a reference at every node, so the caller's visit function may lower it while the
 * traversal runs. `<=` (not `<`) keeps boxes whose bound equals the current best: they may hold
 * a seed with the same f and a lower index, which wins the tie. Usually the round-off margin of the
 * lower bounds already makes a tied seed's bound slightly smaller than its f, but not always: the
 * margin is proportional to d^2 and to the bin's largest |weight|, so when the query lies exactly
 * on a seed (d^2 = 0) in a bin whose weights are all zero, the bound equals the tied f exactly and
 * `<=` is the only thing that keeps that seed. For every leaf reached,
 * visit(seed) is called with the representative seed index of that leaf.
 *
 * Host and device (called inside query kernels).
 */
template <typename View, typename Visit>
MFEM_HOST_DEVICE inline void TraverseBin(const View& view, int b, const SeedPoint& p,
                                         const double& threshold, Visit& visit) {
    const int* bin_ids = view.indexed_ids + view.bin_begin[b];
    const BinBound bound = view.bound[b];
    const auto& metric = view.metric;

    // Leaf action: leaf_nodes[leaf] is the index of the box in this bin's build array (it is not
    // the tree node number), which maps to the seed through bin_ids
    auto leaf_action = [&](std::int32_t leaf, const std::int32_t* leaf_nodes) {
        visit(bin_ids[leaf_nodes[leaf]]);
    };

    // Predicate: open the box unless even the most favorable seed in it cannot reach threshold.
    // The box corners are passed along with the squared distance because the global-tensor
    // Quadric metric uses them for its projected-interval bound (metric_functors.hpp); the other
    // metrics use only the squared distance.
    auto predicate = [&](const SeedPoint& q, const SeedBox& box) -> bool {
        const double sqdist = axom::primal::squared_distance(q, box);
        return metric.LowerBound(q.data(), box.getMin().data(), box.getMax().data(), sqdist,
                                 bound) <= threshold;
    };

    view.trav[b].traverse_tree(p, leaf_action, predicate);
}

/**
 * @brief Orders the bins of `view` by the lower bound of their root box for point p (insertion
 * sort; at most SEED_INDEX_MAX_BINS entries). Bins that are likely to contain the answer are
 * then searched first, so the running best is small early and later bins are pruned at the
 * root. Ties keep the bin order, so the visiting order is deterministic. Host and device.
 */
template <typename View>
MFEM_HOST_DEVICE inline void OrderBins(const View& view, const SeedPoint& p, double* root_lb,
                                       int* order) {
    for (int b = 0; b < view.num_bins; ++b) {
        const SeedBox& root = view.root[b];
        root_lb[b] = view.metric.LowerBound(p.data(), root.getMin().data(), root.getMax().data(),
                                            axom::primal::squared_distance(p, root),
                                            view.bound[b]);
        order[b] = b;
    }
    for (int k = 1; k < view.num_bins; ++k) {
        const int b = order[k];
        int j = k - 1;
        while (j >= 0 && root_lb[order[j]] > root_lb[b]) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = b;
    }
}

/**
 * @brief Query kernel of FindBest: one loop iteration per query point.
 *
 * Per point: order the bins by root lower bound; search them in that order with the pruning
 * threshold = current best f (or current second-best f when the exact second is wanted, since
 * a seed that beats the second-best still matters); stop at the first bin whose root bound
 * exceeds the threshold, because the bins are sorted and no later bin can do better. Every leaf
 * evaluates its representative seed and all seeds at the same position (coincident chain).
 * When `leaf_visits` is not null, leaf_visits[q] receives the number of BVH leaves (distinct
 * seed positions) the search of point q reached: a direct measure of how well the lower bound
 * prunes (N for no pruning at all, a few dozen for good pruning).
 */
template <typename ExecSpace, typename View>
void FindBestKernel(const View& view, int nq, const double* qx, const double* qy,
                    const double* qz, BestPair* out, bool exact_second, int* leaf_visits) {
    axom::for_all<ExecSpace>(
        nq, AXOM_LAMBDA(axom::IndexType q) {
            const double x = qx[q];
            const double y = qy[q];
            const double z = qz[q];
            const SeedPoint p = SeedPoint::make_point(x, y, z);

            // Step 1: visiting order of the bins
            double root_lb[SEED_INDEX_MAX_BINS];
            int order[SEED_INDEX_MAX_BINS];
            OrderBins(view, p, root_lb, order);

            // Step 2: running best and the pruning threshold derived from it
            RunningBest best;
            double threshold = DBL_MAX;
            int num_leaves = 0;
            auto visit = [&](int seed) {
                ++num_leaves;
                for (int s = seed; s >= 0; s = view.coincident_next[s]) {
                    best.Offer(view.metric.Eval(s, x, y, z), s);
                }
                threshold = exact_second ? best.f2 : best.f1;
            };

            // Step 3: search the bins in order, carrying the best value across bins
            for (int k = 0; k < view.num_bins; ++k) {
                const int b = order[k];
                if (root_lb[b] > threshold) {
                    break;
                }
                TraverseBin(view, b, p, threshold, visit);
            }

            // Step 4: store; "none" (INT32_MAX) becomes -1. id1 is "none" only if no f compared
            // smaller than DBL_MAX, e.g. a NaN query coordinate (all NaN comparisons are false)
            BestPair result;
            result.id1 = (best.id1 != INT32_MAX) ? best.id1 : -1;
            result.f1 = best.f1;
            result.id2 = (exact_second && best.id2 != INT32_MAX) ? best.id2 : -1;
            result.f2 = best.f2;
            out[q] = result;
            if (leaf_visits != nullptr) {
                leaf_visits[q] = num_leaves;
            }
        });
}

/**
 * @brief Range-query kernel of FindWithin. With `ids == nullptr` it counts, for each query, the
 * seeds with f <= f_max into counts[q]; otherwise it writes their indices to
 * ids[offsets[q] ...] (in traversal order; sorted later on the host). Both passes use the same
 * fixed threshold, so they visit exactly the same seeds.
 */
template <typename ExecSpace, typename View>
void WithinKernel(const View& view, int nq, const double* qx, const double* qy, const double* qz,
                  double f_max, int* counts, const int* offsets, int* ids) {
    axom::for_all<ExecSpace>(
        nq, AXOM_LAMBDA(axom::IndexType q) {
            const double x = qx[q];
            const double y = qy[q];
            const double z = qz[q];
            const SeedPoint p = SeedPoint::make_point(x, y, z);
            const double threshold = f_max;

            int found = 0;
            auto visit = [&](int seed) {
                for (int s = seed; s >= 0; s = view.coincident_next[s]) {
                    if (view.metric.Eval(s, x, y, z) <= f_max) {
                        if (ids != nullptr) {
                            ids[offsets[q] + found] = s;
                        }
                        ++found;
                    }
                }
            };
            for (int b = 0; b < view.num_bins; ++b) {
                TraverseBin(view, b, p, threshold, visit);
            }
            if (ids == nullptr) {
                counts[q] = found;
            }
        });
}

/**
 * @brief Checks the device precondition of seed_index.hpp: a GPU execution space needs an
 * mfem::Device with a CUDA or HIP backend, because only then does mfem Read/Write(true) return
 * device pointers (mfem general/device.hpp: GetMemoryClass, Backend::CUDA_MASK / HIP_MASK). The
 * DEBUG_DEVICE backend is deliberately not accepted: it only emulates device memory on the host.
 * Host only.
 *
 * @param on_device axom::execution_space<ExecSpace>::onDevice().
 * @param caller    name used in the error message.
 * @throws std::invalid_argument when the precondition is violated.
 */
inline void RequireMatchingMfemDevice(bool on_device, const char* caller) {
    const bool mfem_gpu =
        mfem::Device::Allows(mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK);
    if (on_device && !mfem_gpu) {
        throw std::invalid_argument(
            std::string(caller) +
            ": the Axom execution space runs on a GPU, but the mfem::Device has no CUDA/HIP "
            "backend, so MFEM arrays would hand host pointers to GPU kernels. Configure the "
            "mfem::Device for the GPU (ConfigureMfemDevice) or use a host execution space.");
    }
}

/** @brief Throws std::invalid_argument with a "SeedIndex::Build: " prefix. Host only. */
[[noreturn]] inline void BuildError(const std::string& what) {
    throw std::invalid_argument("SeedIndex::Build: " + what);
}

} // namespace detail

/*
 * Default constructor and destructor. Defined here (not in the header) so that the BVH type is
 * only instantiated in this file, together with the explicit instantiations at the end.
 */
template <typename ExecSpace, typename Metric>
SeedIndex<ExecSpace, Metric>::SeedIndex() = default;

template <typename ExecSpace, typename Metric>
SeedIndex<ExecSpace, Metric>::~SeedIndex() = default;

/*
 * Build: host preprocessing of the seeds, then one BVH per bin.
 *
 * Steps:
 *  1. Validate the input sizes and values.
 *  2. Per-seed metric parameters on the host: weight w_i, and for the Quadric metric the factor
 *     M_i of A_i = M_i^T M_i plus the extreme eigenvalues of A_i (ComputeQuadricFactor).
 *  3. Coincident seeds: sort seed indices by (x, y, z, index) so exactly equal positions are
 *     adjacent; the lowest index of each group becomes the representative (one BVH leaf), the
 *     others are chained behind it in ascending index (m_coincident_next) and always evaluated
 *     with it, so results stay exactly brute-force.
 *  4. Quantile bins: sort the representatives by a scalar "bin key" (the parameter that the
 *     lower bound depends on: largest weight of the group, or smallest eigenvalue for per-seed
 *     quadric tensors) and cut the sorted list into bins of equal count. Each bin gets a
 *     BinBound covering every seed of every group in it.
 *  5. Copy the seed data into mfem::Vectors owned by the index and create the metric functor from
 *     their Read(on_device) pointers (MFEM moves the data to the device when ExecSpace runs
 *     there).
 *  6. For each bin, fill an Axom array of zero-size boxes on ExecSpace and build the BVH
 *     (BVH::initialize, spin/BVH.hpp:425-479; it handles 1-box inputs by adding an invalid box,
 *     which the traversal skips).
 * The host sorts are O(N log N) and run once; N is at most a few million seeds.
 */
template <typename ExecSpace, typename Metric>
void SeedIndex<ExecSpace, Metric>::Build(const SeedSet& seeds, const MetricSetup& setup,
                                         int num_bins, int allocator_id) {
    m_built = false;
    m_bvh.clear();
    detail::RequireMatchingMfemDevice(axom::execution_space<ExecSpace>::onDevice(),
                                      "SeedIndex::Build");

    // ---- Step 1: validation ----
    const std::size_t n_size = seeds.Size();
    if (n_size == 0) {
        detail::BuildError("the seed set is empty");
    }
    if (n_size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        detail::BuildError("more than 2^31 - 1 seeds");
    }
    const int n = static_cast<int>(n_size);
    if (seeds.y.size() != n_size || seeds.z.size() != n_size) {
        detail::BuildError("x, y and z must have the same size");
    }
    if (num_bins < 1) {
        detail::BuildError("num_bins must be >= 1");
    }
    constexpr MetricType type = Metric::TYPE;
    const bool uses_weight = (type != MetricType::VORONOI);
    if (uses_weight && !seeds.weight.empty() && seeds.weight.size() != n_size) {
        detail::BuildError("weight must be empty or have one entry per seed");
    }
    const bool per_seed_shape = (type == MetricType::QUADRIC) && setup.per_seed_shape;
    if (per_seed_shape && seeds.shape.size() != 6 * n_size) {
        detail::BuildError("per-seed quadric shape needs 6 entries per seed in SeedSet::shape");
    }
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(seeds.x[i]) || !std::isfinite(seeds.y[i]) ||
            !std::isfinite(seeds.z[i])) {
            detail::BuildError("seed " + std::to_string(i) + " has a non-finite coordinate");
        }
    }

    // ---- Step 2: per-seed parameters (host) ----
    // weight[i]: 0 for Voronoi or when no weights are given
    std::vector<double> weight(n_size, 0.0);
    if (uses_weight && !seeds.weight.empty()) {
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(seeds.weight[i])) {
                detail::BuildError("seed " + std::to_string(i) + " has a non-finite weight");
            }
            weight[i] = seeds.weight[i];
        }
    }
    // Quadric factor (9 per seed, or 9 shared) and eigenvalue range per seed; mu = 1 otherwise
    std::vector<double> mu_min(n_size, 1.0);
    std::vector<double> mu_max(n_size, 1.0);
    std::vector<double> factor(9, 0.0);
    int factor_stride = 0;
    if (type == MetricType::QUADRIC) {
        if (per_seed_shape) {
            factor.assign(9 * n_size, 0.0);
            factor_stride = 9;
            for (int i = 0; i < n; ++i) {
                ComputeQuadricFactor(&seeds.shape[6 * i], &factor[9 * i], mu_min[i], mu_max[i]);
            }
        } else {
            double lo = 0.0;
            double hi = 0.0;
            ComputeQuadricFactor(setup.global_tensor.data(), factor.data(), lo, hi);
            std::fill(mu_min.begin(), mu_min.end(), lo);
            std::fill(mu_max.begin(), mu_max.end(), hi);
        }
    }

    // ---- Step 3: coincident seeds ----
    // Sort indices lexicographically by (x, y, z), then by index, so equal positions are adjacent
    // and each group starts with its lowest index
    std::vector<int> by_position(n_size);
    std::iota(by_position.begin(), by_position.end(), 0);
    std::sort(by_position.begin(), by_position.end(), [&](int a, int b) {
        if (seeds.x[a] != seeds.x[b]) {
            return seeds.x[a] < seeds.x[b];
        }
        if (seeds.y[a] != seeds.y[b]) {
            return seeds.y[a] < seeds.y[b];
        }
        if (seeds.z[a] != seeds.z[b]) {
            return seeds.z[a] < seeds.z[b];
        }
        return a < b;
    });

    m_representative.SetSize(n);
    m_coincident_next.SetSize(n);
    int* representative = m_representative.HostWrite();
    int* coincident_next = m_coincident_next.HostWrite();
    std::vector<int> reps; // representatives (one per distinct position)
    reps.reserve(n_size);
    int num_coincident = 0;
    for (int k = 0; k < n; ++k) {
        const int i = by_position[k];
        coincident_next[i] = -1;
        const bool same_as_previous = (k > 0) && seeds.x[i] == seeds.x[by_position[k - 1]] &&
                                      seeds.y[i] == seeds.y[by_position[k - 1]] &&
                                      seeds.z[i] == seeds.z[by_position[k - 1]];
        if (same_as_previous) {
            // Chain i behind the previous member (indices ascend within the group)
            const int prev = by_position[k - 1];
            coincident_next[prev] = i;
            representative[i] = representative[prev];
            ++num_coincident;
        } else {
            representative[i] = i;
            reps.push_back(i);
        }
    }
    const int num_reps = static_cast<int>(reps.size());

    // Per-group parameter summaries, indexed by representative: largest weight (and |weight|),
    // smallest / largest eigenvalue over all members of the group
    std::vector<BinBound> group_bound(n_size);
    for (const int r : reps) {
        BinBound g;
        g.w_max = -DBL_MAX;
        g.w_absmax = 0.0;
        g.mu_min = DBL_MAX;
        g.mu_max = 0.0;
        for (int s = r; s >= 0; s = coincident_next[s]) {
            g.w_max = std::max(g.w_max, weight[s]);
            g.w_absmax = std::max(g.w_absmax, std::fabs(weight[s]));
            g.mu_min = std::min(g.mu_min, mu_min[s]);
            g.mu_max = std::max(g.mu_max, mu_max[s]);
        }
        group_bound[r] = g;
    }

    // ---- Step 4: quantile bins ----
    // Bin key: what the lower bound depends on. Laguerre/Additive and global-tensor Quadric: the
    // weight; per-seed Quadric: the smallest eigenvalue (multiplicative weighting has w = 0 and
    // varies only the tensor); Voronoi: nothing.
    auto bin_key = [&](int r) -> double {
        if (type == MetricType::VORONOI) {
            return 0.0;
        }
        if (per_seed_shape) {
            return group_bound[r].mu_min;
        }
        return group_bound[r].w_max;
    };
    std::sort(reps.begin(), reps.end(), [&](int a, int b) {
        const double ka = bin_key(a);
        const double kb = bin_key(b);
        return (ka != kb) ? (ka < kb) : (a < b);
    });
    // One bin if the key is the same for every seed (nothing to separate)
    const bool key_varies = bin_key(reps.front()) != bin_key(reps.back());
    int bins = key_varies ? std::min({num_bins, SEED_INDEX_MAX_BINS, num_reps}) : 1;

    // Equal-count cut points; within a bin, representatives in ascending index (deterministic
    // BVH input order)
    for (int b = 0; b <= SEED_INDEX_MAX_BINS; ++b) {
        // slots past the last bin repeat the end, so they describe empty bins
        const int cut = std::min(b, bins);
        m_bin_begin[b] = static_cast<int>((static_cast<std::int64_t>(num_reps) * cut) / bins);
    }
    for (int b = 0; b < bins; ++b) {
        std::sort(reps.begin() + m_bin_begin[b], reps.begin() + m_bin_begin[b + 1]);
    }

    // Bin bounds: combine the group summaries of the bin
    for (int b = 0; b < bins; ++b) {
        BinBound bb;
        bb.w_max = -DBL_MAX;
        bb.w_absmax = 0.0;
        bb.mu_min = DBL_MAX;
        bb.mu_max = 0.0;
        for (int k = m_bin_begin[b]; k < m_bin_begin[b + 1]; ++k) {
            const BinBound& g = group_bound[reps[k]];
            bb.w_max = std::max(bb.w_max, g.w_max);
            bb.w_absmax = std::max(bb.w_absmax, g.w_absmax);
            bb.mu_min = std::min(bb.mu_min, g.mu_min);
            bb.mu_max = std::max(bb.mu_max, g.mu_max);
        }
        m_bin_bound[b] = bb;
    }

    // ---- Step 5: device copies of the seed data and the metric functor ----
    const bool on_device = axom::execution_space<ExecSpace>::onDevice();
    m_x.SetSize(n);
    m_y.SetSize(n);
    m_z.SetSize(n);
    m_weight.SetSize(n);
    m_factor.SetSize(static_cast<int>(factor.size()));
    {
        double* hx = m_x.HostWrite();
        double* hy = m_y.HostWrite();
        double* hz = m_z.HostWrite();
        double* hw = m_weight.HostWrite();
        for (int i = 0; i < n; ++i) {
            hx[i] = seeds.x[i];
            hy[i] = seeds.y[i];
            hz[i] = seeds.z[i];
            hw[i] = weight[i];
        }
        double* hf = m_factor.HostWrite();
        for (std::size_t k = 0; k < factor.size(); ++k) {
            hf[k] = factor[k];
        }
    }
    m_indexed_ids.SetSize(num_reps);
    {
        int* hids = m_indexed_ids.HostWrite();
        for (int k = 0; k < num_reps; ++k) {
            hids[k] = reps[k];
        }
    }

    SeedDeviceView seed_view;
    seed_view.x = m_x.Read(on_device);
    seed_view.y = m_y.Read(on_device);
    seed_view.z = m_z.Read(on_device);
    seed_view.weight = m_weight.Read(on_device);
    seed_view.factor = m_factor.Read(on_device);
    seed_view.factor_stride = factor_stride;
    m_metric = Metric{seed_view};

    // ---- Step 6: one BVH per bin over zero-size boxes at the representative positions ----
    m_allocator_id = allocator_id;
    const double* dx = seed_view.x;
    const double* dy = seed_view.y;
    const double* dz = seed_view.z;
    const int* dids = m_indexed_ids.Read(on_device);
    for (int b = 0; b < bins; ++b) {
        const int begin = m_bin_begin[b];
        const int count = m_bin_begin[b + 1] - begin;

        // Box j of this bin is the point of seed dids[begin + j]. Uninitialized allocation
        // because BoundingBox is not trivially constructible (LinearBVH.hpp:280-284 does the
        // same); every entry is written by the kernel.
        axom::Array<detail::SeedBox> boxes(axom::ArrayOptions::Uninitialized{}, count, count,
                                           allocator_id);
        auto boxes_view = boxes.view();
        axom::for_all<ExecSpace>(
            count, AXOM_LAMBDA(axom::IndexType j) {
                const int s = dids[begin + j];
                boxes_view[j] = detail::SeedBox(detail::SeedPoint::make_point(dx[s], dy[s], dz[s]));
            });

        auto bvh = std::make_unique<BVHType>();
        bvh->setAllocatorID(allocator_id);
        bvh->initialize(boxes_view, count);
        m_bin_root[b] = bvh->getBounds();
        m_bvh.push_back(std::move(bvh));
    }

    // ---- Report ----
    m_report.num_seeds = n;
    m_report.num_indexed = num_reps;
    m_report.num_coincident = num_coincident;
    m_report.num_bins = bins;
    m_built = true;
}

/*
 * FindBest: assemble the kernel view (metric, traversers, bin data) and run FindBestKernel over
 * the query points. Query coordinates are read and results written in the memory space of
 * ExecSpace (mfem Read/Write with on_device), so no host round trip is needed on a GPU.
 */
template <typename ExecSpace, typename Metric>
void SeedIndex<ExecSpace, Metric>::FindBest(const mfem::Vector& qx, const mfem::Vector& qy,
                                            const mfem::Vector& qz, mfem::Array<BestPair>& out,
                                            bool exact_second,
                                            mfem::Array<int>* leaf_visits) const {
    MFEM_VERIFY(m_built, "SeedIndex::FindBest called before Build");
    detail::RequireMatchingMfemDevice(axom::execution_space<ExecSpace>::onDevice(),
                                      "SeedIndex::FindBest");
    MFEM_VERIFY(qx.Size() == qy.Size() && qx.Size() == qz.Size(),
                "SeedIndex::FindBest: qx, qy, qz must have the same size");
    const int nq = qx.Size();
    out.SetSize(nq);
    if (leaf_visits != nullptr) {
        leaf_visits->SetSize(nq);
    }
    if (nq == 0) {
        return;
    }

    // Kernel view: traversers of the bins plus copies of the per-bin host data (by value)
    const bool on_device = axom::execution_space<ExecSpace>::onDevice();
    const auto view = detail::MakeSearchView(
        m_metric, m_bvh, m_bin_root, m_bin_bound, m_bin_begin, m_indexed_ids.Read(on_device),
        m_coincident_next.Read(on_device), std::make_index_sequence<SEED_INDEX_MAX_BINS>{});

    // Optional diagnostic output (null pointer = not requested)
    int* visits = (leaf_visits != nullptr) ? leaf_visits->Write(on_device) : nullptr;
    detail::FindBestKernel<ExecSpace>(view, nq, qx.Read(on_device), qy.Read(on_device),
                                      qz.Read(on_device), out.Write(on_device), exact_second,
                                      visits);
}

/*
 * FindWithin: count pass (kernel), exclusive prefix sum of the counts on the host (offsets[q] =
 * number of results of all queries before q), fill pass (kernel), then sort each query's ids on
 * the host so the output does not depend on the traversal order.
 */
template <typename ExecSpace, typename Metric>
void SeedIndex<ExecSpace, Metric>::FindWithin(const mfem::Vector& qx, const mfem::Vector& qy,
                                              const mfem::Vector& qz, double f_max,
                                              mfem::Array<int>& offsets,
                                              mfem::Array<int>& ids) const {
    MFEM_VERIFY(m_built, "SeedIndex::FindWithin called before Build");
    detail::RequireMatchingMfemDevice(axom::execution_space<ExecSpace>::onDevice(),
                                      "SeedIndex::FindWithin");
    MFEM_VERIFY(qx.Size() == qy.Size() && qx.Size() == qz.Size(),
                "SeedIndex::FindWithin: qx, qy, qz must have the same size");
    const int nq = qx.Size();
    offsets.SetSize(nq + 1);
    offsets = 0;
    ids.SetSize(0);
    if (nq == 0) {
        return;
    }

    const bool on_device = axom::execution_space<ExecSpace>::onDevice();
    const auto view = detail::MakeSearchView(
        m_metric, m_bvh, m_bin_root, m_bin_bound, m_bin_begin, m_indexed_ids.Read(on_device),
        m_coincident_next.Read(on_device), std::make_index_sequence<SEED_INDEX_MAX_BINS>{});
    const double* dqx = qx.Read(on_device);
    const double* dqy = qy.Read(on_device);
    const double* dqz = qz.Read(on_device);

    // Pass 1: counts per query
    mfem::Array<int> counts(nq);
    detail::WithinKernel<ExecSpace>(view, nq, dqx, dqy, dqz, f_max, counts.Write(on_device),
                                    nullptr, nullptr);

    // Exclusive prefix sum on the host
    const int* hcounts = counts.HostRead();
    int* hoffsets = offsets.HostWrite();
    hoffsets[0] = 0;
    for (int q = 0; q < nq; ++q) {
        hoffsets[q + 1] = hoffsets[q] + hcounts[q];
    }

    // Pass 2: indices
    ids.SetSize(hoffsets[nq]);
    if (hoffsets[nq] > 0) {
        detail::WithinKernel<ExecSpace>(view, nq, dqx, dqy, dqz, f_max, nullptr,
                                        offsets.Read(on_device), ids.Write(on_device));
    }

    // Ascending seed index within each query (host)
    int* hids = ids.HostReadWrite();
    const int* hoff = offsets.HostRead();
    for (int q = 0; q < nq; ++q) {
        std::sort(hids + hoff[q], hids + hoff[q + 1]);
    }
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiations: every metric on every execution space this translation unit can see.
// GPU spaces exist only when this file is compiled as CUDA/HIP (exec_dispatch.hpp file comment).
// -------------------------------------------------------------------------------------------------
#define EXACONSTIT_MS_INSTANTIATE_SEED_INDEX(EXEC)            \
    template class SeedIndex<EXEC, VoronoiMetric>;            \
    template class SeedIndex<EXEC, LaguerreMetric>;           \
    template class SeedIndex<EXEC, QuadricMetric>;            \
    template class SeedIndex<EXEC, AdditiveMetric>;

EXACONSTIT_MS_INSTANTIATE_SEED_INDEX(axom::SEQ_EXEC)
#if defined(AXOM_RUNTIME_POLICY_USE_OPENMP)
EXACONSTIT_MS_INSTANTIATE_SEED_INDEX(axom::OMP_EXEC)
#endif
#if defined(AXOM_RUNTIME_POLICY_USE_CUDA)
EXACONSTIT_MS_INSTANTIATE_SEED_INDEX(axom::CUDA_EXEC<256>)
#endif
#if defined(AXOM_RUNTIME_POLICY_USE_HIP)
EXACONSTIT_MS_INSTANTIATE_SEED_INDEX(axom::HIP_EXEC<256>)
#endif

#undef EXACONSTIT_MS_INSTANTIATE_SEED_INDEX

} // namespace exaconstit::microstructure
