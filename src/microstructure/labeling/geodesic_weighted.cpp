/*
 * Implementation of the weighted multi-seed geodesic labeler (geodesic_weighted.hpp): sources ->
 * synchronous rounds of per-seed FIM updates into bounded, pruned seed lists, with list exchanges
 * between ranks -> labels from the lists -> element labels.
 *
 * Terms used in this file (see also geodesic_fim.cpp):
 * - Entry: one (seed i, T_i) pair of a node's list. Lists are stored packed (entries first, then
 *   slots with seed -1) and sorted by (f, seed), f = T^2 - w_seed, so the first entry is the label.
 * - Dominated: entry i is dominated by entry j when T_j <= T_i and f_j <= f_i, one strictly
 *   (geodesic_weighted.hpp, "Pruning").
 * - Merge: inserting entries into a list: per seed the smaller T, dominated entries removed, at
 *   most K kept (the worst (f, seed) cut, counted as an overflow).
 */
#include "microstructure/labeling/geodesic_weighted.hpp"

#include "microstructure/labeling/eikonal_local_solver.hpp"
#include "microstructure/labeling/fim_scan.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief +infinity, the "not reached" distance (host constant, captured by value in kernels). */
constexpr double INF = std::numeric_limits<double>::infinity();

/** @brief Largest number of vertices of an element (linear hex); 7 other corners per cell. */
constexpr int MAX_ELEMENT_VERTICES = 8;

/** @brief Room of the working lists in the kernels (one more than the largest K). */
constexpr int WORK = GEODESIC_MAX_SEEDS_PER_NODE_LIMIT + 1;

/** @brief "j dominates i" for entries (T, f) (one comparison strict). Host and device. */
MFEM_HOST_DEVICE inline bool Dominates(double tj, double fj, double ti, double fi) {
    return tj <= ti && fj <= fi && (tj < ti || fj < fi);
}

/** @brief (fa, a) before (fb, b): smaller f, then smaller seed. Host and device. */
MFEM_HOST_DEVICE inline bool Before(double fa, int a, double fb, int b) {
    return fa < fb || (fa == fb && a < b);
}

/**
 * @brief True when seed j's tensor is nowhere larger than seed i's, A_j <= A_i (A_i - A_j positive
 * semidefinite), so j's length of any path is at most i's; always true without per-seed tensors
 * (tensors == nullptr) and for identical tensors. Conservative: the principal minors of
 * A_i - A_j must exceed a small positive slack (1e-12 of the scale), so rounding never claims an
 * order that does not hold; a semidefinite difference with a zero eigenvalue then counts as "not
 * ordered" (no pruning, never a wrong one). Tensors as 6 entries per seed
 * (xx yy zz yz xz xy). Host and device.
 */
MFEM_HOST_DEVICE inline bool TensorLeq(const double* tensors, int j, int i) {
    if (tensors == nullptr || i == j) {
        return true;
    }
    const double* a = &tensors[6 * j];
    const double* b = &tensors[6 * i];
    double d[6];
    bool zero = true;
    double scale = 0.0;
    for (int k = 0; k < 6; ++k) {
        d[k] = b[k] - a[k];
        zero = zero && d[k] == 0.0;
        scale = fmax(scale, fmax(fabs(a[k]), fabs(b[k])));
    }
    if (zero) {
        return true;
    }
    const double eps = 1e-12 * scale;
    // d = [xx yy zz yz xz xy]
    const double m01 = d[0] * d[1] - d[5] * d[5];
    const double m02 = d[0] * d[2] - d[4] * d[4];
    const double m12 = d[1] * d[2] - d[3] * d[3];
    const double det = d[0] * (d[1] * d[2] - d[3] * d[3]) - d[5] * (d[5] * d[2] - d[3] * d[4]) +
                       d[4] * (d[5] * d[3] - d[1] * d[4]);
    return d[0] > eps && d[1] > eps && d[2] > eps && m01 > eps * scale && m02 > eps * scale &&
           m12 > eps * scale && det > eps * scale * scale;
}

/**
 * @brief A node's working list: n entries (seed, T, f) in no particular order, at most cap.
 * Host and device (lives in registers / local memory of one kernel iteration).
 */
struct WorkList {
    int seed[WORK];
    double t[WORK];
    double f[WORK];
    int n = 0;
    int cap = 0;
    bool overflow = false;
    const double* tensors = nullptr; ///< per-seed tensors (6 per seed) or nullptr (one metric)

    /** @brief Entry a dominates entry b (TensorLeq for per-seed tensors). */
    MFEM_HOST_DEVICE bool Dom(int a, int b) const {
        return Dominates(t[a], f[a], t[b], f[b]) && TensorLeq(tensors, seed[a], seed[b]);
    }

    /** @brief Removes entry k (order is not kept). */
    MFEM_HOST_DEVICE void Remove(int k) {
        --n;
        seed[k] = seed[n];
        t[k] = t[n];
        f[k] = f[n];
    }

    /** @brief Removes every entry dominated by entry k. */
    MFEM_HOST_DEVICE void RemoveDominatedBy(int k) {
        int q = 0;
        while (q < n) {
            if (q != k && Dom(k, q)) {
                Remove(q);
                if (k == n) {
                    k = q; // entry k was moved into slot q
                }
            } else {
                ++q;
            }
        }
    }

    /**
     * @brief Merges (s, value): per seed the smaller T; a dominated entry is not added; entries
     * the new or lowered one dominates are removed; above cap the worst (f, seed) is cut and
     * overflow is set. w = the weight array (w[s]).
     */
    MFEM_HOST_DEVICE void Insert(int s, double value, const double* w) {
        if (!(value < EikonalInfinity())) {
            return;
        }
        const double fv = value * value - w[s];
        for (int k = 0; k < n; ++k) {
            if (seed[k] == s) {
                if (value < t[k]) {
                    t[k] = value;
                    f[k] = fv;
                    RemoveDominatedBy(k);
                }
                return;
            }
        }
        for (int k = 0; k < n; ++k) {
            if (Dominates(t[k], f[k], value, fv) && TensorLeq(tensors, seed[k], s)) {
                return;
            }
        }
        seed[n] = s;
        t[n] = value;
        f[n] = fv;
        ++n;
        RemoveDominatedBy(n - 1);
        if (n > cap) {
            int worst = 0;
            for (int k = 1; k < n; ++k) {
                if (Before(f[worst], seed[worst], f[k], seed[k])) {
                    worst = k;
                }
            }
            Remove(worst);
            overflow = true;
        }
    }

    /** @brief Sorts by (f, seed) (insertion sort; lists are short). */
    MFEM_HOST_DEVICE void Sort() {
        for (int a = 1; a < n; ++a) {
            const int s = seed[a];
            const double ta = t[a];
            const double fa = f[a];
            int b = a - 1;
            while (b >= 0 && Before(fa, s, f[b], seed[b])) {
                seed[b + 1] = seed[b];
                t[b + 1] = t[b];
                f[b + 1] = f[b];
                --b;
            }
            seed[b + 1] = s;
            t[b + 1] = ta;
            f[b + 1] = fa;
        }
    }
};

/** @brief y = S x for a row-major 3x3 stretch S, in a fixed order. Host and device. */
MFEM_HOST_DEVICE inline void MapPoint(const double* st, const double* x, double* y) {
    y[0] = st[0] * x[0] + st[1] * x[1] + st[2] * x[2];
    y[1] = st[3] * x[0] + st[4] * x[1] + st[5] * x[2];
    y[2] = st[6] * x[0] + st[7] * x[1] + st[8] * x[2];
}

/** @brief T of seed s in node u's packed list (K slots), +infinity if absent. */
MFEM_HOST_DEVICE inline double ListValue(const int* ls, const double* lt, int K, int u, int s) {
    for (int q = 0; q < K; ++q) {
        const int seed = ls[K * u + q];
        if (seed < 0) {
            break;
        }
        if (seed == s) {
            return lt[K * u + q];
        }
    }
    return EikonalInfinity();
}

/**
 * @brief True when node v's new list differs from its old one so that its neighbors must be
 * updated: a seed added or removed, or a T lowered by more than tol. Smaller improvements are kept
 * in the values but do not count (stopping tolerance, geodesic_weighted.hpp). Lists are sorted by
 * (f, seed), but a lowered T can reorder them, so seeds are matched by search. Host and device.
 */
MFEM_HOST_DEVICE inline bool ListChanged(const int* old_s, const double* old_t, const int* new_s,
                                         const double* new_t, int K, double tol) {
    int n_old = 0;
    int n_new = 0;
    while (n_old < K && old_s[n_old] >= 0) {
        ++n_old;
    }
    while (n_new < K && new_s[n_new] >= 0) {
        ++n_new;
    }
    if (n_old != n_new) {
        return true;
    }
    for (int a = 0; a < n_new; ++a) {
        bool found = false;
        for (int b = 0; b < n_old && !found; ++b) {
            if (old_s[b] == new_s[a]) {
                found = true;
                if (new_t[a] < old_t[b] - tol) {
                    return true;
                }
            }
        }
        if (!found) {
            return true;
        }
    }
    return false;
}

/** @brief Writes a sorted working list into node v's K slots (rest seed -1, T +infinity). */
MFEM_HOST_DEVICE inline void StoreList(const WorkList& list, int K, int v, int* ls, double* lt) {
    for (int q = 0; q < K; ++q) {
        ls[K * v + q] = (q < list.n) ? list.seed[q] : -1;
        lt[K * v + q] = (q < list.n) ? list.t[q] : EikonalInfinity();
    }
}

/**
 * @brief One synchronous round of the seed lists.
 *
 * Every node with a changed cell neighbor (changed[u] = 1 for a node u of one of its cells)
 * rebuilds its list: its old entries, then for every cell of every element around it and every
 * seed s in the lists of the cell's other corners (each seed once per cell) the per-cell FIM
 * update from those corners' T_s (ScanCell; a corner without s is unreached), merged into the
 * working list. Other nodes copy their list. new_overflow[v] = 1 when a merge cut a seed. Reads
 * only the previous round's lists. One mfem::forall iteration per node. Not collective.
 */
void ListSweep(const VertexField& field, int K, const mfem::Array<int>& changed,
               const mfem::Vector& weights, const mfem::Vector& seed_stretch,
               const mfem::Vector& tensors, const mfem::Array<int>& old_seed,
               const mfem::Vector& old_t, mfem::Array<int>& new_seed, mfem::Vector& new_t,
               mfem::Array<int>& new_overflow) {
    const int nv = field.NumVertices();
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const int* ve_items = field.VertexElements().Read();
    const FieldCellView view = field.CellView();
    const double* xyz = field.MetricCoordinates().Read();
    const int* chg = changed.Read();
    const double* w = weights.Read();
    // Per-seed tensors: seed s's lengths are measured with S_s (9 per seed), and dominance needs
    // A_j <= A_i (6 per seed); empty = one metric for all seeds
    const double* sst = (seed_stretch.Size() > 0) ? seed_stretch.Read() : nullptr;
    const double* ten = (tensors.Size() > 0) ? tensors.Read() : nullptr;
    const int* ls = old_seed.Read();
    const double* lt = old_t.Read();
    int* ns = new_seed.Write();
    double* nt = new_t.Write();
    int* ovf = new_overflow.Write();
    const bool curved = view.curved != 0;
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        // Active: a node that a cell of v reads has changed
        bool active = false;
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1] && !active; ++k) {
            const int e = ve_items[k];
            const int lattice = view.LatticeOfIncidence(k);
            for (int j = 0; j < view.NumNeighborBits(e) && !active; ++j) {
                const int u = view.NeighborNode(e, lattice, j);
                active = u >= 0 && chg[u] != 0;
            }
        }
        if (!active) {
            for (int q = 0; q < K; ++q) {
                ns[K * v + q] = ls[K * v + q];
                nt[K * v + q] = lt[K * v + q];
            }
            ovf[v] = 0;
            return;
        }
        WorkList list;
        list.cap = K;
        list.tensors = ten;
        for (int q = 0; q < K && ls[K * v + q] >= 0; ++q) {
            list.Insert(ls[K * v + q], lt[K * v + q], w);
        }
        const double* xv = &xyz[3 * v];
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
            const int e = ve_items[k];
            view.ForEachCellOfIncidence(v, e, view.LatticeOfIncidence(k),
                                        [&](const int* others, const int* bits, int n) {
                const double* xo[MAX_ELEMENT_VERTICES - 1];
                for (int i = 0; i < n; ++i) {
                    xo[i] = &xyz[3 * others[i]];
                }
                int far[MAX_ELEMENT_VERTICES - 1];
                const bool hex = CellFarMasks(curved, bits, n, far);
                // Every seed of the corners' lists, once per cell (first occurrence only)
                for (int i = 0; i < n; ++i) {
                    for (int q = 0; q < K; ++q) {
                        const int s = ls[K * others[i] + q];
                        if (s < 0) {
                            break;
                        }
                        bool seen = false;
                        for (int i2 = 0; i2 < i && !seen; ++i2) {
                            seen = ListValue(ls, lt, K, others[i2], s) < EikonalInfinity();
                        }
                        for (int q2 = 0; q2 < q && !seen; ++q2) {
                            seen = ls[K * others[i] + q2] == s;
                        }
                        if (seen) {
                            continue;
                        }
                        double to[MAX_ELEMENT_VERTICES - 1];
                        for (int i2 = 0; i2 < n; ++i2) {
                            to[i2] = ListValue(ls, lt, K, others[i2], s);
                        }
                        int mask = 0;
                        double value;
                        if (sst != nullptr) {
                            // Seed s's own metric: map v and the corners with S_s
                            const double* m = &sst[9 * s];
                            double yv[3];
                            double yo[MAX_ELEMENT_VERTICES - 1][3];
                            const double* yp[MAX_ELEMENT_VERTICES - 1];
                            MapPoint(m, xv, yv);
                            for (int i2 = 0; i2 < n; ++i2) {
                                MapPoint(m, xo[i2], yo[i2]);
                                yp[i2] = yo[i2];
                            }
                            value = ScanCell(yv, yp, to, bits, hex ? far : nullptr, n, &mask);
                        } else {
                            value = ScanCell(xv, xo, to, bits, hex ? far : nullptr, n, &mask);
                        }
                        list.Insert(s, value, w);
                    }
                }
            });
        }
        list.Sort();
        StoreList(list, K, v, ns, nt);
        ovf[v] = list.overflow ? 1 : 0;
    });
}

/**
 * @brief Makes every copy of a shared node hold the merge of all copies' lists (exchange, then
 * the same order-independent merge on every rank). Returns, per node, whether a merge cut a seed
 * (overflow) through `overflow` (OR-ed in). Collective. Host.
 */
void MergeSharedLists(const VertexField& field, int K, const std::vector<double>& weights,
                      const std::vector<double>& tensors, mfem::Array<int>& seeds,
                      mfem::Vector& dist, mfem::Array<int>& overflow) {
    const int nv = field.NumVertices();
    const int width = 2 * K;
    int* ls = seeds.HostReadWrite();
    double* lt = dist.HostReadWrite();
    int* ovf = overflow.HostReadWrite();
    std::vector<double> send(static_cast<std::size_t>(width) * static_cast<std::size_t>(nv));
    const int* shared = field.SharedVertexMarker().HostRead();
    for (int v = 0; v < nv; ++v) {
        for (int q = 0; q < K; ++q) {
            send[static_cast<std::size_t>(width * v + 2 * q)] = ls[K * v + q];
            send[static_cast<std::size_t>(width * v + 2 * q + 1)] = lt[K * v + q];
        }
    }
    std::vector<double> recv;
    std::vector<int> offsets;
    field.ExchangeSharedPayload(width, send, recv, offsets);
    for (int v = 0; v < nv; ++v) {
        if (shared[v] == 0 || offsets[static_cast<std::size_t>(v) + 1] ==
                                  offsets[static_cast<std::size_t>(v)]) {
            continue;
        }
        WorkList list;
        list.cap = K;
        list.tensors = tensors.empty() ? nullptr : tensors.data();
        for (int q = 0; q < K && ls[K * v + q] >= 0; ++q) {
            list.Insert(ls[K * v + q], lt[K * v + q], weights.data());
        }
        for (int c = offsets[static_cast<std::size_t>(v)];
             c < offsets[static_cast<std::size_t>(v) + 1]; ++c) {
            const double* copy = &recv[static_cast<std::size_t>(c) * width];
            for (int q = 0; q < K; ++q) {
                const int s = static_cast<int>(copy[2 * q]);
                if (s < 0) {
                    break;
                }
                list.Insert(s, copy[2 * q + 1], weights.data());
            }
        }
        list.Sort();
        StoreList(list, K, v, ls, lt);
        ovf[v] = (ovf[v] != 0 || list.overflow) ? 1 : 0;
    }
}

/** @brief Global sum of a local count (MPI_Allreduce). Collective. */
std::int64_t GlobalSum(std::int64_t local, MPI_Comm comm) {
    long long value = static_cast<long long>(local);
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_LONG_LONG, MPI_SUM, comm);
    return static_cast<std::int64_t>(value);
}

/** @brief Diagonal of the metric bounding box of the nodes (exact reductions). Collective. */
double MetricDiameter(const VertexField& field, MPI_Comm comm) {
    double bounds[6] = {INF, INF, INF, INF, INF, INF};
    const double* xyz = field.MetricCoordinates().HostRead();
    for (int v = 0; v < field.NumVertices(); ++v) {
        for (int d = 0; d < 3; ++d) {
            bounds[d] = std::min(bounds[d], xyz[3 * v + d]);
            bounds[3 + d] = std::min(bounds[3 + d], -xyz[3 * v + d]);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, bounds, 6, MPI_DOUBLE, MPI_MIN, comm);
    double sum = 0.0;
    for (int d = 0; d < 3; ++d) {
        const double extent = (bounds[d] < INF) ? (-bounds[3 + d] - bounds[d]) : 0.0;
        sum += extent * extent;
    }
    return std::sqrt(sum);
}

} // namespace

/*
 * LabelGeodesicWeighted
 *
 * Steps:
 *  1. Parameter checks (replicated inputs: every rank throws together).
 *  2. VertexField with the metric, InitializeSources (source elements and cells, snapping; its
 *     single-front T0 is not used), then the source lists: on the rank owning a seed's source
 *     element, every corner of the source cell gets (i, |S(v - s_i)|); MergeSharedLists.
 *  3. Rounds: ListSweep -> MergeSharedLists -> changed = ListChanged(old, new, tol) per node ->
 *     stop when nothing changed on any rank (changed is computed from lists that are identical on
 *     every copy, so every copy decides alike).
 *  4. Labels: the first entry of each list (T and seed into the VertexField), SetSeedLists, element
 *     labels (centroid or vote) and the statistics.
 */
GeodesicLabels LabelGeodesicWeighted(mfem::ParMesh& pmesh, MPI_Comm comm, const SeedSet& seeds,
                                     const std::vector<double>& weights,
                                     const GeodesicParameters& params) {
    // Step 1: parameters
    if (params.max_rounds < 1) {
        throw std::invalid_argument("LabelGeodesicWeighted: max_rounds must be >= 1");
    }
    if (!(params.tolerance >= 0.0 && params.tolerance < 1.0)) {
        throw std::invalid_argument("LabelGeodesicWeighted: tolerance must be in [0, 1)");
    }
    const int K = params.max_seeds_per_node;
    if (K < 1 || K > GEODESIC_MAX_SEEDS_PER_NODE_LIMIT) {
        throw std::invalid_argument("LabelGeodesicWeighted: max_seeds_per_node must be in 1..64");
    }
    if (params.sampling == SamplingMode::QUADRATURE_VOTE &&
        (params.vote_levels < 0 || params.vote_levels > 3 || params.max_points_per_chunk < 1)) {
        throw std::invalid_argument("LabelGeodesicWeighted: vote_levels must be 0..3 and "
                                    "max_points_per_chunk >= 1");
    }
    const std::size_t n = seeds.Size();
    if (!weights.empty() && weights.size() != n) {
        throw std::invalid_argument("LabelGeodesicWeighted: weights must be empty or have one "
                                    "entry per seed");
    }
    std::vector<double> w = weights.empty() ? std::vector<double>(n, 0.0) : weights;
    for (const double x : w) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("LabelGeodesicWeighted: weights must be finite");
        }
    }
    // Per-seed tensors: one stretch per seed, the field itself unstretched; A_i = S_i^T S_i for
    // the dominance test (TensorLeq)
    const bool per_seed = !params.seed_stretch.empty();
    std::vector<double> tensors;
    if (per_seed) {
        const std::array<double, 9> identity = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        if (params.seed_stretch.size() != 9 * n || params.stretch != identity) {
            throw std::invalid_argument("LabelGeodesicWeighted: seed_stretch must have 9 entries "
                                        "per seed, with stretch = identity");
        }
        tensors.resize(6 * n);
        for (std::size_t i = 0; i < n; ++i) {
            const double* m = &params.seed_stretch[9 * i];
            auto a = [&](int r, int c) {
                return m[r] * m[c] + m[3 + r] * m[3 + c] + m[6 + r] * m[6 + c];
            };
            const double entries[6] = {a(0, 0), a(1, 1), a(2, 2), a(1, 2), a(0, 2), a(0, 1)};
            for (int k = 0; k < 6; ++k) {
                if (!std::isfinite(entries[k])) {
                    throw std::invalid_argument("LabelGeodesicWeighted: seed_stretch must be "
                                                "finite");
                }
                tensors[6 * i + static_cast<std::size_t>(k)] = entries[k];
            }
        }
    }
    {
        int same_comm = MPI_UNEQUAL;
        MPI_Comm_compare(comm, pmesh.GetComm(), &same_comm);
        if (same_comm != MPI_IDENT && same_comm != MPI_CONGRUENT) {
            throw std::invalid_argument("LabelGeodesicWeighted: comm must be the mesh "
                                        "communicator");
        }
    }
    {
        bool too_many_vertices = false;
        for (int e = 0; e < pmesh.GetNE(); ++e) {
            too_many_vertices = too_many_vertices ||
                                pmesh.GetElement(e)->GetNVertices() > MAX_ELEMENT_VERTICES;
        }
        ThrowIfAnyRank<std::runtime_error>(
            comm, too_many_vertices,
            "LabelGeodesicWeighted: elements with more than 8 vertices are not supported");
    }

    // Step 2: field, sources, source lists
    GeodesicLabels result;
    result.field =
        std::make_unique<VertexField>(pmesh, params.field_nodes, ContextMask(params.context));
    VertexField& field = *result.field;
    field.SetMetricStretch(params.stretch);
    result.sources = field.InitializeSources(seeds, {}, params.context);
    result.report.seeds_outside_material = result.sources.num_seeds_outside;
    const int nv = field.NumVertices();
    const double tolerance = params.tolerance * MetricDiameter(field, comm);

    mfem::Vector stretch_vec(per_seed ? static_cast<int>(9 * n) : 0);
    mfem::Vector tensor_vec(per_seed ? static_cast<int>(6 * n) : 0);
    stretch_vec.UseDevice(true);
    tensor_vec.UseDevice(true);
    if (per_seed) {
        double* sv = stretch_vec.HostWrite();
        double* tv = tensor_vec.HostWrite();
        for (std::size_t k = 0; k < 9 * n; ++k) {
            sv[k] = params.seed_stretch[k];
        }
        for (std::size_t k = 0; k < 6 * n; ++k) {
            tv[k] = tensors[k];
        }
    }
    mfem::Vector weight_vec(static_cast<int>(n));
    weight_vec.UseDevice(true);
    {
        double* wv = weight_vec.HostWrite();
        for (std::size_t i = 0; i < n; ++i) {
            wv[i] = w[i];
        }
    }
    mfem::Array<int> list_seed(K * nv);
    mfem::Vector list_t(K * nv);
    list_t.UseDevice(true);
    mfem::Array<int> overflow(nv);
    overflow = 0;
    {
        int* ls = list_seed.HostWrite();
        double* lt = list_t.HostWrite();
        for (int q = 0; q < K * nv; ++q) {
            ls[q] = -1;
            lt[q] = INF;
        }
        // Per node, the entries of the seeds whose source cell has it as a corner
        const FieldCellView view = field.HostCellView();
        const std::array<double, 9>& st = field.MetricStretch();
        const double* y = field.MetricCoordinates().HostRead();
        std::vector<WorkList> lists;
        std::vector<int> list_of(static_cast<std::size_t>(nv), -1);
        int corners[FieldCellView::MAX_CELL_NODES];
        for (std::size_t i = 0; i < n; ++i) {
            const int e = result.sources.source_element_local[i];
            if (e < 0) {
                continue;
            }
            // T0 = |S (v - s_i)| with the common stretch, or seed i's own (per-seed tensors)
            const double* m = per_seed ? &params.seed_stretch[9 * i] : st.data();
            const double sx[3] = {seeds.x[i], seeds.y[i], seeds.z[i]};
            double ys[3];
            MapPoint(m, sx, ys);
            const int nc = view.CellNodes(e, result.sources.source_cell_local[i], corners);
            for (int c = 0; c < nc; ++c) {
                const int v = corners[c];
                double yv[3] = {y[3 * v + 0], y[3 * v + 1], y[3 * v + 2]};
                if (per_seed) {
                    MapPoint(m, &y[3 * v], yv);
                }
                const double dx = yv[0] - ys[0];
                const double dy = yv[1] - ys[1];
                const double dz = yv[2] - ys[2];
                if (list_of[static_cast<std::size_t>(v)] < 0) {
                    list_of[static_cast<std::size_t>(v)] = static_cast<int>(lists.size());
                    lists.emplace_back();
                    lists.back().cap = K;
                    lists.back().tensors = per_seed ? tensors.data() : nullptr;
                }
                lists[static_cast<std::size_t>(list_of[static_cast<std::size_t>(v)])].Insert(
                    static_cast<int>(i), std::sqrt(dx * dx + dy * dy + dz * dz), w.data());
            }
        }
        int* ovf = overflow.HostReadWrite();
        for (int v = 0; v < nv; ++v) {
            const int l = list_of[static_cast<std::size_t>(v)];
            if (l >= 0) {
                WorkList& list = lists[static_cast<std::size_t>(l)];
                list.Sort();
                StoreList(list, K, v, ls, lt);
                ovf[v] = list.overflow ? 1 : 0;
            }
        }
    }
    MergeSharedLists(field, K, w, tensors, list_seed, list_t, overflow);

    // Step 3: rounds. changed[v] = 1 at the start for every node with a list.
    mfem::Array<int> changed(nv);
    {
        const int* ls = list_seed.HostRead();
        int* chg = changed.HostWrite();
        for (int v = 0; v < nv; ++v) {
            chg[v] = (ls[K * v] >= 0) ? 1 : 0;
        }
    }
    mfem::Array<int> new_seed(K * nv);
    mfem::Vector new_t(K * nv);
    new_t.UseDevice(true);
    mfem::Array<int> new_overflow(nv);
    for (int round = 1;; ++round) {
        ListSweep(field, K, changed, weight_vec, stretch_vec, tensor_vec, list_seed, list_t,
                  new_seed, new_t, new_overflow);
        {
            const int* no = new_overflow.HostRead();
            int* ovf = overflow.HostReadWrite();
            for (int v = 0; v < nv; ++v) {
                ovf[v] = (ovf[v] != 0 || no[v] != 0) ? 1 : 0;
            }
        }
        MergeSharedLists(field, K, w, tensors, new_seed, new_t, overflow);
        bool local_change = false;
        {
            const int* os = list_seed.HostRead();
            const double* ot = list_t.HostRead();
            const int* ns = new_seed.HostRead();
            const double* nt = new_t.HostRead();
            int* chg = changed.HostWrite();
            for (int v = 0; v < nv; ++v) {
                const bool c = ListChanged(&os[K * v], &ot[K * v], &ns[K * v], &nt[K * v], K,
                                           tolerance);
                chg[v] = c ? 1 : 0;
                local_change = local_change || c;
            }
        }
        list_seed = new_seed;
        list_t = new_t;
        result.report.distance_rounds = round;
        if (!AnyRank(local_change, comm)) {
            break;
        }
        ThrowIfAnyRank<std::runtime_error>(
            comm, round >= params.max_rounds,
            "LabelGeodesicWeighted: max_rounds exceeded in the distance rounds");
    }

    // Step 4: node labels (first entry), element labels, statistics
    std::int64_t unreached = 0;
    std::int64_t overflow_count = 0;
    int longest = 0;
    {
        const int* ls = list_seed.HostRead();
        const double* lt = list_t.HostRead();
        const int* ovf = overflow.HostRead();
        const int* owned = field.OwnedVertexMarker().HostRead();
        const int* used = field.NodeUsed().HostRead();
        const bool masked = ContextMask(params.context) != nullptr &&
                            ContextMask(params.context)->active.Size() != 0;
        double* t = field.Distance().HostWrite();
        int* lab = field.Label().HostWrite();
        for (int v = 0; v < nv; ++v) {
            lab[v] = ls[K * v];
            t[v] = (ls[K * v] >= 0) ? lt[K * v] : INF;
            int len = 0;
            while (len < K && ls[K * v + len] >= 0) {
                ++len;
            }
            longest = std::max(longest, len);
            overflow_count += owned[v] * ovf[v];
            if (ls[K * v] < 0 && (!masked || used[v] != 0)) {
                unreached += owned[v];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &longest, 1, MPI_INT, MPI_MAX, comm);
    field.SetSeedLists(K, list_seed, list_t, w, params.seed_stretch);
    std::int64_t split = 0;
    if (params.sampling == SamplingMode::QUADRATURE_VOTE) {
        split = field.LabelElementsByVote(params.vote_levels, params.max_points_per_chunk,
                                          result.element_seed);
    } else {
        field.LabelElementCentroids(result.element_seed);
    }
    std::int64_t unreached_elements = 0;
    {
        const int* es = result.element_seed.HostRead();
        for (int e = 0; e < result.element_seed.Size(); ++e) {
            unreached_elements += (es[e] < 0 && field.ElementActive(e)) ? 1 : 0;
        }
    }
    result.report.unreached_vertices = GlobalSum(unreached, comm);
    result.report.unreached_elements = GlobalSum(unreached_elements, comm);
    result.report.split_elements = GlobalSum(split, comm);
    result.report.overflow_nodes = GlobalSum(overflow_count, comm);
    result.report.max_list_length = longest;
    return result;
}

} // namespace exaconstit::microstructure
