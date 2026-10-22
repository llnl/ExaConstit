/*
 * Implementation of the graph-geodesic labeler (geodesic_graph.hpp): sources -> Dijkstra on the
 * element edges and diagonals with distance exchanges between ranks -> labels from tight
 * neighbors -> element labels.
 *
 * Terms used in this file:
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 * - Dijkstra's algorithm: computes shortest path lengths from a set of start vertices by always
 *   finishing the not-yet-finished vertex with the smallest tentative distance, taken from a
 *   binary heap (std::priority_queue: the smallest entry comes out first).
 * - Relaxation: offering a vertex u the candidate T(v) + w(v,u) and keeping it if smaller.
 * - Tight neighbor of u: a vertex v of a common element with T(v) < T(u) and
 *   T(v) + w(v,u) == T(u) exactly, i.e. a vertex on a shortest path into u.
 */
#include "microstructure/labeling/geodesic_graph.hpp"

#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief +infinity, the "not reached" distance. */
constexpr double INF = std::numeric_limits<double>::infinity();

/** @brief Label of a reached vertex whose label is not known yet (larger than any seed index). */
constexpr int UNSET_LABEL = INT_MAX;

/**
 * @brief Straight-line length of the graph edge between local vertices a and b.
 *
 * sqrt(dx^2 + dy^2 + dz^2) in this fixed order. Swapping a and b only negates dx, dy, dz, which
 * does not change their squares, so every rank computes the same bits for the same pair (the
 * coordinates of a shared vertex are bitwise equal on all ranks, VertexField::Coordinates). Host.
 */
inline double EdgeLength(const double* xyz, int a, int b) {
    const double dx = xyz[3 * a + 0] - xyz[3 * b + 0];
    const double dy = xyz[3 * a + 1] - xyz[3 * b + 1];
    const double dz = xyz[3 * a + 2] - xyz[3 * b + 2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/** @brief Host views of the VertexField connectivity used by both phases. */
struct GraphView {
    int num_vertices = 0;
    const int* elem_vert_offsets = nullptr; ///< element -> vertex CSR offsets
    const int* elem_verts = nullptr;        ///< element -> vertex CSR items
    const int* vert_elem_offsets = nullptr; ///< vertex -> element CSR offsets
    const int* vert_elems = nullptr;        ///< vertex -> element CSR items
    const double* xyz = nullptr;            ///< coordinates xyz[3*v + d]
};

/**
 * @brief Calls visit(u) for every vertex u != v that shares a local element with v.
 *
 * Goes through the elements of v (vertex -> element table) and their vertices (element -> vertex
 * table). A neighbor that shares several elements with v is visited once per shared element;
 * that is harmless for both phases (the same candidate is offered again) and avoids a
 * de-duplication buffer. Host.
 */
template <typename Visit>
inline void ForEachGraphNeighbor(const GraphView& g, int v, Visit&& visit) {
    for (int ke = g.vert_elem_offsets[v]; ke < g.vert_elem_offsets[v + 1]; ++ke) {
        const int e = g.vert_elems[ke];
        for (int kv = g.elem_vert_offsets[e]; kv < g.elem_vert_offsets[e + 1]; ++kv) {
            const int u = g.elem_verts[kv];
            if (u != v) {
                visit(u);
            }
        }
    }
}

/**
 * @brief Multi-source Dijkstra on this rank's elements, started from the listed vertices with
 * their current T.
 *
 * The heap holds (T, vertex) pairs; an entry whose T no longer matches T(vertex) is outdated (the
 * vertex was improved after the entry was pushed) and skipped ("lazy deletion", simpler than
 * updating entries inside the heap). Each vertex taken from the heap relaxes all its graph
 * neighbors with a strict "<", so a vertex is only updated when its distance really drops. The
 * order of equal-T entries in the heap is irrelevant: the converged T does not depend on it (see
 * the header). Host, not collective.
 *
 * @param g      connectivity.
 * @param t      T per local vertex (updated).
 * @param starts vertices to start from (any order; vertices with T = +inf are ignored).
 */
void LocalDijkstra(const GraphView& g, double* t, const std::vector<int>& starts) {
    using Entry = std::pair<double, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (const int v : starts) {
        if (t[v] < INF) {
            heap.push({t[v], v});
        }
    }
    while (!heap.empty()) {
        const Entry top = heap.top();
        heap.pop();
        const int v = top.second;
        if (top.first != t[v]) {
            continue; // outdated entry
        }
        // Offer T(v) + w(v,u) to every neighbor u
        ForEachGraphNeighbor(g, v, [&](int u) {
            const double candidate = t[v] + EdgeLength(g.xyz, v, u);
            if (candidate < t[u]) {
                t[u] = candidate;
                heap.push({candidate, u});
            }
        });
    }
}

/**
 * @brief One local label pass over the reached vertices in increasing (T, vertex id) order.
 *
 * label(u) = min(label(u), labels of u's tight neighbors), where the current label(u) already
 * holds the source label (if u is a source vertex attaining its T) or a smaller value received
 * from another rank. A tight neighbor v satisfies fl(T(v) + w(v,u)) == T(u) with w > 0 and
 * T(v) <= T(u). Normally T(v) < T(u), so v comes earlier in the order and is final for this pass
 * when u is visited; one pass then propagates labels along whole shortest-path trees inside this
 * rank. T(v) == T(u) happens only when the edge is shorter than the rounding error of T (a
 * degenerate mesh); such neighbors are accepted too (otherwise u could be left without any label),
 * and the caller repeats the pass until nothing changes, because a neighbor with equal T may be
 * visited after u. Host, not collective.
 *
 * @param g     connectivity.
 * @param t     converged T (read only).
 * @param label labels (updated; UNSET_LABEL = not known yet).
 * @param order reached vertices sorted by (T, vertex id).
 * @return true if any label changed.
 */
bool LocalLabelPass(const GraphView& g, const double* t, int* label,
                    const std::vector<int>& order) {
    bool changed = false;
    for (const int u : order) {
        int best = label[u];
        ForEachGraphNeighbor(g, u, [&](int v) {
            // Tight: v lies on a shortest path into u (same arithmetic as in LocalDijkstra)
            const double w = EdgeLength(g.xyz, v, u);
            if (t[v] <= t[u] && w > 0.0 && t[v] + w == t[u] && label[v] < best) {
                best = label[v];
            }
        });
        changed = changed || best != label[u];
        label[u] = best;
    }
    return changed;
}

/** @brief Global sum of a local count (MPI_Allreduce). Collective. */
std::int64_t GlobalSum(std::int64_t local, MPI_Comm comm) {
    long long value = static_cast<long long>(local);
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_LONG_LONG, MPI_SUM, comm);
    return static_cast<std::int64_t>(value);
}

} // namespace

/*
 * LabelGeodesicGraph
 *
 * Steps (see the header for the algorithm and the partition-independence argument):
 *  1. Parameter checks (identical on every rank: no communication needed to throw together).
 *  2. VertexField (collective) and InitializeSources (collective).
 *  3. Phase 1: rounds of LocalDijkstra + ReduceSharedDistance until no T changes anywhere. Round
 *     1 starts from every reached vertex (the sources); later rounds start from the vertices
 *     whose T dropped in the exchange.
 *  4. Phase 2: labels start at the source label where T0(v) == T(v), UNSET_LABEL elsewhere on
 *     reached vertices and -1 on unreached ones; rounds of LocalLabelPass + ReduceSharedLabels
 *     until no label changes anywhere.
 *  5. Element labels (centroid or vote) and the global statistics.
 */
GeodesicLabels LabelGeodesicGraph(mfem::ParMesh& pmesh, MPI_Comm comm, const SeedSet& seeds,
                                  bool additive, const GeodesicParameters& params) {
    // Step 1: parameters (seeds are validated by InitializeSources)
    if (params.max_rounds < 1) {
        throw std::invalid_argument("LabelGeodesicGraph: max_rounds must be >= 1");
    }
    if (params.sampling == SamplingMode::QUADRATURE_VOTE) {
        if (params.vote_levels < 0 || params.vote_levels > 3) {
            throw std::invalid_argument("LabelGeodesicGraph: vote_levels must be 0..3");
        }
        if (params.max_points_per_chunk < 1) {
            throw std::invalid_argument("LabelGeodesicGraph: max_points_per_chunk must be >= 1");
        }
    }
    if (additive && !seeds.weight.empty() && seeds.weight.size() != seeds.Size()) {
        throw std::invalid_argument("LabelGeodesicGraph: seeds.weight must be empty or have one "
                                    "entry per seed");
    }
    {
        int same_comm = MPI_UNEQUAL;
        MPI_Comm_compare(comm, pmesh.GetComm(), &same_comm);
        if (same_comm != MPI_IDENT && same_comm != MPI_CONGRUENT) {
            throw std::invalid_argument(
                "LabelGeodesicGraph: comm must be the mesh communicator");
        }
    }

    // Step 2: vertex field and sources
    GeodesicLabels result;
    result.field = std::make_unique<VertexField>(pmesh);
    VertexField& field = *result.field;
    const std::vector<double> no_weights;
    result.sources = field.InitializeSources(seeds, additive ? seeds.weight : no_weights);
    result.report.seeds_outside_material = result.sources.num_seeds_outside;

    const int nv = field.NumVertices();
    GraphView g;
    g.num_vertices = nv;
    g.elem_vert_offsets = field.ElementVertexOffsets().HostRead();
    g.elem_verts = field.ElementVertices().HostRead();
    g.vert_elem_offsets = field.VertexElementOffsets().HostRead();
    g.vert_elems = field.VertexElements().HostRead();
    g.xyz = field.Coordinates().HostRead();

    // Step 3: phase 1, distances
    std::vector<int> starts;
    for (int v = 0; v < nv; ++v) {
        starts.push_back(v); // LocalDijkstra skips unreached vertices
    }
    mfem::Array<int> changed;
    for (int round = 1;; ++round) {
        // HostReadWrite each round: ReduceShared* leaves the valid copy on the host
        LocalDijkstra(g, field.Distance().HostReadWrite(), starts);
        const bool any_change = field.ReduceSharedDistance(&changed);
        result.report.distance_rounds = round;
        if (!any_change) {
            break;
        }
        // any_change is global, so every rank takes this branch together
        ThrowIfAnyRank<std::runtime_error>(
            comm, round >= params.max_rounds,
            "LabelGeodesicGraph: max_rounds exceeded in the distance rounds");
        starts.clear();
        for (int v = 0; v < nv; ++v) {
            if (changed[v] != 0) {
                starts.push_back(v);
            }
        }
    }

    // Step 4: phase 2, labels from tight neighbors
    const double* t = field.Distance().HostRead();
    const double* t0 = field.SourceDistance().HostRead();
    const int* label0 = field.SourceLabel().HostRead();
    int* label = field.Label().HostWrite();
    std::vector<int> order;
    for (int v = 0; v < nv; ++v) {
        if (t[v] == INF) {
            label[v] = -1;
            continue;
        }
        label[v] = (label0[v] >= 0 && t0[v] == t[v]) ? label0[v] : UNSET_LABEL;
        order.push_back(v);
    }
    std::sort(order.begin(), order.end(), [t](int a, int b) {
        return (t[a] < t[b]) || (t[a] == t[b] && a < b);
    });
    for (int round = 1;; ++round) {
        // Local passes until locally stable (one pass unless equal-T tight neighbors exist)
        while (LocalLabelPass(g, t, field.Label().HostReadWrite(), order)) {
        }
        const bool any_change = field.ReduceSharedLabels();
        result.report.label_rounds = round;
        if (!any_change) {
            break;
        }
        ThrowIfAnyRank<std::runtime_error>(
            comm, round >= params.max_rounds,
            "LabelGeodesicGraph: max_rounds exceeded in the label rounds");
    }
    // A reached vertex left without a label (degenerate mesh only) becomes -1 and is counted
    std::int64_t unlabeled = 0;
    std::int64_t unreached = 0;
    {
        int* lab = field.Label().HostReadWrite();
        const int* owned = field.OwnedVertexMarker().HostRead();
        for (int v = 0; v < nv; ++v) {
            if (lab[v] == UNSET_LABEL) {
                lab[v] = -1;
                unlabeled += owned[v];
            }
            // Count only vertices of at least one element (an unused mesh vertex is no material)
            const bool in_element = g.vert_elem_offsets[v + 1] > g.vert_elem_offsets[v];
            if (t[v] == INF && in_element) {
                unreached += owned[v];
            }
        }
    }

    // Step 5: element labels and statistics
    std::int64_t split = 0;
    if (params.sampling == SamplingMode::QUADRATURE_VOTE) {
        split = field.LabelElementsByVote(params.vote_levels, params.max_points_per_chunk,
                                          result.element_seed);
    } else {
        field.LabelElementCentroids(result.element_seed);
    }
    // An element labeled -1 is "unreached" when none of its vertices has a finite T, otherwise
    // "unlabeled" (a reached element that only saw unlabeled vertices). Element-local data only.
    std::int64_t unreached_elements = 0;
    std::int64_t unlabeled_elements = 0;
    {
        const int* es = result.element_seed.HostRead();
        const double* t_final = field.Distance().HostRead();
        const int* ev_offsets = field.ElementVertexOffsets().HostRead();
        const int* ev_items = field.ElementVertices().HostRead();
        for (int e = 0; e < result.element_seed.Size(); ++e) {
            if (es[e] >= 0) {
                continue;
            }
            bool any_reached = false;
            for (int j = ev_offsets[e]; j < ev_offsets[e + 1]; ++j) {
                any_reached = any_reached || t_final[ev_items[j]] < INF;
            }
            unlabeled_elements += any_reached ? 1 : 0;
            unreached_elements += any_reached ? 0 : 1;
        }
    }
    result.report.unreached_vertices = GlobalSum(unreached, comm);
    result.report.unlabeled_vertices = GlobalSum(unlabeled, comm);
    result.report.unreached_elements = GlobalSum(unreached_elements, comm);
    result.report.unlabeled_elements = GlobalSum(unlabeled_elements, comm);
    result.report.split_elements = GlobalSum(split, comm);
    return result;
}

} // namespace exaconstit::microstructure
