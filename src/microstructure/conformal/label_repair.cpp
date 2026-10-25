/*
 * Label repair before conformal fitting (label_repair.hpp).
 */
#include "microstructure/conformal/label_repair.hpp"

#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "axom/core.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace exaconstit::microstructure {

namespace {

constexpr double NONE_PRIORITY = -HUGE_VAL;
constexpr int BAD_NODE_WEIGHT = 8;

/** @brief Lexicographic pair (hi, lo) of doubles; (-inf, -inf) = no priority. */
struct Priority {
    double hi = NONE_PRIORITY;
    double lo = NONE_PRIORITY;
    bool operator<(const Priority& o) const {
        return hi < o.hi || (hi == o.hi && lo < o.lo);
    }
    bool operator==(const Priority& o) const {
        return hi == o.hi && lo == o.lo;
    }
};

/**
 * @brief Node maximum of element priorities: per node the largest priority of its local elements,
 * then the largest over the copies, as two exact minimum reductions (hi, then lo among the copies
 * holding the largest hi). Host, collective.
 */
std::vector<Priority> NodeMaximum(const VertexField& field, const std::vector<Priority>& elem) {
    const int nv = field.NumVertices();
    const int* ve_off = field.VertexElementOffsets().HostRead();
    const int* ve = field.VertexElements().HostRead();
    std::vector<Priority> node(static_cast<std::size_t>(nv));
    for (int v = 0; v < nv; ++v) {
        for (int k = ve_off[v]; k < ve_off[v + 1]; ++k) {
            node[static_cast<std::size_t>(v)] =
                std::max(node[static_cast<std::size_t>(v)], elem[static_cast<std::size_t>(ve[k])]);
        }
    }
    mfem::Vector neg(nv);
    for (int v = 0; v < nv; ++v) {
        neg(v) = -node[static_cast<std::size_t>(v)].hi;
    }
    field.ReduceSharedMinimum(neg);
    mfem::Vector neg_lo(nv);
    for (int v = 0; v < nv; ++v) {
        Priority& p = node[static_cast<std::size_t>(v)];
        neg_lo(v) = (p.hi == -neg(v)) ? -p.lo : HUGE_VAL;
        p.hi = -neg(v);
    }
    field.ReduceSharedMinimum(neg_lo);
    for (int v = 0; v < nv; ++v) {
        node[static_cast<std::size_t>(v)].lo = -neg_lo(v);
    }
    return node;
}

/** @brief Element maximum over its nodes of node priorities. Host. */
std::vector<Priority> ElementMaximum(const VertexField& field, const std::vector<Priority>& node) {
    const int ne = field.NumElements();
    const int* en_off = field.ElementVertexOffsets().HostRead();
    const int* en = field.ElementVertices().HostRead();
    std::vector<Priority> elem(static_cast<std::size_t>(ne));
    for (int e = 0; e < ne; ++e) {
        for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
            elem[static_cast<std::size_t>(e)] =
                std::max(elem[static_cast<std::size_t>(e)], node[static_cast<std::size_t>(en[k])]);
        }
    }
    return elem;
}

/** @brief Per-pass view of the labels: ghost labels, peninsula flags and their ghost copies. */
struct LabelState {
    mfem::Array<int> ghost_label;
    mfem::Array<int> peninsula;       ///< per local element: 1 = peninsula
    mfem::Array<int> ghost_peninsula; ///< per ghost element
};

/**
 * @brief Labels as seen by one element during a preview: the current labels, with element
 * `changed` (local, or -1) carrying `changed_label` instead.
 */
struct LabelView {
    const MeshAdjacency& adj;
    const mfem::Array<int>& label;
    const mfem::Array<int>& ghost_label;
    int changed = -1;
    int changed_label = 0;

    /** @brief Label of local element e. */
    int Local(int e) const {
        return (e == changed) ? changed_label : label[e];
    }
    /** @brief Label of neighbor index n of the adjacency (local element or ghost). */
    int Neighbor(int n) const {
        return adj.IsGhost(n) ? ghost_label[n - adj.NumElements()] : Local(n);
    }
};

/**
 * @brief Face-neighbor labels of element e other than its own, with the number of faces each
 * shares (ascending label). Host.
 */
std::vector<std::pair<int, int>> NeighborLabelCounts(const LabelView& view, int e) {
    const int* off = view.adj.Offsets().HostRead();
    const int* nbr = view.adj.Neighbors().HostRead();
    std::vector<std::pair<int, int>> counts; // (label, faces)
    const int own = view.Local(e);
    for (int k = off[e]; k < off[e + 1]; ++k) {
        const int b = view.Neighbor(nbr[k]);
        if (b == own) {
            continue;
        }
        auto it = std::lower_bound(counts.begin(), counts.end(), std::make_pair(b, 0));
        if (it != counts.end() && it->first == b) {
            ++it->second;
        } else {
            counts.insert(it, {b, 1});
        }
    }
    return counts;
}

/**
 * @brief True when element e (a label that may change, not fixed) shares all but at most one of
 * its faces with one other such grain. Host.
 */
bool IsPeninsula(const LabelView& view, int e, const NodeLabels& labels) {
    if (labels.IsFixedLabel(view.Local(e))) {
        return false;
    }
    const int faces = view.adj.GetParMesh().GetElement(e)->GetNFaces();
    for (const auto& [b, count] : NeighborLabelCounts(view, e)) {
        if (!labels.IsFixedLabel(b) && count >= faces - 1) {
            return true;
        }
    }
    return false;
}

/** @brief Peninsula flags of the local elements and their ghost copies. Collective. */
void ComputePeninsulas(const MeshAdjacency& adj, const mfem::Array<int>& label,
                       const NodeLabels& labels, LabelState& state) {
    mfem::ParMesh& pmesh = adj.GetParMesh();
    adj.ExchangeLabels(label, state.ghost_label);
    const LabelView view{adj, label, state.ghost_label};
    state.peninsula.SetSize(pmesh.GetNE());
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        state.peninsula[e] = IsPeninsula(view, e, labels) ? 1 : 0;
    }
    adj.ExchangeLabels(state.peninsula, state.ghost_peninsula);
}

/**
 * @brief Number of face-connected pieces of every grain 1..num_grains (index = grain; global,
 * partition independent: LabelComponents names each piece by its smallest element key, and the
 * distinct (grain, piece) pairs of all ranks are counted). Collective.
 */
std::vector<std::int64_t> GrainPieces(const MeshAdjacency& adj, const mfem::Array<int>& label,
                                      const std::vector<std::uint64_t>& keys, int num_grains) {
    std::vector<std::uint64_t> component;
    LabelComponents(adj, label, keys, component);
    std::set<std::pair<int, std::uint64_t>> mine;
    for (int e = 0; e < label.Size(); ++e) {
        mine.insert({label[e], component[static_cast<std::size_t>(e)]});
    }
    std::vector<std::uint64_t> send;
    for (const auto& [g, c] : mine) {
        send.push_back(static_cast<std::uint64_t>(g));
        send.push_back(c);
    }
    const MPI_Comm comm = adj.GetComm();
    int size = 1;
    MPI_Comm_size(comm, &size);
    int count = static_cast<int>(send.size());
    std::vector<int> counts(static_cast<std::size_t>(size));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r) - 1] + counts[static_cast<std::size_t>(r) - 1];
    }
    std::vector<std::uint64_t> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(send.data(), count, MPI_UINT64_T, all.data(), counts.data(), displs.data(),
                   MPI_UINT64_T, comm);
    std::set<std::pair<std::uint64_t, std::uint64_t>> pieces;
    for (std::size_t k = 0; k + 1 < all.size(); k += 2) {
        pieces.insert({all[k], all[k + 1]});
    }
    std::vector<std::int64_t> per_grain(static_cast<std::size_t>(num_grains) + 1, 0);
    for (const auto& piece : pieces) {
        ++per_grain[static_cast<std::size_t>(piece.first)];
    }
    return per_grain;
}

bool IsBad(TargetStatus s) {
    return s == TargetStatus::FAKE_JUNCTION || s == TargetStatus::OVER_CONSTRAINED;
}

std::int64_t GlobalSum(std::int64_t local, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT64_T, MPI_SUM, comm);
    return local;
}

/** @brief Local potential of element e (file comment, pass step 5). Host. */
int LocalPotential(const MeshAdjacency& adj, const VertexField& field, const NodeTargets& targets,
                   const LabelState& state, int e) {
    const int* en_off = field.ElementVertexOffsets().HostRead();
    const int* en = field.ElementVertices().HostRead();
    int bad = 0;
    for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
        bad += IsBad(targets.Status(en[k])) ? 1 : 0;
    }
    int pen = state.peninsula[e];
    const int* off = adj.Offsets().HostRead();
    const int* nbr = adj.Neighbors().HostRead();
    for (int k = off[e]; k < off[e + 1]; ++k) {
        const int n = nbr[k];
        pen += adj.IsGhost(n) ? state.ghost_peninsula[n - adj.NumElements()] : state.peninsula[n];
    }
    return BAD_NODE_WEIGHT * bad + pen;
}

} // namespace

/*
 * RepairLabels: the passes of the file comment. The node labels are built once and relabeled
 * (NodeLabels::SetElementLabels); the targets are rebuilt after every change of labels.
 */
LabelRepairResult RepairLabels(mfem::ParMesh& pmesh, mfem::Array<int>& element_label,
                               MetricType metric, const SeedSet& seeds, const MetricSetup& setup,
                               const std::vector<int>& grain_seed,
                               const LabelRepairOptions& options) {
    const MPI_Comm comm = pmesh.GetComm();
    ThrowIfAnyRank(comm, options.max_passes < 0 || options.min_grain_elements < 1,
                   "RepairLabels: max_passes must be >= 0 and min_grain_elements >= 1");
    LabelRepairResult result;
    LabelRepairReport& report = result.report;
    result.labels = std::make_unique<NodeLabels>(pmesh, element_label, options.labels);
    const VertexField& field = result.labels->Field();
    // During the passes the targets follow the analytic rules only (the smoothed fallback would
    // hide the bad nodes the repair is for); the final targets add it (options.targets)
    NodeTargetOptions pass_targets = options.targets;
    pass_targets.smoothed = false;
    pass_targets.hold_under_resolved = false; // thin grains' bad nodes are still repaired
    const auto rebuild_targets = [&]() {
        result.targets.reset();
        result.targets = std::make_unique<NodeTargets>(*result.labels, metric, seeds, setup,
                                                       grain_seed, pass_targets);
    };
    rebuild_targets();

    const MeshAdjacency adj(pmesh);
    LabelState state;
    ComputePeninsulas(adj, element_label, *result.labels, state);
    const auto count_bad = [&]() {
        const auto& s = result.targets->Summary();
        return s.status[static_cast<int>(TargetStatus::FAKE_JUNCTION)] +
               s.status[static_cast<int>(TargetStatus::OVER_CONSTRAINED)];
    };
    const auto count_peninsulas = [&]() {
        std::int64_t local = 0;
        for (int e = 0; e < state.peninsula.Size(); ++e) {
            local += state.peninsula[e];
        }
        return GlobalSum(local, comm);
    };
    report.bad_nodes_before = count_bad();
    report.peninsulas_before = count_peninsulas();

    const int ne = pmesh.GetNE();
    const std::vector<std::uint64_t> keys = ComputeElementKeys(pmesh);
    int num_grains = 0;
    for (int e = 0; e < ne; ++e) {
        num_grains = std::max(num_grains, element_label[e]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &num_grains, 1, MPI_INT, MPI_MAX, comm);
    std::set<std::pair<int, int>> undone_flips; // (local element, label)
    // Face-connected pieces per grain: a flip must not split a grain (raise its count); the counts
    // are refreshed after every pass
    std::vector<std::int64_t> pieces = GrainPieces(adj, element_label, keys, num_grains);
    const int* en_off = field.ElementVertexOffsets().HostRead();
    const int* en = field.ElementVertices().HostRead();

    for (int pass = 0; pass < options.max_passes; ++pass) {
        // Step 2: proposals. For every element of an analytic grain at a bad node or that is a
        // peninsula, and every analytic face-neighbor label b, preview the flip: the label sets
        // its nodes would have (a label leaves a node when the element was its only element
        // with it), evaluated with the target rules, and the element's own peninsula flag (its
        // face neighbors keep theirs in the preview). The best flip with a positive gain in
        // potential is proposed.
        const int* loff = result.labels->LabelOffsets().HostRead();
        const int* litems = result.labels->Labels().HostRead();
        const int* lcount = result.labels->LabelCounts().HostRead();
        const LabelView current{adj, element_label, state.ghost_label};
        struct Preview {
            int element;
            int label;
            int first_query;
        };
        std::vector<Preview> previews;
        std::vector<NodeLabelQuery> queries;
        for (int e = 0; e < ne; ++e) {
            const int a = element_label[e];
            if (result.labels->IsFixedLabel(a)) {
                continue;
            }
            bool bad = state.peninsula[e] != 0;
            for (int k = en_off[e]; k < en_off[e + 1] && !bad; ++k) {
                bad = IsBad(result.targets->Status(en[k]));
            }
            if (!bad) {
                continue;
            }
            for (const auto& [b, faces] : NeighborLabelCounts(current, e)) {
                (void)faces;
                if (result.labels->IsFixedLabel(b) || undone_flips.count({e, b}) != 0) {
                    continue;
                }
                previews.push_back({e, b, static_cast<int>(queries.size())});
                for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                    const int v = en[k];
                    std::vector<int> set;
                    for (int j = loff[v]; j < loff[v + 1]; ++j) {
                        if (!(litems[j] == a && lcount[j] == 1)) {
                            set.push_back(litems[j]);
                        }
                    }
                    if (std::find(set.begin(), set.end(), b) == set.end()) {
                        set.insert(std::lower_bound(set.begin(), set.end(), b), b);
                    }
                    NodeLabelQuery q;
                    q.node = v;
                    q.num_labels = static_cast<int>(set.size());
                    for (int j = 0; j < std::min(q.num_labels, NODE_QUERY_LABELS); ++j) {
                        q.labels[j] = set[static_cast<std::size_t>(j)];
                    }
                    queries.push_back(q);
                }
            }
        }
        const std::vector<NodeTargetResult> evaluated = result.targets->Evaluate(queries);
        std::vector<int> proposal(static_cast<std::size_t>(ne), -1);
        std::vector<int> best_gain(static_cast<std::size_t>(ne), 0);
        std::vector<Priority> priority(static_cast<std::size_t>(ne));
        const int* adj_off = adj.Offsets().HostRead();
        const int* adj_nbr = adj.Neighbors().HostRead();
        for (const Preview& pv : previews) {
            const int e = pv.element;
            int bad = 0;
            for (int k = 0; k < en_off[e + 1] - en_off[e]; ++k) {
                bad += IsBad(evaluated[static_cast<std::size_t>(pv.first_query + k)].status) ? 1
                                                                                             : 0;
            }
            // The neighbors keep their flags: a ghost neighbor's other faces are not known here,
            // and treating local neighbors differently would make the gain partition dependent
            LabelView changed{adj, element_label, state.ghost_label, e, pv.label};
            int pen = IsPeninsula(changed, e, *result.labels) ? 1 : 0;
            for (int k = adj_off[e]; k < adj_off[e + 1]; ++k) {
                const int n = adj_nbr[k];
                pen += adj.IsGhost(n) ? state.ghost_peninsula[n - adj.NumElements()]
                                      : state.peninsula[n];
            }
            const int gain = LocalPotential(adj, field, *result.targets, state, e) -
                             (BAD_NODE_WEIGHT * bad + pen);
            const auto es = static_cast<std::size_t>(e);
            if (gain > best_gain[es]) { // ties keep the lower label (previews are ascending)
                best_gain[es] = gain;
                proposal[es] = pv.label;
            }
        }
        std::int64_t proposed = 0;
        for (int e = 0; e < ne; ++e) {
            const auto es = static_cast<std::size_t>(e);
            if (proposal[es] > 0) {
                const std::uint64_t key = keys[es];
                priority[es] = {static_cast<double>(best_gain[es]) * 4294967296.0 +
                                    static_cast<double>(key >> 32),
                                static_cast<double>(key & 0xffffffffULL)};
                ++proposed;
            }
        }
        if (GlobalSum(proposed, comm) == 0) {
            break;
        }
        ++report.passes;

        // Step 3: largest priority within two element rings
        const std::vector<Priority> ring1 = ElementMaximum(field, NodeMaximum(field, priority));
        const std::vector<Priority> ring2 = ElementMaximum(field, NodeMaximum(field, ring1));
        std::vector<int> chosen;
        for (int e = 0; e < ne; ++e) {
            if (proposal[static_cast<std::size_t>(e)] > 0 &&
                priority[static_cast<std::size_t>(e)] == ring2[static_cast<std::size_t>(e)]) {
                chosen.push_back(e);
            }
        }

        // Step 4: grain size guard
        std::vector<std::int64_t> grain_count(static_cast<std::size_t>(num_grains) + 1, 0);
        std::vector<std::int64_t> leaving(static_cast<std::size_t>(num_grains) + 1, 0);
        for (int e = 0; e < ne; ++e) {
            ++grain_count[static_cast<std::size_t>(element_label[e])];
        }
        for (const int e : chosen) {
            ++leaving[static_cast<std::size_t>(element_label[e])];
        }
        MPI_Allreduce(MPI_IN_PLACE, grain_count.data(), num_grains + 1, MPI_INT64_T, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, leaving.data(), num_grains + 1, MPI_INT64_T, MPI_SUM, comm);
        std::int64_t blocked = 0;
        std::vector<int> apply;
        for (const int e : chosen) {
            const auto a = static_cast<std::size_t>(element_label[e]);
            if (grain_count[a] - leaving[a] < options.min_grain_elements) {
                ++blocked;
            } else {
                apply.push_back(e);
            }
        }
        report.blocked += GlobalSum(blocked, comm);

        // Step 5: apply, undo the flips that split a grain, recompute, undo the flips that did
        // not lower the local potential
        std::vector<int> before(apply.size());
        std::vector<int> old_label(apply.size());
        for (std::size_t i = 0; i < apply.size(); ++i) {
            const int e = apply[i];
            before[i] = LocalPotential(adj, field, *result.targets, state, e);
            old_label[i] = element_label[e];
            element_label[e] = proposal[static_cast<std::size_t>(e)];
        }
        {
            const std::vector<std::int64_t> pieces_after =
                GrainPieces(adj, element_label, keys, num_grains);
            std::vector<int> kept_apply;
            std::vector<int> kept_before;
            std::vector<int> kept_old;
            std::int64_t splits = 0;
            for (std::size_t i = 0; i < apply.size(); ++i) {
                const int e = apply[i];
                const auto from = static_cast<std::size_t>(old_label[i]);
                const auto to = static_cast<std::size_t>(element_label[e]);
                if (pieces_after[from] > pieces[from] || pieces_after[to] > pieces[to]) {
                    undone_flips.insert({e, element_label[e]});
                    element_label[e] = old_label[i];
                    ++splits;
                } else {
                    kept_apply.push_back(e);
                    kept_before.push_back(before[i]);
                    kept_old.push_back(old_label[i]);
                }
            }
            splits = GlobalSum(splits, comm);
            report.split_undone += splits;
            if (splits > 0) {
                apply.swap(kept_apply);
                before.swap(kept_before);
                old_label.swap(kept_old);
            }
        }
        result.labels->SetElementLabels(element_label);
        rebuild_targets();
        ComputePeninsulas(adj, element_label, *result.labels, state);
        std::int64_t kept = 0;
        std::int64_t undone = 0;
        for (std::size_t i = 0; i < apply.size(); ++i) {
            const int e = apply[i];
            if (LocalPotential(adj, field, *result.targets, state, e) < before[i]) {
                ++kept;
            } else {
                undone_flips.insert({e, element_label[e]});
                element_label[e] = old_label[i];
                ++undone;
            }
        }
        kept = GlobalSum(kept, comm);
        undone = GlobalSum(undone, comm);
        report.flips += kept;
        report.undone += undone;
        if (undone > 0) {
            result.labels->SetElementLabels(element_label);
            rebuild_targets();
            ComputePeninsulas(adj, element_label, *result.labels, state);
        }
        pieces = GrainPieces(adj, element_label, keys, num_grains);
        if (kept == 0 && GlobalSum(static_cast<std::int64_t>(apply.size()), comm) == 0) {
            break; // everything proposed was blocked
        }
    }

    report.peninsulas_after = count_peninsulas();
    if (options.targets.smoothed || options.targets.hold_under_resolved) {
        result.targets.reset();
        result.targets = std::make_unique<NodeTargets>(*result.labels, metric, seeds, setup,
                                                       grain_seed, options.targets);
    }
    report.bad_nodes_after = count_bad(); // final targets: smoothed fallback, thin grains held
    return result;
}

/*
 * FlipToCurrentOwner: exact best seed at the element centers (one batched search), the distance
 * past the boundary, and the label of a face neighbor of the best seed (ghost labels from
 * MeshAdjacency).
 */
std::int64_t FlipToCurrentOwner(mfem::ParMesh& pmesh, mfem::Array<int>& element_label,
                                MetricType metric, const SeedSet& seeds, const MetricSetup& setup,
                                const std::vector<int>& grain_seed, double hysteresis,
                                std::vector<char>& flipped) {
    const int ne = pmesh.GetNE();
    if (flipped.empty()) {
        flipped.assign(static_cast<std::size_t>(ne), 0);
    }
    const MeshAdjacency adj(pmesh);
    mfem::Array<int> ghost_label;
    adj.ExchangeLabels(element_label, ghost_label);
    mfem::Vector cx(ne), cy(ne), cz(ne), c(3);
    for (int e = 0; e < ne; ++e) {
        pmesh.GetElementCenter(e, c);
        cx(e) = c(0);
        cy(e) = c(1);
        cz(e) = c(2);
    }
    std::vector<int> new_label(static_cast<std::size_t>(ne), -1);
    DispatchMetric(metric, [&](auto metric_tag) {
        using Metric = decltype(metric_tag);
        SeedIndex<axom::SEQ_EXEC, Metric> index;
        index.Build(seeds, setup, 8, axom::execution_space<axom::SEQ_EXEC>::allocatorID());
        mfem::Array<BestPair> best;
        if (ne > 0) {
            index.FindBest(cx, cy, cz, best, false);
        }
        const BestPair* b = (ne > 0) ? best.HostRead() : nullptr;
        const Metric& f = index.GetMetric();
        const int* off = adj.Offsets().HostRead();
        const int* nbr = adj.Neighbors().HostRead();
        for (int e = 0; e < ne; ++e) {
            const int own = grain_seed[static_cast<std::size_t>(element_label[e] - 1)];
            if (own < 0 || flipped[static_cast<std::size_t>(e)] != 0 || b[e].id1 < 0 ||
                b[e].id1 == own) {
                continue;
            }
            double ga[3];
            double gb[3];
            f.Grad(own, cx(e), cy(e), cz(e), ga);
            f.Grad(b[e].id1, cx(e), cy(e), cz(e), gb);
            const double g = std::sqrt((ga[0] - gb[0]) * (ga[0] - gb[0]) +
                                       (ga[1] - gb[1]) * (ga[1] - gb[1]) +
                                       (ga[2] - gb[2]) * (ga[2] - gb[2]));
            const double gap = (f.Eval(own, cx(e), cy(e), cz(e)) - b[e].f1) / g;
            if (!(g > 0.0) || gap <= hysteresis * pmesh.GetElementSize(e, 0)) {
                continue;
            }
            int target = -1;
            for (int k = off[e]; k < off[e + 1]; ++k) {
                const int n = nbr[k];
                const int lb =
                    adj.IsGhost(n) ? ghost_label[n - adj.NumElements()] : element_label[n];
                if (grain_seed[static_cast<std::size_t>(lb - 1)] == b[e].id1 &&
                    (target < 0 || lb < target)) {
                    target = lb;
                }
            }
            new_label[static_cast<std::size_t>(e)] = target;
        }
    });
    std::int64_t flips = 0;
    for (int e = 0; e < ne; ++e) {
        if (new_label[static_cast<std::size_t>(e)] > 0) {
            element_label[e] = new_label[static_cast<std::size_t>(e)];
            flipped[static_cast<std::size_t>(e)] = 1;
            ++flips;
        }
    }
    return GlobalSum(flips, pmesh.GetComm());
}

} // namespace exaconstit::microstructure
