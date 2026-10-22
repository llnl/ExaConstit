#include "microstructure/labeling/fragment_cleanup.hpp"

#include "microstructure/mesh/element_key.hpp"
#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <limits>
#include <map>
#include <mpi.h>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace exaconstit::microstructure {

namespace {

/*
 * Mix64: the splitmix64 finalizer, a fixed one-to-one scrambling of 64-bit integers in which every
 * input bit affects every output bit (see mesh/element_key.cpp). Used to derive source ids.
 */
std::uint64_t Mix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/** Bit-for-bit reinterpretation between unsigned and signed 64-bit integers (for MPI_INT64_T). */
std::int64_t BitsToSigned(std::uint64_t u) {
    std::int64_t s = 0;
    std::memcpy(&s, &u, sizeof(s));
    return s;
}
std::uint64_t BitsToUnsigned(std::int64_t s) {
    std::uint64_t u = 0;
    std::memcpy(&u, &s, sizeof(u));
    return u;
}

// AnyRank (a logical OR of a flag over the ranks: one MPI_Allreduce with MPI_MAX, so a loop
// controlled by it stops on all ranks in the same iteration) comes from
// util/collective_errors.hpp.

/*
 * UnionFind (also called a disjoint-set forest): a forest over the local elements. parent[i] is
 * another element of the same group (or i itself for the group's root); two elements are in the
 * same group exactly when Find gives the same root.
 *  - Find follows the parent pointers to the root and, on the way, points every visited element at
 *    its grandparent ("path halving"), which keeps the trees shallow so later Finds are fast.
 *  - Union finds both roots and hangs one under the other (the larger index under the smaller;
 *    which one does not matter for the result, only for speed).
 */
class UnionFind {
public:
    explicit UnionFind(int n) : m_parent(static_cast<std::size_t>(n)) {
        for (int i = 0; i < n; i++) {
            m_parent[static_cast<std::size_t>(i)] = i;
        }
    }

    int Find(int i) {
        while (m_parent[static_cast<std::size_t>(i)] != i) {
            const int grandparent =
                m_parent[static_cast<std::size_t>(m_parent[static_cast<std::size_t>(i)])];
            m_parent[static_cast<std::size_t>(i)] = grandparent;
            i = grandparent;
        }
        return i;
    }

    void Union(int a, int b) {
        const int root_a = Find(a);
        const int root_b = Find(b);
        if (root_a == root_b) {
            return;
        }
        if (root_a < root_b) {
            m_parent[static_cast<std::size_t>(root_b)] = root_a;
        } else {
            m_parent[static_cast<std::size_t>(root_a)] = root_b;
        }
    }

private:
    std::vector<int> m_parent;
};

/*
 * AllGatherInt64: every rank contributes a list of int64 values (any length, possibly empty) and
 * every rank receives the concatenation of all lists in rank order.
 *  1. MPI_Allgather of the list lengths (each rank learns every rank's length).
 *  2. Exclusive prefix sum of the lengths = where each rank's list starts in the result (an
 *     "exclusive scan": entry r is the sum of the lengths of ranks 0..r-1).
 *  3. MPI_Allgatherv moves the lists.
 * MPI counts are ints, so the total is checked against INT_MAX; every rank sees the same lengths
 * and therefore throws together. Collective.
 */
std::vector<std::int64_t> AllGatherInt64(const std::vector<std::int64_t>& local, MPI_Comm comm) {
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    const long long local_len = static_cast<long long>(local.size());
    const int local_count = (local_len > INT_MAX) ? -1 : static_cast<int>(local_len);

    // Step 1: everyone's list length
    std::vector<int> counts(static_cast<std::size_t>(nranks), 0);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    // Step 2: start offsets (exclusive prefix sum), with an overflow check
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    long long total = 0;
    bool overflow = false;
    for (int r = 0; r < nranks; r++) {
        if (counts[static_cast<std::size_t>(r)] < 0) {
            overflow = true;
            break;
        }
        displs[static_cast<std::size_t>(r)] = static_cast<int>(total);
        total += counts[static_cast<std::size_t>(r)];
        if (total > INT_MAX) {
            overflow = true;
            break;
        }
    }
    if (overflow) {
        throw std::runtime_error("CleanupFragments: fragment tables exceed the MPI count limit");
    }

    // Step 3: move the data
    std::vector<std::int64_t> all(static_cast<std::size_t>(total));
    MPI_Allgatherv(local.data(), local_count, MPI_INT64_T, all.data(), counts.data(),
                   displs.data(), MPI_INT64_T, comm);
    return all;
}

/*
 * CombineRecords: `flat` holds records of `stride` int64 values each. Records whose first
 * `key_len` values are equal are combined into one record by summing the remaining values. The
 * output is sorted by the key fields (as signed integers). Used to add up the per-rank parts of a
 * fragment's element count or of a face count after AllGatherInt64. Integer sums are exact and
 * independent of order, so the result does not depend on the rank count.
 */
std::vector<std::int64_t> CombineRecords(const std::vector<std::int64_t>& flat, int stride,
                                         int key_len) {
    const std::size_t s = static_cast<std::size_t>(stride);
    const std::size_t num_records = flat.size() / s;
    std::vector<std::size_t> order(num_records);
    std::iota(order.begin(), order.end(), std::size_t{0});
    auto key_less = [&](std::size_t a, std::size_t b) {
        for (int k = 0; k < key_len; k++) {
            const std::int64_t va = flat[a * s + static_cast<std::size_t>(k)];
            const std::int64_t vb = flat[b * s + static_cast<std::size_t>(k)];
            if (va != vb) {
                return va < vb;
            }
        }
        return false;
    };
    std::sort(order.begin(), order.end(), key_less);

    std::vector<std::int64_t> out;
    for (std::size_t i = 0; i < num_records; i++) {
        const std::size_t r = order[i];
        const bool same_as_last = !out.empty() && !key_less(order[i - 1], r) &&
                                  !key_less(r, order[i - 1]);
        if (same_as_last) {
            // Same key as the previous record: add the value fields into the last output record
            const std::size_t last = out.size() - s;
            for (int k = key_len; k < stride; k++) {
                const std::size_t field = static_cast<std::size_t>(k);
                out[last + field] += flat[r * s + field];
            }
        } else {
            for (int k = 0; k < stride; k++) {
                out.push_back(flat[r * s + static_cast<std::size_t>(k)]);
            }
        }
    }
    return out;
}

/*
 * CheckAllRanks: throws std::invalid_argument on every rank when `bad` is true on any rank
 * (collective; see AnyRank).
 */
void CheckAllRanks(bool bad, const std::string& message, MPI_Comm comm) {
    if (AnyRank(bad, comm)) {
        throw std::invalid_argument(message + (bad ? " (on this rank)" : " (on another rank)"));
    }
}

/*
 * ComponentsImpl: the two-pass component search described in the header (LabelComponents),
 * without input checks.
 *
 *  Pass 1 (local union-find): every local face between two elements with the same label joins
 *  their groups. root_id[root] = smallest element key in the group.
 *  Pass 2 (min-propagation across ranks), repeated:
 *   a. every element takes its group's current id;
 *   b. the ids are sent to the neighboring ranks (MeshAdjacency::ExchangeKeys);
 *   c. for every shared face whose ghost element has the same label and a smaller id, the local
 *      group takes that id;
 *   d. stop when no rank changed anything (AnyRank, collective).
 *  The ids only decrease and always name an element of the same fragment, and a smaller id
 *  crosses at least one more rank boundary per round, so at the end every element holds the
 *  smallest key of its whole fragment; the number of rounds is one more than the largest number
 *  of rank-boundary crossings on a path from the smallest-key element. When the loop stops, the
 *  ghost ids received in the last round are final too (nobody changed after sending them).
 */
int ComponentsImpl(const MeshAdjacency& adjacency, const int* label, const int* ghost_label,
                   const std::vector<std::uint64_t>& key, std::vector<std::uint64_t>& component,
                   std::vector<std::uint64_t>& ghost_component) {
    const int ne = adjacency.NumElements();
    const int* offsets = adjacency.Offsets().HostRead();
    const int* neighbors = adjacency.Neighbors().HostRead();

    // Pass 1: local groups of equally labeled, face-connected elements
    UnionFind groups(ne);
    for (int e = 0; e < ne; e++) {
        for (int k = offsets[e]; k < offsets[e + 1]; k++) {
            const int n = neighbors[k];
            if (n < ne && label[n] == label[e]) {
                groups.Union(e, n);
            }
        }
    }
    std::vector<std::uint64_t> root_id(static_cast<std::size_t>(ne),
                                       std::numeric_limits<std::uint64_t>::max());
    for (int e = 0; e < ne; e++) {
        const std::size_t root = static_cast<std::size_t>(groups.Find(e));
        root_id[root] = std::min(root_id[root], key[static_cast<std::size_t>(e)]);
    }

    // Pass 2: min-propagation across shared faces until no rank changes
    component.assign(static_cast<std::size_t>(ne), 0);
    int rounds = 0;
    while (true) {
        rounds++;
        // a. every element takes its group's id
        for (int e = 0; e < ne; e++) {
            component[static_cast<std::size_t>(e)] =
                root_id[static_cast<std::size_t>(groups.Find(e))];
        }
        // b. send ids to the neighboring ranks
        adjacency.ExchangeKeys(component, ghost_component);
        // c. adopt smaller ids of same-label ghost elements
        bool changed = false;
        for (int e = 0; e < ne; e++) {
            for (int k = offsets[e]; k < offsets[e + 1]; k++) {
                const int n = neighbors[k];
                if (n < ne) {
                    continue;
                }
                const int g = n - ne;
                const std::size_t root = static_cast<std::size_t>(groups.Find(e));
                if (ghost_label[g] == label[e] &&
                    ghost_component[static_cast<std::size_t>(g)] < root_id[root]) {
                    root_id[root] = ghost_component[static_cast<std::size_t>(g)];
                    changed = true;
                }
            }
        }
        // d. collective stop test
        if (!AnyRank(changed, adjacency.GetComm())) {
            break;
        }
    }
    return rounds;
}

/** One fragment in the replicated fragment table. */
struct Fragment {
    int label;          ///< working grain label
    std::uint64_t id;   ///< fragment id (smallest element key)
    std::int64_t count; ///< global number of elements
    bool anchor;        ///< contains the grain's anchor element
    bool tiny;          ///< every element is marked tiny (SPLIT_LARGE_MERGE_TINY; else false)
};

/** Mutable state of one CleanupFragments run (host data). */
struct CleanupState {
    const MeshAdjacency* adjacency = nullptr;
    MPI_Comm comm = MPI_COMM_NULL;
    int ne = 0;
    const int* offsets = nullptr;
    const int* neighbors = nullptr;
    std::vector<std::uint64_t> key;    ///< size NE, element keys
    mfem::Array<int> label;            ///< size NE, working labels 1..num_labels
    mfem::Array<int> ghost_label;      ///< size NG, refreshed by RefreshGhostLabels
    int num_labels = 0;                ///< working labels in use are 1..num_labels
    int num_input_grains = 0;
    std::vector<int> parent;           ///< index label-1: input grain (1-based) it came from
    std::vector<std::int64_t> source;  ///< index label-1: source id
    std::vector<std::int64_t> splits;  ///< index input grain-1: fragments split off so far
    const std::vector<std::uint64_t>* anchor = nullptr; ///< null unless KeepRule::SEED_ELEMENT
    /**
     * Size NE (SPLIT_LARGE_MERGE_TINY only, else empty): 1 for the elements of fragments that
     * were classified tiny at the first fragment search. The classification is stored per
     * element, not per fragment id, so it survives relabeling: a fragment is tiny exactly when
     * all of its elements carry the mark (pieces of tiny fragments that join stay tiny; a tiny
     * piece that joins a large fragment does not make it tiny).
     */
    std::vector<char> tiny_element;
};

/** Sends the current working labels to the neighboring ranks (collective). */
void RefreshGhostLabels(CleanupState& state) {
    state.adjacency->ExchangeLabels(state.label, state.ghost_label);
}

/** Label of neighbor n (local element or NE + ghost index). */
int NeighborLabel(const CleanupState& state, int n) {
    return (n < state.ne) ? state.label[n] : state.ghost_label[n - state.ne];
}

/*
 * GlobalLabelCounts: number of elements of every working label over all ranks. Each rank counts
 * its own elements, then MPI_Allreduce with MPI_SUM adds the per-rank arrays (exact integer
 * sums). Result index label-1. Collective.
 */
std::vector<std::int64_t> GlobalLabelCounts(const CleanupState& state) {
    std::vector<std::int64_t> local(static_cast<std::size_t>(state.num_labels), 0);
    for (int e = 0; e < state.ne; e++) {
        local[static_cast<std::size_t>(state.label[e] - 1)] += 1;
    }
    std::vector<std::int64_t> global(local.size(), 0);
    MPI_Allreduce(local.data(), global.data(), state.num_labels, MPI_INT64_T, MPI_SUM,
                  state.comm);
    return global;
}

/*
 * FragmentTable: the replicated list of fragments of every grain that has more than one.
 *
 * Steps:
 *  1. Which grains have several fragments: per label, the smallest and the largest fragment id on
 *     this rank, combined with MPI_Allreduce (MIN and MAX, unsigned 64-bit). A grain has several
 *     fragments exactly when the global smallest and largest ids differ. This costs two
 *     reductions of one number per grain instead of gathering every grain's fragments.
 *  2. Each rank lists (label, fragment id, local element count, local anchor count, local count
 *     of tiny-marked elements) for the fragments of those grains that it holds.
 *  3. AllGatherInt64 + CombineRecords add up the per-rank parts.
 *  4. The table is sorted by (label, fragment id), comparing ids as unsigned numbers.
 * Collective.
 */
std::vector<Fragment> FragmentTable(const CleanupState& state,
                                    const std::vector<std::uint64_t>& component) {
    const std::size_t num_labels = static_cast<std::size_t>(state.num_labels);

    // Step 1: grains with more than one fragment
    std::vector<std::uint64_t> lo(num_labels, std::numeric_limits<std::uint64_t>::max());
    std::vector<std::uint64_t> hi(num_labels, 0);
    for (int e = 0; e < state.ne; e++) {
        const std::size_t l = static_cast<std::size_t>(state.label[e] - 1);
        lo[l] = std::min(lo[l], component[static_cast<std::size_t>(e)]);
        hi[l] = std::max(hi[l], component[static_cast<std::size_t>(e)]);
    }
    std::vector<std::uint64_t> global_lo(num_labels, 0);
    std::vector<std::uint64_t> global_hi(num_labels, 0);
    MPI_Allreduce(lo.data(), global_lo.data(), state.num_labels, MPI_UINT64_T, MPI_MIN,
                  state.comm);
    MPI_Allreduce(hi.data(), global_hi.data(), state.num_labels, MPI_UINT64_T, MPI_MAX,
                  state.comm);

    // Step 2: local parts of those grains' fragments, keyed by (label, id)
    std::map<std::pair<int, std::uint64_t>, std::array<std::int64_t, 3>> local;
    for (int e = 0; e < state.ne; e++) {
        const int l = state.label[e];
        const std::size_t li = static_cast<std::size_t>(l - 1);
        if (global_hi[li] == global_lo[li]) {
            continue; // one fragment (or no element): nothing to decide
        }
        std::array<std::int64_t, 3>& entry = local[{l, component[static_cast<std::size_t>(e)]}];
        entry[0] += 1;
        const bool is_anchor = state.anchor != nullptr && l <= state.num_input_grains &&
                               state.key[static_cast<std::size_t>(e)] ==
                                   (*state.anchor)[li];
        entry[1] += is_anchor ? 1 : 0;
        const bool is_tiny = !state.tiny_element.empty() &&
                             state.tiny_element[static_cast<std::size_t>(e)] != 0;
        entry[2] += is_tiny ? 1 : 0;
    }

    // Step 3: records (label, id bits, count, anchor count, tiny count), gathered and summed
    std::vector<std::int64_t> flat;
    for (const auto& item : local) {
        flat.push_back(item.first.first);
        flat.push_back(BitsToSigned(item.first.second));
        flat.push_back(item.second[0]);
        flat.push_back(item.second[1]);
        flat.push_back(item.second[2]);
    }
    const std::vector<std::int64_t> combined =
        CombineRecords(AllGatherInt64(flat, state.comm), 5, 2);

    // Step 4: unpack and sort by (label, unsigned id)
    std::vector<Fragment> table;
    for (std::size_t r = 0; r + 4 < combined.size(); r += 5) {
        table.push_back({static_cast<int>(combined[r]), BitsToUnsigned(combined[r + 1]),
                         combined[r + 2], combined[r + 3] > 0,
                         combined[r + 4] == combined[r + 2]});
    }
    std::sort(table.begin(), table.end(), [](const Fragment& a, const Fragment& b) {
        return std::tie(a.label, a.id) < std::tie(b.label, b.id);
    });
    return table;
}

/*
 * ExtraFragments: from the fragment table (sorted by label, then id), the fragments that are not
 * kept. Per grain the kept fragment is chosen among the fragments that are not tiny (all of them
 * except under SPLIT_LARGE_MERGE_TINY): the anchor fragment if one exists (SEED_ELEMENT), else
 * the one with the most elements, ties going to the smallest id (the first in table order). If
 * every fragment of the grain is tiny, the choice is made among all of them. Preferring non-tiny
 * fragments means a seed that lies in a tiny piece never keeps that piece while the grain's large
 * body is split off with a derived id (and a new orientation). The returned list keeps the table
 * order (label, then id).
 */
std::vector<Fragment> ExtraFragments(const std::vector<Fragment>& table) {
    std::vector<Fragment> extra;
    std::size_t begin = 0;
    while (begin < table.size()) {
        // Fragments [begin, end) belong to one grain
        std::size_t end = begin;
        bool any_large = false;
        while (end < table.size() && table[end].label == table[begin].label) {
            any_large = any_large || !table[end].tiny;
            end++;
        }
        // Kept fragment among the candidates: anchor first, else largest (strict > keeps the
        // smallest id on ties)
        std::size_t kept = end;
        for (std::size_t i = begin; i < end; i++) {
            if (any_large && table[i].tiny) {
                continue; // not a candidate
            }
            if (kept == end) {
                kept = i;
                continue;
            }
            if (table[kept].anchor) {
                break;
            }
            if (table[i].anchor || table[i].count > table[kept].count) {
                kept = i;
            }
        }
        for (std::size_t i = begin; i < end; i++) {
            if (i != kept) {
                extra.push_back(table[i]);
            }
        }
        begin = end;
    }
    return extra;
}

/*
 * GrainTotals: element count of every grain listed in the fragment table (the sum of its
 * fragments' counts; the table lists all fragments of every grain with more than one), keyed by
 * label. Exact integer sums of replicated data, so identical on every rank.
 */
std::map<int, std::int64_t> GrainTotals(const std::vector<Fragment>& table) {
    std::map<int, std::int64_t> total;
    for (const Fragment& f : table) {
        total[f.label] += f.count;
    }
    return total;
}

/*
 * IsTinyFragment (SPLIT_LARGE_MERGE_TINY): a fragment is tiny when it has fewer than
 * min_fragment_elements elements, or less than min_fragment_fraction of its grain's elements.
 * Both tests use the replicated integer counts; the fraction test count < fraction * total is
 * evaluated in the same floating-point operations on every rank, so every rank decides alike.
 */
bool IsTinyFragment(const Fragment& f, std::int64_t grain_total,
                    const FragmentCleanupParameters& params) {
    if (f.count < static_cast<std::int64_t>(params.min_fragment_elements)) {
        return true;
    }
    return static_cast<double>(f.count) <
           params.min_fragment_fraction * static_cast<double>(grain_total);
}

/*
 * NewGrainFromFragment: gives a fragment a new working label (num_labels + 1). Its parent is the
 * input grain of the fragment's label, and its source id is derived from the parent's source id
 * and the number of fragments split off that parent so far (1 for the first).
 */
int NewGrainFromFragment(CleanupState& state, int old_label) {
    const int parent = state.parent[static_cast<std::size_t>(old_label - 1)];
    std::int64_t& ordinal = state.splits[static_cast<std::size_t>(parent - 1)];
    ordinal += 1;
    const std::int64_t parent_source = state.source[static_cast<std::size_t>(parent - 1)];
    state.num_labels += 1;
    state.parent.push_back(parent);
    state.source.push_back(DeriveFragmentSourceId(parent_source, ordinal));
    return state.num_labels;
}

/**
 * Fragment key (label, id). A fragment id (the smallest element key of the fragment) identifies
 * a fragment only together with its label: keying by the id alone would mix up two fragments of
 * different grains if two elements ever had the same element key (a 64-bit hash collision).
 */
using FragmentKey = std::pair<int, std::uint64_t>;

/** Relabels every local element whose (label, fragment id) is in `new_label_of_fragment`. */
void RelabelFragments(CleanupState& state, const std::vector<std::uint64_t>& component,
                      const std::map<FragmentKey, int>& new_label_of_fragment) {
    if (new_label_of_fragment.empty()) {
        return;
    }
    for (int e = 0; e < state.ne; e++) {
        const auto it =
            new_label_of_fragment.find({state.label[e], component[static_cast<std::size_t>(e)]});
        if (it != new_label_of_fragment.end()) {
            state.label[e] = it->second;
        }
    }
}

/** Face counts between one moving piece and one neighboring label. */
struct FaceCount {
    std::int64_t stable = 0; ///< faces to stable (not moving) elements of the label
    std::int64_t moving = 0; ///< faces to elements that are themselves about to move
};

/*
 * GatherFaceCounts: for moving pieces identified by `piece_of_element[e]` (a piece key >= 0, or
 * -1 for elements that do not move), counts the faces between each piece and each other label,
 * split into faces to stable and to moving neighbors (`moving_neighbor(n)`). Each face is counted
 * once, from the side of the moving element that owns it (a shared face is seen by exactly one
 * rank from the moving side). The per-rank counts are gathered and summed (exact integers).
 * Result: map (piece key, neighbor label) -> counts, identical on every rank. Collective.
 */
template <typename MovingNeighbor>
std::map<std::pair<std::int64_t, int>, FaceCount>
GatherFaceCounts(const CleanupState& state, const std::vector<std::int64_t>& piece_of_element,
                 MovingNeighbor moving_neighbor) {
    // Local counts keyed by (piece, neighbor label, stable flag)
    std::map<std::tuple<std::int64_t, int, int>, std::int64_t> local;
    for (int e = 0; e < state.ne; e++) {
        const std::int64_t piece = piece_of_element[static_cast<std::size_t>(e)];
        if (piece < 0) {
            continue; // this element does not move
        }
        for (int k = state.offsets[e]; k < state.offsets[e + 1]; k++) {
            const int n = state.neighbors[k];
            const int nlabel = NeighborLabel(state, n);
            if (nlabel == state.label[e]) {
                continue; // same grain: not a boundary face of the piece
            }
            const int stable = moving_neighbor(n) ? 0 : 1;
            local[{piece, nlabel, stable}] += 1;
        }
    }

    // Gather records (piece, label, stable flag, count) and add them up
    std::vector<std::int64_t> flat;
    for (const auto& item : local) {
        flat.push_back(std::get<0>(item.first));
        flat.push_back(std::get<1>(item.first));
        flat.push_back(std::get<2>(item.first));
        flat.push_back(item.second);
    }
    const std::vector<std::int64_t> combined =
        CombineRecords(AllGatherInt64(flat, state.comm), 4, 3);

    std::map<std::pair<std::int64_t, int>, FaceCount> counts;
    for (std::size_t r = 0; r + 3 < combined.size(); r += 4) {
        FaceCount& c = counts[{combined[r], static_cast<int>(combined[r + 1])}];
        if (combined[r + 2] == 1) {
            c.stable += combined[r + 3];
        } else {
            c.moving += combined[r + 3];
        }
    }
    return counts;
}

/** Decision for one moving piece in a merge round. */
struct MergeDecision {
    int target = 0;        ///< label to merge into (0: none)
    bool isolated = false; ///< no face to any other label at all
};

/*
 * DecideMerge: target = the label with the most faces to stable elements, ties going to the
 * lowest label (labels are visited in increasing order and only a strictly larger count
 * replaces the current best). No stable faces: no target; no faces at all: isolated.
 */
MergeDecision DecideMerge(const std::map<std::pair<std::int64_t, int>, FaceCount>& counts,
                          std::int64_t piece) {
    MergeDecision decision;
    std::int64_t best = 0;
    std::int64_t total = 0;
    // The map is sorted by (piece, label): walk this piece's entries in increasing label
    for (auto it = counts.lower_bound({piece, std::numeric_limits<int>::min()});
         it != counts.end() && it->first.first == piece; ++it) {
        total += it->second.stable + it->second.moving;
        if (it->second.stable > best) {
            best = it->second.stable;
            decision.target = it->first.second;
        }
    }
    decision.isolated = (total == 0);
    return decision;
}

/*
 * MostFacesLabel: the label with the most faces to the piece, counting faces to stable and to
 * moving elements alike; ties go to the lowest label. Used by the no-progress fallback of the
 * merge rounds, where every candidate touches only moving pieces. Precondition: the piece has
 * at least one face to another label (isolated pieces are handled before the fallback).
 */
int MostFacesLabel(const std::map<std::pair<std::int64_t, int>, FaceCount>& counts,
                   std::int64_t piece) {
    int target = 0;
    std::int64_t best = 0;
    for (auto it = counts.lower_bound({piece, std::numeric_limits<int>::min()});
         it != counts.end() && it->first.first == piece; ++it) {
        const std::int64_t faces = it->second.stable + it->second.moving;
        if (faces > best) {
            best = faces;
            target = it->first.second;
        }
    }
    return target;
}

/*
 * MarkTinyElements (SPLIT_LARGE_MERGE_TINY, first fragment search only): classifies every
 * fragment of every grain with several fragments against the element count of its input grain
 * (IsTinyFragment with GrainTotals) and marks the elements of the tiny ones in
 * state.tiny_element. Classifying once, against the grain as it was labeled, is what the
 * thresholds mean: a piece that is tiny relative to its grain stays tiny after the grain's large
 * pieces have been split off (re-classifying against the shrinking remainder could turn it into
 * a "large" piece and split it). Host only; the table is replicated, so every rank marks alike.
 * Returns the table with the tiny flags set.
 */
std::vector<Fragment> MarkTinyElements(CleanupState& state, const std::vector<Fragment>& table,
                                       const std::vector<std::uint64_t>& component,
                                       const FragmentCleanupParameters& params) {
    const std::map<int, std::int64_t> total = GrainTotals(table);
    std::set<FragmentKey> tiny;
    std::vector<Fragment> marked = table;
    for (Fragment& f : marked) {
        f.tiny = IsTinyFragment(f, total.at(f.label), params);
        if (f.tiny) {
            tiny.insert({f.label, f.id});
        }
    }
    state.tiny_element.assign(static_cast<std::size_t>(state.ne), 0);
    for (int e = 0; e < state.ne; e++) {
        if (tiny.count({state.label[e], component[static_cast<std::size_t>(e)]}) != 0) {
            state.tiny_element[static_cast<std::size_t>(e)] = 1;
        }
    }
    return marked;
}

/*
 * FragmentPhase: applies the fragment policy to the working labels. The first search's number
 * of extra fragments is recorded as report.fragments_found.
 *
 * SPLIT: one search; every extra fragment becomes a new grain (in table order: label, then id).
 * MERGE: rounds of
 *   1. refresh ghost labels, search fragments, list extra fragments (stop when there are none);
 *   2. face counts of every extra fragment toward stable / moving neighbors;
 *   3. decide: merge into the stable neighbor label with most faces; isolated fragments (no face
 *      to any other label) become new grains; if nothing merged and nothing was isolated (every
 *      waiting fragment touches only other waiting fragments), the largest waiting fragment
 *      (ties: smallest id) alone merges into the neighbor label with the most faces, moving
 *      neighbors included. Only one piece moves in such a round, so no two pieces can swap
 *      labels, and the piece joins a fragment of its target label, so the number of extra
 *      fragments drops by at least one: every round makes progress and the loop terminates;
 *   4. relabel.
 * SPLIT_LARGE_MERGE_TINY: at the first search every fragment is classified tiny or large once
 *   (MarkTinyElements). In a round with large extra fragments those are split into new grains
 *   and the search repeats (each split removes an extra fragment, so this happens finitely
 *   often; max_rounds does not count these rounds); rounds with only tiny extra fragments merge
 *   them exactly like MERGE (rounds counted in report.merge_rounds and capped by max_rounds).
 * NONE: one search, for the report only.
 * Collective.
 */
void FragmentPhase(CleanupState& state, const FragmentCleanupParameters& params,
                   FragmentCleanupReport& report) {
    std::vector<std::uint64_t> component;
    std::vector<std::uint64_t> ghost_component;
    bool first_search = true;
    int round = 0;
    const bool split_large = (params.policy == FragmentPolicy::SPLIT_LARGE_MERGE_TINY);
    while (true) {
        // Step 1: fragments of the current labels
        RefreshGhostLabels(state);
        const int propagation_rounds =
            ComponentsImpl(*state.adjacency, state.label.HostRead(),
                           state.ghost_label.HostRead(), state.key, component, ghost_component);
        report.max_propagation_rounds = std::max(report.max_propagation_rounds, propagation_rounds);
        std::vector<Fragment> table = FragmentTable(state, component);
        if (first_search && split_large) {
            table = MarkTinyElements(state, table, component, params);
        }
        std::vector<Fragment> extra = ExtraFragments(table);
        if (first_search) {
            report.fragments_found = static_cast<std::int64_t>(extra.size());
            first_search = false;
        }
        if (extra.empty() || params.policy == FragmentPolicy::NONE) {
            return;
        }

        // SPLIT_LARGE_MERGE_TINY: large extra fragments become new grains, then search again;
        // once only tiny ones are left they go through the merge rounds below like MERGE
        if (split_large) {
            std::vector<Fragment> tiny;
            std::map<FragmentKey, int> new_label;
            for (const Fragment& f : extra) {
                if (f.tiny) {
                    tiny.push_back(f);
                } else {
                    new_label[{f.label, f.id}] = NewGrainFromFragment(state, f.label);
                }
            }
            if (!new_label.empty()) {
                RelabelFragments(state, component, new_label);
                report.fragments_split += static_cast<std::int64_t>(new_label.size());
                continue; // search again with the new labels
            }
            extra = tiny; // only tiny fragments left: merge them
        }

        // SPLIT: every extra fragment becomes a new grain, then done
        if (params.policy == FragmentPolicy::SPLIT) {
            std::map<FragmentKey, int> new_label;
            for (const Fragment& f : extra) {
                new_label[{f.label, f.id}] = NewGrainFromFragment(state, f.label);
            }
            RelabelFragments(state, component, new_label);
            report.fragments_split += static_cast<std::int64_t>(extra.size());
            return;
        }

        // MERGE (and the tiny fragments of SPLIT_LARGE_MERGE_TINY)
        round++;
        if (round > params.max_rounds) {
            throw std::runtime_error("CleanupFragments: fragment merging exceeded max_rounds");
        }
        report.merge_rounds = round;

        // Step 2: face counts of the moving (extra) fragments; pieces are keyed by their
        // position in `extra`, found through the fragment key (label, id)
        std::map<FragmentKey, std::size_t> extra_index;
        for (std::size_t i = 0; i < extra.size(); i++) {
            extra_index[{extra[i].label, extra[i].id}] = i;
        }
        std::vector<std::int64_t> piece(static_cast<std::size_t>(state.ne), -1);
        for (int e = 0; e < state.ne; e++) {
            const auto it =
                extra_index.find({state.label[e], component[static_cast<std::size_t>(e)]});
            if (it != extra_index.end()) {
                piece[static_cast<std::size_t>(e)] = static_cast<std::int64_t>(it->second);
            }
        }
        auto moving_neighbor = [&](int n) {
            const std::uint64_t id = (n < state.ne)
                                         ? component[static_cast<std::size_t>(n)]
                                         : ghost_component[static_cast<std::size_t>(n - state.ne)];
            return extra_index.count({NeighborLabel(state, n), id}) != 0;
        };
        const auto counts = GatherFaceCounts(state, piece, moving_neighbor);

        // Step 3: decisions (identical on every rank: same replicated data, same order)
        std::map<FragmentKey, int> new_label;
        bool progress = false;
        for (std::size_t i = 0; i < extra.size(); i++) {
            const MergeDecision d = DecideMerge(counts, static_cast<std::int64_t>(i));
            if (d.target > 0) {
                new_label[{extra[i].label, extra[i].id}] = d.target;
                report.fragments_merged += 1;
                progress = true;
            } else if (d.isolated) {
                new_label[{extra[i].label, extra[i].id}] =
                    NewGrainFromFragment(state, extra[i].label);
                report.fragments_split += 1;
                progress = true;
            }
        }
        if (!progress) {
            // Every waiting fragment touches only other waiting fragments: the largest one
            // (ties: smallest id) alone merges into its neighbor label with the most faces,
            // moving neighbors included (ties: lowest label)
            std::size_t best = 0;
            for (std::size_t i = 1; i < extra.size(); i++) {
                const bool larger = extra[i].count > extra[best].count;
                const bool tie_smaller_id =
                    extra[i].count == extra[best].count && extra[i].id < extra[best].id;
                if (larger || tie_smaller_id) {
                    best = i;
                }
            }
            new_label[{extra[best].label, extra[best].id}] =
                MostFacesLabel(counts, static_cast<std::int64_t>(best));
            report.fragments_merged += 1;
        }

        // Step 4: relabel
        RelabelFragments(state, component, new_label);
    }
}

/*
 * SmallGrainPhase: merges grains with fewer than min_grain_elements elements. Rounds of
 *   1. global element counts; candidates = labels with 0 < count < min that are not settled;
 *      stop when there are none;
 *   2. refresh ghost labels; face counts of every candidate grain toward stable / candidate
 *      neighbors (pieces keyed by label);
 *   3. decide: merge into the stable neighbor label with most faces (ties: lowest label);
 *      a candidate with no neighbor at all is settled (kept as is); if nothing merged and nothing
 *      settled, the largest candidate (ties: lowest label) is settled;
 *   4. relabel.
 * Merge targets are never candidates, so no grain moves twice in one round. Collective.
 */
void SmallGrainPhase(CleanupState& state, const FragmentCleanupParameters& params,
                     FragmentCleanupReport& report) {
    if (params.min_grain_elements <= 1) {
        return;
    }
    std::vector<char> settled(static_cast<std::size_t>(state.num_labels), 0);
    int round = 0;
    while (true) {
        // Step 1: candidates
        const std::vector<std::int64_t> counts = GlobalLabelCounts(state);
        std::vector<char> candidate(static_cast<std::size_t>(state.num_labels), 0);
        std::vector<int> candidates;
        for (int l = 1; l <= state.num_labels; l++) {
            const std::size_t li = static_cast<std::size_t>(l - 1);
            if (counts[li] > 0 && counts[li] < params.min_grain_elements && settled[li] == 0) {
                candidate[li] = 1;
                candidates.push_back(l);
            }
        }
        if (candidates.empty()) {
            return;
        }
        round++;
        if (round > params.max_rounds) {
            throw std::runtime_error("CleanupFragments: small-grain merging exceeded max_rounds");
        }
        report.small_grain_rounds = round;

        // Step 2: face counts, pieces keyed by label
        RefreshGhostLabels(state);
        std::vector<std::int64_t> piece(static_cast<std::size_t>(state.ne), -1);
        for (int e = 0; e < state.ne; e++) {
            if (candidate[static_cast<std::size_t>(state.label[e] - 1)] != 0) {
                piece[static_cast<std::size_t>(e)] = state.label[e];
            }
        }
        auto moving_neighbor = [&](int n) {
            return candidate[static_cast<std::size_t>(NeighborLabel(state, n) - 1)] != 0;
        };
        const auto faces = GatherFaceCounts(state, piece, moving_neighbor);

        // Step 3: decisions
        std::vector<int> target(static_cast<std::size_t>(state.num_labels) + 1, 0);
        bool progress = false;
        for (const int l : candidates) {
            const MergeDecision d = DecideMerge(faces, l);
            if (d.target > 0) {
                target[static_cast<std::size_t>(l)] = d.target;
                report.small_grains_merged += 1;
                progress = true;
            } else if (d.isolated) {
                settled[static_cast<std::size_t>(l - 1)] = 1;
                report.small_grains_kept += 1;
                progress = true;
            }
        }
        if (!progress) {
            int best = candidates[0];
            for (const int l : candidates) {
                if (counts[static_cast<std::size_t>(l - 1)] >
                    counts[static_cast<std::size_t>(best - 1)]) {
                    best = l; // strict >: the lowest label wins ties (candidates are ascending)
                }
            }
            settled[static_cast<std::size_t>(best - 1)] = 1;
            report.small_grains_kept += 1;
        }

        // Step 4: relabel
        for (int e = 0; e < state.ne; e++) {
            const int t = target[static_cast<std::size_t>(state.label[e])];
            if (t > 0) {
                state.label[e] = t;
            }
        }
    }
}

} // namespace

/*
 * FragmentCleanupReport::Summary: the counts in one readable paragraph (for the rank-0 log).
 */
std::string FragmentCleanupReport::Summary() const {
    std::ostringstream out;
    out << "fragment cleanup: " << initial_grains << " grains in, " << final_grains
        << " grains out; " << fragments_found << " extra fragment(s) found, " << fragments_split
        << " split into new grains, " << fragments_merged << " merged into neighbors";
    if (small_grains_merged > 0 || small_grains_kept > 0 || small_grain_rounds > 0) {
        out << "; " << small_grains_merged << " small grain(s) merged, " << small_grains_kept
            << " small grain(s) kept (no stable neighbor)";
    }
    out << " (" << merge_rounds << " fragment merge round(s), " << small_grain_rounds
        << " small-grain round(s), up to " << max_propagation_rounds
        << " neighbor exchange round(s) per component search)";
    return out.str();
}

/*
 * DeriveFragmentSourceId: two splitmix64 rounds over (source_id, ordinal), the first salted with
 * an arbitrary odd constant so that (source_id, k) and (source_id + 1, k - 1)-like pairs do not
 * line up; the top bit is then set, which makes the result negative as a signed integer
 * (std::memcpy reads the bits as int64 without relying on implementation-defined conversions).
 */
std::int64_t DeriveFragmentSourceId(std::int64_t source_id, std::int64_t ordinal) {
    std::uint64_t h = Mix64(BitsToUnsigned(source_id) ^ 0xD1B54A32D192ED03ULL);
    h = Mix64(h + BitsToUnsigned(ordinal));
    h |= (std::uint64_t{1} << 63);
    return BitsToSigned(h);
}

/*
 * LabelComponents: checks the input sizes on every rank (collective), exchanges the element keys
 * and labels with the neighboring ranks, and runs the two-pass search (ComponentsImpl).
 */
int LabelComponents(const MeshAdjacency& adjacency,
                    const mfem::Array<int>& element_label,
                    const std::vector<std::uint64_t>& element_key,
                    std::vector<std::uint64_t>& component,
                    std::vector<std::uint64_t>* ghost_component) {
    const int ne = adjacency.NumElements();
    const bool bad = element_label.Size() != ne || static_cast<int>(element_key.size()) != ne;
    CheckAllRanks(bad, "LabelComponents: inputs must have one entry per local element",
                  adjacency.GetComm());
    mfem::Array<int> ghost_label;
    adjacency.ExchangeLabels(element_label, ghost_label);
    std::vector<std::uint64_t> ghost;
    const int rounds = ComponentsImpl(adjacency, element_label.HostRead(),
                                      ghost_label.HostRead(), element_key, component, ghost);
    if (ghost_component != nullptr) {
        *ghost_component = std::move(ghost);
    }
    return rounds;
}

/*
 * CleanupFragments
 *
 * Steps:
 *  1. Validate the parameters (identical on every rank, so every rank throws together) and the
 *     grain ids (checked collectively).
 *  2. Working state: element keys (ElementKey), labels = input grain ids, per-label parent and
 *     source ids.
 *  3. Fragment phase (policy), then small-grain phase (min_grain_elements).
 *  4. Compact renumbering: global element count per working label (MPI_Allreduce SUM); labels
 *     with elements, in increasing order, become 1..N'; elements, parents and source ids are
 *     mapped accordingly.
 */
FragmentCleanupResult CleanupFragments(const MeshAdjacency& adjacency,
                                       const mfem::Array<int>& element_grain,
                                       int num_grains,
                                       const std::vector<std::int64_t>& grain_source_id,
                                       const FragmentCleanupParameters& params) {
    const MPI_Comm comm = adjacency.GetComm();
    const int ne = adjacency.NumElements();

    // Step 1: parameters (replicated) and grain ids (per rank, reduced)
    if (num_grains < 0) {
        throw std::invalid_argument("CleanupFragments: num_grains must be >= 0");
    }
    if (!grain_source_id.empty() && static_cast<int>(grain_source_id.size()) != num_grains) {
        throw std::invalid_argument("CleanupFragments: grain_source_id must have num_grains "
                                    "entries (or be empty)");
    }
    if (params.keep == KeepRule::SEED_ELEMENT &&
        static_cast<int>(params.grain_anchor_key.size()) != num_grains) {
        throw std::invalid_argument("CleanupFragments: KeepRule::SEED_ELEMENT needs one anchor "
                                    "key per grain");
    }
    if (params.min_grain_elements < 0) {
        throw std::invalid_argument("CleanupFragments: min_grain_elements must be >= 0");
    }
    if (params.max_rounds < 1) {
        throw std::invalid_argument("CleanupFragments: max_rounds must be >= 1");
    }
    if (params.min_fragment_elements < 0) {
        throw std::invalid_argument("CleanupFragments: min_fragment_elements must be >= 0");
    }
    if (!(params.min_fragment_fraction >= 0.0 && params.min_fragment_fraction <= 1.0)) {
        throw std::invalid_argument("CleanupFragments: min_fragment_fraction must be in [0, 1]");
    }
    bool bad_ids = element_grain.Size() != ne;
    if (!bad_ids) {
        const int* grain = element_grain.HostRead();
        for (int e = 0; e < ne; e++) {
            if (grain[e] < 1 || grain[e] > num_grains) {
                bad_ids = true;
                break;
            }
        }
    }
    CheckAllRanks(bad_ids,
                  "CleanupFragments: element_grain must hold one grain id in [1, num_grains] per "
                  "local element",
                  comm);

    // Step 2: working state
    CleanupState state;
    state.adjacency = &adjacency;
    state.comm = comm;
    state.ne = ne;
    state.offsets = adjacency.Offsets().HostRead();
    state.neighbors = adjacency.Neighbors().HostRead();
    state.key = ComputeElementKeys(adjacency.GetParMesh());
    state.label.SetSize(ne);
    {
        const int* grain = element_grain.HostRead();
        for (int e = 0; e < ne; e++) {
            state.label[e] = grain[e];
        }
    }
    state.num_labels = num_grains;
    state.num_input_grains = num_grains;
    state.parent.resize(static_cast<std::size_t>(num_grains));
    state.source.resize(static_cast<std::size_t>(num_grains));
    state.splits.assign(static_cast<std::size_t>(num_grains), 0);
    for (int g = 1; g <= num_grains; g++) {
        const std::size_t gi = static_cast<std::size_t>(g - 1);
        state.parent[gi] = g;
        state.source[gi] = grain_source_id.empty() ? static_cast<std::int64_t>(g - 1)
                                                   : grain_source_id[gi];
    }
    if (params.keep == KeepRule::SEED_ELEMENT) {
        state.anchor = &params.grain_anchor_key;
    }

    // Step 3: fragments, then small grains
    FragmentCleanupResult result;
    result.report.initial_grains = num_grains;
    if (num_grains > 0) {
        FragmentPhase(state, params, result.report);
        SmallGrainPhase(state, params, result.report);
    }

    // Step 4: compact renumbering in increasing working label
    std::vector<std::int64_t> counts;
    if (state.num_labels > 0) {
        counts = GlobalLabelCounts(state);
    }
    std::vector<int> compact(static_cast<std::size_t>(state.num_labels) + 1, 0);
    for (int l = 1; l <= state.num_labels; l++) {
        if (counts[static_cast<std::size_t>(l - 1)] > 0) {
            result.num_grains += 1;
            compact[static_cast<std::size_t>(l)] = result.num_grains;
            result.grain_parent.push_back(state.parent[static_cast<std::size_t>(l - 1)]);
            result.grain_source_id.push_back(state.source[static_cast<std::size_t>(l - 1)]);
        }
    }
    result.element_grain.SetSize(ne);
    for (int e = 0; e < ne; e++) {
        result.element_grain[e] = compact[static_cast<std::size_t>(state.label[e])];
    }
    result.report.final_grains = result.num_grains;
    return result;
}

} // namespace exaconstit::microstructure
