/*
 * Implementation of the connected-cell labeler (connected_labeler.hpp).
 *
 * Terms used in this file:
 * - Element key: 64-bit hash of an element's exact vertex coordinates (mesh/element_key.hpp), the
 *   same on every rank count; used as the element's name and as a tie-break.
 * - Component: a maximal set of face-connected elements with the same label, named by the
 *   smallest element key in it (LabelComponents, labeling/fragment_cleanup.hpp).
 * - Anchor: the element of seed s's own label that holds the seed (or, if that element went to
 *   another seed or the seed is outside the material, the s-labeled element nearest to it).
 * - Orphan: an element labeled s that lies in a component of s not holding s's anchor.
 * - Exclusion list: per element, the seeds it has been an orphan of; the relabeling search skips
 *   them (SeedIndex::FindBestExcluding, geometry/seed_index.hpp). Exclusions are permanent and all
 *   orphans of a round move at once, so the rounds are a fixed greedy order, not the strict
 *   "nearest seed whose cell reaches the element" (example in connected_labeler.hpp).
 * - CSR (compressed rows): several variable-length lists stored back to back in one array `ids`,
 *   with `offsets[q] .. offsets[q+1] - 1` the entries of list q.
 * - BVH (bounding volume hierarchy): a tree of nested bounding boxes that lets us skip elements
 *   (or seeds) that are provably too far away.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 */
#include "microstructure/labeling/connected_labeler.hpp"

#include "microstructure/exec_dispatch.hpp"
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/seeding/seed_element_locator.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief "No element / no component": larger than every real key, so MPI MIN ignores it. */
constexpr std::uint64_t NO_KEY = std::numeric_limits<std::uint64_t>::max();

/**
 * @brief Largest element range sampled at once in vote sampling, in voting points: the sampler
 * output holds 4 doubles (x, y, z, w detJ) = 32 B per point, so one range needs at most 32 MB.
 */
constexpr int VOTE_POINTS_PER_RANGE = 1 << 20;

/**
 * @brief Memory budget of the searched points of one vote chunk. Each searched point needs about
 * 100 B: its gathered coordinates and weight (32 B), its search result (BestPair, 24 B), its CSR
 * offset (4 B) and its share of the sampled range (32 B), plus 4 B per entry of its element's
 * exclusion list, which is copied once per point. That is about 160 B per point at 16 rounds, so
 * 2^20 points (160 MB) fit; chunks hold fewer points when the lists are longer.
 */
constexpr std::size_t VOTE_CHUNK_BYTES = std::size_t(160) << 20;
constexpr std::size_t VOTE_BYTES_PER_POINT = 100;

/**
 * @brief Exclusion lists of the elements that have been orphans, in CSR form, grown round by
 * round.
 *
 * Only elements with a non-empty list are stored: listed[k] (ascending local element id) owns
 * ids[offsets[k] .. offsets[k+1] - 1], and slot[e] = k (or -1 for an element that was never an
 * orphan). Memory is therefore one int per local element (slot) plus one int per list entry
 * actually created, i.e. (number of orphans) x (rounds they took part in), never NE x max_rounds.
 * Host data.
 */
struct ExclusionLists {
    std::vector<int> slot;    ///< size NE: list index of element e, or -1
    std::vector<int> listed;  ///< elements with a list, ascending
    std::vector<int> offsets; ///< size listed.size() + 1
    std::vector<int> ids;     ///< all lists back to back

    /** @brief Number of seeds on element e's list. */
    int Length(int e) const {
        const int k = slot[static_cast<std::size_t>(e)];
        if (k < 0) {
            return 0;
        }
        const auto uk = static_cast<std::size_t>(k);
        return offsets[uk + 1] - offsets[uk];
    }
    /** @brief First entry of element e's list (only valid when Length(e) > 0). */
    const int* List(int e) const {
        const int k = slot[static_cast<std::size_t>(e)];
        return ids.data() + offsets[static_cast<std::size_t>(k)];
    }

    /**
     * @brief Appends labels[e] to the list of every element in `elems` (ascending, distinct).
     *
     * Rebuilds the CSR by merging the already-listed elements with `elems` in ascending order;
     * each list is copied and, for an element of `elems`, extended by one entry. Cost is linear in
     * the stored entries, which is small next to the component search of a round.
     */
    void Append(const std::vector<int>& elems, const int* labels) {
        std::vector<int> new_listed;
        std::vector<int> new_offsets(1, 0);
        std::vector<int> new_ids;
        new_listed.reserve(listed.size() + elems.size());
        new_ids.reserve(ids.size() + elems.size());
        std::size_t a = 0; // position in listed
        std::size_t b = 0; // position in elems
        while (a < listed.size() || b < elems.size()) {
            // Next element in ascending order, and whether it is old, new, or both
            int e = 0;
            bool old_list = false;
            bool add = false;
            if (b >= elems.size() || (a < listed.size() && listed[a] < elems[b])) {
                e = listed[a++];
                old_list = true;
            } else if (a >= listed.size() || elems[b] < listed[a]) {
                e = elems[b++];
                add = true;
            } else { // the same element in both
                e = listed[a++];
                ++b;
                old_list = true;
                add = true;
            }
            if (old_list) {
                const int len = Length(e);
                const int* list = List(e);
                new_ids.insert(new_ids.end(), list, list + len);
            }
            if (add) {
                new_ids.push_back(labels[e]);
            }
            new_listed.push_back(e);
            new_offsets.push_back(static_cast<int>(new_ids.size()));
        }
        listed.swap(new_listed);
        offsets.swap(new_offsets);
        ids.swap(new_ids);
        for (std::size_t k = 0; k < listed.size(); ++k) {
            slot[static_cast<std::size_t>(listed[k])] = static_cast<int>(k);
        }
    }
};

/**
 * @brief Writes the exclusion lists of the elements `elems` in CSR form, each list repeated
 * `repeat` times in a row (once per voting point of the element; 1 for centroid sampling).
 *
 * Query k = j * repeat + q (point q of elems[j]) gets ids[offsets[k] .. offsets[k+1] - 1] = the
 * list of elems[j]. Host loop; the arrays are moved to the device by the search.
 */
void BuildExclusionCsr(const ExclusionLists& lists, const int* elems, int num_elems, int repeat,
                       mfem::Array<int>& offsets, mfem::Array<int>& ids) {
    // Total size first, so both arrays are allocated once
    long long total = 0;
    for (int j = 0; j < num_elems; ++j) {
        total += static_cast<long long>(lists.Length(elems[j])) * repeat;
    }
    if (total > std::numeric_limits<int>::max()) {
        throw std::runtime_error("ConstrainToConnectedCells: exclusion lists of one search chunk "
                                 "exceed 2^31 entries");
    }
    offsets.SetSize(num_elems * repeat + 1);
    ids.SetSize(static_cast<int>(total));
    int* off = offsets.HostWrite();
    int* out = ids.HostWrite();

    // Copy each element's list once per point
    int next = 0;
    off[0] = 0;
    for (int j = 0; j < num_elems; ++j) {
        const int len = lists.Length(elems[j]);
        const int* list = (len > 0) ? lists.List(elems[j]) : nullptr;
        for (int q = 0; q < repeat; ++q) {
            for (int k = 0; k < len; ++k) {
                out[next++] = list[k];
            }
            off[j * repeat + q + 1] = next;
        }
    }
}

/**
 * @brief Copies the voting points of selected elements out of a sampled element range.
 *
 * `points` holds the points of local elements [points.elem_begin, points.elem_end) in the
 * QuadraturePointSamples layout (entry (e - elem_begin) * nq + q). For selected element j, whose
 * position inside the range is rel[j], its nq points are copied to entries j * nq .. j * nq +
 * nq - 1 of x, y, z, w. One mfem::forall iteration per copied point, so it runs on the device.
 *
 * A named function (not a lambda inside the dispatch lambdas) because CUDA does not allow a device
 * lambda to be defined inside a generic lambda.
 */
void GatherSelectedPoints(const QuadraturePointSamples& points, const mfem::Array<int>& rel,
                          int nq, mfem::Vector& x, mfem::Vector& y, mfem::Vector& z,
                          mfem::Vector& w) {
    const int num_sel = rel.Size();
    const int n = num_sel * nq;
    x.SetSize(n);
    y.SetSize(n);
    z.SetSize(n);
    w.SetSize(n);
    const double* px = points.x.Read();
    const double* py = points.y.Read();
    const double* pz = points.z.Read();
    const double* pw = points.weight.Read();
    const int* r = rel.Read();
    double* ox = x.Write();
    double* oy = y.Write();
    double* oz = z.Write();
    double* ow = w.Write();
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int k) {
        // k = j * nq + q: point q of selected element j, found at rel[j] * nq + q in the range
        const int j = k / nq;
        const int q = k - j * nq;
        const int src = r[j] * nq + q;
        ox[k] = px[src];
        oy[k] = py[src];
        oz[k] = pz[src];
        ow[k] = pw[src];
    });
}

/**
 * @brief Volume-weighted vote of each selected element's points (one mfem::forall iteration per
 * element).
 *
 * The points of element j are entries j * nq .. j * nq + nq - 1 of `best` and `weight`. The label
 * of a point is best[.].id1 (the best seed not on the element's exclusion list); a point with
 * id1 = -1 does not vote. The rule is that of the quadrature vote (labeling/quadrature_vote.cpp,
 * VoteChunk), with every seed its own owner: for each label at its first point, its total weight is
 * summed over the element's points in increasing order (so the sum is the same on every rank and
 * device); the largest total wins, an exactly equal total goes to the lower seed index. No scratch
 * memory, so it runs on a GPU. winner[j] = -1 when no point voted.
 */
void VoteSelected(int num_sel, int nq, const mfem::Array<BestPair>& best,
                  const mfem::Vector& weight, mfem::Array<int>& winner) {
    winner.SetSize(num_sel);
    if (num_sel == 0) {
        return;
    }
    const BestPair* point_best = best.Read();
    const double* point_weight = weight.Read();
    int* out = winner.Write();
    mfem::forall(num_sel, [=] MFEM_HOST_DEVICE(int j) {
        const int base = j * nq;
        int win = -1;
        double win_weight = 0.0;
        for (int q = 0; q < nq; ++q) {
            const int label = point_best[base + q].id1;
            if (label < 0) {
                continue;
            }
            // Count each label once, at its first point
            bool seen_before = false;
            for (int p = 0; p < q; ++p) {
                if (point_best[base + p].id1 == label) {
                    seen_before = true;
                    break;
                }
            }
            if (seen_before) {
                continue;
            }
            // Total weight of this label, in increasing point order
            double total = 0.0;
            for (int p = q; p < nq; ++p) {
                if (point_best[base + p].id1 == label) {
                    total += point_weight[base + p];
                }
            }
            if (win < 0 || total > win_weight || (total == win_weight && label < win)) {
                win = label;
                win_weight = total;
            }
        }
        out[j] = win;
    });
}

/**
 * @brief Everything the rounds need besides the seed search; filled by ConstrainToConnectedCells.
 * Host data unless noted.
 */
struct RoundContext {
    const MeshAdjacency* adjacency = nullptr;
    MPI_Comm comm = MPI_COMM_NULL;
    int num_elements = 0;
    int num_seeds = 0;
    ConnectedLabelParameters params;
    std::vector<std::uint64_t> keys;      ///< element keys, size NE
    std::vector<int> anchor_local;        ///< size N: local element holding seed s's anchor, or -1
    std::vector<std::uint64_t> anchor_key;///< size N: anchor key of every seed (replicated)
    std::vector<char> exempt;             ///< size NE: 1 for elements of unanchored bodies
    const ElementSamples* centroids = nullptr; ///< element centroids (any memory space)
    QuadraturePointSampler* sampler = nullptr; ///< vote sampling only, null on empty ranks
};

/**
 * @brief Relabels the elements `movable` (after their current label was appended to their lists)
 * with the best non-excluded seed: at the centroid, or by the vote of their voting points.
 *
 * Centroid: one FindBestExcluding over the centroids of the movable elements. Vote: the movable
 * elements are walked in increasing local id; a chunk is a contiguous element range starting at
 * the next movable element and holding at most VOTE_POINTS_PER_RANGE points; within it, movable
 * elements are taken until the searched points would exceed VOTE_CHUNK_BYTES (at least one
 * element). The range is sampled (QuadraturePointSampler::Sample), the points of the selected
 * elements are gathered, every point is searched with its element's list and the elements vote.
 *
 * Not collective. Returns new_label[j] for movable[j] (-1 = every seed excluded).
 */
template <typename Index>
std::vector<int> RelabelMovable(const Index& index, const RoundContext& ctx,
                                const ExclusionLists& lists, const std::vector<int>& movable) {
    const int m = static_cast<int>(movable.size());
    std::vector<int> new_label(static_cast<std::size_t>(m), -1);
    if (m == 0) {
        return new_label;
    }
    mfem::Array<int> offsets;
    mfem::Array<int> ids;
    mfem::Array<BestPair> best;

    if (ctx.params.sampling != SamplingMode::QUADRATURE_VOTE) {
        // Centroid sampling: query point j = centroid of movable[j]
        mfem::Vector qx(m), qy(m), qz(m);
        const double* cx = ctx.centroids->cx.HostRead();
        const double* cy = ctx.centroids->cy.HostRead();
        const double* cz = ctx.centroids->cz.HostRead();
        double* hx = qx.HostWrite();
        double* hy = qy.HostWrite();
        double* hz = qz.HostWrite();
        for (int j = 0; j < m; ++j) {
            const int e = movable[static_cast<std::size_t>(j)];
            hx[j] = cx[e];
            hy[j] = cy[e];
            hz[j] = cz[e];
        }
        BuildExclusionCsr(lists, movable.data(), m, 1, offsets, ids);
        index.FindBestExcluding(qx, qy, qz, offsets, ids, best);
        const BestPair* hb = best.HostRead();
        for (int j = 0; j < m; ++j) {
            new_label[static_cast<std::size_t>(j)] = hb[j].id1;
        }
        return new_label;
    }

    // Vote sampling: chunks of contiguous element ranges that start at a movable element
    const int nq = ctx.sampler->NumPointsPerElement();
    const int range_elems = std::max(1, VOTE_POINTS_PER_RANGE / nq);
    QuadraturePointSamples points;
    mfem::Vector qx, qy, qz, qw;
    mfem::Array<int> winner;
    int pos = 0;
    while (pos < m) {
        // Element range [begin, end) and the movable elements inside it: movable[pos .. stop)
        const int begin = movable[static_cast<std::size_t>(pos)];
        const int end = std::min(ctx.num_elements, begin + range_elems);
        int stop = pos;
        std::size_t bytes = 0;
        while (stop < m && movable[static_cast<std::size_t>(stop)] < end) {
            // Memory of this element's searched points, including its list copy per point
            const int e = movable[static_cast<std::size_t>(stop)];
            const auto list_len = static_cast<std::size_t>(lists.Length(e));
            const std::size_t cost =
                static_cast<std::size_t>(nq) * (VOTE_BYTES_PER_POINT + 4 * list_len);
            if (stop > pos && bytes + cost > VOTE_CHUNK_BYTES) {
                break;
            }
            bytes += cost;
            ++stop;
        }
        const int num_sel = stop - pos;

        // Positions of the selected elements inside the range
        mfem::Array<int> rel(num_sel);
        for (int j = 0; j < num_sel; ++j) {
            rel[j] = movable[static_cast<std::size_t>(pos + j)] - begin;
        }

        // Points of the range, then only those of the selected elements
        ctx.sampler->Sample(begin, end, points);
        GatherSelectedPoints(points, rel, nq, qx, qy, qz, qw);

        // Search every point with its element's list, then vote per element
        BuildExclusionCsr(lists, movable.data() + pos, num_sel, nq, offsets, ids);
        index.FindBestExcluding(qx, qy, qz, offsets, ids, best);
        VoteSelected(num_sel, nq, best, qw, winner);
        const int* hw = winner.HostRead();
        for (int j = 0; j < num_sel; ++j) {
            new_label[static_cast<std::size_t>(pos + j)] = hw[j];
        }
        pos = stop;
    }
    return new_label;
}

/**
 * @brief The rounds of ConstrainToConnectedCells for one metric/execution-space pair.
 *
 * Each round:
 *  1. Components of the current labels (LabelComponents; collective).
 *  2. Per seed s, the component holding s's anchor if the anchor carries label s, else NO_KEY.
 *     Only the anchor's owner rank fills the entry; MPI_Allreduce(MIN) replicates the table.
 *  3. Orphans: non-exempt elements whose component is not their seed's entry. Movable orphans are
 *     those that still have a candidate seed (not "stuck"). Global counts by MPI_Allreduce(SUM).
 *  4. Stop when no orphan is movable or max_rounds rounds were done; otherwise every movable
 *     orphan appends its label to its list and is relabeled (RelabelMovable). An orphan for which
 *     every seed is excluded keeps its label and becomes stuck.
 * See the header for the termination and rank-independence arguments.
 */
template <typename Index>
ConnectedLabelReport RunRounds(const Index& index, const RoundContext& ctx, int* labels) {
    ConnectedLabelReport report;
    const int ne = ctx.num_elements;
    const int n = ctx.num_seeds;

    ExclusionLists lists;
    lists.slot.assign(static_cast<std::size_t>(ne), -1);
    lists.offsets.assign(1, 0);
    std::vector<char> stuck(static_cast<std::size_t>(ne), 0);

    // Label array view for LabelComponents (host memory owned by the caller's array)
    mfem::Array<int> label_view(labels, ne);
    std::vector<std::uint64_t> component;
    std::vector<std::uint64_t> seed_component(static_cast<std::size_t>(n));
    std::vector<int> movable;

    for (;;) {
        // Step 1: components of the current labels (ids = smallest element key)
        LabelComponents(*ctx.adjacency, label_view, ctx.keys, component);

        // Step 2: component of each seed's anchor, if the anchor still carries that seed
        std::fill(seed_component.begin(), seed_component.end(), NO_KEY);
        for (int s = 0; s < n; ++s) {
            const int a = ctx.anchor_local[static_cast<std::size_t>(s)];
            if (a >= 0 && labels[a] == s) { // always true for an anchor (see the header)
                seed_component[static_cast<std::size_t>(s)] =
                    component[static_cast<std::size_t>(a)];
            }
        }
        if (n > 0) {
            MPI_Allreduce(MPI_IN_PLACE, seed_component.data(), n, MPI_UINT64_T, MPI_MIN, ctx.comm);
        }

        // Step 3: orphans (local) and the global counts
        movable.clear();
        std::int64_t counts[2] = {0, 0}; // orphans, movable orphans
        for (int e = 0; e < ne; ++e) {
            if (ctx.exempt[static_cast<std::size_t>(e)] != 0) {
                continue;
            }
            const auto s = static_cast<std::size_t>(labels[e]);
            if (component[static_cast<std::size_t>(e)] == seed_component[s]) {
                continue; // connected to its seed's anchor: settled
            }
            ++counts[0];
            if (stuck[static_cast<std::size_t>(e)] == 0) {
                movable.push_back(e); // ascending local id
                ++counts[1];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, ctx.comm);

        // Step 4: done, or one more round
        if (counts[1] == 0 || report.rounds >= ctx.params.max_rounds) {
            report.unresolved_elements = counts[0];
            // A seed without an anchor (no element carried its label) has no cell. An anchor,
            // once chosen, carries its seed's label in every round, so no other seed is counted.
            for (int s = 0; s < n; ++s) {
                if (seed_component[static_cast<std::size_t>(s)] == NO_KEY) {
                    ++report.seeds_without_cell;
                }
            }
            break;
        }
        ++report.rounds;

        // Step 4a: every movable orphan excludes its current seed (never already listed)
        lists.Append(movable, labels);

        // Step 4b: best seed not on the list (element-local search)
        const std::vector<int> new_label = RelabelMovable(index, ctx, lists, movable);
        for (std::size_t j = 0; j < movable.size(); ++j) {
            const int e = movable[j];
            if (new_label[j] >= 0) {
                labels[e] = new_label[j];
            } else {
                stuck[static_cast<std::size_t>(e)] = 1; // no candidate left: keep the label
            }
        }
    }
    return report;
}

} // namespace

/*
 * SeedAnchorElementKeys
 *
 * Steps (both collective, seeding/seed_element_locator.hpp):
 *  1. Size checks, reduced over all ranks so every rank throws together.
 *  2. ContainingElementKeys: smallest key of an element containing each seed (point-in-element
 *     test behind an Axom BVH over the element boxes), MIN over ranks.
 *  3. For the seeds no element contains (the same list on every rank): NearestElementCentroidKeys,
 *     the global lexicographic minimum of (squared distance to the element centroid, key).
 */
std::vector<std::uint64_t> SeedAnchorElementKeys(mfem::ParMesh& pmesh,
                                                 MPI_Comm comm,
                                                 const SeedSet& seeds,
                                                 const std::vector<std::uint64_t>& element_key,
                                                 const mfem::Vector& centroid_x,
                                                 const mfem::Vector& centroid_y,
                                                 const mfem::Vector& centroid_z) {
    // Step 1: checks
    const int ne = pmesh.GetNE();
    const bool bad_size = static_cast<int>(element_key.size()) != ne || centroid_x.Size() != ne ||
                          centroid_y.Size() != ne || centroid_z.Size() != ne;
    ThrowIfAnyRank(comm, bad_size,
                   "SeedAnchorElementKeys: element_key and the centroids must have one entry per "
                   "local element");
    const int n = static_cast<int>(seeds.Size());

    // Step 2: containing element
    std::vector<std::uint64_t> anchor = ContainingElementKeys(pmesh, comm, seeds, &element_key);

    // Step 3: seeds outside the material go to the nearest element centroid
    std::vector<int> outside;
    for (int s = 0; s < n; ++s) {
        if (anchor[static_cast<std::size_t>(s)] == NO_KEY) {
            outside.push_back(s);
        }
    }
    if (outside.empty()) {
        return anchor; // identical on every rank, so every rank skips the collective below
    }
    ElementSamples centroids;
    centroids.cx = centroid_x;
    centroids.cy = centroid_y;
    centroids.cz = centroid_z;
    const NearestKeys nearest =
        NearestElementCentroidKeys(pmesh, comm, seeds, outside, &centroids, &element_key);
    for (std::size_t k = 0; k < outside.size(); ++k) {
        anchor[static_cast<std::size_t>(outside[k])] = nearest.key[k];
    }
    return anchor;
}

/*
 * ConnectedAnchorKeys
 *
 * Steps:
 *  1. Size and label checks (reduced over the ranks).
 *  2. ContainingElementKeys (collective). On the rank that owns seed s's containing element, that
 *     element is s's anchor if it carries label s: every local element e with
 *     key(e) == containing[label(e)] is such an anchor (one pass over the local elements, no
 *     lookup table). MPI_Allreduce(MIN) shares the result.
 *  3. For the seeds still without an anchor (the same set on every rank): one pass over the local
 *     elements keeps, per seed s, the s-labeled element with the smallest (squared distance from
 *     its centroid to s, key); then the global lexicographic minimum: MPI_Allreduce(MIN) on the
 *     distances, then MIN on the keys of the ranks that hold exactly that distance. The cost is
 *     O(NE) per rank, because every element is compared only with its own seed.
 */
std::vector<std::uint64_t> ConnectedAnchorKeys(mfem::ParMesh& pmesh,
                                               MPI_Comm comm,
                                               const SeedSet& seeds,
                                               const mfem::Array<int>& element_seed,
                                               const std::vector<std::uint64_t>& element_key,
                                               const mfem::Vector& centroid_x,
                                               const mfem::Vector& centroid_y,
                                               const mfem::Vector& centroid_z) {
    // Step 1: checks
    const int ne = pmesh.GetNE();
    const int n = static_cast<int>(seeds.Size());
    bool bad = static_cast<int>(element_key.size()) != ne || element_seed.Size() != ne ||
               centroid_x.Size() != ne || centroid_y.Size() != ne || centroid_z.Size() != ne;
    const int* label = (!bad && ne > 0) ? element_seed.HostRead() : nullptr;
    for (int e = 0; !bad && e < ne; ++e) {
        bad = label[e] < 0 || label[e] >= n;
    }
    ThrowIfAnyRank(comm, bad,
                   "ConnectedAnchorKeys: element_seed, element_key and the centroids must have one "
                   "entry per local element, with labels in [0, N)");

    // Step 2: the containing element, if it carries the seed's own label
    const std::vector<std::uint64_t> containing =
        ContainingElementKeys(pmesh, comm, seeds, &element_key);
    std::vector<std::uint64_t> anchor(static_cast<std::size_t>(n), NO_KEY);
    for (int e = 0; e < ne; ++e) {
        const auto s = static_cast<std::size_t>(label[e]);
        if (element_key[static_cast<std::size_t>(e)] == containing[s]) {
            anchor[s] = containing[s];
        }
    }
    if (n > 0) {
        MPI_Allreduce(MPI_IN_PLACE, anchor.data(), n, MPI_UINT64_T, MPI_MIN, comm);
    }

    // Step 3: otherwise the s-labeled element whose centroid is nearest to s
    bool any_missing = false;
    for (int s = 0; s < n; ++s) {
        any_missing = any_missing || anchor[static_cast<std::size_t>(s)] == NO_KEY;
    }
    if (!any_missing) {
        return anchor; // identical on every rank
    }
    std::vector<double> best_d2(static_cast<std::size_t>(n), DBL_MAX);
    std::vector<std::uint64_t> best_key(static_cast<std::size_t>(n), NO_KEY);
    const double* cx = (ne > 0) ? centroid_x.HostRead() : nullptr;
    const double* cy = (ne > 0) ? centroid_y.HostRead() : nullptr;
    const double* cz = (ne > 0) ? centroid_z.HostRead() : nullptr;
    for (int e = 0; e < ne; ++e) {
        const auto s = static_cast<std::size_t>(label[e]);
        if (anchor[s] != NO_KEY) {
            continue;
        }
        const double dx = cx[e] - seeds.x[s];
        const double dy = cy[e] - seeds.y[s];
        const double dz = cz[e] - seeds.z[s];
        const double d2 = dx * dx + dy * dy + dz * dz;
        const std::uint64_t key = element_key[static_cast<std::size_t>(e)];
        if (d2 < best_d2[s] || (d2 == best_d2[s] && key < best_key[s])) {
            best_d2[s] = d2;
            best_key[s] = key;
        }
    }
    std::vector<double> global_d2(best_d2);
    MPI_Allreduce(MPI_IN_PLACE, global_d2.data(), n, MPI_DOUBLE, MPI_MIN, comm);
    for (int s = 0; s < n; ++s) {
        const auto us = static_cast<std::size_t>(s);
        if (best_d2[us] != global_d2[us]) {
            best_key[us] = NO_KEY; // another rank holds a nearer element of this seed
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, best_key.data(), n, MPI_UINT64_T, MPI_MIN, comm);
    for (int s = 0; s < n; ++s) {
        const auto us = static_cast<std::size_t>(s);
        if (anchor[us] == NO_KEY) {
            anchor[us] = best_key[us]; // stays NO_KEY when no element carries label s
        }
    }
    return anchor;
}

/*
 * ConstrainToConnectedCells
 *
 * Steps:
 *  1. Parameter checks (identical on every rank) and input checks (reduced over the ranks, so all
 *     ranks throw together).
 *  2. Element keys and volume centroids (ComputeElementSamples), and the anchor key of every seed
 *     from the input labels (ConnectedAnchorKeys); each rank finds which anchors are its own
 *     local elements (sorted (key, element) pairs and a binary search per seed).
 *  3. Exempt bodies: components of the whole mesh (LabelComponents with every label 0); the body
 *     of each seed's anchor is reduced with MPI_Allreduce(MIN); elements of bodies that hold no
 *     anchor are exempt.
 *  4. The seed search for the metric and execution space (DispatchMetric, DispatchAxomExec,
 *     SeedIndex::Build with the global tensor of the tessellation, or per-seed tensors when the
 *     seeds carry them), the vote sampler when needed, and the rounds (RunRounds).
 *  5. Relabeled count (final vs input label), reduced with MPI_Allreduce(SUM). The anchor keys
 *     of step 2 are returned in the report (ConnectedLabelReport::anchor_key).
 */
ConnectedLabelReport ConstrainToConnectedCells(mfem::ParMesh& pmesh,
                                               MPI_Comm comm,
                                               const MeshAdjacency& adjacency,
                                               const SeedSet& seeds,
                                               const TessellationOptions& tessellation,
                                               mfem::Array<int>& element_seed,
                                               const ConnectedLabelParameters& params) {
    // Step 1a: parameters (the same on every rank)
    if (seeds.Size() == 0) {
        throw std::invalid_argument("ConstrainToConnectedCells: the seed set is empty");
    }
    if (seeds.Size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("ConstrainToConnectedCells: more than 2^31 - 1 seeds");
    }
    if (params.max_rounds < 0 || params.max_rounds > CONNECTED_MAX_ROUNDS_LIMIT) {
        throw std::invalid_argument("ConstrainToConnectedCells: max_rounds must be in 0.." +
                                    std::to_string(CONNECTED_MAX_ROUNDS_LIMIT));
    }
    const bool vote = (params.sampling == SamplingMode::QUADRATURE_VOTE);
    if (vote && (params.vote_levels < 0 || params.vote_levels > 3)) {
        throw std::invalid_argument("ConstrainToConnectedCells: vote_levels must be in 0..3");
    }
    if (adjacency.HasPeriodicPairing()) {
        throw std::invalid_argument("ConstrainToConnectedCells: periodic face pairing is not "
                                    "supported by the connected distance mode");
    }

    // Step 1b: inputs (reduced, so every rank throws together)
    const int ne = pmesh.GetNE();
    const int n = static_cast<int>(seeds.Size());
    ThrowIfAnyRank(comm, &adjacency.GetParMesh() != &pmesh || adjacency.NumElements() != ne,
                   "ConstrainToConnectedCells: the adjacency must describe the given mesh");
    bool bad_labels = element_seed.Size() != ne;
    if (!bad_labels && ne > 0) {
        const int* in = element_seed.HostRead();
        for (int e = 0; e < ne; ++e) {
            bad_labels = bad_labels || in[e] < 0 || in[e] >= n;
        }
    }
    ThrowIfAnyRank(comm, bad_labels,
                   "ConstrainToConnectedCells: element_seed must hold one seed index in [0, N) "
                   "per local element");

    // Step 2: keys, centroids, anchors
    RoundContext ctx;
    ctx.adjacency = &adjacency;
    ctx.comm = comm;
    ctx.num_elements = ne;
    ctx.num_seeds = n;
    ctx.params = params;
    ctx.keys = ComputeElementKeys(pmesh);
    ElementSamples centroids;
    ComputeElementSamples(pmesh, -1, centroids);
    ctx.centroids = &centroids;
    ctx.anchor_key = ConnectedAnchorKeys(pmesh, comm, seeds, element_seed, ctx.keys, centroids.cx,
                                         centroids.cy, centroids.cz);

    std::vector<std::pair<std::uint64_t, int>> key_to_element(static_cast<std::size_t>(ne));
    for (int e = 0; e < ne; ++e) {
        key_to_element[static_cast<std::size_t>(e)] = {ctx.keys[static_cast<std::size_t>(e)], e};
    }
    std::sort(key_to_element.begin(), key_to_element.end());
    ctx.anchor_local.assign(static_cast<std::size_t>(n), -1);
    for (int s = 0; s < n; ++s) {
        const std::uint64_t key = ctx.anchor_key[static_cast<std::size_t>(s)];
        const auto it = std::lower_bound(key_to_element.begin(), key_to_element.end(),
                                         std::make_pair(key, std::numeric_limits<int>::min()));
        if (key != NO_KEY && it != key_to_element.end() && it->first == key) {
            ctx.anchor_local[static_cast<std::size_t>(s)] = it->second;
        }
    }

    // Step 3: bodies (material components) and which of them hold an anchor
    mfem::Array<int> zeros(ne);
    zeros = 0;
    std::vector<std::uint64_t> body;
    LabelComponents(adjacency, zeros, ctx.keys, body);
    std::vector<std::uint64_t> anchored_body(static_cast<std::size_t>(n), NO_KEY);
    for (int s = 0; s < n; ++s) {
        const int a = ctx.anchor_local[static_cast<std::size_t>(s)];
        if (a >= 0) {
            anchored_body[static_cast<std::size_t>(s)] = body[static_cast<std::size_t>(a)];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, anchored_body.data(), n, MPI_UINT64_T, MPI_MIN, comm);
    std::sort(anchored_body.begin(), anchored_body.end());
    ctx.exempt.assign(static_cast<std::size_t>(ne), 0);
    std::int64_t num_exempt = 0;
    for (int e = 0; e < ne; ++e) {
        const std::uint64_t b = body[static_cast<std::size_t>(e)];
        if (!std::binary_search(anchored_body.begin(), anchored_body.end(), b)) {
            ctx.exempt[static_cast<std::size_t>(e)] = 1;
            ++num_exempt;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &num_exempt, 1, MPI_INT64_T, MPI_SUM, comm);

    // Step 4: seed search, vote sampler, rounds
    std::unique_ptr<QuadraturePointSampler> sampler;
    if (vote && ne > 0) {
        sampler = std::make_unique<QuadraturePointSampler>(
            pmesh, VoteRule(pmesh.GetElementGeometry(0), params.vote_levels));
        ctx.sampler = sampler.get();
    }
    MetricSetup setup;
    setup.global_tensor = EffectiveGlobalTensor(tessellation);
    setup.per_seed_shape = !seeds.shape.empty();

    const mfem::Array<int> input_labels(element_seed); // copy, for the relabeled count
    int* labels = (ne > 0) ? element_seed.HostReadWrite() : nullptr;
    ConnectedLabelReport report;
    DispatchMetric(tessellation.metric, [&](auto metric_tag) {
        using Metric = decltype(metric_tag);
        DispatchAxomExec([&](auto exec_tag) {
            using Exec = decltype(exec_tag);
            SeedIndex<Exec, Metric> index;
            index.Build(seeds, setup, tessellation.num_bins, AxomAllocatorIdForCurrentDevice());
            report = RunRounds(index, ctx, labels);
        });
    });
    report.unanchored_component_elements = num_exempt;
    report.anchor_key = ctx.anchor_key; // the anchors the rounds used, for the caller

    // Step 5: elements whose label changed
    std::int64_t changed = 0;
    if (ne > 0) {
        const int* in = input_labels.HostRead();
        const int* out = element_seed.HostRead();
        for (int e = 0; e < ne; ++e) {
            changed += (in[e] != out[e]) ? 1 : 0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT64_T, MPI_SUM, comm);
    report.relabeled_elements = changed;
    return report;
}

} // namespace exaconstit::microstructure
