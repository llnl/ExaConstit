/*
 * Discrete vs analytic grain adjacency (adjacency_check.hpp).
 */
#include "microstructure/conformal/adjacency_check.hpp"

#include <algorithm>
#include <mpi.h>
#include <utility>

namespace exaconstit::microstructure {

namespace {

constexpr std::int64_t DISCRETE = 1;
constexpr std::int64_t ANALYTIC = 2;
constexpr std::int64_t SEED_PAIR = 0;  ///< a pair of seeds (compared)
constexpr std::int64_t GRAIN_PAIR = 1; ///< a pair of grains with a grain without seed

/** @brief One pair with its kind and flags: info = 4 * kind + flags. */
struct PairRecord {
    std::int64_t key; ///< first << 32 | second
    std::int64_t info;
};

std::int64_t PairKey(int a, int b) {
    if (a > b) {
        std::swap(a, b);
    }
    return (static_cast<std::int64_t>(a) << 32) | static_cast<std::int64_t>(b);
}

/** @brief Owner rank of a pair (splitmix64 of the key and kind). */
int OwnerOf(const PairRecord& r, int size) {
    std::uint64_t z = static_cast<std::uint64_t>(r.key) ^ static_cast<std::uint64_t>(r.info >> 2);
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<int>(z % static_cast<std::uint64_t>(size));
}

/** @brief Sorts and merges records with the same key and kind (flags or-ed). */
void MergeRecords(std::vector<PairRecord>& records) {
    std::sort(records.begin(), records.end(), [](const PairRecord& x, const PairRecord& y) {
        return (x.info >> 2) != (y.info >> 2) ? (x.info >> 2) < (y.info >> 2) : x.key < y.key;
    });
    std::size_t out = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (out > 0 && records[out - 1].key == records[i].key &&
            (records[out - 1].info >> 2) == (records[i].info >> 2)) {
            records[out - 1].info |= records[i].info;
        } else {
            records[out++] = records[i];
        }
    }
    records.resize(out);
}

/** @brief The smallest ADJACENCY_EXAMPLES keys over all ranks (each rank's keys sorted). */
std::vector<std::array<int, 2>> GatherExamples(const std::vector<std::int64_t>& local,
                                               MPI_Comm comm) {
    const int n = std::min<int>(static_cast<int>(local.size()), ADJACENCY_EXAMPLES);
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(static_cast<std::size_t>(size));
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(size) + 1, 0);
    for (int r = 0; r < size; ++r) {
        displs[static_cast<std::size_t>(r) + 1] = displs[static_cast<std::size_t>(r)] +
                                                  counts[static_cast<std::size_t>(r)];
    }
    std::vector<std::int64_t> all(static_cast<std::size_t>(std::max(1, displs.back())));
    MPI_Allgatherv(
        local.data(), n, MPI_INT64_T, all.data(), counts.data(), displs.data(), MPI_INT64_T, comm);
    all.resize(static_cast<std::size_t>(displs.back()));
    std::sort(all.begin(), all.end());
    std::vector<std::array<int, 2>> out;
    for (std::size_t i = 0; i < all.size() && i < static_cast<std::size_t>(ADJACENCY_EXAMPLES);
         ++i) {
        out.push_back({static_cast<int>(all[i] >> 32), static_cast<int>(all[i] & 0xffffffffLL)});
    }
    return out;
}

} // namespace

/*
 * CompareAdjacency
 *  1. Per owned node: every label pair (discrete); seed pairs are queried at the node (analytic
 *     when FIT), grain pairs with a grain without seed are kept unchecked.
 *  2. Per owned node: the nearby seeds; pairs (best, other) queried at the node with the first
 *     grain of each seed (seeds without grain cannot be queried and are skipped).
 *  3. Records to their owner ranks, merged; counts summed; examples gathered.
 */
AdjacencyComparison CompareAdjacency(const NodeTargets& targets,
                                     const std::vector<int>& grain_seed) {
    const NodeLabels& labels = targets.Labels();
    const VertexField& field = labels.Field();
    const MPI_Comm comm = field.Comm();
    const int nv = labels.NumNodes();
    const int* owned = field.OwnedVertexMarker().HostRead();
    const int* off = labels.LabelOffsets().HostRead();
    const int* items = labels.Labels().HostRead();
    const bool analytic = targets.Options().analytic;

    // First grain of each seed
    int num_seeds = 0;
    for (const int s : grain_seed) {
        num_seeds = std::max(num_seeds, s + 1);
    }
    std::vector<int> seed_grain(static_cast<std::size_t>(num_seeds), -1);
    for (std::size_t g = grain_seed.size(); g-- > 0;) {
        if (grain_seed[g] >= 0) {
            seed_grain[static_cast<std::size_t>(grain_seed[g])] = static_cast<int>(g) + 1;
        }
    }
    const auto grain_of_seed = [&](int s) {
        return (s >= 0 && s < num_seeds) ? seed_grain[static_cast<std::size_t>(s)] : -1;
    };
    const auto seed_of = [&](int grain) {
        return grain_seed[static_cast<std::size_t>(grain - 1)];
    };
    const auto pair_query = [](int v, int ga, int gb) {
        NodeLabelQuery q;
        q.node = v;
        q.num_labels = 2;
        q.labels[0] = std::min(ga, gb);
        q.labels[1] = std::max(ga, gb);
        return q;
    };

    // Step 1
    std::vector<PairRecord> records;
    std::vector<NodeLabelQuery> queries;
    std::vector<std::int64_t> query_key;
    std::vector<int> owned_nodes;
    for (int v = 0; v < nv; ++v) {
        if (owned[v] == 0) {
            continue;
        }
        owned_nodes.push_back(v);
        for (int i = off[v]; i < off[v + 1]; ++i) {
            for (int j = i + 1; j < off[v + 1]; ++j) {
                const int ga = items[i];
                const int gb = items[j];
                const int sa = seed_of(ga);
                const int sb = seed_of(gb);
                if (!analytic || sa < 0 || sb < 0) {
                    records.push_back({PairKey(ga, gb), 4 * GRAIN_PAIR + DISCRETE});
                } else if (sa != sb) {
                    records.push_back({PairKey(sa, sb), 4 * SEED_PAIR + DISCRETE});
                    queries.push_back(pair_query(v, ga, gb));
                    query_key.push_back(PairKey(sa, sb));
                }
            }
        }
    }

    // Step 2
    if (analytic) {
        std::vector<int> nearby_off;
        std::vector<int> nearby;
        targets.NearbySeeds(owned_nodes,
                            ADJACENCY_NEARBY_SEEDS,
                            2.0 * targets.Options().max_move,
                            nearby_off,
                            nearby);
        for (std::size_t i = 0; i < owned_nodes.size(); ++i) {
            const int b = nearby_off[i];
            if (nearby_off[i + 1] <= b) {
                continue;
            }
            const int s1 = nearby[static_cast<std::size_t>(b)];
            const int ga = grain_of_seed(s1);
            for (int k = b + 1; k < nearby_off[i + 1] && ga > 0; ++k) {
                const int sk = nearby[static_cast<std::size_t>(k)];
                const int gk = grain_of_seed(sk);
                if (gk > 0) {
                    queries.push_back(pair_query(owned_nodes[i], ga, gk));
                    query_key.push_back(PairKey(s1, sk));
                }
            }
        }
        const std::vector<NodeTargetResult> results = targets.EvaluateAnalytic(queries);
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i].status == TargetStatus::FIT) {
                records.push_back({query_key[i], 4 * SEED_PAIR + ANALYTIC});
            }
        }
    }
    MergeRecords(records);

    // Step 3
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<std::vector<PairRecord>> by_owner(static_cast<std::size_t>(size));
    for (const PairRecord& r : records) {
        by_owner[static_cast<std::size_t>(OwnerOf(r, size))].push_back(r);
    }
    std::vector<int> send_counts(static_cast<std::size_t>(size));
    std::vector<int> send_displs(static_cast<std::size_t>(size) + 1, 0);
    std::vector<std::int64_t> send;
    for (int r = 0; r < size; ++r) {
        for (const PairRecord& rec : by_owner[static_cast<std::size_t>(r)]) {
            send.push_back(rec.key);
            send.push_back(rec.info);
        }
        send_counts[static_cast<std::size_t>(r)] =
            2 * static_cast<int>(by_owner[static_cast<std::size_t>(r)].size());
        send_displs[static_cast<std::size_t>(r) + 1] = static_cast<int>(send.size());
    }
    std::vector<int> recv_counts(static_cast<std::size_t>(size));
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    std::vector<int> recv_displs(static_cast<std::size_t>(size) + 1, 0);
    for (int r = 0; r < size; ++r) {
        recv_displs[static_cast<std::size_t>(r) + 1] = recv_displs[static_cast<std::size_t>(r)] +
                                                       recv_counts[static_cast<std::size_t>(r)];
    }
    send.resize(std::max<std::size_t>(send.size(), 1));
    std::vector<std::int64_t> recv(static_cast<std::size_t>(std::max(1, recv_displs.back())));
    MPI_Alltoallv(send.data(),
                  send_counts.data(),
                  send_displs.data(),
                  MPI_INT64_T,
                  recv.data(),
                  recv_counts.data(),
                  recv_displs.data(),
                  MPI_INT64_T,
                  comm);
    std::vector<PairRecord> mine(static_cast<std::size_t>(recv_displs.back() / 2));
    for (std::size_t i = 0; i < mine.size(); ++i) {
        mine[i] = {recv[2 * i], recv[2 * i + 1]};
    }
    MergeRecords(mine);

    std::int64_t counts[5] = {0, 0, 0, 0, 0}; // discrete, analytic, fake, missing, unchecked
    std::vector<std::int64_t> fake;
    std::vector<std::int64_t> missing;
    for (const PairRecord& r : mine) {
        const std::int64_t flags = r.info & 3;
        if ((r.info >> 2) == GRAIN_PAIR) {
            ++counts[4];
            continue;
        }
        counts[0] += (flags & DISCRETE) ? 1 : 0;
        counts[1] += (flags & ANALYTIC) ? 1 : 0;
        if (flags == DISCRETE) {
            ++counts[2];
            fake.push_back(r.key);
        } else if (flags == ANALYTIC) {
            ++counts[3];
            missing.push_back(r.key);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, counts, 5, MPI_INT64_T, MPI_SUM, comm);
    AdjacencyComparison out;
    out.discrete = counts[0];
    out.analytic = counts[1];
    out.fake = counts[2];
    out.missing = counts[3];
    out.unchecked = counts[4];
    out.fake_examples = GatherExamples(fake, comm);
    out.missing_examples = GatherExamples(missing, comm);
    return out;
}

} // namespace exaconstit::microstructure
