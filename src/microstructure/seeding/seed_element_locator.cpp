/*
 * Implementation of the seed element locator (seed_element_locator.hpp).
 *
 * Terms used in this file:
 * - BVH (bounding volume hierarchy): a tree of nested bounding boxes that lets us skip elements
 *   or vertices that are provably too far away (axom::spin::BVH).
 * - Traversal: walking the BVH from the root; Axom's traverser (LinearBVH.hpp, traverse_tree)
 *   opens a child box only if a predicate on (point, box) is true and calls a leaf action for
 *   every leaf it reaches, with leaf_nodes[leaf] = index of the box in the build array
 *   (bvh_traverse.hpp:186-191). geometry/seed_index.cpp uses the same pattern.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 */
#include "microstructure/seeding/seed_element_locator.hpp"

#include "microstructure/mesh/element_key.hpp"
#include "microstructure/seeding/material_point_locator.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "axom/core.hpp"
#include "axom/primal/operators/squared_distance.hpp"
#include "axom/spin/BVH.hpp"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

constexpr std::uint64_t NO_KEY = std::numeric_limits<std::uint64_t>::max();

using Point = MaterialPointLocator::PointType;
using Box = MaterialPointLocator::BoxType;
using HostBVH = axom::spin::BVH<3, axom::SEQ_EXEC, double>;

/** @brief The splitmix64 mixing step (same constants as element_key.cpp). */
std::uint64_t Mix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/** @brief Throws on every rank if any listed seed id is outside [0, n). Collective. */
void CheckSeedIds(MPI_Comm comm, const std::vector<int>& seed_ids, int n, const char* caller) {
    bool bad = false;
    for (const int s : seed_ids) {
        bad = bad || s < 0 || s >= n;
    }
    ThrowIfAnyRank(comm, bad, std::string(caller) + ": a seed id is outside [0, N)");
}

/**
 * @brief Nearest of a set of points (with keys) to each query, on this rank only.
 *
 * Builds a host BVH over zero-size boxes at the points and, for every query, walks it with the
 * predicate "squared distance from the query to the box <= best d^2 so far". Using `<=` keeps every
 * box that may hold a point at exactly the best distance, so all exact ties reach the leaf action,
 * which keeps the lexicographically smallest (d^2, key). The result is therefore exactly the
 * brute-force minimum of (d^2, key) over all points. Axom defers leaf actions by one leaf
 * (bvh_traverse.hpp:160-192), so the predicate sometimes sees a slightly stale best; that only
 * opens extra boxes, never skips a needed one.
 *
 * @param px, py, pz, keys the points and their keys (size np; np may be 0).
 * @param qx, qy, qz       the queries (size nq).
 * @param best_d2, best_key outputs, size nq: DBL_MAX / NO_KEY when np == 0.
 * @param best_point       optional output, size nq: index of the winning point, -1 if none.
 */
void LocalNearest(const std::vector<double>& px, const std::vector<double>& py,
                  const std::vector<double>& pz, const std::vector<std::uint64_t>& keys,
                  const std::vector<double>& qx, const std::vector<double>& qy,
                  const std::vector<double>& qz, std::vector<double>& best_d2,
                  std::vector<std::uint64_t>& best_key, std::vector<int>* best_point) {
    const std::size_t np = px.size();
    const std::size_t nq = qx.size();
    best_d2.assign(nq, DBL_MAX);
    best_key.assign(nq, NO_KEY);
    if (best_point != nullptr) {
        best_point->assign(nq, -1);
    }
    if (np == 0 || nq == 0) {
        return;
    }

    // Zero-size box per point, then the BVH (host, SEQ_EXEC)
    std::vector<Box> boxes(np);
    for (std::size_t i = 0; i < np; ++i) {
        boxes[i] = Box(Point{px[i], py[i], pz[i]});
    }
    HostBVH bvh;
    bvh.initialize(boxes.data(), static_cast<axom::IndexType>(np));
    const auto traverser = bvh.getTraverser();

    for (std::size_t q = 0; q < nq; ++q) {
        const Point p{qx[q], qy[q], qz[q]};
        double d2_best = DBL_MAX;
        std::uint64_t key_best = NO_KEY;
        int point_best = -1;

        // Leaf: exact squared distance (the same formula for every rank), then (d^2, key) order
        auto leaf_action = [&](std::int32_t leaf, const std::int32_t* leaf_nodes) {
            const auto i = static_cast<std::size_t>(leaf_nodes[leaf]);
            const double dx = px[i] - qx[q];
            const double dy = py[i] - qy[q];
            const double dz = pz[i] - qz[q];
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < d2_best || (d2 == d2_best && keys[i] < key_best)) {
                d2_best = d2;
                key_best = keys[i];
                point_best = static_cast<int>(i);
            }
        };
        // Predicate: open a box unless it is provably farther than the best so far
        auto predicate = [&](const Point& query, const Box& box) -> bool {
            return axom::primal::squared_distance(query, box) <= d2_best;
        };
        traverser.traverse_tree(p, leaf_action, predicate);

        best_d2[q] = d2_best;
        best_key[q] = key_best;
        if (best_point != nullptr) {
            (*best_point)[q] = point_best;
        }
    }
}

/**
 * @brief Global lexicographic minimum of (d^2, key) per entry: MIN over d^2, then MIN over the
 * keys of the ranks whose local d^2 equals the global one. Collective; updates both arrays.
 * Returns, per entry, whether this rank holds the winning key.
 */
std::vector<char> ReduceNearest(MPI_Comm comm, std::vector<double>& d2,
                                std::vector<std::uint64_t>& key) {
    const int n = static_cast<int>(d2.size());
    std::vector<double> global_d2(d2);
    if (n > 0) {
        MPI_Allreduce(MPI_IN_PLACE, global_d2.data(), n, MPI_DOUBLE, MPI_MIN, comm);
    }
    for (int k = 0; k < n; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        if (d2[uk] != global_d2[uk]) {
            key[uk] = NO_KEY; // another rank is nearer
        }
    }
    std::vector<std::uint64_t> local_key(key);
    if (n > 0) {
        MPI_Allreduce(MPI_IN_PLACE, key.data(), n, MPI_UINT64_T, MPI_MIN, comm);
    }
    std::vector<char> mine(static_cast<std::size_t>(n), 0);
    for (int k = 0; k < n; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        mine[uk] = (key[uk] != NO_KEY && local_key[uk] == key[uk]) ? 1 : 0;
    }
    d2 = global_d2;
    return mine;
}

} // namespace

/* VertexKey: ElementKey's hash for a one-vertex "element" (start value, then each coordinate's
 * raw bits folded through Mix64). */
std::uint64_t VertexKey(double x, double y, double z) {
    std::uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (const double c : {x, y, z}) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &c, sizeof(bits));
        h = Mix64(h ^ bits);
    }
    return h;
}

/*
 * ContainingElementKeys
 *
 * Steps: element keys (given or computed); a host BVH over MaterialPointLocator's element boxes;
 * one batched findPoints (spin/BVH.hpp:343; candidates of point j are
 * candidates[offsets[j] .. offsets[j] + counts[j]), local element ids); the exact
 * PointInElement test on every candidate, keeping the smallest key (all candidates are tested, not
 * just the first hit, so a seed on a shared face gets the same answer on every partition); one
 * MPI_Allreduce(MIN).
 */
std::vector<std::uint64_t> ContainingElementKeys(mfem::ParMesh& pmesh,
                                                 MPI_Comm comm,
                                                 const SeedSet& seeds,
                                                 const std::vector<std::uint64_t>* element_key) {
    const int ne = pmesh.GetNE();
    ThrowIfAnyRank(comm, element_key != nullptr && static_cast<int>(element_key->size()) != ne,
                   "ContainingElementKeys: element_key must have one entry per local element");
    const int n = static_cast<int>(seeds.Size());
    std::vector<std::uint64_t> result(static_cast<std::size_t>(n), NO_KEY);
    if (n == 0) {
        return result;
    }

    if (ne > 0) {
        const std::vector<std::uint64_t> computed =
            (element_key == nullptr) ? ComputeElementKeys(pmesh) : std::vector<std::uint64_t>{};
        const std::vector<std::uint64_t>& keys = (element_key == nullptr) ? computed : *element_key;

        // Candidate elements per seed from the element boxes
        const MaterialPointLocator locator(pmesh);
        std::vector<Box> boxes(static_cast<std::size_t>(ne));
        for (int e = 0; e < ne; ++e) {
            boxes[static_cast<std::size_t>(e)] = locator.ElementBox(e);
        }
        HostBVH bvh;
        bvh.initialize(boxes.data(), ne);
        std::vector<Point> points(static_cast<std::size_t>(n));
        for (int s = 0; s < n; ++s) {
            const auto us = static_cast<std::size_t>(s);
            points[us] = Point{seeds.x[us], seeds.y[us], seeds.z[us]};
        }
        axom::Array<axom::IndexType> offsets(n, n);
        axom::Array<axom::IndexType> counts(n, n);
        axom::Array<axom::IndexType> candidates;
        bvh.findPoints(offsets.view(), counts.view(), candidates, n, points.data());

        // Exact test, smallest key of all containing elements
        for (int s = 0; s < n; ++s) {
            const auto us = static_cast<std::size_t>(s);
            const std::array<double, 3> p = {seeds.x[us], seeds.y[us], seeds.z[us]};
            for (axom::IndexType c = 0; c < counts[s]; ++c) {
                const int e = static_cast<int>(candidates[offsets[s] + c]);
                if (locator.PointInElement(e, p)) {
                    result[us] = std::min(result[us], keys[static_cast<std::size_t>(e)]);
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, result.data(), n, MPI_UINT64_T, MPI_MIN, comm);
    return result;
}

/*
 * NearestElementCentroidKeys: local nearest centroid per listed seed (LocalNearest over the
 * element centroids with the element keys), then the global (d^2, key) minimum (ReduceNearest).
 */
NearestKeys NearestElementCentroidKeys(mfem::ParMesh& pmesh,
                                       MPI_Comm comm,
                                       const SeedSet& seeds,
                                       const std::vector<int>& seed_ids,
                                       const ElementSamples* centroids,
                                       const std::vector<std::uint64_t>* element_key) {
    const int ne = pmesh.GetNE();
    const bool bad_size =
        (element_key != nullptr && static_cast<int>(element_key->size()) != ne) ||
        (centroids != nullptr && centroids->cx.Size() != ne);
    ThrowIfAnyRank(comm, bad_size,
                   "NearestElementCentroidKeys: centroids and element_key must have one entry per "
                   "local element");
    CheckSeedIds(comm, seed_ids, static_cast<int>(seeds.Size()), "NearestElementCentroidKeys");

    // Centroids and keys of the local elements (computed when not given)
    ElementSamples computed_centroids;
    if (centroids == nullptr && ne > 0) {
        ComputeElementSamples(pmesh, -1, computed_centroids);
    }
    const ElementSamples& c = (centroids != nullptr) ? *centroids : computed_centroids;
    std::vector<std::uint64_t> keys =
        (element_key != nullptr) ? *element_key : ComputeElementKeys(pmesh);
    std::vector<double> px(static_cast<std::size_t>(ne));
    std::vector<double> py(static_cast<std::size_t>(ne));
    std::vector<double> pz(static_cast<std::size_t>(ne));
    if (ne > 0) {
        const double* cx = c.cx.HostRead();
        const double* cy = c.cy.HostRead();
        const double* cz = c.cz.HostRead();
        for (int e = 0; e < ne; ++e) {
            const auto ue = static_cast<std::size_t>(e);
            px[ue] = cx[e];
            py[ue] = cy[e];
            pz[ue] = cz[e];
        }
    }

    // Queries: the listed seeds
    const std::size_t nq = seed_ids.size();
    std::vector<double> qx(nq), qy(nq), qz(nq);
    for (std::size_t k = 0; k < nq; ++k) {
        const auto s = static_cast<std::size_t>(seed_ids[k]);
        qx[k] = seeds.x[s];
        qy[k] = seeds.y[s];
        qz[k] = seeds.z[s];
    }

    NearestKeys out;
    LocalNearest(px, py, pz, keys, qx, qy, qz, out.distance_squared, out.key, nullptr);
    ReduceNearest(comm, out.distance_squared, out.key);
    return out;
}

/*
 * NearestVertex: LocalNearest over the local vertices (Mesh::GetVertex) with VertexKey, then the
 * global (d^2, key) minimum; the coordinates of the winning vertex are shared with one more
 * MPI_Allreduce(MIN) in which ranks that do not hold the winner contribute DBL_MAX.
 */
NearestVertexResult NearestVertex(mfem::ParMesh& pmesh,
                                  MPI_Comm comm,
                                  const SeedSet& seeds,
                                  const std::vector<int>& seed_ids) {
    CheckSeedIds(comm, seed_ids, static_cast<int>(seeds.Size()), "NearestVertex");

    // Local vertices and their keys
    const int nv = pmesh.GetNV();
    std::vector<double> px(static_cast<std::size_t>(nv));
    std::vector<double> py(static_cast<std::size_t>(nv));
    std::vector<double> pz(static_cast<std::size_t>(nv));
    std::vector<std::uint64_t> keys(static_cast<std::size_t>(nv));
    for (int v = 0; v < nv; ++v) {
        const double* x = pmesh.GetVertex(v);
        const auto uv = static_cast<std::size_t>(v);
        px[uv] = x[0];
        py[uv] = x[1];
        pz[uv] = x[2];
        keys[uv] = VertexKey(x[0], x[1], x[2]);
    }

    // Queries
    const std::size_t nq = seed_ids.size();
    std::vector<double> qx(nq), qy(nq), qz(nq);
    for (std::size_t k = 0; k < nq; ++k) {
        const auto s = static_cast<std::size_t>(seed_ids[k]);
        qx[k] = seeds.x[s];
        qy[k] = seeds.y[s];
        qz[k] = seeds.z[s];
    }

    // Local search, global reduction, then the winner's coordinates
    NearestVertexResult out;
    std::vector<int> winner;
    LocalNearest(px, py, pz, keys, qx, qy, qz, out.distance_squared, out.key, &winner);
    const std::vector<char> mine = ReduceNearest(comm, out.distance_squared, out.key);
    std::vector<double> pos(3 * nq, DBL_MAX);
    for (std::size_t k = 0; k < nq; ++k) {
        if (mine[k] != 0) {
            const auto v = static_cast<std::size_t>(winner[k]);
            pos[3 * k + 0] = px[v];
            pos[3 * k + 1] = py[v];
            pos[3 * k + 2] = pz[v];
        }
    }
    if (nq > 0) {
        MPI_Allreduce(MPI_IN_PLACE, pos.data(), static_cast<int>(3 * nq), MPI_DOUBLE, MPI_MIN,
                      comm);
    }
    out.position.resize(nq);
    for (std::size_t k = 0; k < nq; ++k) {
        out.position[k] = {pos[3 * k + 0], pos[3 * k + 1], pos[3 * k + 2]};
    }
    return out;
}

} // namespace exaconstit::microstructure
