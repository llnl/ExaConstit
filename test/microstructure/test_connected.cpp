/**
 * @file test_connected.cpp
 * @brief Tests of the connected distance mode (labeling/connected_labeler) and of the
 * exclusion-aware seed search SeedIndex::FindBestExcluding (geometry/seed_index).
 *
 * The connected mode keeps the straight-line (metric) labels, but every seed keeps only the
 * elements its cell connects to the seed's anchor element through material; an element cut off
 * from its seed's anchor (an *orphan*) excludes that seed and takes the best remaining one, in
 * rounds. The tests check:
 * - FindBestExcluding equals a brute-force loop over the non-excluded seeds, for all four metrics;
 * - on the U-shaped part, a seed whose straight-line cell reaches across the slot into the other
 *   arm loses those elements, and every element's label equals an independent serial reference
 *   implementation of the rounds written below (breadth-first search over the serial face graph
 *   and a brute-force seed search);
 * - a body without any seed is exempt and counted;
 * - the max_rounds cap leaves counted orphans;
 * - a seed whose own element is won by a neighbor keeps its other elements (Voronoi corner seed,
 *   Laguerre), and a seed that labels no element has no cell (counted);
 * - the shared seed element locator (containing element, nearest centroid, nearest vertex);
 * - a seed outside the material is anchored to the nearest element;
 * - vote sampling.
 *
 * Rank-count independence is checked inside every run: the distributed result over
 * MPI_COMM_WORLD (METIS partition, and a round-robin partition with an empty last rank) is
 * gathered into serial element order and compared with the result every rank computes alone on
 * the serial mesh (MPI_COMM_SELF), which is the np = 1 answer. Running at np = 1..4
 * (test_connected.cmake) therefore compares np = 2, 3, 4 against np = 1.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/labeling/connected_labeler.hpp"
#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/seeding/seed_element_locator.hpp"
#include "microstructure/microstructure_options.hpp"

#include "mfem.hpp"

#include "axom/core/execution/execution_space.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <mpi.h>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

const std::string data_dir = MS_CONNECTED_DATA_DIR;

// ------------------------------------------------------------------------------------------------
// Meshes and distribution (same construction as test_fragments.cpp)
// ------------------------------------------------------------------------------------------------

/**
 * Hex mesh of the cells (i, j, k) of an nx x ny x nz grid with cell size h for which kept(i, j, k)
 * is true; cell (i, j, k) is [i h, (i+1) h] x [j h, (j+1) h] x [k h, (k+1) h].
 */
template <typename Kept>
mfem::Mesh BuildGrid(int nx, int ny, int nz, double h, Kept kept) {
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    int num_cells = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                num_cells += kept(i, j, k) ? 1 : 0;
            }
        }
    }
    mfem::Mesh mesh(3, (nx + 1) * (ny + 1) * (nz + 1), num_cells, 0, 3);
    for (int k = 0; k <= nz; k++) {
        for (int j = 0; j <= ny; j++) {
            for (int i = 0; i <= nx; i++) {
                mesh.AddVertex(i * h, j * h, k * h);
            }
        }
    }
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (!kept(i, j, k)) {
                    continue;
                }
                int c[8];
                for (int b = 0; b < 8; b++) {
                    c[b] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                }
                const int hex[8] = {c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]};
                mesh.AddHex(hex);
            }
        }
    }
    mesh.FinalizeHexMesh(1, 1, true);
    mesh.RemoveUnusedVertices();
    return mesh;
}

/** The U-shaped part of the end-to-end tests: [0,3] x [0,2] x [0,1] minus the slot
 * [1,2] x [1,2] x [0,1], 4 hexes per unit length (320 elements). */
mfem::Mesh LoadUShape() {
    return mfem::Mesh((data_dir + "/microstructure/u_shape_hex.mesh").c_str(), 1, 1, true);
}

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** METIS partition of `serial` over `comm`. Collective. */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/**
 * Round-robin partition over ranks 0 .. nranks-2, so the last rank owns no element (for >= 2
 * ranks) and the partitions are scattered (many shared faces). Collective.
 */
Distributed DistributeWithEmptyLastRank(mfem::Mesh& serial, MPI_Comm comm) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const int owners = (nranks > 1) ? nranks - 1 : 1;
    Distributed d;
    d.order.partitioning.SetSize(serial.GetNE());
    int num_local = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        d.order.partitioning[e] = e % owners;
        num_local += (e % owners == rank) ? 1 : 0;
    }
    d.order.local_to_serial.SetSize(num_local);
    int next = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        if (d.order.partitioning[e] == rank) {
            d.order.local_to_serial[next++] = e;
        }
    }
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** Per-element values gathered into serial order on every rank. Collective. */
std::vector<int> GatherAll(MPI_Comm comm, const Distributed& d, const mfem::Array<int>& local,
                           int serial_ne) {
    std::vector<int> out = ms::GatherToSerial(comm, d.order, local, 0);
    out.resize(static_cast<std::size_t>(serial_ne));
    MPI_Bcast(out.data(), serial_ne, MPI_INT, 0, comm);
    return out;
}

// ------------------------------------------------------------------------------------------------
// Distributed runs of the library
// ------------------------------------------------------------------------------------------------

/** Input (Euclidean) and output (connected) labels in serial order, and the report. */
struct RunResult {
    std::vector<int> input;
    std::vector<int> output;
    ms::ConnectedLabelReport report;
};

/**
 * Euclidean labels (centroid, or the quadrature vote when params.sampling asks for it) followed
 * by ConstrainToConnectedCells, on `serial` distributed over `comm`. Collective.
 */
RunResult RunConnected(mfem::Mesh& serial, MPI_Comm comm, const ms::SeedSet& seeds,
                       const ms::TessellationOptions& tess,
                       const ms::ConnectedLabelParameters& params, bool empty_last_rank) {
    Distributed d = empty_last_rank ? DistributeWithEmptyLastRank(serial, comm)
                                    : Distribute(serial, comm);
    mfem::Array<int> element_seed;
    if (params.sampling == ms::SamplingMode::QUADRATURE_VOTE) {
        ms::QuadratureVoteParameters vote;
        vote.metric = tess.metric;
        vote.global_tensor = tess.global_tensor;
        vote.num_bins = tess.num_bins;
        vote.levels = params.vote_levels;
        ms::LabelElementsByQuadratureVote(*d.pmesh, seeds, vote, element_seed);
    } else {
        ms::LabelElementsWithSeeds(*d.pmesh, seeds, tess, element_seed);
    }
    RunResult r;
    r.input = GatherAll(comm, d, element_seed, serial.GetNE());
    const ms::MeshAdjacency adjacency(*d.pmesh);
    r.report = ms::ConstrainToConnectedCells(*d.pmesh, comm, adjacency, seeds, tess, element_seed,
                                             params);
    r.output = GatherAll(comm, d, element_seed, serial.GetNE());
    return r;
}

/**
 * RunConnected over MPI_COMM_WORLD (METIS and round-robin with an empty last rank) and over
 * MPI_COMM_SELF (the np = 1 reference); all three must agree exactly. Returns the world result.
 */
RunResult RunAndCompare(mfem::Mesh& serial, const ms::SeedSet& seeds,
                        const ms::TessellationOptions& tess,
                        const ms::ConnectedLabelParameters& params) {
    const RunResult world = RunConnected(serial, MPI_COMM_WORLD, seeds, tess, params, false);
    const RunResult scattered = RunConnected(serial, MPI_COMM_WORLD, seeds, tess, params, true);
    const RunResult self = RunConnected(serial, MPI_COMM_SELF, seeds, tess, params, false);
    for (const RunResult* other : {&scattered, &self}) {
        EXPECT_EQ(world.input, other->input);
        EXPECT_EQ(world.output, other->output);
        EXPECT_EQ(world.report.rounds, other->report.rounds);
        EXPECT_EQ(world.report.relabeled_elements, other->report.relabeled_elements);
        EXPECT_EQ(world.report.unresolved_elements, other->report.unresolved_elements);
        EXPECT_EQ(world.report.unanchored_component_elements,
                  other->report.unanchored_component_elements);
        EXPECT_EQ(world.report.seeds_without_cell, other->report.seeds_without_cell);
    }
    return world;
}

// ------------------------------------------------------------------------------------------------
// Independent serial reference of the connected rounds
// ------------------------------------------------------------------------------------------------

/** Neighbors of every element of a serial mesh through interior faces. */
std::vector<std::vector<int>> FaceGraph(const mfem::Mesh& mesh) {
    std::vector<std::vector<int>> nbrs(static_cast<std::size_t>(mesh.GetNE()));
    for (int f = 0; f < mesh.GetNumFaces(); f++) {
        int e1 = -1;
        int e2 = -1;
        mesh.GetFaceElements(f, &e1, &e2);
        if (e1 >= 0 && e2 >= 0) {
            nbrs[static_cast<std::size_t>(e1)].push_back(e2);
            nbrs[static_cast<std::size_t>(e2)].push_back(e1);
        }
    }
    return nbrs;
}

/** Breadth-first search: component index (0, 1, 2, ...) of every element; elements are joined
 * across a face when their labels are equal. */
std::vector<int> Components(const std::vector<std::vector<int>>& nbrs,
                            const std::vector<int>& label) {
    const std::size_t ne = nbrs.size();
    std::vector<int> comp(ne, -1);
    int next = 0;
    for (std::size_t start = 0; start < ne; ++start) {
        if (comp[start] >= 0) {
            continue;
        }
        std::queue<int> todo;
        todo.push(static_cast<int>(start));
        comp[start] = next;
        while (!todo.empty()) {
            const int e = todo.front();
            todo.pop();
            for (const int n : nbrs[static_cast<std::size_t>(e)]) {
                if (comp[static_cast<std::size_t>(n)] < 0 &&
                    label[static_cast<std::size_t>(n)] == label[static_cast<std::size_t>(e)]) {
                    comp[static_cast<std::size_t>(n)] = next;
                    todo.push(n);
                }
            }
        }
        ++next;
    }
    return comp;
}

/** Number of face-connected pieces of every label (label -> count). */
std::vector<int> PiecesPerLabel(const std::vector<std::vector<int>>& nbrs,
                                const std::vector<int>& label, int num_labels) {
    const std::vector<int> comp = Components(nbrs, label);
    std::vector<std::vector<int>> seen(static_cast<std::size_t>(num_labels));
    for (std::size_t e = 0; e < label.size(); ++e) {
        auto& list = seen[static_cast<std::size_t>(label[e])];
        if (std::find(list.begin(), list.end(), comp[e]) == list.end()) {
            list.push_back(comp[e]);
        }
    }
    std::vector<int> pieces(static_cast<std::size_t>(num_labels));
    for (int s = 0; s < num_labels; ++s) {
        const auto us = static_cast<std::size_t>(s);
        pieces[us] = static_cast<int>(seen[us].size());
    }
    return pieces;
}

/** Sample points of every element of the serial mesh: nq points per element, x/y/z/w. */
struct SerialSamples {
    int nq = 1;
    std::vector<double> x, y, z, w;
};

/** Centroids (nq = 1) or voting points of the serial mesh, from the shared samplers. */
SerialSamples MakeSamples(mfem::Mesh& mesh, const ms::ConnectedLabelParameters& params) {
    SerialSamples s;
    const int ne = mesh.GetNE();
    if (params.sampling == ms::SamplingMode::QUADRATURE_VOTE) {
        ms::QuadraturePointSampler sampler(
            mesh, ms::VoteRule(mesh.GetElementGeometry(0), params.vote_levels));
        ms::QuadraturePointSamples pts;
        sampler.Sample(0, ne, pts);
        s.nq = pts.num_qp;
        const int n = ne * s.nq;
        s.x.assign(pts.x.HostRead(), pts.x.HostRead() + n);
        s.y.assign(pts.y.HostRead(), pts.y.HostRead() + n);
        s.z.assign(pts.z.HostRead(), pts.z.HostRead() + n);
        s.w.assign(pts.weight.HostRead(), pts.weight.HostRead() + n);
    } else {
        ms::ElementSamples c;
        ms::ComputeElementSamples(mesh, -1, c);
        s.nq = 1;
        s.x.assign(c.cx.HostRead(), c.cx.HostRead() + ne);
        s.y.assign(c.cy.HostRead(), c.cy.HostRead() + ne);
        s.z.assign(c.cz.HostRead(), c.cz.HostRead() + ne);
        s.w.assign(static_cast<std::size_t>(ne), 1.0);
    }
    return s;
}

/** Voronoi (d^2) or Laguerre (d^2 - w) metric value of seed s at a point, written out here. */
double RefMetric(ms::MetricType metric, const ms::SeedSet& seeds, int s, double x, double y,
                 double z) {
    const auto us = static_cast<std::size_t>(s);
    const double dx = x - seeds.x[us];
    const double dy = y - seeds.y[us];
    const double dz = z - seeds.z[us];
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (metric == ms::MetricType::LAGUERRE && !seeds.weight.empty()) {
        return d2 - seeds.weight[us];
    }
    return d2;
}

/**
 * Label of element e by brute force over the seeds not in `excluded`: per point the smallest
 * (f, index); then (vote) the label with the largest total point weight, summed in point order,
 * ties to the lower index. -1 when every seed is excluded.
 */
int RefLabel(ms::MetricType metric, const ms::SeedSet& seeds, const SerialSamples& samples, int e,
             const std::vector<int>& excluded) {
    const int n = static_cast<int>(seeds.Size());
    std::vector<int> point_label(static_cast<std::size_t>(samples.nq), -1);
    for (int q = 0; q < samples.nq; ++q) {
        const auto k = static_cast<std::size_t>(e * samples.nq + q);
        double best_f = DBL_MAX;
        int best = -1;
        for (int s = 0; s < n; ++s) {
            if (std::find(excluded.begin(), excluded.end(), s) != excluded.end()) {
                continue;
            }
            const double f = RefMetric(metric, seeds, s, samples.x[k], samples.y[k], samples.z[k]);
            if (best < 0 || f < best_f) { // ascending s, so an equal f keeps the lower index
                best_f = f;
                best = s;
            }
        }
        point_label[static_cast<std::size_t>(q)] = best;
    }
    if (samples.nq == 1) {
        return point_label[0];
    }
    int winner = -1;
    double winner_weight = 0.0;
    for (int s = 0; s < n; ++s) { // ascending s: an exactly equal total keeps the lower index
        double total = 0.0;
        bool any = false;
        for (int q = 0; q < samples.nq; ++q) {
            if (point_label[static_cast<std::size_t>(q)] == s) {
                total += samples.w[static_cast<std::size_t>(e * samples.nq + q)];
                any = true;
            }
        }
        if (any && (winner < 0 || total > winner_weight)) {
            winner = s;
            winner_weight = total;
        }
    }
    return winner;
}

/** Result of the serial reference. */
struct RefResult {
    std::vector<int> input;
    std::vector<int> output;
    std::vector<std::uint64_t> anchor_key;           ///< label-independent (SeedAnchorElementKeys)
    std::vector<std::uint64_t> connected_anchor_key; ///< label-aware (ConnectedAnchorKeys)
    int rounds = 0;
    std::int64_t relabeled = 0;
    std::int64_t unresolved = 0;
    std::int64_t unanchored = 0;
    int seeds_without_cell = 0;
};

/**
 * Serial reference that reproduces the library's rounds (the fixed greedy order documented in
 * connected_labeler.hpp, not the strict "nearest seed whose cell reaches the element"), written
 * independently of the library code from the algorithm statement:
 * - containing element of seed s: the element whose vertex bounding box contains s (exact for the
 *   axis-aligned hexes used here; smallest element key if several);
 * - anchor of seed s: the containing element if the input labels give it to s, else the s-labeled
 *   element with the nearest centroid (smallest key on ties), else none (no cell);
 * - bodies (components of the whole mesh) without an anchor are exempt;
 * - each round: components per label by breadth-first search; an element is an orphan when its
 *   seed has no anchor or its anchor lies in another component; every orphan with a candidate
 *   left permanently excludes its seed and takes the brute-force best non-excluded seed, all
 *   orphans of a round at once.
 * Also returns the label-independent anchors (containing element, else nearest centroid) for the
 * SeedAnchorElementKeys check.
 */
RefResult RoundsReference(mfem::Mesh& mesh, const ms::SeedSet& seeds, ms::MetricType metric,
                    const ms::ConnectedLabelParameters& params) {
    const int ne = mesh.GetNE();
    const int n = static_cast<int>(seeds.Size());
    const auto nbrs = FaceGraph(mesh);
    const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(mesh);
    const SerialSamples samples = MakeSamples(mesh, params);
    ms::ElementSamples centroids;
    ms::ComputeElementSamples(mesh, -1, centroids);
    const double* cx = centroids.cx.HostRead();
    const double* cy = centroids.cy.HostRead();
    const double* cz = centroids.cz.HostRead();

    // Label-independent anchors (element index and key) and the containing elements
    RefResult ref;
    std::vector<int> anchor(static_cast<std::size_t>(n), -1);
    std::vector<int> containing(static_cast<std::size_t>(n), -1);
    for (int s = 0; s < n; ++s) {
        const auto us = static_cast<std::size_t>(s);
        const double p[3] = {seeds.x[us], seeds.y[us], seeds.z[us]};
        for (int e = 0; e < ne; ++e) {
            mfem::Array<int> v;
            mesh.GetElementVertices(e, v);
            double lo[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
            double hi[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
            for (int i = 0; i < v.Size(); ++i) {
                const double* c = mesh.GetVertex(v[i]);
                for (int d = 0; d < 3; ++d) {
                    lo[d] = std::min(lo[d], c[d]);
                    hi[d] = std::max(hi[d], c[d]);
                }
            }
            bool inside = true;
            for (int d = 0; d < 3; ++d) {
                inside = inside && p[d] >= lo[d] && p[d] <= hi[d];
            }
            if (inside && (anchor[us] < 0 || keys[static_cast<std::size_t>(e)] <
                                                 keys[static_cast<std::size_t>(anchor[us])])) {
                anchor[us] = e;
            }
        }
        containing[us] = anchor[us];
        if (anchor[us] < 0) {
            double best = DBL_MAX;
            for (int e = 0; e < ne; ++e) {
                const double dx = cx[e] - p[0];
                const double dy = cy[e] - p[1];
                const double dz = cz[e] - p[2];
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (anchor[us] < 0 || d2 < best ||
                    (d2 == best && keys[static_cast<std::size_t>(e)] <
                                       keys[static_cast<std::size_t>(anchor[us])])) {
                    best = d2;
                    anchor[us] = e;
                }
            }
        }
        ref.anchor_key.push_back(keys[static_cast<std::size_t>(anchor[us])]);
    }

    // Euclidean input labels
    std::vector<int> label(static_cast<std::size_t>(ne));
    for (int e = 0; e < ne; ++e) {
        label[static_cast<std::size_t>(e)] = RefLabel(metric, seeds, samples, e, {});
    }
    ref.input = label;

    // Connected anchors: containing element if labeled s, else nearest s-labeled centroid
    for (int s = 0; s < n; ++s) {
        const auto us = static_cast<std::size_t>(s);
        const double p[3] = {seeds.x[us], seeds.y[us], seeds.z[us]};
        int a = -1;
        if (containing[us] >= 0 && label[static_cast<std::size_t>(containing[us])] == s) {
            a = containing[us];
        } else {
            double best = DBL_MAX;
            for (int e = 0; e < ne; ++e) {
                if (label[static_cast<std::size_t>(e)] != s) {
                    continue;
                }
                const double dx = cx[e] - p[0];
                const double dy = cy[e] - p[1];
                const double dz = cz[e] - p[2];
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (a < 0 || d2 < best ||
                    (d2 == best && keys[static_cast<std::size_t>(e)] <
                                       keys[static_cast<std::size_t>(a)])) {
                    best = d2;
                    a = e;
                }
            }
        }
        anchor[us] = a;
        ref.connected_anchor_key.push_back(
            (a >= 0) ? keys[static_cast<std::size_t>(a)]
                     : std::numeric_limits<std::uint64_t>::max());
    }

    // Exempt bodies
    const std::vector<int> all_zero(static_cast<std::size_t>(ne), 0);
    const std::vector<int> body = Components(nbrs, all_zero);
    std::vector<char> exempt(static_cast<std::size_t>(ne), 1);
    for (int e = 0; e < ne; ++e) {
        for (int s = 0; s < n; ++s) {
            const int a = anchor[static_cast<std::size_t>(s)];
            if (a >= 0 && body[static_cast<std::size_t>(a)] == body[static_cast<std::size_t>(e)]) {
                exempt[static_cast<std::size_t>(e)] = 0;
            }
        }
        ref.unanchored += exempt[static_cast<std::size_t>(e)];
    }

    // Rounds
    std::vector<std::vector<int>> excluded(static_cast<std::size_t>(ne));
    std::vector<char> stuck(static_cast<std::size_t>(ne), 0);
    for (;;) {
        const std::vector<int> comp = Components(nbrs, label);
        auto is_orphan = [&](int e) {
            const auto ue = static_cast<std::size_t>(e);
            const int a = anchor[static_cast<std::size_t>(label[ue])];
            if (exempt[ue] != 0) {
                return false;
            }
            return a < 0 || label[static_cast<std::size_t>(a)] != label[ue] ||
                   comp[static_cast<std::size_t>(a)] != comp[ue];
        };
        std::vector<int> movable;
        std::int64_t orphans = 0;
        for (int e = 0; e < ne; ++e) {
            if (is_orphan(e)) {
                ++orphans;
                if (stuck[static_cast<std::size_t>(e)] == 0) {
                    movable.push_back(e);
                }
            }
        }
        if (movable.empty() || ref.rounds >= params.max_rounds) {
            ref.unresolved = orphans;
            for (int s = 0; s < n; ++s) {
                const int a = anchor[static_cast<std::size_t>(s)];
                if (a < 0 || label[static_cast<std::size_t>(a)] != s) {
                    ++ref.seeds_without_cell;
                }
            }
            break;
        }
        ++ref.rounds;
        std::vector<int> next = label;
        for (const int e : movable) {
            const auto ue = static_cast<std::size_t>(e);
            excluded[ue].push_back(label[ue]);
            const int l = RefLabel(metric, seeds, samples, e, excluded[ue]);
            if (l >= 0) {
                next[ue] = l;
            } else {
                stuck[ue] = 1;
            }
        }
        label = next;
    }
    ref.output = label;
    for (int e = 0; e < ne; ++e) {
        ref.relabeled += (ref.output[static_cast<std::size_t>(e)] !=
                          ref.input[static_cast<std::size_t>(e)])
                             ? 1
                             : 0;
    }
    return ref;
}

/** Library run (all partitions) and serial reference must agree in every label and count. */
RefResult ExpectMatchesReference(mfem::Mesh& mesh, const ms::SeedSet& seeds,
                                 const ms::TessellationOptions& tess,
                                 const ms::ConnectedLabelParameters& params, RunResult* run_out) {
    const RunResult run = RunAndCompare(mesh, seeds, tess, params);
    const RefResult ref = RoundsReference(mesh, seeds, tess.metric, params);
    EXPECT_EQ(run.input, ref.input) << "Euclidean input labels differ from the reference";
    EXPECT_EQ(run.output, ref.output);
    EXPECT_EQ(run.report.rounds, ref.rounds);
    EXPECT_EQ(run.report.relabeled_elements, ref.relabeled);
    EXPECT_EQ(run.report.unresolved_elements, ref.unresolved);
    EXPECT_EQ(run.report.unanchored_component_elements, ref.unanchored);
    EXPECT_EQ(run.report.seeds_without_cell, ref.seeds_without_cell);
    if (run_out != nullptr) {
        *run_out = run;
    }
    return ref;
}

/** Seed set from positions (and optional weights); source ids 0..N-1. */
ms::SeedSet MakeSeedSet(const std::vector<std::array<double, 3>>& p,
                        const std::vector<double>& w = {}) {
    ms::SeedSet s;
    for (std::size_t i = 0; i < p.size(); ++i) {
        s.x.push_back(p[i][0]);
        s.y.push_back(p[i][1]);
        s.z.push_back(p[i][2]);
        s.source_id.push_back(static_cast<std::int64_t>(i));
    }
    s.weight = w;
    return s;
}

/** True if the element centroid (x, y) lies in the left arm (x < 1, y > 1) or the right arm
 * (x > 2, y > 1) of the U-shaped part. */
void ArmMembership(mfem::Mesh& mesh, std::vector<char>& left, std::vector<char>& right) {
    ms::ElementSamples c;
    ms::ComputeElementSamples(mesh, -1, c);
    const double* cx = c.cx.HostRead();
    const double* cy = c.cy.HostRead();
    left.assign(static_cast<std::size_t>(mesh.GetNE()), 0);
    right.assign(static_cast<std::size_t>(mesh.GetNE()), 0);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        left[static_cast<std::size_t>(e)] = (cx[e] < 1.0 && cy[e] > 1.0) ? 1 : 0;
        right[static_cast<std::size_t>(e)] = (cx[e] > 2.0 && cy[e] > 1.0) ? 1 : 0;
    }
}

// ------------------------------------------------------------------------------------------------
// FindBestExcluding vs brute force
// ------------------------------------------------------------------------------------------------

/**
 * Brute force: smallest (f, index) over the seeds not on `list`; {-1, DBL_MAX} if all are listed.
 */
template <typename Metric>
std::pair<int, double> BruteExcluding(const Metric& metric, int n, double x, double y, double z,
                                      const std::vector<int>& list) {
    int best = -1;
    double best_f = DBL_MAX;
    for (int s = 0; s < n; ++s) {
        if (std::find(list.begin(), list.end(), s) != list.end()) {
            continue;
        }
        const double f = metric.Eval(s, x, y, z);
        if (best < 0 || f < best_f) {
            best = s;
            best_f = f;
        }
    }
    return {best, best_f};
}

/**
 * Builds a SeedIndex for `seeds`, makes random queries with random exclusion lists (lengths
 * 0..6; entries drawn from the query's nearest few seeds, which are the ones that matter, and
 * from all seeds, sometimes repeated), and expects FindBestExcluding to give exactly the brute
 * force index and bitwise the same f. The first query of a 2-seed set excludes both seeds and
 * must report -1.
 */
template <typename Metric>
void ExpectExcludingMatchesBrute(const ms::SeedSet& seeds, const ms::MetricSetup& setup,
                                 const std::string& tag) {
    const int n = static_cast<int>(seeds.Size());
    ms::SeedIndex<axom::SEQ_EXEC, Metric> index;
    index.Build(seeds, setup, 4, axom::execution_space<axom::SEQ_EXEC>::allocatorID());

    std::mt19937_64 gen(1234 + n);
    std::uniform_real_distribution<double> uni(-0.1, 1.1);
    const int nq = 400;
    mfem::Vector qx(nq), qy(nq), qz(nq);
    std::vector<std::vector<int>> lists(static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        qx(q) = uni(gen);
        qy(q) = uni(gen);
        qz(q) = uni(gen);
        // Seeds ordered by f at this query: excluding the best few changes the answer
        std::vector<std::pair<double, int>> order;
        for (int s = 0; s < n; ++s) {
            order.push_back({index.GetMetric().Eval(s, qx(q), qy(q), qz(q)), s});
        }
        std::sort(order.begin(), order.end());
        const int len = q % 7;
        for (int k = 0; k < len; ++k) {
            const int pick = (k % 2 == 0) ? order[static_cast<std::size_t>(
                                                std::min(k, n - 1))].second
                                          : static_cast<int>(gen() % static_cast<unsigned>(n));
            lists[static_cast<std::size_t>(q)].push_back(pick);
        }
    }
    if (n == 2) {
        lists[0] = {1, 0}; // everything excluded
    }
    mfem::Array<int> offsets(nq + 1);
    std::vector<int> flat;
    offsets[0] = 0;
    for (int q = 0; q < nq; ++q) {
        for (const int s : lists[static_cast<std::size_t>(q)]) {
            flat.push_back(s);
        }
        offsets[q + 1] = static_cast<int>(flat.size());
    }
    mfem::Array<int> ids(static_cast<int>(flat.size()));
    for (std::size_t k = 0; k < flat.size(); ++k) {
        ids[static_cast<int>(k)] = flat[k];
    }

    mfem::Array<ms::BestPair> out;
    index.FindBestExcluding(qx, qy, qz, offsets, ids, out);
    ASSERT_EQ(out.Size(), nq);
    const ms::BestPair* hout = out.HostRead();
    int mismatches = 0;
    for (int q = 0; q < nq; ++q) {
        const auto& list = lists[static_cast<std::size_t>(q)];
        const auto ref = BruteExcluding(index.GetMetric(), n, qx(q), qy(q), qz(q), list);
        if (hout[q].id1 != ref.first || (ref.first >= 0 && hout[q].f1 != ref.second) ||
            hout[q].id2 != -1) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << tag;
    if (n == 2) {
        EXPECT_EQ(hout[0].id1, -1) << tag << ": all seeds excluded";
    }
}

} // namespace

// Exactness of the exclusion-aware search: for every metric (Voronoi, Laguerre, Additive, Quadric
// with a global and with per-seed tensors), with coincident seeds and random exclusion lists, the
// result must be exactly the brute-force minimum over the non-excluded seeds, because excluded
// seeds are skipped only at evaluation and the pruning bounds are true lower bounds.
TEST(SeedIndex, FindBestExcludingMatchesBruteForce) {
    for (const int n : {1, 2, 300}) {
        std::mt19937_64 gen(77 + n);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        ms::SeedSet seeds;
        for (int i = 0; i < n; ++i) {
            seeds.x.push_back(uni(gen));
            seeds.y.push_back(uni(gen));
            seeds.z.push_back(uni(gen));
            seeds.source_id.push_back(i);
        }
        // Coincident seeds (same position as a lower index)
        if (n >= 10) {
            for (const int i : {5, 9}) {
                seeds.x[static_cast<std::size_t>(i)] = seeds.x[0];
                seeds.y[static_cast<std::size_t>(i)] = seeds.y[0];
                seeds.z[static_cast<std::size_t>(i)] = seeds.z[0];
            }
        }
        const double h = std::cbrt(1.0 / n);
        const std::string tag = "N=" + std::to_string(n);
        ExpectExcludingMatchesBrute<ms::VoronoiMetric>(seeds, ms::MetricSetup{}, "Voronoi " + tag);

        ms::SeedSet weighted = seeds;
        for (int i = 0; i < n; ++i) {
            weighted.weight.push_back(h * h * uni(gen));
        }
        ExpectExcludingMatchesBrute<ms::LaguerreMetric>(weighted, ms::MetricSetup{},
                                                        "Laguerre " + tag);
        ms::SeedSet additive = seeds;
        for (int i = 0; i < n; ++i) {
            additive.weight.push_back(h * uni(gen));
        }
        ExpectExcludingMatchesBrute<ms::AdditiveMetric>(additive, ms::MetricSetup{},
                                                        "Additive " + tag);

        ms::MetricSetup global;
        global.global_tensor = {2.0, 1.0, 0.5, 0.1, 0.0, 0.2};
        ExpectExcludingMatchesBrute<ms::QuadricMetric>(weighted, global, "Quadric global " + tag);

        ms::SeedSet shaped = weighted;
        for (int i = 0; i < n; ++i) {
            const double a = 0.2 + uni(gen);
            const double b = 0.2 + uni(gen);
            const double c = 0.2 + uni(gen);
            shaped.shape.insert(shaped.shape.end(), {a, b, c, 0.0, 0.0, 0.0});
        }
        ms::MetricSetup per_seed;
        per_seed.per_seed_shape = true;
        ExpectExcludingMatchesBrute<ms::QuadricMetric>(shaped, per_seed, "Quadric per-seed " + tag);
    }
}

// A seed near the top of the left arm (A) reaches straight across the slot: its Voronoi cell holds
// the tip of the right arm, which is connected to A only through the base, where seed B's cell
// is. The constraint must give those elements to the next-best seed whose cell reaches them
// through material (B), so afterwards every grain is one face-connected piece; in particular no
// grain has elements in both arms without a path through its own elements in the base. Labels
// and counts must match the serial reference exactly.
TEST(ConnectedLabeler, UShapeArmCrossingMatchesReference) {
    mfem::Mesh mesh = LoadUShape();
    const ms::SeedSet seeds =
        MakeSeedSet({{0.9, 1.9, 0.45}, {1.55, 0.2, 0.45}, {0.3, 0.3, 0.45}});
    ms::TessellationOptions tess;
    ms::ConnectedLabelParameters params;
    RunResult run;
    ExpectMatchesReference(mesh, seeds, tess, params, &run);

    std::vector<char> left;
    std::vector<char> right;
    ArmMembership(mesh, left, right);
    // Input: seed 0 owns elements in both arms (the crossing this mode removes)
    bool crossing_in = false;
    bool crossing_out = false;
    for (std::size_t e = 0; e < run.input.size(); ++e) {
        crossing_in = crossing_in || (right[e] != 0 && run.input[e] == 0);
        crossing_out = crossing_out || (right[e] != 0 && run.output[e] == 0);
    }
    EXPECT_TRUE(crossing_in) << "test setup: seed 0's Euclidean cell should cross the slot";
    EXPECT_FALSE(crossing_out);

    // Output: one piece per grain, nothing unresolved, the crossing elements were relabeled
    const auto pieces = PiecesPerLabel(FaceGraph(mesh), run.output, 3);
    for (const int p : pieces) {
        EXPECT_LE(p, 1);
    }
    EXPECT_EQ(run.report.unresolved_elements, 0);
    EXPECT_GE(run.report.rounds, 1);
    EXPECT_GT(run.report.relabeled_elements, 0);
    EXPECT_EQ(run.report.unanchored_component_elements, 0);
    EXPECT_EQ(run.report.seeds_without_cell, 0);
}

// Two seeds in the left arm (A at the top, A2 lower) are the first and second nearest to the tip
// of the right arm, so its elements need two rounds (A, then A2, are both disconnected) before
// they reach B in the base. With max_rounds = 1 some of them are left over as counted orphans
// with their last label, and with max_rounds = 0 nothing moves and the input orphans are counted;
// both must match the reference with the same cap.
TEST(ConnectedLabeler, MaxRoundsCapLeavesCountedOrphans) {
    mfem::Mesh mesh = LoadUShape();
    const ms::SeedSet seeds =
        MakeSeedSet({{0.9, 1.9, 0.45}, {0.9, 1.4, 0.45}, {1.55, 0.2, 0.45}});
    ms::TessellationOptions tess;
    ms::ConnectedLabelParameters params;

    RunResult full;
    const RefResult ref_full = ExpectMatchesReference(mesh, seeds, tess, params, &full);
    EXPECT_GE(ref_full.rounds, 2) << "test setup: two rounds should be needed";
    EXPECT_EQ(full.report.unresolved_elements, 0);

    params.max_rounds = 1;
    RunResult capped;
    ExpectMatchesReference(mesh, seeds, tess, params, &capped);
    EXPECT_EQ(capped.report.rounds, 1);
    EXPECT_GT(capped.report.unresolved_elements, 0);

    params.max_rounds = 0;
    RunResult none;
    ExpectMatchesReference(mesh, seeds, tess, params, &none);
    EXPECT_EQ(none.report.rounds, 0);
    EXPECT_EQ(none.output, none.input);
    EXPECT_EQ(none.report.relabeled_elements, 0);
    EXPECT_GT(none.report.unresolved_elements, capped.report.unresolved_elements);
}

// Two separate boxes (a slot of one cell between them), seeds only in the left box. The right box
// holds no seed's anchor, so it cannot be connected to any seed: its 36 elements keep their
// Euclidean labels and are counted as unanchored, and the left box is constrained as usual.
TEST(ConnectedLabeler, BodyWithoutSeedIsExempt) {
    const double h = 1.0 / 3.0;
    mfem::Mesh mesh = BuildGrid(9, 3, 3, h, [](int i, int, int) { return i != 4; });
    const ms::SeedSet seeds = MakeSeedSet({{0.2, 0.5, 0.55}, {1.1, 0.4, 0.6}});
    ms::TessellationOptions tess;
    ms::ConnectedLabelParameters params;
    RunResult run;
    ExpectMatchesReference(mesh, seeds, tess, params, &run);
    EXPECT_EQ(run.report.unanchored_component_elements, 4 * 3 * 3);

    // Every element of the right box (centroid x > 5/3) keeps its input label
    ms::ElementSamples c;
    ms::ComputeElementSamples(mesh, -1, c);
    const double* cx = c.cx.HostRead();
    for (int e = 0; e < mesh.GetNE(); ++e) {
        if (cx[e] > 5.0 * h) {
            EXPECT_EQ(run.output[static_cast<std::size_t>(e)],
                      run.input[static_cast<std::size_t>(e)]);
        }
    }
}

/** Number of elements carrying each label (labels 0..num_labels-1). */
std::vector<int> CountLabels(const std::vector<int>& label, int num_labels) {
    std::vector<int> count(static_cast<std::size_t>(num_labels), 0);
    for (const int l : label) {
        ++count[static_cast<std::size_t>(l)];
    }
    return count;
}

// Laguerre f_i = |x - s_i|^2 - w_i.
// - Seed 0 sits at the centroid of its containing element (0.375)^3, where f_0 = 0, but seed 1
//   one element further along the diagonal with weight 0.2 has f_1 = 3 (0.25)^2 - 0.2 = -0.0125
//   there, so seed 1 wins seed 0's own element. Seed 0 still wins the elements on its far side
//   (e.g. centroid (0.125)^3: f_0 = 0.1875 < f_1 = 0.55), so it is anchored to its nearest own
//   element and keeps them all: it has a cell.
// - Seed 4 at (0.6)^3 with weight 0 never wins anywhere: f_4 - f_1 = 0.05 (x + y + z) + 0.108 > 0
//   for every point of the box, so it labels no element and is the one seed without a cell.
TEST(ConnectedLabeler, LaguerreDominatedSeedHasNoCell) {
    mfem::Mesh mesh = BuildGrid(4, 4, 4, 0.25, [](int, int, int) { return true; });
    const ms::SeedSet seeds = MakeSeedSet({{0.375, 0.375, 0.375},
                                           {0.625, 0.625, 0.625},
                                           {0.125, 0.875, 0.125},
                                           {0.875, 0.125, 0.875},
                                           {0.6, 0.6, 0.6}},
                                          {0.0, 0.2, 0.0, 0.05, 0.0});
    ms::TessellationOptions tess;
    tess.metric = ms::MetricType::LAGUERRE;
    ms::ConnectedLabelParameters params;
    RunResult run;
    ExpectMatchesReference(mesh, seeds, tess, params, &run);
    EXPECT_EQ(run.report.seeds_without_cell, 1);
    EXPECT_EQ(run.report.unresolved_elements, 0);
    const auto in = CountLabels(run.input, 5);
    const auto out = CountLabels(run.output, 5);
    EXPECT_GT(in[0], 0);
    EXPECT_EQ(out[0], in[0]) << "seed 0 must keep its far-side elements";
    EXPECT_EQ(in[4], 0);
}

// The reviewer's corner case under plain Voronoi: unit hexes on [-2,2]^3, seed s = (0.05)^3 in the
// corner of its element A = [0,1]^3 and seed t = (1.2, 0.5, 0.5). A's centroid (0.5)^3 is nearer
// to t (d^2 = 0.49) than to s (d^2 = 3 x 0.45^2 = 0.6075), so t labels A, while s labels the
// elements on the far side (e.g. centroid (-0.5)^3: d^2 = 0.9075 to s, 4.89 to t). s must be
// anchored to its nearest own element and keep every one of them: nothing is relabeled.
TEST(ConnectedLabeler, CornerSeedKeepsFarSideElements) {
    mfem::Mesh mesh = BuildGrid(4, 4, 4, 1.0, [](int, int, int) { return true; });
    for (int v = 0; v < mesh.GetNV(); ++v) {
        double* x = mesh.GetVertex(v);
        for (int d = 0; d < 3; ++d) {
            x[d] -= 2.0;
        }
    }
    const ms::SeedSet seeds = MakeSeedSet({{0.05, 0.05, 0.05}, {1.2, 0.5, 0.5}});
    ms::TessellationOptions tess;
    ms::ConnectedLabelParameters params;
    RunResult run;
    ExpectMatchesReference(mesh, seeds, tess, params, &run);
    const auto in = CountLabels(run.input, 2);
    EXPECT_GT(in[0], 0);
    EXPECT_EQ(run.output, run.input);
    EXPECT_EQ(run.report.relabeled_elements, 0);
    EXPECT_EQ(run.report.seeds_without_cell, 0);
    EXPECT_EQ(run.report.unresolved_elements, 0);
}

// A seed in the slot of the U (outside the material) is anchored to the element with the nearest
// centroid, the same on every partition (SeedAnchorElementKeys against the reference anchors, and
// the full labeling against the reference). Seeds inside the material are anchored to the element
// containing them.
TEST(ConnectedLabeler, OutsideSeedSnapsToNearestElement) {
    mfem::Mesh mesh = LoadUShape();
    const ms::SeedSet seeds = MakeSeedSet(
        {{1.5, 1.6, 0.45}, {0.6, 1.7, 0.45}, {2.4, 1.2, 0.45}, {1.55, 0.3, 0.45}});
    ms::TessellationOptions tess;
    ms::ConnectedLabelParameters params;
    const RefResult ref = ExpectMatchesReference(mesh, seeds, tess, params, nullptr);

    for (const bool empty_last : {false, true}) {
        Distributed d = empty_last ? DistributeWithEmptyLastRank(mesh, MPI_COMM_WORLD)
                                   : Distribute(mesh, MPI_COMM_WORLD);
        ms::ElementSamples c;
        ms::ComputeElementSamples(*d.pmesh, -1, c);
        const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(*d.pmesh);
        const std::vector<std::uint64_t> anchors =
            ms::SeedAnchorElementKeys(*d.pmesh, MPI_COMM_WORLD, seeds, keys, c.cx, c.cy, c.cz);
        EXPECT_EQ(anchors, ref.anchor_key);

        // Label-aware anchors from the Euclidean labels
        mfem::Array<int> labels;
        ms::LabelElementsWithSeeds(*d.pmesh, seeds, tess, labels);
        const std::vector<std::uint64_t> connected = ms::ConnectedAnchorKeys(
            *d.pmesh, MPI_COMM_WORLD, seeds, labels, keys, c.cx, c.cy, c.cz);
        EXPECT_EQ(connected, ref.connected_anchor_key);
    }
}

// ------------------------------------------------------------------------------------------------
// Shared seed element locator (seeding/seed_element_locator)
// ------------------------------------------------------------------------------------------------

namespace {

/** Brute-force smallest key of an element whose vertex box contains p (serial, axis-aligned). */
std::uint64_t BruteContaining(const mfem::Mesh& mesh, const std::vector<std::uint64_t>& keys,
                              const double p[3]) {
    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mfem::Array<int> v;
        mesh.GetElementVertices(e, v);
        double lo[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
        double hi[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
        for (int i = 0; i < v.Size(); ++i) {
            const double* c = mesh.GetVertex(v[i]);
            for (int d = 0; d < 3; ++d) {
                lo[d] = std::min(lo[d], c[d]);
                hi[d] = std::max(hi[d], c[d]);
            }
        }
        bool inside = true;
        for (int d = 0; d < 3; ++d) {
            inside = inside && p[d] >= lo[d] && p[d] <= hi[d];
        }
        if (inside) {
            best = std::min(best, keys[static_cast<std::size_t>(e)]);
        }
    }
    return best;
}

} // namespace

// The three locator functions against brute force on the serial mesh, on a METIS partition and on
// a round-robin partition whose last rank owns no element (that rank contributes "nothing found"
// to every reduction). The seeds include a point on a face shared by two elements and one on an
// edge shared by four (the smallest key must win), points outside the mesh, and points at exactly
// equal distance from several vertices (e.g. (0.125, 0.125, -0.5) is equidistant from the four
// vertices of the bottom face of a corner element, with exactly representable coordinates), so
// the (distance, key) tie-break is exercised.
TEST(SeedElementLocator, MatchesBruteForceWithTiesAndEmptyRank) {
    mfem::Mesh mesh = BuildGrid(4, 4, 4, 0.25, [](int, int, int) { return true; });
    const ms::SeedSet seeds = MakeSeedSet({{0.3, 0.6, 0.7},     // interior
                                           {0.5, 0.33, 0.61},   // on a shared face (x = 0.5)
                                           {0.25, 0.75, 0.4},   // on a shared edge
                                           {0.125, 0.125, -0.5}, // outside, vertex tie
                                           {0.5, 0.5, 1.3},     // outside, centroid tie
                                           {-0.2, 1.4, 0.55}}); // outside
    const int n = static_cast<int>(seeds.Size());
    const std::vector<std::uint64_t> serial_keys = ms::ComputeElementKeys(mesh);
    ms::ElementSamples sc;
    ms::ComputeElementSamples(mesh, -1, sc);
    const double* cx = sc.cx.HostRead();
    const double* cy = sc.cy.HostRead();
    const double* cz = sc.cz.HostRead();

    // References
    std::vector<std::uint64_t> ref_contain;
    std::vector<std::uint64_t> ref_centroid_key;
    std::vector<double> ref_centroid_d2;
    std::vector<std::uint64_t> ref_vertex_key;
    std::vector<double> ref_vertex_d2;
    std::vector<std::array<double, 3>> ref_vertex_pos;
    int centroid_ties = 0;
    int vertex_ties = 0;
    std::vector<int> all_ids;
    for (int s = 0; s < n; ++s) {
        const auto us = static_cast<std::size_t>(s);
        const double p[3] = {seeds.x[us], seeds.y[us], seeds.z[us]};
        all_ids.push_back(s);
        ref_contain.push_back(BruteContaining(mesh, serial_keys, p));

        double best = DBL_MAX;
        std::uint64_t key = std::numeric_limits<std::uint64_t>::max();
        int at_best = 0;
        for (int e = 0; e < mesh.GetNE(); ++e) {
            const double dx = cx[e] - p[0];
            const double dy = cy[e] - p[1];
            const double dz = cz[e] - p[2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            const std::uint64_t k = serial_keys[static_cast<std::size_t>(e)];
            if (d2 < best) {
                at_best = 1;
            } else if (d2 == best) {
                ++at_best;
            }
            if (d2 < best || (d2 == best && k < key)) {
                best = d2;
                key = k;
            }
        }
        centroid_ties += (at_best > 1) ? 1 : 0;
        ref_centroid_key.push_back(key);
        ref_centroid_d2.push_back(best);

        best = DBL_MAX;
        key = std::numeric_limits<std::uint64_t>::max();
        std::array<double, 3> pos = {0.0, 0.0, 0.0};
        at_best = 0;
        for (int v = 0; v < mesh.GetNV(); ++v) {
            const double* x = mesh.GetVertex(v);
            const double dx = x[0] - p[0];
            const double dy = x[1] - p[1];
            const double dz = x[2] - p[2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            const std::uint64_t k = ms::VertexKey(x[0], x[1], x[2]);
            if (d2 < best) {
                at_best = 1;
            } else if (d2 == best) {
                ++at_best;
            }
            if (d2 < best || (d2 == best && k < key)) {
                best = d2;
                key = k;
                pos = {x[0], x[1], x[2]};
            }
        }
        vertex_ties += (at_best > 1) ? 1 : 0;
        ref_vertex_key.push_back(key);
        ref_vertex_d2.push_back(best);
        ref_vertex_pos.push_back(pos);
    }
    EXPECT_GE(vertex_ties, 1) << "test setup: some seed should be equidistant from vertices";
    EXPECT_GE(centroid_ties, 1) << "test setup: some seed should be equidistant from centroids";
    EXPECT_NE(ref_contain[1], std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(ref_contain[3], std::numeric_limits<std::uint64_t>::max());

    for (const bool empty_last : {false, true}) {
        Distributed d = empty_last ? DistributeWithEmptyLastRank(mesh, MPI_COMM_WORLD)
                                   : Distribute(mesh, MPI_COMM_WORLD);
        EXPECT_EQ(ms::ContainingElementKeys(*d.pmesh, MPI_COMM_WORLD, seeds), ref_contain);

        const ms::NearestKeys nc =
            ms::NearestElementCentroidKeys(*d.pmesh, MPI_COMM_WORLD, seeds, all_ids);
        EXPECT_EQ(nc.key, ref_centroid_key);
        EXPECT_EQ(nc.distance_squared, ref_centroid_d2);

        const ms::NearestVertexResult nv =
            ms::NearestVertex(*d.pmesh, MPI_COMM_WORLD, seeds, all_ids);
        EXPECT_EQ(nv.key, ref_vertex_key);
        EXPECT_EQ(nv.distance_squared, ref_vertex_d2);
        EXPECT_EQ(nv.position, ref_vertex_pos);

        // A sub-list in another order gives the matching entries
        const std::vector<int> some = {5, 3};
        const ms::NearestKeys part =
            ms::NearestElementCentroidKeys(*d.pmesh, MPI_COMM_WORLD, seeds, some);
        EXPECT_EQ(part.key[0], ref_centroid_key[5]);
        EXPECT_EQ(part.key[1], ref_centroid_key[3]);
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::printf("[ locator ] seeds with exact centroid ties: %d, exact vertex ties: %d\n",
                    centroid_ties, vertex_ties);
    }
}

// Vote sampling: orphans are relabeled by the volume-weighted vote of their voting points, every
// point searched with the element's exclusion list. The reference repeats the vote by brute force
// on the same points; labels and counts must agree on every partition.
TEST(ConnectedLabeler, VoteSamplingMatchesReference) {
    mfem::Mesh mesh = LoadUShape();
    const ms::SeedSet seeds = MakeSeedSet(
        {{0.9, 1.9, 0.45}, {0.9, 1.4, 0.45}, {1.55, 0.2, 0.45}, {2.7, 0.6, 0.3}});
    ms::TessellationOptions tess;
    ms::ConnectedLabelParameters params;
    params.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    params.vote_levels = 1;
    RunResult run;
    ExpectMatchesReference(mesh, seeds, tess, params, &run);
    EXPECT_GE(run.report.rounds, 1);
    EXPECT_EQ(run.report.unresolved_elements, 0);
}

// Invalid input is refused on every rank together, and valid input still works afterwards.
TEST(ConnectedLabeler, InvalidInputThrowsOnAllRanks) {
    mfem::Mesh mesh = LoadUShape();
    Distributed d = Distribute(mesh, MPI_COMM_WORLD);
    const ms::MeshAdjacency adjacency(*d.pmesh);
    const ms::SeedSet seeds = MakeSeedSet({{0.5, 0.5, 0.45}, {2.5, 0.5, 0.45}});
    ms::TessellationOptions tess;
    ms::ConnectedLabelParameters params;
    const int ne = d.pmesh->GetNE();

    // A label outside [0, N) on rank 0 only
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mfem::Array<int> labels(ne);
    labels = 0;
    if (rank == 0 && ne > 0) {
        labels[0] = 7;
    }
    EXPECT_THROW(ms::ConstrainToConnectedCells(*d.pmesh, MPI_COMM_WORLD, adjacency, seeds, tess,
                                               labels, params),
                 std::invalid_argument);

    params.max_rounds = -1;
    labels = 0;
    EXPECT_THROW(ms::ConstrainToConnectedCells(*d.pmesh, MPI_COMM_WORLD, adjacency, seeds, tess,
                                               labels, params),
                 std::invalid_argument);

    params.max_rounds = ms::CONNECTED_MAX_ROUNDS_LIMIT + 1;
    EXPECT_THROW(ms::ConstrainToConnectedCells(*d.pmesh, MPI_COMM_WORLD, adjacency, seeds, tess,
                                               labels, params),
                 std::invalid_argument);

    params.max_rounds = 4;
    ms::LabelElementsWithSeeds(*d.pmesh, seeds, tess, labels);
    const ms::ConnectedLabelReport report = ms::ConstrainToConnectedCells(
        *d.pmesh, MPI_COMM_WORLD, adjacency, seeds, tess, labels, params);
    EXPECT_EQ(report.unresolved_elements, 0);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
