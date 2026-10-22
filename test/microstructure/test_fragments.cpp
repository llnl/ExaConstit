/**
 * @file test_fragments.cpp
 * @brief Tests of the face adjacency of a distributed mesh (mesh/mesh_adjacency), the
 * partition-independent element key (mesh/element_key) and fragment cleanup
 * (labeling/fragment_cleanup).
 *
 * Every test runs on all ranks: MeshAdjacency and CleanupFragments are collective (every rank of
 * the communicator must call them, because the ranks exchange data inside). Rank-count
 * independence is checked inside each run: the distributed result over MPI_COMM_WORLD is gathered
 * into serial element order and compared with the result every rank computes alone on the whole
 * serial mesh (MPI_COMM_SELF), which is exactly the np = 1 answer. Running the executable at
 * np = 1..4 (test_fragments.cmake) therefore compares np = 2, 3, 4 against np = 1.
 *
 * Meshes are built cell by cell on a regular grid (cell size h), either as hexes or as the 6-tet
 * Kuhn split of every cell, with some cells left out to make non-convex or disconnected parts.
 * Element order is the cell order (x fastest, then y, then z), 6 consecutive tets per cell.
 *
 * A *fragment* is a maximal face-connected set of elements of one grain; a healthy grain has one.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever in that
 * collective call.
 */
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/util/counter_rng.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <set>
#include <limits>
#include <memory>
#include <mpi.h>
#include <utility>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

/** A grid-built mesh and the grid cell (i, j, k) of every element. */
struct GridMesh {
    mfem::Mesh mesh;
    std::vector<std::array<int, 3>> cell;
};

/**
 * Mesh of the cells (i, j, k), 0 <= i < nx, 0 <= j < ny, 0 <= k < nz, for which kept(i, j, k) is
 * true; cell (i, j, k) is [i h, (i+1) h] x [j h, (j+1) h] x [k h, (k+1) h]. Hexes, or the 6-tet
 * Kuhn split (the 6 tets along the cell diagonal from its (0,0,0) to its (1,1,1) corner; the same
 * split in every cell, so neighboring cells share face diagonals and the mesh is conforming).
 */
template <typename Kept>
GridMesh BuildGrid(int nx, int ny, int nz, double h, bool tets, Kept kept) {
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    int num_cells = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                num_cells += kept(i, j, k) ? 1 : 0;
            }
        }
    }
    GridMesh grid{mfem::Mesh(3, (nx + 1) * (ny + 1) * (nz + 1),
                             tets ? 6 * num_cells : num_cells, 0, 3),
                  {}};
    mfem::Mesh& mesh = grid.mesh;
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
                // Corner c = bx + 2 by + 4 bz of the cell
                int c[8];
                for (int b = 0; b < 8; b++) {
                    c[b] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                }
                if (!tets) {
                    const int hex[8] = {c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]};
                    mesh.AddHex(hex);
                    grid.cell.push_back({i, j, k});
                } else {
                    const int orders[6][3] = {{1, 2, 4}, {1, 4, 2}, {2, 1, 4},
                                              {2, 4, 1}, {4, 1, 2}, {4, 2, 1}};
                    for (const auto& o : orders) {
                        const int tet[4] = {c[0], c[o[0]], c[o[0] + o[1]], c[7]};
                        mesh.AddTet(tet);
                        grid.cell.push_back({i, j, k});
                    }
                }
            }
        }
    }
    if (tets) {
        mesh.FinalizeTetMesh(1, 1, true);
    } else {
        mesh.FinalizeHexMesh(1, 1, true);
    }
    mesh.RemoveUnusedVertices();
    return grid;
}

/** Full box of nx x ny x nz cells. */
GridMesh Box(int nx, int ny, int nz, double h, bool tets) {
    return BuildGrid(nx, ny, nz, h, tets, [](int, int, int) { return true; });
}

/**
 * U-shaped part with n cells per unit length: [0,3] x [0,2] x [0,1] without the notch
 * [1,2] x [1,2] x [0,1] (an open slot between the two arms x < 1 and x > 2).
 */
GridMesh UShape(int n, bool tets) {
    return BuildGrid(3 * n, 2 * n, n, 1.0 / n, tets, [n](int i, int j, int) {
        return !(i >= n && i < 2 * n && j >= n);
    });
}

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** Partitions `serial` over `comm` (collective), keeping local -> serial element ids. */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/**
 * A partition of `serial` in which the last rank owns no element (when there are at least 2
 * ranks): elements are dealt round-robin to ranks 0 .. nranks-2, so the partitions are also
 * scattered (many shared faces). Collective.
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

/** Cleanup result in serial element order, identical on every rank. */
struct SerialResult {
    std::vector<int> grain;
    int num_grains = 0;
    std::vector<int> parent;
    std::vector<std::int64_t> source;
    ms::FragmentCleanupReport report;
};

/**
 * Runs CleanupFragments on `serial` distributed over `comm` with the serial labels
 * `serial_grain`, and gathers the result into serial order on every rank. Collective over comm.
 */
SerialResult RunCleanup(mfem::Mesh& serial, MPI_Comm comm, const std::vector<int>& serial_grain,
                        int num_grains, const std::vector<std::int64_t>& source,
                        const ms::FragmentCleanupParameters& params, bool empty_last_rank) {
    Distributed d = empty_last_rank ? DistributeWithEmptyLastRank(serial, comm)
                                    : Distribute(serial, comm);
    const int ne = d.pmesh->GetNE();
    mfem::Array<int> local(ne);
    for (int e = 0; e < ne; e++) {
        local[e] = serial_grain[static_cast<std::size_t>(d.order.local_to_serial[e])];
    }
    const ms::MeshAdjacency adjacency(*d.pmesh);
    const ms::FragmentCleanupResult result =
        ms::CleanupFragments(adjacency, local, num_grains, source, params);

    SerialResult out;
    out.grain = ms::GatherToSerial(comm, d.order, result.element_grain, 0);
    out.grain.resize(static_cast<std::size_t>(serial.GetNE()));
    MPI_Bcast(out.grain.data(), serial.GetNE(), MPI_INT, 0, comm);
    out.num_grains = result.num_grains;
    out.parent = result.grain_parent;
    out.source = result.grain_source_id;
    out.report = result.report;
    return out;
}

/**
 * RunCleanup over MPI_COMM_WORLD (METIS partition, and again with a round-robin partition whose
 * last rank is empty) and over MPI_COMM_SELF (the np = 1 reference); all three must agree exactly
 * in labels, grain tables and report counts. Returns the MPI_COMM_WORLD result.
 */
SerialResult RunAndCompare(mfem::Mesh& serial, const std::vector<int>& serial_grain,
                           int num_grains, const std::vector<std::int64_t>& source,
                           const ms::FragmentCleanupParameters& params) {
    const SerialResult world =
        RunCleanup(serial, MPI_COMM_WORLD, serial_grain, num_grains, source, params, false);
    const SerialResult scattered =
        RunCleanup(serial, MPI_COMM_WORLD, serial_grain, num_grains, source, params, true);
    const SerialResult self =
        RunCleanup(serial, MPI_COMM_SELF, serial_grain, num_grains, source, params, false);
    for (const SerialResult* other : {&scattered, &self}) {
        EXPECT_EQ(world.grain, other->grain);
        EXPECT_EQ(world.num_grains, other->num_grains);
        EXPECT_EQ(world.parent, other->parent);
        EXPECT_EQ(world.source, other->source);
        EXPECT_EQ(world.report.fragments_found, other->report.fragments_found);
        EXPECT_EQ(world.report.fragments_split, other->report.fragments_split);
        EXPECT_EQ(world.report.fragments_merged, other->report.fragments_merged);
        EXPECT_EQ(world.report.small_grains_merged, other->report.small_grains_merged);
        EXPECT_EQ(world.report.small_grains_kept, other->report.small_grains_kept);
        EXPECT_EQ(world.report.merge_rounds, other->report.merge_rounds);
        EXPECT_EQ(world.report.small_grain_rounds, other->report.small_grain_rounds);
    }
    return world;
}

/**
 * Independent check on the serial mesh: sizes of the face-connected components of every label,
 * by breadth-first search over the faces of the serial mfem::Mesh (Mesh::GetFaceElements gives
 * both elements of an interior face). Result: label -> sorted component sizes.
 */
std::map<int, std::vector<int>> SerialComponentSizes(const mfem::Mesh& mesh,
                                                     const std::vector<int>& label) {
    const int ne = mesh.GetNE();
    std::vector<std::vector<int>> nbrs(static_cast<std::size_t>(ne));
    for (int f = 0; f < mesh.GetNumFaces(); f++) {
        int e1 = -1;
        int e2 = -1;
        mesh.GetFaceElements(f, &e1, &e2);
        if (e1 >= 0 && e2 >= 0) {
            nbrs[static_cast<std::size_t>(e1)].push_back(e2);
            nbrs[static_cast<std::size_t>(e2)].push_back(e1);
        }
    }
    std::map<int, std::vector<int>> sizes;
    std::vector<char> seen(static_cast<std::size_t>(ne), 0);
    for (int start = 0; start < ne; start++) {
        if (seen[static_cast<std::size_t>(start)]) {
            continue;
        }
        const int l = label[static_cast<std::size_t>(start)];
        std::vector<int> queue = {start};
        seen[static_cast<std::size_t>(start)] = 1;
        for (std::size_t q = 0; q < queue.size(); q++) {
            for (const int n : nbrs[static_cast<std::size_t>(queue[q])]) {
                if (!seen[static_cast<std::size_t>(n)] && label[static_cast<std::size_t>(n)] == l) {
                    seen[static_cast<std::size_t>(n)] = 1;
                    queue.push_back(n);
                }
            }
        }
        sizes[l].push_back(static_cast<int>(queue.size()));
    }
    for (auto& item : sizes) {
        std::sort(item.second.begin(), item.second.end());
    }
    return sizes;
}

/** True if every label of `label` forms exactly one face-connected piece. */
bool AllGrainsConnected(const mfem::Mesh& mesh, const std::vector<int>& label) {
    for (const auto& item : SerialComponentSizes(mesh, label)) {
        if (item.second.size() != 1) {
            return false;
        }
    }
    return true;
}

/** Per-label element counts of a serial labeling (labels 1..num_grains, index g-1). */
std::vector<int> GrainSizes(const std::vector<int>& label, int num_grains) {
    std::vector<int> sizes(static_cast<std::size_t>(num_grains), 0);
    for (const int l : label) {
        sizes[static_cast<std::size_t>(l - 1)] += 1;
    }
    return sizes;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Element key
// ---------------------------------------------------------------------------------------------

// The key must name the same element on every partition: each local element of the ParMesh has
// the key of its serial element (vertex coordinates are copied bit for bit into the partition),
// all keys of a mesh are distinct (no two elements share a vertex set; the 6 Kuhn tets of a cell
// share their smallest vertex, so a key from one vertex only would collide), the local vertex
// order does not matter, and a 1-ulp change of one coordinate changes the key.
TEST(ElementKey, PartitionIndependentUniqueAndOrderFree) {
    for (const bool tets : {false, true}) {
        GridMesh grid = UShape(2, tets);
        const std::vector<std::uint64_t> serial_keys = ms::ComputeElementKeys(grid.mesh);
        std::vector<std::uint64_t> sorted = serial_keys;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end());

        Distributed d = Distribute(grid.mesh, MPI_COMM_WORLD);
        const std::vector<std::uint64_t> local_keys = ms::ComputeElementKeys(*d.pmesh);
        for (int e = 0; e < d.pmesh->GetNE(); e++) {
            EXPECT_EQ(local_keys[static_cast<std::size_t>(e)],
                      serial_keys[static_cast<std::size_t>(d.order.local_to_serial[e])]);
        }
    }

    // One tet, stored with two different vertex orders (same vertex set): same key
    const double xyz[4][3] = {{0.1, 0.2, 0.3}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    auto single_tet = [&](const int order[4], double nudge) {
        mfem::Mesh mesh(3, 4, 1, 0, 3);
        for (int v = 0; v < 4; v++) {
            mesh.AddVertex(xyz[v][0], xyz[v][1], xyz[v][2] + (v == 3 ? nudge : 0.0));
        }
        mesh.AddTet(order);
        mesh.FinalizeTetMesh(1, 1, false);
        return ms::ElementKey(mesh, 0);
    };
    const int order_a[4] = {0, 1, 2, 3};
    const int order_b[4] = {2, 0, 3, 1};
    EXPECT_EQ(single_tet(order_a, 0.0), single_tet(order_b, 0.0));
    // z of vertex 3 moved by one unit in the last place (1 ulp of 1.0 is 2^-52)
    EXPECT_NE(single_tet(order_a, 0.0), single_tet(order_a, std::ldexp(1.0, -52)));
}

// ---------------------------------------------------------------------------------------------
// Mesh adjacency
// ---------------------------------------------------------------------------------------------

// The distributed face graph must be the serial one: for every local element, the keys of its
// neighbors (local elements, or ghost elements on other ranks whose keys arrive through
// ExchangeKeys) and the shared face areas must equal what a single rank finds on the whole mesh
// (MPI_COMM_SELF), and every interior face appears exactly twice globally. ExchangeLabels and
// ExchangeValues are checked on the same data: a label derived from the key, and a double
// derived from it, must arrive unchanged on the ghost side. Covers hexes and tets, the METIS
// partition and a scattered partition with an empty last rank.
TEST(MeshAdjacency, MatchesSerialFaceGraph) {
    for (const bool tets : {false, true}) {
        GridMesh grid = UShape(2, tets);
        // Reference: one rank, whole mesh; key -> sorted (neighbor key, area)
        Distributed ref = Distribute(grid.mesh, MPI_COMM_SELF);
        const ms::MeshAdjacency ref_adj(*ref.pmesh);
        EXPECT_EQ(ref_adj.NumGhostElements(), 0);
        const std::vector<std::uint64_t> ref_keys = ms::ComputeElementKeys(*ref.pmesh);
        std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, double>>> expected;
        for (int e = 0; e < ref_adj.NumElements(); e++) {
            auto& row = expected[ref_keys[static_cast<std::size_t>(e)]];
            for (int k = ref_adj.Offsets()[e]; k < ref_adj.Offsets()[e + 1]; k++) {
                row.emplace_back(ref_keys[static_cast<std::size_t>(ref_adj.Neighbors()[k])],
                                 ref_adj.FaceAreas()[k]);
            }
            std::sort(row.begin(), row.end());
        }
        const long long ref_entries = ref_adj.Offsets()[ref_adj.NumElements()];

        for (const bool scattered : {false, true}) {
            Distributed d = scattered ? DistributeWithEmptyLastRank(grid.mesh, MPI_COMM_WORLD)
                                      : Distribute(grid.mesh, MPI_COMM_WORLD);
            const ms::MeshAdjacency adj(*d.pmesh);
            const int ne = adj.NumElements();
            const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(*d.pmesh);
            std::vector<std::uint64_t> ghost_keys;
            adj.ExchangeKeys(keys, ghost_keys);
            EXPECT_EQ(static_cast<int>(ghost_keys.size()), adj.NumGhostElements());

            // Labels and doubles derived from the key must arrive unchanged
            mfem::Array<int> labels(ne);
            mfem::Vector values(ne);
            for (int e = 0; e < ne; e++) {
                labels[e] = static_cast<int>(keys[static_cast<std::size_t>(e)] % 1000003ULL);
                values[e] = 0.5 + static_cast<double>(keys[static_cast<std::size_t>(e)] >> 20);
            }
            mfem::Array<int> ghost_labels;
            mfem::Vector ghost_values;
            adj.ExchangeLabels(labels, ghost_labels);
            adj.ExchangeValues(values, ghost_values);
            for (int g = 0; g < adj.NumGhostElements(); g++) {
                const std::uint64_t key = ghost_keys[static_cast<std::size_t>(g)];
                EXPECT_EQ(ghost_labels[g], static_cast<int>(key % 1000003ULL));
                EXPECT_EQ(ghost_values[g], 0.5 + static_cast<double>(key >> 20));
            }

            // Every row matches the reference (areas to round-off: the same face can be
            // integrated from a different vertex order on another rank)
            for (int e = 0; e < ne; e++) {
                std::vector<std::pair<std::uint64_t, double>> row;
                for (int k = adj.Offsets()[e]; k < adj.Offsets()[e + 1]; k++) {
                    const int n = adj.Neighbors()[k];
                    const std::uint64_t nkey =
                        adj.IsGhost(n) ? ghost_keys[static_cast<std::size_t>(n - ne)]
                                       : keys[static_cast<std::size_t>(n)];
                    row.emplace_back(nkey, adj.FaceAreas()[k]);
                }
                std::sort(row.begin(), row.end());
                const auto& want = expected[keys[static_cast<std::size_t>(e)]];
                EXPECT_EQ(row.size(), want.size());
                for (std::size_t i = 0; i < std::min(row.size(), want.size()); i++) {
                    EXPECT_EQ(row[i].first, want[i].first);
                    EXPECT_NEAR(row[i].second, want[i].second, 1e-14);
                }
            }
            long long entries = adj.Offsets()[ne];
            long long total = 0;
            MPI_Allreduce(&entries, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            EXPECT_EQ(total, ref_entries);
        }
    }
}

// Analytic face graph of a hex box: in an nx x ny x nz box of cubes of side h, element (i,j,k)
// has one neighbor per direction that stays inside the box, every shared face has area h^2, and
// the number of adjacency entries (2 per interior face) is
// 2 [(nx-1) ny nz + nx (ny-1) nz + nx ny (nz-1)]. The same holds with order-2 Nodes
// (SetCurvature(2) on straight cubes: the face areas then come from the curved face
// transformation, which must still give exactly h^2 up to round-off).
TEST(MeshAdjacency, HexBoxCountsAndAreas) {
    const int nx = 5;
    const int ny = 3;
    const int nz = 4;
    const double h = 0.25;
    for (const bool curved : {false, true}) {
        GridMesh grid = Box(nx, ny, nz, h, false);
        if (curved) {
            grid.mesh.SetCurvature(2);
        }
        Distributed d = Distribute(grid.mesh, MPI_COMM_WORLD);
        const ms::MeshAdjacency adj(*d.pmesh);
        for (int e = 0; e < adj.NumElements(); e++) {
            const auto& c = grid.cell[static_cast<std::size_t>(d.order.local_to_serial[e])];
            const int expected = (c[0] > 0) + (c[0] < nx - 1) + (c[1] > 0) + (c[1] < ny - 1) +
                                 (c[2] > 0) + (c[2] < nz - 1);
            EXPECT_EQ(adj.Offsets()[e + 1] - adj.Offsets()[e], expected);
            for (int k = adj.Offsets()[e]; k < adj.Offsets()[e + 1]; k++) {
                EXPECT_NEAR(adj.FaceAreas()[k], h * h, 1e-15);
            }
        }
        long long entries = adj.Offsets()[adj.NumElements()];
        long long total = 0;
        MPI_Allreduce(&entries, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(total, 2LL * ((nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)));
    }
}

// ---------------------------------------------------------------------------------------------
// Fragment cleanup
// ---------------------------------------------------------------------------------------------

// A constructed disconnected grain: a row of 8 x 2 x 2 unit cubes, grain 1 in cells i = 0-1
// (8 elements, the "left block") and i = 4-7 (16 elements, the "right block"), grain 2 in
// i = 2-3 (8 elements) between them. Grain 1 has two fragments; the right block is the larger.
// Expected results, worked out by hand for every policy:
//  - SPLIT, keep largest: the left block becomes grain 3, source id derived from (10, 1);
//  - MERGE, keep largest: the left block touches only grain 2, so it joins grain 2;
//  - SPLIT / MERGE, keep the anchor (an element of the left block): the right block is the
//    extra fragment and becomes grain 3, or joins grain 2 (its only neighbor);
//  - SPLIT then min_grain_elements = 9: grains 2 and 3 (8 elements each) are too small; grain 3
//    touches only grain 2, which is itself moving, so it waits; grain 2 joins grain 1 (its only
//    stable neighbor); next round grain 3 now touches grain 1 and joins it. One grain is left;
//  - NONE: labels unchanged, one extra fragment reported.
TEST(FragmentCleanup, ConstructedDisconnectedGrain) {
    GridMesh grid = Box(8, 2, 2, 1.0, false);
    const int ne = grid.mesh.GetNE();
    std::vector<int> label(static_cast<std::size_t>(ne));
    std::vector<int> left_block_elements;
    for (int e = 0; e < ne; e++) {
        const int i = grid.cell[static_cast<std::size_t>(e)][0];
        label[static_cast<std::size_t>(e)] = (i >= 2 && i < 4) ? 2 : 1;
        if (i < 2) {
            left_block_elements.push_back(e);
        }
    }
    const std::vector<std::int64_t> source = {10, 20};
    auto expect_labels = [&](const SerialResult& r, auto expected_of_i) {
        for (int e = 0; e < ne; e++) {
            EXPECT_EQ(r.grain[static_cast<std::size_t>(e)],
                      expected_of_i(grid.cell[static_cast<std::size_t>(e)][0]));
        }
    };

    ms::FragmentCleanupParameters split;
    split.policy = ms::FragmentPolicy::SPLIT;
    const SerialResult a = RunAndCompare(grid.mesh, label, 2, source, split);
    expect_labels(a, [](int i) { return i < 2 ? 3 : (i < 4 ? 2 : 1); });
    EXPECT_EQ(a.num_grains, 3);
    EXPECT_EQ(a.parent, (std::vector<int>{1, 2, 1}));
    EXPECT_EQ(a.source,
              (std::vector<std::int64_t>{10, 20, ms::DeriveFragmentSourceId(10, 1)}));
    EXPECT_LT(a.source[2], 0); // derived ids are negative, never a generated (>= 0) id
    EXPECT_EQ(a.report.fragments_found, 1);
    EXPECT_EQ(a.report.fragments_split, 1);
    EXPECT_EQ(a.report.fragments_merged, 0);

    ms::FragmentCleanupParameters merge;
    merge.policy = ms::FragmentPolicy::MERGE;
    const SerialResult b = RunAndCompare(grid.mesh, label, 2, source, merge);
    expect_labels(b, [](int i) { return i < 4 ? 2 : 1; });
    EXPECT_EQ(b.num_grains, 2);
    EXPECT_EQ(b.source, source);
    EXPECT_EQ(b.report.fragments_merged, 1);
    EXPECT_EQ(b.report.fragments_split, 0);
    EXPECT_EQ(b.report.merge_rounds, 1);

    // Anchor of grain 1 = an element of the left block; grain 2 gets a key matching nothing
    ms::FragmentCleanupParameters anchored = split;
    anchored.keep = ms::KeepRule::SEED_ELEMENT;
    anchored.grain_anchor_key = {ms::ElementKey(grid.mesh, left_block_elements[3]), 0};
    const SerialResult c = RunAndCompare(grid.mesh, label, 2, source, anchored);
    expect_labels(c, [](int i) { return i < 2 ? 1 : (i < 4 ? 2 : 3); });
    EXPECT_EQ(c.parent, (std::vector<int>{1, 2, 1}));

    anchored.policy = ms::FragmentPolicy::MERGE;
    const SerialResult c2 = RunAndCompare(grid.mesh, label, 2, source, anchored);
    expect_labels(c2, [](int i) { return i < 2 ? 1 : 2; });

    ms::FragmentCleanupParameters split_min = split;
    split_min.min_grain_elements = 9;
    const SerialResult e = RunAndCompare(grid.mesh, label, 2, source, split_min);
    expect_labels(e, [](int) { return 1; });
    EXPECT_EQ(e.num_grains, 1);
    EXPECT_EQ(e.parent, (std::vector<int>{1}));
    EXPECT_EQ(e.source, (std::vector<std::int64_t>{10}));
    EXPECT_EQ(e.report.small_grains_merged, 2);
    EXPECT_EQ(e.report.small_grain_rounds, 2);

    ms::FragmentCleanupParameters none;
    none.policy = ms::FragmentPolicy::NONE;
    const SerialResult f = RunAndCompare(grid.mesh, label, 2, source, none);
    EXPECT_EQ(f.grain, label);
    EXPECT_EQ(f.report.fragments_found, 1);
    EXPECT_EQ(f.num_grains, 2);
}

// Simultaneous merges may only use stable neighbors. A row of 8 cubes labeled
// 1 1 3 1 2 4 2 2: cell 3 (grain 1) and cell 4 (grain 2) are extra fragments that touch each
// other. Counting every neighbor, cell 3 has one face with grain 3 and one with grain 2 (tie:
// lowest id, grain 2) and cell 4 one with grain 1 and one with grain 4 (grain 1): the two would
// swap labels and stay extra fragments forever. Only faces to stable (not moving) elements count,
// so cell 3 joins grain 3 and cell 4 joins grain 4 in one round: 1 1 3 3 4 4 2 2.
TEST(FragmentCleanup, SimultaneousMergesUseStableNeighbors) {
    GridMesh grid = Box(8, 1, 1, 1.0, false);
    const std::vector<int> label = {1, 1, 3, 1, 2, 4, 2, 2};
    ms::FragmentCleanupParameters merge;
    merge.policy = ms::FragmentPolicy::MERGE;
    merge.max_rounds = 50; // a swap would loop; fail fast instead
    const SerialResult r = RunAndCompare(grid.mesh, label, 4, {}, merge);
    EXPECT_EQ(r.grain, (std::vector<int>{1, 1, 3, 3, 4, 4, 2, 2}));
    EXPECT_EQ(r.report.fragments_merged, 2);
    EXPECT_EQ(r.report.merge_rounds, 1);
    EXPECT_EQ(r.source, (std::vector<std::int64_t>{0, 1, 2, 3})); // empty input: id = g - 1
}

// A fragment with no neighboring grain cannot merge. Cells 0, 1, 2 and 4 of a row of 5 (cell 3
// left out, so cell 4 is a separate body) labeled 1 1 2 _ 1: grain 1 has the fragments
// {0, 1} (kept, larger) and {4}, which touches nothing; with MERGE it is split off as grain 3.
TEST(FragmentCleanup, IsolatedFragmentIsSplit) {
    GridMesh grid = BuildGrid(5, 1, 1, 1.0, false, [](int i, int, int) { return i != 3; });
    const std::vector<int> label = {1, 1, 2, 1};
    ms::FragmentCleanupParameters merge;
    merge.policy = ms::FragmentPolicy::MERGE;
    const SerialResult r = RunAndCompare(grid.mesh, label, 2, {7, 8}, merge);
    EXPECT_EQ(r.grain, (std::vector<int>{1, 1, 2, 3}));
    EXPECT_EQ(r.report.fragments_split, 1);
    EXPECT_EQ(r.report.fragments_merged, 0);
    EXPECT_EQ(r.parent, (std::vector<int>{1, 2, 1}));
    EXPECT_EQ(r.source[2], ms::DeriveFragmentSourceId(7, 1));
}

// When every grain is too small, nothing has a stable neighbor; the largest candidate (ties:
// lowest id) is kept so the others can merge into it. Row of 4 cubes labeled 1 2 3 4 with
// min_grain_elements = 10: round 1 keeps grain 1; round 2 grain 2 joins grain 1 (grains 3 and 4
// only touch moving grains); round 3 grain 3; round 4 grain 4. One grain remains, below the
// minimum but kept.
TEST(FragmentCleanup, AllGrainsSmall) {
    GridMesh grid = Box(4, 1, 1, 1.0, false);
    const std::vector<int> label = {1, 2, 3, 4};
    ms::FragmentCleanupParameters params;
    params.policy = ms::FragmentPolicy::NONE;
    params.min_grain_elements = 10;
    const SerialResult r = RunAndCompare(grid.mesh, label, 4, {}, params);
    EXPECT_EQ(r.grain, (std::vector<int>{1, 1, 1, 1}));
    EXPECT_EQ(r.num_grains, 1);
    EXPECT_EQ(r.report.small_grains_merged, 3);
    EXPECT_EQ(r.report.small_grains_kept, 1);
    EXPECT_EQ(r.report.small_grain_rounds, 4);
}

// Islands injected at random. An 8^3 box (h = 1/8) is labeled in 4 slabs along x
// (grain = 1 + i / 2). Random cells with i in {0, 2} and even j, k (chosen with the counter-based
// RNG, so every rank picks the same cells) get grain 4, which lives at i >= 6: each is an island
// cut off from grain 4's slab. No two candidate cells share a face (they differ by 2 in some
// index), so every island is its own fragment. Expected:
//  - SPLIT: one new single-cell grain per island, slabs unchanged;
//  - MERGE: every island goes back to the slab around it (an island at i = 0 touches only slab 1;
//    at i = 2 it touches slab 1 through one face and slab 2 through at least 3), so the labels
//    are exactly the clean slabs;
//  - SPLIT + min_grain_elements just above the island size: the island grains merge back the
//    same way.
// Hexes (1 element per cell) and Kuhn tets (6 face-connected elements per cell).
TEST(FragmentCleanup, RandomIslands) {
    for (const bool tets : {false, true}) {
        GridMesh grid = Box(8, 8, 8, 1.0 / 8, tets);
        const int ne = grid.mesh.GetNE();
        const int per_cell = tets ? 6 : 1;
        std::vector<int> slabs(static_cast<std::size_t>(ne));
        std::vector<int> label(static_cast<std::size_t>(ne));
        int num_islands = 0;
        std::map<std::array<int, 3>, bool> island_cell;
        for (int i : {0, 2}) {
            for (int j = 0; j < 8; j += 2) {
                for (int k = 0; k < 8; k += 2) {
                    const std::uint64_t id = static_cast<std::uint64_t>(i + 8 * (j + 8 * k));
                    const ms::CounterRng rng(1234, ms::RngPurpose::GENERIC, id);
                    const bool island = rng.UniformDouble(0u) < 0.5;
                    island_cell[{i, j, k}] = island;
                    num_islands += island ? 1 : 0;
                }
            }
        }
        for (int e = 0; e < ne; e++) {
            const auto& c = grid.cell[static_cast<std::size_t>(e)];
            slabs[static_cast<std::size_t>(e)] = 1 + c[0] / 2;
            const auto it = island_cell.find(c);
            const bool island = it != island_cell.end() && it->second;
            label[static_cast<std::size_t>(e)] = island ? 4 : slabs[static_cast<std::size_t>(e)];
        }
        EXPECT_GT(num_islands, 0);

        ms::FragmentCleanupParameters split;
        split.policy = ms::FragmentPolicy::SPLIT;
        const SerialResult s = RunAndCompare(grid.mesh, label, 4, {0, 1, 2, 3}, split);
        EXPECT_EQ(s.report.fragments_found, num_islands);
        EXPECT_EQ(s.report.fragments_split, num_islands);
        EXPECT_EQ(s.num_grains, 4 + num_islands);
        const std::vector<int> sizes = GrainSizes(s.grain, s.num_grains);
        for (int g = 5; g <= s.num_grains; g++) {
            EXPECT_EQ(sizes[static_cast<std::size_t>(g - 1)], per_cell);
            EXPECT_EQ(s.parent[static_cast<std::size_t>(g - 1)], 4);
        }
        for (int e = 0; e < ne; e++) {
            if (label[static_cast<std::size_t>(e)] == slabs[static_cast<std::size_t>(e)]) {
                EXPECT_EQ(s.grain[static_cast<std::size_t>(e)], slabs[static_cast<std::size_t>(e)]);
            }
        }
        EXPECT_TRUE(AllGrainsConnected(grid.mesh, s.grain));

        ms::FragmentCleanupParameters merge;
        merge.policy = ms::FragmentPolicy::MERGE;
        const SerialResult m = RunAndCompare(grid.mesh, label, 4, {0, 1, 2, 3}, merge);
        EXPECT_EQ(m.grain, slabs);
        EXPECT_EQ(m.report.fragments_merged, num_islands);
        EXPECT_EQ(m.num_grains, 4);

        ms::FragmentCleanupParameters split_min = split;
        split_min.min_grain_elements = per_cell + 1;
        const SerialResult sm = RunAndCompare(grid.mesh, label, 4, {0, 1, 2, 3}, split_min);
        EXPECT_EQ(sm.grain, slabs);
        EXPECT_EQ(sm.report.small_grains_merged, num_islands);
        EXPECT_EQ(sm.parent, (std::vector<int>{1, 2, 3, 4}));
        EXPECT_EQ(sm.source, (std::vector<std::int64_t>{0, 1, 2, 3}));
    }
}

// A Voronoi cell crossing the gap of a U-shaped part ([0,3] x [0,2] x [0,1] minus the slot
// [1,2] x [1,2] x [0,1], 8 cells per unit). Seeds: A = (1.45, 1.5, 0.5) in the slot (void),
// B = (1.5, 0.75, 0.5) below it, C = (0.25, 1.5, 0.5) in the left arm, D = (2.95, 1.5, 0.5) in
// the right arm. A is closest to the inner edges of both arms above y ~ 1.1 but not to the
// material under the slot (B is closer there), so A's grain has two fragments on either side of
// the slot. Worked out from the bisector planes at the element centroids (x = 0.9375 on the
// left, x = 2.0625 and 2.1875 on the right; y centroids 1.1875 .. 1.9375): the left fragment is
// 1 column x 7 rows x 8 layers = 56 elements, the right one 2 x 7 x 8 = 112.
//  - SPLIT: the right fragment keeps A's grain, the left one (56 elements) becomes a new grain.
//  - MERGE: the left fragment shares 7 x 8 = 56 faces with C's grain and 8 with B's grain, so it
//    joins C's grain.
// After either policy every grain is one connected piece (independent breadth-first search).
TEST(FragmentCleanup, UShapedVoronoiCellAcrossGap) {
    GridMesh grid = UShape(8, false);
    ms::SeedSet seeds;
    seeds.x = {1.45, 1.5, 0.25, 2.95};
    seeds.y = {1.5, 0.75, 1.5, 1.5};
    seeds.z = {0.5, 0.5, 0.5, 0.5};
    seeds.source_id = {0, 1, 2, 3};
    ms::TessellationOptions tessellation;
    const ms::TessellationLabels labels =
        ms::LabelTessellation(grid.mesh, MPI_COMM_SELF, seeds, tessellation);
    ASSERT_EQ(labels.num_grains, 4); // same on every rank: labeled locally on MPI_COMM_SELF
    std::vector<int> label(static_cast<std::size_t>(grid.mesh.GetNE()));
    const int* lg = labels.element_grain.HostRead();
    for (int e = 0; e < grid.mesh.GetNE(); e++) {
        label[static_cast<std::size_t>(e)] = lg[e];
    }
    const int grain_a = labels.seed_grain[0];
    const int grain_c = labels.seed_grain[2];
    const auto before = SerialComponentSizes(grid.mesh, label);
    EXPECT_EQ(before.at(grain_a), (std::vector<int>{56, 112}));
    std::vector<std::int64_t> source(4);
    for (int g = 1; g <= 4; g++) {
        const int seed = labels.grain_seed[static_cast<std::size_t>(g - 1)];
        source[static_cast<std::size_t>(g - 1)] = seeds.source_id[static_cast<std::size_t>(seed)];
    }
    // Left fragment: elements of A's grain in the left arm
    auto in_left_fragment = [&](int e) {
        return label[static_cast<std::size_t>(e)] == grain_a &&
               grid.cell[static_cast<std::size_t>(e)][0] < 8;
    };

    ms::FragmentCleanupParameters split;
    split.policy = ms::FragmentPolicy::SPLIT;
    const SerialResult s = RunAndCompare(grid.mesh, label, 4, source, split);
    EXPECT_EQ(s.report.fragments_found, 1);
    EXPECT_EQ(s.num_grains, 5);
    EXPECT_EQ(GrainSizes(s.grain, s.num_grains)[static_cast<std::size_t>(grain_a - 1)], 112);
    EXPECT_EQ(GrainSizes(s.grain, s.num_grains)[4], 56);
    EXPECT_EQ(s.parent[4], grain_a);
    for (int e = 0; e < grid.mesh.GetNE(); e++) {
        if (in_left_fragment(e)) {
            EXPECT_EQ(s.grain[static_cast<std::size_t>(e)], 5);
        }
    }
    EXPECT_TRUE(AllGrainsConnected(grid.mesh, s.grain));

    ms::FragmentCleanupParameters merge;
    merge.policy = ms::FragmentPolicy::MERGE;
    const SerialResult m = RunAndCompare(grid.mesh, label, 4, source, merge);
    EXPECT_EQ(m.num_grains, 4);
    EXPECT_EQ(m.report.fragments_merged, 1);
    for (int e = 0; e < grid.mesh.GetNE(); e++) {
        if (in_left_fragment(e)) {
            EXPECT_EQ(m.grain[static_cast<std::size_t>(e)], grain_c);
        }
    }
    EXPECT_TRUE(AllGrainsConnected(grid.mesh, m.grain));
}

// Random Voronoi tessellations of the U-shaped part (tets and hexes, 40 seeds in the bounding
// box, many of them in the slot) with every policy: whatever fragments appear, the result must be
// identical at every rank count and partition (RunAndCompare), every grain must end up connected
// (SPLIT, MERGE), every grain must have at least min_grain_elements elements unless it was kept
// for lack of a stable neighbor, and the element count is conserved.
TEST(FragmentCleanup, RandomVoronoiRankInvariance) {
    for (const bool tets : {false, true}) {
        GridMesh grid = UShape(tets ? 3 : 4, tets);
        ms::SeedSet seeds;
        for (int i = 0; i < 40; i++) {
            const ms::CounterRng rng(99, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
            seeds.x.push_back(3.0 * rng.UniformDouble(0u));
            seeds.y.push_back(2.0 * rng.UniformDouble(1u));
            seeds.z.push_back(rng.UniformDouble(2u));
            seeds.source_id.push_back(i);
        }
        ms::TessellationOptions tessellation;
        const ms::TessellationLabels labels =
            ms::LabelTessellation(grid.mesh, MPI_COMM_SELF, seeds, tessellation);
        std::vector<int> label(static_cast<std::size_t>(grid.mesh.GetNE()));
        const int* lg = labels.element_grain.HostRead();
        for (int e = 0; e < grid.mesh.GetNE(); e++) {
            label[static_cast<std::size_t>(e)] = lg[e];
        }
        std::vector<std::int64_t> source;
        for (const int seed : labels.grain_seed) {
            source.push_back(seeds.source_id[static_cast<std::size_t>(seed)]);
        }

        for (const auto policy : {ms::FragmentPolicy::SPLIT, ms::FragmentPolicy::MERGE}) {
            for (const int min_elements : {1, 12}) {
                ms::FragmentCleanupParameters params;
                params.policy = policy;
                params.min_grain_elements = min_elements;
                const SerialResult r =
                    RunAndCompare(grid.mesh, label, labels.num_grains, source, params);
                EXPECT_TRUE(AllGrainsConnected(grid.mesh, r.grain));
                const std::vector<int> sizes = GrainSizes(r.grain, r.num_grains);
                int small = 0;
                long long total = 0;
                for (const int n : sizes) {
                    EXPECT_GT(n, 0);
                    small += (n < min_elements) ? 1 : 0;
                    total += n;
                }
                EXPECT_LE(small, r.report.small_grains_kept);
                EXPECT_EQ(total, grid.mesh.GetNE());
                EXPECT_EQ(static_cast<int>(r.parent.size()), r.num_grains);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Split large / merge tiny (the generator's default policy)
// ---------------------------------------------------------------------------------------------

/**
 * A strip of 41 x 6 x 1 hex cells. Along x: grain 1 on cells i < 20 (100 elements in rows
 * j < 5), grain 2 on i = 20, grain 1 on 21 <= i < 24 (15 elements), grain 2 on i = 24, grain 1
 * on i >= 25 (80 elements); the row j = 5 is grain 2 everywhere, so grain 2 is one piece and
 * grain 1 has three pieces of 100, 80 and 15 elements (195 in total).
 */
std::vector<int> ThreePieceLabels(const GridMesh& grid) {
    std::vector<int> label(grid.cell.size(), 2);
    for (std::size_t e = 0; e < grid.cell.size(); e++) {
        const int i = grid.cell[e][0];
        const int j = grid.cell[e][1];
        const bool grain1 = j < 5 && (i < 20 || (i >= 21 && i < 24) || i >= 25);
        label[e] = grain1 ? 1 : 2;
    }
    return label;
}

// Tiny pieces are classified once, against the grain as labeled. With the defaults (8 elements,
// 10%) the threshold is 0.1 x 195 = 19.5 elements: the 80-piece is large and split off, the
// 15-piece is tiny and merges into grain 2. (Re-classifying after the split would compare the 15
// with 0.1 x 115 = 11.5 and wrongly split it too.) With min_fragment_elements = 20 and no
// fraction the result is the same (15 < 20 <= 80); with both thresholds 0 nothing is tiny and
// both extra pieces are split.
TEST(FragmentCleanup, SplitLargeMergeTinyClassifiesOnce) {
    GridMesh grid = Box(41, 6, 1, 1.0 / 41, false);
    const std::vector<int> label = ThreePieceLabels(grid);
    struct Case {
        int min_elements;
        double fraction;
        std::int64_t split;
        std::int64_t merged;
        int grains;
    };
    for (const Case& c : {Case{8, 0.1, 1, 1, 3}, Case{20, 0.0, 1, 1, 3}, Case{0, 0.0, 2, 0, 4}}) {
        SCOPED_TRACE("min_fragment_elements " + std::to_string(c.min_elements));
        ms::FragmentCleanupParameters params;
        params.policy = ms::FragmentPolicy::SPLIT_LARGE_MERGE_TINY;
        params.min_fragment_elements = c.min_elements;
        params.min_fragment_fraction = c.fraction;
        const SerialResult result = RunAndCompare(grid.mesh, label, 2, {10, 20}, params);
        EXPECT_EQ(result.report.fragments_found, 2);
        EXPECT_EQ(result.report.fragments_split, c.split);
        EXPECT_EQ(result.report.fragments_merged, c.merged);
        EXPECT_EQ(result.num_grains, c.grains);
        EXPECT_TRUE(AllGrainsConnected(grid.mesh, result.grain));
        // The 100-piece keeps grain 1 and its source id
        EXPECT_EQ(result.grain[0], 1);
        EXPECT_EQ(result.source[0], 10);
        // The 15-piece: grain 2 when merged, a new grain otherwise
        for (std::size_t e = 0; e < grid.cell.size(); e++) {
            if (grid.cell[e][0] == 22 && grid.cell[e][1] == 0) {
                EXPECT_EQ(result.grain[e] == 2, c.merged == 1);
            }
        }
    }
}

// Keep-seed under split-large/merge-tiny: the anchor (seed element) of grain 1 lies in the
// 15-piece, which is tiny, so it may not be kept: the 100-piece keeps the grain (and its source
// id, hence its orientation), the 80-piece is split, the 15-piece merges.
TEST(FragmentCleanup, TinyAnchorPieceIsNotKept) {
    GridMesh grid = Box(41, 6, 1, 1.0 / 41, false);
    const std::vector<int> label = ThreePieceLabels(grid);
    ms::FragmentCleanupParameters params;
    params.policy = ms::FragmentPolicy::SPLIT_LARGE_MERGE_TINY;
    params.keep = ms::KeepRule::SEED_ELEMENT;
    const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(grid.mesh);
    std::uint64_t anchor = 0;
    for (std::size_t e = 0; e < grid.cell.size(); e++) {
        if (grid.cell[e][0] == 22 && grid.cell[e][1] == 2) {
            anchor = keys[e];
        }
    }
    params.grain_anchor_key = {anchor, std::numeric_limits<std::uint64_t>::max()};
    const SerialResult result = RunAndCompare(grid.mesh, label, 2, {10, 20}, params);
    EXPECT_EQ(result.report.fragments_split, 1);
    EXPECT_EQ(result.report.fragments_merged, 1);
    EXPECT_EQ(result.grain[0], 1);
    EXPECT_EQ(result.source[0], 10);
    // With SPLIT the anchor rule is unchanged: the anchor piece (15) keeps grain 1
    params.policy = ms::FragmentPolicy::SPLIT;
    const SerialResult split = RunAndCompare(grid.mesh, label, 2, {10, 20}, params);
    for (std::size_t e = 0; e < grid.cell.size(); e++) {
        if (grid.cell[e][0] == 22 && grid.cell[e][1] == 2) {
            EXPECT_EQ(split.grain[e], 1);
        }
    }
    EXPECT_NE(split.grain[0], 1);
}

// No-progress fallback: a separate two-cell body of the mesh (cells (6,0,0) and (7,0,0), apart
// from the 4 x 2 x 1 main block) where one cell belongs to grain 1 and the other to grain 2,
// whose main pieces are in the block. Both cells are extra, tiny fragments that touch only each
// other, so no stable neighbor exists. The fallback merges just one of them (equal size: the
// smaller fragment id) into the other's grain; the joined 2-cell piece then has no neighbor and
// becomes a grain of its own. The body is never split into two 1-element grains. Same under
// MERGE.
TEST(FragmentCleanup, FallbackMergesOnePieceIntoMovingNeighbor) {
    GridMesh grid = BuildGrid(8, 2, 1, 0.25, false, [](int i, int j, int) {
        return i < 4 || (j == 0 && (i == 6 || i == 7));
    });
    std::vector<int> label(grid.cell.size(), 0);
    for (std::size_t e = 0; e < grid.cell.size(); e++) {
        const int i = grid.cell[e][0];
        label[e] = (i < 2 || i == 6) ? 1 : 2;
    }
    for (const auto policy :
         {ms::FragmentPolicy::SPLIT_LARGE_MERGE_TINY, ms::FragmentPolicy::MERGE}) {
        ms::FragmentCleanupParameters params;
        params.policy = policy;
        const SerialResult result = RunAndCompare(grid.mesh, label, 2, {}, params);
        EXPECT_EQ(result.report.fragments_found, 2);
        EXPECT_EQ(result.report.fragments_merged, 1);
        EXPECT_EQ(result.report.fragments_split, 1);
        EXPECT_EQ(result.num_grains, 3);
        int body_label = -1;
        for (std::size_t e = 0; e < grid.cell.size(); e++) {
            if (grid.cell[e][0] >= 6) {
                if (body_label < 0) {
                    body_label = result.grain[e];
                }
                EXPECT_EQ(result.grain[e], body_label);
            }
        }
        EXPECT_EQ(body_label, 3);
    }
}

// ---------------------------------------------------------------------------------------------
// Periodic face pairing
// ---------------------------------------------------------------------------------------------

/** Periodic parameters of the unit cube with all axes periodic. */
ms::PeriodicFaceParameters UnitCubePairing() {
    ms::PeriodicFaceParameters pairing;
    pairing.periodic = {true, true, true};
    return pairing;
}

// A tet mesh (MFEM's 6-tet split of a 3^3 box, the same split in every cell, so opposite faces
// match): every tet has 4 face neighbors when all axes are periodic, 3 x 9 x 2 = 54 face pairs,
// none unmatched. With the METIS partition (np >= 2) the box is cut into blocks, so the faces
// paired across at least one periodic axis belong to different ranks (periodic ghosts), and the
// exchanged keys are those of the paired elements: every pair is seen from both sides with the
// same element keys.
TEST(MeshAdjacency, PeriodicTetMeshAndRemotePairs) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::TETRAHEDRON);
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    const ms::MeshAdjacency adjacency(*d.pmesh, UnitCubePairing());
    EXPECT_EQ(adjacency.PeriodicReport().matched_pairs, 54);
    EXPECT_EQ(adjacency.PeriodicReport().unmatched_faces, 0);
    const int ne = adjacency.NumElements();
    for (int e = 0; e < ne; e++) {
        EXPECT_EQ(adjacency.Offsets()[e + 1] - adjacency.Offsets()[e], 4) << "element " << e;
    }
    long long local_ghosts = adjacency.NumPeriodicGhosts();
    long long total_ghosts = 0;
    MPI_Allreduce(&local_ghosts, &total_ghosts, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    int nranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks > 1) {
        EXPECT_GT(total_ghosts, 0);
    }

    // Symmetry: the multiset of (my key, neighbor key) pairs over all ranks contains every pair
    // in both orders
    const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(*d.pmesh);
    std::vector<std::uint64_t> ghost_keys;
    adjacency.ExchangeKeys(keys, ghost_keys);
    std::vector<std::int64_t> flat;
    for (int e = 0; e < ne; e++) {
        for (int k = adjacency.Offsets()[e]; k < adjacency.Offsets()[e + 1]; k++) {
            const int n = adjacency.Neighbors()[k];
            const std::uint64_t other = adjacency.IsGhost(n)
                                            ? ghost_keys[static_cast<std::size_t>(n - ne)]
                                            : keys[static_cast<std::size_t>(n)];
            std::int64_t a = 0;
            std::int64_t b = 0;
            std::memcpy(&a, &keys[static_cast<std::size_t>(e)], sizeof(a));
            std::memcpy(&b, &other, sizeof(b));
            flat.push_back(a);
            flat.push_back(b);
        }
    }
    int count = static_cast<int>(flat.size());
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    for (int r = 1; r < nranks; r++) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r - 1)] + counts[static_cast<std::size_t>(r - 1)];
    }
    std::vector<std::int64_t> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(flat.data(), count, MPI_INT64_T, all.data(), counts.data(), displs.data(),
                   MPI_INT64_T, MPI_COMM_WORLD);
    std::multiset<std::pair<std::int64_t, std::int64_t>> forward;
    std::multiset<std::pair<std::int64_t, std::int64_t>> backward;
    for (std::size_t k = 0; k + 1 < all.size(); k += 2) {
        forward.insert({all[k], all[k + 1]});
        backward.insert({all[k + 1], all[k]});
    }
    EXPECT_EQ(forward, backward);
    EXPECT_EQ(forward.size(), static_cast<std::size_t>(4 * serial.GetNE()));
}

// Faces that do not match: one vertex inside the x = 1 face of a 3^3 hex box moves by 0.05 along
// y, so 4 faces on that plane have no partner with coinciding vertices. The constructor must
// throw on every rank (never silently leave faces unpaired). A move of 1e-9 (h = 1/3) is within
// the default tolerance (1e-6 x 1/3), but not within relative_tolerance = 1e-10, which throws.
TEST(MeshAdjacency, PeriodicUnmatchedFacesAndTolerance) {
    auto perturbed = [](double dy) {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::HEXAHEDRON);
        for (int v = 0; v < mesh.GetNV(); v++) {
            double* x = mesh.GetVertex(v);
            if (std::fabs(x[0] - 1.0) < 1e-12 && std::fabs(x[1] - 1.0 / 3.0) < 1e-12 &&
                std::fabs(x[2] - 1.0 / 3.0) < 1e-12) {
                x[1] += dy;
            }
        }
        return mesh;
    };
    auto throws = [](mfem::Mesh& serial, double rel_tol) {
        Distributed d = Distribute(serial, MPI_COMM_WORLD);
        ms::PeriodicFaceParameters pairing = UnitCubePairing();
        pairing.relative_tolerance = rel_tol;
        try {
            const ms::MeshAdjacency adjacency(*d.pmesh, pairing);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    mfem::Mesh far = perturbed(0.05);
    EXPECT_TRUE(throws(far, 1e-6));
    mfem::Mesh near = perturbed(1e-9);
    EXPECT_FALSE(throws(near, 1e-6));
    EXPECT_TRUE(throws(near, 1e-10));

    // Parameter checks: a tolerance outside (0, 0.5) and different parameters on different ranks
    mfem::Mesh box = mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::HEXAHEDRON);
    Distributed d = Distribute(box, MPI_COMM_WORLD);
    ms::PeriodicFaceParameters bad = UnitCubePairing();
    bad.relative_tolerance = 0.0;
    EXPECT_THROW(ms::MeshAdjacency(*d.pmesh, bad), std::invalid_argument);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks > 1) {
        ms::PeriodicFaceParameters mismatched = UnitCubePairing();
        if (rank == nranks - 1) {
            mismatched.periodic[2] = false;
        }
        EXPECT_THROW(ms::MeshAdjacency(*d.pmesh, mismatched), std::invalid_argument);
    }
}

// Invalid input is refused on every rank together: a grain id outside [1, num_grains] on only
// the last rank's elements, and replicated parameter errors (missing anchors, negative minimum).
TEST(FragmentCleanup, InvalidInputThrowsOnAllRanks) {
    GridMesh grid = Box(4, 2, 2, 1.0, false);
    Distributed d = Distribute(grid.mesh, MPI_COMM_WORLD);
    const ms::MeshAdjacency adjacency(*d.pmesh);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    mfem::Array<int> local(d.pmesh->GetNE());
    local = 1;
    if (rank == nranks - 1 && local.Size() > 0) {
        local[0] = 3; // outside [1, 2]
    }
    ms::FragmentCleanupParameters params;
    EXPECT_THROW(ms::CleanupFragments(adjacency, local, 2, {}, params), std::invalid_argument);

    local = 1;
    params.keep = ms::KeepRule::SEED_ELEMENT; // but no anchors
    EXPECT_THROW(ms::CleanupFragments(adjacency, local, 2, {}, params), std::invalid_argument);
    params.keep = ms::KeepRule::LARGEST;
    params.min_grain_elements = -1;
    EXPECT_THROW(ms::CleanupFragments(adjacency, local, 2, {}, params), std::invalid_argument);
    params.min_grain_elements = 1;
    EXPECT_THROW(ms::CleanupFragments(adjacency, local, 2, {5}, params), std::invalid_argument);

    // And valid input still works afterwards (all ranks stayed in step)
    const ms::FragmentCleanupResult ok = ms::CleanupFragments(adjacency, local, 2, {}, params);
    EXPECT_EQ(ok.num_grains, 1); // grain 2 has no element and is dropped by the renumbering
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
