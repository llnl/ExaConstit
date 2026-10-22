/**
 * @file test_geodesic.cpp
 * @brief Tests of the per-vertex distance field (labeling/vertex_field) and the graph geodesic
 * labeler (labeling/geodesic_graph).
 *
 * Every test runs on all ranks: VertexField construction, InitializeSources, ReduceShared* and
 * LabelGeodesicGraph are collective (every rank of the communicator must call them, because the
 * ranks exchange data inside). Rank-count independence is checked inside each run: the result is
 * computed three times,
 *   - on MPI_COMM_WORLD with a METIS partition,
 *   - on MPI_COMM_WORLD with a scattered round-robin partition in which the last rank owns no
 *     element (when there are at least 2 ranks),
 *   - on MPI_COMM_SELF, i.e. every rank alone on the whole mesh (the np = 1 answer),
 * gathered into serial order (elements by serial id, vertices by their exact coordinates) and
 * compared bitwise. Running the executable at np = 1..4 (test_geodesic.cmake) therefore compares
 * np = 2, 3, 4 against np = 1.
 *
 * Meshes are built cell by cell on a regular grid (cell size h), as hexes or as the 6-tet Kuhn
 * split of every cell (the same split in every cell, so the mesh is conforming), with cells left
 * out to make L-shaped or disconnected parts.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/labeling/geodesic_graph.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_options.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

int WorldRank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/** A grid-built mesh and the grid cell (i, j, k) of every element. */
struct GridMesh {
    mfem::Mesh mesh;
    std::vector<std::array<int, 3>> cell;
};

/**
 * Mesh of the cells (i, j, k), 0 <= i < nx, 0 <= j < ny, 0 <= k < nz, with kept(i, j, k) true;
 * cell (i, j, k) is [i h, (i+1) h] x [j h, (j+1) h] x [k h, (k+1) h]. Hexes, or the 6-tet Kuhn
 * split along the cell diagonal from its (0,0,0) to its (1,1,1) corner. (Same builder as
 * test_fragments.cpp.)
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
    GridMesh grid{mfem::Mesh(3, (nx + 1) * (ny + 1) * (nz + 1), tets ? 6 * num_cells : num_cells,
                             0, 3),
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

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** How to distribute a serial mesh for one run. */
enum class Layout { METIS_WORLD, SCATTERED_WORLD, SELF };

/**
 * Partitions `serial` (collective over the communicator of the layout). SCATTERED_WORLD deals the
 * elements round-robin to ranks 0 .. nranks-2, so the last rank owns no element (at >= 2 ranks)
 * and the partitions have many shared vertices.
 */
Distributed Distribute(mfem::Mesh& serial, Layout layout) {
    const MPI_Comm comm = (layout == Layout::SELF) ? MPI_COMM_SELF : MPI_COMM_WORLD;
    Distributed d;
    if (layout != Layout::SCATTERED_WORLD) {
        d.order = ms::BuildSerialOrder(serial, comm);
        d.pmesh = ms::MakeParMesh(comm, serial, d.order);
        return d;
    }
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const int owners = (nranks > 1) ? nranks - 1 : 1;
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

/** Serial vertex id by exact coordinates (vertex coordinates are copied bitwise by ParMesh). */
std::map<std::array<double, 3>, int> SerialVertexIds(const mfem::Mesh& serial) {
    std::map<std::array<double, 3>, int> ids;
    for (int v = 0; v < serial.GetNV(); ++v) {
        const double* p = serial.GetVertex(v);
        ids[{p[0], p[1], p[2]}] = v;
    }
    return ids;
}

/** Per-vertex values of a VertexField in serial vertex order (identical on every rank). */
struct SerialVertexValues {
    std::vector<double> T;
    std::vector<int> label;
    bool copies_agree = true; ///< all copies of every shared vertex held the same (T, label)
    bool all_found = true;    ///< every local vertex was found in the serial mesh
};

/**
 * Gathers (serial vertex id, T, label) of every local vertex of every rank (MPI_Allgatherv) and
 * arranges them in serial order; vertices touched by no element keep (inf, -1). Collective over
 * comm.
 */
SerialVertexValues GatherVertexValues(const mfem::Mesh& serial, const ms::VertexField& field,
                                      MPI_Comm comm) {
    const auto ids = SerialVertexIds(serial);
    const int nv = field.NumVertices();
    const double* xyz = field.Coordinates().HostRead();
    const double* t = field.Distance().HostRead();
    const int* lab = field.Label().HostRead();
    SerialVertexValues out;
    std::vector<int> local_ids(static_cast<std::size_t>(nv));
    std::vector<double> local_t(static_cast<std::size_t>(nv));
    std::vector<int> local_lab(static_cast<std::size_t>(nv));
    bool found = true;
    for (int v = 0; v < nv; ++v) {
        const auto it = ids.find({xyz[3 * v], xyz[3 * v + 1], xyz[3 * v + 2]});
        found = found && it != ids.end();
        local_ids[static_cast<std::size_t>(v)] = (it != ids.end()) ? it->second : -1;
        local_t[static_cast<std::size_t>(v)] = t[v];
        local_lab[static_cast<std::size_t>(v)] = lab[v];
    }
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    MPI_Allgather(&nv, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    for (int r = 1; r < nranks; ++r) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r - 1)] + counts[static_cast<std::size_t>(r - 1)];
    }
    const int total = displs.back() + counts.back();
    std::vector<int> all_ids(static_cast<std::size_t>(total));
    std::vector<double> all_t(static_cast<std::size_t>(total));
    std::vector<int> all_lab(static_cast<std::size_t>(total));
    MPI_Allgatherv(local_ids.data(), nv, MPI_INT, all_ids.data(), counts.data(), displs.data(),
                   MPI_INT, comm);
    MPI_Allgatherv(local_t.data(), nv, MPI_DOUBLE, all_t.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, comm);
    MPI_Allgatherv(local_lab.data(), nv, MPI_INT, all_lab.data(), counts.data(), displs.data(),
                   MPI_INT, comm);
    int found_int = found ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &found_int, 1, MPI_INT, MPI_MIN, comm);
    out.all_found = found_int != 0;

    const auto nsv = static_cast<std::size_t>(serial.GetNV());
    out.T.assign(nsv, INF);
    out.label.assign(nsv, -1);
    std::vector<char> seen(nsv, 0);
    for (int k = 0; k < total; ++k) {
        const int sid = all_ids[static_cast<std::size_t>(k)];
        if (sid < 0) {
            continue;
        }
        const auto s = static_cast<std::size_t>(sid);
        const double tk = all_t[static_cast<std::size_t>(k)];
        const int lk = all_lab[static_cast<std::size_t>(k)];
        if (seen[s] != 0 && (out.T[s] != tk || out.label[s] != lk)) {
            out.copies_agree = false;
        }
        seen[s] = 1;
        out.T[s] = tk;
        out.label[s] = lk;
    }
    return out;
}

/** Element values gathered into serial order on every rank (GatherToSerial + MPI_Bcast). */
std::vector<int> GatherElements(MPI_Comm comm, const ms::SerialOrder& order,
                                const mfem::Array<int>& local, int serial_ne) {
    std::vector<int> values = ms::GatherToSerial(comm, order, local, 0);
    values.resize(static_cast<std::size_t>(serial_ne));
    MPI_Bcast(values.data(), serial_ne, MPI_INT, 0, comm);
    return values;
}

/** A geodesic run in serial order. */
struct GeodesicRun {
    std::vector<int> element_seed;
    SerialVertexValues vertex;
    ms::GeodesicReport report;
    ms::SourceReport sources;
};

/** LabelGeodesicGraph on `serial` distributed per `layout`, gathered. Collective. */
GeodesicRun RunGeodesic(mfem::Mesh& serial, Layout layout, const ms::SeedSet& seeds,
                        bool additive, const ms::GeodesicParameters& params) {
    const MPI_Comm comm = (layout == Layout::SELF) ? MPI_COMM_SELF : MPI_COMM_WORLD;
    Distributed d = Distribute(serial, layout);
    ms::GeodesicLabels labels = ms::LabelGeodesicGraph(*d.pmesh, comm, seeds, additive, params);
    GeodesicRun run;
    run.element_seed = GatherElements(comm, d.order, labels.element_seed, serial.GetNE());
    run.vertex = GatherVertexValues(serial, *labels.field, comm);
    run.report = labels.report;
    run.sources = labels.sources;
    return run;
}

/**
 * Runs the three layouts and checks that they agree bitwise (element labels, vertex T and labels,
 * source elements, partition-independent statistics) and that every rank's copies of a shared
 * vertex agree. Returns the single-rank (MPI_COMM_SELF) run; `world_report`, if given, receives
 * the report of the METIS run (its round counts show the exchanges). Collective over
 * MPI_COMM_WORLD.
 */
GeodesicRun RunAndCompare(mfem::Mesh& serial, const ms::SeedSet& seeds, bool additive,
                          const ms::GeodesicParameters& params = {},
                          ms::GeodesicReport* world_report = nullptr) {
    const GeodesicRun world = RunGeodesic(serial, Layout::METIS_WORLD, seeds, additive, params);
    const GeodesicRun scattered =
        RunGeodesic(serial, Layout::SCATTERED_WORLD, seeds, additive, params);
    const GeodesicRun self = RunGeodesic(serial, Layout::SELF, seeds, additive, params);
    for (const GeodesicRun* other : {&world, &scattered}) {
        EXPECT_TRUE(other->vertex.copies_agree);
        EXPECT_TRUE(other->vertex.all_found);
        EXPECT_EQ(other->element_seed, self.element_seed);
        EXPECT_EQ(other->vertex.label, self.vertex.label);
        // Bitwise: == on doubles, with +inf == +inf
        EXPECT_EQ(other->vertex.T, self.vertex.T);
        EXPECT_EQ(other->sources.source_element_key, self.sources.source_element_key);
        EXPECT_EQ(other->sources.seed_outside, self.sources.seed_outside);
        EXPECT_EQ(other->report.seeds_outside_material, self.report.seeds_outside_material);
        EXPECT_EQ(other->report.unreached_vertices, self.report.unreached_vertices);
        EXPECT_EQ(other->report.unreached_elements, self.report.unreached_elements);
        EXPECT_EQ(other->report.split_elements, self.report.split_elements);
        EXPECT_EQ(other->report.unlabeled_vertices, 0);
    }
    EXPECT_EQ(self.report.distance_rounds, 1); // one rank: nothing to exchange
    EXPECT_EQ(self.report.label_rounds, 1);
    if (world_report != nullptr) {
        *world_report = world.report;
    }
    return self;
}

/** One seed at (x, y, z) with source id 0 appended to `seeds`. */
void AddSeed(ms::SeedSet& seeds, double x, double y, double z) {
    seeds.x.push_back(x);
    seeds.y.push_back(y);
    seeds.z.push_back(z);
    seeds.source_id.push_back(static_cast<std::int64_t>(seeds.source_id.size()));
}

/** Element centroid of serial element e (vertex average; exact for the affine grid cells). */
std::array<double, 3> Centroid(const mfem::Mesh& mesh, int e) {
    mfem::Array<int> v;
    mesh.GetElementVertices(e, v);
    std::array<double, 3> c = {0.0, 0.0, 0.0};
    for (int k = 0; k < v.Size(); ++k) {
        for (int d = 0; d < 3; ++d) {
            c[static_cast<std::size_t>(d)] += mesh.GetVertex(v[k])[d] / v.Size();
        }
    }
    return c;
}

/**
 * Largest relative error (T - exact) / exact over the vertices for which `exact(p)` returns a
 * positive value (the vertices the caller wants measured; others return <= 0).
 */
template <typename Exact>
double MaxRelativeError(const mfem::Mesh& serial, const std::vector<double>& T, Exact exact,
                        double* mean_error = nullptr) {
    double worst = 0.0;
    double sum = 0.0;
    int count = 0;
    for (int v = 0; v < serial.GetNV(); ++v) {
        const double* p = serial.GetVertex(v);
        const double reference = exact(p);
        if (reference <= 0.0) {
            continue;
        }
        const double rel = std::fabs(T[static_cast<std::size_t>(v)] - reference) / reference;
        worst = std::max(worst, rel);
        sum += rel;
        ++count;
    }
    if (mean_error != nullptr) {
        *mean_error = (count > 0) ? sum / count : 0.0;
    }
    return worst;
}

} // namespace

// ReduceShared must give every copy of a shared vertex the lexicographic minimum of (T, label)
// over the copies: the smallest T, and among copies with that T the smallest label. Every rank
// writes rank-dependent values T = (sid + rank) % 3, label = (7 sid + 13 rank) % 5 into its copy
// of serial vertex sid (so copies differ in T, and some copies tie in T with different labels);
// the expected result is computed from the copies of all ranks. A second call changes
// nothing (the exchange is idempotent). The same partition also checks that the vertex numbering
// and coordinates match the mesh vertices, and that the owned marker counts each vertex once.
TEST(VertexField, ReduceSharedIsLexicographicMinimum) {
    for (const bool tets : {false, true}) {
        GridMesh grid = BuildGrid(4, 3, 5, 0.5, tets, [](int, int, int) { return true; });
        Distributed d = Distribute(grid.mesh, Layout::SCATTERED_WORLD);
        ms::VertexField field(*d.pmesh);
        const int rank = WorldRank();
        const auto ids = SerialVertexIds(grid.mesh);

        // Coordinates are the mesh vertices; owned marker counts every vertex once
        const double* xyz = field.Coordinates().HostRead();
        bool coords_ok = field.NumVertices() == d.pmesh->GetNV();
        long long owned = 0;
        std::vector<int> sid(static_cast<std::size_t>(field.NumVertices()));
        for (int v = 0; v < field.NumVertices(); ++v) {
            const double* p = d.pmesh->GetVertex(v);
            for (int k = 0; k < 3; ++k) {
                coords_ok = coords_ok && xyz[3 * v + k] == p[k];
            }
            owned += field.OwnedVertexMarker()[v];
            sid[static_cast<std::size_t>(v)] = ids.at({p[0], p[1], p[2]});
        }
        EXPECT_TRUE(coords_ok);
        MPI_Allreduce(MPI_IN_PLACE, &owned, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(owned, grid.mesh.GetNV());

        // Rank-dependent copies
        double* t = field.Distance().HostWrite();
        int* lab = field.Label().HostWrite();
        for (int v = 0; v < field.NumVertices(); ++v) {
            const int s = sid[static_cast<std::size_t>(v)];
            t[v] = static_cast<double>((s + rank) % 3);
            lab[v] = (7 * s + 13 * rank) % 5;
        }
        // Expected: lexicographic minimum over all copies. Every rank's copy of vertex s is the
        // formula above evaluated with that rank, so it is enough to know which ranks hold s
        // (presence flags, reduced with MPI_MAX).
        std::vector<std::pair<double, int>> expected(static_cast<std::size_t>(grid.mesh.GetNV()),
                                                     {INF, -1});
        int nranks = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        std::vector<int> present(static_cast<std::size_t>(grid.mesh.GetNV() * nranks), 0);
        for (int v = 0; v < field.NumVertices(); ++v) {
            present[static_cast<std::size_t>(sid[static_cast<std::size_t>(v)] * nranks + rank)] =
                1;
        }
        MPI_Allreduce(MPI_IN_PLACE, present.data(), static_cast<int>(present.size()), MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD);
        bool any_difference = false;
        for (int s = 0; s < grid.mesh.GetNV(); ++s) {
            int holders = 0;
            for (int r = 0; r < nranks; ++r) {
                if (present[static_cast<std::size_t>(s * nranks + r)] == 0) {
                    continue;
                }
                ++holders;
                const std::pair<double, int> c = {static_cast<double>((s + r) % 3),
                                                  (7 * s + 13 * r) % 5};
                expected[static_cast<std::size_t>(s)] =
                    std::min(expected[static_cast<std::size_t>(s)], c);
            }
            any_difference = any_difference || holders > 1;
        }

        mfem::Array<int> changed;
        const bool any_change = field.ReduceShared(&changed);
        EXPECT_EQ(any_change, any_difference);
        const SerialVertexValues after = GatherVertexValues(grid.mesh, field, MPI_COMM_WORLD);
        EXPECT_TRUE(after.copies_agree);
        for (int s = 0; s < grid.mesh.GetNV(); ++s) {
            EXPECT_EQ(after.T[static_cast<std::size_t>(s)],
                      expected[static_cast<std::size_t>(s)].first);
            EXPECT_EQ(after.label[static_cast<std::size_t>(s)],
                      expected[static_cast<std::size_t>(s)].second);
        }
        // Changed marker: exactly the local copies that differ from the result
        const double* t_after = field.Distance().HostRead();
        const int* lab_after = field.Label().HostRead();
        bool marker_ok = true;
        for (int v = 0; v < field.NumVertices(); ++v) {
            const int s = sid[static_cast<std::size_t>(v)];
            const bool differs = static_cast<double>((s + rank) % 3) != t_after[v] ||
                                 (7 * s + 13 * rank) % 5 != lab_after[v];
            marker_ok = marker_ok && (changed[v] != 0) == differs;
        }
        EXPECT_TRUE(marker_ok);
        EXPECT_FALSE(field.ReduceShared());
        EXPECT_FALSE(field.ReduceSharedDistance());
        EXPECT_FALSE(field.ReduceSharedLabels());
    }
}

// Curved meshes: the vertex coordinates come from the mesh Nodes. After SetCurvature(2) on an
// affine grid the Nodes at the vertex dofs are the original vertex coordinates (nodal
// interpolation of the linear map, exact at the corners), so Coordinates() must equal the mesh
// vertices bitwise.
TEST(VertexField, CurvedMeshUsesNodeVertices) {
    GridMesh grid = BuildGrid(3, 3, 3, 1.0 / 3.0, false, [](int, int, int) { return true; });
    grid.mesh.SetCurvature(2);
    Distributed d = Distribute(grid.mesh, Layout::METIS_WORLD);
    ms::VertexField field(*d.pmesh);
    const double* xyz = field.Coordinates().HostRead();
    const auto ids = SerialVertexIds(grid.mesh);
    bool all_found = true;
    for (int v = 0; v < field.NumVertices(); ++v) {
        all_found = all_found && ids.count({xyz[3 * v], xyz[3 * v + 1], xyz[3 * v + 2]}) == 1;
    }
    EXPECT_TRUE(all_found);
}

// Straight strut, accuracy: a 1 x 1 x 10 box of 8 x 8 x 80 cells (h = 1/8), point source at the
// center of the end z = 0 (a vertex). The geodesic distance in a convex body is the straight-line
// distance, so T(v) is compared with |v - s| for every vertex more than 2 length units away.
// Graph paths can only follow the element edge and diagonal directions, so T over-estimates: on
// the hex grid by at most 12.8% (the worst direction of the 26-direction path norm, near
// (0.89, 0.37, 0.28)); the strut corners at z ~ 2 are close to that direction. The Kuhn tet grid
// has fewer directions (one body diagonal, one diagonal per face) and errs more. The work package
// asked for < 5%, which no vertex-pair graph reaches here (see the WP1e.2 report); the bounds below
// are regression bounds just above the measured values (hex: max 11.8%, mean 3.7%; tet: max
// 28.9%, mean 6.7%), printed on rank 0. The Fast Iterative Method (WP1e.3) is the accurate solver.
TEST(GeodesicGraph, StraightStrutAccuracy) {
    ms::SeedSet seeds;
    AddSeed(seeds, 0.5, 0.5, 0.0);
    for (const bool tets : {false, true}) {
        GridMesh grid = BuildGrid(8, 8, 80, 0.125, tets, [](int, int, int) { return true; });
        ms::GeodesicReport world_report;
        const GeodesicRun run = RunAndCompare(grid.mesh, seeds, false, {}, &world_report);
        // With several ranks the distances must cross partition boundaries: more than one round
        int nranks = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        if (nranks > 1) {
            EXPECT_GE(world_report.distance_rounds, 2);
            EXPECT_GE(world_report.label_rounds, 2);
        }
        double mean = 0.0;
        const double worst = MaxRelativeError(grid.mesh, run.vertex.T, [](const double* p) {
            const double d = std::sqrt((p[0] - 0.5) * (p[0] - 0.5) +
                                       (p[1] - 0.5) * (p[1] - 0.5) + p[2] * p[2]);
            return (d > 2.0) ? d : 0.0;
        }, &mean);
        if (WorldRank() == 0) {
            std::cout << "[ geodesic ] straight strut " << (tets ? "tet" : "hex")
                      << ": max relative error " << worst << ", mean " << mean
                      << " (METIS run: " << world_report.distance_rounds << " distance rounds, "
                      << world_report.label_rounds << " label rounds)" << std::endl;
        }
        EXPECT_LT(worst, tets ? 0.30 : 0.125);
        EXPECT_LT(mean, tets ? 0.07 : 0.04);
        EXPECT_EQ(run.report.unreached_vertices, 0);
        EXPECT_EQ(run.report.unreached_elements, 0);
    }
}

// L-shaped strut, accuracy around the inner corner: arms [0,4] x [0,1] and [0,1] x [0,4], height
// 1, h = 1/8; source at (3.5, 0.5, 0.5) at the end of the x arm. For a vertex p in the y arm with
// y >= 2 the straight line to the source crosses the void, and the shortest path inside the
// material bends around the vertical inner corner edge x = y = 1: unfolding the two arms into a
// plane gives length sqrt((r1 + r2)^2 + (z_p - z_s)^2), r1 and r2 the horizontal distances of the
// source and of p from the edge. As for the strut, the graph over-estimates; regression bounds
// just above the measured values (hex: max 10.1%, mean 7.5%; tet: max 30.5%, mean 20.8%).
TEST(GeodesicGraph, LShapeCornerAccuracy) {
    constexpr int n = 8;
    ms::SeedSet seeds;
    AddSeed(seeds, 3.5, 0.5, 0.5);
    const double r1 = std::sqrt(2.5 * 2.5 + 0.5 * 0.5);
    for (const bool tets : {false, true}) {
        GridMesh grid = BuildGrid(4 * n, 4 * n, n, 1.0 / n, tets,
                                  [](int i, int j, int) { return i < n || j < n; });
        const GeodesicRun run = RunAndCompare(grid.mesh, seeds, false);
        double mean = 0.0;
        const double worst = MaxRelativeError(grid.mesh, run.vertex.T, [r1](const double* p) {
            if (p[1] < 2.0) {
                return 0.0;
            }
            const double r2 = std::sqrt((p[0] - 1.0) * (p[0] - 1.0) + (p[1] - 1.0) * (p[1] - 1.0));
            const double dz = p[2] - 0.5;
            return std::sqrt((r1 + r2) * (r1 + r2) + dz * dz);
        }, &mean);
        if (WorldRank() == 0) {
            std::cout << "[ geodesic ] L-shape corner " << (tets ? "tet" : "hex")
                      << ": max relative error " << worst << ", mean " << mean << std::endl;
        }
        EXPECT_LT(worst, tets ? 0.32 : 0.11);
        EXPECT_LT(mean, tets ? 0.22 : 0.08);
    }
}

// U-shaped part (test/data/microstructure/u_shape_hex.mesh: [0,3] x [0,2] x [0,1] minus the slot
// [1,2] x [1,2] x [0,1], 4 cells per unit). Seed A = (0.9, 1.9, 0.5) at the tip of the left arm
// next to the slot, seed B = (2.9, 0.5, 0.5) low on the outer side of the right arm. The top of
// the right arm is closer to A in a straight line (across the slot: 1.23 vs 1.58 for the element
// centered at (2.125, 1.875)), so Euclidean labels put elements of the right arm into A's grain.
// Inside the material A is about 2.9 away from there (down the left arm, across the base, up),
// so the geodesic labels keep every element of the right arm above the base (x > 2, y > 1) in B's
// grain and every element of the left arm above the base (x < 1, y > 1) in A's grain. Checked for
// centroid and vote sampling.
TEST(GeodesicGraph, UShapeLabelsDoNotCrossTheGap) {
    mfem::Mesh mesh(std::string(MS_GEODESIC_DATA_DIR) + "/microstructure/u_shape_hex.mesh");
    ms::SeedSet seeds;
    AddSeed(seeds, 0.9, 1.9, 0.5);
    AddSeed(seeds, 2.9, 0.5, 0.5);

    // Euclidean labels (every rank alone on the serial mesh): A crosses the slot
    const ms::TessellationLabels euclid =
        ms::LabelTessellation(mesh, MPI_COMM_SELF, seeds, ms::TessellationOptions{});
    const int* es = euclid.element_seed.HostRead();
    int euclid_crossing = 0;
    for (int e = 0; e < mesh.GetNE(); ++e) {
        const auto c = Centroid(mesh, e);
        euclid_crossing += (c[0] > 2.0 && c[1] > 1.0 && es[e] == 0) ? 1 : 0;
    }
    EXPECT_GT(euclid_crossing, 0);

    for (const auto sampling : {ms::SamplingMode::CENTROID, ms::SamplingMode::QUADRATURE_VOTE}) {
        ms::GeodesicParameters params;
        params.sampling = sampling;
        const GeodesicRun run = RunAndCompare(mesh, seeds, false, params);
        int wrong = 0;
        for (int e = 0; e < mesh.GetNE(); ++e) {
            const auto c = Centroid(mesh, e);
            const int label = run.element_seed[static_cast<std::size_t>(e)];
            if (c[1] > 1.0 && c[0] > 2.0) {
                wrong += (label != 1) ? 1 : 0;
            }
            if (c[1] > 1.0 && c[0] < 1.0) {
                wrong += (label != 0) ? 1 : 0;
            }
        }
        EXPECT_EQ(wrong, 0);
        EXPECT_EQ(run.report.unreached_elements, 0);
    }
}

// Additive weights: seeds at the two end centers of a 1 x 1 x 10 strut (4 x 4 x 40 cells,
// h = 1/4), a0 = 0 and a1 = 2. The Additive grain boundary is where d0 - a0 = d1 - a1, i.e.
// z = 10 - z - 2 -> z = 4 on the axis; every graph path from either end to a point of the cross
// section at height z costs the same lateral detour, so the boundary is the plane z = 4 for every
// vertex. Elements whose centroid lies more than one element (h) below 4 must belong to seed 0,
// more than h above to seed 1; the axis vertices get the exact path sums min(z, 8 - z).
TEST(GeodesicGraph, AdditiveBoundaryPosition) {
    GridMesh grid = BuildGrid(4, 4, 40, 0.25, false, [](int, int, int) { return true; });
    ms::SeedSet seeds;
    AddSeed(seeds, 0.5, 0.5, 0.0);
    AddSeed(seeds, 0.5, 0.5, 10.0);
    seeds.weight = {0.0, 2.0};
    const GeodesicRun run = RunAndCompare(grid.mesh, seeds, true);
    int wrong = 0;
    for (int e = 0; e < grid.mesh.GetNE(); ++e) {
        const double z = Centroid(grid.mesh, e)[2];
        const int label = run.element_seed[static_cast<std::size_t>(e)];
        if (z < 4.0 - 0.25) {
            wrong += (label != 0) ? 1 : 0;
        } else if (z > 4.0 + 0.25) {
            wrong += (label != 1) ? 1 : 0;
        }
    }
    EXPECT_EQ(wrong, 0);
    for (int v = 0; v < grid.mesh.GetNV(); ++v) {
        const double* p = grid.mesh.GetVertex(v);
        if (p[0] == 0.5 && p[1] == 0.5) {
            EXPECT_NEAR(run.vertex.T[static_cast<std::size_t>(v)],
                        std::min(p[2], 8.0 - p[2]), 1e-12);
        }
    }
}

// Edge cases on one mesh: two boxes [0,1] x [0,1] x [0,1] and [0,1] x [0,1] x [1.25, 2.25]
// (h = 1/4, one empty cell layer between them, so they share no vertex). Seed 0 lies outside the
// material below the lower box at (0.5, 0.5, -0.25): its nearest vertex is (0.5, 0.5, 0), shared
// by 4 hexes, so it must snap to the one with the smallest element key, and that vertex starts at
// T = |v - s| = 0.25 exactly (and keeps it: seed 1 is 0.66 away). Seed 1 lies inside the lower
// box. The upper box has no seed: its 125
// vertices keep T = +inf and label -1 and its 64 elements get -1. The scattered layout of
// RunAndCompare also covers a rank owning no element.
TEST(GeodesicGraph, SnappingUnreachedBodyAndEmptyRanks) {
    GridMesh grid = BuildGrid(4, 4, 9, 0.25, false, [](int, int, int k) { return k != 4; });
    ms::SeedSet seeds;
    AddSeed(seeds, 0.5, 0.5, -0.25);
    AddSeed(seeds, 0.3, 0.7, 0.6);
    const GeodesicRun run = RunAndCompare(grid.mesh, seeds, false);

    // Snapping: smallest key among the elements touching vertex (0.5, 0.5, 0)
    const auto ids = SerialVertexIds(grid.mesh);
    const int corner = ids.at({0.5, 0.5, 0.0});
    std::uint64_t expected_key = std::numeric_limits<std::uint64_t>::max();
    int touching = 0;
    for (int e = 0; e < grid.mesh.GetNE(); ++e) {
        mfem::Array<int> v;
        grid.mesh.GetElementVertices(e, v);
        if (v.Find(corner) >= 0) {
            expected_key = std::min(expected_key, ms::ElementKey(grid.mesh, e));
            ++touching;
        }
    }
    EXPECT_EQ(touching, 4);
    EXPECT_EQ(run.sources.seed_outside, (std::vector<char>{1, 0}));
    EXPECT_EQ(run.sources.num_seeds_outside, 1);
    EXPECT_EQ(run.report.seeds_outside_material, 1);
    EXPECT_EQ(run.sources.source_element_key[0], expected_key);
    EXPECT_EQ(run.vertex.T[static_cast<std::size_t>(corner)], 0.25);
    EXPECT_EQ(run.vertex.label[static_cast<std::size_t>(corner)], 0);

    // The seedless upper box
    int upper_vertices_ok = 0;
    for (int v = 0; v < grid.mesh.GetNV(); ++v) {
        if (grid.mesh.GetVertex(v)[2] > 1.1) {
            upper_vertices_ok += (run.vertex.T[static_cast<std::size_t>(v)] == INF &&
                                  run.vertex.label[static_cast<std::size_t>(v)] == -1)
                                     ? 1
                                     : 0;
        }
    }
    EXPECT_EQ(upper_vertices_ok, 125);
    EXPECT_EQ(run.report.unreached_vertices, 125);
    EXPECT_EQ(run.report.unreached_elements, 64);
    for (int e = 0; e < grid.mesh.GetNE(); ++e) {
        const bool upper = Centroid(grid.mesh, e)[2] > 1.1;
        EXPECT_EQ(run.element_seed[static_cast<std::size_t>(e)] < 0, upper);
    }
}

// Independent reference for phase 1 and the labels, on a small non-convex part (hexes and tets):
// a 4 x 3 x 2 grid with one column of cells removed, three seeds inside elements, one with an
// Additive offset. The reference is written here from the definitions, on the serial mesh:
//  - T0: |v - s_i| - a_i on the vertices of each seed's source element (the source element is
//    taken from the run's SourceReport; containment is checked by the snapping test), keeping the
//    lexicographic minimum;
//  - T: plain Bellman-Ford sweeps (relax every vertex pair of every element, T(b) = T(a) + |a - b|
//    when smaller) until nothing changes, which reaches the same smallest path sums as Dijkstra;
//  - labels: brute-force fixed point of "label(u) = smallest seed among the source seed (if
//    T0(u) == T(u)) and the labels of the tight neighbors v (T(v) < T(u), T(v) + |u - v| == T(u))",
//    iterated over all pairs until nothing changes.
// The distances are compared bitwise: both sides add the same edge lengths (sqrt of
// dx^2 + dy^2 + dz^2 of the same coordinates) along the same shortest paths.
TEST(GeodesicGraph, MatchesSerialReference) {
    for (const bool tets : {false, true}) {
        GridMesh grid = BuildGrid(4, 3, 2, 0.5, tets,
                                  [](int i, int j, int) { return !(i == 2 && j >= 1); });
        ms::SeedSet seeds;
        AddSeed(seeds, 0.3, 0.2, 0.3);
        AddSeed(seeds, 1.85, 1.3, 0.7);
        AddSeed(seeds, 1.2, 0.15, 0.6);
        seeds.weight = {0.0, 0.1, 0.05};
        const GeodesicRun run = RunAndCompare(grid.mesh, seeds, true);
        const mfem::Mesh& mesh = grid.mesh;
        const int nv = mesh.GetNV();

        // Source elements by key, then T0 (lexicographic minimum, seeds in increasing order)
        std::vector<double> t0(static_cast<std::size_t>(nv), INF);
        std::vector<int> l0(static_cast<std::size_t>(nv), -1);
        for (int i = 0; i < 3; ++i) {
            const auto si = static_cast<std::size_t>(i);
            for (int e = 0; e < mesh.GetNE(); ++e) {
                if (ms::ElementKey(mesh, e) != run.sources.source_element_key[si]) {
                    continue;
                }
                mfem::Array<int> verts;
                mesh.GetElementVertices(e, verts);
                for (int k = 0; k < verts.Size(); ++k) {
                    const double* p = mesh.GetVertex(verts[k]);
                    const double dx = seeds.x[si] - p[0];
                    const double dy = seeds.y[si] - p[1];
                    const double dz = seeds.z[si] - p[2];
                    const double value = std::sqrt(dx * dx + dy * dy + dz * dz) - seeds.weight[si];
                    const auto v = static_cast<std::size_t>(verts[k]);
                    if (value < t0[v]) {
                        t0[v] = value;
                        l0[v] = i;
                    }
                }
            }
        }
        // Every vertex pair of every element, with its length
        struct Pair {
            int a, b;
            double w;
        };
        std::vector<Pair> pairs;
        for (int e = 0; e < mesh.GetNE(); ++e) {
            mfem::Array<int> verts;
            mesh.GetElementVertices(e, verts);
            for (int i = 0; i < verts.Size(); ++i) {
                for (int j = 0; j < verts.Size(); ++j) {
                    if (i == j) {
                        continue;
                    }
                    const double* pa = mesh.GetVertex(verts[i]);
                    const double* pb = mesh.GetVertex(verts[j]);
                    const double dx = pa[0] - pb[0];
                    const double dy = pa[1] - pb[1];
                    const double dz = pa[2] - pb[2];
                    pairs.push_back({verts[i], verts[j], std::sqrt(dx * dx + dy * dy + dz * dz)});
                }
            }
        }
        // Bellman-Ford sweeps
        std::vector<double> t = t0;
        for (bool changed = true; changed;) {
            changed = false;
            for (const Pair& q : pairs) {
                const double cand = t[static_cast<std::size_t>(q.a)] + q.w;
                if (cand < t[static_cast<std::size_t>(q.b)]) {
                    t[static_cast<std::size_t>(q.b)] = cand;
                    changed = true;
                }
            }
        }
        // Brute-force label fixed point
        std::vector<int> label(static_cast<std::size_t>(nv));
        for (int v = 0; v < nv; ++v) {
            const auto sv = static_cast<std::size_t>(v);
            label[sv] = (t[sv] == INF) ? -1 : ((l0[sv] >= 0 && t0[sv] == t[sv]) ? l0[sv] : INT_MAX);
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (const Pair& q : pairs) {
                const auto a = static_cast<std::size_t>(q.a);
                const auto b = static_cast<std::size_t>(q.b);
                if (t[a] < t[b] && t[a] + q.w == t[b] && label[a] < label[b]) {
                    label[b] = label[a];
                    changed = true;
                }
            }
        }
        EXPECT_EQ(run.vertex.T, t);
        EXPECT_EQ(run.vertex.label, label);
        int seen[3] = {0, 0, 0};
        for (const int l : label) {
            if (l >= 0 && l < 3) {
                seen[l] = 1;
            }
        }
        EXPECT_EQ(seen[0] + seen[1] + seen[2], 3); // every seed labels some vertex
    }
}

// Exact ties go to the lower seed index. Two seeds at mirror positions (0.625, 0.5625, 0.125) and
// (0.625, 0.5625, 1.875) of a 1 x 1 x 2 strut (h = 1/4, all coordinates exact in binary), so the
// mesh and the seeds are symmetric about the plane z = 1 bit for bit: every path sum from one seed
// has a mirror path sum from the other with the same bits, T(x, y, z) == T(x, y, 2 - z) exactly,
// and the vertices of the plane z = 1 are reached at exactly the same T from both seeds. They must
// get label 0, whichever seed is listed first.
TEST(GeodesicGraph, ExactTieGoesToLowerSeedIndex) {
    GridMesh grid = BuildGrid(4, 4, 8, 0.25, false, [](int, int, int) { return true; });
    const auto ids = SerialVertexIds(grid.mesh);
    for (const bool bottom_first : {true, false}) {
        ms::SeedSet seeds;
        AddSeed(seeds, 0.625, 0.5625, bottom_first ? 0.125 : 1.875);
        AddSeed(seeds, 0.625, 0.5625, bottom_first ? 1.875 : 0.125);
        const GeodesicRun run = RunAndCompare(grid.mesh, seeds, false);
        int mirror_mismatch = 0;
        int plane_vertices = 0;
        int plane_wrong = 0;
        for (int v = 0; v < grid.mesh.GetNV(); ++v) {
            const double* p = grid.mesh.GetVertex(v);
            const int mirror = ids.at({p[0], p[1], 2.0 - p[2]});
            mirror_mismatch += (run.vertex.T[static_cast<std::size_t>(v)] !=
                                run.vertex.T[static_cast<std::size_t>(mirror)])
                                   ? 1
                                   : 0;
            if (p[2] == 1.0) {
                ++plane_vertices;
                plane_wrong += (run.vertex.label[static_cast<std::size_t>(v)] != 0) ? 1 : 0;
            }
        }
        EXPECT_EQ(mirror_mismatch, 0);
        EXPECT_EQ(plane_vertices, 25);
        EXPECT_EQ(plane_wrong, 0);
    }
}

// Direct term inside a source element: on a 2 x 2 x 2 grid of unit hexes, seed 1 at
// (0.5, 0.5, 0.75) is 0.25 from the centroid (0.5, 0.5, 0.5) of its element but at least 0.75
// from every corner, while seed 0 at (0.02, 0.02, 0.02), in the same element, sits next to the
// corner (0, 0, 0). Through the corners alone seed 0 would win the centroid (about 0.90 against
// 1.62 for seed 1); with the straight lines from the seeds inside their source element, seed 1
// wins (0.25 against 0.83).
TEST(GeodesicGraph, DirectTermInSourceElement) {
    GridMesh grid = BuildGrid(2, 2, 2, 1.0, false, [](int, int, int) { return true; });
    ms::SeedSet seeds;
    AddSeed(seeds, 0.02, 0.02, 0.02);
    AddSeed(seeds, 0.5, 0.5, 0.75);
    const GeodesicRun run = RunAndCompare(grid.mesh, seeds, false);
    for (int e = 0; e < grid.mesh.GetNE(); ++e) {
        if (grid.cell[static_cast<std::size_t>(e)] == std::array<int, 3> {0, 0, 0}) {
            EXPECT_EQ(run.element_seed[static_cast<std::size_t>(e)], 1);
        }
    }
}

// Invalid input throws on every rank (and leaves the ranks in step for the next collective call):
// an empty seed set, a weight array of the wrong size, max_rounds < 1.
TEST(GeodesicGraph, InvalidInputThrowsOnAllRanks) {
    GridMesh grid = BuildGrid(2, 2, 2, 0.5, false, [](int, int, int) { return true; });
    Distributed d = Distribute(grid.mesh, Layout::METIS_WORLD);
    ms::SeedSet empty;
    EXPECT_THROW(ms::LabelGeodesicGraph(*d.pmesh, MPI_COMM_WORLD, empty, false, {}),
                 std::invalid_argument);
    ms::SeedSet seeds;
    AddSeed(seeds, 0.5, 0.5, 0.5);
    seeds.weight = {1.0, 2.0};
    EXPECT_THROW(ms::LabelGeodesicGraph(*d.pmesh, MPI_COMM_WORLD, seeds, true, {}),
                 std::invalid_argument);
    seeds.weight.clear();
    ms::GeodesicParameters bad;
    bad.max_rounds = 0;
    EXPECT_THROW(ms::LabelGeodesicGraph(*d.pmesh, MPI_COMM_WORLD, seeds, false, bad),
                 std::invalid_argument);
    // Valid input still works afterwards
    const ms::GeodesicLabels ok =
        ms::LabelGeodesicGraph(*d.pmesh, MPI_COMM_WORLD, seeds, false, {});
    EXPECT_EQ(ok.report.unreached_elements, 0);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
