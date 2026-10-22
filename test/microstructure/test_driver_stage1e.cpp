/**
 * @file test_driver_stage1e.cpp
 * @brief Driver-level tests of the Stage 1e distance modes wired into GenerateMicrostructure
 * (microstructure_driver.cpp): Tessellation.distance = "euclidean", "connected" and "geodesic".
 *
 * Covered paths:
 *  1. the U-shaped part in all three modes: a grain crosses the empty notch (owns elements in
 *     both arms that are not connected through the grain) only in euclidean mode;
 *  2. the driver's connected and geodesic labels equal direct calls of the labeling libraries
 *     (LabelTessellation -> ConstrainToConnectedCells -> CompactSeedLabels, and
 *     LabelGeodesicFim / LabelGeodesicGraph -> CompactSeedLabels, for both geodesic solvers), so
 *     the wiring adds nothing and loses nothing;
 *  3. geodesic Johnson-Mehl growth (additive metric, offsets a_i = the nucleation weights);
 *  4. both modes on a box whose notch is a passthrough region: paths stay in the active part, so
 *     the active labels equal those of the U-shaped part alone;
 *  5. a body that no seed reaches: geodesic labels it by straight-line distance and the cleanup
 *     splits it off; connected mode leaves it to the cleanup as an unanchored body; and two
 *     blocks touching only along an edge, which geodesic paths cross (the face-based cleanup
 *     then splits the second block off);
 *  6. the refused combinations (Stage 1e scope, unavailable solver);
 *  7. the Stage 1e example options files.
 *
 * **Fixtures.** Rank 0 writes the seed tables (CSV) into `test_driver_stage1e_data/np<N>/` under
 * the working directory (the build tree when run by ctest; never /tmp), then all ranks meet at a
 * barrier. Meshes are built in memory (grid of unit-spaced hexahedra with some cells removed).
 *
 * **Rank-count independence.** Every driver run is done twice in the same process: distributed
 * over MPI_COMM_WORLD (METIS partition) and alone on every rank with MPI_COMM_SELF (exactly the
 * np = 1 answer). The labels gathered into serial element order, the per-grain tables and the
 * distance-mode counts must agree. Running the executable at np = 1..4 therefore compares np = 2,
 * 3, 4 against np = 1.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/labeling/connected_labeler.hpp"
#include "microstructure/labeling/geodesic_fim.hpp"
#include "microstructure/labeling/geodesic_graph.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_options.hpp"
#include "utilities/auto_mesh.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mpi.h>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

const fs::path data_dir = fs::path(MS_DRIVER_DATA_DIR);

int Rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int NumRanks() {
    int nranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    return nranks;
}

/** Directory of this run's fixtures (one per rank count, so concurrent np runs never collide). */
fs::path DataDir() {
    return fs::path("test_driver_stage1e_data") / ("np" + std::to_string(NumRanks()));
}

// ---------------------------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------------------------

/**
 * Grid of nx x ny x nz hexahedra of edge h, cell (i, j, k) = [i h, (i+1) h] x ..., keeping only
 * the cells for which kept(i, j, k) is true, with element attribute attribute(i, j, k).
 * Lexicographic cell order (x fastest); unused vertices are removed.
 */
mfem::Mesh GridMesh(int nx, int ny, int nz, double h,
                    const std::function<bool(int, int, int)>& kept,
                    const std::function<int(int, int, int)>& attribute) {
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    int cells = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                cells += kept(i, j, k) ? 1 : 0;
            }
        }
    }
    mfem::Mesh mesh(3, (nx + 1) * (ny + 1) * (nz + 1), cells, 0, 3);
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                mesh.AddVertex(i * h, j * h, k * h);
            }
        }
    }
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (kept(i, j, k)) {
                    const int hex[8] = {vid(i, j, k),         vid(i + 1, j, k),
                                        vid(i + 1, j + 1, k), vid(i, j + 1, k),
                                        vid(i, j, k + 1),     vid(i + 1, j, k + 1),
                                        vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)};
                    mesh.AddHex(hex, attribute(i, j, k));
                }
            }
        }
    }
    mesh.FinalizeHexMesh(1, 1, true);
    mesh.RemoveUnusedVertices();
    mesh.SetAttributes();
    return mesh;
}

/** Cells per unit length of the test meshes. */
constexpr int N = 4;

/** True for the cells of the notch [1,2] x [1,2] x [0,1] of the U-shaped part. */
bool InNotch(int i, int j) {
    return i >= N && i < 2 * N && j >= N;
}

/** U-shaped part [0,3] x [0,2] x [0,1] without the notch: two arms joined at y < 1. */
mfem::Mesh UShape() {
    return GridMesh(3 * N, 2 * N, N, 1.0 / N, [](int i, int j, int) { return !InNotch(i, j); },
                    [](int, int, int) { return 1; });
}

/** The full box [0,3] x [0,2] x [0,1]; the notch cells have attribute 2, the U attribute 1. */
mfem::Mesh BoxWithNotchAttribute() {
    return GridMesh(3 * N, 2 * N, N, 1.0 / N, [](int, int, int) { return true; },
                    [](int i, int j, int) { return InNotch(i, j) ? 2 : 1; });
}

/** Two unit cubes [0,1]^3 and [2,3] x [0,1]^2 that do not touch (64 hexahedra each). */
mfem::Mesh TwoBodies() {
    return GridMesh(3 * N, N, N, 1.0 / N, [](int i, int, int) { return i < N || i >= 2 * N; },
                    [](int, int, int) { return 1; });
}

/**
 * Two unit cubes [0,1] x [0,1] x [0,1] and [1,2] x [1,2] x [0,1] (64 hexahedra each) that touch
 * only along the edge x = 1, y = 1 (shared mesh vertices, no shared face).
 */
mfem::Mesh EdgeContactBlocks() {
    return GridMesh(2 * N, 2 * N, N, 1.0 / N,
                    [](int i, int j, int) { return (i < N && j < N) || (i >= N && j >= N); },
                    [](int, int, int) { return 1; });
}

/** Element centers (vertex average) of a serial mesh, 3 per element. */
std::vector<std::array<double, 3>> Centers(mfem::Mesh& mesh) {
    std::vector<std::array<double, 3>> out(static_cast<std::size_t>(mesh.GetNE()));
    mfem::Vector c(3);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.GetElementCenter(e, c);
        out[static_cast<std::size_t>(e)] = {c(0), c(1), c(2)};
    }
    return out;
}

/**
 * Grains that "cross the notch" of the U-shaped part: grains with elements in both upper arms
 * (y > 1, x < 1 and y > 1, x > 2) whose elements are not all connected through shared faces of
 * the grain's own elements. Such a grain jumps across the empty notch. Serial face graph
 * (Mesh::ElementToElementTable) and a breadth-first search per grain; `labels` in serial order.
 */
std::set<int> GrainsCrossingNotch(mfem::Mesh& serial, const std::vector<int>& labels) {
    const std::vector<std::array<double, 3>> center = Centers(serial);
    const mfem::Table& neighbors = serial.ElementToElementTable();
    const int ne = serial.GetNE();
    std::map<int, std::array<bool, 2>> arms; // grain -> (in left arm, in right arm)
    for (int e = 0; e < ne; ++e) {
        const auto& p = center[static_cast<std::size_t>(e)];
        auto& a = arms[labels[static_cast<std::size_t>(e)]];
        if (p[1] > 1.0 && p[0] < 1.0) {
            a[0] = true;
        }
        if (p[1] > 1.0 && p[0] > 2.0) {
            a[1] = true;
        }
    }
    std::set<int> crossing;
    for (const auto& entry : arms) {
        if (!(entry.second[0] && entry.second[1])) {
            continue;
        }
        // Count the face-connected pieces of this grain
        const int g = entry.first;
        std::vector<char> seen(static_cast<std::size_t>(ne), 0);
        int pieces = 0;
        for (int start = 0; start < ne; ++start) {
            const auto si = static_cast<std::size_t>(start);
            if (labels[si] != g || seen[si]) {
                continue;
            }
            ++pieces;
            std::queue<int> queue;
            queue.push(start);
            seen[static_cast<std::size_t>(start)] = 1;
            while (!queue.empty()) {
                const int e = queue.front();
                queue.pop();
                for (int k = neighbors.GetI()[e]; k < neighbors.GetI()[e + 1]; ++k) {
                    const int f = neighbors.GetJ()[k];
                    const auto fi = static_cast<std::size_t>(f);
                    if (labels[fi] == g && !seen[fi]) {
                        seen[fi] = 1;
                        queue.push(f);
                    }
                }
            }
        }
        if (pieces > 1) {
            crossing.insert(g);
        }
    }
    return crossing;
}

// ---------------------------------------------------------------------------------------------
// Seed tables
// ---------------------------------------------------------------------------------------------

/** Writes `text` to DataDir()/name (rank 0 only; see WriteFixtures). */
void WriteText(const std::string& name, const std::string& text) {
    std::ofstream out(DataDir() / name);
    if (!out) {
        throw std::runtime_error("cannot write " + (DataDir() / name).string());
    }
    out << text;
}

/**
 * Writes every fixture once (rank 0), then synchronizes (collective):
 *  - ushape.csv: one seed near the tip of each arm, three along the base, and seed 42 inside the
 *    notch (no material there). Seed 42's straight-line cell is the material next to the notch
 *    in both arms (x in (0.85, 1) and (2, 2.15) above y ~ 1), which touch only across the notch;
 *  - two_bodies.csv: three seeds, all in the first cube of TwoBodies();
 *  - one_seed.csv: a single seed (id 7) in the first block of EdgeContactBlocks().
 */
void WriteFixtures() {
    static bool written = false;
    if (written) {
        return;
    }
    written = true;
    std::string error;
    if (Rank() == 0) {
        try {
            fs::create_directories(DataDir());
            WriteText("ushape.csv",
                      "id,x,y,z\n"
                      "1,0.2,1.5,0.5\n"
                      "2,2.8,1.5,0.5\n"
                      "3,0.4,0.4,0.5\n"
                      "4,1.5,0.45,0.5\n"
                      "5,2.6,0.4,0.5\n"
                      "42,1.5,1.5,0.5\n");
            WriteText("two_bodies.csv",
                      "id,x,y,z\n"
                      "1,0.3,0.3,0.3\n"
                      "2,0.7,0.6,0.5\n"
                      "3,0.4,0.8,0.7\n");
            WriteText("one_seed.csv",
                      "id,x,y,z\n"
                      "7,0.3,0.4,0.5\n");
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    int failed = error.empty() ? 0 : 1;
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (failed != 0) {
        throw std::runtime_error("fixture writing failed on rank 0: " + error);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/** Options reading seed table `name` (source = "file"). */
ms::MicrostructureOptions TableOptions(const std::string& name) {
    ms::MicrostructureOptions options;
    options.seeds.source = "file";
    options.seeds.placement = ms::SeedPlacement::FILE;
    options.seeds.file = DataDir() / name;
    options.seeds.rng_seed = 5u;
    return options;
}

// ---------------------------------------------------------------------------------------------
// Driver runs
// ---------------------------------------------------------------------------------------------

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** Partitions `serial` over `comm` with METIS (collective), keeping local -> serial ids. */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** Per-element values gathered into serial element order and broadcast to every rank. */
std::vector<int> GatherToAll(MPI_Comm comm, const ms::SerialOrder& order,
                             const mfem::Array<int>& local) {
    local.HostRead();
    std::vector<int> all = ms::GatherToSerial(comm, order, local, 0);
    int size = static_cast<int>(all.size());
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);
    all.resize(static_cast<std::size_t>(size));
    MPI_Bcast(all.data(), size, MPI_INT, 0, comm);
    return all;
}

/** One driver run, with the labels in serial element order (on every rank). */
struct DriverRun {
    std::vector<int> labels;
    ms::MicrostructureResult result;
    ms::GenerationReport report;
};

/** GenerateMicrostructure on `serial` distributed over `comm` (collective over comm). */
DriverRun RunDriver(mfem::Mesh& serial, const ms::MicrostructureOptions& options, MPI_Comm comm) {
    Distributed d = Distribute(serial, comm);
    DriverRun run;
    run.result = ms::GenerateMicrostructure(options, *d.pmesh, comm, &run.report);
    run.labels = GatherToAll(comm, d.order, run.result.element_grain);
    run.report.adjacency.reset(); // it refers to d.pmesh, which dies at the end of this call
    return run;
}

/** Distributed run (WORLD) and the single-rank reference (SELF) of the same options. */
struct RunPair {
    DriverRun world;
    DriverRun alone;
};

/**
 * Runs the options over MPI_COMM_WORLD and alone (MPI_COMM_SELF) and checks that the two agree
 * exactly: labels in serial order, per-grain tables, and the distance-mode counts that do not
 * depend on the partition (the geodesic exchange round counts do, so they are not compared).
 * Also checks that the grain ids are compact (1..N, every grain used). Collective over WORLD.
 */
RunPair RunAndCompare(mfem::Mesh& serial, const ms::MicrostructureOptions& options) {
    RunPair pair;
    pair.world = RunDriver(serial, options, MPI_COMM_WORLD);
    pair.alone = RunDriver(serial, options, MPI_COMM_SELF);
    const ms::GenerationReport& w = pair.world.report;
    const ms::GenerationReport& a = pair.alone.report;
    EXPECT_EQ(pair.world.labels, pair.alone.labels);
    EXPECT_EQ(pair.world.result.num_grains, pair.alone.result.num_grains);
    EXPECT_EQ(pair.world.result.grain_quat, pair.alone.result.grain_quat);
    EXPECT_EQ(pair.world.result.grain_source_id, pair.alone.result.grain_source_id);
    EXPECT_EQ(pair.world.result.grain_region, pair.alone.result.grain_region);
    EXPECT_EQ(w.grain_seed, a.grain_seed);
    EXPECT_EQ(w.num_labeled_grains, a.num_labeled_grains);
    EXPECT_EQ(w.num_coincident_seeds, a.num_coincident_seeds);
    EXPECT_EQ(w.num_split_elements, a.num_split_elements);
    EXPECT_EQ(w.cleanup.fragments_found, a.cleanup.fragments_found);
    EXPECT_EQ(w.distance, a.distance);
    EXPECT_EQ(w.used_connected, a.used_connected);
    EXPECT_EQ(w.connected.rounds, a.connected.rounds);
    EXPECT_EQ(w.connected.relabeled_elements, a.connected.relabeled_elements);
    EXPECT_EQ(w.connected.unresolved_elements, a.connected.unresolved_elements);
    EXPECT_EQ(w.connected.unanchored_component_elements, a.connected.unanchored_component_elements);
    EXPECT_EQ(w.connected.seeds_without_cell, a.connected.seeds_without_cell);
    EXPECT_EQ(w.used_geodesic, a.used_geodesic);
    EXPECT_EQ(w.geodesic_solver, a.geodesic_solver);
    EXPECT_EQ(w.geodesic.unreached_vertices, a.geodesic.unreached_vertices);
    EXPECT_EQ(w.geodesic.unreached_elements, a.geodesic.unreached_elements);
    EXPECT_EQ(w.geodesic.split_elements, a.geodesic.split_elements);
    EXPECT_EQ(w.geodesic.seeds_outside_material, a.geodesic.seeds_outside_material);
    EXPECT_EQ(w.num_unreached_elements, a.num_unreached_elements);
    EXPECT_EQ(w.num_unlabeled_elements, a.num_unlabeled_elements);
    EXPECT_EQ(a.num_unlabeled_elements, 0); // no test mesh is degenerate
    // The FIM rounds are synchronous sweeps over the whole mesh, so their counts do not depend on
    // the partition (geodesic_fim.hpp); the graph solver's do (Dijkstra restarts per exchange)
    if (a.used_geodesic && a.geodesic_solver == "fim") {
        EXPECT_EQ(w.geodesic.distance_rounds, a.geodesic.distance_rounds);
        EXPECT_EQ(w.geodesic.label_rounds, a.geodesic.label_rounds);
    }

    // Compact ids: 1..N, every grain used
    const int n = pair.alone.result.num_grains;
    std::vector<int> count(static_cast<std::size_t>(n) + 1, 0);
    for (const int g : pair.alone.labels) {
        EXPECT_GE(g, 1);
        EXPECT_LE(g, n);
        if (g >= 1 && g <= n) {
            count[static_cast<std::size_t>(g)]++;
        }
    }
    for (int g = 1; g <= n; ++g) {
        EXPECT_GT(count[static_cast<std::size_t>(g)], 0) << "grain " << g;
    }
    return pair;
}

/** Calls the driver and returns true if it throws std::invalid_argument (on this rank). */
bool Refused(mfem::ParMesh& pmesh, const ms::MicrostructureOptions& options) {
    try {
        ms::GenerateMicrostructure(options, pmesh, MPI_COMM_WORLD);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

/** Index of the seed with source id `id` in `seeds`, or -1. */
int SeedIndexOf(const ms::SeedSet& seeds, std::int64_t id) {
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        if (seeds.source_id[i] == id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

/** The grains (1-based) that grew from seed index `s`. */
std::vector<int> GrainsOfSeed(const ms::GenerationReport& report, int s) {
    std::vector<int> grains;
    for (std::size_t g = 0; g < report.grain_seed.size(); ++g) {
        if (report.grain_seed[g] == s) {
            grains.push_back(static_cast<int>(g) + 1);
        }
    }
    return grains;
}

/**
 * The connected labels computed directly with the labeling libraries on one rank (MPI_COMM_SELF),
 * in serial element order: straight-line labels, ConstrainToConnectedCells with the options'
 * rounds and sampling, CompactSeedLabels.
 */
std::vector<int> DirectConnectedLabels(mfem::Mesh& serial, const ms::SeedSet& seeds,
                                       const ms::TessellationOptions& tess) {
    Distributed d = Distribute(serial, MPI_COMM_SELF);
    ms::TessellationOptions euclidean = tess;
    euclidean.distance = ms::DistanceMode::EUCLIDEAN;
    const ms::TessellationLabels straight =
        ms::LabelTessellation(*d.pmesh, MPI_COMM_SELF, seeds, euclidean);
    mfem::Array<int> element_seed;
    element_seed = straight.element_seed;
    const ms::MeshAdjacency adjacency(*d.pmesh);
    ms::ConnectedLabelParameters params;
    params.max_rounds = tess.connected_max_rounds;
    params.sampling = tess.sampling;
    params.vote_levels = tess.vote_levels;
    ms::ConstrainToConnectedCells(*d.pmesh, MPI_COMM_SELF, adjacency, seeds, tess, element_seed,
                                  params);
    ms::TessellationLabels labels;
    ms::CompactSeedLabels(MPI_COMM_SELF, element_seed, static_cast<int>(seeds.Size()), labels);
    return GatherToAll(MPI_COMM_SELF, d.order, labels.element_grain);
}

/**
 * The geodesic labels computed directly on one rank, in serial element order: the solver named by
 * tess.geodesic_solver (LabelGeodesicFim for "fim", LabelGeodesicGraph for "graph"), then
 * CompactSeedLabels. Every element must be reached (no body without a seed).
 */
std::vector<int> DirectGeodesicLabels(mfem::Mesh& serial, const ms::SeedSet& seeds,
                                      const ms::TessellationOptions& tess) {
    Distributed d = Distribute(serial, MPI_COMM_SELF);
    ms::GeodesicParameters params;
    params.sampling = tess.sampling;
    params.vote_levels = tess.vote_levels;
    const bool additive = (tess.metric == ms::MetricType::ADDITIVE);
    const ms::GeodesicLabels geodesic =
        (tess.geodesic_solver == "fim")
            ? ms::LabelGeodesicFim(*d.pmesh, MPI_COMM_SELF, seeds, additive, params)
            : ms::LabelGeodesicGraph(*d.pmesh, MPI_COMM_SELF, seeds, additive, params);
    EXPECT_EQ(geodesic.report.unreached_elements, 0);
    ms::TessellationLabels labels;
    ms::CompactSeedLabels(MPI_COMM_SELF, geodesic.element_seed, static_cast<int>(seeds.Size()),
                          labels);
    return GatherToAll(MPI_COMM_SELF, d.order, labels.element_grain);
}

} // namespace

// ---------------------------------------------------------------------------------------------
// 1-2. The U-shaped part in all three modes; driver labels against the libraries
// ---------------------------------------------------------------------------------------------

// ushape.csv puts seed 42 inside the notch. With straight-line distance its cell is the material
// on both sides of the notch, so with fragments = "none" (no repair) grain 42 owns two pieces, one
// in each arm: it crosses the notch. With connected distance the piece not connected to 42's
// anchor element goes to the next best seed; with geodesic distance 42 grows only from its
// (snapped) source element, and the other arm is reached first by the arm-tip seeds. So no grain
// crosses the notch in those modes, and grain 42 lies in one arm only. The driver's labels equal
// the direct library calls (seed order is grain order in both). Seed 42 lies outside the material
// in all modes; geodesic reports it as snapped.
TEST(DriverStage1e, UShapeThreeModes) {
    WriteFixtures();
    mfem::Mesh serial = UShape();
    ms::MicrostructureOptions options = TableOptions("ushape.csv");
    options.cleanup.fragments = "none";

    // Euclidean: grain 42 crosses the notch
    const RunPair euclidean = RunAndCompare(serial, options);
    const int s42 = SeedIndexOf(euclidean.alone.report.seeds, 42);
    EXPECT_EQ(s42, 5);
    const std::set<int> crossing = GrainsCrossingNotch(serial, euclidean.alone.labels);
    const std::vector<int> grains42 = GrainsOfSeed(euclidean.alone.report, s42);
    EXPECT_EQ(grains42.size(), 1u);
    if (grains42.size() == 1u) {
        EXPECT_EQ(crossing.count(grains42[0]), 1u);
    }
    EXPECT_GT(euclidean.alone.report.cleanup.fragments_found, 0);
    EXPECT_FALSE(euclidean.alone.report.used_connected);
    EXPECT_FALSE(euclidean.alone.report.used_geodesic);

    // Connected: no crossing; labels as the direct library calls
    options.tessellation.distance = ms::DistanceMode::CONNECTED;
    const RunPair connected = RunAndCompare(serial, options);
    EXPECT_TRUE(GrainsCrossingNotch(serial, connected.alone.labels).empty());
    EXPECT_TRUE(connected.alone.report.used_connected);
    EXPECT_EQ(connected.alone.report.distance, ms::DistanceMode::CONNECTED);
    EXPECT_GE(connected.alone.report.connected.rounds, 1);
    EXPECT_GT(connected.alone.report.connected.relabeled_elements, 0);
    EXPECT_EQ(connected.alone.report.connected.unresolved_elements, 0);
    EXPECT_EQ(connected.alone.report.connected.unanchored_component_elements, 0);
    EXPECT_EQ(connected.alone.labels,
              DirectConnectedLabels(serial, connected.alone.report.seeds, options.tessellation));

    // Geodesic, both solvers (the default "fim" first): no crossing; labels as the direct library
    // calls
    options.tessellation.distance = ms::DistanceMode::GEODESIC;
    EXPECT_EQ(options.tessellation.geodesic_solver, "fim");
    for (const char* solver : {"fim", "graph"}) {
        SCOPED_TRACE(solver);
        options.tessellation.geodesic_solver = solver;
        const RunPair geodesic = RunAndCompare(serial, options);
        EXPECT_TRUE(GrainsCrossingNotch(serial, geodesic.alone.labels).empty());
        EXPECT_TRUE(geodesic.alone.report.used_geodesic);
        EXPECT_EQ(geodesic.alone.report.geodesic_solver, solver);
        EXPECT_EQ(geodesic.alone.report.geodesic.seeds_outside_material, 1);
        EXPECT_EQ(geodesic.alone.report.num_unreached_elements, 0);
        EXPECT_EQ(geodesic.alone.labels,
                  DirectGeodesicLabels(serial, geodesic.alone.report.seeds, options.tessellation));
    }

    // With the default cleanup the connected and geodesic runs still never cross the notch
    options.cleanup.fragments = "auto";
    for (const auto mode : {ms::DistanceMode::CONNECTED, ms::DistanceMode::GEODESIC}) {
        options.tessellation.distance = mode;
        const RunPair cleaned = RunAndCompare(serial, options);
        EXPECT_TRUE(GrainsCrossingNotch(serial, cleaned.alone.labels).empty());
    }
}

// Quadrature-vote sampling in both modes (the orphan relabeling and the geodesic point labels
// use the 8^vote_levels voting points): the driver still equals the direct library calls and no
// grain crosses the notch. Keep rule "seed" runs with the mode's own anchors.
TEST(DriverStage1e, VoteSamplingAndSeedKeepRule) {
    WriteFixtures();
    mfem::Mesh serial = UShape();
    ms::MicrostructureOptions options = TableOptions("ushape.csv");
    options.cleanup.fragments = "none";
    options.tessellation.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    options.tessellation.vote_levels = 1;

    options.tessellation.distance = ms::DistanceMode::CONNECTED;
    const RunPair connected = RunAndCompare(serial, options);
    EXPECT_TRUE(GrainsCrossingNotch(serial, connected.alone.labels).empty());
    EXPECT_EQ(connected.alone.labels,
              DirectConnectedLabels(serial, connected.alone.report.seeds, options.tessellation));

    options.tessellation.distance = ms::DistanceMode::GEODESIC;
    const RunPair geodesic = RunAndCompare(serial, options);
    EXPECT_TRUE(GrainsCrossingNotch(serial, geodesic.alone.labels).empty());
    EXPECT_EQ(geodesic.alone.labels,
              DirectGeodesicLabels(serial, geodesic.alone.report.seeds, options.tessellation));

    // Keep rule "seed" with splitting: runs on every rank count with the same answer
    options.cleanup.fragments = "split";
    options.cleanup.keep_component = "seed";
    for (const auto mode : {ms::DistanceMode::CONNECTED, ms::DistanceMode::GEODESIC}) {
        options.tessellation.distance = mode;
        const RunPair kept = RunAndCompare(serial, options);
        EXPECT_TRUE(GrainsCrossingNotch(serial, kept.alone.labels).empty());
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Geodesic Johnson-Mehl
// ---------------------------------------------------------------------------------------------

// Johnson-Mehl nuclei on the U-shaped part with geodesic distance: the grains are the cells of
// the additive metric d_path - a_i with a_i = -v t_i (the nucleation weights, all <= 0), measured
// along paths in the material. The driver passes additive = true: for both solvers its labels
// equal a direct solver call with additive = true on the nuclei it reports, and no grain crosses
// the notch.
TEST(DriverStage1e, GeodesicJohnsonMehl) {
    mfem::Mesh serial = UShape();
    ms::MicrostructureOptions options;
    options.seeds.placement = ms::SeedPlacement::JOHNSON_MEHL;
    options.seeds.rng_seed = 3u;
    options.seeds.johnson_mehl.rate = 5.0;
    options.seeds.johnson_mehl.growth_speed = 1.0;
    options.tessellation.metric = ms::MetricType::ADDITIVE;
    options.tessellation.distance = ms::DistanceMode::GEODESIC;
    options.cleanup.fragments = "none";
    for (const char* solver : {"fim", "graph"}) {
        SCOPED_TRACE(solver);
        options.tessellation.geodesic_solver = solver;
        const RunPair runs = RunAndCompare(serial, options);
        const ms::GenerationReport& report = runs.alone.report;
        EXPECT_TRUE(report.used_johnson_mehl);
        EXPECT_TRUE(report.used_geodesic);
        EXPECT_GT(report.seeds.Size(), 2u);
        EXPECT_EQ(report.seeds.weight.size(), report.seeds.Size());
        for (const double a : report.seeds.weight) {
            EXPECT_LE(a, 0.0);
        }
        EXPECT_EQ(runs.alone.labels,
                  DirectGeodesicLabels(serial, report.seeds, options.tessellation));
        EXPECT_TRUE(GrainsCrossingNotch(serial, runs.alone.labels).empty());
        // The nucleation limitation is reported as a warning
        bool warned = false;
        for (const std::string& w : report.warnings) {
            warned = warned || w.find("Johnson-Mehl") != std::string::npos;
        }
        EXPECT_TRUE(warned);
    }
}

// ---------------------------------------------------------------------------------------------
// 4. Passthrough notch: paths stay in the active part
// ---------------------------------------------------------------------------------------------

// The full box with the notch cells as a passthrough region (attribute 2 -> region 2). The
// connected and geodesic modes run on the active sub-mesh, i.e. exactly the U-shaped part, so
// neither paths nor connections may pass through the notch: the labels of the active elements
// equal those of the U-shaped part alone (matched by element center), and the notch is the extra
// last grain. (Seed 42 lies inside the passthrough notch, i.e. outside the active material, in
// both runs.)
TEST(DriverStage1e, PassthroughNotchKeepsPathsInActivePart) {
    WriteFixtures();
    mfem::Mesh box = BoxWithNotchAttribute();
    mfem::Mesh ushape = UShape();
    // Element of the U-shaped part at each center
    const std::vector<std::array<double, 3>> ucenter = Centers(ushape);
    std::map<std::array<long long, 3>, int> u_at;
    auto key = [](const std::array<double, 3>& p) {
        return std::array<long long, 3>{std::llround(p[0] * 1e6), std::llround(p[1] * 1e6),
                                        std::llround(p[2] * 1e6)};
    };
    for (std::size_t e = 0; e < ucenter.size(); ++e) {
        u_at[key(ucenter[e])] = static_cast<int>(e);
    }
    const std::vector<std::array<double, 3>> bcenter = Centers(box);

    for (const auto mode : {ms::DistanceMode::CONNECTED, ms::DistanceMode::GEODESIC}) {
        SCOPED_TRACE(mode == ms::DistanceMode::CONNECTED ? "connected" : "geodesic");
        ms::MicrostructureOptions options = TableOptions("ushape.csv");
        options.cleanup.fragments = "none";
        options.tessellation.distance = mode;
        const RunPair alone_u = RunAndCompare(ushape, options);
        options.mesh.passthrough_attributes = {{2, 2}};
        const RunPair with_plate = RunAndCompare(box, options);
        const ms::GenerationReport& report = with_plate.alone.report;
        EXPECT_TRUE(report.used_submesh);
        EXPECT_EQ(report.num_active_elements, ushape.GetNE());
        EXPECT_EQ(with_plate.alone.result.num_grains, alone_u.alone.result.num_grains + 1);
        const int notch_grain = with_plate.alone.result.num_grains;
        int compared = 0;
        for (std::size_t e = 0; e < bcenter.size(); ++e) {
            const int label = with_plate.alone.labels[e];
            const auto it = u_at.find(key(bcenter[e]));
            if (it == u_at.end()) {
                EXPECT_EQ(label, notch_grain) << "notch element " << e;
                continue;
            }
            EXPECT_EQ(label, alone_u.alone.labels[static_cast<std::size_t>(it->second)]) << e;
            ++compared;
        }
        EXPECT_EQ(compared, ushape.GetNE());
    }
}

// ---------------------------------------------------------------------------------------------
// 5. A body that no seed reaches
// ---------------------------------------------------------------------------------------------

// Two separate cubes with all three seeds in the first. Geodesic: no path reaches the second
// cube, so its 64 elements are unreached; they take their straight-line nearest-seed labels and
// the cleanup (fragments = "auto", which merges for seed tables) splits them off, since an
// isolated body has no neighbor to merge into. Connected: the second cube holds no seed's anchor,
// so its 64 elements are exempt (unanchored) and the cleanup splits them off the same way. Either
// way no grain owns elements in both cubes, the second cube's grains descend from the seeds, and
// every seed keeps its own grain (source id = its table id) in the first cube: in these modes the
// driver always keeps the piece holding the seed's anchor / source element, so the larger
// leftover cube cannot take over a seed's identity (with keep rule "largest" it would).
TEST(DriverStage1e, UnreachedBodySplitOff) {
    WriteFixtures();
    mfem::Mesh serial = TwoBodies();
    const std::vector<std::array<double, 3>> center = Centers(serial);
    for (const auto mode : {ms::DistanceMode::GEODESIC, ms::DistanceMode::CONNECTED}) {
        const bool geodesic = (mode == ms::DistanceMode::GEODESIC);
        SCOPED_TRACE(geodesic ? "geodesic" : "connected");
        ms::MicrostructureOptions options = TableOptions("two_bodies.csv");
        options.tessellation.distance = mode;
        const RunPair runs = RunAndCompare(serial, options);
        const ms::GenerationReport& report = runs.alone.report;
        if (geodesic) {
            EXPECT_EQ(report.num_unreached_elements, 64);
            EXPECT_EQ(report.geodesic.unreached_elements, 64);
        } else {
            EXPECT_EQ(report.connected.unanchored_component_elements, 64);
            EXPECT_EQ(report.num_unreached_elements, 0);
        }
        EXPECT_GE(report.cleanup.fragments_split, 1) << report.cleanup.Summary();
        std::set<int> first, second;
        for (std::size_t e = 0; e < center.size(); ++e) {
            (center[e][0] < 1.5 ? first : second).insert(runs.alone.labels[e]);
        }
        for (const int g : second) {
            EXPECT_EQ(first.count(g), 0u) << "grain " << g << " spans both bodies";
            const int s = report.grain_seed[static_cast<std::size_t>(g - 1)];
            EXPECT_GE(s, 0);
            EXPECT_LT(s, 3);
        }
        // Every seed keeps its own grain (its table id, not a derived one) in the first cube:
        // the leftover body never takes over a seed's identity, although it is larger than any
        // seed's grain
        for (const std::int64_t id : {1, 2, 3}) {
            const auto& ids = runs.alone.result.grain_source_id;
            const auto it = std::find(ids.begin(), ids.end(), id);
            EXPECT_NE(it, ids.end()) << "seed " << id;
            if (it != ids.end()) {
                EXPECT_EQ(first.count(static_cast<int>(it - ids.begin()) + 1), 1u) << id;
            }
        }
    }
}

// Two blocks that touch only along one edge, one seed in the first block. The geodesic solvers
// connect elements through shared vertices (the solvers work on mesh vertices), so the path from
// the seed reaches the second block through the shared edge: no element is unreached and every
// element gets the seed's label. The cleanup is face based, so the second block is a separate
// fragment of that grain; it has no face neighbor to merge into, so even "merge" (the "auto"
// policy for seed tables) splits it off into a new grain with a derived source id, as documented.
// With fragments = "none" the run keeps one grain with one extra fragment.
TEST(DriverStage1e, GeodesicThroughEdgeContact) {
    WriteFixtures();
    mfem::Mesh serial = EdgeContactBlocks();
    const std::vector<std::array<double, 3>> center = Centers(serial);
    for (const char* solver : {"fim", "graph"}) {
        SCOPED_TRACE(solver);
        ms::MicrostructureOptions options = TableOptions("one_seed.csv");
        options.tessellation.distance = ms::DistanceMode::GEODESIC;
        options.tessellation.geodesic_solver = solver;
        const RunPair runs = RunAndCompare(serial, options);
        const ms::GenerationReport& report = runs.alone.report;
        EXPECT_EQ(report.num_unreached_elements, 0);
        EXPECT_EQ(report.geodesic.unreached_elements, 0);
        EXPECT_EQ(report.num_labeled_grains, 1);
        EXPECT_EQ(report.cleanup.fragments_found, 1);
        EXPECT_EQ(report.cleanup.fragments_split, 1);
        EXPECT_EQ(runs.alone.result.num_grains, 2);
        // The block holding the seed keeps the seed's identity; the other block is the new grain
        for (std::size_t e = 0; e < center.size(); ++e) {
            EXPECT_EQ(runs.alone.labels[e], center[e][0] < 1.0 ? 1 : 2) << e;
        }
        EXPECT_EQ(runs.alone.result.grain_source_id[0], 7);
        EXPECT_NE(runs.alone.result.grain_source_id[1], 7);
        EXPECT_EQ(report.grain_seed, (std::vector<int>{0, 0}));

        options.cleanup.fragments = "none";
        const RunPair kept = RunAndCompare(serial, options);
        EXPECT_EQ(kept.alone.result.num_grains, 1);
        EXPECT_EQ(kept.alone.report.cleanup.fragments_found, 1);
    }
}

// ---------------------------------------------------------------------------------------------
// 6. Refused combinations
// ---------------------------------------------------------------------------------------------

// The driver refuses (std::invalid_argument on every rank, before any collective work) every
// combination outside the Stage 1e scope: periodic axes, Lloyd, the volume fit and voxel maps with
// either mode; geodesic with Laguerre / Quadric or with an anisotropy; a rounds cap outside
// 0..CONNECTED_MAX_ROUNDS_LIMIT; an
// unknown solver (in any mode).
TEST(DriverStage1e, RefusedCombinations) {
    mfem::Mesh serial = exaconstit::MakeAutoHexMesh({4, 4, 4}, {1.0, 1.0, 1.0});
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    using O = ms::MicrostructureOptions;
    auto refused = [&](const O& options) { return Refused(*d.pmesh, options); };
    O base;
    base.seeds.num_seeds = 5;
    base.seeds.rng_seed = 2u;

    for (const auto mode : {ms::DistanceMode::CONNECTED, ms::DistanceMode::GEODESIC}) {
        SCOPED_TRACE(mode == ms::DistanceMode::CONNECTED ? "connected" : "geodesic");
        O o = base;
        o.tessellation.distance = mode;
        O periodic = o;
        periodic.tessellation.periodic = {true, false, false};
        EXPECT_TRUE(refused(periodic));
        O lloyd = o;
        lloyd.tessellation.lloyd_iterations = 1;
        EXPECT_TRUE(refused(lloyd));
        O fit = o;
        fit.tessellation.metric = ms::MetricType::ADDITIVE;
        fit.tessellation.fit_volumes.enabled = true;
        EXPECT_TRUE(refused(fit));
        O voxel = o;
        voxel.source.type = "voxel_map";
        voxel.voxel_map.file = "any.h5";
        EXPECT_TRUE(refused(voxel));
    }

    O geodesic = base;
    geodesic.tessellation.distance = ms::DistanceMode::GEODESIC;
    O laguerre = geodesic;
    laguerre.tessellation.metric = ms::MetricType::LAGUERRE;
    EXPECT_TRUE(refused(laguerre));
    O quadric = geodesic;
    quadric.tessellation.metric = ms::MetricType::QUADRIC;
    EXPECT_TRUE(refused(quadric));
    O tensor = geodesic;
    tensor.tessellation.anisotropy_kind = "tensor";
    EXPECT_TRUE(refused(tensor));
    O unknown = geodesic;
    unknown.tessellation.geodesic_solver = "dijkstra";
    EXPECT_TRUE(refused(unknown));
    // A value invalid in itself is refused in any mode, as by validate()
    O typo = base;
    typo.tessellation.geodesic_solver = "x";
    EXPECT_TRUE(refused(typo));

    O rounds = base;
    rounds.tessellation.distance = ms::DistanceMode::CONNECTED;
    rounds.tessellation.connected_max_rounds = -1;
    EXPECT_TRUE(refused(rounds));
    rounds.tessellation.connected_max_rounds = ms::CONNECTED_MAX_ROUNDS_LIMIT + 1;
    EXPECT_TRUE(refused(rounds));

    // Direct callers of the straight-line labeler cannot get silently wrong labels
    O direct = base;
    direct.tessellation.distance = ms::DistanceMode::CONNECTED;
    ms::SeedSet seeds;
    seeds.x = {0.25, 0.75};
    seeds.y = {0.5, 0.5};
    seeds.z = {0.5, 0.5};
    seeds.source_id = {0, 1};
    EXPECT_THROW(ms::LabelTessellation(*d.pmesh, MPI_COMM_WORLD, seeds, direct.tessellation),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// 7. Example options files
// ---------------------------------------------------------------------------------------------

// The Stage 1e examples (all on u_shape_hex.mesh, whose path is relative to the directory the
// tool runs in) validate, run through the driver and give the same result on every rank count;
// each runs the mode it names and no grain crosses the notch.
TEST(DriverStage1e, ExampleOptionsFilesRun) {
    const fs::path dir = data_dir / "microstructure";
    for (const char* name : {"connected_ushape.toml", "geodesic_ushape.toml",
                             "geodesic_johnson_mehl_ushape.toml"}) {
        SCOPED_TRACE(name);
        ms::MicrostructureOptions options =
            ms::MicrostructureOptions::parse_options_file(dir / name);
        options.mesh.floc = dir / options.mesh.floc;
        EXPECT_TRUE(options.validate());
        mfem::Mesh serial(options.mesh.floc.string().c_str(), 1, 1, true);
        const RunPair runs = RunAndCompare(serial, options);
        EXPECT_GT(runs.alone.result.num_grains, 1);
        EXPECT_EQ(runs.alone.report.distance, options.tessellation.distance);
        EXPECT_EQ(runs.alone.report.used_connected,
                  options.tessellation.distance == ms::DistanceMode::CONNECTED);
        EXPECT_EQ(runs.alone.report.used_geodesic,
                  options.tessellation.distance == ms::DistanceMode::GEODESIC);
        EXPECT_TRUE(GrainsCrossingNotch(serial, runs.alone.labels).empty());
    }
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
