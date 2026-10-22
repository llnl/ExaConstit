/**
 * @file test_driver_stage1bc.cpp
 * @brief Driver-level tests of the Stage 1b/1c features wired into GenerateMicrostructure
 * (microstructure_driver.cpp) and of the shared utilities added with them.
 *
 * Covered paths: fragment cleanup (split, merge, small grains, keep the seed's piece) on a
 * non-convex U-shaped part; quadrature-vote labeling; periodic microstructures (periodic seed
 * images in the labeler and periodic face pairing in the cleanup adjacency); log-normal size
 * targets with polydisperse placement and a Laguerre volume fit; Lloyd relaxation and centroidal
 * power mode; columnar grains; Johnson-Mehl nucleation; the example options files of these
 * features; and the shared helpers (collective errors, polydisperse RSA in the seed generator,
 * the IntegrationRule sampler overload).
 *
 * **Rank-count independence.** Every driver run is done twice in the same process: distributed
 * over MPI_COMM_WORLD (METIS partition, or a scattered partition whose last rank is empty), and
 * alone on every rank with MPI_COMM_SELF, which is exactly the np = 1 answer. The labels are
 * gathered into serial element order and must be identical, as must the grain count, the
 * orientations (bitwise) and the source ids. Running the executable at np = 1..4
 * (test_driver_stage1bc.cmake) therefore compares np = 2, 3, 4 against np = 1.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever in that
 * collective call.
 */
#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/seeding/johnson_mehl.hpp"
#include "microstructure/seeding/seed_generators.hpp"
#include "microstructure/util/collective_errors.hpp"
#include "utilities/auto_mesh.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>
#include <iostream>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mpi.h>
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

/** ExaConstit's auto box: nx x ny x nz hexes of the box [0, Lx] x [0, Ly] x [0, Lz]. */
mfem::Mesh AutoHex(int nx, int ny, int nz, double lx = 1.0, double ly = 1.0, double lz = 1.0) {
    return exaconstit::MakeAutoHexMesh({nx, ny, nz}, {lx, ly, lz});
}

/**
 * A structured hex mesh of nx x ny x nz cells of size h, keeping only the cells for which
 * kept(i, j, k) is true. cell[e] is the (i, j, k) of element e.
 */
struct GridMesh {
    mfem::Mesh mesh;
    std::vector<std::array<int, 3>> cell;
};

template <typename Kept>
GridMesh BuildHexGrid(int nx, int ny, int nz, double h, Kept kept) {
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    int num_cells = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                num_cells += kept(i, j, k) ? 1 : 0;
            }
        }
    }
    GridMesh grid{mfem::Mesh(3, (nx + 1) * (ny + 1) * (nz + 1), num_cells, 0, 3), {}};
    for (int k = 0; k <= nz; k++) {
        for (int j = 0; j <= ny; j++) {
            for (int i = 0; i <= nx; i++) {
                grid.mesh.AddVertex(i * h, j * h, k * h);
            }
        }
    }
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (!kept(i, j, k)) {
                    continue;
                }
                const int hex[8] = {vid(i, j, k),         vid(i + 1, j, k),
                                    vid(i + 1, j + 1, k), vid(i, j + 1, k),
                                    vid(i, j, k + 1),     vid(i + 1, j, k + 1),
                                    vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)};
                grid.mesh.AddHex(hex);
                grid.cell.push_back({i, j, k});
            }
        }
    }
    grid.mesh.FinalizeHexMesh(1, 1, true);
    grid.mesh.RemoveUnusedVertices();
    return grid;
}

/**
 * U-shaped part with n cells per unit length: [0,3] x [0,2] x [0,1] without the notch
 * [1,2] x [1,2] x [0,1], i.e. two arms (x < 1 and x > 2) joined at the bottom (y < 1). A
 * Voronoi cell of a seed near the notch can reach across the empty slot, so its elements form
 * two pieces that touch only through the void: a real fragment.
 */
GridMesh UShape(int n) {
    return BuildHexGrid(3 * n, 2 * n, n, 1.0 / n, [n](int i, int j, int) {
        return !(i >= n && i < 2 * n && j >= n);
    });
}

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

/**
 * A scattered partition whose last rank owns no element (np >= 2): elements are dealt
 * round-robin to ranks 0 .. nranks-2, which gives many shared faces and exercises the empty-rank
 * branches of every collective step. Collective.
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

/** GenerateMicrostructure on `serial` distributed over `comm`. Collective over comm. */
DriverRun RunDriver(mfem::Mesh& serial, const ms::MicrostructureOptions& options, MPI_Comm comm,
              bool empty_last_rank = false) {
    Distributed d = empty_last_rank ? DistributeWithEmptyLastRank(serial, comm)
                                    : Distribute(serial, comm);
    DriverRun run;
    run.result = ms::GenerateMicrostructure(options, *d.pmesh, comm, &run.report);
    run.labels = GatherToAll(comm, d.order, run.result.element_grain);
    return run;
}

/** Distributed run (WORLD) and the single-rank reference (SELF) of the same options. */
struct RunPair {
    DriverRun world;
    DriverRun alone;
};

/**
 * Runs the options over MPI_COMM_WORLD and alone (MPI_COMM_SELF) and checks that the two agree
 * exactly: labels in serial order, grain count, orientations (bitwise), source ids, and the
 * report counts of seeding and cleanup. Collective over MPI_COMM_WORLD.
 */
RunPair RunAndCompare(mfem::Mesh& serial, const ms::MicrostructureOptions& options,
                      bool empty_last_rank = false) {
    RunPair pair;
    pair.world = RunDriver(serial, options, MPI_COMM_WORLD, empty_last_rank);
    pair.alone = RunDriver(serial, options, MPI_COMM_SELF);
    EXPECT_EQ(pair.world.labels, pair.alone.labels);
    EXPECT_EQ(pair.world.result.num_grains, pair.alone.result.num_grains);
    EXPECT_EQ(pair.world.result.grain_quat, pair.alone.result.grain_quat);
    EXPECT_EQ(pair.world.result.grain_source_id, pair.alone.result.grain_source_id);
    EXPECT_EQ(pair.world.report.num_seeds, pair.alone.report.num_seeds);
    EXPECT_EQ(pair.world.report.num_labeled_grains, pair.alone.report.num_labeled_grains);
    EXPECT_EQ(pair.world.report.cleanup.fragments_found, pair.alone.report.cleanup.fragments_found);
    EXPECT_EQ(pair.world.report.cleanup.fragments_split, pair.alone.report.cleanup.fragments_split);
    EXPECT_EQ(pair.world.report.cleanup.fragments_merged,
              pair.alone.report.cleanup.fragments_merged);
    EXPECT_EQ(pair.world.report.num_split_elements, pair.alone.report.num_split_elements);
    EXPECT_EQ(pair.world.report.grain_seed, pair.alone.report.grain_seed);
    // Grain ids are compact: 1..N, every grain used
    std::vector<int> count(static_cast<std::size_t>(pair.alone.result.num_grains) + 1, 0);
    for (const int g : pair.alone.labels) {
        EXPECT_GE(g, 1);
        EXPECT_LE(g, pair.alone.result.num_grains);
        if (g >= 1 && g <= pair.alone.result.num_grains) {
            count[static_cast<std::size_t>(g)]++;
        }
    }
    for (int g = 1; g <= pair.alone.result.num_grains; ++g) {
        EXPECT_GT(count[static_cast<std::size_t>(g)], 0) << "grain " << g;
    }
    return pair;
}

/** Quick-path options: `n` random Voronoi seeds with the given RNG key. */
ms::MicrostructureOptions QuickOptions(std::int64_t n, std::uint64_t rng_seed) {
    ms::MicrostructureOptions options;
    options.seeds.num_seeds = n;
    options.seeds.rng_seed = rng_seed;
    return options;
}

/**
 * Number of face-connected pieces of the serial labels on `serial` (connected components of
 * equal labels through shared faces, optionally also through matching periodic faces of the
 * mesh box). Computed alone on every rank (MPI_COMM_SELF).
 */
std::size_t CountPieces(mfem::Mesh& serial, const std::vector<int>& labels, bool periodic_pairs,
                        const std::array<bool, 3>& periodic = {true, true, true}) {
    Distributed alone = Distribute(serial, MPI_COMM_SELF);
    std::unique_ptr<ms::MeshAdjacency> adjacency;
    if (periodic_pairs) {
        ms::PeriodicFaceParameters pairing;
        pairing.periodic = periodic;
        double lo[3] = {1e300, 1e300, 1e300};
        double hi[3] = {-1e300, -1e300, -1e300};
        for (int v = 0; v < serial.GetNV(); ++v) {
            for (int d = 0; d < 3; ++d) {
                lo[d] = std::min(lo[d], serial.GetVertex(v)[d]);
                hi[d] = std::max(hi[d], serial.GetVertex(v)[d]);
            }
        }
        pairing.lo = {lo[0], lo[1], lo[2]};
        pairing.hi = {hi[0], hi[1], hi[2]};
        adjacency = std::make_unique<ms::MeshAdjacency>(*alone.pmesh, pairing);
    } else {
        adjacency = std::make_unique<ms::MeshAdjacency>(*alone.pmesh);
    }
    mfem::Array<int> local(static_cast<int>(labels.size()));
    for (int e = 0; e < local.Size(); ++e) {
        local[e] = labels[static_cast<std::size_t>(alone.order.local_to_serial[e])];
    }
    std::vector<std::uint64_t> component;
    ms::LabelComponents(*adjacency, local, ms::ComputeElementKeys(*alone.pmesh), component);
    return std::set<std::uint64_t>(component.begin(), component.end()).size();
}

/** Element count of every grain (index g-1) of serial labels with grains 1..num_grains. */
std::vector<int> GrainSizes(const std::vector<int>& labels, int num_grains) {
    std::vector<int> size(static_cast<std::size_t>(num_grains), 0);
    for (const int g : labels) {
        if (g >= 1 && g <= num_grains) {
            size[static_cast<std::size_t>(g - 1)]++;
        }
    }
    return size;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Fragment cleanup through the driver
// ---------------------------------------------------------------------------------------------

// Seeds anywhere in the bounding box of the U-shaped part (sampling_domain = "bbox"): Voronoi
// cells near the notch reach across the empty slot, so some grains come out in two pieces.
// With fragments = "split" every extra piece becomes its own grain: afterwards every grain is
// exactly one face-connected piece (pieces == grains), the grain count grew by the number of
// split fragments, the split grains carry derived (negative) source ids, and their orientations
// differ from their parent's. The result is identical on every rank count, also on a scattered
// partition with an empty rank.
TEST(DriverStage1bc, FragmentSplitOnUShapedPart) {
    GridMesh grid = UShape(4);
    ms::MicrostructureOptions options = QuickOptions(14, 21u);
    options.seeds.sampling_domain = "bbox";
    options.cleanup.fragments = "split";
    for (const bool empty_last_rank : {false, true}) {
        SCOPED_TRACE(empty_last_rank ? "empty last rank" : "METIS partition");
        const RunPair runs = RunAndCompare(grid.mesh, options, empty_last_rank);
        const DriverRun& run = runs.alone;
        EXPECT_GT(run.report.cleanup.fragments_found, 0);
        EXPECT_EQ(run.report.cleanup.fragments_split, run.report.cleanup.fragments_found);
        EXPECT_EQ(static_cast<std::int64_t>(run.result.num_grains),
                  run.report.num_labeled_grains + run.report.cleanup.fragments_split);
        EXPECT_EQ(CountPieces(grid.mesh, run.labels, false),
                  static_cast<std::size_t>(run.result.num_grains));
        int derived = 0;
        for (const std::int64_t id : run.result.grain_source_id) {
            derived += (id < 0) ? 1 : 0;
        }
        EXPECT_EQ(derived, run.report.cleanup.fragments_split);
    }
}

// fragments = "merge" with min_grain_elements = 4 on the same part: the extra pieces join a
// neighboring grain instead, so the grain count never exceeds the labeled count, every grain is
// one piece, and no grain is smaller than 4 elements unless the cleanup reports it as kept.
TEST(DriverStage1bc, FragmentMergeAndSmallGrains) {
    GridMesh grid = UShape(4);
    ms::MicrostructureOptions options = QuickOptions(14, 21u);
    options.seeds.sampling_domain = "bbox";
    options.cleanup.fragments = "merge";
    options.cleanup.min_grain_elements = 4;
    const RunPair runs = RunAndCompare(grid.mesh, options);
    const DriverRun& run = runs.alone;
    EXPECT_GT(run.report.cleanup.fragments_merged, 0);
    EXPECT_LE(run.result.num_grains, run.report.num_labeled_grains);
    EXPECT_EQ(CountPieces(grid.mesh, run.labels, false),
              static_cast<std::size_t>(run.result.num_grains));
    int small = 0;
    for (const int size : GrainSizes(run.labels, run.result.num_grains)) {
        small += (size < 4) ? 1 : 0;
    }
    EXPECT_EQ(small, run.report.cleanup.small_grains_kept);
    // Merging never invents source ids: every grain keeps a seed's id
    for (const std::int64_t id : run.result.grain_source_id) {
        EXPECT_GE(id, 0);
    }
}

// keep_component = "seed": for every grain whose seed lies in an element that the plain
// labeling (fragments = "none") gave to that grain, the element containing the seed must still
// belong to the grain with the seed's own source id after splitting, i.e. the seed's piece kept
// the identity (and orientation), and the other pieces were split off. Seeds are found with the
// element-key map of the structured grid (cell = floor(p / h)).
TEST(DriverStage1bc, KeepSeedComponent) {
    const int n = 4;
    GridMesh grid = UShape(n);
    ms::MicrostructureOptions plain = QuickOptions(14, 21u);
    plain.seeds.sampling_domain = "bbox";
    plain.cleanup.fragments = "none";
    ms::MicrostructureOptions keep_seed = plain;
    keep_seed.cleanup.fragments = "split";
    keep_seed.cleanup.keep_component = "seed";

    const RunPair none_runs = RunAndCompare(grid.mesh, plain);
    const RunPair seed_runs = RunAndCompare(grid.mesh, keep_seed);
    const DriverRun& none_run = none_runs.alone;
    const DriverRun& seed_run = seed_runs.alone;
    EXPECT_GT(seed_run.report.cleanup.fragments_split, 0);

    // The seeds themselves (same parameters as the driver)
    Distributed alone = Distribute(grid.mesh, MPI_COMM_SELF);
    const ms::SeedSet seeds =
        ms::GenerateSeeds(*alone.pmesh, ms::MakeSeedingParameters(keep_seed));
    std::map<std::array<int, 3>, int> element_of_cell;
    for (std::size_t e = 0; e < grid.cell.size(); ++e) {
        element_of_cell[grid.cell[e]] = static_cast<int>(e);
    }
    int checked = 0;
    for (int g = 1; g <= none_run.result.num_grains; ++g) {
        const int s = none_run.report.grain_seed[static_cast<std::size_t>(g - 1)];
        const std::array<int, 3> cell = {static_cast<int>(std::floor(seeds.x[s] * n)),
                                         static_cast<int>(std::floor(seeds.y[s] * n)),
                                         static_cast<int>(std::floor(seeds.z[s] * n))};
        const auto it = element_of_cell.find(cell);
        if (it == element_of_cell.end()) {
            continue; // the seed lies in the notch (bbox sampling)
        }
        const int e = it->second;
        if (none_run.labels[static_cast<std::size_t>(e)] != g) {
            continue; // the seed's element belongs to another grain
        }
        const int kept = seed_run.labels[static_cast<std::size_t>(e)];
        EXPECT_EQ(seed_run.result.grain_source_id[static_cast<std::size_t>(kept - 1)],
                  seeds.source_id[static_cast<std::size_t>(s)])
            << "grain " << g;
        ++checked;
    }
    EXPECT_GT(checked, 0);
}

// The default fragments = "auto" (split large pieces, merge tiny ones) on the U-shaped part,
// with 8 seeds anywhere in its bounding box (rng_seed 11, chosen because its Voronoi cells give
// both kinds of piece): one grain reaches across the empty slot with a large piece, which is
// still split into a grain of its own (at least min_fragment_elements = 8 elements), and one
// tiny piece is merged ("split" would have split both). Every grain is one piece afterwards.
TEST(DriverStage1bc, AutoStillSplitsLargeLatticeGapPieces) {
    GridMesh grid = UShape(4);
    ms::MicrostructureOptions options = QuickOptions(8, 11u);
    options.seeds.sampling_domain = "bbox";
    const RunPair runs = RunAndCompare(grid.mesh, options);
    const DriverRun& run = runs.alone;
    EXPECT_EQ(run.report.cleanup.fragments_found, 2);
    EXPECT_EQ(run.report.cleanup.fragments_split, 1);
    EXPECT_EQ(run.report.cleanup.fragments_merged, 1);
    EXPECT_EQ(CountPieces(grid.mesh, run.labels, false),
              static_cast<std::size_t>(run.result.num_grains));
    const std::vector<int> sizes = GrainSizes(run.labels, run.result.num_grains);
    for (int g = 0; g < run.result.num_grains; ++g) {
        if (run.result.grain_source_id[static_cast<std::size_t>(g)] < 0) { // split off
            EXPECT_GE(sizes[static_cast<std::size_t>(g)], 8) << "grain " << g + 1;
        }
    }

    ms::MicrostructureOptions split = options;
    split.cleanup.fragments = "split";
    EXPECT_EQ(RunAndCompare(grid.mesh, split).alone.report.cleanup.fragments_split, 2);
}

// The default cleanup on the end-to-end example (cube_tet, 20 random Voronoi seeds, centroid
// labels): 6 single elements touch the rest of their grain only along an edge (measured). They
// are tiny (1 < 8 elements), so "auto" merges all of them and the run ends with the 20 seed
// grains, each one face-connected piece; "split" would have made 26 grains of them.
TEST(DriverStage1bc, AutoMergesEdgeConnectedArtifacts) {
    mfem::Mesh serial((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
    const ms::MicrostructureOptions options = QuickOptions(20, 42u);
    const RunPair runs = RunAndCompare(serial, options);
    const DriverRun& run = runs.alone;
    EXPECT_EQ(run.report.cleanup.fragments_found, 6);
    EXPECT_EQ(run.report.cleanup.fragments_merged, 6);
    EXPECT_EQ(run.report.cleanup.fragments_split, 0);
    EXPECT_EQ(run.result.num_grains, 20);
    EXPECT_EQ(CountPieces(serial, run.labels, false), 20u);

    ms::MicrostructureOptions split = options;
    split.cleanup.fragments = "split";
    const RunPair split_runs = RunAndCompare(serial, split);
    EXPECT_EQ(split_runs.alone.result.num_grains, 26);
}

// SPLIT_LARGE_MERGE_TINY directly, on a constructed labeling of a 10 x 4 x 4 hex box (cells of
// size 0.1 along x): grain 1 = the slab of cells i < 3 (48 elements) plus the slab i >= 6 (64
// elements) plus the single cell (4, 0, 0); grain 2 = the rest (cells 3 <= i < 6 without the
// single cell, one connected piece). Grain 1 thus has three pieces of 64, 48 and 1 elements (113
// in total); the 64-piece is kept (largest). With the defaults (8 elements, 10%) the 48-piece is
// large and split off, the single cell is tiny and merges into grain 2 (its only neighbor): 3
// grains. With min_fragment_elements = 0 and min_fragment_fraction = 0.5 the 48-piece is tiny by
// the fraction rule (48 < 0.5 x 113) and merges into grain 2 too: 2 grains. Both are identical
// for every rank count and on a partition with an empty rank.
TEST(DriverStage1bc, SplitLargeMergeTinyPolicy) {
    mfem::Mesh serial = AutoHex(10, 4, 4);
    auto label_of = [](mfem::Mesh& mesh, int e) {
        mfem::Vector c(3);
        mesh.GetElementCenter(e, c);
        const int i = static_cast<int>(std::floor(c[0] * 10));
        const int j = static_cast<int>(std::floor(c[1] * 4));
        const int k = static_cast<int>(std::floor(c[2] * 4));
        const bool grain1 = i < 3 || i >= 6 || (i == 4 && j == 0 && k == 0);
        return grain1 ? 1 : 2;
    };
    struct Expect {
        int min_elements;
        double fraction;
        int grains;
        std::int64_t split;
        std::int64_t merged;
    };
    for (const Expect& x : {Expect{8, 0.1, 3, 1, 1}, Expect{0, 0.5, 2, 0, 2}}) {
        SCOPED_TRACE("min_fragment_fraction " + std::to_string(x.fraction));
        ms::FragmentCleanupParameters params;
        params.policy = ms::FragmentPolicy::SPLIT_LARGE_MERGE_TINY;
        params.min_fragment_elements = x.min_elements;
        params.min_fragment_fraction = x.fraction;
        std::vector<std::vector<int>> results;
        for (const int mode : {0, 1, 2}) { // WORLD (METIS), WORLD (empty last rank), SELF
            const MPI_Comm comm = (mode == 2) ? MPI_COMM_SELF : MPI_COMM_WORLD;
            Distributed d = (mode == 1) ? DistributeWithEmptyLastRank(serial, comm)
                                        : Distribute(serial, comm);
            mfem::Array<int> labels(d.pmesh->GetNE());
            for (int e = 0; e < labels.Size(); ++e) {
                labels[e] = label_of(*d.pmesh, e);
            }
            const ms::MeshAdjacency adjacency(*d.pmesh);
            const ms::FragmentCleanupResult result =
                ms::CleanupFragments(adjacency, labels, 2, {10, 20}, params);
            EXPECT_EQ(result.num_grains, x.grains);
            EXPECT_EQ(result.report.fragments_found, 2);
            EXPECT_EQ(result.report.fragments_split, x.split);
            EXPECT_EQ(result.report.fragments_merged, x.merged);
            results.push_back(GatherToAll(comm, d.order, result.element_grain));
        }
        EXPECT_EQ(results[0], results[2]);
        EXPECT_EQ(results[1], results[2]);
        // The single cell (4, 0, 0) ends up in grain 2 in both cases
        for (int e = 0; e < serial.GetNE(); ++e) {
            mfem::Vector c(3);
            serial.GetElementCenter(e, c);
            if (std::floor(c[0] * 10) == 4 && std::floor(c[1] * 4) == 0 &&
                std::floor(c[2] * 4) == 0) {
                EXPECT_EQ(results[2][static_cast<std::size_t>(e)], 2);
            }
        }
    }
    // Parameter checks (identical on every rank)
    ms::FragmentCleanupParameters bad;
    bad.min_fragment_fraction = 1.5;
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    mfem::Array<int> ones(d.pmesh->GetNE());
    ones = 1;
    const ms::MeshAdjacency adjacency(*d.pmesh);
    EXPECT_THROW(ms::CleanupFragments(adjacency, ones, 1, {}, bad), std::invalid_argument);
    bad.min_fragment_fraction = 0.1;
    bad.min_fragment_elements = -1;
    EXPECT_THROW(ms::CleanupFragments(adjacency, ones, 1, {}, bad), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Quadrature voting and periodic microstructures
// ---------------------------------------------------------------------------------------------

// sampling = "quadrature_vote" through the driver gives exactly the labels of the library vote
// (LabelTessellationByQuadratureVote with the same seeds and vote level) when cleanup is off,
// reports the elements cut by grain boundaries, and is rank-count independent (cube_tet mesh,
// levels 1 and 2).
TEST(DriverStage1bc, QuadratureVoteMatchesLibrary) {
    mfem::Mesh serial((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
    for (const int levels : {1, 2}) {
        SCOPED_TRACE("vote_levels " + std::to_string(levels));
        ms::MicrostructureOptions options = QuickOptions(12, 4u);
        options.tessellation.sampling = ms::SamplingMode::QUADRATURE_VOTE;
        options.tessellation.vote_levels = levels;
        options.cleanup.fragments = "none";
        const RunPair runs = RunAndCompare(serial, options);
        EXPECT_GT(runs.alone.report.num_split_elements, 0);

        // Library reference on the whole mesh
        Distributed alone = Distribute(serial, MPI_COMM_SELF);
        const ms::SeedSet seeds =
            ms::GenerateSeeds(*alone.pmesh, ms::MakeSeedingParameters(options));
        ms::QuadratureVoteParameters vote;
        vote.levels = levels;
        const ms::TessellationLabels labels =
            ms::LabelTessellationByQuadratureVote(*alone.pmesh, MPI_COMM_SELF, seeds, vote);
        std::vector<int> expected(static_cast<std::size_t>(serial.GetNE()));
        const int* grain = labels.element_grain.HostRead();
        for (int e = 0; e < alone.pmesh->GetNE(); ++e) {
            expected[static_cast<std::size_t>(alone.order.local_to_serial[e])] = grain[e];
        }
        EXPECT_EQ(runs.alone.labels, expected);
    }
}

// Periodic face pairing of MeshAdjacency on a fully periodic 4 x 4 x 4 hex box: every element
// has exactly 6 face neighbors (the box has no boundary any more), 3 x 16 = 48 face pairs are
// found and none is left unmatched, and the exchanged element keys of every neighbor are the
// keys of the element in the wrapped neighboring cell. With only x periodic, 16 pairs are found
// and the y/z boundary faces stay unpaired (fewer neighbors). Checked on the METIS partition and
// on a scattered partition with an empty rank, where most periodic partners live on other ranks.
TEST(DriverStage1bc, PeriodicFacePairing) {
    const int n = 4;
    mfem::Mesh serial = AutoHex(n, n, n);
    // Serial element id of cell (i, j, k): from the element centroid
    auto cell_of = [&](mfem::Mesh& mesh, int e) {
        mfem::Vector c(3);
        mesh.GetElementCenter(e, c);
        return std::array<int, 3>{static_cast<int>(std::floor(c[0] * n)),
                                  static_cast<int>(std::floor(c[1] * n)),
                                  static_cast<int>(std::floor(c[2] * n))};
    };
    std::map<std::array<int, 3>, std::uint64_t> key_of_cell;
    for (int e = 0; e < serial.GetNE(); ++e) {
        key_of_cell[cell_of(serial, e)] = ms::ElementKey(serial, e);
    }

    for (const bool empty_last_rank : {false, true}) {
        SCOPED_TRACE(empty_last_rank ? "empty last rank" : "METIS partition");
        Distributed d = empty_last_rank ? DistributeWithEmptyLastRank(serial, MPI_COMM_WORLD)
                                        : Distribute(serial, MPI_COMM_WORLD);
        ms::PeriodicFaceParameters pairing;
        pairing.periodic = {true, true, true};
        const ms::MeshAdjacency adjacency(*d.pmesh, pairing);
        EXPECT_EQ(adjacency.PeriodicReport().plane_faces, 6 * n * n);
        EXPECT_EQ(adjacency.PeriodicReport().matched_pairs, 3 * n * n);
        EXPECT_EQ(adjacency.PeriodicReport().unmatched_faces, 0);

        const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(*d.pmesh);
        std::vector<std::uint64_t> ghost_keys;
        adjacency.ExchangeKeys(keys, ghost_keys);
        const int ne = adjacency.NumElements();
        for (int e = 0; e < ne; ++e) {
            const int begin = adjacency.Offsets()[e];
            const int end = adjacency.Offsets()[e + 1];
            EXPECT_EQ(end - begin, 6) << "element " << e;
            // The 6 expected neighbors: the wrapped cells at +-1 along each axis
            const std::array<int, 3> c = cell_of(*d.pmesh, e);
            std::multiset<std::uint64_t> expected;
            for (int axis = 0; axis < 3; ++axis) {
                for (const int step : {-1, 1}) {
                    std::array<int, 3> nb = c;
                    nb[static_cast<std::size_t>(axis)] =
                        (nb[static_cast<std::size_t>(axis)] + step + n) % n;
                    expected.insert(key_of_cell[nb]);
                }
            }
            std::multiset<std::uint64_t> found;
            for (int k = begin; k < end; ++k) {
                const int nb = adjacency.Neighbors()[k];
                found.insert(adjacency.IsGhost(nb) ? ghost_keys[static_cast<std::size_t>(nb - ne)]
                                                   : keys[static_cast<std::size_t>(nb)]);
            }
            EXPECT_EQ(found, expected) << "element " << e;
        }

        // Only x periodic: 16 pairs, interior + x neighbors only
        ms::PeriodicFaceParameters x_only = pairing;
        x_only.periodic = {true, false, false};
        const ms::MeshAdjacency adjacency_x(*d.pmesh, x_only);
        EXPECT_EQ(adjacency_x.PeriodicReport().matched_pairs, n * n);
        EXPECT_EQ(adjacency_x.PeriodicReport().unmatched_faces, 0);
        long long local_entries = adjacency_x.Offsets()[ne];
        long long total_entries = 0;
        MPI_Allreduce(&local_entries, &total_entries, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        // Every element sees 2 x-neighbors (wrapped) plus its y/z interior neighbors:
        // x: 2 n^3, y and z: 2 n^2 (n - 1) each
        EXPECT_EQ(total_entries, 2LL * n * n * n + 4LL * n * n * (n - 1));
    }
}

// A periodic microstructure through the driver (centroid and vote sampling): with images, grains
// wrap around the box, and the periodic-aware cleanup must see a wrapped grain as one grain. So
// every grain is one piece when matching periodic faces count as connected, while at least one
// grain falls apart into several pieces when they do not (a grain that wraps around a face). The
// result is rank-count independent.
TEST(DriverStage1bc, PeriodicGrainWrapsAsOneGrain) {
    mfem::Mesh serial = AutoHex(8, 8, 8);
    for (const bool vote : {false, true}) {
        SCOPED_TRACE(vote ? "quadrature vote" : "centroid");
        ms::MicrostructureOptions options = QuickOptions(6, 13u);
        options.tessellation.periodic = {true, true, true};
        options.cleanup.fragments = "split";
        if (vote) {
            options.tessellation.sampling = ms::SamplingMode::QUADRATURE_VOTE;
        }
        const RunPair runs = RunAndCompare(serial, options);
        const DriverRun& run = runs.alone;
        EXPECT_EQ(run.report.periodic_layers, 1);
        EXPECT_EQ(run.report.num_periodic_images, 26 * 6);
        EXPECT_EQ(run.report.periodic_faces.matched_pairs, 3 * 64);
        EXPECT_EQ(run.report.periodic_faces.unmatched_faces, 0);
        EXPECT_EQ(CountPieces(serial, run.labels, true),
                  static_cast<std::size_t>(run.result.num_grains));
        EXPECT_GT(CountPieces(serial, run.labels, false),
                  static_cast<std::size_t>(run.result.num_grains));
    }
}

// ---------------------------------------------------------------------------------------------
// Size control, relaxation, morphology and nucleation
// ---------------------------------------------------------------------------------------------

// Log-normal size targets with polydisperse placement and a Laguerre volume fit, the seed count
// taken from the mean diameter (num_seeds = 0): the seeds respect their exclusion radii
// (|s_i - s_j| >= kappa (r_i + r_j), with r from the reported per-seed target volumes), the
// targets fill the material, the fit increased the dual function monotonically and ended below
// its starting error, and everything is rank-count independent.
TEST(DriverStage1bc, LognormalPolydisperseFit) {
    mfem::Mesh serial = AutoHex(10, 10, 10);
    ms::MicrostructureOptions options = QuickOptions(0, 3u);
    options.seeds.size.distribution = "lognormal";
    options.seeds.size.mean_diameter = 0.3;
    options.seeds.size.sigma = 0.3;
    options.seeds.size.polydisperse_placement = true;
    options.seeds.size.exclusion_factor = 0.5;
    options.tessellation.metric = ms::MetricType::LAGUERRE;
    options.tessellation.fit_volumes.enabled = true;
    options.tessellation.fit_volumes.tolerance = 0.05;
    options.tessellation.fit_volumes.max_iterations = 60;
    const RunPair runs = RunAndCompare(serial, options);
    const ms::GenerationReport& report = runs.alone.report;

    EXPECT_TRUE(report.used_polydisperse_placement);
    EXPECT_TRUE(report.ran_weight_fit);
    EXPECT_GT(report.num_seeds, 20);
    EXPECT_NEAR(report.material_volume, 1.0, 1e-12);
    EXPECT_EQ(report.seed_target_volume.size(), static_cast<std::size_t>(report.num_seeds));
    double sum = 0.0;
    for (const double v : report.seed_target_volume) {
        sum += v;
    }
    EXPECT_NEAR(sum, report.material_volume, 1e-12);
    // Largest first: slot volumes never increase in seed order
    for (std::size_t j = 1; j < report.seed_target_volume.size(); ++j) {
        EXPECT_LE(report.seed_target_volume[j], report.seed_target_volume[j - 1]);
    }
    // Monotone dual and a smaller final error
    for (std::size_t k = 1; k < report.weight_fit.phi.size(); ++k) {
        EXPECT_GE(report.weight_fit.phi[k], report.weight_fit.phi[k - 1]);
    }
    EXPECT_LT(report.weight_fit.max_relative_error.back(),
              report.weight_fit.max_relative_error.front());

    // Exclusion radii respected by the actual seeds
    Distributed alone = Distribute(serial, MPI_COMM_SELF);
    ms::SeedingParameters params = ms::MakeSeedingParameters(options);
    params.num_seeds = report.num_seeds;
    const double kappa = options.seeds.size.exclusion_factor;
    for (const double v : report.seed_target_volume) {
        params.rsa_radius.push_back(kappa * std::cbrt(3.0 * v / (4.0 * 3.14159265358979323846)));
    }
    const ms::SeedSet seeds = ms::GenerateSeeds(*alone.pmesh, params);
    EXPECT_EQ(static_cast<std::int64_t>(seeds.Size()), report.num_seeds);
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        for (std::size_t j = i + 1; j < seeds.Size(); ++j) {
            const double dx = seeds.x[i] - seeds.x[j];
            const double dy = seeds.y[i] - seeds.y[j];
            const double dz = seeds.z[i] - seeds.z[j];
            EXPECT_GE(std::sqrt(dx * dx + dy * dy + dz * dz),
                      params.rsa_radius[i] + params.rsa_radius[j]);
        }
    }
}

// Lloyd relaxation (Voronoi, 4 steps) lowers the CVT energy monotonically; centroidal power mode
// (Laguerre, equal targets) refits after every step. Both are rank-count independent.
TEST(DriverStage1bc, LloydAndCentroidalPower) {
    mfem::Mesh serial = AutoHex(8, 8, 8);
    ms::MicrostructureOptions lloyd = QuickOptions(10, 9u);
    lloyd.tessellation.lloyd_iterations = 4;
    const RunPair lloyd_runs = RunAndCompare(serial, lloyd);
    const ms::LloydReport& history = lloyd_runs.alone.report.lloyd;
    EXPECT_TRUE(lloyd_runs.alone.report.ran_lloyd);
    EXPECT_EQ(history.energy.size(), 5u);
    for (std::size_t k = 1; k < history.energy.size(); ++k) {
        EXPECT_LE(history.energy[k], history.energy[k - 1] * (1.0 + 1e-12));
    }

    ms::MicrostructureOptions power = QuickOptions(10, 9u);
    power.tessellation.metric = ms::MetricType::LAGUERRE;
    power.tessellation.lloyd_iterations = 2;
    power.tessellation.centroidal_power = true;
    power.tessellation.fit_volumes.tolerance = 0.05;
    power.tessellation.fit_volumes.max_iterations = 50;
    const RunPair power_runs = RunAndCompare(serial, power);
    EXPECT_EQ(power_runs.alone.report.lloyd.fits.size(), 3u);
    EXPECT_FALSE(power_runs.alone.report.ran_weight_fit); // the refits are inside Lloyd
}

// Columnar grains along z (anisotropy kind "columnar", A = I - z z^T): the metric ignores z, so
// every element of one (x, y) column of the box carries the same grain (cleanup off). Lamellar
// grains with normal z are the opposite: every z layer of elements carries one grain.
TEST(DriverStage1bc, ColumnarAndLamellarGrains) {
    const int n = 6;
    mfem::Mesh serial = AutoHex(n, n, 2 * n, 1.0, 1.0, 2.0);
    auto check = [&](const std::string& kind) {
        ms::MicrostructureOptions options = QuickOptions(5, 11u);
        options.tessellation.metric = ms::MetricType::QUADRIC;
        options.tessellation.anisotropy_kind = kind;
        options.cleanup.fragments = "none";
        const RunPair runs = RunAndCompare(serial, options);
        std::map<std::array<int, 2>, std::set<int>> grains_of;
        for (int e = 0; e < serial.GetNE(); ++e) {
            mfem::Vector c(3);
            serial.GetElementCenter(e, c);
            const int i = static_cast<int>(std::floor(c[0] * n));
            const int j = static_cast<int>(std::floor(c[1] * n));
            const int k = static_cast<int>(std::floor(c[2] * n));
            const std::array<int, 2> key =
                (kind == "columnar") ? std::array<int, 2>{i, j} : std::array<int, 2>{k, 0};
            grains_of[key].insert(runs.alone.labels[static_cast<std::size_t>(e)]);
        }
        for (const auto& entry : grains_of) {
            EXPECT_EQ(entry.second.size(), 1u) << kind;
        }
        EXPECT_GT(runs.alone.result.num_grains, 1);
    };
    check("columnar");
    check("lamellar");
}

// Johnson-Mehl nucleation through the driver: the seeds are the nuclei of
// GenerateJohnsonMehlSeeds (same parameters), labeled with the Additive metric and their weights
// a = -v t, so the driver labels equal LabelTessellation with those seeds (cleanup off). The
// report carries the nucleation counts, and the run is rank-count independent.
TEST(DriverStage1bc, JohnsonMehlSeeds) {
    mfem::Mesh serial = AutoHex(6, 6, 6);
    ms::MicrostructureOptions options = QuickOptions(0, 5u);
    options.seeds.placement = ms::SeedPlacement::JOHNSON_MEHL;
    options.seeds.johnson_mehl.rate = 60.0;
    options.tessellation.metric = ms::MetricType::ADDITIVE;
    options.cleanup.fragments = "none";
    const RunPair runs = RunAndCompare(serial, options);
    const ms::GenerationReport& report = runs.alone.report;
    EXPECT_TRUE(report.used_johnson_mehl);
    EXPECT_EQ(report.johnson_mehl.accepted, report.num_seeds);
    EXPECT_EQ(report.seeding.placement_used, ms::SeedPlacement::JOHNSON_MEHL);
    EXPECT_GT(report.num_seeds, 5);

    Distributed alone = Distribute(serial, MPI_COMM_SELF);
    ms::JohnsonMehlParameters params;
    params.nucleation_rate = 60.0;
    params.rng_seed = 5u;
    const ms::SeedSet nuclei = ms::GenerateJohnsonMehlSeeds(*alone.pmesh, params);
    EXPECT_EQ(static_cast<std::int64_t>(nuclei.Size()), report.num_seeds);
    const ms::TessellationLabels labels =
        ms::LabelTessellation(*alone.pmesh, MPI_COMM_SELF, nuclei, options.tessellation);
    std::vector<int> expected(static_cast<std::size_t>(serial.GetNE()));
    const int* grain = labels.element_grain.HostRead();
    for (int e = 0; e < alone.pmesh->GetNE(); ++e) {
        expected[static_cast<std::size_t>(alone.order.local_to_serial[e])] = grain[e];
    }
    EXPECT_EQ(runs.alone.labels, expected);
}

// Every example options file of the new features (test/data/microstructure) runs through the
// driver on its auto mesh and gives the same microstructure for every rank count.
TEST(DriverStage1bc, ExampleFilesRun) {
    for (const char* name : {"periodic_vote_auto.toml", "lognormal_fit_auto.toml",
                             "columnar_auto.toml", "johnson_mehl_auto.toml",
                             "lloyd_cvt_auto.toml"}) {
        SCOPED_TRACE(name);
        const ms::MicrostructureOptions options =
            ms::MicrostructureOptions::parse_options_file(data_dir / "microstructure" / name);
        EXPECT_TRUE(options.validate());
        mfem::Mesh serial = AutoHex(options.mesh.nxyz[0], options.mesh.nxyz[1],
                                    options.mesh.nxyz[2], options.mesh.mxyz[0],
                                    options.mesh.mxyz[1], options.mesh.mxyz[2]);
        const RunPair runs = RunAndCompare(serial, options);
        EXPECT_GT(runs.alone.result.num_grains, 1);
    }
}

// Options the pipeline refuses before any work (on every rank, std::invalid_argument): an
// orientation source other than "random" (from_source / list), Johnson-Mehl with a periodic axis,
// and polydisperse placement with a sampling box. These used to fail only at the end of the run
// or deep inside seeding.
TEST(DriverStage1bc, RefusedCombinations) {
    mfem::Mesh serial = AutoHex(4, 4, 4);
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    auto refused = [&](auto&& mutate) {
        ms::MicrostructureOptions options = QuickOptions(4, 1u);
        mutate(options);
        try {
            ms::GenerateMicrostructure(options, *d.pmesh, MPI_COMM_WORLD);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    using O = ms::MicrostructureOptions;
    EXPECT_TRUE(refused([](O& o) { o.orientations.source = "from_source"; }));
    EXPECT_TRUE(refused([](O& o) { o.orientations.source = "list"; }));
    EXPECT_TRUE(refused([](O& o) {
        o.seeds.placement = ms::SeedPlacement::JOHNSON_MEHL;
        o.seeds.johnson_mehl.rate = 60.0;
        o.tessellation.metric = ms::MetricType::ADDITIVE;
        o.tessellation.periodic = {true, false, false};
    }));
    EXPECT_TRUE(refused([](O& o) {
        o.seeds.size.polydisperse_placement = true;
        o.seeds.use_sampling_box = true;
        o.seeds.sampling_box_lo = {0.0, 0.0, 0.0};
        o.seeds.sampling_box_hi = {0.5, 0.5, 0.5};
    }));
}

// ---------------------------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------------------------

// ThrowIfAnyRankFailed: when only the last rank fails, every rank throws the last rank's message
// with its rank number; when ranks 0 and last fail, rank 0's message wins (the lowest rank).
// ThrowIfAnyRank throws the common message everywhere; AnyRank and LowestRankWith agree on all
// ranks. Nothing is thrown when no rank fails.
TEST(DriverStage1bc, CollectiveErrorsAgree) {
    const int rank = Rank();
    const int last = NumRanks() - 1;
    EXPECT_NO_THROW(ms::ThrowIfAnyRankFailed(MPI_COMM_WORLD, ""));
    EXPECT_EQ(ms::LowestRankWith(false, MPI_COMM_WORLD), -1);
    EXPECT_EQ(ms::LowestRankWith(rank == last, MPI_COMM_WORLD), last);
    EXPECT_TRUE(ms::AnyRank(rank == last, MPI_COMM_WORLD));
    EXPECT_FALSE(ms::AnyRank(false, MPI_COMM_WORLD));

    std::string message;
    try {
        ms::ThrowIfAnyRankFailed(MPI_COMM_WORLD,
                                 (rank == last) ? "bad element on the last rank" : "");
    } catch (const std::runtime_error& e) {
        message = e.what();
    }
    EXPECT_EQ(message, "bad element on the last rank (reported by rank " + std::to_string(last) +
                           ")");

    message.clear();
    try {
        const bool fails = (rank == 0 || rank == last);
        ms::ThrowIfAnyRankFailed(MPI_COMM_WORLD,
                                 fails ? "failure on rank " + std::to_string(rank) : "", false);
    } catch (const std::runtime_error& e) {
        message = e.what();
    }
    EXPECT_EQ(message, "failure on rank 0");

    EXPECT_THROW(ms::ThrowIfAnyRank(MPI_COMM_WORLD, rank == last, "common message"),
                 std::invalid_argument);
}

// Polydisperse RSA in the seed generator: with exclusion radii per slot (largest first) the
// accepted seeds never come closer than rho_i + rho_j (periodic minimum image on a periodic
// box too), the seeds are identical for every rank count, and radii whose spheres would fill
// more than 30% of the volume are refused on every rank.
TEST(DriverStage1bc, PolydisperseRsaSeeds) {
    mfem::Mesh serial = AutoHex(6, 6, 6);
    for (const bool periodic : {false, true}) {
        SCOPED_TRACE(periodic ? "periodic" : "not periodic");
        ms::SeedingParameters params;
        params.num_seeds = 30;
        params.rng_seed = 17u;
        if (periodic) {
            params.periodic = {true, true, true};
        }
        for (int j = 0; j < 30; ++j) {
            params.rsa_radius.push_back(0.09 - 0.0015 * j); // decreasing: largest first
        }
        Distributed world = Distribute(serial, MPI_COMM_WORLD);
        const ms::SeedSet seeds = ms::GenerateSeeds(*world.pmesh, params);
        Distributed alone = Distribute(serial, MPI_COMM_SELF);
        const ms::SeedSet reference = ms::GenerateSeeds(*alone.pmesh, params);
        EXPECT_EQ(seeds.x, reference.x);
        EXPECT_EQ(seeds.y, reference.y);
        EXPECT_EQ(seeds.z, reference.z);
        EXPECT_EQ(seeds.source_id, reference.source_id);
        EXPECT_EQ(seeds.Size(), 30u);
        for (std::size_t i = 0; i < seeds.Size(); ++i) {
            for (std::size_t j = i + 1; j < seeds.Size(); ++j) {
                double d2 = 0.0;
                const double delta[3] = {seeds.x[i] - seeds.x[j], seeds.y[i] - seeds.y[j],
                                         seeds.z[i] - seeds.z[j]};
                for (double dx : delta) {
                    if (periodic) {
                        dx -= std::round(dx); // box length 1
                    }
                    d2 += dx * dx;
                }
                EXPECT_GE(std::sqrt(d2), params.rsa_radius[i] + params.rsa_radius[j]);
            }
        }
    }
    // Infeasible: 30 spheres of radius 0.15 fill 0.42 of the unit cube
    ms::SeedingParameters infeasible;
    infeasible.num_seeds = 30;
    infeasible.rsa_radius.assign(30, 0.15);
    Distributed world = Distribute(serial, MPI_COMM_WORLD);
    EXPECT_THROW(ms::GenerateSeeds(*world.pmesh, infeasible), std::runtime_error);
    // Inconsistent parameters
    infeasible.rsa_radius.assign(29, 0.01);
    EXPECT_THROW(ms::GenerateSeeds(*world.pmesh, infeasible), std::invalid_argument);
}

// The IntegrationRule overload of ComputeQuadraturePointSamples gives bitwise the same points and
// weights as the quadrature-order overload with the same MFEM rule, and QuadraturePointSampler
// returns the same values chunk by chunk as in one piece (hex box and cube_tet, distributed; an
// empty rank returns empty arrays).
TEST(DriverStage1bc, SamplerRuleOverload) {
    mfem::Mesh hex = AutoHex(3, 3, 3);
    mfem::Mesh tet((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
    for (mfem::Mesh* serial : {&hex, &tet}) {
        Distributed d = DistributeWithEmptyLastRank(*serial, MPI_COMM_WORLD);
        mfem::ParMesh& mesh = *d.pmesh;
        const int ne = mesh.GetNE();
        ms::QuadraturePointSamples by_order;
        ms::ComputeQuadraturePointSamples(mesh, 3, 0, ne, by_order);
        ms::QuadraturePointSamples by_rule;
        if (ne > 0) {
            const mfem::IntegrationRule& rule = mfem::IntRules.Get(mesh.GetElementGeometry(0), 3);
            ms::ComputeQuadraturePointSamples(mesh, rule, 0, ne, by_rule);

            ms::QuadraturePointSampler sampler(mesh, rule);
            EXPECT_EQ(sampler.NumPointsPerElement(), rule.GetNPoints());
            const int half = ne / 2;
            ms::QuadraturePointSamples first;
            ms::QuadraturePointSamples second;
            sampler.Sample(0, half, first);
            sampler.Sample(half, ne, second);
            const double* x = by_order.x.HostRead();
            const double* w = by_order.weight.HostRead();
            const int nq = by_order.num_qp;
            for (int i = 0; i < first.x.Size(); ++i) {
                EXPECT_EQ(first.x.HostRead()[i], x[i]);
                EXPECT_EQ(first.weight.HostRead()[i], w[i]);
            }
            for (int i = 0; i < second.x.Size(); ++i) {
                EXPECT_EQ(second.x.HostRead()[i], x[half * nq + i]);
                EXPECT_EQ(second.weight.HostRead()[i], w[half * nq + i]);
            }
        } else {
            EXPECT_EQ(by_order.num_qp, 0);
            EXPECT_EQ(by_order.x.Size(), 0);
            continue;
        }
        EXPECT_EQ(by_rule.num_qp, by_order.num_qp);
        EXPECT_EQ(by_rule.x.Size(), by_order.x.Size());
        const int common = std::min(by_rule.x.Size(), by_order.x.Size());
        for (int i = 0; i < common; ++i) {
            EXPECT_EQ(by_rule.x.HostRead()[i], by_order.x.HostRead()[i]);
            EXPECT_EQ(by_rule.y.HostRead()[i], by_order.y.HostRead()[i]);
            EXPECT_EQ(by_rule.z.HostRead()[i], by_order.z.HostRead()[i]);
            EXPECT_EQ(by_rule.weight.HostRead()[i], by_order.weight.HostRead()[i]);
        }
    }
}

// MFEM v4.10 quadrature rules have only positive weights (ExaConstit requires the v4.10 fork):
// the quadrature samples of the volume fit use MFEM's rule directly (order 3 for linear tets, the
// exact default, whose pre-v4.10 rule had the weight -2/15), every sample weight is positive, and
// the weights of each order add up to the material volume (the unit cube). The guard inside
// MakeQuadratureWeightedSamples would throw on every rank for a non-positive rule.
TEST(DriverStage1bc, QuadratureSamplesArePositive) {
    mfem::Mesh tet((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
    Distributed d = Distribute(tet, MPI_COMM_WORLD);
    for (int order = -1; order <= 12; ++order) {
        SCOPED_TRACE("order " + std::to_string(order));
        ms::WeightedSamples samples;
        ms::MakeQuadratureWeightedSamples(*d.pmesh, order, samples);
        const int effective = (order < 0) ? 3 : order;
        const int nq = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, effective).GetNPoints();
        EXPECT_EQ(samples.Size(), nq * d.pmesh->GetNE());
        double local = 0.0;
        const double* w = samples.weight.HostRead();
        for (int q = 0; q < samples.Size(); ++q) {
            EXPECT_GT(w[q], 0.0);
            local += w[q];
        }
        double total = 0.0;
        MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_NEAR(total, 1.0, 1e-12);
    }
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
