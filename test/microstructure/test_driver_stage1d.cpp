/**
 * @file test_driver_stage1d.cpp
 * @brief Driver-level tests of the Stage 1d features wired into GenerateMicrostructure
 * (microstructure_driver.cpp): seed tables, the region of interest (ROI) with random filler
 * seeds, volume targets from the table, voxel maps, phases and regions, and passthrough mesh
 * parts.
 *
 * Covered paths:
 *  1. a seed table: the labels equal a direct tessellation of the same seeds, the quaternions are
 *     normalized to unit length (sign and order kept), phases become regions through a
 *     many-to-one map, and non-crystalline regions get identity quaternions;
 *  2. ROI and filler: rows outside the ROI are dropped, no filler seed lies inside the ROI, the
 *     filler ids start at max table id + 1, the automatic filler count;
 *  3. distribution = "from_source" with fit_volumes: targets proportional to the table volumes,
 *     the fit reaches its tolerance; with filler the table targets fill the ROI volume share;
 *  4. voxel maps in the canonical and DREAM.3D layouts: hole elements filled, two regions from
 *     three phases, the DREAM.3D quaternions reordered per the declarations;
 *  5. synthetic phase fractions within the documented bound;
 *  6. a build plate kept as a passthrough region: region ids, the identity quaternion of the
 *     plate grain, and the tool-facing adjacency that belongs to the whole mesh;
 *  7. refused combinations;
 *  8. the Stage 1d example options files that run on an auto mesh.
 *
 * **Fixtures.** Rank 0 writes the seed tables (CSV) and the voxel maps (HDF5, from the shared
 * common/voxel_fixture.hpp) into `test_driver_stage1d_data/np<N>/` under the working directory
 * (the build tree when run by ctest; never /tmp), then all ranks meet at a barrier.
 *
 * **Rank-count independence.** Every driver run is done twice in the same process: distributed
 * over MPI_COMM_WORLD and alone on every rank with MPI_COMM_SELF (exactly the np = 1 answer). The
 * labels gathered into serial element order and every per-grain table (orientations bitwise,
 * source ids, regions, phases, filler flags) must agree. Running the executable at np = 1..4
 * (test_driver_stage1d.cmake) therefore compares np = 2, 3, 4 against np = 1.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/io/seed_table_io.hpp"
#include "microstructure/labeling/grain_statistics.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/orientation/orientation_assigner.hpp"
#include "microstructure/orientation/random_orientation.hpp"
#include "microstructure/util/counter_rng.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_options.hpp"
#include "utilities/auto_mesh.hpp"

#include "common/voxel_fixture.hpp"

#include "axom/config.hpp"
#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

const fs::path data_dir = fs::path(MS_DRIVER_DATA_DIR);

constexpr double PI = 3.14159265358979323846;

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

/** Directory of this run's fixtures (one per rank count, so concurrent np runs never collide). */
fs::path DataDir() {
    return fs::path("test_driver_stage1d_data") / ("np" + std::to_string(NumRanks()));
}

/** Formats a double with 17 significant digits (round-trips exactly through strtod). */
std::string Num(double v) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", v);
    return buffer;
}

// ---------------------------------------------------------------------------------------------
// Seed tables
// ---------------------------------------------------------------------------------------------

/** Positions of the 8 "measured" seeds in the unit cube (generic: no exact ties). */
const std::array<std::array<double, 3>, 8> kTablePos = {{{0.21, 0.19, 0.22},
                                                          {0.76, 0.24, 0.31},
                                                          {0.29, 0.71, 0.26},
                                                          {0.81, 0.79, 0.18},
                                                          {0.24, 0.32, 0.77},
                                                          {0.69, 0.21, 0.79},
                                                          {0.18, 0.82, 0.71},
                                                          {0.77, 0.72, 0.78}}};
/** Their ids (not 0..7), phases {0, 3, 7} and table volumes (ratios 1..8). */
const std::array<std::int64_t, 8> kTableId = {10, 4, 7, 22, 15, 3, 9, 30};
const std::array<int, 8> kTablePhase = {0, 3, 7, 0, 3, 7, 0, 3};

/**
 * Quaternion row i of the table (w x y z). Row 0 has norm 1.0004 (accepted: | |q| - 1 | <=
 * 1e-3), so a normalizing driver would change it; the others are a normalized rotation family.
 */
std::array<double, 4> TableQuat(int i) {
    if (i == 0) {
        return {1.0004, 0.0, 0.0, 0.0};
    }
    std::array<double, 4> q = {1.0, 0.1 * i, -0.05 * i, 0.02 * i};
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (double& v : q) {
        v /= n;
    }
    return q;
}

/** Writes `text` to DataDir()/name on rank 0. Not collective (see WriteFixtures). */
void WriteText(const std::string& name, const std::string& text) {
    std::ofstream out(DataDir() / name);
    if (!out) {
        throw std::runtime_error("cannot write " + (DataDir() / name).string());
    }
    out << text;
}

/**
 * The seed tables:
 *  - measured.csv: the 8 seeds with id, quaternion and phase;
 *  - sized.csv: the 8 seeds with a volume column (volume of seed i = (i + 1) * 0.01) and no
 *    quaternions or phases;
 *  - roi.csv: 6 seeds inside the ROI [0, 0.5]^3 (ids 1..6), 3 outside (ids 50, 99, 7), with
 *    volumes; no quaternions and no phases.
 */
void WriteSeedTables() {
    std::ostringstream measured;
    measured << "# measured seeds\nid,x,y,z,qw,qx,qy,qz,phase\n";
    for (int i = 0; i < 8; ++i) {
        const auto& p = kTablePos[static_cast<std::size_t>(i)];
        const std::array<double, 4> q = TableQuat(i);
        measured << kTableId[static_cast<std::size_t>(i)] << ',' << Num(p[0]) << ',' << Num(p[1])
                 << ',' << Num(p[2]) << ',' << Num(q[0]) << ',' << Num(q[1]) << ','
                 << Num(q[2]) << ',' << Num(q[3]) << ',' << kTablePhase[static_cast<std::size_t>(i)]
                 << "\n";
    }
    WriteText("measured.csv", measured.str());

    std::ostringstream sized;
    sized << "id,x,y,z,volume\n";
    for (int i = 0; i < 8; ++i) {
        const auto& p = kTablePos[static_cast<std::size_t>(i)];
        sized << kTableId[static_cast<std::size_t>(i)] << ',' << Num(p[0]) << ',' << Num(p[1])
              << ',' << Num(p[2]) << ',' << Num((i + 1) * 0.01) << "\n";
    }
    WriteText("sized.csv", sized.str());

    WriteText("roi.csv",
              "id,x,y,z,volume\n"
              "1,0.11,0.12,0.13,1\n"
              "2,0.37,0.14,0.11,2\n"
              "50,0.8,0.8,0.8,1\n"
              "3,0.13,0.38,0.16,1\n"
              "4,0.36,0.37,0.12,3\n"
              "99,0.9,0.1,0.1,1\n"
              "5,0.14,0.16,0.39,1\n"
              "6,0.38,0.33,0.36,2\n"
              "7,0.6,0.45,0.2,1\n");

    // roi.csv with every table weight carrying the large common offset C = 100 (100 x the
    // squared box size): weighted metrics only see differences, so the filler seeds must follow
    WriteText("offset.csv",
              "id,x,y,z,volume,weight\n"
              "1,0.11,0.12,0.13,1,100\n"
              "2,0.37,0.14,0.11,2,100\n"
              "3,0.13,0.38,0.16,1,100\n"
              "4,0.36,0.37,0.12,3,100\n"
              "5,0.14,0.16,0.39,1,100\n"
              "6,0.38,0.33,0.36,2,100\n");

    // U-shaped part [0,3] x [0,2] x [0,1] without the notch [1,2] x [1,2]: seed 42 sits in the
    // notch (no material there), so its cell holds only material next to the notch in both arms
    // (x in (0.85, 1) and (2, 2.15) above y ~ 1): two disconnected pieces. Phases 0 / 3.
    WriteText("ushape.csv",
              "id,x,y,z,qw,qx,qy,qz,phase\n"
              "1,0.2,1.5,0.5,1,0,0,0,0\n"
              "2,2.8,1.5,0.5,0,1,0,0,0\n"
              "3,0.4,0.4,0.5,0,0,1,0,0\n"
              "4,1.5,0.45,0.5,0,0,0,1,0\n"
              "5,2.6,0.4,0.5,0.6,0.8,0,0,0\n"
              "42,1.5,1.5,0.5,0.5004,0.5,-0.5,0.5,3\n");

    // One measured seed in the bottom-left block of the U (ROI [0,1]^3); filler seeds elsewhere
    WriteText("ushape_roi.csv",
              "id,x,y,z,qw,qx,qy,qz\n"
              "1,0.5,0.5,0.5,1,0,0,0\n");

    // Seeds above the build plate of PlateMesh (z > 1/4) with phases 0 and 1
    WriteText("plate.csv",
              "id,x,y,z,phase\n"
              "1,0.21,0.23,0.41,0\n"
              "2,0.74,0.27,0.46,1\n"
              "3,0.27,0.76,0.62,0\n"
              "4,0.71,0.73,0.67,1\n"
              "5,0.33,0.31,0.86,1\n"
              "6,0.68,0.66,0.91,0\n");
}

/** Writes every fixture once (rank 0), then synchronizes (collective). */
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
            WriteSeedTables();
#ifdef AXOM_USE_HDF5
            voxel_fixture::CanonicalOptions canonical;
            canonical.ids = voxel_fixture::MapData().ids;
            voxel_fixture::WriteCanonical(DataDir() / "canonical.h5", canonical);
            voxel_fixture::WriteSimpl(DataDir() / "simpl.h5");
#endif
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

/** The table as every rank reads it (alone, MPI_COMM_SELF), for expected values. */
ms::SeedTable ReadTableAlone(const std::string& name) {
    ms::SeedTableParameters params;
    params.path = DataDir() / name;
    return ms::ReadSeedTable(params, MPI_COMM_SELF);
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
    /** Element count of every grain from the report's adjacency (ComputeGrainStatistics). */
    std::vector<std::int64_t> stats_elements;
};

/**
 * GenerateMicrostructure on `serial` distributed over `comm` (collective over comm). The grain
 * statistics are computed from the report's adjacency while the mesh is still alive, which also
 * checks that the adjacency matches the returned per-element labels (whole mesh).
 */
DriverRun RunDriverOn(Distributed& d, const ms::MicrostructureOptions& options, MPI_Comm comm) {
    DriverRun run;
    run.result = ms::GenerateMicrostructure(options, *d.pmesh, comm, &run.report);
    run.labels = GatherToAll(comm, d.order, run.result.element_grain);
    EXPECT_EQ(run.report.adjacency->NumElements(), d.pmesh->GetNE());
    const ms::GrainStatistics stats = ms::ComputeGrainStatistics(
        *run.report.adjacency, run.result.element_grain, run.result.num_grains, 0);
    run.stats_elements = stats.num_elements;
    run.report.adjacency.reset(); // it refers to d.pmesh, which may die after this call
    return run;
}

/** RunDriverOn with a METIS partition of `serial` over `comm`. Collective over comm. */
DriverRun RunDriver(mfem::Mesh& serial, const ms::MicrostructureOptions& options, MPI_Comm comm) {
    Distributed d = Distribute(serial, comm);
    return RunDriverOn(d, options, comm);
}

/**
 * A partition given as an owner rank per serial element (collective over WORLD): builds the
 * local -> serial map and the ParMesh. Owners must be valid ranks; a rank may own nothing.
 */
Distributed DistributeWithOwners(mfem::Mesh& serial, const std::vector<int>& owner) {
    const int rank = Rank();
    Distributed d;
    d.order.partitioning.SetSize(serial.GetNE());
    int num_local = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        d.order.partitioning[e] = owner[static_cast<std::size_t>(e)];
        num_local += (owner[static_cast<std::size_t>(e)] == rank) ? 1 : 0;
    }
    d.order.local_to_serial.SetSize(num_local);
    int next = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        if (owner[static_cast<std::size_t>(e)] == rank) {
            d.order.local_to_serial[next++] = e;
        }
    }
    d.pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, d.order);
    return d;
}

/**
 * U-shaped hex part with n cells per unit length: [0,3] x [0,2] x [0,1] without the notch
 * [1,2] x [1,2] x [0,1] (two arms joined at y < 1), lexicographic cell order.
 */
mfem::Mesh UShape(int n) {
    const int nx = 3 * n, ny = 2 * n, nz = n;
    const double h = 1.0 / n;
    auto kept = [n](int i, int j) { return !(i >= n && i < 2 * n && j >= n); };
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    int cells = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                cells += kept(i, j) ? 1 : 0;
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
                if (kept(i, j)) {
                    const int hex[8] = {vid(i, j, k),         vid(i + 1, j, k),
                                        vid(i + 1, j + 1, k), vid(i, j + 1, k),
                                        vid(i, j, k + 1),     vid(i + 1, j, k + 1),
                                        vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)};
                    mesh.AddHex(hex);
                }
            }
        }
    }
    mesh.FinalizeHexMesh(1, 1, true);
    mesh.RemoveUnusedVertices();
    return mesh;
}

/**
 * Every orientation row is a unit quaternion: |norm - 1| <= 4e-16 (about two units of round-off
 * of a norm near 1), whatever its origin (table, voxel feature, random, identity).
 */
void ExpectUnitQuaternions(const ms::MicrostructureResult& result) {
    for (std::size_t g = 0; 4 * g < result.grain_quat.size(); ++g) {
        const double* q = &result.grain_quat[4 * g];
        const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        EXPECT_LE(std::fabs(norm - 1.0), 4e-16) << "grain " << g + 1;
    }
}

/**
 * Expects `q` (4 values, w x y z) to be the direction of `input`: input / |input| within 1e-15
 * per component, i.e. normalized, with the sign and component order of the input kept.
 */
void ExpectNormalizedFrom(const double* q, const std::array<double, 4>& input,
                          const std::string& what) {
    const double norm = std::sqrt(input[0] * input[0] + input[1] * input[1] +
                                  input[2] * input[2] + input[3] * input[3]);
    for (std::size_t k = 0; k < 4; ++k) {
        EXPECT_NEAR(q[k], input[k] / norm, 1e-15) << what << " component " << k;
    }
}

/** Distributed run (WORLD) and the single-rank reference (SELF) of the same options. */
struct RunPair {
    DriverRun world;
    DriverRun alone;
};

/**
 * Runs the options over MPI_COMM_WORLD and alone (MPI_COMM_SELF) and checks that the two agree
 * exactly: labels in serial order and every per-grain and per-region table. Also checks the
 * table sizes and that grain ids are compact (1..N, every grain used). Collective over WORLD.
 */
RunPair RunAndCompare(mfem::Mesh& serial, const ms::MicrostructureOptions& options) {
    RunPair pair;
    pair.world = RunDriver(serial, options, MPI_COMM_WORLD);
    pair.alone = RunDriver(serial, options, MPI_COMM_SELF);
    const ms::MicrostructureResult& w = pair.world.result;
    const ms::MicrostructureResult& a = pair.alone.result;
    EXPECT_EQ(pair.world.labels, pair.alone.labels);
    EXPECT_EQ(w.num_grains, a.num_grains);
    EXPECT_EQ(w.grain_quat, a.grain_quat);
    EXPECT_EQ(w.grain_source_id, a.grain_source_id);
    EXPECT_EQ(w.grain_region, a.grain_region);
    EXPECT_EQ(w.grain_phase, a.grain_phase);
    EXPECT_EQ(w.grain_filler, a.grain_filler);
    EXPECT_EQ(w.num_regions, a.num_regions);
    EXPECT_EQ(w.region_phases, a.region_phases);
    EXPECT_EQ(w.region_crystalline, a.region_crystalline);
    EXPECT_EQ(pair.world.report.grain_seed, pair.alone.report.grain_seed);
    EXPECT_EQ(pair.world.report.grain_feature, pair.alone.report.grain_feature);
    EXPECT_EQ(pair.world.report.cleanup.fragments_found, pair.alone.report.cleanup.fragments_found);
    EXPECT_EQ(pair.world.stats_elements, pair.alone.stats_elements);
    ExpectUnitQuaternions(w);
    ExpectUnitQuaternions(a);

    // Sizes of the per-grain / per-region tables
    const auto n = static_cast<std::size_t>(a.num_grains);
    EXPECT_EQ(a.grain_quat.size(), 4 * n);
    EXPECT_EQ(a.grain_region.size(), n);
    EXPECT_EQ(a.grain_source_id.size(), n);
    EXPECT_EQ(a.grain_phase.size(), n);
    EXPECT_EQ(a.grain_filler.size(), n);
    EXPECT_EQ(pair.alone.report.grain_seed.size(), n);
    EXPECT_EQ(pair.alone.report.grain_feature.size(), n);
    EXPECT_EQ(a.region_phases.size(), static_cast<std::size_t>(a.num_regions));
    EXPECT_EQ(a.region_crystalline.size(), static_cast<std::size_t>(a.num_regions));

    // Compact ids: 1..N, every grain used; regions in 1..R
    std::vector<int> count(n + 1, 0);
    for (const int g : pair.alone.labels) {
        EXPECT_GE(g, 1);
        EXPECT_LE(g, a.num_grains);
        if (g >= 1 && g <= a.num_grains) {
            count[static_cast<std::size_t>(g)]++;
        }
    }
    for (std::size_t g = 1; g <= n; ++g) {
        EXPECT_GT(count[g], 0) << "grain " << g;
    }
    for (const int r : a.grain_region) {
        EXPECT_GE(r, 1);
        EXPECT_LE(r, a.num_regions);
    }
    return pair;
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

/** True when p lies strictly inside the box (lo, hi). */
bool StrictlyInside(const std::array<double, 3>& p, const std::array<double, 3>& lo,
                    const std::array<double, 3>& hi) {
    return p[0] > lo[0] && p[0] < hi[0] && p[1] > lo[1] && p[1] < hi[1] && p[2] > lo[2] &&
           p[2] < hi[2];
}

/** True when p lies in the closed box [lo, hi]. */
bool InsideClosed(const std::array<double, 3>& p, const std::array<double, 3>& lo,
                  const std::array<double, 3>& hi) {
    return p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1] && p[2] >= lo[2] &&
           p[2] <= hi[2];
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

} // namespace

// ---------------------------------------------------------------------------------------------
// 1. Seed tables
// ---------------------------------------------------------------------------------------------

// A seed table is tessellated exactly like the same seeds given directly: with fragments =
// "none" the driver's labels equal LabelTessellation on a SeedSet with the table positions
// (grain ids in seed order in both). The orientations are the table rows normalized to unit
// length with sign and order kept (row 0 has norm 1.0004 and becomes (1, 0, 0, 0)); source ids
// are the table ids. The phases
// {0, 3, 7} go to regions through the many-to-one map {0: 1, 3: 1, 7: 2}, so region 1 holds
// phases {0, 3} and region 2 phase 7. With phase 7 non-crystalline, region 2 is flagged and its
// grains get the identity quaternion while the others keep their measured ones.
TEST(DriverStage1d, SeedTableLabelsQuaternionsAndRegions) {
    WriteFixtures();
    mfem::Mesh serial = AutoHex(8, 8, 8);
    ms::MicrostructureOptions options = TableOptions("measured.csv");
    options.cleanup.fragments = "none";
    options.orientations.source = "from_source";
    options.regions.mode = "from_phase";
    options.regions.phase_to_region = {{0, 1}, {3, 1}, {7, 2}};
    const RunPair runs = RunAndCompare(serial, options);
    const ms::MicrostructureResult& result = runs.alone.result;
    const ms::GenerationReport& report = runs.alone.report;
    EXPECT_TRUE(report.used_seed_table);
    EXPECT_EQ(report.num_table_rows, 8);
    EXPECT_EQ(report.num_table_seeds, 8);
    EXPECT_TRUE(report.table_has_quat);
    EXPECT_TRUE(report.table_has_phase);
    EXPECT_EQ(result.num_grains, 8);

    // Reference: the same seeds tessellated directly, alone on every rank
    const ms::SeedTable table = ReadTableAlone("measured.csv");
    ms::SeedSet seeds;
    seeds.x = table.seeds.x;
    seeds.y = table.seeds.y;
    seeds.z = table.seeds.z;
    seeds.source_id = table.seeds.source_id;
    Distributed alone = Distribute(serial, MPI_COMM_SELF);
    const ms::TessellationLabels direct =
        ms::LabelTessellation(*alone.pmesh, MPI_COMM_SELF, seeds, options.tessellation);
    EXPECT_EQ(GatherToAll(MPI_COMM_SELF, alone.order, direct.element_grain), runs.alone.labels);

    // Per grain: normalized table quaternion, table id, phase and mapped region of its seed
    for (int g = 0; g < result.num_grains; ++g) {
        const auto gi = static_cast<std::size_t>(g);
        const auto s = static_cast<std::size_t>(report.grain_seed[gi]);
        const std::array<double, 4> input = {table.seeds.quat[4 * s], table.seeds.quat[4 * s + 1],
                                             table.seeds.quat[4 * s + 2],
                                             table.seeds.quat[4 * s + 3]};
        ExpectNormalizedFrom(&result.grain_quat[4 * gi], input, "grain " + std::to_string(g));
        if (s == 0) {
            EXPECT_EQ(result.grain_quat[4 * gi], 1.0); // 1.0004 / 1.0004
        }
        EXPECT_EQ(result.grain_source_id[gi], kTableId[s]);
        EXPECT_EQ(result.grain_phase[gi], kTablePhase[s]);
        EXPECT_EQ(result.grain_region[gi], kTablePhase[s] == 7 ? 2 : 1);
        EXPECT_EQ(result.grain_filler[gi], 0);
    }
    EXPECT_EQ(result.num_regions, 2);
    EXPECT_EQ(result.region_phases, (std::vector<std::vector<int>>{{0, 3}, {7}}));
    EXPECT_EQ(result.region_crystalline, (std::vector<char>{1, 1}));

    // Phase 7 non-crystalline: identity quaternions in region 2 only
    options.regions.non_crystalline_phases = {7};
    const RunPair amorphous = RunAndCompare(serial, options);
    const ms::MicrostructureResult& r2 = amorphous.alone.result;
    EXPECT_EQ(r2.region_crystalline, (std::vector<char>{1, 0}));
    for (int g = 0; g < r2.num_grains; ++g) {
        const auto gi = static_cast<std::size_t>(g);
        const auto s = static_cast<std::size_t>(amorphous.alone.report.grain_seed[gi]);
        if (r2.grain_region[gi] == 2) {
            EXPECT_EQ(r2.grain_quat[4 * gi], 1.0);
            EXPECT_EQ(r2.grain_quat[4 * gi + 1], 0.0);
            EXPECT_EQ(r2.grain_quat[4 * gi + 2], 0.0);
            EXPECT_EQ(r2.grain_quat[4 * gi + 3], 0.0);
        } else {
            const std::array<double, 4> input = {
                table.seeds.quat[4 * s], table.seeds.quat[4 * s + 1], table.seeds.quat[4 * s + 2],
                table.seeds.quat[4 * s + 3]};
            ExpectNormalizedFrom(&r2.grain_quat[4 * gi], input, "grain " + std::to_string(g));
        }
    }

    // Random orientations for a table: keyed by the table ids, not copied
    options.orientations.source = "random";
    options.regions.non_crystalline_phases.clear();
    const RunPair random = RunAndCompare(serial, options);
    EXPECT_NE(random.alone.result.grain_quat, result.grain_quat);
}

// ---------------------------------------------------------------------------------------------
// 2. ROI and filler seeds
// ---------------------------------------------------------------------------------------------

// roi.csv has 6 rows inside the ROI [0, 0.5]^3 and 3 outside (ids 50, 99, 7). The 3 outside rows
// are dropped. The filler ids start at 100 = max id of the whole table (99, a dropped row) + 1.
// The ROI faces lie on element faces of the 8^3 mesh, so its quadrature-point volumes are exact:
// V_roi = 1/8, V_outside = 7/8, and the automatic count is round(6 * 7) = 42. Every table seed
// lies in the closed ROI, no filler seed strictly inside it; filler grains carry the filler flag
// and ids >= 100, and with fragments = "auto" (merge for data sources) nothing is split.
TEST(DriverStage1d, RoiAndFillerSeeds) {
    WriteFixtures();
    mfem::Mesh serial = AutoHex(8, 8, 8);
    ms::MicrostructureOptions options = TableOptions("roi.csv");
    options.seeds.use_roi = true;
    options.seeds.roi_lo = {0.0, 0.0, 0.0};
    options.seeds.roi_hi = {0.5, 0.5, 0.5};
    options.seeds.fill_outside_roi = "random";
    const RunPair runs = RunAndCompare(serial, options);
    const ms::GenerationReport& report = runs.alone.report;
    const ms::MicrostructureResult& result = runs.alone.result;
    EXPECT_EQ(report.num_table_rows, 9);
    EXPECT_EQ(report.num_table_dropped, 3);
    EXPECT_EQ(report.num_table_seeds, 6);
    EXPECT_EQ(report.filler_source_id_offset, 100);
    EXPECT_DOUBLE_EQ(report.roi_material_volume, 0.125);
    EXPECT_DOUBLE_EQ(report.outside_material_volume, 0.875);
    EXPECT_EQ(report.num_filler_requested, 42);
    EXPECT_EQ(report.num_filler_seeds, 42);
    EXPECT_EQ(report.seeds.Size(), 48u);
    EXPECT_EQ(runs.world.report.seeds.x, report.seeds.x);

    // Table seeds in the ROI, filler seeds outside it, filler ids from the offset
    for (std::size_t i = 0; i < report.seeds.Size(); ++i) {
        const std::array<double, 3> p = {report.seeds.x[i], report.seeds.y[i], report.seeds.z[i]};
        if (i < 6) {
            EXPECT_TRUE(InsideClosed(p, options.seeds.roi_lo, options.seeds.roi_hi)) << i;
            EXPECT_LT(report.seeds.source_id[i], 7);
        } else {
            EXPECT_FALSE(StrictlyInside(p, options.seeds.roi_lo, options.seeds.roi_hi)) << i;
            EXPECT_GE(report.seeds.source_id[i], 100);
        }
    }
    // Filler flags follow the parent seed
    int fillers = 0;
    for (int g = 0; g < result.num_grains; ++g) {
        const auto gi = static_cast<std::size_t>(g);
        const bool filler = report.grain_seed[gi] >= 6;
        EXPECT_EQ(result.grain_filler[gi], filler ? 1 : 0);
        if (filler) {
            ++fillers;
            EXPECT_GE(result.grain_source_id[gi], 100);
        }
    }
    EXPECT_EQ(report.num_filler_grains, fillers);
    EXPECT_GT(fillers, 30);
    EXPECT_EQ(report.cleanup.fragments_split, 0);

    // Synthetic phases with filler: every grain, filler grains included, gets one of the phases
    options.regions.mode = "phase_fractions";
    options.regions.phase_ids = {0, 1};
    options.regions.phase_fractions = {0.5, 0.5};
    const RunPair fractions = RunAndCompare(serial, options);
    for (const int phase : fractions.alone.result.grain_phase) {
        EXPECT_TRUE(phase == 0 || phase == 1);
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Volume targets from the table
// ---------------------------------------------------------------------------------------------

// distribution = "from_source": the target of seed i is its table volume scaled to the material
// volume (1 for the unit cube), so target_i / sum = volume_i / sum of volumes. A Laguerre volume
// fit then reaches the 5% tolerance on the quadrature points (the fit report's last error).
// With ROI and filler the table targets share the ROI volume V_roi = 1/8 and the filler seeds
// share V_outside = 7/8 equally.
TEST(DriverStage1d, VolumeTargetsFromTable) {
    WriteFixtures();
    mfem::Mesh serial = AutoHex(12, 12, 12);
    ms::MicrostructureOptions options = TableOptions("sized.csv");
    options.seeds.size.distribution = "from_source";
    options.tessellation.metric = ms::MetricType::LAGUERRE;
    options.tessellation.fit_volumes.enabled = true;
    options.tessellation.fit_volumes.tolerance = 0.05;
    options.tessellation.fit_volumes.max_iterations = 200;
    const RunPair runs = RunAndCompare(serial, options);
    const ms::GenerationReport& report = runs.alone.report;
    EXPECT_TRUE(report.table_has_volume);
    EXPECT_EQ(report.seed_target_volume.size(), 8u);
    double target_sum = 0.0;
    for (const double v : report.seed_target_volume) {
        target_sum += v;
    }
    EXPECT_NEAR(target_sum, 1.0, 1e-12);
    for (std::size_t i = 0; i < 8; ++i) {
        // table volume (i + 1) * 0.01, sum 0.36
        EXPECT_NEAR(report.seed_target_volume[i], (i + 1) * 0.01 / 0.36, 1e-12) << i;
    }
    EXPECT_TRUE(report.ran_weight_fit);
    EXPECT_FALSE(report.weight_fit.max_relative_error.empty());
    if (!report.weight_fit.max_relative_error.empty()) {
        EXPECT_LE(report.weight_fit.max_relative_error.back(), 0.05);
    }
    EXPECT_EQ(runs.world.report.seeds.weight, report.seeds.weight);

    // ROI + filler: table targets fill the ROI share, filler targets are equal
    mfem::Mesh coarse = AutoHex(8, 8, 8);
    ms::MicrostructureOptions roi = TableOptions("roi.csv");
    roi.seeds.use_roi = true;
    roi.seeds.roi_lo = {0.0, 0.0, 0.0};
    roi.seeds.roi_hi = {0.5, 0.5, 0.5};
    roi.seeds.fill_outside_roi = "random";
    roi.seeds.filler_num_seeds = 10;
    roi.seeds.size.distribution = "from_source";
    roi.tessellation.metric = ms::MetricType::LAGUERRE;
    const RunPair filled = RunAndCompare(coarse, roi);
    const std::vector<double>& t = filled.alone.report.seed_target_volume;
    EXPECT_EQ(t.size(), 16u);
    if (t.size() == 16u) {
        double table_sum = 0.0;
        for (std::size_t i = 0; i < 6; ++i) {
            table_sum += t[i];
        }
        EXPECT_NEAR(table_sum, 0.125, 1e-12);
        // roi.csv volumes of the kept rows: 1, 2, 1, 3, 1, 2 (sum 10)
        EXPECT_NEAR(t[3], 0.125 * 3.0 / 10.0, 1e-12);
        for (std::size_t j = 6; j < 16; ++j) {
            EXPECT_NEAR(t[j], 0.0875, 1e-12) << j;
        }
        // Laguerre starting weights w = r^2 of the targets (no table weights)
        const double r3 = std::cbrt(3.0 * t[3] / (4.0 * PI));
        EXPECT_NEAR(filled.alone.report.seeds.weight[3], r3 * r3, 1e-12);
    }
}

// ---------------------------------------------------------------------------------------------
// 4. Voxel maps
// ---------------------------------------------------------------------------------------------

#ifdef AXOM_USE_HDF5

// The 20 x 16 x 12 voxel map of common/voxel_fixture.hpp on the voxel-aligned hex mesh (one
// element per voxel, the same lexicographic order), read in the canonical layout and in the
// DREAM.3D SIMPL v6 layout. Every element gets its voxel's feature (the 8 hole elements are
// filled from their neighbors), grains are numbered in ascending feature id (3, 5, 8, 13, 21,
// 34 -> 1..6), and the phases r % 3 = {0, 2, 2, 1, 0, 1} go to 2 regions through
// {0: 1, 1: 1, 2: 2}. Canonical quaternions are the stored rows normalized; the DREAM.3D file
// stores float32 x y z w, so with quat_layout = "xyzw" and conjugate = false the result is the
// float-rounded w x y z quaternion divided by its norm (float rounding moves the norm off 1 by
// ~1e-8, so normalization is visible here).
TEST(DriverStage1d, VoxelMapsBothFormats) {
    WriteFixtures();
    using voxel_fixture::H;
    using voxel_fixture::NX;
    using voxel_fixture::NY;
    using voxel_fixture::NZ;
    mfem::Mesh serial =
        mfem::Mesh::MakeCartesian3D(NX, NY, NZ, mfem::Element::HEXAHEDRON, NX * H, NY * H, NZ * H,
                                    false);
    for (const bool dream3d : {false, true}) {
        SCOPED_TRACE(dream3d ? "dream3d" : "canonical");
        ms::MicrostructureOptions options;
        options.source.type = "voxel_map";
        options.voxel_map.format = dream3d ? "dream3d" : "exaconstit";
        options.voxel_map.file = DataDir() / (dream3d ? "simpl.h5" : "canonical.h5");
        if (dream3d) {
            options.voxel_map.quat_layout = "xyzw";
            options.voxel_map.conjugate = 0;
        }
        options.orientations.source = "from_source";
        options.regions.mode = "from_phase";
        options.regions.phase_to_region = {{0, 1}, {1, 1}, {2, 2}};
        const RunPair runs = RunAndCompare(serial, options);
        const ms::MicrostructureResult& result = runs.alone.result;
        const ms::GenerationReport& report = runs.alone.report;
        EXPECT_TRUE(report.used_voxel_map);
        EXPECT_EQ(report.voxel_num_filled, 8);
        EXPECT_EQ(report.voxel_num_outside_points, 0);
        EXPECT_EQ(report.voxel_grid.dims, (std::array<std::int64_t, 3>{NX, NY, NZ}));
        EXPECT_EQ(result.num_grains, 6);
        EXPECT_EQ(report.grain_feature, (std::vector<std::int64_t>{3, 5, 8, 13, 21, 34}));

        // Labels: the grain of every element's (filled) voxel feature
        const std::vector<std::int64_t>& filled = voxel_fixture::MapData().filled;
        int mismatches = 0;
        for (std::size_t e = 0; e < runs.alone.labels.size(); ++e) {
            const auto it = std::find(voxel_fixture::kUsedIds.begin(),
                                      voxel_fixture::kUsedIds.end(), filled[e]);
            const int grain = static_cast<int>(it - voxel_fixture::kUsedIds.begin()) + 1;
            mismatches += (runs.alone.labels[e] != grain) ? 1 : 0;
        }
        EXPECT_EQ(mismatches, 0);

        // Orientations, phases and regions per feature
        for (int g = 0; g < result.num_grains; ++g) {
            const auto gi = static_cast<std::size_t>(g);
            const int feature = static_cast<int>(report.grain_feature[gi]);
            std::array<double, 4> stored = voxel_fixture::FeatureQuat(feature);
            if (dream3d) {
                for (double& v : stored) {
                    v = double(float(v));
                }
            }
            ExpectNormalizedFrom(&result.grain_quat[4 * gi], stored,
                                 "feature " + std::to_string(feature));
            const int phase = voxel_fixture::FeaturePhase(feature);
            EXPECT_EQ(result.grain_phase[gi], phase);
            EXPECT_EQ(result.grain_region[gi], phase == 2 ? 2 : 1);
            EXPECT_EQ(result.grain_source_id[gi], feature);
            EXPECT_EQ(report.grain_seed[gi], -1);
        }
        EXPECT_EQ(result.num_regions, 2);
        EXPECT_EQ(result.region_phases, (std::vector<std::vector<int>>{{0, 1}, {2}}));

        // Single region and random orientations: the phases are still read (informative
        // grain_phase, region 1 lists them), as for a seed table's phase column
        ms::MicrostructureOptions single = options;
        single.regions = ms::RegionOptions{};
        single.orientations.source = "random";
        const RunPair one = RunAndCompare(serial, single);
        EXPECT_EQ(one.alone.result.num_regions, 1);
        EXPECT_EQ(one.alone.result.region_phases, (std::vector<std::vector<int>>{{0, 1, 2}}));
        for (int g = 0; g < one.alone.result.num_grains; ++g) {
            const auto gi = static_cast<std::size_t>(g);
            EXPECT_EQ(one.alone.result.grain_phase[gi],
                      voxel_fixture::FeaturePhase(
                          static_cast<int>(one.alone.report.grain_feature[gi])));
        }
    }
}

// DREAM.3D quaternions without the storage declarations are refused by validate(); a run that
// skips validate() fails on every rank when the features are read (the reader refuses, so the
// orientation would never be guessed). Voxel maps refuse tessellation options (here a Laguerre
// metric) before any work.
TEST(DriverStage1d, VoxelMapRefusals) {
    WriteFixtures();
    mfem::Mesh serial = AutoHex(4, 4, 4, 1.25, 1.0, 0.75);
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    ms::MicrostructureOptions options;
    options.mesh.type = "auto";
    options.source.type = "voxel_map";
    options.voxel_map.format = "dream3d";
    options.voxel_map.file = DataDir() / "simpl.h5";
    options.orientations.source = "from_source";
    EXPECT_FALSE(options.validate());
    bool threw = false;
    try {
        ms::GenerateMicrostructure(options, *d.pmesh, MPI_COMM_WORLD);
    } catch (const std::exception&) {
        threw = true;
    }
    EXPECT_TRUE(threw);

    ms::MicrostructureOptions laguerre;
    laguerre.source.type = "voxel_map";
    laguerre.voxel_map.file = DataDir() / "canonical.h5";
    laguerre.tessellation.metric = ms::MetricType::LAGUERRE;
    EXPECT_TRUE(Refused(*d.pmesh, laguerre));
}

#endif // AXOM_USE_HDF5

// ---------------------------------------------------------------------------------------------
// 5. Phase fractions
// ---------------------------------------------------------------------------------------------

// Synthetic phases 1 (70%) and 2 (30%) over 30 random grains on a 10^3 unit-cube mesh. The
// greedy assignment guarantees |achieved_k - f_k| <= v_max / V (regions/region_assigner.hpp);
// on this uniform mesh a grain's volume is its element count / 1000, so v_max / V = (largest
// element count) / 1000. The phases are compact-numbered into regions 1 and 2, and the result
// is identical on every rank count (RunAndCompare compares grain_phase).
TEST(DriverStage1d, PhaseFractionsWithinBound) {
    mfem::Mesh serial = AutoHex(10, 10, 10);
    ms::MicrostructureOptions options;
    options.seeds.num_seeds = 30;
    options.seeds.rng_seed = 17u;
    options.regions.mode = "phase_fractions";
    options.regions.phase_ids = {1, 2};
    options.regions.phase_fractions = {0.7, 0.3};
    const RunPair runs = RunAndCompare(serial, options);
    const ms::MicrostructureResult& result = runs.alone.result;
    const ms::GenerationReport& report = runs.alone.report;
    std::vector<int> count(static_cast<std::size_t>(result.num_grains), 0);
    for (const int g : runs.alone.labels) {
        count[static_cast<std::size_t>(g - 1)]++;
    }
    const double bound = *std::max_element(count.begin(), count.end()) / 1000.0;
    EXPECT_EQ(report.phase_fraction_ids, (std::vector<int>{1, 2}));
    EXPECT_EQ(report.phase_fraction_achieved.size(), 2u);
    if (report.phase_fraction_achieved.size() == 2u) {
        EXPECT_LE(std::fabs(report.phase_fraction_achieved[0] - 0.7), bound + 1e-12);
        EXPECT_LE(std::fabs(report.phase_fraction_achieved[1] - 0.3), bound + 1e-12);
    }
    // Element-count fractions agree with the reported ones
    double phase1 = 0.0;
    for (int g = 0; g < result.num_grains; ++g) {
        const auto gi = static_cast<std::size_t>(g);
        EXPECT_TRUE(result.grain_phase[gi] == 1 || result.grain_phase[gi] == 2);
        EXPECT_EQ(result.grain_region[gi], result.grain_phase[gi]);
        if (result.grain_phase[gi] == 1) {
            phase1 += count[gi] / 1000.0;
        }
    }
    EXPECT_NEAR(phase1, report.phase_fraction_achieved[0], 1e-12);
    EXPECT_EQ(result.num_regions, 2);
    EXPECT_EQ(result.region_phases, (std::vector<std::vector<int>>{{1}, {2}}));
}

// ---------------------------------------------------------------------------------------------
// 6. Build plate passthrough
// ---------------------------------------------------------------------------------------------

namespace {

/** 6 x 6 x 8 unit box whose two bottom element layers (z < 1/4) have attribute 2. */
mfem::Mesh PlateMesh() {
    mfem::Mesh mesh = AutoHex(6, 6, 8);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        const int layer = e / 36; // lexicographic order: x fastest, then y, then z
        mesh.SetAttribute(e, layer < 2 ? 2 : 1);
    }
    mesh.SetAttributes();
    return mesh;
}

} // namespace

// The two bottom layers (72 elements, attribute 2) are a build plate kept as one grain in region
// 2; the 216 elements above get 10 random seeds (region 1). The plate grain is the last grain
// N + 1 with the identity quaternion, phase -1, source id -2 and no filler flag; every plate
// element carries it and no other element does. Every seed lies above the plate (seeding ran on
// the sub-mesh). The report's adjacency belongs to the whole mesh: RunDriver checks its element
// count and computes grain statistics on the returned labels, which give the plate grain its 72
// elements. A passthrough region equal to the grains' region (1) is allowed (one region); a gap
// (region 3 with no region 2) is refused.
TEST(DriverStage1d, BuildPlatePassthrough) {
    mfem::Mesh serial = PlateMesh();
    ms::MicrostructureOptions options;
    options.seeds.num_seeds = 10;
    options.seeds.rng_seed = 3u;
    options.mesh.passthrough_attributes = {{2, 2}};
    const RunPair runs = RunAndCompare(serial, options);
    const ms::MicrostructureResult& result = runs.alone.result;
    const ms::GenerationReport& report = runs.alone.report;
    EXPECT_TRUE(report.used_submesh);
    EXPECT_EQ(report.num_active_elements, 216);
    EXPECT_EQ(report.num_elements, 288);
    EXPECT_EQ(report.passthrough_attributes, (std::vector<int>{2}));
    EXPECT_EQ(report.passthrough_regions, (std::vector<int>{2}));
    EXPECT_EQ(report.passthrough_element_counts, (std::vector<std::int64_t>{72}));
    EXPECT_EQ(report.num_passthrough_grains, 1);
    const int plate = result.num_grains;
    const auto pi = static_cast<std::size_t>(plate - 1);
    EXPECT_EQ(result.grain_quat[4 * pi], 1.0);
    EXPECT_EQ(result.grain_quat[4 * pi + 1], 0.0);
    EXPECT_EQ(result.grain_quat[4 * pi + 2], 0.0);
    EXPECT_EQ(result.grain_quat[4 * pi + 3], 0.0);
    EXPECT_EQ(result.grain_region[pi], 2);
    EXPECT_EQ(result.grain_phase[pi], -1);
    EXPECT_EQ(result.grain_source_id[pi], -2);
    EXPECT_EQ(result.grain_filler[pi], 0);
    EXPECT_EQ(report.grain_seed[pi], -1);
    EXPECT_EQ(result.num_regions, 2);
    EXPECT_EQ(result.region_crystalline, (std::vector<char>{1, 0}));
    for (std::size_t e = 0; e < runs.alone.labels.size(); ++e) {
        const bool in_plate = (e / 36) < 2;
        EXPECT_EQ(runs.alone.labels[e] == plate, in_plate) << e;
        if (!in_plate) {
            EXPECT_EQ(result.grain_region[static_cast<std::size_t>(runs.alone.labels[e] - 1)], 1);
        }
    }
    for (std::size_t i = 0; i < report.seeds.Size(); ++i) {
        EXPECT_GE(report.seeds.z[i], 0.25) << i;
    }
    EXPECT_EQ(runs.alone.stats_elements.size(), static_cast<std::size_t>(result.num_grains));
    if (!runs.alone.stats_elements.empty()) {
        EXPECT_EQ(runs.alone.stats_elements.back(), 72);
    }

    // The plate in the grains' region: one region, still crystalline
    options.mesh.passthrough_attributes = {{2, 1}};
    const RunPair same = RunAndCompare(serial, options);
    EXPECT_EQ(same.alone.result.num_regions, 1);
    EXPECT_EQ(same.alone.result.region_crystalline, (std::vector<char>{1}));

    // A gap in the region ids is refused on every rank
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    options.mesh.passthrough_attributes = {{2, 3}};
    EXPECT_TRUE(Refused(*d.pmesh, options));
}

// ---------------------------------------------------------------------------------------------
// 7. Refused combinations
// ---------------------------------------------------------------------------------------------

// Each combination is refused by the driver with std::invalid_argument on every rank, before
// any long work: a boundary clearance or periodic axes with passthrough attributes (the
// interface would count as an outer boundary), Lloyd relaxation or Johnson-Mehl nucleation with
// a seed table (measured positions would move / be replaced), filler without an ROI, sizes from
// the source without a table, and data the options need but the table lacks (quaternions,
// phases, shape tensors, a size column).
TEST(DriverStage1d, RefusedCombinations) {
    WriteFixtures();
    mfem::Mesh plate = PlateMesh();
    Distributed d = Distribute(plate, MPI_COMM_WORLD);
    using O = ms::MicrostructureOptions;
    auto refused = [&](const O& options) { return Refused(*d.pmesh, options); };

    O pass;
    pass.seeds.num_seeds = 5;
    pass.mesh.passthrough_attributes = {{2, 2}};
    O clearance = pass;
    clearance.seeds.boundary_clearance = 0.05;
    EXPECT_TRUE(refused(clearance));
    O periodic = pass;
    periodic.tessellation.periodic = {true, false, false};
    EXPECT_TRUE(refused(periodic));

    O lloyd = TableOptions("measured.csv");
    lloyd.tessellation.lloyd_iterations = 2;
    EXPECT_TRUE(refused(lloyd));
    O jm = TableOptions("measured.csv");
    jm.seeds.placement = ms::SeedPlacement::JOHNSON_MEHL;
    jm.tessellation.metric = ms::MetricType::ADDITIVE;
    EXPECT_TRUE(refused(jm));
    O filler = TableOptions("measured.csv");
    filler.seeds.fill_outside_roi = "random";
    EXPECT_TRUE(refused(filler));
    O sizes;
    sizes.seeds.num_seeds = 5;
    sizes.seeds.size.distribution = "from_source";
    EXPECT_TRUE(refused(sizes));

    // Data the table lacks (roi.csv has no quaternions, phases or shape columns)
    O quats = TableOptions("roi.csv");
    quats.orientations.source = "from_source";
    EXPECT_TRUE(refused(quats));
    O phases = TableOptions("roi.csv");
    phases.regions.mode = "from_phase";
    EXPECT_TRUE(refused(phases));
    O shapes = TableOptions("roi.csv");
    shapes.tessellation.metric = ms::MetricType::QUADRIC;
    shapes.tessellation.anisotropy_kind = "per_seed";
    EXPECT_TRUE(refused(shapes));
    O no_volume = TableOptions("measured.csv");
    no_volume.seeds.size.distribution = "from_source";
    no_volume.tessellation.metric = ms::MetricType::LAGUERRE;
    EXPECT_TRUE(refused(no_volume));
}

// The orientation assigner normalizes supplied rows (here norm 2 -> the same direction with norm
// 1, sign kept), normalizes random rows the same way, and refuses a zero or non-finite supplied row
// instead of silently writing the identity. Host only, no MPI.
TEST(DriverStage1d, AssignerNormalizesSuppliedQuaternions) {
    ms::MicrostructureResult result;
    const std::vector<std::int64_t> ids = {5, 6};
    std::vector<double> rows = {-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    ms::AssignMixedGrainOrientations(1u, ids, rows, {1, 0}, result);
    EXPECT_EQ(result.grain_quat[0], -1.0);
    EXPECT_EQ(result.grain_quat[1], 0.0);
    ExpectUnitQuaternions(result);
    EXPECT_THROW(ms::AssignMixedGrainOrientations(1u, ids, rows, {1, 1}, result),
                 std::invalid_argument);
    rows[4] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(ms::AssignMixedGrainOrientations(1u, ids, rows, {0, 1}, result),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// R11 additions: filler weights, data-driven splits, passthrough with phases, empty active rank
// ---------------------------------------------------------------------------------------------

// Laguerre table weights with a large common offset C = 100 (offset.csv): if the filler seeds got
// weight 0, f = |x - s|^2 - w would be ~100 larger for them everywhere and every filler grain
// would be empty. With the mean table weight (no size targets) or their own target weight plus
// the mean table offset (size targets from the table), every seed owns elements. Filler spacing
// (filler_min_distance_factor) keeps two filler seeds from sharing one element.
TEST(DriverStage1d, FillerWeightsFollowTableWeights) {
    WriteFixtures();
    mfem::Mesh serial = AutoHex(8, 8, 8);
    ms::MicrostructureOptions options = TableOptions("offset.csv");
    options.tessellation.metric = ms::MetricType::LAGUERRE;
    options.seeds.use_roi = true;
    options.seeds.roi_lo = {0.0, 0.0, 0.0};
    options.seeds.roi_hi = {0.5, 0.5, 0.5};
    options.seeds.fill_outside_roi = "random";
    options.seeds.filler_min_distance_factor = 0.5;
    for (const bool targets : {false, true}) {
        SCOPED_TRACE(targets ? "size targets from the table" : "no size targets");
        options.seeds.size.distribution = targets ? "from_source" : "none";
        const RunPair runs = RunAndCompare(serial, options);
        const ms::GenerationReport& report = runs.alone.report;
        EXPECT_GT(report.num_filler_seeds, 0);
        EXPECT_EQ(report.num_empty_seeds, 0);
        // The filler weights carry the offset: no filler weight is far below the table's 100
        for (std::size_t j = 6; j < report.seeds.weight.size(); ++j) {
            EXPECT_GT(report.seeds.weight[j], 99.0) << j;
        }
        if (!targets) {
            EXPECT_EQ(report.seeds.weight.back(), 100.0); // the mean of the table weights
        }
    }
}

// Fragment split on measured grains. Seed 42 of ushape.csv sits in the notch of the U, so its
// grain comes out in two pieces, one in each arm; with fragments = "split" the extra piece
// becomes a new grain with a derived source id that keeps the parent's normalized quaternion,
// phase and region (from_source orientations, from_phase regions).
// Filler pieces: a random filler grain split the same way gets a random orientation of its own
// derived id, not the parent's. Which filler seeds produce a split depends on their random
// positions, so the first RNG key (1..40) that yields a split filler grain is used; the check is
// deterministic for that key.
TEST(DriverStage1d, SplitDataDrivenGrainsInheritParentData) {
    WriteFixtures();
    mfem::Mesh serial = UShape(4);
    ms::MicrostructureOptions options = TableOptions("ushape.csv");
    options.cleanup.fragments = "split";
    options.orientations.source = "from_source";
    options.regions.mode = "from_phase";
    const RunPair runs = RunAndCompare(serial, options);
    const ms::MicrostructureResult& result = runs.alone.result;
    const ms::GenerationReport& report = runs.alone.report;
    EXPECT_GE(report.cleanup.fragments_split, 1);
    std::vector<std::size_t> pieces; // grains grown from seed 42 (index 5)
    for (std::size_t g = 0; g < static_cast<std::size_t>(result.num_grains); ++g) {
        if (report.grain_seed[g] == 5) {
            pieces.push_back(g);
        }
    }
    EXPECT_GE(pieces.size(), 2u);
    int derived = 0;
    for (const std::size_t g : pieces) {
        ExpectNormalizedFrom(&result.grain_quat[4 * g], {0.5004, 0.5, -0.5, 0.5},
                             "piece " + std::to_string(g + 1));
        EXPECT_EQ(result.grain_phase[g], 3);
        EXPECT_EQ(result.grain_region[g], 2); // phases {0, 3} -> regions {1, 2}
        derived += (result.grain_source_id[g] != 42) ? 1 : 0;
    }
    EXPECT_EQ(derived, static_cast<int>(pieces.size()) - 1);

    // Filler pieces
    ms::MicrostructureOptions filler = TableOptions("ushape_roi.csv");
    filler.seeds.use_roi = true;
    filler.seeds.roi_lo = {0.0, 0.0, 0.0};
    filler.seeds.roi_hi = {1.0, 1.0, 1.0};
    filler.seeds.fill_outside_roi = "random";
    filler.seeds.filler_num_seeds = 10;
    filler.cleanup.fragments = "split";
    filler.orientations.source = "from_source";
    bool found = false;
    for (std::uint64_t key = 1; key <= 40 && !found; ++key) {
        filler.seeds.rng_seed = key;
        const DriverRun run = RunDriver(serial, filler, MPI_COMM_SELF);
        const ms::MicrostructureResult& r = run.result;
        for (std::size_t g = 0; g < static_cast<std::size_t>(r.num_grains); ++g) {
            const int s = run.report.grain_seed[g];
            const std::int64_t parent_id = run.report.seeds.source_id[static_cast<std::size_t>(s)];
            if (r.grain_filler[g] == 0 || r.grain_source_id[g] == parent_id) {
                continue;
            }
            found = true;
            SCOPED_TRACE("rng_seed " + std::to_string(key));
            // Random orientation of the derived id, normalized
            double expected[4];
            ms::RandomQuaternion(ms::CounterRng(key, ms::RngPurpose::ORIENTATION,
                                                static_cast<std::uint64_t>(r.grain_source_id[g])),
                                 0u, expected);
            ExpectNormalizedFrom(&r.grain_quat[4 * g],
                                 {expected[0], expected[1], expected[2], expected[3]},
                                 "split filler grain");
            // and not the parent's orientation
            for (std::size_t h = 0; h < static_cast<std::size_t>(r.num_grains); ++h) {
                if (r.grain_source_id[h] == parent_id) {
                    EXPECT_NE(r.grain_quat[4 * h], r.grain_quat[4 * g]);
                }
            }
            // The distributed run agrees for this key
            const RunPair check = RunAndCompare(serial, filler);
            EXPECT_EQ(check.alone.result.grain_quat, r.grain_quat);
            break;
        }
    }
    EXPECT_TRUE(found);
}

// Passthrough together with regions from phases: plate.csv phases {0, 1} give regions 1 and 2,
// and the plate (attribute 2) is mapped to region 3, so the regions are exactly 1..3 with region
// 3 non-crystalline and without phases. Mapping the plate to region 4 leaves region 3 unused,
// which only the driver can see (the number of phase regions is known after the table is read):
// it is refused on every rank.
TEST(DriverStage1d, PassthroughWithPhaseRegions) {
    WriteFixtures();
    mfem::Mesh serial = PlateMesh();
    ms::MicrostructureOptions options = TableOptions("plate.csv");
    options.regions.mode = "from_phase";
    options.mesh.passthrough_attributes = {{2, 3}};
    EXPECT_TRUE([&] {
        ms::MicrostructureOptions v = options;
        v.mesh.type = "auto";
        return v.validate();
    }());
    const RunPair runs = RunAndCompare(serial, options);
    const ms::MicrostructureResult& result = runs.alone.result;
    EXPECT_EQ(result.num_regions, 3);
    EXPECT_EQ(result.region_phases, (std::vector<std::vector<int>>{{0}, {1}, {}}));
    EXPECT_EQ(result.region_crystalline, (std::vector<char>{1, 1, 0}));
    EXPECT_EQ(result.grain_region.back(), 3);
    for (int g = 0; g + 1 < result.num_grains; ++g) {
        const auto gi = static_cast<std::size_t>(g);
        EXPECT_EQ(result.grain_region[gi], result.grain_phase[gi] + 1);
    }

    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    options.mesh.passthrough_attributes = {{2, 4}};
    EXPECT_TRUE(Refused(*d.pmesh, options));
}

// A partition in which the last rank owns only build-plate elements, so it has zero active
// elements (np >= 2; at np = 1 rank 0 owns everything): the sub-mesh is empty on that rank, yet
// seeding, labeling, cleanup, the passthrough mapping and the whole-mesh adjacency complete, and
// the labels equal the single-rank reference.
TEST(DriverStage1d, RankWithoutActiveElements) {
    mfem::Mesh serial = PlateMesh();
    const int nranks = NumRanks();
    std::vector<int> owner(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        const bool plate = (e / 36) < 2;
        owner[static_cast<std::size_t>(e)] =
            (nranks == 1) ? 0 : (plate ? nranks - 1 : e % (nranks - 1));
    }
    ms::MicrostructureOptions options;
    options.seeds.num_seeds = 10;
    options.seeds.rng_seed = 4u;
    options.mesh.passthrough_attributes = {{2, 2}};
    Distributed d = DistributeWithOwners(serial, owner);
    const DriverRun world = RunDriverOn(d, options, MPI_COMM_WORLD);
    const DriverRun alone = RunDriver(serial, options, MPI_COMM_SELF);
    EXPECT_EQ(world.labels, alone.labels);
    EXPECT_EQ(world.result.grain_quat, alone.result.grain_quat);
    EXPECT_EQ(world.result.grain_region, alone.result.grain_region);
    EXPECT_EQ(world.stats_elements, alone.stats_elements);
    EXPECT_EQ(world.report.num_active_elements, 216);
    ExpectUnitQuaternions(world.result);
}

// ---------------------------------------------------------------------------------------------
// 8. Example options files
// ---------------------------------------------------------------------------------------------

// The Stage 1d examples that run on an auto mesh go through the driver unchanged (apart from the
// seed table path, which is relative to the directory the tool runs in), and give the same
// result on every rank count. seed_table_auto.toml uses every column of the checked-in
// all_columns.csv: ROI with filler, normalized table quaternions, phases -> regions with a
// filler phase, table weights, volume targets with a volume fit, and per-seed Quadric shape
// tensors (the filler seeds get their own spherical tensors, so every seed has 6 tensor
// entries).
// phase_fractions_auto.toml: synthetic phases on random seeds, one of them non-crystalline.
TEST(DriverStage1d, ExampleOptionsFilesRun) {
    for (const char* name : {"seed_table_auto.toml", "phase_fractions_auto.toml"}) {
        SCOPED_TRACE(name);
        ms::MicrostructureOptions options =
            ms::MicrostructureOptions::parse_options_file(data_dir / "microstructure" / name);
        if (!options.seeds.file.empty()) {
            options.seeds.file = data_dir / "microstructure" / options.seeds.file;
        }
        EXPECT_TRUE(options.validate());
        mfem::Mesh serial = exaconstit::MakeAutoHexMesh(options.mesh.nxyz, options.mesh.mxyz);
        const RunPair runs = RunAndCompare(serial, options);
        EXPECT_GT(runs.alone.result.num_grains, 1);
        EXPECT_GE(runs.alone.result.num_regions, 2);
        if (options.tessellation.anisotropy_kind == "per_seed") {
            EXPECT_EQ(runs.alone.report.seeds.shape.size(), 6 * runs.alone.report.seeds.Size());
        }
    }
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
