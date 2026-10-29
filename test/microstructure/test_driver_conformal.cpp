/**
 * @file test_driver_conformal.cpp
 * @brief Driver-level tests of the conformal grain boundary step ([Conformal] enabled)
 * in GenerateMicrostructure (exaconstit_microstructure).
 *
 * What is checked, and why:
 *  - Hex box without Nodes, 8 Voronoi grains: the step runs and reports, the mesh keeps no Nodes
 *    (the fitted positions become its vertices), the quality floor holds with no inverted
 *    element, the grain boundary nodes end closer to the analytic boundaries, the box keeps its
 *    volume (its faces stay in place), the grain count is the same as without the step, and only
 *    a few elements change grain (label repair, the count the report gives).
 *  - Tet box: the tet floor (mean ratio) holds.
 *  - A mesh with order-2 Nodes (as ExaConstit's mesh at [Mesh] order = 2) is fitted and keeps its
 *    Nodes; order 3 is refused on every rank with std::invalid_argument.
 *  - Grain sources without analytic boundaries: a voxel map (the shared voxelized
 *    Voronoi fixture, one element per voxel) and geodesic labels on the U-shaped part fit every
 *    grain boundary node to the smoothed staircase; connected labels on the U-shaped part use
 *    analytic targets where valid and the smoothed staircase elsewhere. Each keeps the floor, no
 *    inversion, the part's volume and a smaller distance to the targets.
 *  - A passthrough plate (bottom quarter of a box): its elements keep their grain, the nodes on
 *    its top face keep z bit for bit, and its volume is unchanged.
 *  - mono_def essential dofs: on the hex, tet and order-2 boxes the nodes that ExaConstit's
 *    mono_def boundary conditions select (1e-12 tests against the global min / max coordinates,
 *    system_driver.cpp) are the same before and after the fit.
 *  - Without the step nothing moves (the vertices of the input mesh are kept bit for bit).
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows.
 */
#include "microstructure/conformal/conformal_fitter.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_options.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/voxel_fixture.hpp"
#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

namespace fs = std::filesystem;

/** Directory of the example options files and meshes (set by CMake). */
const fs::path data_dir = fs::path(MS_DRIVER_DATA_DIR);

/** Options of the tests: 8 random Voronoi seeds, the conformal step on or off. */
ms::MicrostructureOptions Options(bool conformal) {
    ms::MicrostructureOptions o;
    o.seeds.num_seeds = 8;
    o.seeds.rng_seed = 3;
    o.conformal.enabled = conformal;
    return o;
}

/** One driver run: the (possibly fitted) mesh, the result and the report. */
struct DriverRun {
    std::unique_ptr<mfem::ParMesh> pmesh; ///< the distributed mesh
    ms::MicrostructureResult result;      ///< the driver's result
    ms::GenerationReport report;          ///< the driver's report
};

/** Distributes `serial` over MPI_COMM_WORLD and runs GenerateMicrostructure (collective). */
DriverRun Generate(mfem::Mesh& serial, const ms::MicrostructureOptions& options) {
    DriverRun run;
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    run.pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    run.result = ms::GenerateMicrostructure(options, *run.pmesh, MPI_COMM_WORLD, &run.report);
    return run;
}

/** Generate with Options(conformal). */
DriverRun Generate(mfem::Mesh& serial, bool conformal) {
    return Generate(serial, Options(conformal));
}

/** Mesh volume, integrated exactly (the Jacobian of a trilinear hex is tri-quadratic). */
double Volume(mfem::ParMesh& pmesh) {
    double v = 0.0;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        mfem::ElementTransformation* tr = pmesh.GetElementTransformation(e);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(pmesh.GetElementGeometry(e), 6);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            tr->SetIntPoint(&ir.IntPoint(q));
            v += ir.IntPoint(q).weight * tr->Weight();
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return v;
}

/** Volume of the elements with grain `grain` (exact integration, global). */
double GrainVolume(const DriverRun& run, int grain) {
    double v = 0.0;
    mfem::ParMesh& pmesh = *run.pmesh;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        if (run.result.element_grain[e] != grain) {
            continue;
        }
        mfem::ElementTransformation* tr = pmesh.GetElementTransformation(e);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(pmesh.GetElementGeometry(e), 6);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            tr->SetIntPoint(&ir.IntPoint(q));
            v += ir.IntPoint(q).weight * tr->Weight();
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return v;
}

/** The checks every fit must pass; prints a summary line on rank 0. */
void CheckFit(const char* name, DriverRun& run, double floor, double volume) {
    const ms::GenerationReport& r = run.report;
    ASSERT_TRUE(r.ran_conformal);
    EXPECT_GT(r.conformal_fit.fit_nodes, 0);
    EXPECT_GE(r.conformal_fit.rounds, 1);
    EXPECT_EQ(r.conformal_fit.quality_after.inverted, 0);
    EXPECT_GE(r.conformal_fit.quality_after.min_scaled_jacobian, floor);
    EXPECT_LE(r.conformal_fit.quality_after.max_aspect_ratio, ms::MESH_ASPECT_RATIO_MAX);
    EXPECT_LT(r.conformal_fit.mean_deviation, r.conformal_fit.initial_deviation);
    EXPECT_NEAR(Volume(*run.pmesh), volume, 1e-12 * volume);
    if (mfem::Mpi::Root()) {
        std::printf("%s: %d grains, %lld fitted nodes (%lld smoothed), deviation %.3f -> %.3f h, "
                    "sj %.3f, ar %.2f, rounds %d, solves %d\n",
                    name,
                    run.result.num_grains,
                    static_cast<long long>(r.conformal_fit.fit_nodes),
                    static_cast<long long>(r.conformal_smoothed_nodes),
                    r.conformal_fit.initial_deviation,
                    r.conformal_fit.mean_deviation,
                    r.conformal_fit.quality_after.min_scaled_jacobian,
                    r.conformal_fit.quality_after.max_aspect_ratio,
                    r.conformal_fit.rounds,
                    r.conformal_fit.solves);
    }
}

/** The U-shaped part of the distance-mode tests (320 hexahedra, volume 5). */
mfem::Mesh UShape() {
    return mfem::Mesh((data_dir / "microstructure/u_shape_hex.mesh").string().c_str());
}

/** Elements whose grain differs between two runs on the same partition (global). */
int Differences(const DriverRun& a, const DriverRun& b) {
    int n = 0;
    for (int e = 0; e < a.pmesh->GetNE(); ++e) {
        n += (a.result.element_grain[e] != b.result.element_grain[e]) ? 1 : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return n;
}

/** Local vertices that moved from their input position (global count). */
int MovedVertices(mfem::ParMesh& pmesh, mfem::ParMesh& input) {
    int n = 0;
    for (int v = 0; v < pmesh.GetNV(); ++v) {
        for (int d = 0; d < 3; ++d) {
            n += (pmesh.GetVertex(v)[d] != input.GetVertex(v)[d]) ? 1 : 0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return n;
}

/**
 * Per node of the mesh (Nodes dofs, or vertices without Nodes), the essential-dof mask of
 * ExaConstit's mono_def boundary conditions (system_driver.cpp, mono_def_flag): with the global
 * min / max coordinates o, bit 0 = x dof (|x - ox| < 1e-12 and |z - oz| < 1e-12), bit 1 = y dof
 * (at the min corner), bit 2 = z dof (|z - oz| < 1e-12 or |z - oz_max| < 1e-12).
 */
std::vector<int> MonoDefMasks(mfem::ParMesh& pmesh) {
    std::vector<std::array<double, 3>> x;
    if (const mfem::GridFunction* nodes = pmesh.GetNodes()) {
        const mfem::FiniteElementSpace& fes = *nodes->FESpace();
        x.resize(static_cast<std::size_t>(fes.GetNDofs()));
        for (int i = 0; i < fes.GetNDofs(); ++i) {
            for (int d = 0; d < 3; ++d) {
                x[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] = (*nodes)(
                    fes.DofToVDof(i, d));
            }
        }
    } else {
        x.resize(static_cast<std::size_t>(pmesh.GetNV()));
        for (int v = 0; v < pmesh.GetNV(); ++v) {
            for (int d = 0; d < 3; ++d) {
                x[static_cast<std::size_t>(v)][static_cast<std::size_t>(d)] = pmesh.GetVertex(v)[d];
            }
        }
    }
    double lo[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL};
    double hi[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
    for (const auto& p : x) {
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], p[static_cast<std::size_t>(d)]);
            hi[d] = std::max(hi[d], p[static_cast<std::size_t>(d)]);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, lo, 3, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
    MPI_Allreduce(MPI_IN_PLACE, hi, 3, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
    std::vector<int> mask(x.size(), 0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const bool x0 = std::abs(x[i][0] - lo[0]) < 1e-12;
        const bool y0 = std::abs(x[i][1] - lo[1]) < 1e-12;
        const bool z0 = std::abs(x[i][2] - lo[2]) < 1e-12;
        const bool z1 = std::abs(x[i][2] - hi[2]) < 1e-12;
        mask[i] = ((x0 && z0) ? 1 : 0) | ((x0 && y0 && z0) ? 2 : 0) | ((z0 || z1) ? 4 : 0);
    }
    return mask;
}

/**
 * Nodes whose mono_def mask differs between the fitted mesh and the input (global count), and in
 * `marked` the number of nodes with a nonzero mask on the input (global).
 */
int MonoDefChanges(mfem::ParMesh& fitted, mfem::ParMesh& input, int& marked) {
    const std::vector<int> a = MonoDefMasks(fitted);
    const std::vector<int> b = MonoDefMasks(input);
    int n = (a.size() == b.size()) ? 0 : 1;
    marked = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        n += (a[i] != b[i]) ? 1 : 0;
        marked += (b[i] != 0) ? 1 : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &marked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return n;
}

} // namespace

// Hex box without Nodes: the fit moves vertices only when enabled, keeps the hex quality floor
// and the box volume (the outer faces stay planar), brings the nodes closer to the analytic
// boundaries, changes the grain of exactly the elements the repair reports, and leaves the
// mono_def boundary nodes as they were.
TEST(DriverConformal, HexBox) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(12, 12, 12, mfem::Element::HEXAHEDRON);
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    std::unique_ptr<mfem::ParMesh> input = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    DriverRun plain = Generate(serial, false);
    DriverRun fitted = Generate(serial, true);
    const ms::GenerationReport& r = fitted.report;
    EXPECT_FALSE(plain.report.ran_conformal);
    EXPECT_EQ(MovedVertices(*plain.pmesh, *input), 0);
    EXPECT_GT(MovedVertices(*fitted.pmesh, *input), 0);
    ASSERT_TRUE(r.ran_conformal);
    EXPECT_EQ(fitted.pmesh->GetNodes(), nullptr);
    EXPECT_GT(r.conformal_fit.fit_nodes, 0);
    EXPECT_GE(r.conformal_fit.rounds, 1);
    EXPECT_EQ(r.conformal_fit.quality_after.inverted, 0);
    EXPECT_GE(r.conformal_fit.quality_after.min_scaled_jacobian, ms::MESH_QUALITY_FLOOR_HEX);
    EXPECT_LT(r.conformal_fit.mean_deviation, 0.75 * r.conformal_fit.initial_deviation);
    // The quality of the mesh as it is now (vertices, no Nodes) is the reported one
    const ms::MeshQuality now = ms::ComputeMeshQuality(*fitted.pmesh);
    EXPECT_DOUBLE_EQ(now.min_scaled_jacobian, r.conformal_fit.quality_after.min_scaled_jacobian);
    EXPECT_NEAR(Volume(*fitted.pmesh), 1.0, 1e-12);
    EXPECT_EQ(fitted.result.num_grains, plain.result.num_grains);
    const int changed = Differences(plain, fitted);
    EXPECT_EQ(changed, r.conformal_repair.flips);
    EXPECT_LT(changed, serial.GetNE() / 20);
    EXPECT_NE(r.adjacency, nullptr);
    // mono_def boundary conditions find the same nodes
    int marked = 0;
    EXPECT_EQ(MonoDefChanges(*fitted.pmesh, *input, marked), 0);
    EXPECT_GT(marked, 0);
    if (mfem::Mpi::Root()) {
        std::printf("hex box: %d grains, %d flips, deviation %.3f -> %.3f h, sj %.3f, "
                    "aspect %.3f, %d rounds, %d solves\n",
                    fitted.result.num_grains,
                    changed,
                    r.conformal_fit.initial_deviation,
                    r.conformal_fit.mean_deviation,
                    r.conformal_fit.quality_after.min_scaled_jacobian,
                    r.conformal_fit.quality_after.max_aspect_ratio,
                    r.conformal_fit.rounds,
                    r.conformal_fit.solves);
    }
}

// Tet box: the tet mean-ratio floor holds, nothing inverts, the deviation does not grow and the
// volume is kept; mono_def finds the same nodes.
TEST(DriverConformal, TetBox) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    std::unique_ptr<mfem::ParMesh> input = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    DriverRun fitted = Generate(serial, true);
    int marked = 0;
    EXPECT_EQ(MonoDefChanges(*fitted.pmesh, *input, marked), 0);
    EXPECT_GT(marked, 0);
    const ms::GenerationReport& r = fitted.report;
    ASSERT_TRUE(r.ran_conformal);
    EXPECT_EQ(r.conformal_fit.quality_after.inverted, 0);
    EXPECT_GE(r.conformal_fit.quality_after.min_mean_ratio, ms::MESH_MEAN_RATIO_FLOOR_TET);
    EXPECT_LE(r.conformal_fit.mean_deviation, r.conformal_fit.initial_deviation);
    EXPECT_NEAR(Volume(*fitted.pmesh), 1.0, 1e-12);
}

// Order-3 Nodes are not supported by the fit: every rank throws std::invalid_argument.
TEST(DriverConformal, RefusesHigherOrder) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    std::unique_ptr<mfem::ParMesh> pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    pmesh->SetCurvature(3);
    EXPECT_THROW(ms::GenerateMicrostructure(Options(true), *pmesh, MPI_COMM_WORLD),
                 std::invalid_argument);
}

// Order-2 Nodes (ExaConstit's [Mesh] order = 2) are fitted at every nodal point and stay order 2.
TEST(DriverConformal, OrderTwoHook) {
    // A mesh with order-2 Nodes, as ExaConstit's mesh at [Mesh] order = 2: the fit moves every
    // nodal point and the mesh keeps its order-2 Nodes
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    DriverRun run;
    run.pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    run.pmesh->SetCurvature(2);
    std::unique_ptr<mfem::ParMesh> input = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    input->SetCurvature(2);
    run.result = ms::GenerateMicrostructure(Options(true), *run.pmesh, MPI_COMM_WORLD, &run.report);
    CheckFit("order 2", run, ms::MESH_QUALITY_FLOOR_HEX, 1.0);
    int marked = 0;
    EXPECT_EQ(MonoDefChanges(*run.pmesh, *input, marked), 0); // every nodal point
    EXPECT_GT(marked, 0);
    ASSERT_NE(run.pmesh->GetNodes(), nullptr);
    EXPECT_EQ(run.pmesh->GetNodes()->FESpace()->GetMaxElementOrder(), 2);
}

#ifdef AXOM_USE_HDF5
// Voxel grains have no analytic boundary, so every fitted node targets the smoothed staircase.
TEST(DriverConformal, VoxelMap) {
    using voxel_fixture::H;
    using voxel_fixture::NX;
    using voxel_fixture::NY;
    using voxel_fixture::NZ;
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const fs::path dir = fs::path("test_driver_conformal_data") / ("np" + std::to_string(size));
    if (mfem::Mpi::Root()) {
        fs::create_directories(dir);
        voxel_fixture::CanonicalOptions canonical;
        canonical.ids = voxel_fixture::MapData().ids;
        voxel_fixture::WriteCanonical(dir / "canonical.h5", canonical);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        NX, NY, NZ, mfem::Element::HEXAHEDRON, NX * H, NY * H, NZ * H, false);
    ms::MicrostructureOptions options;
    options.seeds.num_seeds = 0;
    options.source.type = "voxel_map";
    options.voxel_map.format = "exaconstit";
    options.voxel_map.file = dir / "canonical.h5";
    options.conformal.enabled = true;
    DriverRun run = Generate(serial, options);
    CheckFit("voxel map", run, ms::MESH_QUALITY_FLOOR_HEX, NX * H * NY * H * NZ * H);
    EXPECT_EQ(run.report.conformal_smoothed_nodes, run.report.conformal_fit.fit_nodes);
}
#endif

// Geodesic labels have no analytic boundary (every node on the smoothed staircase); connected
// labels keep analytic targets where the cells are unchanged, so fewer nodes are smoothed.
TEST(DriverConformal, GeodesicAndConnected) {
    mfem::Mesh serial = UShape();
    ms::MicrostructureOptions options = Options(true);
    options.seeds.num_seeds = 10;
    options.tessellation.distance = ms::DistanceMode::GEODESIC;
    DriverRun geodesic = Generate(serial, options);
    CheckFit("geodesic", geodesic, ms::MESH_QUALITY_FLOOR_HEX, 5.0);
    EXPECT_EQ(geodesic.report.conformal_smoothed_nodes, geodesic.report.conformal_fit.fit_nodes);

    options.tessellation.distance = ms::DistanceMode::CONNECTED;
    DriverRun connected = Generate(serial, options);
    CheckFit("connected", connected, ms::MESH_QUALITY_FLOOR_HEX, 5.0);
    EXPECT_LT(connected.report.conformal_smoothed_nodes, connected.report.conformal_fit.fit_nodes);
}

// A passthrough plate keeps its shape: its elements keep their grain, its top face nodes keep z
// exactly, and its volume stays 1/4.
TEST(DriverConformal, PassthroughPlate) {
    // Box 8 x 8 x 8; the bottom two layers (z < 1/4) are a passthrough plate (attribute 2)
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(8, 8, 8, mfem::Element::HEXAHEDRON);
    mfem::Vector c(3);
    for (int e = 0; e < serial.GetNE(); ++e) {
        serial.GetElementCenter(e, c);
        serial.SetAttribute(e, c(2) < 0.25 ? 2 : 1);
    }
    serial.SetAttributes();
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    std::unique_ptr<mfem::ParMesh> input = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    ms::MicrostructureOptions options = Options(true);
    options.mesh.active_attributes = {1};
    options.mesh.passthrough_attributes = {{2, 2}};
    DriverRun run = Generate(serial, options);
    CheckFit("passthrough plate", run, ms::MESH_QUALITY_FLOOR_HEX, 1.0);
    const int plate = run.result.num_grains; // the passthrough grain comes last
    int wrong_grain = 0;
    int moved_off_plate = 0;
    for (int e = 0; e < run.pmesh->GetNE(); ++e) {
        input->GetElementCenter(e, c);
        wrong_grain += ((c(2) < 0.25) != (run.result.element_grain[e] == plate)) ? 1 : 0;
    }
    for (int v = 0; v < run.pmesh->GetNV(); ++v) {
        if (input->GetVertex(v)[2] == 0.25) {
            moved_off_plate += (run.pmesh->GetVertex(v)[2] != 0.25) ? 1 : 0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &wrong_grain, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &moved_off_plate, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(wrong_grain, 0);
    EXPECT_EQ(moved_off_plate, 0);
    EXPECT_NEAR(GrainVolume(run, plate), 0.25, 1e-14);
}

/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    mfem::Hypre::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
