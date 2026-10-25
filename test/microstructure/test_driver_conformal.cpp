/**
 * @file test_driver_conformal.cpp
 * @brief Driver-level tests of the Stage 2 conformal grain boundary step ([Conformal] enabled)
 * in GenerateMicrostructure (exaconstit_microstructure).
 *
 * What is checked, and why:
 *  - Hex box without Nodes, 8 Voronoi grains: the step runs and reports, the mesh keeps no Nodes
 *    (the fitted positions become its vertices), the quality floor holds with no inverted
 *    element, the grain boundary nodes end closer to the analytic boundaries, the box keeps its
 *    volume (its faces stay in place), the grain count is the same as without the step, and only
 *    a few elements change grain (label repair, the count the report gives).
 *  - Tet box: the tet floor holds.
 *  - A mesh with order-2 Nodes is refused on every rank with std::invalid_argument.
 *  - Without the step nothing moves (the vertices of the input mesh are kept bit for bit).
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows.
 */
#include "microstructure/conformal/conformal_fitter.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_options.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

ms::MicrostructureOptions Options(bool conformal) {
    ms::MicrostructureOptions o;
    o.seeds.num_seeds = 8;
    o.seeds.rng_seed = 3;
    o.conformal.enabled = conformal;
    return o;
}

struct DriverRun {
    std::unique_ptr<mfem::ParMesh> pmesh;
    ms::MicrostructureResult result;
    ms::GenerationReport report;
};

DriverRun Generate(mfem::Mesh& serial, bool conformal) {
    DriverRun run;
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    run.pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    run.result = ms::GenerateMicrostructure(Options(conformal), *run.pmesh, MPI_COMM_WORLD,
                                            &run.report);
    return run;
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

} // namespace

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
    if (mfem::Mpi::Root()) {
        std::printf("hex box: %d grains, %d flips, deviation %.3f -> %.3f h, sj %.3f\n",
                    fitted.result.num_grains, changed, r.conformal_fit.initial_deviation,
                    r.conformal_fit.mean_deviation,
                    r.conformal_fit.quality_after.min_scaled_jacobian);
    }
}

TEST(DriverConformal, TetBox) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    DriverRun fitted = Generate(serial, true);
    const ms::GenerationReport& r = fitted.report;
    ASSERT_TRUE(r.ran_conformal);
    EXPECT_EQ(r.conformal_fit.quality_after.inverted, 0);
    EXPECT_GE(r.conformal_fit.quality_after.min_scaled_jacobian, ms::MESH_QUALITY_FLOOR_TET);
    EXPECT_LE(r.conformal_fit.mean_deviation, r.conformal_fit.initial_deviation);
    EXPECT_NEAR(Volume(*fitted.pmesh), 1.0, 1e-12);
}

TEST(DriverConformal, RefusesHigherOrder) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    std::unique_ptr<mfem::ParMesh> pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
    pmesh->SetCurvature(2);
    EXPECT_THROW(ms::GenerateMicrostructure(Options(true), *pmesh, MPI_COMM_WORLD),
                 std::invalid_argument);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    mfem::Hypre::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
