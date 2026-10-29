/*
 * test_curved_pipeline: the whole generator (GenerateMicrostructure) on genuinely curved meshes
 * (the curved end-to-end matrix).
 *
 * Meshes: a 4 x 4 x 4 box of hexes or tets with order-2 or order-3 Nodes in the Gauss-Lobatto or
 * the closed-uniform H1 basis, moved off the straight map by a smooth warp (so the elements are
 * really curved: StraightSidedLocal is false). The bottom layer carries attribute 2 (a plate for
 * the passthrough case; an ordinary part of the material otherwise).
 *
 * Pipelines (one MicrostructureOptions each): Euclidean centroid labels with the default cleanup
 * and grain statistics; quadrature vote; connected distance; geodesic distance with the FIM and
 * the graph solver; Laguerre with log-normal sizes, Lloyd relaxation and the volume fit;
 * Johnson-Mehl nucleation (Additive metric); and the plate as a passthrough attribute.
 *
 * Every pipeline runs on every mesh over MPI_COMM_WORLD and alone on MPI_COMM_SELF, and the two
 * must agree exactly: element labels (serial order), grain count, orientations, source ids,
 * regions and the fixed-point grain volumes. Running the executable at np = 1..4 therefore
 * compares np = 2, 3, 4 against np = 1. A curved mesh distributed from a serial mesh has bitwise
 * equal copies of every shared node, so exact agreement is the right expectation (for meshes
 * refined in parallel see test_curved_determinism). Sanity checks on the single-rank run: more
 * than one grain, compact ids, every element labeled, and for the passthrough case the plate
 * elements carry the last grain id.
 */
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include "microstructure/labeling/grain_statistics.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_options.hpp"
#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

constexpr int PLATE_ATTRIBUTE = 2;

/** Smooth warp moving every node (also on the boundary): genuinely curved elements. */
void Warp(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    y(0) += 0.04 * std::sin(3.0 * x(1)) * std::cos(2.0 * x(2));
    y(1) += 0.03 * std::sin(2.0 * x(0) + x(2));
    y(2) += 0.03 * std::cos(3.0 * x(0) * x(1));
}

/**
 * The curved box: Nodes of `order` in the H1 basis `basis` (BasisType), set with
 * Mesh::SetNodalFESpace (which projects the straight geometry onto the new space) and then
 * warped. The Nodes grid function owns its space and collection (GridFunction::MakeOwner).
 */
mfem::Mesh CurvedBox(mfem::Element::Type type, int order, int basis) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, type, 1.0, 1.0, 1.0, false);
    for (int e = 0; e < mesh.GetNE(); e++) {
        mfem::Vector center(3);
        mesh.GetElementTransformation(e)->Transform(
            mfem::Geometries.GetCenter(mesh.GetElementBaseGeometry(e)), center);
        mesh.SetAttribute(e, center(2) < 0.25 ? PLATE_ATTRIBUTE : 1);
    }
    mesh.SetAttributes();
    auto* fec = new mfem::H1_FECollection(order, 3, basis);
    auto* fes = new mfem::FiniteElementSpace(&mesh, fec, 3);
    mesh.SetNodalFESpace(fes);
    mesh.GetNodes()->MakeOwner(fec);
    mesh.Transform(Warp);
    return mesh;
}

/** Per-element values gathered into serial element order and broadcast to every rank. */
std::vector<int>
GatherToAll(MPI_Comm comm, const ms::SerialOrder& order, const mfem::Array<int>& local) {
    local.HostRead();
    std::vector<int> all = ms::GatherToSerial(comm, order, local, 0);
    int size = static_cast<int>(all.size());
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);
    all.resize(static_cast<std::size_t>(size));
    MPI_Bcast(all.data(), size, MPI_INT, 0, comm);
    return all;
}

/** One driver run with its labels in serial order and the grain volumes. */
struct DriverRun {
    std::vector<int> labels;
    std::vector<int> attributes; // serial order, of the input mesh
    ms::MicrostructureResult result;
    std::vector<double> grain_volume;
};

/** GenerateMicrostructure on `serial` distributed over `comm` (collective), plus statistics. */
DriverRun RunDriver(mfem::Mesh& serial, const ms::MicrostructureOptions& options, MPI_Comm comm) {
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, comm);
    std::unique_ptr<mfem::ParMesh> pmesh = ms::MakeParMesh(comm, serial, order);
    mfem::Array<int> attributes(pmesh->GetNE());
    for (int e = 0; e < pmesh->GetNE(); e++) {
        attributes[e] = pmesh->GetAttribute(e);
    }
    DriverRun run;
    ms::GenerationReport report;
    run.result = ms::GenerateMicrostructure(options, *pmesh, comm, &report);
    run.labels = GatherToAll(comm, order, run.result.element_grain);
    run.attributes = GatherToAll(comm, order, attributes);
    const ms::GrainStatistics stats = ms::ComputeGrainStatistics(
        *report.adjacency, run.result.element_grain, run.result.num_grains);
    run.grain_volume = stats.volume;
    return run;
}

/** A named pipeline. */
struct Pipeline {
    std::string name;
    ms::MicrostructureOptions options;
};

std::vector<Pipeline> Pipelines() {
    std::vector<Pipeline> out;
    ms::MicrostructureOptions base;
    base.seeds.num_seeds = 10;
    base.seeds.rng_seed = 2026;

    out.push_back({"euclidean", base});

    ms::MicrostructureOptions vote = base;
    vote.tessellation.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    vote.tessellation.vote_levels = 1;
    out.push_back({"vote", vote});

    ms::MicrostructureOptions connected = base;
    connected.tessellation.distance = ms::DistanceMode::CONNECTED;
    out.push_back({"connected", connected});

    ms::MicrostructureOptions fim = base;
    fim.tessellation.distance = ms::DistanceMode::GEODESIC;
    fim.tessellation.geodesic_solver = "fim";
    out.push_back({"geodesic fim", fim});

    ms::MicrostructureOptions graph = fim;
    graph.tessellation.geodesic_solver = "graph";
    out.push_back({"geodesic graph", graph});

    ms::MicrostructureOptions fit = base;
    fit.tessellation.metric = ms::MetricType::LAGUERRE;
    fit.seeds.size.distribution = "lognormal";
    fit.seeds.size.sigma = 0.3;
    fit.tessellation.lloyd_iterations = 2;
    fit.tessellation.fit_volumes.enabled = true;
    fit.tessellation.fit_volumes.tolerance = 0.1;
    fit.tessellation.fit_volumes.max_iterations = 30;
    out.push_back({"lloyd + fit", fit});

    ms::MicrostructureOptions jm = base;
    jm.seeds.num_seeds = 0;
    jm.seeds.placement = ms::SeedPlacement::JOHNSON_MEHL;
    jm.tessellation.metric = ms::MetricType::ADDITIVE;
    jm.seeds.johnson_mehl.rate = 40.0;
    jm.seeds.johnson_mehl.growth_speed = 1.0;
    jm.seeds.johnson_mehl.num_nuclei = 12;
    out.push_back({"johnson-mehl", jm});

    ms::MicrostructureOptions pass = base;
    pass.mesh.active_attributes = {1};
    pass.mesh.passthrough_attributes = {{PLATE_ATTRIBUTE, 2}};
    out.push_back({"passthrough", pass});
    return out;
}

struct MeshVariant {
    mfem::Element::Type type;
    int order;
    int basis;
    std::string name;
};

const MeshVariant kMeshes[] = {
    {mfem::Element::HEXAHEDRON, 2, mfem::BasisType::GaussLobatto, "hex p2 GL"},
    {mfem::Element::HEXAHEDRON, 3, mfem::BasisType::GaussLobatto, "hex p3 GL"},
    {mfem::Element::HEXAHEDRON, 2, mfem::BasisType::ClosedUniform, "hex p2 uniform"},
    {mfem::Element::HEXAHEDRON, 3, mfem::BasisType::ClosedUniform, "hex p3 uniform"},
    {mfem::Element::TETRAHEDRON, 2, mfem::BasisType::GaussLobatto, "tet p2 GL"},
    {mfem::Element::TETRAHEDRON, 3, mfem::BasisType::GaussLobatto, "tet p3 GL"},
    {mfem::Element::TETRAHEDRON, 2, mfem::BasisType::ClosedUniform, "tet p2 uniform"},
    {mfem::Element::TETRAHEDRON, 3, mfem::BasisType::ClosedUniform, "tet p3 uniform"}};

/** The matrix for the meshes of one element type (split per type to keep each test short). */
void RunMatrix(mfem::Element::Type type) {
    for (const MeshVariant& m : kMeshes) {
        if (m.type != type) {
            continue;
        }
        mfem::Mesh serial = CurvedBox(m.type, m.order, m.basis);
        EXPECT_FALSE(ms::StraightSidedLocal(serial)) << m.name;
        for (const Pipeline& p : Pipelines()) {
            const std::string what = m.name + ", " + p.name;
            const DriverRun world = RunDriver(serial, p.options, MPI_COMM_WORLD);
            const DriverRun alone = RunDriver(serial, p.options, MPI_COMM_SELF);
            EXPECT_EQ(world.labels, alone.labels) << what;
            EXPECT_EQ(world.result.num_grains, alone.result.num_grains) << what;
            EXPECT_EQ(world.result.grain_quat, alone.result.grain_quat) << what;
            EXPECT_EQ(world.result.grain_source_id, alone.result.grain_source_id) << what;
            EXPECT_EQ(world.result.grain_region, alone.result.grain_region) << what;
            EXPECT_EQ(world.grain_volume, alone.grain_volume) << what;

            // Sanity of the single-rank result
            const int n = alone.result.num_grains;
            EXPECT_GT(n, 1) << what;
            std::vector<int> count(static_cast<std::size_t>(n) + 1, 0);
            for (const int g : alone.labels) {
                EXPECT_GE(g, 1) << what;
                EXPECT_LE(g, n) << what;
                if (g >= 1 && g <= n) {
                    count[static_cast<std::size_t>(g)]++;
                }
            }
            for (int g = 1; g <= n; g++) {
                EXPECT_GT(count[static_cast<std::size_t>(g)], 0) << what << ", grain " << g;
            }
            if (p.name == "passthrough") {
                for (std::size_t e = 0; e < alone.labels.size(); e++) {
                    if (alone.attributes[e] == PLATE_ATTRIBUTE) {
                        EXPECT_EQ(alone.labels[e], n) << what;
                    } else {
                        EXPECT_LT(alone.labels[e], n) << what;
                    }
                }
            }
        }
    }
}

} // namespace

// Hexahedra: order 2 and 3, Gauss-Lobatto and closed-uniform Nodes, every pipeline.
TEST(CurvedPipeline, HexMatrix) {
    RunMatrix(mfem::Element::HEXAHEDRON);
}

// Tetrahedra: order 2 and 3, Gauss-Lobatto and closed-uniform Nodes, every pipeline.
TEST(CurvedPipeline, TetMatrix) {
    RunMatrix(mfem::Element::TETRAHEDRON);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
