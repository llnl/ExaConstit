/*
 * test_straight_sided: recognition of straight-sided meshes (mesh/straight_sided.hpp) and its
 * effect on seeding and labeling.
 *
 * ExaConstit gives every mesh high-order "Nodes" (a coordinate field, Mesh::SetCurvature), also
 * when the elements are straight. The generator must treat such a mesh exactly like the same mesh
 * without Nodes (same seeds, same labels), while genuinely curved meshes keep the curved
 * treatment (padded element boxes, Newton inverse map, labeling through the Nodes).
 *
 * Runs at 1-4 MPI ranks; distributed results are compared with single-rank references.
 */
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/seeding/material_point_locator.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

/** Unit cube [0,1]^3 of n^3 hexes or 6 n^3 tets (MFEM's Cartesian generator). */
mfem::Mesh UnitCube(int n, mfem::Element::Type type) {
    return mfem::Mesh::MakeCartesian3D(n, n, n, type, 1.0, 1.0, 1.0, false);
}

/** Bulges the top face of the unit cube: z -> z (1 + 0.2 sin(pi x) sin(pi y)). */
void BulgeTop(const mfem::Vector& x, mfem::Vector& y) {
    const double pi = 3.141592653589793;
    y = x;
    y(2) = x(2) * (1.0 + 0.2 * std::sin(pi * x(0)) * std::sin(pi * x(1)));
}

/** Distributes `serial` over MPI_COMM_WORLD with the generator's own partitioner (collective). */
std::unique_ptr<mfem::ParMesh> Distribute(mfem::Mesh& serial) {
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    return ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
}

/** Bitwise equality of seed positions and source ids. */
void ExpectSameSeeds(const ms::SeedSet& a, const ms::SeedSet& b) {
    ASSERT_EQ(a.Size(), b.Size());
    for (std::size_t i = 0; i < a.Size(); i++) {
        EXPECT_EQ(a.x[i], b.x[i]) << "seed " << i;
        EXPECT_EQ(a.y[i], b.y[i]) << "seed " << i;
        EXPECT_EQ(a.z[i], b.z[i]) << "seed " << i;
        EXPECT_EQ(a.source_id[i], b.source_id[i]) << "seed " << i;
    }
}

const mfem::Element::Type kTypes[] = {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON};

} // namespace

// The classification rules: no Nodes, order-1 Nodes and higher-order Nodes projected from the
// straight cube are straight-sided (the projection reproduces the multilinear map, so every
// nodal point lies on it to round-off); the bulged order-2 cube is not (its top nodes are up to
// 0.2 above the corner map); discontinuous (L2) Nodes, MFEM's periodic representation, are never
// straight-sided. Checked alone on every rank and collectively on the distributed mesh.
TEST(StraightSided, Classification) {
    for (const auto type : kTypes) {
        for (int order = 0; order <= 3; ++order) {
            mfem::Mesh serial = UnitCube(3, type);
            if (order > 0) {
                serial.SetCurvature(order);
            }
            EXPECT_TRUE(ms::StraightSidedLocal(serial)) << "order " << order;
            auto pmesh = Distribute(serial);
            EXPECT_TRUE(ms::IsStraightSided(*pmesh, MPI_COMM_WORLD)) << "order " << order;
        }
        mfem::Mesh bulged = UnitCube(3, type);
        bulged.SetCurvature(2);
        bulged.Transform(BulgeTop);
        EXPECT_FALSE(ms::StraightSidedLocal(bulged));
        auto pbulged = Distribute(bulged);
        EXPECT_FALSE(ms::IsStraightSided(*pbulged, MPI_COMM_WORLD));

        mfem::Mesh discontinuous = UnitCube(3, type);
        discontinuous.SetCurvature(2, true);
        EXPECT_FALSE(ms::StraightSidedLocal(discontinuous));
    }
}

// Corner points of a straight-sided mesh come from the Nodes vertex dofs, which for the
// projected cube equal the vertex coordinates exactly; for a mesh without Nodes they are the
// vertices themselves. The corner transformation maps the reference corners to them.
TEST(StraightSided, CornerPointsFromNodes) {
    for (const auto type : kTypes) {
        mfem::Mesh plain = UnitCube(2, type);
        mfem::Mesh curved_space = UnitCube(2, type);
        curved_space.SetCurvature(2);
        for (int e = 0; e < plain.GetNE(); ++e) {
            mfem::DenseMatrix a;
            mfem::DenseMatrix b;
            ms::ElementCornerPoints(plain, e, a);
            ms::ElementCornerPoints(curved_space, e, b);
            ASSERT_EQ(a.Width(), b.Width());
            for (int k = 0; k < a.Width(); ++k) {
                for (int d = 0; d < 3; ++d) {
                    EXPECT_EQ(a(d, k), b(d, k)) << "element " << e << " corner " << k;
                }
            }
        }
    }
}

// Collective agreement: on an order-2 straight cube, rank 0 moves the element-interior node of
// its first element (a dof no other element or rank shares) by 1% of the element size, so only
// rank 0 holds a curved element. StraightSidedLocal then differs between ranks (when there are
// several), but IsStraightSided must answer "curved" on every rank.
TEST(StraightSided, CollectiveAgreement) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mfem::Mesh serial = UnitCube(3, mfem::Element::HEXAHEDRON);
    serial.SetCurvature(2);
    auto pmesh = Distribute(serial);
    ASSERT_TRUE(ms::IsStraightSided(*pmesh, MPI_COMM_WORLD));
    if (rank == 0 && pmesh->GetNE() > 0) {
        mfem::GridFunction* nodes = pmesh->GetNodes();
        mfem::Array<int> vdofs;
        nodes->FESpace()->GetElementVDofs(0, vdofs);
        const int num_dofs = vdofs.Size() / 3;
        // The last dof of an order-2 H1 hex is its interior (cell) node
        (*nodes)(vdofs[num_dofs - 1]) += 0.01 / 3.0;
    }
    const bool local = ms::StraightSidedLocal(*pmesh);
    if (rank == 0 && pmesh->GetNE() > 0) {
        EXPECT_FALSE(local);
    } else {
        EXPECT_TRUE(local);
    }
    EXPECT_FALSE(ms::IsStraightSided(*pmesh, MPI_COMM_WORLD));
}

// Seeds: order-1 and order-2 Nodes on the straight cube give bit-identical seeds to the cube
// without Nodes (same corner boxes, same sampling box, same inside test), serially and
// distributed, for hexes and tets, including the surface clearance (straight faces are
// triangulated from their corners). The curved locator is kept for the bulged mesh.
TEST(StraightSided, SameSeedsAsWithoutNodes) {
    for (const auto type : kTypes) {
        ms::SeedingParameters params;
        params.num_seeds = 40;
        params.rng_seed = 5;
        params.min_distance_factor = 0.3;
        params.boundary_clearance = 0.02;
        mfem::Mesh plain = UnitCube(3, type);
        const ms::SeedSet reference = ms::GenerateSeeds(plain, MPI_COMM_SELF, params);
        for (int order = 1; order <= 2; ++order) {
            mfem::Mesh serial = UnitCube(3, type);
            serial.SetCurvature(order);
            ExpectSameSeeds(ms::GenerateSeeds(serial, MPI_COMM_SELF, params), reference);
            auto pmesh = Distribute(serial);
            ExpectSameSeeds(ms::GenerateSeeds(*pmesh, params), reference);
        }
        mfem::Mesh bulged = UnitCube(3, type);
        bulged.SetCurvature(2);
        bulged.Transform(BulgeTop);
        const ms::MaterialPointLocator locator(bulged, ms::IsStraightSided(bulged, MPI_COMM_SELF));
        EXPECT_FALSE(locator.StraightSided());
    }
}

// Labels: the whole pipeline on the distributed cube with no Nodes, order-1 and order-2 Nodes.
// The pipeline works in place on each mesh (no copy), so the seeds must be bit-identical (the
// seeding uses the exact straight-sided geometry), while element centroids are evaluated through
// the Nodes and may differ from the no-Nodes values in the last bits. A centroid exactly
// equidistant from two seeds could then change grain, so the labels are required to agree on at
// least 99% of the elements (counted over all ranks); in practice all agree, and the measured
// fraction is printed. The caller's mesh keeps its Nodes, and the bulged mesh is still labeled
// as curved (its seed sampling box reaches into the bulge).
TEST(StraightSided, SameSeedsAndLabelsAsWithoutNodes) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ms::MicrostructureOptions options;
    options.seeds.num_seeds = 12;
    options.seeds.rng_seed = 3;
    for (const auto type : kTypes) {
        mfem::Mesh plain = UnitCube(4, type);
        auto pplain = Distribute(plain);
        ms::GenerationReport plain_report;
        const ms::MicrostructureResult reference =
            ms::GenerateMicrostructure(options, *pplain, MPI_COMM_WORLD, &plain_report);
        for (int order = 1; order <= 2; ++order) {
            mfem::Mesh serial = UnitCube(4, type);
            serial.SetCurvature(order);
            auto pmesh = Distribute(serial);
            ms::GenerationReport report;
            const ms::MicrostructureResult result =
                ms::GenerateMicrostructure(options, *pmesh, MPI_COMM_WORLD, &report);
            ExpectSameSeeds(report.seeds, plain_report.seeds);
            ASSERT_EQ(result.element_grain.Size(), reference.element_grain.Size());
            long long counts[2] = {0, result.element_grain.Size()};
            for (int e = 0; e < result.element_grain.Size(); ++e) {
                counts[0] += (result.element_grain[e] == reference.element_grain[e]) ? 1 : 0;
            }
            MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            const double fraction = double(counts[0]) / double(counts[1]);
            if (rank == 0) {
                std::cout << "order " << order << ": " << counts[0] << " of " << counts[1]
                          << " elements keep their grain (" << fraction << ")" << std::endl;
            }
            EXPECT_GE(fraction, 0.99) << "order " << order;
            EXPECT_NE(pmesh->GetNodes(), nullptr);
        }
        mfem::Mesh bulged = UnitCube(4, type);
        bulged.SetCurvature(2);
        bulged.Transform(BulgeTop);
        auto pbulged = Distribute(bulged);
        ms::GenerationReport curved_report;
        ms::GenerateMicrostructure(options, *pbulged, MPI_COMM_WORLD, &curved_report);
        EXPECT_GE(curved_report.seeding.sampling_hi[2], 1.2);
    }
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
