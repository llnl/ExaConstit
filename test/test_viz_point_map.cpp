/**
 * @file test_viz_point_map.cpp
 * @brief Checks the table that places quadrature point data on visualization nodes
 *
 * VizPointMap (src/postprocessing/viz_point_map.hpp) decides, for every node of the L2 field
 * written to the visualization files, which quadrature point value it receives:
 * - zone output: the single element-average point of each element goes to the single node of
 *   that element (L2 order 0);
 * - quadrature output: each Gauss-Lobatto node of an L2 element of the mesh order takes the
 *   value of the closest quadrature point of the element.
 *
 * The checks, on a curved (high-order) mesh where only part of the elements belong to the
 * "region" (like one material of a multi-material run):
 * - every output node of every region element is written exactly once, and nodes of elements
 *   outside the region are never written;
 * - each source index points at a quadrature point of the same element;
 * - hexahedra in quadrature mode: with the default 2p+1 integration rule the match is one to
 *   one, and node (i, j, k) of the Gauss-Lobatto grid gets Gauss point (i, j, k), i.e. in each
 *   direction the closest Gauss point to the node coordinate;
 * - zone mode: one entry per element, source = that element's average.
 *
 * Part of the extensive suite (EXACONSTIT_ENABLE_EXTENSIVE_TESTS).
 */
#include "postprocessing/viz_point_map.hpp"

#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace {

/// Small cube mesh of the given order, bent so the elements are curved
std::shared_ptr<mfem::ParMesh> MakeCurvedMesh(mfem::Element::Type type, int order) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(3, 3, 3, type, 1.0, 1.0, 1.0);
    serial.SetCurvature(order);
    serial.Transform([](const mfem::Vector& x, mfem::Vector& y) {
        y = x;
        y(0) += 0.05 * std::sin(M_PI * x(1)) * std::sin(M_PI * x(2));
        y(1) += 0.05 * std::sin(M_PI * x(0));
    });
    return std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, serial);
}

/// Every other element is in the region (mimics one material of several)
mfem::Array<bool> RegionElements(const mfem::ParMesh& mesh) {
    mfem::Array<bool> in_region(mesh.GetNE());
    for (int e = 0; e < mesh.GetNE(); e++) {
        in_region[e] = (e % 2 == 0);
    }
    return in_region;
}

/// Checks shared by both modes: coverage of output nodes and sources inside their element
void CheckCoverage(const VizPointMap& map,
                   const mfem::expt::PartialQuadratureSpace& qspace,
                   const mfem::ParFiniteElementSpace& fes) {
    const auto& l2g = qspace.GetLocal2Global();
    const auto& offsets = qspace.Offsets(mfem::QSpaceOffsetStorage::COMPRESSED);
    std::vector<int> written(fes.GetNDofs(), 0);
    mfem::Array<int> dofs;

    const mfem::Array<int>& h_src = map.GetSource();
    const mfem::Array<int>& h_dst = map.GetDest();
    const mfem::Array<int>& h_eoff = map.GetElementOffsets();
    h_src.HostRead();
    h_dst.HostRead();
    h_eoff.HostRead();

    ASSERT_EQ(map.GetNumElements(), qspace.GetNE());
    for (int ie = 0; ie < qspace.GetNE(); ie++) {
        fes.GetElementDofs(l2g[ie], dofs);
        ASSERT_EQ(h_eoff[ie + 1] - h_eoff[ie], dofs.Size());
        for (int k = h_eoff[ie]; k < h_eoff[ie + 1]; k++) {
            EXPECT_GE(h_src[k], offsets[ie]);
            EXPECT_LT(h_src[k], offsets[ie + 1]);
            EXPECT_GE(dofs.Find(h_dst[k]), 0) << "node outside its element";
            written[h_dst[k]] += 1;
        }
    }
    for (int e = 0; e < fes.GetNE(); e++) {
        fes.GetElementDofs(e, dofs);
        const int expected = (qspace.GetGlobal2Local().Size() == 1 ||
                              qspace.GetGlobal2Local()[e] >= 0)
                                 ? 1
                                 : 0;
        for (int d : dofs) {
            EXPECT_EQ(written[d], expected) << "element " << e << " node " << d;
        }
    }
}

class VizPointMapTest : public ::testing::TestWithParam<std::tuple<mfem::Element::Type, int>> {};

TEST_P(VizPointMapTest, Quadrature) {
    const auto [type, order] = GetParam();
    auto mesh = MakeCurvedMesh(type, order);
    auto in_region = RegionElements(*mesh);
    mfem::expt::PartialQuadratureSpace qspace(mesh, 2 * order + 1, in_region);

    mfem::L2_FECollection fec(order, 3, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace fes(mesh.get(), &fec);

    VizPointMap map(VizFieldLocation::QUADRATURE, qspace, fes, qspace.GetLocal2Global());
    CheckCoverage(map, qspace, fes);

    if (type != mfem::Element::HEXAHEDRON) {
        return;
    }

    // Hexahedra: closest Gauss point to a Gauss-Lobatto node, direction by direction
    const int n1d = order + 1;
    ASSERT_EQ(qspace.GetIntRule(0).GetNPoints(), n1d * n1d * n1d);
    const auto& offsets = qspace.Offsets(mfem::QSpaceOffsetStorage::COMPRESSED);
    const mfem::Array<int>& h_src = map.GetSource();
    const mfem::Array<int>& h_eoff = map.GetElementOffsets();
    h_src.HostRead();
    h_eoff.HostRead();

    // 1D Gauss-Legendre points of the p + 1 point rule (the hexahedron rule is their tensor
    // product)
    const mfem::IntegrationRule& gauss_1d = mfem::IntRules.Get(mfem::Geometry::SEGMENT,
                                                               2 * order + 1);
    ASSERT_EQ(gauss_1d.GetNPoints(), n1d);
    auto nearest_1d = [&](double xn) {
        int best = 0;
        for (int i = 1; i < n1d; i++) {
            if (std::abs(gauss_1d.IntPoint(i).x - xn) < std::abs(gauss_1d.IntPoint(best).x - xn)) {
                best = i;
            }
        }
        return gauss_1d.IntPoint(best).x;
    };

    std::vector<int> used;
    for (int ie = 0; ie < qspace.GetNE(); ie++) {
        const auto& ir = qspace.GetIntRule(ie);
        const auto& nodes = fes.GetFE(qspace.GetLocal2Global()[ie])->GetNodes();
        used.assign(ir.GetNPoints(), 0);
        for (int jn = 0; jn < nodes.GetNPoints(); jn++) {
            const int iq = h_src[h_eoff[ie] + jn] - offsets[ie];
            used[iq] += 1;
            const auto& node = nodes.IntPoint(jn);
            const auto& qp = ir.IntPoint(iq);
            EXPECT_NEAR(qp.x, nearest_1d(node.x), 1e-14);
            EXPECT_NEAR(qp.y, nearest_1d(node.y), 1e-14);
            EXPECT_NEAR(qp.z, nearest_1d(node.z), 1e-14);
        }
        for (int c : used) {
            EXPECT_EQ(c, 1) << "hexahedron match is not one to one";
        }
    }
}

TEST_P(VizPointMapTest, Zone) {
    const auto [type, order] = GetParam();
    auto mesh = MakeCurvedMesh(type, order);
    auto in_region = RegionElements(*mesh);
    // Element-average space, as used by the "_avg" quadrature functions
    mfem::expt::PartialQuadratureSpace qspace(mesh, 1, in_region);

    mfem::L2_FECollection fec(0, 3);
    mfem::ParFiniteElementSpace fes(mesh.get(), &fec);

    VizPointMap map(VizFieldLocation::ZONE, qspace, fes, qspace.GetLocal2Global());
    CheckCoverage(map, qspace, fes);
    ASSERT_EQ(map.GetNumPoints(), qspace.GetNE());
    EXPECT_EQ(map.SourceName("state_var"), "state_var_avg");

    const mfem::Array<int>& h_src = map.GetSource();
    h_src.HostRead();
    for (int ie = 0; ie < qspace.GetNE(); ie++) {
        EXPECT_EQ(h_src[ie], ie);
    }
}

INSTANTIATE_TEST_SUITE_P(HexAndTet,
                         VizPointMapTest,
                         ::testing::Combine(::testing::Values(mfem::Element::HEXAHEDRON,
                                                              mfem::Element::TETRAHEDRON),
                                            ::testing::Values(1, 2, 3)));

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
