/*
 * test_curved_geometry: the generator's geometric building blocks on curved and warped elements
 *.
 *
 * Every curved mesh here is genuinely curved: the nodes are moved off the straight map after
 * Mesh::SetCurvature (a "bulge" of the top face), so a code path that silently used the element
 * corners instead of the Nodes would give visibly wrong answers. References are computed
 * independently of the code under test: by dense sampling of the face maps (brute force), from
 * the analytic bulge formula, or by inverting the simple column structure of the bulged mesh.
 *
 * Runs at 1-4 MPI ranks. Distributed results are combined over the ranks exactly as the seeding
 * code does (bitwise OR of the per-rank masks) and compared with the serial references.
 */
#include "microstructure/io/output_writer.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/preflight.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/seeding/boundary_clearance.hpp"
#include "microstructure/seeding/material_point_locator.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;
namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.141592653589793; ///< pi.

/** This process's rank in MPI_COMM_WORLD. */
int Rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/** The number of ranks of MPI_COMM_WORLD. */
int NumRanks() {
    int n = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

/** Unit cube [0,1]^3 of n^3 hexes or 6 n^3 tets (MFEM's Cartesian generator). */
mfem::Mesh UnitCube(int n, mfem::Element::Type type) {
    return mfem::Mesh::MakeCartesian3D(n, n, n, type, 1.0, 1.0, 1.0, false);
}

/** Bulge amplitude used by BulgeTop (set before calling Mesh::Transform). */
double g_bulge_amplitude = 0.2;

/** Bulges the top face of the unit cube: z -> z (1 + a sin(pi x) sin(pi y)); x, y unchanged. */
void BulgeTop(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    y(2) = x(2) * (1.0 + g_bulge_amplitude * std::sin(kPi * x(0)) * std::sin(kPi * x(1)));
}

/** A genuinely curved unit cube: order-`order` Nodes, then the top bulged with amplitude a. */
mfem::Mesh BulgedCube(int n, mfem::Element::Type type, int order, double amplitude) {
    mfem::Mesh mesh = UnitCube(n, type);
    mesh.SetCurvature(order);
    g_bulge_amplitude = amplitude;
    mesh.Transform(BulgeTop);
    return mesh;
}

/** Distributes `serial` over MPI_COMM_WORLD with the generator's own partitioner (collective). */
std::unique_ptr<mfem::ParMesh> Distribute(mfem::Mesh& serial) {
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    return ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
}

/** Output directory shared by all ranks of this run (created by rank 0, then a barrier). */
fs::path SharedDir(const std::string& test) {
    const fs::path dir = fs::path("test_curved_geometry_output") /
                         ("np" + std::to_string(NumRanks())) / test;
    if (Rank() == 0) {
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return dir;
}

/** Bit i of a point mask (word i / 64, bit i % 64), the layout of MarkInside / MarkNear. */
bool Bit(const std::vector<std::uint64_t>& mask, std::size_t i) {
    return ((mask[i / 64] >> (i % 64)) & 1u) != 0;
}

/**
 * Brute-force distance from p to the outer surface of a serial mesh: the minimum over all
 * boundary faces (faces with one element) of the distance to the face map evaluated on a dense
 * reference lattice (`dense` subdivisions per edge). For a point at distance d from a smooth
 * surface, sampling with spacing s overestimates the distance by about (s/2)^2 / (2 d), far below
 * the tolerances used here.
 */
double BruteForceSurfaceDistance(mfem::Mesh& mesh, const std::array<double, 3>& p, int dense) {
    double best_sq = std::numeric_limits<double>::infinity();
    mfem::DenseMatrix x;
    for (int f = 0; f < mesh.GetNumFaces(); f++) {
        int e1 = -1;
        int e2 = -1;
        mesh.GetFaceElements(f, &e1, &e2);
        if (e2 >= 0) {
            continue;
        }
        mfem::ElementTransformation* trans = mesh.GetFaceTransformation(f);
        mfem::RefinedGeometry* refined = mfem::GlobGeometryRefiner.Refine(mesh.GetFaceGeometry(f),
                                                                          dense);
        trans->Transform(refined->RefPts, x);
        for (int q = 0; q < x.Width(); q++) {
            const double dx = x(0, q) - p[0];
            const double dy = x(1, q) - p[1];
            const double dz = x(2, q) - p[2];
            best_sq = std::min(best_sq, dx * dx + dy * dy + dz * dz);
        }
    }
    return std::sqrt(best_sq);
}

/**
 * Height of the top surface of a bulged unit cube at (x, y), from the mesh itself: the bulge only
 * moves nodes in z, so the x and y parts of every top face map are the straight (affine) map of
 * the face. The face is located by solving that affine map from three reference points, then the
 * face map gives z. Top faces are the boundary faces whose corners all lie at z >= top_min_z
 * (1 + min(0, a) for bulge amplitude a, which separates the top corners from the corners of the
 * layer below on the 2x2x2 cube used here). Returns -infinity if (x, y) is on no top face.
 */
double TopSurfaceHeight(mfem::Mesh& mesh, double x, double y, double top_min_z) {
    double height = -std::numeric_limits<double>::infinity();
    mfem::Vector a(3);
    mfem::Vector b(3);
    mfem::Vector c(3);
    mfem::Vector out(3);
    for (int f = 0; f < mesh.GetNumFaces(); f++) {
        int e1 = -1;
        int e2 = -1;
        mesh.GetFaceElements(f, &e1, &e2);
        if (e2 >= 0) {
            continue;
        }
        mfem::ElementTransformation* trans = mesh.GetFaceTransformation(f);
        const mfem::Geometry::Type geom = mesh.GetFaceGeometry(f);
        mfem::IntegrationPoint ip;
        ip.Set2(0.0, 0.0);
        trans->Transform(ip, a);
        ip.Set2(1.0, 0.0);
        trans->Transform(ip, b);
        ip.Set2(0.0, 1.0);
        trans->Transform(ip, c);
        // Top face test: all corners at z >= top_min_z (the fourth corner of a square too)
        const double zmin = top_min_z - 1e-12;
        bool top = a(2) >= zmin && b(2) >= zmin && c(2) >= zmin;
        if (geom == mfem::Geometry::SQUARE) {
            ip.Set2(1.0, 1.0);
            trans->Transform(ip, out);
            top = top && out(2) >= zmin;
        }
        if (!top) {
            continue;
        }
        // Affine xy map: (x, y) = a + s (b - a) + t (c - a); solve the 2x2 system for (s, t)
        const double m00 = b(0) - a(0);
        const double m01 = c(0) - a(0);
        const double m10 = b(1) - a(1);
        const double m11 = c(1) - a(1);
        const double det = m00 * m11 - m01 * m10;
        const double rx = x - a(0);
        const double ry = y - a(1);
        const double s = (rx * m11 - m01 * ry) / det;
        const double t = (m00 * ry - rx * m10) / det;
        const double tol = 1e-12;
        const bool inside = (geom == mfem::Geometry::TRIANGLE)
                                ? (s >= -tol && t >= -tol && s + t <= 1.0 + tol)
                                : (s >= -tol && t >= -tol && s <= 1.0 + tol && t <= 1.0 + tol);
        if (!inside) {
            continue;
        }
        ip.Set2(std::clamp(s, 0.0, 1.0), std::clamp(t, 0.0, 1.0));
        trans->Transform(ip, out);
        height = std::max(height, out(2));
    }
    return height;
}

/**
 * A simpler point-in-element test, for comparison only: plain Newton from
 * the element center, an "Outside" answer trusted as final, and the projected fallback only when
 * Newton did not converge. Used to report how many classifications the retries fix.
 */
bool SingleStartInside(mfem::Mesh& mesh, int e, const std::array<double, 3>& p) {
    mfem::IsoparametricTransformation trans;
    mesh.GetElementTransformation(e, &trans);
    mfem::Vector point(3);
    point(0) = p[0];
    point(1) = p[1];
    point(2) = p[2];
    mfem::IntegrationPoint ip;
    mfem::InverseElementTransformation inverse(&trans);
    inverse.SetInitialGuessType(mfem::InverseElementTransformation::Center);
    inverse.SetSolverType(mfem::InverseElementTransformation::Newton);
    inverse.SetReferenceTol(1.0e-12);
    inverse.SetElementTol(1.0e-8);
    const int result = inverse.Transform(point, ip);
    if (result != mfem::InverseElementTransformation::Unknown) {
        return result == mfem::InverseElementTransformation::Inside;
    }
    inverse.SetInitialGuessType(mfem::InverseElementTransformation::ClosestRefNode);
    inverse.SetInitGuessRelOrder(1);
    inverse.SetSolverType(mfem::InverseElementTransformation::NewtonElementProject);
    return inverse.Transform(point, ip) == mfem::InverseElementTransformation::Inside;
}

/** The element types every test runs on. */
const mfem::Element::Type kTypes[] = {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON};

} // namespace

//------------------------------------------------------------------------------------------------
// Boundary clearance
//------------------------------------------------------------------------------------------------

// Planar faces keep their two triangles: a 2x2x2 hex cube (no Nodes) has 24 boundary quads, all
// planar, so the surface is 48 triangles (no planar face is subdivided).
TEST(CurvedGeometry, ClearancePlanarFacesUnchanged) {
    mfem::Mesh mesh = UnitCube(2, mfem::Element::HEXAHEDRON);
    const ms::BoundaryClearance clearance(mesh, 0.1, true);
    EXPECT_EQ(clearance.NumTriangles(), 48u);
}

// A single hex (no Nodes) whose top corners (0,0) and (1,1) are raised to z = 1.4 has a warped
// bilinear top face z = 1 + 0.4 [(1-x)(1-y) + x y], with height 1.2 and a horizontal tangent
// plane at its center. The point (0.5, 0.5, 1.15) is 0.05 below that surface, so with clearance
// 0.06 it is near the surface. The old two-triangle split 0-1-2 / 0-2-3 put the face center on
// the diagonal at z = 1.4, about 0.22 from the point, and missed it. The warp |w.n|/4 = 0.2 and
// the face size sqrt(2) give k = ceil(sqrt(0.2 / (1e-3 sqrt(2)))) = 12 subdivisions, i.e.
// 2 * 144 triangles on top plus 2 on each of the 5 planar faces. A grid of points under the top
// face is then classified against the brute-force distance to the bilinear faces.
TEST(CurvedGeometry, ClearanceWarpedBilinearFace) {
    mfem::Mesh mesh = UnitCube(1, mfem::Element::HEXAHEDRON);
    for (int v = 0; v < mesh.GetNV(); v++) {
        double* x = mesh.GetVertex(v);
        const bool diagonal_corner = (x[0] == x[1]); // (0,0) and (1,1)
        if (x[2] == 1.0 && diagonal_corner) {
            x[2] = 1.4;
        }
    }
    const double c = 0.06;
    const ms::BoundaryClearance clearance(mesh, c, true);
    EXPECT_EQ(clearance.NumTriangles(), 10u + 288u);

    std::vector<std::array<double, 3>> points = {{0.5, 0.5, 1.15}};
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            const double x = 0.125 * i;
            const double y = 0.125 * j;
            const double top = 1.0 + 0.4 * ((1 - x) * (1 - y) + x * y);
            for (double depth : {0.02, 0.04, 0.055, 0.065, 0.08, 0.12}) {
                points.push_back({x, y, top - depth});
            }
        }
    }
    const auto n = static_cast<std::int64_t>(points.size());
    std::vector<double> px(points.size()), py(points.size()), pz(points.size());
    for (std::size_t i = 0; i < points.size(); i++) {
        px[i] = points[i][0];
        py[i] = points[i][1];
        pz[i] = points[i][2];
    }
    std::vector<std::uint64_t> near((points.size() + 63) / 64, 0);
    clearance.MarkNear(n, px.data(), py.data(), pz.data(), near.data());
    EXPECT_TRUE(Bit(near, 0)) << "the center point is 0.05 below the bilinear surface";

    // Tolerance band: subdivision error <= 1e-3 * sqrt(2), sampling error ~1e-5
    const double band = 2.0e-3;
    int checked = 0;
    for (std::size_t i = 0; i < points.size(); i++) {
        const double d = BruteForceSurfaceDistance(mesh, points[i], 200);
        if (std::fabs(d - c) < band) {
            continue;
        }
        checked++;
        EXPECT_EQ(Bit(near, i), d < c) << "point " << i << " at distance " << d;
    }
    EXPECT_GT(checked, 200);
}

// Genuinely curved surface: the bulged order-2 cube (top z = 1 + 0.2 sin(pi x) sin(pi y)),
// distributed over the ranks, every rank marking against its own boundary faces and the masks
// OR-combined (as seeding does). Curved faces are triangulated at 2 * order = 4 subdivisions
// per edge, so the error against the mesh's own (order-2) surface is the sag of one
// sub-triangle, about (1/12)^2 / 8 * 2 ~ 2e-3: points farther than 3e-3 from the clearance
// distance are checked against the brute-force distance to the mesh faces. Against the analytic
// bulge, whose order-2 interpolant on elements of size 1/3 differs from it by up to ~1e-2, the
// classification is checked outside a band of 1.5e-2. Points stay in x, y in [0.3, 0.7], at
// least 0.3 from the side faces.
TEST(CurvedGeometry, ClearanceBulgedSurface) {
    for (const auto type : kTypes) {
        mfem::Mesh serial = BulgedCube(3, type, 2, 0.2);
        auto pmesh = Distribute(serial);
        const double c = 0.05;
        const ms::BoundaryClearance clearance(*pmesh, c, false);

        std::vector<double> px, py, pz;
        for (int i = 0; i <= 4; i++) {
            for (int j = 0; j <= 4; j++) {
                const double x = 0.3 + 0.1 * i;
                const double y = 0.3 + 0.1 * j;
                const double top = 1.0 + 0.2 * std::sin(kPi * x) * std::sin(kPi * y);
                for (double depth : {0.01, 0.03, 0.04, 0.045, 0.055, 0.06, 0.07, 0.09}) {
                    px.push_back(x);
                    py.push_back(y);
                    pz.push_back(top - depth);
                }
            }
        }
        const std::size_t n = px.size();
        std::vector<std::uint64_t> near((n + 63) / 64, 0);
        clearance.MarkNear(
            static_cast<std::int64_t>(n), px.data(), py.data(), pz.data(), near.data());
        MPI_Allreduce(MPI_IN_PLACE,
                      near.data(),
                      static_cast<int>(near.size()),
                      MPI_UINT64_T,
                      MPI_BOR,
                      MPI_COMM_WORLD);

        int checked_mesh = 0;
        int checked_analytic = 0;
        for (std::size_t i = 0; i < n; i++) {
            const std::array<double, 3> p = {px[i], py[i], pz[i]};
            // Mesh surface reference
            const double d_mesh = BruteForceSurfaceDistance(serial, p, 48);
            if (std::fabs(d_mesh - c) > 3.0e-3) {
                checked_mesh++;
                EXPECT_EQ(Bit(near, i), d_mesh < c)
                    << "type " << type << " point " << i << " mesh distance " << d_mesh;
            }
            // Analytic surface reference: dense sampling of z = 1 + 0.2 sin sin near the point
            double best_sq = std::numeric_limits<double>::infinity();
            for (int a = -100; a <= 100; a++) {
                for (int b = -100; b <= 100; b++) {
                    const double x = p[0] + 0.001 * a;
                    const double y = p[1] + 0.001 * b;
                    const double z = 1.0 + 0.2 * std::sin(kPi * x) * std::sin(kPi * y);
                    const double dx = x - p[0];
                    const double dy = y - p[1];
                    const double dz = z - p[2];
                    best_sq = std::min(best_sq, dx * dx + dy * dy + dz * dz);
                }
            }
            const double d_exact = std::sqrt(best_sq);
            if (std::fabs(d_exact - c) > 1.5e-2) {
                checked_analytic++;
                EXPECT_EQ(Bit(near, i), d_exact < c)
                    << "type " << type << " point " << i << " analytic distance " << d_exact;
            }
        }
        EXPECT_GT(checked_mesh, 100);
        EXPECT_GT(checked_analytic, 50);
    }
}

//------------------------------------------------------------------------------------------------
// Mesh writers: curved round trips
//------------------------------------------------------------------------------------------------

/** Volumes of all local elements (through the Nodes for a curved mesh). */
std::vector<double> ElementVolumes(mfem::Mesh& mesh) {
    std::vector<double> volumes(static_cast<std::size_t>(mesh.GetNE()));
    for (int e = 0; e < mesh.GetNE(); e++) {
        volumes[static_cast<std::size_t>(e)] = mesh.GetElementVolume(e);
    }
    return volumes;
}

/** All ranks' values gathered on every rank and sorted. Collective. */
std::vector<double> GatherSorted(const std::vector<double>& local) {
    const int nranks = NumRanks();
    int count = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> offsets(static_cast<std::size_t>(nranks), 0);
    for (int r = 1; r < nranks; r++) {
        offsets[r] = offsets[r - 1] + counts[r - 1];
    }
    std::vector<double> all(static_cast<std::size_t>(offsets.back() + counts.back()));
    MPI_Allgatherv(local.data(),
                   count,
                   MPI_DOUBLE,
                   all.data(),
                   counts.data(),
                   offsets.data(),
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);
    std::sort(all.begin(), all.end());
    return all;
}

// The bulged order-2 mesh written with each mesh writer and read back the way ExaConstit reads
// it keeps its curved geometry: Nodes of order 2, and element volumes equal to the original's to
// round-off (numbers are written with 17 significant digits). The volumes differ from element to
// element because of the bulge, so a writer that dropped the Nodes (straight elements) would
// change the top-layer volumes by several percent. Checked for the serial writer, the
// partitioned writer (one part per rank) and the ParPrint writer of a distributed mesh (local
// element order preserved, so volumes are compared element by element).
TEST(CurvedGeometry, MeshRoundTripsKeepCurvedVolumes) {
    for (const auto type : kTypes) {
        const std::string tag = (type == mfem::Element::HEXAHEDRON) ? "hex" : "tet";
        mfem::Mesh serial = BulgedCube(3, type, 2, 0.2);
        for (int e = 0; e < serial.GetNE(); e++) {
            serial.SetAttribute(e, 1 + e % 5);
        }
        serial.SetAttributes();
        const std::vector<double> reference = ElementVolumes(serial);
        std::vector<double> reference_sorted = reference;
        std::sort(reference_sorted.begin(), reference_sorted.end());
        const double tol = 1e-13;

        // Serial writer, read back with mfem::Mesh(path) on every rank
        const fs::path dir = SharedDir("round_trip_" + tag);
        fs::path serial_path = dir / "bulged.mesh";
        if (Rank() == 0) {
            serial_path = ms::WriteSerialMesh(serial, serial_path, 0, "curved round trip");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        {
            mfem::Mesh back(serial_path.string().c_str(), 1, 1, true);
            ASSERT_NE(back.GetNodes(), nullptr);
            EXPECT_EQ(back.GetNodes()->FESpace()->GetMaxElementOrder(), 2);
            const std::vector<double> volumes = ElementVolumes(back);
            ASSERT_EQ(volumes.size(), reference.size());
            for (std::size_t e = 0; e < volumes.size(); e++) {
                EXPECT_NEAR(volumes[e], reference[e], tol * std::fabs(reference[e]))
                    << tag << " serial element " << e;
            }
        }

        // Partitioned writer, one part per rank, read back as ParMesh(comm, istream)
        const std::string prefix = (dir / "part.").string();
        ms::WritePartitionedMesh(serial, NumRanks(), prefix, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        {
            mfem::ifgzstream in(mfem::MakeParFilename(prefix, Rank()));
            mfem::ParMesh back(MPI_COMM_WORLD, in, true, 1, true);
            ASSERT_NE(back.GetNodes(), nullptr);
            const std::vector<double> all = GatherSorted(ElementVolumes(back));
            ASSERT_EQ(all.size(), reference_sorted.size());
            for (std::size_t e = 0; e < all.size(); e++) {
                EXPECT_NEAR(all[e], reference_sorted[e], tol * std::fabs(reference_sorted[e]))
                    << tag << " partitioned, sorted volume " << e;
            }
        }

        // ParPrint writer of a distributed mesh, local element order preserved
        auto pmesh = Distribute(serial);
        const std::vector<double> local_reference = ElementVolumes(*pmesh);
        const std::string par_prefix = (dir / "par.").string();
        ms::WriteParallelMesh(*pmesh, par_prefix, 0, "curved round trip");
        MPI_Barrier(MPI_COMM_WORLD);
        {
            mfem::ifgzstream in(mfem::MakeParFilename(par_prefix, Rank()));
            mfem::ParMesh back(MPI_COMM_WORLD, in, true, 1, true);
            ASSERT_NE(back.GetNodes(), nullptr);
            const std::vector<double> volumes = ElementVolumes(back);
            ASSERT_EQ(volumes.size(), local_reference.size());
            for (std::size_t e = 0; e < volumes.size(); e++) {
                EXPECT_NEAR(volumes[e], local_reference[e], tol * std::fabs(local_reference[e]))
                    << tag << " ParPrint element " << e;
            }
        }
    }
}

//------------------------------------------------------------------------------------------------
// Point location
//------------------------------------------------------------------------------------------------

// The sampled curved element boxes contain the element: every point of a much finer reference
// lattice (8p subdivisions per direction, 4 times the lattice the box was built from) mapped
// through the strongly bulged order-3 elements lies in its element's box. Also checks that the
// boxes are tight for elements that are straight in fact (the bottom layer is only slightly
// curved; its boxes stay within 1% of the element size of the corner box).
TEST(CurvedGeometry, CurvedElementBoxesContainElements) {
    for (const auto type : kTypes) {
        mfem::Mesh mesh = BulgedCube(2, type, 3, 0.5);
        const ms::MaterialPointLocator locator(mesh, false);
        mfem::IsoparametricTransformation trans;
        mfem::DenseMatrix x;
        for (int e = 0; e < mesh.GetNE(); e++) {
            mesh.GetElementTransformation(e, &trans);
            mfem::RefinedGeometry* fine = mfem::GlobGeometryRefiner.Refine(
                mesh.GetElementGeometry(e), 24);
            trans.Transform(fine->RefPts, x);
            const auto& box = locator.ElementBox(e);
            for (int q = 0; q < x.Width(); q++) {
                for (int d = 0; d < 3; d++) {
                    EXPECT_GE(x(d, q), box.getMin()[d]) << "type " << type << " element " << e;
                    EXPECT_LE(x(d, q), box.getMax()[d]) << "type " << type << " element " << e;
                }
            }
        }
    }
}

// Points next to a strongly curved face: the 2x2x2 cube with order-3 Nodes and the top bulged by
// z -> z (1 + 0.5 sin(pi x) sin(pi y)), so the top face rises by up to half the element size
// over one element. Points straddle the mesh's own top surface at vertical offsets from 1e-6 to
// 0.05 above and below it; the reference answer is "inside exactly when z is below the top
// surface height", which TopSurfaceHeight computes from the face maps alone (no inverse map).
// The locator runs serially and distributed (per-rank masks OR-combined). For information, the
// number of points that the previous single-start Newton test misclassified is printed.
TEST(CurvedGeometry, PointsNearStronglyCurvedFace) {
    for (const auto type : kTypes) {
        for (const double amplitude : {0.5, -0.4}) {
            mfem::Mesh serial = BulgedCube(2, type, 3, amplitude);
            const double top_min_z = 1.0 + std::min(0.0, amplitude);

            std::vector<double> px, py, pz;
            std::vector<char> expected;
            for (int i = 0; i <= 12; i++) {
                for (int j = 0; j <= 12; j++) {
                    const double x = 0.02 + 0.08 * i;
                    const double y = 0.02 + 0.08 * j;
                    const double top = TopSurfaceHeight(serial, x, y, top_min_z);
                    ASSERT_TRUE(std::isfinite(top)) << "no top face above " << x << ", " << y;
                    for (double offset : {1e-6, 1e-4, 1e-3, 1e-2, 5e-2}) {
                        for (double sign : {-1.0, 1.0}) {
                            px.push_back(x);
                            py.push_back(y);
                            pz.push_back(top + sign * offset);
                            expected.push_back(sign < 0 ? 1 : 0);
                        }
                    }
                }
            }
            const std::size_t n = px.size();

            // Serial locator
            {
                const ms::MaterialPointLocator locator(serial, false);
                std::vector<std::uint64_t> inside((n + 63) / 64, 0);
                locator.MarkInside(
                    static_cast<std::int64_t>(n), px.data(), py.data(), pz.data(), inside.data());
                int wrong_before = 0;
                for (std::size_t i = 0; i < n; i++) {
                    EXPECT_EQ(Bit(inside, i), expected[i] != 0)
                        << "type " << type << " point (" << px[i] << ", " << py[i] << ", " << pz[i]
                        << ")";
                    // Old single-start test over all elements, for the information line
                    bool old_inside = false;
                    const std::array<double, 3> p = {px[i], py[i], pz[i]};
                    for (int e = 0; e < serial.GetNE() && !old_inside; e++) {
                        old_inside = SingleStartInside(serial, e, p);
                    }
                    wrong_before += (old_inside != (expected[i] != 0)) ? 1 : 0;
                }
                if (Rank() == 0) {
                    std::cout << "[ info ] type " << type << ", amplitude " << amplitude << ": "
                              << n << " points, single-start "
                              << "Newton misclassified " << wrong_before << std::endl;
                }
            }

            // Distributed locator, masks OR-combined over the ranks
            auto pmesh = Distribute(serial);
            const ms::MaterialPointLocator locator(*pmesh, false);
            std::vector<std::uint64_t> inside((n + 63) / 64, 0);
            locator.MarkInside(
                static_cast<std::int64_t>(n), px.data(), py.data(), pz.data(), inside.data());
            MPI_Allreduce(MPI_IN_PLACE,
                          inside.data(),
                          static_cast<int>(inside.size()),
                          MPI_UINT64_T,
                          MPI_BOR,
                          MPI_COMM_WORLD);
            for (std::size_t i = 0; i < n; i++) {
                EXPECT_EQ(Bit(inside, i), expected[i] != 0)
                    << "distributed, type " << type << " point " << i;
            }
        }
    }
}

/** Bends the unit cube into an annular sector: r = 1 + x, theta = 3.6 y, z unchanged. */
void BendToSector(const mfem::Vector& x, mfem::Vector& y) {
    const double r = 1.0 + x(0);
    const double theta = 3.6 * x(1);
    y.SetSize(3);
    y(0) = r * std::cos(theta);
    y(1) = r * std::sin(theta);
    y(2) = x(2);
}

// Randomly distorted curved elements keep their interior points. A 4^3 box with order-2 or
// order-3 Nodes has every node moved by a random amount of up to amp h / 2 in each coordinate;
// elements with detJ <= 0 anywhere on a 9^3 reference lattice are skipped (preflight would
// reject most of them). Points mapped from random reference points inside each remaining element
// are points of that element, so PointInElement must return true for all of them. This guards
// the early reject of the curved retries (MaterialPointLocator::PointInCurvedElement): with a
// distance-only rule about 1 in 1000 of these points was lost (spurious Newton roots 0.04 - 0.5
// reference units outside the element); the rule with the unfolded-root and same-root conditions
// loses none here. (Heavier distortions, e.g. order-3 tets at amp = 0.3, have a few points that
// none of the four attempts finds; they are outside the range this test covers.)
TEST(CurvedGeometry, PerturbedElementsKeepInteriorPoints) {
    std::mt19937_64 gen(12345);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    struct Case {
        mfem::Element::Type type;
        int order;
        double amp;
    };
    const Case cases[] = {{mfem::Element::HEXAHEDRON, 2, 0.15},
                          {mfem::Element::HEXAHEDRON, 2, 0.3},
                          {mfem::Element::HEXAHEDRON, 3, 0.15},
                          {mfem::Element::TETRAHEDRON, 2, 0.15},
                          {mfem::Element::TETRAHEDRON, 2, 0.3},
                          {mfem::Element::TETRAHEDRON, 2, 0.45},
                          {mfem::Element::TETRAHEDRON, 3, 0.15}};
    for (const Case& c : cases) {
        const bool tet = c.type == mfem::Element::TETRAHEDRON;
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, c.type, 1.0, 1.0, 1.0, false);
        mesh.SetCurvature(c.order);
        mfem::GridFunction& nodes = *mesh.GetNodes();
        for (int i = 0; i < nodes.Size(); i++) {
            nodes(i) += c.amp * 0.25 * (uniform(gen) - 0.5);
        }
        const ms::MaterialPointLocator locator(mesh, false);
        long tested = 0;
        long lost = 0;
        for (int e = 0; e < mesh.GetNE(); e++) {
            mfem::IsoparametricTransformation trans;
            mesh.GetElementTransformation(e, &trans);
            // Skip elements folded anywhere on a 9^3 reference lattice
            bool folded = false;
            const int n = 8;
            for (int a = 0; a <= n && !folded; a++) {
                for (int b = 0; b <= n && !folded; b++) {
                    for (int k = 0; k <= n && !folded; k++) {
                        if (tet && a + b + k > n) {
                            continue;
                        }
                        mfem::IntegrationPoint ip;
                        ip.Set3(a / double(n), b / double(n), k / double(n));
                        trans.SetIntPoint(&ip);
                        folded = !(trans.Weight() > 0.0);
                    }
                }
            }
            if (folded) {
                continue;
            }
            for (int k = 0; k < 100; k++) {
                double x = uniform(gen);
                double y = uniform(gen);
                double z = uniform(gen);
                while (tet && x + y + z > 1.0) {
                    x = uniform(gen);
                    y = uniform(gen);
                    z = uniform(gen);
                }
                mfem::IntegrationPoint ip;
                ip.Set3(x, y, z);
                mfem::Vector point(3);
                trans.Transform(ip, point);
                tested++;
                lost += locator.PointInElement(e, {point(0), point(1), point(2)}) ? 0 : 1;
            }
        }
        EXPECT_GT(tested, 0);
        EXPECT_EQ(lost, 0) << (tet ? "tet" : "hex") << " order " << c.order << " amp " << c.amp;
    }
}

// One element bent into an annular sector of 3.6 rad (r from 1 to 2), with order-2 and order-3
// Nodes: the element is strongly curved and its center is far, in the Newton sense, from its
// ends. Points mapped from a lattice of reference points strictly inside the element must be
// found inside (they are points of the element by construction), and points mapped from
// reference points just outside the faces r = 1 and r = 2 (radius exactly linear in the first
// reference coordinate, so they lie at r < 1 or r > 2, outside the sector) must be found
// outside. Newton started only from the element center (the simpler test) converges for
// some interior points near the ends of the arc to a second solution outside the reference
// element and calls them outside (60 of 1200 at order 2, 90 at order 3 when written); the
// retries from the closest nodal point and the closest lattice point find them. The test checks
// that the single-start count is positive, so it really exercises the retries.
TEST(CurvedGeometry, PointsInStronglyBentElement) {
    for (int order : {2, 3}) {
        mfem::Mesh mesh = UnitCube(1, mfem::Element::HEXAHEDRON);
        mesh.SetCurvature(order);
        mesh.Transform(BendToSector);
        const ms::PreflightReport report = ms::CheckMesh(mesh, MPI_COMM_SELF);
        ASSERT_TRUE(report.Passed()) << ms::DescribePreflightFailures(report);

        std::vector<double> px, py, pz;
        std::vector<char> expected;
        mfem::IsoparametricTransformation trans;
        mesh.GetElementTransformation(0, &trans);
        mfem::Vector x(3);
        auto add = [&](double a, double b, double c, bool inside) {
            mfem::IntegrationPoint ip;
            ip.Set3(a, b, c);
            trans.Transform(ip, x);
            px.push_back(x(0));
            py.push_back(x(1));
            pz.push_back(x(2));
            expected.push_back(inside ? 1 : 0);
        };
        for (int i = 0; i <= 9; i++) {
            for (int j = 0; j <= 9; j++) {
                for (int k = 0; k <= 9; k++) {
                    add(0.02 + 0.106 * i, 0.02 + 0.106 * j, 0.02 + 0.106 * k, true);
                }
                add(-0.01, 0.05 + 0.1 * i, 0.05 + 0.1 * j, false);
                add(1.01, 0.05 + 0.1 * i, 0.05 + 0.1 * j, false);
            }
        }
        const std::size_t n = px.size();
        const ms::MaterialPointLocator locator(mesh, false);
        std::vector<std::uint64_t> inside((n + 63) / 64, 0);
        locator.MarkInside(
            static_cast<std::int64_t>(n), px.data(), py.data(), pz.data(), inside.data());
        int wrong_before = 0;
        for (std::size_t i = 0; i < n; i++) {
            EXPECT_EQ(Bit(inside, i), expected[i] != 0) << "order " << order << " point " << i;
            const std::array<double, 3> p = {px[i], py[i], pz[i]};
            wrong_before += (SingleStartInside(mesh, 0, p) != (expected[i] != 0)) ? 1 : 0;
        }
        if (Rank() == 0) {
            std::cout << "[ info ] bent element, order " << order << ": " << n
                      << " points, single-start Newton misclassified " << wrong_before << std::endl;
        }
        EXPECT_GT(wrong_before, 0) << "the case no longer needs the Newton retries";
    }
}

//------------------------------------------------------------------------------------------------
// Preflight
//------------------------------------------------------------------------------------------------

// A trilinear hex folded at one corner: in element-relative coordinates, moving the corner
// (1,1,1) of a cube element to (t,t,t) gives detJ proportional to 3t - 2 at that corner (the
// triple product of its three edges (t, t-1, t-1) and permutations), negative for t = 0.65,
// while detJ stays positive at all quadrature points of the sampling rule (checked below; at
// t = 0.6 the order-2 rule already sees the fold). The quadrature-only check passed this
// element; the nodal-point check (the element's nodal points plus its corners: the 8 corners
// without Nodes, the 27 nodes with order-2 Nodes) catches it and counts it as folded between the
// quadrature points.
// Distributed over the ranks: exactly one bad element globally, and every rank sees the same
// counts.
TEST(CurvedGeometry, PreflightCatchesFoldBetweenQuadraturePoints) {
    for (int order : {0, 2}) {
        mfem::Mesh serial = UnitCube(2, mfem::Element::HEXAHEDRON);
        // Move the vertex (1,1,1) of the corner element to (t, t, t) relative to the element
        // [0.5,1]^3: (0.5 + 0.5 t, ...) with t = 0.65
        for (int v = 0; v < serial.GetNV(); v++) {
            double* x = serial.GetVertex(v);
            if (x[0] == 1.0 && x[1] == 1.0 && x[2] == 1.0) {
                x[0] = x[1] = x[2] = 0.5 + 0.5 * 0.65;
            }
        }
        if (order > 0) {
            serial.SetCurvature(order);
        }

        // The fold is invisible at the quadrature points of the sampling rule
        {
            const int quad_order = ms::DefaultSampleQuadOrder(serial);
            mfem::IsoparametricTransformation trans;
            for (int e = 0; e < serial.GetNE(); e++) {
                serial.GetElementTransformation(e, &trans);
                const mfem::IntegrationRule& ir = mfem::IntRules.Get(serial.GetElementGeometry(e),
                                                                     quad_order);
                for (int q = 0; q < ir.GetNPoints(); q++) {
                    trans.SetIntPoint(&ir.IntPoint(q));
                    ASSERT_GT(trans.Weight(), 0.0) << "order " << order << " element " << e;
                }
            }
        }

        auto pmesh = Distribute(serial);
        const ms::PreflightReport report = ms::CheckMesh(*pmesh);
        EXPECT_FALSE(report.Passed()) << "order " << order;
        EXPECT_EQ(report.num_bad_elements, 1) << "order " << order;
        EXPECT_EQ(report.num_folded_between_points, 1) << "order " << order;
        EXPECT_LT(report.min_detj, 0.0);
        long long local = static_cast<long long>(report.local_bad_elements.size());
        long long total = 0;
        MPI_Allreduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(total, 1);
        EXPECT_NE(ms::DescribePreflightFailures(report).find("fold between them"),
                  std::string::npos);
    }
}

// A fold at a non-corner node of an order-2 hex: the unit cube element, with the node at the
// middle of the edge x = y = 1 pushed inward by (-0.2, -0.2, 0), or the node at the center of the
// face x = 1 pushed inward by 0.4. detJ is then negative at the moved node (-0.2 in both cases)
// but positive at every quadrature point of the sampling rule and at all 8 corners (detJ = 1
// there, measured), so neither a quadrature-only nor a corner-only check would catch it; the
// nodal points do, and the element counts as folded between the quadrature points. Serial
// (MPI_COMM_SELF) on every rank.
TEST(CurvedGeometry, PreflightCatchesFoldAtNonCornerNode) {
    for (const bool edge : {true, false}) {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
            1, 1, 1, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
        mesh.SetCurvature(2);
        mfem::GridFunction& nodes = *mesh.GetNodes();
        const mfem::FiniteElementSpace* fes = nodes.FESpace();
        const double target[3] = {1.0, edge ? 1.0 : 0.5, 0.5};
        int moved = 0;
        for (int d = 0; d < fes->GetNDofs(); d++) {
            bool match = true;
            for (int c = 0; c < 3; c++) {
                match = match && std::fabs(nodes(fes->DofToVDof(d, c)) - target[c]) < 1e-12;
            }
            if (match) {
                nodes(fes->DofToVDof(d, 0)) -= edge ? 0.2 : 0.4;
                if (edge) {
                    nodes(fes->DofToVDof(d, 1)) -= 0.2;
                }
                moved++;
            }
        }
        ASSERT_EQ(moved, 1);
        const ms::PreflightReport report = ms::CheckMesh(mesh, MPI_COMM_SELF);
        EXPECT_EQ(report.num_bad_elements, 1) << (edge ? "edge" : "face");
        EXPECT_EQ(report.num_folded_between_points, 1) << (edge ? "edge" : "face");
        EXPECT_NEAR(report.min_detj, -0.2, 1e-12);
    }
}

// Valid curved meshes still pass: the bulged order-2 and order-3 cubes have positive detJ at
// every quadrature and lattice point.
TEST(CurvedGeometry, PreflightPassesBulgedMeshes) {
    for (const auto type : kTypes) {
        for (int order : {2, 3}) {
            mfem::Mesh serial = BulgedCube(2, type, order, 0.5);
            auto pmesh = Distribute(serial);
            const ms::PreflightReport report = ms::CheckMesh(*pmesh);
            EXPECT_TRUE(report.Passed()) << ms::DescribePreflightFailures(report);
            EXPECT_EQ(report.num_folded_between_points, 0);
        }
    }
}

//------------------------------------------------------------------------------------------------
// ParaView
//------------------------------------------------------------------------------------------------

/** NumberOfPoints and NumberOfCells of this rank's .vtu piece below `dir` (proc<rank>.vtu). */
std::array<long, 2> PieceSizes(const fs::path& dir) {
    char name[32];
    std::snprintf(name, sizeof(name), "proc%06d.vtu", Rank());
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.path().filename() != name) {
            continue;
        }
        std::ifstream in(entry.path());
        std::string line;
        while (std::getline(in, line)) {
            const auto pos = line.find("<Piece NumberOfPoints=\"");
            if (pos == std::string::npos) {
                continue;
            }
            long points = 0;
            long cells = 0;
            std::sscanf(line.c_str() + pos,
                        "<Piece NumberOfPoints=\"%ld\" NumberOfCells=\"%ld\"",
                        &points,
                        &cells);
            return {points, cells};
        }
    }
    return {-1, -1};
}

// High-order ParaView output for curved meshes: the bulged order-2 hex mesh is written as one
// Lagrange cell of order 2 per element, i.e. (2+1)^3 = 27 points per element (Mesh::PrintVTU
// writes each element's own points), and the same mesh without Nodes as plain cells (8 points
// per hex at level of detail 1). An explicit levels_of_detail = 1 forces the plain output.
TEST(CurvedGeometry, ParaViewHighOrderForCurvedMesh) {
    mfem::Mesh curved = BulgedCube(2, mfem::Element::HEXAHEDRON, 2, 0.2);
    mfem::Mesh straight = UnitCube(2, mfem::Element::HEXAHEDRON);
    auto pcurved = Distribute(curved);
    auto pstraight = Distribute(straight);
    const fs::path dir = SharedDir("paraview");

    auto grains = [](const mfem::ParMesh& pmesh) {
        mfem::Array<int> g(pmesh.GetNE());
        for (int e = 0; e < pmesh.GetNE(); e++) {
            g[e] = 1 + e % 3;
        }
        return g;
    };
    const long ne = pcurved->GetNE();

    ms::WriteParaView(*pcurved, grains(*pcurved), {}, dir / "curved");
    auto sizes = PieceSizes(dir / "curved");
    EXPECT_EQ(sizes[0], 27 * ne);
    EXPECT_EQ(sizes[1], ne);

    ms::WriteParaView(*pstraight, grains(*pstraight), {}, dir / "straight");
    sizes = PieceSizes(dir / "straight");
    EXPECT_EQ(sizes[0], 8 * ne);
    EXPECT_EQ(sizes[1], ne);

    ms::WriteParaView(*pcurved, grains(*pcurved), {}, dir / "forced_linear", 1);
    sizes = PieceSizes(dir / "forced_linear");
    EXPECT_EQ(sizes[0], 8 * ne);
    EXPECT_EQ(sizes[1], ne);
}

/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
