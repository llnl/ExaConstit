/**
 * @file test_node_targets.cpp
 * @brief Tests of conformal/manifold_projector.hpp and conformal/node_targets:
 * targets of grain boundary nodes on the analytic boundaries, and their validation.
 *
 * What is checked, and why:
 *  - Voronoi bicrystal with an inclined bisector (hex): every target is the orthogonal projection
 *    of the node onto the plane (closed form in the test), restricted to the box faces it lies
 *    on, whose coordinates are kept bit for bit; no node is rejected.
 *  - Laguerre quad point (tet): triple line targets satisfy both equations and are closest points
 *    (x - y lies in the span of the two gradients); quad point targets equal the analytic point.
 *  - Additive (Johnson-Mehl) bicrystal (curved hyperboloid boundary): targets on the boundary and
 *    closest points (x - y parallel to the gradient).
 *  - Curved boundary (tets with a domed top): targets of the bisector's nodes on the dome lie on
 *    the bisector and on the dome's own triangles; nodes where the dome meets a side face slide
 *    along that feature edge (also on the side face, its coordinate kept bit for bit).
 *  - Validation: a grain boundary whose points are all closer to an unused seed (FOREIGN_SEED),
 *    two grains of the same seed (SAME_SEED), a grain without analytic boundary (HELD), a grain
 *    boundary along a creased top face (FIT inside the face and on its feature edges), and the
 *    projector's singular case.
 *  - Far target on a curved surface (max_move = 2): dome nodes 1.2 to 1.8 element sizes from the
 *    bisector reach it on the dome through two-ring patches; the one-ring patch does not hold
 *    the target.
 *  - Smoothed staircase (the same bicrystal on a finer mesh without seeds, as for a voxel map):
 *    every boundary node is FIT on the smoothed staircase, which lies far closer to the true
 *    bisector than the staircase nodes, keeps box face coordinates bit for bit, stays within the
 *    move limit, and is bit for bit the same on MPI_COMM_SELF; also with order-2 Nodes (smoothing
 *    over the lattice of the nodal points).
 *  - Rank invariance: the distributed targets and statuses equal, bit for bit, those of the same
 *    mesh on MPI_COMM_SELF (meshes distributed round robin), at np = 1, 2, 3, 4.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows.
 */
#include "microstructure/conformal/boundary_projector.hpp"
#include "microstructure/conformal/manifold_projector.hpp"
#include "microstructure/conformal/node_labels.hpp"
#include "microstructure/conformal/node_targets.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

/** Point coordinates rounded to 1e-9, to match points across meshes and ranks. */
using Key = std::array<long long, 3>;

/** The Key of point x. */
Key MakeKey(const double* x) {
    return {std::llround(x[0] * 1e9), std::llround(x[1] * 1e9), std::llround(x[2] * 1e9)};
}

/** Seeds with optional weights. */
ms::SeedSet MakeSeeds(const std::vector<std::array<double, 3>>& p,
                      const std::vector<double>& w = {}) {
    ms::SeedSet s;
    for (const auto& q : p) {
        s.x.push_back(q[0]);
        s.y.push_back(q[1]);
        s.z.push_back(q[2]);
    }
    s.weight = w;
    return s;
}

/** f_i(c) of the metric, written out in the test (not the library's functor). */
double MetricValue(ms::MetricType metric, const ms::SeedSet& s, int i, const double* c) {
    const double dx = c[0] - s.x[i], dy = c[1] - s.y[i], dz = c[2] - s.z[i];
    const double w = s.weight.empty() ? 0.0 : s.weight[static_cast<std::size_t>(i)];
    const double d2 = dx * dx + dy * dy + dz * dz;
    return metric == ms::MetricType::ADDITIVE ? std::sqrt(d2) - w : d2 - w;
}

/** Element labels: 1 + the seed with the smallest f at the element center. */
mfem::Array<int> SeedLabels(mfem::Mesh& mesh, ms::MetricType metric, const ms::SeedSet& s) {
    mfem::Array<int> labels(mesh.GetNE());
    mfem::Vector c(3);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.GetElementCenter(e, c);
        int best = 0;
        for (int i = 1; i < static_cast<int>(s.Size()); ++i) {
            const double* x = c.GetData();
            if (MetricValue(metric, s, i, x) < MetricValue(metric, s, best, x)) {
                best = i;
            }
        }
        labels[e] = best + 1;
    }
    return labels;
}

/** A mesh with its node labels and targets (members destroyed in reverse order). */
struct Fitted {
    std::unique_ptr<mfem::ParMesh> pmesh;     ///< the distributed mesh
    std::unique_ptr<ms::NodeLabels> labels;   ///< the node labels
    std::unique_ptr<ms::NodeTargets> targets; ///< the node targets
};

/**
 * Distributes `serial` on `comm` (element e on rank e mod size), labels the elements with
 * `labels_of` and computes the targets.
 */
Fitted Fit(mfem::Mesh& serial,
           MPI_Comm comm,
           const std::function<mfem::Array<int>(mfem::Mesh&)>& labels_of,
           ms::MetricType metric,
           const ms::SeedSet& seeds,
           const std::vector<int>& grain_seed,
           const ms::NodeLabelOptions& label_options = {},
           const ms::NodeTargetOptions& target_options = {}) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> part(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        part[static_cast<std::size_t>(e)] = e % size;
    }
    Fitted f;
    f.pmesh = std::make_unique<mfem::ParMesh>(comm, serial, part.data());
    const mfem::Array<int> labels = labels_of(*f.pmesh);
    f.labels = std::make_unique<ms::NodeLabels>(*f.pmesh, labels, label_options);
    f.targets = std::make_unique<ms::NodeTargets>(
        *f.labels, metric, seeds, ms::MetricSetup{}, grain_seed, target_options);
    return f;
}

/** Number of local nodes whose target or status differs from the MPI_COMM_SELF run. */
int CompareWithSerial(const Fitted& dist, const Fitted& self) {
    std::map<Key, int> index;
    const double* xs = self.labels->Field().Coordinates().HostRead();
    for (int v = 0; v < self.labels->NumNodes(); ++v) {
        index[MakeKey(xs + 3 * v)] = v;
    }
    const double* xd = dist.labels->Field().Coordinates().HostRead();
    const double* td = dist.targets->Targets().HostRead();
    const double* ts = self.targets->Targets().HostRead();
    int bad = 0;
    for (int v = 0; v < dist.labels->NumNodes(); ++v) {
        const auto it = index.find(MakeKey(xd + 3 * v));
        if (it == index.end()) {
            ++bad;
            continue;
        }
        const int w = it->second;
        bad += (td[3 * v] != ts[3 * w] || td[3 * v + 1] != ts[3 * w + 1] ||
                td[3 * v + 2] != ts[3 * w + 2] ||
                dist.targets->Status(v) != self.targets->Status(w))
                   ? 1
                   : 0;
    }
    return bad;
}

/** Boundary triangles of a tet mesh on its top (z > 0.5 at all corners). */
std::vector<std::array<double, 9>> TopTriangles(mfem::Mesh& serial) {
    std::vector<std::array<double, 9>> top;
    mfem::Array<int> bv;
    for (int b = 0; b < serial.GetNBE(); ++b) {
        serial.GetBdrElementVertices(b, bv);
        std::array<double, 9> t{};
        bool up = true;
        for (int c = 0; c < 3; ++c) {
            for (int d = 0; d < 3; ++d) {
                t[static_cast<std::size_t>(3 * c + d)] = serial.GetVertex(bv[c])[d];
            }
            up = up && t[static_cast<std::size_t>(3 * c + 2)] > 0.5;
        }
        if (up) {
            top.push_back(t);
        }
    }
    return top;
}

/** y lies on some triangle: in its plane and inside it (barycentric coordinates >= 0). */
bool OnTriangles(const std::vector<std::array<double, 9>>& tris, const double* y) {
    for (const auto& t : tris) {
        const double u[3] = {t[3] - t[0], t[4] - t[1], t[5] - t[2]};
        const double w[3] = {t[6] - t[0], t[7] - t[1], t[8] - t[2]};
        const double r[3] = {y[0] - t[0], y[1] - t[1], y[2] - t[2]};
        const double n[3] = {
            u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
        const double nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        const double off = (r[0] * n[0] + r[1] * n[1] + r[2] * n[2]) / std::sqrt(nn);
        // Barycentric coordinates of the projection onto the plane
        const double uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        const double ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        const double uw = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
        const double ru = r[0] * u[0] + r[1] * u[1] + r[2] * u[2];
        const double rw = r[0] * w[0] + r[1] * w[1] + r[2] * w[2];
        const double det = uu * ww - uw * uw;
        const double a = (ru * ww - rw * uw) / det;
        const double b = (rw * uu - ru * uw) / det;
        if (std::fabs(off) < 1e-12 && a >= -1e-12 && b >= -1e-12 && a + b <= 1.0 + 1e-12) {
            return true;
        }
    }
    return false;
}

/** Sum of `local` over MPI_COMM_WORLD (collective). */
int GlobalSum(int local) {
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return local;
}

/** True when coordinate c lies on a face of the unit box. */
bool OnBoxFace(double c) {
    return c == 0.0 || c == 1.0;
}

constexpr int FIT = static_cast<int>(ms::TargetStatus::FIT); ///< summary index of FIT
/** summary index of FAKE_JUNCTION */
constexpr int FAKE = static_cast<int>(ms::TargetStatus::FAKE_JUNCTION);
constexpr int HELD = static_cast<int>(ms::TargetStatus::HELD); ///< summary index of HELD

} // namespace

// Two planes (x + y = 0, z = 0.5): the closest point (-0.2, 0.2, 0.5) is found in one Newton step
// (the second only confirms it, the constraints being linear); two parallel planes have no common
// point and are reported SINGULAR.
TEST(ManifoldProjector, PlanesAndSingularRows) {
    ms::VoronoiMetric unused{};
    const double x[3] = {0.3, 0.7, -0.2};
    ms::ManifoldRow rows[2];
    rows[0].normal[0] = 1.0;
    rows[0].normal[1] = 1.0;
    rows[0].offset = 0.0; // x + y = 0
    rows[1].normal[2] = 2.0;
    rows[1].offset = 1.0; // z = 0.5
    double y[3];
    int iterations = 0;
    EXPECT_EQ(ms::ProjectOntoManifold(unused, rows, 2, x, 1e-14, 5, y, &iterations),
              ms::ProjectionResult::CONVERGED);
    EXPECT_NEAR(y[0], -0.2, 1e-15);
    EXPECT_NEAR(y[1], 0.2, 1e-15);
    EXPECT_NEAR(y[2], 0.5, 1e-15);
    EXPECT_EQ(iterations, 2); // linear: the second step confirms the first
    rows[1] = rows[0];
    rows[1].offset = 1.0; // parallel planes
    EXPECT_EQ(ms::ProjectOntoManifold(unused, rows, 2, x, 1e-14, 5, y),
              ms::ProjectionResult::SINGULAR);
}

// Inclined Voronoi bicrystal: every two-grain node targets its closest point on the bisector
// within the coordinates it may move (box-face coordinates stay), computed here in closed form;
// the distributed run equals the single-rank one.
TEST(NodeTargets, VoronoiInclinedBicrystal) {
    const ms::SeedSet seeds = MakeSeeds({{0.3, 0.4, 0.5}, {0.7, 0.55, 0.45}});
    const auto labels_of = [&](mfem::Mesh& m) {
        return SeedLabels(m, ms::MetricType::VORONOI, seeds);
    };
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    Fitted dist = Fit(serial, MPI_COMM_WORLD, labels_of, ms::MetricType::VORONOI, seeds, {0, 1});
    Fitted self = Fit(serial, MPI_COMM_SELF, labels_of, ms::MetricType::VORONOI, seeds, {0, 1});
    EXPECT_EQ(CompareWithSerial(dist, self), 0);

    // Bisector n . y = c with n = s1 - s0, c = (|s1|^2 - |s0|^2) / 2
    const double n[3] = {0.4, 0.15, -0.05};
    const double c = 0.5 * ((0.49 + 0.3025 + 0.2025) - (0.09 + 0.16 + 0.25));
    const ms::NodeLabels& nl = *dist.labels;
    const double* xyz = nl.Field().Coordinates().HostRead();
    const double* t = dist.targets->Targets().HostRead();
    int bad = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (nl.NumLabels(v) != 2) {
            continue;
        }
        const double* x = xyz + 3 * v;
        const double* y = t + 3 * v;
        // Expected: project x onto the plane within the subspace of the free coordinates
        double m[3];
        for (int d = 0; d < 3; ++d) {
            m[d] = OnBoxFace(x[d]) ? 0.0 : n[d];
        }
        const double mm = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
        const double s = (n[0] * x[0] + n[1] * x[1] + n[2] * x[2] - c) / mm;
        for (int d = 0; d < 3; ++d) {
            const double expect = x[d] - s * m[d];
            const bool kept = !OnBoxFace(x[d]) || y[d] == x[d];
            bad += (std::fabs(y[d] - expect) > 1e-13 || !kept) ? 1 : 0;
        }
        bad += (dist.targets->Status(v) != ms::TargetStatus::FIT) ? 1 : 0;
    }
    EXPECT_EQ(GlobalSum(bad), 0);
    const auto& s = dist.targets->Summary();
    EXPECT_EQ(s.status[FAKE], 0);
    EXPECT_EQ(s.status[FIT],
              nl.Summary().count[1][0] + nl.Summary().count[1][1] + nl.Summary().count[1][2]);
    EXPECT_GT(s.status[FIT], 0);
    EXPECT_LE(s.max_move_ratio, 1.0);
}

// Regular tetrahedron of seeds around the center; one weight moves the quad point
TEST(NodeTargets, LaguerreQuadPoint) {
    const double r = 0.3 / std::sqrt(3.0);
    const ms::SeedSet seeds = MakeSeeds({{0.5 + r, 0.5 + r, 0.5 + r},
                                         {0.5 + r, 0.5 - r, 0.5 - r},
                                         {0.5 - r, 0.5 + r, 0.5 - r},
                                         {0.5 - r, 0.5 - r, 0.5 + r}},
                                        {0.004, 0.0, 0.0, 0.0});
    const auto labels_of = [&](mfem::Mesh& m) {
        return SeedLabels(m, ms::MetricType::LAGUERRE, seeds);
    };
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    Fitted dist = Fit(
        serial, MPI_COMM_WORLD, labels_of, ms::MetricType::LAGUERRE, seeds, {0, 1, 2, 3});
    Fitted self = Fit(
        serial, MPI_COMM_SELF, labels_of, ms::MetricType::LAGUERRE, seeds, {0, 1, 2, 3});
    EXPECT_EQ(CompareWithSerial(dist, self), 0);

    // Analytic quad point: 2 y . (s_k - s_0) = |s_k|^2 - |s_0|^2 - w_k + w_0, k = 1, 2, 3
    mfem::DenseMatrix a(3);
    mfem::Vector b(3), q(3);
    const auto sq = [&](int i) {
        return seeds.x[i] * seeds.x[i] + seeds.y[i] * seeds.y[i] + seeds.z[i] * seeds.z[i];
    };
    for (int k = 1; k <= 3; ++k) {
        a(k - 1, 0) = 2.0 * (seeds.x[k] - seeds.x[0]);
        a(k - 1, 1) = 2.0 * (seeds.y[k] - seeds.y[0]);
        a(k - 1, 2) = 2.0 * (seeds.z[k] - seeds.z[0]);
        b(k - 1) = sq(k) - sq(0) - seeds.weight[k] + seeds.weight[0];
    }
    mfem::DenseMatrixInverse(a).Mult(b, q);

    const ms::NodeLabels& nl = *dist.labels;
    const double* xyz = nl.Field().Coordinates().HostRead();
    const double* t = dist.targets->Targets().HostRead();
    const int* off = nl.LabelOffsets().HostRead();
    const int* items = nl.Labels().HostRead();
    int bad = 0;
    int quad_fit = 0;
    int triple_fit = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (dist.targets->Status(v) != ms::TargetStatus::FIT || nl.BoundarySurfaces(v) > 0) {
            continue;
        }
        const double* x = xyz + 3 * v;
        const double* y = t + 3 * v;
        if (nl.NumLabels(v) == 4) {
            ++quad_fit;
            bad += (std::fabs(y[0] - q(0)) + std::fabs(y[1] - q(1)) + std::fabs(y[2] - q(2)) >
                    1e-13)
                       ? 1
                       : 0;
        } else if (nl.NumLabels(v) == 3) {
            ++triple_fit;
            // Both equations hold and x - y is in the span of the two plane normals
            const int i0 = items[off[v]] - 1;
            const int i1 = items[off[v] + 1] - 1;
            const int i2 = items[off[v] + 2] - 1;
            const double f0 = MetricValue(ms::MetricType::LAGUERRE, seeds, i0, y);
            bad += (std::fabs(f0 - MetricValue(ms::MetricType::LAGUERRE, seeds, i1, y)) > 1e-13 ||
                    std::fabs(f0 - MetricValue(ms::MetricType::LAGUERRE, seeds, i2, y)) > 1e-13)
                       ? 1
                       : 0;
            double g1[3], g2[3];
            const int ids[3] = {i0, i1, i2};
            const double sx[3] = {seeds.x[ids[0]], seeds.y[ids[0]], seeds.z[ids[0]]};
            const double s1[3] = {seeds.x[ids[1]], seeds.y[ids[1]], seeds.z[ids[1]]};
            const double s2[3] = {seeds.x[ids[2]], seeds.y[ids[2]], seeds.z[ids[2]]};
            for (int d = 0; d < 3; ++d) {
                g1[d] = s1[d] - sx[d];
                g2[d] = s2[d] - sx[d];
            }
            const double cr[3] = {g1[1] * g2[2] - g1[2] * g2[1],
                                  g1[2] * g2[0] - g1[0] * g2[2],
                                  g1[0] * g2[1] - g1[1] * g2[0]};
            const double dot = (x[0] - y[0]) * cr[0] + (x[1] - y[1]) * cr[1] +
                               (x[2] - y[2]) * cr[2];
            bad += (std::fabs(dot) > 1e-13) ? 1 : 0;
        }
    }
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GT(GlobalSum(quad_fit), 0);
    EXPECT_GT(GlobalSum(triple_fit), 0);
}

// |y - s0| - 0.15 = |y - s1|: one sheet of a hyperboloid around s1
TEST(NodeTargets, AdditiveCurvedBoundary) {
    const ms::SeedSet seeds = MakeSeeds({{0.2, 0.5, 0.5}, {0.8, 0.45, 0.5}}, {0.15, 0.0});
    const auto labels_of = [&](mfem::Mesh& m) {
        return SeedLabels(m, ms::MetricType::ADDITIVE, seeds);
    };
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    Fitted dist = Fit(serial, MPI_COMM_WORLD, labels_of, ms::MetricType::ADDITIVE, seeds, {0, 1});
    Fitted self = Fit(serial, MPI_COMM_SELF, labels_of, ms::MetricType::ADDITIVE, seeds, {0, 1});
    EXPECT_EQ(CompareWithSerial(dist, self), 0);

    const ms::NodeLabels& nl = *dist.labels;
    const double* xyz = nl.Field().Coordinates().HostRead();
    const double* t = dist.targets->Targets().HostRead();
    int bad = 0;
    int fit = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (nl.NumLabels(v) != 2) {
            continue;
        }
        bad += (dist.targets->Status(v) != ms::TargetStatus::FIT) ? 1 : 0;
        const double* x = xyz + 3 * v;
        const double* y = t + 3 * v;
        const double phi = MetricValue(ms::MetricType::ADDITIVE, seeds, 0, y) -
                           MetricValue(ms::MetricType::ADDITIVE, seeds, 1, y);
        bad += (std::fabs(phi) > 1e-12) ? 1 : 0;
        // Interior nodes: x - y parallel to grad phi = unit(y - s0) - unit(y - s1)
        if (nl.BoundarySurfaces(v) == 0) {
            ++fit;
            double g[3];
            const double r0 = std::sqrt(std::pow(y[0] - 0.2, 2) + std::pow(y[1] - 0.5, 2) +
                                        std::pow(y[2] - 0.5, 2));
            const double r1 = std::sqrt(std::pow(y[0] - 0.8, 2) + std::pow(y[1] - 0.45, 2) +
                                        std::pow(y[2] - 0.5, 2));
            const double s0[3] = {0.2, 0.5, 0.5}, s1[3] = {0.8, 0.45, 0.5};
            for (int d = 0; d < 3; ++d) {
                g[d] = (y[d] - s0[d]) / r0 - (y[d] - s1[d]) / r1;
            }
            const double u[3] = {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
            const double cr[3] = {
                u[1] * g[2] - u[2] * g[1], u[2] * g[0] - u[0] * g[2], u[0] * g[1] - u[1] * g[0]};
            bad += (std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]) > 1e-10) ? 1 : 0;
        } else {
            for (int d = 0; d < 3; ++d) {
                bad += (OnBoxFace(x[d]) && y[d] != x[d]) ? 1 : 0;
            }
        }
    }
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GT(GlobalSum(fit), 0);
}

// Tet box with a domed top z = 1 + 0.3 x (1 - x) + 0.3 y (1 - y): flat triangles with
// slightly different normals, one curved surface. The inclined Voronoi bisector crosses it;
// its nodes on the dome must reach the bisector while staying on the dome's triangles.
TEST(NodeTargets, CurvedTopSurface) {
    const ms::SeedSet seeds = MakeSeeds({{0.3, 0.4, 0.5}, {0.7, 0.55, 0.45}});
    const auto labels_of = [&](mfem::Mesh& m) {
        return SeedLabels(m, ms::MetricType::VORONOI, seeds);
    };
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    serial.Transform([](const mfem::Vector& x, mfem::Vector& y) {
        y = x;
        y(2) = x(2) * (1.0 + 0.3 * x(0) * (1.0 - x(0)) + 0.3 * x(1) * (1.0 - x(1)));
    });
    Fitted dist = Fit(serial, MPI_COMM_WORLD, labels_of, ms::MetricType::VORONOI, seeds, {0, 1});
    Fitted self = Fit(serial, MPI_COMM_SELF, labels_of, ms::MetricType::VORONOI, seeds, {0, 1});
    EXPECT_EQ(CompareWithSerial(dist, self), 0);

    const std::vector<std::array<double, 9>> dome = TopTriangles(serial);
    const auto on_dome = [&](const double* y) {
        return OnTriangles(dome, y);
    };
    const double n[3] = {0.4, 0.15, -0.05};
    const double c = 0.5 * ((0.49 + 0.3025 + 0.2025) - (0.09 + 0.16 + 0.25));
    const ms::NodeLabels& nl = *dist.labels;
    const double* xyz = nl.Field().Coordinates().HostRead();
    const double* t = dist.targets->Targets().HostRead();
    const int* surf_off = nl.SurfaceOffsets().HostRead();
    const int* planar = nl.SurfacePlanar().HostRead();
    int bad = 0;
    int moved_on_dome = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (nl.NumLabels(v) != 2 || nl.BoundarySurfaces(v) != 1 || planar[surf_off[v]] != 0) {
            continue;
        }
        const double* x = xyz + 3 * v;
        const double* y = t + 3 * v;
        bad += (dist.targets->Status(v) != ms::TargetStatus::FIT) ? 1 : 0;
        bad += (std::fabs(n[0] * y[0] + n[1] * y[1] + n[2] * y[2] - c) > 1e-13) ? 1 : 0;
        bad += on_dome(y) ? 0 : 1;
        moved_on_dome += (y[0] != x[0] || y[1] != x[1] || y[2] != x[2]) ? 1 : 0;
    }
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GT(GlobalSum(moved_on_dome), 0);
    // Where the bisector meets the dome's edges with the side faces (a feature edge of the curved
    // surface) the nodes slide along the edge: on the bisector, on the dome's triangles, and on
    // the side face with its coordinate kept bit for bit
    int edge_nodes = 0;
    bad = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (nl.NumLabels(v) != 2 || nl.BoundarySurfaces(v) != 2) {
            continue;
        }
        int flat = -1;
        bool curved = false;
        for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
            curved = curved || planar[s] == 0;
            flat = (planar[s] != 0) ? s : flat;
        }
        if (!curved || flat < 0) {
            continue;
        }
        ++edge_nodes;
        const double* x = xyz + 3 * v;
        const double* y = t + 3 * v;
        const double* fn = nl.SurfaceNormals().HostRead() + 3 * flat;
        const int axis = (fn[0] != 0.0) ? 0 : 1;
        bad += (dist.targets->Status(v) != ms::TargetStatus::FIT) ? 1 : 0;
        bad += (std::fabs(n[0] * y[0] + n[1] * y[1] + n[2] * y[2] - c) > 1e-13) ? 1 : 0;
        bad += on_dome(y) ? 0 : 1;
        bad += (y[axis] != x[axis]) ? 1 : 0;
    }
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GT(GlobalSum(edge_nodes), 0);
    EXPECT_EQ(dist.targets->Summary().reason[static_cast<int>(ms::TargetReason::CURVED_FEATURE)],
              0);
    EXPECT_EQ(dist.targets->Summary().status[FAKE], 0);
}

// The domed tet box again, with the bisector x = 0.47 and max_move = 2: dome nodes 1.2 to 1.8
// element sizes from the bisector reach it only through a patch of two face rings (the
// one-ring ends one element away and its closest point would stick to the patch border)
TEST(NodeTargets, FarTargetOnCurvedSurface) {
    const ms::SeedSet seeds = MakeSeeds({{0.27, 0.5, 0.5}, {0.67, 0.5, 0.5}});
    const auto labels_of = [&](mfem::Mesh& m) {
        return SeedLabels(m, ms::MetricType::VORONOI, seeds);
    };
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    serial.Transform([](const mfem::Vector& x, mfem::Vector& y) {
        y = x;
        y(2) = x(2) * (1.0 + 0.3 * x(0) * (1.0 - x(0)) + 0.3 * x(1) * (1.0 - x(1)));
    });
    ms::NodeTargetOptions options;
    options.max_move = 2.0;
    Fitted f = Fit(
        serial, MPI_COMM_WORLD, labels_of, ms::MetricType::VORONOI, seeds, {0, 1}, {}, options);
    EXPECT_EQ(f.targets->PatchRings(), 2);
    const std::vector<std::array<double, 9>> dome = TopTriangles(serial);
    const ms::NodeLabels& nl = *f.labels;
    const auto one_ring = ms::MakeCurvedPatches(nl, 1);
    const auto two_ring = ms::MakeCurvedPatches(nl, 2);
    const double* xyz = nl.Field().Coordinates().HostRead();
    const int* surf_off = nl.SurfaceOffsets().HostRead();
    const int* planar = nl.SurfacePlanar().HostRead();
    std::vector<ms::NodeLabelQuery> queries;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        const double* x = xyz + 3 * v;
        const double gap = std::fabs(x[0] - 0.47) / f.targets->MeshSize(v);
        if (nl.BoundarySurfaces(v) == 1 && planar[surf_off[v]] == 0 && gap > 1.2 && gap < 1.8) {
            ms::NodeLabelQuery q;
            q.node = v;
            q.num_labels = 2;
            q.labels[0] = 1;
            q.labels[1] = 2;
            queries.push_back(q);
        }
    }
    const std::vector<ms::NodeTargetResult> results = f.targets->Evaluate(queries);
    int bad = 0;
    int stuck = 0;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const int v = queries[i].node;
        const double* y = results[i].target;
        bad += (results[i].status != ms::TargetStatus::FIT) ? 1 : 0;
        bad += (std::fabs(y[0] - 0.47) > 1e-12) ? 1 : 0;
        bad += OnTriangles(dome, y) ? 0 : 1;
        // The target lies on the two-ring patch but off the one-ring patch
        double p[3];
        double pn[3];
        bool interior = false;
        bad += (two_ring->ClosestPoint(v, y, p, pn, interior) > 1e-12) ? 1 : 0;
        stuck += (one_ring->ClosestPoint(v, y, p, pn, interior) > 1e-3) ? 1 : 0;
        bad += (two_ring->NumTriangles(v) <= one_ring->NumTriangles(v)) ? 1 : 0;
    }
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GT(GlobalSum(static_cast<int>(queries.size())), 0);
    EXPECT_EQ(GlobalSum(stuck), GlobalSum(static_cast<int>(queries.size())));
    // With NODE_TARGETS_DUMP_PATCHES = <file>, the patches, node, target and the one-ring
    // closest point of the first query (for plots; run at np = 1)
    const char* dump = std::getenv("NODE_TARGETS_DUMP_PATCHES");
    if (dump != nullptr && !queries.empty()) {
        const int v = queries[0].node;
        FILE* out = std::fopen(dump, "w");
        for (int k = 1; k <= 2; ++k) {
            const auto& patch = (k == 1) ? *one_ring : *two_ring;
            for (int tri = 0; tri < patch.NumTriangles(v); ++tri) {
                const double* q = patch.Triangle(v, tri);
                std::fprintf(out, "T%d", k);
                for (int j = 0; j < 9; ++j) {
                    std::fprintf(out, " %.10g", q[j]);
                }
                std::fprintf(out, "\n");
            }
        }
        double p[3];
        double pn[3];
        bool interior = false;
        one_ring->ClosestPoint(v, results[0].target, p, pn, interior);
        const double* x = xyz + 3 * v;
        const double* y = results[0].target;
        std::fprintf(out,
                     "X %.10g %.10g %.10g\nY %.10g %.10g %.10g\nP %.10g %.10g %.10g\n",
                     x[0],
                     x[1],
                     x[2],
                     y[0],
                     y[1],
                     y[2],
                     p[0],
                     p[1],
                     p[2]);
        std::fclose(out);
    }
    // Distributed patches give the MPI_COMM_SELF targets, node by node
    Fitted self = Fit(
        serial, MPI_COMM_SELF, labels_of, ms::MetricType::VORONOI, seeds, {0, 1}, {}, options);
    EXPECT_EQ(CompareWithSerial(f, self), 0);
}

// The inclined Voronoi bicrystal on a finer mesh, treated as if its boundary had no analytic
// description (no seeds): every boundary node gets the smoothed staircase, which must lie
// much closer to the true bisector than the staircase nodes, keep the box face coordinates
// bit for bit and be the same bits for every partition.
TEST(NodeTargets, SmoothedStaircase) {
    const ms::SeedSet seeds = MakeSeeds({{0.3, 0.4, 0.5}, {0.7, 0.55, 0.45}});
    const auto labels_of = [&](mfem::Mesh& m) {
        return SeedLabels(m, ms::MetricType::VORONOI, seeds);
    };
    ms::NodeTargetOptions options;
    options.analytic = false;
    options.smoothed = true;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(12, 12, 12, mfem::Element::HEXAHEDRON);
    const ms::SeedSet none;
    Fitted dist = Fit(
        serial, MPI_COMM_WORLD, labels_of, ms::MetricType::VORONOI, none, {-1, -1}, {}, options);
    Fitted self = Fit(
        serial, MPI_COMM_SELF, labels_of, ms::MetricType::VORONOI, none, {-1, -1}, {}, options);
    EXPECT_EQ(CompareWithSerial(dist, self), 0);

    const double n[3] = {0.4, 0.15, -0.05};
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const double c = 0.5 * ((0.49 + 0.3025 + 0.2025) - (0.09 + 0.16 + 0.25));
    const auto distance = [&](const double* y) {
        return std::fabs(n[0] * y[0] + n[1] * y[1] + n[2] * y[2] - c) / len;
    };
    const ms::NodeLabels& nl = *dist.labels;
    const double* xyz = nl.Field().Coordinates().HostRead();
    const double* t = dist.targets->Targets().HostRead();
    const int* owned = nl.Field().OwnedVertexMarker().HostRead();
    double sums[3] = {0.0, 0.0, 0.0}; // staircase distance, target distance, count
    int bad = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (nl.NumLabels(v) != 2 || nl.OverConstrained(v)) {
            continue;
        }
        const double* x = xyz + 3 * v;
        const double* y = t + 3 * v;
        bad += (dist.targets->Status(v) != ms::TargetStatus::FIT || !dist.targets->Smoothed(v)) ? 1
                                                                                                : 0;
        for (int d = 0; d < 3; ++d) {
            bad += (OnBoxFace(x[d]) && y[d] != x[d]) ? 1 : 0;
        }
        const double move = std::sqrt((y[0] - x[0]) * (y[0] - x[0]) +
                                      (y[1] - x[1]) * (y[1] - x[1]) +
                                      (y[2] - x[2]) * (y[2] - x[2]));
        bad += (move > 2.0 * dist.targets->MeshSize(v) * (1.0 + 1e-12)) ? 1 : 0;
        if (owned[v] != 0) {
            sums[0] += distance(x);
            sums[1] += distance(y);
            sums[2] += 1.0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GT(sums[2], 0.0);
    EXPECT_LT(sums[1], 0.5 * sums[0]);
    EXPECT_EQ(dist.targets->Summary().smoothed, dist.targets->Summary().status[FIT]);
    if (mfem::Mpi::Root()) {
        std::printf("smoothed staircase: mean distance to the bisector %.4f -> %.4f (h = %.4f)\n",
                    sums[0] / sums[2],
                    sums[1] / sums[2],
                    1.0 / 12.0);
    }

    // Order-2 Nodes (smoothing over the lattice of the nodal points): same checks, coarser mesh
    mfem::Mesh coarse = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    coarse.SetCurvature(2);
    Fitted dist2 = Fit(
        coarse, MPI_COMM_WORLD, labels_of, ms::MetricType::VORONOI, none, {-1, -1}, {}, options);
    Fitted self2 = Fit(
        coarse, MPI_COMM_SELF, labels_of, ms::MetricType::VORONOI, none, {-1, -1}, {}, options);
    EXPECT_EQ(CompareWithSerial(dist2, self2), 0);
    const ms::NodeLabels& nl2 = *dist2.labels;
    const double* xyz2 = nl2.Field().Coordinates().HostRead();
    const double* t2 = dist2.targets->Targets().HostRead();
    const int* owned2 = nl2.Field().OwnedVertexMarker().HostRead();
    double sums2[3] = {0.0, 0.0, 0.0};
    for (int v = 0; v < nl2.NumNodes(); ++v) {
        if (nl2.NumLabels(v) == 2 && !nl2.OverConstrained(v) && owned2[v] != 0) {
            sums2[0] += distance(xyz2 + 3 * v);
            sums2[1] += distance(t2 + 3 * v);
            sums2[2] += 1.0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, sums2, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_GT(sums2[2], 0.0);
    // The order-2 staircase nodes start closer (half of them are edge and face midpoints)
    EXPECT_LT(sums2[1], 0.6 * sums2[0]);
}

// Grain 1 -> s0, grain 2 -> s2: their bisector x = 1/2 is where the labels meet, but every
// point of it is closer to the unused seed s1 at the center line
TEST(NodeTargets, Validation) {
    const ms::SeedSet seeds = MakeSeeds({{0.0, 0.5, 0.5}, {0.5, 0.5, 0.5}, {1.0, 0.5, 0.5}});
    const auto halves = [](mfem::Mesh& m) {
        mfem::Array<int> labels(m.GetNE());
        mfem::Vector c(3);
        for (int e = 0; e < m.GetNE(); ++e) {
            m.GetElementCenter(e, c);
            labels[e] = c(0) < 0.5 ? 1 : 2;
        }
        return labels;
    };
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
    const std::int64_t face_nodes = 25;
    {
        Fitted f = Fit(serial, MPI_COMM_WORLD, halves, ms::MetricType::VORONOI, seeds, {0, 2});
        const auto& s = f.targets->Summary();
        EXPECT_EQ(s.status[FAKE], face_nodes);
        EXPECT_EQ(s.reason[static_cast<int>(ms::TargetReason::FOREIGN_SEED)], face_nodes);
        EXPECT_EQ(s.status[FIT], 0);
    }
    {
        Fitted f = Fit(serial, MPI_COMM_WORLD, halves, ms::MetricType::VORONOI, seeds, {0, 0});
        EXPECT_EQ(f.targets->Summary().reason[static_cast<int>(ms::TargetReason::SAME_SEED)],
                  face_nodes);
    }
    {
        Fitted f = Fit(serial, MPI_COMM_WORLD, halves, ms::MetricType::VORONOI, seeds, {0, -1});
        EXPECT_EQ(f.targets->Summary().status[HELD], face_nodes);
        EXPECT_EQ(f.targets->Summary().reason[static_cast<int>(ms::TargetReason::NON_ANALYTIC)],
                  face_nodes);
        // Targets of nodes that are not FIT stay at x
        const double* xyz = f.labels->Field().Coordinates().HostRead();
        const double* t = f.targets->Targets().HostRead();
        int moved = 0;
        for (int k = 0; k < 3 * f.labels->NumNodes(); ++k) {
            moved += (t[k] != xyz[k]) ? 1 : 0;
        }
        EXPECT_EQ(GlobalSum(moved), 0);
    }
    {
        // Roof-shaped top face, crease along x = 1/2 on the grain boundary: the 5 crease nodes
        // see one curved top surface (30-degree feature angle). The 3 inside the top face are
        // already on the grain boundary and the surface (FIT, no move); so are the 2 on the side
        // faces, on a feature edge of the curved surface (FIT along the edge, no move)
        mfem::Mesh roof = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
        roof.Transform([](const mfem::Vector& x, mfem::Vector& y) {
            y = x;
            y(2) = x(2) * (1.0 + 0.18 * std::fabs(x(0) - 0.5));
        });
        const ms::SeedSet two = MakeSeeds({{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}});
        Fitted f = Fit(roof, MPI_COMM_WORLD, halves, ms::MetricType::VORONOI, two, {0, 1});
        EXPECT_EQ(f.targets->Summary().reason[static_cast<int>(ms::TargetReason::CURVED_FEATURE)],
                  0);
        EXPECT_EQ(f.targets->Summary().status[FIT], face_nodes);
        EXPECT_EQ(f.targets->Summary().max_move_ratio, 0.0);
    }
}

/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
