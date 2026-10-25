/**
 * @file test_conformal_fit.cpp
 * @brief Tests of conformal/conformal_fitter and conformal/quality (Stage 2d): the mesh nodes
 * moved toward the analytic grain boundaries by a spring-balanced pseudo-elastic morph, with a
 * fixed quality floor.
 *
 * What is checked, and why:
 *  - Quality of simple elements: a unit cube and a regular tet score 1, an inverted hex is counted.
 *  - Inclined bicrystal (Voronoi bisector x + 0.2 y + 0.1 z = 0.65 through the unit box), hex and
 *    tet meshes, after label repair: the quality floor holds, no element is inverted, nodes on the
 *    box faces keep their face coordinate bit for bit, the grain boundary nodes end much closer to
 *    the analytic plane than the staircase (hex: below 0.4 of its deviation, the aspect ratio
 *    ceiling holds the shallow 1:5 steps back; tet: below three quarters, the bulk quality guard
 *    stops the Kuhn tets earlier), no element above the aspect ratio ceiling, and the volume of
 *    grain 1 is close to the analytic 1/2.
 *  - A 20-grain Voronoi polycrystal (hex 16^3): the floor holds, almost no element is below 0.5,
 *    and the mean deviation drops well below the staircase's.
 *  - Order-2 Nodes (hex 6^3, 8 grains): every nodal point is fitted (more than twice the fitted
 *    nodes of order 1), the limits hold, no element folds on a denser Gauss-Lobatto grid, box faces
 *    are kept, and WORLD agrees with SELF to the solver tolerance.
 *  - Curved and tilted outer boundary (box with a domed top, faces x = 0, 1 tilted; hex and tet,
 *    12 grains): nodes on those surfaces slide along them and end on them (the planes, or the
 *    input boundary faces around the node, to rounding), some of them move, the floor holds and
 *    the deviation drops.
 *  - Rank count: the fitted positions on MPI_COMM_WORLD (round-robin partition) agree with those on
 *    MPI_COMM_SELF to the linear solver tolerance.
 *  - With CONFORMAL_DUMP_DIR set, ConformalFit.DumpSlices writes the elements of a polycrystal
 *    before and after fitting (label, quality, original center, corners) for plotting
 *    (CONFORMAL_DUMP_N, _GRAINS, _SKIP, _TET, _CURVED, _SMOOTHED select the case; _SMOOTHED fits
 *    to the smoothed staircase instead of the analytic boundaries); skipped otherwise.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows.
 */
#include "microstructure/conformal/boundary_projector.hpp"
#include "microstructure/conformal/conformal_fitter.hpp"
#include "microstructure/conformal/label_repair.hpp"
#include "microstructure/conformal/quality.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

using Key = std::array<long long, 3>;

Key MakeKey(const double* x) {
    return {std::llround(x[0] * 1e9), std::llround(x[1] * 1e9), std::llround(x[2] * 1e9)};
}

/** Two seeds mirrored about the plane x + 0.2 y + 0.1 z = 0.65 through (1/2, 1/2, 1/2). */
ms::SeedSet BisectorSeeds() {
    const double n[3] = {1.0, 0.2, 0.1};
    const double len = std::sqrt(1.0 + 0.04 + 0.01);
    ms::SeedSet s;
    for (const double side : {-1.0, 1.0}) {
        s.x.push_back(0.5 + side * 0.25 * n[0] / len);
        s.y.push_back(0.5 + side * 0.25 * n[1] / len);
        s.z.push_back(0.5 + side * 0.25 * n[2] / len);
    }
    return s;
}

/** n seeds from a fixed pseudo-random sequence in [0.05, 0.95]^3, after `skip` seeds. */
ms::SeedSet ScatteredSeeds(int n, int skip = 0) {
    double r = 0.1234;
    const auto next = [&r]() {
        r = std::fmod(r * 9301.0 + 0.49297, 1.0);
        return 0.05 + 0.9 * r;
    };
    for (int i = 0; i < 3 * skip; ++i) {
        next();
    }
    ms::SeedSet s;
    for (int i = 0; i < n; ++i) {
        s.x.push_back(next());
        s.y.push_back(next());
        s.z.push_back(next());
    }
    return s;
}

mfem::Array<int> VoronoiLabels(mfem::Mesh& mesh, const ms::SeedSet& s) {
    mfem::Array<int> labels(mesh.GetNE());
    mfem::Vector c(3);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.GetElementCenter(e, c);
        double best = HUGE_VAL;
        for (int i = 0; i < static_cast<int>(s.Size()); ++i) {
            const double d = std::pow(c(0) - s.x[i], 2) + std::pow(c(1) - s.y[i], 2) +
                             std::pow(c(2) - s.z[i], 2);
            if (d < best) {
                best = d;
                labels[e] = i + 1;
            }
        }
    }
    return labels;
}

/** A fitted mesh with what produced it. */
struct Fitted {
    std::unique_ptr<mfem::ParMesh> pmesh;
    mfem::Array<int> labels;
    ms::LabelRepairResult repair;
    ms::ConformalFitReport report;
    mfem::Vector x0; ///< node positions before fitting (Nodes layout)
};

void WriteElements(mfem::ParMesh& pmesh, const mfem::Array<int>& labels, const mfem::Vector& x0,
                   const std::string& path);
void WriteSections(mfem::ParMesh& pmesh, const mfem::Array<int>& labels, const mfem::Vector& x0,
                   const std::string& path);

/**
 * Distribute round robin on `comm`, label (Voronoi of `seeds`), repair, fit. With `dump_before`
 * set, the repaired labels on the unfitted mesh are written there first (DumpSlices).
 */
Fitted Fit(mfem::Mesh& serial, MPI_Comm comm, const ms::SeedSet& seeds,
           const std::string& dump_before = "", bool smoothed_only = false, int order = 1) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> part(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        part[static_cast<std::size_t>(e)] = e % size;
    }
    Fitted f;
    f.pmesh = std::make_unique<mfem::ParMesh>(comm, serial, part.data());
    f.pmesh->SetCurvature(order, false, 3, mfem::Ordering::byNODES);
    f.x0 = *f.pmesh->GetNodes();
    f.labels = VoronoiLabels(*f.pmesh, seeds);
    std::vector<int> grain_seed(seeds.Size());
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        grain_seed[i] = static_cast<int>(i);
    }
    ms::LabelRepairOptions repair_options;
    if (smoothed_only) { // the staircase as if it had no analytic description
        repair_options.targets.analytic = false;
        repair_options.targets.smoothed = true;
    }
    f.repair = ms::RepairLabels(*f.pmesh, f.labels, ms::MetricType::VORONOI, seeds, {},
                                grain_seed, repair_options);
    if (!dump_before.empty()) {
        WriteElements(*f.pmesh, f.labels, f.x0, dump_before);
    }
    f.report = ms::FitConformal(*f.pmesh, *f.repair.targets);
    return f;
}

double GrainVolume(mfem::ParMesh& pmesh, const mfem::Array<int>& labels, int grain) {
    double v = 0.0;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        v += (labels[e] == grain) ? pmesh.GetElementVolume(e) : 0.0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    return v;
}

int GlobalSum(int local) {
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return local;
}

/** Share of elements below a scaled Jacobian of `limit` (global). */
double ShareBelow(mfem::ParMesh& pmesh, double limit) {
    double c[2] = {0.0, static_cast<double>(pmesh.GetNE())};
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        double sj, ar, det;
        ms::ElementQuality(pmesh, e, sj, ar, det);
        c[0] += (sj < limit) ? 1.0 : 0.0;
    }
    MPI_Allreduce(MPI_IN_PLACE, c, 2, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    return c[0] / c[1];
}

/**
 * Nodes that left a box face they were on (face coordinate not kept bit for bit). With `sheared`,
 * only the faces that stay axis planes under DomeAndShear (y = 0, 1 and z = 0) are checked.
 */
int FaceViolations(const Fitted& f, bool sheared = false) {
    const mfem::GridFunction& x = *f.pmesh->GetNodes();
    const mfem::FiniteElementSpace& fes = *x.FESpace();
    int bad = 0;
    for (int v = 0; v < fes.GetNDofs(); ++v) {
        for (int d = 0; d < 3; ++d) {
            const double before = f.x0(fes.DofToVDof(v, d));
            const bool face = sheared ? (d == 1 && (before == 0.0 || before == 1.0)) ||
                                            (d == 2 && before == 0.0)
                                      : before == 0.0 || before == 1.0;
            if (face) {
                bad += (x(fes.DofToVDof(v, d)) != before) ? 1 : 0;
            }
        }
    }
    return GlobalSum(bad);
}

/** Largest position difference between the WORLD and SELF fits, matched by input position. */
double WorstRankDifference(const Fitted& dist, const Fitted& self) {
    std::map<Key, std::array<double, 3>> moved;
    const mfem::GridFunction& xs = *self.pmesh->GetNodes();
    const mfem::FiniteElementSpace& fs = *xs.FESpace();
    for (int v = 0; v < fs.GetNDofs(); ++v) {
        const double p0[3] = {self.x0(fs.DofToVDof(v, 0)), self.x0(fs.DofToVDof(v, 1)),
                              self.x0(fs.DofToVDof(v, 2))};
        moved[MakeKey(p0)] = {xs(fs.DofToVDof(v, 0)), xs(fs.DofToVDof(v, 1)),
                              xs(fs.DofToVDof(v, 2))};
    }
    const mfem::GridFunction& xd = *dist.pmesh->GetNodes();
    const mfem::FiniteElementSpace& fd = *xd.FESpace();
    double worst = 0.0;
    for (int v = 0; v < fd.GetNDofs(); ++v) {
        const double p0[3] = {dist.x0(fd.DofToVDof(v, 0)), dist.x0(fd.DofToVDof(v, 1)),
                              dist.x0(fd.DofToVDof(v, 2))};
        const auto& q = moved.at(MakeKey(p0));
        for (int d = 0; d < 3; ++d) {
            worst = std::max(worst,
                             std::fabs(xd(fd.DofToVDof(v, d)) - q[static_cast<std::size_t>(d)]));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return worst;
}

/**
 * `fit_ratio`: the mean deviation must end below this fraction of the staircase's (hexes fit
 * closely; Kuhn tets start at a scaled Jacobian of 0.58, so the bulk quality guard stops them
 * earlier).
 */
void CheckBisector(mfem::Mesh& serial, double floor, double fit_ratio) {
    const ms::SeedSet seeds = BisectorSeeds();
    Fitted f = Fit(serial, MPI_COMM_WORLD, seeds);
    const ms::ConformalFitReport& r = f.report;
    EXPECT_GT(r.fit_nodes, 0);
    EXPECT_GE(r.rounds, 1);
    EXPECT_EQ(r.quality_after.inverted, 0);
    EXPECT_GE(r.quality_after.min_scaled_jacobian, floor);
    EXPECT_LE(r.quality_after.max_aspect_ratio, ms::MESH_ASPECT_RATIO_MAX);
    EXPECT_EQ(FaceViolations(f), 0);
    EXPECT_LT(r.mean_deviation, fit_ratio * r.initial_deviation);
    EXPECT_NEAR(GrainVolume(*f.pmesh, f.labels, 1), 0.5, 0.01);
    Fitted self = Fit(serial, MPI_COMM_SELF, seeds);
    EXPECT_LE(WorstRankDifference(f, self), 1e-6);
    if (mfem::Mpi::Root()) {
        std::printf("bisector: deviation %.3f -> %.3f h (max %.3f), sj %.3f, rounds %d, "
                    "solves %d\n",
                    r.initial_deviation, r.mean_deviation, r.max_deviation,
                    r.quality_after.min_scaled_jacobian, r.rounds, r.solves);
    }
}

/**
 * Largest distance of a sliding node from its surfaces, in units of h: a node on one curved
 * surface from the input boundary around it (its one-ring triangle patch on an unfitted copy of the
 * mesh, same partition); a node on one or two flat surfaces, at least one tilted, from each of
 * their planes. `count` gets the number of such nodes that moved.
 */
double SurfaceDrift(mfem::Mesh& serial, const Fitted& f, int& count) {
    int size = 1;
    MPI_Comm_size(f.pmesh->GetComm(), &size);
    std::vector<int> part(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        part[static_cast<std::size_t>(e)] = e % size;
    }
    mfem::ParMesh ref(f.pmesh->GetComm(), serial, part.data());
    ref.SetCurvature(1, false, 3, mfem::Ordering::byNODES);
    const ms::VertexField field(ref, ms::FieldNodes::MESH_NODES);
    const ms::SurfacePatches patches(
        field, std::vector<char>(static_cast<std::size_t>(field.NumVertices()), 1));
    const ms::NodeLabels& nl = f.repair.targets->Labels();
    const int* surf_off = nl.SurfaceOffsets().HostRead();
    const double* normal = nl.SurfaceNormals().HostRead();
    const int* planar = nl.SurfacePlanar().HostRead();
    const mfem::GridFunction& x = *f.pmesh->GetNodes();
    const mfem::FiniteElementSpace& fes = *x.FESpace();
    double worst = 0.0;
    int moved = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        bool tilted = false;
        bool curved = false;
        for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
            const double* n = normal + 3 * s;
            const int zeros = (n[0] == 0.0) + (n[1] == 0.0) + (n[2] == 0.0);
            curved = curved || planar[s] == 0;
            tilted = tilted || (planar[s] != 0 && zeros != 2);
        }
        const int num = nl.BoundarySurfaces(v);
        if (!(curved && num == 1) && !(tilted && !curved && num <= 2)) {
            continue;
        }
        double y[3];
        double y0[3];
        bool same = true;
        for (int d = 0; d < 3; ++d) {
            y[d] = x(fes.DofToVDof(v, d));
            y0[d] = f.x0(fes.DofToVDof(v, d));
            same = same && y[d] == y0[d];
        }
        double dist = 0.0;
        if (curved) {
            double p[3], pn[3];
            bool interior = false;
            dist = patches.ClosestPoint(v, y, p, pn, interior);
        } else {
            for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
                const double* n = normal + 3 * s;
                const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                dist = std::max(dist, std::fabs(n[0] * (y[0] - y0[0]) + n[1] * (y[1] - y0[1]) +
                                                n[2] * (y[2] - y0[2])) /
                                          len);
            }
        }
        worst = std::max(worst, dist / f.repair.targets->MeshSize(v));
        moved += (same || field.OwnedVertexMarker().HostRead()[v] == 0) ? 0 : 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    count = GlobalSum(moved);
    return worst;
}

/**
 * Unit box with a domed top z = 1 + 0.3 x (1 - x) + 0.3 y (1 - y) and the faces x = 0, 1 tilted
 * (x shifted by 0.2 z): one curved surface and two tilted planes besides the axis planes.
 */
void DomeAndShear(mfem::Mesh& serial, double shear) {
    serial.Transform([shear](const mfem::Vector& p, mfem::Vector& y) {
        y = p;
        y(2) = p(2) * (1.0 + 0.3 * p(0) * (1.0 - p(0)) + 0.3 * p(1) * (1.0 - p(1)));
        y(0) = p(0) + shear * p(2);
    });
}

void CheckCurved(mfem::Element::Type type, int n, double floor, double shear) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(n, n, n, type);
    DomeAndShear(serial, shear);
    ms::SeedSet seeds = ScatteredSeeds(12);
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        seeds.x[i] += shear * seeds.z[i];
        seeds.z[i] *= 1.1;
    }
    Fitted f = Fit(serial, MPI_COMM_WORLD, seeds);
    const ms::ConformalFitReport& r = f.report;
    int moved = 0;
    const double drift = SurfaceDrift(serial, f, moved);
    EXPECT_EQ(r.quality_after.inverted, 0);
    EXPECT_GE(r.quality_after.min_scaled_jacobian, floor);
    EXPECT_EQ(FaceViolations(f, true), 0);
    EXPECT_LT(r.mean_deviation, r.initial_deviation);
    EXPECT_LE(drift, 1e-9);
    EXPECT_GT(moved, 0);
    if (mfem::Mpi::Root()) {
        std::printf("curved %s: deviation %.3f -> %.3f h (max %.3f), sj %.3f -> %.3f, rounds %d, "
                    "surface drift %.2e h, moved on surface %d\n",
                    type == mfem::Element::TETRAHEDRON ? "tet" : "hex", r.initial_deviation,
                    r.mean_deviation, r.max_deviation, r.quality_before.min_scaled_jacobian,
                    r.quality_after.min_scaled_jacobian, r.rounds, drift, moved);
    }
}

} // namespace

TEST(MeshQuality, SimpleElements) {
    mfem::Mesh cube = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
    double sj, ar, det;
    ms::ElementQuality(cube, 0, sj, ar, det);
    EXPECT_NEAR(sj, 1.0, 1e-14);
    EXPECT_NEAR(ar, 1.0, 1e-14);
    mfem::Mesh tet(3, 4, 1, 0, 3);
    tet.AddVertex(0.0, 0.0, 0.0);
    tet.AddVertex(1.0, 0.0, 0.0);
    tet.AddVertex(0.5, std::sqrt(3.0) / 2.0, 0.0);
    tet.AddVertex(0.5, std::sqrt(3.0) / 6.0, std::sqrt(2.0 / 3.0));
    const int tv[4] = {0, 1, 2, 3};
    tet.AddTet(tv);
    tet.FinalizeTetMesh();
    ms::ElementQuality(tet, 0, sj, ar, det);
    EXPECT_NEAR(std::fabs(sj), 1.0, 1e-12);
    EXPECT_NEAR(ar, 1.0, 1e-12); // aspect ratio relative to the regular tet
    mfem::ParMesh pcube(MPI_COMM_SELF, cube);
    pcube.SetCurvature(1, false, 3, mfem::Ordering::byNODES);
    mfem::GridFunction& x = *pcube.GetNodes();
    x(x.FESpace()->DofToVDof(0, 0)) = 1.5; // vertex (0,0,0) moved past x = 1
    const ms::MeshQuality q = ms::ComputeMeshQuality(pcube);
    EXPECT_EQ(q.inverted, 1);
    EXPECT_LT(q.min_scaled_jacobian, 0.0);
}

TEST(ConformalFit, HexInclinedBicrystal) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(8, 8, 8, mfem::Element::HEXAHEDRON);
    CheckBisector(serial, ms::MESH_QUALITY_FLOOR_HEX, 0.4);
}

TEST(ConformalFit, TetInclinedBicrystal) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    CheckBisector(serial, ms::MESH_QUALITY_FLOOR_TET, 0.75);
}

TEST(ConformalFit, HexPolycrystal) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(16, 16, 16, mfem::Element::HEXAHEDRON);
    Fitted f = Fit(serial, MPI_COMM_WORLD, ScatteredSeeds(20));
    const ms::ConformalFitReport& r = f.report;
    EXPECT_EQ(r.quality_after.inverted, 0);
    EXPECT_GE(r.quality_after.min_scaled_jacobian, ms::MESH_QUALITY_FLOOR_HEX);
    EXPECT_LE(r.quality_after.max_aspect_ratio, ms::MESH_ASPECT_RATIO_MAX);
    const double below = ShareBelow(*f.pmesh, 0.5); // collective: outside the Root() block
    EXPECT_LT(below, 0.02);
    EXPECT_LT(r.mean_deviation, 0.75 * r.initial_deviation);
    EXPECT_EQ(FaceViolations(f), 0);
    if (mfem::Mpi::Root()) {
        std::printf("polycrystal: deviation %.3f -> %.3f h (max %.3f), sj %.3f, <0.5 %.4f, rounds "
                    "%d, solves %d\n",
                    r.initial_deviation, r.mean_deviation, r.max_deviation,
                    r.quality_after.min_scaled_jacobian, below, r.rounds,
                    r.solves);
    }
}

TEST(ConformalFit, HexOrderTwoPolycrystal) {
    // Order-2 Nodes: every nodal point is fitted (curved grain boundaries). The quality limits
    // hold at the nodal points; an independent check on a denser Gauss-Lobatto grid (5 points
    // per direction) finds no folded element either.
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    const ms::SeedSet seeds = ScatteredSeeds(8);
    Fitted f = Fit(serial, MPI_COMM_WORLD, seeds, "", false, 2);
    const ms::ConformalFitReport& r = f.report;
    EXPECT_EQ(r.quality_after.inverted, 0);
    EXPECT_GE(r.quality_after.min_scaled_jacobian, ms::MESH_QUALITY_FLOOR_HEX);
    EXPECT_LE(r.quality_after.max_aspect_ratio, ms::MESH_ASPECT_RATIO_MAX);
    EXPECT_LT(r.mean_deviation, 0.85 * r.initial_deviation);
    EXPECT_EQ(FaceViolations(f), 0);
    mfem::IntegrationRules lobatto(0, mfem::Quadrature1D::GaussLobatto);
    const mfem::IntegrationRule& ir = lobatto.Get(mfem::Geometry::CUBE, 7);
    int folded = 0;
    for (int e = 0; e < f.pmesh->GetNE(); ++e) {
        mfem::ElementTransformation* T = f.pmesh->GetElementTransformation(e);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            T->SetIntPoint(&ir.IntPoint(q));
            folded += (T->Weight() <= 0.0) ? 1 : 0;
        }
    }
    EXPECT_EQ(GlobalSum(folded), 0);
    // More nodes are fitted than at order 1 (edge and face nodes of the grain boundaries)
    Fitted linear = Fit(serial, MPI_COMM_WORLD, seeds);
    EXPECT_GT(r.fit_nodes, 2 * linear.report.fit_nodes);
    Fitted self = Fit(serial, MPI_COMM_SELF, seeds, "", false, 2);
    EXPECT_LE(WorstRankDifference(f, self), 1e-6);
    if (mfem::Mpi::Root()) {
        std::printf("order 2: deviation %.3f -> %.3f h (max %.3f), sj %.3f, ar %.2f, rounds %d, "
                    "fitted nodes %lld (order 1: %lld)\n",
                    r.initial_deviation, r.mean_deviation, r.max_deviation,
                    r.quality_after.min_scaled_jacobian, r.quality_after.max_aspect_ratio,
                    r.rounds, static_cast<long long>(r.fit_nodes),
                    static_cast<long long>(linear.report.fit_nodes));
    }
}

TEST(ConformalFit, HexCurvedAndTiltedBoundary) {
    CheckCurved(mfem::Element::HEXAHEDRON, 12, ms::MESH_QUALITY_FLOOR_HEX, 0.2);
}

TEST(ConformalFit, TetCurvedAndTiltedBoundary) {
    CheckCurved(mfem::Element::TETRAHEDRON, 8, ms::MESH_QUALITY_FLOOR_TET, 0.2);
}

TEST(ConformalFit, DumpSlices) {
    const char* dir = std::getenv("CONFORMAL_DUMP_DIR");
    if (dir == nullptr) {
        GTEST_SKIP();
    }
    const int n = std::getenv("CONFORMAL_DUMP_N") ? std::atoi(std::getenv("CONFORMAL_DUMP_N")) : 16;
    const int grains =
        std::getenv("CONFORMAL_DUMP_GRAINS") ? std::atoi(std::getenv("CONFORMAL_DUMP_GRAINS")) : 20;
    const int skip =
        std::getenv("CONFORMAL_DUMP_SKIP") ? std::atoi(std::getenv("CONFORMAL_DUMP_SKIP")) : 0;
    const bool tet = std::getenv("CONFORMAL_DUMP_TET") != nullptr;
    const bool curved = std::getenv("CONFORMAL_DUMP_CURVED") != nullptr;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n, n, n, tet ? mfem::Element::TETRAHEDRON : mfem::Element::HEXAHEDRON);
    ms::SeedSet seeds = ScatteredSeeds(grains, skip);
    if (curved) { // the case of CheckCurved
        DomeAndShear(serial, 0.2);
        for (std::size_t i = 0; i < seeds.Size(); ++i) {
            seeds.x[i] += 0.2 * seeds.z[i];
            seeds.z[i] *= 1.1;
        }
    }
    const bool smoothed_only = std::getenv("CONFORMAL_DUMP_SMOOTHED") != nullptr;
    const int order =
        std::getenv("CONFORMAL_DUMP_ORDER") ? std::atoi(std::getenv("CONFORMAL_DUMP_ORDER")) : 1;
    Fitted f = Fit(serial, MPI_COMM_WORLD, seeds, std::string(dir) + "/orig.txt", smoothed_only,
                   order);
    double ar_counts[4] = {0.0, 0.0, 0.0, static_cast<double>(f.pmesh->GetNE())};
    for (int e = 0; e < f.pmesh->GetNE(); ++e) {
        double sj, ar, det;
        ms::ElementQuality(*f.pmesh, e, sj, ar, det);
        ar_counts[0] += (ar > 3.0) ? 1.0 : 0.0;
        ar_counts[1] += (ar > 4.0) ? 1.0 : 0.0;
        ar_counts[2] += (ar > 5.0) ? 1.0 : 0.0;
    }
    MPI_Allreduce(MPI_IN_PLACE, ar_counts, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (mfem::Mpi::Root()) {
        std::printf("DUMP aspect ratio: max %.2f, share > 3: %.4f, > 4: %.4f, > 5: %.4f\n",
                    f.report.quality_after.max_aspect_ratio, ar_counts[0] / ar_counts[3],
                    ar_counts[1] / ar_counts[3], ar_counts[2] / ar_counts[3]);
    }
    if (mfem::Mpi::Root()) {
        const ms::ConformalFitReport& r = f.report;
        std::printf("DUMP %s %d^3 %d grains: deviation %.3f -> %.3f h (max %.3f), sj %.3f, rounds "
                    "%d, solves %d\n",
                    tet ? "tet" : "hex", n, grains, r.initial_deviation, r.mean_deviation,
                    r.max_deviation,
                    r.quality_after.min_scaled_jacobian, r.rounds, r.solves);
    }
    WriteElements(*f.pmesh, f.labels, f.x0, std::string(dir) + "/fit.txt");
    if (!tet) {
        WriteSections(*f.pmesh, f.labels, f.x0, std::string(dir) + "/fit_sections.txt");
    }
}

namespace {

/** One line per local element: label, quality, original center, current corners (.r<rank>). */
void WriteElements(mfem::ParMesh& pmesh, const mfem::Array<int>& labels, const mfem::Vector& x0,
                   const std::string& path) {
    int rank = 0;
    MPI_Comm_rank(pmesh.GetComm(), &rank);
    FILE* out = std::fopen((path + ".r" + std::to_string(rank)).c_str(), "w");
    const mfem::GridFunction& x = *pmesh.GetNodes();
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        double sj, ar, det;
        ms::ElementQuality(pmesh, e, sj, ar, det);
        mfem::Array<int> vd;
        x.FESpace()->GetElementVDofs(e, vd);
        const int m = vd.Size() / 3;
        double c0[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < m; ++j) {
            for (int d = 0; d < 3; ++d) {
                c0[d] += x0(vd[d * m + j]) / m;
            }
        }
        std::fprintf(out, "%d %.6g %.6g %.6g %.6g", labels[e], sj, c0[0], c0[1], c0[2]);
        for (int j = 0; j < m; ++j) {
            std::fprintf(out, " %.8g %.8g %.8g", x(vd[j]), x(vd[m + j]), x(vd[2 * m + j]));
        }
        std::fprintf(out, "\n");
    }
    std::fclose(out);
}

/**
 * One line per local hex: label, quality, original center, then 24 points (x y) of the boundary
 * of the element's mid-section (reference z = 1/2) through the element map, so curved (order-2)
 * faces show as curves (.r<rank>).
 */
void WriteSections(mfem::ParMesh& pmesh, const mfem::Array<int>& labels, const mfem::Vector& x0,
                   const std::string& path) {
    int rank = 0;
    MPI_Comm_rank(pmesh.GetComm(), &rank);
    FILE* out = std::fopen((path + ".r" + std::to_string(rank)).c_str(), "w");
    const mfem::GridFunction& x = *pmesh.GetNodes();
    mfem::Vector X(3);
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        double sj, ar, det;
        ms::ElementQuality(pmesh, e, sj, ar, det);
        mfem::Array<int> vd;
        x.FESpace()->GetElementVDofs(e, vd);
        const int m = vd.Size() / 3;
        double c0[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < m; ++j) {
            for (int d = 0; d < 3; ++d) {
                c0[d] += x0(vd[d * m + j]) / m;
            }
        }
        std::fprintf(out, "%d %.6g %.6g %.6g %.6g", labels[e], sj, c0[0], c0[1], c0[2]);
        mfem::ElementTransformation* T = pmesh.GetElementTransformation(e);
        const double side[4][2][2] = {{{0, 0}, {1, 0}}, {{1, 0}, {1, 1}}, {{1, 1}, {0, 1}},
                                      {{0, 1}, {0, 0}}};
        for (const auto& s : side) {
            for (int k = 0; k < 6; ++k) {
                const double t = k / 6.0;
                mfem::IntegrationPoint ip;
                ip.Set3(s[0][0] + t * (s[1][0] - s[0][0]), s[0][1] + t * (s[1][1] - s[0][1]),
                        0.5);
                T->Transform(ip, X);
                std::fprintf(out, " %.8g %.8g", X(0), X(1));
            }
        }
        std::fprintf(out, "\n");
    }
    std::fclose(out);
}

} // namespace

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    mfem::Hypre::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
