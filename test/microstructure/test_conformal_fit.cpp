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
 *    ceiling holds the shallow 1:5 steps back; tet: below 0.55), the quality limits hold (hex:
 *    scaled Jacobian floor and aspect ratio ceiling; tet: mean ratio floor), and the volume of
 *    grain 1 is close to the analytic 1/2.
 *  - A 20-grain Voronoi polycrystal (hex 16^3): the floor holds, almost no element is below 0.5,
 *    and the mean deviation drops well below the staircase's.
 *  - Thin grains: a grain two elements thick squeezed by two inclined boundaries keeps the limits
 *    and its volume (E1); a grain one element thick is reported as under-resolved and its grain
 *    boundary nodes are held (E5).
 *  - Jittered tets (8^3 x 6 Kuhn mesh with interior vertices moved up to 0.2 element sizes): poor
 *    input tets below the mean ratio floor, the fit still reduces the deviation by a quarter or
 * more and keeps the limits relative to the input.
 *  - Order-2 Nodes (hex 6^3, 8 grains): every nodal point is fitted (more than twice the fitted
 *    nodes of order 1), the limits hold, no element folds on a denser Gauss-Lobatto grid, box faces
 *    are kept, and WORLD agrees with SELF to the solver tolerance.
 *  - 120° tricrystal (V1; hex 12^3, tet 8^3): the triple line nodes end below 0.6 of their
 *    starting distance from the analytic line, all of them FIT, limits kept.
 *  - Curved boundary convergence (V2; Additive bicrystal, hyperboloid boundary, hex 8^3 and
 *    16^3): grain volume and boundary area against an exact reference; the fitted area error is
 *    well below the staircase's and below 30 %, the fitted volume error halves from 8^3 to 16^3
 *    and ends below 2 %. (The fitted area does not converge, see the test.)
 *  - Curved and tilted outer boundary (box with a domed top, faces x = 0, 1 tilted; hex and tet,
 *    12 grains): nodes on those surfaces slide along them and end on them (the planes, or the
 *    input boundary faces around the node, to rounding; nodes on the rims where the dome meets a
 *    side slide along the rim), some of them move, the floor holds and the deviation drops.
 *  - Curved wall rims (box with the side x = 1 bulged, inclined bicrystal crossing the rims where
 *    the wall meets the flat top and bottom): the grain boundary nodes on a rim are FIT, end much
 *    closer to the bisector, keep their z bit for bit and stay on the wall; WORLD agrees with
 *    SELF.
 *  - Rank count: the fitted positions on MPI_COMM_WORLD (round-robin partition) agree with those on
 *    MPI_COMM_SELF to the linear solver tolerance.
 *  - With CONFORMAL_DUMP_DIR set, ConformalFit.DumpSlices writes the elements of a polycrystal
 *    before and after fitting (label, quality, original center, corners) for plotting
 *    (CONFORMAL_DUMP_N, _GRAINS, _SKIP, _TET, _CURVED, _WALL, _SMOOTHED, _MESH (a mesh file)
 *    select the case; _SMOOTHED fits to the smoothed staircase instead of the analytic boundaries);
 * skipped otherwise.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows.
 */
#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include "microstructure/conformal/boundary_projector.hpp"
#include "microstructure/conformal/conformal_fitter.hpp"
#include "microstructure/conformal/label_repair.hpp"
#include "microstructure/conformal/quality.hpp"
#include <gtest/gtest.h>

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

/** Element labels of the Additive (Johnson-Mehl) metric |c - s_i| - w_i at the element centers. */
mfem::Array<int> AdditiveLabels(mfem::Mesh& mesh, const ms::SeedSet& s) {
    mfem::Array<int> labels(mesh.GetNE());
    mfem::Vector c(3);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.GetElementCenter(e, c);
        double best = HUGE_VAL;
        for (int i = 0; i < static_cast<int>(s.Size()); ++i) {
            const double d = std::sqrt(std::pow(c(0) - s.x[i], 2) + std::pow(c(1) - s.y[i], 2) +
                                       std::pow(c(2) - s.z[i], 2)) -
                             s.weight[static_cast<std::size_t>(i)];
            if (d < best) {
                best = d;
                labels[e] = i + 1;
            }
        }
    }
    return labels;
}

/**
 * Area of the faces between elements of grains `a` and `b` (global): local interior faces once,
 * faces shared between ranks at half weight on each side (the neighbor's label from an L2
 * order-0 field exchanged with ExchangeFaceNbrData).
 */
double InterfaceArea(mfem::ParMesh& pmesh, const mfem::Array<int>& labels, int a, int b) {
    mfem::L2_FECollection l2(0, 3);
    mfem::ParFiniteElementSpace l2fes(&pmesh, &l2);
    mfem::ParGridFunction g(&l2fes);
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        g(e) = labels[e];
    }
    g.ExchangeFaceNbrData();
    const auto pair = [&](int la, int lb) {
        return (la == a && lb == b) || (la == b && lb == a);
    };
    const auto face_area = [&](mfem::ElementTransformation& t, mfem::Geometry::Type geom) {
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, 4);
        double area = 0.0;
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            t.SetIntPoint(&ir.IntPoint(q));
            area += ir.IntPoint(q).weight * t.Weight();
        }
        return area;
    };
    double area = 0.0;
    for (int f = 0; f < pmesh.GetNumFaces(); ++f) {
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        if (e2 >= 0 && pair(labels[e1], labels[e2])) {
            area += face_area(*pmesh.GetFaceTransformation(f), pmesh.GetFaceGeometry(f));
        }
    }
    const mfem::Vector& nbr = g.FaceNbrData();
    for (int i = 0; i < pmesh.GetNSharedFaces(); ++i) {
        mfem::FaceElementTransformations* ft = pmesh.GetSharedFaceTransformations(i);
        const int e1 = ft->Elem1No;
        const int k = ft->Elem2No - pmesh.GetNE();
        if (pair(labels[e1], static_cast<int>(nbr(k)))) {
            area += 0.5 * face_area(*ft->Face, pmesh.GetFaceGeometry(pmesh.GetSharedFace(i)));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    return area;
}

/** A fitted mesh with what produced it. */
struct Fitted {
    std::unique_ptr<mfem::ParMesh> pmesh;
    mfem::Array<int> labels;
    ms::LabelRepairResult repair;
    ms::ConformalFitReport report;
    mfem::Vector x0; ///< node positions before fitting (Nodes layout)
};

void WriteElements(mfem::ParMesh& pmesh,
                   const mfem::Array<int>& labels,
                   const mfem::Vector& x0,
                   const std::string& path);
void WriteSections(mfem::ParMesh& pmesh,
                   const mfem::Array<int>& labels,
                   const mfem::Vector& x0,
                   const std::string& path);

/**
 * Distribute round robin on `comm`, label (Voronoi of `seeds`), repair, fit. With `dump_before`
 * set, the repaired labels on the unfitted mesh are written there first (DumpSlices).
 */
Fitted Fit(mfem::Mesh& serial,
           MPI_Comm comm,
           const ms::SeedSet& seeds,
           const std::string& dump_before = "",
           bool smoothed_only = false,
           int order = 1,
           ms::MetricType metric = ms::MetricType::VORONOI) {
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
    f.labels = (metric == ms::MetricType::ADDITIVE) ? AdditiveLabels(*f.pmesh, seeds)
                                                    : VoronoiLabels(*f.pmesh, seeds);
    std::vector<int> grain_seed(seeds.Size());
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        grain_seed[i] = static_cast<int>(i);
    }
    ms::LabelRepairOptions repair_options;
    if (smoothed_only) { // the staircase as if it had no analytic description
        repair_options.targets.analytic = false;
        repair_options.targets.smoothed = true;
    }
    f.repair = ms::RepairLabels(*f.pmesh, f.labels, metric, seeds, {}, grain_seed, repair_options);
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
        const double p0[3] = {
            self.x0(fs.DofToVDof(v, 0)), self.x0(fs.DofToVDof(v, 1)), self.x0(fs.DofToVDof(v, 2))};
        moved[MakeKey(p0)] = {
            xs(fs.DofToVDof(v, 0)), xs(fs.DofToVDof(v, 1)), xs(fs.DofToVDof(v, 2))};
    }
    const mfem::GridFunction& xd = *dist.pmesh->GetNodes();
    const mfem::FiniteElementSpace& fd = *xd.FESpace();
    double worst = 0.0;
    for (int v = 0; v < fd.GetNDofs(); ++v) {
        const double p0[3] = {
            dist.x0(fd.DofToVDof(v, 0)), dist.x0(fd.DofToVDof(v, 1)), dist.x0(fd.DofToVDof(v, 2))};
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
 * The quality limits of the fit (conformal_fitter.hpp): hexes the scaled Jacobian floor and the
 * aspect ratio ceiling, tets the mean ratio floor; an input already beyond a limit (or within 10 %
 * of it) may lose at most 10 % of its value.
 */
void ExpectLimits(const ms::ConformalFitReport& r, bool tet) {
    const ms::MeshQuality& in = r.quality_before;
    const ms::MeshQuality& out = r.quality_after;
    EXPECT_EQ(out.inverted, 0);
    if (tet) {
        EXPECT_GE(out.min_mean_ratio,
                  std::min(ms::MESH_MEAN_RATIO_FLOOR_TET, 0.9 * in.min_mean_ratio) - 1e-12);
    } else {
        EXPECT_GE(out.min_scaled_jacobian,
                  std::min(ms::MESH_QUALITY_FLOOR_HEX, 0.9 * in.min_scaled_jacobian) - 1e-12);
        EXPECT_LE(out.max_aspect_ratio,
                  std::max(ms::MESH_ASPECT_RATIO_MAX, 1.1 * in.max_aspect_ratio) + 1e-12);
    }
}

/**
 * `fit_ratio`: the mean deviation must end below this fraction of the staircase's (the shallow
 * 1:5 steps of this bisector are the worst case for the aspect ratio ceiling of hexes).
 */
void CheckBisector(mfem::Mesh& serial, bool tet, double fit_ratio) {
    const ms::SeedSet seeds = BisectorSeeds();
    Fitted f = Fit(serial, MPI_COMM_WORLD, seeds);
    const ms::ConformalFitReport& r = f.report;
    EXPECT_GT(r.fit_nodes, 0);
    EXPECT_GE(r.rounds, 1);
    ExpectLimits(r, tet);
    EXPECT_EQ(FaceViolations(f), 0);
    EXPECT_LT(r.mean_deviation, fit_ratio * r.initial_deviation);
    EXPECT_NEAR(GrainVolume(*f.pmesh, f.labels, 1), 0.5, 0.01);
    Fitted self = Fit(serial, MPI_COMM_SELF, seeds);
    EXPECT_LE(WorstRankDifference(f, self), 1e-6);
    if (mfem::Mpi::Root()) {
        std::printf("bisector: deviation %.3f -> %.3f h (max %.3f), sj %.3f, rounds %d, "
                    "solves %d\n",
                    r.initial_deviation,
                    r.mean_deviation,
                    r.max_deviation,
                    r.quality_after.min_scaled_jacobian,
                    r.rounds,
                    r.solves);
    }
}

/**
 * V1: tricrystal with 120 degree junctions. Three Voronoi seeds 0.3 from (0.47, 0.53) at 90, 210
 * and 330 degrees, all at z = 0.5: the three bisectors meet at 120 degrees along the vertical line
 * x = 0.47, y = 0.53 (off the lattice). The nodes with three labels (the discrete triple line)
 * must end much closer to it (below 0.6 of their starting distance; with the tenth-strength
 * triple line springs of the first fitter they did not move at all), every node with three
 * labels must be FIT, and the limits hold.
 */
void CheckTricrystal(mfem::Element::Type type, int n) {
    const double cx = 0.47, cy = 0.53, r = 0.3;
    ms::SeedSet seeds;
    for (const double deg : {90.0, 210.0, 330.0}) {
        seeds.x.push_back(cx + r * std::cos(deg * M_PI / 180.0));
        seeds.y.push_back(cy + r * std::sin(deg * M_PI / 180.0));
        seeds.z.push_back(0.5);
    }
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(n, n, n, type);
    Fitted f = Fit(serial, MPI_COMM_WORLD, seeds);
    const bool tet = type == mfem::Element::TETRAHEDRON;
    ExpectLimits(f.report, tet);
    EXPECT_EQ(FaceViolations(f), 0);
    const ms::NodeLabels& nl = f.repair.targets->Labels();
    const int* owned = nl.Field().OwnedVertexMarker().HostRead();
    const mfem::GridFunction& x = *f.pmesh->GetNodes();
    const mfem::FiniteElementSpace& fes = *x.FESpace();
    double sums[3] = {0.0, 0.0, 0.0}; // before, after, count
    double worst = 0.0;
    int bad = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (nl.NumLabels(v) != 3 || owned[v] == 0) {
            continue;
        }
        const double h = f.repair.targets->MeshSize(v);
        const double d0 = std::hypot(f.x0(fes.DofToVDof(v, 0)) - cx,
                                     f.x0(fes.DofToVDof(v, 1)) - cy);
        const double d1 = std::hypot(x(fes.DofToVDof(v, 0)) - cx, x(fes.DofToVDof(v, 1)) - cy);
        sums[0] += d0 / h;
        sums[1] += d1 / h;
        sums[2] += 1.0;
        worst = std::max(worst, d1 / h);
        bad += (f.repair.targets->Status(v) != ms::TargetStatus::FIT) ? 1 : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GE(sums[2], n + 1.0); // the line crosses the box: at least one node per layer
    EXPECT_LT(sums[1], 0.6 * sums[0]);
    EXPECT_LT(f.report.mean_deviation, 0.75 * f.report.initial_deviation);
    if (mfem::Mpi::Root()) {
        std::printf("tricrystal %s %d^3: %d triple line nodes, distance to the line %.3f -> %.3f h "
                    "(max %.3f); deviation %.3f -> %.3f h\n",
                    tet ? "tet" : "hex",
                    n,
                    static_cast<int>(sums[2]),
                    sums[0] / sums[2],
                    sums[1] / sums[2],
                    worst,
                    f.report.initial_deviation,
                    f.report.mean_deviation);
    }
}

/**
 * Largest distance of a sliding node from its surfaces, in units of h: a node on a curved surface
 * (one, or two on a feature edge) from the input boundary around it (its two-ring triangle patch
 * per surface on an unfitted copy of the mesh, same partition) and from the plane of a flat second
 * surface; a node on one or two flat surfaces, at least one tilted, from each of their planes.
 * `count` gets the number of such nodes that moved.
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
    const ms::NodeLabels& nl = f.repair.targets->Labels();
    ms::PatchSurfaces surfaces;
    surfaces.offsets = &nl.SurfaceOffsets();
    surfaces.normals = &nl.SurfaceNormals();
    surfaces.min_cosine = nl.FeatureCosine();
    const ms::SurfacePatches patches(
        field,
        std::vector<char>(static_cast<std::size_t>(field.NumVertices()), 1),
        nullptr,
        2,
        &surfaces);
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
        if (!(curved && num <= 2) && !(tilted && !curved && num <= 2)) {
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
        for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
            const double* n = normal + 3 * s;
            if (planar[s] == 0) {
                double p[3], pn[3];
                bool interior = false;
                dist = std::max(dist, patches.ClosestPoint(v, y, p, pn, interior, s - surf_off[v]));
                continue;
            }
            const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            dist = std::max(
                dist,
                std::fabs(n[0] * (y[0] - y0[0]) + n[1] * (y[1] - y0[1]) + n[2] * (y[2] - y0[2])) /
                    len);
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

void CheckCurved(mfem::Element::Type type, int n, double shear) {
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
    ExpectLimits(r, type == mfem::Element::TETRAHEDRON);
    EXPECT_EQ(FaceViolations(f, true), 0);
    EXPECT_LT(r.mean_deviation, r.initial_deviation);
    EXPECT_LE(drift, 1e-9);
    EXPECT_GT(moved, 0);
    if (mfem::Mpi::Root()) {
        std::printf("curved %s: deviation %.3f -> %.3f h (max %.3f), sj %.3f -> %.3f, rounds %d, "
                    "surface drift %.2e h, moved on surface %d\n",
                    type == mfem::Element::TETRAHEDRON ? "tet" : "hex",
                    r.initial_deviation,
                    r.mean_deviation,
                    r.max_deviation,
                    r.quality_before.min_scaled_jacobian,
                    r.quality_after.min_scaled_jacobian,
                    r.rounds,
                    drift,
                    moved);
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
    CheckBisector(serial, false, 0.4);
}

TEST(ConformalFit, TetInclinedBicrystal) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    CheckBisector(serial, true, 0.55);
}

TEST(ConformalFit, HexTricrystal) {
    CheckTricrystal(mfem::Element::HEXAHEDRON, 12);
}

TEST(ConformalFit, TetTricrystal) {
    CheckTricrystal(mfem::Element::TETRAHEDRON, 8);
}

TEST(ConformalFit, CurvedBoundaryConvergence) {
    // V2: an Additive (Johnson-Mehl) bicrystal, seeds (0.3, 0.5, 0.5) and (0.7, 0.5, 0.5) with
    // weights 0 and 0.3: the boundary |x - s1| = |x - s2| - 0.3 is one sheet of a hyperboloid of
    // revolution about the seed axis (foci the seeds, a = 0.15, c = 0.2, b^2 = c^2 - a^2),
    // x(rho) = 0.5 - a sqrt(1 + rho^2 / b^2), which meets the face x = 0 in a circle of radius
    // rho0 = b sqrt((0.5 / a)^2 - 1) = 0.42. Grain 1 volume = 2 pi int_0^rho0 x(rho) rho drho and
    // boundary area = 2 pi int_0^rho0 sqrt(1 + x'(rho)^2) rho drho (Gauss-Legendre, smooth
    // integrands: exact to rounding).
    // Measured (hex, order 1, 6 to 32 elements per unit): the fitted volume error falls from
    // 3-5 % (6-8 per unit) to 0.1-1.2 % (12-32), the staircase's from 16 % to 0.6-2 %; the
    // fitted area stays 10-25 % too large at every resolution (the staircase's 40-60 %), because
    // the fit leaves a waviness of a fraction of h at the stair corners whose slope does not
    // shrink with h (the quality limits keep the corner elements from flattening). Order 2 and
    // tets behave the same way. So area is checked against the staircase, not for convergence.
    ms::SeedSet seeds;
    seeds.x = {0.3, 0.7};
    seeds.y = {0.5, 0.5};
    seeds.z = {0.5, 0.5};
    seeds.weight = {0.0, 0.3};
    const double a = 0.15;
    const double b2 = 0.2 * 0.2 - a * a;
    const double rho0 = std::sqrt(b2 * ((0.5 / a) * (0.5 / a) - 1.0));
    const mfem::IntegrationRule& gl = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 63);
    double volume = 0.0;
    double area = 0.0;
    for (int q = 0; q < gl.GetNPoints(); ++q) {
        const double rho = rho0 * gl.IntPoint(q).x;
        const double w = rho0 * gl.IntPoint(q).weight * 2.0 * M_PI * rho;
        const double root = std::sqrt(1.0 + rho * rho / b2);
        const double slope = a * rho / (b2 * root);
        volume += w * (0.5 - a * root);
        area += w * std::sqrt(1.0 + slope * slope);
    }
    double volume_errors[2] = {0.0, 0.0};
    int level = 0;
    for (const int n : {8, 16}) {
        mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::HEXAHEDRON);
        Fitted f = Fit(serial, MPI_COMM_WORLD, seeds, "", false, 1, ms::MetricType::ADDITIVE);
        ExpectLimits(f.report, false);
        // The staircase: the same labels on the input positions
        mfem::ParMesh input(*f.pmesh);
        *input.GetNodes() = f.x0;
        input.NodesUpdated();
        const double stair_area = std::fabs(InterfaceArea(input, f.labels, 1, 2) - area) / area;
        const double stair_volume = std::fabs(GrainVolume(input, f.labels, 1) - volume) / volume;
        const double fit_area = std::fabs(InterfaceArea(*f.pmesh, f.labels, 1, 2) - area) / area;
        const double fit_volume = std::fabs(GrainVolume(*f.pmesh, f.labels, 1) - volume) / volume;
        EXPECT_LT(fit_area, 0.6 * stair_area);
        EXPECT_LT(fit_area, 0.3);
        volume_errors[level++] = fit_volume;
        if (mfem::Mpi::Root()) {
            std::printf("curved boundary n = %d: area error staircase %.4f, fitted %.4f; volume "
                        "error staircase %.4f, fitted %.4f\n",
                        n,
                        stair_area,
                        fit_area,
                        stair_volume,
                        fit_volume);
        }
    }
    EXPECT_LT(volume_errors[1], 0.5 * volume_errors[0]);
    EXPECT_LT(volume_errors[1], 0.02);
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
                    r.initial_deviation,
                    r.mean_deviation,
                    r.max_deviation,
                    r.quality_after.min_scaled_jacobian,
                    below,
                    r.rounds,
                    r.solves);
    }
}

TEST(ConformalFit, ThinGrains) {
    // E1: the middle of three grains in a row is two elements thick and both its (inclined)
    // boundaries pull its nodes inward; the limits hold and it keeps its volume. E5: with the seeds
    // closer, the middle grain is one element thick: it has no interior node, is reported as
    // under-resolved, and its nodes keep their positions.
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(10, 10, 10, mfem::Element::HEXAHEDRON);
    const auto run = [&](double middle, double spread, double tilt) {
        ms::SeedSet s;
        const double p[3][3] = {{middle - spread, 0.5 - tilt, 0.5},
                                {middle, 0.5, 0.5},
                                {middle + spread, 0.5 + tilt, 0.5 - 0.3 * tilt}};
        for (const auto& q : p) {
            s.x.push_back(q[0]);
            s.y.push_back(q[1]);
            s.z.push_back(q[2]);
        }
        return Fit(serial, MPI_COMM_WORLD, s);
    };
    Fitted thick = run(0.5, 0.2, 0.03); // middle grain about [0.4, 0.6]: centers 0.45, 0.55
    ExpectLimits(thick.report, false);
    EXPECT_EQ(thick.repair.targets->Summary().under_resolved_grains, 0);
    EXPECT_LT(thick.report.mean_deviation, 0.6 * thick.report.initial_deviation);
    EXPECT_NEAR(GrainVolume(*thick.pmesh, thick.labels, 2), 0.2, 0.02);

    // Middle grain about [0.4, 0.5], tilted by at most 0.02: element centers 0.45 only
    Fitted thin = run(0.45, 0.1, 0.002);
    ExpectLimits(thin.report, false);
    EXPECT_EQ(thin.repair.targets->Summary().under_resolved_grains, 1);
    // The held nodes of the thin grain did not get springs: their labels hold grain 2
    const ms::NodeLabels& nl = thin.repair.targets->Labels();
    int held_wrong = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        const int* items = nl.Labels().HostRead() + nl.LabelOffsets()[v];
        const bool has2 = std::find(items, items + nl.NumLabels(v), 2) != items + nl.NumLabels(v);
        if (has2 && nl.NumLabels(v) >= 2) {
            held_wrong += (thin.repair.targets->Reason(v) != ms::TargetReason::UNDER_RESOLVED &&
                           thin.repair.targets->Status(v) != ms::TargetStatus::OVER_CONSTRAINED)
                              ? 1
                              : 0;
        }
    }
    EXPECT_EQ(GlobalSum(held_wrong), 0);
}

TEST(ConformalFit, JitteredTetPolycrystal) {
    // An unstructured-like tet mesh: the interior vertices of an 8^3 x 6 Kuhn mesh moved by up to
    // 0.2 element sizes (fixed pseudo-random sequence), so the worst tets start below the mean
    // ratio floor, as in meshes from a mesh generator (a gmsh cube's worst tet: 0.33). The fit must
    // still move the boundaries (it could hardly move with the scaled Jacobian rules), keep the
    // limits relative to the input, and invert nothing.
    const int n = 8;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::TETRAHEDRON);
    double r = 0.4321;
    for (int v = 0; v < serial.GetNV(); ++v) {
        double* x = serial.GetVertex(v);
        for (int d = 0; d < 3; ++d) {
            r = std::fmod(r * 9301.0 + 0.49297, 1.0);
            const bool face = x[d] == 0.0 || x[d] == 1.0;
            x[d] += face ? 0.0 : 0.2 * (2.0 * r - 1.0) / n;
        }
    }
    Fitted f = Fit(serial, MPI_COMM_WORLD, ScatteredSeeds(12));
    const ms::ConformalFitReport& r_fit = f.report;
    EXPECT_LT(r_fit.quality_before.min_mean_ratio, ms::MESH_MEAN_RATIO_FLOOR_TET);
    ExpectLimits(r_fit, true);
    EXPECT_LT(r_fit.mean_deviation, 0.75 * r_fit.initial_deviation);
    EXPECT_EQ(FaceViolations(f), 0);
    if (mfem::Mpi::Root()) {
        std::printf("jittered tets: deviation %.3f -> %.3f h, mean ratio %.3f -> %.3f, rounds %d\n",
                    r_fit.initial_deviation,
                    r_fit.mean_deviation,
                    r_fit.quality_before.min_mean_ratio,
                    r_fit.quality_after.min_mean_ratio,
                    r_fit.rounds);
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
                    r.initial_deviation,
                    r.mean_deviation,
                    r.max_deviation,
                    r.quality_after.min_scaled_jacobian,
                    r.quality_after.max_aspect_ratio,
                    r.rounds,
                    static_cast<long long>(r.fit_nodes),
                    static_cast<long long>(linear.report.fit_nodes));
    }
}

TEST(ConformalFit, HexCurvedAndTiltedBoundary) {
    CheckCurved(mfem::Element::HEXAHEDRON, 12, 0.2);
}

TEST(ConformalFit, TetCurvedAndTiltedBoundary) {
    CheckCurved(mfem::Element::TETRAHEDRON, 8, 0.2);
}

TEST(ConformalFit, HexCurvedWallRim) {
    // A hex box whose side x = 1 bulges out (x = 1 + 0.3 y (1 - y)), so the curved wall meets
    // the flat top and bottom at rims. The inclined bisector crosses both rims: its nodes on a
    // rim slide along it (E7) instead of being held, and every sliding node stays on its surfaces
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(8, 8, 8, mfem::Element::HEXAHEDRON);
    serial.Transform([](const mfem::Vector& p, mfem::Vector& y) {
        y = p;
        y(0) = p(0) * (1.0 + 0.3 * p(1) * (1.0 - p(1)));
    });
    ms::SeedSet seeds;
    seeds.x = {0.75, 1.05};
    seeds.y = {0.3, 0.7};
    seeds.z = {0.4, 0.65};
    Fitted f = Fit(serial, MPI_COMM_WORLD, seeds);
    const ms::ConformalFitReport& r = f.report;
    ExpectLimits(r, false);
    EXPECT_LT(r.mean_deviation, 0.7 * r.initial_deviation);
    int moved = 0;
    EXPECT_LE(SurfaceDrift(serial, f, moved), 1e-9);
    EXPECT_GT(moved, 0);
    // Rim nodes of the grain boundary: FIT, closer to the bisector, top/bottom z kept bit for bit
    // (rim nodes: two surfaces, one curved, one flat)
    const ms::NodeLabels& nl = f.repair.targets->Labels();
    const int* surf_off = nl.SurfaceOffsets().HostRead();
    const int* planar = nl.SurfacePlanar().HostRead();
    const int* owned = nl.Field().OwnedVertexMarker().HostRead();
    const double n[3] = {seeds.x[1] - seeds.x[0], seeds.y[1] - seeds.y[0], seeds.z[1] - seeds.z[0]};
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const double c =
        0.5 * ((seeds.x[1] * seeds.x[1] + seeds.y[1] * seeds.y[1] + seeds.z[1] * seeds.z[1]) -
               (seeds.x[0] * seeds.x[0] + seeds.y[0] * seeds.y[0] + seeds.z[0] * seeds.z[0]));
    const mfem::GridFunction& x = *f.pmesh->GetNodes();
    const mfem::FiniteElementSpace& fes = *x.FESpace();
    int rim = 0;
    int bad = 0;
    double before = 0.0;
    double after = 0.0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        if (nl.NumLabels(v) != 2 || nl.BoundarySurfaces(v) != 2 || owned[v] == 0 ||
            (planar[surf_off[v]] != 0) == (planar[surf_off[v] + 1] != 0)) {
            continue;
        }
        double y[3];
        double y0[3];
        for (int d = 0; d < 3; ++d) {
            y[d] = x(fes.DofToVDof(v, d));
            y0[d] = f.x0(fes.DofToVDof(v, d));
        }
        const double h = f.repair.targets->MeshSize(v);
        before += std::fabs(n[0] * y0[0] + n[1] * y0[1] + n[2] * y0[2] - c) / len / h;
        after += std::fabs(n[0] * y[0] + n[1] * y[1] + n[2] * y[2] - c) / len / h;
        bad += (f.repair.targets->Status(v) != ms::TargetStatus::FIT) ? 1 : 0;
        bad += (y[2] != y0[2]) ? 1 : 0;
        ++rim;
    }
    MPI_Allreduce(MPI_IN_PLACE, &before, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &after, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rim = GlobalSum(rim);
    EXPECT_EQ(GlobalSum(bad), 0);
    EXPECT_GE(rim, 2);
    EXPECT_LT(after, 0.5 * before);
    Fitted self = Fit(serial, MPI_COMM_SELF, seeds);
    EXPECT_LE(WorstRankDifference(f, self), 1e-6);
    if (mfem::Mpi::Root()) {
        std::printf("curved wall rim: %d rim nodes, distance to the bisector %.3f -> %.3f h; "
                    "deviation %.3f -> %.3f h, sj %.3f -> %.3f\n",
                    rim,
                    before / std::max(rim, 1),
                    after / std::max(rim, 1),
                    r.initial_deviation,
                    r.mean_deviation,
                    r.quality_before.min_scaled_jacobian,
                    r.quality_after.min_scaled_jacobian);
    }
}

TEST(ConformalFit, DumpSlices) {
    const char* dir = std::getenv("CONFORMAL_DUMP_DIR");
    if (dir == nullptr) {
        GTEST_SKIP();
    }
    const int n = std::getenv("CONFORMAL_DUMP_N") ? std::atoi(std::getenv("CONFORMAL_DUMP_N")) : 16;
    const int grains = std::getenv("CONFORMAL_DUMP_GRAINS")
                           ? std::atoi(std::getenv("CONFORMAL_DUMP_GRAINS"))
                           : 20;
    const int skip = std::getenv("CONFORMAL_DUMP_SKIP")
                         ? std::atoi(std::getenv("CONFORMAL_DUMP_SKIP"))
                         : 0;
    const bool tet = std::getenv("CONFORMAL_DUMP_TET") != nullptr;
    const bool curved = std::getenv("CONFORMAL_DUMP_CURVED") != nullptr;
    const char* mesh_file = std::getenv("CONFORMAL_DUMP_MESH");
    mfem::Mesh serial = (mesh_file != nullptr)
                            ? mfem::Mesh(mesh_file)
                            : mfem::Mesh::MakeCartesian3D(n,
                                                          n,
                                                          n,
                                                          tet ? mfem::Element::TETRAHEDRON
                                                              : mfem::Element::HEXAHEDRON);
    ms::SeedSet seeds = ScatteredSeeds(grains, skip);
    if (std::getenv("CONFORMAL_DUMP_WALL") != nullptr) { // the case of HexCurvedWallRim
        serial.Transform([](const mfem::Vector& p, mfem::Vector& y) {
            y = p;
            y(0) = p(0) * (1.0 + 0.3 * p(1) * (1.0 - p(1)));
        });
        seeds = ms::SeedSet();
        seeds.x = {0.75, 1.05};
        seeds.y = {0.3, 0.7};
        seeds.z = {0.4, 0.65};
    }
    if (curved) { // the case of CheckCurved
        DomeAndShear(serial, 0.2);
        for (std::size_t i = 0; i < seeds.Size(); ++i) {
            seeds.x[i] += 0.2 * seeds.z[i];
            seeds.z[i] *= 1.1;
        }
    }
    const bool smoothed_only = std::getenv("CONFORMAL_DUMP_SMOOTHED") != nullptr;
    const int order = std::getenv("CONFORMAL_DUMP_ORDER")
                          ? std::atoi(std::getenv("CONFORMAL_DUMP_ORDER"))
                          : 1;
    Fitted f = Fit(
        serial, MPI_COMM_WORLD, seeds, std::string(dir) + "/orig.txt", smoothed_only, order);
    if (mfem::Mpi::Root() && std::getenv("CONFORMAL_DUMP_STATUS")) {
        const auto& ts = f.repair.targets->Summary();
        std::printf(
            "DUMP status: none %lld fit %lld over %lld held %lld fake %lld (smoothed %lld); "
            "reasons:",
            (long long)ts.status[0],
            (long long)ts.status[1],
            (long long)ts.status[2],
            (long long)ts.status[3],
            (long long)ts.status[4],
            (long long)ts.smoothed);
        for (int k = 0; k < ms::NUM_TARGET_REASONS; ++k) {
            std::printf(" %lld", (long long)ts.reason[static_cast<std::size_t>(k)]);
        }
        std::printf("\n");
    }
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
                    f.report.quality_after.max_aspect_ratio,
                    ar_counts[0] / ar_counts[3],
                    ar_counts[1] / ar_counts[3],
                    ar_counts[2] / ar_counts[3]);
    }
    if (mfem::Mpi::Root()) {
        const ms::ConformalFitReport& r = f.report;
        std::printf("DUMP %s %d^3 %d grains: deviation %.3f -> %.3f h (max %.3f), sj %.3f, rounds "
                    "%d, solves %d\n",
                    tet ? "tet" : "hex",
                    n,
                    grains,
                    r.initial_deviation,
                    r.mean_deviation,
                    r.max_deviation,
                    r.quality_after.min_scaled_jacobian,
                    r.rounds,
                    r.solves);
    }
    WriteElements(*f.pmesh, f.labels, f.x0, std::string(dir) + "/fit.txt");
    if (!tet) {
        WriteSections(*f.pmesh, f.labels, f.x0, std::string(dir) + "/fit_sections.txt");
    }
}

namespace {

/** One line per local element: label, quality, original center, current corners (.r<rank>). */
void WriteElements(mfem::ParMesh& pmesh,
                   const mfem::Array<int>& labels,
                   const mfem::Vector& x0,
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
void WriteSections(mfem::ParMesh& pmesh,
                   const mfem::Array<int>& labels,
                   const mfem::Vector& x0,
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
        const double side[4][2][2] = {
            {{0, 0}, {1, 0}}, {{1, 0}, {1, 1}}, {{1, 1}, {0, 1}}, {{0, 1}, {0, 0}}};
        for (const auto& s : side) {
            for (int k = 0; k < 6; ++k) {
                const double t = k / 6.0;
                mfem::IntegrationPoint ip;
                ip.Set3(s[0][0] + t * (s[1][0] - s[0][0]), s[0][1] + t * (s[1][1] - s[0][1]), 0.5);
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
