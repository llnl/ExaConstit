/**
 * @file test_geodesic_curved.cpp
 * @brief Tests of the geodesic labelers on curved (high-order) meshes: the nodal field of
 * labeling/vertex_field.hpp (every nodal point of the mesh's own Nodes space, p^3 virtual
 * sub-cells per element, no refined mesh) used by the FIM and graph solvers.
 *
 * What is checked, and why:
 *  - Lattice: the virtual sub-cells of hexes and tets of order 2 and 3 (Gauss-Lobatto, closed
 *    uniform and Bernstein Nodes) tile the reference element (reference volumes sum to the
 *    reference volume, all positive), every element has p^3 of them, the per-node enumeration
 *    visits each (cell, corner) pair exactly once, and their straight physical volumes sum to the
 *    element volume (exactly for straight elements, approximately for curved ones).
 *  - Concave wall: on a half annulus (inner radius a = 1) meshed with genuinely curved elements
 *    (Nodes moved onto the circles), the FIM distance behind the hole is compared with the
 *    analytic shortest path around the inner wall (tangent + arc + tangent, combined with the
 *    height difference on the unrolled cylinder). At fixed h the error falls from p = 1 to p = 3,
 *    and the corner-only field on the p = 3 mesh underestimates (its paths cut through the hole
 *    along the chords) while the nodal field does not cut beyond its own (h/p)-chords.
 *  - Convex domain: a cylinder with curved order-2 elements; the distance equals the straight
 *    line distance.
 *  - Rank count: every curved run is repeated on MPI_COMM_WORLD (METIS and a scattered partition
 *    with an empty rank) and on MPI_COMM_SELF and compared bitwise (node T and labels, element
 *    labels, sources, FIM round counts), so np = 2, 3, 4 are compared against np = 1.
 *  - Straight meshes: FieldNodes::AUTO keeps the corner field on straight-sided meshes with Nodes,
 *    giving the same bits as the mesh without Nodes.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/labeling/geodesic_fim.hpp"
#include "microstructure/labeling/geodesic_graph.hpp"
#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/serial_order.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.14159265358979323846;

int WorldRank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/**
 * Box mesh [x0, x0 + nx hx] x [y0, ...] x [z0, ...] of nx x ny x nz cells, hexes or the 6-tet
 * Kuhn split of every cell (same construction as test_geodesic_fim.cpp's BuildGrid).
 */
mfem::Mesh BuildBox(int nx, int ny, int nz, const double lo[3], const double h[3], bool tets) {
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    const int num_cells = nx * ny * nz;
    mfem::Mesh mesh(3, (nx + 1) * (ny + 1) * (nz + 1), tets ? 6 * num_cells : num_cells, 0, 3);
    for (int k = 0; k <= nz; k++) {
        for (int j = 0; j <= ny; j++) {
            for (int i = 0; i <= nx; i++) {
                mesh.AddVertex(lo[0] + i * h[0], lo[1] + j * h[1], lo[2] + k * h[2]);
            }
        }
    }
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int c[8];
                for (int b = 0; b < 8; b++) {
                    c[b] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                }
                if (!tets) {
                    const int hex[8] = {c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]};
                    mesh.AddHex(hex);
                } else {
                    const int orders[6][3] = {{1, 2, 4}, {1, 4, 2}, {2, 1, 4},
                                              {2, 4, 1}, {4, 1, 2}, {4, 2, 1}};
                    for (const auto& o : orders) {
                        const int tet[4] = {c[0], c[o[0]], c[o[0] + o[1]], c[7]};
                        mesh.AddTet(tet);
                    }
                }
            }
        }
    }
    if (tets) {
        mesh.FinalizeTetMesh(1, 1, true);
    } else {
        mesh.FinalizeHexMesh(1, 1, true);
    }
    return mesh;
}

/**
 * Gives `mesh` order-p continuous Nodes in the given basis (Gauss-Lobatto, closed uniform or
 * Bernstein = BasisType::Positive), interpolating its current straight geometry. The Nodes
 * GridFunction owns the collection and space (GridFunction::MakeOwner, as Mesh::SetCurvature
 * does, mesh.cpp:7230-7237).
 */
void SetNodes(mfem::Mesh& mesh, int p, int btype) {
    mfem::FiniteElementCollection* fec = nullptr;
    if (btype == mfem::BasisType::Positive) {
        fec = new mfem::H1Pos_FECollection(p, 3);
    } else {
        fec = new mfem::H1_FECollection(p, 3, btype);
    }
    auto* fes = new mfem::FiniteElementSpace(&mesh, fec, 3, mfem::Ordering::byVDIM);
    mesh.SetNodalFESpace(fes);
    mesh.GetNodes()->MakeOwner(fec);
}

/** Inner and outer radius and height of the half annulus. */
constexpr double ANNULUS_A = 1.0;
constexpr double ANNULUS_B = 2.0;
constexpr double ANNULUS_H = 0.5;

/**
 * Half annulus a <= r <= b, 0 <= theta <= pi, 0 <= z <= H with nr x nth x nz cells, as order-p
 * curved elements: a box in (r, theta, z) with order-p Nodes, mapped by x = r cos(theta),
 * y = r sin(theta). Mesh::Transform interpolates the map at the nodal points (mesh.cpp:14056), so
 * every nodal point lies exactly on its circle: the elements are genuinely curved for p >= 2
 * (p = 1: straight chords between the vertices).
 */
mfem::Mesh HalfAnnulus(int nr, int nth, int nz, int p, bool tets,
                       int btype = mfem::BasisType::GaussLobatto) {
    const double lo[3] = {ANNULUS_A, 0.0, 0.0};
    const double h[3] = {(ANNULUS_B - ANNULUS_A) / nr, PI / nth, ANNULUS_H / nz};
    mfem::Mesh mesh = BuildBox(nr, nth, nz, lo, h, tets);
    SetNodes(mesh, p, btype);
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(3);
        y(0) = x(0) * std::cos(x(1));
        y(1) = x(0) * std::sin(x(1));
        y(2) = x(2);
    });
    return mesh;
}

/**
 * Shortest path length inside the half annulus from s to t (both with r >= a) around the inner
 * cylinder r < a. In the (x, y) plane: the straight segment if it does not enter the disk r < a,
 * otherwise tangent + arc + tangent, sqrt(rs^2 - a^2) + a phi + sqrt(rt^2 - a^2) with
 * phi = |theta_t - theta_s| - acos(a/rs) - acos(a/rt). The cylinder is developable, so with the
 * height difference dz the 3D length is sqrt(L^2 + dz^2). Returns the 2D blocking flag through
 * `blocked`.
 */
double AnnulusGeodesic(const double* s, const double* t, bool* blocked) {
    const double a = ANNULUS_A;
    // Distance from the origin to the segment s-t in the plane
    const double dx = t[0] - s[0];
    const double dy = t[1] - s[1];
    const double len2 = dx * dx + dy * dy;
    double u = (len2 > 0.0) ? -(s[0] * dx + s[1] * dy) / len2 : 0.0;
    u = std::min(1.0, std::max(0.0, u));
    const double cx = s[0] + u * dx;
    const double cy = s[1] + u * dy;
    const double dz = t[2] - s[2];
    *blocked = std::sqrt(cx * cx + cy * cy) < a;
    if (!*blocked) {
        return std::sqrt(len2 + dz * dz);
    }
    const double rs = std::hypot(s[0], s[1]);
    const double rt = std::hypot(t[0], t[1]);
    // Angles in [0, pi]: a node on the plane y = 0 may carry y = -1e-16
    const double ts = std::atan2(std::max(0.0, s[1]), s[0]);
    const double tt = std::atan2(std::max(0.0, t[1]), t[0]);
    // Points on the wall itself may have r = a - 1 ulp: clamp so acos and sqrt stay real
    const double phi = std::fabs(tt - ts) - std::acos(std::min(1.0, a / rs)) -
                       std::acos(std::min(1.0, a / rt));
    const double planar = std::sqrt(std::max(0.0, rs * rs - a * a)) + a * phi +
                          std::sqrt(std::max(0.0, rt * rt - a * a));
    return std::sqrt(planar * planar + dz * dz);
}

/** Field nodes of a run: coordinates, T and label of every local node. */
struct NodeValues {
    std::vector<std::array<double, 3>> xyz;
    std::vector<double> T;
    std::vector<int> label;
};

/** The local nodes of a field. Not collective. */
NodeValues LocalNodeValues(const ms::VertexField& field) {
    NodeValues out;
    const double* xyz = field.Coordinates().HostRead();
    const double* t = field.Distance().HostRead();
    const int* lab = field.Label().HostRead();
    for (int v = 0; v < field.NumVertices(); ++v) {
        out.xyz.push_back({xyz[3 * v], xyz[3 * v + 1], xyz[3 * v + 2]});
        out.T.push_back(t[v]);
        out.label.push_back(lab[v]);
    }
    return out;
}

/** All ranks' nodes (MPI_Allgatherv), copies included. Collective over comm. */
NodeValues GatherNodeValues(const ms::VertexField& field, MPI_Comm comm) {
    const NodeValues local = LocalNodeValues(field);
    const int n = static_cast<int>(local.T.size());
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    std::vector<int> counts3(static_cast<std::size_t>(nranks));
    std::vector<int> displs3(static_cast<std::size_t>(nranks), 0);
    for (int r = 0; r < nranks; ++r) {
        const auto ur = static_cast<std::size_t>(r);
        if (r > 0) {
            displs[ur] = displs[ur - 1] + counts[ur - 1];
        }
        counts3[ur] = 3 * counts[ur];
        displs3[ur] = 3 * displs[ur];
    }
    const int total = displs.back() + counts.back();
    std::vector<double> local_xyz(3 * static_cast<std::size_t>(n));
    for (int v = 0; v < n; ++v) {
        for (int d = 0; d < 3; ++d) {
            local_xyz[3 * static_cast<std::size_t>(v) + static_cast<std::size_t>(d)] =
                local.xyz[static_cast<std::size_t>(v)][static_cast<std::size_t>(d)];
        }
    }
    std::vector<double> all_xyz(3 * static_cast<std::size_t>(total));
    NodeValues out;
    out.T.resize(static_cast<std::size_t>(total));
    out.label.resize(static_cast<std::size_t>(total));
    MPI_Allgatherv(local_xyz.data(), 3 * n, MPI_DOUBLE, all_xyz.data(), counts3.data(),
                   displs3.data(), MPI_DOUBLE, comm);
    MPI_Allgatherv(local.T.data(), n, MPI_DOUBLE, out.T.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, comm);
    MPI_Allgatherv(local.label.data(), n, MPI_INT, out.label.data(), counts.data(),
                   displs.data(), MPI_INT, comm);
    for (int k = 0; k < total; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        out.xyz.push_back({all_xyz[3 * uk], all_xyz[3 * uk + 1], all_xyz[3 * uk + 2]});
    }
    return out;
}

/** Element values gathered into serial order on every rank (GatherToSerial + MPI_Bcast). */
std::vector<int> GatherElements(MPI_Comm comm, const ms::SerialOrder& order,
                                const mfem::Array<int>& local, int serial_ne) {
    std::vector<int> values = ms::GatherToSerial(comm, order, local, 0);
    values.resize(static_cast<std::size_t>(serial_ne));
    MPI_Bcast(values.data(), serial_ne, MPI_INT, 0, comm);
    return values;
}

/** How to distribute a serial mesh for one run. */
enum class Layout { METIS_WORLD, SCATTERED_WORLD, SELF };

/** The distributed mesh with its serial-order bookkeeping. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/**
 * Partitions `serial` (collective over the communicator of the layout). SCATTERED_WORLD deals the
 * elements round-robin to ranks 0 .. nranks-2, so the last rank owns no element (at >= 2 ranks)
 * and the partitions have many shared nodes. (As in test_geodesic_fim.cpp.)
 */
Distributed Distribute(mfem::Mesh& serial, Layout layout) {
    const MPI_Comm comm = (layout == Layout::SELF) ? MPI_COMM_SELF : MPI_COMM_WORLD;
    Distributed d;
    if (layout != Layout::SCATTERED_WORLD) {
        d.order = ms::BuildSerialOrder(serial, comm);
        d.pmesh = ms::MakeParMesh(comm, serial, d.order);
        return d;
    }
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const int owners = (nranks > 1) ? nranks - 1 : 1;
    d.order.partitioning.SetSize(serial.GetNE());
    int num_local = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        d.order.partitioning[e] = e % owners;
        num_local += (e % owners == rank) ? 1 : 0;
    }
    d.order.local_to_serial.SetSize(num_local);
    int next = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        if (d.order.partitioning[e] == rank) {
            d.order.local_to_serial[next++] = e;
        }
    }
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** One geodesic run, gathered. */
struct GeodesicRun {
    bool curved = false;
    int order = 0;
    std::vector<int> element_seed;
    NodeValues nodes;
    ms::GeodesicReport report;
    std::vector<std::uint64_t> source_keys;
    std::vector<char> seed_outside;
};

/** A labeler run (FIM or graph) on `serial` distributed per `layout`, gathered. Collective. */
GeodesicRun RunGeodesic(mfem::Mesh& serial, Layout layout, const ms::SeedSet& seeds, bool additive,
                const ms::GeodesicParameters& params, bool graph) {
    const MPI_Comm comm = (layout == Layout::SELF) ? MPI_COMM_SELF : MPI_COMM_WORLD;
    Distributed d = Distribute(serial, layout);
    ms::GeodesicLabels g = graph ? ms::LabelGeodesicGraph(*d.pmesh, comm, seeds, additive, params)
                                 : ms::LabelGeodesicFim(*d.pmesh, comm, seeds, additive, params);
    GeodesicRun run;
    run.curved = g.field->UsesMeshNodes();
    run.order = g.field->LatticeOrder();
    run.element_seed = GatherElements(comm, d.order, g.element_seed, serial.GetNE());
    run.nodes = GatherNodeValues(*g.field, comm);
    run.report = g.report;
    run.source_keys = g.sources.source_element_key;
    run.seed_outside = g.sources.seed_outside;
    return run;
}

/**
 * Runs the three layouts and checks bitwise agreement with the single-rank run: element labels;
 * T and label of every node copy of every rank against the single-rank value at the same exact
 * coordinates (owner-broadcast coordinates, identical on all copies); sources; FIM round counts.
 * Returns the single-rank run. Collective over MPI_COMM_WORLD.
 */
GeodesicRun RunAndCompare(mfem::Mesh& serial, const ms::SeedSet& seeds, bool additive,
                  const ms::GeodesicParameters& params, bool graph) {
    const GeodesicRun self = RunGeodesic(serial, Layout::SELF, seeds, additive, params, graph);
    std::map<std::array<double, 3>, std::size_t> index;
    for (std::size_t k = 0; k < self.nodes.xyz.size(); ++k) {
        index[self.nodes.xyz[k]] = k;
    }
    EXPECT_EQ(index.size(), self.nodes.xyz.size()); // distinct node coordinates
    for (const Layout layout : {Layout::METIS_WORLD, Layout::SCATTERED_WORLD}) {
        const GeodesicRun other = RunGeodesic(serial, layout, seeds, additive, params, graph);
        EXPECT_EQ(other.curved, self.curved);
        EXPECT_EQ(other.element_seed, self.element_seed);
        std::size_t missing = 0;
        std::size_t differ = 0;
        for (std::size_t k = 0; k < other.nodes.xyz.size(); ++k) {
            const auto it = index.find(other.nodes.xyz[k]);
            if (it == index.end()) {
                ++missing;
                continue;
            }
            // Bitwise: == on doubles, +inf == +inf
            if (other.nodes.T[k] != self.nodes.T[it->second] ||
                other.nodes.label[k] != self.nodes.label[it->second]) {
                ++differ;
            }
        }
        EXPECT_EQ(missing, 0u);
        EXPECT_EQ(differ, 0u);
        EXPECT_EQ(other.source_keys, self.source_keys);
        EXPECT_EQ(other.seed_outside, self.seed_outside);
        EXPECT_EQ(other.report.unreached_vertices, self.report.unreached_vertices);
        EXPECT_EQ(other.report.split_elements, self.report.split_elements);
        EXPECT_EQ(other.report.unlabeled_vertices, 0);
        EXPECT_EQ(other.report.unlabeled_elements, 0);
        if (!graph) {
            EXPECT_EQ(other.report.distance_rounds, self.report.distance_rounds);
            EXPECT_EQ(other.report.label_rounds, self.report.label_rounds);
        }
    }
    return self;
}

/** One seed at (x, y, z) appended to `seeds`. */
void AddSeed(ms::SeedSet& seeds, double x, double y, double z) {
    seeds.x.push_back(x);
    seeds.y.push_back(y);
    seeds.z.push_back(z);
    seeds.source_id.push_back(static_cast<std::int64_t>(seeds.source_id.size()));
}

/** Volume of the tet (a, b, c, d) (positive for MFEM's orientation). */
double TetVolume(const double* a, const double* b, const double* c, const double* d) {
    const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const double w[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
            u[2] * (v[0] * w[1] - v[1] * w[0])) /
           6.0;
}

/**
 * Volume of a straight cell from its corners: a tet directly, a hex (MFEM corner order) as the
 * 6 tets around its diagonal 0-6 (exact for a parallelepiped, the trilinear volume to
 * second order otherwise).
 */
double CellVolume(const std::vector<const double*>& c) {
    if (c.size() == 4) {
        return TetVolume(c[0], c[1], c[2], c[3]);
    }
    const int tets[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
                            {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
    double vol = 0.0;
    for (const auto& t : tets) {
        vol += TetVolume(c[static_cast<std::size_t>(t[0])], c[static_cast<std::size_t>(t[1])],
                         c[static_cast<std::size_t>(t[2])], c[static_cast<std::size_t>(t[3])]);
    }
    return vol;
}

/** Largest relative error and smallest signed relative error of a set of samples. */
struct ErrorStats {
    double max_abs_rel = 0.0;
    double min_signed_rel = 0.0;
    double mean_abs_rel = 0.0;
    int count = 0;
};

/**
 * Error of the converged FIM distance on the half annulus (single-rank run, MPI_COMM_SELF, not
 * collective) against AnnulusGeodesic from the seed s, over the nodes selected by `use(node
 * coordinates, exact distance, blocked)`.
 */
template <typename Use>
ErrorStats AnnulusError(int nr, int nth, int nz, int p, bool tets, ms::FieldNodes field_nodes,
                        const double* s, Use use, int btype = mfem::BasisType::GaussLobatto,
                        bool* used_nodes = nullptr) {
    mfem::Mesh serial = HalfAnnulus(nr, nth, nz, p, tets, btype);
    Distributed d = Distribute(serial, Layout::SELF);
    ms::SeedSet seeds;
    AddSeed(seeds, s[0], s[1], s[2]);
    ms::GeodesicParameters params;
    params.field_nodes = field_nodes;
    const ms::GeodesicLabels g = ms::LabelGeodesicFim(*d.pmesh, MPI_COMM_SELF, seeds, false,
                                                      params);
    if (used_nodes != nullptr) {
        *used_nodes = g.field->UsesMeshNodes();
    }
    const NodeValues nodes = LocalNodeValues(*g.field);
    ErrorStats stats;
    stats.min_signed_rel = INF;
    for (std::size_t k = 0; k < nodes.T.size(); ++k) {
        bool blocked = false;
        const double exact = AnnulusGeodesic(s, nodes.xyz[k].data(), &blocked);
        if (!use(nodes.xyz[k].data(), exact, blocked)) {
            continue;
        }
        const double rel = (nodes.T[k] - exact) / exact;
        stats.max_abs_rel = std::max(stats.max_abs_rel, std::fabs(rel));
        stats.min_signed_rel = std::min(stats.min_signed_rel, rel);
        stats.mean_abs_rel += std::fabs(rel);
        ++stats.count;
    }
    if (stats.count > 0) {
        stats.mean_abs_rel /= stats.count;
    }
    return stats;
}

} // namespace

// Lattice: on genuinely curved hexes and tets (the half annulus) of order 2 and 3, for
// Gauss-Lobatto, closed uniform and Bernstein Nodes, every element has p^3 sub-cells; their
// reference corners tile the reference element (volumes positive, summing to 1 for the cube and
// 1/6 for the tet: the lattice is a deformation of the uniform one that keeps the reference
// faces flat); the per-node enumeration ForEachCellOfIncidence visits every (cell, corner) pair
// exactly once; LocateCell returns a cell containing the point for random reference points; and
// the straight physical sub-cells fill the curved element up to the chord error, which shrinks
// with p. On the straight (affine) box with order-p Nodes forced to the nodal field, the
// sub-cell volumes sum to the element volume to round-off.
TEST(CurvedLattice, SubCellsTileTheElement) {
    for (const bool tets : {false, true}) {
        for (const int btype : {mfem::BasisType::GaussLobatto, mfem::BasisType::ClosedUniform,
                                mfem::BasisType::Positive}) {
            double previous_volume_error = INF;
            for (const int p : {2, 3}) {
                mfem::Mesh serial = HalfAnnulus(2, 6, 1, p, tets, btype);
                Distributed d = Distribute(serial, Layout::SELF);
                ms::VertexField field(*d.pmesh, ms::FieldNodes::MESH_NODES);
                EXPECT_TRUE(field.UsesMeshNodes());
                EXPECT_EQ(field.LatticeOrder(), p);
                const ms::FieldCellView view = field.HostCellView();
                const int lattice_size = (p + 1) * (p + 1) * (p + 1);
                const double* xyz = field.Coordinates().HostRead();
                double worst_volume_error = 0.0;
                bool counts_ok = true;
                bool reference_ok = true;
                bool incidence_ok = true;
                for (int e = 0; e < field.NumElements(); ++e) {
                    // All cells of the element, their reference and physical volumes
                    int num_cells = 0;
                    double ref_volume = 0.0;
                    double phys_volume = 0.0;
                    std::map<int, int> corner_visits; // cell id -> visits from its corners
                    for (int cell = 0; cell < view.num_types * lattice_size; ++cell) {
                        int corners[ms::FieldCellView::MAX_CELL_NODES];
                        const int nc = view.CellNodes(e, cell, corners);
                        if (nc == 0) {
                            continue;
                        }
                        ++num_cells;
                        std::vector<const double*> ref;
                        std::vector<const double*> phys;
                        for (int k = 0; k < nc; ++k) {
                            const int lat = view.IncidenceLattice(corners[k], e);
                            const int i = lat % (p + 1);
                            const int j = (lat / (p + 1)) % (p + 1);
                            const int kk = lat / ((p + 1) * (p + 1));
                            ref.push_back(&view.ref_nodes[3 * view.lattice_to_local[view.Pack(
                                                                  i, j, kk)]]);
                            phys.push_back(&xyz[3 * corners[k]]);
                        }
                        const double rv = CellVolume(ref);
                        reference_ok = reference_ok && rv > 0.0;
                        ref_volume += rv;
                        phys_volume += CellVolume(phys);
                        corner_visits[cell] = 0;
                    }
                    counts_ok = counts_ok && num_cells == p * p * p;
                    const double expected_ref = tets ? 1.0 / 6.0 : 1.0;
                    reference_ok = reference_ok && std::fabs(ref_volume - expected_ref) < 1e-13;
                    // Each node of the element visits each of its cells once
                    int visits = 0;
                    const int* ev_off = field.ElementVertexOffsets().HostRead();
                    const int* ev = field.ElementVertices().HostRead();
                    for (int k = ev_off[e]; k < ev_off[e + 1]; ++k) {
                        view.ForEachCellOfIncidence(ev[k], e, view.IncidenceLattice(ev[k], e),
                                                    [&](const int*, const int*, int n) {
                                                        visits += 1;
                                                        incidence_ok = incidence_ok &&
                                                                       n == view.nodes_per_cell - 1;
                                                    });
                    }
                    incidence_ok = incidence_ok && visits == num_cells * view.nodes_per_cell;
                    const double elem_volume = d.pmesh->GetElementVolume(e);
                    worst_volume_error = std::max(worst_volume_error,
                                                  std::fabs(phys_volume - elem_volume) /
                                                      elem_volume);
                }
                EXPECT_TRUE(counts_ok);
                EXPECT_TRUE(reference_ok);
                EXPECT_TRUE(incidence_ok);
                // Chord error of the straight sub-cells: below 6% on these coarse 30-degree
                // elements, and smaller at p = 3 than at p = 2
                EXPECT_LT(worst_volume_error, 0.06);
                if (p == 3) {
                    EXPECT_LT(worst_volume_error, previous_volume_error);
                }
                previous_volume_error = worst_volume_error;
                if (WorldRank() == 0) {
                    std::cout << "  [lattice] " << (tets ? "tet" : "hex") << " basis " << btype
                              << " p=" << p << ": sub-cell volume vs element volume, worst "
                              << "relative difference " << worst_volume_error << "\n";
                }
                // LocateCell: random reference points land in a cell containing them
                std::uint64_t state = 12345;
                auto uniform = [&state]() {
                    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                    return static_cast<double>(state >> 11) / 9007199254740992.0;
                };
                bool locate_ok = true;
                for (int trial = 0; trial < 200; ++trial) {
                    double x = uniform();
                    double y = uniform();
                    double z = uniform();
                    if (tets && x + y + z > 1.0) {
                        continue;
                    }
                    const int cell = view.LocateCell(x, y, z);
                    int corners[ms::FieldCellView::MAX_CELL_NODES];
                    locate_ok = locate_ok && view.CellNodes(0, cell, corners) > 0;
                    if (!tets) {
                        // The hex cell's reference box holds the point
                        const int anchor = cell % lattice_size;
                        const int a[3] = {anchor % (p + 1), (anchor / (p + 1)) % (p + 1),
                                          anchor / ((p + 1) * (p + 1))};
                        const double xi[3] = {x, y, z};
                        for (int dd = 0; dd < 3; ++dd) {
                            locate_ok = locate_ok &&
                                        xi[dd] >= view.axis_points[3 * a[dd] + dd] - 1e-14 &&
                                        xi[dd] <= view.axis_points[3 * (a[dd] + 1) + dd] + 1e-14;
                        }
                    }
                }
                EXPECT_TRUE(locate_ok);
            }
        }
    }
    // Straight elements: the sub-cells fill the element exactly (to round-off)
    for (const bool tets : {false, true}) {
        const double lo[3] = {0.0, 0.0, 0.0};
        const double h[3] = {0.3, 0.2, 0.25};
        mfem::Mesh serial = BuildBox(2, 2, 2, lo, h, tets);
        SetNodes(serial, 3, mfem::BasisType::GaussLobatto);
        Distributed d = Distribute(serial, Layout::SELF);
        ms::VertexField field(*d.pmesh, ms::FieldNodes::MESH_NODES);
        const ms::FieldCellView view = field.HostCellView();
        const double* xyz = field.Coordinates().HostRead();
        double worst = 0.0;
        for (int e = 0; e < field.NumElements(); ++e) {
            double vol = 0.0;
            for (int cell = 0; cell < view.num_types * 64; ++cell) {
                int corners[ms::FieldCellView::MAX_CELL_NODES];
                const int nc = view.CellNodes(e, cell, corners);
                std::vector<const double*> phys;
                for (int k = 0; k < nc; ++k) {
                    phys.push_back(&xyz[3 * corners[k]]);
                }
                if (nc > 0) {
                    vol += CellVolume(phys);
                }
            }
            const double elem_volume = d.pmesh->GetElementVolume(e);
            worst = std::max(worst, std::fabs(vol - elem_volume) / elem_volume);
        }
        EXPECT_LT(worst, 1e-12);
    }
}

// Concave wall, analytic (accuracy behind the hole): half annulus, inner radius 1, 3 x 12 x 1
// cells (h = 1/3 radially, 15 degrees around), one seed at the mesh vertex (5/3, 0, 0). Behind the
// hole (no line of sight) the shortest path wraps the inner wall: tangent + arc + tangent
// (AnnulusGeodesic). Nodes with a path length above 1.5 are measured. At fixed h the max error
// must fall from p = 1 (straight chords, corner field) to p = 2 and p = 3 (nodal field: the
// solver works on p^3 sub-cells per element, like a mesh refined p times, and the wall nodes lie
// on the true circle), for hexes and tets and for Gauss-Lobatto and closed uniform Nodes.
// The seed sits on a vertex, i.e. on a field node for every p: for a seed inside a cell, the
// first-order solver's error next to the point source depends on where the seed falls in its
// (sub-)cell, and that offset, carried along every path, can hide the p trend (measured: a seed
// at r = 1.6 gave hex Gauss-Lobatto 5.1%, 3.1%, 5.6% for p = 1, 2, 3, exactly the numbers of the
// corner field on meshes refined 1, 2 and 3 times, which are not monotone either).
TEST(CurvedGeodesic, ConcaveWallErrorFallsWithOrder) {
    const double s[3] = {5.0 / 3.0, 0.0, 0.0};
    auto behind_hole = [](const double*, double exact, bool blocked) {
        return blocked && exact > 1.5;
    };
    for (const bool tets : {false, true}) {
        for (const int btype : {mfem::BasisType::GaussLobatto, mfem::BasisType::ClosedUniform}) {
            double previous = INF;
            for (const int p : {1, 2, 3}) {
                bool used_nodes = false;
                const ErrorStats e = AnnulusError(3, 12, 1, p, tets, ms::FieldNodes::AUTO, s,
                                                  behind_hole, btype, &used_nodes);
                EXPECT_GT(e.count, 10);
                EXPECT_EQ(used_nodes, p >= 2); // p = 1 is straight-sided: the corner field
                EXPECT_LT(e.max_abs_rel, previous);
                previous = e.max_abs_rel;
                if (WorldRank() == 0) {
                    std::cout << "  [concave wall, behind the hole] " << (tets ? "tet" : "hex")
                              << " basis " << btype << " p=" << p << " ("
                              << (used_nodes ? "nodal" : "corner")
                              << " field): max |rel err| " << e.max_abs_rel
                              << ", min signed rel err " << e.min_signed_rel << " over "
                              << e.count << " nodes\n";
                }
            }
            // The corner field on the same p = 3 mesh sees only the chords: same as p = 1
            const ErrorStats corners =
                AnnulusError(3, 12, 1, 3, tets, ms::FieldNodes::CORNERS, s, behind_hole, btype);
            const ErrorStats nodal =
                AnnulusError(3, 12, 1, 3, tets, ms::FieldNodes::MESH_NODES, s, behind_hole, btype);
            if (WorldRank() == 0) {
                std::cout << "  [concave wall, behind the hole] " << (tets ? "tet" : "hex")
                          << " basis " << btype << " p=3 mesh: corner-only field max |rel err| "
                          << corners.max_abs_rel << " vs nodal field " << nodal.max_abs_rel
                          << "\n";
            }
            EXPECT_LT(nodal.max_abs_rel, corners.max_abs_rel);
        }
    }
}

// Concave wall, chords cut through the hole: the same half annulus with 2 x 8 x 2 cells (22.5
// degrees per element around, nodes at z = 0.25), a seed on the inner wall at (1, 0, 0.25) (a
// vertex). The true shortest path to another inner-wall point at the same height is the arc,
// a theta. A field whose nodes are joined by straight chords can only follow the inscribed
// polygon, which lies inside the hole and is shorter: k chords of angle D have the length
// 2 k sin(D/2), i.e. the relative error is sin(D/2)/(D/2) - 1 (-0.64% for D = 22.5 degrees, with
// the vertex candidate T(a) + |v - a| reproducing the chord chain exactly). So the corner field
// underestimates by about that much, while the nodal field's chords are p times shorter (cut
// about p^2 times smaller, a little less for the non-uniform Gauss-Lobatto spacing). This is the
// "paths must not cut through the concave wall" check: the nodal field may cut only by its own
// (h/p)-chord sag.
TEST(CurvedGeodesic, WallChordsCutThroughTheHole) {
    const double s[3] = {1.0, 0.0, 0.25};
    auto wall_same_height = [](const double* v, double exact, bool) {
        return std::fabs(std::hypot(v[0], v[1]) - 1.0) < 1e-12 && std::fabs(v[2] - 0.25) < 1e-12 &&
               exact > 0.5;
    };
    const double half = 0.5 * PI / 8.0;
    const double chord_cut = std::sin(half) / half - 1.0; // -0.0064
    for (const bool tets : {false, true}) {
        const ErrorStats corners =
            AnnulusError(2, 8, 2, 3, tets, ms::FieldNodes::CORNERS, s, wall_same_height);
        const ErrorStats nodal =
            AnnulusError(2, 8, 2, 3, tets, ms::FieldNodes::MESH_NODES, s, wall_same_height);
        if (WorldRank() == 0) {
            std::cout << "  [inner wall, chord cut] " << (tets ? "tet" : "hex")
                      << " p=3: corner field min signed rel err " << corners.min_signed_rel
                      << ", max |err| " << corners.max_abs_rel << "; nodal field min signed "
                      << nodal.min_signed_rel << ", max |err| " << nodal.max_abs_rel
                      << " (chord deficit of 22.5 deg: " << chord_cut << ")\n";
        }
        EXPECT_GT(corners.count, 3);
        EXPECT_GE(nodal.count, 3 * corners.count);
        // Corner field: the inscribed polygon, shorter than the arc by the chord deficit
        EXPECT_LT(corners.min_signed_rel, 0.8 * chord_cut);
        // Nodal field: at most a fifth of that cut
        EXPECT_GT(nodal.min_signed_rel, 0.2 * chord_cut);
        EXPECT_LT(nodal.max_abs_rel, corners.max_abs_rel);
    }
}

// Convex curved domain: a cylinder of radius 1 (the square [-1,1]^2 x [0,0.5] mapped onto the disk
// by the elliptical map x' = x sqrt(1 - y^2/2), y' = y sqrt(1 - x^2/2)), 8 x 8 x 2 order-2 elements
// with nodal points on the true curved geometry. The domain is convex, so the geodesic distance
// is the straight line: nodes farther than 1 from the seed (a mesh vertex near one side, so the
// source cell is the same for both fields) must be within 4% (hex) / 10% (tet) of |v - s|, and
// the nodal field must be more accurate than the corner field on the same mesh.
TEST(CurvedGeodesic, ConvexCylinderMatchesEuclidean) {
    for (const bool tets : {false, true}) {
        double max_err[2] = {0.0, 0.0};
        for (const ms::FieldNodes mode : {ms::FieldNodes::CORNERS, ms::FieldNodes::AUTO}) {
            const double lo[3] = {-1.0, -1.0, 0.0};
            const double h[3] = {0.25, 0.25, 0.25};
            mfem::Mesh serial = BuildBox(8, 8, 2, lo, h, tets);
            SetNodes(serial, 2, mfem::BasisType::GaussLobatto);
            serial.Transform([](const mfem::Vector& x, mfem::Vector& y) {
                y.SetSize(3);
                y(0) = x(0) * std::sqrt(1.0 - 0.5 * x(1) * x(1));
                y(1) = x(1) * std::sqrt(1.0 - 0.5 * x(0) * x(0));
                y(2) = x(2);
            });
            Distributed d = Distribute(serial, Layout::SELF);
            ms::SeedSet seeds;
            const double s[3] = {-0.75, 0.0, 0.25}; // a mesh vertex (see the test comment)
            AddSeed(seeds, s[0], s[1], s[2]);
            ms::GeodesicParameters params;
            params.field_nodes = mode;
            const ms::GeodesicLabels g =
                ms::LabelGeodesicFim(*d.pmesh, MPI_COMM_SELF, seeds, false, params);
            EXPECT_EQ(g.field->UsesMeshNodes(), mode == ms::FieldNodes::AUTO);
            const NodeValues nodes = LocalNodeValues(*g.field);
            double worst = 0.0;
            for (std::size_t k = 0; k < nodes.T.size(); ++k) {
                const double* v = nodes.xyz[k].data();
                const double exact = std::sqrt((v[0] - s[0]) * (v[0] - s[0]) +
                                               (v[1] - s[1]) * (v[1] - s[1]) +
                                               (v[2] - s[2]) * (v[2] - s[2]));
                if (exact > 1.0) {
                    worst = std::max(worst, std::fabs(nodes.T[k] - exact) / exact);
                }
            }
            max_err[mode == ms::FieldNodes::AUTO ? 1 : 0] = worst;
        }
        if (WorldRank() == 0) {
            std::cout << "  [convex cylinder] " << (tets ? "tet" : "hex")
                      << " p=2: corner field max rel err " << max_err[0]
                      << ", nodal field max rel err " << max_err[1] << "\n";
        }
        // First-order eikonal solver next to a point source (1 to 1.8 length units = 8 to 15
        // sub-cells away; compare test_geodesic_fim's < 2% beyond 16 cells): hex < 4%, tet < 10%
        EXPECT_LT(max_err[1], tets ? 0.10 : 0.04);
        EXPECT_LT(max_err[1], max_err[0]);
    }
}

// Rank count: curved half annulus (hex and tet, p = 2 and 3), three seeds plus one in the hole
// (outside the material: snapped to the nearest nodal point), FIM and graph, centroid and vote
// sampling, Voronoi and Additive. Every run is compared bitwise with the single-rank run
// (RunAndCompare). The labels must also be complete (no unlabeled nodes or elements).
TEST(CurvedGeodesic, RankCountIndependent) {
    for (const bool tets : {false, true}) {
        for (const int p : {2, 3}) {
            mfem::Mesh serial = HalfAnnulus(2, 6, 2, p, tets);
            ms::SeedSet seeds;
            AddSeed(seeds, 1.4 * std::cos(0.2), 1.4 * std::sin(0.2), 0.1);
            AddSeed(seeds, 1.7 * std::cos(1.5), 1.7 * std::sin(1.5), 0.3);
            AddSeed(seeds, 1.2 * std::cos(2.9), 1.2 * std::sin(2.9), 0.4);
            AddSeed(seeds, 0.2, 0.3, 0.25); // in the hole
            seeds.weight = {0.0, 0.05, 0.1, 0.0};
            for (const bool graph : {false, true}) {
                for (const bool vote : {false, true}) {
                    ms::GeodesicParameters params;
                    if (vote) {
                        params.sampling = ms::SamplingMode::QUADRATURE_VOTE;
                        params.vote_levels = 1;
                    }
                    const GeodesicRun run = RunAndCompare(serial, seeds, !vote, params, graph);
                    EXPECT_TRUE(run.curved);
                    EXPECT_EQ(run.order, p);
                    EXPECT_EQ(run.seed_outside, (std::vector<char> {0, 0, 0, 1}));
                    // Every element is labeled, and each seed owns at least one element
                    std::vector<int> owned(4, 0);
                    bool all_labeled = true;
                    for (const int s : run.element_seed) {
                        all_labeled = all_labeled && s >= 0 && s < 4;
                        if (s >= 0 && s < 4) {
                            owned[static_cast<std::size_t>(s)] += 1;
                        }
                    }
                    EXPECT_TRUE(all_labeled);
                    for (int s = 0; s < 3; ++s) {
                        EXPECT_GT(owned[static_cast<std::size_t>(s)], 0);
                    }
                }
            }
        }
    }
}

namespace {

/**
 * Element labels keyed by element key over all ranks of comm (MPI_Allgatherv of (key, label)
 * pairs). Collective.
 */
std::map<std::uint64_t, int> LabelsByElementKey(mfem::ParMesh& pmesh,
                                                const mfem::Array<int>& element_seed,
                                                MPI_Comm comm) {
    const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(pmesh);
    std::vector<std::uint64_t> pairs;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        pairs.push_back(keys[static_cast<std::size_t>(e)]);
        pairs.push_back(static_cast<std::uint64_t>(static_cast<std::int64_t>(element_seed[e])));
    }
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    const int count = static_cast<int>(pairs.size());
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    int total = 0;
    for (int r = 0; r < nranks; ++r) {
        displs[static_cast<std::size_t>(r)] = total;
        total += counts[static_cast<std::size_t>(r)];
    }
    std::vector<std::uint64_t> all(static_cast<std::size_t>(total));
    MPI_Allgatherv(pairs.data(), count, MPI_UINT64_T, all.data(), counts.data(), displs.data(),
                   MPI_UINT64_T, comm);
    std::map<std::uint64_t, int> out;
    for (std::size_t i = 0; i + 1 < all.size(); i += 2) {
        out[all[i]] = static_cast<int>(static_cast<std::int64_t>(all[i + 1]));
    }
    return out;
}

} // namespace

// Rank count after a parallel refinement (the hook path: distribute, then UniformRefinement of
// the ParMesh). MFEM interpolates the refined Nodes per element, so the copies of a shared node
// can differ in the last bits between partitions, and the nodal field broadcasts the owner's
// copy: node values can no longer be compared bitwise across rank counts. Elements are therefore
// matched by element key (the same multiset on every rank count; keys round coordinates to a
// grid, mesh/element_key.hpp), and the element labels must agree on at least 99.9% of the
// elements (only a centroid within round-off of a geodesic boundary may go either way). Curved
// half annulus, hex and tet, p = 2, FIM and graph, with an outside seed.
TEST(CurvedGeodesic, RankCountIndependentAfterParallelRefinement) {
    for (const bool tets : {false, true}) {
        mfem::Mesh serial = HalfAnnulus(2, 6, 2, 2, tets);
        ms::SeedSet seeds;
        AddSeed(seeds, 1.4 * std::cos(0.2), 1.4 * std::sin(0.2), 0.1);
        AddSeed(seeds, 1.7 * std::cos(1.5), 1.7 * std::sin(1.5), 0.3);
        AddSeed(seeds, 1.2 * std::cos(2.9), 1.2 * std::sin(2.9), 0.4);
        AddSeed(seeds, 0.2, 0.3, 0.25); // in the hole
        for (const bool graph : {false, true}) {
            std::map<std::uint64_t, int> labels[2];
            for (const Layout layout : {Layout::SELF, Layout::METIS_WORLD}) {
                const MPI_Comm comm = (layout == Layout::SELF) ? MPI_COMM_SELF : MPI_COMM_WORLD;
                Distributed d = Distribute(serial, layout);
                d.pmesh->UniformRefinement();
                const ms::GeodesicParameters params;
                ms::GeodesicLabels g =
                    graph ? ms::LabelGeodesicGraph(*d.pmesh, comm, seeds, false, params)
                          : ms::LabelGeodesicFim(*d.pmesh, comm, seeds, false, params);
                EXPECT_TRUE(g.field->UsesMeshNodes());
                EXPECT_EQ(g.report.unlabeled_elements, 0);
                labels[layout == Layout::SELF ? 0 : 1] =
                    LabelsByElementKey(*d.pmesh, g.element_seed, comm);
            }
            EXPECT_EQ(labels[0].size(), static_cast<std::size_t>(8 * serial.GetNE()));
            EXPECT_EQ(labels[1].size(), labels[0].size()); // same key multiset, no collisions
            std::size_t agree = 0;
            for (const auto& entry : labels[1]) {
                const auto it = labels[0].find(entry.first);
                agree += (it != labels[0].end() && it->second == entry.second) ? 1 : 0;
            }
            if (WorldRank() == 0) {
                std::cout << "  [parallel refinement] " << (tets ? "tet" : "hex")
                          << (graph ? " graph" : " FIM") << ": " << agree << " of "
                          << labels[0].size() << " element labels agree with np 1\n";
            }
            EXPECT_GE(static_cast<double>(agree), 0.999 * static_cast<double>(labels[0].size()));
        }
    }
}

// Concave curved surface, labels: two seeds on either side of the hole near theta = 0 and
// theta = pi at the same radius. By symmetry of the half annulus, the geodesic boundary is the
// plane theta = pi/2 (x = 0): every element whose centroid has x > 0.1 belongs to the seed at
// theta = 0, every element with x < -0.1 to the other. (A Euclidean bisector would agree here;
// the check is that the curved field labels whole elements consistently along the wall.)
TEST(CurvedGeodesic, SymmetricSeedsSplitAtTheMidplane) {
    for (const bool tets : {false, true}) {
        mfem::Mesh serial = HalfAnnulus(2, 8, 1, 3, tets);
        ms::SeedSet seeds;
        AddSeed(seeds, 1.5 * std::cos(0.3), 1.5 * std::sin(0.3), 0.25);
        AddSeed(seeds, 1.5 * std::cos(PI - 0.3), 1.5 * std::sin(PI - 0.3), 0.25);
        const GeodesicRun run = RunAndCompare(serial, seeds, false, {}, false);
        bool ok = true;
        for (int e = 0; e < serial.GetNE(); ++e) {
            mfem::Vector c(3);
            mfem::ElementTransformation* tr = serial.GetElementTransformation(e);
            tr->Transform(mfem::Geometries.GetCenter(serial.GetElementBaseGeometry(e)), c);
            if (c(0) > 0.1) {
                ok = ok && run.element_seed[static_cast<std::size_t>(e)] == 0;
            } else if (c(0) < -0.1) {
                ok = ok && run.element_seed[static_cast<std::size_t>(e)] == 1;
            }
        }
        EXPECT_TRUE(ok);
    }
}

// Straight meshes are unchanged: a hex and a tet box with straight order-3 Nodes
// (Mesh::SetCurvature) is straight-sided, so FieldNodes::AUTO keeps the corner field, and the
// node T, node labels and element labels equal, bit for bit, those of the same mesh without
// Nodes (field coordinates come from the Nodes vertex dofs, which SetCurvature sets to the
// vertices exactly). Forcing the nodal field on the straight strut instead gives the h/p
// resolution: its max error against the straight-line distance is below the corner field's.
TEST(CurvedGeodesic, StraightMeshesKeepTheCornerField) {
    for (const bool tets : {false, true}) {
        const double lo[3] = {0.0, 0.0, 0.0};
        const double h[3] = {0.25, 0.25, 0.25};
        mfem::Mesh plain = BuildBox(6, 4, 3, lo, h, tets);
        mfem::Mesh curved_nodes = BuildBox(6, 4, 3, lo, h, tets);
        curved_nodes.SetCurvature(3);
        ms::SeedSet seeds;
        AddSeed(seeds, 0.1, 0.1, 0.1);
        AddSeed(seeds, 1.4, 0.2, 0.7);
        AddSeed(seeds, 0.6, 0.9, 0.3);
        AddSeed(seeds, 2.0, 0.5, 0.4); // outside
        for (const bool graph : {false, true}) {
            const GeodesicRun a = RunAndCompare(plain, seeds, false, {}, graph);
            const GeodesicRun b = RunAndCompare(curved_nodes, seeds, false, {}, graph);
            EXPECT_FALSE(a.curved);
            EXPECT_FALSE(b.curved);
            EXPECT_EQ(a.element_seed, b.element_seed);
            EXPECT_EQ(a.nodes.xyz, b.nodes.xyz);
            EXPECT_EQ(a.nodes.T, b.nodes.T);
            EXPECT_EQ(a.nodes.label, b.nodes.label);
            EXPECT_EQ(a.source_keys, b.source_keys);
        }
        // Forced nodal field on the straight order-3 mesh: finer resolution
        double worst[2] = {0.0, 0.0};
        for (const ms::FieldNodes mode : {ms::FieldNodes::CORNERS, ms::FieldNodes::MESH_NODES}) {
            const double hs[3] = {0.25, 0.25, 0.25};
            mfem::Mesh strut = BuildBox(4, 4, 16, lo, hs, tets);
            strut.SetCurvature(3);
            Distributed d = Distribute(strut, Layout::SELF);
            ms::SeedSet one;
            AddSeed(one, 0.5, 0.5, 0.0);
            ms::GeodesicParameters params;
            params.field_nodes = mode;
            const ms::GeodesicLabels g =
                ms::LabelGeodesicFim(*d.pmesh, MPI_COMM_SELF, one, false, params);
            const NodeValues nodes = LocalNodeValues(*g.field);
            for (std::size_t k = 0; k < nodes.T.size(); ++k) {
                const double* v = nodes.xyz[k].data();
                const double exact =
                    std::sqrt((v[0] - 0.5) * (v[0] - 0.5) + (v[1] - 0.5) * (v[1] - 0.5) +
                              v[2] * v[2]);
                if (exact > 2.0) {
                    const std::size_t m = (mode == ms::FieldNodes::MESH_NODES) ? 1 : 0;
                    worst[m] = std::max(worst[m], std::fabs(nodes.T[k] - exact) / exact);
                }
            }
        }
        if (WorldRank() == 0) {
            std::cout << "  [straight strut] " << (tets ? "tet" : "hex")
                      << " order-3 Nodes: corner field max rel err " << worst[0]
                      << ", forced nodal field " << worst[1] << "\n";
        }
        EXPECT_LT(worst[1], worst[0]);
    }
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
