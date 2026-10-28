/*
 * Fitting of the mesh nodes to the grain boundaries: an elastic body pulled toward the targets by
 * springs, in rounds of increasing spring strength, with local back-off and stiffening that keep
 * every element within its quality limits (conformal_fitter.hpp explains the method).
 */
#include "microstructure/conformal/conformal_fitter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include "microstructure/conformal/boundary_projector.hpp"
#include "microstructure/util/collective_errors.hpp"

namespace exaconstit::microstructure {

namespace {

/** @brief After a failed solve, the springs of the fitted nodes of every element within this
 * margin above its scaled Jacobian or mean ratio limit are halved (not only those of the elements
 * that broke the limit, which would just move the problem next door). */
constexpr double BACKOFF_MARGIN = 0.05;
/** @brief Elements with a scaled Jacobian below this get a stiffness factor (0.6 / sj)^exponent. */
constexpr double STIFFEN_BELOW = 0.6;
/** @brief Same as BACKOFF_MARGIN for the aspect ratio: elements within this margin below their
 * aspect ratio limit also back off. */
constexpr double ASPECT_RATIO_BACKOFF_MARGIN = 0.5;
/** @brief Slack for elements that are poor in the input: an element already beyond a limit, or
 * within this fraction of it, gets as its limit its own input value minus this fraction (for the
 * aspect ratio: plus). */
constexpr double POOR_INPUT_SLACK = 0.1;
/** @brief The aspect ratio of an input element already beyond the ceiling may grow by this
 * fraction of its input value. Such elements are thin (a pillow or buffer layer along a curved
 * surface), and their aspect ratio changes by the relative change of their thickness, so a small
 * move across them already costs 10 %: with 0.1 the nodes next to such a layer hardly fit (a
 * strut lattice with pillows: 0.38 -> 0.36 h; with 0.3: 0.28 h, the smallest scaled Jacobian
 * 0.42 -> 0.35 and the largest aspect ratio 17 -> 20). */
constexpr double POOR_INPUT_ASPECT_SLACK = 0.3;
/** @brief Bulk quality guard for tets: a tet with a mean ratio below this counts as mediocre. */
constexpr double TET_GUARD_MEAN_RATIO = 0.5;
/** @brief Largest stiffness factor of an element (1 = unstiffened). */
constexpr double MAX_STIFFNESS = 1e3;
/** @brief Bulk quality guard: a round may leave at most this share of all elements newly
 * mediocre (beyond those mediocre in the input). */
constexpr double BULK_SHARE = 0.01;
/** @brief Bulk quality guard for hexes: a hex with a scaled Jacobian below floor + this counts
 * as mediocre. */
constexpr double BULK_MARGIN = 0.2;
/** @brief Retries of a first round that fails the bulk guard, each with a quarter of the spring
 * of the try before. */
constexpr int WEAK_TRIES = 3;
/**
 * @brief Stiffness of the spring that holds a sliding node near its tilted or curved surface,
 * along the surface normal, per unit of element size h. The exact projection after each solve
 * puts the node back on its surface; this spring only keeps the solve from moving it far off.
 * Weaker springs (10 to 100) let the projection shear thin boundary elements (on a pillowed strut
 * lattice the fit got worse: 0.265 -> 0.29 h at 100). Stronger ones make the system harder to
 * solve: at 1e4 CG needed about 3 times the iterations on curved surfaces, for the same fit.
 */
constexpr double NORMAL_PENALTY = 1e3;

/** @brief How a node on the outer boundary moves (box-face coordinates are fixed separately). */
enum class Slide : char {
    NONE = 0,   ///< on no tilted or curved surface (only on box faces, or inside the mesh)
    PLANES = 1, ///< on one or two flat surfaces, at least one tilted: in the plane, or along the
                ///< line where the two meet
    CURVED = 2, ///< on one or two surfaces, at least one curved: on the input boundary faces
                ///< around it (patches), along the sharp edge when there are two
    FIXED = 3   ///< on three or more surfaces (a corner): does not move
};

/**
 * @brief Moves y to the nearest point on one plane, or on the line where two planes meet: the
 * planes are n_k . x = c_k with unit normals n_k (num = 1 or 2, two planes not parallel), and the
 * nearest point is y - N^T (N N^T)^-1 (N y - c) with N the matrix whose rows are the normals.
 * (boundary_projector.cpp has the same formula with a guard for nearly parallel planes; the
 * planes here are surfaces separated by a sharp edge, never parallel.)
 */
void ProjectOntoPlanes(const double (*n)[3], const double* c, int num, double y[3]) {
    double r[2];
    for (int k = 0; k < num; ++k) {
        r[k] = n[k][0] * y[0] + n[k][1] * y[1] + n[k][2] * y[2] - c[k];
    }
    double a[2] = {r[0], 0.0};
    if (num == 2) {
        const double g = n[0][0] * n[1][0] + n[0][1] * n[1][1] + n[0][2] * n[1][2];
        const double det = 1.0 - g * g;
        a[0] = (r[0] - g * r[1]) / det;
        a[1] = (r[1] - g * r[0]) / det;
    }
    for (int k = 0; k < num; ++k) {
        for (int d = 0; d < 3; ++d) {
            y[d] -= a[k] * n[k][d];
        }
    }
}

/** @brief True when n points exactly along a coordinate axis (two components exactly zero), as
 * the normals of box faces do. */
bool IsAxis(const double n[3]) {
    int zeros = 0;
    for (int d = 0; d < 3; ++d) {
        zeros += (n[d] == 0.0) ? 1 : 0;
    }
    return zeros == 2;
}

/** @brief Sum of a per-rank count over all ranks (collective). */
std::int64_t GlobalSum(std::int64_t local, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT64_T, MPI_SUM, comm);
    return local;
}

/** @brief Copies the current node positions from the Nodes grid function into a plain array,
 * 3 doubles (x, y, z) per node in node order (the layout NodeLabelQuery::position expects). */
void GatherPositions(const mfem::GridFunction& nodes,
                     const mfem::FiniteElementSpace& fes,
                     int nv,
                     std::vector<double>& out) {
    out.resize(3 * static_cast<std::size_t>(nv));
    for (int v = 0; v < nv; ++v) {
        for (int d = 0; d < 3; ++d) {
            out[3 * static_cast<std::size_t>(v) + d] = nodes(fes.DofToVDof(v, d));
        }
    }
}

/**
 * @brief Keeps one fitted node per quad point (a point where four grains meet).
 *
 * Several nodes can carry the same four grains and so share one target point; only the node
 * closest to it stays fitted, the others are unmarked (marker[v] = false). Every rank sends, for
 * each fitted 4-grain node it owns, a record of 8 doubles: the four grains, the squared distance
 * to the target and the position. All records go to every rank (MPI_Allgatherv), and per set of
 * four grains the record with the smallest (distance, then position) wins. Since every rank sees
 * all records, every copy of a node reaches the same decision. Collective.
 */
void KeepClosestQuadPointNodes(const NodeLabels& labels,
                               const NodeTargets& targets,
                               mfem::Array<bool>& marker) {
    const VertexField& field = labels.Field();
    const MPI_Comm comm = field.Comm();
    const int nv = labels.NumNodes();
    const double* xyz = field.Coordinates().HostRead();
    const double* t = targets.Targets().HostRead();
    const int* owned = field.OwnedVertexMarker().HostRead();
    const int* loff = labels.LabelOffsets().HostRead();
    const int* litems = labels.Labels().HostRead();
    const auto is_quad = [&](int v) {
        return marker[v] && labels.NumLabels(v) == 4;
    };
    std::vector<double> mine;
    for (int v = 0; v < nv; ++v) {
        if (!is_quad(v) || owned[v] == 0) {
            continue;
        }
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            d2 += (t[3 * v + d] - xyz[3 * v + d]) * (t[3 * v + d] - xyz[3 * v + d]);
        }
        mine.insert(mine.end(), litems + loff[v], litems + loff[v] + 4);
        mine.push_back(d2);
        mine.insert(mine.end(), xyz + 3 * v, xyz + 3 * v + 3);
    }
    int count = static_cast<int>(mine.size());
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(static_cast<std::size_t>(size));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        displs[static_cast<std::size_t>(r)] = displs[static_cast<std::size_t>(r) - 1] +
                                              counts[static_cast<std::size_t>(r) - 1];
    }
    std::vector<double> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(
        mine.data(), count, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
    // Per set of four grains, the best (squared distance, x, y, z); std::array compares
    // lexicographically, so ties in distance go to the smaller position
    std::map<std::array<int, 4>, std::array<double, 4>> best;
    for (std::size_t k = 0; k + 8 <= all.size(); k += 8) {
        const std::array<int, 4> key = {static_cast<int>(all[k]),
                                        static_cast<int>(all[k + 1]),
                                        static_cast<int>(all[k + 2]),
                                        static_cast<int>(all[k + 3])};
        const std::array<double, 4> val = {all[k + 4], all[k + 5], all[k + 6], all[k + 7]};
        const auto it = best.find(key);
        if (it == best.end() || val < it->second) {
            best[key] = val;
        }
    }
    // A 4-grain node stays fitted only when it is the winner of its set (same position)
    for (int v = 0; v < nv; ++v) {
        if (!is_quad(v)) {
            continue;
        }
        const std::array<int, 4> key = {
            litems[loff[v]], litems[loff[v] + 1], litems[loff[v] + 2], litems[loff[v] + 3]};
        const auto it = best.find(key);
        marker[v] = it != best.end() && it->second[1] == xyz[3 * v] &&
                    it->second[2] == xyz[3 * v + 1] && it->second[3] == xyz[3 * v + 2];
    }
}

} // namespace

namespace {

/**
 * @brief What a fitting pass moves (FitConformal).
 *  - ALL: every fitted node, displacement in the Nodes space (order-1 meshes);
 *  - LINEAR_VERTICES: the vertex nodes (dofs 0..NV-1 of the Nodes space, numbered first by MFEM)
 *    with an order-1 displacement field, interpolated to every nodal point (higher orders, first
 *    pass: the elements move as straight-sided order-1 elements would);
 *  - CURVATURE: the vertices fixed, the other fitted nodal points pulled toward the closest point
 *    of their grain boundary from where they are now (higher orders, second pass).
 */
enum class PassKind { ALL, LINEAR_VERTICES, CURVATURE };

/*
 * FitPass: one fitting pass from the current Nodes (the method is described in
 * conformal_fitter.hpp).
 *  1. Checks the inputs and measures the quality of the input mesh.
 *  2. Decides which nodes are fitted (status FIT, one per quad point) and how every node may move:
 *     fixed coordinates on box faces, sliding on tilted planes or curved patches, or fixed.
 *  3. Measures the starting deviation and, for the passes of a higher-order mesh, chooses what
 *     the springs pull: the vertices toward their targets (LINEAR_VERTICES), or the other nodal
 *     points toward the closest point on their grain boundary from where they are (CURVATURE).
 *  4. Runs the rounds of spring-balanced elastic solves with back-off and stiffening, keeping the
 *     last round that respects the quality limits and the bulk guard.
 *  5. Measures the final deviation and quality.
 * Collective over the mesh communicator.
 */
ConformalFitReport FitPass(mfem::ParMesh& pmesh,
                           const NodeTargets& targets,
                           const ConformalFitOptions& options,
                           PassKind kind_of_pass) {
    const bool vertices_only = kind_of_pass == PassKind::LINEAR_VERTICES;
    const bool curvature = kind_of_pass == PassKind::CURVATURE;
    const MPI_Comm comm = pmesh.GetComm();
    const NodeLabels& labels = targets.Labels();
    const VertexField& field = labels.Field();

    // Step 1: the mesh must carry Nodes and be the one the node labels were built on
    auto* nodes = dynamic_cast<mfem::ParGridFunction*>(pmesh.GetNodes());
    ThrowIfAnyRank(comm,
                   nodes == nullptr || &field.Mesh() != &pmesh,
                   "FitConformal: the mesh needs Nodes (a ParGridFunction; call "
                   "ParMesh::SetCurvature(1)) and must be the mesh of the targets");
    mfem::ParFiniteElementSpace& fes = *nodes->ParFESpace();
    ThrowIfAnyRank(comm,
                   !field.UsesMeshNodes() || field.NumVertices() != fes.GetNDofs(),
                   "FitConformal: the node labels must use the mesh Nodes "
                   "(FieldNodes::MESH_NODES)");
    ThrowIfAnyRank(comm,
                   !(options.initial_spring > 0.0) || options.max_rounds < 1 ||
                       options.max_attempts < 1 || options.min_improvement < 0.0 ||
                       options.stiffening_exponent < 0.0 || options.stiffness_smoothing < 0 ||
                       !(options.poisson_ratio >= 0.0 && options.poisson_ratio < 0.5),
                   "FitConformal: invalid options");
    ConformalFitReport report;
    report.quality_before = ComputeMeshQuality(pmesh);
    const int nv = labels.NumNodes();
    const int ne = pmesh.GetNE();
    // The scaled Jacobian floor of hexes; tets use the mean ratio floor instead, and -1 makes
    // the scaled Jacobian checks below always pass for them
    const double floor = (pmesh.GetTypicalElementGeometry() == mfem::Geometry::CUBE)
                             ? MESH_QUALITY_FLOOR_HEX
                             : -1.0;
    const mfem::ParGridFunction x0(*nodes);

    // Step 2: marker[v] = node v is pulled toward a target
    mfem::Array<bool> marker(nv);
    for (int v = 0; v < nv; ++v) {
        marker[v] = targets.Status(v) == TargetStatus::FIT;
    }
    KeepClosestQuadPointNodes(labels, targets, marker);
    const double* t = targets.Targets().HostRead();
    const int* owned = field.OwnedVertexMarker().HostRead();
    const int* surf_off = labels.SurfaceOffsets().HostRead();
    const double* surf_normal = labels.SurfaceNormals().HostRead();
    const int* surf_planar = labels.SurfacePlanar().HostRead();
    const int* loff = labels.LabelOffsets().HostRead();
    const int* litems = labels.Labels().HostRead();
    const double* xyz0 = field.Coordinates().HostRead();
    // How every node may move: fixed[3 v + d] = coordinate d of node v keeps its input value
    // (a box face, a corner, or a vertex in the CURVATURE pass); slide[v] = how it follows its
    // tilted or curved surfaces. A box face normal fixes its axis; tilted and curved surfaces
    // decide the Slide kind.
    std::vector<char> fixed(3 * static_cast<std::size_t>(nv), 0);
    std::vector<Slide> slide(static_cast<std::size_t>(nv), Slide::NONE);
    std::int64_t fit_nodes = 0;
    std::int64_t fixed_nodes = 0;
    for (int v = 0; v < nv; ++v) {
        const auto vs = 3 * static_cast<std::size_t>(v);
        const int num_surfaces = surf_off[v + 1] - surf_off[v];
        bool tilted = false;
        bool curved = false;
        for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
            const double* n = surf_normal + 3 * s;
            if (surf_planar[s] != 0 && IsAxis(n)) {
                for (int d = 0; d < 3; ++d) {
                    fixed[vs + d] |= (n[d] != 0.0) ? 1 : 0;
                }
            } else {
                tilted = tilted || surf_planar[s] != 0;
                curved = curved || surf_planar[s] == 0;
            }
        }
        Slide& kind = slide[static_cast<std::size_t>(v)];
        if (curved) {
            kind = (num_surfaces <= 2) ? Slide::CURVED : Slide::FIXED;
        } else if (tilted) {
            kind = (num_surfaces <= 2) ? Slide::PLANES : Slide::FIXED;
        }
        if (kind == Slide::FIXED || (curvature && v < pmesh.GetNV())) {
            fixed[vs] = fixed[vs + 1] = fixed[vs + 2] = 1;
        }
        fit_nodes += (marker[v] && owned[v]) ? 1 : 0;
        fixed_nodes += (fixed[vs] && fixed[vs + 1] && fixed[vs + 2] && owned[v]) ? 1 : 0;
    }
    report.fit_nodes = GlobalSum(fit_nodes, comm);
    report.fixed_nodes = GlobalSum(fixed_nodes, comm);

    // What the sliding nodes slide on. PLANES: planes[v] holds up to two planes (n0, c0, n1, c1)
    // through the node's input position. CURVED: the patches of MakeCurvedPatches (one per
    // surface; on a sharp edge the node slides along the edge). penalty[v] holds the normals of
    // the stiff spring that keeps the node near its surfaces during the solve: count, n0, n1.
    const std::unique_ptr<SurfacePatches> patches_ptr = MakeCurvedPatches(labels,
                                                                          targets.PatchRings());
    const SurfacePatches& patches = *patches_ptr;
    std::vector<std::array<double, 8>> planes(static_cast<std::size_t>(nv));  // n0 c0 n1 c1
    std::vector<std::array<double, 7>> penalty(static_cast<std::size_t>(nv)); // count, n0, n1
    for (int v = 0; v < nv; ++v) {
        const Slide kind = slide[static_cast<std::size_t>(v)];
        auto& pl = planes[static_cast<std::size_t>(v)];
        auto& pen = penalty[static_cast<std::size_t>(v)];
        pen[0] = 0.0;
        if (kind == Slide::PLANES) {
            int k = 0;
            for (int s = surf_off[v]; s < surf_off[v + 1]; ++s, ++k) {
                const double* n = surf_normal + 3 * s;
                const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                double c = 0.0;
                for (int d = 0; d < 3; ++d) {
                    pl[4 * k + d] = n[d] / len;
                    c += pl[4 * k + d] * xyz0[3 * v + d];
                }
                pl[4 * k + 3] = c;
                if (!IsAxis(n)) {
                    const int m = static_cast<int>(pen[0]);
                    for (int d = 0; d < 3; ++d) {
                        pen[1 + 3 * m + d] = pl[4 * k + d];
                    }
                    pen[0] += 1.0;
                }
            }
        } else if (kind == Slide::CURVED) {
            // Per surface: the area-weighted normal of a curved surface's input faces around the
            // node, or the normal of a tilted plane (axis planes fix a coordinate instead)
            for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
                double n[3] = {0.0, 0.0, 0.0};
                if (surf_planar[s] != 0) {
                    if (IsAxis(surf_normal + 3 * s)) {
                        continue;
                    }
                    std::copy(surf_normal + 3 * s, surf_normal + 3 * s + 3, n);
                }
                const int k = s - surf_off[v];
                for (int tri = 0; surf_planar[s] == 0 && tri < patches.NumTriangles(v, k); ++tri) {
                    const double* q = patches.Triangle(v, tri, k);
                    const double a[3] = {q[3] - q[0], q[4] - q[1], q[5] - q[2]};
                    const double b[3] = {q[6] - q[0], q[7] - q[1], q[8] - q[2]};
                    // Orient like the surface normal (faces of interfaces can point either way)
                    const double c[3] = {a[1] * b[2] - a[2] * b[1],
                                         a[2] * b[0] - a[0] * b[2],
                                         a[0] * b[1] - a[1] * b[0]};
                    const double* ns = surf_normal + 3 * s;
                    const double sign = (c[0] * ns[0] + c[1] * ns[1] + c[2] * ns[2] < 0.0) ? -1.0
                                                                                           : 1.0;
                    for (int d = 0; d < 3; ++d) {
                        n[d] += sign * c[d];
                    }
                }
                const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (len > 0.0) {
                    const int m = static_cast<int>(pen[0]);
                    for (int d = 0; d < 3; ++d) {
                        pen[1 + 3 * m + d] = n[d] / len;
                    }
                    pen[0] += 1.0;
                }
            }
        }
    }
    // Puts the sliding nodes back on their surfaces after a solve. Every copy of a node has the
    // same position, planes and patches, so every copy lands on the same point; fixed coordinates
    // (box faces) are restored from the input exactly.
    const auto project_to_surfaces = [&]() {
        for (int v = 0; v < nv; ++v) {
            const Slide kind = slide[static_cast<std::size_t>(v)];
            if (kind != Slide::PLANES && kind != Slide::CURVED) {
                continue;
            }
            double y[3];
            for (int d = 0; d < 3; ++d) {
                y[d] = (*nodes)(fes.DofToVDof(v, d));
            }
            if (kind == Slide::PLANES) {
                const auto& pl = planes[static_cast<std::size_t>(v)];
                const double n[2][3] = {{pl[0], pl[1], pl[2]}, {pl[4], pl[5], pl[6]}};
                const double c[2] = {pl[3], pl[7]};
                ProjectOntoPlanes(n, c, surf_off[v + 1] - surf_off[v], y);
            } else {
                ProjectOnNodeSurfaces(labels,
                                      patches,
                                      v,
                                      xyz0 + 3 * v,
                                      1e-10 * targets.MeshSize(v),
                                      SURFACE_PROJECTION_ITERATIONS,
                                      y);
            }
            for (int d = 0; d < 3; ++d) {
                const int vd = fes.DofToVDof(v, d);
                (*nodes)(vd) = fixed[3 * static_cast<std::size_t>(v) + d] ? x0(vd) : y[d];
            }
        }
    };

    // Deviation of the fitted nodes from their grain boundaries at the current positions: the
    // distance to the closest point on the boundary from where the node is, in element sizes,
    // largest and mean over the fitted nodes (each counted on its owner rank)
    std::vector<double> here;
    const auto deviation = [&](double& max_dev, double& mean_dev) {
        GatherPositions(*nodes, fes, nv, here);
        std::vector<NodeLabelQuery> queries;
        for (int v = 0; v < nv; ++v) {
            if (!marker[v]) {
                continue;
            }
            NodeLabelQuery q;
            q.node = v;
            q.num_labels = labels.NumLabels(v);
            for (int k = 0; k < std::min(q.num_labels, NODE_QUERY_LABELS); ++k) {
                q.labels[k] = litems[loff[v] + k];
            }
            q.position = &here[3 * static_cast<std::size_t>(v)];
            queries.push_back(q);
        }
        const std::vector<NodeTargetResult> res = targets.Evaluate(queries);
        double sums[2] = {0.0, 0.0};
        max_dev = 0.0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const int v = queries[i].node;
            if (res[i].status != TargetStatus::FIT || owned[v] == 0) {
                continue;
            }
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                d2 += (res[i].target[d] - queries[i].position[d]) *
                      (res[i].target[d] - queries[i].position[d]);
            }
            const double dev = std::sqrt(d2) / targets.MeshSize(v);
            max_dev = std::max(max_dev, dev);
            sums[0] += dev;
            sums[1] += 1.0;
        }
        MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &max_dev, 1, MPI_DOUBLE, MPI_MAX, comm);
        mean_dev = (sums[1] > 0.0) ? sums[0] / sums[1] : 0.0;
    };
    // Step 3: the starting deviation covers every fitted node. Then, for the passes of a
    // higher-order mesh, only the vertices (LINEAR_VERTICES) or only the other nodal points
    // (CURVATURE) keep their springs. pull[3 v + d] is where the spring of node v pulls: its
    // target, or for CURVATURE the closest point on its grain boundary from where it is now
    // (its current position when there is none, i.e. no pull).
    double unused_max = 0.0;
    deviation(unused_max, report.initial_deviation);
    std::vector<double> pull(t, t + 3 * static_cast<std::size_t>(nv));
    if (vertices_only) {
        for (int v = pmesh.GetNV(); v < nv; ++v) {
            marker[v] = false;
        }
    }
    if (curvature) {
        GatherPositions(*nodes, fes, nv, here);
        std::vector<NodeLabelQuery> queries;
        for (int v = 0; v < nv; ++v) {
            marker[v] = marker[v] && v >= pmesh.GetNV();
            if (!marker[v]) {
                continue;
            }
            NodeLabelQuery q;
            q.node = v;
            q.num_labels = labels.NumLabels(v);
            for (int k = 0; k < std::min(q.num_labels, NODE_QUERY_LABELS); ++k) {
                q.labels[k] = litems[loff[v] + k];
            }
            q.position = &here[3 * static_cast<std::size_t>(v)];
            queries.push_back(q);
        }
        const std::vector<NodeTargetResult> res = targets.Evaluate(queries);
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const int v = queries[i].node;
            for (int d = 0; d < 3; ++d) {
                pull[3 * static_cast<std::size_t>(v) + d] = (res[i].status == TargetStatus::FIT)
                                                                ? res[i].target[d]
                                                                : here[3 * v + d];
            }
        }
    }
    // The space of the displacement u: the Nodes space, or an order-1 space for LINEAR_VERTICES.
    // MFEM numbers the vertex dofs first in both, so dof v is node v in either.
    std::unique_ptr<mfem::H1_FECollection> linear_fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> linear_fes;
    if (vertices_only) {
        linear_fec = std::make_unique<mfem::H1_FECollection>(1, 3);
        linear_fes = std::make_unique<mfem::ParFiniteElementSpace>(
            &pmesh, linear_fec.get(), 3, fes.GetOrdering());
    }
    mfem::ParFiniteElementSpace& sfes = vertices_only ? *linear_fes : fes;
    const int sn = sfes.GetNDofs();

    // Step 4: the rounds (only when there is something to fit)
    if (report.fit_nodes > 0) {
        // Per-element fields (piecewise constant, L2 order 0): quality_factor = the stiffening
        // factor of the element, stiffness = that factor spread to the neighbors, used as the
        // Young's modulus. Per node: multiplier = the spring multiplier m_v.
        mfem::L2_FECollection l2(0, 3);
        mfem::ParFiniteElementSpace l2_fes(&pmesh, &l2);
        mfem::ParGridFunction stiffness(&l2_fes);
        mfem::ParGridFunction quality_factor(&l2_fes);
        quality_factor = 1.0;
        mfem::Vector multiplier(nv);
        multiplier = 1.0;
        const int* en_off = field.ElementVertexOffsets().HostRead();
        const int* en = field.ElementVertices().HostRead();
        std::vector<double> sj(static_cast<std::size_t>(ne), 1.0);
        std::vector<char> fails(static_cast<std::size_t>(ne), 0);

        // Fixed coordinates become essential (Dirichlet) true dofs with zero displacement. MFEM
        // solves on "true dofs": the dofs each rank owns, without the copies of shared ones.
        mfem::Array<int> ess_marker(sfes.GetVSize());
        ess_marker = 0;
        for (int v = 0; v < sn; ++v) {
            for (int d = 0; d < 3; ++d) {
                ess_marker[sfes.DofToVDof(v, d)] = fixed[3 * static_cast<std::size_t>(v) + d];
            }
        }
        mfem::Array<int> ess_tdof_marker;
        mfem::Array<int> ess_tdof;
        sfes.GetRestrictionMatrix()->BooleanMult(ess_marker, ess_tdof_marker);
        mfem::FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof);

        // Stiffness for the next solve: each element's quality factor, spread
        // `stiffness_smoothing` times (each node takes the largest factor of its elements over
        // all copies, then each element the mean of its nodes when that is larger)
        const auto update_stiffness = [&]() {
            for (int e = 0; e < ne; ++e) {
                stiffness(e) = quality_factor(e);
            }
            for (int pass = 0; pass < options.stiffness_smoothing; ++pass) {
                mfem::Vector nodal(nv);
                nodal = 0.0;
                for (int e = 0; e < ne; ++e) {
                    for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                        nodal(en[k]) = std::max(nodal(en[k]), stiffness(e));
                    }
                }
                nodal.Neg(); // maximum over the copies as the minimum of the negatives
                field.ReduceSharedMinimum(nodal);
                nodal.Neg();
                for (int e = 0; e < ne; ++e) {
                    double sum = 0.0;
                    for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                        sum += nodal(en[k]);
                    }
                    stiffness(e) = std::max(stiffness(e), sum / (en_off[e + 1] - en_off[e]));
                }
            }
        };

        // One spring-balanced elastic solve with spring factor c: assembles K on the input
        // geometry (the Nodes are reset to x0 first), adds the springs and the sliding penalty,
        // solves, sets the Nodes to x0 + u and puts the sliding nodes back on their surfaces
        mfem::ParGridFunction u(&sfes);
        mfem::ParGridFunction u_nodes(&fes);
        // Higher orders: a Gauss rule of order 2 p (27 points per order-2 hex) instead of MFEM's
        // default for curved elements, which is several times larger; enough for a mesh morph
        const int order = sfes.GetMaxElementOrder();
        const mfem::IntegrationRule* quad_rule =
            (order > 1) ? &mfem::IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * order)
                        : nullptr;
        const double nu = options.poisson_ratio;
        const auto solve = [&](double c) {
            *nodes = x0;
            update_stiffness();
            mfem::GridFunctionCoefficient e_coeff(&stiffness);
            mfem::ProductCoefficient mu(1.0 / (2.0 * (1.0 + nu)), e_coeff);
            mfem::ProductCoefficient lambda(nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), e_coeff);
            mfem::ParBilinearForm a(&sfes);
            auto* elasticity = new mfem::ElasticityIntegrator(lambda, mu);
            if (quad_rule != nullptr) {
                elasticity->SetIntRule(quad_rule);
            }
            a.AddDomainIntegrator(elasticity);
            a.Assemble();
            // Springs and their loads on the local vdofs, then on the true dofs (owned values
            // only, so the spring of a shared node counts once)
            mfem::ParGridFunction k_local(&sfes);
            mfem::ParGridFunction load_local(&sfes);
            k_local = 0.0;
            load_local = 0.0;
            for (int v = 0; v < sn; ++v) {
                if (!marker[v]) {
                    continue;
                }
                const double k = c * multiplier(v) * targets.MeshSize(v);
                for (int d = 0; d < 3; ++d) {
                    const int vd = sfes.DofToVDof(v, d);
                    k_local(vd) = k;
                    load_local(vd) = k * (pull[3 * static_cast<std::size_t>(v) + d] -
                                          x0(fes.DofToVDof(v, d)));
                }
            }
            u = 0.0;
            mfem::ParLinearForm rhs(&sfes);
            rhs = 0.0;
            mfem::HypreParMatrix A;
            mfem::Vector X;
            mfem::Vector B;
            a.FormLinearSystem(ess_tdof, u, rhs, A, X, B);
            mfem::Vector k_true(sfes.GetTrueVSize());
            mfem::Vector load_true(sfes.GetTrueVSize());
            sfes.GetRestrictionMatrix()->Mult(k_local, k_true);
            sfes.GetRestrictionMatrix()->Mult(load_local, load_true);
            for (const int i : ess_tdof) {
                k_true(i) = 0.0;
                load_true(i) = 0.0;
            }
            mfem::SparseMatrix diag(sfes.GetTrueVSize());
            for (int i = 0; i < sfes.GetTrueVSize(); ++i) {
                diag.Add(i, i, k_true(i));
            }
            // Normal penalty k n n^T of the sliding nodes, on the owner's true dofs (not on fixed
            // components, whose rows are already eliminated)
            for (int v = 0; v < sn; ++v) {
                const auto& pen = penalty[static_cast<std::size_t>(v)];
                if (pen[0] == 0.0 || owned[v] == 0) {
                    continue;
                }
                int td[3];
                for (int d = 0; d < 3; ++d) {
                    td[d] = fixed[3 * static_cast<std::size_t>(v) + d]
                                ? -1
                                : sfes.GetLocalTDofNumber(sfes.DofToVDof(v, d));
                }
                const double k = NORMAL_PENALTY * targets.MeshSize(v);
                for (int m = 0; m < static_cast<int>(pen[0]); ++m) {
                    const double* n = pen.data() + 1 + 3 * m;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            if (td[i] >= 0 && td[j] >= 0) {
                                diag.Add(td[i], td[j], k * n[i] * n[j]);
                            }
                        }
                    }
                }
            }
            diag.Finalize();
            mfem::HypreParMatrix P(comm, sfes.GlobalTrueVSize(), sfes.GetTrueDofOffsets(), &diag);
            std::unique_ptr<mfem::HypreParMatrix> AP(mfem::ParAdd(&A, &P));
            B += load_true;
            // l1 Gauss-Seidel (one of mechanics' preconditioners): the same fit as BoomerAMG in
            // systems mode at a fraction of the setup, which AMG pays again for every solve; it
            // needs more CG iterations (about 400 on a 72 000-hex strut lattice), hence the cap.
            // 1e-8 moves the nodes by far less than the targets' accuracy.
            mfem::HypreSmoother smoother(*AP, mfem::HypreSmoother::l1GS);
            smoother.SetPositiveDiagonal(true);
            mfem::CGSolver cg(comm);
            cg.SetRelTol(1e-8);
            cg.SetMaxIter(2000);
            cg.SetPrintLevel(0);
            cg.SetPreconditioner(smoother);
            cg.SetOperator(*AP);
            cg.Mult(B, X);
            a.RecoverFEMSolution(X, rhs, u);
            if (vertices_only) {
                u_nodes.ProjectGridFunction(u); // the order-1 field at every nodal point
            } else {
                u_nodes = u;
            }
            for (int i = 0; i < nodes->Size(); ++i) {
                (*nodes)(i) = x0(i) + u_nodes(i);
            }
            project_to_surfaces();
            ++report.solves;
        };

        // Quality limits per element, from its input quality. Hexes: the scaled Jacobian floor
        // (sj_limit) and the aspect ratio ceiling (ar_limit); tets: the mean ratio floor
        // (eta_limit; see MESH_MEAN_RATIO_FLOOR_TET for why tets use it). An element already
        // beyond a limit, or within POOR_INPUT_SLACK of it, gets a limit relative to its own input
        // value instead and is marked poor. Mediocre elements, for the bulk guard: hexes below
        // floor + BULK_MARGIN, tets below a mean ratio of TET_GUARD_MEAN_RATIO; input_below
        // counts those of the input, which the guard does not hold against the fit.
        const bool tets = pmesh.GetTypicalElementGeometry() == mfem::Geometry::TETRAHEDRON;
        const auto Mediocre = [&](int e, double q) {
            return tets ? ElementMeanRatio(pmesh, e) < TET_GUARD_MEAN_RATIO
                        : q < floor + BULK_MARGIN;
        };
        std::vector<double> eta_limit(static_cast<std::size_t>(ne), 0.0);
        std::vector<double> eta_e(static_cast<std::size_t>(ne), 1.0);
        std::vector<double> ar_e(static_cast<std::size_t>(ne), 1.0);
        std::vector<double> sj_limit(static_cast<std::size_t>(ne), floor);
        std::vector<double> ar_limit(static_cast<std::size_t>(ne), MESH_ASPECT_RATIO_MAX);
        std::vector<char> poor(static_cast<std::size_t>(ne), 0); // limit relaxed to the input
        double input_below = 0.0;
        for (int e = 0; e < ne; ++e) {
            double q = 0.0;
            double ar = 0.0;
            double det = 0.0;
            ElementQuality(pmesh, e, q, ar, det);
            const auto es = static_cast<std::size_t>(e);
            sj_limit[es] = std::min(floor, (1.0 - POOR_INPUT_SLACK) * q);
            const double ar_slack = (ar > MESH_ASPECT_RATIO_MAX) ? POOR_INPUT_ASPECT_SLACK
                                                                 : POOR_INPUT_SLACK;
            ar_limit[es] = std::max(MESH_ASPECT_RATIO_MAX, (1.0 + ar_slack) * ar);
            if (tets) {
                eta_limit[es] = std::min(MESH_MEAN_RATIO_FLOOR_TET,
                                         (1.0 - POOR_INPUT_SLACK) * ElementMeanRatio(pmesh, e));
            }
            poor[es] = (tets ? eta_limit[es] < MESH_MEAN_RATIO_FLOOR_TET
                             : sj_limit[es] < floor || ar_limit[es] > MESH_ASPECT_RATIO_MAX)
                           ? 1
                           : 0;
            input_below += Mediocre(e, q) ? 1.0 : 0.0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &input_below, 1, MPI_DOUBLE, MPI_SUM, comm);

        // good = the nodes of the last kept round (the input until one is kept); c = the spring
        // factor of the current round
        mfem::Vector good(x0);
        double c = options.initial_spring;
        double last_mean = report.initial_deviation;
        int weak_tries = 0;
        for (int round = 0; round < options.max_rounds; ++round) {
            bool ok = false;
            for (int attempt = 0; attempt < options.max_attempts && !ok; ++attempt) {
                solve(c);
                bool failed = false;
                for (int e = 0; e < ne; ++e) {
                    const auto es = static_cast<std::size_t>(e);
                    double det = 0.0;
                    ElementQuality(pmesh, e, sj[es], ar_e[es], det);
                    if (tets) {
                        eta_e[es] = ElementMeanRatio(pmesh, e);
                        fails[es] = det <= 0.0 || eta_e[es] < eta_limit[es];
                    } else {
                        fails[es] = det <= 0.0 || sj[es] < sj_limit[es] || ar_e[es] > ar_limit[es];
                    }
                    failed = failed || fails[es] != 0;
                }
                ok = !AnyRank(failed, comm);
                // After every solve: stiffen the distorted elements (never softened again). After
                // a failed solve also: double the factor of the poor input elements that failed,
                // and mark (hit) the nodes of every element near a limit; an element near a limit
                // without fitted nodes marks its nodes in `spread` instead (-1)
                mfem::Vector next(multiplier);
                mfem::Vector hit(nv);
                hit = 0.0;
                mfem::Vector spread(nv); // -1: a node of a near element without fitted nodes
                spread = 0.0;
                for (int e = 0; e < ne; ++e) {
                    const double q = sj[static_cast<std::size_t>(e)];
                    const double a = ar_e[static_cast<std::size_t>(e)];
                    if (options.stiffening_exponent > 0.0 && q < STIFFEN_BELOW) {
                        const double ratio = STIFFEN_BELOW / std::max(q, 0.01);
                        const double f = std::min(MAX_STIFFNESS,
                                                  std::pow(ratio, options.stiffening_exponent));
                        quality_factor(e) = std::max(quality_factor(e), f);
                    }
                    // A poor input element (its limit relaxed to its input value) that broke its
                    // limit resists more in the next attempt: thin elements, e.g. a pillow layer,
                    // otherwise keep shearing past their limit as their neighbors move
                    if (!ok && fails[static_cast<std::size_t>(e)] != 0 &&
                        poor[static_cast<std::size_t>(e)] != 0) {
                        quality_factor(e) = std::min(MAX_STIFFNESS, 2.0 * quality_factor(e));
                    }
                    const auto es = static_cast<std::size_t>(e);
                    const bool near = tets ? eta_e[es] < eta_limit[es] + BACKOFF_MARGIN
                                           : (q < sj_limit[es] + BACKOFF_MARGIN ||
                                              a > ar_limit[es] - ASPECT_RATIO_BACKOFF_MARGIN);
                    if (!ok && near) {
                        bool fitted = false;
                        for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                            hit(en[k]) = 1.0;
                            fitted = fitted || marker[en[k]];
                        }
                        for (int k = en_off[e]; k < en_off[e + 1] && !fitted; ++k) {
                            spread(en[k]) = -1.0;
                        }
                    }
                }
                // An element without fitted nodes is distorted through its neighbors: back off
                // the nodes of every element that touches one of its nodes (flags or-ed over the
                // node copies first, so every rank sees the same flags)
                field.ReduceSharedMinimum(spread);
                for (int e = 0; e < ne; ++e) {
                    bool touches = false;
                    for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                        touches = touches || spread(en[k]) < 0.0;
                    }
                    for (int k = en_off[e]; k < en_off[e + 1] && touches; ++k) {
                        hit(en[k]) = 1.0;
                    }
                }
                // Halve the springs of the fitted nodes of the elements near a limit
                for (int v = 0; v < nv; ++v) {
                    if (hit(v) > 0.0 && marker[v]) {
                        next(v) = 0.5 * multiplier(v);
                    }
                }
                field.ReduceSharedMinimum(next);
                multiplier = next;
            }
            if (!ok) {
                break;
            }
            // Bulk quality guard: a round that leaves more than BULK_SHARE of the elements newly
            // mediocre is not kept (quality first: the small extra fit is not worth it). A first
            // round that fails it is retried with a quarter of the spring.
            double below[2] = {0.0, static_cast<double>(ne)};
            for (int e = 0; e < ne; ++e) {
                below[0] += Mediocre(e, sj[static_cast<std::size_t>(e)]) ? 1.0 : 0.0;
            }
            MPI_Allreduce(MPI_IN_PLACE, below, 2, MPI_DOUBLE, MPI_SUM, comm);
            if (below[0] - input_below > BULK_SHARE * below[1]) {
                if (report.rounds == 0 && weak_tries < WEAK_TRIES) {
                    ++weak_tries;
                    c *= 0.25;
                    continue;
                }
                break;
            }
            // The round is kept
            good = *nodes;
            ++report.rounds;
            report.final_spring = c;
            double max_dev = 0.0;
            double mean_dev = 0.0;
            deviation(max_dev, mean_dev);
            // Weak early springs improve little, so the stop rule applies from the fourth round
            if (report.rounds >= 4 && mean_dev > (1.0 - options.min_improvement) * last_mean) {
                break;
            }
            last_mean = mean_dev;
            c *= 2.0;
        }
        *nodes = good;
    }

    // Step 5
    deviation(report.max_deviation, report.mean_deviation);
    report.quality_after = ComputeMeshQuality(pmesh);
    return report;
}

} // namespace

/*
 * FitConformal: one pass (ALL) for order-1 Nodes. For higher orders, two passes:
 *  1. LINEAR_VERTICES: the vertices are fitted with an order-1 displacement field, interpolated
 *     to every nodal point, so the elements move as straight-sided elements would. Pulling on
 *     single nodal points of a higher-order element instead makes a sharp local spike in its shape
 *     that folds the element.
 *  2. CURVATURE: with the vertices fixed, the other nodal points are pulled onto their grain
 *     boundaries, which bends the element faces to follow the curved boundary.
 * The report combines both: the input quality and starting deviation of the first pass, the
 * rest from the second, with rounds and solves summed.
 */
ConformalFitReport
FitConformal(mfem::ParMesh& pmesh, const NodeTargets& targets, const ConformalFitOptions& options) {
    if (targets.Labels().Field().LatticeOrder() == 1) {
        return FitPass(pmesh, targets, options, PassKind::ALL);
    }
    const ConformalFitReport first = FitPass(pmesh, targets, options, PassKind::LINEAR_VERTICES);
    ConformalFitReport report = FitPass(pmesh, targets, options, PassKind::CURVATURE);
    report.quality_before = first.quality_before;
    report.initial_deviation = first.initial_deviation;
    report.rounds += first.rounds;
    report.solves += first.solves;
    return report;
}

} // namespace exaconstit::microstructure
