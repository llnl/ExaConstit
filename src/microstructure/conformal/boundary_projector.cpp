/*
 * Patches of boundary triangles around selected nodes, closest points on them, and the
 * projection of a moved node back onto its boundary surfaces (boundary_projector.hpp).
 */
#include "microstructure/conformal/boundary_projector.hpp"

#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace exaconstit::microstructure {

namespace {

/** @brief Dot product of two 3-vectors. */
double Dot(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** @brief p = a + s u + t w. */
void Combine(
    const double a[3], double s, const double u[3], double t, const double w[3], double p[3]) {
    for (int d = 0; d < 3; ++d) {
        p[d] = a[d] + s * u[d] + t * w[d];
    }
}

/** @brief The axis d when n points exactly along +-e_d (two components exactly zero, as for
 * the faces of an axis-aligned box), else -1. */
int AxisOf(const double n[3]) {
    int zeros = 0;
    int axis = -1;
    for (int d = 0; d < 3; ++d) {
        if (n[d] == 0.0) {
            ++zeros;
        } else {
            axis = d;
        }
    }
    return (zeros == 2) ? axis : -1;
}

/** @brief Two planes count as one when 1 - cos^2 of the angle between their normals is at most
 * this (normals within about 0.06 degrees): the two-plane formula below divides by 1 - cos^2. */
constexpr double PARALLEL_PLANES = 1e-6;

/**
 * @brief Moves y to the nearest point on one plane, or on the line where two planes meet.
 *
 * The planes are n_k . x = c_k with unit normals n_k (k < num, num = 1 or 2). The nearest point
 * is y - N^T (N N^T)^-1 (N y - c), N the matrix whose rows are the normals: the smallest
 * correction that satisfies every plane (the "least-norm" correction). For two planes N N^T is
 * the 2 x 2 matrix [1 g; g 1] with g = n_0 . n_1, inverted in closed form. Two (nearly) parallel
 * planes are the same plane for this purpose, and only the first is used.
 */
void ProjectOntoPlanes(const double n[2][3], const double c[2], int num, double y[3]) {
    double r[2] = {Dot(n[0], y) - c[0], (num == 2) ? Dot(n[1], y) - c[1] : 0.0};
    double a[2] = {r[0], 0.0};
    if (num == 2) {
        const double g = Dot(n[0], n[1]);
        const double det = 1.0 - g * g;
        if (!(det > PARALLEL_PLANES)) {
            num = 1; // (nearly) the same plane: the first one only
        } else {
            a[0] = (r[0] - g * r[1]) / det;
            a[1] = (r[1] - g * r[0]) / det;
        }
    }
    for (int d = 0; d < 3; ++d) {
        y[d] -= a[0] * n[0][d] + ((num == 2) ? a[1] * n[1][d] : 0.0);
    }
}

} // namespace

/*
 * ClosestPointOnTriangle: Ericson's region test (Real-Time Collision Detection, 2005, section
 * 5.1.5). Space around a triangle splits into seven regions by which feature is closest: each
 * corner, each edge, or the inside. The test uses dot products of the edge vectors with the
 * vectors from the corners to y (d1 .. d6) to find the region without computing any distance:
 * first the corner regions of a, b and c, then the edge regions ab, ac and bc (where the
 * barycentric-like ratios vc, vb, va are <= 0), and otherwise y projects inside the triangle and
 * p is its projection, with barycentric weights from va, vb, vc.
 */
void ClosestPointOnTriangle(const double y[3],
                            const double a[3],
                            const double b[3],
                            const double c[3],
                            double p[3],
                            bool& interior) {
    interior = false;
    const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const double ap[3] = {y[0] - a[0], y[1] - a[1], y[2] - a[2]};
    const double d1 = Dot(ab, ap);
    const double d2 = Dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        std::copy(a, a + 3, p);
        return;
    }
    const double bp[3] = {y[0] - b[0], y[1] - b[1], y[2] - b[2]};
    const double d3 = Dot(ab, bp);
    const double d4 = Dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        std::copy(b, b + 3, p);
        return;
    }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        Combine(a, d1 / (d1 - d3), ab, 0.0, ac, p);
        return;
    }
    const double cp[3] = {y[0] - c[0], y[1] - c[1], y[2] - c[2]};
    const double d5 = Dot(ab, cp);
    const double d6 = Dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        std::copy(c, c + 3, p);
        return;
    }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        Combine(a, 0.0, ab, d2 / (d2 - d6), ac, p);
        return;
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
        Combine(b, (d4 - d3) / ((d4 - d3) + (d5 - d6)), bc, 0.0, bc, p);
        return;
    }
    const double inv = 1.0 / (va + vb + vc);
    Combine(a, vb * inv, ab, vc * inv, ac, p);
    interior = true;
}

namespace {

/** @brief A triangle as 9 doubles: x y z of each of its three corners. */
using Tri = std::array<double, 9>;

/**
 * @brief Gives every copy of a shared node the union of the triangle lists of all its copies.
 *
 * Collective. VertexField::ExchangeSharedLists sends each node's list to the other ranks that
 * hold the node and returns what they sent; the received triangles are appended, and every list
 * is sorted (lexicographically on the 9 doubles) and stripped of duplicates, so all copies end
 * with the same list in the same order.
 */
void ExchangeTriangles(const VertexField& field, std::vector<std::vector<Tri>>& lists) {
    const int nv = field.NumVertices();
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    for (int v = 0; v < nv; ++v) {
        for (const Tri& t : lists[static_cast<std::size_t>(v)]) {
            send.insert(send.end(), t.begin(), t.end());
        }
        send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
    }
    std::vector<int> recv_off;
    std::vector<double> recv;
    field.ExchangeSharedLists(send_off, send, recv_off, recv);
    for (int v = 0; v < nv; ++v) {
        auto& list = lists[static_cast<std::size_t>(v)];
        for (int k = recv_off[static_cast<std::size_t>(v)];
             k + 9 <= recv_off[static_cast<std::size_t>(v) + 1];
             k += 9) {
            Tri t;
            std::copy(recv.data() + k, recv.data() + k + 9, t.begin());
            list.push_back(t);
        }
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
}

/** @brief Sets a per-node flag on every copy of a node when it is set on any copy (a logical
 * or over the copies). Collective. */
void ExchangeFlags(const VertexField& field, std::vector<char>& flag) {
    const int nv = field.NumVertices();
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    for (int v = 0; v < nv; ++v) {
        if (flag[static_cast<std::size_t>(v)] != 0) {
            send.push_back(1.0);
        }
        send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
    }
    std::vector<int> recv_off;
    std::vector<double> recv;
    field.ExchangeSharedLists(send_off, send, recv_off, recv);
    for (int v = 0; v < nv; ++v) {
        if (recv_off[static_cast<std::size_t>(v) + 1] > recv_off[static_cast<std::size_t>(v)]) {
            flag[static_cast<std::size_t>(v)] = 1;
        }
    }
}

/** @brief Unit normal of a triangle, (b - a) x (c - a) normalized (a zero vector for a
 * triangle of zero area). */
void TriangleNormal(const Tri& t, double n[3]) {
    const double u[3] = {t[3] - t[0], t[4] - t[1], t[5] - t[2]};
    const double w[3] = {t[6] - t[0], t[7] - t[1], t[8] - t[2]};
    n[0] = u[1] * w[2] - u[2] * w[1];
    n[1] = u[2] * w[0] - u[0] * w[2];
    n[2] = u[0] * w[1] - u[1] * w[0];
    const double len = std::sqrt(Dot(n, n));
    for (int d = 0; d < 3; ++d) {
        n[d] = (len > 0.0) ? n[d] / len : 0.0;
    }
}

} // namespace

/*
 * SurfacePatches constructor. The r-ring of a node is the union of the (r - 1)-rings of the
 * nodes on its boundary faces, so the rings are built from the inside out, each time over the
 * nodes that need them:
 *  1. The surface faces this rank counts (the outer boundary faces with one element that are not
 *     shared with another rank, or the faces marked in surface_faces) are cut into triangles (file
 *     comment); each face keeps the list of its nodes and its triangles.
 *  2. Which nodes need which ring: the selected nodes need ring `rings`; ring r - 1 is needed by
 *     every node that shares a surface face with a node needing ring r. The flags are or-ed over
 *     the copies of each node, so they are the same on every copy.
 *  3. One-rings of the nodes that need them: the triangles of every face around the node, then
 *     exchanged between copies, sorted and stripped of duplicates (ExchangeTriangles).
 *  4. Rings 2 .. `rings`: the union of the previous rings of the nodes of every face around the
 *     node, exchanged, sorted and stripped of duplicates. Each face is handled by exactly one
 *     rank, so after the exchange every copy has the same triangles.
 *  5. The patches of the selected nodes: without `surfaces`, one patch with every triangle of the
 *     ring; with them, one patch per surface of the node, each triangle going to the surface
 *     whose normal is closest to its own (largest |cosine|). A triangle outside the one-ring is
 *     kept only when that |cosine| reaches surfaces->min_cosine, so patches stop at sharp edges.
 */
SurfacePatches::SurfacePatches(const VertexField& field,
                               const std::vector<char>& selected,
                               const std::vector<char>* surface_faces,
                               int rings,
                               const PatchSurfaces* surfaces) {
    const int nv = field.NumVertices();
    ThrowIfAnyRank(field.Comm(),
                   selected.size() != static_cast<std::size_t>(nv),
                   "SurfacePatches: selected needs one entry per node");
    ThrowIfAnyRank(field.Comm(),
                   rings < 1 || rings > SURFACE_PATCH_MAX_RINGS,
                   "SurfacePatches: rings must be between 1 and SURFACE_PATCH_MAX_RINGS");
    mfem::ParMesh& pmesh = field.Mesh();

    // Step 1: triangulate the surface faces. shared[f] = 1 for faces on the partition boundary
    // (they have a second element on another rank, so they are not outer boundary faces).
    // face_node / face_tri hold, per face, its nodes (the field's numbering: the Nodes dofs on a
    // curved mesh, else the vertices) and its triangles, in offsets arrays face_*_off.
    std::vector<char> shared(static_cast<std::size_t>(pmesh.GetNumFaces()), 0);
    for (int i = 0; i < pmesh.GetNSharedFaces(); ++i) {
        shared[static_cast<std::size_t>(pmesh.GetSharedFace(i))] = 1;
    }
    const mfem::GridFunction* nodes = pmesh.GetNodes();
    const mfem::FiniteElementSpace* node_fes = field.UsesMeshNodes() ? nodes->FESpace() : nullptr;
    const int order = (nodes != nullptr) ? nodes->FESpace()->GetMaxElementOrder() : 1;
    std::vector<int> face_node_off(1, 0);
    std::vector<int> face_node;
    std::vector<int> face_tri_off(1, 0);
    std::vector<Tri> face_tri;
    mfem::IsoparametricTransformation trans;
    mfem::Array<int> fnodes;
    mfem::Vector X(3);
    std::vector<double> points;
    for (int f = 0; f < pmesh.GetNumFaces(); ++f) {
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        const bool use = (surface_faces != nullptr)
                             ? (*surface_faces)[static_cast<std::size_t>(f)] != 0
                             : (e2 < 0 && shared[static_cast<std::size_t>(f)] == 0);
        if (!use) {
            continue;
        }
        // The face's nodes. MFEM may return a dof as -1 - dof (its sign convention for some
        // spaces); only the dof number matters here, so negative entries are decoded.
        if (node_fes != nullptr) {
            node_fes->GetFaceDofs(f, fnodes);
        } else {
            pmesh.GetFaceVertices(f, fnodes);
        }
        for (int j = 0; j < fnodes.Size(); ++j) {
            face_node.push_back((fnodes[j] >= 0) ? fnodes[j] : -1 - fnodes[j]);
        }
        face_node_off.push_back(static_cast<int>(face_node.size()));
        // The face mapped to space at the points of a reference grid with `subdivisions` cells
        // per edge (file comment): points[3 i .. 3 i + 2] is grid point i
        const mfem::Geometry::Type geom = pmesh.GetFaceGeometry(f);
        int subdivisions = 2 * order;
        if (nodes == nullptr) {
            subdivisions = (geom == mfem::Geometry::TRIANGLE) ? 1 : 2;
        }
        pmesh.GetFaceTransformation(f, &trans);
        mfem::RefinedGeometry* refined = mfem::GlobGeometryRefiner.Refine(geom, subdivisions);
        const int np = refined->RefPts.GetNPoints();
        points.resize(3 * static_cast<std::size_t>(np));
        for (int i = 0; i < np; ++i) {
            trans.Transform(refined->RefPts.IntPoint(i), X);
            std::copy(X.GetData(), X.GetData() + 3, &points[3 * static_cast<std::size_t>(i)]);
        }
        // Sub-elements: 3 (triangles) or 4 (quadrilaterals) point indices each
        const int corners = (geom == mfem::Geometry::TRIANGLE) ? 3 : 4;
        const mfem::Array<int>& sub = refined->RefGeoms;
        for (int k = 0; k + corners <= sub.Size(); k += corners) {
            const int split[2][3] = {{0, 1, 2}, {0, 2, 3}};
            for (int h = 0; h < (corners == 4 ? 2 : 1); ++h) {
                Tri t;
                for (int c = 0; c < 3; ++c) {
                    const double* q = &points[3 * static_cast<std::size_t>(sub[k + split[h][c]])];
                    std::copy(q, q + 3, t.begin() + 3 * c);
                }
                face_tri.push_back(t);
            }
        }
        face_tri_off.push_back(static_cast<int>(face_tri.size()));
    }
    const int num_faces = static_cast<int>(face_node_off.size()) - 1;

    // Step 2: need[r - 1][v] != 0 when node v needs its r-ring. Starts with the selected nodes at
    // every level, then each level adds the nodes on the faces around the nodes of the level above.
    std::vector<std::vector<char>> need(static_cast<std::size_t>(rings), selected);
    for (int r = rings - 1; r >= 1; --r) {
        std::vector<char>& inner = need[static_cast<std::size_t>(r)];
        std::vector<char>& outer = need[static_cast<std::size_t>(r) - 1];
        for (int f = 0; f < num_faces; ++f) {
            bool any = false;
            for (int j = face_node_off[f]; j < face_node_off[f + 1]; ++j) {
                any = any || inner[static_cast<std::size_t>(face_node[j])] != 0;
            }
            for (int j = face_node_off[f]; j < face_node_off[f + 1] && any; ++j) {
                outer[static_cast<std::size_t>(face_node[j])] = 1;
            }
        }
        ExchangeFlags(field, outer);
    }

    // Step 3: one-rings, the triangles of every face around each node that needs one
    std::vector<std::vector<Tri>> one(static_cast<std::size_t>(nv));
    for (int f = 0; f < num_faces; ++f) {
        for (int j = face_node_off[f]; j < face_node_off[f + 1]; ++j) {
            const auto v = static_cast<std::size_t>(face_node[j]);
            if (need[0][v] != 0) {
                one[v].insert(one[v].end(),
                              face_tri.begin() + face_tri_off[f],
                              face_tri.begin() + face_tri_off[f + 1]);
            }
        }
    }
    ExchangeTriangles(field, one);

    // Step 4: ring r from the (r - 1)-rings of the nodes of the faces around each node
    std::vector<std::vector<Tri>> ring = one;
    for (int r = 2; r <= rings; ++r) {
        const std::vector<char>& here = need[static_cast<std::size_t>(r) - 1];
        std::vector<std::vector<Tri>> next(static_cast<std::size_t>(nv));
        for (int f = 0; f < num_faces; ++f) {
            for (int j = face_node_off[f]; j < face_node_off[f + 1]; ++j) {
                const auto v = static_cast<std::size_t>(face_node[j]);
                if (here[v] == 0) {
                    continue;
                }
                for (int i = face_node_off[f]; i < face_node_off[f + 1]; ++i) {
                    const auto& add = ring[static_cast<std::size_t>(face_node[i])];
                    next[v].insert(next[v].end(), add.begin(), add.end());
                }
            }
        }
        ExchangeTriangles(field, next);
        ring = std::move(next);
    }

    // Step 5: sort each selected node's triangles into its patches and append them to the
    // flat storage (m_patch_offsets per node, m_offsets per patch, m_triangles)
    const int* surf_off = (surfaces != nullptr) ? surfaces->offsets->HostRead() : nullptr;
    const double* surf_normal = (surfaces != nullptr) ? surfaces->normals->HostRead() : nullptr;
    m_patch_offsets.assign(static_cast<std::size_t>(nv) + 1, 0);
    m_offsets.assign(1, 0);
    for (int v = 0; v < nv; ++v) {
        const auto vs = static_cast<std::size_t>(v);
        int num_patches = 0;
        if (selected[vs] != 0) {
            num_patches = (surfaces != nullptr) ? surf_off[v + 1] - surf_off[v] : 1;
        }
        std::vector<std::vector<Tri>> patch(static_cast<std::size_t>(num_patches));
        for (const Tri& t : ring[vs]) {
            if (num_patches == 0) {
                break;
            }
            if (surfaces == nullptr) {
                patch[0].push_back(t);
                continue;
            }
            double n[3];
            TriangleNormal(t, n);
            int best = 0;
            double best_cos = -1.0;
            for (int k = 0; k < num_patches; ++k) {
                const double c = std::fabs(Dot(n, surf_normal + 3 * (surf_off[v] + k)));
                if (c > best_cos) {
                    best_cos = c;
                    best = k;
                }
            }
            if (best_cos >= surfaces->min_cosine ||
                std::binary_search(one[vs].begin(), one[vs].end(), t)) {
                patch[static_cast<std::size_t>(best)].push_back(t);
            }
        }
        for (const auto& list : patch) {
            for (const Tri& t : list) {
                m_triangles.insert(m_triangles.end(), t.begin(), t.end());
            }
            m_offsets.push_back(static_cast<int>(m_triangles.size()));
        }
        m_patch_offsets[vs + 1] = m_patch_offsets[vs] + num_patches;
    }
}

/*
 * SurfacePatches::ClosestPoint: the closest point on each triangle of the patch
 * (ClosestPointOnTriangle), keeping the nearest one; on equal distances the first triangle in
 * the sorted order wins, so every copy of a node returns the same point. Squared distances are
 * compared and the square root is taken once at the end.
 */
double SurfacePatches::ClosestPoint(
    int v, const double y[3], double p[3], double normal[3], bool& interior, int k) const {
    double best = -1.0;
    for (int t = 0; t < NumTriangles(v, k); ++t) {
        const double* tri = Triangle(v, t, k);
        double q[3];
        bool inside = false;
        ClosestPointOnTriangle(y, tri, tri + 3, tri + 6, q, inside);
        const double d2 = (y[0] - q[0]) * (y[0] - q[0]) + (y[1] - q[1]) * (y[1] - q[1]) +
                          (y[2] - q[2]) * (y[2] - q[2]);
        if (best < 0.0 || d2 < best) {
            best = d2;
            std::copy(q, q + 3, p);
            interior = inside;
            const double u[3] = {tri[3] - tri[0], tri[4] - tri[1], tri[5] - tri[2]};
            const double w[3] = {tri[6] - tri[0], tri[7] - tri[1], tri[8] - tri[2]};
            normal[0] = u[1] * w[2] - u[2] * w[1];
            normal[1] = u[2] * w[0] - u[0] * w[2];
            normal[2] = u[0] * w[1] - u[1] * w[0];
            const double len = std::sqrt(Dot(normal, normal));
            for (int d = 0; d < 3; ++d) {
                normal[d] = (len > 0.0) ? normal[d] / len : 0.0;
            }
        }
    }
    return (best < 0.0) ? -1.0 : std::sqrt(best);
}

} // namespace exaconstit::microstructure

namespace exaconstit::microstructure {

/*
 * MakeCurvedPatches: selects the nodes on a curved surface that the fit may move (on one or two
 * boundary surfaces, at least one of them curved; nodes on three or more surfaces are corners and
 * never move) and builds their patches, one per surface, from the node labels' surfaces, feature
 * cosine and surface faces.
 */
std::unique_ptr<SurfacePatches> MakeCurvedPatches(const NodeLabels& labels, int rings) {
    const int nv = labels.NumNodes();
    const int* off = labels.SurfaceOffsets().HostRead();
    const int* planar = labels.SurfacePlanar().HostRead();
    std::vector<char> selected(static_cast<std::size_t>(nv), 0);
    for (int v = 0; v < nv; ++v) {
        bool curved = false;
        for (int s = off[v]; s < off[v + 1]; ++s) {
            curved = curved || planar[s] == 0;
        }
        selected[static_cast<std::size_t>(v)] = (curved && off[v + 1] - off[v] <= 2) ? 1 : 0;
    }
    PatchSurfaces surfaces;
    surfaces.offsets = &labels.SurfaceOffsets();
    surfaces.normals = &labels.SurfaceNormals();
    surfaces.min_cosine = labels.FeatureCosine();
    return std::make_unique<SurfacePatches>(
        labels.Field(), selected, &labels.SurfaceFaces(), rings, &surfaces);
}

/*
 * CurvedPatchesReady: checks the patch of every curved surface of node v. A patch can be empty
 * when the node was not selected (more than two surfaces), so the node cannot follow that surface.
 */
bool CurvedPatchesReady(const NodeLabels& labels, const SurfacePatches& patches, int v) {
    const int* off = labels.SurfaceOffsets().HostRead();
    const int* planar = labels.SurfacePlanar().HostRead();
    for (int s = off[v]; s < off[v + 1]; ++s) {
        if (planar[s] == 0 && patches.NumTriangles(v, s - off[v]) == 0) {
            return false;
        }
    }
    return true;
}

/*
 * ProjectOnNodeSurfaces (see the header for the cases). Each pass replaces every surface of the
 * node by a plane: a flat surface by itself (through x0), a curved surface by the tangent plane
 * at the closest point p on its patch. When p lies on an edge or a corner of the patch (y beyond
 * the patch side or across a crease) the triangle normal is not the right direction, and the
 * plane normal to y - p through p is used instead. Then y moves to the nearest point on those
 * planes (ProjectOntoPlanes). With only flat surfaces one pass is exact; with a curved one the
 * tangent planes change as y moves, so the passes repeat until they settle.
 */
void ProjectOnNodeSurfaces(const NodeLabels& labels,
                           const SurfacePatches& patches,
                           int v,
                           const double* x0,
                           double tol,
                           int max_iterations,
                           double y[3]) {
    const int* off = labels.SurfaceOffsets().HostRead();
    const double* nrm = labels.SurfaceNormals().HostRead();
    const int* planar = labels.SurfacePlanar().HostRead();
    const int num = off[v + 1] - off[v];
    if (num == 0 || num > 2) {
        return;
    }
    bool curved = false;
    for (int s = off[v]; s < off[v + 1]; ++s) {
        curved = curved || planar[s] == 0;
    }
    if (curved && !CurvedPatchesReady(labels, patches, v)) {
        return;
    }
    // One curved surface: the closest point on its patch is the answer
    if (curved && num == 1) {
        double p[3];
        double pn[3];
        bool interior = false;
        patches.ClosestPoint(v, y, p, pn, interior);
        std::copy(p, p + 3, y);
        return;
    }
    // Planes n[k] . x = c[k] of the node's surfaces, rebuilt in every pass; dist_max is the
    // largest distance from y to a curved patch before the pass
    double n[2][3];
    double c[2];
    for (int it = 0; it < (curved ? max_iterations : 1); ++it) {
        double dist_max = 0.0;
        for (int k = 0; k < num; ++k) {
            const int s = off[v] + k;
            if (planar[s] != 0) {
                std::copy(nrm + 3 * s, nrm + 3 * s + 3, n[k]);
                c[k] = Dot(n[k], x0);
                continue;
            }
            double p[3];
            bool interior = false;
            const double dist = patches.ClosestPoint(v, y, p, n[k], interior, k);
            if (!interior && dist > tol) {
                for (int d = 0; d < 3; ++d) {
                    n[k][d] = (y[d] - p[d]) / dist;
                }
            }
            c[k] = Dot(n[k], p);
            dist_max = std::max(dist_max, dist);
        }
        const double before[3] = {y[0], y[1], y[2]};
        ProjectOntoPlanes(n, c, num, y);
        const double step2 = (y[0] - before[0]) * (y[0] - before[0]) +
                             (y[1] - before[1]) * (y[1] - before[1]) +
                             (y[2] - before[2]) * (y[2] - before[2]);
        if (step2 <= tol * tol && dist_max <= tol) {
            break;
        }
    }
    // Coordinates fixed by axis-aligned planes are restored exactly (the projection leaves them
    // only up to round-off)
    for (int s = off[v]; s < off[v + 1]; ++s) {
        const int axis = (planar[s] != 0) ? AxisOf(nrm + 3 * s) : -1;
        if (axis >= 0) {
            y[axis] = x0[axis];
        }
    }
}

} // namespace exaconstit::microstructure
