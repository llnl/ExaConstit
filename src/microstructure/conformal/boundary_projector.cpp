/*
 * Boundary triangle patches of selected nodes and closest points on them (boundary_projector.hpp).
 */
#include "microstructure/conformal/boundary_projector.hpp"

#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace exaconstit::microstructure {

namespace {

double Dot(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** @brief p = a + s u + t w. */
void Combine(const double a[3], double s, const double u[3], double t, const double w[3],
             double p[3]) {
    for (int d = 0; d < 3; ++d) {
        p[d] = a[d] + s * u[d] + t * w[d];
    }
}

} // namespace

/*
 * ClosestPointOnTriangle: Ericson's Voronoi-region test (Real-Time Collision Detection, 2005,
 * section 5.1.5): the vertex regions, then the edge regions, then the interior.
 */
void ClosestPointOnTriangle(const double y[3], const double a[3], const double b[3],
                            const double c[3], double p[3], bool& interior) {
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

/*
 * Constructor
 *  1. Boundary faces (one element, not shared; or the faces marked in surface_faces) with a
 *     selected node: triangulate (file comment)
 *     and append the triangles to the list of every selected node of the face.
 *  2. Exchange the lists (VertexField::ExchangeSharedLists).
 *  3. Per selected node: own and received triangles, sorted lexicographically (9 doubles) and made
 *     distinct, into the CSR arrays.
 */
SurfacePatches::SurfacePatches(const VertexField& field, const std::vector<char>& selected,
                               const std::vector<char>* surface_faces) {
    const int nv = field.NumVertices();
    ThrowIfAnyRank(field.Comm(), selected.size() != static_cast<std::size_t>(nv),
                   "SurfacePatches: selected needs one entry per node");
    mfem::ParMesh& pmesh = field.Mesh();

    // Step 1
    std::vector<char> shared(static_cast<std::size_t>(pmesh.GetNumFaces()), 0);
    for (int i = 0; i < pmesh.GetNSharedFaces(); ++i) {
        shared[static_cast<std::size_t>(pmesh.GetSharedFace(i))] = 1;
    }
    const mfem::GridFunction* nodes = pmesh.GetNodes();
    const mfem::FiniteElementSpace* node_fes =
        field.UsesMeshNodes() ? nodes->FESpace() : nullptr;
    const int order = (nodes != nullptr) ? nodes->FESpace()->GetMaxElementOrder() : 1;
    std::vector<std::vector<double>> lists(static_cast<std::size_t>(nv));
    mfem::IsoparametricTransformation trans;
    mfem::Array<int> face_nodes;
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
        if (node_fes != nullptr) {
            node_fes->GetFaceDofs(f, face_nodes);
        } else {
            pmesh.GetFaceVertices(f, face_nodes);
        }
        bool any = false;
        for (int j = 0; j < face_nodes.Size(); ++j) {
            face_nodes[j] = (face_nodes[j] >= 0) ? face_nodes[j] : -1 - face_nodes[j];
            any = any || selected[static_cast<std::size_t>(face_nodes[j])] != 0;
        }
        if (!any) {
            continue;
        }
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
        std::vector<double> tris;
        for (int k = 0; k + corners <= sub.Size(); k += corners) {
            const int split[2][3] = {{0, 1, 2}, {0, 2, 3}};
            for (int h = 0; h < (corners == 4 ? 2 : 1); ++h) {
                for (int c = 0; c < 3; ++c) {
                    const double* q = &points[3 * static_cast<std::size_t>(sub[k + split[h][c]])];
                    tris.insert(tris.end(), q, q + 3);
                }
            }
        }
        for (int j = 0; j < face_nodes.Size(); ++j) {
            const auto v = static_cast<std::size_t>(face_nodes[j]);
            if (selected[v] != 0) {
                lists[v].insert(lists[v].end(), tris.begin(), tris.end());
            }
        }
    }

    // Step 2
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    for (int v = 0; v < nv; ++v) {
        const auto& list = lists[static_cast<std::size_t>(v)];
        send.insert(send.end(), list.begin(), list.end());
        send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
    }
    std::vector<int> recv_off;
    std::vector<double> recv;
    field.ExchangeSharedLists(send_off, send, recv_off, recv);

    // Step 3
    m_offsets.assign(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<std::array<double, 9>> tri;
    for (int v = 0; v < nv; ++v) {
        tri.clear();
        const auto vs = static_cast<std::size_t>(v);
        const auto add = [&](const double* data, int count) {
            for (int k = 0; k + 9 <= count; k += 9) {
                std::array<double, 9> t;
                std::copy(data + k, data + k + 9, t.begin());
                tri.push_back(t);
            }
        };
        add(lists[vs].data(), static_cast<int>(lists[vs].size()));
        add(recv.data() + recv_off[vs], recv_off[vs + 1] - recv_off[vs]);
        std::sort(tri.begin(), tri.end());
        tri.erase(std::unique(tri.begin(), tri.end()), tri.end());
        for (const auto& t : tri) {
            m_triangles.insert(m_triangles.end(), t.begin(), t.end());
        }
        m_offsets[vs + 1] = static_cast<int>(m_triangles.size());
    }
}

double SurfacePatches::ClosestPoint(int v, const double y[3], double p[3], double normal[3],
                                    bool& interior) const {
    double best = -1.0;
    for (int t = 0; t < NumTriangles(v); ++t) {
        const double* tri = Triangle(v, t);
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
                normal[d] /= len;
            }
        }
    }
    return (best < 0.0) ? -1.0 : std::sqrt(best);
}

} // namespace exaconstit::microstructure
