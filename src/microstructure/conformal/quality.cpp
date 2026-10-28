/*
 * Element quality at the corners (quality.hpp).
 */
#include "microstructure/conformal/quality.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "microstructure/util/collective_errors.hpp"

namespace exaconstit::microstructure {

namespace {

double Det3(const double a[3], const double b[3], const double c[3]) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
           a[2] * (b[0] * c[1] - b[1] * c[0]);
}

double Norm3(const double a[3]) {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

/**
 * @brief Inverse of the corner edge matrix of the regular unit tetrahedron (columns: the edges
 * (1, 0, 0), (1/2, sqrt(3)/2, 0), (1/2, sqrt(3)/6, sqrt(2/3)) from one corner), so that
 * E W^-1 is the identity, up to a rotation and scale, for a regular tetrahedron.
 */
mfem::DenseMatrix RegularTetInverse() {
    mfem::DenseMatrix w(3);
    const double cols[3][3] = {{1.0, 0.0, 0.0},
                               {0.5, std::sqrt(3.0) / 2.0, 0.0},
                               {0.5, std::sqrt(3.0) / 6.0, std::sqrt(2.0 / 3.0)}};
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            w(r, c) = cols[c][r];
        }
    }
    w.Invert();
    return w;
}

/**
 * @brief Scaled Jacobian and aspect ratio of the corner with edge vectors e (columns). The aspect
 * ratio is the ratio of the extreme singular values of E W^-1, W the corner edge matrix of the
 * ideal element (the identity for a hex corner, RegularTetInverse for a tet corner), so the
 * ideal element scores 1.
 */
void CornerQuality(const double e[3][3],
                   double scale,
                   const mfem::DenseMatrix* w_inv,
                   double& sj,
                   double& ar,
                   double& det) {
    det = Det3(e[0], e[1], e[2]);
    const double lengths = Norm3(e[0]) * Norm3(e[1]) * Norm3(e[2]);
    sj = (lengths > 0.0) ? scale * det / lengths : -1.0;
    mfem::DenseMatrix edges(3);
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            edges(r, c) = e[c][r];
        }
    }
    mfem::DenseMatrix m(3);
    if (w_inv != nullptr) {
        mfem::Mult(edges, *w_inv, m);
    } else {
        m = edges;
    }
    const double smax = m.CalcSingularvalue(0);
    const double smin = m.CalcSingularvalue(2);
    ar = (smin > 0.0) ? smax / smin : HUGE_VAL;
}

} // namespace

/*
 * ElementQuality: corners from the geometry's reference vertices (every nodal point for curved
 * elements). Hex: the Jacobian of the
 * element map at the corner (columns = edges along the reference axes). Tet: the edges from the
 * corner to the three others, ordered so that the determinant has the sign of the element map.
 */
void ElementQuality(
    mfem::Mesh& mesh, int e, double& scaled_jacobian, double& aspect_ratio, double& min_det) {
    const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(e);
    if (geom != mfem::Geometry::CUBE && geom != mfem::Geometry::TETRAHEDRON) {
        throw std::invalid_argument("ElementQuality: only hex and tet elements are supported");
    }
    mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
    const mfem::IntegrationRule* corners = mfem::Geometries.GetVertices(geom);
    scaled_jacobian = HUGE_VAL;
    aspect_ratio = 0.0;
    min_det = HUGE_VAL;
    // Curved elements (Nodes of order > 1): every nodal point of the element (its corners
    // included), where the edges are the Jacobian applied to the reference corner edges
    const mfem::GridFunction* nodes = mesh.GetNodes();
    const bool curved = nodes != nullptr && nodes->FESpace()->GetFE(e)->GetOrder() > 1;
    const mfem::IntegrationRule* points = curved ? &nodes->FESpace()->GetFE(e)->GetNodes()
                                                 : corners;
    if (geom == mfem::Geometry::CUBE) {
        for (int i = 0; i < points->GetNPoints(); ++i) {
            T->SetIntPoint(&points->IntPoint(i));
            const mfem::DenseMatrix& J = T->Jacobian();
            double edges[3][3];
            for (int c = 0; c < 3; ++c) {
                for (int r = 0; r < 3; ++r) {
                    edges[c][r] = J(r, c);
                }
            }
            double sj, ar, det;
            CornerQuality(edges, 1.0, nullptr, sj, ar, det);
            scaled_jacobian = std::min(scaled_jacobian, sj);
            aspect_ratio = std::max(aspect_ratio, ar);
            min_det = std::min(min_det, det);
        }
        return;
    }
    // Tet: physical corners, and corner i's edges to the others in an even permutation so that
    // det keeps the sign of det(x1 - x0, x2 - x0, x3 - x0)
    const int others[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    static const mfem::DenseMatrix w_inv = RegularTetInverse();
    if (curved) {
        // At every nodal point, the four corner patterns with edges J (r_k - r_i), r the
        // reference vertices (for a straight tet these are the physical corner edges)
        for (int q = 0; q < points->GetNPoints(); ++q) {
            T->SetIntPoint(&points->IntPoint(q));
            const mfem::DenseMatrix& J = T->Jacobian();
            for (int i = 0; i < 4; ++i) {
                const mfem::IntegrationPoint& ri = corners->IntPoint(i);
                double edges[3][3];
                for (int k = 0; k < 3; ++k) {
                    const mfem::IntegrationPoint& rk = corners->IntPoint(others[i][k]);
                    const double dr[3] = {rk.x - ri.x, rk.y - ri.y, rk.z - ri.z};
                    for (int d = 0; d < 3; ++d) {
                        edges[k][d] = J(d, 0) * dr[0] + J(d, 1) * dr[1] + J(d, 2) * dr[2];
                    }
                }
                double sj, ar, det;
                CornerQuality(edges, std::sqrt(2.0), &w_inv, sj, ar, det);
                scaled_jacobian = std::min(scaled_jacobian, sj);
                aspect_ratio = std::max(aspect_ratio, ar);
                min_det = std::min(min_det, det);
            }
        }
        return;
    }
    mfem::Vector p[4];
    for (int i = 0; i < 4; ++i) {
        p[i].SetSize(3);
        T->Transform(corners->IntPoint(i), p[i]);
    }
    for (int i = 0; i < 4; ++i) {
        double edges[3][3];
        for (int k = 0; k < 3; ++k) {
            for (int d = 0; d < 3; ++d) {
                edges[k][d] = p[others[i][k]](d) - p[i](d);
            }
        }
        double sj, ar, det;
        CornerQuality(edges, std::sqrt(2.0), &w_inv, sj, ar, det);
        scaled_jacobian = std::min(scaled_jacobian, sj);
        aspect_ratio = std::max(aspect_ratio, ar);
        min_det = std::min(min_det, det);
    }
}

/*
 * ElementMeanRatio: per corner (every nodal point for curved elements) S = E W^-1 as for the
 * aspect ratio, eta = 3 |det S|^(2/3) / |S|_F^2 with the sign of det S; the minimum.
 */
double ElementMeanRatio(mfem::Mesh& mesh, int e) {
    const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(e);
    if (geom != mfem::Geometry::CUBE && geom != mfem::Geometry::TETRAHEDRON) {
        throw std::invalid_argument("ElementMeanRatio: only hex and tet elements are supported");
    }
    mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
    const mfem::IntegrationRule* corners = mfem::Geometries.GetVertices(geom);
    const mfem::GridFunction* nodes = mesh.GetNodes();
    const bool curved = nodes != nullptr && nodes->FESpace()->GetFE(e)->GetOrder() > 1;
    const bool tet = geom == mfem::Geometry::TETRAHEDRON;
    const mfem::IntegrationRule* points = (curved || !tet)
                                              ? (curved ? &nodes->FESpace()->GetFE(e)->GetNodes()
                                                        : corners)
                                              : corners;
    static const mfem::DenseMatrix w_inv = RegularTetInverse();
    const int others[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    double eta = HUGE_VAL;
    mfem::DenseMatrix E(3);
    mfem::DenseMatrix S(3);
    const auto measure = [&](const mfem::DenseMatrix& m) {
        const double det = m.Det();
        const double fro2 = m.FNorm2();
        const double value = (fro2 > 0.0) ? 3.0 * std::cbrt(det * det) / fro2 : 0.0;
        eta = std::min(eta, (det > 0.0) ? value : -value);
    };
    for (int q = 0; q < points->GetNPoints(); ++q) {
        T->SetIntPoint(&points->IntPoint(q));
        const mfem::DenseMatrix& J = T->Jacobian();
        if (!tet) {
            measure(J);
            continue;
        }
        for (int i = 0; i < 4; ++i) {
            const mfem::IntegrationPoint& ri = corners->IntPoint(i);
            for (int k = 0; k < 3; ++k) {
                const mfem::IntegrationPoint& rk = corners->IntPoint(others[i][k]);
                const double dr[3] = {rk.x - ri.x, rk.y - ri.y, rk.z - ri.z};
                for (int d = 0; d < 3; ++d) {
                    E(d, k) = J(d, 0) * dr[0] + J(d, 1) * dr[1] + J(d, 2) * dr[2];
                }
            }
            mfem::Mult(E, w_inv, S);
            measure(S);
        }
    }
    return eta;
}

MeshQuality ComputeMeshQuality(mfem::ParMesh& pmesh) {
    const MPI_Comm comm = pmesh.GetComm();
    bool bad = false;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        const mfem::Geometry::Type g = pmesh.GetElementBaseGeometry(e);
        bad = bad || (g != mfem::Geometry::CUBE && g != mfem::Geometry::TETRAHEDRON);
    }
    ThrowIfAnyRank(comm, bad, "ComputeMeshQuality: only hex and tet elements are supported");
    double mins[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL}; // scaled Jacobian, det, mean ratio
    double max_ar = 0.0;
    std::int64_t counts[2] = {0, pmesh.GetNE()};
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        double sj, ar, det;
        ElementQuality(pmesh, e, sj, ar, det);
        mins[0] = std::min(mins[0], sj);
        mins[1] = std::min(mins[1], det);
        mins[2] = std::min(mins[2], ElementMeanRatio(pmesh, e));
        max_ar = std::max(max_ar, ar);
        counts[0] += (det <= 0.0) ? 1 : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, mins, 3, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &max_ar, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, comm);
    MeshQuality q;
    q.min_scaled_jacobian = mins[0];
    q.min_det = mins[1];
    q.min_mean_ratio = mins[2];
    q.max_aspect_ratio = max_ar;
    q.inverted = counts[0];
    q.num_elements = counts[1];
    return q;
}

} // namespace exaconstit::microstructure
