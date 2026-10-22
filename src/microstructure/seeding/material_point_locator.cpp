#include "microstructure/seeding/material_point_locator.hpp"

#include "microstructure/mesh/straight_sided.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace exaconstit::microstructure {

namespace {

/**
 * Tolerance of the closed-form tet test, relative to 1: a barycentric coordinate down to -1e-12
 * still counts as inside. It only has to cover round-off, so that a point on a face shared by two
 * tets is inside at least one of them.
 */
constexpr double TET_BARYCENTRIC_TOL = 1.0e-12;

/**
 * Reference-space tolerance of mfem::Geometry::CheckPoint after the Newton inverse map (the same
 * value MFEM uses by default, eltrans.hpp InverseElementTransformation constructor).
 */
constexpr double REFERENCE_INSIDE_TOL = 1.0e-8;

/**
 * Relative enlargement of the box of a curved element: 10% of the box diagonal on every side.
 * The unpadded box already contains every nodal point of the element and a refined grid of mapped
 * points, so the padding only has to cover the bulge of the polynomial map between those points,
 * which for well-shaped order-p elements is a small fraction of the element size. It is a safety
 * margin, not a proven bound.
 */
constexpr double CURVED_BOX_PADDING = 0.10;

/** Determinant of the 3x3 matrix with columns a, b, c. */
double Det3(const double a[3], const double b[3], const double c[3]) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
           a[2] * (b[0] * c[1] - b[1] * c[0]);
}

} // namespace

/*
 * MaterialPointLocator constructor
 *
 * Steps:
 *  1. Element-box method from the caller's straight-sided flag (IsStraightSided,
 *     mesh/straight_sided.hpp, decided over all ranks): corner boxes for straight-sided meshes
 *     (corners from the Nodes when present, else the mesh vertices, ElementCornerPoints),
 *     nodal + sampled + padded boxes otherwise.
 *  2. Compute one box per local element and their unions: LocalBounds (the boxes as used by the
 *     BVH, padded for curved elements; GenerateSeeds samples candidates from their global union,
 *     so every part of an element inside its padded box can be reached) and LocalPointBounds
 *     (only the evaluated points, the best estimate of the true extent).
 *  3. Build the Axom BVH over the boxes (axom::spin::BVH::initialize, BVH.hpp:427). The BVH
 *     itself enlarges each box by its default scale factor 1.000123 (BVH.hpp:411), which only
 *     adds candidates; the exact test decides.
 *
 * For curved elements the box covers:
 *  - the element's nodal coordinates (the Nodes grid function values at the element's vector
 *    dofs, FiniteElementSpace::GetElementVDofs, laid out x dofs, then y dofs, then z dofs). For a
 *    Lagrange (interpolating) basis these are points of the element; for a Bernstein (positive)
 *    basis they are control points whose convex hull contains the element, so their box is then a
 *    rigorous bound;
 *  - the points of an n = 2 * order times subdivided reference element
 *    (mfem::GlobGeometryRefiner.Refine(geom, n): (n+1)^3 points for a hex, (n+1)(n+2)(n+3)/6 for
 *    a tet), mapped to physical space by IsoparametricTransformation::Transform;
 * and is then padded by CURVED_BOX_PADDING. Only the mapped points enter LocalPointBounds.
 */
MaterialPointLocator::MaterialPointLocator(const mfem::Mesh& mesh, bool straight_sided)
    : m_mesh(mesh) {
    m_num_elements = mesh.GetNE();
    if (m_num_elements > 0 && (mesh.Dimension() != 3 || mesh.SpaceDimension() != 3)) {
        throw std::runtime_error("MaterialPointLocator: the mesh must be 3D");
    }

    // Step 1: the caller's collective decision. A mesh without Nodes is always straight-sided,
    // whatever the caller passed (nothing else could be meant).
    m_straight_sided = straight_sided || mesh.GetNodes() == nullptr;
    int node_order = 1;
    if (!m_straight_sided) {
        node_order = std::max(1, mesh.GetNodes()->FESpace()->GetMaxElementOrder());
    }

    // Step 2: per-element boxes and their union
    m_element_boxes.resize(m_num_elements);
    mfem::DenseMatrix corners;
    mfem::Array<int> vdofs;
    mfem::Vector nodal_values;
    mfem::IsoparametricTransformation trans;
    mfem::Vector x(3);
    for (int e = 0; e < m_num_elements; e++) {
        BoxType box;
        if (m_straight_sided) {
            // Corner box: exact for straight-sided elements (a linear tet or trilinear hex lies
            // in the convex hull of its corners). Without Nodes the corners are the vertices.
            ElementCornerPoints(mesh, e, corners);
            for (int k = 0; k < corners.Width(); k++) {
                box.addPoint(PointType {corners(0, k), corners(1, k), corners(2, k)});
            }
            m_local_point_bounds.addBox(box);
        } else {
            // Sampled points: map a refined grid of reference points (2 * order subdivisions)
            mesh.GetElementTransformation(e, &trans);
            const mfem::Geometry::Type geom = mesh.GetElementGeometry(e);
            mfem::RefinedGeometry* refined =
                mfem::GlobGeometryRefiner.Refine(geom, 2 * node_order);
            for (int q = 0; q < refined->RefPts.GetNPoints(); q++) {
                trans.Transform(refined->RefPts.IntPoint(q), x);
                box.addPoint(PointType {x(0), x(1), x(2)});
            }
            m_local_point_bounds.addBox(box);

            // Nodal coordinates: vdofs hold all x dofs, then all y dofs, then all z dofs
            const mfem::GridFunction* nodes = mesh.GetNodes();
            nodes->FESpace()->GetElementVDofs(e, vdofs);
            nodes->GetSubVector(vdofs, nodal_values);
            const int num_dofs = vdofs.Size() / 3;
            for (int k = 0; k < num_dofs; k++) {
                box.addPoint(PointType {nodal_values(k), nodal_values(k + num_dofs),
                                        nodal_values(k + 2 * num_dofs)});
            }

            // Pad by 10% of the diagonal to cover bulges between the points
            const auto range = box.range();
            const double diagonal =
                std::sqrt(range[0] * range[0] + range[1] * range[1] + range[2] * range[2]);
            box.expand(CURVED_BOX_PADDING * diagonal);
        }
        m_element_boxes[e] = box;
        m_local_bounds.addBox(box);
    }

    // Step 3: the BVH (only when there is at least one element; BVH::initialize needs > 0)
    if (m_num_elements > 0) {
        m_bvh.initialize(m_element_boxes.data(), m_num_elements);
    }
}

/*
 * MarkInside
 *
 * Steps:
 *  1. Keep only the points inside the local bounds (cheap rejection of points that belong to
 *     other ranks' partitions or to holes far from this partition).
 *  2. One batched BVH query (findPoints, BVH.hpp:343): for each kept point, the list of elements
 *     whose box contains it. Output layout: point j's candidate elements are
 *     candidates[offsets[j] .. offsets[j] + counts[j]), stored as element indices (the BVH maps
 *     its leaves back to the input box index, LinearBVH.hpp:451).
 *  3. Exact test against each candidate element; stop at the first element that contains the
 *     point and set its bit.
 */
void MaterialPointLocator::MarkInside(std::int64_t n, const double* x, const double* y,
                                      const double* z, std::uint64_t* inside_mask) const {
    if (m_num_elements == 0 || n == 0) {
        return;
    }

    // Step 1: points inside the local bounds, remembering their index in the batch
    std::vector<PointType> points;
    std::vector<std::int64_t> point_index;
    points.reserve(static_cast<std::size_t>(n));
    point_index.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; i++) {
        const PointType p {x[i], y[i], z[i]};
        if (m_local_bounds.contains(p)) {
            points.push_back(p);
            point_index.push_back(i);
        }
    }
    const axom::IndexType num_points = static_cast<axom::IndexType>(points.size());
    if (num_points == 0) {
        return;
    }

    // Step 2: candidate elements per point from the BVH (host arrays, SEQ execution space)
    axom::Array<axom::IndexType> offsets(num_points, num_points);
    axom::Array<axom::IndexType> counts(num_points, num_points);
    axom::Array<axom::IndexType> candidates;
    m_bvh.findPoints(offsets.view(), counts.view(), candidates, num_points, points.data());

    // Step 3: exact point-in-element test, first hit wins
    for (axom::IndexType j = 0; j < num_points; j++) {
        const std::array<double, 3> p = {points[j][0], points[j][1], points[j][2]};
        for (axom::IndexType c = 0; c < counts[j]; c++) {
            const int e = static_cast<int>(candidates[offsets[j] + c]);
            if (PointInElement(e, p)) {
                const std::int64_t i = point_index[j];
                inside_mask[i / 64] |= std::uint64_t(1) << (i % 64);
                break;
            }
        }
    }
}

/* PointInElement: closed form for straight-sided tets, Newton inverse map otherwise. */
bool MaterialPointLocator::PointInElement(int e, const std::array<double, 3>& p) const {
    if (m_straight_sided && m_mesh.GetElementGeometry(e) == mfem::Geometry::TETRAHEDRON) {
        return PointInLinearTet(e, p);
    }
    return PointInGeneralElement(e, p);
}

/*
 * PointInLinearTet: barycentric coordinates by signed volumes.
 *
 * With tet vertices v0..v3 and edge vectors a = v1-v0, b = v2-v0, c = v3-v0, the point
 * p = v0 + l1 a + l2 b + l3 c has l1 = det(p-v0, b, c) / det(a, b, c), and similarly for l2, l3
 * (Cramer's rule); l0 = 1 - l1 - l2 - l3. Each l_i is the signed volume of the sub-tet that
 * replaces vertex i by p, divided by the tet volume, so the point is inside exactly when all four
 * are >= 0. Dividing by det(a, b, c) makes the test independent of the vertex orientation.
 */
bool MaterialPointLocator::PointInLinearTet(int e, const std::array<double, 3>& p) const {
    // Corner points (vertices without Nodes, the Nodes vertex dofs otherwise)
    mfem::DenseMatrix corners;
    ElementCornerPoints(m_mesh, e, corners);
    const double* v0 = corners.GetColumn(0);
    const double* v1 = corners.GetColumn(1);
    const double* v2 = corners.GetColumn(2);
    const double* v3 = corners.GetColumn(3);

    // Edge vectors from v0 and the offset of the point from v0
    double a[3], b[3], c[3], r[3];
    for (int d = 0; d < 3; d++) {
        a[d] = v1[d] - v0[d];
        b[d] = v2[d] - v0[d];
        c[d] = v3[d] - v0[d];
        r[d] = p[d] - v0[d];
    }

    // Cramer's rule for (l1, l2, l3); a degenerate (zero-volume) tet contains nothing
    const double det = Det3(a, b, c);
    if (det == 0.0) {
        return false;
    }
    const double l1 = Det3(r, b, c) / det;
    const double l2 = Det3(a, r, c) / det;
    const double l3 = Det3(a, b, r) / det;
    const double l0 = 1.0 - l1 - l2 - l3;

    return l0 >= -TET_BARYCENTRIC_TOL && l1 >= -TET_BARYCENTRIC_TOL &&
           l2 >= -TET_BARYCENTRIC_TOL && l3 >= -TET_BARYCENTRIC_TOL;
}

/*
 * PointInGeneralElement: Newton inverse map with mfem::InverseElementTransformation.
 *
 * First attempt: plain (unprojected) Newton from the element center, solver type Newton
 * (eltrans.hpp SolverType). For a candidate inside or near a hex this converges in a few
 * iterations, and NewtonSolve then checks the reference point with
 * Geometry::CheckPoint(geom, ip, ip_tol) and returns Inside or Outside (eltrans.cpp:236, 320).
 * Unlike the projected solvers, plain Newton does not report "Inside" for a point outside the
 * element that converged onto the element boundary.
 *
 * Fallback (plain Newton returned Unknown, i.e. did not converge in 16 iterations): start from
 * the closest point of a reference grid (ClosestRefNode) and use NewtonElementProject, which keeps
 * the iterate inside the reference element and is more robust on distorted elements; "Unknown"
 * again counts as outside. The grid has max(trans_order + rel_order, 0) + 1 points per direction
 * (eltrans.hpp:336-343). MFEM's default rel_order is -1, which for a linear (order-1) element
 * gives a single point, the element center, i.e. no better than the first attempt; rel_order = 1
 * gives 3 points per direction for linear elements and order + 2 in general.
 *
 * The reference tolerance is relaxed to 1e-12 so that convergence does not depend on reaching
 * 1e-15 for points near the origin (the physical tolerance 4e-15 * |p|_inf is ~0 there,
 * eltrans.cpp:178).
 */
bool MaterialPointLocator::PointInGeneralElement(int e, const std::array<double, 3>& p) const {
    // Straight-sided: the multilinear corner map, bitwise the transformation of the same mesh
    // without Nodes; otherwise the mesh's own (Nodes) transformation
    mfem::IsoparametricTransformation trans;
    if (m_straight_sided) {
        StraightElementTransformation(m_mesh, e, trans);
    } else {
        m_mesh.GetElementTransformation(e, &trans);
    }

    mfem::Vector point(3);
    point(0) = p[0];
    point(1) = p[1];
    point(2) = p[2];
    mfem::IntegrationPoint ip;

    // First attempt: plain Newton from the element center
    mfem::InverseElementTransformation inverse(&trans);
    inverse.SetInitialGuessType(mfem::InverseElementTransformation::Center);
    inverse.SetSolverType(mfem::InverseElementTransformation::Newton);
    inverse.SetReferenceTol(1.0e-12);
    inverse.SetElementTol(REFERENCE_INSIDE_TOL);
    const int result = inverse.Transform(point, ip);
    if (result == mfem::InverseElementTransformation::Inside) {
        return true;
    }
    if (result == mfem::InverseElementTransformation::Outside) {
        return false;
    }

    // Fallback: projected Newton from the closest reference node
    inverse.SetInitialGuessType(mfem::InverseElementTransformation::ClosestRefNode);
    inverse.SetInitGuessRelOrder(1);
    inverse.SetSolverType(mfem::InverseElementTransformation::NewtonElementProject);
    const int retry = inverse.Transform(point, ip);
    if (retry != mfem::InverseElementTransformation::Inside) {
        return false;
    }
    // The projected solver can stop on the element boundary for an outside point; confirm that
    // the mapped reference point really reproduces p (to 1e-10 of the element size) before
    // accepting it.
    mfem::Vector mapped(3);
    trans.Transform(ip, mapped);
    const auto range = m_element_boxes[e].range();
    const double element_size = std::max({range[0], range[1], range[2]});
    const double mismatch = std::max({std::fabs(mapped(0) - p[0]), std::fabs(mapped(1) - p[1]),
                                      std::fabs(mapped(2) - p[2])});
    return mismatch <= 1.0e-10 * element_size;
}

} // namespace exaconstit::microstructure
