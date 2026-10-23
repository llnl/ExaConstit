#include "microstructure/seeding/material_point_locator.hpp"

#include "microstructure/mesh/straight_sided.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
 * Relative enlargement of the box of a curved element of a geometry without a reference lattice
 * (anything but hex and tet; the generator's pipeline only uses hexes and tets, so this is a
 * fallback): 10% of the box diagonal on every side, the heuristic used for all curved elements
 * before the sampled bulge margin. A safety margin, not a proven bound.
 */
constexpr double CURVED_BOX_PADDING = 0.10;

/**
 * Round-off allowance added to the bulge margin of a curved element box: this many machine
 * epsilons times the largest coordinate magnitude of the element, so that an element that is
 * straight in fact (zero sampled deviation) still gets a box that covers the round-off of the
 * map evaluation.
 */
constexpr double BOX_ROUNDOFF_EPS = 16.0;

/**
 * Relative mismatch accepted after the projected Newton solver: the mapped reference point must
 * reproduce p to this fraction of the element size (the solver can stop on the element boundary
 * for a point outside the element).
 */
constexpr double PROJECTED_MISMATCH_TOL = 1.0e-10;

/**
 * Early reject of the curved-element retries (PointInCurvedElement): attempts 3 and 4 are skipped
 * when attempts 1 and 2 converged to the same root of x(xi) = p, at most this distance outside
 * the reference element (reference units, element size 1), with the map clearly unfolded there. Box candidates
 * that are really outside (points of neighboring elements) converge to such roots. Interior
 * points of distorted elements can also have spurious roots this close (0.04 - 0.5 measured on
 * randomly perturbed order 2-4 meshes), but in those cases the root had a small or negative detJ
 * or the two starts disagreed (see the stress test CurvedGeometry.PerturbedElementsKeepInteriorPoints).
 */
constexpr double RETRY_REFERENCE_DISTANCE = 0.5;

/**
 * Early reject: the map at attempt 1's root must be clearly unfolded, detJ(root) > this fraction
 * of detJ at the element center (detJ shrinks toward 0 where the continued map folds back).
 */
constexpr double ROOT_DETJ_FRACTION = 0.2;

/** Reference-space distance below which two Newton roots count as the same root. */
constexpr double SAME_ROOT_TOL = 1.0e-8;

/**
 * Distance of reference point ip outside the reference hex or tet (<= 0 inside): the largest
 * violated constraint (x_d >= 0, x_d <= 1 for a cube; x_d >= 0, x + y + z <= 1 for a tet).
 * Returns +infinity for other geometries, so they never skip the retries.
 */
double ReferenceDistanceOutside(mfem::Geometry::Type geom, const mfem::IntegrationPoint& ip) {
    const double lower = std::max({-ip.x, -ip.y, -ip.z});
    if (geom == mfem::Geometry::CUBE) {
        return std::max({lower, ip.x - 1.0, ip.y - 1.0, ip.z - 1.0});
    }
    if (geom == mfem::Geometry::TETRAHEDRON) {
        return std::max(lower, ip.x + ip.y + ip.z - 1.0);
    }
    return std::numeric_limits<double>::infinity();
}

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
 * For curved elements the box is built by CurvedElementBox (see there for the bulge margin):
 * nodal points, the mapped 2p + 1 per direction reference lattice and the lattice edge midpoints,
 * enlarged by the sampled bulge margin. Only the mapped lattice points enter LocalPointBounds
 * (the same points as the refined grid used before the bulge margin, so the global mesh box of a
 * curved mesh did not change; only the element boxes did).
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
        // Reference lattices (n = 2p) of the geometries present on this rank
        for (int e = 0; e < m_num_elements; e++) {
            const mfem::Geometry::Type geom = mesh.GetElementGeometry(e);
            if (!m_lattices[geom].valid) {
                BuildLattice(geom, 2 * node_order);
            }
        }
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
        } else if (m_lattices[mesh.GetElementGeometry(e)].valid) {
            // Curved hex or tet: nodal points, lattice points and edge midpoints plus the sampled
            // bulge margin (CurvedElementBox)
            BoxType point_box;
            CurvedElementBox(e, box, point_box);
            m_local_point_bounds.addBox(point_box);
        } else {
            // Curved element of another geometry (fallback, not used by the generator's
            // pipeline): refined grid of mapped points plus nodal points, padded by 10% of the
            // diagonal
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
 * BuildLattice
 *
 * Builds the reference lattice of one geometry (see ReferenceLattice in the header) for n
 * subdivisions per direction:
 *  1. Points: (i, j, k) / n for the hex (all 0 <= i, j, k <= n, i fastest) and for the tet
 *     (i + j + k <= n, the same points as the uniform refinement of the reference tet with vertices
 *     (0,0,0), (1,0,0), (0,1,0), (0,0,1)). A table index_of(i, j, k) records each point's index.
 *  2. Edges: from every lattice point, one step along each lattice direction that stays in the
 *     lattice. The hex lattice has the 3 axis directions. The tet lattice is cut by the planes
 *     i, j, k = const and i + j + k = const into small tets and octahedra, whose edges point along
 *     the 6 directions (1,0,0), (0,1,0), (0,0,1), (-1,1,0), (-1,0,1), (0,-1,1).
 *  3. Midpoints: the reference midpoint of every edge, as an integration rule so that the whole
 *     set is mapped by one IsoparametricTransformation::Transform(IntegrationRule&, DenseMatrix&)
 *     call (eltrans.hpp:689).
 */
void MaterialPointLocator::BuildLattice(mfem::Geometry::Type geom, int n) {
    ReferenceLattice& lattice = m_lattices[geom];
    const bool is_hex = geom == mfem::Geometry::CUBE;
    if (!is_hex && geom != mfem::Geometry::TETRAHEDRON) {
        return; // no lattice: the constructor uses the padded fallback box
    }

    // Step 1: lattice points and the (i, j, k) -> point index table (-1 = not in the lattice)
    const int side = n + 1;
    std::vector<int> index_of(static_cast<std::size_t>(side * side * side), -1);
    auto table = [side](int i, int j, int k) {
        return static_cast<std::size_t>(i + side * (j + side * k));
    };
    std::vector<std::array<int, 3>> ijk;
    for (int k = 0; k <= n; k++) {
        for (int j = 0; j <= n; j++) {
            for (int i = 0; i <= n; i++) {
                if (!is_hex && i + j + k > n) {
                    continue;
                }
                index_of[table(i, j, k)] = static_cast<int>(ijk.size());
                ijk.push_back({i, j, k});
            }
        }
    }
    lattice.points.SetSize(static_cast<int>(ijk.size()));
    for (std::size_t q = 0; q < ijk.size(); q++) {
        mfem::IntegrationPoint& ip = lattice.points.IntPoint(static_cast<int>(q));
        ip.Set3(static_cast<double>(ijk[q][0]) / n, static_cast<double>(ijk[q][1]) / n,
                static_cast<double>(ijk[q][2]) / n);
        ip.weight = 0.0; // not a quadrature rule
    }

    // Step 2: lattice edges along the 3 (hex) or 6 (tet) lattice directions
    const int directions[6][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                  {-1, 1, 0}, {-1, 0, 1}, {0, -1, 1}};
    const int num_directions = is_hex ? 3 : 6;
    lattice.edge_a.clear();
    lattice.edge_b.clear();
    for (std::size_t q = 0; q < ijk.size(); q++) {
        for (int d = 0; d < num_directions; d++) {
            const int i = ijk[q][0] + directions[d][0];
            const int j = ijk[q][1] + directions[d][1];
            const int k = ijk[q][2] + directions[d][2];
            if (i < 0 || j < 0 || k < 0 || i > n || j > n || k > n) {
                continue;
            }
            const int other = index_of[table(i, j, k)];
            if (other < 0) {
                continue; // outside the tet lattice
            }
            lattice.edge_a.push_back(static_cast<int>(q));
            lattice.edge_b.push_back(other);
        }
    }

    // Step 3: reference midpoints of the edges
    const int num_edges = static_cast<int>(lattice.edge_a.size());
    lattice.midpoints.SetSize(num_edges);
    for (int m = 0; m < num_edges; m++) {
        const mfem::IntegrationPoint& a = lattice.points.IntPoint(lattice.edge_a[m]);
        const mfem::IntegrationPoint& b = lattice.points.IntPoint(lattice.edge_b[m]);
        mfem::IntegrationPoint& mid = lattice.midpoints.IntPoint(m);
        mid.Set3(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z));
        mid.weight = 0.0;
    }
    lattice.valid = true;
}

/*
 * CurvedElementBox
 *
 * Box of a curved hex or tet, and why the margin covers the element.
 *
 * Every point of the element is x(xi) for a reference point xi in some cell of the reference
 * lattice (a small hex of the hex lattice, or a small tet/octahedron of the tet lattice). Write
 * x(xi) = I(xi) + err(xi), where I is the piecewise-linear (hex cells: trilinear) interpolant of
 * the map at the lattice points of that cell. I(xi) is a weighted average of the cell's mapped
 * lattice points with non-negative weights, so it lies in the box of the mapped lattice points.
 * Hence x(xi) lies in that box enlarged by max |err| in each coordinate. For a map that is smooth
 * on the scale of the lattice, err is dominated by the second derivatives along the lattice
 * directions: along one lattice edge of reference length h the interpolation error of a quadratic
 * is h^2 / 8 times its second derivative, which is exactly the deviation measured at the edge
 * midpoint, x(mid) - (x(a) + x(b)) / 2. Inside a cell the error is at most the sum over the cell's
 * directions (3 for a hex cell, at most 1.5 times the largest edge for a tet cell), and
 * CURVED_BOX_MARGIN_FACTOR = 6 adds a factor 2 for the higher-order terms. The bound is sampled:
 * it rests on the largest second derivative being seen at some edge midpoint, which holds for
 * well-shaped elements but is not proven for arbitrary polynomial maps (hence "sampled bound").
 * The argument covers the hex cells and the corner tets of the tet lattice, whose directions are
 * all lattice edges. It does not cover the octahedra of the tet lattice: their interior diagonals
 * are not lattice edges, so the second derivative along them is never sampled. Empirical support
 * instead: a stress test (hex and tet, p = 2 - 4, every node moved at random by up to 30% of the
 * element size) found no element point outside its box.
 *
 * Steps:
 *  1. Element transformation through the Nodes (Mesh::GetElementTransformation). Its point
 *     matrix (IsoparametricTransformation::GetPointMat) holds the element's nodal coordinates,
 *     one column per dof: points of the element for a Lagrange basis, control points whose
 *     convex hull contains the element for a Bernstein basis.
 *  2. Map the lattice points and the edge midpoints (one Transform call each).
 *  3. Box of the nodal points, the lattice points and the midpoints; `point_box` = box of the
 *     lattice points only.
 *  4. Largest edge-midpoint deviation per coordinate, and the margin
 *     CURVED_BOX_MARGIN_FACTOR * deviation + BOX_ROUNDOFF_EPS * eps * max |coordinate|.
 */
void MaterialPointLocator::CurvedElementBox(int e, BoxType& box, BoxType& point_box) const {
    const ReferenceLattice& lattice = m_lattices[m_mesh.GetElementGeometry(e)];

    // Step 1: the Nodes transformation of the element
    mfem::IsoparametricTransformation trans;
    m_mesh.GetElementTransformation(e, &trans);
    const mfem::DenseMatrix& nodal = trans.GetPointMat();

    // Step 2: mapped lattice points and edge midpoints, 3 x n_points and 3 x n_edges
    mfem::DenseMatrix lattice_x;
    mfem::DenseMatrix midpoint_x;
    trans.Transform(lattice.points, lattice_x);
    trans.Transform(lattice.midpoints, midpoint_x);

    // Step 3: per-axis extremes of the lattice points, then of the midpoints and nodal points
    double lo[3];
    double hi[3];
    for (int d = 0; d < 3; d++) {
        lo[d] = std::numeric_limits<double>::infinity();
        hi[d] = -std::numeric_limits<double>::infinity();
    }
    auto include = [&lo, &hi](const mfem::DenseMatrix& pts) {
        for (int q = 0; q < pts.Width(); q++) {
            for (int d = 0; d < 3; d++) {
                lo[d] = std::min(lo[d], pts(d, q));
                hi[d] = std::max(hi[d], pts(d, q));
            }
        }
    };
    include(lattice_x);
    point_box = BoxType(PointType {lo[0], lo[1], lo[2]}, PointType {hi[0], hi[1], hi[2]});
    include(midpoint_x);
    include(nodal);

    // Step 4: largest deviation of the edge midpoints from the edge chords, per coordinate
    double deviation[3] = {0.0, 0.0, 0.0};
    for (int m = 0; m < midpoint_x.Width(); m++) {
        const int a = lattice.edge_a[static_cast<std::size_t>(m)];
        const int b = lattice.edge_b[static_cast<std::size_t>(m)];
        for (int d = 0; d < 3; d++) {
            const double chord_mid = 0.5 * (lattice_x(d, a) + lattice_x(d, b));
            deviation[d] = std::max(deviation[d], std::fabs(midpoint_x(d, m) - chord_mid));
        }
    }
    double magnitude = 0.0;
    for (int d = 0; d < 3; d++) {
        magnitude = std::max({magnitude, std::fabs(lo[d]), std::fabs(hi[d])});
    }
    const double roundoff =
        BOX_ROUNDOFF_EPS * std::numeric_limits<double>::epsilon() * magnitude;
    for (int d = 0; d < 3; d++) {
        const double margin = CURVED_BOX_MARGIN_FACTOR * deviation[d] + roundoff;
        lo[d] -= margin;
        hi[d] += margin;
    }
    box = BoxType(PointType {lo[0], lo[1], lo[2]}, PointType {hi[0], hi[1], hi[2]});
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
 * Straight-sided elements (unchanged since before curved elements were handled, so seeds on
 * straight meshes are bit for bit the same):
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
 * Curved elements (PointInCurvedElement below): an "Outside" answer is not trusted on its own,
 * because the polynomial map continued outside the reference element can have a second solution
 * there; see PointInCurvedElement.
 *
 * The reference tolerance is relaxed to 1e-12 so that convergence does not depend on reaching
 * 1e-15 for points near the origin (the physical tolerance 4e-15 * |p|_inf is ~0 there,
 * eltrans.cpp:178).
 */
bool MaterialPointLocator::PointInGeneralElement(int e, const std::array<double, 3>& p) const {
    if (!m_straight_sided) {
        return PointInCurvedElement(e, p);
    }

    // Straight-sided: the multilinear corner map, bitwise the transformation of the same mesh
    // without Nodes
    mfem::IsoparametricTransformation trans;
    StraightElementTransformation(m_mesh, e, trans);

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
    return mismatch <= PROJECTED_MISMATCH_TOL * element_size;
}

/*
 * PointInCurvedElement: Newton inverse map of a curved element, retried from several starts.
 *
 * Why retries. For a curved element, x(xi) is a polynomial of degree p in each reference
 * coordinate, defined (as a formula) also outside the reference element. x(xi) = p can then have
 * more than one solution: the one inside the element (when p is in the element) and others
 * outside it, where the continued map folds back. Plain Newton converges to whichever solution
 * is closest to its start in the Newton sense, so an "Outside" answer from one start only means
 * "this start found a solution outside", not "p is outside". MFEM itself documents that none of
 * its start/solver combinations guarantees finding an inside point (eltrans.hpp, the
 * InverseElementTransformation constructor comment). An "Inside" answer, in contrast, is
 * reliable: Newton converged (physical residual below 4e-15 |p| or step below 1e-12) to a
 * reference point that Geometry::CheckPoint accepts.
 *
 * Attempts, stopping at the first "Inside" (the answer does not depend on the order, except for
 * points within the round-off tolerances of the element boundary):
 *  1. plain Newton from the element center (InitGuessType Center);
 *  2. plain Newton from the reference position of the element's nodal point closest to p
 *     (the transformation's point matrix holds the nodal coordinates, column k belonging to the
 *     reference node GetFE()->GetNodes().IntPoint(k); cheap, no extra map evaluations);
 *  3. plain Newton from the reference lattice point (n = 2p, as in the element box) whose mapped
 *     position is closest to p (InitGuessType GivenPoint, eltrans.hpp:130);
 *  4. projected Newton (NewtonElementProject) from that lattice point, accepted only if the
 *     mapped reference point reproduces p to PROJECTED_MISMATCH_TOL times the element size
 *     (the projected solver can stop on the boundary for an outside point).
 * Only when all four fail is p declared outside.
 *
 * Early reject (cost). Most candidates tested here are outside the element: they lie in its box
 * but in a neighboring element. For them attempts 1 and 2 converge to the same true root of
 * x(xi) = p just outside the reference element, and attempts 3 and 4 (two more Newton solves and a
 * mapping of the (2p+1)^3 lattice) would only confirm it. So p is declared outside after attempt 2
 * when (a) attempt 1 converged (Outside) at most RETRY_REFERENCE_DISTANCE outside the element,
 * (b) the map is clearly unfolded there (detJ > ROOT_DETJ_FRACTION x detJ at the element center),
 * and (c) attempt 2, from a different start, converged
 * to the same root (within SAME_ROOT_TOL). Retries still run when attempt 1 did not converge,
 * converged far outside or near a fold of the continued map (small or negative detJ), or the two
 * starts disagree:
 * that is where spurious roots of interior points were found. The rule is a tested heuristic, not
 * a proof: a randomly perturbed order 2-4 stress test (test_curved_geometry,
 * PerturbedElementsKeepInteriorPoints) loses no interior point to it, while a distance-only rule
 * lost about 1 in 1000 there. A heavier stress run (order 2-4, node noise up to 45% of the
 * element size, about 386 000 interior points) lost one point to it, in an order-4 tet whose
 * node noise was 60% of its node spacing: there the continued map has a genuine second root
 * close to the element.
 */
bool MaterialPointLocator::PointInCurvedElement(int e, const std::array<double, 3>& p) const {
    mfem::IsoparametricTransformation trans;
    m_mesh.GetElementTransformation(e, &trans);

    mfem::Vector point(3);
    point(0) = p[0];
    point(1) = p[1];
    point(2) = p[2];
    mfem::IntegrationPoint ip;

    // Attempt 1: plain Newton from the element center
    mfem::InverseElementTransformation inverse(&trans);
    inverse.SetInitialGuessType(mfem::InverseElementTransformation::Center);
    inverse.SetSolverType(mfem::InverseElementTransformation::Newton);
    inverse.SetReferenceTol(1.0e-12);
    inverse.SetElementTol(REFERENCE_INSIDE_TOL);
    const int first = inverse.Transform(point, ip);
    if (first == mfem::InverseElementTransformation::Inside) {
        return true;
    }
    // Attempt 1's root, kept for the early reject below: converged (Outside) close to the
    // element, at a point where the map is clearly unfolded (detJ there above ROOT_DETJ_FRACTION
    // of its value at the center; preflight makes detJ positive inside the element, and it
    // decreases to 0 and below where the continuation folds back)
    const mfem::IntegrationPoint first_root = ip;
    bool first_near = false;
    if (first == mfem::InverseElementTransformation::Outside &&
        ReferenceDistanceOutside(m_mesh.GetElementGeometry(e), first_root) <=
            RETRY_REFERENCE_DISTANCE) {
        trans.SetIntPoint(&first_root);
        const double root_detj = trans.Weight();
        trans.SetIntPoint(&mfem::Geometries.GetCenter(m_mesh.GetElementGeometry(e)));
        first_near = root_detj > ROOT_DETJ_FRACTION * trans.Weight();
    }

    // Attempt 2: plain Newton from the reference node of the closest nodal point
    const mfem::DenseMatrix& nodal = trans.GetPointMat();
    const mfem::IntegrationRule& ref_nodes = trans.GetFE()->GetNodes();
    int closest_node = 0;
    double closest_node_dist = std::numeric_limits<double>::infinity();
    for (int k = 0; k < nodal.Width(); k++) {
        const double dist = point.DistanceSquaredTo(nodal.GetColumn(k));
        if (dist < closest_node_dist) {
            closest_node_dist = dist;
            closest_node = k;
        }
    }
    inverse.SetInitialGuess(ref_nodes.IntPoint(closest_node)); // switches to GivenPoint
    const int second = inverse.Transform(point, ip);
    if (second == mfem::InverseElementTransformation::Inside) {
        return true;
    }
    // Early reject: two different starts converged to the same unfolded root just outside the
    // element (see the function comment)
    if (first_near && second == mfem::InverseElementTransformation::Outside &&
        std::max({std::fabs(ip.x - first_root.x), std::fabs(ip.y - first_root.y),
                  std::fabs(ip.z - first_root.z)}) <= SAME_ROOT_TOL) {
        return false;
    }

    // Attempt 3: plain Newton from the closest mapped lattice point (hex/tet only; other
    // geometries use MFEM's own closest-point search on a grid of 2p + 1 points per direction)
    const ReferenceLattice& lattice = m_lattices[m_mesh.GetElementGeometry(e)];
    mfem::IntegrationPoint lattice_start;
    bool have_lattice_start = false;
    if (lattice.valid) {
        mfem::DenseMatrix lattice_x;
        trans.Transform(lattice.points, lattice_x);
        int closest = 0;
        double closest_dist = std::numeric_limits<double>::infinity();
        for (int q = 0; q < lattice_x.Width(); q++) {
            const double dist = point.DistanceSquaredTo(lattice_x.GetColumn(q));
            if (dist < closest_dist) {
                closest_dist = dist;
                closest = q;
            }
        }
        lattice_start = lattice.points.IntPoint(closest);
        have_lattice_start = true;
        inverse.SetInitialGuess(lattice_start);
    } else {
        inverse.SetInitialGuessType(mfem::InverseElementTransformation::ClosestPhysNode);
        inverse.SetInitGuessPointsType(mfem::Quadrature1D::ClosedUniform);
        inverse.SetInitGuessOrder(2 * std::max(1, trans.Order()));
    }
    if (inverse.Transform(point, ip) == mfem::InverseElementTransformation::Inside) {
        return true;
    }

    // Attempt 4: projected Newton from the same start, then check the mapped point
    if (have_lattice_start) {
        inverse.SetInitialGuess(lattice_start);
    }
    inverse.SetSolverType(mfem::InverseElementTransformation::NewtonElementProject);
    if (inverse.Transform(point, ip) != mfem::InverseElementTransformation::Inside) {
        return false;
    }
    mfem::Vector mapped(3);
    trans.Transform(ip, mapped);
    const auto range = m_element_boxes[e].range();
    const double element_size = std::max({range[0], range[1], range[2]});
    const double mismatch = std::max({std::fabs(mapped(0) - p[0]), std::fabs(mapped(1) - p[1]),
                                      std::fabs(mapped(2) - p[2])});
    return mismatch <= PROJECTED_MISMATCH_TOL * element_size;
}

} // namespace exaconstit::microstructure
