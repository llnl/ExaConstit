#include "microstructure/seeding/boundary_clearance.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "axom/primal/operators/closest_point.hpp"
#include "axom/primal/operators/squared_distance.hpp"
#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/mesh/vertex_coordinates.hpp"

namespace exaconstit::microstructure {

namespace {

/*
 * WarpedQuadSubdivisions
 *
 * Number of subdivisions per edge for a straight-sided (bilinear) quadrilateral face, chosen so
 * that the two-triangle split of every sub-quadrilateral stays within WARP_REL_TOL times the face
 * size of the bilinear surface.
 *
 * The bilinear face map with corners v0, v1, v2, v3 (counterclockwise) is
 *     x(s, t) = v0 + (v1 - v0) s + (v3 - v0) t + (v0 - v1 + v2 - v3) s t,
 * so its only non-planar part is the twist vector w = v0 - v1 + v2 - v3 times s t. The triangles
 * 0-1-2 and 0-2-3 contain the diagonal 0-2, whose midpoint is (v0 + v2) / 2, while the bilinear
 * surface passes through the face center (v0 + v1 + v2 + v3) / 4; the two differ by w / 4. Only
 * the part of that offset normal to the face moves the surface (for a planar quad, w lies in the
 * plane of the face and the bilinear surface is the flat quad itself, exactly covered by the two
 * triangles), so the warp is
 *     warp = |w . n| / 4,   n = unit normal from the cross product of the two diagonals.
 * Splitting the face into k x k sub-quadrilaterals divides the twist of each one by k^2 (the s t
 * term scales with the square of the sub-quadrilateral size), so k = ceil(sqrt(warp / (tol *
 * size))) brings every sub-quadrilateral's warp below tol * size. A planar face gives warp at the
 * round-off level and k = 1, i.e. exactly the two triangles used before warped faces were
 * subdivided, so straight-sided meshes with planar faces keep their surface bit for bit.
 * k is capped at WARP_MAX_SUBDIVISIONS; at the cap the remaining warp is warp / k^2.
 *
 * @param corners the 4 face corners in face order (columns of a 3 x 4 matrix).
 */
int WarpedQuadSubdivisions(const mfem::DenseMatrix& corners) {
    double twist[3];
    double diag_a[3];
    double diag_b[3];
    double size_sq = 0.0;
    for (int d = 0; d < 3; d++) {
        twist[d] = corners(d, 0) - corners(d, 1) + corners(d, 2) - corners(d, 3);
        diag_a[d] = corners(d, 2) - corners(d, 0);
        diag_b[d] = corners(d, 3) - corners(d, 1);
    }
    // Face size: the longer diagonal
    double len_a_sq = 0.0;
    double len_b_sq = 0.0;
    for (int d = 0; d < 3; d++) {
        len_a_sq += diag_a[d] * diag_a[d];
        len_b_sq += diag_b[d] * diag_b[d];
    }
    size_sq = std::max(len_a_sq, len_b_sq);
    // Normal from the diagonals' cross product (its length is twice the projected face area)
    const double normal[3] = {diag_a[1] * diag_b[2] - diag_a[2] * diag_b[1],
                              diag_a[2] * diag_b[0] - diag_a[0] * diag_b[2],
                              diag_a[0] * diag_b[1] - diag_a[1] * diag_b[0]};
    const double normal_len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                        normal[2] * normal[2]);
    if (!(normal_len > 0.0) || !(size_sq > 0.0)) {
        return 1; // degenerate face: nothing sensible to refine
    }
    const double warp = std::fabs(twist[0] * normal[0] + twist[1] * normal[1] +
                                  twist[2] * normal[2]) /
                        (4.0 * normal_len);
    const double allowed = BOUNDARY_WARP_REL_TOL * std::sqrt(size_sq);
    if (warp <= allowed) {
        return 1;
    }
    const int k = static_cast<int>(std::ceil(std::sqrt(warp / allowed)));
    return std::min(k, BOUNDARY_WARP_MAX_SUBDIVISIONS);
}

} // namespace

/*
 * BoundaryClearance constructor
 *
 * Steps:
 *  0. Straight path only: owner-broadcast vertex positions (VertexCoordinates,
 *     mesh/vertex_coordinates.hpp; collective for a ParMesh). Each rank's own copy of the Nodes
 *     can differ in the last bits at a vertex shared with another rank after a parallel
 *     refinement; with the broadcast, the corners (and therefore the warp test and the
 *     triangles) of a surface face are the same whichever rank's data they come from. For a mesh
 *     distributed from a serial mesh the values are unchanged, bit for bit.
 *  1. Mark the faces shared with other ranks (ParMesh only): ParMesh::GetSharedFace(i) returns the
 *     local face index of shared face i (pmesh.hpp:688).
 *  2. A local face f is a surface face if GetFaceElements gives no second element (e2 < 0), f
 *     is not shared, and its element is active under the mask. (Before
 *     ParMesh::ExchangeFaceNbrData, shared faces also have e2 < 0, hence step 1.) A face between
 *     an active and an inactive element has two elements, so it is never a surface face.
 *  3. Triangulate each surface face: map the points of an n-times subdivided reference face
 *     (mfem::GlobGeometryRefiner.Refine(face_geom, n)) through the face transformation and emit
 *     one triangle per sub-triangle, or two per sub-quadrilateral (corners 0-1-2 and 0-2-3).
 *     - Straight-sided meshes: the corner transformation StraightFaceTransformation with the
 *       owner-broadcast corners of step 0 (so a mesh with straight-sided Nodes gives the
 *       triangles of the same mesh without Nodes). n = 1 for
 *       triangles and planar quadrilaterals (the face itself); a warped (non-planar bilinear)
 *       quadrilateral gets n = WarpedQuadSubdivisions(corners) >= 1, so the triangles follow the
 *       bilinear surface to BOUNDARY_WARP_REL_TOL of the face size instead of cutting across it.
 *     - Curved meshes: n = 2 * order through the Nodes face transformation (already subdivided;
 *       the error is the sag of one sub-triangle, see the header).
 *  4. Build the BVH over the triangle boxes (axom::spin::BVH::initialize, BVH.hpp:427).
 */
BoundaryClearance::BoundaryClearance(const mfem::Mesh& mesh,
                                     double clearance,
                                     bool straight_sided,
                                     const ElementMask* mask)
    : m_clearance(clearance) {
    if (!(clearance > 0.0)) {
        throw std::invalid_argument("BoundaryClearance: the clearance must be positive");
    }
    if (mask != nullptr && mask->active.Size() != 0 && mask->active.Size() != mesh.GetNE()) {
        throw std::invalid_argument("BoundaryClearance: the element mask needs one entry per "
                                    "local element");
    }

    // Step 1: faces shared with other ranks are not surface faces
    std::vector<char> is_shared(static_cast<std::size_t>(mesh.GetNumFaces()), 0);
    if (const auto* pmesh = dynamic_cast<const mfem::ParMesh*>(&mesh)) {
        for (int s = 0; s < pmesh->GetNSharedFaces(); s++) {
            is_shared[static_cast<std::size_t>(pmesh->GetSharedFace(s))] = 1;
        }
    }

    // Subdivisions per face edge: 1 (the face itself) when straight-sided
    const bool straight = straight_sided || mesh.GetNodes() == nullptr;

    // Owner-broadcast vertex positions for the straight path (collective for a ParMesh, so it is
    // built here, before any step that could throw on one rank only; `straight` is the same on
    // every rank: straight_sided comes from the collective IsStraightSided and every rank of a
    // ParMesh has Nodes or none does)
    std::unique_ptr<VertexCoordinates> coords;
    if (straight) {
        coords = std::make_unique<VertexCoordinates>(mesh);
    }

    int subdivisions = 1;
    if (!straight) {
        subdivisions = 2 * std::max(1, mesh.GetNodes()->FESpace()->GetMaxElementOrder());
    }

    mfem::IsoparametricTransformation face_trans;
    mfem::Vector x(3);
    std::vector<PointType> face_points;
    for (int f = 0; f < mesh.GetNumFaces(); f++) {
        // Step 2: keep only faces on the outer surface of the material
        int e1 = -1;
        int e2 = -1;
        mesh.GetFaceElements(f, &e1, &e2);
        if (e2 >= 0 || is_shared[static_cast<std::size_t>(f)] || !ElementIsActive(mask, e1)) {
            continue;
        }

        // Step 3a: physical positions of the refined reference points of the face
        const mfem::Geometry::Type geom = mesh.GetFaceGeometry(f);
        int face_subdivisions = subdivisions;
        if (straight) {
            StraightFaceTransformation(mesh, *coords, f, face_trans);
            // Warped bilinear quadrilaterals are subdivided; the point matrix of the corner
            // transformation holds the 4 corners in face order
            if (geom == mfem::Geometry::SQUARE) {
                face_subdivisions = WarpedQuadSubdivisions(face_trans.GetPointMat());
            }
        } else {
            mesh.GetFaceTransformation(f, &face_trans);
        }
        mfem::RefinedGeometry* refined = mfem::GlobGeometryRefiner.Refine(geom, face_subdivisions);
        face_points.resize(static_cast<std::size_t>(refined->RefPts.GetNPoints()));
        for (int q = 0; q < refined->RefPts.GetNPoints(); q++) {
            face_trans.Transform(refined->RefPts.IntPoint(q), x);
            face_points[static_cast<std::size_t>(q)] = PointType{x(0), x(1), x(2)};
        }

        // Step 3b: triangles from the sub-elements (RefGeoms lists 3 or 4 point indices each)
        const mfem::Array<int>& sub = refined->RefGeoms;
        if (geom == mfem::Geometry::TRIANGLE) {
            for (int s = 0; s + 2 < sub.Size(); s += 3) {
                m_triangles.emplace_back(
                    face_points[sub[s]], face_points[sub[s + 1]], face_points[sub[s + 2]]);
            }
        } else if (geom == mfem::Geometry::SQUARE) {
            for (int s = 0; s + 3 < sub.Size(); s += 4) {
                m_triangles.emplace_back(
                    face_points[sub[s]], face_points[sub[s + 1]], face_points[sub[s + 2]]);
                m_triangles.emplace_back(
                    face_points[sub[s]], face_points[sub[s + 2]], face_points[sub[s + 3]]);
            }
        } else {
            throw std::runtime_error("BoundaryClearance: unsupported face geometry");
        }
    }

    // Step 4: triangle boxes, their union and the BVH
    m_boxes.resize(m_triangles.size());
    for (std::size_t t = 0; t < m_triangles.size(); t++) {
        BoxType box;
        for (int k = 0; k < 3; k++) {
            box.addPoint(m_triangles[t][k]);
        }
        m_boxes[t] = box;
        m_bounds.addBox(box);
    }
    if (!m_triangles.empty()) {
        m_bvh.initialize(m_boxes.data(), static_cast<axom::IndexType>(m_boxes.size()));
    }
}

/*
 * MarkNear
 *
 * Steps:
 *  1. Query box of point i: the cube [p - c, p + c]. Points whose cube misses the union of the
 *     local triangle boxes cannot be near the local surface and are skipped.
 *  2. One batched BVH query (findBoundingBoxes, BVH.hpp:395): the triangles whose box intersects
 *     each query cube. Layout as for findPoints: candidates[offsets[j] .. offsets[j] + counts[j]).
 *  3. Exact distance: axom::primal::closest_point(p, triangle) (closest_point.hpp:164) gives the
 *     closest point q of the triangle, and |p - q|^2 < c^2 marks the point as too close.
 */
void BoundaryClearance::MarkNear(std::int64_t n,
                                 const double* x,
                                 const double* y,
                                 const double* z,
                                 std::uint64_t* near_mask) const {
    if (m_triangles.empty() || n == 0) {
        return;
    }

    // Step 1: query cubes of the points that can be near the local surface
    std::vector<BoxType> queries;
    std::vector<PointType> query_points;
    std::vector<std::int64_t> point_index;
    BoxType reach = m_bounds;
    reach.expand(m_clearance);
    for (std::int64_t i = 0; i < n; i++) {
        const PointType p{x[i], y[i], z[i]};
        if (!reach.contains(p)) {
            continue;
        }
        BoxType cube(p);
        cube.expand(m_clearance);
        queries.push_back(cube);
        query_points.push_back(p);
        point_index.push_back(i);
    }
    const axom::IndexType num_queries = static_cast<axom::IndexType>(queries.size());
    if (num_queries == 0) {
        return;
    }

    // Step 2: candidate triangles per point
    axom::Array<axom::IndexType> offsets(num_queries, num_queries);
    axom::Array<axom::IndexType> counts(num_queries, num_queries);
    axom::Array<axom::IndexType> candidates;
    m_bvh.findBoundingBoxes(offsets.view(), counts.view(), candidates, num_queries, queries.data());

    // Step 3: exact squared distance to each candidate triangle
    const double clearance_sq = m_clearance * m_clearance;
    for (axom::IndexType j = 0; j < num_queries; j++) {
        for (axom::IndexType c = 0; c < counts[j]; c++) {
            const auto t = static_cast<std::size_t>(candidates[offsets[j] + c]);
            const TriangleType& tri = m_triangles[t];
            const PointType closest = axom::primal::closest_point(query_points[j], tri);
            if (axom::primal::squared_distance(query_points[j], closest) < clearance_sq) {
                const std::int64_t i = point_index[j];
                near_mask[i / 64] |= std::uint64_t(1) << (i % 64);
                break;
            }
        }
    }
}

} // namespace exaconstit::microstructure
