#include "microstructure/seeding/boundary_clearance.hpp"

#include "axom/primal/operators/closest_point.hpp"
#include "axom/primal/operators/squared_distance.hpp"

#include <algorithm>
#include <stdexcept>

namespace exaconstit::microstructure {

/*
 * BoundaryClearance constructor
 *
 * Steps:
 *  1. Mark the faces shared with other ranks (ParMesh only): ParMesh::GetSharedFace(i) returns the
 *     local face index of shared face i (pmesh.hpp:688).
 *  2. A local face f is a surface face if GetFaceElements gives no second element (e2 < 0) and f
 *     is not shared. (Before ParMesh::ExchangeFaceNbrData, shared faces also have e2 < 0, hence
 *     step 1.)
 *  3. Triangulate each surface face: map the points of an n-times subdivided reference face
 *     (mfem::GlobGeometryRefiner.Refine(face_geom, n)) through the face transformation and emit
 *     one triangle per sub-triangle, or two per sub-quadrilateral (corners 0-1-2 and 0-2-3).
 *     n = 1 for straight-sided meshes (the face itself), 2 * order for curved ones.
 *  4. Build the BVH over the triangle boxes (axom::spin::BVH::initialize, BVH.hpp:427).
 */
BoundaryClearance::BoundaryClearance(const mfem::Mesh& mesh, double clearance)
    : m_clearance(clearance) {
    if (!(clearance > 0.0)) {
        throw std::invalid_argument("BoundaryClearance: the clearance must be positive");
    }

    // Step 1: faces shared with other ranks are not surface faces
    std::vector<char> is_shared(static_cast<std::size_t>(mesh.GetNumFaces()), 0);
    if (const auto* pmesh = dynamic_cast<const mfem::ParMesh*>(&mesh)) {
        for (int s = 0; s < pmesh->GetNSharedFaces(); s++) {
            is_shared[static_cast<std::size_t>(pmesh->GetSharedFace(s))] = 1;
        }
    }

    // Subdivisions per face edge: 1 (the face itself) when straight-sided
    int subdivisions = 1;
    if (mesh.GetNodes() != nullptr) {
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
        if (e2 >= 0 || is_shared[static_cast<std::size_t>(f)]) {
            continue;
        }

        // Step 3a: physical positions of the refined reference points of the face
        mesh.GetFaceTransformation(f, &face_trans);
        const mfem::Geometry::Type geom = mesh.GetFaceGeometry(f);
        mfem::RefinedGeometry* refined = mfem::GlobGeometryRefiner.Refine(geom, subdivisions);
        face_points.resize(static_cast<std::size_t>(refined->RefPts.GetNPoints()));
        for (int q = 0; q < refined->RefPts.GetNPoints(); q++) {
            face_trans.Transform(refined->RefPts.IntPoint(q), x);
            face_points[static_cast<std::size_t>(q)] = PointType {x(0), x(1), x(2)};
        }

        // Step 3b: triangles from the sub-elements (RefGeoms lists 3 or 4 point indices each)
        const mfem::Array<int>& sub = refined->RefGeoms;
        if (geom == mfem::Geometry::TRIANGLE) {
            for (int s = 0; s + 2 < sub.Size(); s += 3) {
                m_triangles.emplace_back(face_points[sub[s]], face_points[sub[s + 1]],
                                         face_points[sub[s + 2]]);
            }
        } else if (geom == mfem::Geometry::SQUARE) {
            for (int s = 0; s + 3 < sub.Size(); s += 4) {
                m_triangles.emplace_back(face_points[sub[s]], face_points[sub[s + 1]],
                                         face_points[sub[s + 2]]);
                m_triangles.emplace_back(face_points[sub[s]], face_points[sub[s + 2]],
                                         face_points[sub[s + 3]]);
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
void BoundaryClearance::MarkNear(std::int64_t n, const double* x, const double* y,
                                 const double* z, std::uint64_t* near_mask) const {
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
        const PointType p {x[i], y[i], z[i]};
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
    m_bvh.findBoundingBoxes(offsets.view(), counts.view(), candidates, num_queries,
                            queries.data());

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
