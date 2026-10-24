#pragma once

#include "microstructure/labeling/vertex_field.hpp"

#include "mfem.hpp"

#include <vector>

/**
 * @file boundary_projector.hpp
 * @brief The outer boundary of the mesh around selected nodes, as small triangle patches, and the
 * closest point on such a patch (Stage 2b of conformal fitting, curved boundaries).
 *
 * **What it is for.** A node where a grain boundary meets a curved part of the outer boundary
 * (the surface of a strut from a tomography scan, a curved mesh) must move along the grain
 * boundary and stay on the outer boundary. Flat boundaries are handled exactly by a plane
 * (node_targets.hpp); a curved boundary is known only through the mesh's own boundary faces, so
 * the target must lie on those faces.
 *
 * **Patches.** The patch of node v is the set of boundary faces that contain v (its *one-ring*),
 * cut into triangles. Targets move at most about one element size h (NodeTargetOptions::max_move),
 * so the one-ring is the part of the boundary a target can reach. Each boundary face belongs to
 * one rank only, so the triangles of a shared node are collected over all ranks
 * (VertexField::ExchangeSharedLists), sorted and made distinct: every copy of v gets the same
 * triangles in the same order, and closest points computed from them are bitwise identical on
 * every copy, for any partition. (A global search structure over the whole boundary, the
 * alternative, would need the boundary of the neighbor ranks on every rank to give the same
 * answer on every copy.)
 *
 * **Triangles of a face.** The face is mapped through its face transformation (the mesh Nodes
 * when present) at the points of a reference grid with s subdivisions per edge
 * (mfem::GlobGeometryRefiner), and every sub-triangle (a sub-quadrilateral as the two triangles
 * 0-1-2 and 0-2-3) becomes a triangle: s = 1 for a triangle of a mesh without Nodes (the face
 * itself), s = 2 for a quadrilateral without Nodes (a warped bilinear face is not flat), and
 * s = 2 p for a face of a mesh with order-p Nodes.
 *
 * **Closest point.** The closest point of y on the patch is the closest point over its triangles
 * (Ericson's region test, "Real-Time Collision Detection", 2005, section 5.1.5); ties go to the
 * first triangle in the sorted order.
 *
 * Host only, collective construction.
 */

namespace exaconstit::microstructure {

/** @brief Boundary triangle patches of selected nodes (see the file comment). */
class SurfacePatches {
public:
    /**
     * @brief Collects the patches of the selected nodes.
     *
     * **Collective** over the field's communicator.
     * @param field    node numbering, coordinates and exchange (node v of the field).
     * @param selected per node: nonzero = build its patch. Must be the same on every copy of a
     *                 node (e.g. derived from the node labels), size NumVertices().
     * @throws std::invalid_argument on every rank if `selected` has the wrong size.
     */
    SurfacePatches(const VertexField& field, const std::vector<char>& selected);

    /** @brief Number of triangles of node v's patch (0 for nodes not selected). */
    int NumTriangles(int v) const {
        const auto vs = static_cast<std::size_t>(v);
        return (m_offsets[vs + 1] - m_offsets[vs]) / 9;
    }
    /** @brief The 3 corners (9 doubles, x y z each) of triangle t of node v's patch. */
    const double* Triangle(int v, int t) const {
        return m_triangles.data() + m_offsets[static_cast<std::size_t>(v)] + 9 * t;
    }

    /**
     * @brief Closest point p of y on node v's patch.
     * @param v        a selected node with a non-empty patch.
     * @param y        query point.
     * @param p        out: the closest point.
     * @param normal   out: unit normal of the triangle holding p.
     * @param interior out: true when p lies inside that triangle (not on an edge or corner).
     * @return |y - p|, or a negative value for an empty patch.
     */
    double ClosestPoint(int v, const double y[3], double p[3], double normal[3],
                        bool& interior) const;

private:
    std::vector<int> m_offsets;      ///< per node, offset into m_triangles (NV + 1)
    std::vector<double> m_triangles; ///< 9 doubles per triangle
};

/**
 * @brief Closest point p of y on the triangle (a, b, c), with `interior` true when p is inside
 * the triangle (not on an edge or corner). Host.
 */
void ClosestPointOnTriangle(const double y[3], const double a[3], const double b[3],
                            const double c[3], double p[3], bool& interior);

} // namespace exaconstit::microstructure
