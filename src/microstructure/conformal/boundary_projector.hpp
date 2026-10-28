#pragma once

#include "microstructure/conformal/node_labels.hpp"
#include "microstructure/labeling/vertex_field.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
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
 * **Patches.** The *one-ring* of node v is the set of boundary faces that contain v, cut into
 * triangles; the *r-ring* adds the faces that touch the (r - 1)-ring (r = `rings`). A patch must
 * hold every point a target can reach: a target moves at most NodeTargetOptions::max_move element
 * sizes (2 for smoothed targets), and the one-ring reaches about one element size from a vertex
 * but only half of one from a mid-edge or mid-face nodal point of a curved mesh
 * (PatchRingsFor). A closest point outside the patch would stick to the patch border. Each
 * boundary face belongs to one rank only, so the triangles of a shared node are collected over
 * all ranks (VertexField::ExchangeSharedLists), sorted and made distinct, ring after ring (each
 * ring is the union of the previous rings of the face's nodes): every copy of v gets the same
 * triangles in the same order, and closest points computed from them are bitwise identical on
 * every copy, for any partition. (A global search structure over the whole boundary, the
 * alternative, would need the boundary of the neighbor ranks on every rank to give the same
 * answer on every copy.)
 *
 * **One patch per surface.** With PatchSurfaces (the boundary surfaces of the node labels,
 * node_labels.hpp) a selected node gets one patch per boundary surface at the node: a triangle
 * goes to the surface whose normal is closest to its own (|cos|, so a face's orientation does not
 * matter). A triangle outside the one-ring is kept only when that |cos| reaches the cosine of the
 * feature angle, so a larger ring does not spill over a feature edge onto another surface (the
 * side wall next to a domed top, the far side of a thin strut). On a feature edge of a curved
 * boundary (two surfaces) the node then has a patch per surface, and the edge is where they
 * meet (node_targets.hpp).
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

/** @brief Iterations of ProjectOnNodeSurfaces on a feature edge of a curved boundary. */
inline constexpr int SURFACE_PROJECTION_ITERATIONS = 25;

/** @brief Largest number of face rings of a patch. */
inline constexpr int SURFACE_PATCH_MAX_RINGS = 3;

/**
 * @brief Face rings a patch needs for targets that move up to `max_move` element sizes on a mesh
 * whose nodes are of order `order`: ceil(max_move) for order 1 (the one-ring reaches one element
 * size from a vertex), ceil(max_move + 1/2) for higher orders (half an element from a mid-edge
 * nodal point), at least 1 and at most SURFACE_PATCH_MAX_RINGS.
 */
inline int PatchRingsFor(double max_move, int order) {
    const double reach = (order > 1) ? max_move + 0.5 : max_move;
    const int rings = static_cast<int>(std::ceil(reach - 1e-12));
    return std::min(SURFACE_PATCH_MAX_RINGS, std::max(1, rings));
}

/**
 * @brief The boundary surfaces of every node (NodeLabels::SurfaceOffsets, SurfaceNormals and
 * FeatureCosine): with them a selected node gets one patch per surface (see the file comment).
 * Kept by pointer during the construction only.
 */
struct PatchSurfaces {
    const mfem::Array<int>* offsets = nullptr; ///< per node, CSR offsets into the surfaces
    const mfem::Vector* normals = nullptr;     ///< unit normal per surface, 3 doubles
    double min_cosine = 1.0;                   ///< cosine of the feature angle
};

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
     * @param surface_faces optional, per local face: 1 = a face of the surfaces, counted by this
     *                 rank only (NodeLabels::SurfaceFaces: the outer boundary plus the interface
     *                 faces of fixed labels). nullptr = the outer boundary faces (one element,
     *                 not shared).
     * @param rings    face rings of a patch, 1 .. SURFACE_PATCH_MAX_RINGS (PatchRingsFor).
     * @param surfaces optional: one patch per boundary surface of each selected node (the file
     *                 comment); nullptr = one patch per selected node with every triangle.
     * @throws std::invalid_argument on every rank if `selected` has the wrong size or `rings` is
     *         out of range.
     */
    SurfacePatches(const VertexField& field, const std::vector<char>& selected,
                   const std::vector<char>* surface_faces = nullptr, int rings = 1,
                   const PatchSurfaces* surfaces = nullptr);

    /** @brief Number of patches of node v: 0 when not selected, else 1 without PatchSurfaces
     * and the number of boundary surfaces of v with them (patch k = surface k of the node). */
    int NumPatches(int v) const {
        const auto vs = static_cast<std::size_t>(v);
        return m_patch_offsets[vs + 1] - m_patch_offsets[vs];
    }
    /** @brief Number of triangles of patch k of node v (0 for nodes not selected). */
    int NumTriangles(int v, int k = 0) const {
        if (k >= NumPatches(v)) {
            return 0;
        }
        const auto p = static_cast<std::size_t>(m_patch_offsets[static_cast<std::size_t>(v)] + k);
        return (m_offsets[p + 1] - m_offsets[p]) / 9;
    }
    /** @brief The 3 corners (9 doubles, x y z each) of triangle t of patch k of node v. */
    const double* Triangle(int v, int t, int k = 0) const {
        const auto p = static_cast<std::size_t>(m_patch_offsets[static_cast<std::size_t>(v)] + k);
        return m_triangles.data() + m_offsets[p] + 9 * t;
    }

    /**
     * @brief Closest point p of y on patch k of node v.
     * @param v        a selected node with a non-empty patch k.
     * @param y        query point.
     * @param p        out: the closest point.
     * @param normal   out: unit normal of the triangle holding p.
     * @param interior out: true when p lies inside that triangle (not on an edge or corner).
     * @param k        the patch (surface k of the node with PatchSurfaces).
     * @return |y - p|, or a negative value for an empty patch.
     */
    double ClosestPoint(int v, const double y[3], double p[3], double normal[3], bool& interior,
                        int k = 0) const;

private:
    std::vector<int> m_patch_offsets; ///< per node, first patch (NV + 1)
    std::vector<int> m_offsets;       ///< per patch, offset into m_triangles (patches + 1)
    std::vector<double> m_triangles;  ///< 9 doubles per triangle
};

/**
 * @brief The patches of the nodes of `labels` on a curved boundary surface (at most two boundary
 * surfaces, at least one of them curved: a smooth curved part, or a feature edge where a curved
 * surface meets another surface): `rings` face rings, one patch per surface of the node
 * (PatchSurfaces from the labels), over NodeLabels::SurfaceFaces. **Collective.**
 */
std::unique_ptr<SurfacePatches> MakeCurvedPatches(const NodeLabels& labels, int rings);

/** @brief True when every curved surface of node v has a non-empty patch (true without curved
 * surfaces). Host. */
bool CurvedPatchesReady(const NodeLabels& labels, const SurfacePatches& patches, int v);

/**
 * @brief Puts y back on the boundary surfaces of node v (at most two; nothing for a node on three
 * or more, or on none): the planes of its flat surfaces through x0 (the least-norm correction),
 * and the patches of its curved surfaces (MakeCurvedPatches). One curved surface: the closest
 * point on its patch. Two surfaces with a curved one: the planes through the closest points on
 * the curved patches (tangent planes; the plane normal to y - p at a patch edge) and the flat
 * planes, alternated until the step and the distance to every patch are at most `tol` or
 * `max_iterations` steps are done. Axis-normal planes restore that coordinate of x0 bit for bit.
 * Host.
 */
void ProjectOnNodeSurfaces(const NodeLabels& labels, const SurfacePatches& patches, int v,
                           const double* x0, double tol, int max_iterations, double y[3]);

/**
 * @brief Closest point p of y on the triangle (a, b, c), with `interior` true when p is inside
 * the triangle (not on an edge or corner). Host.
 */
void ClosestPointOnTriangle(const double y[3], const double a[3], const double b[3],
                            const double c[3], double p[3], bool& interior);

} // namespace exaconstit::microstructure
