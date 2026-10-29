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
 * @brief Local copies of the curved outer surface of the mesh around selected nodes, as small
 * patches of triangles, and the closest point on such a patch. Used by the conformal fit to keep
 * nodes on curved outer surfaces while they slide onto grain boundaries.
 *
 * **What it is for.** When the conformal fit moves a node onto a grain boundary, a node on the
 * outer surface of the part must stay on that surface, or the part would change shape. A flat
 * surface is a plane and is handled exactly (node_targets.hpp). A curved surface, such as the
 * strut of a lattice or the face of a curved (high-order) mesh, is known only through the mesh's
 * own boundary faces, so the node is kept on those faces: on a patch of triangles that covers the
 * boundary faces around it.
 *
 * **Patches and rings.** The patch of a node v is built from rings of boundary faces. The
 * one-ring is the set of boundary faces that contain v; the two-ring adds every boundary face
 * that touches a face of the one-ring, and so on (`rings` rings in all). A patch must contain
 * every point a node can be moved to, otherwise its closest point would stick to the patch edge.
 * A node moves at most NodeTargetOptions::max_move element sizes (2 for targets on the smoothed
 * staircase). The one-ring reaches about one element size from a vertex, but only half an element
 * size from a nodal point in the middle of an edge or face of a high-order element, hence
 * PatchRingsFor. ("Nodal point" means any node of the mesh's finite element space: the vertices,
 * and on a curved mesh of order p also the extra points on edges, faces and inside elements.)
 *
 * **The same patch on every rank.** A node on a partition boundary has a copy on each rank that
 * shares it, and every boundary face belongs to exactly one rank. The triangles of a shared node
 * are therefore collected from all ranks (VertexField::ExchangeSharedLists), sorted and stripped
 * of duplicates, ring after ring. Every copy of a node gets the same triangles in the same order,
 * so the closest points computed from them are identical to the last bit on every copy, for any
 * partition. (The alternative, one search structure over the whole outer surface, would need the
 * surfaces of the neighbor ranks on every rank to give the same answer on every copy.)
 *
 * **One patch per surface.** With PatchSurfaces, a node gets one patch for each boundary surface
 * it lies on (the surfaces of node_labels.hpp: groups of boundary faces separated by sharp edges,
 * such as a strut's side wall and the box face it ends on). Each triangle goes to the surface
 * whose normal is closest to its own; the absolute value of the cosine is used, so the
 * orientation of a face does not matter. A triangle beyond the one-ring is kept only when that
 * cosine reaches the cosine of the feature angle (the angle above which an edge counts as sharp),
 * so a wide patch does not spill over a sharp edge onto the neighboring surface (from a domed top
 * onto the side wall, or around a thin strut to its far side). A node on a sharp edge of a curved
 * boundary then has one patch per surface, and the edge is where the patches meet.
 *
 * **Triangles of a face.** A boundary face is mapped from its reference element to space (through
 * the mesh Nodes on a curved mesh) at the points of a grid with s subdivisions per edge
 * (mfem::GlobGeometryRefiner), and every small triangle of the grid becomes a patch triangle (a
 * small quadrilateral becomes the two triangles 0-1-2 and 0-2-3). s = 1 for a triangle face of a
 * straight-sided mesh (the face itself), s = 2 for a quadrilateral face of a straight-sided mesh
 * (four corners need not lie in one plane), and s = 2 p for a face of a mesh with order-p Nodes.
 *
 * **Closest point.** The closest point on a patch is the closest of the closest points on its
 * triangles, each found with Ericson's region test (C. Ericson, "Real-Time Collision Detection",
 * Morgan Kaufmann, 2005, section 5.1.5); a tie goes to the first triangle in the sorted order.
 *
 * Host only. The SurfacePatches constructor and MakeCurvedPatches are collective over the mesh
 * communicator; the queries are local.
 */

namespace exaconstit::microstructure {

/** @brief Most alternations of ProjectOnNodeSurfaces for a node on a sharp edge where a curved
 * surface meets another surface. */
inline constexpr int SURFACE_PROJECTION_ITERATIONS = 25;

/** @brief Largest number of face rings in a patch (wider patches cost memory and exchange time
 * and are not needed: nodes move at most 2 element sizes). */
inline constexpr int SURFACE_PATCH_MAX_RINGS = 3;

/**
 * @brief Number of face rings a patch needs so that it contains every point a node can be moved
 * to.
 *
 * A node moves at most `max_move` element sizes. The one-ring reaches about one element size from
 * a vertex but half an element size from a nodal point inside an edge or face of a high-order
 * element, so the rings are ceil(max_move) for order 1 and ceil(max_move + 1/2) for higher
 * orders, clamped to [1, SURFACE_PATCH_MAX_RINGS].
 * @param max_move largest node move, in element sizes (> 0).
 * @param order    polynomial order of the mesh Nodes (1 for a straight-sided mesh).
 */
inline int PatchRingsFor(double max_move, int order) {
    const double reach = (order > 1) ? max_move + 0.5 : max_move;
    const int rings = static_cast<int>(std::ceil(reach - 1e-12));
    return std::min(SURFACE_PATCH_MAX_RINGS, std::max(1, rings));
}

/**
 * @brief The boundary surfaces of every node, from NodeLabels (SurfaceOffsets, SurfaceNormals,
 * FeatureCosine). Given to SurfacePatches, a node gets one patch per surface (see the file
 * comment). The pointers are only used during the SurfacePatches construction.
 */
struct PatchSurfaces {
    /** @brief Surfaces of node v are offsets[v] .. offsets[v + 1] - 1 (an offsets array: the
     * per-node lists stored one after another, offsets[v] where the list of v starts). */
    const mfem::Array<int>* offsets = nullptr;
    /** @brief Unit outward normal of each surface, 3 doubles per surface in the order of
     * `offsets`. */
    const mfem::Vector* normals = nullptr;
    /** @brief Cosine of the feature angle: a triangle beyond the one-ring joins a surface only
     * when the |cosine| between their normals is at least this. */
    double min_cosine = 1.0;
};

/**
 * @brief Patches of boundary triangles around selected nodes, the same on every copy of a node,
 * and closest-point queries on them (see the file comment).
 */
class SurfacePatches {
public:
    /**
     * @brief Collects the patches of the selected nodes.
     *
     * Host only. Collective over the communicator of `field`.
     * @param field    node numbering, coordinates and the exchange between copies of shared nodes;
     *                 node v below is node v of this field.
     * @param selected one entry per node of the field: nonzero = build a patch for this node. Must
     *                 be the same on every copy of a node (e.g. derived from the node labels).
     * @param surface_faces optional, one entry per local mesh face: 1 = the face belongs to the
     *                 surfaces to cover, and this rank counts it (NodeLabels::SurfaceFaces: the
     *                 outer boundary plus the interfaces to fixed parts). nullptr = the outer
     *                 boundary faces (faces with one element that are not shared with another
     *                 rank).
     * @param rings    face rings per patch, 1 .. SURFACE_PATCH_MAX_RINGS (see PatchRingsFor).
     * @param surfaces optional: one patch per boundary surface of each selected node (see the file
     *                 comment); nullptr = one patch per selected node holding every triangle.
     * @throws std::invalid_argument on every rank if `selected` does not have one entry per node or
     *         `rings` is out of range.
     */
    SurfacePatches(const VertexField& field,
                   const std::vector<char>& selected,
                   const std::vector<char>* surface_faces = nullptr,
                   int rings = 1,
                   const PatchSurfaces* surfaces = nullptr);

    /** @brief Number of patches of node v: 0 when v is not selected; otherwise 1 without
     * PatchSurfaces, or the number of boundary surfaces of v with them (patch k = surface k of the
     * node). Host. */
    int NumPatches(int v) const {
        const auto vs = static_cast<std::size_t>(v);
        return m_patch_offsets[vs + 1] - m_patch_offsets[vs];
    }
    /** @brief Number of triangles in patch k of node v (0 when v has no patch k). Host. */
    int NumTriangles(int v, int k = 0) const {
        if (k >= NumPatches(v)) {
            return 0;
        }
        const auto p = static_cast<std::size_t>(m_patch_offsets[static_cast<std::size_t>(v)] + k);
        return (m_offsets[p + 1] - m_offsets[p]) / 9;
    }
    /** @brief Corners of triangle t (0-based, t < NumTriangles(v, k)) of patch k of node v: 9
     * doubles, x y z of the first, second and third corner (mesh length units). Host. */
    const double* Triangle(int v, int t, int k = 0) const {
        const auto p = static_cast<std::size_t>(m_patch_offsets[static_cast<std::size_t>(v)] + k);
        return m_triangles.data() + m_offsets[p] + 9 * t;
    }

    /**
     * @brief Closest point p to y on patch k of node v. Host, not collective.
     * @param v        a node with a patch k.
     * @param y        query point x, y, z (mesh length units).
     * @param p        out: the closest point on the patch.
     * @param normal   out: unit normal of the triangle that holds p (orientation of the face it
     *                 came from, not necessarily outward).
     * @param interior out: true when p lies inside that triangle, false on one of its edges or
     *                 corners (y then lies off the side of the patch or across a crease).
     * @param k        the patch (surface k of the node when built with PatchSurfaces).
     * @return the distance |y - p|, or a negative value when the patch is empty (p, normal and
     *         interior are then unchanged).
     */
    double ClosestPoint(
        int v, const double y[3], double p[3], double normal[3], bool& interior, int k = 0) const;

private:
    std::vector<int> m_patch_offsets; ///< patches of node v: m_patch_offsets[v] .. [v + 1] - 1
    std::vector<int> m_offsets;       ///< doubles of patch q: m_triangles[m_offsets[q] .. [q + 1])
    std::vector<double> m_triangles;  ///< all patch triangles, 9 doubles each (see Triangle)
};

/**
 * @brief The patches the conformal fit needs: for every node on a curved boundary surface (a node
 * on at most two surfaces, at least one of them curved: a smooth curved part, or a sharp edge
 * where a curved surface meets another surface), one patch per surface, `rings` face rings, over
 * the faces of NodeLabels::SurfaceFaces.
 *
 * Host only. Collective over the mesh communicator of `labels`.
 * @param labels node labels with their boundary surfaces.
 * @param rings  face rings per patch (PatchRingsFor).
 */
std::unique_ptr<SurfacePatches> MakeCurvedPatches(const NodeLabels& labels, int rings);

/**
 * @brief True when every curved surface of node v has a non-empty patch (also true for a node
 * without curved surfaces). A node for which this is false cannot be kept on its surface and is
 * held in place by the fit. Host, not collective.
 */
bool CurvedPatchesReady(const NodeLabels& labels, const SurfacePatches& patches, int v);

/**
 * @brief Puts a moved position y of node v back onto the boundary surfaces of v.
 *
 * Nodes on no boundary surface, or on three or more (a corner, which the fit never moves), are
 * left unchanged. For the others:
 *   - flat surfaces are the planes through the node's input position x0, and y moves to the
 *     nearest point on them (one plane: straight onto it; two planes: onto their line of
 *     intersection);
 *   - one curved surface: y moves to the closest point on its patch;
 *   - two surfaces with a curved one (a sharp edge): each curved surface is replaced by its
 *     tangent plane at the closest point on its patch (when that point lies on a patch edge, the
 *     plane normal to y - p), y moves onto the planes, and this alternates until both the step
 *     and the distance to every patch are at most `tol`, or `max_iterations` steps are done.
 * Finally, a coordinate fixed by an axis-aligned plane (a box face x = const, say) is restored
 * from x0 bit for bit, so box faces stay exactly flat.
 *
 * Host, not collective.
 * @param labels  node labels with the boundary surfaces of each node.
 * @param patches patches from MakeCurvedPatches (needed only for curved surfaces).
 * @param v       the node.
 * @param x0      input position of the node (3 doubles, mesh length units).
 * @param tol     convergence tolerance (length units).
 * @param max_iterations most alternations for a node on a sharp edge with a curved surface.
 * @param y       in: the moved position; out: the position on the node's surfaces.
 */
void ProjectOnNodeSurfaces(const NodeLabels& labels,
                           const SurfacePatches& patches,
                           int v,
                           const double* x0,
                           double tol,
                           int max_iterations,
                           double y[3]);

/**
 * @brief Closest point p to y on the triangle with corners a, b, c (Ericson's region test, see
 * the file comment). Host, not collective.
 * @param interior out: true when p lies strictly inside the triangle, false when it lies on an
 *                 edge or a corner.
 */
void ClosestPointOnTriangle(const double y[3],
                            const double a[3],
                            const double b[3],
                            const double c[3],
                            double p[3],
                            bool& interior);

} // namespace exaconstit::microstructure
