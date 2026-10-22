#pragma once

#include "mfem.hpp"

#include "axom/primal/geometry/BoundingBox.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Triangle.hpp"
#include "axom/spin/BVH.hpp"

#include <array>
#include <cstdint>
#include <vector>

/**
 * @file boundary_clearance.hpp
 * @brief Tests whether points are closer than a clearance distance to the outer surface of the
 * material, for the `boundary_clearance` seeding option.
 *
 * A seed very close to the free surface gives a thin sliver grain along the surface. With
 * boundary_clearance = c > 0, candidates whose distance to the material surface is below c are
 * rejected.
 *
 * **How.** The surface is the set of *boundary faces*: faces of the global mesh with an element on
 * one side only. On a rank's partition these are the local faces with no second local element
 * that are not shared with another rank (faces between partitions are interior to the material,
 * not surface). Each boundary face is split into flat triangles (one per triangle face, two per
 * quadrilateral face, or a finer triangulation of a curved face), and a BVH (bounding volume
 * hierarchy: a tree of nested bounding boxes that lets a query skip every triangle whose box is
 * provably too far away) over the triangle bounding boxes finds the triangles near a point: a
 * triangle can only be within c of p if its box intersects the cube [p - c, p + c]. For those,
 * `axom::primal::closest_point(p, triangle)` gives the exact closest point and therefore the
 * exact distance.
 *
 * Every rank tests every point against its own part of the surface; a point is "too close" if it
 * is too close to the surface on ANY rank, so the caller combines the per-rank answers with a
 * bitwise OR over ranks (MPI_Allreduce with MPI_BOR, a *collective* call: one that every rank
 * must make, in the same order, because it exchanges data between ranks). Nothing in this class
 * is collective itself.
 *
 * Curved faces (meshes with Nodes) are approximated by triangulating the face at its refined
 * reference points (2 * order subdivisions per edge); the distance error is of the order of the
 * curvature sag of one sub-triangle.
 *
 * Host only (axom::SEQ_EXEC), like the material point locator.
 */

namespace exaconstit::microstructure {

/**
 * @brief Distance-to-surface test over the boundary faces of one rank's mesh partition.
 *
 * Construction is local to the rank (not collective). Host only.
 */
class BoundaryClearance {
public:
    using ExecSpace = axom::SEQ_EXEC;
    using PointType = axom::primal::Point<double, 3>;
    using BoxType = axom::primal::BoundingBox<double, 3>;
    using TriangleType = axom::primal::Triangle<double, 3>;

    /**
     * @brief Collects and triangulates the local boundary faces and builds the triangle BVH.
     *
     * @param mesh      this rank's 3D mesh partition. If it is an mfem::ParMesh, faces shared with
     *                  other ranks (ParMesh::GetSharedFace) are excluded; for a plain mfem::Mesh
     *                  every face with one element is a boundary face.
     * @param clearance c > 0, mesh length units.
     * @param straight_sided IsStraightSided(mesh, comm) (mesh/straight_sided.hpp): straight-sided
     *        faces are triangulated from their corner vertices (from the Nodes when present),
     *        exactly as for a mesh without Nodes; curved faces are subdivided 2 * order times.
     *        Ignored (treated as true) for a mesh without Nodes.
     */
    BoundaryClearance(const mfem::Mesh& mesh, double clearance, bool straight_sided);

    /** @brief Number of local surface triangles. */
    std::size_t NumTriangles() const {
        return m_triangles.size();
    }

    /**
     * @brief Sets bit i of `near_mask` for every point i whose distance to a local surface
     * triangle is < c. Bit layout as in MaterialPointLocator::MarkInside (word i / 64, bit i % 64);
     * other bits are unchanged. Host only, not collective.
     */
    void MarkNear(std::int64_t n, const double* x, const double* y, const double* z,
                  std::uint64_t* near_mask) const;

private:
    double m_clearance;                     ///< c
    std::vector<TriangleType> m_triangles;  ///< local surface triangles
    std::vector<BoxType> m_boxes;           ///< bounding box of each triangle
    BoxType m_bounds;                       ///< union of the triangle boxes
    axom::spin::BVH<3, ExecSpace> m_bvh;    ///< BVH over m_boxes (if any triangles)
};

} // namespace exaconstit::microstructure
