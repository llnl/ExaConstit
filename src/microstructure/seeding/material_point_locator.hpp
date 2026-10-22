#pragma once

#include "mfem.hpp"

#include "axom/core.hpp"
#include "axom/primal/geometry/BoundingBox.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/spin/BVH.hpp"

#include <array>
#include <cstdint>
#include <vector>

/**
 * @file material_point_locator.hpp
 * @brief Answers "is this physical point inside the material?" for the elements of one rank's
 * mesh partition.
 *
 * Seeds placed by rejection sampling must lie inside the material, which for a lattice or a
 * tomography mesh is a small, complicated part of its bounding box. The test has two stages:
 *
 * 1. **Coarse: element bounding boxes in a BVH.** A BVH (bounding volume hierarchy) is a tree of
 *    nested bounding boxes: the root box holds all elements, each child box holds about half of
 *    its parent's elements, down to one element per leaf. A point is pushed down only the
 *    branches whose box contains it, so finding the few elements whose box contains a point
 *    costs O(log n_elements) instead of a loop over all elements. Axom's `spin::BVH::findPoints`
 *    does this for a whole batch of points.
 * 2. **Exact: point-in-element.** A box test only says "maybe": the box of a tet is up to 6 times
 *    its volume. For each candidate element the exact test maps the point back to the reference
 *    element and checks that the reference coordinates lie inside it:
 *    - linear tets (straight-sided mesh, mesh/straight_sided.hpp): closed form, the 4
 *      barycentric coordinates lambda_i (the volume fractions of the 4 sub-tets formed by the
 *      point and each face) must all be >= -tol;
 *    - hexes and any curved element: mfem::InverseElementTransformation solves x(xi) = p for the
 *      reference point xi by Newton's method, then mfem::Geometry::CheckPoint tests xi.
 *
 * A point on a face shared by two elements counts as inside both (the tolerances are slightly
 * positive), so no point of the material falls into a crack between elements because of
 * round-off. A point is reported inside if it is inside ANY local element; combining the answer
 * over ranks is the caller's job (seed_generators.cpp uses a bitwise-OR reduction).
 *
 * **Host only.** The exact test uses MFEM's host transformation objects, so the BVH is built and
 * queried on the host with axom::SEQ_EXEC, whatever mfem::Device is configured. Seeding tests
 * a few candidates per seed, which is cheap compared with labeling every element.
 */

namespace exaconstit::microstructure {

/**
 * @brief Point-in-material test over the local elements of one mesh partition.
 *
 * Construction is local to the rank (not collective); the straight-sided flag it takes comes
 * from the collective IsStraightSided. The mesh must stay alive and unchanged
 * while the locator is used (it keeps a reference). Host only.
 */
class MaterialPointLocator {
public:
    /** @brief Axom execution space of the element BVH: host serial (see the file comment). */
    using ExecSpace = axom::SEQ_EXEC;
    using PointType = axom::primal::Point<double, 3>;
    using BoxType = axom::primal::BoundingBox<double, 3>;

    /**
     * @brief Computes one bounding box per local element and builds the BVH over them.
     *
     * Element boxes:
     * - straight-sided mesh (no Nodes, or Nodes that describe straight elements, see
     *   mesh/straight_sided.hpp): the box of the element's corner points, which contains the
     *   element exactly (a linear tet or trilinear hex lies in the convex hull of its corners).
     *   The corners come from the Nodes when present, else from the mesh vertices; without Nodes
     *   this is the vertex box used before straight-sided Nodes were recognized, bit for bit;
     * - curved mesh: the box of the element's nodal coordinates and of the element mapped at a
     *   refined grid of reference points, enlarged by 10% of its diagonal on every side to cover
     *   any bulge between those points (a safety margin, not a proven bound; for Bernstein bases
     *   the nodal box alone is already a rigorous bound).
     * A straight-sided mesh with Nodes therefore gives the same boxes and inside tests as the
     * same mesh without Nodes, so seeding draws the same seeds on both.
     *
     * @param mesh this rank's 3D mesh partition (serial mfem::Mesh or the local part of a
     *        ParMesh). May have 0 elements, in which case every point is outside.
     * @param straight_sided the result of IsStraightSided(mesh, comm) (collective, so every rank
     *        uses the same box method and the global sampling box does not depend on the
     *        partition). Ignored (treated as true) for a mesh without Nodes.
     * @throws std::runtime_error if the mesh is not 3D.
     */
    MaterialPointLocator(const mfem::Mesh& mesh, bool straight_sided);

    /** @brief Number of local elements. */
    int NumElements() const {
        return m_num_elements;
    }

    /**
     * @brief Bounding box of all local elements (union of the element boxes, padded for curved
     * elements): every point of the local material lies in it. Invalid (empty) when the rank has
     * no elements.
     */
    const BoxType& LocalBounds() const {
        return m_local_bounds;
    }

    /**
     * @brief Box of the element points actually evaluated (vertices, or the sampled points of
     * curved elements before the 10% padding): the best estimate of the true extent of the local
     * material, used for the global mesh bounding box. Invalid when the rank has no elements.
     */
    const BoxType& LocalPointBounds() const {
        return m_local_point_bounds;
    }

    /** @brief Bounding box of local element e (0-based local index), as built above. */
    const BoxType& ElementBox(int e) const {
        return m_element_boxes[e];
    }

    /** @brief True if the elements are treated as straight-sided (exact corner boxes). */
    bool StraightSided() const {
        return m_straight_sided;
    }

    /**
     * @brief Sets bit i of `inside_mask` for every point i inside some local element.
     *
     * Bit i lives in word i / 64 at position i % 64 (LSB first). Bits of points that are not
     * inside a local element are left unchanged, so the caller zeroes the mask first. Host only,
     * not collective.
     *
     * @param n           number of points.
     * @param x,y,z       point coordinates, n entries each (mesh length units).
     * @param inside_mask at least (n + 63) / 64 words.
     */
    void MarkInside(std::int64_t n, const double* x, const double* y, const double* z,
                    std::uint64_t* inside_mask) const;

    /**
     * @brief Exact test: true if p lies in local element e (0-based), up to the tolerances in the
     * file comment. Host only.
     */
    bool PointInElement(int e, const std::array<double, 3>& p) const;

private:
    /** @brief Closed-form barycentric test for a straight-sided tet. */
    bool PointInLinearTet(int e, const std::array<double, 3>& p) const;

    /** @brief Newton inverse map (mfem::InverseElementTransformation) for any element. */
    bool PointInGeneralElement(int e, const std::array<double, 3>& p) const;

    const mfem::Mesh& m_mesh;                 ///< the local mesh (not owned)
    int m_num_elements = 0;                   ///< local element count
    bool m_straight_sided = true;             ///< element = convex hull of its corners
    std::vector<BoxType> m_element_boxes;     ///< one box per local element
    BoxType m_local_bounds;                   ///< union of the element boxes
    BoxType m_local_point_bounds;             ///< box of the evaluated points (no padding)
    axom::spin::BVH<3, ExecSpace> m_bvh;      ///< BVH over m_element_boxes (if any elements)
};

} // namespace exaconstit::microstructure
