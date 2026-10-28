#pragma once

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <vector>

#include "axom/core.hpp"
#include "axom/primal/geometry/BoundingBox.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/spin/BVH.hpp"
#include "microstructure/mesh/element_mask.hpp"

/**
 * @file material_point_locator.hpp
 * @brief Answers "is this physical point inside the material?" for the elements of one rank's
 * mesh partition.
 *
 * Seeds placed by rejection sampling must lie inside the material, which for a lattice or a
 * tomography mesh is a small, complicated part of its bounding box. The test has two steps:
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
 *      reference point xi by Newton's method, then mfem::Geometry::CheckPoint tests xi. On a
 *      curved element the polynomial map, continued outside the reference element, can have a
 *      second solution xi outside it even when p is inside the element, and Newton started from
 *      the element center can converge to that wrong solution. Curved elements are therefore
 *      retried from several starting points (the element center, the reference position of the
 *      element's closest nodal point, the closest point of a sampled reference lattice, and a
 *      Newton variant that keeps the iterate inside the element) before a point is declared
 *      outside.
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
 * @brief Margin factor of the curved element boxes: the box of the sampled points of a curved
 * element is enlarged on each axis by this factor times the largest edge-midpoint deviation of
 * that coordinate (see MaterialPointLocator's constructor). 6 = 3 x 2: within one lattice cell
 * the deviation of the map from its piecewise-linear interpolant is at most about 3 times the
 * largest edge-midpoint deviation (the three directions of a hex cell add up; for a tet cell the
 * factor is 1.5), and the extra factor 2 is a safety factor for the higher-order terms that a
 * single midpoint sample per edge does not see.
 */
constexpr double CURVED_BOX_MARGIN_FACTOR = 6.0;

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
    using PointType = axom::primal::Point<double, 3>;     ///< a 3D point (Axom primal)
    using BoxType = axom::primal::BoundingBox<double, 3>; ///< an axis-aligned 3D box

    /**
     * @brief Computes one bounding box per local element and builds the BVH over them.
     *
     * Element boxes:
     * - straight-sided mesh (no Nodes, or Nodes that describe straight elements, see
     *   mesh/straight_sided.hpp): the box of the element's corner points, which contains the
     *   element exactly (a linear tet or trilinear hex lies in the convex hull of its corners).
     *   The corners come from the Nodes when present, else from the mesh vertices;
     * - curved mesh: the box of the element's nodal coordinates and of the element mapped at a
     *   reference lattice of 2p + 1 points per direction (p = Nodes order) and at the midpoints of
     *   the lattice edges, enlarged by a margin that covers the bulge of the element between the
     *   lattice points. The margin is CURVED_BOX_MARGIN_FACTOR times the largest deviation, over
     *   all lattice edges, between the element map at the edge midpoint and the average of the
     *   map at the edge ends (a sampled estimate of how far the element bends away from its
     *   piecewise-linear interpolant on the lattice, see the constructor comment in the .cpp).
     *   This is a **sampled bound, not a proven one**: it is reliable for elements whose map is
     *   smooth on the scale of the lattice (any reasonable order-p element), but an element that
     *   folds sharply between lattice points could stick out of its box. For Bernstein bases the
     *   nodal box alone is already a rigorous bound.
     * A straight-sided mesh with Nodes therefore gives the same boxes and inside tests as the
     * same mesh without Nodes, so seeding draws the same seeds on both.
     *
     * @param mesh this rank's 3D mesh partition (serial mfem::Mesh or the local part of a
     *        ParMesh). May have 0 elements, in which case every point is outside.
     * @param straight_sided the result of IsStraightSided(mesh, comm) (collective, so every rank
     *        uses the same box method and the global sampling box does not depend on the
     *        partition). Ignored (treated as true) for a mesh without Nodes.
     * @param mask active elements (mesh/element_mask.hpp); nullptr = every element. Inactive
     *        elements get no box (ElementBox returns an empty box), are left out of the BVH and
     *        of LocalBounds / LocalPointBounds, so no point is ever found inside them. The flags
     *        are copied; the mask need not outlive the locator.
     * @throws std::runtime_error if the mesh is not 3D; std::invalid_argument if a non-empty
     *         mask does not have one entry per local element.
     */
    MaterialPointLocator(const mfem::Mesh& mesh,
                         bool straight_sided,
                         const ElementMask* mask = nullptr);

    /** @brief Number of local elements (active or not; element ids run over all of them). */
    int NumElements() const {
        return m_num_elements;
    }

    /** @brief True when local element e takes part (always true without a mask). */
    bool ElementActive(int e) const {
        return m_active_flags.empty() || m_active_flags[static_cast<std::size_t>(e)] != 0;
    }

    /** @brief Number of local active elements (NumElements() without a mask). */
    int NumActiveElements() const {
        return static_cast<int>(m_active_elements.size());
    }

    /**
     * @brief Bounding box of all local elements (union of the element boxes, with the bulge
     * margin for curved elements): every point of the local material lies in it (for curved
     * elements, up to the sampled-bound caveat of the constructor). Invalid (empty) when the rank
     * has no elements.
     */
    const BoxType& LocalBounds() const {
        return m_local_bounds;
    }

    /**
     * @brief Box of the element points actually evaluated (corners, or the 2p + 1 per direction
     * lattice points of curved elements, without the bulge margin): the best estimate of the true
     * extent of the local material, used for the global mesh bounding box. Invalid when the rank
     * has no elements.
     */
    const BoxType& LocalPointBounds() const {
        return m_local_point_bounds;
    }

    /** @brief Bounding box of local element e (0-based local index), as built above; an empty
     * (invalid) box for an inactive element. */
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
    void MarkInside(std::int64_t n,
                    const double* x,
                    const double* y,
                    const double* z,
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

    /** @brief Newton inverse map of a curved element, retried from several starting points. */
    bool PointInCurvedElement(int e, const std::array<double, 3>& p) const;

    /**
     * @brief Reference lattice of one element geometry, used for the curved element boxes and
     * as Newton starting points: the points (i, j, k) / n (hex: 0 <= i, j, k <= n; tet:
     * i + j + k <= n) with n = 2p, plus the midpoints of the lattice edges (hex: the 3 axis
     * directions; tet: the 6 edge directions of the simplex lattice). Edge m joins
     * points.IntPoint(edge_a[m]) and points.IntPoint(edge_b[m]); its midpoint is
     * midpoints.IntPoint(m). Built once per geometry in the constructor; empty (valid = false)
     * for geometries that are not present or not hex/tet.
     */
    struct ReferenceLattice {
        bool valid = false;              ///< built for this geometry
        mfem::IntegrationRule points;    ///< (n+1)^3 (hex) or (n+1)(n+2)(n+3)/6 (tet) points
        mfem::IntegrationRule midpoints; ///< one per lattice edge
        std::vector<int> edge_a;         ///< first end of each edge (index into points)
        std::vector<int> edge_b;         ///< second end of each edge
    };

    /** @brief Builds m_lattices[geom] for lattice size n (see ReferenceLattice). */
    void BuildLattice(mfem::Geometry::Type geom, int n);

    /**
     * @brief Box of curved local element e (see the constructor): nodal points, lattice points
     * and edge midpoints, enlarged by the bulge margin. `point_box` receives the box of the
     * lattice points only. Host only.
     */
    void CurvedElementBox(int e, BoxType& box, BoxType& point_box) const;

    const mfem::Mesh& m_mesh;             ///< the local mesh (not owned)
    int m_num_elements = 0;               ///< local element count
    bool m_straight_sided = true;         ///< element = convex hull of its corners
    std::vector<BoxType> m_element_boxes; ///< one box per local element (empty if inactive)
    std::vector<char> m_active_flags;     ///< per local element, empty = all active
    std::vector<int> m_active_elements;   ///< local ids of the active elements, increasing
    std::vector<BoxType> m_active_boxes;  ///< BVH input with a mask (empty: all active)
    BoxType m_local_bounds;               ///< union of the element boxes
    BoxType m_local_point_bounds;         ///< box of the evaluated points (no padding)
    /** @brief Reference lattices of the curved path, indexed by mfem::Geometry::Type. */
    std::array<ReferenceLattice, mfem::Geometry::NumGeom> m_lattices;
    axom::spin::BVH<3, ExecSpace> m_bvh; ///< BVH over m_active_boxes (if any)
};

/**
 * @brief The active material of one generation run, shared by the steps that place or move
 * seeds: which elements take part (`mask`) and a point locator built once over them (`locator`).
 *
 * Both are optional. Functions taking a `const MaterialContext*` treat nullptr, or a null member,
 * as "every element is active" and "build a locator locally". When `locator` is given it must
 * have been built on the same mesh with the same mask (the driver builds it once,
 * microstructure_driver.cpp). Building the locator of a curved mesh is the expensive part (its
 * element boxes map a reference lattice per element), which is why it is shared.
 *
 * Whether `locator` is set must be the same on every rank, since building a missing locator is
 * collective. `mask` may be null on some ranks only: the driver's mask
 * (AttributeDomain::Mask()) is null on a rank without local elements, so functions that change
 * their communication pattern with a mask reduce the decision over the ranks.
 */
struct MaterialContext {
    /** @brief Active elements; nullptr = every element. */
    const ElementMask* mask = nullptr;
    /** @brief Locator over the active elements; nullptr = build one where needed. */
    const MaterialPointLocator* locator = nullptr;
};

/** @brief The mask of a context (nullptr for a null context). */
inline const ElementMask* ContextMask(const MaterialContext* context) {
    return context != nullptr ? context->mask : nullptr;
}

/**
 * @brief The context's locator, or one built into `owned` (IsStraightSided over `comm`, then
 * MaterialPointLocator(mesh, straight_sided, ContextMask(context))).
 *
 * **Collective over `comm`** when no locator is given (IsStraightSided reduces over the ranks);
 * otherwise local. Host only.
 *
 * @param mesh    this rank's mesh partition (the one the context belongs to).
 * @param comm    communicator over which the mesh is distributed.
 * @param context optional shared mask and locator.
 * @param owned   receives the locator when one has to be built; kept alive by the caller.
 * @return the locator to use.
 */
const MaterialPointLocator& ContextLocator(const mfem::Mesh& mesh,
                                           MPI_Comm comm,
                                           const MaterialContext* context,
                                           std::unique_ptr<MaterialPointLocator>& owned);

} // namespace exaconstit::microstructure
