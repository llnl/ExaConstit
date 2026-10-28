#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <vector>

#include "microstructure/mesh/element_mask.hpp"

/**
 * @file attribute_domain.hpp
 * @brief Restricts grain generation to part of a mesh, selected by element attribute, and maps
 *        the grain labels back to the whole mesh.
 *
 * **Use case.** A mesh may contain parts that should not receive grains: e.g. the build plate
 * under an additively manufactured part, or a grip region modeled as a homogeneous material. The
 * user lists the element attributes of the *active* parts (they get a tessellation) and of the
 * *passthrough* parts (each passthrough attribute becomes one extra ExaConstit region, kept as a
 * single "grain" without a crystal orientation of its own).
 *
 * **How.** AttributeDomain validates the classification and builds an element mask
 * (mesh/element_mask.hpp: 1 for active, 0 for passthrough elements) over the mesh itself. Every
 * generation stage takes the mask and skips the inactive elements; no sub-mesh or other copy of
 * the mesh is built (production meshes are large, and an mfem::ParSubMesh copy would also
 * distort non-Gauss-Lobatto curved Nodes of order >= 3). After labeling, ToParent writes the
 * passthrough ids: active elements keep their grain id 1..N, and the elements of the k-th
 * passthrough attribute (sorted ascending) get id N + 1 + k. Those ids are then written as the
 * element attributes of the output mesh.
 *
 * **Consequences for the stages** (see each stage for details):
 * - Seeding places seeds in active elements only; the material volume and boxes count only them.
 * - The interface between active and passthrough elements is bonded, not a surface: the surface
 *   clearance (seeding/boundary_clearance.hpp) ignores it, so seeds may lie next to the top of a
 *   build plate.
 * - For connectivity (fragment cleanup, connected mode, geodesic paths) a face to a passthrough
 *   element is a free surface: grains do not connect through the passthrough part.
 * - Periodic axes cannot be combined with passthrough attributes (refused by the options and the
 *   labeler).
 * - A rank may own no active element; every collective call still completes there.
 */

namespace exaconstit::microstructure {

/**
 * @brief The active (tessellated) part of a mesh selected by element attribute, plus the
 * passthrough parts that are kept as one region each.
 *
 * Construction and ToParent are collective; the accessors are not. Host data structure. Not
 * copyable (it refers to its parent mesh).
 */
class AttributeDomain {
public:
    /**
     * @brief Classifies the element attributes and builds the element mask.
     *
     * Steps: gather the attributes present on any rank with their global element counts; check
     * the classification; with passthrough attributes, mark the active elements in the mask.
     *
     * **Collective** over the parent's communicator (MPI_Allgather / MPI_Allgatherv of the local
     * attribute counts). Host only. Every validation error is computed from globally gathered
     * data, so all ranks throw the same exception together.
     *
     * @param parent  the whole distributed mesh. Must outlive this object and must not be
     *                refined or repartitioned while it is in use.
     * @param active_attributes element attributes (> 0) that receive grains. Empty means "every
     *                attribute that is not passthrough". Order and duplicates do not matter.
     * @param passthrough_attributes element attributes (> 0) kept as one region each; sorted
     *                ascending internally (duplicates removed). May be empty.
     * @throws std::invalid_argument on every rank when: a listed attribute is <= 0; an attribute
     *         is listed both as active and passthrough; a mesh attribute is listed in neither
     *         (the message lists the offending attributes); a passthrough attribute has no
     *         elements; or the active set has no elements globally.
     */
    AttributeDomain(mfem::ParMesh& parent,
                    const std::vector<int>& active_attributes,
                    const std::vector<int>& passthrough_attributes);

    /** @brief Not copyable: the object refers to its parent mesh. */
    AttributeDomain(const AttributeDomain&) = delete;
    /** @brief Not copy-assignable (see the copy constructor). */
    AttributeDomain& operator=(const AttributeDomain&) = delete;

    /**
     * @brief The active elements of the parent mesh, or nullptr when every element is active
     * (no passthrough attribute). Valid for the lifetime of this object. Not collective.
     */
    const ElementMask* Mask() const;

    /** @brief The parent mesh passed to the constructor. */
    mfem::ParMesh& ParentMesh() {
        return m_parent;
    }

    /** @brief The active attributes actually used, sorted ascending (all ranks identical). */
    const std::vector<int>& ActiveAttributes() const {
        return m_active_attributes;
    }

    /** @brief The passthrough attributes, sorted ascending (all ranks identical). */
    const std::vector<int>& PassthroughAttributes() const {
        return m_passthrough_attributes;
    }

    /**
     * @brief Grain id of every element.
     *
     * Active elements keep their grain from `element_grain`; elements with passthrough attribute
     * PassthroughAttributes()[k] get num_grains + 1 + k.
     *
     * **Collective** over the parent's communicator (one MPI_Allreduce for the input check).
     * Host only.
     *
     * @param element_grain size = local NE of the parent: grain id (1..num_grains) of every local
     *                      active element; the entries of inactive elements are not read (the
     *                      labelers' placeholder 0). Read with HostRead().
     * @param num_grains    global number of grains N >= 0 (same on every rank).
     * @return size = local NE: grain id of every local element, in 1..num_grains + number of
     *         passthrough attributes.
     * @throws std::invalid_argument on every rank if, on any rank, the input size is wrong or an
     *         active element's grain id lies outside [1, num_grains].
     */
    mfem::Array<int> ToParent(const mfem::Array<int>& element_grain, int num_grains) const;

    /**
     * @brief Global number of elements of each passthrough attribute, in the order of
     * PassthroughAttributes(). Identical on every rank; not collective (computed once in the
     * constructor).
     */
    std::vector<std::int64_t> PassthroughElementCounts() const {
        return m_passthrough_counts;
    }

    /** @brief Global number of active elements (same on every rank). */
    std::int64_t GlobalActiveElements() const {
        return m_global_active_elements;
    }

private:
    mfem::ParMesh& m_parent;                        ///< the whole mesh
    ElementMask m_mask;                             ///< active elements; empty = all active
    std::vector<int> m_active_attributes;           ///< sorted ascending
    std::vector<int> m_passthrough_attributes;      ///< sorted ascending
    std::vector<std::int64_t> m_passthrough_counts; ///< global, per passthrough attribute
    std::int64_t m_global_active_elements = 0;      ///< global number of active elements
};

} // namespace exaconstit::microstructure
