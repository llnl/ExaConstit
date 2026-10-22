#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <memory>
#include <vector>

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
 * AttributeDomain builds the mesh on which seeding, labeling and cleanup run (ActiveMesh): the
 * parent mesh itself when every element is active, otherwise an `mfem::ParSubMesh` that holds
 * only the active elements. After labeling, ToParent turns the per-active-element grain ids into
 * one id per parent element: active elements keep their grain id 1..N, and the elements of the
 * k-th passthrough attribute (sorted ascending) get id N + 1 + k. Those ids are then written as
 * the element attributes of the output mesh.
 *
 * **Verified `mfem::ParSubMesh` behavior** (MFEM v4.10, `builds/mfem/mesh/submesh/`):
 * - `ParSubMesh::CreateFromDomain(parent, attributes)` (`psubmesh.hpp:72`) keeps every local
 *   parent element whose attribute is listed, in increasing parent element order
 *   (`submesh_utils.cpp:64-89`), on the same rank: the parallel distribution is reused and
 *   nothing is repartitioned. `GetParentElementIDMap()` (`psubmesh.hpp:112`) maps sub-mesh
 *   element i to its local parent element id.
 * - Each sub-mesh element keeps the parent element's attribute (`submesh_utils.cpp:86`); the
 *   global attribute list is rebuilt by `SetAttributes()` (`psubmesh.cpp:312`, which reduces
 *   over the communicator in `ParMesh::SetAttributes`, `pmesh.cpp:1593-1614`). Vertices are
 *   copied bit-for-bit from the parent (`submesh_utils.cpp:78`), so vertex-based element keys
 *   (mesh/element_key.hpp) agree between the parent and the sub-mesh.
 * - Curved meshes: when the parent has Nodes (a high-order geometry), the sub-mesh gets Nodes of
 *   the same order, continuity and ordering, filled by transferring the parent's Nodes
 *   (`psubmesh.cpp:289-311`).
 * - Shared faces (faces between elements on two ranks) are rebuilt for the sub-mesh
 *   (`psubmesh.cpp:231-266`, `BuildSharedFacesMapping`), and `ExchangeFaceNbrData()` is already
 *   called once during construction (`psubmesh.cpp:269`), so face-neighbor data and
 *   MeshAdjacency work on the sub-mesh. A parent face between an active and a passthrough
 *   element becomes a boundary face of the sub-mesh.
 * - A rank may own zero active elements; the construction is collective and still completes on
 *   that rank (checked by test_regions with a partition whose ranks own no active element).
 * - The header comment says the attributes should mark "exactly one connected subset"
 *   (`psubmesh.hpp:66-67`), but the element selection only filters by attribute
 *   (`submesh_utils.cpp:68`); disconnected active parts are accepted.
 * - The ParSubMesh keeps a reference to its parent (`psubmesh.cpp:40`), so the parent must
 *   outlive the AttributeDomain.
 * - Building the sub-mesh calls `ExchangeFaceNbrData()` on the *parent* (`psubmesh.cpp:51`), so
 *   the parent caches copies of its neighbor-rank elements, including their attributes. Those
 *   cached attributes go stale if the parent's element attributes are changed later (e.g. when
 *   grain ids are written into them), and calling `ExchangeFaceNbrData()` again does not
 *   refresh them: it returns at once when the data already exist (`pmesh.cpp:2100-2103`), and
 *   `DeleteFaceNbrData` is protected (`pmesh.hpp:204`). Code that needs neighbor values of the
 *   parent after relabeling should exchange them explicitly (e.g. MeshAdjacency::
 *   ExchangeLabels) instead of reading face-neighbor element attributes.
 * - **Limit: at most 32 ranks may share one vertex, edge or face.** The construction stores the
 *   ranks of each shared entity as bits of an int and hard-aborts otherwise
 *   (`MFEM_VERIFY(group_sz <= 8*sizeof(int))`, `psubmesh.cpp:332,375,469,526`). AttributeDomain
 *   checks the largest communication group of the parent first
 *   (`ParMesh::GetNGroups`, `pmesh.hpp:472`; `GroupTopology::GetGroupSize`,
 *   `communication.hpp:210`) and throws std::invalid_argument on all ranks instead.
 * - The parent must have boundary elements: the sub-mesh reads
 *   `parent.bdr_attributes.Max()` (`submesh_utils.cpp:288`), which is invalid for an empty
 *   array. AttributeDomain checks this collectively first.
 *
 * **Consequences for callers** (the driver decides the policy):
 * - The interface between active and passthrough elements is a *free surface* of the
 *   sub-mesh. Anything that looks at the sub-mesh surface treats it as an outer boundary: e.g.
 *   seeding with `boundary_clearance > 0` keeps seeds away from a bonded interface (such as the
 *   top of a build plate), which is usually not wanted there.
 * - The interface faces get a new boundary attribute, `parent.bdr_attributes.Max() + 1`
 *   (`submesh_utils.cpp:288,316-318`). When such faces lie on a plane of the active part's
 *   bounding box (the top of a build plate is the bottom plane of the part), periodic-plane
 *   detection on the sub-mesh (periodic seeds / MeshAdjacency periodic pairing) would treat
 *   them as faces of a periodic boundary.
 */

namespace exaconstit::microstructure {

/**
 * @brief The active (tessellated) part of a mesh selected by element attribute, plus the
 * passthrough parts that are kept as one region each.
 *
 * Construction and ToParent are collective; ActiveMesh, UsesSubMesh and the accessors are not.
 * Host data structure. Not copyable or movable (the ParSubMesh refers to its parent and is kept
 * at a fixed address).
 */
class AttributeDomain {
public:
    /**
     * @brief Classifies the element attributes and builds the active mesh.
     *
     * Steps: gather the attributes present on any rank with their global element counts; check
     * the classification; build an `mfem::ParSubMesh` of the active attributes when there are
     * passthrough attributes (otherwise ActiveMesh is the parent itself).
     *
     * **Collective** over the parent's communicator (MPI_Allgather / MPI_Allgatherv of the local
     * attribute counts, then the ParSubMesh construction). Host only. Every validation error is
     * computed from globally gathered data, so all ranks throw the same exception together.
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
     *         elements; the active set has no elements globally; or (only when a sub-mesh is
     *         needed) the parent has no boundary elements or more than 32 ranks share one
     *         vertex, edge or face (MFEM's ParSubMesh limit, see the file comment).
     */
    AttributeDomain(mfem::ParMesh& parent,
                    const std::vector<int>& active_attributes,
                    const std::vector<int>& passthrough_attributes);

    ~AttributeDomain();
    AttributeDomain(const AttributeDomain&) = delete;
    AttributeDomain& operator=(const AttributeDomain&) = delete;

    /**
     * @brief The mesh to seed, label and clean up: the parent itself when every element is
     * active (UsesSubMesh() == false), else the ParSubMesh of the active elements. Not
     * collective. The reference stays valid for the lifetime of this object.
     */
    mfem::ParMesh& ActiveMesh();

    /** @brief True when ActiveMesh() is a ParSubMesh (there are passthrough attributes). */
    bool UsesSubMesh() const;

    /** @brief The parent mesh passed to the constructor. */
    mfem::ParMesh& ParentMesh() {
        return m_parent;
    }

    /**
     * @brief Size: local NE of ActiveMesh(). Entry i = 0-based local parent element id of local
     * active element i (strictly increasing). The identity map when UsesSubMesh() is false.
     *
     * A per-element result on ActiveMesh() (index i) belongs to serial element
     * `order.local_to_serial[ActiveToParentElement()[i]]`, with `order` the SerialOrder the
     * parent was built with (mesh/serial_order.hpp).
     */
    const mfem::Array<int>& ActiveToParentElement() const {
        return m_active_to_parent;
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
     * @brief Grain id of every parent element.
     *
     * Parent elements that are active get the grain of their active element; elements with
     * passthrough attribute PassthroughAttributes()[k] get num_grains + 1 + k.
     *
     * **Collective** over the parent's communicator (one MPI_Allreduce for the input check).
     * Host only.
     *
     * @param active_element_grain size = local NE of ActiveMesh(): grain id (1..num_grains) of
     *                             every local active element. Read with HostRead().
     * @param num_grains           global number of grains N >= 0 (same on every rank).
     * @return size = local NE of the parent: grain id of every local parent element, in
     *         1..num_grains + number of passthrough attributes.
     * @throws std::invalid_argument on every rank if, on any rank, the input size is wrong or a
     *         grain id lies outside [1, num_grains].
     */
    mfem::Array<int> ToParent(const mfem::Array<int>& active_element_grain, int num_grains) const;

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
    /** Holds the ParSubMesh so it is constructed in place (see attribute_domain.cpp). */
    struct SubMeshHolder;

    mfem::ParMesh& m_parent;                      ///< the whole mesh
    std::unique_ptr<SubMeshHolder> m_submesh;     ///< null when every element is active
    std::vector<int> m_active_attributes;         ///< sorted ascending
    std::vector<int> m_passthrough_attributes;    ///< sorted ascending
    std::vector<std::int64_t> m_passthrough_counts; ///< global, per passthrough attribute
    std::int64_t m_global_active_elements = 0;    ///< global number of active elements
    mfem::Array<int> m_active_to_parent;          ///< local active element -> parent element
};

} // namespace exaconstit::microstructure
