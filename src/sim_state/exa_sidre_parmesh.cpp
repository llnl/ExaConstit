// Compiled only when ExaConstit is built with Axom, so this file can always be listed in
// CMake; without EXACONSTIT_USE_AXOM it is an empty translation unit.
#if defined(EXACONSTIT_USE_AXOM)

#include "sim_state/exa_sidre_parmesh.hpp"

#include "sim_state/restart_sidre_utils.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "axom/sidre.hpp"

// =========================================================================
// BACKGROUND: HOW AN mfem::ParMesh DESCRIBES ITS PARALLEL STRUCTURE
// =========================================================================
//
// Everything in this file writes or rebuilds the pieces listed here, so it is worth reading
// once before changing anything below. References are to MFEM's mesh/pmesh.cpp,
// general/communication.cpp and fem/pfespace.cpp.
//
// Local mesh. Each rank holds an ordinary serial mfem::Mesh for its own elements: local
// vertices, local elements, local boundary elements, and the local edges and faces MFEM
// derives from the element connectivity (FinalizeTopology()). Vertices, edges and faces on
// the interface between ranks exist on every rank that touches them, each rank with its own
// local index for them.
//
// Communication groups (ParMesh::gtopo, an mfem::GroupTopology). A group is a set of ranks
// that share at least one vertex, edge or face. Group 0 is always this rank on its own and
// shares nothing. Every other group has at least two members and a local index
// 1..NGroups()-1 on each member rank; the same group generally has different local indices
// on different ranks. Each group has a master rank (GroupTopology::Create() picks the
// smallest rank in the group), and the master owns the group's shared DOFs when the finite
// element spaces assign true DOFs. Inside a GroupTopology, ranks are referred to by "local
// processor" (lproc) numbers: this rank is lproc 0 and the other ranks are numbered in the
// order they first appear when walking the groups in local group order
// (GroupTopology::ProcToLProc()). The local group order therefore also fixes the neighbor
// numbering and the order of the messages the group communicators exchange.
//
// Shared entities. The shared vertices are stored as local vertex indices (svert_lvert).
// The shared edges and faces are stored as vertex tuples (shared_edges, shared_trias,
// shared_quads) in a "stored orientation" that every rank in the group agrees on, together
// with the local edge or face each one corresponds to (sedge_ledge, sface_lface).
//
// Group tables (group_svert, group_sedge, group_stria, group_squad). mfem::Table objects in
// compressed-row (CSR) form with one row per communication group 1..N: row r describes group
// r+1, since group 0 shares nothing. Row r lists the indices, into the shared-entity arrays
// above, of the entities group r+1 shares.
//
// How the finite element spaces use this. ParFiniteElementSpace builds its parallel DOF
// numbering from these structures. For each group it collects the DOFs on the group's shared
// entities, in the order the entities appear in the group's row, and the group's master
// assigns them true DOFs that the other members then receive by position. Two consequences
// drive the whole design of this file:
//   1. Entity i of a group on one rank is matched with entity i of the same group on every
//      other member, so the order of the entities within each group must be identical on
//      every member rank.
//   2. A shared edge or face with several interior DOFs lists them in a specific order, and
//      GetSharedEdgeDofs() / GetSharedTriangleDofs() / GetSharedQuadrilateralDofs() permute
//      each rank's local DOF order by the orientation that GroupEdge() / GroupTriangle() /
//      GroupQuadrilateral() report, so that every rank lists them in the stored order. A
//      wrong stored orientation silently matches the wrong DOFs across ranks. It only shows
//      up once a shared entity has two or more interior DOFs (H1 order >= 3 on hex faces),
//      and the mesh otherwise looks perfectly valid.
//
// MFEM normally builds all of this in the partitioning ParMesh constructor, which walks the
// global serial mesh in global entity order; that is what makes the per-group entity order
// and the stored orientations agree across ranks. A restarted run has no global mesh, so the
// writer below records exactly what each rank holds and the reader rebuilds it unchanged.

namespace exaconstit {

namespace {

using axom::sidre::Group;
using axom::sidre::IndexType;
using axom::sidre::View;
using restart_detail::ReadSidreScalar;
using restart_detail::VerifyOnAllRanks;

static_assert(std::is_same<mfem::real_t, double>::value,
              "Restart mesh I/O stores coordinates as FLOAT64; build MFEM in double precision.");

/**
 * @brief Conduit Blueprint shape name for an MFEM geometry
 *
 * @param geometry MFEM geometry type (mfem::Geometry::Type)
 * @return Blueprint shape name, or nullptr if the restart mesh I/O does not support the
 * geometry
 */
const char* ShapeNameFromGeometry(const int geometry) {
    switch (geometry) {
    case mfem::Geometry::SEGMENT:
        return "line";
    case mfem::Geometry::TRIANGLE:
        return "tri";
    case mfem::Geometry::SQUARE:
        return "quad";
    case mfem::Geometry::TETRAHEDRON:
        return "tet";
    case mfem::Geometry::CUBE:
        return "hex";
    default:
        return nullptr;
    }
}

/**
 * @brief Create an integer array view owned by Sidre
 *
 * @param group Group to create the view in
 * @param path Path of the new view relative to the group
 * @param count Number of entries
 * @return Pointer to the allocated entries, or nullptr when count is 0
 *
 * @details The arrays are owned by Sidre, never external views of the mesh's memory, because
 * Group::load() only restores the description of an external view and not its data, so the
 * reader would get nothing back. Zero-length arrays are described but not allocated, which
 * is why callers must not dereference the returned pointer when count is 0. Ranks with no
 * boundary elements, or groups with no shared edges, produce such arrays all the time.
 */
int* CreateOwnedIntArray(Group* group, const std::string& path, const IndexType count) {
    if (count == 0) {
        group->createView(path, axom::sidre::INT_ID, 0);
        return nullptr;
    }
    return group->createViewAndAllocate(path, axom::sidre::INT_ID, count)->getData<int*>();
}

/**
 * @brief Create a double array view owned by Sidre
 *
 * @param group Group to create the view in
 * @param path Path of the new view relative to the group
 * @param count Number of entries
 * @return Pointer to the allocated entries, or nullptr when count is 0
 *
 * @details See CreateOwnedIntArray() for why the arrays are owned and why zero-length arrays
 * are not allocated.
 */
double* CreateOwnedDoubleArray(Group* group, const std::string& path, const IndexType count) {
    if (count == 0) {
        group->createView(path, axom::sidre::FLOAT64_ID, 0);
        return nullptr;
    }
    return group->createViewAndAllocate(path, axom::sidre::FLOAT64_ID, count)->getData<double*>();
}

/**
 * @brief Look up an array written by CreateOwnedIntArray() or CreateOwnedDoubleArray()
 *
 * @tparam ValueType int or double
 * @param group Group that holds the view
 * @param path Path of the view relative to the group
 * @param expected_count Number of entries the array must have
 * @return Pointer to the entries (nullptr when the array is empty) and whether the view
 * exists, is owned by Sidre, has the expected type and has expected_count entries
 *
 * @details The type and ownership checks catch a checkpoint written by a different layout
 * version or a corrupted file. Without them a wrong-typed array would be reinterpreted
 * silently, and an external view would come back with a null data pointer.
 */
template <typename ValueType>
std::pair<const ValueType*, bool>
ReadOwnedArray(Group* group, const std::string& path, const IndexType expected_count) {
    const axom::sidre::TypeID expected_type = std::is_same<ValueType, double>::value
                                                  ? axom::sidre::FLOAT64_ID
                                                  : axom::sidre::INT_ID;
    if (!group->hasView(path)) {
        return {nullptr, false};
    }
    View* view = group->getView(path);
    if (view->getNumElements() != expected_count) {
        return {nullptr, false};
    }
    if (expected_count == 0) {
        return {nullptr, true};
    }
    if (view->getTypeID() != expected_type || view->isExternal()) {
        return {nullptr, false};
    }
    return {view->getData<ValueType*>(), true};
}

/**
 * @brief Look up an owned integer array of unknown length that holds fixed-size tuples
 *
 * @param group Group that holds the view
 * @param path Path of the view relative to the group
 * @param components Number of integers per tuple (for example 2 for an edge)
 * @param ok Set to false if the view is missing, malformed, or its length is not a multiple
 *           of components; left unchanged otherwise, so one flag can collect several checks
 * @return Pointer to the entries (nullptr when empty or malformed) and the number of tuples
 *
 * @details Used for the adjacency-set arrays, whose lengths are not recorded separately.
 */
std::pair<const int*, IndexType>
ReadIntTuples(Group* group, const std::string& path, const int components, bool& ok) {
    if (!group->hasView(path)) {
        ok = false;
        return {nullptr, 0};
    }
    View* view = group->getView(path);
    const IndexType scalar_count = view->getNumElements();
    if (scalar_count % components != 0) {
        ok = false;
        return {nullptr, 0};
    }
    if (scalar_count == 0) {
        return {nullptr, 0};
    }
    if (view->getTypeID() != axom::sidre::INT_ID || view->isExternal()) {
        ok = false;
        return {nullptr, 0};
    }
    return {view->getData<int*>(), scalar_count / components};
}

/**
 * @brief Agree across ranks on the element and boundary element geometries
 *
 * @param mesh Parallel mesh being written
 * @return Mesh-wide element geometry and boundary element geometry; the boundary geometry is
 * -1 if no rank has boundary elements
 *
 * @details Collective. The reader needs the geometry to size the connectivity arrays and to
 * create elements, and it has to be the same on every rank. A rank deep inside the domain may
 * own no boundary elements, though, and a rank could in principle own no elements at all, so
 * a rank cannot simply look at its own first element. Each rank reports what it has, using -1
 * for "none", and an MPI_Allreduce with MPI_MAX picks the mesh-wide value (valid geometry
 * values are all >= 0).
 *
 * The same pass also checks that no rank mixes geometries, that every rank agrees with the
 * mesh-wide value, and that the geometry is one the Blueprint layout here supports. Any
 * violation aborts on every rank, since a single connectivity array per topology cannot
 * describe mixed meshes.
 */
std::pair<int, int> AgreeOnMeshGeometries(const mfem::ParMesh& mesh) {
    int local_element_geometry = -1;
    int local_boundary_geometry = -1;
    bool local_mixed_geometry = false;
    for (int element_index = 0; element_index < mesh.GetNE(); ++element_index) {
        const int geometry = mesh.GetElementGeometry(element_index);
        if (local_element_geometry < 0) {
            local_element_geometry = geometry;
        } else if (geometry != local_element_geometry) {
            local_mixed_geometry = true;
        }
    }
    for (int boundary_index = 0; boundary_index < mesh.GetNBE(); ++boundary_index) {
        const int geometry = mesh.GetBdrElementGeometry(boundary_index);
        if (local_boundary_geometry < 0) {
            local_boundary_geometry = geometry;
        } else if (geometry != local_boundary_geometry) {
            local_mixed_geometry = true;
        }
    }

    std::array<int, 2> local_geometries = {local_element_geometry, local_boundary_geometry};
    std::array<int, 2> global_geometries = {-1, -1};
    MPI_Allreduce(
        local_geometries.data(), global_geometries.data(), 2, MPI_INT, MPI_MAX, mesh.GetComm());
    const int element_geometry = global_geometries[0];
    const int boundary_geometry = global_geometries[1];

    const bool local_consistent =
        !local_mixed_geometry &&
        (local_element_geometry < 0 || local_element_geometry == element_geometry) &&
        (local_boundary_geometry < 0 || local_boundary_geometry == boundary_geometry) &&
        ShapeNameFromGeometry(element_geometry) != nullptr &&
        (boundary_geometry < 0 || ShapeNameFromGeometry(boundary_geometry) != nullptr);
    VerifyOnAllRanks(local_consistent,
                     mesh.GetComm(),
                     "the mesh mixes element geometries or uses a geometry other than segment, "
                     "triangle, quadrilateral, tetrahedron or hexahedron",
                     "Unable to write the mesh to the restart checkpoint");
    return {element_geometry, boundary_geometry};
}

/**
 * @brief Write the vertex coordinates under `coordsets/coords`
 *
 * @param mesh Parallel mesh being written
 * @param mesh_group Group being written into
 *
 * @details One array per axis (x, y and, in 3D, z), each indexed by local vertex number, as
 * Blueprint's explicit coordsets expect. The local vertex numbering has to be preserved
 * exactly: element connectivity, the shared vertices and the shared edge and face tuples all
 * refer to vertices by these indices.
 *
 * These are the mesh's vertex positions, not its Nodes. For a high-order or deformed mesh the
 * vertices only carry the element corners. SimulationState's coordinate fields are restored
 * separately from the checkpoint, so nothing is lost.
 */
void WriteCoordinates(const mfem::ParMesh& mesh, Group* mesh_group) {
    const std::array<const char*, 3> axis_names = {"x", "y", "z"};
    mesh_group->createViewString("coordsets/coords/type", "explicit");
    for (int axis = 0; axis < mesh.SpaceDimension(); ++axis) {
        double* axis_values = CreateOwnedDoubleArray(
            mesh_group, std::string("coordsets/coords/values/") + axis_names[axis], mesh.GetNV());
        for (int vertex_index = 0; vertex_index < mesh.GetNV(); ++vertex_index) {
            axis_values[vertex_index] = mesh.GetVertex(vertex_index)[axis];
        }
    }
}

/**
 * @brief Write the element connectivity and element attributes
 *
 * @param mesh Parallel mesh being written
 * @param mesh_group Group being written into
 * @param element_geometry Mesh-wide element geometry from AgreeOnMeshGeometries()
 *
 * @details The connectivity is one flat array, vertices_per_element entries per element, in
 * local element order and in each element's own vertex order. Both orders matter. The finite
 * element spaces number their local DOFs element by element following each element's vertex
 * order, so a different element order or a rotated vertex list would put restored values on
 * the wrong DOFs even though the mesh looks the same. That is also why the reader must not
 * let MFEM "fix" element orientations (see the reader's Finalize() call).
 *
 * In ExaConstit the element attributes hold the region IDs once SimulationState has been set
 * up (it overwrites the grain IDs read from the mesh file), and that is what a restarted run
 * uses to rebuild its regions. The grain IDs are checkpointed separately by the caller.
 */
void WriteVolumeTopology(const mfem::ParMesh& mesh, Group* mesh_group, const int element_geometry) {
    const int num_elements = mesh.GetNE();
    const int vertices_per_element = mfem::Geometry::NumVerts[element_geometry];
    mesh_group->createViewString("topologies/mesh/type", "unstructured");
    mesh_group->createViewString("topologies/mesh/coordset", "coords");
    mesh_group->createViewString("topologies/mesh/elements/shape",
                                 ShapeNameFromGeometry(element_geometry));

    int* connectivity = CreateOwnedIntArray(mesh_group,
                                            "topologies/mesh/elements/connectivity",
                                            static_cast<IndexType>(num_elements) *
                                                vertices_per_element);
    for (int element_index = 0; element_index < num_elements; ++element_index) {
        const int* element_vertices = mesh.GetElement(element_index)->GetVertices();
        for (int local_vertex = 0; local_vertex < vertices_per_element; ++local_vertex) {
            connectivity[static_cast<std::size_t>(element_index) * vertices_per_element +
                         local_vertex] = element_vertices[local_vertex];
        }
    }

    mesh_group->createViewString("fields/mesh_material_attribute/association", "element");
    mesh_group->createViewString("fields/mesh_material_attribute/topology", "mesh");
    int* element_attributes = CreateOwnedIntArray(
        mesh_group, "fields/mesh_material_attribute/values", num_elements);
    for (int element_index = 0; element_index < num_elements; ++element_index) {
        element_attributes[element_index] = mesh.GetAttribute(element_index);
    }
}

/**
 * @brief Write the boundary element connectivity and boundary attributes
 *
 * @param mesh Parallel mesh being written
 * @param mesh_group Group being written into
 * @param boundary_geometry Mesh-wide boundary geometry from AgreeOnMeshGeometries(); nothing
 *                          is written when it is -1 (no rank has boundary elements)
 *
 * @details Boundary elements are stored explicitly rather than regenerated on read, because
 * their attributes are what the boundary conditions refer to, and MFEM's generated boundary
 * would carry default attributes. A rank with no boundary elements still writes the
 * (zero-length) arrays whenever some other rank has boundary elements, so every rank's group
 * has the same structure.
 */
void WriteBoundaryTopology(const mfem::ParMesh& mesh,
                           Group* mesh_group,
                           const int boundary_geometry) {
    if (boundary_geometry < 0) {
        return;
    }
    const int num_boundary_elements = mesh.GetNBE();
    const int vertices_per_boundary_element = mfem::Geometry::NumVerts[boundary_geometry];
    mesh_group->createViewString("topologies/mesh/boundary_topology", "boundary");
    mesh_group->createViewString("topologies/boundary/type", "unstructured");
    mesh_group->createViewString("topologies/boundary/coordset", "coords");
    mesh_group->createViewString("topologies/boundary/elements/shape",
                                 ShapeNameFromGeometry(boundary_geometry));

    int* boundary_connectivity = CreateOwnedIntArray(mesh_group,
                                                     "topologies/boundary/elements/connectivity",
                                                     static_cast<IndexType>(num_boundary_elements) *
                                                         vertices_per_boundary_element);
    for (int boundary_index = 0; boundary_index < num_boundary_elements; ++boundary_index) {
        const int* boundary_vertices = mesh.GetBdrElement(boundary_index)->GetVertices();
        for (int local_vertex = 0; local_vertex < vertices_per_boundary_element; ++local_vertex) {
            boundary_connectivity[static_cast<std::size_t>(boundary_index) *
                                      vertices_per_boundary_element +
                                  local_vertex] = boundary_vertices[local_vertex];
        }
    }

    mesh_group->createViewString("fields/boundary_material_attribute/association", "element");
    mesh_group->createViewString("fields/boundary_material_attribute/topology", "boundary");
    int* boundary_attributes = CreateOwnedIntArray(
        mesh_group, "fields/boundary_material_attribute/values", num_boundary_elements);
    for (int boundary_index = 0; boundary_index < num_boundary_elements; ++boundary_index) {
        boundary_attributes[boundary_index] = mesh.GetBdrAttribute(boundary_index);
    }
}

/**
 * @brief Write the communication groups and their shared entities under `adjsets/mesh`
 *
 * @param mesh Parallel mesh being written
 * @param mesh_group Group being written into
 *
 * @details Writes one Sidre group per communication group g = 1..NGroups()-1 (see the
 * background notes at the top of this file for what a communication group is). Each holds:
 * - `local_group_index`: g itself. The reader puts the groups back in this order. The order
 *   determines the lproc numbering and the message order of the group communicators, so
 *   rebuilding it exactly keeps the restarted run's parallel communication, and therefore
 *   its floating-point reduction order, identical to the original run's. The reader does not
 *   rely on Sidre or HDF5 returning the groups in creation order.
 * - `neighbors`: the MPI ranks of the other group members.
 * - `values`: the shared vertices, as local vertex indices.
 * - `edges` (2D and 3D): the shared edges as vertex pairs, in the stored orientation.
 * - `triangles`, `quadrilaterals` (3D): the shared faces as vertex tuples, in the stored
 *   orientation.
 *
 * The shared entities are written in the order the group's table row lists them, which the
 * partitioning ParMesh constructor made identical on every member rank (it walks the global
 * entities in global order). The reader rebuilds each row in that same order, so the
 * per-group order agreement survives the round trip.
 *
 * Every group is written, including groups that share only vertices, so the rebuilt mesh has
 * exactly the original set of groups. The group name `g<master rank>_<master group>` is the
 * same on every member rank, which makes the files easier to inspect; nothing depends on it.
 *
 * **Recovering the stored orientation.** ParMesh does not expose the stored edge and face
 * tuples directly. GroupEdge(), GroupTriangle() and GroupQuadrilateral() return the local
 * edge or face index and the orientation o of the stored entity relative to that local
 * entity, which is what ParFiniteElementSpace consumes. Writing the local entity's vertices
 * instead of the stored ones would be wrong whenever o is not the identity. Every rank would
 * rebuild its shared entity in its own local orientation, every rank would then report o as
 * the identity, and ranks whose local orientations differ would list the interior DOFs of
 * that edge or face in different orders. So the stored tuple is rebuilt from the local one:
 * - Edges: GetEdgeVertices() always returns a local edge as (smaller, larger), and
 *   GroupEdge() reports o = +1 exactly when the stored edge's first vertex is the smaller one
 *   (`o = (v[0] < v[1]) ? +1 : -1`). Hence stored = (o > 0) ? (smaller, larger) : (larger,
 *   smaller).
 * - Faces: GroupTriangle() / GroupQuadrilateral() compute
 *   o = GetTriOrientation() / GetQuadOrientation(base = local face, test = stored face), and
 *   those return the o for which test[Orient[o][i]] == base[i] for every i
 *   (Geometry::Constants<...>::Orient in fem/geom.hpp). So the stored face is obtained by
 *   scattering: stored[Orient[o][i]] = local_face[i].
 * On read, the rebuilt mesh holds these stored tuples, and GroupEdge(), GroupTriangle() and
 * GroupQuadrilateral() return the same orientations as in the writing run.
 */
void WriteAdjacencySets(const mfem::ParMesh& mesh, Group* mesh_group) {
    const int dimension = mesh.Dimension();
    Group* adjset_group = mesh_group->createGroup("adjsets/mesh");
    adjset_group->createViewString("association", "vertex");
    adjset_group->createViewString("topology", "mesh");
    Group* communication_groups = adjset_group->createGroup("groups");

    mfem::Array<int> local_entity_vertices;
    const auto& triangle_orient = mfem::Geometry::Constants<mfem::Geometry::TRIANGLE>::Orient;
    const auto& quad_orient = mfem::Geometry::Constants<mfem::Geometry::SQUARE>::Orient;

    // Group 0 is this rank on its own and shares nothing, so it is not written.
    for (int group_index = 1; group_index < mesh.GetNGroups(); ++group_index) {
        const std::string group_name = "g" +
                                       std::to_string(mesh.gtopo.GetGroupMasterRank(group_index)) +
                                       "_" +
                                       std::to_string(mesh.gtopo.GetGroupMasterGroup(group_index));
        Group* shared_group = communication_groups->createGroup(group_name);
        shared_group->createViewScalar("local_group_index", group_index);

        // Neighbor ranks. GetGroup() lists the members as lproc numbers; this rank is always
        // lproc 0, and every other lproc is translated back to its MPI rank, since lproc
        // numbers mean nothing outside this GroupTopology.
        const int member_count = mesh.gtopo.GetGroupSize(group_index);
        const int* member_lprocs = mesh.gtopo.GetGroup(group_index);
        int* neighbor_ranks = CreateOwnedIntArray(shared_group, "neighbors", member_count - 1);
        int neighbor_slot = 0;
        for (int member = 0; member < member_count; ++member) {
            if (member_lprocs[member] != 0) {
                neighbor_ranks[neighbor_slot++] = mesh.gtopo.GetNeighborRank(member_lprocs[member]);
            }
        }

        // Shared vertices: a vertex has no orientation, so its local index is all there is.
        const int shared_vertex_count = mesh.GroupNVertices(group_index);
        int* shared_vertices = CreateOwnedIntArray(shared_group, "values", shared_vertex_count);
        for (int i = 0; i < shared_vertex_count; ++i) {
            shared_vertices[i] = mesh.GroupVertex(group_index, i);
        }

        if (dimension >= 2) {
            // Shared edges, converted from the local (smaller, larger) order to the stored
            // order using GroupEdge()'s orientation (see the function documentation).
            const int shared_edge_count = mesh.GroupNEdges(group_index);
            int* shared_edges_out = CreateOwnedIntArray(
                shared_group, "edges", 2 * static_cast<IndexType>(shared_edge_count));
            for (int i = 0; i < shared_edge_count; ++i) {
                int local_edge = -1;
                int orientation = 0;
                mesh.GroupEdge(group_index, i, local_edge, orientation);
                mesh.GetEdgeVertices(local_edge, local_entity_vertices);
                const int smaller = local_entity_vertices[0];
                const int larger = local_entity_vertices[1];
                shared_edges_out[2 * i + 0] = (orientation > 0) ? smaller : larger;
                shared_edges_out[2 * i + 1] = (orientation > 0) ? larger : smaller;
            }
        }

        if (dimension >= 3) {
            // Shared faces, converted from the local face's vertex order to the stored order
            // by scattering through Orient[o] (see the function documentation). A group can
            // share both triangles and quadrilaterals only in mixed meshes, which are not
            // supported, but both lists are always written so the layout does not depend on
            // the element type.
            const int shared_triangle_count = mesh.GroupNTriangles(group_index);
            int* shared_triangles_out = CreateOwnedIntArray(
                shared_group, "triangles", 3 * static_cast<IndexType>(shared_triangle_count));
            for (int i = 0; i < shared_triangle_count; ++i) {
                int local_face = -1;
                int orientation = 0;
                mesh.GroupTriangle(group_index, i, local_face, orientation);
                mesh.GetFaceVertices(local_face, local_entity_vertices);
                for (int j = 0; j < 3; ++j) {
                    shared_triangles_out[3 * i + triangle_orient[orientation][j]] =
                        local_entity_vertices[j];
                }
            }

            const int shared_quad_count = mesh.GroupNQuadrilaterals(group_index);
            int* shared_quads_out = CreateOwnedIntArray(
                shared_group, "quadrilaterals", 4 * static_cast<IndexType>(shared_quad_count));
            for (int i = 0; i < shared_quad_count; ++i) {
                int local_face = -1;
                int orientation = 0;
                mesh.GroupQuadrilateral(group_index, i, local_face, orientation);
                mesh.GetFaceVertices(local_face, local_entity_vertices);
                for (int j = 0; j < 4; ++j) {
                    shared_quads_out[4 * i + quad_orient[orientation][j]] =
                        local_entity_vertices[j];
                }
            }
        }
    }
}

/**
 * @brief Write the ExaConstit metadata and validation counts under `exaconstit/`
 *
 * @param mesh Parallel mesh being written
 * @param mesh_group Group being written into
 * @param element_geometry Mesh-wide element geometry
 * @param boundary_geometry Mesh-wide boundary geometry (-1 if there are no boundary elements)
 *
 * @details The format version lets the reader reject layouts it does not understand. The
 * dimensions and geometries are what the reader needs to size and create the mesh, and the
 * topology counts are compared after the rebuild (see MeshTopologyCounts). The vertex,
 * element and boundary element counts also give the expected lengths of the stored arrays.
 */
void WriteMeshMetadata(const mfem::ParMesh& mesh,
                       Group* mesh_group,
                       const int element_geometry,
                       const int boundary_geometry) {
    Group* metadata = mesh_group->createGroup("exaconstit");
    metadata->createViewScalar("format_version", MESH_CHECKPOINT_FORMAT_VERSION);
    metadata->createViewScalar("dimension", mesh.Dimension());
    metadata->createViewScalar("space_dimension", mesh.SpaceDimension());
    metadata->createViewScalar("element_geometry", element_geometry);
    metadata->createViewScalar("boundary_geometry", boundary_geometry);
    const MeshTopologyCounts counts = CountMeshTopology(mesh);
    metadata->createViewScalar("num_vertices", counts.num_vertices);
    metadata->createViewScalar("num_elements", counts.num_elements);
    metadata->createViewScalar("num_boundary_elements", counts.num_boundary_elements);
    metadata->createViewScalar("num_edges", counts.num_edges);
    metadata->createViewScalar("num_faces", counts.num_faces);
    metadata->createViewScalar("num_communication_groups", counts.num_communication_groups);
}

} // namespace

// =========================================================================
// TOPOLOGY COUNTS
// =========================================================================

bool MeshTopologyCounts::Matches(const MeshTopologyCounts& other) const {
    return num_vertices == other.num_vertices && num_elements == other.num_elements &&
           num_boundary_elements == other.num_boundary_elements && num_edges == other.num_edges &&
           num_faces == other.num_faces &&
           num_communication_groups == other.num_communication_groups;
}

MeshTopologyCounts CountMeshTopology(const mfem::ParMesh& mesh) {
    MeshTopologyCounts counts;
    counts.num_vertices = mesh.GetNV();
    counts.num_elements = mesh.GetNE();
    counts.num_boundary_elements = mesh.GetNBE();
    counts.num_edges = mesh.GetNEdges();
    counts.num_faces = mesh.GetNFaces();
    // GetNGroups() includes group 0 (this rank alone), which shares nothing.
    counts.num_communication_groups = mesh.GetNGroups() - 1;
    return counts;
}

MeshTopologyCounts ReadStoredMeshTopologyCounts(Group* mesh_group) {
    Group* metadata = mesh_group->getGroup("exaconstit");
    MeshTopologyCounts counts;
    counts.num_vertices = ReadSidreScalar<std::int64_t>(metadata, "num_vertices");
    counts.num_elements = ReadSidreScalar<std::int64_t>(metadata, "num_elements");
    counts.num_boundary_elements = ReadSidreScalar<std::int64_t>(metadata, "num_boundary_elements");
    counts.num_edges = ReadSidreScalar<std::int64_t>(metadata, "num_edges");
    counts.num_faces = ReadSidreScalar<std::int64_t>(metadata, "num_faces");
    counts.num_communication_groups = ReadSidreScalar<std::int64_t>(metadata,
                                                                    "num_communication_groups");
    return counts;
}

// =========================================================================
// WRITER
// =========================================================================

/**
 * @details Implementation notes. The writer records exactly what this rank's ParMesh holds,
 * with no renumbering of any kind, because the reader has to reproduce every numbering and
 * ordering listed in the background notes at the top of this file. Each piece is written by
 * its own helper, documented above:
 * - AgreeOnMeshGeometries(): the only collective step (MPI_Allreduce).
 * - WriteCoordinates(): local vertex positions.
 * - WriteVolumeTopology(): element connectivity and attributes (region IDs).
 * - WriteBoundaryTopology(): boundary element connectivity and attributes.
 * - WriteAdjacencySets(): communication groups and shared entities in stored orientation.
 * - WriteMeshMetadata(): format version, dimensions, geometries and validation counts.
 *
 * Nonconforming and NURBS meshes keep extra structures (NCMesh, NURBSExtension) that this
 * layout does not describe, so they are rejected up front rather than written incompletely.
 */
void WriteParMeshBlueprint(const mfem::ParMesh& mesh, Group* mesh_group) {
    CALI_CXX_MARK_SCOPE("exaconstit::WriteParMeshBlueprint");

    MFEM_VERIFY(mesh_group != nullptr, "WriteParMeshBlueprint was given a null mesh group");
    MFEM_VERIFY_0(!mesh.Nonconforming() && mesh.NURBSext == nullptr,
                  "Restart checkpoints do not support nonconforming or NURBS meshes");

    const auto [element_geometry, boundary_geometry] = AgreeOnMeshGeometries(mesh);
    WriteCoordinates(mesh, mesh_group);
    WriteVolumeTopology(mesh, mesh_group, element_geometry);
    WriteBoundaryTopology(mesh, mesh_group, boundary_geometry);
    WriteAdjacencySets(mesh, mesh_group);
    WriteMeshMetadata(mesh, mesh_group, element_geometry, boundary_geometry);
}

// =========================================================================
// READER
// =========================================================================

/**
 * @details Implementation notes. This replaces MFEM's partitioning ParMesh constructor for a
 * restarted run. That constructor builds the local mesh piece and the parallel structures
 * from the global serial mesh. Here they are rebuilt from what each rank wrote, using the same
 * sequence of protected Mesh / ParMesh calls the MFEM constructors use, in the same order.
 * Read the background notes at the top of this file first; the steps below refer to them.
 *
 * The order of the steps is not arbitrary:
 * - The communicator has to be set before the first collective call.
 * - Every array is checked before anything is built, so a bad checkpoint aborts on every rank
 *   instead of crashing one rank halfway through construction while the others wait in a
 *   collective call.
 * - FinalizeTopology() has to run before the shared structures are rebuilt, and Finalize()
 *   after, because Finalize() looks up the local edge and face of every shared entity in the
 *   edge and face tables FinalizeTopology() built.
 *
 * Any change here should be checked with test_restart_mesh_roundtrip, which compares the
 * rebuilt mesh with the original on 1 to 4 ranks, including shared-entity orientations and the
 * true-DOF layout of H1 spaces up to order 4.
 */
ExaSidreParMesh::ExaSidreParMesh(Group* mesh_group, MPI_Comm comm) {
    CALI_CXX_MARK_SCOPE("exaconstit::ExaSidreParMesh");

    MFEM_VERIFY(mesh_group != nullptr, "ExaSidreParMesh was given a null mesh group");
    MFEM_VERIFY(comm != MPI_COMM_NULL, "ExaSidreParMesh was given a null communicator");
    const std::string abort_summary = "Unable to rebuild the mesh from the restart checkpoint";

    // ---------------------------------------------------------------------
    // Step 0: communicator
    //
    // The ParMesh base class was default constructed, which leaves MyComm, NRanks, MyRank
    // and gtopo's communicator unset. They must be set before the first collective call:
    // FinalizeTopology() calls ParMesh::HasBoundaryElements() and ParMesh::SetAttributes(),
    // both of which use MyComm, and BuildSharedStructures() uses gtopo's communicator. The
    // MFEM constructors do the same thing first.
    // ---------------------------------------------------------------------
    MyComm = comm;
    gtopo.SetComm(comm);
    MPI_Comm_size(MyComm, &NRanks);
    MPI_Comm_rank(MyComm, &MyRank);

    // ---------------------------------------------------------------------
    // Step 1: metadata
    //
    // The format version comes first, since nothing else can be interpreted without it. The
    // dimensions, geometries and local sizes then determine how many entries each array must
    // have. A boundary geometry of -1 means no rank had boundary elements when the checkpoint
    // was written, which is only consistent if this rank has none either.
    // ---------------------------------------------------------------------
    const bool has_metadata = mesh_group->hasGroup("exaconstit");
    int format_version = -1;
    if (has_metadata) {
        format_version = ReadSidreScalar<int>(mesh_group->getGroup("exaconstit"), "format_version");
    }
    VerifyOnAllRanks(has_metadata && format_version == MESH_CHECKPOINT_FORMAT_VERSION,
                     comm,
                     "the mesh group has no exaconstit/ metadata or an unsupported "
                     "format_version (" +
                         std::to_string(format_version) + ", expected " +
                         std::to_string(MESH_CHECKPOINT_FORMAT_VERSION) + ")",
                     abort_summary);

    Group* metadata = mesh_group->getGroup("exaconstit");
    const int dimension = ReadSidreScalar<int>(metadata, "dimension");
    const int space_dimension = ReadSidreScalar<int>(metadata, "space_dimension");
    const int element_geometry = ReadSidreScalar<int>(metadata, "element_geometry");
    const int boundary_geometry = ReadSidreScalar<int>(metadata, "boundary_geometry");
    const MeshTopologyCounts stored_counts = ReadStoredMeshTopologyCounts(mesh_group);
    const int num_vertices = static_cast<int>(stored_counts.num_vertices);
    const int num_elements = static_cast<int>(stored_counts.num_elements);
    const int num_boundary_elements = static_cast<int>(stored_counts.num_boundary_elements);

    const bool geometry_supported = element_geometry >= 0 &&
                                    element_geometry < mfem::Geometry::NUM_GEOMETRIES &&
                                    (boundary_geometry < 0 ||
                                     boundary_geometry < mfem::Geometry::NUM_GEOMETRIES) &&
                                    (boundary_geometry >= 0 || num_boundary_elements == 0);
    VerifyOnAllRanks(geometry_supported,
                     comm,
                     "the mesh metadata has an invalid element or boundary geometry",
                     abort_summary);

    const int vertices_per_element = mfem::Geometry::NumVerts[element_geometry];
    const int vertices_per_boundary_element = boundary_geometry >= 0
                                                  ? mfem::Geometry::NumVerts[boundary_geometry]
                                                  : 0;

    // ---------------------------------------------------------------------
    // Step 2: check every array before building anything
    //
    // Each array must exist, be owned by Sidre (an external view would have no data after
    // load), have the expected type, and have exactly the length the metadata implies. The
    // adjacency-set arrays are checked in BuildSharedStructures(); only the presence of their
    // parent group is checked here. All results are combined into one flag and agreed on by
    // every rank, so either every rank goes on to build the mesh or every rank aborts.
    // ---------------------------------------------------------------------
    bool arrays_ok = true;
    std::array<const double*, 3> axis_values = {nullptr, nullptr, nullptr};
    const std::array<const char*, 3> axis_names = {"x", "y", "z"};
    for (int axis = 0; axis < space_dimension; ++axis) {
        const auto [values, ok] = ReadOwnedArray<double>(
            mesh_group, std::string("coordsets/coords/values/") + axis_names[axis], num_vertices);
        axis_values[axis] = values;
        arrays_ok = arrays_ok && ok;
    }
    const auto [element_connectivity, connectivity_ok] = ReadOwnedArray<int>(
        mesh_group,
        "topologies/mesh/elements/connectivity",
        static_cast<IndexType>(num_elements) * vertices_per_element);
    const auto [element_attributes, element_attributes_ok] = ReadOwnedArray<int>(
        mesh_group, "fields/mesh_material_attribute/values", num_elements);
    arrays_ok = arrays_ok && connectivity_ok && element_attributes_ok;

    const int* boundary_connectivity = nullptr;
    const int* boundary_attributes = nullptr;
    if (boundary_geometry >= 0) {
        const auto [connectivity, boundary_connectivity_ok] = ReadOwnedArray<int>(
            mesh_group,
            "topologies/boundary/elements/connectivity",
            static_cast<IndexType>(num_boundary_elements) * vertices_per_boundary_element);
        const auto [attributes, boundary_attributes_ok] = ReadOwnedArray<int>(
            mesh_group, "fields/boundary_material_attribute/values", num_boundary_elements);
        boundary_connectivity = connectivity;
        boundary_attributes = attributes;
        arrays_ok = arrays_ok && boundary_connectivity_ok && boundary_attributes_ok;
    }
    arrays_ok = arrays_ok && mesh_group->hasGroup("adjsets/mesh/groups");
    VerifyOnAllRanks(arrays_ok,
                     comm,
                     "the mesh arrays are missing, external, or have unexpected sizes or types",
                     abort_summary);

    // ---------------------------------------------------------------------
    // Step 3: build the local (serial) mesh piece
    //
    // InitMesh() only reserves space; AddVertex(), AddElement() and AddBdrElement() then
    // append in order, so local vertex i, element i and boundary element i get exactly the
    // indices they had in the writing run. Every value is copied into storage MFEM owns:
    // AddVertex() copies the coordinates, and SetVertices() copies the vertex indices into
    // the element MFEM allocated with NewElement(). Nothing points into the Sidre buffers
    // afterwards. The Element objects are owned by the mesh once added and are deleted with
    // it.
    //
    // InitMesh() takes the counts from the metadata, but AddVertex() and AddElement() grow
    // the arrays as needed, so the counts are only a size hint; the array checks in step 2
    // are what guarantee the loops below stay in bounds.
    // ---------------------------------------------------------------------
    InitMesh(dimension, space_dimension, num_vertices, num_elements, num_boundary_elements);
    for (int vertex_index = 0; vertex_index < num_vertices; ++vertex_index) {
        mfem::real_t coordinates[3] = {0.0, 0.0, 0.0};
        for (int axis = 0; axis < space_dimension; ++axis) {
            coordinates[axis] = axis_values[axis][vertex_index];
        }
        AddVertex(coordinates);
    }
    for (int element_index = 0; element_index < num_elements; ++element_index) {
        mfem::Element* element = NewElement(element_geometry);
        element->SetVertices(element_connectivity +
                             static_cast<std::size_t>(element_index) * vertices_per_element);
        element->SetAttribute(element_attributes[element_index]);
        AddElement(element);
    }
    for (int boundary_index = 0; boundary_index < num_boundary_elements; ++boundary_index) {
        mfem::Element* boundary_element = NewElement(boundary_geometry);
        boundary_element->SetVertices(boundary_connectivity +
                                      static_cast<std::size_t>(boundary_index) *
                                          vertices_per_boundary_element);
        boundary_element->SetAttribute(boundary_attributes[boundary_index]);
        AddBdrElement(boundary_element);
    }

    // ---------------------------------------------------------------------
    // Step 4: local topology
    //
    // FinalizeTopology() sets the local mesh type (SetMeshGen()), builds the element-to-face
    // and element-to-edge tables, generates the local faces and edges with their local
    // numbering, and sets the attribute lists. The shared-entity lookups in Finalize() (step
    // 6) need these edge and face tables.
    //
    // generate_bdr is false because the boundary was read from the checkpoint. Generating it
    // would add boundary elements with default attributes, which the boundary conditions
    // would not recognize. Two calls inside are collective in a ParMesh and are reached on
    // every rank: HasBoundaryElements() is an MPI_Allreduce and is evaluated before
    // generate_bdr in `!HasBoundaryElements() && generate_bdr`, and SetAttributes() is
    // ParMesh::SetAttributes(), which gathers the attribute lists from every rank so that
    // `attributes` and `bdr_attributes` are mesh-wide.
    // ---------------------------------------------------------------------
    const bool generate_boundary_elements = false;
    FinalizeTopology(generate_boundary_elements);

    // ---------------------------------------------------------------------
    // Step 5: parallel structure
    //
    // FinalizeTopology() computed `meshgen` (which element families are present) from this
    // rank's elements only. ReduceMeshGen() combines it across ranks with a bitwise OR, so
    // every rank, including one with no elements, sees the mesh-wide value that MFEM's
    // parallel code expects. The MFEM constructors do the same.
    //
    // BuildSharedStructures() then rebuilds the communication groups and the shared-entity
    // tables; see its implementation notes.
    // ---------------------------------------------------------------------
    ReduceMeshGen();
    BuildSharedStructures(mesh_group->getGroup("adjsets/mesh/groups"), dimension);

    // ---------------------------------------------------------------------
    // Step 6: finalize
    //
    // ParMesh::Finalize() calls Mesh::Finalize() and then FinalizeParTopo(), which fills in
    // sedge_ledge and sface_lface (the local edge and face of every shared edge and face) by
    // looking up each stored vertex tuple in the local vertex-to-vertex and face tables.
    // ParMesh::Finalize() also saves and restores the mesh-wide meshgen around
    // Mesh::Finalize(), which may recompute it from the local elements.
    //
    // Both flags must stay false. Either one would rewrite element vertex lists that were
    // stored exactly as the writing run had them, and that run's mesh was already finalized:
    // - fix_orientation = true lets CheckElementOrientation() reorder the vertices of any
    //   element it considers inverted.
    // - refine = true runs MarkForRefinement(), which for tetrahedral meshes reorders every
    //   tet's vertices to mark its refinement edge and can rotate the shared triangles.
    // Either change renumbers local DOFs relative to the writing run and can break the stored
    // shared-face orientations, so the restored fields would land on the wrong DOFs.
    // ---------------------------------------------------------------------
    const bool refine = false;
    const bool fix_orientation = false;
    Finalize(refine, fix_orientation);
}

/**
 * @details Implementation notes. This rebuilds everything listed under "Communication
 * groups", "Shared entities" and "Group tables" in the background notes at the top of this
 * file, from the per-group arrays WriteAdjacencySets() wrote. It does the same job as the
 * group-building part of MFEM's partitioning ParMesh constructor (FindSharedFaces(),
 * FindSharedEdges(), FindSharedVertices(), gtopo.Create(), BuildFaceGroup(), BuildEdgeGroup(),
 * BuildVertexGroup() and the shared-element builders), except that the groups and entities
 * are read rather than discovered.
 *
 * Three properties carry over from the writing run, and the rest of the parallel machinery
 * depends on each of them:
 * - Local group order: group g is rebuilt with local index g (from `local_group_index`). This
 *   fixes the lproc numbering and the communication pattern, so the restarted run exchanges
 *   data exactly as the original run did.
 * - Per-group entity order: each table row lists its entities in the stored order, which is
 *   the order every member rank agrees on.
 * - Stored orientation: the shared edge and face tuples are taken verbatim, so GroupEdge(),
 *   GroupTriangle() and GroupQuadrilateral() report the original orientations.
 *
 * `sedge_ledge` and `sface_lface` are only sized here. Their values depend on the local edge
 * and face numbering, and Finalize() computes them from the stored tuples.
 */
void ExaSidreParMesh::BuildSharedStructures(Group* adjset_groups, const int dimension) {
    CALI_CXX_MARK_SCOPE("exaconstit::ExaSidreParMesh::BuildSharedStructures");

    // Pointers into the Sidre arrays of one communication group, plus their tuple counts.
    struct SharedGroupArrays {
        const int* neighbor_ranks = nullptr;
        IndexType neighbor_count = 0;
        const int* vertices = nullptr;
        IndexType vertex_count = 0;
        const int* edges = nullptr;
        IndexType edge_count = 0;
        const int* triangles = nullptr;
        IndexType triangle_count = 0;
        const int* quadrilaterals = nullptr;
        IndexType quadrilateral_count = 0;
    };

    // ---------------------------------------------------------------------
    // Read the stored groups into their original local order
    //
    // shared_groups[g - 1] receives the group whose stored local_group_index is g, no matter
    // in what order Sidre returns the groups. Each index 1..group_count has to appear exactly
    // once; a missing, duplicated or out-of-range index means the adjacency sets are corrupt.
    // The tuple sizes (1 for vertices and neighbors, 2 for edges, 3 for triangles, 4 for
    // quadrilaterals) are checked by ReadIntTuples().
    // ---------------------------------------------------------------------
    const int group_count = static_cast<int>(adjset_groups->getNumGroups());
    std::vector<SharedGroupArrays> shared_groups(static_cast<std::size_t>(group_count));
    std::vector<bool> group_filled(static_cast<std::size_t>(group_count), false);
    bool arrays_ok = true;
    for (Group& group : adjset_groups->groups()) {
        const int local_group_index = group.hasView("local_group_index")
                                          ? ReadSidreScalar<int>(&group, "local_group_index")
                                          : -1;
        if (local_group_index < 1 || local_group_index > group_count ||
            group_filled[static_cast<std::size_t>(local_group_index - 1)]) {
            arrays_ok = false;
            continue;
        }
        group_filled[static_cast<std::size_t>(local_group_index - 1)] = true;
        SharedGroupArrays& arrays = shared_groups[static_cast<std::size_t>(local_group_index - 1)];
        std::tie(arrays.neighbor_ranks,
                 arrays.neighbor_count) = ReadIntTuples(&group, "neighbors", 1, arrays_ok);
        std::tie(arrays.vertices,
                 arrays.vertex_count) = ReadIntTuples(&group, "values", 1, arrays_ok);
        if (dimension >= 2) {
            std::tie(arrays.edges,
                     arrays.edge_count) = ReadIntTuples(&group, "edges", 2, arrays_ok);
        }
        if (dimension >= 3) {
            std::tie(arrays.triangles,
                     arrays.triangle_count) = ReadIntTuples(&group, "triangles", 3, arrays_ok);
            std::tie(arrays.quadrilaterals, arrays.quadrilateral_count) = ReadIntTuples(
                &group, "quadrilaterals", 4, arrays_ok);
        }
    }

    // ---------------------------------------------------------------------
    // Rebuild the group topology
    //
    // GroupTopology::Create() takes the groups as a list of rank sets, in local group order,
    // with this rank alone first (group 0). Each IntegerSet must contain MPI ranks in sorted
    // order: Create() picks each group's master with ListOfIntegerSets::PickElementInSet(),
    // which returns the set's first entry, so sorting is what makes the master the smallest
    // rank, the same choice MFEM's partitioning constructor makes on every member rank.
    //
    // ListOfIntegerSets::Insert() returns the index of an equal set if one is already in the
    // list instead of adding a new one. Two stored groups with the same members would
    // therefore collapse into one and shift every later group's index, so the returned index
    // is checked against the expected one. A valid mesh never has duplicate groups.
    //
    // Create() exchanges group information with each neighbor rank (each non-master member
    // learns the group's index on its master), so every rank must reach it. That is why the
    // array checks above are agreed on across ranks first. The MPI tag is the one MFEM's
    // partitioning constructor uses for the same call.
    // ---------------------------------------------------------------------
    mfem::ListOfIntegerSets communication_sets;
    {
        mfem::IntegerSet local_only_set;
        mfem::Array<int>& local_only = local_only_set;
        local_only.Append(MyRank);
        communication_sets.Insert(local_only_set);
    }
    for (int group_offset = 0; group_offset < group_count; ++group_offset) {
        const SharedGroupArrays& arrays = shared_groups[static_cast<std::size_t>(group_offset)];
        mfem::IntegerSet member_set;
        mfem::Array<int>& members = member_set;
        members.Append(MyRank);
        members.Append(arrays.neighbor_ranks, static_cast<int>(arrays.neighbor_count));
        members.Sort();
        const int inserted_index = communication_sets.Insert(member_set);
        arrays_ok = arrays_ok && (inserted_index == group_offset + 1);
    }
    VerifyOnAllRanks(arrays_ok,
                     MyComm,
                     "the adjacency-set arrays are malformed, have missing or duplicate local "
                     "group indices, or contain duplicate communication groups",
                     "Unable to rebuild the mesh from the restart checkpoint");
    const int group_topology_mpi_tag = 822;
    gtopo.Create(communication_sets, group_topology_mpi_tag);

    // ---------------------------------------------------------------------
    // Size the shared-entity arrays and group tables
    //
    // The shared-entity arrays (svert_lvert, shared_edges, shared_trias, shared_quads) hold
    // every group's entities back to back, group 1 first. The group tables have one row per
    // group 1..N (row r is group r+1), and each row lists the indices of that group's
    // entities in those arrays. Since the arrays are filled group by group in row order, each
    // row's indices are simply consecutive.
    //
    // ParFiniteElementSpace::GetGroupComm() calls GroupNTriangles() and GroupNQuadrilaterals()
    // for every group in every dimension, so below 3D the triangle and quadrilateral tables
    // still need group_count (empty) rows instead of being left unsized. The partitioning
    // constructor also sizes them in every dimension (BuildFaceGroup()).
    // ---------------------------------------------------------------------
    IndexType total_shared_vertices = 0;
    IndexType total_shared_edges = 0;
    IndexType total_shared_faces = 0;
    for (const SharedGroupArrays& arrays : shared_groups) {
        total_shared_vertices += arrays.vertex_count;
        total_shared_edges += arrays.edge_count;
        total_shared_faces += arrays.triangle_count + arrays.quadrilateral_count;
    }

    svert_lvert.SetSize(static_cast<int>(total_shared_vertices));
    group_svert.SetDims(group_count, static_cast<int>(total_shared_vertices));
    if (dimension >= 2) {
        sedge_ledge.SetSize(static_cast<int>(total_shared_edges));
        shared_edges.SetSize(static_cast<int>(total_shared_edges));
        group_sedge.SetDims(group_count, static_cast<int>(total_shared_edges));
    }
    if (dimension >= 3) {
        sface_lface.SetSize(static_cast<int>(total_shared_faces));
        group_stria.MakeI(group_count);
        group_squad.MakeI(group_count);
    } else {
        group_stria.SetSize(group_count, 0);
        group_squad.SetSize(group_count, 0);
    }

    // ---------------------------------------------------------------------
    // Fill the shared-entity arrays and group tables, group by group in local order
    //
    // Vertex and edge tables are written directly in compressed-row form. Table::SetDims()
    // allocated I (row offsets, size group_count + 1) and J (column entries) and set I[0] = 0;
    // after row r is filled, I[r + 1] is set to the running total, which is both the end of
    // row r and the start of row r + 1. J receives consecutive indices into svert_lvert and
    // shared_edges.
    //
    // The face tables use mfem::Table's two-pass construction instead, because triangles and
    // quadrilaterals are counted per row first: MakeI() above zeroed the per-row counts,
    // AddColumnsInRow() records each row's count here, and MakeJ() below turns the counts
    // into offsets.
    //
    // Within each group the entities are appended in stored order, which is the order every
    // member rank agrees on, and the edge and face tuples are copied verbatim, keeping their
    // stored orientation. Each shared edge becomes an mfem::Segment owned by the mesh,
    // with attribute 1 as in MFEM's own constructors. Shared faces are plain vertex tuples
    // (Vert3 / Vert4).
    // ---------------------------------------------------------------------
    int vertices_added = 0;
    int edges_added = 0;
    for (int row = 0; row < group_count; ++row) {
        const SharedGroupArrays& arrays = shared_groups[static_cast<std::size_t>(row)];

        for (IndexType i = 0; i < arrays.vertex_count; ++i) {
            group_svert.GetJ()[vertices_added] = vertices_added;
            svert_lvert[vertices_added] = arrays.vertices[i];
            ++vertices_added;
        }
        group_svert.GetI()[row + 1] = vertices_added;

        if (dimension >= 2) {
            for (IndexType i = 0; i < arrays.edge_count; ++i) {
                group_sedge.GetJ()[edges_added] = edges_added;
                shared_edges[edges_added] = new mfem::Segment(arrays.edges + 2 * i, 1);
                ++edges_added;
            }
            group_sedge.GetI()[row + 1] = edges_added;
        }

        if (dimension >= 3) {
            for (IndexType i = 0; i < arrays.triangle_count; ++i) {
                const int* v = arrays.triangles + 3 * i;
                shared_trias.Append(Vert3(v[0], v[1], v[2]));
            }
            for (IndexType i = 0; i < arrays.quadrilateral_count; ++i) {
                const int* v = arrays.quadrilaterals + 4 * i;
                shared_quads.Append(Vert4(v[0], v[1], v[2], v[3]));
            }
            group_stria.AddColumnsInRow(row, static_cast<int>(arrays.triangle_count));
            group_squad.AddColumnsInRow(row, static_cast<int>(arrays.quadrilateral_count));
        }
    }

    if (dimension >= 3) {
        // sface_lface holds the triangles first and then the quadrilaterals, so its size has
        // to equal the two lists combined for FinalizeParTopo() to index it correctly.
        MFEM_VERIFY(shared_trias.Size() + shared_quads.Size() == sface_lface.Size(),
                    "Shared face count mismatch while rebuilding the mesh from the restart "
                    "checkpoint");
        // Row offsets from the per-row counts, then consecutive column entries: triangle
        // (or quadrilateral) j of the whole list is entry j of the table.
        group_stria.MakeJ();
        for (int i = 0; i < shared_trias.Size(); ++i) {
            group_stria.GetJ()[i] = i;
        }
        group_squad.MakeJ();
        for (int i = 0; i < shared_quads.Size(); ++i) {
            group_squad.GetJ()[i] = i;
        }
    }
}

} // namespace exaconstit

#endif // EXACONSTIT_USE_AXOM
