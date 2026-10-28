/*
 * Implementation of VertexCoordinates (vertex_coordinates.hpp).
 *
 * Terms used in this file:
 * - Group (MFEM GroupTopology): a set of ranks that share some mesh entities (vertices, edges,
 *   faces) on a partition boundary. Group 0 is "this rank alone" and holds nothing shared. Every
 *   group has a master rank, the owner of the group's entities.
 * - GroupCommunicator: MFEM's object that sends values of shared entities between the ranks of
 *   each group; Bcast copies the master's values to the other ranks (communication.hpp:420-436).
 *   It only exchanges messages with neighboring ranks.
 * - Collective: a function every rank of the communicator must call.
 */
#include "microstructure/mesh/vertex_coordinates.hpp"

namespace exaconstit::microstructure {

/*
 * VertexCoordinates constructor
 *
 * Steps:
 *  1. Coordinate source: continuous (H1) Nodes when the mesh has them, else the vertex array
 *     (no Nodes, or discontinuous L2 Nodes, which have no vertex dofs).
 *  2. Local values: per vertex v, either the Nodes value at its dof (GetVertexDofs gives the one
 *     scalar dof of vertex v, dof = v for H1, fespace.cpp:3838-3846; DofToVDof the entry of each
 *     component), or Mesh::GetVertex(v). Missing components (2D meshes) are 0.
 *  3. ParMesh only (dynamic_cast): owner broadcast. The group -> local-entry table of a new
 *     GroupCommunicator over the mesh's group topology (ParMesh::gtopo) is filled the way
 *     ParFiniteElementSpace::GetGroupComm fills it for vertex dofs (pfespace.cpp, the "vertices"
 *     block): row g >= 1 lists, for every shared vertex j of group g (ParMesh::GroupVertex, the
 *     same order on every rank of the group), the three entries 3 k, 3 k + 1, 3 k + 2 of local
 *     vertex k in our xyz array; row 0 (unshared) stays empty. Finalize() allocates the buffers,
 *     then Bcast(data, 0) (layout 0: the array has one entry per local entry) copies the master's
 *     entries to every other rank of the group. Unshared vertices are not touched.
 */
VertexCoordinates::VertexCoordinates(const mfem::Mesh& mesh) {
    // Step 1: which description of the geometry to read
    m_num_vertices = mesh.GetNV();
    const int sdim = mesh.SpaceDimension();
    const mfem::GridFunction* nodes = mesh.GetNodes();
    if (nodes != nullptr &&
        nodes->FESpace()->FEColl()->GetContType() == mfem::FiniteElementCollection::CONTINUOUS) {
        m_from_nodes = true;
    }

    // Step 2: local values, layout xyz[d + 3 v]
    m_xyz.SetSize(3 * m_num_vertices);
    double* xyz = m_xyz.HostWrite();
    const double* node_data = m_from_nodes ? nodes->HostRead() : nullptr;
    mfem::Array<int> dofs;
    for (int v = 0; v < m_num_vertices; ++v) {
        for (int d = 0; d < 3; ++d) {
            xyz[d + 3 * v] = 0.0;
        }
        if (m_from_nodes) {
            const mfem::FiniteElementSpace* fes = nodes->FESpace();
            fes->GetVertexDofs(v, dofs);
            for (int d = 0; d < sdim; ++d) {
                xyz[d + 3 * v] = node_data[fes->DofToVDof(dofs[0], d)];
            }
        } else {
            const double* x = mesh.GetVertex(v);
            for (int d = 0; d < sdim; ++d) {
                xyz[d + 3 * v] = x[d];
            }
        }
    }

    // Step 3: owner broadcast over the shared-vertex groups (ParMesh only; collective)
    const auto* pmesh = dynamic_cast<const mfem::ParMesh*>(&mesh);
    if (pmesh == nullptr) {
        return;
    }
    m_broadcast = true;
    mfem::GroupCommunicator gcomm(pmesh->gtopo);
    mfem::Table& group_entries = gcomm.GroupLDofTable();
    const int num_groups = pmesh->GetNGroups();

    // Count the entries of all shared groups (3 per shared vertex), then fill row by row
    int num_entries = 0;
    for (int g = 1; g < num_groups; ++g) {
        num_entries += 3 * pmesh->GroupNVertices(g);
    }
    group_entries.SetDims(num_groups, num_entries);
    int* row_start = group_entries.GetI();
    int* entry = group_entries.GetJ();
    row_start[0] = 0;
    int next = 0;
    if (num_groups > 1) {
        row_start[1] = 0; // group 0 (this rank alone) has no shared entries
    }
    for (int g = 1; g < num_groups; ++g) {
        for (int j = 0; j < pmesh->GroupNVertices(g); ++j) {
            const int k = pmesh->GroupVertex(g, j);
            for (int d = 0; d < 3; ++d) {
                entry[next++] = 3 * k + d;
            }
        }
        row_start[g + 1] = next;
    }
    gcomm.Finalize();

    // Master's values to every copy (host array, layout 0)
    gcomm.Bcast(xyz, 0);
}

} // namespace exaconstit::microstructure
