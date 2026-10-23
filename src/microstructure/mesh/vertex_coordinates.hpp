#pragma once

#include "mfem.hpp"

/**
 * @file vertex_coordinates.hpp
 * @brief The position of every mesh vertex, read from the geometry the mesh really has and made
 * identical on every rank that holds a copy of the vertex.
 *
 * **Why MFEM's vertex array is not enough.** mfem::Mesh stores two descriptions of the geometry:
 * the vertex array (Mesh::GetVertex) and, for high-order or curved meshes, the "Nodes" (an H1
 * GridFunction holding the coordinates of every nodal point). For a mesh with Nodes the Nodes are
 * the geometry; the vertex array is only a convenience copy, and it is not reliable:
 *  - ParMesh fills it by averaging the Nodes vertex values over the local elements around each
 *    vertex (pmesh.cpp:307-309 -> Mesh::SetVerticesFromNodes, mesh.cpp:7243-7254 ->
 *    GridFunction::GetNodalValues). The average is summed over a different set of elements on
 *    each partition, so the same vertex can get different last bits on different ranks and for
 *    different rank counts;
 *  - Mesh::Transform and other node motion change only the Nodes (mesh.cpp:14072-14078), so the
 *    vertex array then describes the old geometry ("stale vertices").
 *
 * Even the Nodes vertex values can differ between the copies of a shared vertex: after
 * ParMesh::UniformRefinement every element interpolates its own new nodal values, and two ranks
 * holding the same vertex can disagree in the last bits (up to about 5e-15 in the audit
 * reproducer, also for straight meshes refined in parallel as in the ExaConstit hook).
 *
 * **What this class stores.** For every local vertex v (0 <= v < Mesh::GetNV()) three coordinates
 * (2D meshes get z = 0):
 *  - mesh with continuous (H1) Nodes: the Nodes value at the vertex's degree of freedom. The H1
 *    space numbers one dof per vertex, dof = vertex index (FiniteElementSpace::GetVertexDofs,
 *    fespace.cpp:3838-3846); FiniteElementSpace::DofToVDof gives each component's entry for either
 *    storage order (byNODES or byVDIM);
 *  - mesh without Nodes, or with discontinuous (L2) Nodes (MFEM's topologically periodic meshes,
 *    which have no vertex dofs): the vertex array, which is then the geometry (or the only
 *    per-vertex data there is).
 *
 * **Owner broadcast (ParMesh only).** A vertex on a partition boundary belongs to a *group* of
 * ranks (MFEM's GroupTopology, ParMesh::gtopo); one rank of the group is its *master* (owner).
 * The owner's three coordinates are copied to every other copy with MFEM's GroupCommunicator
 * (an object that knows, for every shared entity, which ranks hold it and in which order), so all
 * copies of a vertex are bitwise identical afterwards. The communicator is set up over the
 * ParMesh's own shared-vertex lists (ParMesh::GroupNVertices / GroupVertex, the same lists the
 * H1 space uses for its vertex dofs, pfespace.cpp GetGroupComm), with three entries per vertex.
 * The owner is the group master in both cases, so the result equals a broadcast through the
 * Nodes space's own GroupComm(); building it from the mesh groups avoids a dof-sized copy of the
 * Nodes and also covers meshes without Nodes. No finite element space and no mesh copy is
 * created.
 *
 * When the broadcast matters:
 *  - meshes refined in parallel (with or without Nodes: a new face-center vertex of a hex is an
 *    average of four vertices, whose summation order can differ between the two ranks of the
 *    face);
 *  - meshes whose Nodes were moved on each rank separately.
 * For a mesh distributed from a serial mesh and not changed since, all copies are already
 * identical (ParMesh copies the serial values bit for bit, pmesh.cpp:296-304), and the broadcast
 * changes nothing; in particular results for such meshes without Nodes are unchanged.
 *
 * **What it does not do.** It cannot make two different runs agree: the owner's value itself is
 * computed by one of the owner's elements and depends on the partition after a parallel
 * refinement. Consumers that need the same answer for every rank count (element keys) quantize
 * the coordinates as well (mesh/element_key.hpp).
 *
 * **Memory.** 3 doubles per local vertex, plus the group communicator's lists of shared vertices
 * while the constructor runs. Host data.
 */

namespace exaconstit::microstructure {

/**
 * @brief Per-vertex coordinates from the mesh's true geometry, identical on every rank holding
 * the vertex (see the file comment).
 */
class VertexCoordinates {
public:
    /**
     * @brief Reads the coordinates of every local vertex and, for a ParMesh, broadcasts the
     * owner's values to all copies.
     *
     * **Collective when `mesh` is an mfem::ParMesh** (detected with dynamic_cast): every rank of
     * the mesh communicator must construct it, in the same order relative to other collective
     * calls, even a rank without vertices (the broadcast exchanges messages with the
     * neighboring ranks). For a serial mfem::Mesh nothing is communicated. Host only.
     *
     * @param mesh a serial mesh or this rank's ParMesh partition; spaceDim <= 3. Read only
     *             during construction (the coordinates are copied), never modified.
     */
    explicit VertexCoordinates(const mfem::Mesh& mesh);

    /** @brief Number of local vertices NV (= Mesh::GetNV()). */
    int NumVertices() const {
        return m_num_vertices;
    }

    /**
     * @brief Pointer to the 3 coordinates (x, y, z; mesh length units) of local vertex v,
     * 0 <= v < NumVertices(). Host memory, valid while the object lives.
     */
    const double* Vertex(int v) const {
        // GetData() is the host pointer; the constructor filled the host copy (HostWrite) and
        // nothing writes the Vector afterwards, so it stays valid
        return m_xyz.GetData() + 3 * v;
    }

    /**
     * @brief All coordinates, size 3 NV, layout xyz[d + 3 v] (component d of vertex v). Host
     * data (the Vector's host copy is valid).
     */
    const mfem::Vector& Data() const {
        return m_xyz;
    }

    /** @brief True when the values come from continuous Nodes, false for the vertex array. */
    bool FromNodes() const {
        return m_from_nodes;
    }

    /** @brief True when the owner values were broadcast (the mesh is a ParMesh). */
    bool Broadcast() const {
        return m_broadcast;
    }

private:
    int m_num_vertices = 0;
    bool m_from_nodes = false;
    bool m_broadcast = false;
    mfem::Vector m_xyz;
};

} // namespace exaconstit::microstructure
