#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <memory>
#include <mpi.h>

namespace axom {
namespace sidre {
class Group;
} // namespace sidre
} // namespace axom

namespace exaconstit {

/**
 * @brief Version of the mesh subtree layout written by WriteParMeshBlueprint()
 *
 * Stored in `mesh/exaconstit/format_version` of every checkpoint rank file. Bump it whenever
 * the layout written by WriteParMeshBlueprint() changes, so older checkpoints are rejected
 * with a clear message instead of being misread.
 */
inline constexpr int MESH_CHECKPOINT_FORMAT_VERSION = 1;

/**
 * @brief Per-rank topology counts used to validate a reconstructed mesh
 *
 * @details Edges and faces are not stored in a checkpoint; MFEM regenerates them from the
 * element connectivity. Comparing these counts with the ones recorded at write time checks
 * that the regenerated local topology matches the original.
 */
struct MeshTopologyCounts {
    /** @brief Number of local vertices (Mesh::GetNV) */
    std::int64_t num_vertices = 0;
    /** @brief Number of local elements (Mesh::GetNE) */
    std::int64_t num_elements = 0;
    /** @brief Number of local boundary elements (Mesh::GetNBE) */
    std::int64_t num_boundary_elements = 0;
    /** @brief Number of local edges (Mesh::GetNEdges) */
    std::int64_t num_edges = 0;
    /** @brief Number of local faces (Mesh::GetNFaces) */
    std::int64_t num_faces = 0;
    /** @brief Number of communication groups, not counting group 0 (this rank alone) */
    std::int64_t num_communication_groups = 0;

    /**
     * @brief Check whether every count agrees with another set of counts
     *
     * @param other Counts to compare against
     * @return True if all six counts are equal
     */
    bool Matches(const MeshTopologyCounts& other) const;
};

/**
 * @brief Count the local topology of a parallel mesh
 *
 * @param mesh Parallel mesh to count
 * @return Topology counts of this rank's piece of the mesh
 *
 * @details Purely local; no MPI communication takes place.
 */
MeshTopologyCounts CountMeshTopology(const mfem::ParMesh& mesh);

/**
 * @brief Read the topology counts recorded by WriteParMeshBlueprint()
 *
 * @param mesh_group Sidre group that WriteParMeshBlueprint() wrote into
 * @return Topology counts stored under `exaconstit/` in the group
 */
MeshTopologyCounts ReadStoredMeshTopologyCounts(axom::sidre::Group* mesh_group);

/**
 * @brief Write this rank's piece of a parallel mesh into a Sidre group
 *
 * @param mesh Parallel mesh to write
 * @param mesh_group Empty Sidre group to write into
 *
 * @details Writes the local vertices, elements, boundary elements and attributes, plus the
 * description of how this piece connects to the other ranks' pieces (communication groups
 * and shared vertices, edges and faces), so that ExaSidreParMesh can rebuild the identical
 * ParMesh on the same rank. The layout follows the Conduit Blueprint mesh conventions, with
 * ExaConstit additions under `adjsets/` and `exaconstit/`. All arrays are owned by Sidre.
 *
 * Collective over mesh.GetComm().
 *
 * @pre The mesh is conforming and not NURBS, all elements share one geometry and all boundary
 * elements share one geometry (segment, triangle, quadrilateral, tetrahedron or hexahedron),
 * and mfem::real_t is double. Violations abort on every rank.
 */
void WriteParMeshBlueprint(const mfem::ParMesh& mesh, axom::sidre::Group* mesh_group);

/**
 * @brief Parallel mesh rebuilt from the output of WriteParMeshBlueprint()
 *
 * @details Restarted runs use this in place of reading the original mesh file, which may no
 * longer exist. The result is a finalized, conforming mfem::ParMesh identical to the one that
 * was written: the same local vertex and element numbering, element vertex order,
 * attributes, communication groups, shared entities and shared-entity orientation. Finite
 * element spaces built on it therefore number their local and true DOFs exactly as in the
 * writing run.
 *
 * Invariants callers must respect:
 * - The run must use the same number of MPI ranks as the writing run, and rank r must read
 *   the group that rank r wrote.
 * - The mesh has no Nodes grid function. Code that needs high-order or deformed coordinates
 *   must supply them (SimulationState overlays the checkpointed coordinates).
 *
 * This is a subclass only because the members needed to rebuild a ParMesh from its pieces
 * are protected; it adds nothing beyond its constructor.
 */
class ExaSidreParMesh : public mfem::ParMesh {
public:
    /**
     * @brief Rebuild a parallel mesh from a checkpoint's mesh group
     *
     * @param mesh_group Group written by WriteParMeshBlueprint(). Non-const because Sidre's
     *                   typed accessor View::getData<T>() is non-const.
     * @param comm Communicator with the same number of ranks as the writing run
     *
     * @details Collective over comm. Every array is checked on every rank before anything is
     * built, and a bad checkpoint aborts on every rank. Nothing points into the Sidre buffers
     * once the constructor returns, so the DataStore holding mesh_group may be destroyed
     * right away.
     */
    ExaSidreParMesh(axom::sidre::Group* mesh_group, MPI_Comm comm);

    ExaSidreParMesh(const ExaSidreParMesh&) = delete;
    ExaSidreParMesh& operator=(const ExaSidreParMesh&) = delete;
    ~ExaSidreParMesh() override = default;

    /**
     * @brief Rebuild a parallel mesh and return it as a shared pointer
     *
     * @param mesh_group Group written by WriteParMeshBlueprint()
     * @param comm Communicator with the same number of ranks as the writing run
     * @return Shared pointer to the rebuilt mesh, ready to assign to SimulationState's mesh
     */
    static std::shared_ptr<ExaSidreParMesh> Create(axom::sidre::Group* mesh_group, MPI_Comm comm) {
        return std::make_shared<ExaSidreParMesh>(mesh_group, comm);
    }

private:
    /**
     * @brief Rebuild the communication groups and the shared-entity tables
     *
     * @param adjset_groups The `adjsets/mesh/groups` group written by WriteParMeshBlueprint()
     * @param dimension Topological dimension of the mesh
     *
     * @details Collective over the mesh communicator. Must be called after ReduceMeshGen()
     * and before Finalize().
     */
    void BuildSharedStructures(axom::sidre::Group* adjset_groups, int dimension);
};

} // namespace exaconstit
