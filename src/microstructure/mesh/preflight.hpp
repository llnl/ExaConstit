#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file preflight.hpp
 * @brief Collective checks that a mesh meets ExaConstit's and the generator's requirements.
 *
 * Run these before labeling so that an unusable mesh fails early with a readable message
 * instead of failing deep inside a kernel or the restart writer. The requirements are:
 * - a 3D volume mesh (the generator pipeline is 3D only);
 * - one element geometry over the whole mesh, e.g. all hexes or all tets (ExaConstit's
 *   quadrature-space kernels and restart checkpoint, `partial_qspace.cpp`,
 *   `exa_sidre_parmesh.cpp`);
 * - a conforming mesh, i.e. no hanging nodes from local refinement (same reason);
 * - a positive Jacobian determinant (detJ > 0) at every quadrature point, i.e. no inverted or
 *   folded elements (the samplers divide by element volume);
 * - no periodic identification. MFEM represents a periodic mesh with discontinuous (L2) Nodes,
 *   and grain-boundary fitting does not support it.
 *
 * A *collective* function must be called by every rank of the communicator; each rank
 * computes local results and the ranks then combine them with MPI_Allreduce (a reduction whose
 * result is delivered to every rank).
 */

namespace exaconstit::microstructure {

/**
 * @brief Result of CheckMesh.
 *
 * All fields except `local_bad_elements` are global reductions over the communicator and are
 * identical on every rank, so decisions based on them (e.g. throwing) are made consistently.
 */
struct PreflightReport {
    /** @brief Mesh dimension and space dimension are both 3 on every rank with elements. */
    bool is_3d = true;
    /**
     * @brief Bitmask of the element geometries present on any rank: bit g is set when some
     *        element has `mfem::Geometry::Type` g (e.g. CUBE, TETRAHEDRON).
     */
    int geometry_mask = 0;
    /** @brief Exactly one element geometry globally (an empty mesh fails). */
    bool single_geometry = true;
    /** @brief No rank has a nonconforming (NCMesh) mesh. */
    bool conforming = true;
    /** @brief No rank has discontinuous (L2) Nodes, i.e. the mesh is not periodic. */
    bool continuous_nodes = true;
    /**
     * @brief Minimum signed detJ over all quadrature points of all elements (physical volume
     *        per unit reference-element volume). Equals `std::numeric_limits<double>::max()`
     *        for an empty mesh.
     */
    double min_detj = 0.0;
    /** @brief Global number of elements with detJ <= 0 at one or more quadrature points. */
    std::int64_t num_bad_elements = 0;
    /** @brief Global number of elements. */
    std::int64_t num_elements = 0;
    /**
     * @brief 0-based local element ids (on this rank only) of the elements with detJ <= 0, in
     *        increasing order. For a ParMesh built by MakeParMesh, the serial id of local
     *        element `id` is `SerialOrder::local_to_serial[id]`.
     */
    std::vector<int> local_bad_elements;

    /** @brief True when every check passed. Same value on every rank. */
    bool Passed() const {
        return is_3d && single_geometry && conforming && continuous_nodes &&
               num_bad_elements == 0;
    }
};

/**
 * @brief Runs all checks on the local part `mesh` of a mesh distributed over `comm`.
 *
 * Use this overload for a serial mesh (pass MPI_COMM_SELF) or for hand-built per-rank meshes.
 * detJ is evaluated at the quadrature points of DefaultSampleQuadOrder(mesh), the same points
 * the samplers use.
 *
 * Collective over `comm`. Runs on the host. Never throws for a bad mesh; inspect the report or
 * call ThrowOnPreflightFailure.
 *
 * @param mesh this rank's part of the mesh (may have zero elements).
 * @param comm communicator over which the results are combined.
 */
PreflightReport CheckMesh(const mfem::Mesh& mesh, MPI_Comm comm);

/**
 * @brief Runs all checks on a ParMesh.
 *
 * Same as `CheckMesh(pmesh, pmesh.GetComm())`. Collective over `pmesh.GetComm()`; host only.
 */
PreflightReport CheckMesh(const mfem::ParMesh& pmesh);

/**
 * @brief Human-readable summary of the failed checks (empty when the report passed).
 *
 * One line per failed check. For inverted elements it includes up to 10 of this rank's local
 * element ids, so the text can differ between ranks. Local; host only.
 */
std::string DescribePreflightFailures(const PreflightReport& report);

/**
 * @brief Throws std::runtime_error with DescribePreflightFailures() if the report failed.
 *
 * Local (no MPI calls), but because the report is globally reduced, every rank throws or none
 * does, so callers can catch and abort collectively without deadlocking. Host only.
 */
void ThrowOnPreflightFailure(const PreflightReport& report);

} // namespace exaconstit::microstructure
