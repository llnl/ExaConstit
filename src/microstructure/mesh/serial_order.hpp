#pragma once

#include "mfem.hpp"

#include <memory>
#include <mpi.h>
#include <vector>

/**
 * @file serial_order.hpp
 * @brief Links a distributed ParMesh back to the element numbering of the serial mesh it came
 *        from.
 *
 * **Why this is needed.** The generator assigns a grain to every element of a distributed
 * `mfem::ParMesh`, where each MPI rank owns only its own piece of the mesh (its *mesh
 * partition*) and numbers those elements 0..NE_local-1. The output files (serial mesh files and
 * the AUTO-mesh `grain_file`) are indexed by the element id of the original serial mesh, which
 * is the same on every rank count. So we need, for every local element, its serial element id.
 *
 * **How the map is obtained.** A *partitioning* is an array with one entry per serial element
 * holding the rank that owns it. When `mfem::ParMesh` is built from a conforming serial mesh and
 * a partitioning, each rank copies its elements by walking the serial elements in increasing
 * order and keeping those whose partitioning entry equals its rank
 * (`ParMesh::BuildLocalElements`, `pmesh.cpp:365-393`). Local element i is therefore the i-th
 * serial element owned by this rank, and the partitioning alone defines the map.
 *
 * Do not use `ParMesh::GetGlobalElementNum` for this: it numbers rank 0's elements first, then
 * rank 1's, and so on (a rank-offset numbering), which is not the serial id.
 *
 * All functions in this file are *collective*: every rank of the communicator must call them,
 * in the same order, or the program hangs inside MPI.
 */

namespace exaconstit::microstructure {

/**
 * @brief Partitioning of a serial mesh and the map from local to serial element ids.
 *
 * Built by BuildSerialOrder. `partitioning` is identical on every rank; `local_to_serial`
 * describes this rank's mesh partition only.
 */
struct SerialOrder {
    /**
     * @brief Owning rank of each serial element.
     *
     * Size: serial NE. Entry `partitioning[e]` (e = 0-based serial element id) is the 0-based
     * rank in the communicator that owns element e.
     */
    mfem::Array<int> partitioning;
    /**
     * @brief Serial element id of each local element.
     *
     * Size: number of local elements on this rank. Entry `local_to_serial[i]` is the 0-based
     * serial element id of 0-based local element i. Strictly increasing.
     */
    mfem::Array<int> local_to_serial;
};

/**
 * @brief Partitions `serial` over the ranks of `comm` and builds the local-to-serial map.
 *
 * Rank 0 calls `serial.GeneratePartitioning(nranks, 1)` (METIS graph partitioning) and
 * broadcasts the result, so every rank uses the same partitioning even if METIS were not
 * deterministic across processes. Each rank then lists the serial elements it owns, in
 * increasing order, which is the order ParMesh will use for its local elements.
 *
 * Requires MFEM built with METIS (`MFEM_USE_METIS`). There is no fallback: without METIS,
 * `Mesh::GeneratePartitioning` calls `mfem_error` (aborts), even for a single part
 * (`mesh.cpp:9012`, `mesh.cpp:9330`).
 *
 * Collective over `comm`. Runs on the host.
 *
 * Precondition: `serial.GetNE()` is identical on every rank. It is used as the MPI_Bcast
 * count, so a mismatch is undefined behavior in MPI (typically a hang or a truncation error).
 *
 * @param serial the full serial mesh. Every rank must pass the same mesh (same element count
 *        and ordering). Non-const only because `GeneratePartitioning` is non-const; the mesh is
 *        not modified (MFEM only builds and frees a temporary element adjacency table).
 * @param comm communicator whose ranks will own the partitions.
 * @return the partitioning (size serial NE, same on all ranks) and this rank's
 *         `local_to_serial` map.
 */
SerialOrder BuildSerialOrder(mfem::Mesh& serial, MPI_Comm comm);

/**
 * @brief Builds the ParMesh with the partitioning in `order`.
 *
 * Calls `mfem::ParMesh(comm, serial, order.partitioning)`, then checks that this rank received
 * exactly the elements `order.local_to_serial` predicts.
 *
 * Collective over `comm`. Runs on the host.
 *
 * Requires a conforming serial mesh. A nonconforming mesh (one with hanging nodes from local
 * refinement, stored by MFEM in an `NCMesh`) takes a different ParMesh code path through
 * `ParNCMesh`, whose local element order is not tied to the serial order.
 *
 * @param comm communicator; must be the one `order` was built with.
 * @param serial the same serial mesh passed to BuildSerialOrder. Non-const because the MFEM
 *        constructor takes a non-const reference.
 * @param order result of BuildSerialOrder for `serial` and `comm`.
 * @return the distributed mesh; its local element i is serial element
 *         `order.local_to_serial[i]`.
 * @throws std::runtime_error if the mesh is nonconforming, `order` does not match `serial`, or
 *         the local element count differs from `order.local_to_serial`. Every check is reduced
 *         over `comm`, so either all ranks throw or none does (no rank is left waiting in a
 *         later collective call).
 */
std::unique_ptr<mfem::ParMesh>
MakeParMesh(MPI_Comm comm, mfem::Mesh& serial, const SerialOrder& order);

/**
 * @brief Gathers one int per local element onto `root`, arranged in serial element order.
 *
 * Typical use: gather the grain id of every local element so `root` can write the serial
 * `grain_file`. Uses `MPI_Gatherv` (each rank sends a block of a different length to `root`,
 * which stores the blocks one after another), then reorders the received blocks into serial
 * order using the partitioning.
 *
 * Collective over `comm`. Runs on the host.
 *
 * @param comm communicator `order` was built with.
 * @param order result of BuildSerialOrder.
 * @param local_values one value per local element, indexed by 0-based local element id
 *        (size `order.local_to_serial.Size()`).
 * @param root 0-based rank that receives the result.
 * @return on `root`: the values indexed by 0-based serial element id (size serial NE);
 *         on the other ranks: an empty vector.
 * @throws std::runtime_error if `local_values` does not have one entry per local element on
 *         some rank. The check is reduced over `comm`, so all ranks throw together.
 */
std::vector<int> GatherToSerial(MPI_Comm comm,
                                const SerialOrder& order,
                                const mfem::Array<int>& local_values,
                                int root);

} // namespace exaconstit::microstructure
