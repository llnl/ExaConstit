#include "microstructure/mesh/serial_order.hpp"

#include "microstructure/util/collective_errors.hpp"

#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

// Collective error agreement (every rank throws or none does) comes from
// util/collective_errors.hpp: ThrowWithLocalMessageIfAnyRankFailed lets a failing rank report its
// own details while the other ranks report that the check failed elsewhere.

/*
 * BuildSerialOrder
 *
 * Algorithm:
 *  1. Rank 0 asks MFEM for a partitioning of the serial mesh into `nranks` parts
 *     (`Mesh::GeneratePartitioning`, `mesh.cpp:9010-9032`). This needs MFEM built with METIS;
 *     without it MFEM aborts via mfem_error (`mesh.cpp:9330`), there is no fallback.
 *  2. Rank 0 broadcasts it (MPI_Bcast: one rank sends the same array to all others). Computing
 *     it once and broadcasting guarantees all ranks agree, even if the partitioner were not
 *     deterministic across processes.
 *  3. Every rank walks the serial elements in increasing id and records the ids it owns. This
 *     is exactly the order `ParMesh::BuildLocalElements` (`pmesh.cpp:365-393`) uses for local
 *     elements, so `local_to_serial[i]` is the serial id of local element i.
 */
SerialOrder BuildSerialOrder(mfem::Mesh& serial, MPI_Comm comm) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // Step 1: partition on rank 0 only.
    const int num_serial_elements = serial.GetNE();
    SerialOrder order;
    order.partitioning.SetSize(num_serial_elements);
    if (rank == 0) {
        // GeneratePartitioning allocates the array with new[] and hands ownership to the
        // caller, so copy it into our mfem::Array and free it with delete[].
        int* parts = serial.GeneratePartitioning(nranks, 1);
        for (int e = 0; e < num_serial_elements; e++) {
            order.partitioning[e] = parts[e];
        }
        delete[] parts;
    }

    // Step 2: everyone receives rank 0's partitioning. The count is each rank's own
    // serial.GetNE(), so all ranks must pass the same serial mesh (documented precondition).
    MPI_Bcast(order.partitioning.GetData(), num_serial_elements, MPI_INT, 0, comm);

    // Step 3: count the elements this rank owns, then list their serial ids in increasing
    // order (two passes so the array is sized exactly once).
    int num_local = 0;
    for (int e = 0; e < num_serial_elements; e++) {
        if (order.partitioning[e] == rank) {
            num_local++;
        }
    }
    order.local_to_serial.SetSize(num_local);
    int next_local = 0;
    for (int e = 0; e < num_serial_elements; e++) {
        if (order.partitioning[e] == rank) {
            order.local_to_serial[next_local] = e;
            next_local++;
        }
    }
    return order;
}

/*
 * MakeParMesh
 *
 * Algorithm:
 *  1. Check the inputs: the mesh must be conforming (see the header for why) and the
 *     partitioning must have one entry per serial element. The check results are combined
 *     over all ranks before anyone throws, because the ParMesh constructor below is collective.
 *  2. Build the ParMesh with our partitioning (`ParMesh(comm, mesh, partitioning)`,
 *     `pmesh.cpp:106`). The constructor copies the partitioning; `order` keeps its own copy.
 *  3. Confirm that this rank got exactly `order.local_to_serial.Size()` elements. If not, the
 *     local-to-serial map would be wrong, so all ranks throw (again after a reduction, since
 *     callers usually go on to more collective calls).
 */
std::unique_ptr<mfem::ParMesh>
MakeParMesh(MPI_Comm comm, mfem::Mesh& serial, const SerialOrder& order) {
    // Step 1: input checks. The nonconforming test comes first because it explains the more
    // fundamental problem.
    std::string input_error;
    if (serial.Nonconforming()) {
        input_error = "the serial mesh is nonconforming; the local-to-serial element map is only "
                      "defined for conforming meshes";
    } else if (order.partitioning.Size() != serial.GetNE()) {
        input_error = "partitioning has " + std::to_string(order.partitioning.Size()) +
                      " entries but the serial mesh has " + std::to_string(serial.GetNE()) +
                      " elements";
    }
    ThrowWithLocalMessageIfAnyRankFailed(!input_error.empty(), input_error, "MakeParMesh", comm);

    // Step 2: distribute the mesh. Each rank keeps the serial elements whose partitioning entry
    // equals its rank, in increasing serial order.
    auto pmesh = std::make_unique<mfem::ParMesh>(comm, serial, order.partitioning.GetData());

    // Step 3: the local element count must match the map built by BuildSerialOrder.
    const bool count_mismatch = pmesh->GetNE() != order.local_to_serial.Size();
    std::string count_error;
    if (count_mismatch) {
        count_error = "ParMesh has " + std::to_string(pmesh->GetNE()) +
                      " local elements but the serial order expects " +
                      std::to_string(order.local_to_serial.Size());
    }
    ThrowWithLocalMessageIfAnyRankFailed(count_mismatch, count_error, "MakeParMesh", comm);
    return pmesh;
}

/*
 * GatherToSerial
 *
 * Algorithm:
 *  1. Check that every rank passed one value per local element (reduced over all ranks so a
 *     bad size on one rank cannot leave the others waiting in MPI_Gatherv).
 *  2. On `root`, compute how many values each rank sends (`counts[p]` = number of serial
 *     elements owned by rank p) and where each rank's block starts in the receive buffer
 *     (`displs[p]` = counts[0] + ... + counts[p-1]; an *exclusive prefix sum*, i.e. a running
 *     total that excludes the current entry). Only `root` needs these, and it can compute them
 *     from the partitioning without extra communication.
 *  3. MPI_Gatherv: rank p sends its `local_values`, and `root` stores them at
 *     `recv[displs[p] .. displs[p] + counts[p] - 1]`. So `recv` holds rank 0's block, then
 *     rank 1's block, and so on.
 *  4. Reorder on `root`. Within each rank's block the values are in increasing serial id (see
 *     BuildSerialOrder). Walking the serial ids e = 0, 1, 2, ... and taking the next unused
 *     value from the block of rank partitioning[e] therefore yields the value of serial
 *     element e. `cursor[p]` is the position of the next unused value in rank p's block.
 */
std::vector<int> GatherToSerial(MPI_Comm comm,
                                const SerialOrder& order,
                                const mfem::Array<int>& local_values,
                                int root) {
    // Step 1: size check, consistent across ranks.
    const bool size_mismatch = local_values.Size() != order.local_to_serial.Size();
    std::string size_error;
    if (size_mismatch) {
        size_error = "got " + std::to_string(local_values.Size()) + " local values for " +
                     std::to_string(order.local_to_serial.Size()) + " local elements";
    }
    ThrowWithLocalMessageIfAnyRankFailed(size_mismatch, size_error, "GatherToSerial", comm);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const int num_serial_elements = order.partitioning.Size();

    // Step 2: receive counts and block offsets, needed on root only. On the other ranks these
    // vectors stay empty; MPI ignores the receive arguments there.
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<int> recv;
    if (rank == root) {
        counts.assign(static_cast<std::size_t>(nranks), 0);
        for (int e = 0; e < num_serial_elements; e++) {
            const auto owner = static_cast<std::size_t>(order.partitioning[e]);
            counts[owner]++;
        }
        displs.assign(static_cast<std::size_t>(nranks), 0);
        for (std::size_t p = 1; p < static_cast<std::size_t>(nranks); p++) {
            displs[p] = displs[p - 1] + counts[p - 1];
        }
        recv.resize(static_cast<std::size_t>(num_serial_elements));
    }

    // Step 3: gather the variable-length blocks onto root.
    MPI_Gatherv(local_values.GetData(),
                local_values.Size(),
                MPI_INT,
                recv.data(),
                counts.data(),
                displs.data(),
                MPI_INT,
                root,
                comm);

    // Step 4: reorder rank-blocked data into serial element order.
    std::vector<int> serial_values;
    if (rank == root) {
        serial_values.resize(static_cast<std::size_t>(num_serial_elements));
        std::vector<int> cursor = displs;
        for (int e = 0; e < num_serial_elements; e++) {
            const auto owner = static_cast<std::size_t>(order.partitioning[e]);
            const auto recv_index = static_cast<std::size_t>(cursor[owner]);
            serial_values[static_cast<std::size_t>(e)] = recv[recv_index];
            cursor[owner]++;
        }
    }
    return serial_values;
}

} // namespace exaconstit::microstructure
