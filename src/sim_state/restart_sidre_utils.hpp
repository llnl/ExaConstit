#pragma once

#include "utilities/unified_logger.hpp"

#include "mfem.hpp"

#include <mpi.h>
#include <string>
#include <type_traits>

#include "axom/sidre.hpp"

/**
 * @brief Small helpers shared by the checkpoint/restart implementation files
 *
 * @details Only included from exa_sidre_parmesh.cpp and restart_checkpoint.cpp, which are
 * compiled when ExaConstit is built with Axom (EXACONSTIT_USE_AXOM). Not part of the public
 * restart interface.
 */
namespace exaconstit::restart_detail {

/**
 * @brief Read a scalar view, converting from whatever numeric type it was stored as
 *
 * @tparam ScalarType Type to return (integer, unsigned integer or floating point)
 * @param group Group that holds the view
 * @param path Path of the scalar view relative to the group
 * @return The stored value converted to ScalarType
 *
 * @details View::getScalar() does not convert between integer widths, and the JSON protocol
 * does not preserve them: Conduit's JSON reader brings every integer back as int64, so an int
 * written to `commit.json` as 1 reads back as 0 through getScalar() into an int. Going through
 * conduit::Node::to_int64(), to_uint64() and to_float64() makes every read independent of the
 * protocol and of the exact type the writer used.
 */
template <typename ScalarType>
ScalarType ReadSidreScalar(axom::sidre::Group* group, const std::string& path) {
    const auto& node = group->getView(path)->getNode();
    if constexpr (std::is_floating_point<ScalarType>::value) {
        return static_cast<ScalarType>(node.to_float64());
    } else if constexpr (std::is_unsigned<ScalarType>::value) {
        return static_cast<ScalarType>(node.to_uint64());
    } else {
        return static_cast<ScalarType>(node.to_int64());
    }
}

/**
 * @brief Check whether a condition holds on every rank
 *
 * @param local_value This rank's result
 * @param comm Communicator to reduce over
 * @return True on every rank if local_value is true on every rank, false on every rank
 * otherwise
 *
 * @details Collective over comm. Used to turn rank-local checks into a decision every rank
 * agrees on, so that either all ranks continue or all ranks abort together. Aborting on only
 * the ranks that failed would leave the others waiting forever in their next collective call.
 */
inline bool AllRanksTrue(const bool local_value, MPI_Comm comm) {
    int local_flag = local_value ? 1 : 0;
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MIN, comm);
    return global_flag == 1;
}

/**
 * @brief Abort on every rank if a rank-local check failed on any rank
 *
 * @param local_ok This rank's result
 * @param comm Communicator to reduce over
 * @param local_message Description of the problem, reported by each rank where it occurred
 * @param summary Message rank 0 aborts with, shared by every rank
 *
 * @details Collective over comm. Each failing rank prints its own warning, since only that
 * rank knows the details, and then every rank aborts with the same summary through
 * MFEM_ABORT_0, so the summary appears once.
 */
inline void VerifyOnAllRanks(const bool local_ok,
                             MPI_Comm comm,
                             const std::string& local_message,
                             const std::string& summary) {
    if (AllRanksTrue(local_ok, comm)) {
        return;
    }
    if (!local_ok) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        MFEM_WARNING("Rank " << rank << ": " << local_message);
    }
    MFEM_ABORT_0(summary << " (see the per-rank warnings above)");
}

} // namespace exaconstit::restart_detail
