#pragma once

#include <climits>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>

/**
 * @file collective_errors.hpp
 * @brief Helpers that make every MPI rank stop with the same error when a check fails on only
 * some of the ranks.
 *
 * **The problem.** Many steps of the generator are *collective*: every rank of the communicator
 * must call them, in the same order, because the ranks exchange data inside them (MPI_Allreduce,
 * a neighbor exchange, ...). If one rank finds a problem with its own part of the mesh (an
 * unsupported element, a wrong array size) and simply throws, the other ranks carry on into the
 * next collective call and wait there forever for the rank that left: the job hangs instead of
 * stopping with a message.
 *
 * **The pattern.** Every rank first decides locally whether it failed, then all ranks agree on
 * the global answer with one MPI_Allreduce, and then either all ranks throw or none does:
 * - AnyRank: logical OR of a flag over the ranks (MPI_Allreduce with MPI_MAX on 0/1 ints);
 * - LowestRankWith: the lowest rank whose flag is set (MPI_Allreduce with MPI_MIN), used to pick
 *   one rank whose detailed message or element id is shown to everybody;
 * - BroadcastString: sends a message from one rank to all (two MPI_Bcast calls);
 * - ThrowIfAnyRankFailed: the lowest failing rank broadcasts its message and every rank throws it;
 * - ThrowIfAnyRank: every rank throws the same caller-supplied message;
 * - ThrowWithLocalMessageIfAnyRankFailed: failing ranks throw their own message, the others a
 *   short "failed on another rank" note;
 * - AgreeOrReport: no exception; every rank learns whether all succeeded, and the lowest failing
 *   rank prints its message (for programs that stop with an exit code, like the generator tool).
 *
 * All functions are host only and **collective** over `comm`: every rank must call them with its
 * own flag or message, even a rank that has nothing to report.
 */

namespace exaconstit::microstructure {

/**
 * @brief True on every rank if `local` is true on at least one rank (a logical OR over ranks).
 *
 * One MPI_Allreduce with MPI_MAX on 0/1 integers; every rank receives the same answer, so a loop
 * or branch controlled by it behaves identically on all ranks. Collective; host only.
 */
inline bool AnyRank(bool local, MPI_Comm comm) {
    int local_flag = 0;
    if (local) {
        local_flag = 1;
    }
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, comm);
    return global_flag != 0;
}

/**
 * @brief The lowest rank of `comm` whose `local` flag is true, or -1 if no rank has it set.
 *
 * One MPI_Allreduce with MPI_MIN over "my rank if my flag is set, else INT_MAX". The minimum is
 * exact and the same on every rank. Collective; host only.
 */
inline int LowestRankWith(bool local, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int candidate = INT_MAX;
    if (local) {
        candidate = rank;
    }
    int lowest = INT_MAX;
    MPI_Allreduce(&candidate, &lowest, 1, MPI_INT, MPI_MIN, comm);
    if (lowest == INT_MAX) {
        return -1;
    }
    return lowest;
}

/**
 * @brief Sends `text` from rank `root` to every rank of `comm` and returns it (on every rank).
 *
 * Two MPI_Bcast calls: first the length, then the characters. `text` is only read on `root`.
 * Collective; host only.
 */
inline std::string BroadcastString(const std::string& text, int root, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int length = 0;
    if (rank == root) {
        length = static_cast<int>(text.size());
    }
    MPI_Bcast(&length, 1, MPI_INT, root, comm);
    std::string message;
    if (rank == root) {
        message = text;
    } else {
        message.assign(static_cast<std::size_t>(length), ' ');
    }
    if (length > 0) {
        MPI_Bcast(message.data(), length, MPI_CHAR, root, comm);
    }
    return message;
}

/**
 * @brief Throws the same std::runtime_error on every rank if any rank passes a non-empty
 * `local_error`.
 *
 * Steps: (1) the lowest failing rank is found with LowestRankWith; (2) it broadcasts its message
 * (BroadcastString); (3) every rank throws std::runtime_error with that message, followed by
 * " (reported by rank r)" when `append_rank` is true. Returns normally on every rank when no
 * rank failed.
 *
 * Collective; host only.
 *
 * @param comm        communicator.
 * @param local_error this rank's error message; empty means "no error on this rank".
 * @param append_rank append the failing rank's number to the message.
 */
inline void ThrowIfAnyRankFailed(MPI_Comm comm,
                                 const std::string& local_error,
                                 bool append_rank = true) {
    // Step 1: lowest failing rank, or none
    const int failing_rank = LowestRankWith(!local_error.empty(), comm);
    if (failing_rank < 0) {
        return;
    }
    // Step 2: its message on every rank
    std::string message = BroadcastString(local_error, failing_rank, comm);
    // Step 3: the same exception everywhere
    if (append_rank) {
        message += " (reported by rank " + std::to_string(failing_rank) + ")";
    }
    throw std::runtime_error(message);
}

/**
 * @brief Throws `Exception(message)` on every rank if `local_failed` is true on at least one
 * rank (AnyRank). Every rank uses the same message, so it should describe the problem in general
 * terms. Collective; host only.
 *
 * @tparam Exception exception type constructible from a std::string (default
 *                   std::invalid_argument, for bad input data).
 */
template <typename Exception = std::invalid_argument>
void ThrowIfAnyRank(MPI_Comm comm, bool local_failed, const std::string& message) {
    if (AnyRank(local_failed, comm)) {
        throw Exception(message);
    }
}

/**
 * @brief Throws std::runtime_error on every rank if any rank failed: a failing rank uses
 * "caller: local_message", every other rank "caller: a consistency check failed on another rank",
 * so the error is visible in every rank's output. Collective; host only.
 *
 * @param local_failed  this rank's verdict.
 * @param local_message details, used only on failing ranks.
 * @param caller        name of the calling function, prefixed to the message.
 * @param comm          communicator.
 */
inline void ThrowWithLocalMessageIfAnyRankFailed(bool local_failed,
                                                 const std::string& local_message,
                                                 const std::string& caller,
                                                 MPI_Comm comm) {
    if (!AnyRank(local_failed, comm)) {
        return;
    }
    if (local_failed) {
        throw std::runtime_error(caller + ": " + local_message);
    }
    throw std::runtime_error(caller + ": a consistency check failed on another rank");
}

/**
 * @brief Collective verdict on a step that each rank performed on its own (reading a file,
 * validating options, building a mesh), without throwing: returns true on every rank if
 * `local_ok` is true on every rank, false on every rank otherwise.
 *
 * Steps: AnyRank on the failure flag tells every rank whether any rank failed; on failure,
 * LowestRankWith picks the lowest failing rank, which alone prints
 * "<program>: error (rank r): <message>" to std::cerr (the rank part only when there is more than
 * one rank), so the user sees one message even when every rank failed the same way. An empty
 * `message` prints nothing. The caller then stops all ranks together, e.g. by returning an exit
 * code from main.
 *
 * Collective (two MPI_Allreduce calls on failure, one on success); host only.
 *
 * @param local_ok this rank's verdict.
 * @param message  what went wrong on this rank (used only on the reporting rank).
 * @param comm     communicator.
 * @param program  prefix of the printed line (e.g. the program name).
 * @return true on every rank if every rank succeeded.
 */
inline bool AgreeOrReport(bool local_ok,
                          const std::string& message,
                          MPI_Comm comm,
                          const std::string& program) {
    if (!AnyRank(!local_ok, comm)) {
        return true;
    }
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const int reporter = LowestRankWith(!local_ok, comm);
    if (rank == reporter && !message.empty()) {
        std::cerr << program << ": error";
        if (nranks > 1) {
            std::cerr << " (rank " << rank << ")";
        }
        std::cerr << ": " << message << std::endl;
    }
    return false;
}

} // namespace exaconstit::microstructure
