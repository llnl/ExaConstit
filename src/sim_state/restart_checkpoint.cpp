// Compiled only when ExaConstit is built with Axom, so this file can always be listed in
// CMake; without EXACONSTIT_USE_AXOM it is an empty translation unit.
#if defined(EXACONSTIT_USE_AXOM)

#include "sim_state/restart_checkpoint.hpp"

#include "sim_state/exa_sidre_parmesh.hpp"
#include "sim_state/restart_sidre_utils.hpp"
#include "sim_state/simulation_state.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "axom/sidre.hpp"

#if !defined(AXOM_USE_HDF5)
#error "ExaConstit restart uses the sidre_hdf5 protocol; build Axom with HDF5 (AXOM_USE_HDF5)."
#endif

// =========================================================================
// BACKGROUND: WHAT A CHECKPOINT CONTAINS AND HOW A RESTART USES IT
// =========================================================================
//
// Read this once before changing anything below; every function in this file implements a
// piece of it. doc/restart.md has the user-facing description and the full design record.
//
// Goal. A restarted run must continue exactly as if the original run had never stopped,
// using only the checkpoint and the options file. The mesh, grain, region-mapping,
// orientation and state-variable files may no longer exist. For a run with the same number of
// ranks, the restarted run is bitwise identical to the uninterrupted one
// (test/test_restart_runs.py checks this on real runs).
//
// Rank file layout. Every rank writes one Sidre DataStore to
// <directory>/cycle_<completed cycles>/rank_<rank>.hdf5 with these groups:
//
//   restart_info/  format version, rank count, this rank, completed cycles, driver step,
//                  completed time, next dt, time-step type, mesh order, ExaConstit version
//   mesh/          this rank's piece of the mesh (exa_sidre_parmesh.cpp), plus
//                  fields/grain_id (the grain ID of every local element)
//   fields/        mesh_ref/values      reference coordinates (owned copy, see below)
//                  mesh_current/values  current coordinates
//                  primal_field/values  velocity true DOFs
//                  mesh_space/          sizes and a DOF-layout hash of the coordinate space
//   qspaces/<name> validation data for each quadrature space (order, size, element subset)
//   qfuncs/<name>  each quadrature function: its space, vdim and size, plus either its
//                  values or `copy_of = <name>` (see WriteQuadratureFunctions())
//   materials/     name, model type, state-variable count and properties of every region
//
// Registry names are made into single Sidre path segments with SidreSafeName(), because
// Sidre uses '/' as its path separator.
//
// Owned and external views. Sidre can either copy data into memory it owns ("owned" view)
// or record a pointer to memory someone else owns ("external" view). The large arrays
// (current coordinates, velocity, quadrature functions) are written as external views
// pointing straight at SimulationState's host data, so writing a checkpoint copies nothing in
// memory. On read, Group::load() restores an external view as a description only (type and
// size, no data). The reader then points each one at the SimulationState buffer it should
// fill (View::setExternalDataPtr()) and a single Group::loadExternalData() call reads all of
// them from the file straight into place. Small and metadata arrays are owned, and so is
// mesh_ref, because it is needed in phase 1, before the buffers phase 2 fills even exist.
//
// Commit protocol. Every rank saves its file, the ranks agree that all saves worked, and only
// then does rank 0 write commit.json (to a temporary name, then renamed, which is atomic on
// POSIX file systems). A checkpoint directory without commit.json is incomplete and is never
// selected for a restart, so a crash in the middle of a write can never be restarted from.
//
// Two-phase restart. A fresh run builds every object from the input files: SimulationState,
// then SystemDriver (the operator, the material models and the solvers), then
// PostProcessingDriver. A restarted run builds them in exactly the same way and then
// overwrites the evolving state with the checkpoint ("build first, overlay last"). Anything a
// constructor initializes (for example ExaCMechModel::InitStateVars()) is simply replaced,
// and quadrature functions registered by any component are restored as long as they exist
// by then. The one unavoidable exception is that SimulationState's constructor would read
// input files that may be gone, so on restart runs it takes those few things from the
// checkpoint instead (phase 1):
//   - the mesh (ReconstructMesh()), since every space is built on it;
//   - the reference coordinates (RestoreReferenceConfiguration()), see that function for why
//     the exact bits matter even during construction;
//   - the grain IDs and region IDs (ReadElementGrainIds() and the element attributes), since
//     they decide which region spaces and functions the constructor creates;
//   - the orientation and state-variable initialization is skipped, since phase 2 overwrites
//     that state anyway.
// Phase 2 (RestoreState()) is the overlay itself, called from the driver once every object
// exists.
//
// Ranks with no elements. Every function here must work on a rank that owns no elements, and
// such a rank must still reach every collective call. On such a rank the mesh arrays, the
// coordinate and velocity vectors and the global quadrature functions all have zero length,
// and the region quadrature spaces and functions do not exist at all (SimulationState only
// creates them where the region has elements). Zero-length arrays are written as described
// but unallocated views and are never dereferenced; their data pointers may be null. Each
// rank only ever reads the file it wrote, so a region that does not exist on a rank is absent
// from both the checkpoint and the restarted run on that rank. test_restart_mesh_roundtrip
// and test_restart_runs both include cases with empty ranks.
//
// DataStore lifetimes. A DataStore that finds Axom's logger (SLIC) uninitialized initializes
// it and finalizes it again in its destructor. Two DataStores alive at the same time could
// therefore leave the second one logging through a finalized logger, so every DataStore in
// this file is destroyed before the next one is created: the write store before the commit
// store, and the commit store before the read store.

namespace fs = std::filesystem;

namespace exaconstit {

namespace {

using axom::sidre::DataStore;
using axom::sidre::Group;
using axom::sidre::IndexType;
using axom::sidre::View;
using restart_detail::AllRanksTrue;
using restart_detail::ReadSidreScalar;

/** @brief File name of the commit marker that makes a checkpoint directory usable */
constexpr const char* COMMIT_FILE_NAME = "commit.json";
/** @brief Sidre protocol of the commit file (small, human readable) */
constexpr const char* COMMIT_PROTOCOL = "sidre_json";
/** @brief Sidre protocol of the rank files */
constexpr const char* RANK_FILE_PROTOCOL = "sidre_hdf5";
/** @brief Number of digits of the cycle count in a checkpoint directory name */
constexpr std::size_t CYCLE_DIGITS = 10;

/**
 * @brief Name of the checkpoint directory for a completed-cycle count
 *
 * @param completed_cycles Number of completed cycles the checkpoint holds
 * @return `cycle_` followed by the count, zero-padded to CYCLE_DIGITS digits
 *
 * @details Zero padding makes the directories sort in cycle order by name as well.
 */
std::string CycleDirectoryName(const std::size_t completed_cycles) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "cycle_%010zu", completed_cycles);
    return std::string(buffer);
}

/**
 * @brief Name of a rank's file inside a checkpoint directory
 *
 * @param rank MPI rank
 * @return `rank_` followed by the rank, zero-padded to 6 digits, and `.hdf5`
 */
std::string RankFileName(const int rank) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "rank_%06d.hdf5", rank);
    return std::string(buffer);
}

/**
 * @brief Parse a checkpoint directory name
 *
 * @param name Directory name (not a path)
 * @param completed_cycles Receives the cycle count when the name matches
 * @return True if the name is exactly `cycle_` followed by CYCLE_DIGITS digits
 *
 * @details The match is exact, so renamed directories (`cycle_<digits>.superseded`) and anything
 * else a user leaves in the checkpoint directory are never mistaken for checkpoints.
 */
bool ParseCycleDirectoryName(const std::string& name, std::size_t& completed_cycles) {
    const std::string prefix = "cycle_";
    if (name.size() != prefix.size() + CYCLE_DIGITS ||
        name.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    for (std::size_t i = prefix.size(); i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
    }
    completed_cycles = static_cast<std::size_t>(std::stoull(name.substr(prefix.size())));
    return true;
}

/**
 * @brief A checkpoint directory found on disk
 */
struct CheckpointDirectoryEntry {
    /** @brief Completed cycles the checkpoint holds (from its name) */
    std::size_t completed_cycles = 0;
    /** @brief Path of the checkpoint directory */
    fs::path path;
    /** @brief Whether the directory has a commit file, i.e. is complete */
    bool committed = false;
};

/**
 * @brief List the checkpoint directories under a directory, oldest first
 *
 * @param directory Checkpoint root directory
 * @return One entry per `cycle_<digits>` subdirectory, sorted by completed cycles; empty if
 * the directory does not exist
 *
 * @details Only rank 0 calls this. File system errors are treated as "no checkpoints" rather
 * than thrown, since the callers decide what a missing checkpoint means.
 */
std::vector<CheckpointDirectoryEntry> ListCheckpointDirectories(const fs::path& directory) {
    std::vector<CheckpointDirectoryEntry> entries;
    std::error_code error;
    if (!fs::is_directory(directory, error)) {
        return entries;
    }
    for (const auto& item : fs::directory_iterator(directory, error)) {
        std::size_t completed_cycles = 0;
        if (item.is_directory(error) &&
            ParseCycleDirectoryName(item.path().filename().string(), completed_cycles)) {
            entries.push_back(
                {completed_cycles, item.path(), fs::exists(item.path() / COMMIT_FILE_NAME, error)});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.completed_cycles < rhs.completed_cycles;
    });
    return entries;
}

/**
 * @brief Turn a SimulationState registry name into a single Sidre path segment
 *
 * @param name Registry name (quadrature space or quadrature function name)
 * @return The name with '%' encoded as "%25" and '/' as "%2F"
 *
 * @details Sidre treats '/' as a path separator, so a name containing it would otherwise
 * create nested groups. The same encoding is used on write and read, so the round trip only
 * needs the two to agree; scripts/restart/compare_restart_checkpoints.cpp uses it too.
 */
std::string SidreSafeName(const std::string& name) {
    std::string encoded;
    encoded.reserve(name.size());
    for (const char character : name) {
        if (character == '/') {
            encoded += "%2F";
        } else if (character == '%') {
            encoded += "%25";
        } else {
            encoded += character;
        }
    }
    return encoded;
}

/**
 * @brief Broadcast a string from rank 0
 *
 * @param value On rank 0 the string to send; on every other rank receives it
 * @param comm Communicator to broadcast over
 *
 * @details Collective. The length is broadcast first so the receivers can size the string.
 */
void BroadcastString(std::string& value, MPI_Comm comm) {
    int length = static_cast<int>(value.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, comm);
    value.resize(static_cast<std::size_t>(length));
    if (length > 0) {
        MPI_Bcast(value.data(), length, MPI_CHAR, 0, comm);
    }
}

/**
 * @brief 64-bit FNV-1a hash
 *
 * @details A simple, deterministic, order-sensitive hash with no dependencies. It is used to
 * fingerprint arrays that are too large to store just for validation (DOF layouts and element
 * subsets). It is not cryptographic; it only has to make an accidental match between two
 * different layouts very unlikely.
 */
class Fnv1a64 {
public:
    /**
     * @brief Add raw bytes to the hash
     *
     * @param data Bytes to add
     * @param byte_count Number of bytes
     */
    void AddBytes(const void* data, const std::size_t byte_count) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < byte_count; ++i) {
            m_state ^= static_cast<std::uint64_t>(bytes[i]);
            m_state *= 1099511628211ULL;
        }
    }

    /**
     * @brief Current hash value
     *
     * @return The hash of every byte added so far
     */
    std::uint64_t Value() const {
        return m_state;
    }

private:
    /** @brief Running hash state, starting from the FNV-1a 64-bit offset basis */
    std::uint64_t m_state = 1469598103934665603ULL;
};

/**
 * @brief Hash of an integer array's contents
 *
 * @param values Array to hash
 * @return FNV-1a hash of the entries in order; the hash of nothing for an empty array
 *
 * @details Empty arrays (for example the element subset of the global quadrature space on a
 * rank with no elements) are never read, since their host pointer may be null.
 */
std::uint64_t HashIntArray(const mfem::Array<int>& values) {
    Fnv1a64 hasher;
    if (values.Size() > 0) {
        hasher.AddBytes(values.HostRead(), static_cast<std::size_t>(values.Size()) * sizeof(int));
    }
    return hasher.Value();
}

/**
 * @brief Hash of a grid function read element by element in each element's DOF order
 *
 * @param field Grid function to hash
 * @return FNV-1a hash of the values visited element by element
 *
 * @details For each local element, the field's values are visited through
 * GetElementVDofs(), i.e. in the order the element's DOFs are numbered locally. If the
 * rebuilt space numbers the local DOFs differently from the writing run, the stored
 * coordinate vector lands on different DOFs and the visited sequence, and so the hash,
 * changes. Coordinates at distinct nodes are distinct, so a renumbering cannot reproduce the
 * same sequence by accident.
 *
 * MFEM encodes a DOF with reversed orientation as a negative index -1 - dof whose value must
 * be negated; that convention is followed so the hash matches what an element actually sees.
 * On a rank with no elements nothing is visited and the hash is that of an empty sequence,
 * the same in both runs.
 */
std::uint64_t ElementLocalFieldHash(const mfem::ParGridFunction& field) {
    const mfem::ParFiniteElementSpace* space = field.ParFESpace();
    const double* values = field.HostRead();
    mfem::Array<int> element_vdofs;
    Fnv1a64 hasher;
    for (int element_index = 0; element_index < space->GetNE(); ++element_index) {
        space->GetElementVDofs(element_index, element_vdofs);
        for (const int signed_vdof : element_vdofs) {
            const int vdof = (signed_vdof >= 0) ? signed_vdof : (-1 - signed_vdof);
            const double value = (signed_vdof >= 0) ? values[vdof] : -values[vdof];
            hasher.AddBytes(&value, sizeof(value));
        }
    }
    return hasher.Value();
}

/**
 * @brief Largest change of a grid function when restricted to true DOFs and prolongated back
 *
 * @param field Grid function to test
 * @return max |field - P(R(field))| over this rank's local DOFs
 *
 * @details R (GridFunction::GetTrueDofs()) keeps one value per true DOF, the one held by the
 * owning rank, and P (ParGridFunction::SetFromTrueDofs()) copies it back to every rank that
 * shares the DOF. A field that was continuous in the writing run survives this unchanged if,
 * and only if, every shared DOF is matched with the same true DOF on every rank. A
 * mismatched shared DOF, for example from a wrong shared-face orientation, receives another
 * DOF's value and shows up as a jump of the order of the element size.
 *
 * Collective: the prolongation exchanges shared values between ranks, so every rank must
 * call this, including ranks with no DOFs (which return 0).
 */
double SharedDofInconsistency(const mfem::ParGridFunction& field) {
    mfem::Vector true_values;
    field.GetTrueDofs(true_values);
    mfem::ParGridFunction prolongated(field.ParFESpace());
    prolongated.SetFromTrueDofs(true_values);
    const double* original = field.HostRead();
    const double* round_trip = prolongated.HostRead();
    double max_difference = 0.0;
    for (int i = 0; i < field.Size(); ++i) {
        max_difference = std::max(max_difference, std::abs(original[i] - round_trip[i]));
    }
    return max_difference;
}

/**
 * @brief Create an external double array view
 *
 * @param group Group to create the view in
 * @param path Path of the new view relative to the group
 * @param data Host data the view points at; must stay valid until the DataStore is saved
 * @param count Number of entries
 *
 * @details No data is copied; Group::save() reads straight from data. Zero-length arrays are
 * only described, since their data pointer may be null. Sidre's external-view API takes a
 * non-const pointer, but saving only reads through it.
 */
void CreateExternalDoubleView(Group* group,
                              const std::string& path,
                              const double* data,
                              const IndexType count) {
    if (count == 0) {
        group->createView(path, axom::sidre::FLOAT64_ID, 0);
        return;
    }
    group->createView(path, axom::sidre::FLOAT64_ID, count, const_cast<double*>(data));
}

/**
 * @brief Create an owned double array view holding a copy of some data
 *
 * @param group Group to create the view in
 * @param path Path of the new view relative to the group
 * @param data Data to copy (may be null when count is 0)
 * @param count Number of entries
 *
 * @details Zero-length arrays are only described, not allocated.
 */
void CreateOwnedDoubleCopy(Group* group,
                           const std::string& path,
                           const double* data,
                           const IndexType count) {
    if (count == 0) {
        group->createView(path, axom::sidre::FLOAT64_ID, 0);
        return;
    }
    double* stored =
        group->createViewAndAllocate(path, axom::sidre::FLOAT64_ID, count)->getData<double*>();
    std::copy(data, data + count, stored);
}

/**
 * @brief Name under which a quadrature space is registered in SimulationState
 *
 * @param spaces SimulationState's quadrature space registry
 * @param space Space to look up
 * @return The registry name, or an empty string if the space is not registered
 *
 * @details Quadrature functions only hold a pointer to their space, so the name is found by
 * searching the registry. It is stored with each quadrature function so the reader can check
 * that the function still lives on the same space.
 */
std::string FindQuadratureSpaceName(
    const std::map<std::string, std::shared_ptr<mfem::expt::PartialQuadratureSpace>>& spaces,
    const mfem::expt::PartialQuadratureSpace* space) {
    for (const auto& [name, candidate] : spaces) {
        if (candidate.get() == space) {
            return name;
        }
    }
    return std::string();
}

/**
 * @brief Quadrature functions that are recomputed before they are next read, and so are
 *        neither written nor restored
 *
 * @param sim_state Simulation state whose registry names to use
 * @return Registry names of the global and every region's tangent stiffness
 *
 * @details The tangent stiffness is the largest quadrature function (36 values per point,
 * both globally and per region) and it is always recomputed before any read:
 * - Every NonlinearMechOperator::Setup() calls the material model's ModelSetup(), which
 *   recomputes the tangent from the converged state. Setup() runs at the start of every
 *   Newton residual evaluation (Mult()) and in SolveInit() after a boundary-condition change
 *   (GetUpdateBCsAction()).
 * - The integrators read tangent values only while assembling the Jacobian
 *   (AssembleElementGrad(), AssembleGradPA(), AssembleEA(), AssembleGradDiagonalPA(),
 *   AddMult[Transpose]GradPA()), which always happens after such a Setup().
 *   ICExaNLFIntegrator::AddMultPA() only uses the function's space (its integration rule),
 *   which the restarted run's constructor-built function provides.
 * If a future change reads the tangent before the first Setup() of a step, it must be removed
 * from this list so it is checkpointed again.
 *
 * The region names are generated for every region, including regions with no elements on
 * this rank; looking them up in a set is harmless when they are absent from the registry.
 */
std::set<std::string> RecomputedQuadratureFunctionNames(const SimulationState& sim_state) {
    std::set<std::string> names;
    names.insert(sim_state.GetQuadratureFunctionMapName("tangent_stiffness", -1));
    for (std::size_t region = 0; region < sim_state.GetNumberOfRegions(); ++region) {
        names.insert(
            sim_state.GetQuadratureFunctionMapName("tangent_stiffness", static_cast<int>(region)));
    }
    return names;
}

/**
 * @brief Collect the external views that have not been pointed at a buffer
 *
 * @param group Group to search, recursively
 * @param out Receives (group, view name) for every external view with a null data pointer
 *
 * @details Group::loadExternalData() tries to read every external view described in the
 * group, bound or not, so the unbound ones (zero-length arrays and quadrature functions
 * present only in the checkpoint) have to be removed first; see RestoreState().
 */
void CollectUnboundExternalViews(Group* group, std::vector<std::pair<Group*, std::string>>& out) {
    for (View& view : group->views()) {
        if (view.isExternal() && view.getVoidPtr() == nullptr) {
            out.emplace_back(group, view.getName());
        }
    }
    for (Group& child : group->groups()) {
        CollectUnboundExternalViews(&child, out);
    }
}

} // namespace

// =========================================================================
// CONSTRUCTION AND FAILURE REPORTING
// =========================================================================

RestartCheckpoint::RestartCheckpoint(const RestartOptions& options, MPI_Comm comm)
    : m_options(options), m_comm(comm) {
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_num_ranks);
}

RestartCheckpoint::~RestartCheckpoint() = default;

void RestartCheckpoint::RecordFailure(const std::string& message) {
    m_local_failed = true;
    m_local_failure_log += "\n  - " + message;
}

/**
 * @details Implementation notes. Most checks in this file can fail on some ranks and not
 * others, for example a malformed array in one rank's file. Aborting only on the failing rank
 * would leave every other rank waiting forever in its next collective call, so each check
 * records its failure (RecordFailure()) and every stage ends with this function, which every
 * rank calls. The ranks agree on whether anyone failed. If so, each failing rank reports its
 * own problems with MFEM_WARNING, since only it knows them, and every rank then aborts with a
 * single shared MFEM_ABORT_0 message.
 */
void RestartCheckpoint::AbortIfAnyRankFailed(const char* stage) {
    if (AllRanksTrue(!m_local_failed, m_comm)) {
        m_local_failure_log.clear();
        return;
    }
    if (m_local_failed) {
        MFEM_WARNING("Restart failed on rank " << m_rank << " while " << stage << ":"
                                               << m_local_failure_log);
    }
    MFEM_ABORT_0("Restart failed while " << stage << " (see the per-rank warnings above)");
}

// =========================================================================
// WRITING
// =========================================================================

/**
 * @details Implementation notes. The decision uses only values every rank agrees on (the
 * options, the time manager and m_last_written_cycles), so every rank makes the same decision
 * and either all ranks or no ranks enter the collective WriteCheckpoint().
 *
 * - IsCheckpointSafe(): the time manager is not in the middle of a retrial or sub-step
 *   sequence, whose counters are not checkpointed. With the current SystemDriver::Solve() this
 *   always holds after a converged step: retries loop inside Solve() until the step succeeds,
 *   AUTO stepping never sub-steps, and FIXED / CUSTOM stepping always report success. The
 *   guard keeps checkpoints correct if sub-stepping across driver iterations is ever added.
 * - Counting cycles since the last checkpoint, rather than testing cycle % interval, keeps
 *   the interval correct after a restart from an arbitrary cycle and if a write is deferred.
 * - `completed_cycles <= m_last_written_cycles` prevents writing the same cycle twice, for
 *   example a final checkpoint right after an interval checkpoint of the same cycle, and
 *   writing immediately after a restart.
 */
void RestartCheckpoint::WriteCheckpointIfDue(SimulationState& sim_state,
                                             const std::size_t driver_step) {
    if (!m_options.write) {
        return;
    }
    const TimeManagement& time_manager = sim_state.m_time_manager;
    if (!time_manager.IsCheckpointSafe()) {
        return;
    }
    const TimeManagement::CheckpointState time_state = time_manager.GetCheckpointState();
    if (time_state.completed_cycles <= m_last_written_cycles) {
        return;
    }
    const std::size_t cycles_since_last = time_state.completed_cycles - m_last_written_cycles;
    const bool interval_due = m_options.interval > 0 &&
                              cycles_since_last >= static_cast<std::size_t>(m_options.interval);
    const bool final_due = time_manager.IsFinished() && m_options.write_final;
    if (!interval_due && !final_due) {
        return;
    }
    WriteCheckpoint(sim_state, driver_step, time_state);
}

/**
 * @details Implementation notes. A checkpoint is written in three stages:
 *
 * 1. Rank 0 prepares the directory (PrepareCheckpointDirectory()). If that fails the
 *    checkpoint is skipped with a warning and the simulation continues; losing one checkpoint
 *    is better than losing the run.
 * 2. Every rank fills a DataStore with its state and saves it to its rank file. The helpers
 *    each write one top-level group (see the background notes at the top of this file for
 *    the layout):
 *    WriteRestartInfo(), WriteMeshAndGrainIds(), WriteNodalFields(),
 *    WriteQuadratureSpaces(), WriteQuadratureFunctions() and WriteMaterials().
 *    The large arrays are external views of SimulationState's host data, so nothing is copied
 *    in memory, and SimulationState must not change between building the store and saving it.
 *    The DataStore is destroyed right after saving, freeing the owned arrays (the mesh copy,
 *    mesh_ref and the small metadata). The external views never owned their memory.
 * 3. Commit. The ranks agree that every save succeeded. Only then does rank 0 write
 *    commit.json, which is what makes the checkpoint usable, and apply the retention policy.
 *    The result is broadcast so every rank updates m_last_written_cycles consistently, and a
 *    checkpoint that was not committed is reported as a warning and does not count, so the
 *    next due cycle tries again.
 *
 * Collective; every rank, including ranks with no elements, takes part in every stage.
 */
void RestartCheckpoint::WriteCheckpoint(SimulationState& sim_state,
                                        const std::size_t driver_step,
                                        const TimeManagement::CheckpointState& time_state) {
    CALI_CXX_MARK_SCOPE("exaconstit::RestartCheckpoint::WriteCheckpoint");
    const double start_seconds = MPI_Wtime();
    const std::size_t completed_cycles = time_state.completed_cycles;
    const double completed_time = time_state.completed_time;
    const fs::path checkpoint_directory = m_options.directory /
                                          CycleDirectoryName(completed_cycles);

    // ---------------------------------------------------------------------
    // Stage 1: directory
    // ---------------------------------------------------------------------
    if (!PrepareCheckpointDirectory(checkpoint_directory)) {
        MFEM_WARNING_0("Unable to prepare the checkpoint directory "
                       << checkpoint_directory << "; the checkpoint for cycle " << completed_cycles
                       << " was skipped and the simulation continues");
        return;
    }

    // ---------------------------------------------------------------------
    // Stage 2: every rank builds and saves its rank file
    // ---------------------------------------------------------------------
    bool saved = false;
    {
        DataStore store;
        Group* root = store.getRoot();
        WriteRestartInfo(root, sim_state, driver_step, time_state);
        WriteMeshAndGrainIds(root, sim_state);
        WriteNodalFields(root, sim_state);
        WriteQuadratureSpaces(root, sim_state);
        WriteQuadratureFunctions(root, sim_state);
        WriteMaterials(root, sim_state);
        saved = root->save((checkpoint_directory / RankFileName(m_rank)).string(),
                           RANK_FILE_PROTOCOL);
    }

    // ---------------------------------------------------------------------
    // Stage 3: commit (only if every rank saved its file)
    // ---------------------------------------------------------------------
    const bool all_saved = AllRanksTrue(saved, m_comm);
    int committed = 0;
    if (all_saved && m_rank == 0) {
        committed = WriteCommitFile(
                        checkpoint_directory, completed_cycles, driver_step, completed_time)
                        ? 1
                        : 0;
        if (committed != 0) {
            ApplyRetentionPolicy();
        }
    }
    MPI_Bcast(&committed, 1, MPI_INT, 0, m_comm);

    if (committed == 0) {
        MFEM_WARNING_0("The checkpoint " << checkpoint_directory
                                         << " was not committed (a rank file could not be "
                                            "saved or the commit file could not be written); "
                                            "the simulation continues");
        return;
    }
    m_last_written_cycles = completed_cycles;
    if (m_rank == 0) {
        std::cout << "Wrote restart checkpoint " << checkpoint_directory.string() << " (cycle "
                  << completed_cycles << ", driver step " << driver_step
                  << ", t = " << completed_time << ") in " << (MPI_Wtime() - start_seconds) << " s"
                  << std::endl;
    }
}

/**
 * @brief Create the directory for a checkpoint about to be written
 *
 * @param checkpoint_directory Directory of the checkpoint
 * @return True on every rank if the directory is ready, false on every rank otherwise
 *
 * @details Collective. Only rank 0 touches the file system, and its result is broadcast so
 * every rank makes the same decision. Rank 0:
 * - supersedes checkpoints of another timeline on this run's first write
 *   (SupersedeCheckpointsAfter());
 * - creates the directory;
 * - removes an existing commit file, so a directory being rewritten for the same cycle is
 *   marked incomplete until the new commit, rather than appearing complete with a mix of old
 *   and new rank files.
 * The supersede step is attempted only once, even if this write fails, since it renames
 * rather than deletes and retrying it gains nothing.
 */
bool RestartCheckpoint::PrepareCheckpointDirectory(const fs::path& checkpoint_directory) {
    int prepared = 1;
    if (m_rank == 0) {
        try {
            if (m_supersede_pending) {
                SupersedeCheckpointsAfter(m_last_written_cycles);
            }
            fs::create_directories(checkpoint_directory);
            fs::remove(checkpoint_directory / COMMIT_FILE_NAME);
        } catch (const fs::filesystem_error& error) {
            MFEM_WARNING_0("Unable to prepare " << checkpoint_directory << ": " << error.what());
            prepared = 0;
        }
    }
    m_supersede_pending = false;
    MPI_Bcast(&prepared, 1, MPI_INT, 0, m_comm);
    return prepared != 0;
}

/**
 * @brief Write the `restart_info/` group
 *
 * @param root Root group of the rank file being written
 * @param sim_state Simulation state being written
 * @param driver_step Driver loop counter of the step just completed
 * @param time_state Resumable time state of the time manager
 *
 * @details The format version, rank count and rank let OpenForRestart() reject a checkpoint
 * that this build or this run cannot read before anything else is touched. The time state is
 * stored as unambiguous quantities (see TimeManagement::CheckpointState) rather than the time
 * manager's raw members, whose meaning depends on its internal state. The previous and
 * original time steps are only used in log messages; storing them lets the restarted run log
 * exactly what the uninterrupted run would have.
 *
 * The driver step is stored separately from the cycle count even though the two are equal
 * with the current SystemDriver::Solve(). The boundary-condition schedule and post-processing
 * are keyed by the driver step, and storing it separately means a restart never relies on the
 * two staying equal.
 *
 * The time-step type, mesh order and ExaConstit version are informational; the mesh order is
 * checked separately through the mesh space (ValidateMeshSpace()).
 */
void RestartCheckpoint::WriteRestartInfo(Group* root,
                                         const SimulationState& sim_state,
                                         const std::size_t driver_step,
                                         const TimeManagement::CheckpointState& time_state) const {
    Group* info = root->createGroup("restart_info");
    info->createViewScalar("format_version", RESTART_FORMAT_VERSION);
    info->createViewScalar("num_ranks", m_num_ranks);
    info->createViewScalar("rank", m_rank);
    info->createViewScalar("completed_cycles",
                           static_cast<std::int64_t>(time_state.completed_cycles));
    info->createViewScalar("driver_step", static_cast<std::int64_t>(driver_step));
    info->createViewScalar("completed_time", time_state.completed_time);
    info->createViewScalar("next_dt", time_state.next_dt);
    info->createViewScalar("previous_dt", time_state.previous_dt);
    info->createViewScalar("original_dt", time_state.original_dt);
    info->createViewScalar("time_type",
                           static_cast<int>(sim_state.m_time_manager.GetTimeStepType()));
    info->createViewScalar("mesh_order", sim_state.m_options.mesh.order);
    info->createViewString("exaconstit_version", sim_state.m_options.version);
}

/**
 * @brief Write the `mesh/` group: this rank's mesh piece and the element grain IDs
 *
 * @param root Root group of the rank file being written
 * @param sim_state Simulation state being written
 *
 * @details The mesh is written by WriteParMeshBlueprint(), which is collective (the ranks
 * agree on the element geometries), so every rank must call this.
 *
 * The grain IDs are written as an extra element field, `fields/grain_id`. A restarted run
 * needs them because the element attributes no longer hold grain IDs once SimulationState has
 * been set up: its constructor replaces them with region IDs, and the grain-to-region map that
 * links the two comes from the region-mapping file, which may no longer exist. SimulationState
 * keeps the grain IDs as a piecewise-constant grid function with one value per local element,
 * stored as doubles; they are whole numbers and are converted back to int here. On a rank
 * with no elements the array is empty.
 */
void RestartCheckpoint::WriteMeshAndGrainIds(Group* root, const SimulationState& sim_state) const {
    Group* mesh_group = root->createGroup("mesh");
    WriteParMeshBlueprint(*sim_state.m_mesh, mesh_group);

    const int num_elements = sim_state.m_mesh->GetNE();
    mesh_group->createViewString("fields/grain_id/association", "element");
    mesh_group->createViewString("fields/grain_id/topology", "mesh");
    if (num_elements == 0) {
        mesh_group->createView("fields/grain_id/values", axom::sidre::INT_ID, 0);
        return;
    }
    int* grain_ids = mesh_group
                         ->createViewAndAllocate(
                             "fields/grain_id/values", axom::sidre::INT_ID, num_elements)
                         ->getData<int*>();
    const double* grain_values = sim_state.m_grains->HostRead();
    for (int element_index = 0; element_index < num_elements; ++element_index) {
        grain_ids[element_index] = static_cast<int>(grain_values[element_index]);
    }
}

/**
 * @brief Write the `fields/` group: the nodal state and a fingerprint of its DOF layout
 *
 * @param root Root group of the rank file being written
 * @param sim_state Simulation state being written
 *
 * @details Only the state that cannot be recomputed is stored:
 * - `mesh_current`: the current coordinates (all local DOFs, i.e. VSize values).
 * - `primal_field`: the velocity true DOFs (TrueVSize values), which the solver works with.
 * - `mesh_ref`: the reference coordinates. These are stored as an owned copy rather than an
 *   external view, because phase 1 (RestoreReferenceConfiguration()) needs them inside the
 *   SimulationState constructor, before the read side has buffers to bind external views to.
 * Everything else is derived from these after the restart (ApplyPostRestartFixup()):
 * `mesh_t_beg`, the previous velocity, the velocity grid function and the displacement.
 *
 * External views read HostRead() data. In MFEM's CUDA, HIP and Umpire device memory spaces
 * HostRead() makes a synchronous device-to-host copy, so the host data is current when the
 * file is saved. (A blocking device copy only orders against the default stream, which is the
 * stream the solver uses.)
 *
 * `mesh_space/` records the sizes of the coordinate space (local, owned true and global true
 * DOFs), the mesh order and ElementLocalFieldHash() of mesh_ref. The restarted run checks
 * these before interpreting the stored coordinates (ValidateMeshSpace() and
 * VerifyRestoredDofLayout()), since the vectors are only meaningful if the rebuilt space
 * numbers its DOFs exactly as the writing run did.
 *
 * On a rank with no elements every vector here is empty; the views are only described and the
 * hash is that of an empty sequence.
 */
void RestartCheckpoint::WriteNodalFields(Group* root, const SimulationState& sim_state) const {
    Group* fields = root->createGroup("fields");
    const mfem::ParGridFunction& mesh_ref = *sim_state.m_mesh_nodes.at("mesh_ref");
    const mfem::ParGridFunction& mesh_current = *sim_state.m_mesh_nodes.at("mesh_current");
    const mfem::Vector& primal_field = *sim_state.m_primal_field;

    CreateOwnedDoubleCopy(fields,
                          "mesh_ref/values",
                          mesh_ref.Size() > 0 ? mesh_ref.HostRead() : nullptr,
                          mesh_ref.Size());
    CreateExternalDoubleView(fields,
                             "mesh_current/values",
                             mesh_current.Size() > 0 ? mesh_current.HostRead() : nullptr,
                             mesh_current.Size());
    CreateExternalDoubleView(fields,
                             "primal_field/values",
                             primal_field.Size() > 0 ? primal_field.HostRead() : nullptr,
                             primal_field.Size());

    const mfem::ParFiniteElementSpace& mesh_space = *sim_state.m_mesh_fes;
    Group* space_info = fields->createGroup("mesh_space");
    space_info->createViewScalar("order", sim_state.m_options.mesh.order);
    space_info->createViewScalar("vsize", static_cast<std::int64_t>(mesh_space.GetVSize()));
    space_info->createViewScalar("true_vsize", static_cast<std::int64_t>(mesh_space.TrueVSize()));
    space_info->createViewScalar("global_true_vsize",
                                 static_cast<std::int64_t>(mesh_space.GlobalTrueVSize()));
    space_info->createViewScalar("mesh_ref_element_hash", ElementLocalFieldHash(mesh_ref));
}

/**
 * @brief Write the `qspaces/` group: validation data for every quadrature space
 *
 * @param root Root group of the rank file being written
 * @param sim_state Simulation state being written
 *
 * @details Quadrature spaces are not restored; the restarted run rebuilds them in its
 * constructor from the mesh and the regions. What is stored lets ValidateQuadratureSpaces()
 * confirm that each rebuilt space matches the original one point for point, so a quadrature
 * function's stored values can be copied straight into the rebuilt function: the integration
 * order, the number of quadrature point values, the number of local elements, whether the
 * space covers the whole local mesh, and a hash of its local-to-global element map (which
 * elements, in which order).
 *
 * The global spaces exist on every rank, possibly empty. Region spaces only exist on ranks
 * that own elements of the region, so they are only written there; the same rank of the
 * restarted run will not have them either.
 */
void RestartCheckpoint::WriteQuadratureSpaces(Group* root, const SimulationState& sim_state) const {
    Group* qspaces = root->createGroup("qspaces");
    for (const auto& [name, space] : sim_state.m_map_qs) {
        if (!space) {
            continue;
        }
        Group* entry = qspaces->createGroup(SidreSafeName(name));
        entry->createViewScalar("order", space->GetOrder());
        entry->createViewScalar("size", static_cast<std::int64_t>(space->GetSize()));
        entry->createViewScalar("num_local_elements",
                                static_cast<std::int64_t>(space->GetNumLocalElements()));
        entry->createViewScalar("is_full_space", space->IsFullSpace() ? 1 : 0);
        entry->createViewScalar("local2global_hash", HashIntArray(space->GetLocal2Global()));
    }
}

/**
 * @brief Write the `qfuncs/` group: every registered quadrature function
 *
 * @param root Root group of the rank file being written
 * @param sim_state Simulation state being written
 *
 * @details Every non-null function in SimulationState's registry is written, except the ones
 * that are recomputed before they are next read (RecomputedQuadratureFunctionNames()). Writing
 * everything rather than a hand-picked list means functions added by future models or
 * post-processing are checkpointed without any change here. Each entry records the name of
 * the function's space, its vdim and its number of values, which the reader checks before
 * restoring, and then either the values (an external view) or `copy_of`.
 *
 * **Begin/end pairs and `copy_of`.** Material state is kept in pairs: beginning-of-step
 * (`_beg`) and end-of-step (`_end`) functions, registered in m_model_update_qf_pairs.
 * SystemDriver::UpdateModel() ends with SimulationState::SetupModelVariables(), which copies
 * every `_beg` into its `_end` partner, and the checkpoint is written after UpdateModel(), so
 * the two normally hold identical values. Instead of assuming that (post-processing runs in
 * between and its code is not part of this contract), each pair is compared bit for bit on
 * this rank:
 * - identical: the `_end` entry only records `copy_of = <_beg name>`, and the reader restores
 *   it by copying the restored `_beg` (BindFieldsAndQuadratureFunctions()). For the state
 *   variables and stresses this nearly halves the size of the checkpoint;
 * - different: the `_end` values are written like any other function.
 * Either way the restart is bitwise exact. Since the comparison is per rank, one rank's file
 * may use `copy_of` for a pair that another rank's file writes in full; each rank only reads
 * its own file.
 *
 * Empty functions (a global function on a rank with no elements) are never compared, since
 * their data pointers may be null, and are written as described zero-length views.
 */
void RestartCheckpoint::WriteQuadratureFunctions(Group* root,
                                                 const SimulationState& sim_state) const {
    // End-of-step name -> beginning-of-step name, for the pairs that are bitwise identical.
    std::map<std::string, std::string> end_copy_of;
    for (const auto& [beg_name, end_name] : sim_state.m_model_update_qf_pairs) {
        const auto beg_it = sim_state.m_map_qfs.find(beg_name);
        const auto end_it = sim_state.m_map_qfs.find(end_name);
        if (beg_it == sim_state.m_map_qfs.end() || end_it == sim_state.m_map_qfs.end() ||
            !beg_it->second || !end_it->second) {
            continue;
        }
        const mfem::expt::PartialQuadratureFunction& beg_function = *beg_it->second;
        const mfem::expt::PartialQuadratureFunction& end_function = *end_it->second;
        if (beg_function.Size() == 0 || beg_function.Size() != end_function.Size() ||
            beg_function.GetVDim() != end_function.GetVDim() ||
            beg_function.GetPartialSpaceShared() != end_function.GetPartialSpaceShared()) {
            continue;
        }
        const std::size_t byte_count = static_cast<std::size_t>(beg_function.Size()) *
                                       sizeof(double);
        if (std::memcmp(beg_function.HostRead(), end_function.HostRead(), byte_count) == 0) {
            end_copy_of[end_name] = beg_name;
        }
    }

    const std::set<std::string> recomputed = RecomputedQuadratureFunctionNames(sim_state);
    Group* qfuncs = root->createGroup("qfuncs");
    for (const auto& [name, function] : sim_state.m_map_qfs) {
        if (!function || recomputed.count(name) != 0) {
            continue;
        }
        Group* entry = qfuncs->createGroup(SidreSafeName(name));
        entry->createViewString(
            "qspace_name",
            FindQuadratureSpaceName(sim_state.m_map_qs, function->GetPartialSpaceShared().get()));
        entry->createViewScalar("vdim", function->GetVDim());
        entry->createViewScalar("value_count", static_cast<std::int64_t>(function->Size()));
        const auto copy_it = end_copy_of.find(name);
        if (copy_it != end_copy_of.end()) {
            entry->createViewString("copy_of", copy_it->second);
        } else {
            CreateExternalDoubleView(entry,
                                     "values",
                                     function->Size() > 0 ? function->HostRead() : nullptr,
                                     function->Size());
        }
    }
}

/**
 * @brief Write the `materials/` group: the identity and properties of every region
 *
 * @param root Root group of the rank file being written
 * @param sim_state Simulation state being written
 *
 * @details For each region: the material name, the model type and the effective number of
 * state variables (including the orientation components SimulationState adds), and the
 * material properties. ValidateMaterials() aborts the restart if the identity changed, since
 * the stored state variables would then be meaningless, but only warns if the properties
 * changed, since editing properties for a continuation is a legitimate use.
 *
 * Every rank holds the material list and properties for every region, whether or not it owns
 * elements of that region, so every rank writes the same group.
 */
void RestartCheckpoint::WriteMaterials(Group* root, const SimulationState& sim_state) const {
    Group* materials = root->createGroup("materials");
    const int num_regions = static_cast<int>(sim_state.m_material_name_region.size());
    materials->createViewScalar("num_regions", num_regions);
    for (int region = 0; region < num_regions; ++region) {
        const auto region_index = static_cast<std::size_t>(region);
        Group* entry = materials->createGroup("region_" + std::to_string(region));
        entry->createViewString("name", sim_state.m_material_name_region[region_index].first);
        entry->createViewScalar("mech_type",
                                static_cast<int>(sim_state.m_region_material_type[region_index]));
        entry->createViewScalar("num_state_vars",
                                sim_state.m_options.materials[region_index].state_vars.num_vars);
        const std::vector<double>& properties = sim_state.GetMaterialProperties(region);
        CreateOwnedDoubleCopy(
            entry, "properties", properties.data(), static_cast<IndexType>(properties.size()));
    }
}

/**
 * @brief Write the commit file that marks a checkpoint as complete
 *
 * @param checkpoint_directory Directory of the checkpoint
 * @param completed_cycles Number of completed cycles the checkpoint holds
 * @param driver_step Driver loop counter of the checkpoint
 * @param completed_time End time of the last completed step
 * @return True if the commit file is in place
 *
 * @details Called on rank 0 only, after every rank saved its file. The file is written under
 * a temporary name and then renamed. A rename within one file system is atomic on POSIX, so a
 * reader either sees no commit file or a complete one, never a partial one. Besides marking
 * the checkpoint complete, the commit file carries the format version and rank count that
 * OpenForRestart() checks before any rank opens its own file. It uses the JSON protocol so it
 * can be read by eye.
 */
bool RestartCheckpoint::WriteCommitFile(const fs::path& checkpoint_directory,
                                        const std::size_t completed_cycles,
                                        const std::size_t driver_step,
                                        const double completed_time) const {
    DataStore commit_store;
    Group* commit = commit_store.getRoot();
    commit->createViewScalar("format_version", RESTART_FORMAT_VERSION);
    commit->createViewScalar("num_ranks", m_num_ranks);
    commit->createViewScalar("completed_cycles", static_cast<std::int64_t>(completed_cycles));
    commit->createViewScalar("driver_step", static_cast<std::int64_t>(driver_step));
    commit->createViewScalar("completed_time", completed_time);
    commit->createViewString("rank_file_protocol", RANK_FILE_PROTOCOL);

    const fs::path temporary = checkpoint_directory / (std::string(COMMIT_FILE_NAME) + ".tmp");
    if (!commit->save(temporary.string(), COMMIT_PROTOCOL)) {
        return false;
    }
    std::error_code error;
    fs::rename(temporary, checkpoint_directory / COMMIT_FILE_NAME, error);
    return !error;
}

/**
 * @brief Rename checkpoints newer than this run's starting point out of the way
 *
 * @param completed_cycles This run's starting point: the restart cycle, or 0 for a fresh run
 *
 * @details Called on rank 0 only, on this run's first write. Checkpoints in the write
 * directory that are newer than the starting point belong to a different timeline: a fresh
 * run reusing an old directory, or a restart from an older checkpoint. If they stayed, a
 * later restart that picks "the newest committed checkpoint" could pick one of them instead
 * of this run's. They are renamed to `<name>.superseded` (or `.superseded.N`), never deleted,
 * so nothing is lost if this was a mistake; the renamed directories no longer match
 * ParseCycleDirectoryName() and are ignored. A rename failure is reported and otherwise
 * ignored, since it only affects which checkpoint a later automatic selection would pick.
 */
void RestartCheckpoint::SupersedeCheckpointsAfter(const std::size_t completed_cycles) const {
    for (const CheckpointDirectoryEntry& entry : ListCheckpointDirectories(m_options.directory)) {
        if (entry.completed_cycles <= completed_cycles) {
            continue;
        }
        fs::path target = entry.path;
        target += ".superseded";
        for (int suffix = 1; fs::exists(target); ++suffix) {
            target = entry.path;
            target += ".superseded." + std::to_string(suffix);
        }
        std::error_code error;
        fs::rename(entry.path, target, error);
        if (error) {
            MFEM_WARNING_0("Unable to rename the superseded checkpoint " << entry.path << ": "
                                                                         << error.message());
        } else {
            std::cout << "Renamed superseded restart checkpoint " << entry.path.string() << " to "
                      << target.string() << std::endl;
        }
    }
}

/**
 * @brief Remove checkpoints beyond the configured retention
 *
 * @details Called on rank 0 only, right after a successful commit, so the newest checkpoint
 * is always complete before anything older is removed. Two kinds of directories are removed
 * from the write directory:
 * - the oldest committed checkpoints beyond the newest `keep_last` (keep_last = 0 keeps all);
 * - incomplete (uncommitted) checkpoints older than the newest committed one, which are left
 *   over from failed or interrupted writes and can never be restarted from.
 * An incomplete checkpoint newer than the newest committed one is left alone, since it may be
 * a write still in progress by another run. A removal failure is reported and otherwise
 * ignored; it only costs disk space.
 */
void RestartCheckpoint::ApplyRetentionPolicy() const {
    const std::vector<CheckpointDirectoryEntry> entries = ListCheckpointDirectories(
        m_options.directory);
    std::size_t newest_committed = 0;
    bool any_committed = false;
    std::vector<const CheckpointDirectoryEntry*> committed;
    for (const CheckpointDirectoryEntry& entry : entries) {
        if (entry.committed) {
            committed.push_back(&entry);
            newest_committed = entry.completed_cycles;
            any_committed = true;
        }
    }

    std::vector<fs::path> to_remove;
    if (m_options.keep_last > 0 &&
        committed.size() > static_cast<std::size_t>(m_options.keep_last)) {
        // entries is sorted oldest first, so the first remove_count committed ones go.
        const std::size_t remove_count = committed.size() -
                                         static_cast<std::size_t>(m_options.keep_last);
        for (std::size_t i = 0; i < remove_count; ++i) {
            to_remove.push_back(committed[i]->path);
        }
    }
    for (const CheckpointDirectoryEntry& entry : entries) {
        if (!entry.committed && any_committed && entry.completed_cycles < newest_committed) {
            to_remove.push_back(entry.path);
        }
    }

    for (const fs::path& path : to_remove) {
        std::error_code error;
        fs::remove_all(path, error);
        if (error) {
            MFEM_WARNING_0("Unable to remove the old checkpoint " << path << ": "
                                                                  << error.message());
        }
    }
}

// =========================================================================
// RESTARTING, PHASE 1 (inside the SimulationState constructor)
// =========================================================================

/**
 * @brief Pick the checkpoint directory to restart from and check its commit file
 *
 * @return Path of the selected checkpoint directory, the same on every rank
 *
 * @details Collective. Only rank 0 looks at the file system, so every rank is guaranteed to
 * open the same checkpoint even if directories appear or disappear while the run starts.
 * Rank 0 selects `[Restart] read_cycle` in the read directory if it is set, otherwise the
 * newest committed checkpoint there, and reads its commit file. Any problem (requested
 * checkpoint missing or incomplete, nothing committed, unreadable commit file) is broadcast as
 * a message so every rank aborts with it.
 *
 * The format version and rank count from the commit file are then checked on every rank.
 * Checking them from the commit file, before any rank opens its own file, gives a clear
 * message for the common mistake of restarting with a different number of ranks, instead of
 * some ranks finding no file at all. The checkpoint layout is partition dependent (each rank
 * holds its own piece of the mesh and state), so the rank count has to match exactly.
 */
std::string RestartCheckpoint::SelectCheckpointDirectory() {
    std::string selected_directory;
    std::string selection_error;
    int commit_format_version = -1;
    int commit_num_ranks = -1;
    if (m_rank == 0) {
        const fs::path& read_directory = m_options.GetReadDirectory();
        fs::path selected;
        if (m_options.read_cycle.has_value()) {
            selected = read_directory / CycleDirectoryName(*m_options.read_cycle);
            if (!fs::exists(selected / COMMIT_FILE_NAME)) {
                selection_error = "the requested checkpoint " + selected.string() +
                                  " does not exist or is incomplete (no " + COMMIT_FILE_NAME + ")";
            }
        } else {
            const auto entries = ListCheckpointDirectories(read_directory);
            for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
                if (it->committed) {
                    selected = it->path;
                    break;
                }
            }
            if (selected.empty()) {
                selection_error = "no complete checkpoint was found in " + read_directory.string();
            }
        }
        if (selection_error.empty()) {
            DataStore commit_store;
            Group* commit = commit_store.getRoot();
            if (commit->load((selected / COMMIT_FILE_NAME).string(), COMMIT_PROTOCOL) &&
                commit->hasView("format_version") && commit->hasView("num_ranks")) {
                commit_format_version = ReadSidreScalar<int>(commit, "format_version");
                commit_num_ranks = ReadSidreScalar<int>(commit, "num_ranks");
                selected_directory = selected.string();
            } else {
                selection_error = "unable to read " + (selected / COMMIT_FILE_NAME).string();
            }
        }
    }

    BroadcastString(selection_error, m_comm);
    MFEM_VERIFY_0(selection_error.empty(), "Unable to restart: " << selection_error);
    BroadcastString(selected_directory, m_comm);
    MPI_Bcast(&commit_format_version, 1, MPI_INT, 0, m_comm);
    MPI_Bcast(&commit_num_ranks, 1, MPI_INT, 0, m_comm);
    MFEM_VERIFY_0(commit_format_version == RESTART_FORMAT_VERSION,
                  "Unable to restart: the checkpoint "
                      << selected_directory << " has format version " << commit_format_version
                      << ", but this build reads version " << RESTART_FORMAT_VERSION);
    MFEM_VERIFY_0(commit_num_ranks == m_num_ranks,
                  "Unable to restart: the checkpoint "
                      << selected_directory << " was written with " << commit_num_ranks
                      << " MPI ranks, but this run has " << m_num_ranks
                      << "; a restart must use the same number of ranks");
    return selected_directory;
}

/**
 * @details Implementation notes.
 * 1. SelectCheckpointDirectory() picks the checkpoint on rank 0 and checks its commit file on
 *    every rank.
 * 2. Every rank loads its own rank file into m_read_store, which stays alive until the end of
 *    RestoreState(). Group::load() materializes the owned arrays (the mesh, mesh_ref, grain
 *    IDs and metadata) and restores the external views (the bulk state) as descriptions only;
 *    their data is read later, straight into SimulationState's buffers.
 * 3. Each rank checks that its file's restart_info matches: format version, rank count and,
 *    importantly, its own rank, which catches files renamed or copied between ranks. The
 *    stored time state and driver step are kept for phase 2.
 * 4. m_last_written_cycles is set to the restart point, so the interval for this run's first
 *    checkpoint is counted from the restart, and its first write supersedes any checkpoints
 *    newer than the restart point (SupersedeCheckpointsAfter()).
 *
 * A problem with one rank's file is recorded and every rank aborts together at the end of the
 * stage (AbortIfAnyRankFailed()).
 */
void RestartCheckpoint::OpenForRestart() {
    CALI_CXX_MARK_SCOPE("exaconstit::RestartCheckpoint::OpenForRestart");

    const std::string selected_directory = SelectCheckpointDirectory();

    m_read_rank_file = fs::path(selected_directory) / RankFileName(m_rank);
    m_read_store = std::make_unique<DataStore>();
    Group* root = m_read_store->getRoot();
    if (!fs::exists(m_read_rank_file) ||
        !root->load(m_read_rank_file.string(), RANK_FILE_PROTOCOL)) {
        RecordFailure("unable to load " + m_read_rank_file.string());
    } else if (!root->hasGroup("restart_info") || !root->hasGroup("mesh")) {
        RecordFailure(m_read_rank_file.string() + " has no restart_info/ or mesh/ group");
    } else {
        Group* info = root->getGroup("restart_info");
        const int stored_version = ReadSidreScalar<int>(info, "format_version");
        const int stored_rank = ReadSidreScalar<int>(info, "rank");
        const int stored_num_ranks = ReadSidreScalar<int>(info, "num_ranks");
        if (stored_version != RESTART_FORMAT_VERSION || stored_rank != m_rank ||
            stored_num_ranks != m_num_ranks) {
            RecordFailure(m_read_rank_file.string() + " was written by rank " +
                          std::to_string(stored_rank) + " of " + std::to_string(stored_num_ranks) +
                          " with format version " + std::to_string(stored_version) +
                          ", which does not match this rank");
        } else {
            m_restart_driver_step = static_cast<std::size_t>(
                ReadSidreScalar<std::int64_t>(info, "driver_step"));
            m_restart_time_state.completed_cycles = static_cast<std::size_t>(
                ReadSidreScalar<std::int64_t>(info, "completed_cycles"));
            m_restart_time_state.completed_time = ReadSidreScalar<double>(info, "completed_time");
            m_restart_time_state.next_dt = ReadSidreScalar<double>(info, "next_dt");
            m_restart_time_state.previous_dt = ReadSidreScalar<double>(info, "previous_dt");
            m_restart_time_state.original_dt = ReadSidreScalar<double>(info, "original_dt");
        }
    }
    AbortIfAnyRankFailed("opening the checkpoint");

    m_last_written_cycles = m_restart_time_state.completed_cycles;

    if (m_rank == 0) {
        std::cout << "Restarting from checkpoint " << selected_directory << " (cycle "
                  << m_restart_time_state.completed_cycles << ", driver step "
                  << m_restart_driver_step << ", t = " << m_restart_time_state.completed_time << ")"
                  << std::endl;
    }
}

/**
 * @details Implementation notes. ExaSidreParMesh rebuilds the mesh and checks the stored
 * arrays itself (collectively). This function adds a final check that the edges, faces and
 * communication groups MFEM regenerated from the connectivity match the counts recorded at
 * write time. The mesh is returned as a plain mfem::ParMesh, so SimulationState owns it like
 * any other mesh.
 */
std::shared_ptr<mfem::ParMesh> RestartCheckpoint::ReconstructMesh() {
    CALI_CXX_MARK_SCOPE("exaconstit::RestartCheckpoint::ReconstructMesh");
    MFEM_VERIFY_0(m_read_store, "ReconstructMesh() requires OpenForRestart() first");
    Group* mesh_group = m_read_store->getRoot()->getGroup("mesh");

    std::shared_ptr<ExaSidreParMesh> mesh = ExaSidreParMesh::Create(mesh_group, m_comm);

    const MeshTopologyCounts stored = ReadStoredMeshTopologyCounts(mesh_group);
    const MeshTopologyCounts rebuilt = CountMeshTopology(*mesh);
    if (!stored.Matches(rebuilt)) {
        std::ostringstream message;
        message << "the rebuilt mesh topology differs from the checkpoint (vertices, elements, "
                   "boundary elements, edges, faces, communication groups: stored "
                << stored.num_vertices << ", " << stored.num_elements << ", "
                << stored.num_boundary_elements << ", " << stored.num_edges << ", "
                << stored.num_faces << ", " << stored.num_communication_groups << "; rebuilt "
                << rebuilt.num_vertices << ", " << rebuilt.num_elements << ", "
                << rebuilt.num_boundary_elements << ", " << rebuilt.num_edges << ", "
                << rebuilt.num_faces << ", " << rebuilt.num_communication_groups << ")";
        RecordFailure(message.str());
    }
    AbortIfAnyRankFailed("rebuilding the mesh");
    return mesh;
}

/**
 * @details Implementation notes. The grain IDs were written by WriteMeshAndGrainIds() as one
 * int per local element, in local element order. The rebuilt mesh has the same local element
 * order, so they are copied straight across. The count is checked against the local element
 * count, which also covers ranks with no elements (both zero, nothing copied).
 */
void RestartCheckpoint::ReadElementGrainIds(mfem::Array<int>& grain_ids) {
    MFEM_VERIFY_0(m_read_store, "ReadElementGrainIds() requires OpenForRestart() first");
    Group* mesh_group = m_read_store->getRoot()->getGroup("mesh");
    const std::string path = "fields/grain_id/values";
    if (!mesh_group->hasView(path)) {
        RecordFailure("the checkpoint has no mesh/" + path);
    } else {
        View* view = mesh_group->getView(path);
        if (view->getNumElements() != grain_ids.Size()) {
            RecordFailure("the checkpoint has " + std::to_string(view->getNumElements()) +
                          " grain IDs for " + std::to_string(grain_ids.Size()) + " local elements");
        } else if (grain_ids.Size() > 0) {
            const int* stored = view->getData<int*>();
            std::copy(stored, stored + grain_ids.Size(), grain_ids.HostWrite());
        }
    }
    AbortIfAnyRankFailed("reading the grain IDs");
}

/**
 * @details Implementation notes.
 *
 * **Why the coordinates are restored during construction.** "Build first, overlay last"
 * would normally leave every field for phase 2. The coordinates are the exception, because
 * some constructors compute and keep values derived from the starting geometry:
 * - SystemDriver's constructor, with mono_def_bcs, selects essential DOFs from the mesh nodes
 *   using tight min/max coordinate comparisons (1e-12).
 * - AbaqusUmatModel::InitLocSFGrads() stores reference shape-function gradients that every
 *   later step reuses.
 * In a fresh run these see the reference coordinates projected from the original mesh file's
 * Nodes. The rebuilt mesh has no Nodes, so the SimulationState constructor's
 * Mesh::GetNodes(GridFunction&) projects from the vertex positions instead, which agrees with
 * the original only to rounding (a few entries differ by about 1e-16). Small as that is, it
 * changes those stored values and so the restarted run would no longer be bitwise identical.
 * Overwriting the coordinate fields with the stored mesh_ref bits right after they are created
 * gives every constructor exactly the geometry it saw in the original run. This is safe: in
 * the original run mesh_ref, mesh_t_beg and mesh_current all held exactly these values until
 * the first step (the constructor sets them equal and SwapNodes() makes mesh_current the
 * mesh's Nodes; the driver only zeroes the velocity and displacement).
 *
 * **Steps.**
 * 1. ValidateMeshSpace(): the coordinate space must have the same order and sizes as in the
 *    writing run, or the stored vector cannot be interpreted at all.
 * 2. Check that mesh_ref was stored as an owned double array of the space's local size. It
 *    has to be owned: an external view would have no data until loadExternalData(), which
 *    phase 2 does.
 * 3. Copy it into mesh_ref, mesh_t_beg and mesh_current, then tell the mesh its Nodes changed
 *    (mesh_current is the mesh's Nodes) so cached geometric factors are dropped.
 * 4. VerifyRestoredDofLayout(): confirm, on the overlaid coordinates, that the rebuilt space
 *    numbers its DOFs exactly as the writing run did, before anything is built on it.
 * 5. Drop the stored mesh_ref; it has been copied.
 *
 * On a rank with no elements the space and every vector are empty, nothing is copied, and the
 * rank still takes part in the collective checks.
 */
void RestartCheckpoint::RestoreReferenceConfiguration(SimulationState& sim_state) {
    CALI_CXX_MARK_SCOPE("exaconstit::RestartCheckpoint::RestoreReferenceConfiguration");
    MFEM_VERIFY_0(m_read_store, "RestoreReferenceConfiguration() requires OpenForRestart() first");
    Group* root = m_read_store->getRoot();

    // Step 1: the coordinate space must match before the stored vector means anything.
    ValidateMeshSpace(root, sim_state);

    // Step 2: locate the owned copy of the reference coordinates.
    mfem::ParGridFunction& mesh_ref = *sim_state.m_mesh_nodes.at("mesh_ref");
    const std::string path = "fields/mesh_ref/values";
    const double* stored_reference = nullptr;
    if (!root->hasView(path)) {
        RecordFailure("the checkpoint has no " + path);
    } else {
        View* view = root->getView(path);
        if (view->isExternal() || view->getNumElements() != mesh_ref.Size() ||
            (mesh_ref.Size() > 0 && view->getTypeID() != axom::sidre::FLOAT64_ID)) {
            RecordFailure(path + " is not an owned double array with one entry per local DOF "
                                 "of the coordinate space");
        } else if (mesh_ref.Size() > 0) {
            stored_reference = view->getData<double*>();
        }
    }
    AbortIfAnyRankFailed("restoring the reference configuration");

    // Step 3: overwrite all three coordinate fields with the original reference bits.
    for (const char* name : {"mesh_ref", "mesh_t_beg", "mesh_current"}) {
        mfem::ParGridFunction& field = *sim_state.m_mesh_nodes.at(name);
        if (field.Size() > 0) {
            std::copy(stored_reference, stored_reference + field.Size(), field.HostWrite());
        }
    }
    sim_state.m_mesh->NodesUpdated();

    // Step 4: verify the DOF layout before anything is built on these coordinates.
    VerifyRestoredDofLayout(sim_state);

    // Step 5: the stored copy is no longer needed.
    root->destroyGroupAndData("fields/mesh_ref");
}

/**
 * @details Implementation notes. Called at the end of the SimulationState constructor, once
 * the mesh has been rebuilt and the grain IDs copied out. The mesh arrays are often the
 * largest owned data in the file, so freeing them early keeps the peak memory of the restart
 * down; the rest of the store (the bulk state descriptions) is needed until phase 2.
 */
void RestartCheckpoint::ReleaseMeshArrays() {
    if (m_read_store && m_read_store->getRoot()->hasGroup("mesh")) {
        m_read_store->getRoot()->destroyGroupAndData("mesh");
    }
}

/**
 * @brief Check that the coordinate space matches the one the checkpoint was written with
 *
 * @param root Root group of the loaded rank file
 * @param sim_state Simulation state under construction
 *
 * @details Compares the mesh order and the coordinate space's local, true and global true DOF
 * counts with the stored values, and keeps the stored DOF-layout hash for
 * VerifyRestoredDofLayout(). A different order (or, equivalently, different sizes) means the
 * stored coordinate and velocity vectors cannot be mapped onto this space at all, so this is
 * recorded as a failure; the caller aborts at the end of its stage.
 */
void RestartCheckpoint::ValidateMeshSpace(Group* root, const SimulationState& sim_state) {
    if (!root->hasGroup("fields/mesh_space")) {
        RecordFailure("the checkpoint has no fields/mesh_space group");
        return;
    }
    Group* space_info = root->getGroup("fields/mesh_space");
    const mfem::ParFiniteElementSpace& mesh_space = *sim_state.m_mesh_fes;
    const int stored_order = ReadSidreScalar<int>(space_info, "order");
    const auto stored_vsize = ReadSidreScalar<std::int64_t>(space_info, "vsize");
    const auto stored_true_vsize = ReadSidreScalar<std::int64_t>(space_info, "true_vsize");
    const auto stored_global_true_vsize = ReadSidreScalar<std::int64_t>(space_info,
                                                                        "global_true_vsize");
    m_stored_mesh_ref_element_hash = ReadSidreScalar<std::uint64_t>(space_info,
                                                                    "mesh_ref_element_hash");
    if (stored_order != sim_state.m_options.mesh.order) {
        RecordFailure("the mesh order changed (checkpoint " + std::to_string(stored_order) +
                      ", this run " + std::to_string(sim_state.m_options.mesh.order) +
                      "); the mesh order cannot change on a restart");
    }
    if (stored_vsize != mesh_space.GetVSize() || stored_true_vsize != mesh_space.TrueVSize() ||
        stored_global_true_vsize != static_cast<std::int64_t>(mesh_space.GlobalTrueVSize())) {
        RecordFailure(
            "the coordinate space sizes differ (local, true, global true DOFs: checkpoint " +
            std::to_string(stored_vsize) + ", " + std::to_string(stored_true_vsize) + ", " +
            std::to_string(stored_global_true_vsize) + "; this run " +
            std::to_string(mesh_space.GetVSize()) + ", " + std::to_string(mesh_space.TrueVSize()) +
            ", " + std::to_string(static_cast<std::int64_t>(mesh_space.GlobalTrueVSize())) + ")");
    }
}

/**
 * @brief Verify that the rebuilt coordinate space numbers its DOFs as in the writing run
 *
 * @param sim_state Simulation state whose mesh_ref has just been overwritten
 *
 * @details Collective. Matching sizes are not enough: the stored vectors are only correct if
 * every value lands on the same DOF it came from. Two independent checks, each run on the
 * overlaid reference coordinates (a continuous field with distinct values at distinct nodes):
 *
 * (a) Local DOF numbering. ElementLocalFieldHash() of mesh_ref must equal the hash recorded
 *     at write time. It visits the values element by element in each element's DOF order, so
 *     any difference in how this rank numbers its local DOFs changes it.
 *
 * (b) Shared DOF matching across ranks. SharedDofInconsistency() restricts mesh_ref to true
 *     DOFs and prolongates it back. If every shared DOF is matched with the same true DOF on
 *     every rank, the field comes back unchanged up to rounding; a mismatch, such as a wrong
 *     shared-face orientation at H1 order >= 3, moves another node's coordinate onto the DOF,
 *     a jump of the order of the element size. The tolerance is 1e-10 times the largest
 *     coordinate magnitude in the mesh (agreed on with an MPI_Allreduce so every rank uses the
 *     same value), far above rounding and far below any real mismatch. A mesh whose
 *     coordinates are all zero (only possible on a degenerate mesh) skips the check.
 *
 * Every rank computes both checks, including ranks with no DOFs, since (b) and the scale
 * reduction are collective. Failures abort every rank.
 */
void RestartCheckpoint::VerifyRestoredDofLayout(const SimulationState& sim_state) {
    CALI_CXX_MARK_SCOPE("exaconstit::RestartCheckpoint::VerifyRestoredDofLayout");
    const mfem::ParGridFunction& mesh_ref = *sim_state.m_mesh_nodes.at("mesh_ref");

    // (a) Local DOF numbering.
    if (ElementLocalFieldHash(mesh_ref) != m_stored_mesh_ref_element_hash) {
        RecordFailure("the local DOF numbering of the rebuilt mesh differs from the checkpoint "
                      "(the element-local hash of mesh_ref does not match)");
    }

    // (b) Shared DOF matching across ranks.
    double local_scale = 0.0;
    if (mesh_ref.Size() > 0) {
        const double* reference = mesh_ref.HostRead();
        for (int i = 0; i < mesh_ref.Size(); ++i) {
            local_scale = std::max(local_scale, std::abs(reference[i]));
        }
    }
    double global_scale = 0.0;
    MPI_Allreduce(&local_scale, &global_scale, 1, MPI_DOUBLE, MPI_MAX, m_comm);
    const double inconsistency = SharedDofInconsistency(mesh_ref);
    const double tolerance = 1.0e-10 * global_scale;
    if (global_scale > 0.0 && inconsistency > tolerance) {
        std::ostringstream message;
        message << "shared DOFs are matched inconsistently across ranks (max |x - P R x| = "
                << inconsistency << ", tolerance " << tolerance << ")";
        RecordFailure(message.str());
    }
    AbortIfAnyRankFailed("verifying the restored DOF layout");
}

// =========================================================================
// RESTARTING, PHASE 2 (from the driver)
// =========================================================================

/**
 * @details Implementation notes. By the time this runs, SimulationState, SystemDriver and
 * PostProcessingDriver have all been constructed exactly as in a fresh run, so every buffer
 * the checkpoint can fill exists. The steps are ordered so that nothing is overwritten until
 * everything has been checked:
 *
 * 1. Validate (ValidateMaterials(), ValidateQuadratureSpaces()). A failure here aborts before
 *    any buffer is touched, so SimulationState is never left half restored. The coordinate
 *    space and DOF layout were already validated in phase 1.
 * 2. Bind (BindFieldsAndQuadratureFunctions()). Every external view in the checkpoint that
 *    has a matching SimulationState buffer is pointed at that buffer's host memory. Functions
 *    stored as `copy_of` are queued in m_pending_copies instead.
 * 3. Drop unbound external views. Group::loadExternalData() reads every external view
 *    described in the group whether or not it was bound, and an unbound view has nowhere to
 *    put its data. The unbound ones are zero-length arrays and functions that are only in the
 *    checkpoint (with a warning from step 2); they are destroyed first.
 * 4. Load. A single loadExternalData() call per rank reads all bound arrays straight into
 *    SimulationState's buffers. HostReadWrite() during binding made the host copy the valid
 *    one, so on a GPU run the next device access copies the restored values over. The queued
 *    `copy_of` functions are then filled from their restored partners.
 * 5. Resume time (TimeManagement::ResumeFromCheckpoint()) and recompute the fields derived
 *    from the restored state (ApplyPostRestartFixup()).
 * 6. Release the checkpoint. From here on nothing restored is owned by Sidre.
 *
 * Collective, including on ranks with no elements: they bind nothing (every array of theirs
 * is empty) but still take part in every agreement step.
 */
void RestartCheckpoint::RestoreState(SimulationState& sim_state) {
    CALI_CXX_MARK_SCOPE("exaconstit::RestartCheckpoint::RestoreState");
    MFEM_VERIFY_0(m_read_store, "RestoreState() requires OpenForRestart() first");
    const double start_seconds = MPI_Wtime();
    Group* root = m_read_store->getRoot();

    // Step 1: validate everything before anything is overwritten.
    ValidateMaterials(root, sim_state);
    ValidateQuadratureSpaces(root, sim_state);
    AbortIfAnyRankFailed("checking the checkpoint against this run");

    // Step 2: point each stored array at the SimulationState buffer it restores.
    BindFieldsAndQuadratureFunctions(root, sim_state);
    AbortIfAnyRankFailed("matching the checkpoint arrays to this run");

    // Step 3: loadExternalData() would try to read unbound views too, so drop them.
    std::vector<std::pair<Group*, std::string>> unbound;
    CollectUnboundExternalViews(root, unbound);
    for (auto& [group, view_name] : unbound) {
        group->destroyView(view_name);
    }

    // Step 4: one bulk read into the bound buffers, then fill the copy_of functions.
    if (!root->loadExternalData(m_read_rank_file.string())) {
        RecordFailure("unable to read the state arrays from " + m_read_rank_file.string());
    }
    AbortIfAnyRankFailed("reading the checkpoint arrays");
    for (auto& [destination, source] : m_pending_copies) {
        *destination = static_cast<const mfem::Vector&>(*source);
    }
    m_pending_copies.clear();

    // Step 5: time, then the fields derived from the restored state.
    sim_state.m_time_manager.ResumeFromCheckpoint(m_restart_time_state);
    ApplyPostRestartFixup(sim_state);

    // Step 6: nothing restored is owned by Sidre from here on.
    m_read_store.reset();

    const TimeManagement& time_manager = sim_state.m_time_manager;
    if (m_rank == 0) {
        std::cout << "Restored checkpoint state in " << (MPI_Wtime() - start_seconds)
                  << " s; next step: cycle " << (m_restart_time_state.completed_cycles + 1)
                  << ", driver step " << (m_restart_driver_step + 1)
                  << ", t_end = " << time_manager.GetTime()
                  << ", dt = " << time_manager.GetDeltaTime()
                  << ", t_final = " << time_manager.GetFinalTime() << std::endl;
    }
    if (time_manager.IsFinished()) {
        MFEM_WARNING_0("The checkpoint is already at the configured final time, so there is "
                       "nothing left to run; increase t_final (or extend the custom time steps) "
                       "to continue the simulation");
    }
}

/**
 * @brief Check that the material regions match the ones the checkpoint was written with
 *
 * @param root Root group of the loaded rank file
 * @param sim_state Fully constructed simulation state
 *
 * @details The stored state variables are only meaningful for the same material model with
 * the same state-variable layout, so a different number of regions, or a different name,
 * model type or effective state-variable count for any region, is recorded as a failure.
 * Changed material properties only produce a warning: editing properties for a continuation
 * is legitimate, and this run's values are the ones used (the models read them from
 * SimulationState when they are constructed). The warning is issued on every rank through
 * MFEM_WARNING_0, which prints it once; every rank holds the same material data, so every
 * rank reaches the same conclusion.
 */
void RestartCheckpoint::ValidateMaterials(Group* root, const SimulationState& sim_state) {
    if (!root->hasGroup("materials")) {
        RecordFailure("the checkpoint has no materials/ group");
        return;
    }
    Group* materials = root->getGroup("materials");
    const int stored_regions = ReadSidreScalar<int>(materials, "num_regions");
    const int runtime_regions = static_cast<int>(sim_state.m_material_name_region.size());
    if (stored_regions != runtime_regions) {
        RecordFailure("the number of material regions changed (checkpoint " +
                      std::to_string(stored_regions) + ", this run " +
                      std::to_string(runtime_regions) + ")");
        return;
    }
    for (int region = 0; region < runtime_regions; ++region) {
        const auto region_index = static_cast<std::size_t>(region);
        Group* entry = materials->getGroup("region_" + std::to_string(region));
        const std::string stored_name = entry->getView("name")->getString();
        const int stored_type = ReadSidreScalar<int>(entry, "mech_type");
        const int stored_state_vars = ReadSidreScalar<int>(entry, "num_state_vars");
        const std::string& runtime_name = sim_state.m_material_name_region[region_index].first;
        const int runtime_type = static_cast<int>(sim_state.m_region_material_type[region_index]);
        const int runtime_state_vars =
            sim_state.m_options.materials[region_index].state_vars.num_vars;
        if (stored_name != runtime_name || stored_type != runtime_type ||
            stored_state_vars != runtime_state_vars) {
            RecordFailure("region " + std::to_string(region + 1) +
                          " changed its material name, model type or number of state variables "
                          "(checkpoint '" +
                          stored_name + "', type " + std::to_string(stored_type) + ", " +
                          std::to_string(stored_state_vars) + " state variables; this run '" +
                          runtime_name + "', type " + std::to_string(runtime_type) + ", " +
                          std::to_string(runtime_state_vars) + " state variables)");
            continue;
        }

        const std::vector<double>& runtime_properties = sim_state.GetMaterialProperties(region);
        View* stored_view = entry->getView("properties");
        const IndexType stored_count = stored_view->getNumElements();
        bool same = stored_count == static_cast<IndexType>(runtime_properties.size());
        if (same && stored_count > 0) {
            const double* stored = stored_view->getData<double*>();
            same = std::equal(runtime_properties.begin(), runtime_properties.end(), stored);
        }
        if (!same) {
            MFEM_WARNING_0("The material properties of region "
                           << (region + 1) << " ('" << runtime_name
                           << "') differ from the checkpoint; the restart continues with this "
                              "run's properties");
        }
    }
}

/**
 * @brief Check that the rebuilt quadrature spaces match the ones the checkpoint was written
 *        with
 *
 * @param root Root group of the loaded rank file
 * @param sim_state Fully constructed simulation state
 *
 * @details Every quadrature space present both in this run and in the checkpoint must have the
 * same integration order, number of values, number of local elements, full/partial status and
 * local-to-global element map (by hash). Only then can a quadrature function's stored values
 * be copied straight into the rebuilt function, point for point. A space present on only one
 * side is not an error here: the functions on it are handled (and reported) during binding.
 *
 * On a rank with no elements only the (empty) global spaces exist and they match trivially;
 * region spaces exist on neither side.
 */
void RestartCheckpoint::ValidateQuadratureSpaces(Group* root, const SimulationState& sim_state) {
    if (!root->hasGroup("qspaces")) {
        RecordFailure("the checkpoint has no qspaces/ group");
        return;
    }
    Group* qspaces = root->getGroup("qspaces");
    for (const auto& [name, space] : sim_state.m_map_qs) {
        if (!space || !qspaces->hasGroup(SidreSafeName(name))) {
            continue;
        }
        Group* entry = qspaces->getGroup(SidreSafeName(name));
        const bool same = ReadSidreScalar<int>(entry, "order") == space->GetOrder() &&
                          ReadSidreScalar<std::int64_t>(entry, "size") == space->GetSize() &&
                          ReadSidreScalar<std::int64_t>(entry, "num_local_elements") ==
                              space->GetNumLocalElements() &&
                          ReadSidreScalar<int>(entry, "is_full_space") ==
                              (space->IsFullSpace() ? 1 : 0) &&
                          ReadSidreScalar<std::uint64_t>(entry, "local2global_hash") ==
                              HashIntArray(space->GetLocal2Global());
        if (!same) {
            RecordFailure("the quadrature space '" + name +
                          "' differs from the checkpoint (integration order, size, element "
                          "subset or element order)");
        }
    }
}

/**
 * @brief Point every stored state array at the SimulationState buffer it restores
 *
 * @param root Root group of the loaded rank file
 * @param sim_state Fully constructed simulation state
 *
 * @details Nothing is read yet; this only sets data pointers (View::setExternalDataPtr()) so
 * that RestoreState()'s single loadExternalData() call reads every array straight into place.
 *
 * **Nodal fields.** `mesh_current` and `primal_field` must be external double arrays with
 * exactly the size of the target vector. mesh_ref was already restored in phase 1, and the
 * other nodal fields are recomputed (ApplyPostRestartFixup()).
 *
 * **Quadrature functions.** Every function registered in this run, except the recomputed ones
 * (RecomputedQuadratureFunctionNames()), is matched with the checkpoint entry of the same
 * name:
 * - A different space name, vdim or value count is a failure; the values would not line up.
 * - A `copy_of` entry is queued in m_pending_copies and filled after the load from its
 *   partner, which must exist in this run with the same size and have stored values in the
 *   checkpoint.
 * - Otherwise the stored values view must be an external double array of the right size, and
 *   is pointed at the function's host data.
 * - Empty functions (on ranks with no elements of their space) are skipped; their stored view
 *   is zero length and is dropped before the load.
 *
 * **Mismatched sets are warnings, not failures.** A function registered in this run but not in
 * the checkpoint keeps the value its constructor gave it, which is expected when a
 * continuation enables extra post-processing. A function only in the checkpoint is skipped,
 * which is expected when a continuation disables some. Both are reported on rank 0 through
 * MFEM_WARNING_0. Because region functions only exist where the region has elements, the
 * lists reflect rank 0's view; in practice the sets are decided by the options and are the
 * same everywhere a region exists.
 *
 * HostReadWrite() is used for the target pointers so that the host copy becomes the valid one
 * and, on a GPU run, the next device access copies the restored values to the device.
 */
void RestartCheckpoint::BindFieldsAndQuadratureFunctions(Group* root, SimulationState& sim_state) {
    // ---------------------------------------------------------------------
    // Part 1: nodal fields
    //
    // Only two nodal vectors are restored here. mesh_ref was restored in phase 1 (and its
    // stored copy already dropped), and every other nodal field is recomputed from these two
    // afterwards (ApplyPostRestartFixup()):
    // - mesh_current: all local DOFs of the coordinate space (VSize entries). This is also the
    //   mesh's Nodes, so restoring it moves the mesh to its checkpointed configuration.
    // - primal_field: the velocity true DOFs (TrueVSize entries), the solver's unknown.
    //
    // Each stored view must be an external double array with exactly the target's size. It
    // was written as an external view, so a stored owned view or a different type means the
    // file does not have the expected layout. The sizes were already implied by the mesh-space
    // check in phase 1, so a size mismatch here means the file is damaged.
    //
    // Binding only records the target pointer; the data is read later by RestoreState()'s
    // loadExternalData() call. HostReadWrite() marks the host copy as the valid one, so on a
    // GPU run the next device access copies the restored values to the device instead of
    // using stale device data. On a rank with no elements both vectors are empty: the size
    // check still runs (0 == 0), nothing is bound, and the zero-length stored view is dropped
    // before the load.
    // ---------------------------------------------------------------------
    const auto bind_vector = [&](const std::string& path, mfem::Vector& target) {
        if (!root->hasView(path)) {
            RecordFailure("the checkpoint has no " + path);
            return;
        }
        View* view = root->getView(path);
        if (!view->isExternal() || view->getTypeID() != axom::sidre::FLOAT64_ID ||
            view->getNumElements() != target.Size()) {
            RecordFailure(path + " has " + std::to_string(view->getNumElements()) +
                          " entries or the wrong type, but this run needs " +
                          std::to_string(target.Size()) + " doubles");
            return;
        }
        if (target.Size() > 0) {
            view->setExternalDataPtr(target.HostReadWrite());
        }
    };
    bind_vector("fields/mesh_current/values", *sim_state.m_mesh_nodes.at("mesh_current"));
    bind_vector("fields/primal_field/values", *sim_state.m_primal_field);

    // ---------------------------------------------------------------------
    // Part 2: quadrature functions
    //
    // The loop runs over this run's registry rather than over the checkpoint, because the
    // question being answered is "where does each of this run's functions get its value
    // from?". Functions that are only in the checkpoint are reported in part 3.
    //
    // For each registered function, in order:
    //   a. Skip it if it is null (an unused registry slot) or recomputed before it is next
    //      read (RecomputedQuadratureFunctionNames()); it keeps its constructor value until
    //      then, exactly as it would after a normal step.
    //   b. Look up the checkpoint entry of the same (encoded) name. If there is none, remember
    //      the name for the warning in part 3 and leave the function as constructed.
    //   c. Check that the stored function has the same space name, vdim and number of values.
    //      ValidateQuadratureSpaces() has already checked that a space of that name has the
    //      same points in the same order, so matching names plus matching shapes means value i
    //      of the stored array belongs at value i of this function. Anything else would put
    //      values on the wrong quadrature points, so it is a failure, not a warning.
    //   d. If the entry is a `copy_of`, its values were not stored because they were bitwise
    //      equal to its beginning-of-step partner (see WriteQuadratureFunctions()). Queue the
    //      copy for after the load. The partner must exist in this run with the same size, and
    //      must have stored values in the checkpoint; otherwise nothing could be copied.
    //   e. Otherwise bind the stored values to the function's host data, after checking that
    //      the view is an external double array of the right size. An empty function (a
    //      global function on a rank with no elements) is skipped before that check: there is
    //      nothing to read, and its zero-length stored view is dropped before the load.
    //
    // Every problem is recorded rather than aborting immediately, so all of them are reported
    // together and every rank reaches the agreement in RestoreState().
    // ---------------------------------------------------------------------
    if (!root->hasGroup("qfuncs")) {
        RecordFailure("the checkpoint has no qfuncs/ group");
        return;
    }
    Group* qfuncs = root->getGroup("qfuncs");
    const std::set<std::string> recomputed = RecomputedQuadratureFunctionNames(sim_state);
    std::vector<std::string> runtime_only;
    for (const auto& [name, function] : sim_state.m_map_qfs) {
        // (a) Unused or recomputed.
        if (!function || recomputed.count(name) != 0) {
            continue;
        }

        // (b) Not in the checkpoint.
        const std::string encoded = SidreSafeName(name);
        if (!qfuncs->hasGroup(encoded)) {
            runtime_only.push_back(name);
            continue;
        }

        // (c) Same space, vdim and number of values.
        Group* entry = qfuncs->getGroup(encoded);
        const std::string stored_space = entry->getView("qspace_name")->getString();
        const std::string runtime_space = FindQuadratureSpaceName(
            sim_state.m_map_qs, function->GetPartialSpaceShared().get());
        const int stored_vdim = ReadSidreScalar<int>(entry, "vdim");
        const auto stored_count = ReadSidreScalar<std::int64_t>(entry, "value_count");
        if (stored_space != runtime_space || stored_vdim != function->GetVDim() ||
            stored_count != function->Size()) {
            RecordFailure("the quadrature function '" + name +
                          "' changed shape (checkpoint: space '" + stored_space + "', vdim " +
                          std::to_string(stored_vdim) + ", " + std::to_string(stored_count) +
                          " values; this run: space '" + runtime_space + "', vdim " +
                          std::to_string(function->GetVDim()) + ", " +
                          std::to_string(function->Size()) + " values)");
            continue;
        }

        // (d) Stored as a copy of its partner: copy after the load. The source pointer is the
        // partner's SimulationState function, which the load fills from its own stored values
        // before m_pending_copies is processed.
        if (entry->hasView("copy_of")) {
            const std::string source_name = entry->getView("copy_of")->getString();
            const auto source_it = sim_state.m_map_qfs.find(source_name);
            const bool source_restorable =
                source_it != sim_state.m_map_qfs.end() && source_it->second &&
                source_it->second->Size() == function->Size() &&
                qfuncs->hasGroup(SidreSafeName(source_name)) &&
                qfuncs->getGroup(SidreSafeName(source_name))->hasView("values");
            if (!source_restorable) {
                RecordFailure("the quadrature function '" + name + "' is stored as a copy of '" +
                              source_name + "', which cannot be restored in this run");
                continue;
            }
            m_pending_copies.emplace_back(function, source_it->second);
            continue;
        }

        // (e) Stored values: point the stored view at the function's host data.
        if (function->Size() == 0) {
            continue;
        }
        if (!entry->hasView("values")) {
            RecordFailure("the quadrature function '" + name + "' has no stored values");
            continue;
        }
        View* values = entry->getView("values");
        if (!values->isExternal() || values->getTypeID() != axom::sidre::FLOAT64_ID ||
            values->getNumElements() != function->Size()) {
            RecordFailure("the stored values of the quadrature function '" + name +
                          "' have the wrong type or size");
            continue;
        }
        values->setExternalDataPtr(function->HostReadWrite());
    }

    // ---------------------------------------------------------------------
    // Part 3: report the functions present on only one side
    //
    // Neither case stops the restart (see the function documentation for why both are
    // expected when a continuation changes its post-processing options). The checkpoint-only
    // functions are left unbound here, so RestoreState() drops their views before the load.
    //
    // MFEM_WARNING_0 is used so each message appears once, from rank 0. It is not a
    // collective call: each rank evaluates its own lists, and only rank 0's output is shown.
    // Region functions only exist on ranks that own elements of the region, so if rank 0 owns
    // none of a region, a mismatch in that region's functions is not reported. In practice
    // which functions exist is decided by the options and is the same on every rank where a
    // region exists.
    // ---------------------------------------------------------------------
    for (const std::string& name : runtime_only) {
        MFEM_WARNING_0("The quadrature function '"
                       << name
                       << "' is not in the checkpoint; it keeps the value it was initialized "
                          "with");
    }
    for (Group& entry : qfuncs->groups()) {
        bool registered = false;
        for (const auto& [name, function] : sim_state.m_map_qfs) {
            if (function && SidreSafeName(name) == entry.getName()) {
                registered = true;
                break;
            }
        }
        if (!registered) {
            MFEM_WARNING_0("The checkpoint quadrature function '"
                           << entry.getName() << "' is not used by this run and is skipped");
        }
    }
}

/**
 * @brief Recompute the nodal fields derived from the restored state
 *
 * @param sim_state Simulation state whose stored fields were just restored
 *
 * @details This repeats exactly what SimulationState::FinishCycle() did at the end of the step
 * the checkpoint was written after, using the restored coordinates and velocity:
 * - mesh_t_beg = mesh_current (beginning-of-step coordinates);
 * - previous velocity = velocity (true DOFs);
 * - velocity grid function = the velocity true DOFs distributed to local DOFs (collective);
 * - displacement = mesh_current - mesh_ref.
 * mesh_current is the mesh's Nodes (SimulationState swaps it in during construction), and its
 * values just changed underneath the mesh, so the mesh is told to drop its cached geometric
 * factors. If FinishCycle() ever derives another field, it must be added here as well.
 */
void RestartCheckpoint::ApplyPostRestartFixup(SimulationState& sim_state) const {
    mfem::ParGridFunction& mesh_current = *sim_state.m_mesh_nodes.at("mesh_current");
    mfem::ParGridFunction& mesh_ref = *sim_state.m_mesh_nodes.at("mesh_ref");
    (*sim_state.m_mesh_nodes.at("mesh_t_beg")) = mesh_current;
    (*sim_state.m_primal_field_prev) = *sim_state.m_primal_field;
    sim_state.m_mesh_qoi_nodes.at("velocity")->Distribute(*sim_state.m_primal_field);
    mfem::ParGridFunction& displacement = *sim_state.m_mesh_qoi_nodes.at("displacement");
    displacement = mesh_current;
    displacement -= mesh_ref;
    sim_state.m_mesh->NodesUpdated();
}

} // namespace exaconstit

#endif // EXACONSTIT_USE_AXOM
