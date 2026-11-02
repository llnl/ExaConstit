#pragma once

#include "mfem_expt/partial_qfunc.hpp"
#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

namespace axom {
namespace sidre {
class DataStore;
class Group;
} // namespace sidre
} // namespace axom

namespace exaconstit {

/**
 * @brief Version of the checkpoint rank-file layout written by RestartCheckpoint
 *
 * Stored in `restart_info/format_version` of every rank file and in `commit.json`. Bump it
 * whenever the layout changes, so older checkpoints are rejected with a clear message instead
 * of being misread. The mesh subtree has its own version (MESH_CHECKPOINT_FORMAT_VERSION).
 */
inline constexpr int RESTART_FORMAT_VERSION = 1;

/**
 * @brief Binary checkpoint writing and restarting for a SimulationState
 *
 * @details Writes the evolving simulation state to disk with Axom Sidre (one HDF5 file per
 * rank) and restores it in a later run. A restart works from the checkpoint alone: the mesh,
 * grain, region-mapping, orientation and state-variable files of the original run are never
 * opened again. See doc/restart.md for the user-facing description and the design record.
 *
 * **Checkpoint layout on disk:**
 * @code
 *   <directory>/cycle_<completed cycles, 10 digits>/
 *       rank_<rank, 6 digits>.hdf5   written by every rank
 *       commit.json                  written last, by rank 0
 * @endcode
 * A checkpoint without `commit.json` is incomplete ("torn") and is never read.
 *
 * **Restarting is split into two phases**, because the state a restart has to restore is
 * created in two places:
 * - Phase 1 runs inside the SimulationState constructor, in place of reading the original
 *   input files: OpenForRestart(), ReconstructMesh(), RestoreReferenceConfiguration(),
 *   ReadElementGrainIds() and ReleaseMeshArrays(), in that order.
 * - Phase 2, RestoreState(), runs from the driver after SystemDriver and
 *   PostProcessingDriver have been constructed, so every quadrature function any component
 *   registers already exists. It overwrites the evolving state with the checkpointed values,
 *   replacing whatever the constructors initialized.
 *
 * **Ownership.** SimulationState owns the RestartCheckpoint (std::unique_ptr) and makes it a
 * friend so it can reach the field and quadrature-function registries. RestartCheckpoint
 * never stores a pointer back to its owner; the SimulationState& arguments are borrowed for
 * the duration of each call. Everything restored ends up owned by SimulationState, and
 * nothing is owned by Sidre once RestoreState() returns.
 *
 * **Collective calls.** Every public member function except GetRestartDriverStep() is
 * collective over the communicator given to the constructor and must be called on every
 * rank, including ranks that own no elements. A problem found on any rank aborts every rank.
 *
 * **What is checkpointed.**
 * - Quadrature data: every function registered in SimulationState's quadrature-function
 *   registry (SimulationState::AddQuadratureFunction()), whatever component registered it,
 *   except `tangent_stiffness`. Quadrature data kept outside the registry is not saved.
 * - Nodal data: a fixed list, not a registry: the reference and current coordinates and the
 *   velocity true DOFs. The other SimulationState grid functions are derived from these.
 *   Any new grid function or solution vector that carries history, e.g. an additional
 *   primal variable, must be added to WriteNodalFields() and to the binding in
 *   BindFieldsAndQuadratureFunctions(), or it is silently not restored.
 * - Mesh: the volume ParMesh only. Anything built on top of it (submeshes, surface meshes,
 *   other topologies) is not saved and must either be rebuilt deterministically from the
 *   restored mesh or have its own checkpoint data.
 * - Time state, the element grain IDs and the material properties (the properties are only
 *   compared against the restarted run's; that run uses its own).
 *
 * **Restart requirements.** The restarted run must use the same number of MPI ranks, the
 * same mesh order, and the same material list (names, model types and state-variable
 * counts). These are checked and violations abort with a message saying what changed.
 */
class RestartCheckpoint {
public:
    /**
     * @brief Construct the checkpoint handler
     *
     * @param options [Restart] options (copied)
     * @param comm Communicator of the simulation (SimulationState uses MPI_COMM_WORLD)
     */
    RestartCheckpoint(const RestartOptions& options, MPI_Comm comm);
    ~RestartCheckpoint();

    RestartCheckpoint(const RestartCheckpoint&) = delete;
    RestartCheckpoint& operator=(const RestartCheckpoint&) = delete;

    // =========================================================================
    // WRITING
    // =========================================================================

    /**
     * @brief Write a checkpoint if one is due
     *
     * @param sim_state Simulation state to write
     * @param driver_step The driver's loop counter `ti` for the step just completed
     *
     * @details Call once at the end of every driver-loop iteration, after FinishCycle(),
     * UpdateModel() and post-processing. A checkpoint is due when `[Restart] write` is set,
     * the time manager is not in the middle of a retrial or sub-step sequence, at least one
     * cycle has completed since the last checkpoint (or the restart point), and either
     * `interval` cycles have passed or the simulation just finished and `write_final` is set.
     *
     * A failed write is reported as a warning and the simulation continues; the incomplete
     * checkpoint is never read.
     */
    void WriteCheckpointIfDue(SimulationState& sim_state, std::size_t driver_step);

    // =========================================================================
    // RESTARTING, PHASE 1 (inside the SimulationState constructor)
    // =========================================================================

    /**
     * @brief Select the checkpoint to restart from and load this rank's file
     *
     * @details Uses `[Restart] read_cycle` if set, otherwise the newest complete checkpoint
     * in the read directory. Aborts if no usable checkpoint exists, or if it was written with
     * a different format version or number of ranks. Must be called before any other phase 1
     * or phase 2 function.
     */
    void OpenForRestart();

    /**
     * @brief Rebuild the parallel mesh from the loaded checkpoint
     *
     * @return The rebuilt mesh, identical to the one that was written (see ExaSidreParMesh)
     *
     * @details Replaces reading the mesh file. Aborts if the rebuilt topology does not match
     * the counts recorded in the checkpoint.
     */
    std::shared_ptr<mfem::ParMesh> ReconstructMesh();

    /**
     * @brief Fill in the element grain IDs from the checkpoint
     *
     * @param grain_ids Array sized to the local element count; receives one grain ID per
     *                  local element
     *
     * @details Replaces reading the grain IDs from the mesh file's element attributes, which
     * hold region IDs in a checkpointed mesh.
     */
    void ReadElementGrainIds(mfem::Array<int>& grain_ids);

    /**
     * @brief Overwrite the coordinate fields with the checkpointed reference configuration
     *
     * @param sim_state Simulation state under construction
     *
     * @details Call from the SimulationState constructor right after the coordinate fields
     * exist and before anything else reads them. Sets `mesh_ref`, `mesh_t_beg` and
     * `mesh_current` (the mesh's Nodes) to the checkpointed reference coordinates, which is
     * exactly what they held in the original run until its first step, and checks that the
     * mesh's finite element space numbers its DOFs as in the original run. Aborts on a
     * mismatch.
     */
    void RestoreReferenceConfiguration(SimulationState& sim_state);

    /**
     * @brief Free the mesh arrays held by the loaded checkpoint
     *
     * @details Call at the end of the SimulationState constructor; the mesh and grain IDs
     * have been copied out by then. The rest of the checkpoint stays loaded for phase 2.
     */
    void ReleaseMeshArrays();

    // =========================================================================
    // RESTARTING, PHASE 2 (from the driver)
    // =========================================================================

    /**
     * @brief Overwrite the evolving state with the checkpointed values
     *
     * @param sim_state Fully constructed simulation state
     *
     * @details Call once, after SystemDriver and PostProcessingDriver are constructed and
     * before the first time step. Checks the checkpoint against this run (materials and
     * quadrature spaces), restores the nodal fields and every quadrature function present in
     * both, resumes the time state, recomputes the fields derived from the restored state,
     * and releases the checkpoint. Aborts on a mismatch before anything is overwritten.
     */
    void RestoreState(SimulationState& sim_state);

    /**
     * @brief Driver loop counter recorded in the checkpoint
     *
     * @return The `ti` of the last completed step; the first restarted step is this + 1
     */
    std::size_t GetRestartDriverStep() const {
        return m_restart_driver_step;
    }

private:
    // ---------------------------------------------------------------------
    // Writing (each documented in restart_checkpoint.cpp)
    // ---------------------------------------------------------------------
    void WriteCheckpoint(SimulationState& sim_state,
                         std::size_t driver_step,
                         const TimeManagement::CheckpointState& time_state);
    bool PrepareCheckpointDirectory(const std::filesystem::path& checkpoint_directory);
    void WriteRestartInfo(axom::sidre::Group* root,
                          const SimulationState& sim_state,
                          std::size_t driver_step,
                          const TimeManagement::CheckpointState& time_state) const;
    void WriteMeshAndGrainIds(axom::sidre::Group* root, const SimulationState& sim_state) const;
    void WriteNodalFields(axom::sidre::Group* root, const SimulationState& sim_state) const;
    void WriteQuadratureSpaces(axom::sidre::Group* root, const SimulationState& sim_state) const;
    void WriteQuadratureFunctions(axom::sidre::Group* root, const SimulationState& sim_state) const;
    void WriteMaterials(axom::sidre::Group* root, const SimulationState& sim_state) const;
    bool WriteCommitFile(const std::filesystem::path& checkpoint_directory,
                         std::size_t completed_cycles,
                         std::size_t driver_step,
                         double completed_time) const;
    void SupersedeCheckpointsAfter(std::size_t completed_cycles) const;
    void ApplyRetentionPolicy() const;

    // ---------------------------------------------------------------------
    // Restarting (each documented in restart_checkpoint.cpp)
    // ---------------------------------------------------------------------
    std::string SelectCheckpointDirectory();
    void ValidateMeshSpace(axom::sidre::Group* root, const SimulationState& sim_state);
    void VerifyRestoredDofLayout(const SimulationState& sim_state);
    void ValidateMaterials(axom::sidre::Group* root, const SimulationState& sim_state);
    void ValidateQuadratureSpaces(axom::sidre::Group* root, const SimulationState& sim_state);
    void BindFieldsAndQuadratureFunctions(axom::sidre::Group* root, SimulationState& sim_state);
    void ApplyPostRestartFixup(SimulationState& sim_state) const;

    // ---------------------------------------------------------------------
    // Failure reporting
    // ---------------------------------------------------------------------
    /**
     * @brief Record a problem found on this rank
     *
     * @param message Description of the problem
     *
     * @details Problems are collected rather than aborting immediately, so that every rank
     * reaches AbortIfAnyRankFailed() and the abort happens on every rank together.
     */
    void RecordFailure(const std::string& message);

    /**
     * @brief Abort on every rank if any rank recorded a failure
     *
     * @param stage Short description of what was being done, used in the abort message
     *
     * @details Collective. Each failing rank reports its own recorded problems as warnings,
     * then every rank aborts with one shared message.
     */
    void AbortIfAnyRankFailed(const char* stage);

    /** @brief [Restart] options this handler was created with */
    RestartOptions m_options;
    /** @brief Communicator of the simulation */
    MPI_Comm m_comm;
    /** @brief This rank in m_comm */
    int m_rank = 0;
    /** @brief Number of ranks in m_comm */
    int m_num_ranks = 1;

    /** @brief Completed cycles at the last committed checkpoint, or at the restart point */
    std::size_t m_last_written_cycles = 0;
    /** @brief Whether the next write still has to supersede checkpoints of another timeline */
    bool m_supersede_pending = true;

    /** @brief Checkpoint loaded by OpenForRestart(); released at the end of RestoreState() */
    std::unique_ptr<axom::sidre::DataStore> m_read_store;
    /** @brief Path of this rank's file in the checkpoint being restarted from */
    std::filesystem::path m_read_rank_file;
    /** @brief Driver loop counter recorded in the checkpoint being restarted from */
    std::size_t m_restart_driver_step = 0;
    /** @brief Time state recorded in the checkpoint being restarted from */
    TimeManagement::CheckpointState m_restart_time_state;
    /** @brief Element-local hash of mesh_ref recorded in the checkpoint */
    std::uint64_t m_stored_mesh_ref_element_hash = 0;

    /**
     * @brief End-of-step quadrature functions to restore by copying their beginning-of-step
     *        partner once the bulk load is done (destination, source)
     */
    std::vector<std::pair<std::shared_ptr<mfem::expt::PartialQuadratureFunction>,
                          std::shared_ptr<mfem::expt::PartialQuadratureFunction>>>
        m_pending_copies;

    /** @brief Whether this rank recorded a failure since the last AbortIfAnyRankFailed() */
    bool m_local_failed = false;
    /** @brief Problems recorded on this rank since the last AbortIfAnyRankFailed() */
    std::string m_local_failure_log;
};

} // namespace exaconstit
