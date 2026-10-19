#pragma once

#include "boundary_conditions/BCManager.hpp"
#include "mfem_expt/partial_qfunc.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "options/option_parser_v2.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Enumeration for time step status and control
 *
 * @details Tracks the current state of time stepping algorithm and provides
 * control flow information for adaptive time stepping, sub-stepping, and
 * simulation completion detection.
 */
enum class TimeStep {
    NORMAL,  ///< Normal time stepping mode
    RETRIAL, ///< Time step failed, retrying with smaller step
    SUBSTEP, ///< Sub-stepping through an original time step
    FAILED,  ///< Time step failed completely, cannot continue
    FINAL,   ///< This is the final time step of the simulation
    FINISHED ///< Simulation is completely finished
};

/**
 * @brief Comprehensive time step management and adaptive time stepping for ExaConstit simulations
 *
 * @details This class handles all aspects of time management in ExaConstit simulations including:
 * - **Multiple Time Step Types**: Fixed, automatic, and custom time step sequences
 * - **Adaptive Time Stepping**: Automatic adjustment based on Newton-Raphson convergence behavior
 * - **Sub-stepping Recovery**: Automatic sub-stepping when convergence fails
 * - **Boundary Condition Synchronization**: Time step modification to hit exact BC change times
 * - **Failure Handling**: Sophisticated retry logic with progressive time step reduction
 * - **Final Time Detection**: Accurate detection of simulation completion
 * - **Restart Capabilities**: Complete state save/restore for checkpoint/restart
 *
 * **Time Step Types Supported**:
 * - **FIXED**: Constant time step throughout simulation (with failure handling)
 * - **AUTO**: Automatic adjustment based on solver performance
 * - **CUSTOM**: User-defined sequence of time steps from input
 *
 * **Adaptive Algorithm**: For AUTO mode, time step is adjusted using:
 * factor = (max_nr_steps * dt_scale) / actual_nr_steps
 * where fewer Newton iterations lead to larger time steps and vice versa.
 */
class TimeManagement {
private:
    /** @brief Current simulation time */
    double time = 0.0;

    /** @brief Old simulation time */
    double old_time = 0.0;

    /** @brief Final simulation time (target end time) */
    double time_final = 0.0;

    /** @brief Current time step size */
    double dt = 1.0;

    /** @brief Original time step before failure/sub-stepping */
    double dt_orig = 1.0;

    /** @brief Previous time step size (for tracking changes) */
    double prev_dt = 1.0;

    /** @brief Minimum allowed time step size */
    double dt_min = 1.0;

    /** @brief Maximum allowed time step size (AUTO mode) */
    double dt_max = 1.0;

    /** @brief Scaling factor for time step reduction on failure */
    double dt_scale = 0.25;

    /** @brief Fixed time step size (FIXED mode) */
    double dt_fixed = 1.0;

    /** @brief Time stepping mode (FIXED, AUTO, CUSTOM, NOTYPE) */
    TimeStepType time_type = TimeStepType::NOTYPE;

    /** @brief Custom time step sequence (CUSTOM mode) */
    std::vector<double> custom_dt = {};

    /** @brief Current simulation cycle number */
    size_t simulation_cycle = 0;

    /** @brief Maximum Newton-Raphson steps for AUTO time step scaling */
    size_t max_nr_steps = 25;

    /** @brief Maximum number of consecutive failures before giving up */
    size_t max_failures = 4;

    /** @brief Current number of consecutive failures */
    size_t num_failures = 0;

    /** @brief Required number of sub-steps to complete original time step */
    size_t required_num_sub_steps = 0;

    /** @brief Current number of sub-steps completed */
    size_t num_sub_steps = 0;

    /** @brief Internal state tracker for time step status */
    TimeStep internal_tracker = TimeStep::NORMAL;

public:
    /**
     * @brief Constructor - initialize time management from simulation options
     *
     * @param options Reference to ExaOptions containing time step configuration
     *
     * @details Sets up time management based on the specified time step type:
     *
     * **FIXED Mode**:
     * - Uses constant time step from options.time.fixed_time->dt
     * - Sets dt_min based on maximum allowed reductions
     * - Configures final time from options
     *
     * **AUTO Mode**:
     * - Uses starting time step from options.time.auto_time->dt_start
     * - Configures min/max bounds and scaling parameters
     * - Sets up automatic time step logging file
     * - Links to Newton solver iteration limits
     *
     * **CUSTOM Mode**:
     * - Loads user-defined time step sequence
     * - Calculates total simulation time from sequence sum
     * - Sets minimum based on smallest step in sequence
     *
     * Also performs initial time setup and final step detection.
     */
    TimeManagement(ExaOptions& options);

    /**
     * @brief Get current simulation time
     *
     * @return Current time value
     */
    double GetTime() const {
        return time;
    }

    /**
     * @brief Get actual simulation time if auto-time stepping used
     *
     * @return Actual time step value for a step
     */
    double GetTrueCycleTime() const {
        return old_time;
    }

    /**
     * @brief Get current time step size
     *
     * @return Current time step size
     */
    double GetDeltaTime() const {
        return dt;
    }

    /**
     * @brief Get current simulation cycle number
     *
     * @return Current cycle (time step) number
     */
    size_t GetSimulationCycle() const {
        return simulation_cycle;
    }

    /**
     * @brief Update time step based on solver performance and handle time advancement
     *
     * @param nr_steps Number of Newton-Raphson iterations required for convergence
     * @param success Whether the previous time step converged successfully
     * @return TimeStep status indicating the next action required
     *
     * @details This is the core time stepping algorithm that handles multiple scenarios:
     *
     * **Failure Handling** (success = false):
     * 1. Checks if already sub-stepping (returns FAILED if so)
     * 2. Saves original time step on first failure
     * 3. Reduces time step by dt_scale factor
     * 4. Enforces minimum time step limit
     * 5. Increments failure counter and returns RETRIAL or FAILED
     *
     * **Success Cases**:
     * - **Final Step**: Transitions FINAL -> FINISHED
     * - **Sub-stepping Recovery**: Continues sub-steps until original dt is recovered
     * - **Normal Advancement**: Updates cycle and computes next time step
     *
     * **Time Step Calculation by Mode**:
     * - **AUTO**: factor = (max_nr_steps * dt_scale) / nr_steps; dt *= factor
     * - **CUSTOM**: dt = custom_dt[simulation_cycle]
     * - **FIXED**: dt = dt_fixed
     *
     * **Final Time Detection**:
     * - Automatically adjusts final time step to land exactly on time_final
     * - Handles floating-point precision issues with tolerance checking
     * - Returns appropriate FINAL status when approaching end time
     *
     * @note This method is responsible for all time advancement logic and must be
     * called after each Newton solver attempt to properly manage the simulation timeline.
     */
    TimeStep UpdateDeltaTime(const int nr_steps, const bool success = true);

    /**
     * @brief Adjust time step to hit a specific boundary condition time exactly
     *
     * @param desired_bc_time Target time for boundary condition change
     * @return True if time step was adjusted, false if target time already passed
     *
     * @details This method ensures that the simulation hits specific times exactly
     * when boundary conditions need to change. The algorithm:
     * 1. Checks if the desired time hasn't already passed
     * 2. Calculates if the next time step would overshoot the target
     * 3. If overshoot detected, adjusts current time step to land exactly on target
     * 4. Handles the time update internally using ResetTime()/UpdateTime()
     *
     * This is critical for simulations with time-dependent boundary conditions where
     * exact timing is required for physical accuracy.
     *
     * @note Only modifies time step if it would overshoot the target time
     */
    bool BCTime(const double desired_bc_time);

    /**
     * @brief Advance simulation time by current time step
     *
     * @details Updates time = time + dt. Used after successful convergence
     * to move to the next time step. Called internally by UpdateDeltaTime()
     * and BCTime() methods.
     */
    void UpdateTime() {
        time += dt;
    }

    /**
     * @brief Revert time to previous value
     *
     * @details Updates time = time - dt. Used when a time step fails
     * and needs to be retried with a smaller time step. Called internally
     * by UpdateDeltaTime() and BCTime() methods.
     */
    void ResetTime() {
        time -= dt;
    }

    /**
     * @brief Restart simulation from a specific time and cycle
     *
     * @param time_restart Time to restart from
     * @param dt_restart Time step size to use
     * @param cycle Cycle number to restart from
     *
     * @details Used for simulation restarts from checkpoint data.
     * Sets all time-related state to the specified restart values.
     * Does not modify time step type or other configuration parameters.
     */
    void RestartTimeState(const double time_restart, const double dt_restart, const size_t cycle) {
        simulation_cycle = cycle;
        time = time_restart;
        dt = dt_restart;
    }

    /**
     * @brief Print retrial diagnostic information
     *
     * @details Outputs detailed information about cycle time step info including:
     * - Original time step size before we reduced things down
     * - Current time
     * - Current cycle
     * - Current time step size
     *
     * Used for debugging convergence issues and understanding when/why
     * retrying a time step is required.
     */
    void PrintRetrialStats() const {
        std::cout << "[Cycle: " << (simulation_cycle + 1) << " , time: " << time
                  << "] Previous attempts to converge failed step: dt old was " << dt_orig
                  << " new dt is " << dt << std::endl;
    }

    /**
     * @brief Print sub-stepping diagnostic information
     *
     * @details Outputs detailed information about sub-stepping including:
     * - Original time step size before sub-stepping began
     * - Current sub-step size
     * - Total number of sub-steps required to recover
     *
     * Used for debugging convergence issues and understanding when/why
     * sub-stepping is being triggered.
     */
    void PrintSubStepStats() const {
        std::cout << "[Cycle: " << (simulation_cycle + 1) << " , time: " << time
                  << "] Previous attempts to converge failed but now starting sub-stepping of our "
                     "desired time step: desired dt old was "
                  << dt_orig << " sub-stepping dt is " << dt
                  << " and number of sub-steps required is " << required_num_sub_steps << std::endl;
    }

    /**
     * @brief Print time step change information
     *
     * @details Outputs information about time step changes including:
     * - Current simulation time
     * - Previous and current time step sizes
     * - Factor by which time step changed
     *
     * Useful for monitoring adaptive time stepping behavior and understanding
     * how the solver performance affects time step selection.
     */
    void PrintTimeStats() const {
        const double factor = dt / prev_dt;
        std::cout << "Time " << time << " dt old was " << prev_dt << " dt has been updated to "
                  << dt << " and changed by a factor of " << factor << std::endl;
    }

    /**
     * @brief Check if this is the final time step
     *
     * @return True if simulation has reached final time and this is the last step
     */
    bool IsLastStep() const {
        return internal_tracker == TimeStep::FINAL;
    }

    /**
     * @brief Check if simulation is completely finished
     *
     * @return True if simulation has completed all time steps
     */
    bool IsFinished() const {
        return internal_tracker == TimeStep::FINISHED;
    }
};

/**
 * @brief Central simulation state manager for ExaConstit multi-material simulations
 *
 * @details This class serves as the central repository for all simulation data and provides
 * a unified interface for accessing mesh, material properties, quadrature functions, and
 * coordinate information across multiple material regions.
 *
 * **Key Architectural Features**:
 *
 * **Multi-Region Support**:
 * - Manages data for multiple material regions with different models (ExaCMech, UMAT)
 * - Region-aware naming scheme for all data structures
 * - Independent material properties and model types per region
 * - Seamless multi-material simulations with unified interface
 *
 * **Quadrature Function Management**:
 * - Comprehensive storage and access for all simulation data
 * - Region-specific automatic name resolution
 * - State variable mapping for complex models like ExaCMech
 * - Efficient begin/end step data management with O(1) swapping
 *
 * **Mesh and Coordinate Tracking**:
 * - Multiple coordinate systems (reference, current, time-start)
 * - Automatic mesh deformation based on velocity field
 * - Displacement computation and tracking
 * - Device-compatible coordinate management
 *
 * **Material Integration**:
 * - Support for heterogeneous material models
 * - Crystal plasticity grain management
 * - Region-specific material properties
 * - Material model type tracking per region
 *
 * **Time Management Integration**:
 * - Embedded TimeManagement for comprehensive time control
 * - Adaptive time stepping integration
 * - Restart capability support
 *
 * **Device Compatibility**:
 * All data structures support CPU/GPU execution with appropriate device memory management.
 */
class SimulationState {
private:
    // All the various quantities related to our simulations
    // aka the mesh, quadrature functions, finite element spaces,
    // mesh nodes, and various things related to our material systems

    // We might eventually need to make this a map or have a LOR version
    // if we decide to map our quadrature function data from a HOR set to a
    // LOR version to make visualizations easier...
    /** @brief Parallel mesh shared pointer */
    std::shared_ptr<mfem::ParMesh> m_mesh;
    // Get the PFES associated with the mesh
    // The same as below goes for the above as well
    /** @brief Finite element space for mesh coordinates and primary solution */
    std::shared_ptr<mfem::ParFiniteElementSpace> m_mesh_fes;
    // Map of the QuadratureSpaceBase associated with a given name
    // These QuadratureSpaceBase might also be the PartialQuadratureSpace objects
    /** @brief Map of quadrature functions by name (includes region-specific names) */
    std::map<std::string, std::shared_ptr<mfem::expt::PartialQuadratureSpace>> m_map_qs;
    // Map of the QuadratureFunction associated with a given name
    // These QuadratureFunctions might also be a PartialQuadratureFunction class
    // for when we have have multiple materials in a simulation
    /** @brief Map of quadrature spaces by region name */
    std::map<std::string, std::shared_ptr<mfem::expt::PartialQuadratureFunction>> m_map_qfs;
    // Map of the ParallelFiniteElementSpace associated with a given vector dimension
    /** @brief Map of finite element spaces by vector dimension */
    std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>> m_map_pfes;
    // Map of the FiniteElementCollection associated with the typical FEC name
    // Typically would be something like L2_3D_P2 (FECTYPE _ #SPACEDIM D_P #MESHORDER)
    // The name is based on the name that MFEM prints out for along with any GridFunction that
    // tells us what FiniteElementCollection it belongs to
    /** @brief Map of finite element collections by type string */
    std::map<std::string, std::shared_ptr<mfem::FiniteElementCollection>> m_map_fec;
    // Map of the mesh nodes associated with a given region maybe?
    /** @brief Map of mesh coordinate grid functions (current, reference, time-start) */
    std::map<std::string, std::shared_ptr<mfem::ParGridFunction>> m_mesh_nodes;
    // Map of the mesh nodes associated with a QoI aka x_nodes-> time_{0}, time_{i}, time_{i+1},
    // velocity, displacement
    /** @brief Map of mesh quantities of interest (velocity, displacement) */
    std::map<std::string, std::shared_ptr<mfem::ParGridFunction>> m_mesh_qoi_nodes;

    // Our velocity field
    /** @brief Current primal field (velocity degrees of freedom) */
    std::shared_ptr<mfem::Vector> m_primal_field;
    /** @brief Previous time step primal field for rollback capability */
    std::shared_ptr<mfem::Vector> m_primal_field_prev;
    /** @brief Grain ID array for crystal plasticity models */
    std::shared_ptr<mfem::ParGridFunction> m_grains;

    // Map of the material properties associated with a given region name
    /** @brief Material properties organized by region name */
    std::map<std::string, std::vector<double>> m_material_properties;
    // Vector of the material region name and the region index associated with it
    /** @brief Material name and region ID pairs for region management */
    std::vector<std::pair<std::string, int>> m_material_name_region;
    /** @brief Material model type (EXACMECH, UMAT) by region index */
    std::vector<MechType> m_region_material_type;
    // Map of the quadrature function name to the potential offset in the quadrature function and
    // the vector dimension associated with that quadrature function name.
    // This variable is useful to obtain sub-mappings within a quadrature function used for all
    // history variables such as how it's done with ECMech's models.
    /** @brief Quadrature function state variable mappings for ExaCMech models */
    std::map<std::string, std::pair<int, int>> m_map_qf_mappings;
    // Class devoted to updating our time based on various logic we might have.
    /** @brief Time management instance for comprehensive time control */
    TimeManagement m_time_manager;
    // Only need 1 instance of our boundary condition manager
    // BCManager m_bc_manager;

    // Vector of the names of the quadrature function pairs that have their data ptrs
    // swapped when UpdateModel() is called.
    /** @brief Quadrature function pairs for efficient model updates (begin/end step swapping) */
    std::vector<std::pair<std::string, std::string>> m_model_update_qf_pairs;
    /** @brief Reference to simulation options */
    ExaOptions& m_options;

#if defined(EXACONSTIT_USE_AXOM)
    // We want this to be something akin to a axom::sidre::MFEMSidreDataCollection
    // However, we need it flexible enough to handle multiple different mesh topologies in it that
    // we might due to different mfem::SubMesh objects that correspond to each
    // PartialQuadraturePoint
    /** @brief Simulation restart data store (optional Axom/Sidre support) */
    std::unique_ptr<axom::sidre::DataStore> m_simulation_restart;
#endif
    /** @brief MPI rank identifier */
    int my_id;

    /** @brief Map storing whether each region has elements on this MPI rank
     *  @details Key: region_id, Value: true if region has elements on this rank
     */
    std::unordered_map<int, bool> m_is_region_active;

    /** @brief MPI communicators for each region containing only ranks with that region
     *  @details Key: region_id, Value: MPI communicator (MPI_COMM_NULL if region not on this rank)
     */
    std::unordered_map<int, MPI_Comm> m_region_communicators;

    /** @brief Map storing the root (lowest) MPI rank that has each region
     *  @details Key: region_id, Value: lowest rank with this region
     */
    std::unordered_map<int, int> m_region_root_rank;

public:
    /** @brief Runtime model for device execution (CPU/OpenMP/GPU) */
    RTModel class_device;

public:
    /**
     * @brief Constructor - initializes complete simulation state from options
     *
     * @param options Reference to simulation options containing all configuration
     *
     * @details Sets up the complete simulation state including:
     * - Mesh loading and finite element space creation
     * - Material regions and properties setup
     * - Quadrature functions for all regions and data types
     * - Coordinate tracking grid functions (reference, current, time-start)
     * - Time management initialization
     * - Device memory configuration
     * - Multi-material data structure organization
     * - Crystal plasticity grain setup if applicable
     */
    SimulationState(ExaOptions& options);
    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~SimulationState();

    // =========================================================================
    // INITIALIZATION METHODS
    // =========================================================================

    /**
     * @brief Initialize state variables and grain orientation data for all material regions
     *
     * @details This method handles the complete initialization of material-specific data:
     * 1. **Shared Orientation Loading**: Loads crystal orientation data once for all regions
     * 2. **Region-Specific Initialization**: Calls InitializeRegionStateVariables for each region
     * 3. **State Variable Setup**: Copies beginning-of-step to end-of-step data
     * 4. **Memory Cleanup**: Frees shared orientation data after all regions are initialized
     *
     * Replaces the global setStateVarData function with a per-region approach that
     * supports multiple material types and grain distributions.
     */
    void InitializeStateVariables(const std::map<int, int>& grains2region);

    // =========================================================================
    // QUADRATURE FUNCTION MANAGEMENT
    // =========================================================================

    /**
     * @brief Generate region-specific quadrature function name
     *
     * @param qf_name Base quadrature function name
     * @param region Region index (-1 for global/non-region-specific)
     * @return Region-specific quadrature function name
     *
     * @details Creates region-specific names using the pattern:
     * "base_name_materialname_regionid"
     *
     * Examples:
     * - "cauchy_stress_beg" + region 0 (steel) -> "cauchy_stress_beg_steel_0"
     * - "state_var_end" + region 1 (aluminum) -> "state_var_end_aluminum_1"
     * - "def_grad_beg" + region -1 -> "def_grad_beg" (global)
     *
     * This naming scheme enables transparent multi-material support.
     */
    std::string GetQuadratureFunctionMapName(const std::string_view& qf_name,
                                             const int region = -1) const {
        if (region < 0) {
            return std::string(qf_name);
        }
        std::string mat_name = GetRegionName(region);
        std::string qf_name_mat = std::string(qf_name) + "_" + mat_name;
        return qf_name_mat;
    }

    /**
     * @brief Get quadrature function for specific region
     *
     * @param qf_name Quadrature function name
     * @param region Region index (-1 for global)
     * @return Shared pointer to the quadrature function
     * @return The quadrature function, or nullptr if no function is registered under the
     *         resolved name (the lookup never inserts into the registry).
     *
     * @details Primary interface for accessing simulation data. Automatically resolves
     * region-specific names and returns the appropriate quadrature function.
     *
     * **Common Usage Patterns**:
     * ```cpp
     * // Get stress for region 0
     * auto stress = sim_state.GetQuadratureFunction("cauchy_stress_beg", 0);
     *
     * // Get global deformation gradient
     * auto def_grad = sim_state.GetQuadratureFunction("def_grad_beg");
     *
     * // Get state variables for specific region
     * auto state_vars = sim_state.GetQuadratureFunction("state_var_end", region_id);
     * ```
     */
    std::shared_ptr<mfem::expt::PartialQuadratureFunction>
    GetQuadratureFunction(const std::string_view& qf_name, const int region = -1) {
        // find(), not operator[]: a lookup of an unregistered name (e.g. a region inactive on
        // this rank) must not insert a nullptr entry into m_map_qfs.
        const auto it = m_map_qfs.find(GetQuadratureFunctionMapName(qf_name, region));
        return (it != m_map_qfs.end()) ? it->second : nullptr;
    }

    /**
     * @brief Add a new quadrature function for a specific region
     *
     * @param qf_name Base quadrature function name
     * @param vdim Vector dimension of the quadrature function
     * @param region Region index (-1 for global)
     * @return True if added; false if a function of that name is already registered, or if
     *         the region has no quadrature space on this rank (it owns none of the region's
     *         elements)
     *
     * @details Creates a new quadrature function with the specified vector dimension
     * and associates it with the given region. The function is initialized with zeros
     * and uses the appropriate quadrature space for the region.
     *
     * **Vector Dimensions for Common Data**:
     * - Scalars (pressure, von Mises stress): vdim = 1
     * - Stress/strain tensors: vdim = 6 (Voigt notation)
     * - Deformation gradients: vdim = 9 (3x3 matrix)
     * - State variables: vdim = model-dependent
     */
    bool AddQuadratureFunction(const std::string_view& qf_name,
                               const int vdim = 1,
                               const int region = -1);

    /**
     * @brief Get state variable mapping for ExaCMech models
     *
     * @param state_name State variable name (e.g., "slip_rates", "hardness")
     * @param region Region index (-1 for global)
     * @return Pair containing (offset, length) within the state variable vector
     * @throws std::out_of_range if mapping doesn't exist
     *
     * @details ExaCMech models store multiple state variables in a single large vector.
     * This method returns the offset and length for a specific state variable within
     * that vector, enabling efficient access to individual quantities.
     *
     * **Example Usage**:
     * ```cpp
     * auto [offset, length] = sim_state.GetQuadratureFunctionStatePair("slip_rates", 0);
     * // slip_rates for region 0 starts at 'offset' and has 'length' components
     * ```
     */
    std::pair<int, int> GetQuadratureFunctionStatePair(const std::string_view& state_name,
                                                       const int region = -1) const;

    /**
     * @brief Add state variable mapping for ExaCMech models
     *
     * @param state_name State variable name
     * @param state_pair Pair containing (offset, length) within state vector
     * @param region Region index
     * @return True if successfully added, false if already exists
     *
     * @details Used by ExaCMech models during initialization to register the location
     * of specific state variables within the large state variable vector. This enables
     * efficient access without searching or string parsing during simulation.
     */
    bool AddQuadratureFunctionStatePair(const std::string_view state_name,
                                        std::pair<int, int> state_pair,
                                        const int region);

    // =========================================================================
    // MODEL UPDATE MANAGEMENT
    // =========================================================================

    /**
     * @brief Register quadrature function pairs for model updates
     *
     * @param update_var_pair Pair of (beginning_step_name, end_step_name)
     *
     * @details Registers pairs of quadrature functions that need to have their
     * data swapped when UpdateModel() is called. Typically used for begin/end
     * step variables like:
     * - ("cauchy_stress_beg", "cauchy_stress_end")
     * - ("state_var_beg", "state_var_end")
     * - ("def_grad_beg", "def_grad_end")
     */
    void AddUpdateVariablePairNames(std::pair<std::string_view, std::string_view> update_var_pair) {
        std::string view1(update_var_pair.first);
        std::string view2(update_var_pair.second);
        m_model_update_qf_pairs.push_back({view1, view2});
    }

    /**
     * @brief Update model variables by swapping begin/end step data
     *
     * @details Performs efficient O(1) pointer swaps between beginning and end time step
     * variables for all registered quadrature function pairs. This moves end-of-step
     * converged values to beginning-of-step for the next time step without data copying.
     *
     * Called after successful convergence to prepare for the next time step.
     */
    void UpdateModel() {
        for (auto [name_prev, name_cur] : m_model_update_qf_pairs) {
            m_map_qfs.at(name_prev)->Swap(*m_map_qfs.at(name_cur));
        }
    }

    /**
     * @brief Setup model variables by copying begin to end step data
     *
     * @details Copies beginning-of-step data to end-of-step quadrature functions
     * for all registered pairs. Used at the start of a time step to initialize
     * end-step variables with beginning-step values before material model execution.
     */
    void SetupModelVariables() {
        for (auto [name_prev, name_cur] : m_model_update_qf_pairs) {
            // Value copy (PartialQuadratureFunction's explicit copy assignment).
            *m_map_qfs.at(name_cur) = *m_map_qfs.at(name_prev);
        }
    }

    // =========================================================================
    // MESH AND COORDINATE MANAGEMENT
    // =========================================================================

    /**
     * @brief Get the simulation mesh
     *
     * @return Shared pointer to the parallel mesh
     */
    std::shared_ptr<mfem::ParMesh> GetMesh() {
        return m_mesh;
    }

    /**
     * @brief Get current mesh coordinates
     *
     * @return Shared pointer to current coordinate grid function
     *
     * @details Returns the current deformed mesh coordinates. Updated after
     * each converged time step based on the velocity field using:
     * current_coords = time_start_coords + velocity * dt
     */
    std::shared_ptr<mfem::ParGridFunction> GetCurrentCoords() {
        return m_mesh_nodes["mesh_current"];
    }
    /**
     * @brief Get beginning-of-time-step mesh coordinates
     *
     * @return Shared pointer to time step start coordinate grid function
     *
     * @details Coordinates at the beginning of the current time step, used as
     * the reference for computing incremental deformation during the step.
     */
    std::shared_ptr<mfem::ParGridFunction> GetTimeStartCoords() {
        return m_mesh_nodes["mesh_t_beg"];
    }

    /**
     * @brief Get reference mesh coordinates
     *
     * @return Shared pointer to reference coordinate grid function
     *
     * @details Returns the undeformed reference configuration coordinates.
     * Used for computing total deformation gradients and strains from the
     * original configuration.
     */
    std::shared_ptr<mfem::ParGridFunction> GetRefCoords() {
        return m_mesh_nodes["mesh_ref"];
    }

    /**
     * @brief Get displacement field
     *
     * @return Shared pointer to displacement grid function
     *
     * @details Total displacement from reference configuration:
     * displacement = current_coords - reference_coords
     */
    std::shared_ptr<mfem::ParGridFunction> GetDisplacement() {
        return m_mesh_qoi_nodes["displacement"];
    }

    /**
     * @brief Get velocity field
     *
     * @return Shared pointer to velocity grid function
     *
     * @details Current nodal velocity field, which is the primary unknown
     * in ExaConstit's velocity-based formulation.
     */
    std::shared_ptr<mfem::ParGridFunction> GetVelocity() {
        return m_mesh_qoi_nodes["velocity"];
    }

    /**
     * @brief Get global visualization quadrature space
     *
     * @return Shared pointer to global quadrature space for visualization
     */
    std::shared_ptr<mfem::expt::PartialQuadratureSpace> GetGlobalVizQuadSpace() {
        return m_map_qs["global_ord_0"];
    }

    /**
     * @brief Update nodal coordinates based on current velocity solution
     *
     * @details Updates mesh coordinates after Newton solver convergence using:
     * 1. Distribute velocity solution to grid function
     * 2. current_coords = time_start_coords + velocity * dt
     *
     * This implements the updated Lagrangian formulation by moving the mesh
     * according to the computed velocity field.
     */
    void UpdateNodalEndCoords() {
        m_mesh_qoi_nodes["velocity"]->Distribute(*m_primal_field);
        (*m_mesh_nodes["mesh_current"]) = *m_mesh_qoi_nodes["velocity"];
        (*m_mesh_nodes["mesh_current"]) *= GetDeltaTime();
        (*m_mesh_nodes["mesh_current"]) += *m_mesh_nodes["mesh_t_beg"];
    }

    /**
     * @brief Restart cycle by reverting to previous time step state
     *
     * @details Reverts mesh coordinates and primal field to previous time step
     * values when a time step fails and needs to be retried with a smaller
     * time step size. Ensures simulation state consistency for adaptive stepping.
     */
    void RestartCycle() {
        m_mesh_qoi_nodes["velocity"]->Distribute(*m_primal_field_prev);
        (*m_primal_field) = *m_primal_field_prev;
        (*m_mesh_nodes["mesh_current"]) = (*m_mesh_nodes["mesh_t_beg"]);
    }

    /**
     * @brief Finalize current cycle after successful convergence
     *
     * @details Completes the current time step by:
     * 1. Copying current primal field to previous (for next rollback if needed)
     * 2. Computing displacement = current_coords - reference_coords
     * 3. Distributing velocity solution to grid function
     * 4. Updating time-start coordinates = current coordinates
     *
     * Prepares the simulation state for the next time step.
     */
    void FinishCycle();

    // =========================================================================
    // FINITE ELEMENT SPACE MANAGEMENT
    // =========================================================================

    /**
     * @brief Get finite element space for specified vector dimension
     *
     * @param vdim Vector dimension required
     * @return Shared pointer to finite element space
     *
     * @details Creates L2 discontinuous finite element spaces with specified vector
     * dimension on demand. Spaces are cached for reuse. Uses L2 elements appropriate
     * for quadrature data projection and visualization.
     *
     * **Common Vector Dimensions**:
     * - vdim = 1: Scalar fields (pressure, temperature, von Mises stress)
     * - vdim = 3: Vector fields (velocity, displacement)
     * - vdim = 6: Symmetric tensors (stress, strain in Voigt notation)
     * - vdim = 9: Full tensors (deformation gradient, velocity gradient)
     */
    std::shared_ptr<mfem::ParFiniteElementSpace> GetParFiniteElementSpace(const int vdim);

    /**
     * @brief Get the main mesh finite element space
     *
     * @return Shared pointer to mesh finite element space
     *
     * @details Returns the finite element space used for mesh coordinates
     * and primary solution fields. Typically uses H1 continuous elements
     * for the velocity-based formulation.
     */
    std::shared_ptr<mfem::ParFiniteElementSpace> GetMeshParFiniteElementSpace() {
        return m_mesh_fes;
    }

    /**
     * @brief Get finite element collection by string identifier
     *
     * @param fec_str String identifier for the finite element collection
     * @return Shared pointer to finite element collection
     *
     * @details Retrieves or creates finite element collections based on string identifiers.
     * The string format typically follows the pattern "ElementType_SpaceDim_Order"
     * (e.g., "L2_3D_P0", "H1_2D_P1"). Collections are cached for reuse to avoid
     * unnecessary memory allocation.
     *
     * **Common Collection Types**:
     * - "L2_3D_P0": Discontinuous constant elements for 3D
     * - "H1_3D_P1": Continuous linear elements for 3D
     * - "L2_2D_P0": Discontinuous constant elements for 2D
     *
     * Used internally by GetParFiniteElementSpace() and other methods requiring
     * specific finite element collections.
     */
    std::shared_ptr<mfem::FiniteElementCollection>
    GetFiniteElementCollection(const std::string fec_str) {
        return m_map_fec[fec_str];
    }

    // =========================================================================
    // MATERIAL AND REGION MANAGEMENT
    // =========================================================================

    /**
     * @brief Get number of material regions
     *
     * @return Number of regions in the simulation
     */
    size_t GetNumberOfRegions() const {
        return m_material_name_region.size();
    }

    /**
     * @brief Get material model type for a region
     *
     * @param idx Region index
     * @return Material model type (EXACMECH, UMAT, etc.)
     */
    MechType GetRegionModelType(const size_t idx) const {
        return m_region_material_type[idx];
    }

    /**
     * @brief Get region name string
     *
     * @param region Region index (-1 for global)
     * @return Region name string
     *
     * @details Returns formatted region name as "material_name_region_id"
     * (e.g., "steel_1", "aluminum_2") or "global" for region = -1.
     */
    std::string GetRegionName(const int region) const {
        if (region < 0) {
            return "global";
        }
        size_t region_idx = static_cast<size_t>(region);
        return m_material_name_region[region_idx].first + "_" +
               std::to_string(m_material_name_region[region_idx].second + 1);
    }

    /**
     * @brief Get display region name string
     *
     * @param region Region index (-1 for global)
     * @return Region name string
     *
     * @details Returns formatted region name as "material_name_region_id"
     * (e.g., "Steel 1", "Aluminum 2") or "Global" for region = -1.
     */
    std::string GetRegionDisplayName(const int region) const;

    /**
     * @brief Get material properties for a specific region
     *
     * @param region Region index
     * @return Const reference to material properties vector
     *
     * @details Material properties are stored as vectors of doubles containing
     * model-specific parameters (elastic moduli, yield strengths, hardening
     * parameters, etc.).
     */
    const std::vector<double>& GetMaterialProperties(const int region) const {
        const auto region_name = GetRegionName(region);
        return GetMaterialProperties(region_name);
    }

    /**
     * @brief Get material properties by region name
     *
     * @param region_name Name of the region
     * @return Const reference to material properties vector
     */
    const std::vector<double>& GetMaterialProperties(const std::string& region_name) const {
        return m_material_properties.at(region_name);
    }

    /**
     * @brief Get grain ID array
     *
     * @return Shared pointer to grain ID array for crystal plasticity
     *
     * @details Array mapping each element to its corresponding grain ID
     * for crystal plasticity simulations. Used to assign orientations
     * and track grain-specific behavior.
     */
    std::shared_ptr<mfem::ParGridFunction> GetGrains() {
        return m_grains;
    }

    /** @brief Check if a region has any elements on this MPI rank
     *  @param region_id The region identifier to check
     *  @return true if region has elements on this rank, false otherwise
     */
    bool IsRegionActive(int region_id) const {
        auto it = m_is_region_active.find(region_id);
        return it != m_is_region_active.end() && it->second;
    }

    /** @brief Get the MPI communicator for a specific region
     *  @param region_id The region identifier
     *  @return MPI communicator for the region, or MPI_COMM_NULL if region not on this rank
     *  @note Only ranks with elements in the region are part of the returned communicator
     */
    MPI_Comm GetRegionCommunicator(int region_id) const {
        auto it = m_region_communicators.find(region_id);
        return (it != m_region_communicators.end()) ? it->second : MPI_COMM_NULL;
    }

    /** @brief Get the root (lowest) MPI rank that has a specific region
     *  @param region_id The region identifier
     *  @return The lowest rank with this region, or -1 if region doesn't exist
     */
    int GetRegionRootRank(int region_id) const {
        auto it = m_region_root_rank.find(region_id);
        return (it != m_region_root_rank.end()) ? it->second : -1;
    }

    /** @brief Get the root (lowest) MPI rank mapping
     *  @return The root (lowest) MPI rank mapping
     */
    const auto& GetRegionRootRankMapping() const {
        return m_region_root_rank;
    }

    /** @brief Check if this rank is responsible for I/O for a given region
     *  @param region_id The region identifier
     *  @return true if this rank should handle I/O for the region
     */
    bool IsRegionIORoot(int region_id) const {
        return GetRegionRootRank(region_id) == my_id;
    }

    int GetMPIID() const {
        return my_id;
    }

    // =========================================================================
    // SOLUTION FIELD ACCESS
    // =========================================================================

    /**
     * @brief Get current primal field (velocity DOFs)
     *
     * @return Shared pointer to current primal field vector
     *
     * @details The primal field contains velocity degrees of freedom in
     * ExaConstit's velocity-based formulation. This is the primary unknown
     * solved by the Newton-Raphson algorithm.
     */
    std::shared_ptr<mfem::Vector> GetPrimalField() {
        return m_primal_field;
    }

    /**
     * @brief Get previous time step primal field
     *
     * @return Shared pointer to previous primal field vector
     *
     * @details Stores the converged primal field from the previous time step.
     * Used for rollback when time step fails and for providing initial
     * guesses in adaptive time stepping.
     */
    std::shared_ptr<mfem::Vector> GetPrimalFieldPrev() {
        return m_primal_field_prev;
    }

    // =========================================================================
    // SIMULATION CONTROL
    // =========================================================================
    /**
     * @brief Get simulation options
     *
     * @return Const reference to simulation options
     */
    const ExaOptions& GetOptions() const {
        return m_options;
    }

    /**
     * @brief Get current simulation time
     *
     * @return Current time value from TimeManagement
     */
    double GetTime() const {
        return m_time_manager.GetTime();
    }

    /**
     * @brief Get actual simulation time for a given cycle as auto-time step might have changed
     * things
     *
     * @return Current time value from TimeManagement
     */
    double GetTrueCycleTime() const {
        return m_time_manager.GetTrueCycleTime();
    }

    /**
     * @brief Get current time step size
     *
     * @return Current time step size from TimeManagement
     */
    double GetDeltaTime() const {
        return m_time_manager.GetDeltaTime();
    }

    /**
     * @brief Update time step based on solver performance
     *
     * @param nr_steps Number of Newton-Raphson iterations required
     * @param success Whether the time step converged (true) or failed (false)
     * @return Updated time step status
     *
     * @details Delegates to TimeManagement for comprehensive adaptive time step control.
     * See TimeManagement::UpdateDeltaTime() for detailed algorithm description.
     */
    TimeStep UpdateDeltaTime(const int nr_steps, const bool success) {
        return m_time_manager.UpdateDeltaTime(nr_steps, success);
    }

    /**
     * @brief Get current simulation cycle
     *
     * @return Current simulation cycle from TimeManagement
     */
    size_t GetSimulationCycle() const {
        return m_time_manager.GetSimulationCycle();
    }

    /**
     * @brief Check if this is the last time step
     *
     * @return True if simulation has reached final time
     */
    bool IsLastStep() const {
        return m_time_manager.IsLastStep();
    }

    /**
     * @brief Check if simulation is finished
     *
     * @return True if simulation is complete
     */
    bool IsFinished() const {
        return m_time_manager.IsFinished();
    }

    /**
     * @brief Print time step statistics
     *
     * @details Outputs current time and time step information for monitoring
     * adaptive time step behavior. Delegates to TimeManagement.
     */
    void PrintTimeStats() const {
        m_time_manager.PrintTimeStats();
    }

    /**
     * @brief Print retrial time step statistics
     *
     * @details Outputs current time and time step information for monitoring
     * adaptive time step behavior. Delegates to TimeManagement.
     */
    void PrintRetrialTimeStats() const {
        m_time_manager.PrintRetrialStats();
    }

private:
    /** @brief Create MPI communicators for each region containing only ranks with that region
     *  @details This prevents deadlocks in collective operations when some ranks have no
     *           elements for a region. Must be called after m_is_region_active is populated.
     */
    void CreateRegionCommunicators();

    /**
     * @brief Initialize region-specific state variables
     *
     * @param region_id Region identifier
     * @param material Material options for this region
     * @param grains2region Mapping from grain IDs to regions
     *
     * @details Helper method that handles state variable initialization for
     * a single material region, including grain orientation assignment and
     * model-specific state variable setup.
     */
    void InitializeRegionStateVariables(int region_id,
                                        const MaterialOptions& material,
                                        const std::map<int, int>& grains2region);

    /**
     * @brief Utility function to update the number of state variables count in our options if a
     * model uses orientations
     */
    void UpdateExaOptionsWithOrientationCounts();

    /**
     * @brief Shared crystal orientation data for multi-region crystal plasticity
     *
     * @details Temporary storage structure used during initialization to share
     * crystal orientation data across multiple material regions. This avoids
     * loading the same orientation file multiple times when several regions
     * use the same grain structure.
     *
     * The data is loaded once, used by all regions that need it, then cleaned
     * up to free memory. The values are stored exactly as they appear in the file;
     * how they are interpreted depends on each region's material model (see
     * PrepareOrientationForRegion).
     */
    struct SharedOrientationData {
        /** @brief Raw values read from the orientation file */
        std::vector<double> values;
        /** @brief Number of grains loaded */
        int num_grains;
        /** @brief Flag indicating if data has been successfully loaded */
        bool is_loaded;
        /**
         * @brief Default constructor
         *
         * @details Initializes an empty, unloaded state.
         */
        SharedOrientationData() : num_grains(0), is_loaded(false) {}
    };

    /**
     * @brief Per-region orientation configuration for crystal plasticity
     *
     * @details Stores the orientation data for a particular material region along
     * with indexing information. ExaCMech models use unit quaternions, while other
     * models (e.g. UMATs) use the values from the orientation file in whatever
     * format they expect, with ori_stride values per grain.
     *
     * Used during state variable initialization to efficiently assign orientations
     * to elements based on their grain IDs and region membership.
     */
    struct OrientationConfig {
        /** @brief Orientation data in the format required by this region's material model */
        std::vector<double> data;
        /** @brief Number of components per orientation (4 for quaternions, 3 for Euler angles,
         * etc.) */
        int stride;
        /** @brief Starting index in the state variable vector for orientation data */
        int offset_start;
        /** @brief Ending index in the state variable vector for orientation data */
        int offset_end;
        /** @brief Flag indicating if this configuration is valid and ready for use */
        bool is_valid;
        /**
         * @brief Default constructor
         *
         * @details Initializes an invalid configuration that must be properly
         * set up before use.
         */
        OrientationConfig() : stride(0), offset_start(-1), offset_end(0), is_valid(false) {}
    };

    /** @brief Shared orientation data for crystal plasticity (temporary storage during
     * initialization) */
    SharedOrientationData m_shared_orientation_data;

    /**
     * @brief Load shared orientation data for crystal plasticity
     *
     * @param orientation_file Path to orientation data file
     * @param num_grains Number of grains to load
     *
     * @details Reads all of the values in the orientation file so they can be shared
     * across multiple material regions. The values are not interpreted here, as their
     * format depends on each region's material model. Aborts if the file can't be read,
     * has no values, or the grain count doesn't match previously loaded data.
     */
    void LoadSharedOrientationData(const std::string& orientation_file, int num_grains);

    /**
     * @brief Convert quaternions to Euler angles
     *
     * @param quaternions Vector of quaternion data (w,x,y,z format)
     * @param num_grains Number of grains to convert
     * @return Vector of Euler angles in Bunge convention
     *
     * @details Utility function for converting between orientation representations
     * when different material models require different formats.
     */
    std::vector<double> ConvertQuaternionsToEuler(const std::vector<double>& quaternions,
                                                  int num_grains);

    /**
     * @brief Convert unit quaternions to rotation matrices
     * @param quaternions Vector containing unit quaternions (stride 4)
     * @param num_grains Number of grains
     * @return Vector of 3x3 rotation matrices (stride 9)
     */
    std::vector<double> ConvertQuaternionsToMatrix(const std::vector<double>& quaternions,
                                                   int num_grains);

    /**
     * @brief Prepare orientation data for a specific region/material
     * @param material Material options containing grain info and orientation requirements
     * @return OrientationConfig with the orientation data for this material
     *
     * @details ExaCMech models require unit quaternions (w, x, y, z) representing passive
     * rotations from the crystal to the sample frame, which are normalized if needed. Any
     * other model is given the values from the orientation file as is, with a stride set by
     * ori_type (4 for quat, 3 for euler, or ori_stride for custom).
     */
    OrientationConfig PrepareOrientationForRegion(const MaterialOptions& material);

    /**
     * @brief Calculate the effective state variable count including orientations
     * @param material Material options
     * @return Total count including orientation variables if present
     */
    int CalculateEffectiveStateVarCount(const MaterialOptions& material);

    /**
     * @brief Determine placement offsets for orientation data in state variable array
     * @param material Material options
     * @param orientation_stride Number of orientation components per grain
     * @return Pair of (offset_start, offset_end) indices
     */
    std::pair<int, int> CalculateOrientationOffsets(const MaterialOptions& material,
                                                    int orientation_stride);

    /**
     * @brief Fill orientation data into the state variable array at a specific quadrature point
     * @param qf_data Pointer to QuadratureFunction data
     * @param qpt_base_index Base index for current quadrature point
     * @param qf_vdim Vector dimension of QuadratureFunction
     * @param grain_id Grain ID for current element
     * @param orientation_config Orientation configuration with data and offsets
     */
    void FillOrientationData(double* qf_data,
                             int qpt_base_index,
                             int qf_vdim,
                             int grain_id,
                             const OrientationConfig& orientation_config);

    /**
     * @brief Clean up shared orientation data after initialization
     *
     * @details Frees memory used by shared orientation data after all regions
     * have been initialized. Helps reduce memory footprint for large simulations.
     */
    void CleanupSharedOrientationData();
};