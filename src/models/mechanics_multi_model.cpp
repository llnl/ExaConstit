#include "models/mechanics_multi_model.hpp"

#include "mfem_expt/partial_qfunc.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "models/mechanics_ecmech.hpp"
#include "models/mechanics_umat.hpp"
#include "umats/unified_umat_loader.hpp"
#include "utilities/dynamic_function_loader.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

/**
 * @brief Resolve UMAT library paths with search path support
 *
 * @param library_path Relative or absolute path to UMAT library
 * @param search_paths List of directories to search for the library
 * @return Resolved absolute path to the library
 *
 * @details Resolves UMAT library paths by:
 * 1. Using absolute paths as-is
 * 2. Searching through provided search paths for relative paths
 * 3. Checking current directory as fallback
 * 4. Warning if library is not found
 */
fs::path resolveUmatLibraryPath(const fs::path& library_path,
                                const std::vector<fs::path>& search_paths) {
    // If absolute path, use as-is
    if (fs::path(library_path).is_absolute()) {
        return library_path;
    }

    // Search in specified paths
    for (const auto& search_path : search_paths) {
        auto full_path = fs::path(search_path) / library_path;
        if (fs::exists(full_path)) {
            return full_path.string();
        }
    }

    // Try current directory
    if (fs::exists(library_path)) {
        return fs::absolute(library_path);
    }

    MFEM_WARNING_0("UMAT library not found: " << library_path);
    return library_path; // Return original path, let loader handle the error
}

/**
 * @brief Convert string-based load strategy to enum
 *
 * @param strategy_str String representation of load strategy
 * @return Corresponding LoadStrategy enum value
 *
 * @details Converts string-based load strategy specifications from configuration files
 * to the appropriate enum values. Supports "persistent", "load_on_setup", and "lazy_load"
 * strategies.
 */
inline exaconstit::LoadStrategy stringToLoadStrategy(const std::string& strategy_str) {
    if (strategy_str == "persistent")
        return exaconstit::LoadStrategy::PERSISTENT;
    if (strategy_str == "load_on_setup")
        return exaconstit::LoadStrategy::LOAD_ON_SETUP;
    if (strategy_str == "lazy_load")
        return exaconstit::LoadStrategy::LAZY_LOAD;

    MFEM_WARNING_0("Unknown load strategy '" << strategy_str << "', using 'persistent'");
    return exaconstit::LoadStrategy::PERSISTENT;
}

/**
 * @brief Factory function to create appropriate material model type
 *
 * @param mat_config Material configuration options
 * @param sim_state Reference to simulation state
 * @return Unique pointer to created material model
 *
 * @details Factory function that creates the appropriate material model type (UMAT, ExaCMech, etc.)
 * based on the material configuration. Handles library path resolution for UMAT models and
 * parameter setup for all model types.
 */
std::unique_ptr<ExaModel> CreateMaterialModel(const MaterialOptions& mat_config,
                                              std::shared_ptr<SimulationState> sim_state) {
    if (mat_config.mech_type == MechType::UMAT && mat_config.model.umat.has_value()) {
        const auto& umat_config = mat_config.model.umat.value();
        // Resolve library path using search paths
        auto resolved_path = resolveUmatLibraryPath(umat_config.library_path,
                                                    umat_config.search_paths);

        const auto load_strategy = stringToLoadStrategy(umat_config.load_strategy);
        // Create enhanced UMAT model
        auto umat_model = std::make_unique<AbaqusUmatModel>(
            mat_config.region_id - 1,
            mat_config.state_vars.num_vars,
            sim_state,
            umat_config.enable_dynamic_loading ? resolved_path : "",
            load_strategy,
            umat_config.function_name);

        return umat_model;
    }
    // Handle other material types...
    return nullptr;
}

MultiExaModel::MultiExaModel(std::shared_ptr<SimulationState> sim_state, const ExaOptions& options)
    : ExaModel(-1, 0, sim_state) // Region -1, n_state_vars computed later
{
    CALI_CXX_MARK_SCOPE("composite_model_construction");

    // The construction is now beautifully simple because SimulationState
    // already handles all the complex region management for us

    m_num_regions = sim_state->GetNumberOfRegions();

    // Create specialized models for each region
    CreateChildModels(options);

    // Update our base class state variable count to reflect the maximum needed
    // across all child models. This ensures compatibility with existing interfaces.
    int max_state_vars = 0;
    for (const auto& child : m_child_models) {
        max_state_vars = std::max(max_state_vars, child->num_state_vars);
    }
    num_state_vars = max_state_vars;
}

void MultiExaModel::CreateChildModels(const ExaOptions& options) {
    // Create specialized material models for each region
    // SimulationState already knows about regions, so we just create models

    m_child_models.reserve(options.materials.size());

    for (size_t region_idx = 0; region_idx < options.materials.size(); ++region_idx) {
        const auto& material = options.materials[region_idx];
        const int region_id = static_cast<int>(region_idx);
        if (!m_sim_state->IsRegionActive(region_id)) {
            if (material.mech_type == MechType::EXACMECH) {
                std::string model_name = material.model.exacmech ? material.model.exacmech->shortcut
                                                                 : "";
                ECMechSetupQuadratureFuncStatePair(region_id, model_name, m_sim_state);
            }
            continue;
        }
        // Create the appropriate model type based on material specification
        std::unique_ptr<ExaModel> child_model;

        if (material.mech_type == MechType::UMAT) {
            // Create UMAT model for this region
            child_model = CreateMaterialModel(options.materials[region_idx], m_sim_state);
        } else if (material.mech_type == MechType::EXACMECH) {
            // Create ExaCMech model for this region

            // Determine execution strategy based on global solver settings
            ecmech::ExecutionStrategy accel = ecmech::ExecutionStrategy::CPU;
            if (options.solvers.rtmodel == RTModel::OPENMP) {
                accel = ecmech::ExecutionStrategy::OPENMP;
            } else if (options.solvers.rtmodel == RTModel::GPU) {
                accel = ecmech::ExecutionStrategy::GPU;
            }

            // Extract the material model name from the options
            std::string model_name = material.model.exacmech ? material.model.exacmech->shortcut
                                                             : "";

            child_model = std::make_unique<ExaCMechModel>(
                region_idx,                   // Region this model handles
                material.state_vars.num_vars, // State variables
                material.temperature,         // Operating temperature
                accel,                        // Execution strategy (CPU/GPU)
                model_name,                   // ExaCMech model type
                m_sim_state                   // Shared simulation state
            );
        } else {
            throw std::runtime_error("Unknown material type for region " +
                                     std::to_string(region_idx));
        }

        if (!child_model) {
            throw std::runtime_error("Failed to create material model for region " +
                                     std::to_string(region_idx));
        }
        m_child_models.push_back(std::move(child_model));
    }
}

void MultiExaModel::ModelSetup(const int nqpts,
                               const int nelems,
                               const int space_dim,
                               const int nnodes,
                               const mfem::Vector& jacobian,
                               const mfem::Vector& loc_grad,
                               const mfem::Vector& vel) {
    CALI_CXX_MARK_SCOPE("composite_model_setup");

    m_sim_state->SetupModelVariables();

    // This is now incredibly simple because SimulationState handles all the complexity!
    // Each child model automatically gets the right data for its region through SimulationState

    // Process each region - each child model operates on its own region's data
    std::vector<bool> region_success(m_child_models.size());

    for (size_t region_idx = 0; region_idx < m_child_models.size(); ++region_idx) {
        region_success[region_idx] = SetupChildModel(
            region_idx, nqpts, nelems, space_dim, nnodes, jacobian, loc_grad, vel);
    }

    // Verify that all regions completed successfully across all MPI processes
    if (!ValidateAllRegionsSucceeded(region_success)) {
        throw std::runtime_error("One or more material regions failed during setup");
    }

    // No need for explicit result aggregation - SimulationState handles this automatically
    // through the PartialQuadratureFunction system when child models write their results
}

bool MultiExaModel::SetupChildModel(size_t region_idx,
                                    const int nqpts,
                                    const int nelems,
                                    const int space_dim,
                                    const int nnodes,
                                    const mfem::Vector& jacobian,
                                    const mfem::Vector& loc_grad,
                                    const mfem::Vector& vel) const {
    CALI_CXX_MARK_SCOPE("composite_setup_child");
    const int actual_region_id = m_child_models[region_idx]->GetRegionID();
    try {
        // The beauty of this design: we just call the child model with the region index
        // SimulationState automatically routes the right data to the right model!
        auto& child_model = m_child_models[region_idx];

        // The child model uses its region_idx to get region-specific data from SimulationState
        // This is much cleaner than manually extracting and routing data
        child_model->ModelSetup(nqpts, nelems, space_dim, nnodes, jacobian, loc_grad, vel);

        return true;
    } catch (const std::exception& e) {
        MFEM_WARNING_0("[Cycle " << m_sim_state->GetSimulationCycle() + 1 << "] Region "
                                 << actual_region_id << " failed: " << e.what());
        return false;
    } catch (...) {
        MFEM_WARNING_0("Region " << actual_region_id << " failed with unknown error");
        return false;
    }
}

bool MultiExaModel::ValidateAllRegionsSucceeded(const std::vector<bool>& region_success) const {
    // Check if all regions succeeded on this processor
    bool local_success = std::all_of(
        region_success.begin(), region_success.end(), [](bool success) {
            return success;
        });

    // Use MPI collective operation to ensure global consistency
    bool global_success = false;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    return global_success;
}

void MultiExaModel::UpdateModelVars() {
    // Coordinate state variable updates across all child models
    for (auto& child : m_child_models) {
        child->UpdateModelVars();
    }
}

// Utility methods for external access
ExaModel* MultiExaModel::GetChildModel(int region_idx) const {
    if (region_idx < 0 || region_idx >= static_cast<int>(m_child_models.size())) {
        return nullptr;
    }
    return m_child_models[static_cast<size_t>(region_idx)].get();
}