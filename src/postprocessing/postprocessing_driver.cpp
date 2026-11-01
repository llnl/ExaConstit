#include "postprocessing/postprocessing_driver.hpp"

#include "postprocessing/mechanics_lightup.hpp"
#include "postprocessing/postprocessing_file_manager.hpp"
#include "postprocessing/projection_class.hpp"
#include "postprocessing/surface_quantities.hpp"
#include "postprocessing/volume_statistics.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/rotations.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem/linalg/kernels.hpp"

#include "ECMech_const.h"
#include "SNLS_linalg.h"

#include <algorithm>
#include <filesystem>
#include <sstream>
namespace fs = std::filesystem;

namespace {

/**
 * @brief Generic registration template for projection types
 *
 * @tparam T Projection class type to register
 * @param region_model_types Vector of material model types per region
 * @return Vector of shared projection instances, one per region plus global
 *
 * Creates projection instances for all regions plus one additional
 * global instance. Each projection is wrapped in a shared_ptr for
 * efficient memory management and polymorphic behavior.
 *
 * The template design enables type-safe registration of any
 * projection class derived from ProjectionBase.
 */
template <class T>
std::vector<std::shared_ptr<ProjectionBase>>
RegisterGeneric(const std::vector<MechType>& region_model_types) {
    std::vector<std::shared_ptr<ProjectionBase>> base;
    const size_t num_regions = region_model_types.size() + 1;
    for (size_t i = 0; i < num_regions; i++) {
        base.emplace_back(std::make_shared<T>());
    }
    return base;
}

/**
 * @brief Register centroid projections for all regions
 *
 * @param region_model_types Vector of material model types per region
 * @return Vector of CentroidProjection instances
 *
 * Creates centroid projection instances that compute geometric
 * centroids of mesh elements. Compatible with all material model
 * types as it depends only on mesh geometry.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterCentroid(const std::vector<MechType>& region_model_types) {
    return RegisterGeneric<CentroidProjection>(region_model_types);
}

/**
 * @brief Register volume projections for all regions
 *
 * @param region_model_types Vector of material model types per region
 * @return Vector of VolumeProjection instances
 *
 * Creates volume projection instances that compute element volumes
 * from integration of geometric determinants. Provides essential
 * geometric information for visualization and volume averaging.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterVolume(const std::vector<MechType>& region_model_types) {
    return RegisterGeneric<VolumeProjection>(region_model_types);
}

/**
 * @brief Register Cauchy stress projections for all regions
 *
 * @param region_model_types Vector of material model types per region
 * @return Vector of CauchyStressProjection instances
 *
 * Creates projections for full Cauchy stress tensor (6 components
 * in Voigt notation). Compatible with all material models that
 * provide stress state information.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterCauchyStress(const std::vector<MechType>& region_model_types) {
    return RegisterGeneric<CauchyStressProjection>(region_model_types);
}

/**
 * @brief Register Von Mises stress projections for all regions
 *
 * @param region_model_types Vector of material model types per region
 * @return Vector of VonMisesStressProjection instances
 *
 * Creates projections that compute Von Mises equivalent stress
 * from the Cauchy stress tensor. Provides scalar stress measure
 * commonly used for yield and failure analysis.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterVMStress(const std::vector<MechType>& region_model_types) {
    return RegisterGeneric<VonMisesStressProjection>(region_model_types);
}

/**
 * @brief Register hydrostatic stress projections for all regions
 *
 * @param region_model_types Vector of material model types per region
 * @return Vector of HydrostaticStressProjection instances
 *
 * Creates projections that compute hydrostatic (mean) stress
 * component. Essential for analyzing volumetric deformation
 * and pressure-dependent material behavior.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterHydroStress(const std::vector<MechType>& region_model_types) {
    return RegisterGeneric<HydrostaticStressProjection>(region_model_types);
}

/**
 * @brief Register all state variables projections for all regions
 *
 * @param sim_state Reference to simulation state (for the options)
 * @param region_model_types Vector of material model types per region
 * @return Vector of AllStateVariablesProjection instances
 *
 * Creates projections that output all available state variables
 * for debugging and detailed analysis. State variable count and
 * interpretation depend on the specific material model.
 *
 * The vector size must be known here, before the grid functions are created, and must be the
 * same on every rank (ranks without elements of a region still create its grid function), so it
 * is taken from the material options rather than from a quadrature function.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterAllState(const std::shared_ptr<SimulationState> sim_state,
                 const std::vector<MechType>& region_model_types) {
    std::vector<std::shared_ptr<ProjectionBase>> base;
    const auto& materials = sim_state->GetOptions().materials;
    for (size_t i = 0; i < region_model_types.size(); i++) {
        base.emplace_back(
            std::make_shared<AllStateVariablesProjection>(materials[i].state_vars.num_vars));
    }
    return base;
}

/**
 * @brief Generic registration template for ECMech-specific projections
 *
 * @tparam T ECMech projection class type
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @param key State variable key name for ECMech lookup
 * @return Vector of ECMech projection instances
 *
 * Creates ECMech-specific projections with automatic state variable
 * index resolution. Non-ECMech regions receive dummy projections
 * with invalid indices. The maximum state variable length across
 * all regions is tracked for consistent vector dimensions.
 */
template <class T>
std::vector<std::shared_ptr<ProjectionBase>>
RegisterECMech(const std::shared_ptr<SimulationState> sim_state,
               const std::vector<MechType>& region_model_types,
               const std::string key,
               const std::string display_name) {
    std::vector<std::shared_ptr<ProjectionBase>> base;
    const size_t num_regions = region_model_types.size();
    int max_length = -1;
    for (size_t i = 0; i < num_regions; i++) {
        if (region_model_types[i] != MechType::EXACMECH) {
            // Need to do a basic guard against non-ecmech models
            base.emplace_back(std::make_shared<T>("", -1, -1, display_name));
            continue;
        }
        auto [index, length] = sim_state->GetQuadratureFunctionStatePair(key, static_cast<int>(i));
        base.emplace_back(std::make_shared<T>(key, index, length, display_name));
        max_length = (max_length < length) ? length : max_length;
    }

    if (base[0]->CanAggregateGlobally()) {
        base.emplace_back(std::make_shared<T>(key, 0, max_length, display_name));
    }

    return base;
}

/**
 * @brief Register DpEff (effective plastic strain rate) projections for ExaCMech
 *
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of DpEffProjection instances
 *
 * Creates DpEffProjection instances for regions with ExaCMech material models.
 * Uses the "eq_pl_strain_rate" state variable key to access effective plastic
 * strain rate data. Non-ExaCMech regions receive dummy projections.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterDpEffProjection(const std::shared_ptr<SimulationState> sim_state,
                        const std::vector<MechType>& region_model_types) {
    std::string key = "eq_pl_strain_rate";
    std::string display_name = "Equivalent Plastic Strain Rate";
    return RegisterECMech<NNegStateProjection>(sim_state, region_model_types, key, display_name);
}

/**
 * @brief Register EPS (effective plastic strain) projections for ExaCMech
 *
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of EPSProjection instances
 *
 * Creates EPSProjection instances for regions with ExaCMech material models.
 * Uses the "eq_pl_strain" state variable key to access effective plastic
 * strain data. Non-ExaCMech regions receive dummy projections.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterEPSProjection(const std::shared_ptr<SimulationState> sim_state,
                      const std::vector<MechType>& region_model_types) {
    std::string key = "eq_pl_strain";
    std::string display_name = "Equivalent Plastic Strain";
    return RegisterECMech<NNegStateProjection>(sim_state, region_model_types, key, display_name);
}

/**
 * @brief Register crystal orientation projections for ExaCMech
 *
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of XtalOrientationProjection instances
 *
 * Creates crystal orientation projection instances using the "quats" state
 * variable key to access quaternion orientation data. Only compatible with
 * ExaCMech material models that provide crystal orientation information.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterXtalOriProjection(const std::shared_ptr<SimulationState> sim_state,
                          const std::vector<MechType>& region_model_types) {
    std::string key = "quats";
    std::string display_name = "Crystal Orientations";
    return RegisterECMech<XtalOrientationProjection>(
        sim_state, region_model_types, key, display_name);
}

/**
 * @brief Register elastic strain projections for ExaCMech
 *
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of ElasticStrainProjection instances
 *
 * Creates elastic strain projection instances using the "elastic_strain" state
 * variable key. Handles coordinate transformations and tensor reconstruction
 * for ExaCMech elastic strain data.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterElasticStrainProjection(const std::shared_ptr<SimulationState> sim_state,
                                const std::vector<MechType>& region_model_types) {
    std::string key = "elastic_strain";
    std::string display_name = "Elastic Strains";
    return RegisterECMech<ElasticStrainProjection>(
        sim_state, region_model_types, key, display_name);
}

/**
 * @brief Register hardness projections for ExaCMech
 *
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of HardnessProjection instances
 *
 * Creates hardness projection instances using the "hardness" state variable
 * key. Includes post-processing to ensure non-negative hardness values
 * suitable for visualization and analysis.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterHardnessProjection(const std::shared_ptr<SimulationState> sim_state,
                           const std::vector<MechType>& region_model_types) {
    std::string key = "hardness";
    std::string display_name = "Hardness";
    return RegisterECMech<NNegStateProjection>(sim_state, region_model_types, key, display_name);
}

/**
 * @brief Register shear rate projections for ExaCMech
 *
 * @param sim_state Reference to simulation state for state variable queries
 * @param region_model_types Vector of material model types per region
 * @return Vector of ShearingRateProjection instances
 *
 * Creates shear rate projection instances using the "shear_rate" state
 * variable key. Provides access to macroscopic shear rate data for
 * rate-dependent analysis and deformation characterization.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterShearRateProjection(const std::shared_ptr<SimulationState> sim_state,
                            const std::vector<MechType>& region_model_types) {
    std::string key = "shear_rate";
    std::string display_name = "Shearing Rate";
    return RegisterECMech<ShearingRateProjection>(sim_state, region_model_types, key, display_name);
}

/**
 * @brief List the state variables of one region to write as visualization fields
 *
 * @param sim_state Simulation state (state variable names and the material options)
 * @param region Region index (material index, 0-based)
 * @param selection Names to write, or the single entry "all"
 * @return One entry per field: (field name, (first slot, number of slots)) in the region's
 *         state variable vector, in output order
 *
 * With "all": every named state variable of the region whose plot flag is set (see
 * SimulationState::AddQuadratureFunctionStatePair) in the order it was registered. Slots of
 * names registered with plot = false are not written. For models other than ExaCMech this is
 * followed by one scalar field "State Var <k>" for each slot k (0-based) that no name covers, so
 * the whole state variable vector is written: ExaConstit cannot tell which of those slots are worth
 * plotting. ExaCMech names every quantity worth plotting itself, so its unnamed slots (for example
 * the count of function evaluations) are not written. With a list: the listed names in the list's
 * order (a name listed twice is written once). An unknown name stops the run with a message listing
 * the names the region has.
 *
 * Only the registered names and the material's state variable count are used, both of which
 * are the same on every rank, so all ranks get the same list even for regions that have no
 * elements on them.
 */
std::vector<std::pair<std::string, std::pair<int, int>>> StateVariableFieldList(
    const SimulationState& sim_state, const int region, const std::vector<std::string>& selection) {
    const auto named = sim_state.GetQuadratureFunctionStatePairs(region);
    std::vector<std::pair<std::string, std::pair<int, int>>> fields;

    if (selection.size() == 1 && selection[0] == "all") {
        // Every named state variable first, marking the slots each one uses
        const int num_vars =
            sim_state.GetOptions().materials[static_cast<size_t>(region)].state_vars.num_vars;
        std::vector<bool> covered(static_cast<size_t>(std::max(num_vars, 0)), false);
        for (const auto& entry : named) {
            if (sim_state.GetQuadratureFunctionStatePlot(entry.first, region)) {
                fields.push_back(entry);
            }
            const int offset = entry.second.first;
            const int length = entry.second.second;
            for (int k = std::max(offset, 0); k < std::min(offset + length, num_vars); k++) {
                covered[static_cast<size_t>(k)] = true;
            }
        }
        // Then one scalar field per slot without a name, except for ExaCMech, whose names
        // already cover everything worth plotting
        if (sim_state.GetRegionModelType(static_cast<size_t>(region)) == MechType::EXACMECH) {
            return fields;
        }
        for (int k = 0; k < num_vars; k++) {
            if (!covered[static_cast<size_t>(k)]) {
                fields.push_back({"State Var " + std::to_string(k), {k, 1}});
            }
        }
        return fields;
    }

    for (const auto& name : selection) {
        // Skip a name the user listed more than once
        const auto is_name = [&name](const auto& entry) {
            return entry.first == name;
        };
        if (std::find_if(fields.begin(), fields.end(), is_name) != fields.end()) {
            continue;
        }
        const auto match = std::find_if(named.begin(), named.end(), is_name);
        if (match == named.end()) {
            std::ostringstream available;
            for (const auto& entry : named) {
                available << " \"" << entry.first << "\"";
            }
            const std::string available_names = named.empty() ? std::string(" (none)")
                                                              : available.str();
            MFEM_ABORT_0("visualize_state_vars of material " << region + 1 << " lists \"" << name
                                                             << "\", which is not a state variable "
                                                                "name of this material. Available:"
                                                             << available_names);
        }
        fields.push_back(*match);
    }
    return fields;
}
} // namespace

void PostProcessingDriver::RegisterProjection(const std::string& field) {
    std::vector<std::shared_ptr<ProjectionBase>> projection_class;

    if (field == "centroid") {
        projection_class = RegisterCentroid(m_region_model_types);
    } else if (field == "volume") {
        projection_class = RegisterVolume(m_region_model_types);
    } else if (field == "cauchy") {
        projection_class = RegisterCauchyStress(m_region_model_types);
    } else if (field == "von_mises") {
        projection_class = RegisterVMStress(m_region_model_types);
    } else if (field == "hydro") {
        projection_class = RegisterHydroStress(m_region_model_types);
    } else if (field == "all_state") {
        projection_class = RegisterAllState(m_sim_state, m_region_model_types);
    } else if (field == "dpeff") {
        projection_class = RegisterDpEffProjection(m_sim_state, m_region_model_types);
    } else if (field == "eps") {
        projection_class = RegisterEPSProjection(m_sim_state, m_region_model_types);
    } else if (field == "xtal_ori") {
        projection_class = RegisterXtalOriProjection(m_sim_state, m_region_model_types);
    } else if (field == "elastic_strain") {
        projection_class = RegisterElasticStrainProjection(m_sim_state, m_region_model_types);
    } else if (field == "hardness") {
        projection_class = RegisterHardnessProjection(m_sim_state, m_region_model_types);
    } else if (field == "shear_rate") {
        projection_class = RegisterShearRateProjection(m_sim_state, m_region_model_types);
    } else {
        // region_<N> entries are handled by RegisterStateVariableProjections()
        if (ProjectionOptions::region_number(field) < 0) {
            MFEM_WARNING_0("[PostProcessing.projections] enabled_projections entry '"
                           << field
                           << "' is not a known projection and is ignored. Known: region_<N>, "
                              "centroid, volume, cauchy, von_mises, hydro, all_state, dpeff, eps, "
                              "xtal_ori, elastic_strain, hardness, shear_rate");
        }
        return;
    }

    std::string field_name = field;
    std::string display_name = projection_class[0]->GetDisplayName();
    using PTMC = ProjectionTraits::ModelCompatibility;
    PTMC model_compatibility = projection_class[0]->model;
    bool supports_global_aggregation = projection_class[0]->CanAggregateGlobally();

    std::vector<bool> region_enabled;
    std::vector<int> region_length;

    for (size_t i = 0; i < m_region_model_types.size(); i++) {
        const auto model = m_region_model_types[i];
        const auto project_model = projection_class[i]->model;
        region_length.push_back(projection_class[i]->GetVectorDimension());
        if (project_model == PTMC::EXACMECH_ONLY && model == MechType::EXACMECH) {
            region_enabled.push_back(true);
        } else if (project_model == PTMC::EXACMECH_ONLY && model == MechType::UMAT) {
            region_enabled.push_back(false);
        } else if (project_model == PTMC::UMAT_ONLY && model == MechType::EXACMECH) {
            region_enabled.push_back(false);
        } else if (project_model == PTMC::UMAT_ONLY && model == MechType::UMAT) {
            region_enabled.push_back(true);
        } else if (project_model == PTMC::ALL_MODELS) {
            region_enabled.push_back(true);
        } else {
            region_enabled.push_back(false);
        }
    }
    if (supports_global_aggregation) {
        region_enabled.push_back(true);
        region_length.push_back(
            projection_class[m_region_model_types.size()]->GetVectorDimension());
    }

    // Register the projection
    m_registered_projections.push_back({field_name,
                                        display_name,
                                        model_compatibility,
                                        region_enabled,
                                        projection_class,
                                        region_length,
                                        supports_global_aggregation});
}

PostProcessingDriver::PostProcessingDriver(std::shared_ptr<SimulationState> sim_state,
                                           ExaOptions& options)
    : m_sim_state(sim_state), m_mpi_rank(0), m_num_regions(sim_state->GetNumberOfRegions()),
      m_aggregation_mode(AggregationMode::BOTH),
      m_viz_location(options.visualization.field_location),
      m_enable_visualization(options.visualization.visit || options.visualization.conduit ||
                             options.visualization.paraview || options.visualization.adios2),
      m_viz_output_frequency(options.visualization.output_frequency) {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_mpi_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &m_num_mpi_rank);

    // Initialize file manager with proper ExaOptions handling
    m_file_manager = std::make_unique<PostProcessingFileManager>(options);

    // Ensure output directory exists
    if (!m_file_manager->EnsureOutputDirectoryExists()) {
        MFEM_WARNING_0("Failed to create output directory. Volume averaging may fail.");
    }

    // Initialize region-specific data structures
    m_region_model_types.resize(m_num_regions);
    m_region_evec.resize(m_num_regions);

    int max_vdim = 0;
    // Get model types for each region
    for (size_t region = 0; region < m_num_regions; ++region) {
        m_region_model_types[region] = sim_state->GetRegionModelType(region);
        // Initialize region-specific element average buffer
        if (auto pqf = sim_state->GetQuadratureFunction("cauchy_stress_end",
                                                        static_cast<int>(region))) {
            // Find maximum vdim across all possible quadrature functions for this region
            for (const auto& field_name :
                 {"cauchy_stress_end", "state_var_end", "von_mises", "kinetic_grads"}) {
                if (auto qf = sim_state->GetQuadratureFunction(field_name,
                                                               static_cast<int>(region))) {
                    max_vdim = std::max(max_vdim, qf->GetVDim());
                }
            }
            // Create element average buffer with maximum dimension needed
            m_region_evec[region] = std::make_unique<mfem::expt::PartialQuadratureFunction>(
                pqf->GetPartialSpaceShared(), max_vdim);
        }
    }

    // Initialize global element average buffer
    auto fe_space = sim_state->GetMeshParFiniteElementSpace();
    int global_max_vdim = max_vdim; // Accommodate stress tensors and other multi-component fields
    m_global_evec = std::make_unique<mfem::Vector>(global_max_vdim * fe_space->GetNE());
    m_global_evec->UseDevice(true);

    // Register default projections and volume calculations
    RegisterDefaultVolumeCalculations();
    SetupUserFunctions();
    SetupVolumeStatistics();

    // Initialize grid functions and data collections
    if (m_enable_visualization) {
        auto mesh = m_sim_state->GetMesh();

        // Local element of each region's partial quadrature space -> element of the mesh the
        // region is visualized on (the full mesh for one region, its submesh otherwise).
        std::map<int, mfem::Array<int>> map_pqs2submesh;
        if (m_num_regions == 1) {
            auto l2g = sim_state->GetQuadratureFunction("cauchy_stress_end", 0)
                           ->GetPartialSpaceShared()
                           ->GetLocal2Global();
            mfem::Array<int> pqs2submesh(l2g);
            map_pqs2submesh.emplace(0, std::move(pqs2submesh));
            m_map_submesh.emplace(0, mesh);
        } else {
            for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
                mfem::Array<int> domain(1);
                domain[0] = region + 1;
                auto submesh = mfem::ParSubMesh::CreateFromDomain(*mesh.get(), domain);
                auto submesh_ptr = std::make_shared<mfem::ParSubMesh>(std::move(submesh));
                m_map_submesh.emplace(region, std::move(submesh_ptr));

                if (!m_sim_state->IsRegionActive(region)) {
                    continue;
                }

                auto pqs = sim_state->GetQuadratureFunction("cauchy_stress_end", region)
                               ->GetPartialSpaceShared();
                auto l2g = pqs->GetLocal2Global();
                mfem::Array<int> pqs2submesh(l2g.Size());
                for (int i = 0; i < l2g.Size(); i++) {
                    const int mapping = dynamic_cast<mfem::ParSubMesh*>(m_map_submesh[region].get())
                                            ->GetSubMeshElementFromParent(l2g[i]);
                    pqs2submesh[i] = mapping;
                }
                map_pqs2submesh.emplace(region, std::move(pqs2submesh));
            }
        }

        // Build each region's (source quadrature point, output node) table. Zone output reads
        // the element averages ("_avg", one point per element); quadrature output reads the
        // end-of-step quadrature values ("_end") and picks the nearest point for each node.
        const std::string source_qf_name = (m_viz_location == VizFieldLocation::QUADRATURE)
                                               ? "cauchy_stress_end"
                                               : "cauchy_stress_avg";
        for (auto& [region, pqs2submesh] : map_pqs2submesh) {
            auto source_qspace =
                sim_state->GetQuadratureFunction(source_qf_name, region)->GetPartialSpaceShared();
            auto scalar_fes = GetParFiniteElementSpace(region, 1);
            m_map_viz_points.emplace(
                region, VizPointMap(m_viz_location, *source_qspace, *scalar_fes, pqs2submesh));
        }

        RegisterDefaultProjections();
        InitializeGridFunctions();
        InitializeDataCollections(options);
    }

    InitializeLightUpAnalysis();

    if (options.post_processing.surface_quantities.enabled) {
        m_surface_quantities = std::make_unique<SurfaceQuantities>(m_sim_state, options);
    }
}

std::shared_ptr<mfem::ParFiniteElementSpace>
PostProcessingDriver::GetParFiniteElementSpace(const int region, const int vdim) {
    if (!m_enable_visualization) {
        return std::shared_ptr<mfem::ParFiniteElementSpace>();
    }

    if (m_map_pfes.find(region) == m_map_pfes.end()) {
        m_map_pfes.emplace(region, std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>>());
    }

    if (m_map_pfes[region].find(vdim) == m_map_pfes[region].end()) {
        auto mesh = (region < 0) ? m_sim_state->GetMesh() : m_map_submesh[region];
        const int space_dim = mesh->SpaceDimension();
        // Zone output: order 0. Quadrature output: Gauss-Lobatto L2 of the mesh order (both
        // collections are owned by SimulationState).
        const int order = (m_viz_location == VizFieldLocation::QUADRATURE)
                              ? m_sim_state->GetOptions().mesh.order
                              : 0;
        std::string l2_fec_str = "L2_" + std::to_string(space_dim) + "D_P" + std::to_string(order);
        auto l2_fec = m_sim_state->GetFiniteElementCollection(l2_fec_str);
        auto value = std::make_shared<mfem::ParFiniteElementSpace>(
            mesh.get(), l2_fec.get(), vdim, mfem::Ordering::byVDIM);
        m_map_pfes[region].emplace(vdim, std::move(value));
    }
    return m_map_pfes[region][vdim];
}

void PostProcessingDriver::UpdateFields(const int step, [[maybe_unused]] const double time) {
    // Zone output reads element averages, so refresh them. Quadrature output reads the
    // end-of-step quadrature values directly and needs no averaging.
    for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
        if (!m_sim_state->IsRegionActive(region) || m_viz_location != VizFieldLocation::ZONE) {
            continue;
        }
        auto state_qf_avg = m_sim_state->GetQuadratureFunction("state_var_avg", region);
        auto state_qf_end = m_sim_state->GetQuadratureFunction("state_var_end", region);
        CalcElementAvg(state_qf_avg.get(), state_qf_end.get());
        auto cauchy_qf_avg = m_sim_state->GetQuadratureFunction("cauchy_stress_avg", region);
        auto cauchy_qf_end = m_sim_state->GetQuadratureFunction("cauchy_stress_end", region);
        CalcElementAvg(cauchy_qf_avg.get(), cauchy_qf_end.get());
    }

    // Post-processing user functions written as fields compute their values only at
    // visualization output steps (their fields are only written then)
    if (m_enable_visualization && ShouldOutputVisualizationAtStep(step)) {
        UpdateUserFunctionFields();
    }

    // Execute projections based on aggregation mode
    if (m_aggregation_mode == AggregationMode::PER_REGION ||
        m_aggregation_mode == AggregationMode::BOTH) {
        // Process each region separately
        for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
            if (!m_sim_state->IsRegionActive(region)) {
                continue;
            }
            // No table means visualization is off (nothing is registered to project)
            const auto points_it = m_map_viz_points.find(region);
            if (points_it == m_map_viz_points.end()) {
                continue;
            }
            const size_t reg_idx = static_cast<size_t>(region);
            const auto& points = points_it->second;
            for (auto& reg : m_registered_projections) {
                if (reg.region_enabled[reg_idx]) {
                    const auto gf_name = GetGridFunctionName(reg.display_name, region);
                    auto& grid_func = m_map_gfs[gf_name];
                    reg.projection_class[reg_idx]->Execute(m_sim_state, grid_func, points, region);
                }
            }
        }
    }

    if (m_aggregation_mode == AggregationMode::GLOBAL_COMBINED ||
        m_aggregation_mode == AggregationMode::BOTH) {
        // Execute global aggregated projections
        for (auto& reg : m_registered_projections) {
            if (reg.supports_global_aggregation) {
                ExecuteGlobalProjection(reg.display_name);
            }
        }
    }
}

void PostProcessingDriver::Update(const int step, const double time) {
    CALI_CXX_MARK_SCOPE("postprocessing_update");
    UpdateFields(step, time);

    // Update data collections for visualization
    if (ShouldOutputVisualizationAtStep(step) && m_enable_visualization) {
        UpdateDataCollections(step, time);
    }

    // Check if we should output volume averages at this step
    if (ShouldOutputAtStep(step)) {
        PrintVolValues(time, m_aggregation_mode);
    }
    ClearVolumeAverageCache();

    if (m_surface_quantities && m_surface_quantities->ShouldOutputAtStep(step)) {
        m_surface_quantities->Update(time, *m_file_manager);
    }

    if (m_light_up_instances.size() > 0) {
        UpdateLightUpAnalysis();
    }

    // HDF5 output: gather this step's rows to rank 0 and append them to the file
    m_file_manager->FinishOutputStep();
}

void PostProcessingDriver::ContinueAfterRestart(const double restart_time) {
    m_file_manager->ContinueAfterRestart(restart_time);
}

PostProcessingDriver::~PostProcessingDriver() = default;

void PostProcessingDriver::PrintVolValues(const double time, AggregationMode mode) {
    CALI_CXX_MARK_SCOPE("postprocessing_vol_values");

    if (mode == AggregationMode::PER_REGION || mode == AggregationMode::BOTH) {
        // Calculate per-region volume averages
        for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
            if (!m_sim_state->IsRegionActive(region)) {
                continue;
            }

            for (auto& reg : m_registered_volume_calcs) {
                if (reg.region_enabled[static_cast<size_t>(region)]) {
                    reg.region_func(region, time);
                }
            }
        }
    }

    if (mode == AggregationMode::GLOBAL_COMBINED || mode == AggregationMode::BOTH) {
        // Calculate global aggregated volume averages
        for (auto& reg : m_registered_volume_calcs) {
            if (reg.has_global_aggregation && reg.global_func) {
                reg.global_func(time);
            }
        }
    }
}

PostProcessingDriver::CalcType PostProcessingDriver::GetCalcType(const std::string& calc_type_str) {
    // Convert string identifiers to type-safe enums for internal processing
    if (calc_type_str == "stress") {
        return CalcType::STRESS;
    } else if (calc_type_str == "def_grad") {
        return CalcType::DEF_GRAD;
    } else if (calc_type_str == "plastic_work" || calc_type_str == "pl_work") {
        return CalcType::PLASTIC_WORK;
    } else if (calc_type_str == "eq_pl_strain" || calc_type_str == "eps") {
        return CalcType::EQ_PL_STRAIN;
    } else if (calc_type_str == "euler_strain") {
        return CalcType::EULER_STRAIN;
    } else if (calc_type_str == "elastic_strain" || calc_type_str == "estrain") {
        return CalcType::ELASTIC_STRAIN;
    } else {
        // Default fallback - could also throw an exception for strict validation
        MFEM_WARNING_0("Unknown calculation type '" << calc_type_str << "', defaulting to stress");
        return CalcType::STRESS;
    }
}

std::shared_ptr<mfem::expt::PartialQuadratureFunction>
PostProcessingDriver::PrepareVolumePoints(CalcType calc_type, int region, int& data_size) {
    data_size = 0;
    if (!m_sim_state->IsRegionActive(region)) {
        return nullptr;
    }

    std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf;
    std::string qf_name;
    const size_t reg_idx = static_cast<size_t>(region);

    // Configure calculation parameters based on type
    switch (calc_type) {
    case CalcType::STRESS:
        qf_name = "cauchy_stress_end";
        data_size = 6; // Voigt notation: Sxx, Syy, Szz, Syz, Sxz, Sxy
        break;

    case CalcType::DEF_GRAD:
        qf_name = "kinetic_grads";
        data_size = 9; // Full 3x3 tensor, column-major: F11, F21, F31, F12, ..., F33
        break;

    case CalcType::PLASTIC_WORK:
    case CalcType::EQ_PL_STRAIN:
        if (m_region_model_types[reg_idx] == MechType::UMAT) {
            return nullptr;
        }
        qf_name = "scalar";
        data_size = 1; // Scalar quantities
        break;

    case CalcType::EULER_STRAIN:
        qf_name = "kinetic_grads"; // Adjust this to your actual QF name for Euler strain
        data_size = 9;             // Voigt notation: E11, E22, E33, E23, E13, E12
        break;

    case CalcType::ELASTIC_STRAIN:
        if (m_region_model_types[reg_idx] == MechType::UMAT) {
            return nullptr;
        }
        qf_name = "kinetic_grads"; // Adjust this to your actual QF name for elastic strain
        data_size = 9;             // Voigt notation: Ee11, Ee22, Ee33, Ee23, Ee13, Ee12
        break;

    default:
        // This should never happen due to enum type safety, but defensive programming
        MFEM_ABORT_0("Unhandled calculation type in CalculateVolumeAverage");
        return nullptr;
    }

    // Get the quadrature function for this region
    qf = m_sim_state->GetQuadratureFunction(qf_name, region);
    if (!qf) {
        // Region doesn't have this quadrature function - return invalid data
        return nullptr;
    }

    // Handle calculation-specific preprocessing
    switch (calc_type) {
    case CalcType::EQ_PL_STRAIN: {
        // Extract equivalent plastic strain from state variables
        auto state_vars = m_sim_state->GetQuadratureFunction("state_var_end", region)->Read();
        const int vdim = m_sim_state->GetQuadratureFunction("state_var_end", region)->GetVDim();
        const int eps_ind =
            m_sim_state->GetQuadratureFunctionStatePair("eq_pl_strain", region).first;
        auto data = qf->Write();

        // Copy equivalent plastic strain values to scalar quadrature function
        mfem::forall(qf->Size(), [=] MFEM_HOST_DEVICE(int i) {
            data[i] = state_vars[i * vdim + eps_ind];
        });
        break;
    }

    case CalcType::PLASTIC_WORK: {
        // Extract plastic work from state variables
        auto state_vars = m_sim_state->GetQuadratureFunction("state_var_end", region)->Read();
        const int vdim = m_sim_state->GetQuadratureFunction("state_var_end", region)->GetVDim();

        // NOTE: You'll need to update this line to match your actual plastic work state variable
        // This is a placeholder - replace with your actual method to get plastic work index
        const int pl_work_ind =
            m_sim_state->GetQuadratureFunctionStatePair("plastic_work", region).first;
        auto data = qf->Write();

        // Copy plastic work values to scalar quadrature function
        mfem::forall(qf->Size(), [=] MFEM_HOST_DEVICE(int i) {
            data[i] = state_vars[i * vdim + pl_work_ind];
        });
        break;
    }

    case CalcType::EULER_STRAIN:
    case CalcType::DEF_GRAD: {
        // Special handling for deformation gradient - assign global values to region
        auto def_grad_global = m_sim_state->GetQuadratureFunction("kinetic_grads", -1);
        if (def_grad_global) {
            qf->operator=(0.0);
            qf->operator=(*dynamic_cast<mfem::QuadratureFunction*>(def_grad_global.get()));
        }
        break;
    }
    case CalcType::ELASTIC_STRAIN: {
        auto state_vars = m_sim_state->GetQuadratureFunction("state_var_end", region)->Read();
        const int vdim = m_sim_state->GetQuadratureFunction("state_var_end", region)->GetVDim();
        // One value per quadrature point (not per element): state_vars and data both hold
        // every quadrature point of the region
        const int npts = m_sim_state->GetQuadratureFunction("state_var_end", region)->Size() / vdim;
        const int estrain_ind =
            m_sim_state->GetQuadratureFunctionStatePair("elastic_strain", region).first;
        const int quats_ind = m_sim_state->GetQuadratureFunctionStatePair("quats", region).first;
        const int rel_vol_ind =
            m_sim_state->GetQuadratureFunctionStatePair("relative_volume", region).first;
        qf->operator=(0.0);
        auto data = qf->Write();

        mfem::forall(npts, [=] MFEM_HOST_DEVICE(int i) {
            const auto strain_lat = &state_vars[i * vdim + estrain_ind];
            const auto quats = &state_vars[i * vdim + quats_ind];
            const auto rel_vol = state_vars[i * vdim + rel_vol_ind];
            double* strain = &data[i * 9];

            {
                double strainm[3 * 3] = {};
                double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
                const double t1 = ecmech::sqr2i * strain_lat[0];
                const double t2 = ecmech::sqr6i * strain_lat[1];
                //
                // Volume strain is ln(V^e_mean) term aka ln(relative volume)
                // Our plastic deformation has a det(1) aka no change in volume change
                const double elas_vol_strain = log(rel_vol);
                // We output elastic strain formulation such that the relationship
                // between V^e and \varepsilon is just V^e = I + \varepsilon
                strain_m[0][0] = (t1 - t2) + elas_vol_strain;                      // 11
                strain_m[1][1] = (-t1 - t2) + elas_vol_strain;                     // 22
                strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
                strain_m[1][2] = ecmech::sqr2i * strain_lat[4];                    // 23
                strain_m[2][0] = ecmech::sqr2i * strain_lat[3];                    // 31
                strain_m[0][1] = ecmech::sqr2i * strain_lat[2];                    // 12

                strain_m[2][1] = strain_m[1][2];
                strain_m[0][2] = strain_m[2][0];
                strain_m[1][0] = strain_m[0][1];

                double rmat[3 * 3] = {};
                double strain_samp[3 * 3] = {};

                Quat2RMat(quats, rmat);
                snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

                strain_m[0] = &strain_samp[0];
                strain_m[1] = &strain_samp[3];
                strain_m[2] = &strain_samp[6];
                strain[0] = strain_m[0][0];
                strain[1] = strain_m[1][1];
                strain[2] = strain_m[2][2];
                strain[3] = strain_m[1][2];
                strain[4] = strain_m[0][2];
                strain[5] = strain_m[0][1];
                strain[6] = 0.0;
                strain[7] = 0.0;
                strain[8] = 0.0;
            }
        });
        break;
    }

    case CalcType::STRESS:
    default:
        // No special preprocessing needed for these types
        // The quadrature function already contains the correct data
        break;
    }
    return qf;
}

PostProcessingDriver::VolumeAverageData
PostProcessingDriver::CalculateVolumeAverage(CalcType calc_type, int region) {
    // Per-point values of the quantity on the region (null if it has none on this rank)
    int data_size = 0;
    const auto qf = PrepareVolumePoints(calc_type, region, data_size);
    if (!qf) {
        return VolumeAverageData();
    }

    // Perform the volume integration to compute average
    mfem::Vector avg_data(data_size);
    double total_volume = 0.0;

    auto region_comm = m_sim_state->GetRegionCommunicator(region);

    switch (calc_type) {
    case CalcType::PLASTIC_WORK: {
        total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<false>(
            qf.get(), avg_data, data_size, m_sim_state->GetOptions().solvers.rtmodel, region_comm);
        break;
    }
    default: {
        total_volume = exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
            qf.get(), avg_data, data_size, m_sim_state->GetOptions().solvers.rtmodel, region_comm);
        break;
    }
    }
    // Any post processing that might be needed
    switch (calc_type) {
    case CalcType::EULER_STRAIN: {
        mfem::Vector avg_euler_strain(6);
        {
            mfem::DenseMatrix euler_strain(3, 3);
            mfem::DenseMatrix def_grad(avg_data.HostReadWrite(), 3, 3);
            int dim = 3;
            mfem::DenseMatrix Finv(dim), Binv(dim);
            double half = 1.0 / 2.0;

            mfem::CalcInverse(def_grad, Finv);
            mfem::MultAtB(Finv, Finv, Binv);

            euler_strain = 0.0;

            for (int j = 0; j < dim; j++) {
                for (int i = 0; i < dim; i++) {
                    euler_strain(i, j) -= half * Binv(i, j);
                }

                euler_strain(j, j) += half;
            }
            avg_euler_strain(0) = euler_strain(0, 0);
            avg_euler_strain(1) = euler_strain(1, 1);
            avg_euler_strain(2) = euler_strain(2, 2);
            avg_euler_strain(3) = euler_strain(1, 2);
            avg_euler_strain(4) = euler_strain(0, 2);
            avg_euler_strain(5) = euler_strain(0, 1);
        }
        return VolumeAverageData(total_volume, avg_euler_strain);
        break;
    }
    case CalcType::ELASTIC_STRAIN: {
        avg_data.SetSize(6);
        break;
    }
    default:
        break;
    }
    // Return the calculated data
    return VolumeAverageData(total_volume, avg_data);
}

PostProcessingDriver::VolumeAverageData
PostProcessingDriver::GetOrCalculateVolumeAverage(CalcType calc_type, int region) {
    // First, check if we have cached data for this calculation type and region
    auto cache_it = m_region_cache.find(calc_type);
    if (cache_it != m_region_cache.end()) {
        auto region_it = cache_it->second.find(region);
        if (region_it != cache_it->second.end() && region_it->second.is_valid) {
            // Found valid cached data - return it immediately
            return region_it->second;
        }
    }

    // No cached data found - calculate it now
    auto result = CalculateVolumeAverage(calc_type, region);

    // Cache the result for future use (even if invalid, to avoid repeated failed attempts)
    m_region_cache[calc_type][region] = result;

    return result;
}

void PostProcessingDriver::ClearVolumeAverageCache() {
    // Clear all cached data at the beginning of each time step
    m_region_cache.clear();
    for (auto& entry : m_user_function_outputs) {
        entry.results.clear();
    }
    for (auto& output : m_statistics_outputs) {
        output.results.clear();
    }
}

void PostProcessingDriver::VolumeAverage(const std::string& calc_type_str,
                                         int region,
                                         double time) {
    if (region >= 0 && !m_sim_state->IsRegionActive(region)) {
        return;
    }
    // Convert string to enum for internal processing
    CalcType calc_type = GetCalcType(calc_type_str);

    // Calculate and cache the result
    auto result = GetOrCalculateVolumeAverage(calc_type, region);

    if (!result.is_valid) {
        // fix me
        // Calculation failed (e.g., missing quadrature function) - skip output
        // Could maybe add a warning for this so people are aware of which materials
        // didn't have a valid calculation but only do it once per simulation...
        return;
    }

    // Write output using the file manager
    auto region_name = m_sim_state->GetRegionName(region);
    auto region_comm = m_sim_state->GetRegionCommunicator(region);
    if (result.data.Size() == 1) {
        // Scalar quantity
        m_file_manager->WriteVolumeAverage(calc_type_str,
                                           region,
                                           region_name,
                                           time,
                                           result.volume,
                                           result.data[0],
                                           1,
                                           region_comm);
    } else {
        // Vector/tensor quantity
        m_file_manager->WriteVolumeAverage(calc_type_str,
                                           region,
                                           region_name,
                                           time,
                                           result.volume,
                                           result.data,
                                           result.data.Size(),
                                           region_comm);
    }
}

void PostProcessingDriver::GlobalVolumeAverage(const std::string& calc_type_str, double time) {
    CalcType calc_type = GetCalcType(calc_type_str);

    // Determine expected data size for this calculation type
    int data_size = 1; // Default for scalar quantities
    switch (calc_type) {
    case CalcType::STRESS:
    case CalcType::EULER_STRAIN:
    case CalcType::ELASTIC_STRAIN:
        data_size = 6; // Tensor quantities in Voigt notation
        break;
    case CalcType::DEF_GRAD:
        data_size = 9; // Full 3x3 tensor
        break;
    case CalcType::PLASTIC_WORK:
    case CalcType::EQ_PL_STRAIN:
    default:
        data_size = 1; // Scalar quantities
        break;
    }

    // Gather every region's result on rank 0 and combine them over the whole sample
    // (volume-weighted average, or plain sum for the integrated plastic work)
    const bool average = (calc_type != CalcType::PLASTIC_WORK);
    const auto global = CombineRegionVolumeResults(
        [this, calc_type](const int region) {
            return GetOrCalculateVolumeAverage(calc_type, region);
        },
        data_size,
        average);
    if (!global.is_valid) {
        // No valid regions found - issue warning
        MFEM_WARNING_0("No valid regions found for global " << calc_type_str << " calculation");
        return;
    }

    // Write global output (region = -1 indicates global file)
    if (data_size == 1) {
        // Scalar quantity
        m_file_manager->WriteVolumeAverage(
            calc_type_str, -1, "", time, global.volume, global.data[0]);
    } else {
        // Vector/tensor quantity
        m_file_manager->WriteVolumeAverage(calc_type_str, -1, "", time, global.volume, global.data);
    }
}

std::vector<PostProcessingDriver::VolumeAverageData>
PostProcessingDriver::GatherRegionVolumeResults(
    const std::function<VolumeAverageData(int)>& region_result, const int data_size) {
    std::vector<VolumeAverageData> results(m_num_regions);
    for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
        // Collective on the region's communicator; ranks without the region get invalid data
        auto region_data = region_result(region);
        // An invalid result still takes part in the exchange below: send data_size zeros with
        // zero volume so it is skipped
        if (!region_data.is_valid || region_data.data.Size() != data_size) {
            region_data.data.SetSize(data_size);
            region_data.data = 0.0;
            region_data.volume = 0.0;
        }

        // Gather all region data to rank 0
        if (m_mpi_rank == 0) {
            // Rank 0 receives from all region roots
            const int root_rank = m_sim_state->GetRegionRootRank(region);
            if (root_rank > m_mpi_rank) {
                MPI_Recv(region_data.data.HostWrite(),
                         data_size,
                         MPI_DOUBLE,
                         root_rank,
                         region,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Recv(&region_data.volume,
                         1,
                         MPI_DOUBLE,
                         root_rank,
                         static_cast<int>(m_num_regions) * 2 + region,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        } else if (m_sim_state->IsRegionIORoot(region)) {
            // Other ranks send their region data if they're region roots
            MPI_Send(region_data.data.HostRead(), data_size, MPI_DOUBLE, 0, region, MPI_COMM_WORLD);
            MPI_Send(&region_data.volume,
                     1,
                     MPI_DOUBLE,
                     0,
                     static_cast<int>(m_num_regions) * 2 + region,
                     MPI_COMM_WORLD);
        }
        region_data.is_valid = (region_data.volume > 0.0);
        results[static_cast<size_t>(region)] = region_data;
    }
    return results;
}

PostProcessingDriver::VolumeAverageData PostProcessingDriver::CombineRegionVolumeResults(
    const std::function<VolumeAverageData(int)>& region_result,
    const int data_size,
    const bool average) {
    mfem::Vector global_data(data_size);
    global_data = 0.0;
    double global_volume = 0.0;

    for (auto& region_data : GatherRegionVolumeResults(region_result, data_size)) {
        if (region_data.volume > 0.0) {
            // Volume-weighted contribution (averages) or plain sum (integrals)
            region_data.data.HostRead();
            for (int i = 0; i < data_size; ++i) {
                const double value = region_data.data[i];
                global_data[i] += average ? value * region_data.volume : value;
            }
            global_volume += region_data.volume;
        }
    }

    if (global_volume <= 0.0) {
        return VolumeAverageData();
    }
    if (average) {
        global_data /= global_volume;
    }
    return VolumeAverageData(global_volume, global_data);
}

bool PostProcessingDriver::ShouldOutputAtStep(int step) const {
    return m_file_manager->ShouldOutputAtStep(step);
}

bool PostProcessingDriver::ShouldOutputVisualizationAtStep(int step) const {
    return (step % m_viz_output_frequency == 0);
}

void PostProcessingDriver::VolumeAvgStress(const int region, const double time) {
    VolumeAverage("stress", region, time);
}

void PostProcessingDriver::GlobalVolumeAvgStress(const double time) {
    GlobalVolumeAverage("stress", time);
}

void PostProcessingDriver::VolumeAvgDefGrad(const int region, const double time) {
    VolumeAverage("def_grad", region, time);
}

void PostProcessingDriver::GlobalVolumeAvgDefGrad(const double time) {
    GlobalVolumeAverage("def_grad", time);
}

void PostProcessingDriver::VolumeEPS(const int region, const double time) {
    VolumeAverage("eq_pl_strain", region, time);
}

void PostProcessingDriver::GlobalVolumeEPS(const double time) {
    GlobalVolumeAverage("eq_pl_strain", time);
}

void PostProcessingDriver::VolumePlWork(const int region, const double time) {
    VolumeAverage("plastic_work", region, time);
}

void PostProcessingDriver::GlobalVolumePlWork(const double time) {
    GlobalVolumeAverage("plastic_work", time);
}

void PostProcessingDriver::VolumeAvgEulerStrain(const int region, const double time) {
    VolumeAverage("euler_strain", region, time);
}

void PostProcessingDriver::GlobalVolumeAvgEulerStrain(const double time) {
    GlobalVolumeAverage("euler_strain", time);
}

void PostProcessingDriver::VolumeAvgElasticStrain(const int region, const double time) {
    VolumeAverage("elastic_strain", region, time);
}

void PostProcessingDriver::GlobalVolumeAvgElasticStrain(const double time) {
    GlobalVolumeAverage("elastic_strain", time);
}

void PostProcessingDriver::RegisterDefaultProjections() {
    const auto& projection_opts = m_sim_state->GetOptions().post_processing.projections;

    std::vector<std::string> fields;
    if (projection_opts.auto_enable_compatible) {
        std::vector<std::string> defaults = {"centroid",
                                             "volume",
                                             "cauchy",
                                             "von_mises",
                                             "hydro",
                                             "dpeff",
                                             "eps",
                                             "xtal_ori",
                                             "elastic_strain",
                                             "hardness",
                                             "shear_rate"};
        fields = defaults;
    } else if (projection_opts.enabled_projections.size() > 0) {
        fields = projection_opts.enabled_projections;
    }

    // region_<N> entries are not built-in names; RegisterProjection skips them and they are
    // handled below
    for (const auto& field : fields) {
        RegisterProjection(field);
    }

    // State variable fields requested per region (region_<N> entries and visualize_state_vars)
    // are added in every case, also when auto_enable_compatible picked the built-in list
    RegisterStateVariableProjections();

    // Fields from "visualization" post-processing user functions
    RegisterUserFunctionFields();
}

void PostProcessingDriver::RegisterStateVariableProjections() {
    const auto& options = m_sim_state->GetOptions();

    // What each region asks for: nothing (empty), {"all"}, or a list of names.
    // A region_<N> entry asks for everything of material N.
    std::vector<std::vector<std::string>> selection(m_num_regions);
    for (const auto& entry : options.post_processing.projections.enabled_projections) {
        const int number = ProjectionOptions::region_number(entry);
        if (number >= 1 && number <= static_cast<int>(m_num_regions)) {
            selection[static_cast<size_t>(number - 1)] = {"all"};
        }
    }
    // The material's own list is used unless region_<N> already asked for everything (which
    // includes every listed name, so nothing is created twice)
    for (size_t region = 0; region < m_num_regions && region < options.materials.size(); region++) {
        const auto& requested = options.materials[region].visualize_state_vars;
        const bool already_all = (selection[region].size() == 1 && selection[region][0] == "all");
        if (!requested.empty() && !already_all) {
            selection[region] = requested;
        }
    }

    using PTMC = ProjectionTraits::ModelCompatibility;
    for (size_t region = 0; region < m_num_regions; region++) {
        if (selection[region].empty()) {
            continue;
        }
        const int region_int = static_cast<int>(region);
        const auto fields = StateVariableFieldList(*m_sim_state, region_int, selection[region]);
        for (const auto& field : fields) {
            const std::string name = field.first;
            const int offset = field.second.first;
            const int length = field.second.second;

            // A field of the same name already written for this region (for example a user
            // state variable called "Hardness") would share its grid function; keep the first
            const bool name_taken = std::any_of(m_registered_projections.begin(),
                                                m_registered_projections.end(),
                                                [&name, region](const ProjectionRegistration& reg) {
                                                    return reg.display_name == name &&
                                                           reg.region_enabled[region];
                                                });
            if (name_taken) {
                MFEM_WARNING_0("State variable field \""
                               << name << "\" of material " << region + 1
                               << " has the same name as another visualization field of that "
                                  "region and is not written");
                continue;
            }

            // Copy slots [offset, offset + length) of the state variables unchanged. The
            // field is turned on for this region only and never combined into the global
            // output; the other regions' entries are placeholders that are never used.
            auto projection = std::make_shared<StateVariableProjection>(
                name, offset, length, name, PTMC::ALL_MODELS);
            std::vector<std::shared_ptr<ProjectionBase>> projection_class(m_num_regions,
                                                                          projection);
            std::vector<bool> region_enabled(m_num_regions, false);
            std::vector<int> region_length(m_num_regions, 0);
            region_enabled[region] = true;
            region_length[region] = length;

            m_registered_projections.push_back(
                {"state_var_region_" + std::to_string(region + 1) + "_" + name,
                 name,
                 PTMC::ALL_MODELS,
                 region_enabled,
                 projection_class,
                 region_length,
                 false});
        }
    }
}

void PostProcessingDriver::SetupUserFunctions() {
    const auto& entries = m_sim_state->GetOptions().post_processing.user_functions;
    m_user_function_outputs.reserve(entries.size());

    for (size_t index = 0; index < entries.size(); index++) {
        UserFunctionOutput entry;
        entry.options = entries[index];
        const auto& options = entry.options;

        // Selected regions (options use material numbers starting at 1); all when none given
        if (options.regions.empty()) {
            for (int region = 0; region < static_cast<int>(m_num_regions); region++) {
                entry.regions.push_back(region);
            }
        } else {
            for (const int material : options.regions) {
                entry.regions.push_back(material - 1);
            }
        }

        // Already loaded and checked at startup (CheckUserFunctions), so this comes from the cache
        if (options.factory.empty()) {
            entry.function = exaconstit::LoadUserFunction<exaconstit::PostProcessingFunction>(
                options.library_path,
                options.function_name,
                exaconstit::PostProcessingFunctionSignature(options.function_name),
                "post-processing user function of entry '" + options.name + "'");
        } else {
            entry.function = exaconstit::MakeUserPostProcessingCall(
                m_sim_state, index, options, entry.regions);
        }

        // Point values on each selected region this rank owns elements of, on the region's
        // quadrature points; for zone visualization also their element averages (one point per
        // element, the space of the "_avg" functions)
        const bool zone_fields = (options.type == UserFunctionType::VISUALIZATION &&
                                  m_viz_location == VizFieldLocation::ZONE);
        for (const int region : entry.regions) {
            if (!m_sim_state->IsRegionActive(region)) {
                continue;
            }
            const auto point_space = m_sim_state->GetQuadratureFunction("cauchy_stress_end", region)
                                         ->GetPartialSpaceShared();
            entry.values[region] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                point_space, options.num_components, 0.0);
            if (zone_fields) {
                const auto element_space = m_sim_state
                                               ->GetQuadratureFunction("cauchy_stress_avg", region)
                                               ->GetPartialSpaceShared();
                entry.averages[region] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    element_space, options.num_components, 0.0);
            }
        }
        m_user_function_outputs.push_back(std::move(entry));
    }

    // Volume outputs: one registration per entry, run only on its regions
    for (size_t index = 0; index < m_user_function_outputs.size(); index++) {
        const auto& entry = m_user_function_outputs[index];
        const auto& options = entry.options;
        if (options.type == UserFunctionType::VISUALIZATION) {
            if (!m_enable_visualization) {
                MFEM_WARNING_0("Post-processing user function entry '"
                               << options.name
                               << "' is of type visualization but no visualization format is "
                                  "enabled in [Visualizations]; it is not written");
            }
            continue;
        }

        // The output name picks the file names; it must not be one of the built-in averages
        static const std::vector<std::string> built_in = {"stress",
                                                          "def_grad",
                                                          "plastic_work",
                                                          "pl_work",
                                                          "eq_pl_strain",
                                                          "eps",
                                                          "euler_strain",
                                                          "elastic_strain",
                                                          "estrain"};
        MFEM_VERIFY_0(std::find(built_in.begin(), built_in.end(), options.name) == built_in.end(),
                      "Post-processing user function entry name '"
                          << options.name
                          << "' is the name of a built-in volume average; choose another name");
        m_file_manager->SetVolumeAverageColumnNames(options.name, options.component_names);

        std::function<void(double)> global_output = nullptr;
        if (options.global) {
            global_output = [this, index](const double time) {
                GlobalUserFunctionVolumeOutput(index, time);
            };
        }
        RegisterVolumeAverageFunction(
            options.name,
            options.name,
            [this, index](const int region, const double time) {
                UserFunctionVolumeOutput(index, region, time);
            },
            global_output,
            true);
        // Only the entry's regions produce a region file
        auto& region_enabled = m_registered_volume_calcs.back().region_enabled;
        std::fill(region_enabled.begin(), region_enabled.end(), false);
        for (const int region : entry.regions) {
            region_enabled[static_cast<size_t>(region)] = true;
        }
    }
}

void PostProcessingDriver::RegisterUserFunctionFields() {
    using PTMC = ProjectionTraits::ModelCompatibility;
    for (const auto& entry : m_user_function_outputs) {
        const auto& options = entry.options;
        if (options.type != UserFunctionType::VISUALIZATION) {
            continue;
        }
        // A field name already used would share that field's grid function
        const bool name_taken = std::any_of(m_registered_projections.begin(),
                                            m_registered_projections.end(),
                                            [&options](const ProjectionRegistration& reg) {
                                                return reg.display_name == options.name;
                                            });
        MFEM_VERIFY_0(!name_taken,
                      "Post-processing user function entry name '"
                          << options.name
                          << "' is already the name of another visualization field; choose "
                             "another name");

        // One projection per region holding that region's buffers (none where the region is
        // not selected or has no elements on this rank). The field exists, with the same size,
        // on every rank, as the data collections require.
        std::vector<std::shared_ptr<ProjectionBase>> projection_class;
        std::vector<bool> region_enabled(m_num_regions, false);
        std::vector<int> region_length(m_num_regions, options.num_components);
        for (size_t region = 0; region < m_num_regions; region++) {
            const int region_int = static_cast<int>(region);
            const auto values = entry.values.find(region_int);
            const auto averages = entry.averages.find(region_int);
            projection_class.push_back(std::make_shared<QuadratureFunctionProjection>(
                options.name,
                options.num_components,
                (values != entry.values.end()) ? values->second : nullptr,
                (averages != entry.averages.end()) ? averages->second : nullptr,
                options.global));
            region_enabled[region] = std::find(entry.regions.begin(),
                                               entry.regions.end(),
                                               region_int) != entry.regions.end();
        }

        m_registered_projections.push_back({"user_function_" + options.name,
                                            options.name,
                                            PTMC::ALL_MODELS,
                                            region_enabled,
                                            projection_class,
                                            region_length,
                                            options.global});
    }
}

void PostProcessingDriver::UpdateUserFunctionFields() {
    for (auto& entry : m_user_function_outputs) {
        if (entry.options.type != UserFunctionType::VISUALIZATION) {
            continue;
        }
        for (auto& [region, values] : entry.values) {
            entry.function(m_sim_state, region, values);
            // Zone output shows the element average of the values
            const auto averages = entry.averages.find(region);
            if (averages != entry.averages.end()) {
                CalcElementAvg(averages->second.get(), values.get());
            }
        }
    }
}

PostProcessingDriver::VolumeAverageData
PostProcessingDriver::UserFunctionVolumeResult(const size_t index, const int region) {
    auto& entry = m_user_function_outputs[index];
    const auto values = entry.values.find(region);
    if (values == entry.values.end()) {
        return VolumeAverageData(); // Region not selected, or no elements on this rank
    }
    // Computed once per output step (the cache is cleared with the built-in one)
    const auto cached = entry.results.find(region);
    if (cached != entry.results.end()) {
        return cached->second;
    }

    // The user's quantity at every point of the region, then its volume average or integral
    // (collective on the region's communicator)
    entry.function(m_sim_state, region, values->second);
    const int num_components = entry.options.num_components;
    mfem::Vector result(num_components);
    const auto rtmodel = m_sim_state->GetOptions().solvers.rtmodel;
    const auto region_comm = m_sim_state->GetRegionCommunicator(region);
    const double volume =
        (entry.options.type == UserFunctionType::VOLUME_AVERAGE)
            ? exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
                  values->second.get(), result, num_components, rtmodel, region_comm)
            : exaconstit::kernel::ComputeVolAvgTensorFromPartial<false>(
                  values->second.get(), result, num_components, rtmodel, region_comm);
    entry.results[region] = VolumeAverageData(volume, result);
    return entry.results[region];
}

void PostProcessingDriver::UserFunctionVolumeOutput(const size_t index,
                                                    const int region,
                                                    const double time) {
    const auto result = UserFunctionVolumeResult(index, region);
    if (!result.is_valid) {
        return;
    }
    const auto& options = m_user_function_outputs[index].options;
    m_file_manager->WriteVolumeAverage(options.name,
                                       region,
                                       m_sim_state->GetRegionName(region),
                                       time,
                                       result.volume,
                                       result.data,
                                       options.num_components,
                                       m_sim_state->GetRegionCommunicator(region));
}

void PostProcessingDriver::GlobalUserFunctionVolumeOutput(const size_t index, const double time) {
    const auto& entry = m_user_function_outputs[index];
    const bool average = (entry.options.type == UserFunctionType::VOLUME_AVERAGE);
    const auto global = CombineRegionVolumeResults(
        [this, index](const int region) {
            return UserFunctionVolumeResult(index, region);
        },
        entry.options.num_components,
        average);
    if (!global.is_valid) {
        return; // No selected region had any volume
    }
    m_file_manager->WriteVolumeAverage(
        entry.options.name, -1, "", time, global.volume, global.data, entry.options.num_components);
}

void PostProcessingDriver::SetupVolumeStatistics() {
    const auto& vol_opts = m_sim_state->GetOptions().post_processing.volume_averages;

    // Built-in averages: statistics (one list for all) and per-quantity histograms of every
    // enabled average, by the names of their switches
    if (vol_opts.enabled) {
        const std::vector<std::pair<std::string, bool>> built_in = {
            {"stress", vol_opts.stress},
            {"def_grad", vol_opts.def_grad},
            {"euler_strain", vol_opts.euler_strain},
            {"plastic_work", vol_opts.plastic_work},
            {"eq_pl_strain", vol_opts.eq_pl_strain},
            {"elastic_strain", vol_opts.elastic_strain}};
        for (const auto& [name, enabled] : built_in) {
            const auto histogram = vol_opts.histograms.find(name);
            const HistogramOptions histogram_options = (histogram != vol_opts.histograms.end())
                                                           ? histogram->second
                                                           : HistogramOptions();
            if (!enabled || (vol_opts.statistics.empty() && histogram_options.bins == 0)) {
                continue;
            }
            const CalcType calc_type = GetCalcType(name);
            RegisterStatisticsOutput(name,
                                     m_file_manager->GetVolumeAverageColumnNames(name),
                                     vol_opts.statistics,
                                     histogram_options,
                                     std::vector<int>(),
                                     true,
                                     [this, calc_type](const int region) {
                                         return BuiltInStatisticsPoints(calc_type, region);
                                     });
        }
    }

    // "volume_average" user functions: the values the function filled for the volume average
    for (size_t index = 0; index < m_user_function_outputs.size(); index++) {
        const auto& options = m_user_function_outputs[index].options;
        if (options.statistics.empty() && options.histogram.bins == 0) {
            continue;
        }
        RegisterStatisticsOutput(options.name,
                                 options.component_names,
                                 options.statistics,
                                 options.histogram,
                                 m_user_function_outputs[index].regions,
                                 options.global,
                                 [this, index](const int region)
                                     -> std::shared_ptr<mfem::expt::PartialQuadratureFunction> {
                                     // Runs the user function for this output step if the average
                                     // did not already
                                     if (!UserFunctionVolumeResult(index, region).is_valid) {
                                         return nullptr;
                                     }
                                     return m_user_function_outputs[index].values.at(region);
                                 });
    }

    // All statistics files go to the stats folder of the results directory (text output only;
    // the HDF5 output keeps them in its one file)
    if (!m_statistics_outputs.empty() && !m_file_manager->WritesHDF5()) {
        auto stats_directory = m_file_manager->GetOutputDirectory() / "stats";
        m_file_manager->EnsureDirectoryExists(stats_directory, MPI_COMM_WORLD);
    }
}

void PostProcessingDriver::RegisterStatisticsOutput(
    const std::string& name,
    const std::vector<std::string>& component_names,
    const std::vector<std::string>& statistics,
    const HistogramOptions& histogram,
    const std::vector<int>& regions,
    const bool global,
    std::function<std::shared_ptr<mfem::expt::PartialQuadratureFunction>(int)> point_values) {
    const size_t index = m_statistics_outputs.size();
    StatisticsOutput output;
    output.name = name;
    output.num_components = static_cast<int>(component_names.size());
    output.component_names = component_names;
    output.statistics = statistics;
    output.histogram = histogram;
    output.point_values = std::move(point_values);
    m_statistics_outputs.push_back(std::move(output));

    // One file per statistic, stats/<name>_<statistic>_*.txt, with one column per component
    // (as in the average file), and one histogram file, stats/<name>_hist_*.txt, with every
    // component's slots. The keys ("stats:...", "hist:...") only identify the files in the
    // file manager and cannot clash with an average's name.
    for (const auto& statistic : statistics) {
        const std::string key = "stats:" + name + ":" + statistic;
        m_file_manager->SetVolumeAverageFileLocation(key, name + "_" + statistic, "stats");
        m_file_manager->SetVolumeAverageColumnNames(key, component_names);
    }
    if (histogram.bins > 0) {
        const std::string key = "hist:" + name;
        m_file_manager->SetVolumeAverageFileLocation(key, name + "_hist", "stats");
        m_file_manager->SetVolumeAverageColumnNames(
            key, exaconstit::HistogramColumnNames(component_names, histogram.bins));
        m_file_manager->SetVolumeAverageHeaderComment(key,
                                                      exaconstit::HistogramEdgesComment(histogram));
        m_file_manager->SetVolumeAverageBinEdges(key, exaconstit::HistogramEdges(histogram));
    }

    std::function<void(double)> global_output = nullptr;
    if (global) {
        global_output = [this, index](const double time) {
            StatisticsGlobalOutput(index, time);
        };
    }
    RegisterVolumeAverageFunction(
        "statistics:" + name,
        "Volume statistics of " + name,
        [this, index](const int region, const double time) {
            StatisticsRegionOutput(index, region, time);
        },
        global_output,
        true);
    // Only the selected regions produce region files (all when none are given)
    if (!regions.empty()) {
        auto& region_enabled = m_registered_volume_calcs.back().region_enabled;
        std::fill(region_enabled.begin(), region_enabled.end(), false);
        for (const int region : regions) {
            region_enabled[static_cast<size_t>(region)] = true;
        }
    }
}

std::shared_ptr<mfem::expt::PartialQuadratureFunction>
PostProcessingDriver::BuiltInStatisticsPoints(const CalcType calc_type, const int region) {
    int data_size = 0;
    const auto values = PrepareVolumePoints(calc_type, region, data_size);
    if (!values) {
        return nullptr;
    }
    const int num_points = values->Size() / data_size;

    if (calc_type == CalcType::ELASTIC_STRAIN) {
        // The first 6 of the 9 values per point hold the strain (Voigt)
        auto strain = std::make_shared<mfem::expt::PartialQuadratureFunction>(
            values->GetPartialSpaceShared(), 6, 0.0);
        const auto source = mfem::Reshape(values->Read(), data_size, num_points);
        auto target = mfem::Reshape(strain->Write(), 6, num_points);
        mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int ip) {
            for (int i = 0; i < 6; i++) {
                target(i, ip) = source(i, ip);
            }
        });
        return strain;
    }
    if (calc_type == CalcType::EULER_STRAIN) {
        // Euler strain of every point, E = (I - F^-T F^-1) / 2, from its deformation gradient
        // (the euler_strain volume average instead converts the averaged deformation gradient)
        auto strain = std::make_shared<mfem::expt::PartialQuadratureFunction>(
            values->GetPartialSpaceShared(), 6, 0.0);
        const auto def_grad = mfem::Reshape(values->Read(), 3, 3, num_points);
        auto target = mfem::Reshape(strain->Write(), 6, num_points);
        mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int ip) {
            double f_data[9];
            double f_inv_data[9];
            for (int j = 0; j < 3; j++) {
                for (int i = 0; i < 3; i++) {
                    f_data[i + 3 * j] = def_grad(i, j, ip);
                }
            }
            mfem::kernels::CalcInverse<3>(f_data, f_inv_data);
            const auto f_inv = mfem::Reshape(f_inv_data, 3, 3);
            // b_inv = F^-T F^-1
            double b_inv[3][3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    b_inv[i][j] = 0.0;
                    for (int k = 0; k < 3; k++) {
                        b_inv[i][j] += f_inv(k, i) * f_inv(k, j);
                    }
                }
            }
            target(0, ip) = 0.5 * (1.0 - b_inv[0][0]);
            target(1, ip) = 0.5 * (1.0 - b_inv[1][1]);
            target(2, ip) = 0.5 * (1.0 - b_inv[2][2]);
            target(3, ip) = -0.5 * b_inv[1][2];
            target(4, ip) = -0.5 * b_inv[0][2];
            target(5, ip) = -0.5 * b_inv[0][1];
        });
        return strain;
    }
    return values;
}

PostProcessingDriver::VolumeAverageData PostProcessingDriver::StatisticsResult(const size_t index,
                                                                               const int region) {
    auto& output = m_statistics_outputs[index];
    const auto cached = output.results.find(region);
    if (cached != output.results.end()) {
        return cached->second;
    }
    if (!m_sim_state->IsRegionActive(region)) {
        return VolumeAverageData();
    }
    const auto values = output.point_values(region);
    if (!values) {
        output.results[region] = VolumeAverageData();
        return output.results[region];
    }
    mfem::Vector packed;
    const double volume = exaconstit::ComputeRegionStatistics(
        *values,
        output.histogram,
        m_sim_state->GetOptions().solvers.rtmodel,
        m_sim_state->GetRegionCommunicator(region),
        packed);
    output.results[region] = VolumeAverageData(volume, packed);
    return output.results[region];
}

void PostProcessingDriver::StatisticsRegionOutput(const size_t index,
                                                  const int region,
                                                  const double time) {
    const auto result = StatisticsResult(index, region);
    if (!result.is_valid) {
        return;
    }
    const auto& output = m_statistics_outputs[index];
    WriteStatisticsRows(output,
                        result,
                        region,
                        m_sim_state->GetRegionName(region),
                        time,
                        m_sim_state->GetRegionCommunicator(region));
}

void PostProcessingDriver::StatisticsGlobalOutput(const size_t index, const double time) {
    const auto& output = m_statistics_outputs[index];
    const int data_size = exaconstit::StatisticsDataSize(output.num_components,
                                                         output.histogram.bins);
    const auto regions = GatherRegionVolumeResults(
        [this, index](const int region) {
            return StatisticsResult(index, region);
        },
        data_size);
    if (m_mpi_rank != 0) {
        return;
    }
    std::vector<double> volumes;
    std::vector<mfem::Vector> packed;
    for (const auto& region : regions) {
        volumes.push_back(region.volume);
        packed.push_back(region.data);
    }
    mfem::Vector combined;
    const double volume = exaconstit::CombineStatistics(
        volumes, packed, output.num_components, output.histogram.bins, combined);
    if (volume <= 0.0) {
        return;
    }
    WriteStatisticsRows(output, VolumeAverageData(volume, combined), -1, "", time, MPI_COMM_SELF);
}

void PostProcessingDriver::WriteStatisticsRows(const StatisticsOutput& output,
                                               const VolumeAverageData& result,
                                               const int region,
                                               const std::string& region_name,
                                               const double time,
                                               MPI_Comm comm) {
    // One file per statistic: the statistic of every component
    for (const auto& statistic : output.statistics) {
        const auto row = exaconstit::StatisticRow(
            result.data, result.volume, output.num_components, statistic);
        m_file_manager->WriteVolumeAverage("stats:" + output.name + ":" + statistic,
                                           region,
                                           region_name,
                                           time,
                                           result.volume,
                                           row,
                                           row.Size(),
                                           comm);
    }
    // One histogram file with every component
    if (output.histogram.bins > 0) {
        const auto row = exaconstit::HistogramRow(
            result.data, result.volume, output.num_components, output.histogram.bins);
        m_file_manager->WriteVolumeAverage(
            "hist:" + output.name, region, region_name, time, result.volume, row, row.Size(), comm);
    }
}

void PostProcessingDriver::RegisterDefaultVolumeCalculations() {
    // Register volume average calculations with both per-region and global variants
    // Only register if the corresponding option is enabled in ExaOptions

    const auto& vol_opts = m_sim_state->GetOptions().post_processing.volume_averages;

    if (vol_opts.enabled && vol_opts.stress) {
        RegisterVolumeAverageFunction(
            "stress",
            "Volume Average Stress",
            [this](int region, double time) {
                VolumeAvgStress(region, time);
            },
            [this](double time) {
                GlobalVolumeAvgStress(time);
            },
            true);
    }

    if (vol_opts.enabled && vol_opts.def_grad) {
        RegisterVolumeAverageFunction(
            "def_grad",
            "Volume Average Deformation Gradient",
            [this](int region, double time) {
                VolumeAvgDefGrad(region, time);
            },
            [this](double time) {
                GlobalVolumeAvgDefGrad(time);
            },
            true);
    }

    if (vol_opts.enabled && vol_opts.euler_strain) {
        RegisterVolumeAverageFunction(
            "euler_strain",
            "Volume Average Euler Strain",
            [this](int region, double time) {
                VolumeAvgEulerStrain(region, time);
            },
            [this](double time) {
                GlobalVolumeAvgEulerStrain(time);
            },
            true);
    }

    if (vol_opts.enabled && vol_opts.plastic_work) {
        RegisterVolumeAverageFunction(
            "plastic_work",
            "Volume Plastic Work",
            [this](int region, double time) {
                VolumePlWork(region, time);
            },
            [this](double time) {
                GlobalVolumePlWork(time);
            },
            true);
    }

    if (vol_opts.enabled && vol_opts.eq_pl_strain) {
        RegisterVolumeAverageFunction(
            "equivalent_plastic_strain",
            "Volume Equivalent Plastic Strain",
            [this](int region, double time) {
                VolumeEPS(region, time);
            },
            [this](double time) {
                GlobalVolumeEPS(time);
            },
            true);
    }

    if (vol_opts.enabled && vol_opts.elastic_strain) {
        RegisterVolumeAverageFunction(
            "elastic_strain",
            "Volume Average Elastic Strain",
            [this](int region, double time) {
                VolumeAvgElasticStrain(region, time);
            },
            [this](double time) {
                GlobalVolumeAvgElasticStrain(time);
            },
            true);
    }
}

void PostProcessingDriver::RegisterVolumeAverageFunction(
    const std::string& calc_name,
    const std::string& display_name,
    std::function<void(const int, const double)> region_func,
    std::function<void(const double)> global_func,
    bool enabled) {
    std::vector<bool> region_enabled(m_num_regions, enabled);

    m_registered_volume_calcs.push_back(
        {calc_name,
         display_name,
         ProjectionTraits::ModelCompatibility::ALL_MODELS, // Default compatibility
         region_enabled,
         region_func,
         global_func,
         (global_func != nullptr)});
}

bool PostProcessingDriver::RegionHasQuadratureFunction(const std::string& field_name,
                                                       int region) const {
    return m_sim_state->GetQuadratureFunction(field_name, region) != nullptr;
}

std::vector<int>
PostProcessingDriver::GetActiveRegionsForField(const std::string& field_name) const {
    std::vector<int> active_regions;

    auto find_lambda = [&](const int region) -> bool {
        const auto gf_name = this->GetGridFunctionName(field_name, region);
        return (this->m_map_gfs.find(gf_name) != this->m_map_gfs.end()) &&
               (m_sim_state->IsRegionActive(region));
    };

    for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
        active_regions.push_back(find_lambda(region));
    }
    return active_regions;
}

std::string PostProcessingDriver::GetGridFunctionName(const std::string& field_name,
                                                      int region) const {
    return field_name + " " + m_sim_state->GetRegionDisplayName(region);
}

void PostProcessingDriver::ExecuteGlobalProjection(const std::string& field_name) {
    if (m_num_regions == 1) {
        return;
    }
    // Get all active regions for this field
    auto active_regions = GetActiveRegionsForField(field_name);
    if (active_regions.empty()) {
        return;
    }
    // Combine region data into global grid function
    CombineRegionDataToGlobal(field_name);
}

void PostProcessingDriver::CombineRegionDataToGlobal(const std::string& field_name) {
    auto global_gf_name = GetGridFunctionName(field_name, -1); // -1 indicates global
    auto& global_gf = *m_map_gfs[global_gf_name];

    // Initialize global grid function to zero
    global_gf = 0.0;

    // Get active regions for this field
    auto active_regions = GetActiveRegionsForField(field_name);

    int index = 0;
    for (const auto active : active_regions) {
        if (active) {
            auto submesh = std::dynamic_pointer_cast<mfem::ParSubMesh>(m_map_submesh[index]);
            if (submesh) {
                auto gf_name = GetGridFunctionName(field_name, index); // -1 indicates global
                auto& gf = *m_map_gfs[gf_name];
                submesh->Transfer(gf, global_gf);
            }
        }
        index += 1;
    }
}

void PostProcessingDriver::CalcElementAvg(mfem::expt::PartialQuadratureFunction* elemVal,
                                          const mfem::expt::PartialQuadratureFunction* qf) {
    CALI_CXX_MARK_SCOPE("calc_element_avg_partial");

    auto pqs = qf->GetPartialSpaceShared();
    auto mesh = pqs->GetMeshShared();
    const mfem::FiniteElement& el = *m_sim_state->GetMeshParFiniteElementSpace()->GetTypicalFE();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    const int nqpts = ir->GetNPoints();
    const int vdim = qf->GetVDim();
    const int NE = pqs->GetNE(); // Number of elements in this PARTIAL space (key difference!)

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS);

    // KEY DIFFERENCE: Get the local-to-global element mapping for partial space
    auto l2g = pqs->GetLocal2Global().Read(); // Maps local element index to global element index
    auto loc_offsets =
        pqs->Offsets(mfem::QSpaceOffsetStorage::COMPRESSED).Read(); // Offsets for local data layout
    // auto global_offsets = (pqs->GetGlobalOffset().Size() > 1) ?
    //                        pqs->GetGlobalOffset().Read() : loc_offsets; // Offsets for global
    //                        data layout

    auto qf_data = qf->Read(); // Partial quadrature function data (only for this region!)
    auto elem_data = elemVal->ReadWrite(); // Element averages output (only for this region!)
    auto j_data = geom->detJ.Read();       // Global geometric factors

    // Zero out element averages
    *elemVal = 0.0;

    // KEY DIFFERENCE: Process only the elements that exist in this partial space
    // The old version processed ALL elements (0 to nelems-1)
    // The new version processes only local elements (0 to NE-1) and maps to global indices
    mfem::forall(NE, [=] MFEM_HOST_DEVICE(int ie) {
        const int global_elem = l2g[ie];          // Map local element to global element
        const int local_offset = loc_offsets[ie]; // Offset into local data array
        // const int global_offset = global_offsets[global_elem]; // Offset into global layout
        const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element

        double vol = 0.0;

        // Calculate volume and weighted sum using actual quadrature points for this element
        for (int iq = 0; iq < npts_elem; ++iq) {
            // Use global element index for geometric factors (j_data)
            const double wt = j_data[global_elem * nqpts + iq] * W[iq];
            vol += wt;

            for (int iv = 0; iv < vdim; ++iv) {
                // Use local data layout for quadrature function values
                const int local_idx = local_offset * vdim + iq * vdim + iv;
                const double val = qf_data[local_idx];

                // Store in local element index (ie, not global_elem!)
                elem_data[ie * vdim + iv] += val * wt;
            }
        }

        // Normalize by volume to get element average
        const double inv_vol = 1.0 / vol;
        for (int iv = 0; iv < vdim; ++iv) {
            elem_data[ie * vdim + iv] *= inv_vol;
        }
    });
}

void PostProcessingDriver::CalcGlobalElementAvg(mfem::Vector* elemVal,
                                                const std::string& field_name) {
    CALI_CXX_MARK_SCOPE("calc_global_element_avg");

    auto fe_space = m_sim_state->GetMeshParFiniteElementSpace();
    const int nelems = fe_space->GetNE();

    // Find the vector dimension by checking the first available region
    int vdim = 1;
    for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
        if (auto pqf = m_sim_state->GetQuadratureFunction(field_name, region)) {
            if (vdim < pqf->GetVDim()) {
                vdim = pqf->GetVDim();
            }
        }
    }

    // Ensure elemVal is sized correctly
    if (elemVal->Size() != vdim * nelems) {
        elemVal->SetSize(vdim * nelems);
        elemVal->UseDevice(true);
    }

    // Initialize to zero
    *elemVal = 0.0;
    double* global_data = elemVal->ReadWrite();

    // Accumulate from all regions
    for (size_t region = 0; region < m_num_regions; ++region) {
        auto pqf = m_sim_state->GetQuadratureFunction(field_name, static_cast<int>(region));
        if (!pqf || !m_region_evec[region]) {
            continue;
        }

        // Calculate element averages for this region
        CalcElementAvg(m_region_evec[region].get(), pqf.get());

        // Add this region's contribution to global averages
        auto pqs = pqf->GetPartialSpaceShared();
        auto l2g = pqs->GetLocal2Global().Read();
        auto region_data = m_region_evec[region]->Read();
        const int NE_region = pqs->GetNE();
        const int local_vdim = pqf->GetVDim();

        mfem::forall(NE_region, [=] MFEM_HOST_DEVICE(int ie) {
            const int global_elem = l2g[ie];
            for (int iv = 0; iv < local_vdim; ++iv) {
                global_data[global_elem * vdim + iv] = region_data[ie * local_vdim + iv];
            }
        });
    }
}

void PostProcessingDriver::InitializeGridFunctions() {
    for (auto& reg : m_registered_projections) {
        // Create per-region grid functions
        int max_vdim = 0;
        if (m_aggregation_mode == AggregationMode::PER_REGION ||
            m_aggregation_mode == AggregationMode::BOTH) {
            for (size_t region = 0; region < m_num_regions; ++region) {
                const int reg_int = static_cast<int>(region);
                if (reg.region_enabled[region]) {
                    const auto gf_name = GetGridFunctionName(reg.display_name, reg_int);
                    // Determine vector dimension from quadrature function
                    const int vdim = reg.region_length[region];
                    if (vdim < 1) {
                        continue;
                    }
                    max_vdim = (vdim > max_vdim) ? vdim : max_vdim;
                    auto fe_space = GetParFiniteElementSpace(reg_int, vdim);
                    m_map_gfs.emplace(gf_name,
                                      std::make_shared<mfem::ParGridFunction>(fe_space.get()));
                    m_map_gfs[gf_name]->operator=(0.0);
                }
            }
        }
        // Create global grid functions
        if (reg.supports_global_aggregation &&
            (m_aggregation_mode == AggregationMode::GLOBAL_COMBINED ||
             m_aggregation_mode == AggregationMode::BOTH) &&
            (m_num_regions > 1)) {
            if (max_vdim < 1) {
                for (size_t region = 0; region < m_num_regions; ++region) {
                    if (reg.region_enabled[region]) {
                        const auto gf_name = GetGridFunctionName(reg.display_name,
                                                                 static_cast<int>(region));
                        // Determine vector dimension from quadrature function
                        const int vdim = reg.region_length[region];
                        max_vdim = (vdim > max_vdim) ? vdim : max_vdim;
                    }
                }
            }

            auto gf_name = GetGridFunctionName(reg.display_name, -1);
            auto fe_space = GetParFiniteElementSpace(-1, max_vdim);
            m_map_gfs.emplace(gf_name, std::make_shared<mfem::ParGridFunction>(fe_space.get()));
            m_map_gfs[gf_name]->operator=(0.0);
        }
    }

    if (m_aggregation_mode == AggregationMode::PER_REGION ||
        m_aggregation_mode == AggregationMode::BOTH) {
        if (m_num_regions == 1) {
            auto disp_gf_name = GetGridFunctionName("Displacement", 0);
            auto vel_gf_name = GetGridFunctionName("Velocity", 0);
            auto grain_gf_name = GetGridFunctionName("Grain ID", 0);
            m_map_gfs.emplace(disp_gf_name, m_sim_state->GetDisplacement());
            m_map_gfs.emplace(vel_gf_name, m_sim_state->GetVelocity());
            m_map_gfs.emplace(grain_gf_name, m_sim_state->GetGrains());
        }
    }

    if ((m_aggregation_mode == AggregationMode::GLOBAL_COMBINED ||
         m_aggregation_mode == AggregationMode::BOTH) &&
        (m_num_regions > 1)) {
        auto disp_gf_name = GetGridFunctionName("Displacement", -1);
        auto vel_gf_name = GetGridFunctionName("Velocity", -1);
        auto grain_gf_name = GetGridFunctionName("Grain ID", -1);
        m_map_gfs.emplace(disp_gf_name, m_sim_state->GetDisplacement());
        m_map_gfs.emplace(vel_gf_name, m_sim_state->GetVelocity());
        m_map_gfs.emplace(grain_gf_name, m_sim_state->GetGrains());
    }

    UpdateFields(static_cast<int>(m_sim_state->GetSimulationCycle()), m_sim_state->GetTime());
}

void PostProcessingDriver::InitializeDataCollections(ExaOptions& options) {
    auto output_dir_base = m_file_manager->GetVizDirectory();
    std::string visit_key = "visit_";
    std::string paraview_key = "paraview_";
#if defined(MFEM_USE_ADIOS2)
    std::string adios2_key = "adios2_";
#endif

    auto data_collection_name = [](const std::string& input, const std::string& delimiter) {
        auto pos = input.find(delimiter);
        if (pos == std::string::npos) {
            return input; // Delimiter not found, return entire string
        }
        return input.substr(0, pos);
    };

    auto has_registered_fields = [this](const std::string& display_region_postfix) {
        for (const auto& [key, value] : m_map_gfs) {
            (void)value;
            if (key.find(display_region_postfix) != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    if (m_aggregation_mode == AggregationMode::PER_REGION ||
        m_aggregation_mode == AggregationMode::BOTH) {
        for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
            auto mesh = m_map_submesh[region];
            std::string region_postfix = "region_" + std::to_string(region + 1);
            std::string display_region_postfix = " " + m_sim_state->GetRegionDisplayName(region);
            if (!has_registered_fields(display_region_postfix)) {
                continue;
            }
            fs::path output_dir = output_dir_base / region_postfix;
            fs::path output_dir_vizs = output_dir / m_file_manager->GetBaseFilename();
            // The subsequent DataCollection::Save() is a parallel operation on the submesh's
            // communicator, which is still the parent MPI communicator. Prepare directories on
            // that same communicator so all participating ranks observe the same path state.
            m_file_manager->EnsureDirectoryExists(output_dir, MPI_COMM_WORLD);
            std::vector<std::string> dcs_keys;
            if (options.visualization.visit) {
                std::string key = visit_key + region_postfix;
                m_map_dcs.emplace(key,
                                  std::make_unique<mfem::VisItDataCollection>(
                                      output_dir_vizs.string(), mesh.get()));
                m_map_dcs[key]->SetPrecision(10);
                dcs_keys.push_back(key);
            }
            if (options.visualization.paraview) {
                std::string key = paraview_key + region_postfix;
                m_map_dcs.emplace(key,
                                  std::make_unique<mfem::ParaViewDataCollection>(
                                      output_dir_vizs.string(), mesh.get()));
                auto& paraview = *(
                    dynamic_cast<mfem::ParaViewDataCollection*>(m_map_dcs[key].get()));
                paraview.SetLevelsOfDetail(options.mesh.order);
                paraview.SetDataFormat(mfem::VTKFormat::BINARY);
                paraview.SetHighOrderOutput(false);
                dcs_keys.push_back(key);
            }
#ifdef MFEM_USE_ADIOS2
            if (options.visualization.adios2) {
                const std::string basename = output_dir_vizs.string() + ".bp";
                std::string key = adios2_key + region_postfix;
                m_map_dcs.emplace(key,
                                  std::make_unique<mfem::ADIOS2DataCollection>(
                                      MPI_COMM_WORLD, basename, mesh.get()));
                auto& adios2 = *(dynamic_cast<mfem::ADIOS2DataCollection*>(m_map_dcs[key].get()));
                adios2.SetParameter("SubStreams", std::to_string(m_num_mpi_rank / 2));
                dcs_keys.push_back(key);
            }
#endif
            for (auto& dcs_key : dcs_keys) {
                auto& dcs = m_map_dcs[dcs_key];
                for (auto& [key, value] : m_map_gfs) {
                    if (key.find(display_region_postfix) != std::string::npos) {
                        std::string disp_key = data_collection_name(key, display_region_postfix);
                        dcs->RegisterField(disp_key, value.get());
                    }
                }
                dcs->SetCycle(0);
                dcs->SetTime(0.0);
                dcs->Save();
            }
        }
    }

    if ((m_aggregation_mode == AggregationMode::GLOBAL_COMBINED ||
         m_aggregation_mode == AggregationMode::BOTH) &&
        (m_num_regions > 1)) {
        auto mesh = m_sim_state->GetMesh();

        std::string region_postfix = "global";
        std::string display_region_postfix = " " + m_sim_state->GetRegionDisplayName(-1);
        if (!has_registered_fields(display_region_postfix)) {
            return;
        }
        fs::path output_dir = output_dir_base / region_postfix;
        fs::path output_dir_vizs = output_dir / m_file_manager->GetBaseFilename();
        m_file_manager->EnsureDirectoryExists(output_dir);
        std::vector<std::string> dcs_keys;
        if (options.visualization.visit) {
            std::string key = visit_key + region_postfix;
            m_map_dcs.emplace(
                key,
                std::make_unique<mfem::VisItDataCollection>(output_dir_vizs.string(), mesh.get()));
            m_map_dcs[key]->SetPrecision(10);
            dcs_keys.push_back(key);
        }
        if (options.visualization.paraview) {
            std::string key = paraview_key + region_postfix;
            m_map_dcs.emplace(key,
                              std::make_unique<mfem::ParaViewDataCollection>(
                                  output_dir_vizs.string(), mesh.get()));
            auto& paraview = *(dynamic_cast<mfem::ParaViewDataCollection*>(m_map_dcs[key].get()));
            paraview.SetLevelsOfDetail(options.mesh.order);
            paraview.SetDataFormat(mfem::VTKFormat::BINARY);
            paraview.SetHighOrderOutput(false);
            dcs_keys.push_back(key);
        }
#ifdef MFEM_USE_ADIOS2
        if (options.visualization.adios2) {
            const std::string basename = output_dir_vizs.string() + ".bp";
            std::string key = adios2_key + region_postfix;
            m_map_dcs.emplace(
                key,
                std::make_unique<mfem::ADIOS2DataCollection>(MPI_COMM_WORLD, basename, mesh.get()));
            auto& adios2 = *(dynamic_cast<mfem::ADIOS2DataCollection*>(m_map_dcs[key].get()));
            adios2.SetParameter("SubStreams", std::to_string(m_num_mpi_rank / 2));
            dcs_keys.push_back(key);
        }
#endif

        for (auto& dcs_key : dcs_keys) {
            auto& dcs = m_map_dcs[dcs_key];
            for (auto& [key, value] : m_map_gfs) {
                if (key.find(display_region_postfix) != std::string::npos) {
                    std::string disp_key = data_collection_name(key, display_region_postfix);
                    dcs->RegisterField(disp_key, value.get());
                }
            }
            dcs->SetCycle(0);
            dcs->SetTime(0.0);
            dcs->Save();
        }
    }
}

void PostProcessingDriver::UpdateDataCollections(const int step, const double time) {
    for (auto& [tmp, dcs] : m_map_dcs) {
        dcs->SetCycle(step);
        dcs->SetTime(time);
        dcs->Save();
    }
}

void PostProcessingDriver::InitializeLightUpAnalysis() {
    auto options = m_sim_state->GetOptions();
    // Clear any existing instances
    m_light_up_instances.clear();

    // Get enabled light_up configurations
    auto enabled_configs = options.post_processing.get_enabled_light_up_configs();

    if (!enabled_configs.empty() && m_mpi_rank == 0) {
        std::cout << "Initializing LightUp analysis for " << enabled_configs.size()
                  << " material(s)" << std::endl;
    }

    // Create LightUp instance for each enabled configuration
    for (const auto& light_config : enabled_configs) {
        MFEM_VERIFY_0(light_config.region_id.has_value(),
                      "LightUp config for material '" << light_config.material_name
                                                      << "' has unresolved region_id");

        int region_id = light_config.region_id.value() - 1;

        if (!m_sim_state->IsRegionActive(region_id)) {
            continue;
        }

        if (m_sim_state->IsRegionIORoot(region_id)) {
            std::cout << "  Creating LightUp for material '" << light_config.material_name
                      << "' (region " << region_id + 1 << ")" << std::endl;
        }

        fs::path lattice_base = light_config.lattice_basename;
        fs::path lattice_basename = m_file_manager->GetOutputDirectory() / lattice_base;

        auto light_up_instance = std::make_unique<LightUp>(
            light_config.hkl_directions,
            light_config.distance_tolerance,
            light_config.sample_direction,
            m_sim_state->GetQuadratureFunction("cauchy_stress_end", region_id)
                ->GetPartialSpaceShared(),
            m_sim_state,
            region_id, // Use the resolved region_id
            options.solvers.rtmodel,
            lattice_basename,
            light_config.lattice_parameters,
            light_config.lattice_type);

        m_light_up_instances.push_back(std::move(light_up_instance));
    }
}

void PostProcessingDriver::UpdateLightUpAnalysis() {
    // Update all LightUp instances
    for (auto& light_up : m_light_up_instances) {
        const int region_id = light_up->GetRegionID();

        auto state_vars = m_sim_state->GetQuadratureFunction("state_var_end", region_id);
        auto stress = m_sim_state->GetQuadratureFunction("cauchy_stress_end", region_id);

        light_up->CalculateLightUpData(state_vars, stress);
    }
}

void PostProcessingDriver::EnableProjection(const std::string& field_name,
                                            int region,
                                            bool enable) {
    for (auto& reg : m_registered_projections) {
        if (reg.field_name == field_name && region < static_cast<int>(reg.region_enabled.size())) {
            reg.region_enabled[static_cast<size_t>(region)] = enable;
        }
    }
}

void PostProcessingDriver::EnableProjection(const std::string& field_name, bool enable) {
    for (auto& reg : m_registered_projections) {
        if (reg.field_name == field_name) {
            std::fill(reg.region_enabled.begin(), reg.region_enabled.end(), enable);
        }
    }
}

void PostProcessingDriver::EnableAllProjections() {
    for (auto& reg : m_registered_projections) {
        for (size_t region = 0; region < static_cast<size_t>(m_num_regions); ++region) {
            // Check compatibility with region's model type
            bool compatible = true;
            if (reg.model_compatibility == ProjectionTraits::ModelCompatibility::EXACMECH_ONLY &&
                m_region_model_types[region] != MechType::EXACMECH) {
                compatible = false;
            }
            if (reg.model_compatibility == ProjectionTraits::ModelCompatibility::UMAT_ONLY &&
                m_region_model_types[region] != MechType::UMAT) {
                compatible = false;
            }

            // Only enable if compatible and has required data
            if (compatible) {
                reg.region_enabled[region] = true;
            }
        }
    }
}

std::vector<std::pair<std::string, std::string>>
PostProcessingDriver::GetAvailableProjections() const {
    std::vector<std::pair<std::string, std::string>> available;
    for (const auto& reg : m_registered_projections) {
        available.emplace_back(reg.field_name, reg.display_name);
    }
    return available;
}

size_t PostProcessingDriver::GetQuadratureFunctionSize() const {
    // Return size based on one of the region quadrature functions
    for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
        if (auto pqf = m_sim_state->GetQuadratureFunction("cauchy_stress_end", region)) {
            return static_cast<size_t>(pqf->GetSpaceShared()->GetSize());
        }
    }
    return 0;
}