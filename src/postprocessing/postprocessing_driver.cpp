#include "postprocessing/postprocessing_driver.hpp"

#include "postprocessing/mechanics_lightup.hpp"
#include "postprocessing/postprocessing_file_manager.hpp"
#include "postprocessing/projection_class.hpp"
#include "postprocessing/surface_quantities.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/rotations.hpp"
#include "utilities/unified_logger.hpp"

#include "ECMech_const.h"
#include "SNLS_linalg.h"

#include <filesystem>
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
 * @param region_model_types Vector of material model types per region
 * @return Vector of AllStateVariablesProjection instances
 *
 * Creates projections that output all available state variables
 * for debugging and detailed analysis. State variable count and
 * interpretation depend on the specific material model.
 */
std::vector<std::shared_ptr<ProjectionBase>>
RegisterAllState(const std::vector<MechType>& region_model_types) {
    return RegisterGeneric<AllStateVariablesProjection>(region_model_types);
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
        projection_class = RegisterAllState(m_region_model_types);
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

    // Initialize grid functions and data collections
    if (m_enable_visualization) {
        auto mesh = m_sim_state->GetMesh();
        if (m_num_regions == 1) {
            auto l2g = sim_state->GetQuadratureFunction("cauchy_stress_end", 0)
                           ->GetPartialSpaceShared()
                           ->GetLocal2Global();
            mfem::Array<int> pqs2submesh(l2g);
            m_map_pqs2submesh.emplace(0, std::move(pqs2submesh));
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
                m_map_pqs2submesh.emplace(region, std::move(pqs2submesh));
            }
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
        auto mesh = m_map_submesh[region];
        const int space_dim = mesh->SpaceDimension();
        std::string l2_fec_str = "L2_" + std::to_string(space_dim) + "D_P" + std::to_string(0);
        auto l2_fec = m_sim_state->GetFiniteElementCollection(l2_fec_str);
        auto value = std::make_shared<mfem::ParFiniteElementSpace>(
            mesh.get(), l2_fec.get(), vdim, mfem::Ordering::byVDIM);
        m_map_pfes[region].emplace(vdim, std::move(value));
    }
    return m_map_pfes[region][vdim];
}

void PostProcessingDriver::UpdateFields([[maybe_unused]] const int step,
                                        [[maybe_unused]] const double time) {
    for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
        if (!m_sim_state->IsRegionActive(region)) {
            continue;
        }
        auto state_qf_avg = m_sim_state->GetQuadratureFunction("state_var_avg", region);
        auto state_qf_end = m_sim_state->GetQuadratureFunction("state_var_end", region);
        CalcElementAvg(state_qf_avg.get(), state_qf_end.get());
        auto cauchy_qf_avg = m_sim_state->GetQuadratureFunction("cauchy_stress_avg", region);
        auto cauchy_qf_end = m_sim_state->GetQuadratureFunction("cauchy_stress_end", region);
        CalcElementAvg(cauchy_qf_avg.get(), cauchy_qf_end.get());
    }

    // Execute projections based on aggregation mode
    if (m_aggregation_mode == AggregationMode::PER_REGION ||
        m_aggregation_mode == AggregationMode::BOTH) {
        // Process each region separately
        for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
            if (!m_sim_state->IsRegionActive(region)) {
                continue;
            }
            const size_t reg_idx = static_cast<size_t>(region);
            auto qpts2mesh = m_map_pqs2submesh[region];
            for (auto& reg : m_registered_projections) {
                if (reg.region_enabled[reg_idx]) {
                    const auto gf_name = GetGridFunctionName(reg.display_name, region);
                    auto& grid_func = m_map_gfs[gf_name];
                    reg.projection_class[reg_idx]->Execute(
                        m_sim_state, grid_func, qpts2mesh, region);
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

PostProcessingDriver::VolumeAverageData
PostProcessingDriver::CalculateVolumeAverage(CalcType calc_type, int region) {
    if (!m_sim_state->IsRegionActive(region)) {
        return VolumeAverageData();
    }

    std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf;
    int data_size;
    std::string qf_name;
    const size_t reg_idx = static_cast<size_t>(region);

    // Configure calculation parameters based on type
    switch (calc_type) {
    case CalcType::STRESS:
        qf_name = "cauchy_stress_end";
        data_size = 6; // Voigt notation: Sxx, Syy, Szz, Sxy, Sxz, Syz
        break;

    case CalcType::DEF_GRAD:
        qf_name = "kinetic_grads";
        data_size = 9; // Full 3x3 tensor: F11, F12, F13, F21, F22, F23, F31, F32, F33
        break;

    case CalcType::PLASTIC_WORK:
    case CalcType::EQ_PL_STRAIN:
        if (m_region_model_types[reg_idx] == MechType::UMAT) {
            return VolumeAverageData();
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
            return VolumeAverageData();
        }
        qf_name = "kinetic_grads"; // Adjust this to your actual QF name for elastic strain
        data_size = 9;             // Voigt notation: Ee11, Ee22, Ee33, Ee23, Ee13, Ee12
        break;

    default:
        // This should never happen due to enum type safety, but defensive programming
        MFEM_ABORT_0("Unhandled calculation type in CalculateVolumeAverage");
        return VolumeAverageData();
    }

    // Get the quadrature function for this region
    qf = m_sim_state->GetQuadratureFunction(qf_name, region);
    if (!qf) {
        // Region doesn't have this quadrature function - return invalid data
        return VolumeAverageData();
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
        const int ne =
            m_sim_state->GetQuadratureFunction("state_var_end", region)->GetSpaceShared()->GetNE();
        const int estrain_ind =
            m_sim_state->GetQuadratureFunctionStatePair("elastic_strain", region).first;
        const int quats_ind = m_sim_state->GetQuadratureFunctionStatePair("quats", region).first;
        const int rel_vol_ind =
            m_sim_state->GetQuadratureFunctionStatePair("relative_volume", region).first;
        qf->operator=(0.0);
        auto data = qf->Write();

        mfem::forall(ne, [=] MFEM_HOST_DEVICE(int i) {
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

    // Initialize accumulators for volume-weighted averaging
    mfem::Vector global_avg_data(data_size);
    global_avg_data = 0.0;
    double global_volume = 0.0;

    // Accumulate contributions from all regions
    for (int region = 0; region < static_cast<int>(m_num_regions); ++region) {
        // Use cached data if available, calculate if not
        auto region_data = GetOrCalculateVolumeAverage(calc_type, region);
        // Now gather all region data to rank 0
        if (m_mpi_rank == 0) {
            // Rank 0 receives from all region roots
            const int root_rank = m_sim_state->GetRegionRootRank(region);
            if (root_rank > m_mpi_rank) {
                region_data.data.SetSize(data_size);
                region_data.is_valid = true;
                // Receive from the region root
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
        } else {
            // Other ranks send their region data if they're region roots
            if (m_sim_state->IsRegionIORoot(region)) {
                MPI_Send(
                    region_data.data.HostRead(), data_size, MPI_DOUBLE, 0, region, MPI_COMM_WORLD);
                MPI_Send(&region_data.volume,
                         1,
                         MPI_DOUBLE,
                         0,
                         static_cast<int>(m_num_regions) * 2 + region,
                         MPI_COMM_WORLD);
            }
        }

        if (region_data.is_valid && region_data.volume > 0.0) {
            // Add volume-weighted contribution to global average
            for (int i = 0; i < data_size; ++i) {
                if (calc_type != CalcType::PLASTIC_WORK) {
                    global_avg_data[i] += region_data.data[i] * region_data.volume;
                } else {
                    global_avg_data[i] += region_data.data[i];
                }
            }
            global_volume += region_data.volume;
        }
    }

    // Normalize by total volume to get the true global average
    if (global_volume > 0.0) {
        if (calc_type != CalcType::PLASTIC_WORK) {
            global_avg_data /= global_volume;
        }
    } else {
        // No valid regions found - issue warning
        MFEM_WARNING_0("No valid regions found for global " << calc_type_str << " calculation");
        return;
    }

    // Write global output (region = -1 indicates global file)
    if (data_size == 1) {
        // Scalar quantity
        m_file_manager->WriteVolumeAverage(
            calc_type_str, -1, "", time, global_volume, global_avg_data[0]);
    } else {
        // Vector/tensor quantity
        m_file_manager->WriteVolumeAverage(
            calc_type_str, -1, "", time, global_volume, global_avg_data);
    }
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

    for (const auto& field : fields) {
        RegisterProjection(field);
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
            auto fe_space = m_sim_state->GetParFiniteElementSpace(max_vdim);
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