#pragma once

#include "postprocessing/viz_point_map.hpp"
#include "sim_state/simulation_state.hpp"
#include "utilities/cuda_access.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem.hpp"

#include <memory>
#include <string>

namespace ProjectionTraits {
/**
 * @brief Model compatibility enumeration for projections
 */
enum class ModelCompatibility {
    ALL_MODELS,    ///< Compatible with all material models
    EXACMECH_ONLY, ///< Only compatible with ExaCMech models
    UMAT_ONLY      ///< Only compatible with UMAT models
};
} // namespace ProjectionTraits

/**
 * @brief Base projection interface for all projection types in ExaConstit
 *
 * ProjectionBase provides the fundamental interface that all projection classes
 * must implement. It defines the common operations for converting quadrature
 * function data to grid function data suitable for visualization and analysis.
 *
 * Key responsibilities:
 * - Execute projection operations for specific material regions
 * - Provide vector dimension information for grid function creation
 * - Support material model compatibility checking
 * - Enable global aggregation capabilities when appropriate
 *
 * The class uses the template method pattern where derived classes implement
 * specific projection algorithms while the base class handles common interface
 * requirements and material model compatibility checking.
 *
 * Material model compatibility is enforced through the ProjectionTraits::ModelCompatibility
 * enumeration, allowing projections to specify whether they work with all material
 * models, only ExaCMech models, or only UMAT models.
 *
 * @ingroup ExaConstit_projections
 */
class ProjectionBase {
public:
    /**
     * @brief Model compatibility type alias
     *
     * Shorthand for ProjectionTraits::ModelCompatibility enumeration,
     * used throughout projection classes to specify material model
     * compatibility requirements.
     */
    using ptmc = ProjectionTraits::ModelCompatibility;
    /**
     * @brief Material model compatibility for this projection
     *
     * Specifies which material model types are compatible with this projection.
     * Used during registration to ensure projections are only created for
     * appropriate material regions. Defaults to ALL_MODELS for maximum compatibility.
     */
    const ptmc model = ptmc::ALL_MODELS;

public:
    ProjectionBase() = default;
    ProjectionBase(const ptmc mc) : model(mc) {};
    virtual ~ProjectionBase() = default;

    /**
     * @brief Execute the projection for a specific region
     * @param sim_state Reference to simulation state
     * @param grid_function Target grid function to populate
     * @param points Region's table of (source quadrature point, output node) pairs. It decides
     *               both which quadrature function is read ("_avg" for zone output, "_end" for
     *               quadrature output) and where each value lands; see VizPointMap.
     * @param region Region index
     */
    virtual void Execute(std::shared_ptr<SimulationState> sim_state,
                         std::shared_ptr<mfem::ParGridFunction> grid_function,
                         const VizPointMap& points,
                         int region) = 0;

    /**
     * @brief Get the vector dimension for this projection
     */
    virtual int GetVectorDimension() const = 0;

    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const {
        return false;
    }

    /**
     * @brief Get a display name for this projection
     */
    virtual std::string GetDisplayName() const = 0;
};

/**
 * @brief Copy per-point values onto the visualization nodes of one region
 *
 * @param source Region quadrature function (element averages for zone output, end-of-step point
 *               values for quadrature output; see VizPointMap)
 * @param first_component First component of `source` to copy
 * @param num_components Number of components to copy (output components 0 .. num_components-1)
 * @param output Region grid function to fill
 * @param points Region's table of (source quadrature point, output node) pairs
 *
 * The kernel shared by the projections that write values unchanged (state variables, user
 * function quantities): output(j, dest(k)) = source(first_component + j, source_point(k)).
 */
void CopyPointValues(const mfem::expt::PartialQuadratureFunction& source,
                     const int first_component,
                     const int num_components,
                     mfem::ParGridFunction& output,
                     const VizPointMap& points);

//=============================================================================
// GEOMETRY PROJECTIONS
//=============================================================================

/**
 * @brief Base class for geometry-based projections that operate directly on mesh data
 *
 * GeometryProjection specializes ProjectionBase for projections that compute
 * geometric quantities directly from mesh topology and coordinates, without
 * requiring material-specific quadrature function data.
 *
 * These projections are inherently region-independent since they depend only
 * on mesh geometry rather than material state. Examples include element
 * centroids, volumes, and geometric quality measures.
 *
 * Key characteristics:
 * - Region-independent operation (same result regardless of material region)
 * - Direct mesh geometry access through finite element spaces
 * - No dependency on material model type or quadrature function data
 * - Automatic global aggregation support for visualization
 *
 * Derived classes must implement ProjectGeometry() to perform the actual
 * geometric calculations using MFEM's geometric factors and integration rules.
 *
 * @ingroup ExaConstit_projections_geometry
 */
class GeometryProjection : public ProjectionBase {
public:
    GeometryProjection() = default;
    ~GeometryProjection() {};
    /**
     * @brief Execute geometry projection (region-independent)
     *
     * @param sim_state Reference to simulation state (unused for geometry)
     * @param grid_function Target grid function to populate
     * @param points Region's point table (only its element list is used)
     * @param region Region index (unused for geometry)
     *
     * Executes geometry-based projection by calling the pure virtual
     * ProjectGeometry() method. Geometry projections are region-independent
     * since they depend only on mesh topology and element geometry. The value is an
     * element quantity, so in quadrature output it is repeated on every node of the element.
     */
    void Execute([[maybe_unused]] std::shared_ptr<SimulationState> sim_state,
                 std::shared_ptr<mfem::ParGridFunction> grid_function,
                 const VizPointMap& points,
                 [[maybe_unused]] int region) override {
        // Geometry projections don't depend on region-specific data
        ProjectGeometry(grid_function, points);
    }

    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const override {
        return true;
    }

protected:
    /**
     * @brief Pure virtual method for geometry calculations
     *
     * @param grid_function Target grid function to populate with geometry data
     * @param points Region's point table; every output node of an element receives the
     *               element's value
     *
     * Derived classes implement this method to compute geometry-based quantities
     * such as element centroids or volumes. The method has direct access to
     * mesh geometry through the grid function's finite element space.
     */
    virtual void ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                 const VizPointMap& points) = 0;
};

/**
 * @brief Element centroid calculation projection
 *
 * Computes geometric centroids of mesh elements by integrating coordinate
 * positions over element volumes. Provides spatial location information
 * for visualization and spatial analysis of simulation results.
 *
 * The centroid calculation uses numerical integration over each element
 * with proper volume weighting to handle arbitrary element shapes and
 * polynomial orders. Results are stored as 3D coordinate vectors.
 *
 * @ingroup ExaConstit_projections_geometry
 */
class CentroidProjection final : public GeometryProjection {
public:
    CentroidProjection() = default;
    ~CentroidProjection() {};

    int GetVectorDimension() const override {
        return 3;
    } // Always 3D coordinates
    std::string GetDisplayName() const override {
        return "Element Centroids";
    }

EXACONSTIT_CUDA_PROTECTED:
    void ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function,
                         const VizPointMap& points) override;
};

/**
 * @brief Element volume projection for mesh analysis and visualization
 *
 * VolumeProjection computes the volume of each mesh element through numerical
 * integration of the Jacobian determinant over the element domain. This provides
 * essential geometric information for volume averaging operations, mesh quality
 * assessment, and visualization scaling.
 *
 * The volume calculation uses MFEM's geometric factors to access pre-computed
 * Jacobian determinants at integration points, which are then integrated using
 * the appropriate quadrature weights to obtain accurate element volumes for
 * arbitrary element shapes and polynomial orders.
 *
 * Key features:
 * - Accurate volume calculation for arbitrary element geometries
 * - Support for high-order finite elements through appropriate integration rules
 * - Essential for volume-weighted averaging operations in post-processing
 * - Useful for mesh quality assessment and adaptive refinement criteria
 *
 * The computed volumes are stored as scalar values (vector dimension = 1) and
 * can be visualized directly or used internally for volume-weighted calculations
 * in other post-processing operations.
 *
 * @ingroup ExaConstit_projections_geometry
 */
class VolumeProjection final : public GeometryProjection {
public:
    VolumeProjection() = default;
    ~VolumeProjection() {};

    int GetVectorDimension() const override {
        return 1;
    } // Scalar volume
    std::string GetDisplayName() const override {
        return "Element Volumes";
    }

EXACONSTIT_CUDA_PROTECTED:
    void ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function,
                         const VizPointMap& points) override;
};

//=============================================================================
// STRESS-BASED PROJECTIONS
//=============================================================================

/**
 * @brief Base class for stress-based projections using Cauchy stress tensor data
 *
 * StressProjection provides a specialized interface for projections that operate
 * on stress tensor data from material constitutive models. It handles the common
 * pattern of retrieving stress quadrature functions and delegating to derived
 * classes for specific stress calculations.
 *
 * The class expects stress data in Voigt notation with 6 components representing
 * the symmetric Cauchy stress tensor: [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂].
 *
 * Key features:
 * - Automatic stress quadrature function retrieval from simulation state
 * - Support for both element-averaged and quadrature point stress data
 * - Global aggregation capabilities for multi-region stress analysis
 * - Compatible with all material model types that provide stress output
 *
 * Derived classes implement ProjectStress() to perform specific calculations
 * such as equivalent stress measures, stress invariants, or direct component
 * extraction for visualization and post-processing.
 *
 * @ingroup ExaConstit_projections_stress
 */
class StressProjection : public ProjectionBase {
public:
    StressProjection() = default;
    ~StressProjection() {};

    void Execute(std::shared_ptr<SimulationState> sim_state,
                 std::shared_ptr<mfem::ParGridFunction> grid_function,
                 const VizPointMap& points,
                 int region) override {
        // Element averages (zone output) or end-of-step quadrature values (quadrature output)
        auto stress_qf = sim_state->GetQuadratureFunction(points.SourceName("cauchy_stress"),
                                                          region);
        if (!stress_qf)
            return; // Region doesn't have stress data
        MFEM_VERIFY_0(stress_qf->GetVDim() == 6,
                      "StressProjection expects 6 Cauchy stress components (Voigt notation)");

        // Project the stress calculation
        ProjectStress(stress_qf, grid_function, points);
    }

    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const override {
        return true;
    }

protected:
    /**
     * @brief Pure virtual method for stress-specific calculations
     *
     * @param stress_qf Partial quadrature function containing stress tensor data
     * @param grid_function Target grid function to populate with processed stress
     * @param points Table of (source quadrature point, output node) pairs for the region
     *
     * Derived classes implement this method to perform specific stress calculations
     * such as Von Mises equivalent stress, hydrostatic stress, or direct stress
     * component extraction. The stress data is provided in Voigt notation with
     * 6 components: [S11, S22, S33, S23, S13, S12].
     */
    virtual void
    ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                  std::shared_ptr<mfem::ParGridFunction> grid_function,
                  const VizPointMap& points) = 0;
};

/**
 * @brief Full Cauchy stress tensor projection in Voigt notation
 *
 * CauchyStressProjection extracts and projects the complete Cauchy stress tensor
 * from material constitutive models for visualization and analysis. The stress
 * components are output in Voigt notation for efficient storage and compatibility
 * with standard post-processing workflows.
 *
 * The projection preserves all six independent components of the symmetric stress
 * tensor: [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂], enabling detailed stress field analysis
 * and validation of constitutive model predictions.
 *
 * Key applications:
 * - Complete stress field visualization in finite element post-processors
 * - Stress validation against analytical solutions or experimental data
 * - Input for stress-based failure criteria and damage models
 * - Principal stress calculations and stress invariant analysis
 * - Multi-axial loading analysis and stress path characterization
 *
 * The projection is compatible with all material model types that provide
 * Cauchy stress output and supports global aggregation for multi-material
 * simulations with consistent stress field representation.
 *
 * @ingroup ExaConstit_projections_stress
 */
class CauchyStressProjection final : public StressProjection {
public:
    CauchyStressProjection() = default;
    ~CauchyStressProjection() {};

    int GetVectorDimension() const override {
        return 6;
    } // Symmetric tensor in Voigt notation
    std::string GetDisplayName() const override {
        return "Cauchy Stress";
    }

EXACONSTIT_CUDA_PROTECTED:
    virtual void
    ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                  std::shared_ptr<mfem::ParGridFunction> stress_gf,
                  const VizPointMap& points) override;
};

/**
 * @brief Von Mises equivalent stress projection for failure analysis
 *
 * VonMisesStressProjection computes the Von Mises equivalent stress from the
 * Cauchy stress tensor, providing a scalar measure of stress intensity commonly
 * used in plasticity theory and failure analysis. The Von Mises stress is
 * calculated as the second invariant of the stress deviator tensor.
 *
 * Mathematical formulation:
 * σᵥₘ = √(3/2 * sᵢⱼsᵢⱼ) = √(1/2 * [(σ₁₁-σ₂₂)² + (σ₂₂-σ₃₃)² + (σ₃₃-σ₁₁)² + 6(σ₁₂² + σ₁₃² + σ₂₃²)])
 *
 * Key applications:
 * - Yield criterion evaluation in metal plasticity
 * - Fatigue analysis and life prediction
 * - Stress concentration identification
 * - Material failure assessment
 * - Optimization of component design for stress reduction
 *
 * The Von Mises stress provides a material-independent measure of stress state
 * that can be directly compared with material yield strength and used in
 * plasticity models regardless of the specific loading configuration.
 *
 * @ingroup ExaConstit_projections_stress
 */
class VonMisesStressProjection final : public StressProjection {
public:
    VonMisesStressProjection() = default;
    ~VonMisesStressProjection() {};

    int GetVectorDimension() const override {
        return 1;
    } // Scalar quantity
    std::string GetDisplayName() const override {
        return "Von Mises Stress";
    }

EXACONSTIT_CUDA_PROTECTED:
    virtual void
    ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                  std::shared_ptr<mfem::ParGridFunction> von_mises,
                  const VizPointMap& points) override;
};

/**
 * @brief Hydrostatic stress projection for pressure analysis
 *
 * HydrostaticStressProjection computes the hydrostatic (mean normal) stress
 * component from the Cauchy stress tensor. The hydrostatic stress represents
 * the volumetric part of the stress state and is crucial for analyzing
 * pressure-dependent material behavior and volumetric deformation.
 *
 * Mathematical formulation:
 * σₕ = (σ₁₁ + σ₂₂ + σ₃₃) / 3 = (1/3) * tr(σ)
 *
 * Key applications:
 * - Pressure-dependent plasticity models (Drucker-Prager, Mohr-Coulomb)
 * - Volumetric strain analysis and compressibility studies
 * - Geomechanics and soil mechanics applications
 * - Phase transformation analysis under pressure
 * - Cavitation and void nucleation studies
 * - Bulk modulus validation and material characterization
 *
 * The hydrostatic stress is particularly important in materials that exhibit
 * pressure-sensitive behavior, such as geological materials, polymers, and
 * porous media, where the volumetric stress component significantly affects
 * material response.
 *
 * @ingroup ExaConstit_projections_stress
 */
class HydrostaticStressProjection final : public StressProjection {
public:
    HydrostaticStressProjection() = default;
    ~HydrostaticStressProjection() {};

    int GetVectorDimension() const override {
        return 1;
    } // Scalar quantity
    std::string GetDisplayName() const override {
        return "Hydrostatic Stress";
    }

EXACONSTIT_CUDA_PROTECTED:
    virtual void
    ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                  std::shared_ptr<mfem::ParGridFunction> hydro_static,
                  const VizPointMap& points) override;
};

//=============================================================================
// STATE VARIABLE PROJECTIONS
//=============================================================================

/**
 * @brief Base class for state variable projections from material constitutive models
 *
 * StateVariableProjection provides a framework for extracting and projecting
 * specific components from material model state variable arrays. This enables
 * visualization and analysis of internal material state evolution including
 * plastic strains, hardening variables, crystal orientations, and other
 * constitutive model-specific quantities.
 *
 * The class handles the common pattern of:
 * 1. Retrieving state variable quadrature functions from simulation state
 * 2. Extracting specific components based on index and length specifications
 * 3. Copying data to grid functions with proper element mapping
 * 4. Applying optional post-processing for data conditioning
 *
 * Key features:
 * - Flexible component extraction with configurable start index and length
 * - Automatic dimension validation against available data and target grid functions
 * - Material model compatibility checking (ALL_MODELS, EXACMECH_ONLY, UMAT_ONLY)
 * - Optional post-processing hook for derived classes (normalization, clamping, etc.)
 * - Support for both scalar and vector state variable components
 *
 * The state variable array organization depends on the specific material model,
 * but typically follows a consistent layout within each model type. ExaCMech
 * models provide well-defined state variable mappings through SimulationState
 * helper methods.
 *
 * @ingroup ExaConstit_projections_state_variables
 */
class StateVariableProjection : public ProjectionBase {
public:
    StateVariableProjection(const std::string& state_var_name,
                            int component_index = 0,
                            int component_length = -1,
                            const std::string& display_name = "",
                            ptmc mc = ptmc::ALL_MODELS)
        : ProjectionBase(mc), m_state_var_name(state_var_name), m_display_name(display_name),
          m_component_index(component_index), m_component_length(component_length) {}

    ~StateVariableProjection() {};

    void Execute(std::shared_ptr<SimulationState> sim_state,
                 std::shared_ptr<mfem::ParGridFunction> state_gf,
                 const VizPointMap& points,
                 int region) override;

    std::string GetDisplayName() const override {
        return m_display_name;
    }

    int GetVectorDimension() const override {
        return m_component_length;
    }

protected:
    /**
     * @brief Post-processing hook for derived classes
     *
     * @param grid_function Target grid function containing extracted state data
     * @param points Region's point table; the output nodes written by this region are
     *               points.ReadDest()[0 .. points.GetNumPoints() - 1]
     *
     * Virtual method called after state variable extraction to allow derived
     * classes to perform additional processing such as normalization, coordinate
     * transformations, or value clamping. It runs node by node on values that were each
     * copied from a single quadrature point, so per-point constraints hold exactly.
     * Default implementation does nothing.
     */
    virtual void PostProcessStateVariable(
        [[maybe_unused]] std::shared_ptr<mfem::ParGridFunction> grid_function,
        [[maybe_unused]] const VizPointMap& points) const {};

    /**
     * @brief State variable name for SimulationState lookup
     *
     * Key used to retrieve the appropriate state variable quadrature function
     * from SimulationState. Must match the naming conventions used by the
     * material model for proper data access.
     */
    std::string m_state_var_name;

    /**
     * @brief Display name for the state variable name
     */
    std::string m_display_name;

    /**
     * @brief Starting index of component within state variable vector
     *
     * Zero-based index indicating the first component to extract from the
     * state variable vector at each quadrature point. For scalar quantities,
     * this is the direct index. For multi-component data, this is the starting index.
     */
    int m_component_index;

    /**
     * @brief Number of components to extract (-1 for automatic detection)
     *
     * Specifies how many consecutive components to extract starting from
     * m_component_index. When set to -1, the component length is determined
     * automatically based on available data or target grid function dimensions.
     */
    int m_component_length;
};

/**
 * @brief Complete state variable array projection for debugging and analysis
 *
 * AllStateVariablesProjection extracts and projects the entire state variable
 * array from material constitutive models, providing comprehensive access to
 * all internal material state information. This projection is primarily used
 * for debugging material model implementations and detailed analysis of
 * constitutive model behavior.
 *
 * Key characteristics:
 * - Projects all available state variables without filtering
 * - Vector dimension determined automatically from material model
 * - Useful for debugging constitutive model implementations
 * - Enables detailed analysis of material state evolution
 * - Cannot be aggregated globally due to variable interpretation complexity
 *
 * The interpretation of state variable components depends entirely on the
 * specific material model implementation:
 * - ExaCMech: Includes plastic strains, hardening variables, orientations, etc.
 * - UMAT: User-defined state variables with model-specific meanings
 * - Other models: Model-specific internal state representations
 *
 * This projection is typically used during material model development,
 * validation, and debugging rather than for routine post-processing and
 * visualization of simulation results.
 *
 * @note The output requires detailed knowledge of the material model's
 *       state variable organization for proper interpretation.
 *
 * @ingroup ExaConstit_projections_state_variables
 */
class AllStateVariablesProjection final : public StateVariableProjection {
public:
    /**
     * @param num_state_vars Number of state variables per point of the region (the vector size
     *                       of the output field)
     */
    explicit AllStateVariablesProjection(int num_state_vars)
        : StateVariableProjection("all_state_vars", 0, num_state_vars, "All State Variables") {}

    ~AllStateVariablesProjection() {};
    virtual bool CanAggregateGlobally() const override {
        return false;
    }
};

/**
 * @brief Post-processing projection for physically non-negative state variables
 *
 * This class provides post-processing functionality for state variables that must be
 * physically non-negative, ensuring numerical stability and physical consistency in
 * finite element simulations. The projection applies a lower bound of zero to all
 * computed values, preventing numerical artifacts from producing unphysical negative
 * quantities.
 *
 * @details
 * During finite element computations, numerical errors, interpolation artifacts, or
 * convergence issues can occasionally produce small negative values for quantities that
 * should be strictly non-negative (e.g., equivalent plastic strain, damage parameters,
 * void fractions). This projection enforces the physical constraint by applying:
 *
 * \f$ \tilde{q} = \max(q, 0) \f$
 *
 * where \f$q\f$ is the computed state variable and \f$\tilde{q}\f$ is the corrected value.
 *
 * @note Currently optimized for ExaCMech constitutive models but designed to be
 *       extensible to other material model frameworks requiring non-negative state variables.
 *
 * @par Typical Use Cases:
 * - Equivalent plastic strain (\f$\varepsilon^p_{eq}\f$)
 * - Damage variables (\f$D\f$)
 * - Void fraction in porous materials (\f$f\f$)
 * - Hardening variables (\f$\kappa\f$)
 * - Any physically bounded scalar state variables
 *
 * @par Performance Characteristics:
 * - Device-compatible (GPU/CPU) through MFEM forall
 * - O(N) complexity where N is the number of degrees of freedom
 * - Supports global aggregation for parallel post-processing
 *
 * @warning This projection modifies the computed field values. Ensure this is
 *          appropriate for your analysis before enabling.
 *
 * @ingroup ExaConstit_projections_state_variables
 * @see StateVariableProjection for base class functionality
 * @see PostProcessing.projections configuration options
 */
class NNegStateProjection final : public StateVariableProjection {
public:
    /**
     * @brief Construct a non-negative state variable projection
     *
     * @param state_var_name Name of the state variable in the material model
     * @param component_index Index of the specific component (for tensor/vector state vars)
     * @param component_length Total number of components in the state variable
     * @param display_name Human-readable name for visualization and output
     *
     * @note The projection currently defaults to EXACMECH_ONLY compatibility but
     *       the implementation is model-agnostic and could be extended to other
     *       constitutive frameworks.
     */
    NNegStateProjection(const std::string& state_var_name,
                        int component_index,
                        int component_length,
                        const std::string& display_name)
        : StateVariableProjection(state_var_name,
                                  component_index,
                                  component_length,
                                  display_name,
                                  ptmc::EXACMECH_ONLY) {}
    ~NNegStateProjection() = default;

    virtual bool CanAggregateGlobally() const override {
        return true;
    }

EXACONSTIT_CUDA_PROTECTED:
    /**
     * @brief Apply non-negative constraint to state variable field
     *
     * Performs element-wise maximum operation with zero to ensure all field
     * values satisfy the non-negative constraint. Uses MFEM's device-portable
     * forall construct for optimal performance on both CPU and GPU architectures.
     *
     * @param grid_function Shared pointer to the MFEM grid function containing state data
     * @param points Region's point table (output nodes to clamp)
     *
     * @note The mathematical operation applied is: data[i] = max(data[i], 0.0)
     *       for every component of every output node written by this region.
     */
    virtual void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                          const VizPointMap& points) const override;
};

/**
 * @brief Crystal orientation projection with quaternion normalization
 *
 * Projects crystal orientation quaternions from ExaCMech models with
 * automatic normalization to ensure unit quaternions. Handles quaternion
 * data extraction and post-processing for texture analysis applications.
 *
 * The post-processing step normalizes quaternions to unit length to correct for
 * numerical drift (and, in zone output, for the shortening caused by averaging the
 * quaternions of an element) and ensure valid rotation representations.
 * Output quaternions follow the convention [q0, q1, q2, q3] where q0
 * is the scalar component.
 *
 * Only compatible with ExaCMech material models that provide quaternion
 * orientation data in their state variable arrays.
 *
 * @ingroup ExaConstit_projections_crystal_plasticity
 */
class XtalOrientationProjection final : public StateVariableProjection {
public:
    XtalOrientationProjection(const std::string& state_var_name,
                              int component_index,
                              int component_length,
                              const std::string& display_name)
        : StateVariableProjection(state_var_name,
                                  component_index,
                                  component_length,
                                  display_name,
                                  ptmc::EXACMECH_ONLY) {}
    ~XtalOrientationProjection() = default;

    virtual bool CanAggregateGlobally() const override {
        return true;
    }

EXACONSTIT_CUDA_PROTECTED:
    virtual void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                          const VizPointMap& points) const override;
};

/**
 * @brief Elastic strain tensor projection for ExaCMech models
 *
 * Projects elastic strain tensor components from ExaCMech state variables
 * with coordinate system transformations. Performs conversion from lattice
 * coordinates to sample coordinates using crystal orientation data.
 *
 * The projection involves:
 * 1. Extraction of deviatoric elastic strain and relative volume
 * 2. Reconstruction of full elastic strain tensor in lattice coordinates
 * 3. Rotation to sample coordinates using quaternion orientations
 * 4. Output in Voigt notation: [E11, E22, E33, E23, E13, E12]
 *
 * Only compatible with ExaCMech models that provide elastic strain
 * state variables and crystal orientation data.
 *
 * @ingroup ExaConstit_projections_strain
 */
class ElasticStrainProjection final : public StateVariableProjection {
public:
    ElasticStrainProjection(const std::string& state_var_name,
                            int component_index,
                            int component_length,
                            const std::string& display_name)
        : StateVariableProjection(state_var_name,
                                  component_index,
                                  component_length,
                                  display_name,
                                  ptmc::EXACMECH_ONLY) {
        m_component_length = 6;
    }
    /**
     * @brief Execute elastic strain projection with coordinate transformation
     *
     * @param sim_state Reference to simulation state for data access
     * @param elastic_strain_gf Target grid function for elastic strain output
     * @param points Table of (source quadrature point, output node) pairs for the region
     * @param region Material region identifier
     *
     * Overrides the base StateVariableProjection::Execute() method to implement
     * specialized processing for elastic strain data. The method:
     *
     * 1. Retrieves state variables including elastic strain, orientations, and volume
     * 2. Reconstructs full 3x3 elastic strain tensor from deviatoric components
     * 3. Applies coordinate transformation from crystal to sample coordinates
     * 4. Outputs strain components in Voigt notation
     *
     * The coordinate transformation accounts for crystal orientation evolution
     * and ensures strain components are expressed in the global reference frame
     * for visualization and post-processing compatibility.
     *
     * @note This method bypasses the standard StateVariableProjection data flow
     *       due to the complex coordinate transformations required.
     */
    void Execute(std::shared_ptr<SimulationState> sim_state,
                 std::shared_ptr<mfem::ParGridFunction> elastic_strain_gf,
                 const VizPointMap& points,
                 int region) override;

    virtual bool CanAggregateGlobally() const override {
        return true;
    }
};

/**
 * @brief Shear rate projection for crystal plasticity analysis
 *
 * Projects macroscopic shear rate data from ExaCMech state variables.
 * Provides access to overall plastic deformation rates for rate-dependent
 * analysis and comparison with experimental strain rate measurements.
 *
 * This projection extracts aggregate shear rate information rather than
 * individual slip system rates, making it suitable for macroscopic
 * deformation analysis and rate sensitivity studies.
 *
 * Only compatible with ExaCMech material models that compute and store
 * macroscopic shear rate data.
 *
 * @ingroup ExaConstit_projections_crystal_plasticity
 */
class ShearingRateProjection final : public StateVariableProjection {
public:
    ShearingRateProjection(const std::string& state_var_name,
                           int component_index,
                           int component_length,
                           const std::string& display_name)
        : StateVariableProjection(state_var_name,
                                  component_index,
                                  component_length,
                                  display_name,
                                  ptmc::EXACMECH_ONLY) {}

    virtual bool CanAggregateGlobally() const override {
        return false;
    }
};
//=============================================================================
// USER FUNCTION PROJECTIONS
//=============================================================================

/**
 * @brief Writes the values of a post-processing user function as a visualization field
 *
 * A [[PostProcessing.user_functions]] entry of type "visualization" fills, for each selected
 * region, a quadrature function with its quantity at every integration point (see
 * PostProcessingDriver). This projection copies those values onto the region's visualization
 * nodes through the same VizPointMap as every other projection: in zone output it reads the
 * element averages of the values (filled by the driver), in quadrature output the point values
 * themselves.
 *
 * The quadrature functions belong to the driver and are given at construction, unlike the other
 * projections, which look their input up in SimulationState by name. One instance is created per
 * region; regions the quantity is not computed for get an instance without data and are skipped.
 *
 * @ingroup ExaConstit_projections
 */
class QuadratureFunctionProjection final : public ProjectionBase {
public:
    /**
     * @param display_name Field name in the visualization files
     * @param num_components Values per point
     * @param values Point values of the region (null if the region is not computed here)
     * @param averages Element averages of the values (null if the region is not computed here)
     * @param global Whether the field is also combined over the regions into the global output
     */
    QuadratureFunctionProjection(const std::string& display_name,
                                 const int num_components,
                                 std::shared_ptr<mfem::expt::PartialQuadratureFunction> values,
                                 std::shared_ptr<mfem::expt::PartialQuadratureFunction> averages,
                                 const bool global)
        : m_display_name(display_name), m_num_components(num_components),
          m_values(std::move(values)), m_averages(std::move(averages)), m_global(global) {}

    void Execute(std::shared_ptr<SimulationState> sim_state,
                 std::shared_ptr<mfem::ParGridFunction> grid_function,
                 const VizPointMap& points,
                 int region) override;

    int GetVectorDimension() const override {
        return m_num_components;
    }

    bool CanAggregateGlobally() const override {
        return m_global;
    }

    std::string GetDisplayName() const override {
        return m_display_name;
    }

private:
    /// Field name in the visualization files
    std::string m_display_name;
    /// Values per point
    int m_num_components;
    /// Point values of the region, filled by the user function
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> m_values;
    /// Element averages of m_values (zone output)
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> m_averages;
    /// Combined into the global output
    bool m_global;
};
