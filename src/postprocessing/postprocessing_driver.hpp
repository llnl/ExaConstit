#pragma once

#include "postprocessing/projection_class.hpp"
#include "sim_state/simulation_state.hpp"
#include "utilities/cuda_access.hpp"
#include "utilities/mechanics_kernels.hpp"

#include "mfem.hpp"

#include "ECMech_const.h"

// Forward declaration to avoid circular includes
class PostProcessingFileManager;

class LightUp;
/**
 * @brief PostProcessingDriver handles all post-processing operations for ExaConstit simulations
 *
 * This class manages:
 * 1. Projection of quadrature data to grid functions for visualization
 * 2. Calculation of volume-averaged quantities (global and per-region)
 * 3. Registration of data with visualization collections
 * 4. Output of post-processed data at specified intervals
 * 5. Multi-material support with region-specific and combined visualizations
 */
class PostProcessingDriver {
public:
    /**
     * @brief Aggregation mode for multi-region data
     */
    enum class AggregationMode {
        PER_REGION,      // Process each region separately
        GLOBAL_COMBINED, // Combine all regions into global fields
        BOTH             // Both per-region and global combined
    };

    /**
     * @brief Construct a new PostProcessingDriver
     *
     * @param sim_state Reference to global simulation state
     * @param options Simulation options
     */
    PostProcessingDriver(std::shared_ptr<SimulationState> sim_state, ExaOptions& options);

    /**
     * @brief Destructor
     */
    ~PostProcessingDriver();

    /**
     * @brief Update post-processing data for current step
     *
     * @param step Current time step
     * @param time Current simulation time
     */
    void Update(const int step, const double time);

    /**
     * @brief Calculate and output volume-averaged quantities
     *
     * @param time Current simulation time
     * @param mode Aggregation mode (default: BOTH)
     */
    void PrintVolValues(const double time, AggregationMode mode = AggregationMode::BOTH);

    /**
     * @brief Update data collections with current projection data
     *
     * @param step Current time step
     * @param time Current simulation time
     */
    void UpdateDataCollections(const int step, const double time);

    /**
     * @brief Enable or disable a projection for a specific region
     *
     * @param field_name Name of the field
     * @param region region index
     * @param enable Whether to enable the projection
     */
    void EnableProjection(const std::string& field_name, int region, bool enable = true);

    /**
     * @brief Enable or disable a projection for all regions
     *
     * @param field_name Name of the field
     * @param enable Whether to enable the projection
     */
    void EnableProjection(const std::string& field_name, bool enable = true);

    /**
     * @brief Enable or disable all projections based on model compatibility
     *
     * Automatically enables all projections that are compatible with each
     * region's material model type. Uses the model_compatibility field in
     * ProjectionRegistration to determine which projections should be enabled
     * for EXACMECH_ONLY, UMAT_ONLY, or ALL_MODELS compatibility levels.
     *
     * Provides a convenient way to activate all appropriate projections
     * without manually specifying each projection type for each region.
     */
    void EnableAllProjections();

    /**
     * @brief Get list of all available projection types
     *
     * @return Vector of pairs containing field names and display names
     *
     * Returns information about all registered projection types for UI display
     * or programmatic enumeration. Each pair contains the internal field name
     * (for EnableProjection calls) and the user-friendly display name.
     */
    std::vector<std::pair<std::string, std::string>> GetAvailableProjections() const;

    /**
     * @brief Set aggregation mode for multi-region processing
     */
    void SetAggregationMode(AggregationMode mode) {
        m_aggregation_mode = mode;
    }

    /**
     * @brief Check if volume averages should be output at current step
     *
     * @param step Current time step number
     * @return true if output should occur, false otherwise
     *
     * Determines output timing based on the configured output frequency
     * in ExaOptions. Delegates to PostProcessingFileManager for
     * consistent frequency control across all output operations.
     */
    bool ShouldOutputAtStep(int step) const;

    /**
     * @brief Check if visualization data collections should be output at current step
     *
     * @param step Current time step number
     * @return true if output should occur, false otherwise
     *
     * Determines output timing based on the visualization output frequency
     * in ExaOptions, which is independent of the volume average output frequency.
     */
    bool ShouldOutputVisualizationAtStep(int step) const;

    // Returns a pointer to a ParFiniteElementSpace (PFES) that's ordered according to VDIMs
    // and makes use of an L2 FiniteElementCollection
    // If the vdim is not in the internal mapping than a new PFES will be created
    std::shared_ptr<mfem::ParFiniteElementSpace> GetParFiniteElementSpace(const int region,
                                                                          const int vdim);

EXACONSTIT_CUDA_PRIVATE:
    /**
     * @brief Enumeration for volume average calculation types
     *
     * Provides type-safe identification of different calculation types to avoid
     * string comparison overhead and prevent typos. Each type corresponds to
     * a specific physical quantity computed in ExaConstit simulations.
     */
    enum class CalcType {
        STRESS,        ///< Cauchy stress tensor (6 components in Voigt notation)
        DEF_GRAD,      ///< Deformation gradient tensor (9 components)
        PLASTIC_WORK,  ///< Accumulated plastic work (scalar)
        EQ_PL_STRAIN,  ///< Equivalent plastic strain (scalar)
        EULER_STRAIN,  ///< Euler strain tensor (6 components in Voigt notation)
        ELASTIC_STRAIN ///< Elastic strain tensor (6 components in Voigt notation)
    };

    /**
     * @brief Cached volume average data for a single region
     *
     * Stores the computed volume average and associated volume for a specific
     * region to enable efficient reuse in global calculations. The cache prevents
     * redundant quadrature function evaluations and volume integrations.
     */
    struct VolumeAverageData {
        double volume;     ///< Total volume of the region
        mfem::Vector data; ///< Volume-averaged quantity (scalar or tensor components)
        bool is_valid;     ///< Flag indicating whether the data is valid and usable

        /**
         * @brief Default constructor for invalid data
         */
        VolumeAverageData() : volume(0.0), data(1), is_valid(false) {}

        /**
         * @brief Constructor for valid data
         * @param vol Total volume of the region
         * @param vec Volume-averaged data vector
         */
        VolumeAverageData(double vol, const mfem::Vector& vec)
            : volume(vol), data(vec), is_valid(true) {}
    };

private:
    /**
     * @brief Cache storage for volume average data
     *
     * Two-level map structure: m_region_cache[calc_type][region_id] = data
     * Enables O(1) lookup of cached region data during global calculations.
     * Cache is cleared each time step to ensure data freshness.
     */
    std::map<CalcType, std::map<int, VolumeAverageData>> m_region_cache;

    /**
     * @brief Convert string calculation type to enum
     *
     * @param calc_type_str String identifier for calculation type
     * @return Corresponding CalcType enum value
     *
     * Provides mapping from user-friendly string names to type-safe enums.
     * Used to interface between public string-based API and internal enum-based
     * implementation for improved performance and type safety.
     */
    CalcType GetCalcType(const std::string& calc_type_str);

    /**
     * @brief Calculate volume average for a specific region and calculation type
     *
     * @param calc_type Type of calculation to perform
     * @param region Region index to process
     * @return Volume average data containing volume and averaged quantities
     *
     * Core calculation method that handles all the complexity of:
     * - Selecting appropriate quadrature functions
     * - Processing state variables for derived quantities
     * - Handling special cases (deformation gradient global assignment)
     * - Performing volume integration using MFEM kernels
     *
     * This method encapsulates all calculation-specific logic and provides
     * a uniform interface for all volume averaging operations.
     */
EXACONSTIT_CUDA_PRIVATE:
    VolumeAverageData CalculateVolumeAverage(CalcType calc_type, int region);
private:

    /**
     * @brief Get cached data or calculate if not available
     *
     * @param calc_type Type of calculation
     * @param region Region index
     * @return Volume average data (from cache or newly calculated)
     *
     * Implements intelligent caching strategy:
     * 1. Check cache for existing valid data
     * 2. If found, return cached result (O(1) operation)
     * 3. If not found, calculate and cache result for future use
     *
     * This method optimizes performance for workflows that compute both
     * region-specific and global quantities by avoiding redundant calculations.
     */
    VolumeAverageData GetOrCalculateVolumeAverage(CalcType calc_type, int region);

    /**
     * @brief Registration structure for projection operations
     *
     * Contains all metadata and objects needed to manage a projection type
     * across multiple material regions with appropriate model compatibility
     * checking and dynamic enablement control.
     */
    struct ProjectionRegistration {
        /**
         * @brief Unique field identifier for this projection
         *
         * String key used for projection lookup and grid function naming.
         * Examples: "stress", "volume", "centroid", "elastic_strain"
         */
        std::string field_name;

        /**
         * @brief Human-readable display name
         *
         * User-friendly name for UI display and error messages.
         * Examples: "Cauchy Stress", "Element Volumes", "Crystal Orientations"
         */
        std::string display_name;

        /**
         * @brief Material model compatibility requirements
         *
         * Specifies which material model types support this projection.
         * Used during registration to avoid creating incompatible projections.
         */
        ProjectionTraits::ModelCompatibility model_compatibility;

        /**
         * @brief Per-region enablement flags
         *
         * Boolean vector indicating which regions have this projection enabled.
         * Size equals m_num_regions. Allows selective projection execution.
         */
        std::vector<bool> region_enabled;

        /**
         * @brief Projection class instances per region
         *
         * Vector of ProjectionBase-derived objects, one per region plus one
         * for global aggregation if supported. Handles actual projection execution.
         */
        std::vector<std::shared_ptr<ProjectionBase>> projection_class;

        /**
         * @brief Vector dimensions per region
         *
         * Stores the vector dimension (number of components) for this projection
         * in each region. Used for grid function creation and validation.
         */
        std::vector<int> region_length;

        /**
         * @brief Global aggregation support flag
         *
         * Indicates whether this projection type can be aggregated across regions
         * into a unified global field. Affects global grid function creation.
         */
        bool supports_global_aggregation = false;
    };

    /**
     * @brief Registration structure for volume averaging calculations
     *
     * Contains function pointers and metadata for volume averaging operations
     * that compute region-specific and globally aggregated scalar quantities
     * from quadrature function data.
     */
    struct VolumeAverageRegistration {
        /**
         * @brief Unique calculation identifier
         *
         * String key for the volume averaging calculation.
         * Examples: "stress", "def_grad", "plastic_work", "equivalent_plastic_strain"
         */
        std::string calc_name;

        /**
         * @brief Human-readable display name
         *
         * User-friendly name for output headers and error messages.
         * Examples: "Volume Average Stress", "Volume Plastic Work"
         */
        std::string display_name;

        /**
         * @brief Material model compatibility requirements
         *
         * Specifies which material models provide the required data for this
         * calculation. Most volume averages work with all models.
         */
        ProjectionTraits::ModelCompatibility model_compatibility;

        /**
         * @brief Per-region enablement flags
         *
         * Boolean vector indicating which regions should perform this calculation.
         * Enables selective volume averaging based on material properties.
         */
        std::vector<bool> region_enabled;

        /**
         * @brief Per-region calculation function
         *
         * Function pointer for region-specific volume averaging. Signature:
         * void(int region, double time). Called for each enabled region.
         */
        std::function<void(int, double)> region_func;

        /**
         * @brief Global aggregation function
         *
         * Function pointer for global volume averaging across all regions.
         * Signature: void(double time). Called once per output timestep.
         */
        std::function<void(double)> global_func;

        /**
         * @brief Global aggregation availability flag
         *
         * Indicates whether the global_func is available and should be called.
         * True when global_func is not nullptr during registration.
         */
        bool has_global_aggregation = true;
    };

    /**
     * @brief Register a volume averaging calculation function
     *
     * @param calc_name Unique identifier for the calculation
     * @param display_name Human-readable name for output
     * @param region_func Function for per-region calculations
     * @param global_func Function for global aggregation (optional)
     * @param enabled Default enablement state
     *
     * Adds a new volume averaging calculation to the registered calculations list.
     * The region_func is called for each enabled region, while global_func (if
     * provided) is called once per timestep for global aggregation.
     *
     * Used internally by RegisterDefaultVolumeCalculations() to set up
     * standard volume averaging operations like stress and strain averaging.
     */
    void RegisterVolumeAverageFunction(const std::string& calc_name,
                                       const std::string& display_name,
                                       std::function<void(const int, const double)> region_func,
                                       std::function<void(const double)> global_func = nullptr,
                                       bool enabled = true);

    /**
     * @brief Execute global projection for a specific field
     *
     * @param field_name Name of the field to project globally
     *
     * Performs global aggregation of field data across all regions.
     * Combines per-region data into unified global grid functions
     * for visualization and analysis. The aggregation method depends
     * on the field type (additive for extensive quantities,
     * volume-weighted for intensive quantities).
     *
     * Used internally when aggregation mode includes GLOBAL_COMBINED.
     */
    void ExecuteGlobalProjection(const std::string& field_name);

    /**
     * @brief Combine region data into global grid function
     *
     * @param field_name Name of the field to combine
     *
     * Transfers data from region-specific grid functions to the corresponding
     * global grid function using ParSubMesh::Transfer() operations. Accumulates
     * contributions from all active regions to create unified global fields
     * for visualization and analysis.
     */
    void CombineRegionDataToGlobal(const std::string& field_name);

    /**
     * @brief Initialize data collections for visualization output
     *
     * @param options Simulation options containing visualization settings
     *
     * Creates MFEM DataCollection objects (VisIt, ParaView, ADIOS2) based on
     * ExaOptions visualization settings. Data collections are created for:
     * - Each region when aggregation mode includes PER_REGION
     * - Global fields when aggregation mode includes GLOBAL_COMBINED
     *
     * Configures output directories, file formats, and precision settings
     * according to user preferences and registers all grid functions with
     * appropriate data collections.
     */
    void InitializeDataCollections(ExaOptions& options);

    /**
     * @brief Initialize grid functions for all registered projections
     *
     * Creates ParGridFunction objects for all enabled projections based on
     * the current aggregation mode. Grid functions are created for:
     * - Per-region projections when mode includes PER_REGION
     * - Global aggregated projections when mode includes GLOBAL_COMBINED
     *
     * Vector dimensions are determined from projection metadata and region
     * compatibility. All grid functions are initialized to zero and registered
     * with appropriate finite element spaces.
     */
    void InitializeGridFunctions();

    /**
     * @brief Check if a region has the required quadrature function
     *
     * @param field_name Name of the field/quadrature function to check
     * @param region Region index to query
     * @return true if the region has the specified quadrature function
     *
     * Queries SimulationState to determine if a specific region contains
     * the quadrature function data required for a projection or calculation.
     * Used to validate data availability before attempting projections.
     */
    bool RegionHasQuadratureFunction(const std::string& field_name, int region) const;

    /**
     * @brief Get all active regions for a given field
     *
     * @param field_name Name of the field to query
     * @return Vector of integers indicating which regions are active
     *
     * Returns a vector of boolean values (as integers) indicating which regions
     * have grid functions created for the specified field. Used for global
     * aggregation operations to determine which regions to combine.
     */
    std::vector<int> GetActiveRegionsForField(const std::string& field_name) const;

    /**
     * @brief Clear the volume average cache
     *
     * Should be called at the beginning of each time step to ensure cache
     * freshness and prevent stale data from affecting calculations. Also
     * prevents unbounded memory growth over long simulation runs.
     *
     * @note This method should be called before any volume averaging operations
     *       in a new time step to ensure data consistency.
     */
    void ClearVolumeAverageCache();

    /**
     * @brief Generic volume average calculation for region-specific output
     *
     * @param calc_type_str String identifier for calculation type
     * @param region Region index to process
     * @param time Current simulation time for output
     *
     * Unified interface for all region-specific volume averaging operations.
     * This method:
     * 1. Converts string type to enum for internal processing
     * 2. Calculates or retrieves cached volume average data
     * 3. Writes formatted output to appropriate file
     * 4. Caches result for potential reuse in global calculations
     *
     * Supports all calculation types through a single, well-tested code path.
     */
    void VolumeAverage(const std::string& calc_type_str, int region, double time);

    /**
     * @brief Generic global volume average calculation
     *
     * @param calc_type_str String identifier for calculation type
     * @param time Current simulation time for output
     *
     * Unified interface for all global volume averaging operations.
     * This method:
     * 1. Accumulates volume-weighted contributions from all regions
     * 2. Uses cached data when available to avoid redundant calculations
     * 3. Calculates missing region data on-demand
     * 4. Normalizes by total volume to compute global average
     * 5. Writes formatted output to global file
     *
     * The caching system makes this method highly efficient when region-specific
     * calculations have already been performed in the same time step.
     */
    void GlobalVolumeAverage(const std::string& calc_type_str, double time);

    /**
     * @brief Calculate and output volume-averaged stress for a specific region
     *
     * @param region Region index for calculation
     * @param time Current simulation time
     *
     * Computes volume-averaged Cauchy stress tensor (6 components) for the
     * specified region using element-averaged stress values. The calculation
     * performs true volume weighting by integrating stress over element volumes.
     *
     * Output format: Time, Total_Volume, Sxx, Syy, Szz, Sxy, Sxz, Syz
     *
     * Files are written only by MPI rank 0 to region-specific output files
     * managed by PostProcessingFileManager. Headers are added automatically
     * for new files.
     */
    void VolumeAvgStress(const int region, const double time);
    /**
     * @brief Volume-averaged Euler strain calculation for a region
     *
     * @param region Region index for calculation
     * @param time Current simulation time
     *
     * Computes volume-averaged Euler (engineering) strain tensor for the specified
     * region. Euler strain is computed as E = 0.5*(F^T*F - I) where F is the
     * deformation gradient. Output follows Voigt notation for symmetric tensors.
     */
    void VolumeAvgEulerStrain(const int region, const double time);
    /**
     * @brief Calculate and output volume-averaged deformation gradient for a region
     *
     * @param region Region index for calculation
     * @param time Current simulation time
     *
     * Computes volume-averaged deformation gradient tensor (9 components) for
     * the specified region. The deformation gradient captures finite strain
     * deformation including rotation and stretch components.
     *
     * Output format: Time, Total_Volume, F11, F12, F13, F21, F22, F23, F31, F32, F33
     *
     * Essential for finite strain analysis and strain path tracking in
     * large deformation simulations.
     */
    void VolumeAvgDefGrad(const int region, const double time);
    /**
     * @brief Volume-averaged plastic work calculation for a region
     *
     * @param region Region index for calculation
     * @param time Current simulation time
     *
     * Computes volume-averaged plastic work (scalar) for the specified region.
     * Plastic work represents energy dissipated through irreversible deformation
     * and is essential for energy balance analysis in thermomechanical problems.
     */
    void VolumePlWork(const int region, const double time);
    /**
     * @brief Calculate and output volume-averaged equivalent plastic strain for a region
     *
     * @param region Region index for calculation
     * @param time Current simulation time
     *
     * Computes volume-averaged equivalent plastic strain (scalar) for the
     * specified region. Equivalent plastic strain provides a scalar measure
     * of accumulated plastic deformation magnitude.
     *
     * Output format: Time, Total_Volume, Equivalent_Plastic_Strain
     *
     * Critical for strain-based failure criteria and plastic strain
     * accumulation tracking in fatigue and damage analysis.
     */
    void VolumeEPS(const int region, const double time);
    /**
     * @brief Calculate and output volume-averaged elastic strain for a region
     *
     * @param region Region index for calculation
     * @param time Current simulation time
     *
     * Computes volume-averaged elastic strain tensor (6 components) for
     * ExaCMech regions. Elastic strain represents recoverable deformation
     * and is essential for stress-strain relationship analysis.
     *
     * Output format: Time, Total_Volume, Ee11, Ee22, Ee33, Ee23, Ee13, Ee12
     *
     * Only available for ExaCMech material models that explicitly track
     * elastic strain state variables.
     */
    void VolumeAvgElasticStrain(const int region, const double time);

    /**
     * @brief Calculate and output global volume-averaged stress
     *
     * @param time Current simulation time
     *
     * Computes global volume-averaged stress by combining contributions from
     * all regions with proper volume weighting. Each region's contribution
     * is weighted by its total volume before summing and normalizing.
     *
     * The global calculation provides homogenized stress response across
     * all material regions for macroscopic analysis and comparison with
     * experimental data.
     */
    void GlobalVolumeAvgStress(const double time);
    /**
     * @brief Global volume-averaged Euler strain calculation
     *
     * @param time Current simulation time
     *
     * Computes global Euler strain by volume-weighted averaging across all regions.
     * Provides macroscopic strain response for comparison with experimental data
     * and validation of material model predictions.
     */
    void GlobalVolumeAvgEulerStrain(const double time);
    /**
     * @brief Calculate and output global volume-averaged deformation gradient
     *
     * @param time Current simulation time
     *
     * Computes global volume-averaged deformation gradient by combining
     * region contributions with volume weighting. The global deformation
     * gradient represents overall specimen deformation for comparison
     * with experimental displacement boundary conditions.
     */
    void GlobalVolumeAvgDefGrad(const double time);
    /**
     * @brief Global volume-averaged plastic work calculation
     *
     * @param time Current simulation time
     *
     * Computes global plastic work by volume-weighted averaging across all regions.
     * Provides total energy dissipation for specimen-level energy balance and
     * thermomechanical analysis applications.
     */
    void GlobalVolumePlWork(const double time);
    /**
     * @brief Calculate and output global volume-averaged equivalent plastic strain
     *
     * @param time Current simulation time
     *
     * Computes global equivalent plastic strain by volume-weighted averaging
     * across all regions. Provides overall plastic strain accumulation for
     * macroscopic material characterization and model validation.
     */
    void GlobalVolumeEPS(const double time);
    /**
     * @brief Calculate and output global volume-averaged elastic strain
     *
     * @param time Current simulation time
     *
     * Computes global elastic strain by volume-weighted averaging across
     * ExaCMech regions. Provides macroscopic elastic response for
     * elasticity analysis and unloading behavior characterization.
     */
    void GlobalVolumeAvgElasticStrain(const double time);

    /**
     * @brief Calculate element-averaged values from partial quadrature function
     *
     * @param elemVal Output partial quadrature function for element averages
     * @param qf Input partial quadrature function with quadrature point data
     *
     * Computes volume-weighted element averages from quadrature point data.
     * The algorithm integrates quadrature point values over each element using
     * integration weights and Jacobian determinants, then normalizes by element
     * volume to produce true element averages.
     *
     * Essential for converting quadrature point data to element-constant data
     * suitable for visualization and further processing operations.
     */
EXACONSTIT_CUDA_PRIVATE:
    void CalcElementAvg(mfem::expt::PartialQuadratureFunction* elemVal,
                        const mfem::expt::PartialQuadratureFunction* qf);

    /**
     * @brief Calculate global element averages across all regions
     *
     * @param elemVal Output global vector for element averages
     * @param field_name Name of the field to average
     *
     * Computes element averages for a field across all regions and combines
     * them into a single global vector. Each region's contribution is calculated
     * using CalcElementAvg() and then mapped to the appropriate global element
     * indices using partial-to-global mappings.
     *
     * Used for global aggregation operations and cross-region analysis.
     */
    void CalcGlobalElementAvg(mfem::Vector* elemVal, const std::string& field_name);
private:

    /**
     * @brief Get the size of quadrature functions
     *
     * @return Size of quadrature functions in the simulation
     *
     * Returns the total number of quadrature points across all elements
     * by querying one of the available quadrature functions. Used for
     * memory allocation and loop bounds in global calculations.
     */
    size_t GetQuadratureFunctionSize() const;

    /**
     * @brief Generate standardized grid function names
     *
     * @param field_name Base field name
     * @param region Region index (-1 for global)
     * @return Formatted grid function name
     *
     * Creates consistent grid function names following the patterns:
     * - "field_name_region_X" for region-specific functions
     * - "field_name_global" for global aggregated functions
     *
     * Ensures consistent naming across all grid function operations
     * and enables proper lookup in the m_map_gfs container.
     */
    std::string GetGridFunctionName(const std::string& field_name, int region = -1) const;

    /**
     * @brief Update all field projections for current time step
     *
     * @param step Current time step number
     * @param time Current simulation time
     *
     * Executes all registered projections based on the current aggregation mode.
     * Updates both per-region and global fields depending on configuration.
     * This method handles the core projection pipeline including:
     * - Element-averaged value computation from quadrature data
     * - Region-specific projection execution
     * - Global field aggregation when enabled
     *
     * @note Called internally by Update() before visualization updates
     */
    void UpdateFields(const int step, const double time);

    /**
     * @brief Register default set of projections based on material models
     *
     * Automatically registers standard projections based on the material model
     * types present in each region. This includes:
     * - Geometry projections (centroid, volume) for all regions
     * - Stress projections (Cauchy, Von Mises, hydrostatic) for all regions
     * - State variable projections for compatible material models
     * - ECMech-specific projections for ExaCMech regions
     *
     * Registration is conditional on material model compatibility and
     * availability of required quadrature data in each region.
     */
    void RegisterDefaultProjections();

    /**
     * @brief Register default volume average calculations
     *
     * Sets up standard volume averaging operations including:
     * - Stress tensor averaging (per-region and global)
     * - Deformation gradient averaging
     * - Plastic work averaging
     * - Equivalent plastic strain averaging
     * - Elastic strain averaging (ExaCMech only)
     *
     * Each calculation is registered with both per-region and global
     * aggregation functions when applicable.
     */
    void RegisterDefaultVolumeCalculations();

    /**
     * @brief Register a specific projection by field name
     *
     * @param field Name of the field to register for projection
     *
     * Dynamically adds a projection for the specified field name.
     * The projection type is determined automatically based on:
     * - Field name matching known projection types
     * - Material model compatibility in each region
     * - Availability of required quadrature data
     *
     * Supports both built-in projections and custom field names
     * with automatic ECMech state variable detection.
     */
    void RegisterProjection(const std::string& field);

    /**
     * @brief Initialize LightUp analysis instances
     *
     * Creates and configures LightUp analysis objects based on the
     * light_up_configs specified in ExaOptions. Each enabled LightUp
     * configuration is instantiated with:
     * - Specified HKL directions for lattice strain calculations
     * - Distance tolerance for peak detection
     * - Sample direction for orientation reference
     * - Region-specific quadrature spaces and data
     *
     * LightUp instances are created only for regions with enabled
     * configurations and compatible material models (typically ExaCMech).
     */
    void InitializeLightUpAnalysis();
    /**
     * @brief Update LightUp analysis for all configured instances
     *
     * Executes lattice strain calculations for all active LightUp instances.
     * Each instance processes state variables and stress data for its
     * assigned region to compute:
     * - Lattice strains for specified HKL directions
     * - Directional stiffness properties
     * - Taylor factors and plastic strain rates
     *
     * Results are written to region-specific output files for post-processing
     * and comparison with experimental diffraction data.
     */
    void UpdateLightUpAnalysis();

private:
    /**
     * @brief Reference to simulation state for data access
     *
     * Provides access to all simulation data including quadrature functions,
     * mesh information, material properties, and state variables across all regions.
     */
    std::shared_ptr<SimulationState> m_sim_state;

    /**
     * @brief MPI rank of current process
     *
     * Used for controlling parallel I/O operations and ensuring only rank 0
     * performs file writing operations to avoid race conditions.
     */
    int m_mpi_rank;

    /**
     * @brief Total number of MPI processes
     *
     * Total count of parallel processes for coordination of distributed
     * post-processing operations and resource allocation.
     */
    int m_num_mpi_rank;

    /**
     * @brief Material model types for each region
     *
     * Vector containing the material model type (EXACMECH, UMAT, etc.) for
     * each material region. Used to determine projection compatibility and
     * enable appropriate post-processing operations per region.
     */
    std::vector<MechType> m_region_model_types;

    /**
     * @brief Total number of material regions
     *
     * Count of distinct material regions in the simulation. Determines the
     * number of region-specific projections and volume averaging operations.
     */
    size_t m_num_regions;

    /**
     * @brief Current aggregation mode for multi-region processing
     *
     * Controls whether to process regions separately (PER_REGION), combine
     * into global fields (GLOBAL_COMBINED), or both (BOTH). Affects which
     * grid functions and data collections are created and updated.
     */
    AggregationMode m_aggregation_mode;

    /**
     * @brief Buffer for element-averaged values per region
     *
     * Vector of partial quadrature functions used as temporary storage for
     * element-averaged calculations. One buffer per region plus one for
     * global aggregation operations.
     */
    std::vector<std::unique_ptr<mfem::expt::PartialQuadratureFunction>> m_region_evec;

    /**
     * @brief Global element vector for aggregated calculations
     *
     * MFEM Vector used for storing global element-averaged data when
     * combining results from multiple regions. Provides unified storage
     * for cross-region calculations and global aggregation operations.
     */
    std::unique_ptr<mfem::Vector> m_global_evec;

    /**
     * @brief File manager for ExaOptions-compliant output
     *
     * Handles all file I/O operations including directory creation, filename
     * generation, and output frequency control. Ensures consistent file
     * organization and naming conventions across all post-processing output.
     */
    std::unique_ptr<PostProcessingFileManager> m_file_manager;

    /**
     * @brief Nested map for finite element spaces by region and vector dimension
     *
     * Two-level map: outer key is region index, inner key is vector dimension.
     * Stores finite element spaces for different combinations of material regions
     * and field vector dimensions. Enables efficient reuse of compatible spaces.
     */
    std::map<int, std::map<int, std::shared_ptr<mfem::ParFiniteElementSpace>>> m_map_pfes;

    /**
     * @brief Submesh storage for region-specific visualization
     *
     * Maps region index to corresponding ParSubMesh objects. Each submesh
     * contains only the elements belonging to a specific material region,
     * enabling region-specific visualization and data processing.
     */
    std::map<int, std::shared_ptr<mfem::ParMesh>> m_map_submesh;

    /**
     * @brief Mapping from partial quadrature space to submesh elements
     *
     * Maps region index to element index mapping arrays. Provides translation
     * between local element indices in partial quadrature spaces and global
     * element indices in the corresponding submesh.
     */
    std::map<int, mfem::Array<int>> m_map_pqs2submesh;

    /**
     * @brief Grid function storage for all projections
     *
     * Maps grid function names to ParGridFunction objects. Names follow the
     * pattern "field_region_X" for region-specific or "field_global" for
     * aggregated functions. Stores all projected data for visualization.
     */
    std::map<std::string, std::shared_ptr<mfem::ParGridFunction>> m_map_gfs;

    /**
     * @brief Data collection storage for visualization output
     *
     * Maps data collection keys to MFEM DataCollection objects (VisIt, ParaView,
     * ADIOS2). Keys follow patterns like "visit_region_X" or "paraview_global".
     * Manages all visualization output formats and their associated data.
     */
    std::map<std::string, std::unique_ptr<mfem::DataCollection>> m_map_dcs;

    /**
     * @brief Registered projection operations
     *
     * Vector of ProjectionRegistration structures containing metadata and
     * instances for all registered projection operations. Enables dynamic
     * projection management and execution based on field names and regions.
     */
    std::vector<ProjectionRegistration> m_registered_projections;

    /**
     * @brief Registered volume averaging calculations
     *
     * Vector of VolumeAverageRegistration structures containing function
     * pointers and metadata for volume averaging operations. Supports both
     * per-region and global aggregated calculations.
     */
    std::vector<VolumeAverageRegistration> m_registered_volume_calcs;

    /**
     * @brief Visualization enablement flag
     *
     * Controls whether visualization-related operations are performed.
     * When false, grid functions and data collections are not created,
     * reducing memory usage for simulations that only need volume averaging.
     */
    bool m_enable_visualization;

    /**
     * @brief Visualization output frequency (every N time steps)
     *
     * Taken from the visualization options and kept separate from the
     * volume average output frequency used by the file manager.
     */
    int m_viz_output_frequency;

    /**
     * @brief Active LightUp analysis instances
     *
     * Vector of LightUp objects for lattice strain analysis. Each instance
     * corresponds to an enabled LightUp configuration from ExaOptions,
     * providing in-situ diffraction simulation capabilities.
     */
    std::vector<std::unique_ptr<LightUp>> m_light_up_instances;
};
