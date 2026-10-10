#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "TOML_Reader/toml.hpp"

// Enumeration types
/**
 * @brief Enumeration for different mesh generation types
 */
enum class MeshType {
    AUTO,  /**< Automatically generated structured mesh */
    FILE,  /**< Mesh loaded from external file */
    NOTYPE /**< Uninitialized or invalid mesh type */
};

/**
 * @brief Enumeration for time stepping strategies
 */
enum class TimeStepType {
    FIXED,  /**< Fixed time step size */
    AUTO,   /**< Adaptive time stepping */
    CUSTOM, /**< Custom time steps from file */
    NOTYPE  /**< Uninitialized or invalid time step type */
};

/**
 * @brief Enumeration for crystal orientation representation types
 */
enum class OriType {
    EULER,  /**< Euler angles representation */
    QUAT,   /**< Quaternion representation */
    CUSTOM, /**< Custom orientation format */
    NOTYPE  /**< Uninitialized or invalid orientation type */
};

/**
 * @brief Enumeration for material mechanics model types
 */
enum class MechType {
    UMAT,     /**< User-defined material subroutine */
    EXACMECH, /**< ExaCMech crystal plasticity model */
    NOTYPE    /**< Uninitialized or invalid mechanics type */
};

/**
 * @brief Enumeration for runtime execution models
 */
enum class RTModel {
    CPU,    /**< CPU-only execution */
    OPENMP, /**< OpenMP parallel execution */
    GPU,    /**< GPU accelerated execution */
    NOTYPE  /**< Uninitialized or invalid runtime model */
};

/**
 * @brief Enumeration for finite element assembly strategies
 */
enum class AssemblyType {
    FULL,  /**< Full matrix assembly */
    PA,    /**< Partial assembly */
    EA,    /**< Element assembly */
    NOTYPE /**< Uninitialized or invalid assembly type */
};

/**
 * @brief Enumeration for integration models
 */
enum class IntegrationModel {
    DEFAULT, /**< Standard integration */
    BBAR,    /**< B-bar method for incompressible materials */
    NOTYPE   /**< Uninitialized or invalid integration model */
};

/**
 * @brief Enumeration for linear solver types
 */
enum class LinearSolverType {
    CG,       /**< Conjugate Gradient solver */
    GMRES,    /**< Generalized Minimal Residual solver */
    MINRES,   /**< Minimal Residual solver */
    BICGSTAB, /**< BiCGSTAB Solver */
    NOTYPE    /**< Uninitialized or invalid linear solver type */
};

/**
 * @brief Enumeration for nonlinear solver types
 */
enum class NonlinearSolverType {
    NR,    /**< Newton-Raphson method */
    NRLS,  /**< Newton-Raphson with line search */
    TRDOG, /**< Trust-region dogleg method (ported from SNLS) */
    NOTYPE /**< Uninitialized or invalid nonlinear solver type */
};

/**
 * @brief Enumeration for preconditioner types
 */
enum class PreconditionerType {
    JACOBI,    /**< Jacobi preconditioner */
    AMG,       /**< Algebraic multigrid preconditioner (Full assembly only) */
    ILU,       /**< Incomplete LU factorization preconditioner (Full assembly only) */
    L1GS,      /**< l1-scaled block Gauss-Seidel/SSOR preconditioner (Full assembly only) */
    CHEBYSHEV, /**< Chebyshev preconditioner (Full assembly only) */
    NOTYPE     /**< Uninitialized or invalid preconditioner type */
};

enum class LatticeType {
    CUBIC,
    HEXAGONAL,
    TRIGONAL,
    RHOMBOHEDRAL,
    TETRAGONAL,
    ORTHORHOMBIC,
    MONOCLINIC,
    TRICLINIC
};

/**
 * @brief Type alias for a nested unordered map structure used for boundary condition mapping
 *
 * This type represents a map where:
 * - Key (string): Boundary condition type identifier (e.g., "total", "ess_vgrad")
 * - Value: Map where:
 *   - Key (int): Time step or cycle number
 *   - Value (vector<int>): List of boundary condition IDs or components for that step
 */
using map_of_imap = std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>;

/**
 * @brief Mesh configuration info
 */
struct MeshOptions {
    /**
     * @brief Type of mesh generation strategy to use
     */
    MeshType mesh_type = MeshType::FILE;

    /**
     * @brief Path to external mesh file (required when mesh_type = FILE)
     */
    std::filesystem::path mesh_file;

    /**
     * @brief Number of elements in each direction [nx, ny, nz] for auto-generated mesh
     */
    std::array<int, 3> nxyz = {1, 1, 1};

    /**
     * @brief Physical domain size in each direction [x, y, z] for auto-generated mesh
     */
    std::array<double, 3> mxyz = {1.0, 1.0, 1.0};

    /**
     * @brief Number of serial refinement levels to apply to mesh
     */
    int ref_ser = 0;

    /**
     * @brief Number of parallel refinement levels to apply to mesh
     */
    int ref_par = 0;

    /**
     * @brief Polynomial order for finite element basis functions
     */
    int order = 1;

    /**
     * @brief Whether to enforce periodic boundary conditions
     */
    bool periodicity = false;

    // Validation
    bool validate() const;

    // Conversion from toml
    static MeshOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Grain information for crystal plasticity models
 */
struct GrainInfo {
    /**
     * @brief Optional file path containing grain orientation data
     */
    std::optional<std::filesystem::path> orientation_file;

    /**
     * @brief Optional file path containing grain ID mapping data
     */
    std::optional<std::filesystem::path> grain_file;

    /**
     * @brief Location of orientation data within state variables array
     */
    int ori_state_var_loc = -1;

    /**
     * @brief Stride for accessing orientation data in state variables
     */
    int ori_stride = 0;

    /**
     * @brief Type of orientation representation (Euler, quaternion, custom)
     */
    OriType ori_type = OriType::QUAT;

    /**
     * @brief Total number of grains in the simulation
     */
    int num_grains = 0;

    // Validation
    bool validate() const;

    // Conversion from toml
    static GrainInfo from_toml(const toml::value& toml_input);
};

/**
 * @brief Material properties configuration
 */
struct MaterialProperties {
    /**
     * @brief File path containing material property values
     */
    std::filesystem::path properties_file;

    /**
     * @brief Number of material properties expected
     */
    int num_props = 0;

    /**
     * @brief Vector of material property values
     */
    std::vector<double> properties;

    // Validation
    bool validate() const;

    // Conversion from toml
    static MaterialProperties from_toml(const toml::value& toml_input);
};

/**
 * @brief State variables configuration
 */
struct StateVariables {
    /**
     * @brief File path containing initial state variable values
     */
    std::filesystem::path state_file;

    /**
     * @brief Number of state variables per integration point
     */
    int num_vars = 0;

    /**
     * @brief Initial values for state variables
     */
    std::vector<double> initial_values;

    // Validation
    bool validate() const;

    // Conversion from toml
    static StateVariables from_toml(const toml::value& toml_input);
};

/**
 * @brief UMAT-specific options
 */
struct UmatOptions {
    /**
     * @brief Path to the UMAT library file
     */
    std::filesystem::path library_path;

    /**
     * @brief Name of the UMAT function to call (default: "umat_call")
     */
    std::string function_name = "umat_call";

    /**
     * @brief Whether thermal effects are enabled
     */
    bool thermal = false;

    /**
     * @brief Strategy for loading the library ("persistent", "load_on_setup", "lazy_load")
     */
    std::string load_strategy = "persistent";

    /**
     * @brief Whether dynamic loading is enabled
     */
    bool enable_dynamic_loading = true;

    /**
     * @brief Additional search paths for UMAT libraries
     */
    std::vector<std::filesystem::path> search_paths;

    /**
     * @brief Validates if the load strategy is one of the accepted values
     * @return true if load_strategy is valid, false otherwise
     */
    bool is_valid_load_strategy() const;

    // Validation
    bool validate() const;

    // Conversion from toml
    static UmatOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief ExaCMech-specific options
 */
struct ExaCMechModelOptions {
    /**
     * @brief Direct shortcut specification for ExaCMech model
     */
    std::string shortcut;

    /**
     * @brief Size of slip rate tensor
     */
    size_t gdot_size = 0;

    /**
     * @brief Size of hardening matrix
     */
    size_t hard_size = 0;

    /**
     * @brief Crystal type (FCC, BCC, or HCP) - legacy approach
     */
    std::string xtal_type;

    /**
     * @brief Slip type (PowerVoce, PowerVoceNL, or MTSDD) - legacy approach
     */
    std::string slip_type;

    /**
     * @brief Get the effective shortcut name (either directly specified or derived from legacy
     * fields)
     * @return The shortcut string to use for ExaCMech
     */
    std::string get_effective_shortcut() const;

    // Validation
    bool validate() const;

    // Static conversion from TOML
    static ExaCMechModelOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Material model options
 */
struct MaterialModelOptions {
    /**
     * @brief Whether crystal plasticity is enabled for this material
     */
    bool crystal_plasticity = true;

    /**
     * @brief UMAT-specific configuration options (if using UMAT)
     */
    std::optional<UmatOptions> umat;

    /**
     * @brief ExaCMech-specific configuration options (if using ExaCMech)
     */
    std::optional<ExaCMechModelOptions> exacmech;

    // Validation
    bool validate() const;

    // Conversion from toml
    static MaterialModelOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Material options for a specific material/region
 */
struct MaterialOptions {
    /**
     * @brief Descriptive name for this material
     */
    std::string material_name = "default";

    /**
     * @brief Region/material attribute ID associated with this material
     */
    int region_id = 1;

    /**
     * @brief Type of mechanics model to use for this material
     */
    MechType mech_type = MechType::NOTYPE;

    /**
     * @brief Material property configuration
     */
    MaterialProperties properties;

    /**
     * @brief State variable configuration
     */
    StateVariables state_vars;

    /**
     * @brief Grain information (required for crystal plasticity)
     */
    std::optional<GrainInfo> grain_info;

    /**
     * @brief Model-specific configuration options
     */
    MaterialModelOptions model;

    /**
     * @brief Operating temperature in Kelvin
     */
    double temperature = 298.0;

    // Validation
    bool validate() const;

    // Conversion from toml
    static MaterialOptions from_toml(const toml::value& toml_input);

    /**
     * @brief Parse an array of materials from TOML input
     * @param toml_input TOML value containing material array or single material
     * @return Vector of MaterialOptions parsed from the input
     */
    static std::vector<MaterialOptions> from_toml_array(const toml::value& toml_input);
};

/**
 * @brief Time stepping configuration
 */
struct TimeOptions {
    /**
     * @brief Type of time stepping strategy being used
     */
    TimeStepType time_type = TimeStepType::FIXED;

    /**
     * @brief Auto time stepping options
     */
    struct AutoTimeOptions {
        /**
         * @brief Initial time step size for adaptive stepping
         */
        double dt_start = 0.1;

        /**
         * @brief Minimum allowed time step size
         */
        double dt_min = 0.05;

        /**
         * @brief Maximum allowed time step size
         */
        double dt_max = 1e9;

        /**
         * @brief Scaling factor for time step adjustment
         */
        double dt_scale = 0.25;

        /**
         * @brief Final simulation time
         */
        double t_final = 1.0;

        static AutoTimeOptions from_toml(const toml::value& toml_input);
    };

    /**
     * @brief Fixed time stepping options
     */
    struct FixedTimeOptions {
        /**
         * @brief Fixed time step size for uniform stepping
         */
        double dt = 1.0;

        /**
         * @brief Final simulation time
         */
        double t_final = 1.0;

        static FixedTimeOptions from_toml(const toml::value& toml_input);
    };

    /**
     * @brief Custom time stepping options
     */
    struct CustomTimeOptions {
        /**
         * @brief Number of time steps to take
         */
        int nsteps = 1;

        /**
         * @brief File path containing custom time step values
         */
        std::filesystem::path floc = "custom_dt.txt";

        /**
         * @brief Vector of time step values loaded from file
         */
        std::vector<double> dt_values;

        /**
         * @brief Load custom time step values from file
         * @return true if successful, false if file couldn't be loaded
         */
        bool load_custom_dt_values();

        static CustomTimeOptions from_toml(const toml::value& toml_input);
    };

    /**
     * @brief Auto time stepping configuration (if using AUTO mode)
     */
    std::optional<AutoTimeOptions> auto_time;

    /**
     * @brief Fixed time stepping configuration (if using FIXED mode)
     */
    std::optional<FixedTimeOptions> fixed_time;

    /**
     * @brief Custom time stepping configuration (if using CUSTOM mode)
     */
    std::optional<CustomTimeOptions> custom_time;

    /**
     * @brief Whether this is a restart simulation
     */
    bool restart = false;

    /**
     * @brief Time to restart from (if restart is enabled)
     */
    double restart_time = 0.0;

    /**
     * @brief Cycle number to restart from (if restart is enabled)
     */
    size_t restart_cycle = 0;

    /**
     * @brief Determine which time stepping mode is active based on priority (Custom > Auto > Fixed)
     */
    void determine_time_type();

    // Static conversion from TOML
    static TimeOptions from_toml(const toml::value& toml_input);

    // Validation
    bool validate();
};

/**
 * @brief Linear solver configuration
 */
struct LinearSolverOptions {
    /**
     * @brief Type of iterative linear solver to use
     */
    LinearSolverType solver_type = LinearSolverType::CG;

    /**
     * @brief Preconditioner type for linear solver acceleration
     */
    PreconditionerType preconditioner = PreconditionerType::AMG;

    /**
     * @brief Absolute convergence tolerance for linear solver
     */
    double abs_tol = 1e-30;

    /**
     * @brief Relative convergence tolerance for linear solver
     */
    double rel_tol = 1e-10;

    /**
     * @brief Maximum number of linear solver iterations
     */
    int max_iter = 1000;

    /**
     * @brief Verbosity level for linear solver output (0 = silent)
     */
    int print_level = 0;

    // Validation
    bool validate() const;

    // Conversion from toml
    static LinearSolverOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Trust-region dogleg solver configuration
 *
 * @details Controls the trust-region radius management and dogleg step
 * computation for the ExaTrustRegionSolver. Parameters are ported from
 * SNLS's TrDeltaControl with sane defaults suitable for solid mechanics
 * applications. Power users can tune these for difficult crystal plasticity
 * problems.
 *
 * The trust-region radius delta is updated based on the ratio
 *     rho = actual_residual_change / predicted_residual_change
 * where predicted change comes from the linearized model at the current iterate.
 *
 * Acceptance/rejection bands:
 *   - "Good" band [xi_lg, xi_ug]: increase delta when rho falls here
 *   - "OK"  band [xi_lo, xi_uo]: keep delta when rho falls here (outside good)
 *   - Outside [xi_lo, xi_uo]: decrease delta
 *
 * TOML configuration example:
 * @code
 * [Solvers.NR.trust_region]
 *     delta_init      = 1.0
 *     delta_min       = 1e-12
 *     delta_max       = 1e4
 *     xi_lg           = 0.75
 *     xi_ug           = 1.4
 *     xi_lo           = 0.35
 *     xi_uo           = 5.0
 *     xi_inc          = 1.5
 *     xi_dec          = 0.25
 *     xi_forced_inc   = 1.2
 *     reject_increase = true
 * @endcode
 */
struct TrustRegionOptions {
    /**
     * @brief Initial trust-region radius
     */
    double delta_init = 1.0;

    /**
     * @brief Minimum allowed trust-region radius. Solver fails if delta drops below this.
     */
    double delta_min = 1e-12;

    /**
     * @brief Maximum allowed trust-region radius
     */
    double delta_max = 1e4;

    /**
     * @brief Lower bound of the "good" rho band (increase delta when rho > xi_lg)
     */
    double xi_lg = 0.75;

    /**
     * @brief Upper bound of the "good" rho band
     */
    double xi_ug = 1.4;

    /**
     * @brief Lower bound of the "ok" rho band (decrease delta when rho < xi_lo)
     */
    double xi_lo = 0.35;

    /**
     * @brief Upper bound of the "ok" rho band (decrease delta when rho > xi_uo)
     */
    double xi_uo = 5.0;

    /**
     * @brief Factor used to increase delta when a step is accepted in the "good" band
     */
    double xi_inc = 1.5;

    /**
     * @brief Factor used to decrease delta when a step quality is outside the "ok" band
     */
    double xi_dec = 0.25;

    /**
     * @brief Forced-increase factor when the predicted residual change is exactly zero
     */
    double xi_forced_inc = 1.2;

    /**
     * @brief Whether to reject steps that increase the residual norm
     */
    bool reject_increase = true;

    // Validation
    bool validate() const;

    // Conversion from toml
    static TrustRegionOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Nonlinear solver configuration
 */
struct NonlinearSolverOptions {
    /**
     * @brief Maximum number of nonlinear iterations per time step
     */
    int iter = 25;

    /**
     * @brief Relative convergence tolerance for nonlinear solver
     */
    double rel_tol = 1e-5;

    /**
     * @brief Absolute convergence tolerance for nonlinear solver
     */
    double abs_tol = 1e-10;

    /**
     * @brief Type of nonlinear solver algorithm to use
     */
    NonlinearSolverType nl_solver = NonlinearSolverType::NR;

    /**
     * @brief Verbosity of the nonlinear solver output
     *
     * - 0: only a one line summary (number of iterations, final residual, and final
     *      relative residual) once the solver converges or fails
     * - 1: residual and step scale information for every iteration (default)
     * - 2+: also prints detailed per-iteration information for solvers that have it
     *       (currently the trust-region dogleg solver)
     */
    int print_info = 1;

    /**
     * @brief Trust-region configuration (only used when nl_solver == TRDOG).
     *
     * If left empty, default TrustRegionOptions values are used. Users with
     * difficult convergence problems should provide custom values.
     */
    std::optional<TrustRegionOptions> trust_region;

    // Validation
    bool validate() const;

    // Conversion from toml
    static NonlinearSolverOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Global solver configuration
 */
struct SolverOptions {
    /**
     * @brief Finite element assembly strategy for matrix construction
     */
    AssemblyType assembly = AssemblyType::FULL;

    /**
     * @brief Runtime execution model for computations
     */
    RTModel rtmodel = RTModel::CPU;

    /**
     * @brief Integration model for handling material nonlinearities
     */
    IntegrationModel integ_model = IntegrationModel::DEFAULT;

    /**
     * @brief Configuration for iterative linear solver
     */
    LinearSolverOptions linear_solver;

    /**
     * @brief Configuration for nonlinear Newton-Raphson solver
     */
    NonlinearSolverOptions nonlinear_solver;

    // Validation
    bool validate();

    // Conversion from toml
    static SolverOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Time-dependent boundary condition configuration
 */
struct BCTimeInfo {
    /**
     * @brief Whether boundary conditions vary with simulation time
     */
    bool time_dependent = false;

    /**
     * @brief Whether boundary conditions vary with simulation cycle number
     */
    bool cycle_dependent = false;

    /**
     * @brief Time values for time-dependent boundary condition updates
     */
    std::vector<double> times;

    /**
     * @brief Cycle numbers for cycle-dependent boundary condition updates
     */
    std::vector<int> cycles;

    // Validation
    bool validate() const;

    // Conversion from toml
    static BCTimeInfo from_toml(const toml::value& toml_input);
};

/**
 * @brief Velocity boundary condition
 */
struct VelocityBC {
    /**
     * @brief Node or boundary attribute IDs where velocity BCs are applied
     */
    std::vector<int> essential_ids;

    /**
     * @brief Component indices (0=x, 1=y, 2=z) for velocity constraints
     */
    std::vector<int> essential_comps;

    /**
     * @brief Prescribed velocity values corresponding to essential_comps
     */
    std::vector<double> essential_vals;

    // Validation
    bool validate() const;

    // Conversion from toml
    static VelocityBC from_toml(const toml::value& toml_input);
};

/**
 * @brief Velocity gradient boundary condition
 */
struct VelocityGradientBC {
    /**
     * @brief Velocity gradient tensor components (stored as flattened 3x3 matrix)
     */
    std::vector<double> velocity_gradient;

    /**
     * @brief Component IDs for this boundary condition
     */
    std::vector<int> essential_comps;

    /**
     * @brief Node/element IDs where this boundary condition applies
     */
    std::vector<int> essential_ids;

    /**
     * @brief Time-dependent information for this boundary condition
     */
    BCTimeInfo time_info;

    /**
     * @brief Origin point for velocity gradient application
     */
    std::optional<std::array<double, 3>> origin;

    // Validation
    bool validate() const;

    // Conversion from toml
    static VelocityGradientBC from_toml(const toml::value& toml_input);
};

/**
 * @brief Legacy boundary condition format support for backward compatibility
 */
struct LegacyBC {
    /**
     * @brief Whether boundary conditions change over time
     */
    bool changing_ess_bcs = false;

    /**
     * @brief Experimental feature monotonic z-loading BCs better
     * single crystal simulations
     */
    bool mono_def_bcs = false;

    /**
     * @brief Time steps at which boundary conditions are updated
     */
    std::vector<int> update_steps = {1};

    /**
     * @brief Essential boundary condition node IDs
     * Can be either flat vector (constant BCs) or nested vector (time-dependent BCs)
     */
    std::variant<std::vector<int>, std::vector<std::vector<int>>> essential_ids;

    /**
     * @brief Essential boundary condition component IDs
     * Can be either flat vector (constant BCs) or nested vector (time-dependent BCs)
     */
    std::variant<std::vector<int>, std::vector<std::vector<int>>> essential_comps;

    /**
     * @brief Essential boundary condition values
     * Can be either flat vector (constant BCs) or nested vector (time-dependent BCs)
     */
    std::variant<std::vector<double>, std::vector<std::vector<double>>> essential_vals;

    /**
     * @brief Essential velocity gradient values
     * Can be either double-nested (constant BCs) or triple-nested (time-dependent BCs)
     */
    std::variant<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>>
        essential_vel_grad;

    /**
     * @brief Origin point for velocity gradient boundary conditions
     */
    std::vector<double> vgrad_origin = {0.0, 0.0, 0.0};
};

/**
 * @brief Boundary conditions configuration
 */
struct BoundaryOptions {
    /**
     * @brief Modern structured velocity boundary conditions
     */
    std::vector<VelocityBC> velocity_bcs;

    /**
     * @brief Modern structured velocity gradient boundary conditions
     */
    std::vector<VelocityGradientBC> vgrad_bcs;

    /**
     * @brief Legacy format support for direct compatibility
     */
    LegacyBC legacy_bcs;

    /**
     * @brief Type alias for nested boundary condition mapping
     */
    using map_of_imap = std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>;

    /**
     * @brief Maps time steps to velocity values for BCManager compatibility
     */
    std::unordered_map<int, std::vector<double>> map_ess_vel;

    /**
     * @brief Maps time steps to velocity gradient values for BCManager compatibility
     */
    std::unordered_map<int, std::vector<double>> map_ess_vgrad;

    /**
     * @brief Maps BC types and time steps to component IDs for BCManager compatibility
     */
    map_of_imap map_ess_comp;

    /**
     * @brief Maps BC types and time steps to node/element IDs for BCManager compatibility
     */
    map_of_imap map_ess_id;

    /**
     * @brief Time steps at which boundary conditions are updated
     */
    std::vector<int> update_steps;

    /**
     * @brief Time-dependent boundary condition information
     */
    BCTimeInfo time_info;

    /**
     * @brief Experimental feature monotonic z-loading BCs better
     * single crystal simulations
     */
    bool mono_def_bcs = false;

    // Transform raw BC data into structured format during validation
    bool validate();

    /**
     * @brief Transform legacy flat arrays into structured VelocityBC objects
     */
    void transform_legacy_format();

    /**
     * @brief Populate the map structures expected by BCManager
     */
    void populate_bc_manager_maps();

    /**
     * @brief Helper method to create BC objects from legacy arrays
     * @param step Time step number
     * @param ess_ids Essential boundary condition node IDs
     * @param ess_comps Essential boundary condition component IDs
     * @param essential_vals Essential boundary condition values
     * @param essential_vel_grad Essential velocity gradient values
     */
    void create_boundary_conditions(int step,
                                    const std::vector<int>& ess_ids,
                                    const std::vector<int>& ess_comps,
                                    const std::vector<double>& essential_vals,
                                    const std::vector<std::vector<double>>& essential_vel_grad);

    // Conversion from toml
    static BoundaryOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Visualization options for lattice orientation
 */
struct LightUpOptions {
    /**
     * @brief Whether lattice orientation visualization is enabled
     */
    bool enabled = false;

    /**
     * @brief Name to match with MaterialOptions::material_name
     */
    std::string material_name = "";

    /**
     * @brief Region ID (resolved during validation from material_name)
     */
    std::optional<int> region_id;

    /**
     * @brief Crystal directions to visualize as [h,k,l] indices
     */
    std::vector<std::array<double, 3>> hkl_directions;

    /**
     * @brief Angular tolerance for lattice direction matching (radians)
     */
    double distance_tolerance = 0.0873;

    /**
     * @brief Sample reference direction for orientation comparison
     */
    std::array<double, 3> sample_direction = {0.0, 0.0, 1.0};

    /**
     * @brief Lattice parameters
     *  'cubic'          a
     *  'hexagonal'      a, c
     *  'trigonal'       a, c
     *  'rhombohedral'   a, alpha (in radians)
     *  'tetragonal'     a, c
     *  'orthorhombic'   a, b, c
     *  'monoclinic'     a, b, c, beta (in radians)
     *  'triclinic'      a, b, c, alpha, beta, gamma (in radians)
     */
    std::vector<double> lattice_parameters = {3.6};

    /**
     * @brief Base filename for lattice orientation output files
     */
    std::string lattice_basename = "lattice_avg_";

    /**
     * @brief Lattice type that the user has set
     */
    LatticeType lattice_type = LatticeType::CUBIC;

    /**
     * @brief note whether or not a light-up file was auto-generated
     */
    bool is_auto_generated = false;

    /**
     * @brief Equality operator for uniqueness checking
     */
    bool operator==(const LightUpOptions& other) const {
        // Compare all relevant fields except is_auto_generated
        return material_name == other.material_name && region_id == other.region_id;
    }

    // Validation
    bool validate() const;

    // Conversion from toml
    static LightUpOptions from_toml(const toml::value& toml_input);

    /**
     * @brief Parse light up options from TOML with legacy format support
     * @param toml_input TOML value containing light up configuration
     * @return Vector of LightUpOptions (may be empty if disabled)
     */
    static std::vector<LightUpOptions> from_toml_with_legacy(const toml::value& toml_input);

    /**
     * @brief Resolve material_name to region_id using material list
     * @param materials Vector of material configurations to search
     * @return true if material found and region_id resolved, false otherwise
     */
    bool resolve_region_id(const std::vector<MaterialOptions>& materials);
};

/**
 * @brief Visualization and output options
 */
struct VisualizationOptions {
    /**
     * @brief Enable VisIt output format
     */
    bool visit = false;

    /**
     * @brief Enable ParaView output format
     */
    bool paraview = false;

    /**
     * @brief Enable Conduit output format
     */
    bool conduit = false;

    /**
     * @brief Enable ADIOS2 output format
     */
    bool adios2 = false;

    /**
     * @brief Frequency of visualization output (every N time steps)
     */
    int output_frequency = 1;

    /**
     * @brief Base path/filename for visualization output files
     */
    std::filesystem::path floc = "results";

    // Validation
    bool validate() const;

    // Conversion from toml
    static VisualizationOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Volume average calculation options
 */
struct VolumeAverageOptions {
    /**
     * @brief Filename for averaged stress output
     */
    std::filesystem::path avg_stress_fname = "avg_stress.txt";

    /**
     * @brief Filename for averaged deformation gradient output
     */
    std::filesystem::path avg_def_grad_fname = "avg_def_grad.txt";

    /**
     * @brief Filename for averaged plastic work output
     */
    std::filesystem::path avg_pl_work_fname = "avg_pl_work.txt";

    /**
     * @brief Filename for averaged equivalent plastic strain output
     */
    std::filesystem::path avg_eq_pl_strain_fname = "avg_eq_pl_strain.txt";

    /**
     * @brief Filename for averaged Euler strain output
     */
    std::filesystem::path avg_euler_strain_fname = "avg_euler_strain.txt";

    /**
     * @brief Filename for averaged elastic strain output
     */
    std::filesystem::path avg_elastic_strain_fname = "avg_elastic_strain.txt";

    /**
     * @brief Whether volume averaging is enabled
     */
    bool enabled = true;

    /**
     * @brief Whether to output stress averages
     */
    bool stress = true;

    /**
     * @brief Whether to output deformation gradient averages
     */
    bool def_grad = false;

    /**
     * @brief Whether to output Euler strain averages
     */
    bool euler_strain = false;

    /**
     * @brief Whether to output equivalent plastic strain averages
     */
    bool eq_pl_strain = false;

    /**
     * @brief Whether to output plastic work averages
     */
    bool plastic_work = false;

    /**
     * @brief Whether to output elastic strain averages (ExaCMech only)
     */
    bool elastic_strain = false;

    /**
     * @brief Whether to output additional average quantities
     */
    bool additional_avgs = false;

    /**
     * @brief Output directory for volume average files
     */
    std::filesystem::path output_directory = "results";

    /**
     * @brief Frequency of volume average output (every N time steps)
     */
    int output_frequency = 1;

    // Validation
    bool validate() const;

    // Conversion from toml
    static VolumeAverageOptions from_toml(const toml::value& toml_input);

    /**
     * @brief Parse volume average options from TOML with legacy format support
     * @param toml_input TOML value containing volume average configuration
     * @return VolumeAverageOptions parsed from input
     */
    static VolumeAverageOptions from_toml_with_legacy(const toml::value& toml_input);
};

/**
 * @brief Projection options for visualization
 */
struct ProjectionOptions {
    /**
     * @brief List of enabled projection types for visualization
     */
    std::vector<std::string> enabled_projections;

    /**
     * @brief Whether to automatically enable compatible projections
     */
    bool auto_enable_compatible = true;

    // Validation
    bool validate() const;

    // Conversion from toml
    static ProjectionOptions from_toml(const toml::value& toml_input);
};

/**
 * @brief Post-processing options
 */
struct PostProcessingOptions {
    /**
     * @brief Configuration for volume-averaged quantity calculations
     */
    VolumeAverageOptions volume_averages;

    /**
     * @brief Configuration for field projection operations
     */
    ProjectionOptions projections;

    /**
     * @brief Light-up analysis configurations for crystal orientation visualization
     */
    std::vector<LightUpOptions> light_up_configs;

    // Validation
    bool validate() const;

    // Conversion from toml
    static PostProcessingOptions from_toml(const toml::value& toml_input);

    /**
     * @brief Get all enabled light up configurations
     * @return Vector of enabled LightUpOptions
     */
    std::vector<LightUpOptions> get_enabled_light_up_configs() const;

    /**
     * @brief Get light up configuration for a specific region
     * @param region_id Region ID to search for
     * @return Pointer to LightUpOptions if found, nullptr otherwise
     */
    LightUpOptions* get_light_up_config_for_region(int region_id);

    /**
     * @brief Get light up configuration for a specific region (const version)
     * @param region_id Region ID to search for
     * @return Const pointer to LightUpOptions if found, nullptr otherwise
     */
    const LightUpOptions* get_light_up_config_for_region(int region_id) const;
};

/**
 * @brief Main options class for ExaConstit simulation configuration
 */
class ExaOptions {
public:
    /**
     * @brief Base name for output files (derived from input filename)
     */
    std::string basename = "exaconstit";

    /**
     * @brief Version string for ExaConstit
     */
    std::string version = "0.9.0";

    /**
     * @brief Mesh generation and refinement options
     */
    MeshOptions mesh;

    /**
     * @brief Time stepping configuration
     */
    TimeOptions time;

    /**
     * @brief Solver and assembly options
     */
    SolverOptions solvers;

    /**
     * @brief Visualization output options
     */
    VisualizationOptions visualization;

    /**
     * @brief Material configurations for all regions
     */
    std::vector<MaterialOptions> materials;

    /**
     * @brief Boundary condition specifications
     */
    BoundaryOptions boundary_conditions;

    /**
     * @brief Post-processing and analysis options
     */
    PostProcessingOptions post_processing;

    /**
     * @brief Paths to external material configuration files
     */
    std::vector<std::filesystem::path> material_files;

    /**
     * @brief Path to external post-processing configuration file
     */
    std::optional<std::filesystem::path> post_processing_file;

    /**
     * @brief Optional orientation file path for grain data
     */
    std::optional<std::filesystem::path> orientation_file;

    /**
     * @brief Optional grain mapping file path
     */
    std::optional<std::filesystem::path> grain_file;

    /**
     * @brief Optional region mapping file path
     */
    std::optional<std::filesystem::path> region_mapping_file;

    /**
     * @brief Parse the main configuration file and populate all options
     * @param filename Path to the TOML configuration file
     * @param my_id MPI rank for error reporting
     */
    void parse_options(const std::string& filename, int my_id);

    /**
     * @brief Core option parsing from TOML value
     * @param toml_input Parsed TOML configuration data
     */
    void parse_from_toml(const toml::value& toml_input);

    /**
     * @brief Validate all configuration options for consistency
     * @return true if all options are valid, false otherwise
     */
    bool validate();

    /**
     * @brief Print all options in a formatted way for debugging
     */
    void print_options() const;

private:
    /**
     * @brief Parse mesh-specific options from TOML input
     * @param toml_input TOML value containing mesh configuration
     */
    void parse_mesh_options(const toml::value& toml_input);

    /**
     * @brief Parse time stepping options from TOML input
     * @param toml_input TOML value containing time configuration
     */
    void parse_time_options(const toml::value& toml_input);

    /**
     * @brief Parse solver options from TOML input
     * @param toml_input TOML value containing solver configuration
     */
    void parse_solver_options(const toml::value& toml_input);

    /**
     * @brief Parse material options from TOML input
     * @param toml_input TOML value containing material configuration
     */
    void parse_material_options(const toml::value& toml_input);

    /**
     * @brief Parse boundary condition options from TOML input
     * @param toml_input TOML value containing boundary condition configuration
     */
    void parse_boundary_options(const toml::value& toml_input);

    /**
     * @brief Parse visualization options from TOML input
     * @param toml_input TOML value containing visualization configuration
     */
    void parse_visualization_options(const toml::value& toml_input);

    /**
     * @brief Parse post-processing options from TOML input
     * @param toml_input TOML value containing post-processing configuration
     */
    void parse_post_processing_options(const toml::value& toml_input);

    /**
     * @brief Parse model-specific options for a material
     * @param toml_input TOML value containing model configuration
     * @param material Material object to populate with model options
     */
    void parse_model_options(const toml::value& toml_input, MaterialOptions& material);

    /**
     * @brief Load material configurations from external files
     */
    void load_material_files();

    /**
     * @brief Load post-processing configuration from external file
     */
    void load_post_processing_file();

    /**
     * @brief Print mesh options in formatted output
     */
    void print_mesh_options() const;

    /**
     * @brief Print time stepping options in formatted output
     */
    void print_time_options() const;

    /**
     * @brief Print solver options in formatted output
     */
    void print_solver_options() const;

    /**
     * @brief Print material options in formatted output
     */
    void print_material_options() const;

    /**
     * @brief Print boundary condition options in formatted output
     */
    void print_boundary_options() const;

    /**
     * @brief Print visualization options in formatted output
     */
    void print_visualization_options() const;

    /**
     * @brief Print post-processing options in formatted output
     */
    void print_post_processing_options() const;
};

/**
 * @brief Convert string to MeshType enum
 * @param str String representation of mesh type ("file", "auto")
 * @return Corresponding MeshType enum value, or NOTYPE if invalid
 */
MeshType string_to_mesh_type(const std::string& str);

/**
 * @brief Convert string to TimeStepType enum
 * @param str String representation of time step type ("fixed", "auto", "custom")
 * @return Corresponding TimeStepType enum value, or NOTYPE if invalid
 */
TimeStepType string_to_time_step_type(const std::string& str);

/**
 * @brief Convert string to MechType enum
 * @param str String representation of mechanics type ("umat", "exacmech")
 * @return Corresponding MechType enum value, or NOTYPE if invalid
 */
MechType string_to_mech_type(const std::string& str);

/**
 * @brief Convert string to RTModel enum
 * @param str String representation of runtime model ("CPU", "OPENMP", "GPU")
 * @return Corresponding RTModel enum value, or NOTYPE if invalid
 */
RTModel string_to_rt_model(const std::string& str);

/**
 * @brief Convert string to AssemblyType enum
 * @param str String representation of assembly type ("FULL", "PA", "EA")
 * @return Corresponding AssemblyType enum value, or NOTYPE if invalid
 */
AssemblyType string_to_assembly_type(const std::string& str);

/**
 * @brief Convert string to IntegrationModel enum
 * @param str String representation of integration model ("FULL", "BBAR")
 * @return Corresponding IntegrationModel enum value, or NOTYPE if invalid
 */
IntegrationModel string_to_integration_model(const std::string& str);

/**
 * @brief Convert string to LinearSolverType enum
 * @param str String representation of linear solver type ("CG", "GMRES", "MINRES", "BICGSTAB")
 * @return Corresponding LinearSolverType enum value
 */
LinearSolverType string_to_linear_solver_type(const std::string& str);

/**
 * @brief Convert string to NonlinearSolverType enum
 * @param str String representation of nonlinear solver type ("NR", "NRLS")
 * @return Corresponding NonlinearSolverType enum value, or NOTYPE if invalid
 */
NonlinearSolverType string_to_nonlinear_solver_type(const std::string& str);

/**
 * @brief Convert string to PreconditionerType enum
 * @param str String representation of preconditioner type ("JACOBI", "AMG", "ILU", "L1GS",
 * "CHEBYSHEV")
 * @return Corresponding PreconditionerType enum value
 */
PreconditionerType string_to_preconditioner_type(const std::string& str);

/**
 * @brief Convert string to OriType enum
 * @param str String representation of orientation type ("quat", "custom", "euler")
 * @return Corresponding OriType enum value, or NOTYPE if invalid
 */
OriType string_to_ori_type(const std::string& str);

/**
 * @brief Convert string to LatticeType enum
 * @param str String representation of lattice type ("CUBIC", "HEXAGONAL", "TRIGONAL",
 *             "RHOMBOHEDRAL", "TETRAGONAL", "ORTHORHOMBIC", "MONOCLINIC", "TRICLINIC")
 * @return Corresponding LatticeType enum value
 */
LatticeType string_to_lattice_type(const std::string& str);