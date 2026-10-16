#ifndef MECHANICS_UMAT
#define MECHANICS_UMAT

#include "models/mechanics_model.hpp"
#include "umats/unified_umat_loader.hpp"

#include "mfem.hpp"

#include <filesystem>
namespace fs = std::filesystem;

/**
 * @brief Enhanced Abaqus UMAT model with dynamic library loading support
 *
 * @details Implementation of ExaModel for Abaqus UMAT (User Material) interfaces.
 * Supports both static linking and dynamic loading of UMAT shared libraries, enabling
 * flexible material model integration without recompilation.
 *
 * Key features:
 * - Dynamic loading of UMAT shared libraries
 * - Support for multiple UMATs in different regions
 * - Configurable loading strategies (persistent, on-demand, etc.)
 * - Thread-safe library management
 * - Automatic cleanup and error handling
 */
class AbaqusUmatModel : public ExaModel {
protected:
    /** @brief Characteristic element length passed to UMAT */
    double elem_length;

    /** @brief Initial local shape function gradients working space */
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> loc0_sf_grad;

    /**
     * @brief Quadrature weight times det(J) in the reference configuration, one value per point
     *
     * @details Filled with loc0_sf_grad in InitLocSFGrads. With IntegrationModel::BBAR it gives
     * the volume weights of the F-bar treatment (exaconstit::kernel::FBarElement), whose
     * element-averaged det F is the element's current-to-reference volume ratio.
     */
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> loc0_wdetj;

    /** @brief Incremental deformation gradients working space */
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> incr_def_grad;

    /** @brief End-of-step deformation gradients working space */
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> end_def_grad;

    /** @brief Path to UMAT shared library */
    std::filesystem::path umat_library_path;

    /** @brief Pointer to loaded UMAT function */
    UmatFunction umat_function;

    /** @brief Loading strategy for the library */
    exaconstit::LoadStrategy load_strategy;

    /** @brief Flag to enable/disable dynamic loading */
    bool use_dynamic_loading;

    /** @brief UMAT function name if supplied */
    const std::string umat_function_name;

public:
    /**
     * @brief Constructor with dynamic UMAT loading support
     *
     * @param region Region identifier
     * @param n_state_vars Number of state variables
     * @param sim_state Reference to simulation state
     * @param umat_library_path Path to UMAT shared library (empty for static linking)
     * @param load_strategy Strategy for loading/unloading the library
     * @param umat_function_name UMAT function name that the user wants us to load
     *
     * @details Creates an Abaqus UMAT model instance with support for dynamic library loading.
     * Initializes working space for deformation gradients and prepares for UMAT execution.
     */
    AbaqusUmatModel(
        const int region,
        int n_state_vars,
        std::shared_ptr<SimulationState> sim_state,
        const std::filesystem::path& umat_library_path_ = "",
        const exaconstit::LoadStrategy& load_strategy_ = exaconstit::LoadStrategy::PERSISTENT,
        const std::string umat_function_name_ = "");

    /**
     * @brief Destructor - cleans up resources and unloads library if needed
     *
     * @details Cleans up resources and unloads UMAT library if using non-persistent loading
     * strategy.
     */
    virtual ~AbaqusUmatModel();

    /**
     * @brief Get the beginning-of-step deformation gradient quadrature function
     *
     * @return Shared pointer to the beginning-of-step deformation gradient quadrature function
     *
     * @details Retrieves the deformation gradient at the beginning of the time step from
     * SimulationState for this model's region. This replaces direct member variable access
     * and enables dynamic access to the correct region-specific deformation gradient data.
     */
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> GetDefGrad0();

    /**
     * @brief Update beginning-of-step deformation gradient with converged values
     *
     * @details Updates the beginning-of-step deformation gradient with converged end-of-step
     * values after successful solution convergence. We just need to update our beginning of
     * time step def. grad. with our end step def. grad. now that they are equal.
     */
    virtual void UpdateModelVars() override;

    /**
     * @brief Main UMAT execution method
     *
     * @param nqpts Number of quadrature points per element
     * @param nelems Number of elements in this batch
     * @param space_dim Spatial dimension
     * @param nnodes Number of nodes per element (unused in current implementation)
     * @param jacobian Jacobian transformation matrices for elements
     * @param loc_grad Local gradient operators (unused in current implementation)
     * @param vel Velocity field at elemental level (unused in current implementation)
     *
     * @details Main UMAT execution method that:
     * 1. Loads UMAT library if using on-demand loading
     * 2. Computes incremental deformation gradients
     * 3. Calls UMAT for each quadrature point with appropriate strain measures
     * 4. Collects stress and tangent stiffness results
     * 5. Updates state variables
     * 6. Unloads library if using LOAD_ON_SETUP strategy
     *
     * Since, it is just copy and pasted from the old EvalModel function and now
     * has loops added to it. Now uses accessor methods to get QuadratureFunctions from
     * SimulationState.
     */
    virtual void ModelSetup(const int nqpts,
                            const int nelems,
                            const int space_dim,
                            const int /*nnodes*/,
                            const mfem::Vector& jacobian,
                            const mfem::Vector& /*loc_grad*/,
                            const mfem::Vector& vel) override;

    /**
     * @brief Configure dynamic loading of a UMAT library
     *
     * @param library_path Path to the UMAT shared library
     * @param strategy Loading strategy to use
     * @return True if library setup succeeded, false otherwise
     *
     * @details Configures dynamic loading of a UMAT library with the specified loading strategy.
     */
    bool SetUmatLibrary(const std::filesystem::path& library_path,
                        exaconstit::LoadStrategy strategy = exaconstit::LoadStrategy::PERSISTENT);

    /**
     * @brief Get the current UMAT library path
     */
    const std::filesystem::path& GetUmatLibraryPath() const {
        return umat_library_path;
    }

    /**
     * @brief Check if using dynamic loading
     */
    bool UsingDynamicLoading() const {
        return use_dynamic_loading;
    }

    /**
     * @brief Force reload of the current UMAT library
     *
     * @return True if reload succeeded, false otherwise
     *
     * @details Forces unloading and reloading of the current UMAT library,
     * useful for development and testing.
     */
    bool ReloadUmatLibrary();

    /**
     * @brief Load the UMAT shared library
     *
     * @return True if loading succeeded, false otherwise
     *
     * @details Loads the UMAT shared library and retrieves the UMAT function pointer.
     */
    bool LoadUmatLibrary();

    /**
     * @brief Unload the currently loaded UMAT library
     *
     * @details Unloads the currently loaded UMAT library and resets the function pointer.
     */
    void UnloadUmatLibrary();

protected:
    /**
     * @brief Call the UMAT function (either static or dynamic)
     *
     * @param stress Stress tensor components
     * @param statev State variables array
     * @param ddsdde Material tangent matrix
     * @param sse Specific elastic strain energy
     * @param spd Plastic dissipation
     * @param scd Creep dissipation
     * @param rpl Volumetric heat generation
     * @param ddsdt Stress increment due to temperature
     * @param drplde Heat generation rate due to strain
     * @param drpldt Heat generation rate due to temperature
     * @param stran Strain tensor
     * @param dstran Strain increment
     * @param time Current time and time at beginning of increment
     * @param deltaTime Time increment
     * @param tempk Temperature in Kelvin
     * @param dtemp Temperature increment
     * @param predef Predefined field variables
     * @param dpred Predefined field variable increments
     * @param cmname Material name
     * @param ndi Number of direct stress components
     * @param nshr Number of shear stress components
     * @param ntens Total number of stress components
     * @param nstatv Number of state variables
     * @param props Material properties
     * @param nprops Number of material properties
     * @param coords Coordinates
     * @param drot Rotation increment matrix
     * @param pnewdt Suggested new time increment
     * @param celent Characteristic element length
     * @param dfgrd0 Deformation gradient at beginning of increment
     * @param dfgrd1 Deformation gradient at end of increment
     * @param noel Element number
     * @param npt Integration point number
     * @param layer Layer number
     * @param kspt Section point number
     * @param kstep Step number
     * @param kinc Increment number
     *
     * @details Calls the UMAT function (either statically linked or dynamically loaded)
     * with the standard Abaqus UMAT interface.
     */
    void CallUmat(double* stress,
                  double* statev,
                  double* ddsdde,
                  double* sse,
                  double* spd,
                  double* scd,
                  double* rpl,
                  double* ddsdt,
                  double* drplde,
                  double* drpldt,
                  double* stran,
                  double* dstran,
                  double* time,
                  double* deltaTime,
                  double* tempk,
                  double* dtemp,
                  double* predef,
                  double* dpred,
                  char* cmname,
                  int* ndi,
                  int* nshr,
                  int* ntens,
                  int* nstatv,
                  double* props,
                  int* nprops,
                  double* coords,
                  double* drot,
                  double* pnewdt,
                  double* celent,
                  double* dfgrd0,
                  double* dfgrd1,
                  int* noel,
                  int* npt,
                  int* layer,
                  int* kspt,
                  int* kstep,
                  int* kinc);

    /**
     * @brief Initialize local shape function gradients
     *
     * @param fes Parallel finite element space
     *
     * @details Initializes local shape function gradients for UMAT calculations.
     */
    void InitLocSFGrads(const std::shared_ptr<mfem::ParFiniteElementSpace> fes);

    /**
     * @brief Initialize incremental and end-of-step deformation gradient quadrature functions
     *
     * @details Initializes incremental and end-of-step deformation gradient quadrature functions.
     */
    void InitIncrEndDefGrad();

    /**
     * @brief Calculate incremental and end-of-step deformation gradients
     *
     * @param x0 Current coordinates grid function
     *
     * @details Calculates incremental and end-of-step deformation gradients from current mesh
     * coordinates.
     */
    void CalcIncrEndDefGrad(const mfem::ParGridFunction& x0);

    /**
     * @brief Calculate logarithmic strain increment from deformation gradient
     *
     * @param dE Output strain increment matrix
     * @param Jpt Deformation gradient at quadrature point
     *
     * @details Calculates logarithmic strain increment from deformation gradients for UMAT input.
     */
    void CalcLogStrainIncrement(mfem::DenseMatrix& dE, const mfem::DenseMatrix& Jpt);

    /**
     * @brief Calculate Eulerian strain increment from deformation gradient
     *
     * @param dE Output strain increment matrix
     * @param Jpt Deformation gradient at quadrature point
     *
     * @details Calculates Eulerian strain increment from deformation gradients for UMAT input.
     */
    void CalcEulerianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix& Jpt);

    /**
     * @brief Calculate Lagrangian strain increment from deformation gradient
     *
     * @param dE Output strain increment matrix
     * @param Jpt Deformation gradient at quadrature point
     *
     * @details Calculates Lagrangian strain increment from deformation gradients for UMAT input.
     */
    void CalcLagrangianStrainIncr(mfem::DenseMatrix& dE, const mfem::DenseMatrix& Jpt);

    /**
     * @brief Calculate element length from element volume
     *
     * @param elemVol Element volume
     *
     * @details Calculates characteristic element length as cube root of element volume.
     */
    void CalcElemLength(const double elemVol);
};

#endif