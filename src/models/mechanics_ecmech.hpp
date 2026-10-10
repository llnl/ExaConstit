#pragma once

#include "models/mechanics_model.hpp"

#include "mfem.hpp"

#include "ECMech_const.h"
#include "ECMech_matModelBase.h"

#include <memory>

/**
 * @brief Sets up ExaCMech Model quadrature function state pairs
 *
 * @param region_id - the region id associated with this model
 * @param mat_model_name - the exacmech model shortcut name
 * @param sim_stae - the SimulationState generally associated with the ExaModels and which will
 * contain the quadrature function state pair
 */
void ECMechSetupQuadratureFuncStatePair(const int region_id,
                                        const std::string& mat_model_name,
                                        std::shared_ptr<SimulationState> sim_state);

/**
 * @brief ExaCMech crystal plasticity material model implementation
 *
 * @details Implementation of ExaModel for ExaCMech crystal plasticity material models.
 * Supports various crystal plasticity models (FCC, BCC, HCP) with different hardening
 * laws and can execute on CPU, OpenMP, or GPU.
 */
class ExaCMechModel : public ExaModel {
protected:
    /** @brief Current temperature in Kelvin degrees */
    double temp_k;

    /**
     * @brief Pointer to ExaCMech material model base class instance
     *
     * @details The child classes to this class will have also have another variable
     * that actually contains the real material model that is then dynamically casted
     * to this base class during the instantiation of the class.
     */
    ecmech::matModelBase* mat_model_base;

    /** @brief Execution strategy (CPU/OpenMP/GPU) for this model */
    ecmech::ExecutionStrategy accel;

    // RETAINED: Temporary variables that we'll be making use of when running our models.
    // These are working space arrays specific to the ExaCMech model execution,
    // not data storage, so they remain as member variables

    /** @brief Velocity gradient tensor components working array */
    std::unique_ptr<mfem::Vector> vel_grad_array;

    /** @brief Internal energy components working array */
    std::unique_ptr<mfem::Vector> eng_int_array;

    /** @brief Spin tensor components working array */
    std::unique_ptr<mfem::Vector> w_vec_array;

    /** @brief Volume ratio data working array */
    std::unique_ptr<mfem::Vector> vol_ratio_array;

    /** @brief Stress vector in pressure-deviatoric form working array */
    std::unique_ptr<mfem::Vector> stress_svec_p_array;

    /** @brief Deformation rate vector in pressure-deviatoric form working array */
    std::unique_ptr<mfem::Vector> d_svec_p_array;

    /** @brief Temperature array */
    std::unique_ptr<mfem::Vector> tempk_array;

    /** @brief Symmetric deformation rate tensor working array */
    std::unique_ptr<mfem::Vector> sdd_array;

    /** @brief Effective deformation rate working array */
    std::unique_ptr<mfem::Vector> eff_def_rate;

    /**
     * @brief Mapping from variable names to their locations within the state variable vector
     *
     * @details This is ExaCMech-specific and helps locate variables within the large state vector.
     * Enables efficient access to specific quantities like slip rates, hardening variables, etc.
     */
    std::map<std::string, size_t> index_map;

public:
    /**
     * @brief Construct an ExaCMech material model instance
     *
     * @param region Which material region this model manages (key for SimulationState access)
     * @param n_state_vars Number of state variables
     * @param temp_k Temperature in Kelvin
     * @param accel Execution strategy (CPU/OpenMP/GPU)
     * @param mat_model_name ExaCMech model name (e.g., "FCC_PowerVoce", "BCC_KMBalD")
     * @param sim_state Reference to simulation state for data access
     *
     * @details Creates an ExaCMech material model instance for a specific region.
     * Initializes working space arrays and sets up the ExaCMech material model based
     * on the provided model name.
     */
    ExaCMechModel(const int region,
                  int n_state_vars,
                  double temp_k,
                  ecmech::ExecutionStrategy accel,
                  const std::string& mat_model_name,
                  std::shared_ptr<SimulationState> sim_state);

    /**
     * @brief Destructor - cleans up working arrays and ExaCMech model instance
     *
     * @details Deallocates all dynamically allocated working space arrays and
     * the ExaCMech material model instance.
     */
    ~ExaCMechModel() = default;

    /**
     * @brief Initialize working space arrays required for ExaCMech calculations
     *
     * @details Arrays are sized based on the number of quadrature points and allocated
     * on the appropriate device (CPU/GPU). Instead of using stress0 member variable,
     * gets it from SimulationState.
     */
    void SetupDataStructures();

    /**
     * @brief Create the appropriate ExaCMech material model instance
     *
     * @param mat_model_name Name of the ExaCMech material model to instantiate
     *
     * @details Creates the appropriate ExaCMech material model instance based on the
     * model name and sets up the index mapping for state variables.
     */
    void SetupModel(const std::string& mat_model_name);

    /**
     * @brief Initialize state variables at all quadrature points
     *
     * @param hist_init Initial values for state variables
     *
     * @details Initializes state variables at all quadrature points with the provided
     * initial values. Sets up:
     * - Effective shear rate and strain
     * - Plastic work and flow strength
     * - Crystal orientations (quaternions)
     * - Slip rates and hardening variables
     * - Internal energy and volume ratios
     */
    void InitStateVars(std::vector<double> hist_init);

    /**
     * @brief Main ExaCMech model execution method
     *
     * @param nqpts Number of quadrature points per element
     * @param nelems Number of elements in this batch
     * @param space_dim Spatial dimension (unused in current implementation)
     * @param nnodes Number of nodes per element
     * @param jacobian Jacobian transformation matrices for elements
     * @param loc_grad Local gradient operators
     * @param vel Velocity field at elemental level
     *
     * @details This model takes in the velocity, det(jacobian), and local_grad/jacobian.
     * It then computes velocity gradient symm and skw tensors and passes that to our
     * material model in order to get out our Cauchy stress and the material tangent
     * matrix (d \sigma / d Vgrad_{sym}). It also updates all of the state variables
     * that live at the quadrature pts.
     *
     * Implements the three-stage process:
     * 1. **Preprocessing**: Computes velocity gradients, deformation rates, and other kinematic
     * quantities
     * 2. **Material Model**: Calls ExaCMech crystal plasticity kernel to compute stress and tangent
     * stiffness
     * 3. **Postprocessing**: Converts results to appropriate format and updates state variables
     *
     * IMPLEMENTATION NOTE: This method's signature remains unchanged, but internally
     * it uses the new accessor methods to get QuadratureFunctions from SimulationState
     */
    void ModelSetup(const int nqpts,
                    const int nelems,
                    const int /*space_dim*/,
                    const int nnodes,
                    const mfem::Vector& jacobian,
                    const mfem::Vector& loc_grad,
                    const mfem::Vector& vel) override;

    /**
     * @brief Update model state variables after solution convergence
     *
     * @details Empty implementation since ExaCMech handles state variable updates
     * internally during ModelSetup. If we needed to do anything to our state variables
     * once things are solved for we would do that here.
     */
    virtual void UpdateModelVars() override {}
};