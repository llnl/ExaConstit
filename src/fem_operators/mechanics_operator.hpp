
#ifndef mechanics_operator_hpp
#define mechanics_operator_hpp

#include "fem_operators/mechanics_integrators.hpp"
#include "fem_operators/mechanics_operator_ext.hpp"
#include "models/mechanics_model.hpp"
#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <memory>
/**
 * @brief Central nonlinear mechanics operator for updated Lagrangian finite element formulations.
 *
 * NonlinearMechOperator drives the entire ExaConstit nonlinear mechanics system, implementing
 * an updated Lagrangian finite element formulation for large deformation solid mechanics.
 * It manages the Newton-Raphson solver, Krylov iterative solvers, material models, and
 * coordinates the interaction between finite element operations and constitutive models.
 *
 * The class extends MFEM's NonlinearForm to provide specialized mechanics operations including:
 * - Updated Lagrangian formulation with current configuration updates
 * - Material model integration (crystal plasticity, UMAT, multi-model support)
 * - Partial and element assembly support for high-performance computing
 * - Jacobian computation and preconditioning for Newton-Raphson convergence
 * - Deformation gradient calculation and coordinate updates
 * - Essential boundary condition management
 *
 * Key features for large-scale simulations:
 * - GPU/CPU device compatibility through MFEM's device abstraction
 * - Memory-efficient partial assembly operations
 * - Support for heterogeneous material regions
 * - Automatic coordinate updating for finite deformation problems
 * - Integration with ExaConstit's simulation state management
 *
 * The operator works in conjunction with SimulationState to manage:
 * - Current and reference configurations
 * - Material state variables across time steps
 * - Boundary condition updates
 * - Multi-material region handling
 *
 * @ingroup ExaConstit_fem_operators
 */
class NonlinearMechOperator : public mfem::NonlinearForm {
protected:
    /**
     * @brief Stress-divergence (internal force) integrator, shared with h_form
     *
     * h_form holds it as an external integrator, so this class owns it; it is declared
     * before h_form so that it outlives it. Keeping it separate lets ComputeInternalForce()
     * evaluate only the internal forces even when h_form carries other integrators (e.g.
     * traction boundary conditions).
     */
    std::shared_ptr<ExaNLFIntegrator> stress_integrator;

    /** @brief MFEM parallel nonlinear form for distributed memory computations */
    std::unique_ptr<mfem::ParNonlinearForm> h_form;

    /** @brief Diagonal vector for Jacobian preconditioning operations */
    mutable mfem::Vector diag;

    /** @brief Shape function derivatives at quadrature points for element operations */
    mutable mfem::Vector qpts_dshape;

    /** @brief Element-wise solution vector in local element ordering */
    mutable mfem::Vector el_x;

    /** @brief Prolongation operation intermediate vector for assembly operations */
    mutable mfem::Vector px;

    /** @brief Element-wise internal force work vector for ComputeInternalForce() */
    mutable mfem::Vector el_y;

    /** @brief Local (L-vector) internal force work vector for ComputeInternalForce() */
    mutable mfem::Vector py;

    /** @brief Element Jacobian matrices for geometric transformation computations */
    mutable mfem::Vector el_jac;

    /** @brief Pointer to current Jacobian operator for Newton-Raphson iterations */
    mutable mfem::Operator* jacobian;

    /** @brief Jacobi preconditioner for iterative linear solvers */
    mutable std::shared_ptr<MechOperatorJacobiSmoother> prec_oper;

    /** @brief Element restriction operator for local-to-global degree of freedom mapping */
    const mfem::Operator* elem_restrict_lex;

    /** @brief Assembly strategy (FULL, PARTIAL, ELEMENT) controlling computational approach */
    AssemblyType assembly;

    /** @brief Material model manager handling constitutive relationships */
    std::shared_ptr<ExaModel> model;

    /** @brief Essential boundary condition component specification array */
    const mfem::Array2D<bool>& ess_bdr_comps;

    /** @brief Reference to simulation state for accessing mesh, fields, and configuration data */
    std::shared_ptr<SimulationState> m_sim_state;

public:
    /**
     * @brief Construct nonlinear mechanics operator with boundary conditions and simulation state.
     *
     * @param ess_bdr Array of essential boundary attributes for Dirichlet conditions
     * @param ess_bdr_comp Component specification for essential boundary conditions
     * @param sim_state Reference to simulation state containing mesh, fields, and options
     *
     * Initializes the complete nonlinear mechanics system including:
     * - Parallel nonlinear form setup with proper boundary condition handling
     * - Material model instantiation based on simulation options
     * - Assembly strategy configuration (partial/element/full assembly)
     * - Device memory allocation for vectors and working arrays
     * - Integration rule and shape function derivative precomputation
     * - Preconditioner setup for iterative linear solvers
     *
     * The constructor configures the finite element space, sets up domain integrators
     * based on the chosen integration model (default or B-bar), and prepares all
     * necessary data structures for efficient nonlinear solver operations.
     *
     * Memory allocation is device-aware and will utilize GPU memory when available.
     * The operator is ready for Newton-Raphson iterations upon construction completion.
     */
    NonlinearMechOperator(mfem::Array<int>& ess_bdr,
                          mfem::Array2D<bool>& ess_bdr_comp,
                          std::shared_ptr<SimulationState> sim_state);

    /**
     * @brief Compute Jacobian operator for Newton-Raphson linearization.
     *
     * @param x Current solution vector for Jacobian evaluation point
     * @return Reference to assembled Jacobian operator
     *
     * Computes the tangent stiffness matrix (Jacobian) of the nonlinear residual with respect
     * to the solution vector. This is the core linearization operation in Newton-Raphson
     * iterations, providing the linear system operator for computing solution updates.
     *
     * The method:
     * 1. Calls the underlying MFEM nonlinear form Jacobian computation
     * 2. Assembles the diagonal for preconditioner updates
     * 3. Returns reference to the assembled operator for linear solver use
     *
     * The Jacobian includes contributions from:
     * - Material tangent stiffness (constitutive Jacobian)
     * - Geometric stiffness from large deformation effects
     * - Essential boundary condition enforcement
     *
     * Performance is optimized through partial assembly when enabled, avoiding
     * explicit matrix formation while maintaining operator functionality.
     *
     * @note The returned operator is suitable for use with MFEM's iterative solvers
     * @note Diagonal assembly enables efficient Jacobi preconditioning
     */
    virtual mfem::Operator& GetGradient(const mfem::Vector& x) const override;

    /**
     * @brief Compute linearized residual update for boundary condition changes.
     *
     * @param k Current solution vector
     * @param x Linearization point for Jacobian evaluation
     * @param y Output vector for the linearized residual update
     * @return Reference to Jacobian operator for the updated boundary conditions
     *
     * Computes the effect of boundary condition changes on the linearized system
     * by providing both the residual update and modified Jacobian. This enables
     * efficient handling of time-dependent or load-step-dependent boundary conditions
     * without full system reassembly.
     *
     * The algorithm:
     * 1. Temporarily removes essential boundary condition constraints
     * 2. Computes unconstrained Jacobian-vector product
     * 3. Evaluates residual with updated boundary conditions
     * 4. Combines linearized and nonlinear contributions
     * 5. Restores essential boundary condition enforcement
     *
     * Applications include:
     * - Progressive loading with evolving Dirichlet conditions
     * - Contact boundary condition updates during nonlinear iterations
     * - Multi-physics coupling with changing interface conditions
     * - Adaptive boundary condition strategies
     *
     * The method maintains consistency with the Newton-Raphson framework while
     * efficiently handling boundary condition modifications during the solution process.
     *
     * @note Requires proper Setup() call before use to ensure consistent state
     * @note Output vector y contains both Jacobian action and residual contributions
     */
    virtual mfem::Operator&
    GetUpdateBCsAction(const mfem::Vector& k, const mfem::Vector& x, mfem::Vector& y) const;

    /**
     * @brief Evaluate the internal force vector of the current stresses.
     *
     * @param y Output true-DOF vector, y_a = integral of sigma . grad N_a dv
     *
     * Applies only the stress-divergence integrator (through its partial assembly kernels,
     * on the current mesh and the stored "cauchy_stress_end"), not h_form, so no other
     * integrators or essential boundary conditions enter. The material model is not
     * evaluated. Used by post-processing to sum the nodal forces on a surface.
     */
    void ComputeInternalForce(mfem::Vector& y) const;

    /**
     * @brief Evaluate nonlinear residual vector for current solution state.
     *
     * @param k Solution vector (typically velocity or displacement increment)
     * @param y Output residual vector
     *
     * Computes the nonlinear residual vector representing the out-of-balance forces
     * in the discretized momentum balance equation. This is the core function
     * evaluation in Newton-Raphson iterations, measuring how far the current
     * solution is from satisfying the equilibrium equations.
     *
     * The residual computation includes:
     * 1. Current configuration update using solution vector k
     * 2. Deformation gradient calculation at quadrature points
     * 3. Material model evaluation (stress and tangent computation)
     * 4. Integration of internal force contributions
     * 5. Application of essential boundary conditions
     *
     * Performance optimizations:
     * - Device-aware memory operations for GPU execution
     * - Efficient coordinate update and geometric calculations
     * - Optimized material model calls with vectorized operations
     * - Caliper profiling markers for performance analysis
     *
     * The method coordinates with SimulationState to update:
     * - Nodal coordinates for updated Lagrangian formulation
     * - Material state variables based on computed deformation
     * - Boundary condition applications
     *
     * @note Calls Setup<true>() to update coordinates and material state
     * @note Residual is computed in the current (deformed) configuration
     */
    virtual void Mult(const mfem::Vector& k, mfem::Vector& y) const override;

    /// Sets all of the data up for the Mult and GetGradient method
    /// This is of significant interest to be able to do partial assembly operations.
    using mfem::NonlinearForm::Setup;

    /**
     * @brief Setup deformation state and material properties for current solution.
     *
     * @tparam upd_crds Boolean controlling whether to update nodal coordinates
     * @param k Solution vector for deformation gradient computation
     *
     * Prepares all necessary geometric and material quantities for residual and
     * Jacobian computations. This is a critical setup phase that coordinates
     * the updated Lagrangian formulation with material model requirements.
     *
     * The setup process includes:
     * 1. Coordinate update (if upd_crds=true) for current configuration
     * 2. Jacobian matrix computation for geometric transformations
     * 3. Deformation gradient calculation at quadrature points
     * 4. Material model setup with current state variables
     * 5. Error handling for material model convergence issues
     *
     * Template parameter usage:
     * - upd_crds=true: Full setup for residual evaluation (updates mesh geometry)
     * - upd_crds=false: Linearization setup for Jacobian evaluation (geometry fixed)
     *
     * The method ensures consistency between:
     * - Current mesh configuration and solution vector
     * - Material state variables and computed deformation
     * - Integration point data and finite element discretization
     *
     * Error handling includes detection of material model failures and provides
     * descriptive error messages for debugging convergence issues.
     *
     * @note Template instantiation enables compile-time optimization
     * @note Material model setup may fail for extreme deformations
     * @throws std::runtime_error if material model setup fails
     */
    template <bool upd_crds>
    void Setup(const mfem::Vector& k) const;

    /**
     * @brief Compute geometric transformation matrices for finite element operations.
     *
     * Sets up Jacobian matrices needed for coordinate transformations between
     * reference and current configurations in the updated Lagrangian formulation.
     * This includes computation of geometric factors required for integration
     * point operations and material model evaluations.
     *
     * The method:
     * 1. Temporarily swaps mesh coordinates to current configuration
     * 2. Computes geometric transformation matrices
     * 3. Updates integration point geometric data
     * 4. Restores original mesh coordinate state
     * 5. Invalidates cached geometric factors for next update
     *
     * This separation enables efficient recomputation of geometric quantities
     * without affecting the overall mesh data structure and allows material
     * models to access current configuration geometry consistently.
     *
     * @note Requires current nodal coordinates to be available in SimulationState
     * @note Invalidates mesh geometric factors cache for consistency
     */
    void SetupJacobianTerms() const;

    /**
     * @brief Calculate deformation gradient tensor at all quadrature points.
     *
     * @param def_grad Output quadrature function for deformation gradient storage
     *
     * Computes the deformation gradient tensor F = ∂x/∂X at each quadrature point,
     * where x is the current position and X is the reference position. This is
     * the fundamental kinematic quantity for finite deformation mechanics and
     * serves as input to material constitutive models.
     *
     * The calculation involves:
     * 1. Current configuration setup and coordinate transformation
     * 2. Shape function derivative evaluation at quadrature points
     * 3. Deformation gradient computation using nodal displacements
     * 4. Storage in device-compatible quadrature function format
     *
     * The deformation gradient enables material models to:
     * - Compute finite strain measures (Green-Lagrange, logarithmic, etc.)
     * - Evaluate stress in current or reference configurations
     * - Update internal state variables consistently with large deformation
     *
     * Performance considerations:
     * - Optimized kernel operations for GPU execution
     * - Memory-efficient quadrature point data layout
     * - Consistent with partial assembly operations
     *
     * @note Output def_grad must be properly sized for all mesh quadrature points
     * @note Deformation gradient computation assumes updated Lagrangian formulation
     */
    void CalculateDeformationGradient(mfem::QuadratureFunction& def_grad) const;

    /**
     * @brief Update nodal coordinates with computed velocity solution.
     *
     * @param vel Velocity vector solution from Newton-Raphson iteration
     *
     * Updates the current nodal coordinates using the velocity solution from
     * the nonlinear solver. This is essential for the updated Lagrangian
     * formulation where the finite element mesh follows the material deformation.
     *
     * The update process:
     * 1. Copies velocity solution to simulation state primal field
     * 2. Triggers coordinate update in simulation state management
     * 3. Ensures consistency between solution vector and mesh geometry
     *
     * This method maintains the updated Lagrangian framework by ensuring that:
     * - Mesh nodes track material particle positions
     * - Geometric calculations reflect current configuration
     * - Material models receive consistent deformation data
     *
     * @note Must be called after each Newton-Raphson iteration for consistency
     * @note Coordinates are updated in simulation state for global accessibility
     */
    void UpdateEndCoords(const mfem::Vector& vel) const;

    /**
     * @brief Update essential boundary condition specifications.
     *
     * @param ess_bdr New essential boundary attribute specifications
     * @param mono_def_flag Flag controlling monolithic or component-wise BC application
     *
     * Updates the essential boundary condition specification for the nonlinear form,
     * enabling dynamic boundary condition changes during the simulation. This is
     * essential for progressive loading, contact problems, and multi-physics coupling.
     *
     * Update modes:
     * - mono_def_flag=true: Direct DOF specification for monolithic problems
     * - mono_def_flag=false: Component-wise BC specification for vector problems
     *
     * The method updates both:
     * - Internal nonlinear form boundary condition storage
     * - Class-level essential DOF lists for consistent access
     *
     * Applications include:
     * - Time-dependent boundary conditions
     * - Load stepping with evolving constraints
     * - Contact and interface condition updates
     * - Multi-stage loading protocols
     *
     * @note Boundary condition changes affect Jacobian structure and preconditioning
     * @note Component specification must match finite element space dimensions
     */
    void UpdateEssTDofs(const mfem::Array<int>& ess_bdr, bool mono_def_flag);

    /**
     * @brief Retrieve list of essential (constrained) true degrees of freedom.
     *
     * @return Constant reference to array of essential true DOF indices
     *
     * Provides access to the current set of constrained degrees of freedom,
     * which is essential for linear solver setup, preconditioning, and
     * solution vector manipulation in constrained problems.
     *
     * The essential DOF list includes:
     * - Dirichlet boundary condition constraints
     * - Multi-point constraints if present
     * - Any other DOF constraints from the finite element formulation
     *
     * Applications:
     * - Linear solver constraint enforcement
     * - Preconditioner setup for constrained systems
     * - Solution vector post-processing
     * - Convergence monitoring for unconstrained DOFs
     *
     * @note List reflects current boundary condition state
     * @note Indices are in true (globally numbered) DOF space
     */
    const mfem::Array<int>& GetEssTDofList();

    /**
     * @brief Access material model for constitutive relationship queries.
     *
     * @return Pointer to active material model instance
     *
     * Provides access to the material model instance for external queries
     * about constitutive relationships, state variable access, and material
     * property information. This enables coupling with post-processing,
     * adaptive algorithms, and multi-physics solvers.
     *
     * The material model provides:
     * - Stress and tangent stiffness computation
     * - State variable access and manipulation
     * - Material property queries
     * - Constitutive model-specific operations
     *
     * Common uses:
     * - Post-processing stress and strain calculations
     * - Adaptive mesh refinement based on material state
     * - Multi-physics coupling with thermal or electromagnetic models
     * - Material failure and damage assessment
     *
     * @note Returned pointer should not be deleted by caller
     * @note Material model lifetime matches operator lifetime
     */
    std::shared_ptr<ExaModel> GetModel() const {
        return model;
    }

    /**
     * @brief Access Jacobi preconditioner for linear solver operations.
     *
     * @return Pointer to partial assembly preconditioner instance
     *
     * Provides access to the configured Jacobi preconditioner for use with
     * iterative linear solvers in Newton-Raphson iterations. The preconditioner
     * is automatically updated with diagonal information from Jacobian assembly.
     *
     * Preconditioner features:
     * - L1-Jacobi smoothing for effective preconditioning
     * - Essential boundary condition handling
     * - Device-compatible operations for GPU execution
     * - Automatic diagonal updates during Jacobian assembly
     *
     * The preconditioner is optimized for:
     * - Partial assembly operations (matrix-free)
     * - Large-scale parallel computations
     * - Mixed precision iterative solvers
     * - Contact and constrained problems
     *
     * @return nullptr if partial assembly is not enabled
     * @note Preconditioner state automatically maintained during solution
     */
    std::shared_ptr<MechOperatorJacobiSmoother> GetPAPreconditioner() {
        return prec_oper;
    }

    /**
     * @brief Clean up mechanics operator resources and material model.
     *
     * Destructor for NonlinearMechOperator that properly deallocates resources
     * and ensures clean shutdown of the mechanics system. This includes cleanup
     * of both MFEM resources and ExaConstit-specific material model instances.
     *
     * Cleanup responsibilities:
     * - Deletes material model instance and associated resources
     * - Deallocates MFEM parallel nonlinear form
     * - Releases any allocated preconditioner operators
     * - Ensures proper cleanup of device memory allocations
     *
     * The destructor handles:
     * - Material model deletion with proper ExaModel cleanup
     * - MFEM parallel nonlinear form deallocation
     * - Preconditioner operator cleanup if allocated
     * - Device memory management for GPU-allocated vectors
     *
     * @note Material model deletion includes cleanup of constitutive model resources
     * @note MFEM nonlinear form cleanup handles integrator deallocation
     * @note Preconditioner cleanup performed automatically when needed
     */
    virtual ~NonlinearMechOperator() = default;
};

#endif /* mechanics_operator_hpp */
