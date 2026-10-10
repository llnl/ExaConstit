#ifndef MECHANICS_INTEG
#define MECHANICS_INTEG

#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <string>
#include <unordered_map>
#include <utility>

/**
 * @brief Nonlinear form integrator for general solid mechanics problems with material model
 * integration.
 *
 * ExaNLFIntegrator implements a comprehensive finite element integrator specifically designed
 * for ExaConstit's solid mechanics applications, including crystal plasticity, large deformation
 * mechanics, and general material model integration. This integrator serves as the foundation
 * for nonlinear finite element assembly operations in updated Lagrangian formulations.
 *
 * The integrator provides:
 * - Element vector assembly for residual computation (internal forces)
 * - Element matrix assembly for Jacobian computation (tangent stiffness)
 * - Partial assembly (PA) operations for memory-efficient matrix-free methods
 * - Element assembly (EA) operations for minimal memory usage
 * - Device-compatible implementations for CPU and GPU execution
 *
 * Key features for crystal plasticity and micromechanics:
 * - Integration with ExaConstit's material model framework
 * - Support for heterogeneous material regions through SimulationState
 * - Quadrature function data access for stress and tangent stiffness
 * - Optimized assembly operations for large-scale simulations
 * - Compatibility with MFEM's assembly level abstractions
 *
 * Assembly strategy support:
 * - Traditional element-wise assembly for small problems
 * - Partial assembly for memory-efficient large-scale problems
 * - Element assembly for memory-constrained environments
 * - Mixed assembly strategies for heterogeneous hardware
 *
 * The integrator coordinates with SimulationState to access:
 * - Current stress tensors from material model evaluations
 * - Material tangent stiffness matrices for linearization
 * - Geometric data for coordinate transformations
 * - Quadrature point data for integration operations
 *
 * @ingroup ExaConstit_fem_operators
 */
class ExaNLFIntegrator : public mfem::NonlinearFormIntegrator {
protected:
    /** @brief Reference to simulation state for accessing mesh, fields, and material data */
    std::shared_ptr<SimulationState> m_sim_state;

    /** @brief Working vector for material data storage during assembly operations */
    mfem::Vector dmat;

    /** @brief Gradient data vector for partial assembly operations */
    mfem::Vector grad;

    /** @brief Partial assembly material data vector */
    mfem::Vector pa_mat;

    /** @brief Partial assembly diagonal material data vector */
    mfem::Vector pa_dmat;

    /** @brief Jacobian transformation data vector for geometric operations */
    mfem::Vector jacobian;

    /** @brief Geometric factors for mesh transformation operations (not owned) */
    const mfem::GeometricFactors* geom; // Not owned

    /** @brief Spatial dimension of the finite element problem */
    int space_dims;

    /** @brief Number of finite elements in the mesh */
    int nelems;

    /** @brief Number of quadrature points per element */
    int nqpts;

    /** @brief Number of nodes (degrees of freedom) per element */
    int nnodes;

public:
    /**
     * @brief Construct integrator with simulation state reference.
     *
     * @param sim_state Reference to simulation state containing mesh, fields, and material data
     *
     * Initializes the nonlinear form integrator with access to the simulation state,
     * enabling integration with ExaConstit's material model framework and data management.
     * The integrator is ready for element assembly operations upon construction.
     *
     * The constructor establishes:
     * - Reference to simulation state for data access
     * - Foundation for subsequent assembly strategy configuration
     * - Integration with MFEM's NonlinearFormIntegrator interface
     *
     * @note Simulation state reference must remain valid for integrator lifetime
     * @note Working vectors are allocated lazily during first assembly operations
     */
    ExaNLFIntegrator(std::shared_ptr<SimulationState> sim_state) : m_sim_state(sim_state) {}

    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     *
     * Ensures proper cleanup of integrator resources and derived class data.
     * The destructor handles cleanup of working vectors and any allocated
     * data structures used during assembly operations.
     *
     * @note Base class destructor handles MFEM NonlinearFormIntegrator cleanup
     * @note Working vectors are automatically cleaned up by MFEM Vector destructors
     */
    virtual ~ExaNLFIntegrator() {}

    /// This doesn't do anything at this point. We can add the functionality
    /// later on if a use case arises.
    using mfem::NonlinearFormIntegrator::GetElementEnergy;
    /**
     * @brief Compute element energy contribution (placeholder implementation).
     *
     * @param el Finite element for energy computation
     * @param Ttr Element transformation for coordinate mapping
     * @param elfun Element solution vector
     * @return Element energy contribution (currently always returns 0.0)
     *
     * This method provides the interface for element energy computation but
     * currently returns zero. The functionality can be added later if energy
     * calculations become required for the application.
     *
     * Potential future uses:
     * - Total strain energy computation for post-processing
     * - Energy-based error estimation for adaptive refinement
     * - Thermodynamic consistency checks in material models
     * - Variational constitutive updates
     *
     * @note Current implementation is placeholder returning 0.0
     * @note Can be extended for specific energy computation requirements
     */
    virtual double GetElementEnergy([[maybe_unused]] const mfem::FiniteElement& el,
                                    [[maybe_unused]] mfem::ElementTransformation& Ttr,
                                    [[maybe_unused]] const mfem::Vector& elfun) override {
        return 0.0;
    };

    using mfem::NonlinearFormIntegrator::AssembleElementVector;
    /**
     * @brief Assemble element residual vector for internal force computation.
     *
     * @param el Finite element providing shape functions and geometric information
     * @param Ttr Element transformation for coordinate mapping
     * @param elfun Element solution vector (typically nodal velocities or displacements)
     * @param elvect Output element residual vector representing internal forces
     *
     * Computes the element contribution to the nonlinear residual vector, representing
     * the internal forces arising from stress divergence in the current configuration.
     * This is the core element-level computation in Newton-Raphson iterations.
     *
     * The assembly process:
     * 1. Computes shape function derivatives in physical coordinates
     * 2. Retrieves current stress state from quadrature function data
     * 3. Integrates B^T * σ over element volume using Gauss quadrature
     * 4. Accumulates contributions from all quadrature points
     *
     * Stress tensor handling:
     * - Accesses Cauchy stress from simulation state quadrature functions
     * - Uses full 3x3 stress tensor with proper symmetry treatment
     * - Integrates stress divergence contribution to residual vector
     *
     * The residual represents the out-of-balance internal forces:
     * f_internal = ∫_Ω B^T(x) σ(x) dΩ
     *
     * where B is the strain-displacement matrix and σ is the Cauchy stress tensor.
     *
     * Performance optimizations:
     * - Reuses matrices across quadrature points for memory efficiency
     * - Direct external data access for input/output vectors
     * - Optimized matrix-vector operations using MFEM routines
     *
     * @note Assumes 3D problems with symmetric stress tensors
     * @note Integration rule must match quadrature space for stress data
     * @note Caliper profiling enabled for performance monitoring
     */
    virtual void AssembleElementVector(const mfem::FiniteElement& el,
                                       mfem::ElementTransformation& Ttr,
                                       const mfem::Vector& elfun,
                                       mfem::Vector& elvect) override;

    /**
     * @brief Assemble element tangent stiffness matrix for Newton-Raphson linearization.
     *
     * @param el Finite element providing shape functions and geometric information
     * @param Ttr Element transformation for coordinate mapping
     * @param elfun Element solution vector (unused in current implementation)
     * @param elmat Output element stiffness matrix
     *
     * Computes the element tangent stiffness matrix used in Newton-Raphson linearization,
     * representing the derivative of internal forces with respect to nodal displacements.
     * This matrix is essential for convergence of nonlinear iterations.
     *
     * The assembly process:
     * 1. Computes shape function derivatives in physical coordinates
     * 2. Retrieves material tangent stiffness from quadrature function data
     * 3. Constructs strain-displacement B-matrix for current configuration
     * 4. Integrates B^T * C * B over element volume using Gauss quadrature
     *
     * Tangent stiffness computation:
     * K_element = ∫_Ω B^T(x) C(x) B(x) dΩ
     *
     * where:
     * - B is the strain-displacement matrix (6×3n for 3D elements)
     * - C is the material tangent stiffness matrix (6×6 for 3D)
     * - Integration performed over current (deformed) element volume
     *
     * Material tangent matrix:
     * - Accesses 6×6 tangent stiffness from material model evaluations
     * - Uses Voigt notation for symmetric tensor operations
     * - Includes both material and geometric stiffness contributions
     *
     * The algorithm performs the matrix triple product efficiently:
     * 1. Computes temp = C * B (intermediate result)
     * 2. Computes K += B^T * temp (final contribution)
     * 3. Accumulates contributions from all quadrature points
     *
     * Performance considerations:
     * - Optimized matrix operations using MFEM dense matrix routines
     * - Memory reuse for intermediate matrices across quadrature points
     * - Integration weights incorporated efficiently
     *
     * @note Material tangent matrix assumed to be 6×6 in Voigt notation
     * @note B-matrix construction handles 3D elements with proper DOF ordering
     * @note Caliper profiling enabled for performance analysis
     */
    virtual void AssembleElementGrad(const mfem::FiniteElement& el,
                                     mfem::ElementTransformation& Ttr,
                                     const mfem::Vector& /*elfun*/,
                                     mfem::DenseMatrix& elmat) override;

    /**
     * @brief Initialize partial assembly data structures for gradient (Jacobian) operations.
     *
     * @param x Solution vector for state-dependent assembly (unused in current implementation)
     * @param fes Finite element space providing mesh and element information
     *
     * Prepares geometric and material data structures needed for efficient partial
     * assembly Jacobian operations. This method precomputes transformation data
     * and material property layouts optimized for matrix-free operations.
     *
     * The gradient assembly setup includes:
     * 1. Computing and storing shape function derivatives at quadrature points
     * 2. Preparing 4D tensor layouts for material tangent operations
     * 3. Setting up geometric factors for coordinate transformations
     * 4. Organizing data for vectorized element-wise operations
     *
     * 4D tensor transformation:
     * Applies the transformation: D_ijkm = (1/det(J)) * w_qpt * adj(J)^T_ij * C^tan_ijkl *
     * adj(J)_lm where:
     * - D is the transformed 4th order tensor for partial assembly
     * - J is the Jacobian matrix from geometric factors
     * - C^tan is the material tangent stiffness tensor
     * - adj(J) is the adjugate of the Jacobian matrix
     *
     * Performance optimizations:
     * - Precomputes shape function derivatives for all quadrature points
     * - Uses RAJA views with optimized memory layouts for target architecture
     * - Enables vectorization across elements and quadrature points
     * - Supports both CPU and GPU execution
     *
     * @note Current implementation delegates to single-argument version
     * @note Shape function derivatives cached for reuse in gradient operations
     * @note 4D tensor layout optimized for specific hardware architectures
     */
    virtual void AssembleGradPA(const mfem::Vector& /* x */,
                                const mfem::FiniteElementSpace& fes) override;
    /**
     * @brief Initialize partial assembly data structures for gradient operations.
     *
     * @param fes Finite element space providing mesh and element information
     *
     * Performs the core setup for partial assembly gradient operations by precomputing
     * geometric factors and material data layouts. This method transforms material
     * tangent data into optimized formats for efficient matrix-vector operations.
     *
     * The setup process includes:
     * 1. Computing spatial dimensions and element characteristics
     * 2. Precomputing shape function derivatives at all quadrature points
     * 3. Transforming material tangent tensors for partial assembly operations
     * 4. Setting up memory layouts optimized for target hardware
     *
     * Shape function derivative computation:
     * - Calculates ∂N/∂ξ derivatives for all quadrature points
     * - Stores in device-compatible format for GPU execution
     * - Organizes data for efficient vectorized operations
     * - Reuses derivatives across multiple gradient assembly calls
     *
     * Material tensor transformation:
     * - Applies geometric transformations to material tangent matrices
     * - Incorporates quadrature weights and Jacobian determinants
     * - Uses 4D tensor layouts optimized for partial assembly operations
     * - Enables efficient matrix-vector products in AddMultGradPA()
     *
     * The method prepares data structures for:
     * - Fast Jacobian-vector products via AddMultGradPA()
     * - Diagonal assembly for preconditioning via AssembleGradDiagonalPA()
     * - Memory-efficient operations without explicit matrix storage
     *
     * @note Must be called before AddMultGradPA() and diagonal assembly operations
     * @note Material tangent data accessed from simulation state quadrature functions
     * @note Supports only 3D problems (1D and 2D abort with error message)
     */
    virtual void AssembleGradPA(const mfem::FiniteElementSpace& fes) override;

    /**
     * @brief Apply partial assembly gradient (Jacobian) operation.
     *
     * @param x Input vector for Jacobian-vector product
     * @param y Output vector for accumulated result
     *
     * Performs the partial assembly Jacobian-vector product operation using
     * precomputed geometric factors and transformed material tangent data.
     * This operation computes the action of the tangent stiffness matrix
     * without explicit matrix assembly, providing memory-efficient Newton-Raphson iterations.
     *
     * The operation computes: y += K * x, where K is the tangent stiffness matrix
     * represented implicitly through partial assembly data structures.
     *
     * Algorithm overview:
     * 1. Uses precomputed shape function derivatives and material data
     * 2. Performs element-wise matrix-vector operations
     * 3. Applies geometric transformations on-the-fly
     * 4. Accumulates contributions to global vector
     *
     * Memory efficiency features:
     * - No explicit stiffness matrix storage required
     * - Vectorized operations over elements and quadrature points
     * - Device-compatible implementation for GPU acceleration
     * - Minimal working memory requirements
     *
     * Performance characteristics:
     * - Computational complexity: O(nelems × nqpts × ndof²)
     * - Memory complexity: O(nelems × nqpts) for material data
     * - Excellent parallel scaling for large problems
     * - Cache-friendly memory access patterns
     *
     * The method is called repeatedly during Krylov solver iterations
     * within Newton-Raphson steps, making performance optimization critical.
     *
     * @note Requires prior AssembleGradPA() call for data structure setup
     * @note Input and output vectors must match finite element space dimensions
     * @note Essential boundary conditions handled by calling operator
     */
    virtual void AddMultGradPA(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Apply transposed gradient action via partial assembly.
     *
     * @param x Input vector for transposed Jacobian-vector product
     * @param y Output vector for accumulated result
     *
     * Native PA kernel computing y += K^T * x where K = B^T D B is the
     * tangent stiffness operator. The only computational difference from
     * AddMultGradPA is the contraction order with the assembled 4th-order
     * tensor D:
     *
     *   Forward (AddMultGradPA):
     *     T(i,k) = D(i,k,l,n,qpt,elem) * Gx(l,n)   — contract last pair
     *     Y(a,k) += Gt(a,i,qpt) * T(i,k)
     *
     *   Transpose (this method):
     *     T(l,n) = D(i,k,l,n,qpt,elem) * Gx(i,k)   — contract first pair
     *     Y(a,n) += Gt(a,l,qpt) * T(l,n)
     *
     * For symmetric material tangent C, the two operations are identical.
     * For non-symmetric C (crystal plasticity), they differ. The transpose
     * is required for trust-region dogleg solver Cauchy point computation
     * where the merit function gradient is g = J^T * r, not J * r.
     *
     * @note GPU-compatible via mfem::forall
     * @note Requires prior AssembleGradPA() call for the D tensor
     */
    virtual void AddMultTransposeGradPA(const mfem::Vector& x, mfem::Vector& y) const override;

    using mfem::NonlinearFormIntegrator::AssemblePA;
    /**
     * @brief Initialize partial assembly data structures for residual operations.
     *
     * @param fes Finite element space providing mesh and element information
     *
     * Performs the initial setup for partial assembly operations by precomputing
     * and storing geometric factors needed for efficient element-wise operations.
     * This method amortizes setup costs across multiple residual evaluations.
     *
     * The setup process includes:
     * 1. Extracting mesh and finite element information
     * 2. Computing integration rule and weights
     * 3. Storing geometric factors for coordinate transformations
     * 4. Precomputing element-invariant quantities
     *
     * Geometric factor computation:
     * - Retrieves Jacobian matrices for all elements and quadrature points
     * - Stores transformation data in device-compatible format
     * - Enables efficient coordinate mapping during assembly
     *
     * Memory allocation strategy:
     * - Allocates working vectors with appropriate device memory types
     * - Sizes vectors based on problem dimensions and mesh size
     * - Prepares data structures for GPU execution when available
     *
     * The method prepares for:
     * - Fast element vector assembly via AddMultPA()
     * - Reuse of geometric data across multiple assembly calls
     * - Device-compatible data layouts for GPU execution
     *
     * @note Must be called before AddMultPA() operations
     * @note Geometric factors cached for reuse across assembly calls
     * @note Caliper profiling scope for performance monitoring
     */
    virtual void AssemblePA(const mfem::FiniteElementSpace& fes) override;
    /**
     * @brief Apply partial assembly element vector operation.
     *
     * @param x Input vector (unused in current implementation for residual assembly)
     * @param y Output vector for accumulated element contributions
     *
     * Performs the partial assembly element vector operation, computing element
     * residual contributions using precomputed geometric factors and current
     * stress data. This operation is optimized for memory efficiency and
     * computational performance in large-scale simulations.
     *
     * The partial assembly approach:
     * - Uses precomputed geometric factors from AssemblePA()
     * - Accesses stress data directly from quadrature functions
     * - Performs element-wise operations without global matrix assembly
     * - Accumulates results directly into global vector
     *
     * Operation sequence:
     * 1. Initializes output vector appropriately
     * 2. Loops over all elements in parallel-friendly manner
     * 3. Applies element-wise stress integration
     * 4. Accumulates results into global degrees of freedom
     *
     * Memory efficiency features:
     * - Minimal working memory requirements
     * - Direct access to stress quadrature function data
     * - Vectorized operations over elements and quadrature points
     * - Device-compatible implementation for GPU execution
     *
     * This method is called repeatedly during nonlinear iterations,
     * so performance optimization is critical for overall solver efficiency.
     *
     * @note Input vector x currently unused for stress-based residual assembly
     * @note Output vector y must be properly sized for true DOF space
     * @note Requires prior AssemblePA() call for geometric factor setup
     */
    virtual void AddMultPA(const mfem::Vector& /*x*/, mfem::Vector& y) const override;

    /**
     * @brief Assemble diagonal entries for partial assembly preconditioning.
     *
     * @param diag Output vector for diagonal entries of the tangent stiffness matrix
     *
     * Computes diagonal entries of the tangent stiffness matrix using partial
     * assembly techniques, providing diagonal approximations essential for
     * Jacobi preconditioning in iterative linear solvers.
     *
     * The diagonal computation extracts entries: diag[i] = K[i,i] where K is
     * the tangent stiffness matrix represented through partial assembly data.
     *
     * Algorithm approach:
     * 1. Uses precomputed material tangent data from AssembleGradPA()
     * 2. Extracts diagonal contributions element-by-element
     * 3. Applies geometric transformations for diagonal terms
     * 4. Assembles global diagonal through element restriction operations
     *
     * Diagonal extraction strategy:
     * - Computes element-wise diagonal contributions
     * - Uses vectorized operations for efficiency
     * - Handles geometric transformations appropriately
     * - Accumulates to global diagonal vector
     *
     * The diagonal approximation quality affects:
     * - Jacobi preconditioner effectiveness
     * - Krylov solver convergence rates
     * - Overall Newton-Raphson performance
     * - Numerical stability of iterative methods
     *
     * Memory and performance characteristics:
     * - Linear scaling with problem size
     * - Device-compatible implementation
     * - Efficient vectorized operations
     * - Minimal additional memory requirements
     *
     * @note Requires prior AssembleGradPA() call for material data setup
     * @note Output vector must be properly sized for finite element space
     * @note Diagonal quality depends on material tangent matrix conditioning
     */
    virtual void AssembleGradDiagonalPA(mfem::Vector& diag) const override;
    /**
     * @brief Perform element assembly for gradient operations with solution vector.
     *
     * @param x Solution vector for state-dependent assembly (unused in current implementation)
     * @param fes Finite element space providing mesh and element information
     * @param ea_data Output vector for assembled element matrix data
     *
     * Performs element assembly for gradient operations, computing and storing
     * complete element matrices in a format suitable for element assembly (EA)
     * operations. This method delegates to the base element assembly routine.
     *
     * Element assembly characteristics:
     * - Computes full element stiffness matrices
     * - Stores matrices in contiguous device-compatible format
     * - Enables exact matrix-vector products through explicit element matrices
     * - Provides maximum memory efficiency for large problems
     *
     * The method serves as an interface for state-dependent element assembly
     * while currently delegating to the stateless version for implementation.
     *
     * @note Current implementation delegates to AssembleEA(fes, ea_data)
     * @note Solution vector x currently unused but available for future extensions
     * @note Element matrices stored in ea_data with specific layout requirements
     */
    virtual void AssembleGradEA(const mfem::Vector& /* x */,
                                const mfem::FiniteElementSpace& fes,
                                mfem::Vector& ea_data) override;
    /**
     * @brief Perform element assembly for gradient operations.
     *
     * @param fes Finite element space providing mesh and element information
     * @param emat Output vector for assembled element matrix data
     *
     * Computes and stores complete element stiffness matrices for all elements
     * in the mesh, providing an element assembly (EA) representation of the
     * tangent stiffness operator for memory-constrained applications.
     *
     * Element assembly process:
     * 1. Iterates over all elements in the mesh
     * 2. Computes full element stiffness matrices
     * 3. Stores matrices in contiguous device memory format
     * 4. Organizes data for efficient element-wise matrix-vector products
     *
     * Memory layout:
     * - Matrices stored element-by-element in contiguous memory
     * - Dense matrices with row-major ordering within each element
     * - Device-compatible allocation for GPU execution
     * - Total size: nelems × (ndof×ncomps)² entries
     *
     * Performance characteristics:
     * - Higher assembly cost compared to partial assembly
     * - Minimal memory usage compared to global matrix assembly
     * - Exact operator representation without approximation
     * - Excellent performance for high DOF-per-element problems
     *
     * The element matrices enable:
     * - Exact matrix-vector products in element assembly operators
     * - Minimal memory footprint for large-scale problems
     * - Natural parallelization over elements
     * - Cache-friendly memory access patterns
     *
     * @note Supports only 3D problems (1D and 2D problems abort with error)
     * @note Uses RAJA views for optimized memory layouts and vectorization
     * @note Caliper profiling enabled for performance monitoring
     */
    virtual void AssembleEA(const mfem::FiniteElementSpace& fes, mfem::Vector& emat) override;
};

/**
 * @brief B-bar method integrator for incompressible and nearly incompressible solid mechanics.
 *
 * ICExaNLFIntegrator extends ExaNLFIntegrator to implement the B-bar method for handling
 * incompressible and nearly incompressible materials. This integrator is essential for
 * crystal plasticity simulations where volume preservation constraints arise from
 * incompressible plastic deformation or nearly incompressible elastic behavior.
 *
 * The B-bar method (Hughes, 1980):
 * - Modifies the strain-displacement B-matrix to avoid volumetric locking
 * - Uses volume-averaged dilatational strains to improve element performance
 * - Maintains accuracy for incompressible and nearly incompressible materials
 * - Enables stable finite element solutions for high bulk modulus problems
 *
 * Mathematical foundation:
 * The B-bar method splits the strain into volumetric and deviatoric parts:
 * ε = ε_vol + ε_dev, where ε_vol is volume-averaged over the element
 *
 * This approach prevents spurious pressure oscillations and volumetric locking
 * that can occur with standard displacement-based finite elements when dealing
 * with incompressible or nearly incompressible material behavior.
 *
 * Applications in crystal plasticity:
 * - Incompressible plastic deformation in crystal slip
 * - Nearly incompressible elastic response in metals
 * - Volume-preserving deformation in single crystal simulations
 * - Polycrystalline materials with incompressible phases
 *
 * Key features:
 * - Inherits all standard solid mechanics capabilities from ExaNLFIntegrator
 * - Modifies B-matrix construction for volumetric strain averaging
 * - Maintains compatibility with all assembly strategies (PA, EA, standard)
 * - Provides stable solutions for high bulk modulus materials
 * - Supports large deformation kinematics with volume preservation
 *
 * Implementation details:
 * - Computes element-averaged volumetric strain gradients
 * - Modifies standard B-matrix with B-bar corrections
 * - Uses Hughes' formulation from "The Finite Element Method" Section 4.5.2
 * - Maintains computational efficiency comparable to standard elements
 *
 * @ingroup ExaConstit_fem_operators
 */
class ICExaNLFIntegrator : public ExaNLFIntegrator {
private:
    /** @brief Element-averaged shape function derivatives for B-bar computation */
    mfem::Vector elem_deriv_shapes;

public:
    /**
     * @brief Construct B-bar integrator with simulation state reference.
     *
     * @param sim_state Reference to simulation state containing mesh, fields, and material data
     *
     * Initializes the B-bar method integrator by calling the base ExaNLFIntegrator
     * constructor and preparing data structures for B-bar method computations.
     * The integrator is ready for element assembly operations with incompressible
     * material handling upon construction.
     *
     * The constructor establishes:
     * - Base class initialization for standard solid mechanics operations
     * - Foundation for B-bar method implementation
     * - Integration with ExaConstit's material model framework
     *
     * @note Simulation state reference must remain valid for integrator lifetime
     * @note B-bar specific working vectors allocated during first assembly operation
     */
    ICExaNLFIntegrator(std::shared_ptr<SimulationState> sim_state) : ExaNLFIntegrator(sim_state) {}
    /**
     * @brief Virtual destructor for proper cleanup of derived class resources.
     *
     * Ensures proper cleanup of B-bar integrator resources including any
     * working vectors allocated for element-averaged calculations. The
     * destructor handles cleanup of both base class and derived class data.
     *
     * @note Base class destructor handles ExaNLFIntegrator cleanup
     * @note B-bar specific vectors automatically cleaned up by MFEM Vector destructors
     */
    virtual ~ICExaNLFIntegrator() {}

    /// This doesn't do anything at this point. We can add the functionality
    /// later on if a use case arises.
    using ExaNLFIntegrator::GetElementEnergy;

    using mfem::NonlinearFormIntegrator::AssembleElementVector;
    /**
     * @brief Assemble element residual vector using B-bar method for incompressible materials.
     *
     * @param el Finite element providing shape functions and geometric information
     * @param Ttr Element transformation for coordinate mapping
     * @param elfun Element solution vector (typically nodal velocities or displacements)
     * @param elvect Output element residual vector representing internal forces
     *
     * Computes the element residual vector using the B-bar method to handle
     * incompressible and nearly incompressible material behavior. This method
     * modifies the standard residual computation to include volume-averaged
     * strain measures that prevent volumetric locking.
     *
     * B-bar residual computation:
     * 1. Computes element-averaged volumetric strain gradients over element volume
     * 2. Constructs modified B-bar matrix with volumetric strain averaging
     * 3. Retrieves current stress state from quadrature function data
     * 4. Integrates B-bar^T * σ over element volume using Gauss quadrature
     *
     * Volume averaging process:
     * - Integrates shape function derivatives over entire element
     * - Normalizes by total element volume to obtain averages
     * - Uses averaged derivatives to modify B-matrix construction
     * - Maintains consistency with incompressible deformation constraints
     *
     * The B-bar matrix modification:
     * B-bar = B_standard + B_volumetric_correction
     * where B_volumetric_correction ensures proper volume averaging
     *
     * This approach prevents:
     * - Volumetric locking in nearly incompressible materials
     * - Spurious pressure oscillations in incompressible flow
     * - Poor conditioning in high bulk modulus problems
     * - Artificial stiffening due to volumetric constraints
     *
     * Performance considerations:
     * - Requires additional integration loop for volume averaging
     * - Slightly higher computational cost than standard elements
     * - Significantly improved convergence for incompressible problems
     * - Maintains stability for high bulk modulus materials
     *
     * @note Implements Hughes' B-bar method from FEM book Section 4.5.2
     * @note Requires compatible stress tensor data in simulation state
     * @note Caliper profiling enabled for performance monitoring
     */
    virtual void AssembleElementVector(const mfem::FiniteElement& el,
                                       mfem::ElementTransformation& Ttr,
                                       const mfem::Vector& elfun,
                                       mfem::Vector& elvect) override;

    /**
     * @brief Assemble element tangent stiffness matrix using B-bar method.
     *
     * @param el Finite element providing shape functions and geometric information
     * @param Ttr Element transformation for coordinate mapping
     * @param elfun Element solution vector (unused in current implementation)
     * @param elmat Output element stiffness matrix
     *
     * Computes the element tangent stiffness matrix using the B-bar method for
     * proper handling of incompressible and nearly incompressible materials.
     * This method ensures consistent linearization of the B-bar residual formulation.
     *
     * B-bar tangent stiffness computation:
     * K_element = ∫_Ω B-bar^T(x) C(x) B-bar(x) dΩ
     *
     * The algorithm includes:
     * 1. Computing element-averaged volumetric strain gradients
     * 2. Constructing B-bar matrix with volume averaging corrections
     * 3. Retrieving material tangent stiffness from quadrature function data
     * 4. Integrating B-bar^T * C * B-bar over element volume
     *
     * Volume averaging for stiffness:
     * - Uses same element-averaged derivatives as in residual computation
     * - Ensures consistency between residual and tangent matrix
     * - Maintains proper Newton-Raphson convergence properties
     * - Preserves quadratic convergence near solution
     *
     * B-bar matrix construction:
     * - Modifies volumetric strain components with element averages
     * - Preserves deviatoric strain components from standard B-matrix
     * - Ensures proper rank and stability for incompressible problems
     * - Maintains compatibility with material tangent matrix structure
     *
     * Material tangent integration:
     * - Uses full 6×6 material tangent matrix in Voigt notation
     * - Applies B-bar transformation consistently with residual
     * - Incorporates geometric transformations and quadrature weights
     * - Ensures symmetric tangent matrix for proper solver behavior
     *
     * The resulting stiffness matrix provides:
     * - Stable tangent stiffness for incompressible materials
     * - Proper conditioning for nearly incompressible problems
     * - Consistent linearization of B-bar residual formulation
     * - Quadratic Newton-Raphson convergence properties
     *
     * @note Consistent with B-bar residual formulation in AssembleElementVector
     * @note Material tangent matrix assumed to be 6×6 in Voigt notation
     * @note Caliper profiling enabled for performance analysis
     */
    virtual void AssembleElementGrad(const mfem::FiniteElement& el,
                                     mfem::ElementTransformation& Ttr,
                                     const mfem::Vector& /*elfun*/,
                                     mfem::DenseMatrix& elmat) override;

    /**
     * @brief Initialize partial assembly data structures for B-bar gradient operations.
     *
     * @param fes Finite element space providing mesh and element information
     *
     * Sets up the geometric data needed by AddMultGradPA() and
     * AddMultTransposeGradPA() for the B-bar tangent stiffness operator.
     *
     * Unlike the base class AssembleGradPA() which pre-assembles a 4th-order
     * tensor D, the B-bar version stores only the geometric data (Jacobians,
     * reference shape function derivatives, and element-averaged derivatives)
     * and applies the material tangent C on-the-fly inside the kernel. This
     * is because the B-bar correction couples element-constant data (the
     * volume-averaged derivatives) with quadrature-point-local data (C and
     * adj(J)) in a way that doesn't fold cleanly into a single pre-assembled
     * tensor.
     *
     * Setup steps:
     *   1. Cache space_dims, nqpts, nnodes, nelems from the FES
     *   2. Get geometric factors (Jacobians at quadrature points) from the mesh
     *   3. Compute and cache reference shape function derivatives Gt(a, k, qpt)
     *   4. Ensure element-averaged derivatives N̄(a, k, elem) are available
     *      (calling AssemblePA() if not yet computed)
     *
     * @note Must be called before AddMultGradPA() or AddMultTransposeGradPA()
     * @note Material tangent C is accessed directly from the simulation state
     *       quadrature function during the AddMult kernels
     */
    virtual void AssembleGradPA(const mfem::FiniteElementSpace& fes) override;

    /// State-ful overload that ignores the state vector @a x.
    virtual void AssembleGradPA(const mfem::Vector& x,
                                const mfem::FiniteElementSpace& fes) override;

    /**
     * @brief Apply partial-assembly B-bar tangent stiffness action.
     *
     * @param x Input E-vector (nodal velocities)
     * @param y Output E-vector (accumulated)
     *
     * Computes y += K̄ * x where K̄ = ∫ B̄^T C B̄ dΩ is the B-bar tangent.
     *
     * Algorithm per element, per quadrature point:
     *   1. Compute adj(J) and detJ from the cached Jacobian
     *   2. Compute physical derivatives dN(a,j) on-the-fly from Gt and adj(J)
     *   3. Compute physical velocity gradient L(i,j) = dN(a,j) V(a,i)
     *   4. Compute B-bar trace correction Δtr = (tr_bar - tr(L)) / 3
     *      where tr_bar = N̄(a,k) V(a,k) is element-constant (hoisted)
     *   5. Modified velocity gradient L̄ = L + δ_ij * Δtr
     *   6. Apply C: σ'(j,k) = C(j,k,l,m) * L̄(l,m)
     *   7. Pressure correction p' = (1/3) tr(σ')
     *   8. Accumulate into Y: standard force + B-bar pressure redirection
     *      Y(a,k) += [Σ_j dN(a,j) σ'(j,k) + (N̄(a,k) - dN(a,k)) p'] * w * detJ
     *
     * @note GPU-compatible via mfem::forall
     * @note Requires prior AssembleGradPA() call
     */
    virtual void AddMultGradPA(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Apply transposed B-bar tangent stiffness action.
     *
     * @param x Input E-vector
     * @param y Output E-vector (accumulated)
     *
     * Computes y += K̄^T * x. Identical to AddMultGradPA except the C
     * contraction order is swapped:
     *   Forward:   σ'(j,k) = C(j,k,l,m) * L̄(l,m)
     *   Transpose: σ'(j,k) = C(l,m,j,k) * L̄(l,m)
     *
     * The B-bar geometry (N̄, dN, trace correction, pressure redirection)
     * is identical for both directions because B̄ appears on both the
     * trial and test sides of K̄ = B̄^T C B̄, and (B̄^T C B̄)^T = B̄^T C^T B̄.
     *
     * @note For symmetric C, this produces identical results to AddMultGradPA
     */
    virtual void AddMultTransposeGradPA(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Initialize partial assembly data structures for B-bar residual operations.
     *
     * @param fes Finite element space providing mesh and element information
     *
     * Performs setup for B-bar method partial assembly operations by precomputing
     * geometric factors and element-averaged quantities needed for efficient
     * incompressible material handling in matrix-free operations.
     *
     * B-bar partial assembly setup:
     * 1. Calls base class AssemblePA() for standard geometric factors
     * 2. Computes element-averaged shape function derivatives
     * 3. Stores volume-averaged data for B-bar matrix construction
     * 4. Prepares data structures for efficient B-bar operations
     *
     * Element averaging computation:
     * - Integrates shape function derivatives over each element
     * - Normalizes by element volume to obtain averaged quantities
     * - Stores averaged derivatives for use in AddMultPA operations
     * - Enables consistent B-bar method in partial assembly framework
     *
     * The setup enables:
     * - Memory-efficient B-bar residual assembly via AddMultPA()
     * - Reuse of element-averaged data across multiple assembly calls
     * - Device-compatible data layouts for GPU execution
     * - Efficient handling of incompressible material constraints
     *
     * Performance characteristics:
     * - Slightly higher setup cost due to volume averaging
     * - Amortized over multiple assembly operations
     * - Maintains memory efficiency of partial assembly approach
     * - Enables stable solutions for incompressible problems
     *
     * @note Must be called before AddMultPA() operations for B-bar method
     * @note Element averaging data cached for reuse across assembly calls
     * @note Compatible with base class partial assembly infrastructure
     */
    virtual void AssemblePA(const mfem::FiniteElementSpace& fes) override;
    /**
     * @brief Apply partial assembly B-bar element vector operation.
     *
     * @param x Input vector (unused in current implementation for residual assembly)
     * @param y Output vector for accumulated element contributions
     *
     * Performs the partial assembly B-bar element vector operation, computing
     * element residual contributions using precomputed geometric factors and
     * element-averaged quantities. This provides memory-efficient B-bar method
     * implementation for large-scale incompressible material simulations.
     *
     * B-bar partial assembly operation:
     * - Uses precomputed element-averaged shape function derivatives
     * - Constructs B-bar matrices on-the-fly during assembly
     * - Accesses stress data directly from quadrature functions
     * - Accumulates B-bar contributions directly into global vector
     *
     * The operation sequence:
     * 1. Loops over all elements using precomputed geometric data
     * 2. Constructs B-bar matrix using element-averaged derivatives
     * 3. Applies stress integration with B-bar formulation
     * 4. Accumulates results into global degrees of freedom
     *
     * Volume averaging integration:
     * - Uses cached element-averaged derivatives from AssemblePA()
     * - Applies B-bar corrections to volumetric strain components
     * - Maintains computational efficiency of partial assembly
     * - Prevents volumetric locking in incompressible materials
     *
     * Memory efficiency features:
     * - Minimal additional memory for element averaging data
     * - Direct access to stress quadrature function data
     * - Vectorized operations over elements and quadrature points
     * - Device-compatible implementation for GPU execution
     *
     * This method provides the core B-bar computation in Newton-Raphson
     * iterations while maintaining the memory efficiency advantages of
     * partial assembly for large-scale simulations.
     *
     * @note Requires prior AssemblePA() call for B-bar geometric factor setup
     * @note Input vector x currently unused for stress-based residual assembly
     * @note Output vector y must be properly sized for true DOF space
     */
    virtual void AddMultPA(const mfem::Vector& /*x*/, mfem::Vector& y) const override;
    /**
     * @brief Assemble diagonal entries for B-bar partial assembly preconditioning.
     *
     * @param diag Output vector for diagonal entries of the B-bar tangent stiffness matrix
     *
     * Computes diagonal entries of the B-bar tangent stiffness matrix using
     * partial assembly techniques, providing diagonal approximations essential
     * for Jacobi preconditioning in iterative linear solvers for incompressible
     * material problems.
     *
     * B-bar diagonal computation:
     * 1. Uses precomputed element-averaged derivatives from AssembleGradPA()
     * 2. Constructs B-bar matrix modifications for diagonal extraction
     * 3. Applies material tangent data with B-bar transformations
     * 4. Assembles global diagonal through element restriction operations
     *
     * The diagonal extraction process:
     * - Accounts for B-bar modifications in volumetric strain components
     * - Maintains consistency with B-bar tangent stiffness formulation
     * - Uses vectorized operations for computational efficiency
     * - Handles geometric transformations appropriately
     *
     * Diagonal quality considerations:
     * - B-bar method affects diagonal structure and conditioning
     * - Improved conditioning for incompressible material problems
     * - Better preconditioner effectiveness for nearly incompressible materials
     * - Enhanced Krylov solver convergence for high bulk modulus problems
     *
     * Performance characteristics:
     * - Linear scaling with problem size
     * - Device-compatible implementation for GPU execution
     * - Efficient vectorized operations over elements
     * - Minimal additional memory requirements beyond standard diagonal assembly
     *
     * The resulting diagonal provides:
     * - Effective preconditioning for B-bar systems
     * - Stable iterative solver behavior for incompressible problems
     * - Consistent approximation quality across material parameter ranges
     * - Robust performance for nearly incompressible materials
     *
     * @note Requires prior AssembleGradPA() call for B-bar material data setup
     * @note Diagonal entries reflect B-bar modifications for incompressible behavior
     * @note Caliper profiling enabled for performance monitoring
     */
    virtual void AssembleGradDiagonalPA(mfem::Vector& diag) const override;

    /**
     * @brief Perform B-bar element assembly for gradient operations with solution vector.
     *
     * @param x Solution vector for state-dependent assembly (unused in current implementation)
     * @param fes Finite element space providing mesh and element information
     * @param ea_data Output vector for assembled element matrix data
     *
     * Performs B-bar element assembly for gradient operations, computing and storing
     * complete B-bar element stiffness matrices in a format suitable for element
     * assembly (EA) operations. This method delegates to the base element assembly
     * routine while maintaining B-bar method consistency.
     *
     * B-bar element assembly characteristics:
     * - Computes full B-bar element stiffness matrices
     * - Stores matrices in contiguous device-compatible format
     * - Enables exact B-bar matrix-vector products through explicit element matrices
     * - Provides maximum memory efficiency for large incompressible problems
     *
     * The method serves as an interface for state-dependent B-bar element assembly
     * while currently delegating to the stateless version for implementation.
     * Future extensions could include solution-dependent B-bar modifications.
     *
     * @note Current implementation delegates to AssembleEA(fes, ea_data)
     * @note Solution vector x currently unused but available for future B-bar extensions
     * @note Element matrices include B-bar modifications for incompressible behavior
     */
    virtual void AssembleGradEA(const mfem::Vector& /* x */,
                                const mfem::FiniteElementSpace& fes,
                                mfem::Vector& ea_data) override;

    /**
     * @brief Perform B-bar element assembly for gradient operations.
     *
     * @param fes Finite element space providing mesh and element information
     * @param emat Output vector for assembled B-bar element matrix data
     *
     * Computes and stores complete B-bar element stiffness matrices for all elements
     * in the mesh, providing an element assembly (EA) representation of the B-bar
     * tangent stiffness operator for memory-constrained incompressible material applications.
     *
     * B-bar element assembly process:
     * 1. Iterates over all elements in the mesh
     * 2. Computes element-averaged volumetric derivatives for each element
     * 3. Constructs B-bar element stiffness matrices with volume averaging
     * 4. Stores matrices in contiguous device memory format
     *
     * B-bar matrix computation:
     * - Computes element volume through integration of Jacobian determinants
     * - Calculates element-averaged shape function derivatives
     * - Constructs B-bar matrices with volumetric strain averaging
     * - Integrates B-bar^T * C * B-bar over element volume
     *
     * Memory layout:
     * - B-bar matrices stored element-by-element in contiguous memory
     * - Dense matrices with row-major ordering within each element
     * - Device-compatible allocation for GPU execution
     * - Total size: nelems × (ndof×ncomps)² entries
     *
     * Performance characteristics:
     * - Higher assembly cost due to B-bar volume averaging computations
     * - Minimal memory usage compared to global B-bar matrix assembly
     * - Exact B-bar operator representation without approximation
     * - Excellent stability for incompressible material problems
     *
     * The B-bar element matrices enable:
     * - Exact B-bar matrix-vector products in element assembly operators
     * - Stable solutions for incompressible and nearly incompressible materials
     * - Memory-efficient representation for large-scale problems
     * - Natural parallelization over elements with B-bar consistency
     *
     * @note Supports only 3D problems (1D and 2D problems abort with error)
     * @note Uses RAJA views for optimized B-bar memory layouts and vectorization
     * @note Caliper profiling enabled for performance monitoring
     */
    virtual void AssembleEA(const mfem::FiniteElementSpace& fes, mfem::Vector& emat) override;
};

// }

#endif
