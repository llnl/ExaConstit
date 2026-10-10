
#ifndef MECHANICS_SOLVER
#define MECHANICS_SOLVER

#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"

#include <memory>
/**
 * @brief Newton-Raphson solver for nonlinear solid mechanics problems
 *
 * @details This class implements Newton's method for solving nonlinear systems of the form F(x) = b
 * where F is a nonlinear operator. It extends MFEM's IterativeSolver to provide specialized
 * functionality for ExaConstit's solid mechanics applications.
 *
 * The solver uses the Newton-Raphson iteration:
 * x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i) - b]
 *
 * Key features:
 * - Device-compatible implementation for CPU/GPU execution
 * - Integration with MFEM's operator and linear solver framework
 * - Specialized handling for NonlinearForm operators in solid mechanics
 * - Automatic scaling factor adjustment for convergence improvement
 * - Caliper performance profiling integration
 *
 * The method GetGradient() must be implemented for the operator F.
 * The preconditioner is used (in non-iterative mode) to evaluate
 * the action of the inverse gradient of the operator.
 */
class ExaNewtonSolver : public mfem::IterativeSolver {
protected:
    /** @brief Residual vector for Newton iterations */
    mutable mfem::Vector r;

    /** @brief Correction vector for Newton iterations */
    mutable mfem::Vector c;

    /** @brief Pointer to the mechanics nonlinear form operator */
    std::shared_ptr<mfem::NonlinearForm> oper_mech;

    /** @brief Pointer to the preconditioner */
    std::shared_ptr<mfem::Solver> prec_mech;

    /**
     * @brief Print the residual information for a single nonlinear iteration
     *
     * @param it Current nonlinear iteration number
     * @param norm Current residual norm
     * @param norm0 Initial residual norm
     *
     * @details Only prints when print_level >= 1. All nonlinear solvers use this so
     * that the per-iteration output has the same format regardless of the solver.
     */
    void PrintIterationInfo(int it, double norm, double norm0) const;

    /**
     * @brief Print the scale applied to the Newton step taken from an iteration
     *
     * @param it Nonlinear iteration number the step is being taken from
     * @param scale Fraction of the full Newton step that is being taken
     *
     * @details Only prints when print_level >= 1. Each solver provides its own
     * notion of the scale (relaxation factor for NR, line search factor for NRLS,
     * and the trust-region limited fraction of the Newton step for TRDOG), but it is
     * always printed in the same format so the output can be parsed the same way
     * regardless of the solver. A scale that is small and stays small is a sign
     * that the solver is struggling.
     */
    void PrintStepScaleInfo(int it, double scale) const;

    /**
     * @brief Print a one line summary of the nonlinear solve
     *
     * @param it Number of nonlinear iterations that were run
     * @param norm Final residual norm
     * @param norm0 Initial residual norm
     * @param did_converge Whether the nonlinear solve converged
     * @param num_rejected Number of rejected trial steps for solvers that have them
     *                     (e.g. trust-region). Not printed if negative.
     *
     * @details Only prints when print_level == 0. In this mode the per-iteration
     * output is turned off, so this provides the basic information of the number of
     * iterations, the final residual, and the final relative residual.
     */
    void PrintSummaryInfo(
        int it, double norm, double norm0, bool did_converge, int num_rejected = -1) const;

public:
    /**
     * @brief Default constructor
     *
     * @details Creates an ExaNewtonSolver instance for single-processor execution.
     * The operator and linear solver must be set separately using SetOperator() and SetSolver().
     */
    ExaNewtonSolver() {}

#ifdef MFEM_USE_MPI
    /**
     * @brief MPI constructor
     *
     * @param _comm MPI communicator for parallel execution
     *
     * @details Creates an ExaNewtonSolver instance for parallel execution using the specified
     * MPI communicator. This enables the solver to work with distributed finite element spaces
     * and parallel linear solvers.
     */
    ExaNewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) {}
#endif
    /**
     * @brief Set the nonlinear operator to be solved
     *
     * @param op The nonlinear operator representing F in F(x) = b
     *
     * @details Sets up the solver to work with the given operator. The operator must be square
     * (height == width) and must implement the GetGradient() method for computing Jacobians.
     * This method also initializes the internal residual and correction vectors with appropriate
     * device memory settings.
     *
     * @pre The operator must be square (height == width)
     * @post Internal vectors r and c are sized and configured for device execution
     */
    virtual void SetOperator(const mfem::Operator& op);

    /**
     * @brief Set the nonlinear form operator to be solved
     *
     * @param op The nonlinear form representing the mechanics problem
     *
     * @details Specialized version for MFEM NonlinearForm operators, which are commonly used
     * in finite element mechanics problems. This method stores both the general operator
     * interface and the specific NonlinearForm pointer for specialized mechanics operations.
     *
     * @pre The NonlinearForm must be square (height == width)
     * @post Both oper and oper_mech pointers are set, internal vectors are initialized
     */
    virtual void SetOperator(const std::shared_ptr<mfem::NonlinearForm> op);

    /**
     * @brief Set the linear solver for inverting the Jacobian
     *
     * @param solver Linear solver for the Newton correction equation
     *
     * @details This method is equivalent to calling SetPreconditioner(). The linear solver
     * is used to solve the linearized system [DF(x_i)] c = [F(x_i) - b] at each Newton iteration.
     * Common choices include:
     * - CGSolver for symmetric positive definite systems
     * - GMRESSolver for general nonsymmetric systems
     * - MINRESSolver for symmetric indefinite systems
     */
    virtual void SetSolver(mfem::Solver& solver) {
        prec = &solver;
    }

    /**
     * @brief Set the linear solver for inverting the Jacobian
     *
     * @param solver Linear solver for the Newton correction equation
     *
     * @details This method is equivalent to calling SetPreconditioner(). The linear solver
     * is used to solve the linearized system [DF(x_i)] c = [F(x_i) - b] at each Newton iteration.
     * Common choices include:
     * - CGSolver for symmetric positive definite systems
     * - GMRESSolver for general nonsymmetric systems
     * - MINRESSolver for symmetric indefinite systems
     */
    virtual void SetSolver(std::shared_ptr<mfem::Solver> solver) {
        prec_mech = solver;
    }

    /**
     * @brief Solve the linearized Newton correction equation
     *
     * @param oper Linear operator (typically the Jacobian)
     * @param b Right-hand side vector
     * @param x Solution vector (output)
     *
     * @details This method solves the linearized Newton system using the configured linear solver.
     * It sets up the preconditioner/solver with the given operator and applies it to compute
     * the Newton correction. The method is marked with Caliper profiling for performance analysis.
     *
     * The operation performed is: x = [oper]^{-1} b
     *
     * @note This method may use different Krylov solvers (CG, GMRES, MINRES) depending on
     * the configuration provided during solver setup.
     */
    virtual void CGSolver(mfem::Operator& oper, const mfem::Vector& b, mfem::Vector& x) const;

    /**
     * @brief Solve the nonlinear system F(x) = b using Newton-Raphson method
     *
     * @param b Right-hand side vector (if b.Size() != Height(), assumes b = 0)
     * @param x Solution vector (input: initial guess, output: converged solution)
     *
     * @details Main solution method that implements the Newton-Raphson algorithm:
     *
     * 1. **Initialization**: Set up initial residual r = F(x) - b
     * 2. **Newton Iteration Loop**:
     *    - Check convergence: ||r|| <= max(rel_tol * ||r_0||, abs_tol)
     *    - Compute Jacobian: J = DF(x_i)
     *    - Solve linear system: J * c = r
     *    - Apply scaling factor: x_{i+1} = x_i - scale * c
     *    - Update residual: r = F(x_{i+1}) - b
     *    - Adjust scaling factor if convergence stalls
     * 3. **Convergence Check**: Exit when tolerance is met or max iterations reached
     *
     * **Adaptive Scaling**: The solver automatically reduces the scaling factor to 0.5
     * when the residual ratio exceeds 0.5, helping to stabilize convergence for
     * difficult nonlinear problems.
     *
     * **Performance Profiling**: Includes Caliper markers for detailed performance analysis
     * of Newton iterations and linear solver calls.
     *
     * @pre SetOperator() and SetSolver() must be called before Mult()
     * @pre The operator must implement GetGradient() for Jacobian computation
     *
     * @post final_iter contains the number of Newton iterations performed
     * @post final_norm contains the final residual norm
     * @post converged flag indicates whether the solver converged
     */
    virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;

    // We're going to comment this out for now.
    /** @brief This method can be overloaded in derived classes to implement line
        search algorithms. */
    /** The base class implementation (NewtonSolver) simply returns 1. A return
        value of 0 indicates a failure, interrupting the Newton iteration. */
    // virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const
    // { return 1.0; }
};

/**
 * @brief Newton-Raphson solver with line search for enhanced convergence
 *
 * @details This class extends ExaNewtonSolver to include a line search algorithm that
 * improves convergence robustness for highly nonlinear problems. The line search method
 * uses a quadratic variation approach to find an optimal scaling factor for each Newton step.
 *
 * The line search algorithm:
 * 1. Evaluates the residual at three points: x, x - 0.5*c, x - c
 * 2. Fits a quadratic polynomial to these residual norms
 * 3. Finds the minimum of the quadratic to determine optimal step size
 * 4. Falls back to heuristic rules if the quadratic fit is invalid
 *
 * This approach is particularly useful for:
 * - Large deformation problems with geometric nonlinearities
 * - Material models with strong nonlinearities (e.g., plasticity, damage)
 * - Problems where standard Newton-Raphson exhibits oscillatory behavior
 *
 * The method GetGradient() must be implemented for the operator F.
 * The preconditioner is used (in non-iterative mode) to evaluate
 * the action of the inverse gradient of the operator.
 *
 * Reference: Based on quadratic variation line search described in
 * "Numerical Methods for Large Eigenvalue Problems" (https://doi.org/10.1007/978-3-642-01970-8_46)
 */
class ExaNewtonLSSolver : public ExaNewtonSolver {
public:
    /**
     * @brief Default constructor
     *
     * @details Creates an ExaNewtonLSSolver instance for single-processor execution.
     * Inherits all functionality from ExaNewtonSolver and adds line search capabilities.
     */
    ExaNewtonLSSolver() {}

#ifdef MFEM_USE_MPI
    /**
     * @brief MPI constructor
     *
     * @param _comm MPI communicator for parallel execution
     *
     * @details Creates an ExaNewtonLSSolver instance for parallel execution using the specified
     * MPI communicator. The line search algorithm works correctly in parallel environments.
     */
    ExaNewtonLSSolver(MPI_Comm _comm) : ExaNewtonSolver(_comm) {}
#endif
    /** @brief Use parent class SetOperator methods */
    using ExaNewtonSolver::SetOperator;

    /** @brief Use parent class SetSolver methods */
    using ExaNewtonSolver::SetSolver;

    /** @brief Use parent class CGSolver method */
    using ExaNewtonSolver::CGSolver;

    /**
     * @brief Solve the nonlinear system F(x) = b using Newton-Raphson with line search
     *
     * @param b Right-hand side vector (if b.Size() != Height(), assumes b = 0)
     * @param x Solution vector (input: initial guess, output: converged solution)
     *
     * @details Enhanced Newton-Raphson method with quadratic line search for improved robustness:
     *
     * 1. **Standard Newton Setup**: Compute residual and Jacobian as in standard Newton
     * 2. **Line Search Algorithm**:
     *    - Store current state: x_prev = x
     *    - Evaluate residual at x - c: q3 = ||F(x - c) - b||
     *    - Evaluate residual at x - 0.5*c: q2 = ||F(x - 0.5*c) - b||
     *    - Current residual: q1 = ||F(x) - b||
     *    - Fit quadratic: ε = (3*q1 - 4*q2 + q3) / (4*(q1 - 2*q2 + q3))
     *    - Apply optimal step: x = x_prev - ε*c
     * 3. **Fallback Strategy**:
     *    - If quadratic fit is invalid: ε = 1.0 (full Newton step)
     *    - If full step increases residual: ε = 0.05 (heavily damped step)
     *    - If algorithm fails completely: terminate with convergence failure
     *
     * **Line Search Benefits**:
     * - Prevents divergence in highly nonlinear problems
     * - Reduces oscillatory behavior near solution
     * - Maintains quadratic convergence when possible
     * - Provides automatic step size control
     *
     * **Performance Considerations**:
     * - Requires 2 additional function evaluations per iteration
     * - Includes Caliper profiling for line search performance analysis
     * - May increase computational cost but improves robustness
     *
     * @pre SetOperator() and SetSolver() must be called before Mult()
     * @pre The operator must implement GetGradient() for Jacobian computation
     *
     * @post final_iter contains the number of Newton iterations performed
     * @post final_norm contains the final residual norm
     * @post converged flag indicates whether the solver converged
     *
     * @note The line search algorithm prints the relaxation factor when print_level >= 0
     */
    virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;
};

#endif
