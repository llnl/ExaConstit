#include "solvers/mechanics_solver.hpp"

#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem.hpp"
#include "mfem/general/globals.hpp"
#include "mfem/linalg/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

/**
 * @brief Set operator implementation for general Operator
 *
 * @details This implementation:
 * 1. Stores the operator reference and extracts dimensions
 * 2. Validates that the operator is square (required for Newton method)
 * 3. Initializes residual and correction vectors with device memory
 * 4. Configures vectors for GPU execution when available
 */
void ExaNewtonSolver::SetOperator(const mfem::Operator& op) {
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT_0(height == width, "square Operator is required.");

    r.SetSize(width, mfem::Device::GetMemoryType());
    r.UseDevice(true);
    c.SetSize(width, mfem::Device::GetMemoryType());
    c.UseDevice(true);
}

/**
 * @brief Set operator implementation for NonlinearForm
 *
 * @details This specialized implementation:
 * 1. Stores both the NonlinearForm reference and base Operator interface
 * 2. Enables specialized mechanics operations through oper_mech pointer
 * 3. Provides same setup as general Operator version
 * 4. Allows access to mechanics-specific functionality
 */
void ExaNewtonSolver::SetOperator(const std::shared_ptr<mfem::NonlinearForm> op) {
    oper_mech = op;
    oper = op.get();
    height = op->Height();
    width = op->Width();
    MFEM_ASSERT_0(height == width, "square NonlinearForm is required.");

    r.SetSize(width, mfem::Device::GetMemoryType());
    r.UseDevice(true);
    c.SetSize(width, mfem::Device::GetMemoryType());
    c.UseDevice(true);
}

/**
 * @brief Newton-Raphson iteration implementation
 *
 * @details The implementation includes several advanced features:
 *
 * **Adaptive Scaling**: Monitors convergence rate and automatically reduces
 * step size when norm_ratio = norm_current/norm_previous > 0.5
 *
 * **Device Compatibility**: All vector operations are device-aware for GPU execution
 *
 * **Convergence Criteria**: Uses combined absolute and relative tolerance:
 * norm_max = max(rel_tol * norm_0, abs_tol)
 *
 * **Performance Monitoring**: Includes Caliper profiling scopes for:
 * - Overall Newton solver performance ("NR_solver")
 * - Individual Krylov solver calls ("krylov_solver")
 *
 * **Error Handling**: Validates finite residual norms and proper setup
 */
void ExaNewtonSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const {
    CALI_CXX_MARK_SCOPE("NR_solver");
    MFEM_ASSERT_0(oper_mech, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT_0(prec_mech, "the Solver is not set (use SetSolver).");

    int it;
    double norm0, norm, norm_max;
    double norm_prev, norm_ratio;
    const bool have_b = (b.Size() == Height());

    // Might want to use this to fix things later on for example when we have a
    // large residual. We might also want to eventually try and find a converged
    // relaxation factor which would mean resetting our solution vector a few times.
    mfem::Vector x_prev(x.Size());
    x_prev.UseDevice(true);

    if (!iterative_mode) {
        x = 0.0;
    }

    x_prev = x;

    oper_mech->Mult(x, r);
    if (have_b) {
        r -= b;
    }

    norm0 = norm = norm_prev = Norm(r);
    norm_ratio = 1.0;
    // Set the value for the norm that we'll exit on
    norm_max = std::max(rel_tol * norm, abs_tol);

    prec_mech->iterative_mode = false;
    double scale = 1.0;

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++) {
        // Make sure the norm is finite
        MFEM_ASSERT_0(mfem::IsFinite(norm), "norm = " << norm);
        if (print_level >= 0) {
            mfem::out << "Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
            if (it > 0) {
                mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
            }
            mfem::out << '\n';
        }
        // See if our solution has converged and we can quit
        if (norm <= norm_max) {
            converged = 1;
            break;
        }
        // See if we've gone over the max number of desired iterations
        if (it >= max_iter) {
            converged = 0;
            break;
        }

        prec_mech->SetOperator(oper_mech->GetGradient(x));
        CALI_MARK_BEGIN("krylov_solver");
        prec_mech->Mult(r, c); // c = [DF(x_i)]^{-1} [F(x_i)-b]
                               // ExaConstit may use GMRES here

        CALI_MARK_END("krylov_solver");
        const double c_scale = scale;
        if (c_scale == 0.0) {
            converged = 0;
            break;
        }

        add(x, -c_scale, c, x); // full update to the current config
                                // ExaConstit (srw)

        // We now get our new residual
        oper_mech->Mult(x, r);
        if (have_b) {
            r -= b;
        }

        // Find our new norm and save our previous time step value.
        norm_prev = norm;
        norm = Norm(r);
        // We're going to more or less use a heuristic method here for now if
        // our ratio is greater than 1e-1 then we'll set our scaling factor for
        // the next iteration to 0.5.
        // We want to do this since it's not uncommon for us to run into the case
        // where our solution is oscillating over the one we actually want.
        // Eventually, we'll fix this in our scaling factor function.
        norm_ratio = norm / norm_prev;

        if (norm_ratio > 5.0e-1) {
            scale = 0.5;
            if (print_level >= 0) {
                mfem::out << "The relaxation factor for the next iteration has been reduced to "
                          << scale << "\n";
            }
        } else {
            scale = 1.0;
        }
    }

    final_iter = it;
    final_norm = norm;
}

/**
 * @brief Linear solver interface implementation
 *
 * @details Simple wrapper that:
 * 1. Sets up the preconditioner with the current operator (typically Jacobian)
 * 2. Applies the preconditioner to solve the linear system
 * 3. Includes Caliper profiling for linear solver performance
 *
 * @note Despite the name "CGSolver", this method can use any linear solver
 * (CG, GMRES, MINRES) depending on the solver configured via SetSolver()
 */
void ExaNewtonSolver::CGSolver(mfem::Operator& oper, const mfem::Vector& b, mfem::Vector& x) const {
    prec_mech->SetOperator(oper);
    CALI_MARK_BEGIN("krylov_solver");
    prec_mech->Mult(b, x); // c = [DF(x_i)]^{-1} [F(x_i)-b]
                           // ExaConstit may use GMRES here

    CALI_MARK_END("krylov_solver");
}

/**
 * @brief Line search Newton implementation
 *
 * @details The line search algorithm implementation:
 *
 * **Quadratic Line Search Theory**:
 * Given three points and their residual norms (q1, q2, q3), the algorithm
 * fits a quadratic polynomial q(s) = as² + bs + c to find the minimum.
 * The optimal step size is: ε = -b/(2a) = (3*q1 - 4*q2 + q3) / (4*(q1 - 2*q2 + q3))
 *
 * **Robustness Checks**:
 * - Validates quadratic fit: (q1 - 2*q2 + q3) > 0 (convex)
 * - Bounds step size: 0 < ε < 1 (reasonable range)
 * - Fallback logic when quadratic fit fails
 *
 * **Performance Profiling**:
 * - "NRLS_solver" scope for overall line search Newton performance
 * - "Line Search" scope specifically for step size computation
 * - "krylov_solver" scope for linear solver calls
 *
 * **Memory Management**:
 * - Uses device-compatible temporary vectors (x_prev, Jr)
 * - Efficient vector operations with MFEM's device interface
 *
 * **Failure Handling**:
 * - Scale factor of 0.0 triggers immediate convergence failure
 * - Graceful degradation when line search produces invalid results
 */
void ExaNewtonLSSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const {
    CALI_CXX_MARK_SCOPE("NRLS_solver");
    MFEM_ASSERT_0(oper_mech, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT_0(prec_mech, "the Solver is not set (use SetSolver).");

    int it;
    double norm0, norm, norm_max;
    const bool have_b = (b.Size() == Height());

    // Might want to use this to fix things later on for example when we have a
    // large residual. We might also want to eventually try and find a converged
    // relaxation factor which would mean resetting our solution vector a few times.
    mfem::Vector x_prev(x.Size());
    mfem::Vector Jr(x.Size());
    Jr.UseDevice(true);
    x_prev.UseDevice(true);

    if (!iterative_mode) {
        x = 0.0;
    }

    x_prev = x;

    oper_mech->Mult(x, r);
    if (have_b) {
        r -= b;
    }

    norm0 = norm = Norm(r);
    // Set the value for the norm that we'll exit on
    norm_max = std::max(rel_tol * norm, abs_tol);

    prec_mech->iterative_mode = false;
    double scale = 1.0;

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++) {
        // Make sure the norm is finite
        MFEM_ASSERT_0(mfem::IsFinite(norm), "norm = " << norm);
        if (print_level >= 0) {
            mfem::out << "Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
            if (it > 0) {
                mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
            }
            mfem::out << '\n';
        }
        // See if our solution has converged and we can quit
        if (norm <= norm_max) {
            converged = 1;
            break;
        }
        // See if we've gone over the max number of desired iterations
        if (it >= max_iter) {
            converged = 0;
            break;
        }

        prec_mech->SetOperator(oper_mech->GetGradient(x));
        CALI_MARK_BEGIN("krylov_solver");
        prec_mech->Mult(r, c); // c = [DF(x_i)]^{-1} [F(x_i)-b]
                               // ExaConstit may use GMRES here
        CALI_MARK_END("krylov_solver");
        // This line search method is based on the quadratic variation of the norm
        // of the residual line search described in this conference paper:
        // https://doi.org/10.1007/978-3-642-01970-8_46 . We can probably do better
        // than this one.
        {
            CALI_CXX_MARK_SCOPE("Line Search");
            x_prev = x;
            add(x, -1.0, c, x);
            oper_mech->Mult(x, r);
            if (have_b) {
                r -= b;
            }
            double q1 = norm;
            double q3 = Norm(r);
            x = x_prev;
            add(x, -0.5, c, x);
            oper_mech->Mult(x, r);
            if (have_b) {
                r -= b;
            }
            double q2 = Norm(r);

            double eps = (3.0 * q1 - 4.0 * q2 + q3) / (4.0 * (q1 - 2.0 * q2 + q3));

            // Smallest relaxation factor we'll allow the line search to take
            constexpr double min_scale = 0.05;

            if ((q1 - 2.0 * q2 + q3) > 0 && eps > 0 && eps < 1) {
                // The quadratic minimum can land arbitrarily close to 0, so make sure
                // we don't go below our minimum relaxation factor
                scale = std::max(eps, min_scale);
            } else if (q3 < q1) {
                scale = 1.0;
            } else {
                // We should probably just quit if this is the case...
                scale = min_scale;
            }

            if (print_level >= 0) {
                mfem::out << "The relaxation factor for this iteration is " << scale << std::endl;
            }

            x = x_prev;
        }

        const double c_scale = scale;
        if (c_scale == 0.0) {
            converged = 0;
            break;
        }

        add(x, -c_scale, c, x); // full update to the current config
                                // ExaConstit (srw)

        // We now get our new residual
        oper_mech->Mult(x, r);
        if (have_b) {
            r -= b;
        }

        // Find our new norm
        norm = Norm(r);
    }

    final_iter = it;
    final_norm = norm;
}