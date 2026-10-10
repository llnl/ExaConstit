// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other ExaConstit Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "solvers/trust_region_solver.hpp"

#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem.hpp"
#include "mfem/general/globals.hpp"
#include "mfem/linalg/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

/**
 * @brief Compute the Powell dogleg step inside the trust region.
 *
 * @details Step-by-step algorithm:
 *
 * 1. **Full Newton step inside trust region**:
 *    If ||s_N|| <= delta, take the full Newton step. The predicted residual
 *    is zero (the linear model F + J*s_N = 0 is exactly satisfied).
 *
 * 2. **Cauchy point outside trust region**:
 *    Compute the Cauchy point parameters:
 *       - alpha = ||g||^2 / ||J*g||^2   (optimal scaling along steepest descent)
 *       - ||s_sd_opt|| = alpha * ||g||  (norm of the optimal Cauchy step)
 *    If ||s_sd_opt|| >= delta, the optimal Cauchy point is outside the trust
 *    region. Step along the steepest descent direction to the boundary:
 *       delx = -delta * g / ||g||
 *    The predicted residual norm is computed from the linear model evaluated
 *    at this truncated Cauchy step.
 *
 * 3. **Dogleg interpolation (second leg)**:
 *    Otherwise, interpolate along the line segment from the Cauchy point to
 *    the Newton point, finding the parameter beta in [0, 1] such that the
 *    interpolated step lies on the trust-region boundary. The intersection
 *    is found by solving a quadratic:
 *       delx(beta) = beta * s_N - (1 - beta) * alpha * g
 *       ||delx(beta)||^2 = delta^2
 *    yielding qa*beta^2 - 2*qb*beta + qc = 0 where:
 *       qa = ||p||^2,   qb = alpha * (p . g),   qc = ||s_sd_opt||^2 - delta^2
 *       and p = s_N + alpha * g.
 *    Beta is taken from the larger root and clamped to [0, 1] for safety.
 */
void ExaTrustRegionSolver::Dogleg(double delta,
                                  double res_0,
                                  double nr_norm,
                                  double Jg_2,
                                  const mfem::Vector& grad,
                                  const mfem::Vector& nrStep,
                                  mfem::Vector& delx,
                                  double& pred_resid,
                                  DoglegStep& step,
                                  double& beta) const {
    beta = 0.0;

    // --- Case 1: Full Newton step fits inside the trust region ---
    if (nr_norm <= delta) {
        step = DoglegStep::NEWTON;
        delx = nrStep;
        pred_resid = 0.0;
        return;
    }

    // Cauchy point parameters using MPI-aware dot products
    const double norm2_grad = Dot(grad, grad);
    const double norm_grad = std::sqrt(norm2_grad);

    const double alpha = (Jg_2 > 0.0) ? (norm2_grad / Jg_2) : 1.0;
    const double norm_grad_inv = (norm_grad > 0.0) ? (1.0 / norm_grad) : 1.0;
    const double norm_s_sd_opt = alpha * norm_grad;

    // --- Case 2: Cauchy point is outside the trust region ---
    // Take a step along the steepest descent direction to the trust-region boundary
    if (norm_s_sd_opt >= delta) {
        // delx = -delta * (grad / ||grad||)
        const double factor = -delta * norm_grad_inv;
        delx = grad;
        delx *= factor;

        // Predicted residual from linear model at the truncated Cauchy step
        const double val = -(delta * norm_grad) +
                           0.5 * delta * delta * Jg_2 * (norm_grad_inv * norm_grad_inv);
        pred_resid = std::sqrt(std::max(2.0 * val + res_0 * res_0, 0.0));
        step = DoglegStep::STEEPEST_DESCENT;
    }
    // --- Case 3: Cauchy inside, Newton outside; interpolate along the second leg ---
    else {
        // Reuse delx as workspace for p = nrStep + alpha * grad
        mfem::Vector& p = delx;
        add(nrStep, alpha, grad, p);

        // Quadratic coefficients for the trust-region boundary intersection
        double qa = Dot(p, p);
        double qb = Dot(p, grad) * alpha;
        double qc = norm_s_sd_opt * norm_s_sd_opt - delta * delta;

        double discriminant = qb * qb - qa * qc;
        beta = (qa > 0.0) ? (qb + std::sqrt(std::max(discriminant, 0.0))) / qa : 0.0;

        // Clamp beta to [0, 1] to handle any roundoff at the boundary
        beta = std::max(0.0, std::min(1.0, beta));

        // delx = beta * nrStep - (1 - beta) * alpha * grad
        const double omb = 1.0 - beta;
        const double omba = omb * alpha;
        add(beta, nrStep, -omba, grad, delx);

        // Predicted residual from linear model at the dogleg step
        const double res_cauchy = (Jg_2 > 0.0)
                                      ? std::sqrt(std::max(res_0 * res_0 - alpha * norm2_grad, 0.0))
                                      : res_0;
        pred_resid = omb * res_cauchy;
        step = DoglegStep::DOGLEG;
    }
}

/**
 * @brief Detailed per-iteration trust-region output (print_level > 1)
 *
 * @details Example output line:
 *    TR step: dogleg (beta = 0.42), predicted ||r|| = 1.2e-03, trial ||r|| = 1.3e-03,
 *             rho = 0.83, delta = 0.5 -> 1, accepted
 * (printed as a single line)
 */
void ExaTrustRegionSolver::PrintStepDetails(DoglegStep step,
                                            double beta,
                                            double pred_resid,
                                            double trial_resid,
                                            double rho,
                                            double delta_old,
                                            double delta_new,
                                            const char* outcome) const {
    if (print_level < 2) {
        return;
    }
    mfem::out << "    TR step: ";
    switch (step) {
    case DoglegStep::NEWTON:
        mfem::out << "Newton";
        break;
    case DoglegStep::STEEPEST_DESCENT:
        mfem::out << "steepest descent";
        break;
    case DoglegStep::DOGLEG:
        mfem::out << "dogleg (beta = " << beta << ")";
        break;
    }
    mfem::out << ", predicted ||r|| = " << pred_resid << ", trial ||r|| = " << trial_resid
              << ", rho = ";
    if (std::isnan(rho)) {
        mfem::out << "n/a";
    } else {
        mfem::out << rho;
    }
    mfem::out << ", delta = " << delta_old;
    if (delta_new != delta_old) {
        mfem::out << " -> " << delta_new;
    }
    mfem::out << ", " << outcome << '\n';
}

/**
 * @brief Trust-region dogleg Newton iteration implementation.
 *
 * @details Step-by-step algorithm for solving F(x) = b:
 *
 * **Initial setup**:
 *   1. Validate that operator (oper_mech), preconditioner (prec_mech), and
 *      delta_ctrl are properly configured
 *   2. Allocate all device-aware working vectors (nrStep, grad, delx, Jg_temp,
 *      x_prev) once before the iteration loop
 *   3. Evaluate initial residual r = F(x) - b and compute its norm
 *   4. Set the convergence threshold norm_max = max(rel_tol * res, abs_tol)
 *   5. Initialize trust-region radius delta from delta_ctrl.deltaInit
 *
 * **Main iteration loop** (until convergence or max_iter accepted iterations):
 *   1. If the previous step was *not* rejected, recompute Newton machinery:
 *      a. Get Jacobian J = oper_mech->GetGradient(x). The material state is
 *         consistent with x because Mult(x, r) was just evaluated.
 *      b. Compute steepest descent: grad = J^T * r (gradient of f = 0.5 ||F||^2)
 *      c. Compute Jg_2 = ||J * grad||^2 for the optimal Cauchy step length
 *      d. Solve the Newton system J*c = r via the Krylov solver (prec_mech),
 *         then negate: nrStep = -c. The negation matches SNLS convention where
 *         the Newton update is x += nrStep (whereas ExaNewtonSolver uses x -= c).
 *      e. Compute nr_norm = ||nrStep||
 *      If the previous step *was* rejected, all of this data is still valid
 *      from the last accepted iteration and we just recompute the dogleg with
 *      the smaller delta.
 *   2. Save x_prev = x for potential rollback on rejection
 *   3. Compute the dogleg step delx via Dogleg() helper
 *   4. Apply the trial step: x = x_prev + delx
 *   5. Evaluate residual at the trial point: r = F(x) - b
 *   6. Check convergence: if ||r|| <= norm_max, accept and exit
 *   7. Update delta via delta_ctrl.UpdateDelta() based on actual vs predicted
 *      reduction. This may also flag the step for rejection.
 *   8. If rejected: restore x = x_prev, restore residual norm, set reject_prev.
 *      The material state inside the model handles itself analogously to the
 *      ExaNewtonLSSolver line-search behavior — when Mult() is called again at
 *      the next trial point, the model recomputes from the beginning-step state.
 *      Rejected steps do not count as iterations (unlike SNLS) since they are
 *      retries of the same iteration, much like the trial evaluations of a line
 *      search. An internal cap on consecutive rejections guards against an
 *      endless loop, although deltaMin should normally be reached first.
 *   9. If accepted: the iteration count is incremented and the step scale and
 *      new residual are printed in the same format as the other solvers.
 *
 * **Performance Profiling**:
 *   - "TR_dogleg_solver" scope for overall trust-region solver performance
 *   - "TR_newton_setup" scope for J^T*r and J*g computations
 *   - "TR_gradient_transpose" scope for the J^T*r call specifically
 *   - "TR_newton_solve" scope for the Krylov inner solve
 *   - "TR_trial_eval" scope for residual evaluations at trial points
 *   - "krylov_solver" scope for the actual Krylov solver call
 *
 * @note All scalar quantities (norms, dot products) use MFEM's MPI-aware
 *       Norm() and Dot() functions through the IterativeSolver base class
 */
void ExaTrustRegionSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const {
    CALI_CXX_MARK_SCOPE("TR_dogleg_solver");
    MFEM_ASSERT_0(oper_mech, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT_0(prec_mech, "the Solver is not set (use SetSolver).");
    MFEM_ASSERT(delta_ctrl.Validate(), "TrDeltaControl parameters are invalid.");

    const bool have_b = (b.Size() == Height());

    // --- Allocate working vectors once, reused across iterations ---
    mfem::Vector nrStep(width, mfem::Device::GetMemoryType());
    mfem::Vector grad(width, mfem::Device::GetMemoryType());
    mfem::Vector delx(width, mfem::Device::GetMemoryType());
    mfem::Vector Jg_temp(width, mfem::Device::GetMemoryType());
    mfem::Vector x_prev(width, mfem::Device::GetMemoryType());

    nrStep.UseDevice(true);
    grad.UseDevice(true);
    delx.UseDevice(true);
    Jg_temp.UseDevice(true);
    x_prev.UseDevice(true);

    // --- Initial residual evaluation: r = F(x) - b ---
    oper_mech->Mult(x, r);
    if (have_b) {
        r -= b;
    }

    double res = Norm(r);
    // Residual norm at the start of the solve used for the relative residual
    const double norm0 = res;
    // Residual norm at the last accepted state
    double res_0 = res;
    const double norm_max = std::max(rel_tol * res, abs_tol);

    MFEM_ASSERT_0(mfem::IsFinite(res), "norm = " << res);
    PrintIterationInfo(0, res, norm0);

    if (res <= norm_max) {
        converged = true;
        final_iter = 0;
        final_norm = res;
        PrintSummaryInfo(0, res, norm0, true, 0);
        return;
    }

    // --- Initialize trust-region state ---
    double delta = delta_ctrl.deltaInit;
    double rho = 0.0;
    bool reject_prev = false;

    // Persisted across iterations when a step is not rejected
    double Jg_2 = 0.0;
    double nr_norm = 0.0;

    // Only accepted steps count as iterations. A rejected trial step is retried from
    // the same state with a smaller trust region, reusing the Jacobian and Newton step,
    // so it is treated like the extra residual evaluations of a line search rather than
    // a full iteration. This keeps max_iter, final_iter, and the per-iteration output
    // consistent with the other nonlinear solvers. Note, this differs from SNLS where
    // rejected steps also count towards the iteration count.
    int it = 0;
    int num_rejected = 0;
    int num_consecutive_rejected = 0;
    // Internal safeguard against endlessly rejecting steps. In practice the trust region
    // radius should reach deltaMin well before this is hit.
    constexpr int max_consecutive_rejected = 50;
    converged = false;

    // --- Main iteration loop ---
    while (it < max_iter) {
        // If the previous step was not rejected, recompute Newton direction
        // and steepest descent direction at the current x. The Jacobian data
        // is current because oper_mech->Mult(x, r) was just called.
        if (!reject_prev) {
            CALI_CXX_MARK_SCOPE("TR_newton_setup");

            mfem::Operator& J = oper_mech->GetGradient(x);

            // Steepest descent direction: grad = J^T * r
            // This is the gradient of the merit function f(x) = 0.5 * ||F(x)||^2
            {
                CALI_CXX_MARK_SCOPE("TR_gradient_transpose");
                J.MultTranspose(r, grad);
            }

            // Compute ||J * grad||^2 for the optimal Cauchy step length
            //    alpha_cauchy = ||grad||^2 / ||J*grad||^2
            {
                J.Mult(grad, Jg_temp);
                Jg_2 = Dot(Jg_temp, Jg_temp);
            }

            // Solve Newton system: J * c = r, then nrStep = -c
            // CGSolver follows the same convention as ExaNewtonSolver where the
            // Krylov solve produces c such that the Newton update would be x -= c.
            // For the dogleg we need nrStep = -J^{-1}*r, so we negate after the solve.
            {
                CALI_CXX_MARK_SCOPE("TR_newton_solve");
                c = 0.0;
                this->CGSolver(J, r, c);
                nrStep = c;
                nrStep.Neg();
            }

            nr_norm = Norm(nrStep);
        }

        // Save state for potential step rejection
        x_prev = x;

        // Compute the dogleg step
        double pred_resid = 0.0;
        double beta = 0.0;
        DoglegStep step = DoglegStep::NEWTON;
        Dogleg(delta, res_0, nr_norm, Jg_2, grad, nrStep, delx, pred_resid, step, beta);
        const bool use_nr = (step == DoglegStep::NEWTON);
        // Fraction of the full Newton step being taken. This is the trust-region
        // analog of the NR / NRLS relaxation factor. When not taking the Newton step,
        // the step lies on the trust-region boundary so ||delx|| = delta < nr_norm.
        const double step_scale = use_nr ? 1.0 : delta / nr_norm;

        // Apply the trial step: x = x_prev + delx
        x = x_prev;
        x += delx;

        // Evaluate residual at the trial point
        reject_prev = false;
        {
            CALI_CXX_MARK_SCOPE("TR_trial_eval");
            oper_mech->Mult(x, r);
            if (have_b) {
                r -= b;
            }
        }

        res = Norm(r);
        MFEM_ASSERT_0(mfem::IsFinite(res), "norm = " << res);

        // Check convergence
        if (res <= norm_max) {
            converged = true;
            PrintStepDetails(step,
                             beta,
                             pred_resid,
                             res,
                             std::numeric_limits<double>::quiet_NaN(),
                             delta,
                             delta,
                             "converged");
            PrintStepScaleInfo(it, step_scale);
            it++;
            PrintIterationInfo(it, res, norm0);
            break;
        }

        // Update delta from actual vs predicted reduction. May flag for rejection.
        // rho is only computed if the predicted change is non-zero.
        const double delta_old = delta;
        rho = std::numeric_limits<double>::quiet_NaN();
        bool delta_ok = delta_ctrl.UpdateDelta(
            delta, res, res_0, pred_resid, reject_prev, use_nr, nr_norm, rho);

        PrintStepDetails(step,
                         beta,
                         pred_resid,
                         res,
                         rho,
                         delta_old,
                         delta,
                         reject_prev ? "rejected" : "accepted");

        if (reject_prev) {
            // Revert x and the residual norm. On the next pass reject_prev == true so
            // we skip the Newton solve and recompute the dogleg with the updated
            // (smaller) delta. The Jacobian, grad, nrStep, and Jg_2 are still valid
            // from the last accepted state. Nothing is printed at the standard output
            // level since this is the same iteration being retried.
            x = x_prev;
            res = res_0;
            num_rejected++;
            num_consecutive_rejected++;
        } else {
            // Accepted step so this completes an iteration
            PrintStepScaleInfo(it, step_scale);
            it++;
            PrintIterationInfo(it, res, norm0);
            num_consecutive_rejected = 0;
        }

        if (!delta_ok) {
            if (print_level > 0) {
                if (delta <= delta_ctrl.deltaMin) {
                    mfem::out << "The trust region radius reached its minimum value of " << delta
                              << " so the nonlinear solve is stopping\n";
                } else {
                    mfem::out << "The trust region model predicted no change in the residual at "
                                 "its maximum radius so the nonlinear solve is stopping\n";
                }
            }
            converged = false;
            break;
        }

        if (num_consecutive_rejected >= max_consecutive_rejected) {
            if (print_level > 0) {
                mfem::out << "The trust region step was rejected " << num_consecutive_rejected
                          << " times in a row so the nonlinear solve is stopping\n";
            }
            converged = false;
            break;
        }

        res_0 = res;
    }

    final_iter = it;
    final_norm = res;
    PrintSummaryInfo(it, res, norm0, converged, num_rejected);
}