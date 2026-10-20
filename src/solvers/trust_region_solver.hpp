// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other ExaConstit Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "solvers/mechanics_solver.hpp"

#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

/**
 * @brief Trust-region radius control parameters for the dogleg solver.
 *
 * @details Ported from SNLS's TrDeltaControl. Controls how the trust-region
 * radius delta is updated based on the ratio rho = actual_reduction / predicted_reduction.
 *
 * The update logic:
 * - If rho is in the "good" band [xiLG, xiUG] and the step reduced the residual,
 *   increase delta (unless the full Newton step was taken)
 * - If rho is outside the "ok" band [xiLO, xiUO], decrease delta
 * - If the predicted change is zero and delta is not at max, force a small increase
 * - If the residual actually increased, reject the step
 *
 * @ingroup ExaConstit_solvers
 */
struct TrDeltaControl {
    /// @brief Lower bound of the "good" rho interval (increase delta when rho > xiLG)
    double xiLG = 0.75;
    /// @brief Upper bound of the "good" rho interval
    double xiUG = 1.4;
    /// @brief Factor by which to increase delta
    double xiIncDelta = 1.5;
    /// @brief Lower bound of the "ok" rho interval (decrease delta when rho < xiLO)
    double xiLO = 0.35;
    /// @brief Upper bound of the "ok" rho interval (decrease delta when rho > xiUO)
    double xiUO = 5.0;
    /// @brief Factor by which to decrease delta
    double xiDecDelta = 0.25;
    /// @brief Forced increase factor when predicted change is zero
    double xiForcedIncDelta = 1.2;
    /// @brief Initial trust-region radius
    double deltaInit = 1.0;
    /// @brief Minimum allowed trust-region radius (solver fails if hit)
    double deltaMin = 1e-12;
    /// @brief Maximum allowed trust-region radius
    double deltaMax = 1e4;
    /// @brief Whether to reject steps that increase the residual
    bool rejectResIncrease = true;

    /**
     * @brief Validate that the control parameters are self-consistent.
     *
     * @return true if all parameter relationships are valid, false otherwise
     *
     * Verifies the following invariants:
     * - deltaMin > 0 and deltaMax > deltaMin
     * - The "good" rho band [xiLG, xiUG] sits inside the "ok" band [xiLO, xiUO]
     * - The increase factor (xiIncDelta) is greater than 1
     * - The decrease factor (xiDecDelta) is in (0, 1)
     * - The forced-increase factor is greater than 1
     */
    bool Validate() const {
        return (deltaMin > 0.0) && (deltaMax > deltaMin) && (xiLG > xiLO) && (xiUG < xiUO) &&
               (xiIncDelta > 1.0) && (xiDecDelta > 0.0 && xiDecDelta < 1.0) &&
               (xiForcedIncDelta > 1.0);
    }

    /**
     * @brief Decrease the trust-region radius after a rejected/poor step.
     *
     * @param[in,out] delta Current radius, modified on output
     * @param[in] norm_full Norm of the full Newton step
     * @param[in] took_full Whether the full Newton step was used at the last iteration
     * @return true if delta is still above deltaMin, false if solver should fail
     *
     * @details If the full Newton step was taken, uses a geometric mean blend of
     * the current delta and the Newton step norm scaled by xiDecDelta. Otherwise
     * just multiplies delta by xiDecDelta. Returns false (and sets delta to deltaMin)
     * if the resulting delta drops below the minimum allowed value.
     */
    bool DecrDelta(double& delta, double norm_full, bool took_full) const {
        if (took_full) {
            double tempa = delta * xiDecDelta;
            double tempb = norm_full * xiDecDelta;
            delta = std::sqrt(tempa * tempb);
        } else {
            delta *= xiDecDelta;
        }

        if (delta < deltaMin) {
            delta = deltaMin;
            return false;
        }
        return true;
    }

    /**
     * @brief Increase the trust-region radius after a successful step.
     *
     * @param[in,out] delta Current radius, modified on output
     *
     * @details Multiplies delta by xiIncDelta and clamps at deltaMax.
     */
    void IncrDelta(double& delta) const {
        delta *= xiIncDelta;
        if (delta > deltaMax) {
            delta = deltaMax;
        }
    }

    /**
     * @brief Update trust-region radius based on actual vs predicted residual change.
     *
     * @param[in,out] delta Trust-region radius, modified on output
     * @param[in] res New residual norm (after the candidate step)
     * @param[in] res_0 Previous residual norm (before the candidate step)
     * @param[in] pred_resid Predicted residual norm from the dogleg model
     * @param[out] reject Whether the step should be rejected (residual increased)
     * @param[in] took_full Whether the full Newton step was taken
     * @param[in] norm_full Norm of the full Newton step
     * @param[out] rho Actual / predicted reduction ratio (output for diagnostics).
     *                 Left untouched if the predicted change is zero.
     * @return true if the delta update succeeded, false if the solver should fail
     *
     * @details Algorithm (ported from SNLS TrDeltaControl::updateDelta):
     *   1. Compute actual_change = res - res_0 and pred_change = pred_resid - res_0
     *   2. If pred_change is exactly zero, force delta larger (or fail if at max)
     *   3. Otherwise compute rho = actual_change / pred_change
     *   4. If rho is in the "good" band [xiLG, xiUG] and the residual decreased,
     *      increase delta (unless the full Newton step was already taken)
     *   5. If rho is outside the "ok" band [xiLO, xiUO], decrease delta
     *   6. If the residual increased and rejectResIncrease is set, mark for rejection
     *   7. If a rejected step was the full Newton step, keep decreasing delta until
     *      the full Newton step no longer fits in the trust region. Otherwise, the
     *      retried step would be identical to the rejected one. (Not in SNLS)
     */
    bool UpdateDelta(double& delta,
                     double res,
                     double res_0,
                     double pred_resid,
                     bool& reject,
                     bool took_full,
                     double norm_full,
                     double& rho) const {
        bool success = true;
        double actual_change = res - res_0;
        double pred_change = pred_resid - res_0;

        if (pred_change == 0.0) {
            if (delta >= deltaMax) {
                success = false;
            } else {
                delta = std::min(delta * xiForcedIncDelta, deltaMax);
            }
        } else {
            rho = actual_change / pred_change;

            if ((rho > xiLG) && (actual_change < 0.0) && (rho < xiUG)) {
                // Step is in the "good" band and residual actually decreased
                if (!took_full) {
                    IncrDelta(delta);
                }
            } else if ((rho < xiLO) || (rho > xiUO)) {
                // Step quality is outside the acceptable band; shrink delta
                success = DecrDelta(delta, norm_full, took_full);
            }
        }

        reject = false;
        // Do not make this >=, may have res and res_0 both zero and that is ok
        if ((actual_change > 0.0) && rejectResIncrease) {
            reject = true;
        }

        // A rejected full Newton step is retried from the same state with the same
        // Newton step. If delta is still >= norm_full, the full Newton step still fits
        // inside the trust region and the retry would be identical to the step that was
        // just rejected, wasting a residual evaluation only to be rejected again. The
        // took_full branch of DecrDelta does not guarantee delta < norm_full after a
        // single call, since it uses the geometric mean of delta and norm_full.
        // So, keep decreasing delta until the retried step actually differs. This
        // results in the same delta as evaluating those identical retries would have,
        // just without the wasted residual evaluations. DecrDelta strictly decreases
        // delta here and its repeated application converges to xiDecDelta^2 * norm_full,
        // so this loop always terminates.
        if (reject && took_full) {
            while (success && (delta >= norm_full)) {
                success = DecrDelta(delta, norm_full, took_full);
            }
        }

        return success;
    }
};

/**
 * @brief Which part of the dogleg path a trust-region step was taken along.
 */
enum class DoglegStep {
    NEWTON,           ///< Full Newton step fit inside the trust region
    STEEPEST_DESCENT, ///< Step along the steepest descent direction to the trust-region boundary
    DOGLEG            ///< Step between the Cauchy point and the Newton point (second leg)
};

/**
 * @brief Trust-region dogleg solver for nonlinear solid mechanics problems.
 *
 * @details This class implements a Powell-dogleg trust-region method for solving
 * nonlinear systems F(x) = b. It extends ExaNewtonSolver and reuses the same
 * Krylov solver infrastructure (prec_mech) for computing the Newton direction.
 *
 * The trust-region method augments standard Newton with a globalization strategy
 * that interpolates between the steepest descent direction and the full Newton
 * step, constrained to a trust-region radius delta. Step quality is monitored
 * via the ratio rho = actual_reduction / predicted_reduction, and delta is
 * adjusted up or down accordingly.
 *
 * This is a direct port of SNLS's SNLSTrDlDenseG solver, lifted from the
 * material-point dense system to the global FE system.
 *
 * Algorithm at each iteration:
 *   1. Compute steepest descent direction g = J^T * r (gradient of merit f = 0.5 ||F||^2)
 *   2. Compute ||J*g||^2 for the optimal Cauchy step length
 *   3. Solve J * c = r for the full Newton direction (using prec_mech Krylov solver)
 *   4. Compute the dogleg step within the trust region
 *   5. Evaluate the residual at the trial point
 *   6. Accept or reject based on the rho ratio; update delta accordingly
 *
 * Requirements:
 * - The gradient operator must support MultTranspose (for J^T*r computation).
 *   This means the assembly mode must be EA, FA, or PA with the native PA
 *   transpose kernels enabled.
 *
 * @ingroup ExaConstit_solvers
 */
class ExaTrustRegionSolver : public ExaNewtonSolver {
public:
    /**
     * @brief Default constructor
     *
     * @details Creates an ExaTrustRegionSolver instance for single-processor
     * execution. The operator and linear solver must be set separately using
     * SetOperator() and SetSolver(), and the trust-region control parameters
     * may be customized via SetTrustRegionControl().
     */
    ExaTrustRegionSolver() {}

#ifdef MFEM_USE_MPI
    /**
     * @brief MPI constructor
     *
     * @param _comm MPI communicator for parallel execution
     *
     * @details Creates an ExaTrustRegionSolver instance for parallel execution
     * using the specified MPI communicator. All trust-region scalar quantities
     * (norms, dot products) use MPI-aware reductions through MFEM's Dot/Norm.
     */
    ExaTrustRegionSolver(MPI_Comm _comm) : ExaNewtonSolver(_comm) {}
#endif

    /** @brief Use parent class SetOperator methods */
    using ExaNewtonSolver::SetOperator;

    /** @brief Use parent class SetSolver methods */
    using ExaNewtonSolver::SetSolver;

    /** @brief Use parent class CGSolver method (Krylov solve wrapper) */
    using ExaNewtonSolver::CGSolver;

    /**
     * @brief Set trust-region control parameters.
     *
     * @param ctrl TrDeltaControl struct with all tuning parameters
     *
     * @details Replaces the internal control parameters with a user-supplied
     * configuration. Typically called after construction (and before Mult())
     * to wire up parameters parsed from the TOML configuration file.
     */
    void SetTrustRegionControl(const TrDeltaControl& ctrl) {
        delta_ctrl = ctrl;
    }

    /**
     * @brief Get a mutable reference to the trust-region control parameters.
     * @return Reference to the internal TrDeltaControl
     */
    TrDeltaControl& GetTrustRegionControl() {
        return delta_ctrl;
    }

    /**
     * @brief Get a const reference to the trust-region control parameters.
     * @return Const reference to the internal TrDeltaControl
     */
    const TrDeltaControl& GetTrustRegionControl() const {
        return delta_ctrl;
    }

    /**
     * @brief Solve the nonlinear system F(x) = b using trust-region dogleg method.
     *
     * @param b Right-hand side vector (if b.Size() != Height(), assumes b = 0)
     * @param x Solution vector (input: initial guess, output: converged solution)
     *
     * @details Implements the trust-region dogleg algorithm. See class-level
     * documentation for the algorithm description. The Newton direction is
     * computed by the Krylov solver wired in via SetSolver(); J^T*r is
     * computed by calling MultTranspose() on the gradient operator.
     *
     * @pre SetOperator() and SetSolver() must be called before Mult()
     * @pre The gradient operator must support MultTranspose (EA/FA mode, or
     *      PA mode with native transpose kernels)
     *
     * @post final_iter contains the number of accepted iterations performed
     *       (rejected trial steps are retries of the same iteration and do not count)
     * @post final_norm contains the final residual norm
     * @post converged flag indicates whether the solver converged
     */
    virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;

private:
    /**
     * @brief Compute the dogleg step given the current trust-region radius.
     *
     * @param[in] delta Trust-region radius
     * @param[in] res_0 Current residual norm
     * @param[in] nr_norm Norm of the full Newton step
     * @param[in] Jg_2 ||J*g||^2 where g is the steepest descent direction
     * @param[in] grad Steepest descent direction g = J^T * r
     * @param[in] nrStep Full Newton step
     * @param[out] delx The computed dogleg step
     * @param[out] pred_resid Predicted residual norm after the step
     * @param[out] step Which part of the dogleg path the step was taken along
     * @param[out] beta Interpolation factor along the second leg (only meaningful
     *                  when step == DoglegStep::DOGLEG)
     *
     * @details Ported from SNLS's dogleg() kernel. The dogleg path interpolates
     * between the steepest descent direction (Cauchy point) and the full Newton
     * step. Three cases are handled:
     *   - Newton step inside delta: take full Newton step
     *   - Cauchy point outside delta: step along steepest descent to boundary
     *   - Cauchy inside, Newton outside: solve quadratic for the dogleg leg
     *     intersection with the trust-region boundary
     */
    void Dogleg(double delta,
                double res_0,
                double nr_norm,
                double Jg_2,
                const mfem::Vector& grad,
                const mfem::Vector& nrStep,
                mfem::Vector& delx,
                double& pred_resid,
                DoglegStep& step,
                double& beta) const;

    /**
     * @brief Print the detailed trust-region information for an iteration
     *
     * @param step Which part of the dogleg path the step was taken along
     * @param beta Interpolation factor along the second leg of the dogleg path
     * @param pred_resid Predicted residual norm from the dogleg model
     * @param trial_resid Actual residual norm at the trial point
     * @param rho Actual / predicted reduction ratio (NaN if it was not computed)
     * @param delta_old Trust-region radius used for this step
     * @param delta_new Trust-region radius for the next step
     * @param outcome Short description of what happened to the step
     *
     * @details Only prints when print_level > 1. Everything about a trial step,
     * including rejected ones, is provided on a single line. These lines are printed
     * after the residual line of the iteration the trial steps are taken from.
     */
    void PrintStepDetails(DoglegStep step,
                          double beta,
                          double pred_resid,
                          double trial_resid,
                          double rho,
                          double delta_old,
                          double delta_new,
                          const char* outcome) const;

    /// @brief Trust-region control parameters (mutable to allow tuning)
    mutable TrDeltaControl delta_ctrl;
};