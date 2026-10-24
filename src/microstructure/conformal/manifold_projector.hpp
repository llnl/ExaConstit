#pragma once

#include "mfem.hpp"

#include <math.h> // sqrt, fabs in device code (C math library)

/**
 * @file manifold_projector.hpp
 * @brief Closest point of a node position on the intersection of grain boundaries and flat
 * boundary planes (Stage 2b of conformal fitting).
 *
 * **The problem.** A node x that lies between grains a and b must move to the grain boundary
 * f_a(y) = f_b(y) (metric_functors.hpp: f_i is the value whose minimum decides the grain). A node
 * of a triple line must satisfy two such equations, a quad point three; a node on a flat face of
 * the outer boundary must also stay on that plane, n . y = n . x. With the equations written as
 * phi_k(y) = 0, k < m <= 3, the target is the closest point
 *
 *     y = argmin |y - x|^2  subject to  phi_k(y) = 0 for all k.
 *
 * **Algorithm (constrained Gauss-Newton).** Start at y = x and repeat: linearize the equations at
 * y, phi_k(z) ~ phi_k(y) + J_k (z - y) with J_k = grad phi_k(y), and take the closest point of x
 * on the linearized set,
 *
 *     y_new = x - J^T (J J^T)^{-1} (J (x - y) + phi(y)),
 *
 * until the step |y_new - y| is at most `tolerance`. J J^T is an m x m symmetric matrix (m <= 3),
 * solved by a Cholesky factorization. At a fixed point phi(y) = 0 and x - y is a combination of
 * the gradients, which is the condition for the closest point. When every equation is linear
 * (Voronoi, Laguerre, Quadric with one global tensor, and boundary planes) the first step lands
 * exactly on the closest point, y = x - N^T (N N^T)^{-1} (N x - c), and the second step only
 * confirms it. Curved boundaries (Additive / Johnson-Mehl hyperboloids, per-seed Quadric tensors)
 * converge quadratically from a start within a fraction of their radius of curvature.
 *
 * The equations are given as rows (ManifoldRow): a grain boundary between seeds a and b, which
 * uses the metric functor's Eval and Grad (so it is the same function that labeled the elements),
 * or a plane n . y = c.
 *
 * **Failure.** The projection fails when J J^T is singular (two equations with parallel
 * gradients: parallel planes, or a grain boundary parallel to a box face at the node) or when the
 * steps do not converge in `max_iterations`; callers then treat the node as unfit.
 *
 * Header-only, MFEM_HOST_DEVICE, no allocation: usable on the host (Stage 2 runs there) and in
 * kernels.
 */

namespace exaconstit::microstructure {

/** @brief Most equations a node can satisfy in 3D. */
inline constexpr int MANIFOLD_MAX_ROWS = 3;

/** @brief One equation phi(y) = 0 of the projection. */
struct ManifoldRow {
    /** @brief PLANE: n . y - c; GRAIN_BOUNDARY: f_a(y) - f_b(y). */
    enum Kind : int { PLANE = 0, GRAIN_BOUNDARY = 1 };
    int kind = PLANE;
    double normal[3] = {0.0, 0.0, 0.0}; ///< plane normal n (any length)
    double offset = 0.0;                ///< plane offset c
    int seed_a = -1;                    ///< grain boundary: 0-based seed indices
    int seed_b = -1;
};

/** @brief Outcome of ProjectOntoManifold. */
enum class ProjectionResult : int {
    CONVERGED = 0, ///< the last step was at most the tolerance
    SINGULAR = 1,  ///< the gradients of the equations are (nearly) linearly dependent
    NOT_CONVERGED = 2
};

/**
 * @brief Value and gradient of row r at y. Host and device.
 */
template <typename Metric>
MFEM_HOST_DEVICE inline double EvalManifoldRow(const Metric& metric, const ManifoldRow& r,
                                               const double y[3], double grad[3]) {
    if (r.kind == ManifoldRow::PLANE) {
        grad[0] = r.normal[0];
        grad[1] = r.normal[1];
        grad[2] = r.normal[2];
        return r.normal[0] * y[0] + r.normal[1] * y[1] + r.normal[2] * y[2] - r.offset;
    }
    double ga[3];
    double gb[3];
    metric.Grad(r.seed_a, y[0], y[1], y[2], ga);
    metric.Grad(r.seed_b, y[0], y[1], y[2], gb);
    grad[0] = ga[0] - gb[0];
    grad[1] = ga[1] - gb[1];
    grad[2] = ga[2] - gb[2];
    return metric.Eval(r.seed_a, y[0], y[1], y[2]) - metric.Eval(r.seed_b, y[0], y[1], y[2]);
}

/**
 * @brief Closest point y of x on {phi_k = 0, k < num_rows} (see the file comment).
 *
 * @param metric         metric functor with host-valid (or device-valid in a kernel) pointers.
 * @param rows           the equations, 1 <= num_rows <= MANIFOLD_MAX_ROWS.
 * @param x              the node position.
 * @param tolerance      stop when a step is at most this long (length units, > 0).
 * @param max_iterations Gauss-Newton steps allowed (>= 1).
 * @param y              out: the last iterate (the target when CONVERGED).
 * @param iterations     out (optional): steps taken.
 */
template <typename Metric>
MFEM_HOST_DEVICE inline ProjectionResult
ProjectOntoManifold(const Metric& metric, const ManifoldRow* rows, int num_rows,
                    const double x[3], double tolerance, int max_iterations, double y[3],
                    int* iterations = nullptr) {
    y[0] = x[0];
    y[1] = x[1];
    y[2] = x[2];
    for (int it = 1; it <= max_iterations; ++it) {
        double J[MANIFOLD_MAX_ROWS][3];
        double rhs[MANIFOLD_MAX_ROWS];
        for (int k = 0; k < num_rows; ++k) {
            const double phi = EvalManifoldRow(metric, rows[k], y, J[k]);
            // r_k = J_k (x - y) + phi_k(y)
            rhs[k] = J[k][0] * (x[0] - y[0]) + J[k][1] * (x[1] - y[1]) + J[k][2] * (x[2] - y[2]) +
                     phi;
        }
        // Cholesky of G = J J^T (lower factor L in place)
        double L[MANIFOLD_MAX_ROWS][MANIFOLD_MAX_ROWS] = {};
        double scale = 0.0;
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j <= i; ++j) {
                L[i][j] = J[i][0] * J[j][0] + J[i][1] * J[j][1] + J[i][2] * J[j][2];
            }
            scale = (L[i][i] > scale) ? L[i][i] : scale;
        }
        for (int j = 0; j < num_rows; ++j) {
            double d = L[j][j];
            for (int k = 0; k < j; ++k) {
                d -= L[j][k] * L[j][k];
            }
            // Relative pivot test: rows whose gradients are dependent to ~1e-6 radians
            if (!(d > 1e-12 * scale)) {
                if (iterations != nullptr) {
                    *iterations = it;
                }
                return ProjectionResult::SINGULAR;
            }
            L[j][j] = sqrt(d);
            for (int i = j + 1; i < num_rows; ++i) {
                double s = L[i][j];
                for (int k = 0; k < j; ++k) {
                    s -= L[i][k] * L[j][k];
                }
                L[i][j] = s / L[j][j];
            }
        }
        // Solve L L^T lambda = rhs
        double lambda[MANIFOLD_MAX_ROWS];
        for (int i = 0; i < num_rows; ++i) {
            double s = rhs[i];
            for (int k = 0; k < i; ++k) {
                s -= L[i][k] * lambda[k];
            }
            lambda[i] = s / L[i][i];
        }
        for (int i = num_rows - 1; i >= 0; --i) {
            double s = lambda[i];
            for (int k = i + 1; k < num_rows; ++k) {
                s -= L[k][i] * lambda[k];
            }
            lambda[i] = s / L[i][i];
        }
        // y_new = x - J^T lambda
        double step2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            double yn = x[d];
            for (int k = 0; k < num_rows; ++k) {
                yn -= J[k][d] * lambda[k];
            }
            step2 += (yn - y[d]) * (yn - y[d]);
            y[d] = yn;
        }
        if (iterations != nullptr) {
            *iterations = it;
        }
        if (step2 <= tolerance * tolerance) {
            return ProjectionResult::CONVERGED;
        }
    }
    return ProjectionResult::NOT_CONVERGED;
}

} // namespace exaconstit::microstructure
