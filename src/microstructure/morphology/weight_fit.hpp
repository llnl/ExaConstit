#pragma once

#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/microstructure_types.hpp"

#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file weight_fit.hpp
 * @brief Finds seed weights w_i so that every grain of a Laguerre (power) tessellation has a
 * prescribed volume: size control for grain size distributions.
 *
 * **The physical picture.** In a Laguerre tessellation a point x belongs to the grain with the
 * smallest f_i(x) = |x - s_i|^2 - w_i. Raising the weight w_i of one grain pushes all of its
 * faces outward (a face between grains i and j moves by dw / (2 |s_i - s_j|) toward j), so the
 * grain grows at the expense of its neighbors, while the seed positions stay fixed. A weight acts
 * like a pressure inside the grain: we adjust each grain's weight like a pressure until its volume
 * matches the target. Only weight *differences* matter (adding the same constant to every weight
 * changes nothing).
 *
 * **The dual function: one hill to climb.** Adjusting all N pressures at once so that all N
 * volumes match is a coupled nonlinear problem (each grain's volume depends on its neighbors'
 * weights). It has an exact "potential" (this is the dual of semi-discrete optimal transport, the
 * problem of splitting a volume into cells of prescribed sizes at least cost):
 *
 *     Phi(w) = sum_i w_i V_i^target + integral over the material of min_i f_i(x) dx.
 *
 * Its slope with respect to one weight is the volume mismatch of that grain:
 *
 *     dPhi / dw_i = V_i^target - V_i(w),
 *
 * because raising w_i lowers min_i f_i by exactly the same amount everywhere inside grain i and
 * nowhere else (so the integral drops by V_i dw_i) while the first sum rises by V_i^target dw_i.
 * Like a potential energy whose "force" on each weight is the volume mismatch, Phi is stationary
 * exactly when every grain has its target volume. Phi is *concave* (a minimum of functions that
 * are linear in w, integrated, plus a linear term): it is a single hill without false peaks, so
 * any method that keeps climbing reaches the top. Climbing Phi is the same as solving
 * "all volumes match", but with a built-in measure of progress that may never decrease.
 * The integral uses the samples of cell_measures.hpp (quadrature samples give a much smoother
 * volume response than element centroids, see there). The same dual holds for every metric of
 * the form f_i = g_i - w_i (Quadric, Additive), so those metrics are accepted too.
 *
 * **Climbing: L-BFGS.** A Newton method would need the "stiffness matrix" dV_i / dw_j (how much
 * grain i grows when grain j's pressure rises; a sparse, graph-Laplacian-like matrix set by the
 * shared face areas and seed distances) and a linear solve with it per iteration. L-BFGS
 * (limited-memory Broyden-Fletcher-Goldfarb-Shanno) avoids both: like the quasi-Newton (secant)
 * updates of nonlinear FE solvers, it learns an approximate *inverse* stiffness from the last m
 * steps. Every step gives one pair (s = change of the weights, y = change of the volume
 * mismatch), i.e. one measured "secant stiffness" along that direction; the last m pairs define
 * the approximation, which is applied with two short loops over the pairs (no matrix is ever
 * stored). The first step, with no pairs yet, uses a scalar inverse stiffness estimated from the
 * mean cell size and the metric (InitialInverseStiffness).
 *
 * **Never going downhill: Armijo backtracking.** The proposed step is tried in full; if Phi did
 * not increase by at least a small fraction c (1e-4) of the increase its slope predicts, the step
 * is halved and tried again (the same idea as a line search in a Newton-Raphson solver for
 * plasticity). A step is also required to raise the computed Phi strictly (for tiny steps the
 * Armijo bound rounds to the old value). Every accepted step therefore increases Phi: the
 * reported Phi history is strictly increasing.
 * Because the volumes come from a finite set of samples, V_i(w) changes in small jumps and Phi is
 * piecewise linear; when even tiny steps no longer increase Phi, the fit has reached the
 * resolution of the samples and stops.
 *
 * **Empty grains recover.** A grain that currently owns no samples has V_i = 0, so its slope
 * V_i^target - 0 > 0 keeps raising its weight until the grain reappears.
 *
 * **Determinism.** Volumes and Phi are reproducible fixed-point sums (cell_measures.hpp), so every
 * rank sees bitwise the same numbers, performs the same L-BFGS arithmetic and returns the same
 * weights, independent of the rank count.
 */

namespace exaconstit::microstructure {

/** @brief Controls of the weight fit. Host data. */
struct WeightFitParameters {
    /**
     * @brief Metric with weights: LAGUERRE (the usual choice), QUADRIC or ADDITIVE. VORONOI has
     * no weights and is rejected.
     */
    MetricType metric = MetricType::LAGUERRE;
    /** @brief Quadric global tensor (QUADRIC only; per-seed tensors come from seeds.shape). */
    MetricSetup setup{};
    /** @brief Parameter bins of the seed search (performance only, >= 1). */
    int num_bins = 8;
    /** @brief Stop when max_i |V_i - V_i^target| / V_i^target <= tolerance (> 0). */
    double tolerance = 0.02;
    /** @brief Largest number of accepted L-BFGS steps (>= 0). */
    int max_iterations = 200;
    /** @brief Number m of (s, y) pairs L-BFGS remembers (>= 1). */
    int history = 8;
    /** @brief Armijo sufficient-increase fraction c, in (0, 1). */
    double armijo_c = 1e-4;
    /** @brief Largest number of step halvings per line search (>= 1). */
    int max_backtracks = 40;
    /**
     * @brief Scalar inverse stiffness gamma0 = (change of weight) / (change of volume), used for
     * the first step and after every L-BFGS restart (with pairs, L-BFGS uses the measured scale
     * gamma = s.y / y.y instead, the Barzilai-Borwein step length). Its units depend on the metric
     * because the weight's units do: length^-1 for Laguerre (w in length^2), length^-2 for
     * Additive (a in length), (units of A) / length for Quadric (w in units of A x length^2).
     * <= 0 selects the estimate InitialInverseStiffness(metric, ...) (see there).
     */
    double initial_inverse_stiffness = 0.0;
};

/** @brief Convergence history of one fit. Host data, identical on every rank. */
struct WeightFitReport {
    /** @brief True if the tolerance was reached. */
    bool converged = false;
    /** @brief Accepted L-BFGS steps. */
    int iterations = 0;
    /** @brief Evaluations of Phi (labelings of all samples), including rejected trial steps. */
    int evaluations = 0;
    /** @brief Phi at the start and after every accepted step (size iterations + 1); strictly
     * increasing. */
    std::vector<double> phi;
    /** @brief max_i |V_i - V_i^target| / V_i^target at the same iterates. */
    std::vector<double> max_relative_error;
    /** @brief Number of empty cells (V_i = 0) at the same iterates. */
    std::vector<int> num_empty;
    /**
     * @brief Sample resolution: (largest sample weight) / (smallest target volume). Volume errors
     * well below this are not resolvable with these samples.
     */
    double resolution_floor = 0.0;
    /** @brief "converged", "max_iterations" or "line_search_failed" (sample resolution reached). */
    std::string stop_reason;
    /**
     * @brief Number of L-BFGS restarts: times the stored pairs were dropped and the scaled
     * steepest direction -gamma0 G was used instead, because the L-BFGS direction was not an
     * ascent direction for Phi or its line search failed.
     */
    int num_restarts = 0;
    /** @brief The gamma0 actually used (see WeightFitParameters::initial_inverse_stiffness). */
    double initial_inverse_stiffness = 0.0;
};

/** @brief Output of FitWeights. Host data, identical on every rank. */
struct WeightFitResult {
    /** @brief Fitted weight per seed (size N; length^2 for Laguerre). */
    std::vector<double> weight;
    /** @brief Target volumes actually used: the input scaled to sum to the sampled volume. */
    std::vector<double> target_volume;
    /** @brief Cell measures of the final weights (volumes, centroids, sample counts). */
    CellMeasures measures;
    /** @brief Convergence history. */
    WeightFitReport report;
};

/**
 * @brief Estimate of the scalar inverse stiffness gamma0 = dw / dV of one cell, by metric.
 *
 * For a roughly equiaxed cell of size L = (mean cell volume)^(1/3) with about six faces of area
 * L^2 at seed spacing ~ L, raising the cell's weight by dw moves each face by dw / |grad(f_i -
 * f_j)|, so dV/dw ~ 6 L^2 / |grad(f_i - f_j)|:
 * - Laguerre: |grad(f_i - f_j)| = 2 |s_i - s_j| ~ 2 L, so dV/dw ~ 3 L and gamma0 = 0.3 / L;
 * - Additive (f = |x - s| - a): |grad(f_i - f_j)| <= 2 (difference of two unit vectors), about
 *   2, so dV/da ~ 3 L^2 and gamma0 = 0.3 / L^2;
 * - Quadric (f = (x - s)^T A (x - s) - w): |grad(f_i - f_j)| ~ 2 mu L with mu a mean eigenvalue
 *   of A, so dV/dw ~ 3 L / mu and gamma0 = 0.3 mu / L. mu = trace(A) / 3 of the global tensor,
 *   or the average of trace / 3 over the per-seed tensors.
 * The factor 0.3 (a bit below 1/3) makes the first step slightly short; Armijo backtracking
 * corrects any misestimate by halving, and L-BFGS replaces the estimate after the first step.
 * Host only.
 *
 * @param metric      LAGUERRE, ADDITIVE or QUADRIC.
 * @param mean_volume mean cell volume (length^3, > 0).
 * @param setup       Quadric global tensor (used when `shape` is empty).
 * @param shape       per-seed Quadric tensors (SeedSet::shape, 6 per seed) or empty.
 * @return gamma0 (> 0).
 */
double InitialInverseStiffness(MetricType metric, double mean_volume, const MetricSetup& setup,
                               const std::vector<double>& shape);

/**
 * @brief Checks the parameters.
 * @throws std::invalid_argument naming the first invalid field.
 */
void ValidateWeightFitParameters(const WeightFitParameters& params);

/**
 * @brief Fits the seed weights so every cell's sampled volume matches its target (see the file
 * comment for the method).
 *
 * **Collective** over `comm` (every evaluation reduces volumes and Phi over the ranks). Host
 * function; the labeling and accumulation run as kernels on the mfem::Device backend.
 *
 * @param comm           communicator over which the samples are distributed.
 * @param samples        this rank's weighted samples of the material (quadrature samples
 *                       recommended); must stay unchanged during the call.
 * @param seeds          replicated seeds (positions; shape for per-seed quadrics). Their weight
 *                       field is used only as the starting point when `initial_weight` is empty.
 * @param target_volume  one positive, finite target per seed (size N); rescaled internally so
 *                       the targets sum to the sampled material volume.
 * @param initial_weight starting weights (size N), e.g. SizeTargets::initial_weight; if empty,
 *                       seeds.weight (or zeros when that is empty too).
 * @param params         solver controls.
 * @return fitted weights, final measures and the convergence history.
 * @throws std::invalid_argument (on all ranks) for inconsistent sizes, invalid targets, weights
 *         or parameters, and the sample/label errors of CellMeasureAccumulator.
 */
WeightFitResult FitWeights(MPI_Comm comm, const WeightedSamples& samples, const SeedSet& seeds,
                           const std::vector<double>& target_volume,
                           const std::vector<double>& initial_weight,
                           const WeightFitParameters& params);

} // namespace exaconstit::microstructure
