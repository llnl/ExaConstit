/*
 * Implementation of the Laguerre weight fit (weight_fit.hpp): maximize the concave dual
 * Phi(w) = sum_i w_i V_i^target + integral of min_i f_i with L-BFGS and Armijo backtracking.
 *
 * Conventions in this file:
 * - The solver is written as the usual *minimization* of F(w) = -Phi(w). Its gradient is
 *   G_i = dF / dw_i = V_i(w) - V_i^target (the volume excess of grain i). "Descent" for F is
 *   "ascent" for Phi.
 * - Every quantity the solver uses (volumes, Phi) is a reproducible fixed-point sum that is
 *   bitwise identical on every rank (labeling/cell_measures.hpp), and the solver arithmetic is a
 *   plain host loop in seed order. Every rank therefore takes exactly the same steps and the
 *   result does not depend on the rank count.
 * - Collective: a function every rank must call, because the ranks exchange data inside it.
 */
#include "microstructure/morphology/weight_fit.hpp"

#include "microstructure/morphology/sample_labels.hpp"

#include "mfem.hpp"

#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief Dot product of two equal-length vectors, summed in index order (same on all ranks). */
double Dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/** @brief One curvature pair of L-BFGS: step s, gradient change y, and rho = 1 / (y . s). */
struct CurvaturePair {
    std::vector<double> s;
    std::vector<double> y;
    double rho = 0.0;
};

/**
 * @brief Everything known about one trial weight vector: the cell measures, F = -Phi, the
 * gradient G = V - V^target and the error indicators.
 */
struct Evaluation {
    CellMeasures measures;
    double objective = 0.0;          ///< F(w) = -Phi(w)
    std::vector<double> gradient;    ///< G_i = V_i - V_i^target
    double max_relative_error = 0.0; ///< max_i |G_i| / V_i^target
    int num_empty = 0;               ///< cells with V_i == 0
};

/**
 * @brief The objective of the fit, bound to one set of samples, seeds and targets.
 *
 * Holds a working copy of the seeds whose weights are replaced for each evaluation, and the
 * CellMeasureAccumulator whose fixed-point scales are set up once (collective constructor).
 */
class DualObjective {
public:
    /**
     * @brief Copies the seed positions (and shape tensors) into the working seed set and sets up
     * the fixed-point scales of the accumulator. Collective (the accumulator reduces global
     * bounds). `samples` and `params` are referenced, not copied: they must outlive the object.
     */
    DualObjective(MPI_Comm comm, const WeightedSamples& samples, const SeedSet& seeds,
                  const WeightFitParameters& params)
        : m_samples(samples), m_params(params),
          m_accumulator(comm, samples, static_cast<int>(seeds.Size())) {
        // Only positions and shape tensors are needed by the seed search
        m_work.x = seeds.x;
        m_work.y = seeds.y;
        m_work.z = seeds.z;
        m_work.shape = seeds.shape;
    }

    /**
     * @brief Labels all samples with weights w and measures the cells. Collective.
     *
     * Returns the measures and the integral of min_i f_i (= sum_q W_q f_label(q)(x_q), since the
     * label is the minimizing seed and f its value).
     */
    CellMeasures Measure(const std::vector<double>& w, double& integral_min_f) {
        m_work.weight = w;
        mfem::Array<int> labels;
        mfem::Vector min_f;
        LabelSamples(m_samples, m_work, m_params.metric, m_params.setup, m_params.num_bins,
                     labels, &min_f);
        CellMeasures measures = m_accumulator.Measure(labels);
        integral_min_f = m_accumulator.Integrate(min_f);
        ++m_evaluations;
        return measures;
    }

    /**
     * @brief Full evaluation at w for the (normalized) targets: F = -(sum_i w_i V_i^target +
     * integral min f), G = V - V^target, and the error indicators. Collective.
     */
    Evaluation Evaluate(const std::vector<double>& w, const std::vector<double>& target) {
        double integral = 0.0;
        CellMeasures measures = Measure(w, integral);
        return Assemble(w, target, std::move(measures), integral);
    }

    /**
     * @brief F, G and the error indicators from already measured cells (host only, no
     * communication): F = -(sum_i w_i V_i^target + integral), G_i = V_i - V_i^target.
     */
    static Evaluation Assemble(const std::vector<double>& w, const std::vector<double>& target,
                               CellMeasures measures, double integral) {
        Evaluation e;
        e.measures = std::move(measures);
        e.objective = -(Dot(w, target) + integral);
        e.gradient.resize(w.size());
        for (std::size_t i = 0; i < w.size(); ++i) {
            e.gradient[i] = e.measures.volume[i] - target[i];
            e.max_relative_error = std::fmax(e.max_relative_error,
                                             std::fabs(e.gradient[i]) / target[i]);
            if (e.measures.volume[i] == 0.0) {
                ++e.num_empty;
            }
        }
        return e;
    }

    /** @brief Number of labelings of all samples so far. */
    int Evaluations() const {
        return m_evaluations;
    }
    /** @brief The accumulator (for its global sample statistics). */
    const CellMeasureAccumulator& Accumulator() const {
        return m_accumulator;
    }

private:
    const WeightedSamples& m_samples;
    const WeightFitParameters& m_params;
    CellMeasureAccumulator m_accumulator;
    SeedSet m_work;
    int m_evaluations = 0;
};

/**
 * @brief L-BFGS two-loop recursion: returns d = -H G, where H approximates the inverse of the
 * "stiffness" dG/dw from the stored pairs (newest last) and the scalar `gamma` for the oldest
 * level (Nocedal and Wright, Numerical Optimization, Algorithm 7.4).
 *
 * In words: the first loop (newest to oldest) removes from G the components that each stored
 * step explains; the middle multiplies by the scalar inverse stiffness gamma; the second loop
 * (oldest to newest) adds back each stored step scaled by how much it should move. No N x N
 * matrix is formed; the cost is about 4 m N multiply-adds.
 */
std::vector<double> TwoLoopDirection(const std::deque<CurvaturePair>& pairs,
                                     const std::vector<double>& gradient, double gamma) {
    std::vector<double> q = gradient;
    std::vector<double> alpha(pairs.size(), 0.0);
    for (std::size_t k = pairs.size(); k-- > 0;) {
        alpha[k] = pairs[k].rho * Dot(pairs[k].s, q);
        for (std::size_t i = 0; i < q.size(); ++i) {
            q[i] -= alpha[k] * pairs[k].y[i];
        }
    }
    for (double& v : q) {
        v *= gamma;
    }
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        const double beta = pairs[k].rho * Dot(pairs[k].y, q);
        for (std::size_t i = 0; i < q.size(); ++i) {
            q[i] += pairs[k].s[i] * (alpha[k] - beta);
        }
    }
    for (double& v : q) {
        v = -v;
    }
    return q;
}

/** @brief Appends the iterate's values to the report history. */
void Record(WeightFitReport& report, const Evaluation& e) {
    report.phi.push_back(-e.objective);
    report.max_relative_error.push_back(e.max_relative_error);
    report.num_empty.push_back(e.num_empty);
}

} // namespace

/*
 * InitialInverseStiffness: the per-metric estimates derived in the header. The Quadric scale mu
 * is the mean eigenvalue trace(A) / 3 (the first three stored entries are the diagonal), which is
 * positive for any nonzero positive semidefinite A, including the semidefinite columnar and
 * lamellar tensors; if it is not positive (invalid tensors are rejected later by the seed
 * search), 1 is used so the estimate stays finite.
 */
double InitialInverseStiffness(MetricType metric, double mean_volume, const MetricSetup& setup,
                               const std::vector<double>& shape) {
    const double size = std::cbrt(mean_volume);
    if (metric == MetricType::ADDITIVE) {
        return 0.3 / (size * size);
    }
    if (metric == MetricType::QUADRIC) {
        double mu = 0.0;
        if (shape.empty()) {
            mu = (setup.global_tensor[0] + setup.global_tensor[1] + setup.global_tensor[2]) / 3.0;
        } else {
            const std::size_t num = shape.size() / 6;
            for (std::size_t i = 0; i < num; ++i) {
                mu += (shape[6 * i] + shape[6 * i + 1] + shape[6 * i + 2]) / 3.0;
            }
            mu /= static_cast<double>(num);
        }
        if (!(mu > 0.0) || !std::isfinite(mu)) {
            mu = 1.0;
        }
        return 0.3 * mu / size;
    }
    return 0.3 / size; // Laguerre (and the Voronoi limit)
}

/*
 * ValidateWeightFitParameters: each check names the field. Written as !(x > 0) where NaN must
 * fail too.
 */
void ValidateWeightFitParameters(const WeightFitParameters& params) {
    if (params.metric == MetricType::VORONOI) {
        throw std::invalid_argument("Weight fit: the Voronoi metric has no weights; use LAGUERRE");
    }
    if (params.num_bins < 1) {
        throw std::invalid_argument("Weight fit: num_bins must be >= 1");
    }
    if (!(params.tolerance > 0.0) || !std::isfinite(params.tolerance)) {
        throw std::invalid_argument("Weight fit: tolerance must be positive and finite");
    }
    if (params.max_iterations < 0) {
        throw std::invalid_argument("Weight fit: max_iterations must be >= 0");
    }
    if (params.history < 1) {
        throw std::invalid_argument("Weight fit: history must be >= 1");
    }
    if (!(params.armijo_c > 0.0) || !(params.armijo_c < 1.0)) {
        throw std::invalid_argument("Weight fit: armijo_c must be in (0, 1)");
    }
    if (params.max_backtracks < 1) {
        throw std::invalid_argument("Weight fit: max_backtracks must be >= 1");
    }
    if (!std::isfinite(params.initial_inverse_stiffness)) {
        throw std::invalid_argument("Weight fit: initial_inverse_stiffness must be finite");
    }
}

/*
 * FitWeights
 *
 * Algorithm:
 *  1. Check inputs (identical on every rank, so every rank throws together) and pick the
 *     starting weights.
 *  2. Evaluate at the start. The sampled material volume (sum of all cell volumes) is known only
 *     now, so the targets are rescaled to sum to it here, and F is computed with those targets.
 *     Rescaling matters: if sum V^target differed from the material volume, F would decrease
 *     without bound along "add a constant to every weight" and the climb would never end.
 *  3. Loop until converged (max relative volume error <= tolerance), max_iterations accepted
 *     steps, or a failed line search:
 *     a. direction d = -H G from the L-BFGS pairs (scalar gamma0 when there are none);
 *     b. if d is not a descent direction for F (slope G . d >= 0, possible when the sampled F is
 *        not smooth), drop the pairs and use the scaled steepest direction d = -gamma0 G;
 *     c. Armijo backtracking: try alpha = 1, 1/2, 1/4, ... until
 *        F(w + alpha d) <= F(w) + c alpha (G . d), i.e. Phi rises by at least the fraction c of
 *        what its slope predicts, and F(w + alpha d) < F(w) strictly (in floating point the
 *        Armijo bound rounds to F(w) for tiny steps, which would accept "no change"). If all halvings fail with stored pairs, the pairs are dropped
 *        and the line search is retried once along the steepest direction; if that fails too,
 *        the samples cannot resolve a better answer and the fit stops;
 *     d. store the pair s = alpha d, y = G_new - G when y . s > 0 (positive curvature; for a
 *        concave Phi it is never negative, and it is zero on a sampling plateau, where the pair
 *        carries no information), keep the newest `history` pairs, and set the scalar
 *        gamma = (s . y) / (y . y), the inverse stiffness measured along the last step.
 */
WeightFitResult FitWeights(MPI_Comm comm, const WeightedSamples& samples, const SeedSet& seeds,
                           const std::vector<double>& target_volume,
                           const std::vector<double>& initial_weight,
                           const WeightFitParameters& params) {
    // Step 1: input checks (all inputs are replicated, so every rank throws together)
    ValidateWeightFitParameters(params);
    const std::size_t n = seeds.Size();
    if (n == 0) {
        throw std::invalid_argument("Weight fit: the seed set is empty");
    }
    if (target_volume.size() != n) {
        throw std::invalid_argument("Weight fit: target_volume must have one entry per seed");
    }
    for (double v : target_volume) {
        if (!(v > 0.0) || !std::isfinite(v)) {
            throw std::invalid_argument("Weight fit: every target volume must be positive and "
                                        "finite");
        }
    }
    std::vector<double> w;
    if (!initial_weight.empty()) {
        w = initial_weight;
    } else if (!seeds.weight.empty()) {
        w = seeds.weight;
    } else {
        w.assign(n, 0.0);
    }
    if (w.size() != n) {
        throw std::invalid_argument("Weight fit: initial weights must have one entry per seed");
    }
    for (double v : w) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("Weight fit: initial weights must be finite");
        }
    }

    WeightFitResult result;
    WeightFitReport& report = result.report;
    DualObjective objective(comm, samples, seeds, params);

    // Step 2: first labeling, target normalization, first objective value
    double first_integral = 0.0;
    CellMeasures first = objective.Measure(w, first_integral);
    if (!(first.total_volume > 0.0)) {
        throw std::invalid_argument("Weight fit: the samples have zero total volume");
    }
    double target_sum = 0.0;
    for (double v : target_volume) {
        target_sum += v;
    }
    std::vector<double> target = target_volume;
    for (double& v : target) {
        v *= first.total_volume / target_sum;
    }
    double min_target = target[0];
    for (double v : target) {
        min_target = std::fmin(min_target, v);
    }
    report.resolution_floor = objective.Accumulator().MaxSampleWeight() / min_target;

    // Scalar inverse stiffness for steps without curvature pairs (see WeightFitParameters)
    const double mean_cell_volume = first.total_volume / static_cast<double>(n);
    Evaluation current = DualObjective::Assemble(w, target, std::move(first), first_integral);
    Record(report, current);
    double gamma0 =
        InitialInverseStiffness(params.metric, mean_cell_volume, params.setup, seeds.shape);
    if (params.initial_inverse_stiffness > 0.0) {
        gamma0 = params.initial_inverse_stiffness;
    }
    report.initial_inverse_stiffness = gamma0;
    double gamma = gamma0;
    std::deque<CurvaturePair> pairs;

    // Step 3: L-BFGS iterations
    report.stop_reason = "max_iterations";
    while (true) {
        if (current.max_relative_error <= params.tolerance) {
            report.converged = true;
            report.stop_reason = "converged";
            break;
        }
        if (report.iterations >= params.max_iterations) {
            break;
        }

        // 3a-3c: direction and line search, with one steepest-direction retry
        bool accepted = false;
        std::vector<double> w_new;
        Evaluation trial;
        for (int attempt = 0; attempt < 2 && !accepted; ++attempt) {
            std::vector<double> direction;
            double slope = 0.0;
            if (!pairs.empty()) {
                direction = TwoLoopDirection(pairs, current.gradient, gamma);
                slope = Dot(current.gradient, direction);
            }
            if (pairs.empty() || !(slope < 0.0)) {
                // 3b: steepest direction scaled by the initial inverse stiffness
                if (!pairs.empty()) {
                    ++report.num_restarts;
                }
                pairs.clear();
                gamma = gamma0;
                direction = current.gradient;
                for (double& v : direction) {
                    v *= -gamma0;
                }
                slope = Dot(current.gradient, direction);
            }
            if (!(slope < 0.0)) {
                break; // zero gradient: nothing left to improve
            }

            // 3c: Armijo backtracking on F = -Phi
            double alpha = 1.0;
            for (int halving = 0; halving <= params.max_backtracks; ++halving) {
                w_new.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    w_new[i] = w[i] + alpha * direction[i];
                }
                trial = objective.Evaluate(w_new, target);
                // Strictly lower F as well: once c alpha slope is below the rounding unit of F,
                // F + c alpha slope rounds to F and the Armijo test alone would accept a step
                // that does not change Phi at all (an endless walk on a sampling plateau)
                const bool sufficient =
                    trial.objective <= current.objective + params.armijo_c * alpha * slope;
                if (sufficient && trial.objective < current.objective) {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) {
                if (pairs.empty()) {
                    break; // the steepest direction failed too: sample resolution reached
                }
                pairs.clear(); // retry once along the steepest direction
                ++report.num_restarts;
            }
        }
        if (!accepted) {
            report.stop_reason = "line_search_failed";
            break;
        }

        // 3d: curvature pair from the accepted step
        CurvaturePair pair;
        pair.s.resize(n);
        pair.y.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            pair.s[i] = w_new[i] - w[i];
            pair.y[i] = trial.gradient[i] - current.gradient[i];
        }
        const double sy = Dot(pair.s, pair.y);
        const double yy = Dot(pair.y, pair.y);
        if (sy > 0.0 && yy > 0.0) {
            pair.rho = 1.0 / sy;
            gamma = sy / yy;
            pairs.push_back(std::move(pair));
            while (static_cast<int>(pairs.size()) > params.history) {
                pairs.pop_front();
            }
        }

        w = std::move(w_new);
        current = std::move(trial);
        ++report.iterations;
        Record(report, current);
    }

    report.evaluations = objective.Evaluations();
    result.weight = std::move(w);
    result.target_volume = std::move(target);
    result.measures = std::move(current.measures);
    return result;
}

} // namespace exaconstit::microstructure
