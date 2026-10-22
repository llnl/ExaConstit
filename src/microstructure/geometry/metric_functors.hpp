#pragma once

#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <float.h>
#include <math.h>
#include <stdexcept>
#include <string>

/**
 * @file metric_functors.hpp
 * @brief The distance-like functions that decide which grain a point belongs to (the
 * "tessellation metrics"), written as small plain-data objects usable inside GPU kernels.
 *
 * **What a tessellation metric is.** Every grain grows from a seed s_i. Each metric gives every
 * seed a function
 *
 *     f_i(x) = g_i(x) - w_i,
 *
 * where g_i is a distance-like "base function" and w_i a per-seed weight, and a point x belongs
 * to the grain whose seed has the smallest f_i(x): grain(x) = argmin_i f_i(x). Ties (equal f)
 * go to the lowest seed index, so the answer never depends on the order in which seeds are
 * visited, on the execution backend or on the MPI rank count. The grain boundary between grains
 * i and j is the surface f_i(x) = f_j(x); its shape is what distinguishes the metrics:
 *
 * | Metric   | g_i(x)                 | weight | grain boundary shape                          |
 * |----------|------------------------|--------|-----------------------------------------------|
 * | Voronoi  | \|x - s_i\|^2          | none   | planes halfway between seeds                  |
 * | Laguerre | \|x - s_i\|^2          | w_i    | planes, shifted toward the smaller weight     |
 * | Quadric  | (x-s_i)^T A_i (x-s_i)  | w_i    | quadrics (planes for one global A)            |
 * | Additive | \|x - s_i\|            | a_i    | sheets of hyperboloids of revolution          |
 *
 * See each functor for the physical (nucleation and growth) picture behind the metric.
 *
 * **Functor interface.** Each metric is a struct with these members, all `MFEM_HOST_DEVICE`
 * (callable on the host and inside GPU kernels), const, and free of allocation or exceptions:
 * - `double Eval(int i, double x, double y, double z)`: f_i at the point (x, y, z);
 * - `void Grad(int i, double x, double y, double z, double g[3])`: the gradient of f_i
 *   (needed later to move mesh nodes onto grain boundaries);
 * - `double LowerBound(double sqdist_point_box, const BinBound& bound)`: a number that is
 *   guaranteed to be <= f_i(x) for every seed i inside a box whose squared distance to x is
 *   `sqdist_point_box` and whose seed parameters are summarized by `bound`. The seed search
 *   (seed_index.hpp) skips a whole box of seeds when this bound is already larger than the best
 *   f found so far; that is what makes the search fast and still exact.
 * - `double LowerBound(const double p[3], const double box_lo[3], const double box_hi[3],
 *   double sqdist_point_box, const BinBound& bound)`: the same guarantee, but the box itself
 *   (its corners) is also passed, not only its squared distance to the query point p. The seed
 *   search always calls this form. Voronoi, Laguerre, Additive and the per-seed Quadric metric
 *   ignore the corners and return the distance-only bound above. The Quadric metric with one
 *   global tensor A uses the corners for a much sharper "projected-interval" bound (see
 *   QuadricMetric), which is what makes columnar and lamellar grains (semidefinite A) fast.
 *
 * The functors hold raw pointers (a SeedDeviceView) into seed arrays owned by someone else
 * (normally SeedIndex), so they are cheap to copy into kernels. They do not own memory; the
 * pointers must be valid in the memory space where the functor is used (device pointers inside
 * GPU kernels).
 *
 * **Round-off safety of the lower bounds.** The search compares LowerBound against Eval values
 * computed by different floating-point expressions, and compilers may fuse a*b + c into one
 * fused multiply-add (FMA) instruction in one place and not the other. A bound that is
 * mathematically equal to f (for example at the box of a single seed) could then come out one
 * rounding unit *larger* than the computed f and wrongly skip the true nearest seed. Every bound
 * is therefore lowered by a tiny relative margin, METRIC_ROUNDOFF_SLACK (64 machine epsilons,
 * about 1.4e-14) times the size of the terms involved. That is far below any geometric scale, so
 * it costs nothing in search speed, and it keeps LowerBound <= Eval true in floating point.
 *
 * **Columnar and lamellar grains.** The global quadric tensors A = I - a a^T and A = n n^T that
 * produce them are built by MakeColumnarTensor / MakeLamellarTensor in
 * geometry/anisotropy_tensors.hpp.
 *
 * **Math functions.** Device-callable code uses the C math library from <math.h> (sqrt, fabs),
 * not std::sqrt, because the C names are the ones reliably available in CUDA/HIP device code.
 * The host-only helper ComputeQuadricFactor uses <cmath>.
 */

namespace exaconstit::microstructure {

/**
 * @brief Relative safety margin subtracted from every metric lower bound (64 machine epsilons,
 * about 1.4e-14). See "Round-off safety" in the file comment.
 */
inline constexpr double METRIC_ROUNDOFF_SLACK = 64.0 * DBL_EPSILON;

/**
 * @brief Summary of the seed parameters inside one group ("bin") of seeds, used by the metric
 * lower bounds.
 *
 * Plain data, copied into kernels. Units:
 * - w_max, w_absmax: units of the metric value f, i.e. length^2 for Laguerre, length for
 *   Additive, and for Quadric the units of (x - s)^T A (x - s); with the usual dimensionless A
 *   (a pure stretch of directions) that is length^2 as well. Unused (0) for Voronoi.
 * - mu_min, mu_max: units of the quadric tensor A, i.e. (units of f) / length^2; dimensionless
 *   for the usual stretch-only A. Equal to 1 (dimensionless placeholders) for the metrics
 *   without a tensor.
 */
struct BinBound {
    /** @brief Largest weight w_i (or a_i) of the seeds in the bin. */
    double w_max = 0.0;
    /** @brief Largest |w_i| in the bin; scales the round-off margin of the weight term. */
    double w_absmax = 0.0;
    /**
     * @brief Smallest eigenvalue of the quadric tensors A_i in the bin (>= 0); 1 for the
     * metrics without a tensor.
     */
    double mu_min = 1.0;
    /**
     * @brief Largest eigenvalue of the quadric tensors A_i in the bin; 1 for the metrics without
     * a tensor. Scales the round-off margin of the quadric term.
     */
    double mu_max = 1.0;
};

/**
 * @brief Non-owning raw pointers to the per-seed data a metric functor reads.
 *
 * All arrays are indexed by the 0-based seed index i of the SeedSet the data came from. Plain
 * data; pointers must be valid in the memory space where the functor runs.
 */
struct SeedDeviceView {
    /** @brief Seed coordinates, size N each, mesh length units. */
    const double* x = nullptr;
    const double* y = nullptr;
    const double* z = nullptr;
    /** @brief Weight w_i (Laguerre/Quadric) or a_i (Additive), size N. Unused by Voronoi. */
    const double* weight = nullptr;
    /**
     * @brief Quadric only: factor M_i of the tensor A_i = M_i^T M_i, 9 entries per seed, row
     * major: M_i(r, c) = factor[factor_stride * i + 3 * r + c]. With a global tensor the same 9
     * entries serve every seed (factor_stride = 0).
     */
    const double* factor = nullptr;
    /** @brief 9 for per-seed tensors, 0 for one global tensor. */
    int factor_stride = 0;
};

/**
 * @brief Host-side metric parameters that are not stored per seed in the SeedSet.
 *
 * Only the Quadric metric reads these. Plain data; host only.
 */
struct MetricSetup {
    /**
     * @brief Global quadric tensor A, 6 independent entries [Axx, Ayy, Azz, Ayz, Axz, Axy]
     * (same order as SeedSet::shape and TessellationOptions::global_tensor). Used when
     * per_seed_shape is false. Must be symmetric positive semidefinite and nonzero.
     */
    std::array<double, 6> global_tensor = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    /** @brief If true, the Quadric metric uses SeedSet::shape (6 entries per seed) instead. */
    bool per_seed_shape = false;
};

/**
 * @brief Factors a symmetric positive semidefinite 3x3 tensor as A = M^T M and returns the
 * extreme eigenvalues of A.
 *
 * Why a factor: with M available, the quadric distance is computed as a sum of squares,
 * (x - s)^T A (x - s) = |M (x - s)|^2, which is never negative in floating point and has no
 * cancellation between large terms. That keeps the metric lower bound valid in round-off even
 * for semidefinite tensors (columnar or lamellar grains).
 *
 * Algorithm: eigen-decomposition A = Q diag(lambda) Q^T with MFEM's closed-form symmetric 3x3
 * solver (mfem::DenseMatrix::CalcEigenvalues, densemat.hpp:331-333: eigenvalues in increasing
 * order, eigenvector k stored as column k, i.e. vec[3k .. 3k+2]); then row k of M is
 * sqrt(lambda_k) * q_k^T, so M^T M = sum_k lambda_k q_k q_k^T = A. Eigenvalues that are negative
 * only by round-off (>= -1e-12 * lambda_max) are set to zero.
 *
 * **Why the eigenvalues are computed twice.** The quadric lower bound relies on
 * |M d|^2 >= mu_min |d|^2, i.e. on mu_min being a lower bound of the eigenvalues of the matrix
 * M^T M that Eval *actually* uses. That equals lambda only if the computed eigenvectors are
 * exactly orthonormal; with nearly repeated eigenvalues the closed-form solver can return
 * eigenvectors that are slightly non-orthogonal, and the eigenvalues of the computed M^T M can
 * then drift from lambda by more than the 64-epsilon round-off margin of the bound. So after
 * building M we form M^T M explicitly, compute its eigenvalues again, and report mu_min / mu_max
 * as the minimum / maximum over both eigenvalue sets (mu_min clamped at 0). This costs two 3x3
 * eigen-solves per tensor on the host, once at build time.
 *
 * Host only; not collective.
 *
 * @param tensor  6 entries [Axx, Ayy, Azz, Ayz, Axz, Axy].
 * @param factor  output, 9 entries, row-major M(r, c) = factor[3 r + c].
 * @param mu_min  output, smallest eigenvalue of A and of the computed M^T M (>= 0).
 * @param mu_max  output, largest eigenvalue of A and of the computed M^T M (> 0).
 * @throws std::invalid_argument if an entry is not finite, if A is zero, or if A has an
 *         eigenvalue below -1e-12 * lambda_max (A is indefinite, so "distance" could be
 *         negative and grains would not be well defined).
 */
inline void ComputeQuadricFactor(const double tensor[6], double factor[9], double& mu_min,
                                 double& mu_max) {
    for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(tensor[k])) {
            throw std::invalid_argument("ComputeQuadricFactor: tensor entry is not finite");
        }
    }

    // Assemble the full symmetric matrix from the 6 independent entries
    mfem::DenseMatrix a(3);
    a(0, 0) = tensor[0];
    a(1, 1) = tensor[1];
    a(2, 2) = tensor[2];
    a(1, 2) = tensor[3];
    a(2, 1) = tensor[3];
    a(0, 2) = tensor[4];
    a(2, 0) = tensor[4];
    a(0, 1) = tensor[5];
    a(1, 0) = tensor[5];

    // Eigenvalues (increasing) and eigenvectors (column k of vec belongs to lambda[k])
    double lambda[3];
    double vec[9];
    a.CalcEigenvalues(lambda, vec);

    // Reject zero and indefinite tensors; clamp round-off negatives to exactly zero
    const double lambda_max = lambda[2];
    if (!(lambda_max > 0.0)) {
        throw std::invalid_argument("ComputeQuadricFactor: tensor A is zero or negative "
                                    "definite; it must be positive (semi)definite");
    }
    for (int k = 0; k < 3; ++k) {
        if (lambda[k] < -1.0e-12 * lambda_max) {
            throw std::invalid_argument(
                "ComputeQuadricFactor: tensor A is indefinite (eigenvalue " +
                std::to_string(lambda[k]) + "); it must be positive (semi)definite");
        }
        if (lambda[k] < 0.0) {
            lambda[k] = 0.0;
        }
    }

    // Row k of M = sqrt(lambda_k) * (eigenvector k)^T
    for (int k = 0; k < 3; ++k) {
        const double root = std::sqrt(lambda[k]);
        for (int c = 0; c < 3; ++c) {
            factor[3 * k + c] = root * vec[3 * k + c];
        }
    }

    // Eigenvalues of the M^T M actually used by Eval (see "Why the eigenvalues are computed
    // twice"): (M^T M)(r, c) = sum_k M(k, r) M(k, c)
    mfem::DenseMatrix mtm(3);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += factor[3 * k + r] * factor[3 * k + c];
            }
            mtm(r, c) = sum;
        }
    }
    double lambda_mtm[3];
    double vec_mtm[9];
    mtm.CalcEigenvalues(lambda_mtm, vec_mtm);

    // Bounds valid for both A and the computed M^T M
    mu_min = std::min(lambda[0], lambda_mtm[0]);
    mu_max = std::max(lambda_max, lambda_mtm[2]);
    if (mu_min < 0.0) {
        mu_min = 0.0;
    }
}

/**
 * @brief Voronoi metric: f_i(x) = |x - s_i|^2.
 *
 * Physical picture: all grains nucleate at the same time and grow at the same isotropic speed
 * ("site-saturated" nucleation); a point is reached first by the closest seed. Grain boundaries
 * are the planes halfway between neighboring seeds (perpendicular bisectors), and every grain is
 * a convex polyhedron containing its seed. Seed weights are ignored.
 *
 * Squared distance is used instead of distance because it gives the same nearest seed, avoids a
 * square root, and matches the Laguerre form.
 */
struct VoronoiMetric {
    /** @brief Metric family tag, used by SeedIndex to choose host-side preprocessing. */
    static constexpr MetricType TYPE = MetricType::VORONOI;

    /** @brief Pointers to the seed arrays (only x, y, z are read). */
    SeedDeviceView m_seeds;

    /** @brief f_i(x) = |x - s_i|^2 (length^2). @param i 0-based seed index. */
    MFEM_HOST_DEVICE double Eval(int i, double x, double y, double z) const {
        const double dx = x - m_seeds.x[i];
        const double dy = y - m_seeds.y[i];
        const double dz = z - m_seeds.z[i];
        return dx * dx + dy * dy + dz * dz;
    }

    /** @brief grad f_i = 2 (x - s_i). @param g output, 3 components. */
    MFEM_HOST_DEVICE void Grad(int i, double x, double y, double z, double g[3]) const {
        g[0] = 2.0 * (x - m_seeds.x[i]);
        g[1] = 2.0 * (y - m_seeds.y[i]);
        g[2] = 2.0 * (z - m_seeds.z[i]);
    }

    /**
     * @brief Lower bound of f over seeds in a box: |x - s|^2 >= d^2 for every s in the box, where
     * d^2 is the squared distance from x to the box. Lowered by the round-off margin.
     * @param sqdist_point_box squared distance from x to the box (>= 0).
     */
    MFEM_HOST_DEVICE double LowerBound(double sqdist_point_box, const BinBound& bound) const {
        (void)bound;
        return (1.0 - METRIC_ROUNDOFF_SLACK) * sqdist_point_box;
    }

    /**
     * @brief Box form used by the seed search; the box corners are not needed for Voronoi, so
     * this returns LowerBound(sqdist_point_box, bound). @param p, box_lo, box_hi unused.
     */
    MFEM_HOST_DEVICE double LowerBound(const double p[3], const double box_lo[3],
                                       const double box_hi[3], double sqdist_point_box,
                                       const BinBound& bound) const {
        (void)p;
        (void)box_lo;
        (void)box_hi;
        return LowerBound(sqdist_point_box, bound);
    }
};

/**
 * @brief Laguerre (power diagram) metric: f_i(x) = |x - s_i|^2 - w_i.
 *
 * Physical picture: like Voronoi, but a seed with a larger weight claims more space. For two
 * seeds a and b the boundary is still a plane perpendicular to s_b - s_a, at
 * n.x = (|s_b|^2 - |s_a|^2 + w_a - w_b) / 2 with n = s_b - s_a, i.e. shifted toward the seed
 * with the smaller weight. Choosing w_i = r_i^2 for a target grain "radius" r_i gives control of
 * the grain size distribution (this is Neper's default tessellation). Grains are convex
 * polyhedra, but a seed whose weight is too small can lose its grain entirely (empty cell), and
 * a seed is not always inside its own grain.
 *
 * Weights have units of length^2 and may be negative (only differences matter).
 */
struct LaguerreMetric {
    /** @brief Metric family tag, used by SeedIndex to choose host-side preprocessing. */
    static constexpr MetricType TYPE = MetricType::LAGUERRE;

    /** @brief Pointers to the seed arrays (x, y, z and weight are read). */
    SeedDeviceView m_seeds;

    /** @brief f_i(x) = |x - s_i|^2 - w_i (length^2). @param i 0-based seed index. */
    MFEM_HOST_DEVICE double Eval(int i, double x, double y, double z) const {
        const double dx = x - m_seeds.x[i];
        const double dy = y - m_seeds.y[i];
        const double dz = z - m_seeds.z[i];
        return (dx * dx + dy * dy + dz * dz) - m_seeds.weight[i];
    }

    /** @brief grad f_i = 2 (x - s_i) (the weight is constant). @param g output, 3 components. */
    MFEM_HOST_DEVICE void Grad(int i, double x, double y, double z, double g[3]) const {
        g[0] = 2.0 * (x - m_seeds.x[i]);
        g[1] = 2.0 * (y - m_seeds.y[i]);
        g[2] = 2.0 * (z - m_seeds.z[i]);
    }

    /**
     * @brief Lower bound d^2 - w_max: no seed in the box is closer than d and none has a larger
     * weight than w_max. Lowered by the round-off margin.
     */
    MFEM_HOST_DEVICE double LowerBound(double sqdist_point_box, const BinBound& bound) const {
        return (1.0 - METRIC_ROUNDOFF_SLACK) * sqdist_point_box - bound.w_max -
               METRIC_ROUNDOFF_SLACK * bound.w_absmax;
    }

    /**
     * @brief Box form used by the seed search; returns LowerBound(sqdist_point_box, bound)
     * (the box corners add nothing for an isotropic metric). @param p, box_lo, box_hi unused.
     */
    MFEM_HOST_DEVICE double LowerBound(const double p[3], const double box_lo[3],
                                       const double box_hi[3], double sqdist_point_box,
                                       const BinBound& bound) const {
        (void)p;
        (void)box_lo;
        (void)box_hi;
        return LowerBound(sqdist_point_box, bound);
    }
};

/**
 * @brief Quadric (anisotropic) metric: f_i(x) = (x - s_i)^T A_i (x - s_i) - w_i, with A_i
 * symmetric positive semidefinite.
 *
 * A acts as a ruler that measures distance differently along different directions: directions
 * with a small eigenvalue of A "look short", so grains grow long along them.
 * - **One global A** (MetricSetup::global_tensor): the result is a Voronoi/Laguerre tessellation
 *   of space stretched by A, i.e. exactly Voronoi in the coordinates y = M x with A = M^T M.
 *   Boundaries are planes, grains are convex but elongated (e.g. pancake grains of rolled
 *   sheet). A semidefinite A gives infinitely long grains: A = I - a a^T (|a| = 1) gives
 *   columnar grains along a (distance along a is ignored); A = n n^T gives lamellae
 *   perpendicular to n.
 * - **Per-seed A_i** (SeedSet::shape): every grain has its own ellipsoidal growth shape and
 *   orientation (e.g. from measured grain shape tensors). Boundaries are curved quadric surfaces
 *   and a grain can be disconnected.
 * - **Multiplicative weighting**, A_i = I / lambda_i^2 and w_i = 0: grains nucleate at the same
 *   time but grow at different speeds lambda_i; f_i = (|x - s_i| / lambda_i)^2 is the squared
 *   arrival time. The boundary between two grains is a sphere (an Apollonius sphere) around the
 *   slower grain, which ends up enclosed by the faster one.
 *
 * Internally A_i is stored as its factor M_i (A_i = M_i^T M_i, see ComputeQuadricFactor) and
 * evaluated as the sum of squares |M_i (x - s_i)|^2.
 */
struct QuadricMetric {
    /** @brief Metric family tag, used by SeedIndex to choose host-side preprocessing. */
    static constexpr MetricType TYPE = MetricType::QUADRIC;

    /** @brief Pointers to the seed arrays (x, y, z, weight, factor and factor_stride). */
    SeedDeviceView m_seeds;

    /**
     * @brief f_i(x) = |M_i (x - s_i)|^2 - w_i, equal to (x - s_i)^T A_i (x - s_i) - w_i.
     * @param i 0-based seed index.
     */
    MFEM_HOST_DEVICE double Eval(int i, double x, double y, double z) const {
        const double dx = x - m_seeds.x[i];
        const double dy = y - m_seeds.y[i];
        const double dz = z - m_seeds.z[i];
        const double* m = m_seeds.factor + m_seeds.factor_stride * i;
        // Stretched offset y = M d, row by row (M is row major)
        const double y0 = m[0] * dx + m[1] * dy + m[2] * dz;
        const double y1 = m[3] * dx + m[4] * dy + m[5] * dz;
        const double y2 = m[6] * dx + m[7] * dy + m[8] * dz;
        return (y0 * y0 + y1 * y1 + y2 * y2) - m_seeds.weight[i];
    }

    /** @brief grad f_i = 2 A_i (x - s_i) = 2 M_i^T (M_i (x - s_i)). @param g output, 3 comps. */
    MFEM_HOST_DEVICE void Grad(int i, double x, double y, double z, double g[3]) const {
        const double dx = x - m_seeds.x[i];
        const double dy = y - m_seeds.y[i];
        const double dz = z - m_seeds.z[i];
        const double* m = m_seeds.factor + m_seeds.factor_stride * i;
        const double y0 = m[0] * dx + m[1] * dy + m[2] * dz;
        const double y1 = m[3] * dx + m[4] * dy + m[5] * dz;
        const double y2 = m[6] * dx + m[7] * dy + m[8] * dz;
        // g = 2 M^T y: column c of M dotted with y
        g[0] = 2.0 * (m[0] * y0 + m[3] * y1 + m[6] * y2);
        g[1] = 2.0 * (m[1] * y0 + m[4] * y1 + m[7] * y2);
        g[2] = 2.0 * (m[2] * y0 + m[5] * y1 + m[8] * y2);
    }

    /**
     * @brief Lower bound mu_min d^2 - w_max: (x - s)^T A (x - s) >= lambda_min(A) |x - s|^2 >=
     * mu_min d^2. With a semidefinite A (mu_min = 0) this degenerates to -w_max and gives no
     * distance pruning: the search then visits every seed (correct, but slow). For one global A
     * the box form below adds the projected-interval bound, which does prune; per-seed
     * semidefinite tensors still get only this bound.
     *
     * The round-off margin is applied to the eigenvalue (mu_min - slack * mu_max, never below
     * 0), because the rounding error of |M d|^2 scales with the largest eigenvalue times |d|^2,
     * and to the weight term.
     */
    MFEM_HOST_DEVICE double LowerBound(double sqdist_point_box, const BinBound& bound) const {
        double mu = bound.mu_min - METRIC_ROUNDOFF_SLACK * bound.mu_max;
        if (mu < 0.0) {
            mu = 0.0;
        }
        return mu * sqdist_point_box - bound.w_max - METRIC_ROUNDOFF_SLACK * bound.w_absmax;
    }

    /**
     * @brief Box form used by the seed search. Per-seed tensors (factor_stride != 0): the
     * eigenvalue bound above. One global tensor (factor_stride == 0): the larger of the
     * eigenvalue bound and the **projected-interval bound** below; both are valid lower bounds,
     * so their maximum is too.
     *
     * **Projected-interval bound in plain words.** With A = M^T M, f_i(x) + w_i = |M (x - s_i)|^2
     * = sum over the three rows m_k of M of (m_k . x - m_k . s_i)^2. Fix one row m_k. When the
     * seed s moves over the box [box_lo, box_hi], the number m_k . s is a linear function of s,
     * so it takes its smallest and largest values at box corners: P_lo = sum_c m_kc * (box_lo_c
     * if m_kc >= 0, else box_hi_c) and P_hi likewise with lo and hi swapped. Every seed in the
     * box therefore has m_k . s in the interval [P_lo, P_hi] (the "shadow" of the box on the
     * direction m_k), and the term (m_k . x - m_k . s)^2 is at least gap_k^2, where gap_k is
     * the distance from the number m_k . x to that interval (0 if it lies inside). Summing over
     * k and using w_i <= w_max:
     *
     *     f_i(x) >= gap_0^2 + gap_1^2 + gap_2^2 - w_max   for every seed i in the box.
     *
     * Writing A = sum_k lambda_k u_k u_k^T (eigen-decomposition, m_k = sqrt(lambda_k) u_k), this
     * is sum_k lambda_k dist(x . u_k, [min_box s . u_k, max_box s . u_k])^2 - w_max.
     *
     * **Why it is valid, briefly.** For each k and each s in the box, m_k . s lies in [P_lo,
     * P_hi] (a linear function on a box is extreme at a corner), so |m_k . x - m_k . s| >= gap_k
     * (distance from a point to any member of an interval is at least the distance to the
     * interval). Squares of non-negative numbers keep the order; sums of valid term bounds bound
     * the sum; and -w_i >= -w_max. Nothing here assumes that the rows m_k are orthogonal or that
     * A has full rank, so the bound holds for exactly the factor M that Eval uses, including the
     * zero rows of a semidefinite A (they contribute gap 0).
     *
     * **Why it matters.** For columnar grains (A = I - a a^T) or lamellae (A = n n^T) the
     * smallest eigenvalue is 0, so the eigenvalue bound mu_min d^2 - w_max is just -w_max and
     * prunes nothing: every query would visit every seed. The projected-interval bound still
     * measures the distance in the directions that A does see (the plane perpendicular to the
     * column axis, or the layer normal), so boxes of seeds whose columns or layers are far from
     * x are skipped.
     *
     * **Round-off margin.** Eval computes y_k = m_k . (x - s) in floating point, while the bound
     * computes m_k . x and the corner sums separately. Both differ from the exact values by a few
     * rounding units of scale_k = sum_c |m_kc| (|x_c| + max(|box_lo_c|, |box_hi_c|)), which
     * bounds the size of every product involved (|s_c| <= max(|box_lo_c|, |box_hi_c|) for s in
     * the box). Each gap is therefore lowered by METRIC_ROUNDOFF_SLACK * scale_k (64 rounding
     * units, against about 12 that can occur) and clamped at 0, the sum of squares is
     * scaled by (1 - METRIC_ROUNDOFF_SLACK), and the weight term gets the same margin as the
     * other bounds. So the computed bound stays <= the computed f.
     *
     * Cost: 3 x 3 multiply-adds per box, done inside the BVH traversal (host or device).
     *
     * @param p                query point x (3 coordinates, mesh length units).
     * @param box_lo, box_hi   corners of the box of seeds (3 coordinates each; box_lo <= box_hi).
     * @param sqdist_point_box squared distance from p to the box (>= 0), for the eigenvalue bound.
     * @param bound            weight and eigenvalue summary of the seeds in the box's bin.
     */
    MFEM_HOST_DEVICE double LowerBound(const double p[3], const double box_lo[3],
                                       const double box_hi[3], double sqdist_point_box,
                                       const BinBound& bound) const {
        // Bound from the smallest eigenvalue (the only one valid for per-seed tensors)
        const double eigen_bound = LowerBound(sqdist_point_box, bound);
        if (m_seeds.factor_stride != 0) {
            return eigen_bound;
        }

        // One global factor M: projected-interval bound, row by row (M is row major)
        const double* m = m_seeds.factor;
        double sum_gap_sq = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double* row = m + 3 * k;
            double proj_point = 0.0;  // m_k . x
            double proj_lo = 0.0;     // smallest m_k . s over the box corners
            double proj_hi = 0.0;     // largest m_k . s over the box corners
            double scale = 0.0;       // size of the terms, for the round-off margin
            for (int c = 0; c < 3; ++c) {
                const double mc = row[c];
                proj_point += mc * p[c];
                // A positive coefficient is smallest at the low corner, a negative one at the
                // high corner
                if (mc >= 0.0) {
                    proj_lo += mc * box_lo[c];
                    proj_hi += mc * box_hi[c];
                } else {
                    proj_lo += mc * box_hi[c];
                    proj_hi += mc * box_lo[c];
                }
                const double box_abs = (fabs(box_lo[c]) > fabs(box_hi[c])) ? fabs(box_lo[c])
                                                                           : fabs(box_hi[c]);
                scale += fabs(mc) * (fabs(p[c]) + box_abs);
            }
            // Distance from m_k . x to the interval [proj_lo, proj_hi], minus the margin
            double gap = 0.0;
            if (proj_point < proj_lo) {
                gap = proj_lo - proj_point;
            } else if (proj_point > proj_hi) {
                gap = proj_point - proj_hi;
            }
            gap -= METRIC_ROUNDOFF_SLACK * scale;
            if (gap > 0.0) {
                sum_gap_sq += gap * gap;
            }
        }
        const double interval_bound = (1.0 - METRIC_ROUNDOFF_SLACK) * sum_gap_sq - bound.w_max -
                                      METRIC_ROUNDOFF_SLACK * bound.w_absmax;

        // Both are valid lower bounds; the larger one prunes more
        return (interval_bound > eigen_bound) ? interval_bound : eigen_bound;
    }
};

/**
 * @brief Additive (Apollonius / Johnson-Mehl) metric: f_i(x) = |x - s_i| - a_i.
 *
 * Physical picture: grains grow at the same speed v but start at different times t_i. The
 * arrival time at x is t_i + |x - s_i| / v, which orders seeds exactly like |x - s_i| - a_i with
 * a_i = -v t_i (a seed that starts earlier has a larger head start a_i). This is the
 * Johnson-Mehl tessellation of space-time Poisson nucleation. The boundary between two grains,
 * |x - s_a| - |x - s_b| = a_a - a_b, is one sheet of a hyperboloid of revolution with foci s_a
 * and s_b, curving around the grain that started later. Grains are connected (star-shaped around
 * their seed), but a seed that starts too late (inside a grain that already covers it) gets no
 * grain.
 *
 * a_i has units of length and may be negative.
 */
struct AdditiveMetric {
    /** @brief Metric family tag, used by SeedIndex to choose host-side preprocessing. */
    static constexpr MetricType TYPE = MetricType::ADDITIVE;

    /** @brief Pointers to the seed arrays (x, y, z and weight = a_i are read). */
    SeedDeviceView m_seeds;

    /** @brief f_i(x) = |x - s_i| - a_i (length). @param i 0-based seed index. */
    MFEM_HOST_DEVICE double Eval(int i, double x, double y, double z) const {
        const double dx = x - m_seeds.x[i];
        const double dy = y - m_seeds.y[i];
        const double dz = z - m_seeds.z[i];
        return sqrt(dx * dx + dy * dy + dz * dz) - m_seeds.weight[i];
    }

    /**
     * @brief grad f_i = (x - s_i) / |x - s_i|, the unit vector away from the seed; set to zero at
     * the seed itself, where the distance is not differentiable. @param g output, 3 components.
     */
    MFEM_HOST_DEVICE void Grad(int i, double x, double y, double z, double g[3]) const {
        const double dx = x - m_seeds.x[i];
        const double dy = y - m_seeds.y[i];
        const double dz = z - m_seeds.z[i];
        const double r = sqrt(dx * dx + dy * dy + dz * dz);
        const double inv_r = (r > 0.0) ? 1.0 / r : 0.0;
        g[0] = dx * inv_r;
        g[1] = dy * inv_r;
        g[2] = dz * inv_r;
    }

    /**
     * @brief Lower bound sqrt(d^2) - a_max: no seed in the box is closer than d and none has a
     * larger head start than a_max. Lowered by the round-off margin (sqrt is monotone, so
     * shrinking d^2 slightly shrinks the bound).
     */
    MFEM_HOST_DEVICE double LowerBound(double sqdist_point_box, const BinBound& bound) const {
        return sqrt((1.0 - METRIC_ROUNDOFF_SLACK) * sqdist_point_box) - bound.w_max -
               METRIC_ROUNDOFF_SLACK * bound.w_absmax;
    }

    /**
     * @brief Box form used by the seed search; returns LowerBound(sqdist_point_box, bound).
     * @param p, box_lo, box_hi unused.
     */
    MFEM_HOST_DEVICE double LowerBound(const double p[3], const double box_lo[3],
                                       const double box_hi[3], double sqdist_point_box,
                                       const BinBound& bound) const {
        (void)p;
        (void)box_lo;
        (void)box_hi;
        return LowerBound(sqdist_point_box, bound);
    }
};

/**
 * @brief Calls `f(Metric{})` with a default-constructed (null-pointer) functor of the metric
 * family `type`, so runtime options can select a compile-time metric type.
 *
 * Typical use, together with DispatchAxomExec (exec_dispatch.hpp):
 * @code
 *   DispatchMetric(options.metric, [&](auto metric_tag) {
 *       using M = decltype(metric_tag);
 *       DispatchAxomExec([&](auto exec) {
 *           SeedIndex<decltype(exec), M> index;
 *           ...
 *       });
 *   });
 * @endcode
 * The functor passed to `f` has null data pointers; only its type is meant to be used. Host
 * only; not collective.
 *
 * @throws std::invalid_argument for an unknown MetricType value.
 */
template <typename Func>
void DispatchMetric(MetricType type, Func&& f) {
    switch (type) {
    case MetricType::VORONOI:
        f(VoronoiMetric{});
        return;
    case MetricType::LAGUERRE:
        f(LaguerreMetric{});
        return;
    case MetricType::QUADRIC:
        f(QuadricMetric{});
        return;
    case MetricType::ADDITIVE:
        f(AdditiveMetric{});
        return;
    }
    throw std::invalid_argument("DispatchMetric: unknown MetricType");
}

} // namespace exaconstit::microstructure
