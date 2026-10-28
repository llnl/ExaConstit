#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

/**
 * @file anisotropy_tensors.hpp
 * @brief Builders for the global quadric tensor A of the Quadric metric
 * (geometry/metric_functors.hpp) that produce columnar or lamellar grains.
 *
 * The Quadric metric f_i(x) = (x - s_i)^T A (x - s_i) - w_i measures distance with the tensor A.
 * A positive semidefinite A that is blind to some directions gives grains that are infinitely
 * long in those directions:
 * - columnar grains along a unit axis a: A = I - a a^T (MakeColumnarTensor);
 * - lamellae (layers) with unit normal n: A = n n^T (MakeLamellarTensor).
 * The returned 6 entries [Axx, Ayy, Azz, Ayz, Axz, Axy] go into MetricSetup::global_tensor.
 * The seed search stays fast for these tensors: QuadricMetric bounds the metric value over a box of
 * seeds from below using only the directions A measures (it projects the box onto them), so boxes
 * of seeds that cannot win are still skipped although A ignores some directions.
 *
 * Host only (plain C++, std:: math); not collective.
 */

namespace exaconstit::microstructure {

namespace detail {
/**
 * @brief Returns v / |v| for a direction given by the user (host only).
 * @param v    direction, any nonzero finite length.
 * @param name function name used in the error message.
 * @throws std::invalid_argument if an entry is not finite or v is (numerically) zero.
 */
inline std::array<double, 3> UnitDirection(const std::array<double, 3>& v, const char* name) {
    for (int c = 0; c < 3; ++c) {
        if (!std::isfinite(v[c])) {
            throw std::invalid_argument(std::string(name) + ": direction entry is not finite");
        }
    }
    const double length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (!(length > 0.0)) {
        throw std::invalid_argument(std::string(name) + ": direction must be nonzero");
    }
    return {v[0] / length, v[1] / length, v[2] / length};
}
} // namespace detail

/**
 * @brief Global quadric tensor for **columnar grains** along the axis a: A = I - a a^T, with a
 * normalized to unit length.
 *
 * Physical picture: with this A the quadric metric measures only the part of x - s that is
 * perpendicular to a, (x - s)^T A (x - s) = |x - s|^2 - (a . (x - s))^2. Moving along a does not
 * change any f_i, so every grain is an infinitely long prism parallel to a whose cross-section is
 * the 2D Voronoi (or, with weights, Laguerre) cell of the seeds projected onto the plane
 * perpendicular to a. This mimics the columnar grains of directional solidification or of
 * additively manufactured builds, with a = the build / growth direction. Within the mesh the
 * columns are cut off by the material surface, so they span the part from face to face.
 *
 * A is positive semidefinite with eigenvalues {0, 1, 1} (0 along a), which ComputeQuadricFactor
 * accepts. Host only, not collective.
 *
 * @param axis column direction (any nonzero length; normalized here), mesh coordinates.
 * @return the 6 entries [Axx, Ayy, Azz, Ayz, Axz, Axy] for MetricSetup::global_tensor.
 * @throws std::invalid_argument if the axis is zero or not finite.
 */
inline std::array<double, 6> MakeColumnarTensor(const std::array<double, 3>& axis) {
    const std::array<double, 3> a = detail::UnitDirection(axis, "MakeColumnarTensor");
    // A(r, c) = delta_rc - a_r a_c, stored in the Voigt-like order of MetricSetup
    return {1.0 - a[0] * a[0],
            1.0 - a[1] * a[1],
            1.0 - a[2] * a[2],
            -a[1] * a[2],
            -a[0] * a[2],
            -a[0] * a[1]};
}

/**
 * @brief Global quadric tensor for **lamellar grains** (layers) with normal n: A = n n^T, with n
 * normalized to unit length.
 *
 * Physical picture: (x - s)^T A (x - s) = (n . (x - s))^2 measures only the distance along n, so
 * f_i depends on the position x only through its height h = n . x. Every grain is a flat slab
 * between two planes perpendicular to n; the boundary between the slabs of seeds i and j (equal
 * weights) is the plane n . x = (n . s_i + n . s_j) / 2, halfway between the seed heights. This
 * gives lamellar or layered microstructures (e.g. layer-wise deposits, lamellar colonies). Seeds
 * at the same height tie everywhere, and the lowest index wins (the other gets no grain).
 *
 * A is positive semidefinite with eigenvalues {0, 0, 1}. Host only, not collective.
 *
 * @param normal layer normal (any nonzero length; normalized here), mesh coordinates.
 * @return the 6 entries [Axx, Ayy, Azz, Ayz, Axz, Axy] for MetricSetup::global_tensor.
 * @throws std::invalid_argument if the normal is zero or not finite.
 */
inline std::array<double, 6> MakeLamellarTensor(const std::array<double, 3>& normal) {
    const std::array<double, 3> n = detail::UnitDirection(normal, "MakeLamellarTensor");
    // A(r, c) = n_r n_c
    return {n[0] * n[0], n[1] * n[1], n[2] * n[2], n[1] * n[2], n[0] * n[2], n[0] * n[1]};
}

/**
 * @brief Row-major 3x3 "stretch" S with |S r|^2 = r^T A r for a symmetric positive definite A,
 * the coordinate map of the geodesic solvers for the Quadric metric (labeling/geodesic_metric.hpp):
 * the A-length of a path equals the Euclidean length of its image y = S x. S = L^T from the
 * Cholesky factor A = L L^T (upper triangular).
 *
 * Host only, not collective.
 * @param tensor A as [Axx, Ayy, Azz, Ayz, Axz, Axy].
 * @return S row-major: S[3*r + c].
 * @throws std::invalid_argument if an entry is not finite or A is not positive definite (a pivot
 *         <= 1e-14 times the largest diagonal entry).
 */
inline std::array<double, 9> MetricStretchFromTensor(const std::array<double, 6>& tensor) {
    for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(tensor[k])) {
            throw std::invalid_argument("MetricStretchFromTensor: tensor entry is not finite");
        }
    }
    const double a00 = tensor[0], a11 = tensor[1], a22 = tensor[2];
    const double a12 = tensor[3], a02 = tensor[4], a01 = tensor[5];
    const double scale = std::fmax(std::fmax(std::fabs(a00), std::fabs(a11)), std::fabs(a22));
    const double tiny = 1e-14 * scale;
    const auto pivot = [&](double value) {
        if (!(value > tiny)) {
            throw std::invalid_argument("MetricStretchFromTensor: the tensor is not positive "
                                        "definite");
        }
        return std::sqrt(value);
    };
    // A = L L^T, L lower triangular
    const double l00 = pivot(a00);
    const double l10 = a01 / l00;
    const double l20 = a02 / l00;
    const double l11 = pivot(a11 - l10 * l10);
    const double l21 = (a12 - l20 * l10) / l11;
    const double l22 = pivot(a22 - l20 * l20 - l21 * l21);
    // S = L^T
    return {l00, l10, l20, 0.0, l11, l21, 0.0, 0.0, l22};
}

/**
 * @brief Stretch of the **regularized columnar** metric for the geodesic solvers: columns k times
 * longer than wide along the unit axis a instead of infinitely long,
 * S = I - (1 - 1/k) a a^T, so |S r|^2 = |r|^2 - (1 - 1/k^2) (a . r)^2 (the tensor
 * I - (1 - 1/k^2) a a^T). Built directly (no Cholesky of a nearly singular tensor), so large k
 * stays accurate. Host only, not collective.
 * @param axis       column direction (any nonzero length; normalized here).
 * @param elongation k >= 1 (1 = Voronoi).
 * @throws std::invalid_argument for a zero or non-finite axis or k < 1 / not finite.
 */
inline std::array<double, 9> ColumnarStretch(const std::array<double, 3>& axis, double elongation) {
    const std::array<double, 3> a = detail::UnitDirection(axis, "ColumnarStretch");
    if (!std::isfinite(elongation) || !(elongation >= 1.0)) {
        throw std::invalid_argument("ColumnarStretch: elongation must be finite and >= 1");
    }
    const double c = 1.0 - 1.0 / elongation;
    std::array<double, 9> s{};
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) {
            s[3 * r + col] = (r == col ? 1.0 : 0.0) - c * a[r] * a[col];
        }
    }
    return s;
}

/**
 * @brief Stretch of the **regularized lamellar** metric for the geodesic solvers: layers k times
 * wider than thick (normal n) instead of infinitely wide, S = n n^T + (I - n n^T) / k, so
 * |S r|^2 = (n . r)^2 + |r - (n . r) n|^2 / k^2. Host only, not collective.
 * @param normal     layer normal (any nonzero length; normalized here).
 * @param elongation k >= 1 (1 = Voronoi).
 * @throws std::invalid_argument for a zero or non-finite normal or k < 1 / not finite.
 */
inline std::array<double, 9> LamellarStretch(const std::array<double, 3>& normal,
                                             double elongation) {
    const std::array<double, 3> n = detail::UnitDirection(normal, "LamellarStretch");
    if (!std::isfinite(elongation) || !(elongation >= 1.0)) {
        throw std::invalid_argument("LamellarStretch: elongation must be finite and >= 1");
    }
    const double inv = 1.0 / elongation;
    std::array<double, 9> s{};
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) {
            const double nn = n[r] * n[col];
            s[3 * r + col] = nn + inv * ((r == col ? 1.0 : 0.0) - nn);
        }
    }
    return s;
}

} // namespace exaconstit::microstructure
