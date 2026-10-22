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
 * The seed search stays fast for these tensors thanks to the projected-interval lower bound of
 * QuadricMetric.
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
    return {1.0 - a[0] * a[0], 1.0 - a[1] * a[1], 1.0 - a[2] * a[2],
            -a[1] * a[2],      -a[0] * a[2],      -a[0] * a[1]};
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

} // namespace exaconstit::microstructure
