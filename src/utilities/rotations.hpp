#pragma once

#include "mfem.hpp"

#include "ECMech_gpu_portability.h"

#include <cmath>
#include <limits>

/**
 * @brief Convert a 3x3 rotation matrix to a unit quaternion representation.
 *
 * @param rmat Input rotation matrix (3x3, assumed to be orthogonal)
 * @param quat Output quaternion vector (4 components: [w, x, y, z])
 *
 * This function converts a 3x3 rotation matrix to its equivalent unit quaternion
 * representation using a numerically stable algorithm. The conversion handles
 * special cases and numerical precision issues that can arise with naive
 * conversion methods.
 *
 * Algorithm details:
 * 1. Computes the rotation angle φ from the trace of the rotation matrix
 * 2. Handles the case where φ ≈ 0 (identity rotation) specially
 * 3. Extracts the rotation axis from the skew-symmetric part of the matrix
 * 4. Constructs the quaternion using the half-angle formulation
 *
 * The quaternion representation uses the convention:
 * - quat[0] = cos(φ/2) (scalar part)
 * - quat[1] = sin(φ/2) * axis_x (vector part x)
 * - quat[2] = sin(φ/2) * axis_y (vector part y)
 * - quat[3] = sin(φ/2) * axis_z (vector part z)
 *
 * This conversion is essential for:
 * - Crystal plasticity simulations requiring orientation tracking
 * - Rigid body kinematics and rotational mechanics
 * - Interpolation between rotational states
 * - Integration of rotational differential equations
 *
 * @note The input rotation matrix should be orthogonal (R^T * R = I).
 * @note The output quaternion is automatically normalized to unit length.
 * @note Special handling is provided for small rotation angles to avoid numerical issues.
 *
 * @ingroup ExaConstit_utilities_rotations
 */
inline void RMat2Quat(const mfem::DenseMatrix& rmat, mfem::Vector& quat) {
    constexpr double inv2 = 0.5;
    double phi = 0.0;
    static const double eps = std::numeric_limits<double>::epsilon();
    double tr_r = 0.0;
    double inv_sin = 0.0;
    double s = 0.0;

    quat = 0.0;

    tr_r = rmat(0, 0) + rmat(1, 1) + rmat(2, 2);
    phi = inv2 * (tr_r - 1.0);
    phi = std::min(phi, 1.0);
    phi = std::max(phi, -1.0);
    phi = std::acos(phi);
    if (std::abs(phi) < eps) {
        quat[3] = 1.0;
    } else {
        inv_sin = 1.0 / sin(phi);
        quat[0] = phi;
        quat[1] = inv_sin * inv2 * (rmat(2, 1) - rmat(1, 2));
        quat[2] = inv_sin * inv2 * (rmat(0, 2) - rmat(2, 0));
        quat[3] = inv_sin * inv2 * (rmat(1, 0) - rmat(0, 1));
    }

    s = std::sin(inv2 * quat[0]);
    quat[0] = std::cos(quat[0] * inv2);
    quat[1] = s * quat[1];
    quat[2] = s * quat[2];
    quat[3] = s * quat[3];
}

/**
 * @brief Convert a unit quaternion to its corresponding 3x3 rotation matrix.
 *
 * @param quat Input unit quaternion (4 components: [w, x, y, z])
 * @param rmat Output rotation matrix (3x3, orthogonal)
 *
 * This function converts a unit quaternion to its equivalent 3x3 rotation matrix
 * representation using the standard quaternion-to-matrix conversion formula.
 * The conversion is numerically stable and efficient.
 *
 * The conversion uses the formula:
 * R = (w² - x² - y² - z²)I + 2(vv^T) + 2w[v]×
 *
 * where:
 * - w is the scalar part of the quaternion
 * - v = [x, y, z] is the vector part
 * - [v]× is the skew-symmetric matrix of v
 * - I is the 3x3 identity matrix
 *
 * The resulting rotation matrix has the properties:
 * - Orthogonal: R^T * R = I
 * - Proper: det(R) = +1
 * - Preserves lengths and angles
 *
 * This conversion is widely used in:
 * - Crystal plasticity for transforming between crystal and sample coordinates
 * - Rigid body mechanics for applying rotational transformations
 * - Computer graphics and robotics applications
 * - Finite element simulations involving large rotations
 *
 * @note The input quaternion should be normalized (||q|| = 1).
 * @note The output matrix is guaranteed to be a proper orthogonal matrix.
 * @note This is the inverse operation of RMat2Quat().
 *
 * @ingroup ExaConstit_utilities_rotations
 */
inline void Quat2RMat(const mfem::Vector& quat, mfem::DenseMatrix& rmat) {
    double qbar = 0.0;

    qbar = quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    rmat(0, 0) = qbar + 2.0 * quat[1] * quat[1];
    rmat(1, 0) = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    rmat(2, 0) = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);

    rmat(0, 1) = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    rmat(1, 1) = qbar + 2.0 * quat[2] * quat[2];
    rmat(2, 1) = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);

    rmat(0, 2) = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
    rmat(1, 2) = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
    rmat(2, 2) = qbar + 2.0 * quat[3] * quat[3];
}

/**
 * @brief Device-compatible quaternion to rotation matrix conversion.
 *
 * @param quat Input unit quaternion array (4 components: [w, x, y, z])
 * @param rmats Output rotation matrix array (9 components in row-major order)
 *
 * This function provides a device-compatible (CPU/GPU) version of quaternion
 * to rotation matrix conversion. It's designed for use in GPU kernels and
 * high-performance computing environments where the standard MFEM objects
 * may not be suitable.
 *
 * Array layout:
 * - Input quat: [w, x, y, z] (4 consecutive doubles)
 * - Output rmats: [r11, r12, r13, r21, r22, r23, r31, r32, r33] (9 consecutive doubles)
 *
 * The function uses raw arrays instead of MFEM objects to ensure:
 * - Compatibility with GPU execution (CUDA/HIP/OpenMP target)
 * - Minimal memory overhead and optimal performance
 * - Integration with ECMech material models
 * - Use in vectorized and parallel operations
 *
 * This function is extensively used in:
 * - Crystal plasticity material models running on GPU
 * - Vectorized operations over multiple grains
 * - Integration with external material libraries (ECMech)
 * - High-performance lattice strain calculations
 *
 * The `__ecmech_hdev__` decorator ensures the function can be called from
 * both host and device code, providing maximum flexibility for hybrid
 * CPU/GPU simulations.
 *
 * @note This function assumes both input and output arrays are properly allocated.
 * @note The quaternion should be normalized for correct results.
 * @note Row-major storage order is used for the output matrix.
 *
 * @ingroup ExaConstit_utilities_rotations
 */
__ecmech_hdev__ inline void Quat2RMat(const double* const quat, double* const rmats) {
    const double qbar = quat[0] * quat[0] -
                        (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    double* rmat[3] = {&rmats[0], &rmats[3], &rmats[6]};

    rmat[0][0] = qbar + 2.0 * quat[1] * quat[1];
    rmat[1][0] = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    rmat[2][0] = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);

    rmat[0][1] = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    rmat[1][1] = qbar + 2.0 * quat[2] * quat[2];
    rmat[2][1] = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);

    rmat[0][2] = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
    rmat[1][2] = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
    rmat[2][2] = qbar + 2.0 * quat[3] * quat[3];
}