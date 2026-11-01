#pragma once

#include "mfem.hpp"
#include "mfem/linalg/kernels.hpp"

#include <cmath>

/**
 * @file kinematics_kernels.hpp
 * @brief Per-point 3x3 kinematics (deformation gradients, rotations, strains) that run on the
 * CPU or the GPU.
 *
 * Every routine works on one material point and on plain arrays, so it can be called inside an
 * mfem::forall loop on any device. Conventions used throughout:
 * - 3x3 tensors are stored column by column (column-major), the same layout MFEM's DenseMatrix,
 *   GradCalc and the Abaqus UMAT arrays (DFGRD0, DFGRD1, DROT) use: entry (i, j) is at
 *   index i + 3 j.
 * - ExaConstit's 6-component (Voigt) order is (11, 22, 33, 23, 13, 12); the Abaqus UMAT order is
 *   (11, 22, 33, 12, 13, 23). Strains passed to a UMAT use engineering shears (2 e_ij).
 *
 * These are the device versions of the host DenseMatrix routines in strain_measures.hpp; the
 * UMAT model (models/mechanics_umat.cpp) is their main user.
 *
 * @ingroup ExaConstit_utilities
 */
namespace exaconstit {
namespace kernel {

/**
 * @brief Determinant of a 3x3 tensor.
 * @param a tensor, column-major
 * @return det(a)
 */
MFEM_HOST_DEVICE inline double Det3(const double* a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[3] * (a[1] * a[8] - a[2] * a[7]) +
           a[6] * (a[1] * a[5] - a[2] * a[4]);
}

/**
 * @brief Inverse of a 3x3 tensor.
 * @param a tensor, column-major (must be invertible; deformation gradients have det > 0)
 * @param[out] inv a^-1, column-major (must not alias a)
 * @return det(a)
 */
MFEM_HOST_DEVICE inline double Inverse3(const double* a, double* inv) {
    const double det = Det3(a);
    const double c = 1.0 / det;
    inv[0] = c * (a[4] * a[8] - a[7] * a[5]);
    inv[1] = c * (a[7] * a[2] - a[1] * a[8]);
    inv[2] = c * (a[1] * a[5] - a[4] * a[2]);
    inv[3] = c * (a[6] * a[5] - a[3] * a[8]);
    inv[4] = c * (a[0] * a[8] - a[6] * a[2]);
    inv[5] = c * (a[3] * a[2] - a[0] * a[5]);
    inv[6] = c * (a[3] * a[7] - a[6] * a[4]);
    inv[7] = c * (a[6] * a[1] - a[0] * a[7]);
    inv[8] = c * (a[0] * a[4] - a[3] * a[1]);
    return det;
}

/**
 * @brief Product of two 3x3 tensors, c = a b.
 * @param a left tensor, column-major
 * @param b right tensor, column-major
 * @param[out] c a b, column-major (must not alias a or b)
 */
MFEM_HOST_DEVICE inline void Mult3(const double* a, const double* b, double* c) {
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            c[i + 3 * j] = a[i] * b[3 * j] + a[i + 3] * b[1 + 3 * j] + a[i + 6] * b[2 + 3 * j];
        }
    }
}

/**
 * @brief Incremental deformation gradient over a time step, F_incr = F_end F_beg^-1.
 *
 * @param f_beg deformation gradient at the beginning of the step (reference to x_beg)
 * @param f_end deformation gradient at the end of the step (reference to x_end)
 * @param[out] f_incr d x_end / d x_beg, column-major
 *
 * @details Both deformation gradients are measured from the same reference configuration, so
 * their "ratio" maps the beginning-of-step configuration onto the end-of-step one.
 */
MFEM_HOST_DEVICE inline void
IncrementalDefGrad(const double* f_beg, const double* f_end, double* f_incr) {
    double f_beg_inv[9];
    Inverse3(f_beg, f_beg_inv);
    Mult3(f_end, f_beg_inv, f_incr);
}

/**
 * @brief Rotation part R of the polar decomposition F = R U.
 *
 * @param f deformation gradient (det > 0), column-major
 * @param[out] r rotation tensor, column-major (must not alias f)
 *
 * @details The right stretch U is the square root of C = F^T F. With the eigenvalues lambda_k and
 * unit eigenvectors n_k of C, U^-1 = sum_k lambda_k^(-1/2) n_k (x) n_k and R = F U^-1. The
 * eigen-decomposition is MFEM's closed-form 3x3 symmetric solver
 * (mfem::kernels::CalcEigenvalues<3>), which also handles repeated eigenvalues, so the result
 * does not depend on an iteration tolerance.
 */
MFEM_HOST_DEVICE inline void PolarRotation(const double* f, double* r) {
    // Right Cauchy-Green tensor C = F^T F (symmetric)
    double c[9];
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            c[i + 3 * j] = f[3 * i] * f[3 * j] + f[1 + 3 * i] * f[1 + 3 * j] +
                           f[2 + 3 * i] * f[2 + 3 * j];
        }
    }
    double lambda[3];
    double vec[9]; // eigenvector k in vec[3 k .. 3 k + 2]
    mfem::kernels::CalcEigenvalues<3>(c, lambda, vec);

    // U^-1 = sum_k lambda_k^(-1/2) n_k (x) n_k
    double u_inv[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
        const double s = 1.0 / sqrt(lambda[k]);
        const double* n = vec + 3 * k;
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                u_inv[i + 3 * j] += s * n[i] * n[j];
            }
        }
    }
    Mult3(f, u_inv, r);
}

/**
 * @brief Eulerian (Almansi) strain e = 1/2 (I - F^-T F^-1).
 *
 * @param f deformation gradient (det > 0), column-major
 * @param[out] e symmetric strain tensor, column-major
 *
 * @details Same measure as CalcEulerianStrain in strain_measures.hpp. It is the strain the UMAT
 * model passes to UMATs (STRAN from F_beg, DSTRAN from F_incr), as the original
 * implementation did.
 */
MFEM_HOST_DEVICE inline void EulerianStrain(const double* f, double* e) {
    double f_inv[9];
    Inverse3(f, f_inv);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            // (F^-T F^-1)(i, j) = sum_k F^-1(k, i) F^-1(k, j)
            const double b_inv = f_inv[3 * i] * f_inv[3 * j] + f_inv[1 + 3 * i] * f_inv[1 + 3 * j] +
                                 f_inv[2 + 3 * i] * f_inv[2 + 3 * j];
            e[i + 3 * j] = ((i == j) ? 0.5 : 0.0) - 0.5 * b_inv;
        }
    }
}

/**
 * @brief Symmetric strain tensor to the Abaqus 6-vector with engineering shears,
 * (e11, e22, e33, 2 e12, 2 e13, 2 e23).
 * @param e symmetric strain tensor, column-major
 * @param[out] v Abaqus-ordered strain vector
 */
MFEM_HOST_DEVICE inline void StrainToAbaqusVoigt(const double* e, double* v) {
    v[0] = e[0];
    v[1] = e[4];
    v[2] = e[8];
    v[3] = 2.0 * e[3];
    v[4] = 2.0 * e[6];
    v[5] = 2.0 * e[7];
}

/**
 * @brief Stress 6-vector from ExaConstit order (11, 22, 33, 23, 13, 12) to the Abaqus order
 * (11, 22, 33, 12, 13, 23).
 * @param s_exa stress in ExaConstit order
 * @param[out] s_abq stress in Abaqus order (must not alias s_exa)
 */
MFEM_HOST_DEVICE inline void StressToAbaqusVoigt(const double* s_exa, double* s_abq) {
    s_abq[0] = s_exa[0];
    s_abq[1] = s_exa[1];
    s_abq[2] = s_exa[2];
    s_abq[3] = s_exa[5];
    s_abq[4] = s_exa[4];
    s_abq[5] = s_exa[3];
}

/**
 * @brief Stress 6-vector from the Abaqus order back to ExaConstit order. Swapping the 4th and
 * 6th components is its own inverse, so this is the same operation as StressToAbaqusVoigt.
 * @param s_abq stress in Abaqus order
 * @param[out] s_exa stress in ExaConstit order (must not alias s_abq)
 */
MFEM_HOST_DEVICE inline void StressFromAbaqusVoigt(const double* s_abq, double* s_exa) {
    StressToAbaqusVoigt(s_abq, s_exa);
}

/**
 * @brief Convert a UMAT material tangent (DDSDDE) into the tangent ExaConstit's integrators use.
 *
 * @param dt time step
 * @param[in,out] ddsdde 6x6 tangent, 36 values
 *
 * @details Two changes:
 * - component order: rows and columns 4 and 6 are swapped, (12 <-> 23), to go from the Abaqus
 *   order to ExaConstit's;
 * - rate form: DDSDDE is d sigma / d (strain increment); ExaConstit's tangent is with respect to
 *   the deformation rate, d sigma / d D = DDSDDE dt, since strain increment = D dt.
 */
MFEM_HOST_DEVICE inline void AbaqusTangentToExaConstit(const double dt, double* ddsdde) {
    for (int i = 0; i < 6; ++i) {
        const double tmp = ddsdde[6 * i + 3];
        ddsdde[6 * i + 3] = ddsdde[6 * i + 5];
        ddsdde[6 * i + 5] = tmp;
    }
    for (int i = 0; i < 6; ++i) {
        const double tmp = ddsdde[6 * 3 + i];
        ddsdde[6 * 3 + i] = ddsdde[6 * 5 + i];
        ddsdde[6 * 5 + i] = tmp;
    }
    for (int i = 0; i < 36; ++i) {
        ddsdde[i] *= dt;
    }
}

} // namespace kernel
} // namespace exaconstit
