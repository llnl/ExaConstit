/**
 * @file test_umat_functions.cpp
 * @brief UMAT library for the UMAT interface tests (test/umat/CMakeLists.txt)
 *
 * One material model, a hypoelastic isotropic solid, behind both UMAT interfaces:
 * - test_umat_elastic: the standard Abaqus UMAT (one material point per call);
 * - test_umat_elastic_batch: the batched UMAT (UmatBatchFunction in src/umats/userumat.h), a loop
 *   over the NPTS points that calls the same point update.
 * Run on the same problem, both interfaces must give the same results.
 *
 * Point update (Abaqus conventions): the old stress is rotated with DROT, then
 * sigma = R sigma_old R^T + C : DSTRAN with C the isotropic elastic stiffness from
 * PROPS = (E, nu) (engineering shear strains), DDSDDE = C. STATEV(1..6) accumulate DSTRAN and
 * STATEV(7) holds det(DFGRD1), so the state variable path is exercised as well.
 *
 * Like a user's UMAT library, it is built on its own, without ExaConstit or MFEM.
 */
#include <cmath>

namespace {

/// One material point of the hypoelastic model (see the file comment)
void ElasticPoint(double* stress,
                  double* statev,
                  double* ddsdde,
                  const double* dstran,
                  const double* props,
                  const int nstatv,
                  const double* drot,
                  const double* dfgrd1) {
    const double young = props[0];
    const double nu = props[1];
    const double lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double shear = young / (2.0 * (1.0 + nu));
    double c[36] = {0.0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            c[i + 6 * j] = lambda;
        }
        c[i + 6 * i] += 2.0 * shear;
        c[(i + 3) + 6 * (i + 3)] = shear;
    }

    // Rotate the old stress: full tensor from the Abaqus order (11, 22, 33, 12, 13, 23)
    const double s[3][3] = {{stress[0], stress[3], stress[4]},
                            {stress[3], stress[1], stress[5]},
                            {stress[4], stress[5], stress[2]}};
    double rs[3][3] = {{0.0}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    // DROT is stored column by column: R(i, k) = drot[i + 3 k]
                    rs[i][j] += drot[i + 3 * k] * s[k][l] * drot[j + 3 * l];
                }
            }
        }
    }
    const double s_rot[6] = {rs[0][0], rs[1][1], rs[2][2], rs[0][1], rs[0][2], rs[1][2]};
    for (int i = 0; i < 6; ++i) {
        double ds = 0.0;
        for (int j = 0; j < 6; ++j) {
            ds += c[i + 6 * j] * dstran[j];
        }
        stress[i] = s_rot[i] + ds;
    }
    for (int i = 0; i < 36; ++i) {
        ddsdde[i] = c[i];
    }

    if (nstatv >= 7) {
        for (int i = 0; i < 6; ++i) {
            statev[i] += dstran[i];
        }
        const double* f = dfgrd1;
        statev[6] = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[3] * (f[1] * f[8] - f[2] * f[7]) +
                    f[6] * (f[1] * f[5] - f[2] * f[4]);
    }
}

} // namespace

extern "C" {

/// Standard UMAT interface: one point per call
void test_umat_elastic(double* stress, double* statev, double* ddsdde, double* /*sse*/,
                       double* /*spd*/, double* /*scd*/, double* /*rpl*/, double* /*ddsdt*/,
                       double* /*drplde*/, double* /*drpldt*/, double* /*stran*/, double* dstran,
                       double* /*time*/, double* /*dtime*/, double* /*temp*/, double* /*dtemp*/,
                       double* /*predef*/, double* /*dpred*/, char* /*cmname*/, int* /*ndi*/,
                       int* /*nshr*/, int* /*ntens*/, int* nstatv, double* props,
                       int* /*nprops*/, double* /*coords*/, double* drot, double* /*pnewdt*/,
                       double* /*celent*/, double* /*dfgrd0*/, double* dfgrd1, int* /*noel*/,
                       int* /*npt*/, int* /*layer*/, int* /*kspt*/, int* /*kstep*/,
                       int* /*kinc*/) {
    ElasticPoint(stress, statev, ddsdde, dstran, props, *nstatv, drot, dfgrd1);
}

/// Batched UMAT interface: NPTS points per call, point i's data at i * (size per point)
void test_umat_elastic_batch(int* npts, double* stress, double* statev, double* ddsdde,
                             double* /*sse*/, double* /*spd*/, double* /*scd*/, double* /*rpl*/,
                             double* /*ddsdt*/, double* /*drplde*/, double* /*drpldt*/,
                             double* /*stran*/, double* dstran, double* /*time*/,
                             double* /*dtime*/, double* /*temp*/, double* /*dtemp*/,
                             double* /*predef*/, double* /*dpred*/, char* /*cmname*/,
                             int* /*ndi*/, int* /*nshr*/, int* ntens, int* nstatv, double* props,
                             int* /*nprops*/, double* /*coords*/, double* drot,
                             double* /*pnewdt*/, double* /*celent*/, double* /*dfgrd0*/,
                             double* dfgrd1, int* /*noel*/, int* /*npt*/, int* /*layer*/,
                             int* /*kspt*/, int* /*kstep*/, int* /*kinc*/) {
    const int n = *npts;
    const int nt = *ntens;
    const int ns = *nstatv;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; ++i) {
        ElasticPoint(stress + nt * i,
                     statev + ns * i,
                     ddsdde + nt * nt * i,
                     dstran + nt * i,
                     props,
                     ns,
                     drot + 9 * i,
                     dfgrd1 + 9 * i);
    }
}

} // extern "C"
