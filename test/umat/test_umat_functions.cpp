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
 * Point update (Abaqus conventions): sigma = sigma_old + C : DSTRAN with C the isotropic elastic
 * stiffness from PROPS = (E, nu) (engineering shear strains), DDSDDE = C. With rotate_stress =
 * true (the default) ExaConstit has already rotated sigma_old by DROT. With a third property
 * PROPS(3) = 1 the UMAT rotates sigma_old by DROT itself instead (for decks with
 * rotate_stress = false), so both ways must give the same results. STATEV(1..6) accumulate
 * DSTRAN and STATEV(7) holds det(DFGRD1), so the state variable path is exercised as well.
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
                  const int nprops,
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

    // Old stress, rotated by DROT here only when PROPS(3) = 1 (ExaConstit's rotate_stress = false)
    double s_rot[6] = {stress[0], stress[1], stress[2], stress[3], stress[4], stress[5]};
    if (nprops >= 3 && props[2] == 1.0) {
        // Full tensor from the Abaqus order (11, 22, 33, 12, 13, 23)
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
        s_rot[0] = rs[0][0];
        s_rot[1] = rs[1][1];
        s_rot[2] = rs[2][2];
        s_rot[3] = rs[0][1];
        s_rot[4] = rs[0][2];
        s_rot[5] = rs[1][2];
    }
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
                       int* nprops, double* /*coords*/, double* drot, double* /*pnewdt*/,
                       double* /*celent*/, double* /*dfgrd0*/, double* dfgrd1, int* /*noel*/,
                       int* /*npt*/, int* /*layer*/, int* /*kspt*/, int* /*kstep*/,
                       int* /*kinc*/) {
    ElasticPoint(stress, statev, ddsdde, dstran, props, *nprops, *nstatv, drot, dfgrd1);
}

/// Batched UMAT interface: NPTS points per call, point i's data at i * (size per point)
void test_umat_elastic_batch(int* npts, double* stress, double* statev, double* ddsdde,
                             double* /*sse*/, double* /*spd*/, double* /*scd*/, double* /*rpl*/,
                             double* /*ddsdt*/, double* /*drplde*/, double* /*drpldt*/,
                             double* /*stran*/, double* dstran, double* /*time*/,
                             double* /*dtime*/, double* /*temp*/, double* /*dtemp*/,
                             double* /*predef*/, double* /*dpred*/, char* /*cmname*/,
                             int* /*ndi*/, int* /*nshr*/, int* ntens, int* nstatv, double* props,
                             int* nprops, double* /*coords*/, double* drot,
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
                     *nprops,
                     ns,
                     drot + 9 * i,
                     dfgrd1 + 9 * i);
    }
}

} // extern "C"
