// Example batched UMAT (interface = "batched"), in C++.
//
// A batched UMAT receives all material points of a region on an MPI rank in one call. The
// arguments are those of the Abaqus UMAT, in the same order, with NPTS added first; arrays get
// one more, last, dimension NPTS, and the per-point scalars (SSE, PNEWDT, CELENT, NOEL, ...)
// become arrays of length NPTS. Point i's data starts at i times the per-point size:
//     stress + i * ntens        statev + i * nstatv       ddsdde + i * ntens * ntens
//     dstran + i * ntens        dfgrd1 + i * 9            drot + i * 9
//     pnewdt + i                celent + i ...
// Shared by all points: time, dtime, cmname, ndi, nshr, ntens, nstatv, props, nprops, layer,
// kspt, kstep, kinc. The layout is described in ExaConstit's src/umats/userumat.h, and what
// every argument holds (and which outputs ExaConstit uses) in the UMAT section of
// src/options.toml.
//
// The material here is a hypoelastic isotropic solid: the elastic stiffness C (PROPS = Young's
// modulus, Poisson's ratio) times DSTRAN is added to the old stress; DDSDDE = C. As in Abaqus,
// ExaConstit has already rotated the old STRESS (and STRAN) by DROT (rotate_stress = true, the
// default), so the UMAT does no rotation itself; an anisotropic UMAT that handles rotations
// itself sets rotate_stress = false. Writing the update for one point in a function and looping
// over the points is all it takes to turn an existing UMAT into a batched one.
//
// CPU and OpenMP runs: the arrays are ordinary CPU arrays, and the loop below runs over the
// points with OpenMP when the library is compiled with OpenMP.
//
// GPU runs (rtmodel = "GPU"): the arrays are GPU arrays (the sizes ntens, nstatv, npts, ... are
// still CPU memory), so the loop must become GPU work. For example with HIP or CUDA:
//     __global__ void kernel(int n, double* stress, ...) {
//         const int i = blockIdx.x * blockDim.x + threadIdx.x;
//         if (i < n) { ElasticPoint(...point i's data...); }    // ElasticPoint __device__
//     }
//     extern "C" void umat_batched(int* npts, double* stress, ...) {
//         kernel<<<(*npts + 127) / 128, 128>>>(*npts, stress, ...);
//     }
// or with OpenMP offload: #pragma omp target teams distribute parallel for is_device_ptr(...).
// ExaConstit waits for the GPU after the call returns.

namespace {

/// Update of one material point (Abaqus conventions: stress order 11, 22, 33, 12, 13, 23,
/// engineering shear strains)
void ElasticPoint(double* stress,
                  double* ddsdde,
                  const double* dstran,
                  const double* props,
                  double* pnewdt) {
    const double young = props[0];
    const double nu = props[1];
    const double lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double shear = young / (2.0 * (1.0 + nu));

    // Elastic stiffness, stored column by column like the Fortran DDSDDE(NTENS, NTENS)
    for (int k = 0; k < 36; ++k) {
        ddsdde[k] = 0.0;
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ddsdde[i + 6 * j] = lambda;
        }
        ddsdde[i + 6 * i] += 2.0 * shear;
        ddsdde[(i + 3) + 6 * (i + 3)] = shear;
    }

    // Add C : DSTRAN
    for (int i = 0; i < 6; ++i) {
        double ds = 0.0;
        for (int j = 0; j < 6; ++j) {
            ds += ddsdde[i + 6 * j] * dstran[j];
        }
        stress[i] += ds;
    }

    // An elastic update never needs a smaller time step; a UMAT that does sets PNEWDT < 1
    (void)pnewdt;
}

} // namespace

// Arguments this material does not use are left unnamed (name in a comment)
extern "C" void umat_batched(int* npts, double* stress, double* /*statev*/, double* ddsdde,
                             double* /*sse*/, double* /*spd*/, double* /*scd*/, double* /*rpl*/,
                             double* /*ddsdt*/, double* /*drplde*/, double* /*drpldt*/,
                             double* /*stran*/, double* dstran, double* /*time*/,
                             double* /*dtime*/, double* /*temp*/, double* /*dtemp*/,
                             double* /*predef*/, double* /*dpred*/, char* /*cmname*/,
                             int* /*ndi*/, int* /*nshr*/, int* ntens, int* /*nstatv*/,
                             double* props, int* /*nprops*/, double* /*coords*/, double* /*drot*/,
                             double* pnewdt, double* /*celent*/, double* /*dfgrd0*/,
                             double* /*dfgrd1*/, int* /*noel*/, int* /*npt*/, int* /*layer*/,
                             int* /*kspt*/, int* /*kstep*/, int* /*kinc*/) {
    const int n = *npts;
    const int nt = *ntens;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; ++i) {
        ElasticPoint(stress + nt * i,
                     ddsdde + nt * nt * i,
                     dstran + nt * i,
                     props,
                     pnewdt + i);
    }
}
