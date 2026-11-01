#pragma once

/**
 * @brief Function pointer type for UMAT subroutines.
 *
 * This typedef defines the signature for UMAT (User-defined Material) functions
 * that follow the Abaqus UMAT interface standard. The function signature includes
 * all the standard UMAT parameters for stress, state variables, material properties,
 * and various control parameters.
 *
 * @param stress Array of stress components (input/output)
 * @param statev Array of state variables (input/output)
 * @param ddsdde Material tangent stiffness matrix (output)
 * @param sse Specific strain energy (output)
 * @param spd Specific plastic dissipation (output)
 * @param scd Specific creep dissipation (output)
 * @param rpl Volumetric heat generation (output)
 * @param ddsdt Stress variation with temperature (output)
 * @param drplde Energy dissipation variation with strain (output)
 * @param drpldt Energy dissipation variation with temperature (output)
 * @param stran Total strain array (input)
 * @param dstran Strain increment array (input)
 * @param time Step time and total time array (input)
 * @param deltaTime Time increment for current step (input)
 * @param tempk Temperature at start of increment (input)
 * @param dtemp Temperature increment (input)
 * @param predef Predefined field variables (input)
 * @param dpred Predefined field variable increments (input)
 * @param cmname Material name (input)
 * @param ndi Number of direct stress components (input)
 * @param nshr Number of shear stress components (input)
 * @param ntens Total number of stress components (input)
 * @param nstatv Number of state variables (input)
 * @param props Material properties array (input)
 * @param nprops Number of material properties (input)
 * @param coords Coordinates of integration point (input)
 * @param drot Rotation increment matrix (input)
 * @param pnewdt Suggested new time increment (output)
 * @param celent Characteristic element length (input)
 * @param dfgrd0 Deformation gradient at start of increment (input)
 * @param dfgrd1 Deformation gradient at end of increment (input)
 * @param noel Element number (input)
 * @param npt Integration point number (input)
 * @param layer Layer number (input)
 * @param kspt Section point number (input)
 * @param kstep Step number (input)
 * @param kinc Increment number (input)
 */
using UmatFunction = void (*)(double* stress,
                              double* statev,
                              double* ddsdde,
                              double* sse,
                              double* spd,
                              double* scd,
                              double* rpl,
                              double* ddsdt,
                              double* drplde,
                              double* drpldt,
                              double* stran,
                              double* dstran,
                              double* time,
                              double* deltaTime,
                              double* tempk,
                              double* dtemp,
                              double* predef,
                              double* dpred,
                              char* cmname,
                              int* ndi,
                              int* nshr,
                              int* ntens,
                              int* nstatv,
                              double* props,
                              int* nprops,
                              double* coords,
                              double* drot,
                              double* pnewdt,
                              double* celent,
                              double* dfgrd0,
                              double* dfgrd1,
                              int* noel,
                              int* npt,
                              int* layer,
                              int* kspt,
                              int* kstep,
                              int* kinc);

/**
 * @brief Function pointer type for batched UMATs: one call for many material points.
 *
 * Selected with `interface = "batched"` in `[Materials.Model.UMAT]`. The arguments are the
 * UMAT arguments, in the same order, with one extra argument first: NPTS, the number of
 * material points in the call (all points of the material region on this MPI rank).
 *
 * Arrays get one more, last, dimension of size NPTS, and the UMAT scalars that differ from
 * point to point become arrays of length NPTS. In Fortran the declarations read
 * @code{.f}
 *       SUBROUTINE UMATBATCH(NPTS,STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
 *      1 RPL,DDSDDT,DRPLDE,DRPLDT,
 *      1 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
 *      1 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
 *      1 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
 *       INTEGER NPTS, NTENS, NSTATV, NPROPS, NOEL(NPTS), NPT(NPTS), ...
 *       REAL*8 STRESS(NTENS,NPTS), STATEV(NSTATV,NPTS), DDSDDE(NTENS,NTENS,NPTS)
 *       REAL*8 SSE(NPTS), ..., DFGRD0(3,3,NPTS), DROT(3,3,NPTS), PNEWDT(NPTS)
 *       REAL*8 PROPS(NPROPS), TIME(2), DTIME
 * @endcode
 * and point I's data is STRESS(:,I), DFGRD1(:,:,I), and so on. In C or C++ the same memory
 * is indexed as `stress[i * ntens + k]`, `dfgrd1[i * 9 + r + 3 * c]` (point i, 0-based).
 *
 * Per point (last dimension NPTS):
 *   stress(ntens), statev(nstatv), ddsdde(ntens, ntens), sse, spd, scd, rpl, ddsdt(ntens),
 *   drplde(ntens), drpldt, stran(ntens), dstran(ntens), temp, dtemp, predef(1), dpred(1),
 *   coords(3), drot(3, 3), pnewdt, celent, dfgrd0(3, 3), dfgrd1(3, 3), noel, npt
 * Shared by all points (as in UMAT):
 *   time(2), dtime, cmname, ndi, nshr, ntens, nstatv, props(nprops), nprops, layer, kspt,
 *   kstep, kinc
 *
 * What every argument holds, and which outputs ExaConstit uses, is listed in the UMAT section of
 * src/options.toml (the same for both interfaces).
 *
 * Each point reports its own PNEWDT (no reduction needed in user code); ExaConstit uses the
 * smallest value. A value below 1 fails the material update for the current Newton iteration,
 * as for standard UMATs: with automatic time stepping ([Time.Auto]) the step is retried with a
 * smaller time step, otherwise the step is reported as not converged.
 *
 * Memory: the arrays (every argument with an NPTS dimension, and PROPS) are in ExaConstit's
 * memory space for the run. On a GPU run (`rtmodel = "GPU"`) they are device arrays, so the
 * function must do its work on the GPU (CUDA, HIP, OpenMP target with is_device_ptr, ...);
 * ExaConstit waits for the device after the call returns. On CPU and OpenMP runs they are
 * ordinary host arrays and the function may use OpenMP over the points. The remaining
 * arguments (npts, ndi, nshr, ntens, nstatv, nprops, layer, kspt, kstep, kinc, dtime, time,
 * cmname) are always in host memory, so the code that launches the GPU work can read the
 * sizes directly.
 */
using UmatBatchFunction = void (*)(int* npts,
                                   double* stress,
                                   double* statev,
                                   double* ddsdde,
                                   double* sse,
                                   double* spd,
                                   double* scd,
                                   double* rpl,
                                   double* ddsdt,
                                   double* drplde,
                                   double* drpldt,
                                   double* stran,
                                   double* dstran,
                                   double* time,
                                   double* deltaTime,
                                   double* tempk,
                                   double* dtemp,
                                   double* predef,
                                   double* dpred,
                                   char* cmname,
                                   int* ndi,
                                   int* nshr,
                                   int* ntens,
                                   int* nstatv,
                                   double* props,
                                   int* nprops,
                                   double* coords,
                                   double* drot,
                                   double* pnewdt,
                                   double* celent,
                                   double* dfgrd0,
                                   double* dfgrd1,
                                   int* noel,
                                   int* npt,
                                   int* layer,
                                   int* kspt,
                                   int* kstep,
                                   int* kinc);

#ifdef __cplusplus
extern "C" {
#endif

// Default static UMAT (for testing/built-in materials)
// This will be linked from either umat.f or umat.cxx based on ENABLE_FORTRAN
void umat(double* stress,
          double* statev,
          double* ddsdde,
          double* sse,
          double* spd,
          double* scd,
          double* rpl,
          double* ddsdt,
          double* drplde,
          double* drpldt,
          double* stran,
          double* dstran,
          double* time,
          double* deltaTime,
          double* tempk,
          double* dtemp,
          double* predef,
          double* dpred,
          char* cmname,
          int* ndi,
          int* nshr,
          int* ntens,
          int* nstatv,
          double* props,
          int* nprops,
          double* coords,
          double* drot,
          double* pnewdt,
          double* celent,
          double* dfgrd0,
          double* dfgrd1,
          int* noel,
          int* npt,
          int* layer,
          int* kspt,
          int* kstep,
          int* kinc);

#ifdef __cplusplus
}
#endif

// #include <string>
// #include <memory>

// namespace exaconstit {

// /**
//  * @brief Universal UMAT resolver that handles both static and dynamic loading
//  *
//  * This class provides a unified interface for UMAT functions, supporting:
//  * - Built-in/static UMATs compiled into the binary
//  * - Dynamically loaded UMATs from shared libraries
//  * - Runtime symbol resolution with Fortran name mangling handling
//  */
// class UmatResolver {
// public:
//     /**
//      * @brief Get UMAT function from library path or built-in
//      *
//      * @param library_path Path to shared library (empty for built-in)
//      * @param function_name Name of the function to load (default: "umat_call")
//      * @return Function pointer to UMAT, or nullptr on failure
//      */
//     static UmatFunction GetUmat(const std::string& library_path = "",
//                                const std::string& function_name = "umat_call");

//     /**
//      * @brief Get diagnostic information about the last operation
//      */
//     static std::string GetLastError();

//     /**
//      * @brief Check if a library provides a valid UMAT
//      */
//     static bool ValidateLibrary(const std::string& library_path,
//                                const std::string& function_name = "umat_call");

// private:
//     static thread_local std::string last_error_;
// };

// } // namespace ExaConstit
