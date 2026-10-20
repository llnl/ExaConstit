#pragma once

#include "RAJA/config.hpp"

/**
 * @brief Access specifiers for member functions that contain device lambdas
 *
 * nvcc does not allow an extended __host__ __device__ lambda (e.g. a body passed
 * to mfem::forall or RAJA::forall) inside a private or protected member function.
 * Use these in place of `private:` / `protected:` only for the sections holding
 * such methods, and restore the normal specifier right after. In CUDA builds the
 * methods become public; every other build keeps the intended access.
 *
 * Keyed off RAJA_ENABLE_CUDA rather than __CUDACC__ so that a class has the same
 * definition in every translation unit of a build.
 */
#if defined(RAJA_ENABLE_CUDA)
#define EXACONSTIT_CUDA_PRIVATE public
#define EXACONSTIT_CUDA_PROTECTED public
#else
#define EXACONSTIT_CUDA_PRIVATE private
#define EXACONSTIT_CUDA_PROTECTED protected
#endif
