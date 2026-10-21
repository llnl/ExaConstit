# option(BUILD_STATIC_LIBS "Build static libraries" OFF)
# option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

option(ENABLE_THROW "Enable throwing of exceptions" ON)

option(ENABLE_TESTS "Enable tests" OFF)

option(ENABLE_CUDA "Enable CUDA" OFF)

option(ENABLE_HIP  "Enable HIP" OFF)

option(ENABLE_OPENMP "Enable OpenMP" OFF)

# Feature, multi-rank and comparison tests beyond the default CI suite (e.g. the parallel mesh
# input tests). Off by default so CI stays fast; turn on to run them locally.
option(EXACONSTIT_ENABLE_EXTENSIVE_TESTS "Build and register the extensive (non-CI) test suite" OFF)
