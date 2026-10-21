# option(BUILD_STATIC_LIBS "Build static libraries" OFF)
# option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

option(ENABLE_THROW "Enable throwing of exceptions" ON)

option(ENABLE_TESTS "Enable tests" OFF)

option(ENABLE_CUDA "Enable CUDA" OFF)

option(ENABLE_HIP  "Enable HIP" OFF)

option(ENABLE_OPENMP "Enable OpenMP" OFF)

# The microstructure generator (src/microstructure) needs Axom (spin BVHs, primal). The top-level
# CMakeLists.txt turns it off after the third-party setup if Axom was not found.
option(ENABLE_MICROSTRUCTURE "Build the microstructure generator (requires Axom)" ON)

# Disable FMA contraction (fusing a*b+c into one instruction) in the microstructure kernels. This
# removes one source of CPU/GPU round-off differences in grain labels; host and device math
# libraries can still differ, so it does not by itself make results bitwise identical. Costs some
# speed.
option(MICROSTRUCTURE_STRICT_FP "Build the microstructure kernels with -ffp-contract=off" OFF)

# Feature, multi-rank and comparison tests beyond the default CI suite (all microstructure tests,
# the parallel mesh input tests, ...). Off by default so CI stays fast; turn on to run them locally.
option(EXACONSTIT_ENABLE_EXTENSIVE_TESTS "Build and register the extensive (non-CI) test suite" OFF)
