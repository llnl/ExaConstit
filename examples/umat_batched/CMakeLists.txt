#------------------------------------------------------------------------------
# Example: a batched UMAT library (interface = "batched")
#
#   cmake -S . -B build
#   cmake --build build
#
# Builds libumat_batched (.so on Linux, .dylib on macOS) from umat_batched.cpp, and, when a
# Fortran compiler is found, libumat_batched_f from umat_batched.f90 (the same material in
# Fortran). A UMAT library needs nothing from ExaConstit: no headers, no find_package. List the
# library as library_path in the options file (see example_options.toml).
#
# OpenMP: both libraries use OpenMP over the points when CMake finds it (configure prints
# "Could NOT find OpenMP" otherwise and the library is serial; Apple clang needs libomp, e.g.
# -DOpenMP_ROOT=$(brew --prefix libomp)). For a GPU build, compile the point loop as GPU work
# instead (see the comments in umat_batched.cpp).
#------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)
project(umat_batched LANGUAGES CXX)

find_package(OpenMP COMPONENTS CXX)

add_library(umat_batched SHARED umat_batched.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(umat_batched PRIVATE OpenMP::OpenMP_CXX)
endif()

include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    find_package(OpenMP COMPONENTS Fortran)
    add_library(umat_batched_f SHARED umat_batched.f90)
    if(OpenMP_Fortran_FOUND)
        target_link_libraries(umat_batched_f PRIVATE OpenMP::OpenMP_Fortran)
    endif()
endif()
