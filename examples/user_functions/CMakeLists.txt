#------------------------------------------------------------------------------
# Example: building a library of ExaConstit user functions
#
#   cmake -S . -B build -DExaConstit_DIR=<ExaConstit install>/lib/cmake/exaconstit
#   cmake --build build
#
# This builds libmy_exaconstit_functions (.so on Linux, .dylib on macOS). List it as
# library_path in the options file (see example_options.toml). Nothing else is needed:
# find_package(ExaConstit) brings the settings of ExaConstit and of every library it was
# built with (MFEM, RAJA, ExaCMech, SNLS, Axom, ...). Use the same C++ compiler ExaConstit
# was built with (a warning says so if it differs); for an MPI build that is usually the
# MPI wrapper, e.g. -DCMAKE_CXX_COMPILER=mpicxx.
#------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)
project(my_exaconstit_functions LANGUAGES CXX)

find_package(ExaConstit REQUIRED)

exaconstit_add_user_library(NAME my_exaconstit_functions SOURCES my_exaconstit_functions.cpp)
