#------------------------------------------------------------------------------
# Microstructure generator tests
#
# Each module has its own test_<module>.cmake next to this file; they are all included below,
# so adding a module's tests never edits a shared test file. In a .cmake file use:
#
#   ms_add_gtest(NAME          test_mesh
#                SOURCES       test_mesh.cpp
#                DEPENDS_ON    ms_mesh            # only the modules under test
#                NUM_MPI_TASKS 1 2 3 4)           # optional; omit for a serial test
#
# With NUM_MPI_TASKS one test test_mesh_np<N> is registered per rank count up to
# EXACONSTIT_TEST_MAX_MPI_TASKS; without it a single test named NAME runs the executable directly.
# The executables link gtest but not gtest_main: write a main() that initializes MPI.
#
# Test gating: these tests belong to the extensive (non-CI) suite. When
# EXACONSTIT_ENABLE_EXTENSIVE_TESTS is OFF (cmake/ExaConstitOptions.cmake, the default) this file
# returns immediately: no test executable is built and nothing is registered with ctest, so the
# default build and the default `ctest` suite are unchanged. The gate lives here rather than
# around add_subdirectory in the top-level CMakeLists.txt so that all microstructure test logic
# stays in one file.
#------------------------------------------------------------------------------

if(NOT EXACONSTIT_ENABLE_EXTENSIVE_TESTS)
    message(STATUS "Microstructure tests are not built "
                   "(set EXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON to build and run them)")
    return()
endif()

# Defined in test/CMakeLists.txt (before the restart and parallel mesh tests); the fallback only
# matters if that changes
if(NOT DEFINED EXACONSTIT_TEST_MAX_MPI_TASKS)
    set(EXACONSTIT_TEST_MAX_MPI_TASKS 4)
endif()

#------------------------------------------------------------------------------
# ms_add_gtest(NAME <exe> SOURCES <src...> [DEPENDS_ON <targets...>] [NUM_MPI_TASKS <n...>])
#
# Builds the gtest executable <exe> into TEST_OUTPUT_DIR, linked against ms_core, the listed
# ms_* modules, MICROSTRUCTURE_DEPENDS (direct, so BLT compiles GPU sources as device code) and
# gtest, and registers its ctest entries (label "microstructure"). MPI variants above
# EXACONSTIT_TEST_MAX_MPI_TASKS are skipped.
#------------------------------------------------------------------------------
function(ms_add_gtest)
    # Not the usual arg_ prefix: the BLT macros called below parse into arg_* in this scope
    cmake_parse_arguments(_ms "" "NAME" "SOURCES;DEPENDS_ON;NUM_MPI_TASKS" ${ARGN})

    if(NOT _ms_NAME OR NOT _ms_SOURCES)
        message(FATAL_ERROR "ms_add_gtest requires NAME and SOURCES")
    endif()

    blt_add_executable(NAME       ${_ms_NAME}
                       SOURCES    ${_ms_SOURCES}
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON ms_core ${_ms_DEPENDS_ON} ${MICROSTRUCTURE_DEPENDS} gtest)

    if(_ms_NUM_MPI_TASKS)
        # One ctest entry per rank count; blt_add_test wraps the command in mpirun -np <N>
        foreach(_np ${_ms_NUM_MPI_TASKS})
            if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
                blt_add_test(NAME          ${_ms_NAME}_np${_np}
                             COMMAND       ${_ms_NAME}
                             NUM_MPI_TASKS ${_np})
                set_tests_properties(${_ms_NAME}_np${_np} PROPERTIES LABELS "microstructure")
            endif()
        endforeach()
    else()
        # Serial test: run the executable directly
        blt_add_test(NAME    ${_ms_NAME}
                     COMMAND ${_ms_NAME})
        set_tests_properties(${_ms_NAME} PROPERTIES LABELS "microstructure")
    endif()
endfunction()

#------------------------------------------------------------------------------
# Include every per-module test file, in sorted order so the configure output is reproducible.
# CONFIGURE_DEPENDS re-runs the glob at build time, so a new test_<module>.cmake is picked up.
#------------------------------------------------------------------------------
file(GLOB MS_TEST_CMAKE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cmake)
list(SORT MS_TEST_CMAKE_FILES)
foreach(_ms_test_file ${MS_TEST_CMAKE_FILES})
    include(${_ms_test_file})
endforeach()
