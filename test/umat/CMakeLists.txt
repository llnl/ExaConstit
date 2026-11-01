####################################
# UMAT interfaces (standard and batched)
####################################
# Included from test/CMakeLists.txt only with EXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON.
#
# libtest_umat (test_umat_functions.cpp) is a UMAT library as a user would write one: the same
# hypoelastic material behind the standard UMAT interface and the batched one. The mechanics
# driver runs one deck per case, then check_umat.py compares the runs:
#   standard_full / batched_full   one region, default integrator, 1 rank
#   standard_bbar / batched_bbar   one region, B-bar integrator, 1 rank
#   two_regions_full / _bbar       two regions of the same material (region 1 batched, region 2
#                                  standard), 2 ranks: must equal the one-region runs
#     ctest -R umat_interface --output-on-failure

# A plain shared library: a UMAT needs nothing from ExaConstit or MFEM. Part of the default
# build target, so it exists when the tests run.
add_library(test_umat MODULE test_umat_functions.cpp)
set_target_properties(test_umat PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set(UMAT_TEST_LIBRARY "$<TARGET_FILE:test_umat>")
set(EXACONSTIT_TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/test/data)
set(_umat_results ${CMAKE_CURRENT_BINARY_DIR}/results)

#   <case>|<interface>|<function>|<integrator>
set(_umat_cases
    "standard_full|standard|test_umat_elastic|FULL"
    "batched_full|batched|test_umat_elastic_batch|FULL"
    "standard_bbar|standard|test_umat_elastic|BBAR"
    "batched_bbar|batched|test_umat_elastic_batch|BBAR")

# Volume average files are appended to, so every run starts from an empty results folder
add_test(NAME    test_umat_interface_clean
         COMMAND ${CMAKE_COMMAND} -E rm -rf ${_umat_results})
set_tests_properties(test_umat_interface_clean PROPERTIES FIXTURES_SETUP umat_interface_clean)

set(_umat_run_tests)
foreach(_umat_case ${_umat_cases})
    string(REPLACE "|" ";" _umat_fields "${_umat_case}")
    list(GET _umat_fields 0 CASE_NAME)
    list(GET _umat_fields 1 UMAT_INTERFACE)
    list(GET _umat_fields 2 UMAT_FUNCTION)
    list(GET _umat_fields 3 INTEG_MODEL)
    # configure_file fills the @...@ fields; file(GENERATE) then replaces the library generator
    # expression with the library's path
    set(_umat_case_dir ${CMAKE_CURRENT_BINARY_DIR}/${CASE_NAME})
    file(MAKE_DIRECTORY ${_umat_case_dir})
    configure_file(umat.toml.in ${_umat_case_dir}/${CASE_NAME}.toml.in @ONLY)
    file(GENERATE OUTPUT ${_umat_case_dir}/${CASE_NAME}.toml
                  INPUT  ${_umat_case_dir}/${CASE_NAME}.toml.in)
    # Each case runs in its own directory (per-rank log files) and writes its results into the
    # shared results folder through a relative path
    blt_add_test(NAME              test_umat_interface_${CASE_NAME}
                 COMMAND           $<TARGET_FILE:mechanics> -opt ${CASE_NAME}.toml
                 NUM_MPI_TASKS     1
                 WORKING_DIRECTORY ${_umat_case_dir})
    list(APPEND _umat_run_tests test_umat_interface_${CASE_NAME})
endforeach()

# Two regions of the same material, region 1 batched and region 2 standard, on 2 ranks; they
# must equal the one-region runs with the same integrator
#   <case>|<integrator>
set(_umat_np 2)
if(_umat_np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
    set(_umat_np 1)
endif()
set(_umat_region_cases
    "two_regions_full|FULL"
    "two_regions_bbar|BBAR")
foreach(_umat_case ${_umat_region_cases})
    string(REPLACE "|" ";" _umat_fields "${_umat_case}")
    list(GET _umat_fields 0 CASE_NAME)
    list(GET _umat_fields 1 INTEG_MODEL)
    set(_umat_case_dir ${CMAKE_CURRENT_BINARY_DIR}/${CASE_NAME})
    file(MAKE_DIRECTORY ${_umat_case_dir})
    configure_file(umat_two_regions.toml.in ${_umat_case_dir}/${CASE_NAME}.toml.in @ONLY)
    file(GENERATE OUTPUT ${_umat_case_dir}/${CASE_NAME}.toml
                  INPUT  ${_umat_case_dir}/${CASE_NAME}.toml.in)
    blt_add_test(NAME              test_umat_interface_${CASE_NAME}
                 COMMAND           $<TARGET_FILE:mechanics> -opt ${CASE_NAME}.toml
                 NUM_MPI_TASKS     ${_umat_np}
                 WORKING_DIRECTORY ${_umat_case_dir})
    list(APPEND _umat_run_tests test_umat_interface_${CASE_NAME})
endforeach()

foreach(_umat_test ${_umat_run_tests})
    set_tests_properties(${_umat_test} PROPERTIES
                         FIXTURES_REQUIRED umat_interface_clean
                         FIXTURES_SETUP    umat_interface_runs)
endforeach()

add_test(NAME    test_umat_interface
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_umat.py ${_umat_results})
set_tests_properties(test_umat_interface PROPERTIES FIXTURES_REQUIRED umat_interface_runs)

# The user-facing batched UMAT example (examples/umat_batched) builds on its own, as documented
add_test(NAME test_umat_batched_example_build
         COMMAND ${CMAKE_CTEST_COMMAND}
                 --build-and-test ${CMAKE_SOURCE_DIR}/examples/umat_batched
                                  ${CMAKE_CURRENT_BINARY_DIR}/example_build
                 --build-generator ${CMAKE_GENERATOR}
                 --build-options -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
