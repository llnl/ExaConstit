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
#   standard_full_selfrot          rotate_stress = false, the UMAT rotates the stress itself:
#                                  must equal standard_full (ExaConstit rotates)
#   standard_full_eulerian         strain_measure = "eulerian" (sanity checks only)
#   restart (write, read)          standard_full in two runs through a checkpoint (with Axom):
#                                  must equal standard_full; the UMAT checks its energies
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

#   <case>|<interface>|<function>|<integrator>|<rotate_stress>|<strain_measure>|<number of
#   properties>|<properties> (a third property of 1 makes the UMAT rotate the stress itself)
set(_umat_cases
    "standard_full|standard|test_umat_elastic|FULL|true|logarithmic|2|3000.0, 0.3"
    "batched_full|batched|test_umat_elastic_batch|FULL|true|logarithmic|2|3000.0, 0.3"
    "standard_bbar|standard|test_umat_elastic|BBAR|true|logarithmic|2|3000.0, 0.3"
    "batched_bbar|batched|test_umat_elastic_batch|BBAR|true|logarithmic|2|3000.0, 0.3"
    "standard_full_selfrot|standard|test_umat_elastic|FULL|false|logarithmic|3|3000.0, 0.3, 1.0"
    "standard_full_eulerian|standard|test_umat_elastic|FULL|true|eulerian|2|3000.0, 0.3")

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
    list(GET _umat_fields 4 ROTATE_STRESS)
    list(GET _umat_fields 5 STRAIN_MEASURE)
    list(GET _umat_fields 6 NUM_PROPS)
    list(GET _umat_fields 7 PROPS)
    set(T_FINAL 0.5)
    set(RESTART_BLOCK "")
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

# Restart: the standard_full deck runs 3 steps and writes a checkpoint, then a second run
# restarts from it and runs steps 4 and 5 into the same results files, which must then equal
# standard_full. The test UMAT stops the run (PNEWDT < 1) if the accumulated energies SSE, SPD,
# SCD it receives after the restart differ from the copy it keeps in its state variables, so
# this also checks that the energies are restored. Needs Axom (checkpoints).
if(ENABLE_AXOM)
    set(CASE_NAME restart)
    set(UMAT_INTERFACE standard)
    set(UMAT_FUNCTION test_umat_elastic)
    set(INTEG_MODEL FULL)
    set(ROTATE_STRESS true)
    set(STRAIN_MEASURE logarithmic)
    set(NUM_PROPS 2)
    set(PROPS "3000.0, 0.3")
    set(_umat_case_dir ${CMAKE_CURRENT_BINARY_DIR}/${CASE_NAME})
    file(MAKE_DIRECTORY ${_umat_case_dir})
    foreach(_umat_part write read)
        if(_umat_part STREQUAL "write")
            set(T_FINAL 0.3)
            set(RESTART_BLOCK "[Restart]\n    write = true\n    write_final = true")
        else()
            set(T_FINAL 0.5)
            set(RESTART_BLOCK "[Restart]\n    read = true")
        endif()
        configure_file(umat.toml.in ${_umat_case_dir}/${_umat_part}.toml.in @ONLY)
        file(GENERATE OUTPUT ${_umat_case_dir}/${_umat_part}.toml
                      INPUT  ${_umat_case_dir}/${_umat_part}.toml.in)
        blt_add_test(NAME              test_umat_interface_restart_${_umat_part}
                     COMMAND           $<TARGET_FILE:mechanics> -opt ${_umat_part}.toml
                     NUM_MPI_TASKS     1
                     WORKING_DIRECTORY ${_umat_case_dir})
    endforeach()
    # The restart run needs the checkpoint of the first run; both write the shared results
    add_test(NAME    test_umat_interface_restart_clean
             COMMAND ${CMAKE_COMMAND} -E rm -rf ${_umat_case_dir}/restart)
    set_tests_properties(test_umat_interface_restart_clean PROPERTIES
                         FIXTURES_SETUP umat_restart_clean)
    set_tests_properties(test_umat_interface_restart_write PROPERTIES
                         FIXTURES_REQUIRED "umat_interface_clean;umat_restart_clean"
                         FIXTURES_SETUP    umat_restart_written)
    set_tests_properties(test_umat_interface_restart_read PROPERTIES
                         FIXTURES_REQUIRED umat_restart_written
                         FIXTURES_SETUP    umat_interface_runs)
    set(_umat_check_restart "--restart")
else()
    set(_umat_check_restart "")
endif()

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
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_umat.py ${_umat_results}
                 ${_umat_check_restart})
set_tests_properties(test_umat_interface PROPERTIES FIXTURES_REQUIRED umat_interface_runs)

# The user-facing batched UMAT example (examples/umat_batched) builds on its own, as documented
add_test(NAME test_umat_batched_example_build
         COMMAND ${CMAKE_CTEST_COMMAND}
                 --build-and-test ${CMAKE_SOURCE_DIR}/examples/umat_batched
                                  ${CMAKE_CURRENT_BINARY_DIR}/example_build
                 --build-generator ${CMAKE_GENERATOR}
                 --build-options -DCMAKE_CXX_COMPILER=${EXACONSTIT_USER_CXX})
