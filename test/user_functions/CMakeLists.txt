####################################
# User functions loaded from shared libraries
####################################
# Included from test/CMakeLists.txt only with EXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON.
#
# libtest_user_functions (test_user_functions.cpp) plays the part of a user's library: it is
# built against the ExaConstit and MFEM headers but NOT linked with them. The functions it calls
# are found in the program that loads it (mechanics, or test_state_variable_functions), which
# exports its symbols (ENABLE_EXPORTS). Linking ExaConstit or MFEM into the library would give
# it its own copy of their global data (for example MFEM's memory manager), which is wrong.
#     ctest -R user_functions --output-on-failure

# Built exactly as a user would build theirs (cmake/ExaConstitUserLibrary.cmake): ExaConstit's
# headers and definitions and those of its libraries, no ExaConstit or MFEM link.
exaconstit_add_user_library(NAME test_user_functions SOURCES test_user_functions.cpp)
set_target_properties(test_user_functions PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ------------------------------------------------------------------------------------------------
# Input decks: one per case, from user_functions.toml.in. Only the UMAT region's state variable
# functions (and library) change between cases.
#   <case>|<library of the UMAT region's functions>|<UMAT register_function>
# ------------------------------------------------------------------------------------------------
set(_uf_library "$<TARGET_FILE:test_user_functions>")
set(_uf_cases
    "success|${_uf_library}|RegisterUmatStateVariables"
    "wrong_arguments|${_uf_library}|RegisterWrongArguments"
    "extern_c|${_uf_library}|RegisterExternC"
    "missing_library|libtest_user_functions_missing${CMAKE_SHARED_LIBRARY_SUFFIX}|RegisterUmatStateVariables"
    "overlap|${_uf_library}|RegisterOverlappingStateVariables"
    "out_of_range|${_uf_library}|RegisterOutOfRangeStateVariables")

set(EXACONSTIT_TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/test/data)
set(USER_FUNCTION_LIBRARY ${_uf_library})
foreach(_uf_case ${_uf_cases})
    string(REPLACE "|" ";" _uf_fields "${_uf_case}")
    list(GET _uf_fields 0 CASE_NAME)
    list(GET _uf_fields 1 UMAT_STATE_VAR_LIBRARY)
    list(GET _uf_fields 2 UMAT_REGISTER_FUNCTION)
    # configure_file fills the @...@ fields; file(GENERATE) then replaces the library
    # generator expression with the library's path. Each case runs in its own directory, so
    # the per-rank log files of the runs do not collide.
    set(_uf_case_dir ${CMAKE_CURRENT_BINARY_DIR}/${CASE_NAME})
    file(MAKE_DIRECTORY ${_uf_case_dir})
    configure_file(user_functions.toml.in ${_uf_case_dir}/${CASE_NAME}.toml.in @ONLY)
    file(GENERATE OUTPUT ${_uf_case_dir}/${CASE_NAME}.toml
                  INPUT  ${_uf_case_dir}/${CASE_NAME}.toml.in)
endforeach()

# ------------------------------------------------------------------------------------------------
# Success: the register and initialize functions take effect (in-process check)
# ------------------------------------------------------------------------------------------------
blt_add_executable(NAME       test_state_variable_functions
                   SOURCES    test_state_variable_functions.cpp
                   OUTPUT_DIR ${TEST_OUTPUT_DIR}
                   DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS} gtest)
# The user library calls back into this executable, as it does into mechanics, and the decks'
# UMAT region uses the built-in UMAT, which is found by symbol name, so it is added here as in
# mechanics
target_sources(test_state_variable_functions PRIVATE $<TARGET_OBJECTS:exaconstit_builtin_umat>)
set_target_properties(test_state_variable_functions PROPERTIES ENABLE_EXPORTS ON)
add_dependencies(test_state_variable_functions test_user_functions)
foreach(_np 1 2)
    if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
        blt_add_test(NAME              test_user_functions_state_vars_np${_np}
                     COMMAND           test_state_variable_functions
                                       ${CMAKE_CURRENT_BINARY_DIR}/success/success.toml
                     NUM_MPI_TASKS     ${_np}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/success)
        set_tests_properties(test_user_functions_state_vars_np${_np} PROPERTIES
                             RESOURCE_LOCK user_functions_success)
    endif()
endforeach()

# ------------------------------------------------------------------------------------------------
# Failures: the real mechanics driver must stop with a clear message. The runs end with
# MPI_Abort, so each test passes on the message (PASS_REGULAR_EXPRESSION ignores the exit code).
# The signature and library checks happen right after the options are read
# (exaconstit::CheckUserFunctions); the layout checks when the register_function runs. They
# need test_user_functions to be built (it is part of the default build target).
#   <case>|<expected message>
# ------------------------------------------------------------------------------------------------
set(_uf_failures
    "wrong_arguments|'RegisterWrongArguments' was not found in .* with the required signature"
    "extern_c|A function of that name exists but is declared extern"
    "missing_library|Could not open the library of the register_function of material"
    "overlap|'porosity' of region .* overlaps 'damage'"
    "out_of_range|does not fit in the region's 10 state variables")
foreach(_uf_failure ${_uf_failures})
    string(REPLACE "|" ";" _uf_fields "${_uf_failure}")
    list(GET _uf_fields 0 _uf_case)
    list(GET _uf_fields 1 _uf_message)
    blt_add_test(NAME              test_user_functions_${_uf_case}
                 COMMAND           $<TARGET_FILE:mechanics> -opt ${_uf_case}.toml
                 NUM_MPI_TASKS     1
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_uf_case})
    set_tests_properties(test_user_functions_${_uf_case} PROPERTIES
                         PASS_REGULAR_EXPRESSION "${_uf_message}")
endforeach()

# ------------------------------------------------------------------------------------------------
# The user-facing example (examples/user_functions) builds against the ExaConstit CMake package,
# exactly as documented for users (find_package(ExaConstit) + exaconstit_add_user_library), here
# with the build-tree package (a few seconds).
# ------------------------------------------------------------------------------------------------
add_test(NAME test_user_functions_example_build
         COMMAND ${CMAKE_CTEST_COMMAND}
                 --build-and-test ${CMAKE_SOURCE_DIR}/examples/user_functions
                                  ${CMAKE_CURRENT_BINARY_DIR}/example_build
                 --build-generator ${CMAKE_GENERATOR}
                 --build-options -DExaConstit_DIR=${PROJECT_BINARY_DIR}
                                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

# ------------------------------------------------------------------------------------------------
# Post-processing user functions ([[PostProcessing.user_functions]]): a short two-material
# ExaCMech run (3 steps, 2 ranks) per visualization field location, then
# check_postprocessing.py compares the outputs with exact answers (see postprocessing.toml.in).
# ------------------------------------------------------------------------------------------------
foreach(FIELD_LOCATION zone quadrature)
    set(_uf_pp_dir ${CMAKE_CURRENT_BINARY_DIR}/postprocessing_${FIELD_LOCATION})
    file(MAKE_DIRECTORY ${_uf_pp_dir})
    configure_file(postprocessing.toml.in ${_uf_pp_dir}/postprocessing.toml.in @ONLY)
    file(GENERATE OUTPUT ${_uf_pp_dir}/postprocessing.toml
                  INPUT  ${_uf_pp_dir}/postprocessing.toml.in)

    set(_uf_pp_np 2)
    if(_uf_pp_np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
        set(_uf_pp_np 1)
    endif()
    # Volume average files are appended to, so every run starts from an empty results folder
    add_test(NAME    test_user_functions_postprocessing_${FIELD_LOCATION}_clean
             COMMAND ${CMAKE_COMMAND} -E rm -rf ${_uf_pp_dir}/results)
    set_tests_properties(test_user_functions_postprocessing_${FIELD_LOCATION}_clean PROPERTIES
                         FIXTURES_SETUP user_functions_postprocessing_${FIELD_LOCATION}_clean)
    blt_add_test(NAME              test_user_functions_postprocessing_${FIELD_LOCATION}_run
                 COMMAND           $<TARGET_FILE:mechanics> -opt postprocessing.toml
                 NUM_MPI_TASKS     ${_uf_pp_np}
                 WORKING_DIRECTORY ${_uf_pp_dir})
    set_tests_properties(test_user_functions_postprocessing_${FIELD_LOCATION}_run PROPERTIES
                         FIXTURES_REQUIRED user_functions_postprocessing_${FIELD_LOCATION}_clean
                         FIXTURES_SETUP    user_functions_postprocessing_${FIELD_LOCATION})
    add_test(NAME              test_user_functions_postprocessing_${FIELD_LOCATION}
             COMMAND           ${PYTHON_EXECUTABLE}
                               ${CMAKE_CURRENT_SOURCE_DIR}/check_postprocessing.py
                               ${_uf_pp_dir}/results/postprocessing_${FIELD_LOCATION})
    set_tests_properties(test_user_functions_postprocessing_${FIELD_LOCATION} PROPERTIES
                         FIXTURES_REQUIRED user_functions_postprocessing_${FIELD_LOCATION})
endforeach()
