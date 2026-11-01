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

# libtest_user_classes (test_user_classes.cpp) holds the same functions as user classes, built
# as a user on a shared machine builds theirs: a plain shared library that includes only
# exaconstit_user.hpp (no ExaConstit, MFEM or MPI headers or settings). The beginner examples of
# user classes (examples/user_classes) are built the same way, so they keep compiling.
add_library(test_user_classes SHARED test_user_classes.cpp)
add_library(example_user_classes SHARED ${CMAKE_SOURCE_DIR}/examples/user_classes/my_output.cpp
                                        ${CMAKE_SOURCE_DIR}/examples/user_classes/my_state_vars.cpp)
foreach(_uf_target test_user_classes example_user_classes)
    target_include_directories(${_uf_target} PRIVATE ${HEADER_INCLUDE_DIR})
    set_target_properties(${_uf_target} PROPERTIES
                          CXX_STANDARD 17
                          CXX_STANDARD_REQUIRED ON
                          LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

# ------------------------------------------------------------------------------------------------
# Input decks: one per case, from user_functions.toml.in. Only the state variable functions (and
# their libraries) change between cases. The function fields are TOML values: a quoted function
# name, or a factory table of libtest_user_classes ("class:<variant>" below).
#   <case>|<UMAT library>|<UMAT register_function>|<UMAT initialize_function>
# The ExaCMech region uses the function form, or the class form when the UMAT does. The
# restart decks run 2 steps: *_restart_write writes a checkpoint after step 1, *_restart reads it.
# ------------------------------------------------------------------------------------------------
set(_uf_library "$<TARGET_FILE:test_user_functions>")
set(_uf_class_library "$<TARGET_FILE:test_user_classes>")
set(_uf_missing_library "libtest_user_functions_missing${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(_uf_cases
    "success|${_uf_library}|RegisterUmatStateVariables|InitializeUmatStateVariables"
    "wrong_arguments|${_uf_library}|RegisterWrongArguments|InitializeUmatStateVariables"
    "extern_c|${_uf_library}|RegisterExternC|InitializeUmatStateVariables"
    "missing_library|${_uf_missing_library}|RegisterUmatStateVariables|InitializeUmatStateVariables"
    "overlap|${_uf_library}|RegisterOverlappingStateVariables|InitializeUmatStateVariables"
    "out_of_range|${_uf_library}|RegisterOutOfRangeStateVariables|InitializeUmatStateVariables"
    "class_success|${_uf_class_library}|class:umat_names|class:umat_start"
    "class_unknown_variant|${_uf_class_library}|class:no_such_variant|class:umat_start"
    "class_wrong_class|${_uf_class_library}|class:umat_names|class:umat_names"
    "class_duplicate|${_uf_class_library}|class:duplicate_names|class:umat_start"
    "class_unknown_input|${_uf_class_library}|class:umat_names|class:ask_unknown_input"
    "class_throwing|${_uf_class_library}|class:umat_names|class:throwing_initialize"
    "restart_write|${_uf_library}|RegisterUmatStateVariables|InitializeUmatStateVariables"
    "restart|${_uf_library}|RegisterUmatStateVariables|InitializeStopsTheRun"
    "class_restart_write|${_uf_class_library}|class:umat_names|class:umat_start"
    "class_restart|${_uf_class_library}|class:umat_names|class:throwing_initialize")

# TOML value of a user function key: "Name", or the factory table for class:<variant>
function(_uf_toml_value OUT_VAR VALUE)
    if(VALUE MATCHES "^class:(.*)$")
        set(${OUT_VAR} "{ factory = \"MakeTestClasses\", variant = \"${CMAKE_MATCH_1}\" }"
            PARENT_SCOPE)
    else()
        set(${OUT_VAR} "\"${VALUE}\"" PARENT_SCOPE)
    endif()
endfunction()

set(EXACONSTIT_TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/test/data)
set(USER_FUNCTION_LIBRARY ${_uf_library})
foreach(_uf_case ${_uf_cases})
    string(REPLACE "|" ";" _uf_fields "${_uf_case}")
    list(GET _uf_fields 0 CASE_NAME)
    list(GET _uf_fields 1 UMAT_STATE_VAR_LIBRARY)
    list(GET _uf_fields 2 _uf_register)
    list(GET _uf_fields 3 _uf_initialize)
    _uf_toml_value(UMAT_REGISTER_FUNCTION "${_uf_register}")
    _uf_toml_value(UMAT_INITIALIZE_FUNCTION "${_uf_initialize}")
    if(CASE_NAME MATCHES "^class_")
        set(CRYSTAL_STATE_VAR_LIBRARY ${_uf_class_library})
        _uf_toml_value(CRYSTAL_REGISTER_FUNCTION "class:crystal_names")
    else()
        set(CRYSTAL_STATE_VAR_LIBRARY ${_uf_library})
        _uf_toml_value(CRYSTAL_REGISTER_FUNCTION "RegisterExaCMechStateVariables")
    endif()
    set(T_FINAL 0.1)
    set(RESTART_TABLE "")
    if(CASE_NAME MATCHES "restart_write$")
        set(T_FINAL 0.2)
        set(RESTART_TABLE "[Restart]\n    write = true\n    interval = 1\n    write_final = false")
    elseif(CASE_NAME MATCHES "restart$")
        set(T_FINAL 0.2)
        string(CONCAT RESTART_TABLE "[Restart]\n    read = true\n    read_cycle = 1\n"
                                    "    read_directory = \"../${CASE_NAME}_write/restart\"")
    endif()
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
add_dependencies(test_state_variable_functions test_user_functions test_user_classes)
# The same check for both forms: the classes register the same names and set the same values
foreach(_uf_form success class_success)
    string(REPLACE "success" "state_vars" _uf_name "${_uf_form}")
    foreach(_np 1 2)
        if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            set(_uf_test test_user_functions_${_uf_name}_np${_np})
            blt_add_test(NAME              ${_uf_test}
                         COMMAND           test_state_variable_functions
                                           ${CMAKE_CURRENT_BINARY_DIR}/${_uf_form}/${_uf_form}.toml
                         NUM_MPI_TASKS     ${_np}
                         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_uf_form})
            set_tests_properties(${_uf_test} PROPERTIES RESOURCE_LOCK user_functions_${_uf_form})
        endif()
    endforeach()
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
    "out_of_range|does not fit in the region's 10 state variables"
    "class_unknown_variant|does not know the variant 'no_such_variant'"
    "class_wrong_class|is a .*UmatNames, which is not derived from .*InitializeFunction"
    "class_duplicate|names the state variable 'damage', which material umat_2 already has"
    "class_unknown_input|asks for the input 'no_such_input'.*Its inputs are:"
    "class_throwing|deliberate error from ThrowingInitialize")
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
# Restart (needs Axom): the initialize function / class does not run on a restart, since the
# checkpoint has the state. The restart's initialize stops the run if it is called.
# ------------------------------------------------------------------------------------------------
if(ENABLE_AXOM)
    foreach(_uf_rs restart class_restart)
        set(_uf_rs_dir ${CMAKE_CURRENT_BINARY_DIR}/${_uf_rs})
        add_test(NAME    test_user_functions_${_uf_rs}_clean
                 COMMAND ${CMAKE_COMMAND} -E rm -rf ${_uf_rs_dir}_write/restart
                         ${_uf_rs_dir}_write/results ${_uf_rs_dir}/restart ${_uf_rs_dir}/results)
        set_tests_properties(test_user_functions_${_uf_rs}_clean PROPERTIES
                             FIXTURES_SETUP user_functions_${_uf_rs}_clean)
        blt_add_test(NAME              test_user_functions_${_uf_rs}_write
                     COMMAND           $<TARGET_FILE:mechanics> -opt ${_uf_rs}_write.toml
                     NUM_MPI_TASKS     1
                     WORKING_DIRECTORY ${_uf_rs_dir}_write)
        set_tests_properties(test_user_functions_${_uf_rs}_write PROPERTIES
                             FIXTURES_REQUIRED user_functions_${_uf_rs}_clean
                             FIXTURES_SETUP    user_functions_${_uf_rs}_write)
        blt_add_test(NAME              test_user_functions_${_uf_rs}
                     COMMAND           $<TARGET_FILE:mechanics> -opt ${_uf_rs}.toml
                     NUM_MPI_TASKS     1
                     WORKING_DIRECTORY ${_uf_rs_dir})
        set_tests_properties(test_user_functions_${_uf_rs} PROPERTIES
                             FIXTURES_REQUIRED user_functions_${_uf_rs}_write)
    endforeach()
endif()

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
# Each run twice: with the functions, and with the classes (factory tables, plus the
# self-checking check_inputs entry).
string(CONCAT _uf_pp_check_entry
       "    [[PostProcessing.user_functions]]\n"
       "        library_path = \"${_uf_class_library}\"\n"
       "        function_name = { factory = \"MakeTestClasses\", variant = \"check_inputs\" }\n"
       "        type = \"volume_average\"\n")
foreach(_uf_pp_case zone quadrature zone_class quadrature_class)
    string(REGEX REPLACE "_class$" "" FIELD_LOCATION "${_uf_pp_case}")
    if(_uf_pp_case MATCHES "_class$")
        set(PP_LIBRARY ${_uf_class_library})
        _uf_toml_value(PP_UNIT_VALUE "class:unit_value")
        _uf_toml_value(PP_STRESS_INVARIANTS "class:stress_invariants")
        _uf_toml_value(PP_STRESS_TRIAXIALITY "class:stress_triaxiality")
        set(PP_EXTRA_ENTRIES "${_uf_pp_check_entry}")
    else()
        set(PP_LIBRARY ${_uf_library})
        _uf_toml_value(PP_UNIT_VALUE "UnitValue")
        _uf_toml_value(PP_STRESS_INVARIANTS "StressInvariants")
        _uf_toml_value(PP_STRESS_TRIAXIALITY "StressTriaxiality")
        set(PP_EXTRA_ENTRIES "")
    endif()
    set(_uf_pp_dir ${CMAKE_CURRENT_BINARY_DIR}/postprocessing_${_uf_pp_case})
    file(MAKE_DIRECTORY ${_uf_pp_dir})
    configure_file(postprocessing.toml.in ${_uf_pp_dir}/postprocessing.toml.in @ONLY)
    file(GENERATE OUTPUT ${_uf_pp_dir}/postprocessing.toml
                  INPUT  ${_uf_pp_dir}/postprocessing.toml.in)

    set(_uf_pp_np 2)
    if(_uf_pp_np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
        set(_uf_pp_np 1)
    endif()
    # Volume average files are appended to, so every run starts from an empty results folder
    add_test(NAME    test_user_functions_postprocessing_${_uf_pp_case}_clean
             COMMAND ${CMAKE_COMMAND} -E rm -rf ${_uf_pp_dir}/results)
    set_tests_properties(test_user_functions_postprocessing_${_uf_pp_case}_clean PROPERTIES
                         FIXTURES_SETUP user_functions_postprocessing_${_uf_pp_case}_clean)
    blt_add_test(NAME              test_user_functions_postprocessing_${_uf_pp_case}_run
                 COMMAND           $<TARGET_FILE:mechanics> -opt postprocessing.toml
                 NUM_MPI_TASKS     ${_uf_pp_np}
                 WORKING_DIRECTORY ${_uf_pp_dir})
    set_tests_properties(test_user_functions_postprocessing_${_uf_pp_case}_run PROPERTIES
                         FIXTURES_REQUIRED user_functions_postprocessing_${_uf_pp_case}_clean
                         FIXTURES_SETUP    user_functions_postprocessing_${_uf_pp_case})
    add_test(NAME              test_user_functions_postprocessing_${_uf_pp_case}
             COMMAND           ${PYTHON_EXECUTABLE}
                               ${CMAKE_CURRENT_SOURCE_DIR}/check_postprocessing.py
                               ${_uf_pp_dir}/results/postprocessing_${FIELD_LOCATION})
    set_tests_properties(test_user_functions_postprocessing_${_uf_pp_case} PROPERTIES
                         FIXTURES_REQUIRED user_functions_postprocessing_${_uf_pp_case})
endforeach()

# ------------------------------------------------------------------------------------------------
# Volume statistics ([PostProcessing.volume_averages] statistics / histograms, and the
# statistics / histogram keys of user function entries). Material 2 is a 2x2x2 corner block of
# the 5x5x5 auto mesh. The deck runs on 2 ranks, reading a decomposition in which rank 0 owns no
# element of material 2 (written by write_statistics_mesh with an explicit partition, so it is
# the same on every machine), and on 1 rank with the auto mesh; check_statistics.py then checks
# exact values and compares the runs (see statistics.toml.in).
# ------------------------------------------------------------------------------------------------
if(NOT 2 GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
    # Grain per element of the auto mesh (lexicographic order: x fastest, then y, then z):
    # grain 2 for the corner block of the last 2 elements in x, y and z, grain 1 elsewhere;
    # grain g -> material g
    set(STATS_GRAIN_FILE ${CMAKE_CURRENT_BINARY_DIR}/grains_corner.txt)
    set(STATS_REGION_MAP ${CMAKE_CURRENT_BINARY_DIR}/region_mapping_corner.txt)
    set(_uf_grains "")
    foreach(_iz RANGE 4)
        foreach(_iy RANGE 4)
            foreach(_ix RANGE 4)
                if(_ix GREATER 2 AND _iy GREATER 2 AND _iz GREATER 2)
                    string(APPEND _uf_grains "2\n")
                else()
                    string(APPEND _uf_grains "1\n")
                endif()
            endforeach()
        endforeach()
    endforeach()
    file(WRITE ${STATS_GRAIN_FILE} "${_uf_grains}")
    file(WRITE ${STATS_REGION_MAP} "1 1\n2 2\n")

    # The 2-part decomposition of the same mesh: part 0 holds the elements numbered before the
    # first element of material 2, part 1 the rest
    blt_add_executable(NAME       write_statistics_mesh
                       SOURCES    write_statistics_mesh.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS})
    set(_uf_st_mesh_dir ${CMAKE_CURRENT_BINARY_DIR}/statistics_mesh)
    file(MAKE_DIRECTORY ${_uf_st_mesh_dir})
    set(_uf_st_mesh_prefix ${_uf_st_mesh_dir}/corner_np2.mesh.)
    blt_add_test(NAME          test_user_functions_statistics_mesh
                 COMMAND       write_statistics_mesh ${STATS_GRAIN_FILE} 5 5 5
                               ${_uf_st_mesh_prefix} 2
                 NUM_MPI_TASKS 1)
    set_tests_properties(test_user_functions_statistics_mesh PROPERTIES
                         FIXTURES_SETUP user_functions_statistics_mesh)

    # [Mesh] table lines of the two meshes (@STATS_MESH_TABLE@ in statistics.toml.in)
    string(CONCAT _uf_st_auto_mesh "    type = \"auto\"\n    [Mesh.Auto]\n"
                                   "        mxyz = [1.0, 1.0, 1.0]\n        nxyz = [5, 5, 5]")
    set(_uf_st_parallel_mesh "    type = \"parallel\"\n    floc = \"${_uf_st_mesh_prefix}\"")

    set(STATS_OUTPUT_FORMAT "text")
    foreach(NUM_RANKS 2 1)
        if(NUM_RANKS EQUAL 1)
            set(STATS_MESH_TABLE "${_uf_st_auto_mesh}")
        else()
            set(STATS_MESH_TABLE "${_uf_st_parallel_mesh}")
        endif()
        set(_uf_st_dir ${CMAKE_CURRENT_BINARY_DIR}/statistics_np${NUM_RANKS})
        file(MAKE_DIRECTORY ${_uf_st_dir})
        configure_file(statistics.toml.in ${_uf_st_dir}/statistics.toml.in @ONLY)
        file(GENERATE OUTPUT ${_uf_st_dir}/statistics.toml
                      INPUT  ${_uf_st_dir}/statistics.toml.in)

        # Start from an empty folder (files are appended to; marker files are per run)
        set(_uf_st_clean user_functions_statistics_np${NUM_RANKS}_clean)
        add_test(NAME    test_user_functions_statistics_np${NUM_RANKS}_clean
                 COMMAND ${CMAKE_COMMAND} -E rm -rf ${_uf_st_dir}/results
                         ${_uf_st_dir}/active_rank_0_region_1 ${_uf_st_dir}/active_rank_0_region_2
                         ${_uf_st_dir}/active_rank_1_region_1 ${_uf_st_dir}/active_rank_1_region_2)
        set_tests_properties(test_user_functions_statistics_np${NUM_RANKS}_clean PROPERTIES
                             FIXTURES_SETUP ${_uf_st_clean})
        blt_add_test(NAME              test_user_functions_statistics_np${NUM_RANKS}_run
                     COMMAND           $<TARGET_FILE:mechanics> -opt statistics.toml
                     NUM_MPI_TASKS     ${NUM_RANKS}
                     WORKING_DIRECTORY ${_uf_st_dir})
        set_tests_properties(test_user_functions_statistics_np${NUM_RANKS}_run PROPERTIES
                             FIXTURES_REQUIRED "${_uf_st_clean};user_functions_statistics_mesh"
                             FIXTURES_SETUP    user_functions_statistics_runs)
    endforeach()

    add_test(NAME    test_user_functions_statistics
             COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_statistics.py
                     ${CMAKE_CURRENT_BINARY_DIR}/statistics_np2/results/statistics_np2
                     ${CMAKE_CURRENT_BINARY_DIR}/statistics_np1/results/statistics_np1
                     ${CMAKE_CURRENT_BINARY_DIR}/statistics_np2 2)
    set_tests_properties(test_user_functions_statistics PROPERTIES
                         FIXTURES_REQUIRED user_functions_statistics_runs)

    # The same 2-rank run with output_format = "hdf5" (needs Axom): material 2 is only on
    # rank 1, so its rows reach rank 0's file only through the gather of FinishOutputStep().
    # check_hdf5_output.py compares the file with the text run above.
    if(ENABLE_AXOM)
        set(STATS_OUTPUT_FORMAT "hdf5")
        set(NUM_RANKS 2)
        set(STATS_MESH_TABLE "${_uf_st_parallel_mesh}")
        set(_uf_h5_dir ${CMAKE_CURRENT_BINARY_DIR}/statistics_hdf5_np2)
        file(MAKE_DIRECTORY ${_uf_h5_dir})
        configure_file(statistics.toml.in ${_uf_h5_dir}/statistics.toml.in @ONLY)
        file(GENERATE OUTPUT ${_uf_h5_dir}/statistics.toml
                      INPUT  ${_uf_h5_dir}/statistics.toml.in)
        add_test(NAME    test_user_functions_statistics_hdf5_clean
                 COMMAND ${CMAKE_COMMAND} -E rm -rf ${_uf_h5_dir}/results)
        set_tests_properties(test_user_functions_statistics_hdf5_clean PROPERTIES
                             FIXTURES_SETUP user_functions_statistics_hdf5_clean)
        blt_add_test(NAME              test_user_functions_statistics_hdf5_run
                     COMMAND           $<TARGET_FILE:mechanics> -opt statistics.toml
                     NUM_MPI_TASKS     2
                     WORKING_DIRECTORY ${_uf_h5_dir})
        set_tests_properties(test_user_functions_statistics_hdf5_run PROPERTIES
                             FIXTURES_REQUIRED
                             "user_functions_statistics_hdf5_clean;user_functions_statistics_mesh"
                             FIXTURES_SETUP    user_functions_statistics_hdf5_run)
        add_test(NAME    test_user_functions_statistics_hdf5
                 COMMAND ${PYTHON_EXECUTABLE}
                         ${CMAKE_SOURCE_DIR}/test/hdf5_output/check_hdf5_output.py
                         ${_uf_h5_dir}/results/statistics_np2/post_processing.hdf5
                         ${CMAKE_CURRENT_BINARY_DIR}/statistics_np2/results/statistics_np2
                         ${_uf_h5_dir}/results/statistics_np2
                         ${CMAKE_SOURCE_DIR}/scripts/postprocessing)
        set_tests_properties(test_user_functions_statistics_hdf5 PROPERTIES
                             FIXTURES_REQUIRED
                             "user_functions_statistics_hdf5_run;user_functions_statistics_runs")
    endif()
endif()
