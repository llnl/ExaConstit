#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------

set(EXACONSTIT_TEST_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  EXACONSTIT_TEST_DEPENDS
                             DEPENDS_ON  mfem ecmech RAJA mpi snls)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_TEST_DEPENDS blt::cuda_runtime blt::cuda CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_TEST_DEPENDS blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_TEST_DEPENDS cuda cuda_runtime CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_TEST_DEPENDS openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND EXACONSTIT_TEST_DEPENDS blt::hip blt::hip_runtime hipblas rocsparse rocrand)
endif()

if (SNLS_USE_RAJA_PORT_SUITE)
    list(APPEND EXACONSTIT_TEST_DEPENDS chai umpire camp fmt::fmt)
endif()

if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_TEST_DEPENDS caliper)
endif()

list(APPEND EXACONSTIT_TEST_DEPENDS exaconstit_static)

message("-- EXACONSTIT_TEST_DEPENDS: ${EXACONSTIT_TEST_DEPENDS}")

blt_add_executable(NAME      test_grad_oper
                  SOURCES    grad_test.cpp
                  OUTPUT_DIR ${TEST_OUTPUT_DIR}
                  DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS} gtest)

blt_add_test(NAME    test_gradient_operation
             COMMAND test_grad_oper)

# MPI tests never use more ranks than the machine has cores, so the suite does not
# oversubscribe small machines such as the 4-core CI runners (Open MPI refuses to start more
# ranks than slots by default). The default is the number of logical cores CMake detects;
# set EXACONSTIT_TEST_MAX_MPI_TASKS to override it. Rank counts above the cap are skipped.
# It is defined here, outside if(ENABLE_AXOM), because the restart tests, the parallel mesh
# tests below and test/microstructure all use it.
cmake_host_system_information(RESULT _detected_cores QUERY NUMBER_OF_LOGICAL_CORES)
if(NOT _detected_cores OR _detected_cores LESS 1)
    set(_detected_cores 1)
endif()
set(EXACONSTIT_TEST_MAX_MPI_TASKS ${_detected_cores} CACHE STRING
    "Largest number of MPI ranks any test may use (default: detected logical cores)")

####################################
# Parallel mesh decomposition input ([Mesh] type = "parallel")
####################################
# These tests are part of the extensive suite (EXACONSTIT_ENABLE_EXTENSIVE_TESTS, defined in
# cmake/ExaConstitOptions.cmake, default OFF), so the default CI suite does not grow. Nothing
# here is built or registered when the option is off. Turn it on locally to run them:
#     cmake -DEXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON ...
#     ctest -R parallel_mesh --output-on-failure
if(EXACONSTIT_ENABLE_EXTENSIVE_TESTS)
    # test_parallel_mesh_input checks the options (type = "parallel", floc as prefix), compares
    # the makeMesh PARALLEL branch with the serial FILE read rank by rank, and checks that
    # ref_ser/ref_par give the same refined mesh. It also has two helper modes used by the
    # mechanics comparison below (see the comment at the top of test_parallel_mesh_input.cpp).
    blt_add_executable(NAME       test_parallel_mesh_input
                       SOURCES    test_parallel_mesh_input.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS} gtest)
    target_compile_definitions(test_parallel_mesh_input PRIVATE
                               EXACONSTIT_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test/data")

    # A decomposition always has as many parts as ranks, so each rank count tests a different
    # partition (the test writes its own decomposition for the rank count it runs with)
    foreach(_np 1 2 3 4)
        if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            blt_add_test(NAME          test_parallel_mesh_input_np${_np}
                         COMMAND       test_parallel_mesh_input
                         NUM_MPI_TASKS ${_np})
        endif()
    endforeach()

    # mechanics on cube_tet.mesh read as a serial file and as a gzip decomposition must give
    # the same results. The tests form a chain through CTest fixtures (FIXTURES_SETUP /
    # FIXTURES_REQUIRED), which make CTest run them in order and skip a step whose
    # prerequisite failed:
    #   1. test_parallel_mesh_mechanics_setup writes, for each rank count, a decomposition and
    #      a pair of input decks into a scratch directory. It is not the test directory,
    #      because test_mechanics.py runs every .toml it finds there.
    #   2. test_parallel_mesh_mechanics_{serial,parallel}_np<N> run the real mechanics driver
    #      on each deck.
    #   3. test_parallel_mesh_mechanics_compare_np<N> compares the two runs' result files.
    # All mechanics runs share the scratch directory as working directory, and mechanics writes
    # per-rank log files there (mfem_logging_rank_<r>.log, utilities/unified_logger.hpp:688).
    # RESOURCE_LOCK parallel_mesh_case keeps CTest from running two of them at the same time
    # under ctest -j, so they cannot overwrite each other's logs.
    set(_pmi_case_dir ${CMAKE_CURRENT_BINARY_DIR}/parallel_mesh_mechanics)
    set(_pmi_rank_counts)
    foreach(_np 1 2)
        if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            list(APPEND _pmi_rank_counts ${_np})
        endif()
    endforeach()
    blt_add_test(NAME              test_parallel_mesh_mechanics_setup
                 COMMAND           test_parallel_mesh_input --write-mechanics-case
                                   ${_pmi_case_dir} ${_pmi_rank_counts}
                 NUM_MPI_TASKS     1)
    set_tests_properties(test_parallel_mesh_mechanics_setup PROPERTIES
                         FIXTURES_SETUP parallel_mesh_case)
    foreach(_np ${_pmi_rank_counts})
        foreach(_kind serial parallel)
            blt_add_test(NAME              test_parallel_mesh_mechanics_${_kind}_np${_np}
                         COMMAND           $<TARGET_FILE:mechanics> -opt ${_kind}_np${_np}.toml
                         NUM_MPI_TASKS     ${_np}
                         WORKING_DIRECTORY ${_pmi_case_dir})
            set_tests_properties(test_parallel_mesh_mechanics_${_kind}_np${_np} PROPERTIES
                                 FIXTURES_REQUIRED parallel_mesh_case
                                 FIXTURES_SETUP    parallel_mesh_runs_np${_np}
                                 RESOURCE_LOCK     parallel_mesh_case)
        endforeach()
        blt_add_test(NAME    test_parallel_mesh_mechanics_compare_np${_np}
                     COMMAND test_parallel_mesh_input --compare-mechanics
                             ${_pmi_case_dir}/results/serial_np${_np}
                             ${_pmi_case_dir}/results/parallel_np${_np}
                     NUM_MPI_TASKS 1)
        set_tests_properties(test_parallel_mesh_mechanics_compare_np${_np} PROPERTIES
                             FIXTURES_REQUIRED "parallel_mesh_case;parallel_mesh_runs_np${_np}")
    endforeach()

    # The wrong-rank-count abort: running the 2-part decomposition on 1 rank must stop in
    # makeMesh (verifyParallelMeshParts in simulation_state.cpp) with the "exactly as many MPI
    # ranks as it has parts" message. The run ends with MPI_Abort, so the test passes on the
    # message (PASS_REGULAR_EXPRESSION ignores the exit code) rather than on the exit status.
    # It stops before any result file is written, so the parallel_np2 results stay untouched.
    if("2" IN_LIST _pmi_rank_counts)
        blt_add_test(NAME              test_parallel_mesh_mechanics_rank_mismatch
                     COMMAND           $<TARGET_FILE:mechanics> -opt parallel_np2.toml
                     NUM_MPI_TASKS     1
                     WORKING_DIRECTORY ${_pmi_case_dir})
        set_tests_properties(test_parallel_mesh_mechanics_rank_mismatch PROPERTIES
                             FIXTURES_REQUIRED       parallel_mesh_case
                             RESOURCE_LOCK           parallel_mesh_case
                             PASS_REGULAR_EXPRESSION
                                 "exactly as many MPI ranks as it has parts")
    endif()
endif()

####################################
# Checkpoint/restart tests (Axom Sidre)
####################################
if(ENABLE_AXOM)
    # Mesh round trip through the checkpoint mesh I/O, including a case with empty ranks
    blt_add_executable(NAME       test_restart_mesh_roundtrip
                       SOURCES    test_restart_mesh_roundtrip.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS} gtest)

    # SimulationState checkpoint, delete the input files, restart, compare bitwise
    blt_add_executable(NAME       test_restart_e2e
                       SOURCES    test_restart_e2e.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS} gtest)

    foreach(_np 1 2 3 4)
        if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            blt_add_test(NAME          test_restart_mesh_roundtrip_np${_np}
                         COMMAND       test_restart_mesh_roundtrip
                         NUM_MPI_TASKS ${_np})
        endif()
    endforeach()

    foreach(_np 2 3)
        if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            blt_add_test(NAME          test_restart_e2e_np${_np}
                         COMMAND       test_restart_e2e
                         NUM_MPI_TASKS ${_np})
        endif()
    endforeach()

    # Diagnostic: lists how each quadrature function is stored in a checkpoint rank file
    blt_add_executable(NAME       inspect_restart_rank_file
                       SOURCES    ${CMAKE_SOURCE_DIR}/scripts/restart/inspect_restart_rank_file.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON axom)

    # Compares two checkpoint directories view by view (bitwise by default)
    blt_add_executable(NAME       compare_restart_checkpoints
                       SOURCES    ${CMAKE_SOURCE_DIR}/scripts/restart/compare_restart_checkpoints.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON axom)

    # Restart tests on real runs (test_restart_runs.py): cyclic loading chained every 1 and 2
    # cycles, long monotonic tension chained halfway, a run with empty ranks, restarts from past
    # reference checkpoints, and a negative control. They take 10-15 minutes on 8 ranks, so they
    # are not part of the default (CI) test suite. Developers working on the restart code must
    # enable them locally with -DEXACONSTIT_ENABLE_LONG_TESTS=ON and make sure they pass:
    #     ctest -L long --output-on-failure
    option(EXACONSTIT_ENABLE_LONG_TESTS
           "Register the long-running tests (run-level restart tests); not for CI" OFF)
    set(EXACONSTIT_RESTART_RUN_TEST_NP 8 CACHE STRING
        "MPI ranks for test_restart_runs (capped at EXACONSTIT_TEST_MAX_MPI_TASKS)")
    if(EXACONSTIT_ENABLE_LONG_TESTS)
        set(_restart_runs_np ${EXACONSTIT_RESTART_RUN_TEST_NP})
        if(_restart_runs_np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            message(STATUS "test_restart_runs: using ${EXACONSTIT_TEST_MAX_MPI_TASKS} MPI ranks "
                           "instead of ${EXACONSTIT_RESTART_RUN_TEST_NP} "
                           "(EXACONSTIT_TEST_MAX_MPI_TASKS)")
            set(_restart_runs_np ${EXACONSTIT_TEST_MAX_MPI_TASKS})
        endif()
        set(_restart_runs_mpiexec "${MPIEXEC_EXECUTABLE}")
        if(NOT _restart_runs_mpiexec)
            set(_restart_runs_mpiexec "mpirun")
        endif()
        add_custom_command(TARGET compare_restart_checkpoints POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/test/test_restart_runs.py
            $<TARGET_FILE_DIR:test_grad_oper>/../test/test_restart_runs.py)
        add_test(NAME test_restart_runs
                 COMMAND ${PYTHON_EXECUTABLE} test_restart_runs.py
                         --np ${_restart_runs_np}
                         --mpiexec ${_restart_runs_mpiexec}
                         --exe $<TARGET_FILE:mechanics>
                         --compare-tool $<TARGET_FILE:compare_restart_checkpoints>
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test)
        set_tests_properties(test_restart_runs PROPERTIES
                             LABELS "restart;long"
                             PROCESSORS ${_restart_runs_np}
                             TIMEOUT 3600)
    endif()
endif()

if (NOT DEFINED PYTHON_EXECUTABLE)
        message(FATAL_ERROR "cmake: PYTHON_EXECUTABLE must be defined if running test suite current value: ${PYTHON_EXECUTABLE}")
endif()

## Borrowed from Conduit https://github.com/LLNL/conduit
## The license file can be found under 
##------------------------------------------------------------------------------
## - Adds a python based unit test
##
## add_python_test( TEST test)
##------------------------------------------------------------------------------
function(add_python_test )
    set(options)
    set(singleValueArgs TEST )
    set(multiValueArgs )

    # parse our arguments
    cmake_parse_arguments(args
                         "${options}"
                         "${singleValueArgs}"
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Python-based Unit Test: ${args_TEST}")
    add_test(NAME ${args_TEST} COMMAND
             ${PYTHON_EXECUTABLE} ${args_TEST})

    # use proper env var path sep for current platform
    if(WIN32)
        set(ENV_PATH_SEP "\\;")
    else()
        set(ENV_PATH_SEP ":")
    endif()
    # make sure python can pick up the modules we built
    # if python path is already set -- we need to append to it
    # this is important for running in spack's build-env
    set(py_path "")
    if(DEFINED ENV{PYTHONPATH})
        set(py_path "$ENV{PYTHONPATH}${ENV_PATH_SEP}")
    endif()
    set_property(TEST ${args_TEST}
                 PROPERTY
                 ENVIRONMENT "PYTHONPATH=${py_path}${CMAKE_BINARY_DIR}/python-modules/${ENV_PATH_SEP}${CMAKE_CURRENT_SOURCE_DIR}")
    if(WIN32)
        # proper path to dlls for vstudio and proper path for non config based gen (nmake, etc)
        set_property(TEST ${args_TEST}
                     APPEND
                     PROPERTY
                     ENVIRONMENT "PATH=${CMAKE_BINARY_DIR}/bin/${ENV_PATH_SEP}${CMAKE_BINARY_DIR}/bin/$<CONFIG>/${ENV_PATH_SEP}$ENV{PATH}")
    endif()

    # set folder if passed
    if( DEFINED args_FOLDER )
        blt_set_target_folder(TARGET ${args_TEST} FOLDER ${args_FOLDER})
    endif()

endfunction(add_python_test)

####################################
# Add Python Module Tests
####################################
set(PYTHON_MODULE_TESTS test_mechanics.py)


foreach(TEST ${PYTHON_MODULE_TESTS})
    add_python_test(TEST ${TEST}
                    FOLDER ${TEST_OUTPUT_DIR})
endforeach()

# A hack more or less to get the test config files into the problem
add_custom_command(TARGET test_grad_oper PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/test/data/ $<TARGET_FILE_DIR:test_grad_oper>/../test/)

add_custom_command(TARGET test_grad_oper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/test/test_mechanics.py $<TARGET_FILE_DIR:test_grad_oper>/../test/test_mechanics.py
)
