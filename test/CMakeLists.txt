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
# [Mesh.Microstructure]: mechanics generates its grains in memory (Stage 1f)
####################################
# Extensive suite only (EXACONSTIT_ENABLE_EXTENSIVE_TESTS, default OFF), and only when the
# generator is built (ENABLE_MICROSTRUCTURE). Run them with
#     ctest -R microstructure_hook --output-on-failure
if(EXACONSTIT_ENABLE_EXTENSIVE_TESTS AND ENABLE_MICROSTRUCTURE)
    # Unit test: both table forms, every refusal, and a SimulationState whose grains and
    # orientations equal a direct GenerateMicrostructure call (test_microstructure_hook.cpp)
    blt_add_executable(NAME       test_microstructure_hook
                       SOURCES    test_microstructure_hook.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS} gtest)
    target_compile_definitions(test_microstructure_hook PRIVATE
                               EXACONSTIT_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test/data")
    foreach(_np 1 2)
        if(NOT _np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            blt_add_test(NAME          test_microstructure_hook_np${_np}
                         COMMAND       test_microstructure_hook
                         NUM_MPI_TASKS ${_np})
        endif()
    endforeach()

    # End-to-end chain through the real mechanics driver. The decks are hand-written in
    # test/data/microstructure/mechanics/ (hook_*.toml; each explains its role). The steps form a
    # chain through CTest fixtures (FIXTURES_SETUP / FIXTURES_REQUIRED: CTest runs them in order
    # and skips a step whose prerequisite failed), like the generator chains in
    # test/microstructure/test_tool_stage1d.cmake and test_tool_stage1e.cmake:
    #   1. setup: empties <build>/test/microstructure_hook_e2e and copies the decks, the generator
    #      options file and the material files into it.
    #   2. hook_vs_tool: the standalone tool on voronoi_auto.toml and mechanics on its files
    #      (ms_auto_np1.toml); mechanics with the hook at 1 rank (form 1, an options file) and at 2
    #      ranks (form 2, inline tables). Hook vs tool: same seeds, >= 99% equal labels, and equal
    #      mechanics results when the labels are identical. Hook np1 vs np2: byte-identical record
    #      files and equal mechanics results (test_parallel_mesh_input --compare-mechanics, the
    #      test_mechanics.py tolerances).
    #   3. hook_two_phase: phase fractions with 2 regions and 2 [[Materials]] blocks (1 rank),
    #      compared with the tool path (seeds, labels, results when the labels are identical).
    #   4. hook_restart: an uninterrupted 5-step hook run, and a 3-step hook run that writes a
    #      checkpoint followed by a restart that must skip the hook; the two must agree.
    #   5. element order 2 and one serial refinement of the auto box: same seeds (and so
    #      orientations) as the order-1 run and the tool; at order 2 also the same labels.
    #   6. a file mesh with a passthrough part, grain mesh written as parallel parts and read back
    #      by a run without the hook (same results).
    #   7. a genuinely curved order-2 file mesh: hook and tool use the same seeds and give the
    #      same labels (>= 99%).
    # Each mechanics run with the hook goes through a small `cmake -P` script that runs the MPI
    # command, prints its output, and fails unless the exit status is 0 AND the output matches the
    # expected text (PASS_REGULAR_EXPRESSION alone would ignore the exit status). mechanics writes
    # per-rank log files into its working directory, so RESOURCE_LOCK keeps CTest from running
    # two steps of this chain at the same time under ctest -j.
    if(TARGET microstructure_generator AND TARGET test_parallel_mesh_input
       AND NOT 2 GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
        set(_hk_dir  ${CMAKE_CURRENT_BINARY_DIR}/microstructure_hook_e2e)
        set(_hk_data ${CMAKE_SOURCE_DIR}/test/data)
        set(_hk_ms   ${_hk_data}/microstructure)

        # Step 1: setup script, written at configure time with the paths filled in
        set(_hk_setup_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_hook_e2e_setup.cmake)
        file(WRITE ${_hk_setup_script}
             "file(REMOVE_RECURSE \"${_hk_dir}\")\n"
             "file(MAKE_DIRECTORY \"${_hk_dir}\")\n"
             "file(GLOB _decks \"${_hk_ms}/mechanics/hook_*.toml\")\n"
             "file(COPY \${_decks} \"${_hk_ms}/mechanics/ms_auto_np1.toml\" "
             "\"${_hk_ms}/mechanics/ms_two_phase_tool_np2.toml\" "
             "\"${_hk_ms}/mechanics/ms_hook_pass_read_np2.toml\" "
             "\"${_hk_ms}/bulged_cube_order2.mesh\" "
             "\"${_hk_data}/props_cp_voce.txt\" \"${_hk_data}/state_cp_voce.txt\" "
             "DESTINATION \"${_hk_dir}\")\n")
        add_test(NAME    test_microstructure_hook_e2e_setup
                 COMMAND ${CMAKE_COMMAND} -P ${_hk_setup_script})
        set_tests_properties(test_microstructure_hook_e2e_setup PROPERTIES
                             FIXTURES_SETUP ms_hook_case
                             LABELS         "microstructure")
        # ctest needs the working directory to exist when a test starts (also for `ctest -N`)
        file(MAKE_DIRECTORY ${_hk_dir})

        # Run helper: CMD is the command with its arguments separated by "|" (so the list
        # survives the command line as one argument); PASS is a CMake regular expression the
        # merged stdout/stderr must match ("." also matches newlines in CMake)
        set(_hk_run_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_hook_e2e_run.cmake)
        file(WRITE ${_hk_run_script}
             "string(REPLACE \"|\" \";\" _cmd \"\${CMD}\")\n"
             "execute_process(COMMAND \${_cmd} RESULT_VARIABLE _rc "
             "OUTPUT_VARIABLE _out ERROR_VARIABLE _out)\n"
             "message(\"\${_out}\")\n"
             "if(NOT _rc EQUAL 0)\n"
             "  message(FATAL_ERROR \"command failed with exit status \${_rc}\")\n"
             "endif()\n"
             "if(NOT _out MATCHES \"\${PASS}\")\n"
             "  message(FATAL_ERROR \"output does not match the expected text: \${PASS}\")\n"
             "endif()\n")

        # File comparison helper: PAIRS is a comma-separated list of "a=b" file pairs (paths
        # relative to the working directory); every pair must be byte-identical
        set(_hk_files_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_hook_e2e_files.cmake)
        file(WRITE ${_hk_files_script}
             "set(_failed 0)\n"
             "string(REPLACE \",\" \";\" _pairs \"\${PAIRS}\")\n"
             "foreach(_pair \${_pairs})\n"
             "  string(REPLACE \"=\" \";\" _ab \"\${_pair}\")\n"
             "  list(GET _ab 0 _a)\n"
             "  list(GET _ab 1 _b)\n"
             "  execute_process(COMMAND \${CMAKE_COMMAND} -E compare_files \"\${_a}\" \"\${_b}\" "
             "RESULT_VARIABLE _rc)\n"
             "  if(_rc EQUAL 0)\n"
             "    message(STATUS \"identical: \${_a} and \${_b}\")\n"
             "  else()\n"
             "    message(STATUS \"DIFFERENT or missing: \${_a} and \${_b}\")\n"
             "    set(_failed 1)\n"
             "  endif()\n"
             "endforeach()\n"
             "if(_failed)\n"
             "  message(FATAL_ERROR \"record files differ: see the lines above\")\n"
             "endif()\n")

        # The MPI launch line, built as blt_add_test builds it (BLTMacros.cmake, NUM_MPI_TASKS)
        if(MPIEXEC_EXECUTABLE)
            set(_hk_mpiexec ${MPIEXEC_EXECUTABLE})
        else()
            set(_hk_mpiexec ${MPIEXEC})
        endif()

        # One checked run: NAME test name; NP ranks; PASS expected output; SETUP fixture this
        # run provides; NEEDS fixtures it requires (besides the setup); the rest is the program
        # and its arguments
        function(ms_hook_run)
            cmake_parse_arguments(_r "" "NAME;NP;PASS;SETUP" "NEEDS;PROGRAM" ${ARGN})
            set(_cmd ${_hk_mpiexec} ${MPIEXEC_NUMPROC_FLAG} ${_r_NP} ${BLT_MPI_COMMAND_APPEND}
                     ${_r_PROGRAM})
            string(REPLACE ";" "|" _cmd_arg "${_cmd}")
            add_test(NAME              ${_r_NAME}
                     COMMAND           ${CMAKE_COMMAND} "-DCMD=${_cmd_arg}" "-DPASS=${_r_PASS}"
                                       -P ${_hk_run_script}
                     WORKING_DIRECTORY ${_hk_dir})
            set_tests_properties(${_r_NAME} PROPERTIES
                                 FIXTURES_REQUIRED "ms_hook_case;${_r_NEEDS}"
                                 FIXTURES_SETUP    ${_r_SETUP}
                                 RESOURCE_LOCK     ms_hook_mechanics
                                 PROCESSORS        ${_r_NP}
                                 LABELS            "microstructure")
        endfunction()

        # One --compare-mechanics step between two result directories (relative to _hk_dir)
        function(ms_hook_compare)
            cmake_parse_arguments(_c "" "NAME;DIR_A;DIR_B" "NEEDS" ${ARGN})
            blt_add_test(NAME          ${_c_NAME}
                         COMMAND       $<TARGET_FILE:test_parallel_mesh_input> --compare-mechanics
                                       ${_hk_dir}/${_c_DIR_A} ${_hk_dir}/${_c_DIR_B}
                         NUM_MPI_TASKS 1)
            set_tests_properties(${_c_NAME} PROPERTIES
                                 FIXTURES_REQUIRED "ms_hook_case;${_c_NEEDS}"
                                 LABELS            "microstructure")
        endfunction()

        # The summary line the hook prints for the hook_voronoi grains (8 seeds, 8 final grains)
        set(_hk_generated "generated the grains in memory.*8 requested, 8 final grains")

        # Comparisons that are not byte for byte (microstructure_hook_compare.py, see its
        # docstring): NAME test; ARGS script arguments; NEEDS fixtures. Used wherever the hook and
        # another path label different (if equivalent) meshes: the hook works in place on
        # ExaConstit's mesh with Nodes, so element centroids, volumes and face areas may differ in
        # the last bits. The seeds must be identical (same orientation for every shared seed),
        # the labels must agree on at least 99% of the elements (the measured fraction is
        # printed), and the mechanics results are compared only when the labels are identical.
        set(_hk_compare_py ${CMAKE_SOURCE_DIR}/test/microstructure_hook_compare.py)
        function(ms_hook_py)
            cmake_parse_arguments(_p "" "NAME" "ARGS;NEEDS" ${ARGN})
            add_test(NAME              ${_p_NAME}
                     COMMAND           ${PYTHON_EXECUTABLE} -I ${_hk_compare_py} ${_p_ARGS}
                     WORKING_DIRECTORY ${_hk_dir})
            set_tests_properties(${_p_NAME} PROPERTIES
                                 FIXTURES_REQUIRED "ms_hook_case;${_p_NEEDS}"
                                 LABELS            "microstructure")
        endfunction()

        # The --compare-mechanics command line, for the conditional comparisons
        set(_hk_compare_mechanics $<TARGET_FILE:test_parallel_mesh_input> --compare-mechanics)

        # Step 2: hook_vs_tool. The tool writes voronoi_auto.{grains,ori,_grains.csv} into the
        # working directory; ms_auto_np1.toml reads the first two.
        ms_hook_run(NAME    test_microstructure_hook_e2e_tool
                    NP      1
                    PASS    "[^0-9]8 final grains"
                    SETUP   ms_hook_tool_files
                    PROGRAM $<TARGET_FILE:microstructure_generator> -i ${_hk_ms}/voronoi_auto.toml)
        ms_hook_run(NAME    test_microstructure_hook_e2e_tool_mechanics
                    NP      1
                    PASS    "."
                    SETUP   ms_hook_run_tool
                    NEEDS   ms_hook_tool_files
                    PROGRAM $<TARGET_FILE:mechanics> -opt ms_auto_np1.toml)
        foreach(_np 1 2)
            ms_hook_run(NAME    test_microstructure_hook_e2e_hook_np${_np}
                        NP      ${_np}
                        PASS    "${_hk_generated}"
                        SETUP   ms_hook_run_hook_np${_np}
                        PROGRAM $<TARGET_FILE:mechanics> -opt hook_voronoi_np${_np}.toml)
        endforeach()
        # Hook vs tool: same seeds; labels of the 1-rank hook run (its plain-text grain mesh, in
        # serial element order) against the tool's grain file; mechanics when labels identical
        ms_hook_py(NAME  test_microstructure_hook_e2e_hook_vs_tool_seeds
                   ARGS  seeds voronoi_auto_grains.csv
                         results/hook_voronoi_np1/microstructure_grains.csv
                   NEEDS ms_hook_tool_files ms_hook_run_hook_np1)
        ms_hook_py(NAME  test_microstructure_hook_e2e_hook_vs_tool_labels
                   ARGS  labels voronoi_auto.grains
                         results/hook_voronoi_np1/microstructure.mesh.000000 0.99
                         ${_hk_compare_mechanics} ${_hk_dir}/results/ms_auto_np1
                         ${_hk_dir}/results/hook_voronoi_np1
                   NEEDS ms_hook_tool_files ms_hook_run_tool ms_hook_run_hook_np1)
        # Hook at 1 rank vs 2 ranks: the same path on the same mesh (no parallel refinement), so
        # the record files are byte-identical (the grain table holds fixed-point grain
        # statistics) and the mechanics results match
        ms_hook_compare(NAME  test_microstructure_hook_e2e_compare_hook_ranks
                        DIR_A results/hook_voronoi_np1
                        DIR_B results/hook_voronoi_np2
                        NEEDS ms_hook_run_hook_np1 ms_hook_run_hook_np2)
        set(_hk_a results/hook_voronoi_np1/microstructure)
        set(_hk_b results/hook_voronoi_np2/microstructure)
        set(_hk_pairs ${_hk_a}.ori=${_hk_b}.ori ${_hk_a}_grains.csv=${_hk_b}_grains.csv)
        string(REPLACE ";" "," _hk_pairs_arg "${_hk_pairs}")
        add_test(NAME              test_microstructure_hook_e2e_record_files_ranks
                 COMMAND           ${CMAKE_COMMAND} -DPAIRS=${_hk_pairs_arg}
                                   -P ${_hk_files_script}
                 WORKING_DIRECTORY ${_hk_dir})
        set(_hk_needs ms_hook_case ms_hook_run_hook_np1 ms_hook_run_hook_np2)
        set_tests_properties(test_microstructure_hook_e2e_record_files_ranks PROPERTIES
                             FIXTURES_REQUIRED "${_hk_needs}"
                             LABELS            "microstructure")

        # Step 3: hook_two_phase (1 rank). The summary must list both regions with their phases
        # ("." stands for the "; " separator, which cannot appear in the -DPASS argument). Against
        # the tool path (the tool's files read through grain_file / orientation_file /
        # region_mapping_file, ms_two_phase_tool_np2.toml): same seeds, labels, and the mechanics
        # results when the labels are identical.
        string(CONCAT _hk_two_phase "regions: 2.*region 1 .*phase 1. crystalline"
                      ".*region 2 .*phase 2. not crystalline")
        ms_hook_run(NAME    test_microstructure_hook_e2e_two_phase
                    NP      1
                    PASS    "${_hk_two_phase}"
                    SETUP   ms_hook_run_two_phase
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_two_phase_np1.toml)
        ms_hook_run(NAME    test_microstructure_hook_e2e_two_phase_tool
                    NP      1
                    PASS    "[^0-9]40 final grains"
                    SETUP   ms_hook_tool_two_phase_files
                    PROGRAM $<TARGET_FILE:microstructure_generator>
                            -i ${_hk_ms}/phase_fractions_auto.toml)
        ms_hook_run(NAME    test_microstructure_hook_e2e_two_phase_tool_mechanics
                    NP      2
                    PASS    "."
                    SETUP   ms_hook_run_two_phase_tool
                    NEEDS   ms_hook_tool_two_phase_files
                    PROGRAM $<TARGET_FILE:mechanics> -opt ms_two_phase_tool_np2.toml)
        ms_hook_py(NAME  test_microstructure_hook_e2e_two_phase_seeds
                   ARGS  seeds phase_fractions_auto_grains.csv
                         results/hook_two_phase_np1/two_phase_grains.csv
                   NEEDS ms_hook_tool_two_phase_files ms_hook_run_two_phase)
        ms_hook_py(NAME  test_microstructure_hook_e2e_two_phase_labels
                   ARGS  labels phase_fractions_auto.grains
                         results/hook_two_phase_np1/two_phase.mesh.000000 0.99
                         ${_hk_compare_mechanics} ${_hk_dir}/results/tool_two_phase_np2
                         ${_hk_dir}/results/hook_two_phase_np1
                   NEEDS ms_hook_tool_two_phase_files ms_hook_run_two_phase_tool
                         ms_hook_run_two_phase)

        # Step 4: hook_restart. Part B must skip the hook; A + B must equal the reference.
        ms_hook_run(NAME    test_microstructure_hook_e2e_restart_full
                    NP      2
                    PASS    "${_hk_generated}"
                    SETUP   ms_hook_run_restart_full
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_restart_full.toml)
        ms_hook_run(NAME    test_microstructure_hook_e2e_restart_a
                    NP      2
                    PASS    "${_hk_generated}"
                    SETUP   ms_hook_run_restart_a
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_restart_a.toml)
        ms_hook_run(NAME    test_microstructure_hook_e2e_restart_b
                    NP      2
                    PASS    "microstructure hook is skipped"
                    SETUP   ms_hook_run_restart_b
                    NEEDS   ms_hook_run_restart_a
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_restart_b.toml)
        ms_hook_compare(NAME  test_microstructure_hook_e2e_compare_restart
                        DIR_A results/hook_restart_full
                        DIR_B results/hook_restart
                        NEEDS ms_hook_run_restart_full ms_hook_run_restart_b)


        # Step 5: element order and refinement on the straight auto box. Under refinement the
        # seeds (and the orientations keyed by them) stay the same, while the labels follow the
        # mesh. Order 2 (same elements, order-2 Nodes): same seeds as the tool, and at least 99%
        # of the elements with the same grain as the order-1 run (in practice all: the generator
        # treats straight-sided Nodes exactly like no Nodes; the margin only allows for a
        # centroid tie broken differently by round-off). One serial refinement: same seeds for
        # every grain both runs have; labels not compared (different elements).
        ms_hook_run(NAME    test_microstructure_hook_e2e_order2
                    NP      1
                    PASS    "generated the grains in memory"
                    SETUP   ms_hook_run_order2
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_voronoi_p2_np1.toml)
        ms_hook_py(NAME  test_microstructure_hook_e2e_order2_seeds
                   ARGS  seeds voronoi_auto_grains.csv
                         results/hook_voronoi_p2_np1/microstructure_grains.csv
                   NEEDS ms_hook_tool_files ms_hook_run_order2)
        ms_hook_py(NAME  test_microstructure_hook_e2e_order2_labels
                   ARGS  labels results/hook_voronoi_np1/microstructure.mesh.000000
                         results/hook_voronoi_p2_np1/microstructure.mesh.000000 0.99
                   NEEDS ms_hook_run_hook_np1 ms_hook_run_order2)
        ms_hook_run(NAME    test_microstructure_hook_e2e_refined
                    NP      1
                    PASS    "generated the grains in memory"
                    SETUP   ms_hook_run_refined
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_voronoi_h1_np1.toml)
        ms_hook_py(NAME  test_microstructure_hook_e2e_refined_seeds
                   ARGS  seeds results/hook_voronoi_np1/microstructure_grains.csv
                         results/hook_voronoi_h1_np1/microstructure_grains.csv
                   NEEDS ms_hook_run_hook_np1 ms_hook_run_refined)

        # Step 6: file mesh with a passthrough part (attribute 8 of the tool's grain mesh), grain
        # mesh written as parallel parts, then a run without the hook on those parts with the
        # record orientation file and region map: same results.
        ms_hook_run(NAME    test_microstructure_hook_e2e_passthrough
                    NP      2
                    PASS    "generated the grains in memory.*passthrough attribute 8"
                    SETUP   ms_hook_run_passthrough
                    NEEDS   ms_hook_tool_files
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_pass_np2.toml)
        ms_hook_run(NAME    test_microstructure_hook_e2e_passthrough_read
                    NP      2
                    PASS    "."
                    SETUP   ms_hook_run_passthrough_read
                    NEEDS   ms_hook_run_passthrough
                    PROGRAM $<TARGET_FILE:mechanics> -opt ms_hook_pass_read_np2.toml)
        ms_hook_compare(NAME  test_microstructure_hook_e2e_compare_passthrough
                        DIR_A results/hook_pass_np2
                        DIR_B results/hook_pass_read_np2
                        NEEDS ms_hook_run_passthrough ms_hook_run_passthrough_read)

        # Step 7: a genuinely curved file mesh (order-2 hexes, bulged top). The tool and the hook
        # (at element order 2, so the curved Nodes are used; ExaConstit re-projects them onto its
        # own order-2 space, so the geometry may differ in the last bits) must use the same seeds
        # and give the same labels (both 1-rank grain meshes, serial element order), and
        # mechanics must run on the curved mesh.
        ms_hook_run(NAME    test_microstructure_hook_e2e_curved_tool
                    NP      1
                    PASS    "final grains"
                    SETUP   ms_hook_tool_curved_files
                    PROGRAM $<TARGET_FILE:microstructure_generator>
                            -i ${_hk_ms}/mechanics/hook_voronoi_micro.toml
                            -m bulged_cube_order2.mesh -o curved_tool)
        ms_hook_run(NAME    test_microstructure_hook_e2e_curved
                    NP      1
                    PASS    "generated the grains in memory"
                    SETUP   ms_hook_run_curved
                    PROGRAM $<TARGET_FILE:mechanics> -opt hook_curved_np1.toml)
        ms_hook_py(NAME  test_microstructure_hook_e2e_curved_seeds
                   ARGS  seeds curved_tool_grains.csv
                         results/hook_curved_np1/microstructure_grains.csv
                   NEEDS ms_hook_tool_curved_files ms_hook_run_curved)
        ms_hook_py(NAME  test_microstructure_hook_e2e_curved_labels
                   ARGS  labels curved_tool.mesh.000000
                         results/hook_curved_np1/microstructure.mesh.000000 0.99
                   NEEDS ms_hook_tool_curved_files ms_hook_run_curved)
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
