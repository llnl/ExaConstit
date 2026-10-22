# Generator tool outputs: grain statistics, the grain table / grain adjacency CSV
# writers, and end-to-end runs of the Stage 1b/1c features through ExaConstit's mechanics driver.
#
# test_tool_outputs checks ComputeGrainStatistics (hand-computed values, a brute-force loop,
# periodic face pairing, bitwise rank-count independence against a single-rank reference),
# CompareWithTargetVolumes and the two CSV writers. Runs at 1-4 ranks.
ms_add_gtest(NAME          test_tool_outputs
             SOURCES       test_tool_outputs.cpp
             DEPENDS_ON    exaconstit_microstructure ms_io ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)

# cube_tet.mesh lives in the source tree; its path is resolved from here
target_compile_definitions(test_tool_outputs PRIVATE
                           MS_TOOL_OUTPUTS_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")

#------------------------------------------------------------------------------
# End to end, Stage 1b/1c: generator -> byte-identical outputs at 1 and 2 ranks -> mechanics
#
# Five cases, each with a hand-written generator input in test/data/microstructure/e2e/ and
# hand-written mechanics decks in test/data/microstructure/mechanics/:
#   u_shape      non-convex U-shaped part (test/data/microstructure/u_shape_hex.mesh), seeds in the
#                bounding box, default fragment cleanup (pieces split and merged);
#   periodic     periodic 8x8x8 box, quadrature vote, periodic face pairing;
#   lognormal    log-normal sizes, polydisperse placement, Laguerre weight fit (10x10x10 box);
#   columnar     columnar grains along z (quadric metric, A = I - z z^T);
#   johnson_mehl Johnson-Mehl nucleation and growth (additive metric).
#
# The tests form a chain through CTest fixtures (FIXTURES_SETUP / FIXTURES_REQUIRED: CTest runs
# them in order and skips a step whose prerequisite failed):
#   1. test_microstructure_e2e_stage1bc_setup: empties
#      <build>/test/microstructure/end_to_end_stage1bc
#      and creates np1/ and np2/ in it; copies the mechanics decks of these cases and the
#      material files they name (props_cp_voce.txt, state_cp_voce.txt) into np2/.
#   2. test_microstructure_e2e_stage1bc_generate_<case>_np{1,2}: the generator with the same
#      arguments at 1 rank (in np1/) and at 2 ranks (in np2/). Each run must print the expected
#      final grain count (the number the mechanics decks' num_grains hard-code) followed by the
#      summary line of the feature it exercises (PASS_REGULAR_EXPRESSION), e.g. the split and
#      merged fragments for u_shape.
#   3. test_microstructure_e2e_stage1bc_tool_ranks_<case>: the two runs' mesh file (written with
#      compression = 0, so plain text), grain file (auto meshes), orientation file, grain table
#      and grain adjacency table compared byte for byte (cmake -E compare_files). They are
#      identical because every step of the generator is rank-count independent and the grain
#      statistics use fixed-point sums.
#   4. test_microstructure_e2e_stage1bc_mechanics_<deck>: mechanics on the np2/ outputs (decks
#      ms_u_shape_np1/np2, ms_periodic_np1/np2, ms_lognormal_np2, ms_columnar_np1,
#      ms_johnson_mehl_np2). A test passes when mechanics exits normally.
#   5. test_microstructure_e2e_stage1bc_compare_<case>_ranks: the u_shape and periodic results at 1
#      and at 2 ranks compared with test_parallel_mesh_input --compare-mechanics (the
#      test_mechanics.py tolerances).
# mechanics writes per-rank log files into its working directory, so RESOURCE_LOCK keeps CTest
# from running two mechanics steps of this chain at the same time under ctest -j.
#------------------------------------------------------------------------------
if(TARGET mechanics AND TARGET microstructure_generator
   AND NOT 2 GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
    set(_s1bc_dir  ${CMAKE_CURRENT_BINARY_DIR}/end_to_end_stage1bc)
    set(_s1bc_data ${PROJECT_SOURCE_DIR}/test/data)
    set(_s1bc_ms   ${_s1bc_data}/microstructure)

    # Step 1: a small CMake script, written here at configure time with the paths filled in
    set(_s1bc_setup_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1bc_setup.cmake)
    file(WRITE ${_s1bc_setup_script}
         "file(REMOVE_RECURSE \"${_s1bc_dir}\")\n"
         "file(MAKE_DIRECTORY \"${_s1bc_dir}/np1\" \"${_s1bc_dir}/np2\")\n"
         "file(GLOB _decks \"${_s1bc_ms}/mechanics/ms_u_shape_*.toml\" "
         "\"${_s1bc_ms}/mechanics/ms_periodic_*.toml\" "
         "\"${_s1bc_ms}/mechanics/ms_lognormal_*.toml\" "
         "\"${_s1bc_ms}/mechanics/ms_columnar_*.toml\" "
         "\"${_s1bc_ms}/mechanics/ms_johnson_mehl_*.toml\")\n"
         "file(COPY \${_decks} \"${_s1bc_data}/props_cp_voce.txt\" "
         "\"${_s1bc_data}/state_cp_voce.txt\" DESTINATION \"${_s1bc_dir}/np2\")\n")
    add_test(NAME    test_microstructure_e2e_stage1bc_setup
             COMMAND ${CMAKE_COMMAND} -P ${_s1bc_setup_script})
    set_tests_properties(test_microstructure_e2e_stage1bc_setup PROPERTIES
                         FIXTURES_SETUP ms_s1bc_case
                         LABELS         "microstructure")

    # Step 3 helper: compares the files named in FILES (comma-separated, so the list survives
    # the command line as one argument) between DIR1 and DIR2
    set(_s1bc_compare_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1bc_compare.cmake)
    file(WRITE ${_s1bc_compare_script}
         "set(_failed 0)\n"
         "string(REPLACE \",\" \";\" _files \"\${FILES}\")\n"
         "foreach(_f \${_files})\n"
         "  execute_process(COMMAND \${CMAKE_COMMAND} -E compare_files "
         "\"\${DIR1}/\${_f}\" \"\${DIR2}/\${_f}\" RESULT_VARIABLE _rc)\n"
         "  if(_rc EQUAL 0)\n"
         "    message(STATUS \"identical: \${_f}\")\n"
         "  else()\n"
         "    message(STATUS \"DIFFERENT or missing: \${_f}\")\n"
         "    set(_failed 1)\n"
         "  endif()\n"
         "endforeach()\n"
         "if(_failed)\n"
         "  message(FATAL_ERROR \"1-rank and 2-rank generator outputs differ\")\n"
         "endif()\n")

    # Steps 2 and 3 for one case.
    #   NAME    case name; INPUT generator options file; BASE output basename;
    #   MESH    optional mesh file (-m); AUTO set for auto-mesh cases (adds the grain file);
    #   GRAINS  final grain count the run must report (the mechanics decks hard-code it);
    #   REGEX   summary text of the feature, which the summary prints after the grain count.
    function(ms_stage1bc_case)
        cmake_parse_arguments(_s1bc "AUTO" "NAME;INPUT;BASE;MESH;GRAINS;REGEX" "" ${ARGN})
        # One expression for both checks (a list would pass when either matches); CMake's "."
        # also matches newlines, so ".*" spans the summary lines in between
        set(_pass "[^0-9]${_s1bc_GRAINS} final grains.*${_s1bc_REGEX}")
        set(_args -i ${_s1bc_INPUT})
        if(_s1bc_MESH)
            list(APPEND _args -m ${_s1bc_MESH})
        endif()
        foreach(_np 1 2)
            set(_test test_microstructure_e2e_stage1bc_generate_${_s1bc_NAME}_np${_np})
            blt_add_test(NAME              ${_test}
                         COMMAND           $<TARGET_FILE:microstructure_generator> ${_args}
                         NUM_MPI_TASKS     ${_np}
                         WORKING_DIRECTORY ${_s1bc_dir}/np${_np})
            set_tests_properties(${_test} PROPERTIES
                                 FIXTURES_REQUIRED       ms_s1bc_case
                                 FIXTURES_SETUP          ms_s1bc_${_s1bc_NAME}_np${_np}_outputs
                                 PASS_REGULAR_EXPRESSION "${_pass}"
                                 LABELS                  "microstructure")
        endforeach()

        set(_files ${_s1bc_BASE}.mesh ${_s1bc_BASE}.ori ${_s1bc_BASE}_grains.csv
                   ${_s1bc_BASE}_adjacency.csv)
        if(_s1bc_AUTO)
            list(APPEND _files ${_s1bc_BASE}.grains)
        endif()
        string(REPLACE ";" "," _files_arg "${_files}")
        set(_test test_microstructure_e2e_stage1bc_tool_ranks_${_s1bc_NAME})
        add_test(NAME    ${_test}
                 COMMAND ${CMAKE_COMMAND} -DDIR1=${_s1bc_dir}/np1 -DDIR2=${_s1bc_dir}/np2
                         -DFILES=${_files_arg} -P ${_s1bc_compare_script})
        set(_needs ms_s1bc_case ms_s1bc_${_s1bc_NAME}_np1_outputs ms_s1bc_${_s1bc_NAME}_np2_outputs)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "${_needs}"
                             LABELS            "microstructure")
    endfunction()

    ms_stage1bc_case(NAME  u_shape
               INPUT ${_s1bc_ms}/e2e/u_shape_fragments.toml
               MESH  ${_s1bc_ms}/u_shape_hex.mesh
               BASE  e2e_u_shape
               GRAINS 10
               REGEX "found, [1-9][0-9]* split into new grains, [1-9][0-9]* merged")
    ms_stage1bc_case(NAME  periodic AUTO
               INPUT ${_s1bc_ms}/e2e/periodic_box.toml
               BASE  e2e_periodic
               GRAINS 12
               REGEX "periodic: axes \\[xyz\\].* 0 unmatched")
    ms_stage1bc_case(NAME  lognormal AUTO
               INPUT ${_s1bc_ms}/e2e/lognormal_fit.toml
               BASE  e2e_lognormal
               GRAINS 23
               REGEX "element-level volume error vs targets")
    ms_stage1bc_case(NAME  columnar AUTO
               INPUT ${_s1bc_ms}/e2e/columnar.toml
               BASE  e2e_columnar
               GRAINS 10
               REGEX "grain statistics: ")
    ms_stage1bc_case(NAME  johnson_mehl AUTO
               INPUT ${_s1bc_ms}/e2e/johnson_mehl.toml
               BASE  e2e_johnson_mehl
               GRAINS 12
               REGEX "Johnson-Mehl nuclei: [0-9]+ \\(material fully transformed\\)")

    # Step 4: mechanics runs on the 2-rank outputs. Each entry is <deck>:<ranks>:<case>.
    set(_s1bc_runs ms_u_shape_np1:1:u_shape
                 ms_u_shape_np2:2:u_shape
                 ms_periodic_np1:1:periodic
                 ms_periodic_np2:2:periodic
                 ms_lognormal_np2:2:lognormal
                 ms_columnar_np1:1:columnar
                 ms_johnson_mehl_np2:2:johnson_mehl)
    foreach(_run ${_s1bc_runs})
        string(REPLACE ":" ";" _fields ${_run})
        list(GET _fields 0 _deck)
        list(GET _fields 1 _np)
        list(GET _fields 2 _case)
        set(_test test_microstructure_e2e_stage1bc_mechanics_${_deck})
        blt_add_test(NAME              ${_test}
                     COMMAND           $<TARGET_FILE:mechanics> -opt ${_deck}.toml
                     NUM_MPI_TASKS     ${_np}
                     WORKING_DIRECTORY ${_s1bc_dir}/np2)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "ms_s1bc_case;ms_s1bc_${_case}_np2_outputs"
                             FIXTURES_SETUP    ms_s1bc_run_${_deck}
                             RESOURCE_LOCK     ms_s1bc_mechanics
                             LABELS            "microstructure")
    endforeach()

    # Step 5: the same deck at 1 and at 2 ranks gives the same mechanics result
    if(TARGET test_parallel_mesh_input)
        foreach(_case u_shape periodic)
            set(_test test_microstructure_e2e_stage1bc_compare_${_case}_ranks)
            blt_add_test(NAME          ${_test}
                         COMMAND       $<TARGET_FILE:test_parallel_mesh_input> --compare-mechanics
                                       ${_s1bc_dir}/np2/results/ms_${_case}_np1
                                       ${_s1bc_dir}/np2/results/ms_${_case}_np2
                         NUM_MPI_TASKS 1)
            set(_needs ms_s1bc_case ms_s1bc_run_ms_${_case}_np1 ms_s1bc_run_ms_${_case}_np2)
            set_tests_properties(${_test} PROPERTIES
                                 FIXTURES_REQUIRED "${_needs}"
                                 LABELS            "microstructure")
        endforeach()
    endif()
endif()
