#------------------------------------------------------------------------------
# End to end, Stage 1e: the distance modes for parts with holes (connected, geodesic) through the
# generator tool and mechanics
#
# Three cases on the U-shaped part test/data/microstructure/u_shape_hex.mesh (the box
# [0,3] x [0,2] x [0,1] without the notch [1,2] x [1,2] x [0,1]), each with a hand-written
# generator input in test/data/microstructure/e2e/ (and, for the first two, hand-written mechanics
# decks in test/data/microstructure/mechanics/):
#   connected_ushape     distance = "connected": 8 seeds drawn in the bounding box, so several
#                        straight-line cells reach across the notch; the connected mode moves the
#                        cut-off elements to other seeds (the summary must report a nonzero
#                        relabeled count); mechanics at 2 ranks;
#   geodesic_fim_ushape  distance = "geodesic" with the Fast Iterative Method solver and
#                        quadrature-vote sampling, same seeds; mechanics at 1 and at 2 ranks, and
#                        the two mechanics results compared;
#   geodesic_graph_jm    Johnson-Mehl nucleation with geodesic (additive) arrival times from the
#                        graph solver; generator only.
#
# The tests form a chain through CTest fixtures (FIXTURES_SETUP / FIXTURES_REQUIRED: CTest runs
# them in order and skips a step whose prerequisite failed), like the Stage 1d chain in
# test_tool_stage1d.cmake:
#   1. test_microstructure_e2e_stage1e_setup: empties <build>/test/microstructure/
#      end_to_end_stage1e, creates np1/ and np2/ in it, and copies the mechanics decks with their
#      material files (props_cp_voce.txt, state_cp_voce.txt) into np2/.
#   2. test_microstructure_e2e_stage1e_generate_<case>_np{1,2}: the generator with the same input
#      (and -m u_shape_hex.mesh) at 1 rank (in np1/) and at 2 ranks (in np2/). Each run must exit
#      with status 0 AND print the expected final grain count followed by the distance-mode line
#      of the summary. As in Stage 1d the run goes through a small `cmake -P` script
#      (microstructure_e2e_stage1e_run.cmake, written below) that runs the MPI command with
#      execute_process, prints its output, and fails unless the exit status is 0 and the output
#      matches the expected text (PASS_REGULAR_EXPRESSION alone would ignore the exit status).
#   3. test_microstructure_e2e_stage1e_tool_ranks_<case>: the two runs' mesh file (plain text,
#      compression = 0), orientation file, grain table and grain adjacency table compared byte for
#      byte (cmake -E compare_files). The same step checks the 1-rank mesh header: it must record
#      the distance mode (and the geodesic solver) and must not mention rounds, whose counts depend
#      on the rank count for the graph solver.
#   4. test_microstructure_e2e_stage1e_mechanics_<deck>: mechanics on the np2/ outputs
#      (ms_connected_ushape_np2, ms_geodesic_fim_ushape_np1/np2). A test passes when mechanics
#      exits normally.
#   5. test_microstructure_e2e_stage1e_compare_geodesic_fim_ranks: the geodesic_fim_ushape
#      mechanics results at 1 and at 2 ranks compared with test_parallel_mesh_input
#      --compare-mechanics (the test_mechanics.py tolerances).
# mechanics writes per-rank log files into its working directory, so RESOURCE_LOCK keeps CTest
# from running two mechanics steps of this chain at the same time under ctest -j.
#------------------------------------------------------------------------------
if(TARGET mechanics AND TARGET microstructure_generator
   AND NOT 2 GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
    set(_s1e_dir  ${CMAKE_CURRENT_BINARY_DIR}/end_to_end_stage1e)
    set(_s1e_data ${PROJECT_SOURCE_DIR}/test/data)
    set(_s1e_ms   ${_s1e_data}/microstructure)
    set(_s1e_mesh ${_s1e_ms}/u_shape_hex.mesh)

    # Step 1: a small CMake script, written here at configure time with the paths filled in
    set(_s1e_setup_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1e_setup.cmake)
    file(WRITE ${_s1e_setup_script}
         "file(REMOVE_RECURSE \"${_s1e_dir}\")\n"
         "file(MAKE_DIRECTORY \"${_s1e_dir}/np1\" \"${_s1e_dir}/np2\")\n"
         "file(GLOB _decks \"${_s1e_ms}/mechanics/ms_connected_ushape_*.toml\" "
         "\"${_s1e_ms}/mechanics/ms_geodesic_fim_ushape_*.toml\")\n"
         "file(COPY \${_decks} \"${_s1e_data}/props_cp_voce.txt\" "
         "\"${_s1e_data}/state_cp_voce.txt\" DESTINATION \"${_s1e_dir}/np2\")\n")
    add_test(NAME    test_microstructure_e2e_stage1e_setup
             COMMAND ${CMAKE_COMMAND} -P ${_s1e_setup_script})
    set_tests_properties(test_microstructure_e2e_stage1e_setup PROPERTIES
                         FIXTURES_SETUP ms_s1e_case
                         LABELS         "microstructure")
    # ctest needs the working directories to exist when a test starts, and the setup step deletes
    # and recreates them, so they are also created here at configure time (for `ctest -N`)
    file(MAKE_DIRECTORY ${_s1e_dir}/np1 ${_s1e_dir}/np2)

    # Step 2 helper: runs CMD (the command with its arguments separated by "|", so the list
    # survives the command line as one argument) in the current directory, echoes stdout and
    # stderr (merged), and fails unless the exit status is 0 and the output matches PASS (a CMake
    # regular expression, in which "." also matches newlines)
    set(_s1e_run_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1e_run.cmake)
    file(WRITE ${_s1e_run_script}
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

    # Step 3 helper: compares the files named in FILES (comma-separated, so the list survives the
    # command line as one argument) between DIR1 and DIR2, then checks the "# ..." comment lines
    # of DIR1/HEADER_FILE (the mesh header): they must match HEADER and must not contain "round"
    set(_s1e_compare_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1e_compare.cmake)
    file(WRITE ${_s1e_compare_script}
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
         "file(STRINGS \"\${DIR1}/\${HEADER_FILE}\" _header REGEX \"^#\")\n"
         "string(REPLACE \";\" \"\\n\" _header \"\${_header}\")\n"
         "if(NOT _header MATCHES \"\${HEADER}\")\n"
         "  message(STATUS \"mesh header lacks: \${HEADER}\")\n"
         "  set(_failed 1)\n"
         "endif()\n"
         "if(_header MATCHES \"round\")\n"
         "  message(STATUS \"mesh header mentions rounds (rank-count dependent)\")\n"
         "  set(_failed 1)\n"
         "endif()\n"
         "if(_failed)\n"
         "  message(FATAL_ERROR \"Stage 1e generator outputs: see the lines above\")\n"
         "endif()\n")

    # The MPI launch line, built exactly as blt_add_test builds it (BLTMacros.cmake, the
    # NUM_MPI_TASKS branch): MPIEXEC_EXECUTABLE (or the older MPIEXEC), MPIEXEC_NUMPROC_FLAG,
    # the rank count, BLT_MPI_COMMAND_APPEND, then the program
    if(MPIEXEC_EXECUTABLE)
        set(_s1e_mpiexec ${MPIEXEC_EXECUTABLE})
    else()
        set(_s1e_mpiexec ${MPIEXEC})
    endif()

    # Steps 2 and 3 for one case.
    #   NAME    case name; INPUT generator options file; BASE output basename;
    #   GRAINS  final grain count the run must report (the mechanics decks hard-code it);
    #   REGEX   summary text of the distance mode, which the summary prints after the grain count;
    #   HEADER  text the mesh header must contain (the distance mode and solver).
    function(ms_stage1e_case)
        cmake_parse_arguments(_s1e "" "NAME;INPUT;BASE;GRAINS;REGEX;HEADER" "" ${ARGN})
        # One expression for both checks (a list would pass when either matches); CMake's "."
        # also matches newlines, so ".*" spans the summary lines in between
        set(_pass "[^0-9]${_s1e_GRAINS} final grains.*${_s1e_REGEX}")
        foreach(_np 1 2)
            set(_test test_microstructure_e2e_stage1e_generate_${_s1e_NAME}_np${_np})
            set(_cmd ${_s1e_mpiexec} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${BLT_MPI_COMMAND_APPEND}
                     $<TARGET_FILE:microstructure_generator> -i ${_s1e_INPUT} -m ${_s1e_mesh})
            string(REPLACE ";" "|" _cmd_arg "${_cmd}")
            add_test(NAME              ${_test}
                     COMMAND           ${CMAKE_COMMAND} "-DCMD=${_cmd_arg}" "-DPASS=${_pass}"
                                       -P ${_s1e_run_script}
                     WORKING_DIRECTORY ${_s1e_dir}/np${_np})
            set_tests_properties(${_test} PROPERTIES
                                 FIXTURES_REQUIRED ms_s1e_case
                                 FIXTURES_SETUP    ms_s1e_${_s1e_NAME}_np${_np}_outputs
                                 PROCESSORS        ${_np}
                                 LABELS            "microstructure")
        endforeach()

        set(_files ${_s1e_BASE}.mesh ${_s1e_BASE}.ori ${_s1e_BASE}_grains.csv
                   ${_s1e_BASE}_adjacency.csv)
        string(REPLACE ";" "," _files_arg "${_files}")
        set(_test test_microstructure_e2e_stage1e_tool_ranks_${_s1e_NAME})
        add_test(NAME    ${_test}
                 COMMAND ${CMAKE_COMMAND} -DDIR1=${_s1e_dir}/np1 -DDIR2=${_s1e_dir}/np2
                         -DFILES=${_files_arg} -DHEADER_FILE=${_s1e_BASE}.mesh
                         "-DHEADER=${_s1e_HEADER}" -P ${_s1e_compare_script})
        set(_needs ms_s1e_case ms_s1e_${_s1e_NAME}_np1_outputs ms_s1e_${_s1e_NAME}_np2_outputs)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "${_needs}"
                             LABELS            "microstructure")
    endfunction()

    # The regular expressions avoid ";" (it would split the -DPASS argument into a CMake list)
    # and "\n" (CMake's "." already matches newlines, see above).
    set(_s1e_round_regex "[0-9]+ distance round\\(s\\), [0-9]+ label round\\(s\\)")
    string(CONCAT _s1e_fim_regex "solver fim: ${_s1e_round_regex}[^0-9]+"
                  "[0-9]+ seed\\(s\\) outside the material, 0 unreached element")
    string(CONCAT _s1e_graph_regex "solver graph: ${_s1e_round_regex} "
                  "\\(round counts depend on the rank count\\)")
    # Connected: 8 seeds, 32 elements cut off across the notch are relabeled (measured), so the
    # cleanup finds no extra piece; the regex requires a relabeled count of at least 1
    ms_stage1e_case(NAME   connected_ushape
                    INPUT  ${_s1e_ms}/e2e/connected_ushape.toml
                    BASE   e2e_connected_ushape
                    GRAINS 8
                    REGEX  "relabeling round\\(s\\), [1-9][0-9]* element\\(s\\) relabeled"
                    HEADER "# distance: connected")
    # Geodesic, FIM solver with vote sampling: every element reached, 8 grains
    ms_stage1e_case(NAME   geodesic_fim_ushape
                    INPUT  ${_s1e_ms}/e2e/geodesic_fim_ushape.toml
                    BASE   e2e_geodesic_fim_ushape
                    GRAINS 8
                    REGEX  "${_s1e_fim_regex}"
                    HEADER "# distance: geodesic \\(solver fim\\)")
    # Johnson-Mehl along paths, graph solver: 18 nuclei, 2 win no element, 16 grains
    ms_stage1e_case(NAME   geodesic_graph_jm
                    INPUT  ${_s1e_ms}/e2e/geodesic_graph_jm.toml
                    BASE   e2e_geodesic_graph_jm
                    GRAINS 16
                    REGEX  "${_s1e_graph_regex}"
                    HEADER "# metric: additive.# distance: geodesic \\(solver graph\\)")

    # Step 4: mechanics runs on the 2-rank outputs. Each entry is <deck>:<ranks>:<case>.
    set(_s1e_runs ms_connected_ushape_np2:2:connected_ushape
                  ms_geodesic_fim_ushape_np1:1:geodesic_fim_ushape
                  ms_geodesic_fim_ushape_np2:2:geodesic_fim_ushape)
    foreach(_run ${_s1e_runs})
        string(REPLACE ":" ";" _fields ${_run})
        list(GET _fields 0 _deck)
        list(GET _fields 1 _np)
        list(GET _fields 2 _case)
        set(_test test_microstructure_e2e_stage1e_mechanics_${_deck})
        blt_add_test(NAME              ${_test}
                     COMMAND           $<TARGET_FILE:mechanics> -opt ${_deck}.toml
                     NUM_MPI_TASKS     ${_np}
                     WORKING_DIRECTORY ${_s1e_dir}/np2)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "ms_s1e_case;ms_s1e_${_case}_np2_outputs"
                             FIXTURES_SETUP    ms_s1e_run_${_deck}
                             RESOURCE_LOCK     ms_s1e_mechanics
                             LABELS            "microstructure")
    endforeach()

    # Step 5: the same geodesic microstructure at 1 and at 2 ranks gives the same mechanics result
    if(TARGET test_parallel_mesh_input)
        set(_test test_microstructure_e2e_stage1e_compare_geodesic_fim_ranks)
        blt_add_test(NAME          ${_test}
                     COMMAND       $<TARGET_FILE:test_parallel_mesh_input> --compare-mechanics
                                   ${_s1e_dir}/np2/results/ms_geodesic_fim_ushape_np1
                                   ${_s1e_dir}/np2/results/ms_geodesic_fim_ushape_np2
                     NUM_MPI_TASKS 1)
        set(_needs ms_s1e_case ms_s1e_run_ms_geodesic_fim_ushape_np1
                   ms_s1e_run_ms_geodesic_fim_ushape_np2)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "${_needs}"
                             LABELS            "microstructure")
    endif()
endif()
