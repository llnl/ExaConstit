# Element labeling (ms_labeling), orientation assignment and the GenerateMicrostructure driver
# (exaconstit_microstructure), plus the end-to-end chain: microstructure_generator -> mechanics.
#
# test_labeling checks the labeler against a brute-force loop over all seeds, the compact grain
# ids, the orientations and the driver's rank-count independence (each run compares the
# distributed result with a single-rank reference computed in the same run). Runs at 1-4 ranks.
ms_add_gtest(NAME          test_labeling
             SOURCES       test_labeling.cpp
             DEPENDS_ON    exaconstit_microstructure ms_labeling ms_seeding ms_geometry ms_mesh
                           ms_options
             NUM_MPI_TASKS 1 2 3 4)

# cube_tet.mesh lives in the source tree; its path is resolved from here
target_compile_definitions(test_labeling PRIVATE
                           MS_LABELING_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")

#------------------------------------------------------------------------------
# End to end: generator outputs run through ExaConstit's mechanics driver
#
# The tests form a chain through CTest fixtures (FIXTURES_SETUP / FIXTURES_REQUIRED), which make
# CTest run them in order and skip a step whose prerequisite failed:
#   1. test_microstructure_e2e_setup: empties the scratch directory <build>/test/microstructure/
#      end_to_end and copies into it the hand-written mechanics decks
#      (test/data/microstructure/mechanics/*.toml) and the material files they name
#      (props_cp_voce.txt, state_cp_voce.txt). Everything runs there, never in the source tree.
#   2. test_microstructure_e2e_generate_{auto,tet}: the generator writes
#      - voronoi_auto.*: 8 grains on the 8x8x8 auto box (grain file + orientations), 1 rank;
#      - voronoi_cube_tet.*: 20 grains on test/data/cube_tet.mesh, serial gzip mesh plus a 2-part
#        decomposition (--parts 2), 2 ranks.
#      test_microstructure_e2e_generate_tet_np1 repeats the tet run with the same arguments on 1
#      rank in the sibling directory end_to_end_np1, and
#      test_microstructure_e2e_tool_rank_independence compares the two runs' serial mesh, both
#      parts, .ori file and grain table byte for byte. The gzip files are compared as written:
#      deflate is deterministic for the same input, level and zlib library, so identical
#      compressed bytes are expected and imply identical content (a stricter check than comparing
#      the decompressed text). The mesh header comments match too, because both runs get the
#      same arguments.
#   3. test_microstructure_e2e_mechanics_*: mechanics on each deck (auto at 1 and 2 ranks, the
#      serial tet mesh at 1 and 2 ranks, the 2-part decomposition with type = "parallel" at 2
#      ranks). A test passes when mechanics exits normally.
#   4. test_microstructure_e2e_compare_*: result files compared with test_parallel_mesh_input
#      --compare-mechanics (the test_mechanics.py tolerances):
#      - tet_parallel: serial file vs 2-part decomposition, both at 2 ranks;
#      - auto_ranks, tet_ranks: the same deck at 1 and at 2 ranks.
# mechanics writes per-rank log files into its working directory, so RESOURCE_LOCK keeps CTest
# from running two of the mechanics steps at the same time under ctest -j.
#------------------------------------------------------------------------------
if(TARGET mechanics AND TARGET microstructure_generator)
    set(_ms_e2e_dir  ${CMAKE_CURRENT_BINARY_DIR}/end_to_end)
    set(_ms_e2e_data ${PROJECT_SOURCE_DIR}/test/data)

    # Step 1 is a small CMake script, written here at configure time with the paths filled in
    set(_ms_e2e_setup_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_setup.cmake)
    set(_ms_e2e_np1_dir ${CMAKE_CURRENT_BINARY_DIR}/end_to_end_np1)
    file(WRITE ${_ms_e2e_setup_script}
         "file(REMOVE_RECURSE \"${_ms_e2e_dir}\" \"${_ms_e2e_np1_dir}\")\n"
         "file(MAKE_DIRECTORY \"${_ms_e2e_dir}\" \"${_ms_e2e_np1_dir}\")\n"
         "file(GLOB _decks \"${_ms_e2e_data}/microstructure/mechanics/*.toml\")\n"
         "file(COPY \${_decks} \"${_ms_e2e_data}/props_cp_voce.txt\" "
         "\"${_ms_e2e_data}/state_cp_voce.txt\" DESTINATION \"${_ms_e2e_dir}\")\n")
    add_test(NAME    test_microstructure_e2e_setup
             COMMAND ${CMAKE_COMMAND} -P ${_ms_e2e_setup_script})
    set_tests_properties(test_microstructure_e2e_setup PROPERTIES
                         FIXTURES_SETUP ms_e2e_case
                         LABELS         "microstructure")

    # Step 2: generator runs (they write different files, so they may run concurrently)
    blt_add_test(NAME              test_microstructure_e2e_generate_auto
                 COMMAND           $<TARGET_FILE:microstructure_generator>
                                   -i ${_ms_e2e_data}/microstructure/voronoi_auto.toml
                 NUM_MPI_TASKS     1
                 WORKING_DIRECTORY ${_ms_e2e_dir})
    set_tests_properties(test_microstructure_e2e_generate_auto PROPERTIES
                         FIXTURES_REQUIRED ms_e2e_case
                         FIXTURES_SETUP    ms_e2e_auto_outputs
                         LABELS            "microstructure")

    set(_ms_e2e_tet_np 2)
    if(_ms_e2e_tet_np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
        set(_ms_e2e_tet_np 1)
    endif()
    blt_add_test(NAME              test_microstructure_e2e_generate_tet
                 COMMAND           $<TARGET_FILE:microstructure_generator>
                                   -i ${_ms_e2e_data}/microstructure/voronoi_cube_tet.toml
                                   -m ${_ms_e2e_data}/cube_tet.mesh --parts 2
                 NUM_MPI_TASKS     ${_ms_e2e_tet_np}
                 WORKING_DIRECTORY ${_ms_e2e_dir})
    set_tests_properties(test_microstructure_e2e_generate_tet PROPERTIES
                         FIXTURES_REQUIRED ms_e2e_case
                         FIXTURES_SETUP    ms_e2e_tet_outputs
                         LABELS            "microstructure")

    # Step 2b: the tet run again on 1 rank, then a byte-for-byte comparison with the 2-rank run
    # (only meaningful when the main run used 2 ranks)
    if(_ms_e2e_tet_np EQUAL 2)
        blt_add_test(NAME              test_microstructure_e2e_generate_tet_np1
                     COMMAND           $<TARGET_FILE:microstructure_generator>
                                       -i ${_ms_e2e_data}/microstructure/voronoi_cube_tet.toml
                                       -m ${_ms_e2e_data}/cube_tet.mesh --parts 2
                     NUM_MPI_TASKS     1
                     WORKING_DIRECTORY ${_ms_e2e_np1_dir})
        set_tests_properties(test_microstructure_e2e_generate_tet_np1 PROPERTIES
                             FIXTURES_REQUIRED ms_e2e_case
                             FIXTURES_SETUP    ms_e2e_tet_np1_outputs
                             LABELS            "microstructure")

        set(_ms_e2e_compare_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_compare.cmake)
        file(WRITE ${_ms_e2e_compare_script}
             "set(_failed 0)\n"
             "foreach(_f voronoi_cube_tet.mesh.gz voronoi_cube_tet.mesh.000000 "
             "voronoi_cube_tet.mesh.000001 voronoi_cube_tet.ori voronoi_cube_tet_grains.csv)\n"
             "  execute_process(COMMAND \${CMAKE_COMMAND} -E compare_files "
             "\"${_ms_e2e_np1_dir}/\${_f}\" \"${_ms_e2e_dir}/\${_f}\" RESULT_VARIABLE _rc)\n"
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
        add_test(NAME    test_microstructure_e2e_tool_rank_independence
                 COMMAND ${CMAKE_COMMAND} -P ${_ms_e2e_compare_script})
        set_tests_properties(test_microstructure_e2e_tool_rank_independence PROPERTIES
                             FIXTURES_REQUIRED
                                 "ms_e2e_case;ms_e2e_tet_outputs;ms_e2e_tet_np1_outputs"
                             LABELS            "microstructure")
    endif()

    # Step 3: mechanics runs. Each entry is <deck>:<ranks>:<generator fixture>.
    set(_ms_e2e_runs ms_auto_np1:1:ms_e2e_auto_outputs
                     ms_auto_np2:2:ms_e2e_auto_outputs
                     ms_tet_np1:1:ms_e2e_tet_outputs
                     ms_tet_np2:2:ms_e2e_tet_outputs
                     ms_tet_parallel_np2:2:ms_e2e_tet_outputs)
    foreach(_run ${_ms_e2e_runs})
        string(REPLACE ":" ";" _fields ${_run})
        list(GET _fields 0 _deck)
        list(GET _fields 1 _np)
        list(GET _fields 2 _outputs)
        if(_np GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
            continue()
        endif()
        set(_test test_microstructure_e2e_mechanics_${_deck})
        blt_add_test(NAME              ${_test}
                     COMMAND           $<TARGET_FILE:mechanics> -opt ${_deck}.toml
                     NUM_MPI_TASKS     ${_np}
                     WORKING_DIRECTORY ${_ms_e2e_dir})
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "ms_e2e_case;${_outputs}"
                             RESOURCE_LOCK     ms_e2e_case
                             LABELS            "microstructure")
        # Each run sets up a fixture named after its deck, which the comparisons require
        set_tests_properties(${_test} PROPERTIES FIXTURES_SETUP ms_e2e_run_${_deck})
    endforeach()

    # Step 4: serial file vs parallel decomposition (needs the 2-rank runs and the comparison
    # helper of test/test_parallel_mesh_input.cpp)
    # Each entry is <name>:<deck a>:<deck b>
    if(TARGET test_parallel_mesh_input AND NOT 2 GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
        set(_ms_e2e_comparisons tet_parallel:ms_tet_np2:ms_tet_parallel_np2
                                auto_ranks:ms_auto_np1:ms_auto_np2
                                tet_ranks:ms_tet_np1:ms_tet_np2)
        foreach(_cmp ${_ms_e2e_comparisons})
            string(REPLACE ":" ";" _fields ${_cmp})
            list(GET _fields 0 _name)
            list(GET _fields 1 _deck_a)
            list(GET _fields 2 _deck_b)
            blt_add_test(NAME          test_microstructure_e2e_compare_${_name}
                         COMMAND       $<TARGET_FILE:test_parallel_mesh_input> --compare-mechanics
                                       ${_ms_e2e_dir}/results/${_deck_a}
                                       ${_ms_e2e_dir}/results/${_deck_b}
                         NUM_MPI_TASKS 1)
            set_tests_properties(test_microstructure_e2e_compare_${_name} PROPERTIES
                                 FIXTURES_REQUIRED
                                     "ms_e2e_case;ms_e2e_run_${_deck_a};ms_e2e_run_${_deck_b}"
                                 LABELS            "microstructure")
        endforeach()
    endif()
endif()
