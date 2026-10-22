#------------------------------------------------------------------------------
# End to end, Stage 1d: data-driven microstructures through the generator tool and mechanics
#
# Four cases, each with a hand-written generator input in test/data/microstructure/e2e/ (and,
# for the first three, hand-written mechanics decks in test/data/microstructure/mechanics/):
#   seed_table     a measured seed table (e2e/seed_table_roi.csv) on a 2 x 1 x 1 auto bar:
#                  orientations from the table, a region of interest with random filler grains
#                  outside it, phases 0 and 1 -> region 1 and phase 2 -> region 2;
#   voxel_dream3d  a two-phase DREAM.3D (SIMPL v6) voxel map with x y z w quaternions, on a hex
#                  mesh twice as coarse as the voxels (quadrature vote), one region per phase;
#   build_plate    a hex box whose bottom layers (attribute 2) are a passthrough build plate in
#                  its own region 2, the rest (attribute 1) tessellated;
#   converter      a SPPARKS site dump converted by scripts/microstructure/spparks_to_voxels.py,
#                  then read by the generator as a canonical voxel map (grain count checked, no
#                  mechanics).
#
# The tests form a chain through CTest fixtures (FIXTURES_SETUP / FIXTURES_REQUIRED: CTest runs
# them in order and skips a step whose prerequisite failed), like the Stage 1b/1c chain in
# test_tool_outputs.cmake:
#   1. test_microstructure_e2e_stage1d_setup: empties <build>/test/microstructure/
#      end_to_end_stage1d, creates inputs/, np1/ and np2/ in it, copies the seed table into
#      inputs/ and the mechanics decks with their material files (props_cp_voce.txt,
#      state_cp_voce.txt) into np2/.
#   2. test_microstructure_e2e_stage1d_inputs: test_tool_stage1d (a gtest executable, 1 rank,
#      working directory inputs/) writes the DREAM.3D file, the build-plate mesh and the SPPARKS
#      dump, and reads each back to check it.
#   3. test_microstructure_e2e_stage1d_convert_spparks: the converter turns the dump into
#      inputs/e2e_spparks.h5 (only when the configured Python has numpy and h5py; otherwise the
#      converter case is not registered and configure says so).
#   4. test_microstructure_e2e_stage1d_generate_<case>_np{1,2}: the generator with the same
#      input at 1 rank (in np1/) and at 2 ranks (in np2/). Each run must exit with status 0 AND
#      print the expected final grain count (the number the mechanics decks' num_grains
#      hard-code) followed by the summary text of the feature it exercises. CTest's
#      PASS_REGULAR_EXPRESSION alone would ignore the exit status (a run that prints the summary
#      and then aborts, e.g. in MPI_Finalize, would pass), and a FAIL_REGULAR_EXPRESSION can only
#      catch abort messages we thought of. So each run goes through a small `cmake -P` script
#      (microstructure_e2e_stage1d_run.cmake, written below) that runs the MPI command with
#      execute_process, prints its output, and fails unless the exit status is 0 and the output
#      matches the expected text.
#   5. test_microstructure_e2e_stage1d_tool_ranks_<case>: the two runs' mesh file (plain text,
#      compression = 0), grain file (auto meshes), orientation file, region map (two regions),
#      grain table and grain adjacency table compared byte for byte (cmake -E compare_files).
#   6. test_microstructure_e2e_stage1d_mechanics_<deck>: mechanics on the np2/ outputs with two
#      [[Materials]] blocks and the region map (decks ms_seed_table_np1/np2,
#      ms_voxel_dream3d_np2, ms_build_plate_np2). A test passes when mechanics exits normally.
#   7. test_microstructure_e2e_stage1d_compare_seed_table_ranks: the seed_table results at 1 and
#      at 2 ranks compared with test_parallel_mesh_input --compare-mechanics (the
#      test_mechanics.py tolerances).
# mechanics writes per-rank log files into its working directory, so RESOURCE_LOCK keeps CTest
# from running two mechanics steps of this chain at the same time under ctest -j.
#
# The voxel-map cases need HDF5 (AXOM_USE_HDF5, from Axom's CMake config); without it only
# seed_table and build_plate are registered.
#------------------------------------------------------------------------------
if(TARGET mechanics AND TARGET microstructure_generator
   AND NOT 2 GREATER EXACONSTIT_TEST_MAX_MPI_TASKS)
    set(_s1d_dir  ${CMAKE_CURRENT_BINARY_DIR}/end_to_end_stage1d)
    set(_s1d_data ${PROJECT_SOURCE_DIR}/test/data)
    set(_s1d_ms   ${_s1d_data}/microstructure)

    # Which optional cases this build can run
    set(_s1d_hdf5 OFF)
    if(AXOM_USE_HDF5)
        set(_s1d_hdf5 ON)
    else()
        message(STATUS "Stage 1d end-to-end: voxel-map cases not registered (Axom without HDF5)")
    endif()
    set(_s1d_converter OFF)
    if(_s1d_hdf5 AND PYTHON_EXECUTABLE)
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, h5py"
                        RESULT_VARIABLE _s1d_py_rc OUTPUT_QUIET ERROR_QUIET)
        if(_s1d_py_rc EQUAL 0)
            set(_s1d_converter ON)
        else()
            message(STATUS "Stage 1d end-to-end: converter case not registered "
                           "(${PYTHON_EXECUTABLE} lacks numpy or h5py)")
        endif()
    endif()

    # Step 1: a small CMake script, written here at configure time with the paths filled in
    set(_s1d_setup_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1d_setup.cmake)
    file(WRITE ${_s1d_setup_script}
         "file(REMOVE_RECURSE \"${_s1d_dir}\")\n"
         "file(MAKE_DIRECTORY \"${_s1d_dir}/inputs\" \"${_s1d_dir}/np1\" \"${_s1d_dir}/np2\")\n"
         "file(COPY \"${_s1d_ms}/e2e/seed_table_roi.csv\" DESTINATION \"${_s1d_dir}/inputs\")\n"
         "file(GLOB _decks \"${_s1d_ms}/mechanics/ms_seed_table_*.toml\" "
         "\"${_s1d_ms}/mechanics/ms_voxel_dream3d_*.toml\" "
         "\"${_s1d_ms}/mechanics/ms_build_plate_*.toml\")\n"
         "file(COPY \${_decks} \"${_s1d_data}/props_cp_voce.txt\" "
         "\"${_s1d_data}/state_cp_voce.txt\" DESTINATION \"${_s1d_dir}/np2\")\n")
    add_test(NAME    test_microstructure_e2e_stage1d_setup
             COMMAND ${CMAKE_COMMAND} -P ${_s1d_setup_script})
    set_tests_properties(test_microstructure_e2e_stage1d_setup PROPERTIES
                         FIXTURES_SETUP ms_s1d_case
                         LABELS         "microstructure")

    # Step 2: the input writer (a gtest executable; see test_tool_stage1d.cpp). ctest needs the
    # working directory to exist when the test starts, and the setup step deletes and recreates
    # it, so it is also created here at configure time (for `ctest -N` and direct runs).
    blt_add_executable(NAME       test_tool_stage1d
                       SOURCES    test_tool_stage1d.cpp
                       OUTPUT_DIR ${TEST_OUTPUT_DIR}
                       DEPENDS_ON ms_core ms_io ${MICROSTRUCTURE_DEPENDS} gtest)
    file(MAKE_DIRECTORY ${_s1d_dir}/inputs ${_s1d_dir}/np1 ${_s1d_dir}/np2)
    blt_add_test(NAME              test_microstructure_e2e_stage1d_inputs
                 COMMAND           test_tool_stage1d
                 NUM_MPI_TASKS     1
                 WORKING_DIRECTORY ${_s1d_dir}/inputs)
    set_tests_properties(test_microstructure_e2e_stage1d_inputs PROPERTIES
                         FIXTURES_REQUIRED ms_s1d_case
                         FIXTURES_SETUP    ms_s1d_inputs
                         LABELS            "microstructure")

    # Step 3: the SPPARKS converter
    if(_s1d_converter)
        add_test(NAME              test_microstructure_e2e_stage1d_convert_spparks
                 COMMAND           ${PYTHON_EXECUTABLE}
                                   ${PROJECT_SOURCE_DIR}/scripts/microstructure/spparks_to_voxels.py
                                   e2e_spparks.dump e2e_spparks.h5
                 WORKING_DIRECTORY ${_s1d_dir}/inputs)
        set_tests_properties(test_microstructure_e2e_stage1d_convert_spparks PROPERTIES
                             FIXTURES_REQUIRED "ms_s1d_case;ms_s1d_inputs"
                             FIXTURES_SETUP    ms_s1d_spparks_h5
                             LABELS            "microstructure")
    endif()

    # Step 5 helper: compares the files named in FILES (comma-separated, so the list survives
    # the command line as one argument) between DIR1 and DIR2
    set(_s1d_compare_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1d_compare.cmake)
    file(WRITE ${_s1d_compare_script}
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

    # Step 4 helper: runs CMD (the command with its arguments separated by "|", so the list
    # survives the command line as one argument) in the current directory, echoes stdout and
    # stderr (merged), and fails unless the exit status is 0 and the output matches PASS (a CMake
    # regular expression, in which "." also matches newlines)
    set(_s1d_run_script ${CMAKE_CURRENT_BINARY_DIR}/microstructure_e2e_stage1d_run.cmake)
    file(WRITE ${_s1d_run_script}
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
    # The MPI launch line, built exactly as blt_add_test builds it (BLTMacros.cmake, the
    # NUM_MPI_TASKS branch): MPIEXEC_EXECUTABLE (or the older MPIEXEC), MPIEXEC_NUMPROC_FLAG,
    # the rank count, BLT_MPI_COMMAND_APPEND, then the program
    if(MPIEXEC_EXECUTABLE)
        set(_s1d_mpiexec ${MPIEXEC_EXECUTABLE})
    else()
        set(_s1d_mpiexec ${MPIEXEC})
    endif()

    # Steps 4 and 5 for one case.
    #   NAME     case name; INPUT generator options file; BASE output basename;
    #   AUTO     set for auto-mesh cases (adds the grain file to the compared files);
    #   REGIONS  set for cases with more than one region (adds the region map);
    #   GRAINS   final grain count the run must report (the mechanics decks hard-code it);
    #   REGEX    summary text of the feature, which the summary prints after the grain count;
    #   NEEDS    extra fixtures the generator runs require (the written inputs).
    function(ms_stage1d_case)
        cmake_parse_arguments(_s1d "AUTO;REGIONS" "NAME;INPUT;BASE;GRAINS;REGEX" "NEEDS" ${ARGN})
        # One expression for both checks (a list would pass when either matches); CMake's "."
        # also matches newlines, so ".*" spans the summary lines in between
        set(_pass "[^0-9]${_s1d_GRAINS} final grains.*${_s1d_REGEX}")
        foreach(_np 1 2)
            set(_test test_microstructure_e2e_stage1d_generate_${_s1d_NAME}_np${_np})
            set(_cmd ${_s1d_mpiexec} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${BLT_MPI_COMMAND_APPEND}
                     $<TARGET_FILE:microstructure_generator> -i ${_s1d_INPUT})
            string(REPLACE ";" "|" _cmd_arg "${_cmd}")
            add_test(NAME              ${_test}
                     COMMAND           ${CMAKE_COMMAND} "-DCMD=${_cmd_arg}" "-DPASS=${_pass}"
                                       -P ${_s1d_run_script}
                     WORKING_DIRECTORY ${_s1d_dir}/np${_np})
            set_tests_properties(${_test} PROPERTIES
                                 FIXTURES_REQUIRED "ms_s1d_case;${_s1d_NEEDS}"
                                 FIXTURES_SETUP    ms_s1d_${_s1d_NAME}_np${_np}_outputs
                                 PROCESSORS        ${_np}
                                 LABELS            "microstructure")
        endforeach()

        set(_files ${_s1d_BASE}.mesh ${_s1d_BASE}.ori ${_s1d_BASE}_grains.csv
                   ${_s1d_BASE}_adjacency.csv)
        if(_s1d_AUTO)
            list(APPEND _files ${_s1d_BASE}.grains)
        endif()
        if(_s1d_REGIONS)
            list(APPEND _files ${_s1d_BASE}.regions)
        endif()
        string(REPLACE ";" "," _files_arg "${_files}")
        set(_test test_microstructure_e2e_stage1d_tool_ranks_${_s1d_NAME})
        add_test(NAME    ${_test}
                 COMMAND ${CMAKE_COMMAND} -DDIR1=${_s1d_dir}/np1 -DDIR2=${_s1d_dir}/np2
                         -DFILES=${_files_arg} -P ${_s1d_compare_script})
        set(_needs ms_s1d_case ms_s1d_${_s1d_NAME}_np1_outputs ms_s1d_${_s1d_NAME}_np2_outputs)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "${_needs}"
                             LABELS            "microstructure")
    endfunction()

    # The seed table: 6 table seeds (the 7th row is outside the ROI) + 6 filler seeds
    ms_stage1d_case(NAME   seed_table AUTO REGIONS
                    INPUT  ${_s1d_ms}/e2e/seed_table_roi.toml
                    BASE   e2e_seed_table
                    GRAINS 12
                    REGEX  "1 dropped outside the ROI.*, 6 placed.*regions: 2")
    # The two-phase DREAM.3D map: six features, two regions. The 2 x 2 x 2 hole of unassigned
    # voxels is outvoted by the 64-point vote of the coarse elements, so no element is filled
    # (asserted below); the fill is exercised by the converter case
    if(_s1d_hdf5)
        ms_stage1d_case(NAME   voxel_dream3d AUTO REGIONS
                        INPUT  ${_s1d_ms}/e2e/voxel_dream3d.toml
                        BASE   e2e_voxel_dream3d
                        GRAINS 6
                        REGEX  "voxel map: .*; 0 element\\(s\\) filled from.*regions: 2"
                        NEEDS  ms_s1d_inputs)
    endif()
    # The build plate: 8 grains in the part + the plate (grain 9, region 2, 128 elements)
    ms_stage1d_case(NAME   build_plate REGIONS
                    INPUT  ${_s1d_ms}/e2e/build_plate.toml
                    BASE   e2e_build_plate
                    GRAINS 9
                    REGEX  "attribute 2 -> grain 9 -> region 2 \\(128 elements\\)"
                    NEEDS  ms_s1d_inputs)
    # The converted SPPARKS dump: four column grains; the island of a repeated spin is merged
    if(_s1d_converter)
        ms_stage1d_case(NAME   converter AUTO
                        INPUT  ${_s1d_ms}/e2e/converter_spparks.toml
                        BASE   e2e_spparks
                        GRAINS 4
                        REGEX  "; 1 element\\(s\\) filled from their neighbors.*, 1 merged"
                        NEEDS  ms_s1d_inputs ms_s1d_spparks_h5)
    endif()

    # Step 6: mechanics runs on the 2-rank outputs. Each entry is <deck>:<ranks>:<case>.
    set(_s1d_runs ms_seed_table_np1:1:seed_table
                  ms_seed_table_np2:2:seed_table
                  ms_build_plate_np2:2:build_plate)
    if(_s1d_hdf5)
        list(APPEND _s1d_runs ms_voxel_dream3d_np2:2:voxel_dream3d)
    endif()
    foreach(_run ${_s1d_runs})
        string(REPLACE ":" ";" _fields ${_run})
        list(GET _fields 0 _deck)
        list(GET _fields 1 _np)
        list(GET _fields 2 _case)
        set(_test test_microstructure_e2e_stage1d_mechanics_${_deck})
        blt_add_test(NAME              ${_test}
                     COMMAND           $<TARGET_FILE:mechanics> -opt ${_deck}.toml
                     NUM_MPI_TASKS     ${_np}
                     WORKING_DIRECTORY ${_s1d_dir}/np2)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "ms_s1d_case;ms_s1d_${_case}_np2_outputs"
                             FIXTURES_SETUP    ms_s1d_run_${_deck}
                             RESOURCE_LOCK     ms_s1d_mechanics
                             LABELS            "microstructure")
    endforeach()

    # Step 7: the same deck at 1 and at 2 ranks gives the same mechanics result
    if(TARGET test_parallel_mesh_input)
        set(_test test_microstructure_e2e_stage1d_compare_seed_table_ranks)
        blt_add_test(NAME          ${_test}
                     COMMAND       $<TARGET_FILE:test_parallel_mesh_input> --compare-mechanics
                                   ${_s1d_dir}/np2/results/ms_seed_table_np1
                                   ${_s1d_dir}/np2/results/ms_seed_table_np2
                     NUM_MPI_TASKS 1)
        set(_needs ms_s1d_case ms_s1d_run_ms_seed_table_np1 ms_s1d_run_ms_seed_table_np2)
        set_tests_properties(${_test} PROPERTIES
                             FIXTURES_REQUIRED "${_needs}"
                             LABELS            "microstructure")
    endif()
endif()
