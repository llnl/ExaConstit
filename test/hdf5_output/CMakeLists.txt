# ------------------------------------------------------------------------------------------------
# HDF5 post-processing output ([PostProcessing] output_format = "hdf5", needs Axom)
#
# The multi-material regression deck with output_format = "hdf5" runs on 2 ranks, then
# check_hdf5_output.py finds a group in post_processing.hdf5 for every text reference file of the
# deck (test/data/test_results/multi_material_test: averages, statistics, histograms, surfaces)
# and compares every column, reading the file with scripts/postprocessing/read_post_processing.py.
#
# The run also writes checkpoints every 10 steps. Afterwards its file is kept as the reference,
# the same deck restarts from step 20 into the same results folder (whose file then still holds
# all 40 steps, as after a run that stopped past its last checkpoint), and
# compare_hdf5_restart.py checks that the continued file equals the reference: the steps after
# the checkpoint were dropped and written again once.
# ------------------------------------------------------------------------------------------------
set(EXACONSTIT_TEST_DATA ${CMAKE_SOURCE_DIR}/test/data)
set(_h5_dir ${CMAKE_CURRENT_BINARY_DIR}/multi_material)
set(_h5_file ${_h5_dir}/results/multi_material_test/post_processing.hdf5)
file(MAKE_DIRECTORY ${_h5_dir})
set(EXACONSTIT_RESTART_TABLE "[Restart]\n    write = true\n    interval = 10\n    keep_last = 0")
configure_file(multi_material_hdf5.toml.in ${_h5_dir}/multi_material_hdf5.toml @ONLY)
set(EXACONSTIT_RESTART_TABLE "[Restart]\n    read = true\n    read_cycle = 20")
configure_file(multi_material_hdf5.toml.in ${_h5_dir}/multi_material_hdf5_restart.toml @ONLY)

# Start from an empty results folder and no checkpoints
add_test(NAME    test_hdf5_output_clean
         COMMAND ${CMAKE_COMMAND} -E rm -rf ${_h5_dir}/results ${_h5_dir}/restart
                 ${_h5_dir}/reference.hdf5)
set_tests_properties(test_hdf5_output_clean PROPERTIES FIXTURES_SETUP hdf5_output_clean)

blt_add_test(NAME              test_hdf5_output_run
             COMMAND           $<TARGET_FILE:mechanics> -opt multi_material_hdf5.toml
             NUM_MPI_TASKS     2
             WORKING_DIRECTORY ${_h5_dir})
set_tests_properties(test_hdf5_output_run PROPERTIES
                     FIXTURES_REQUIRED hdf5_output_clean
                     FIXTURES_SETUP    hdf5_output_run)

add_test(NAME    test_hdf5_output
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_hdf5_output.py
                 ${_h5_file}
                 ${EXACONSTIT_TEST_DATA}/test_results/multi_material_test
                 ${_h5_dir}/results/multi_material_test
                 ${CMAKE_SOURCE_DIR}/scripts/postprocessing)
set_tests_properties(test_hdf5_output PROPERTIES
                     FIXTURES_REQUIRED hdf5_output_run
                     FIXTURES_SETUP    hdf5_output_checked)

# Restart from step 20 into the same results folder
add_test(NAME    test_hdf5_output_keep_reference
         COMMAND ${CMAKE_COMMAND} -E copy ${_h5_file} ${_h5_dir}/reference.hdf5)
set_tests_properties(test_hdf5_output_keep_reference PROPERTIES
                     FIXTURES_REQUIRED hdf5_output_checked
                     FIXTURES_SETUP    hdf5_output_reference)
blt_add_test(NAME              test_hdf5_output_restart_run
             COMMAND           $<TARGET_FILE:mechanics> -opt multi_material_hdf5_restart.toml
             NUM_MPI_TASKS     2
             WORKING_DIRECTORY ${_h5_dir})
set_tests_properties(test_hdf5_output_restart_run PROPERTIES
                     FIXTURES_REQUIRED hdf5_output_reference
                     FIXTURES_SETUP    hdf5_output_restart_run)
add_test(NAME    test_hdf5_output_restart
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare_hdf5_restart.py
                 ${_h5_dir}/reference.hdf5 ${_h5_file} ${CMAKE_SOURCE_DIR}/scripts/postprocessing)
set_tests_properties(test_hdf5_output_restart PROPERTIES
                     FIXTURES_REQUIRED hdf5_output_restart_run)
