# Microstructure converters (scripts/microstructure/*.py): Python unittest suite.
#
# test_microstructure_converters runs test/microstructure/python/test_converters.py with the
# configured Python. The test writes tiny HEXRD, MIDAS, SPPARKS and ExaCA inputs into its working
# directory (<build>/test/microstructure/converters/converters_test_output), runs each converter
# script on them, and checks the seed tables / voxel maps they write (values, quaternion math,
# required-flag errors). Tests that write HDF5 are skipped, not failed, when h5py is missing.
#
# No C++ target is involved, so the test needs no build step. Like every microstructure test it
# is registered only when EXACONSTIT_ENABLE_EXTENSIVE_TESTS is ON (the gate at the top of
# CMakeLists.txt in this directory).
if(PYTHON_EXECUTABLE)
    set(_ms_conv_dir ${CMAKE_CURRENT_BINARY_DIR}/converters)
    # ctest needs the working directory to exist before the test starts
    file(MAKE_DIRECTORY ${_ms_conv_dir})
    add_test(NAME              test_microstructure_converters
             COMMAND           ${PYTHON_EXECUTABLE}
                               ${PROJECT_SOURCE_DIR}/test/microstructure/python/test_converters.py
             WORKING_DIRECTORY ${_ms_conv_dir})
    # The test finds the scripts through this variable (it falls back to a path relative to
    # itself when run by hand)
    set_tests_properties(test_microstructure_converters PROPERTIES
                         ENVIRONMENT "EXACONSTIT_CONVERTER_DIR=${PROJECT_SOURCE_DIR}/scripts/microstructure"
                         LABELS      "microstructure")
else()
    message(STATUS "test_microstructure_converters not registered: PYTHON_EXECUTABLE is not set")
endif()
