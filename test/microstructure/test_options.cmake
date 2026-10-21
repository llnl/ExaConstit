# Microstructure generator options (ms_options): TOML parsing, validation and CLI overrides
ms_add_gtest(NAME          test_options
             SOURCES       test_options.cpp
             DEPENDS_ON    ms_options
             NUM_MPI_TASKS 1)

# The example options files live in the source tree; their paths are resolved from here
target_compile_definitions(test_options PRIVATE
                           MS_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
