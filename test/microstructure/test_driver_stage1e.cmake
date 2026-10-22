# Driver-level tests of the Stage 1e distance modes wired into GenerateMicrostructure
# (exaconstit_microstructure): "euclidean", "connected" and "geodesic" on a U-shaped part (grains
# cross the notch only in euclidean mode), the driver's labels against direct calls of the labeling
# libraries, geodesic Johnson-Mehl growth, both modes on a part whose notch is a passthrough region
# (paths stay in the active part), a body that no seed reaches, the refused combinations and the
# Stage 1e example options files.
#
# Seed tables are written by the test into its working directory in the build tree. Every driver
# run is compared with a single-rank run in the same process, so the np = 2, 3, 4 variants check
# rank-count independence against np = 1.
ms_add_gtest(NAME          test_driver_stage1e
             SOURCES       test_driver_stage1e.cpp
             DEPENDS_ON    exaconstit_microstructure ms_morphology ms_labeling ms_seeding
                           ms_geometry ms_mesh ms_io ms_options
             NUM_MPI_TASKS 1 2 3 4)

# The example options files and the U-shaped mesh live in the source tree
target_compile_definitions(test_driver_stage1e PRIVATE
                           MS_DRIVER_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
