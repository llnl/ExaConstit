# Driver-level tests of the Stage 1d integration (exaconstit_microstructure): seed tables (labels,
# quaternions normalized to unit length, phase -> region maps), the region of interest with random
# filler seeds, volume targets from the table with a Laguerre volume fit, voxel maps in the
# canonical and DREAM.3D layouts (regions from phases, declared quaternion layout, filled holes),
# synthetic phase fractions, a build plate kept as a passthrough region, and the refused
# combinations.
#
# The seed tables are written by the test into its working directory in the build tree; the voxel
# maps come from the shared fixture common/voxel_fixture.hpp. Every driver run is compared with a
# single-rank reference computed in the same run, so the np = 2, 3, 4 variants check rank-count
# independence against np = 1.
ms_add_gtest(NAME          test_driver_stage1d
             SOURCES       test_driver_stage1d.cpp
             DEPENDS_ON    exaconstit_microstructure ms_morphology ms_labeling ms_seeding
                           ms_geometry ms_mesh ms_io ms_options
             NUM_MPI_TASKS 1 2 3 4)

# The example options files and their seed table live in the source tree
target_compile_definitions(test_driver_stage1d PRIVATE
                           MS_DRIVER_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
