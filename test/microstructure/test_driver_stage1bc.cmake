# Driver-level tests of the Stage 1b/1c integration (exaconstit_microstructure): fragment cleanup
# policies on a non-convex part, quadrature voting, periodic microstructures (seed images and
# periodic face pairing), size targets with polydisperse placement and volume fitting, Lloyd and
# centroidal power relaxation, columnar/lamellar grains, Johnson-Mehl nucleation, the example
# options files of these features, and the shared utilities (collective errors, polydisperse RSA,
# the IntegrationRule sampler overload).
#
# Every driver run is compared with a single-rank reference computed in the same run, so the
# np = 2, 3, 4 variants check rank-count independence against np = 1.
ms_add_gtest(NAME          test_driver_stage1bc
             SOURCES       test_driver_stage1bc.cpp
             DEPENDS_ON    exaconstit_microstructure ms_morphology ms_labeling ms_seeding
                           ms_geometry ms_mesh ms_options
             NUM_MPI_TASKS 1 2 3 4)

# cube_tet.mesh and the example options files live in the source tree
target_compile_definitions(test_driver_stage1bc PRIVATE
                           MS_DRIVER_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
