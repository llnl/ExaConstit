# Driver-level tests of the Stage 2 conformal grain boundary step ([Conformal] enabled) wired into
# GenerateMicrostructure (exaconstit_microstructure): a hex box without Nodes (the step runs, the
# fitted positions become the vertices, quality floor, smaller distance to the analytic boundaries,
# box volume kept, few label changes, nothing moves without the step), a tet box (tet floor), an
# order-2 mesh (fitted, keeps its Nodes) and the refusal of order 3, and the Phase 1 grain sources
# without analytic boundaries (voxel map, geodesic and connected labels on the U-shaped part:
# smoothed staircase targets) and a passthrough plate (interface kept in place), and that the
# nodes ExaConstit's mono_def boundary conditions select are the same before and after the fit
# (hex, tet, order 2), at np = 1, 2, 3, 4.
ms_add_gtest(NAME          test_driver_conformal
             SOURCES       test_driver_conformal.cpp
             DEPENDS_ON    exaconstit_microstructure ms_conformal ms_morphology ms_labeling
                           ms_seeding ms_geometry ms_mesh ms_io ms_options
             NUM_MPI_TASKS 1 2 3 4)

# The U-shaped mesh lives in the source tree
target_compile_definitions(test_driver_conformal PRIVATE
                           MS_DRIVER_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
