# Driver-level tests of the Stage 2 conformal grain boundary step ([Conformal] enabled) wired into
# GenerateMicrostructure (exaconstit_microstructure): a hex box without Nodes (the step runs, the
# fitted positions become the vertices, quality floor, smaller distance to the analytic
# boundaries, box volume kept, few label changes, nothing moves without the step), a tet box
# (tet floor) and the refusal of order-2 meshes, at np = 1, 2, 3, 4.
ms_add_gtest(NAME          test_driver_conformal
             SOURCES       test_driver_conformal.cpp
             DEPENDS_ON    exaconstit_microstructure ms_conformal ms_morphology ms_labeling
                           ms_seeding ms_geometry ms_mesh ms_io ms_options
             NUM_MPI_TASKS 1 2 3 4)
