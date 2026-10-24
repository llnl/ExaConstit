# Element mask in seeding and morphology (WP-CV2a): seeds, surface clearance, weighted samples and
# Lloyd relaxation on a box whose bottom layers are inactive, compared with a mesh of only the
# active part, at np 1-4.
ms_add_gtest(NAME          test_mask_seeding
             SOURCES       test_mask_seeding.cpp
             DEPENDS_ON    ms_morphology ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
