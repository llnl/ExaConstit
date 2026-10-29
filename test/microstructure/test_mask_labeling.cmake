# Element mask in labeling, distance modes and cleanup: Euclidean, connected and
# geodesic labels and fragment cleanup on an inverted-U part above an inactive plate, compared
# with a mesh of only the active part, at np 1-4.
ms_add_gtest(NAME          test_mask_labeling
             SOURCES       test_mask_labeling.cpp
             DEPENDS_ON    ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
