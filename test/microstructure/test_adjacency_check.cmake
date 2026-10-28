# Conformal fitting, Stage 2c, second part: the grain adjacency of the mesh compared with the
# analytic one (ms_conformal: conformal/adjacency_check).
#
# test_adjacency_check checks Voronoi slabs whose mesh and analytic graphs agree, the same slabs
# with one element layer relabeled (a known fake and a known missing pair), that pairs with a grain
# without seed are unchecked, and that the counts and examples of a random polycrystal equal the
# MPI_COMM_SELF ones at np = 1, 2, 3, 4.
ms_add_gtest(NAME          test_adjacency_check
             SOURCES       test_adjacency_check.cpp
             DEPENDS_ON    ms_conformal ms_geometry ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
