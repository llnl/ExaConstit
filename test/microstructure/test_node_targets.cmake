# Conformal fitting: targets on the analytic grain boundaries (ms_conformal:
# conformal/manifold_projector.hpp, conformal/node_targets).
#
# test_node_targets checks the projector on planes and dependent rows; the targets of a Voronoi
# bicrystal (orthogonal projection, box coordinates kept bit for bit), a Laguerre quad point and its
# triple lines (tets), and an Additive (curved) boundary; the validation (foreign seed, same seed,
# non-analytic grain, curved boundary); smoothed staircase targets without seeds (order 1 and 2);
# and that the distributed targets equal the MPI_COMM_SELF ones bit for bit at np = 1, 2, 3, 4.
ms_add_gtest(NAME          test_node_targets
             SOURCES       test_node_targets.cpp
             DEPENDS_ON    ms_conformal ms_geometry ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
