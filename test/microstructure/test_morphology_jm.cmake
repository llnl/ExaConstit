# Johnson-Mehl nucleation (seeding/johnson_mehl.*) and columnar / lamellar grains (global
# semidefinite quadric tensor with the projected-interval pruning bound in
# geometry/metric_functors.hpp and geometry/seed_index.*). The Johnson-Mehl tests compare the
# distributed result with a single-rank reference computed in the same run, so every np variant
# checks rank-count independence; the metric and search tests are serial and run on every rank.
# Runs at 1-4 MPI ranks.
ms_add_gtest(NAME          test_morphology_jm
             SOURCES       test_morphology_jm.cpp
             DEPENDS_ON    ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
