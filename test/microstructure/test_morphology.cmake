# Size control (ms_morphology) and the cell measures it uses (ms_labeling): target grain volumes,
# Laguerre weight fitting (L-BFGS on the concave dual) and Lloyd relaxation.
#
# test_morphology checks analytic cell measures (a bisector that halves the cube), agreement with
# a brute-force sum, the size-target definitions, convergence and monotonicity of the weight fit
# (equal and log-normal targets, empty cells recovering), CVT energy decrease under Lloyd and
# clamping into a non-convex L-shaped block. Each test compares the distributed result bitwise
# with a single-rank reference computed in the same run, so every np variant also checks
# rank-count independence. Runs at 1-4 ranks.
ms_add_gtest(NAME          test_morphology
             SOURCES       test_morphology.cpp
             DEPENDS_ON    ms_morphology ms_seeding ms_labeling ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
