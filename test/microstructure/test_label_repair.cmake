# Conformal fitting, Stage 2c: label repair (ms_conformal: conformal/label_repair).
#
# test_label_repair checks that a peninsula element flips back, that a stray element of a far
# grain flips to its surroundings and leaves no bad node, that a one-element grain is never
# emptied, that non-analytic grains are not flipped into, and that a perturbed 12-grain tet mesh
# is repaired identically on MPI_COMM_WORLD and MPI_COMM_SELF at np = 1, 2, 3, 4.
ms_add_gtest(NAME          test_label_repair
             SOURCES       test_label_repair.cpp
             DEPENDS_ON    ms_conformal ms_geometry ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
