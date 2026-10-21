# Seeding module (ms_seeding): rejection sampling, element-weighted sampling, RSA minimum
# distance, surface clearance and lattices. Each test compares the distributed result with a
# single-rank reference computed inside the same run, so every np variant checks rank-count
# independence. Runs at 1-4 MPI ranks.
ms_add_gtest(NAME          test_seeding
             SOURCES       test_seeding.cpp
             DEPENDS_ON    ms_seeding ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
