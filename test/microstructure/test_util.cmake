# Header-only utilities of ms_core and the orientation sampler, one test source per header:
# counter RNG, random distributions, random orientations and reproducible fixed-point sums.
# Linked into one executable (shared main in test_util_main.cpp); no module in DEPENDS_ON since
# all of them are header-only. Run at np 1-4 so the MPI fixed-point sum is checked at every rank
# count (the other tests run identically on every rank).
ms_add_gtest(NAME          test_util
             SOURCES       test_util_main.cpp
                           test_counter_rng.cpp
                           test_random_distributions.cpp
                           test_random_orientation.cpp
                           test_fixed_point.cpp
             NUM_MPI_TASKS 1 2 3 4)
