/*
 * gtest driver shared by the util/orientation test sources (test_counter_rng.cpp,
 * test_random_distributions.cpp, test_random_orientation.cpp, test_fixed_point.cpp), which are
 * linked into one executable. MFEM's MPI wrapper initializes MPI (needed by the fixed-point MPI
 * test) and finalizes it at exit.
 */
#include "mfem.hpp"

#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
