# Fast Iterative Method eikonal geodesic labeler (ms_labeling: labeling/geodesic_fim,
# labeling/eikonal_local_solver.hpp).
#
# test_geodesic_fim checks the local eikonal update against brute-force minimization over random
# faces (including unreached vertices and a run inside an mfem::forall kernel), the accuracy of
# the FIM distances on a straight strut and around the inner corner of an L-shaped strut (hex and
# tet), geodesic vs Euclidean labels on the U-shaped part, the Additive boundary d1 - a1 =
# d2 - a2, seed snapping, a body without a seed, ranks without elements and invalid input. Every
# labeler run compares the distributed result (and the round counts) with a single-rank reference
# computed in the same run, so np = 2, 3, 4 are compared bitwise against np = 1.
ms_add_gtest(NAME          test_geodesic_fim
             SOURCES       test_geodesic_fim.cpp
             DEPENDS_ON    ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)

# u_shape_hex.mesh lives in the source tree; its path is resolved from here
target_compile_definitions(test_geodesic_fim PRIVATE
                           MS_GEODESIC_FIM_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")

# The serial label reference in the test calls the local solver functions from its own code and
# compares with the library bitwise; give it the same floating-point contraction rule as the
# library (src/microstructure/CMakeLists.txt applies -ffp-contract=off when STRICT_FP is ON).
if(MICROSTRUCTURE_STRICT_FP)
    target_compile_options(test_geodesic_fim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()
