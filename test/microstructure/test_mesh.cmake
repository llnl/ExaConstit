# Mesh module (ms_mesh): serial element order and ParMesh construction, element and
# quadrature-point sampler, mesh preflight checks. Runs at 1-4 MPI ranks.
ms_add_gtest(NAME          test_mesh
             SOURCES       test_mesh.cpp
             DEPENDS_ON    ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
