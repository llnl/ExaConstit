# Build-system smoke test of the header-only ms_core contracts (exec dispatch, options and
# result defaults) and of the auto-mesh helper in src/utilities/auto_mesh.hpp. Serial checks only.
ms_add_gtest(NAME          test_skeleton
             SOURCES       test_skeleton.cpp
             NUM_MPI_TASKS 1)
