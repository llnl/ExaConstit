# Tests of the microstructure output writer (ms_io): serial / partitioned / ParPrint meshes,
# orientation file, region map, grain file, grain table CSV, ParaView, and the exception-safe gzip
# writer. Each file is read back with the reader ExaConstit uses. Runs at np = 1..4 because the
# mesh writers are collective.
ms_add_gtest(NAME          test_io
             SOURCES       test_io.cpp
             DEPENDS_ON    ms_io
             NUM_MPI_TASKS 1 2 3 4)
