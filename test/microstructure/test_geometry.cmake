# Geometry module (ms_geometry): tessellation metric functors and the exact nearest-seed search
# (SeedIndex). Serial: the search is not MPI-parallel (every rank builds its own index from the
# replicated seeds), so one rank checks everything. Only the SEQ execution space runs locally.
ms_add_gtest(NAME       test_geometry
             SOURCES    test_geometry.cpp
             DEPENDS_ON ms_geometry)
