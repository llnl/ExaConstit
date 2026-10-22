# Phases -> regions and synthetic phase fractions (ms_labeling: regions/region_assigner) and
# attribute domains (ms_mesh: mesh/attribute_domain).
#
# test_regions checks the phase -> region numbering (compact, sparse, explicit map, errors,
# non-crystalline flags), the accuracy bound, determinism and input-order independence of the
# phase-fraction assignment, and a build-plate mesh in which only the part above the plate is
# seeded, labeled and cleaned up on an mfem::ParSubMesh while the plate becomes one extra grain.
# Each run compares the distributed result with a single-rank reference computed in the same
# run, so np = 2, 3, 4 are compared against np = 1.
ms_add_gtest(NAME          test_regions
             SOURCES       test_regions.cpp
             DEPENDS_ON    ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
