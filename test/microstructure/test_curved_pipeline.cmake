# The curved end-to-end matrix: GenerateMicrostructure on genuinely curved hex and tet
# meshes of order 2 and 3 in the Gauss-Lobatto and closed-uniform bases, for the Euclidean
# (centroid and vote), connected and geodesic (FIM and graph) modes, Lloyd + volume fit,
# Johnson-Mehl and a passthrough attribute. Each run is compared exactly with a single-rank run in
# the same process, so np = 2, 3, 4 are checked against np = 1.
ms_add_gtest(NAME          test_curved_pipeline
             SOURCES       test_curved_pipeline.cpp
             DEPENDS_ON    exaconstit_microstructure ms_morphology ms_labeling ms_seeding
                           ms_geometry ms_mesh ms_io ms_options
             NUM_MPI_TASKS 1 2 3 4)
