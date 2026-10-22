# Connected distance mode (ms_labeling: connected_labeler) and the exclusion-aware seed search
# (ms_geometry: SeedIndex::FindBestExcluding).
#
# test_connected checks FindBestExcluding against a brute-force loop for all four metrics, and the
# connected-cell constraint on the U-shaped part (test/data/microstructure/u_shape_hex.mesh) and on
# grid meshes against an independent serial reference implementation of the rounds written in the
# test (breadth-first search over the serial face graph, brute-force seed search). Every run also
# compares the distributed result (METIS partition, and a round-robin partition with an empty last
# rank) with a single-rank run in the same process, so np = 2, 3, 4 are compared against np = 1.
ms_add_gtest(NAME          test_connected
             SOURCES       test_connected.cpp
             DEPENDS_ON    ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)

# The U-shaped mesh lives in the source tree; its path is resolved from here
target_compile_definitions(test_connected PRIVATE
                           MS_CONNECTED_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
