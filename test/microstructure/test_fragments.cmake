# Mesh adjacency (ms_mesh: mesh_adjacency, element_key) and fragment cleanup (ms_labeling:
# fragment_cleanup).
#
# test_fragments checks the face adjacency (including neighbors on other ranks and face areas)
# against a single-rank reference, the partition independence of element keys, and the split /
# merge / small-grain rules of fragment cleanup on constructed labelings, random islands and a
# Voronoi tessellation of a U-shaped part. Every run compares the distributed result with a
# single-rank reference computed in the same run, so np = 2, 3, 4 are compared against np = 1.
ms_add_gtest(NAME          test_fragments
             SOURCES       test_fragments.cpp
             DEPENDS_ON    ms_labeling ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
