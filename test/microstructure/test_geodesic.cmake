# Vertex distance field and graph geodesic (ms_labeling: labeling/vertex_field,
# labeling/geodesic_graph).
#
# test_geodesic checks the lexicographic (T, label) exchange of shared vertices, the accuracy of
# the graph geodesic on a straight strut and around the inner corner of an L-shaped strut (hex and
# tet), geodesic vs Euclidean labels on the U-shaped part, the Additive boundary d1 - a1 =
# d2 - a2, seed snapping, a body without a seed and ranks without elements. Every run compares
# the distributed result with a single-rank reference computed in the same run, so np = 2, 3, 4
# are compared bitwise against np = 1.
#
# vertex_field uses seeding/material_point_locator (ms_seeding) for the seed-in-element test, so
# ms_seeding is listed here (see the WP1e.2 report: ms_labeling should list it too).
ms_add_gtest(NAME          test_geodesic
             SOURCES       test_geodesic.cpp
             DEPENDS_ON    ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)

# u_shape_hex.mesh lives in the source tree; its path is resolved from here
target_compile_definitions(test_geodesic PRIVATE
                           MS_GEODESIC_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
