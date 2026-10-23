# Geodesic labelers on curved (high-order) meshes (ms_labeling: labeling/vertex_field with the
# nodal field, labeling/geodesic_fim and labeling/geodesic_graph).
#
# test_geodesic_curved checks the virtual lattice sub-cells (tiling, counts, enumeration, point
# location; hex and tet, order 2-3, Gauss-Lobatto / closed uniform / Bernstein Nodes), the FIM
# distance around the concave inner wall of a genuinely curved half annulus against the analytic
# tangent + arc + tangent path (error falling with p at fixed h; corner-only chords cutting
# through the hole vs the nodal field), a convex curved cylinder against the straight-line
# distance, rank-count independence (np = 2, 3, 4 bitwise against np = 1, FIM and graph, centroid
# and vote, Voronoi and Additive, a seed outside the material), and that straight-sided meshes
# with Nodes keep the corner field bit for bit.
ms_add_gtest(NAME          test_geodesic_curved
             SOURCES       test_geodesic_curved.cpp
             DEPENDS_ON    ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
