# Straight-sided meshes (mesh/straight_sided.hpp): classification of meshes with and without
# Nodes (collectively over the ranks), corner points from the Nodes, and the effect on seeding and
# labeling: straight-sided Nodes of order 1 and 2 give the same seeds and labels as no Nodes,
# genuinely curved (bulged order-2) meshes keep the curved treatment. Runs at 1-4 MPI ranks.
ms_add_gtest(NAME          test_straight_sided
             SOURCES       test_straight_sided.cpp
             DEPENDS_ON    exaconstit_microstructure ms_seeding ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
