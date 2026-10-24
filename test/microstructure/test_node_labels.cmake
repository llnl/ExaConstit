# Conformal fitting, Stage 2a: node label sets (ms_conformal: conformal/node_labels).
#
# test_node_labels compares the label set and boundary surface count of every node of a
# round-robin distributed mesh with a serial reference computed in the test (hex, tet and order-2
# Nodes), the global class counts of a bicrystal, a tricrystal on a fourth grain and four
# quadrants (over-constrained nodes), the feature angle on a creased top face, and the input
# checks. The reference does not depend on the partition, so np = 1, 2, 3, 4 check rank
# invariance.
ms_add_gtest(NAME          test_node_labels
             SOURCES       test_node_labels.cpp
             DEPENDS_ON    ms_conformal ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
