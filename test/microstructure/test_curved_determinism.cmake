# Curved-element determinism (WP-CV1): owner-broadcast vertex positions from the Nodes
# (mesh/vertex_coordinates), quantized element keys (mesh/element_key), element-weighted seeding
# and fragment cleanup on genuinely warped order-2/3 meshes, the hook-style path (SetCurvature +
# parallel refinement) against a single-rank run, and periodic face pairing on a curved periodic
# box, on a mesh with a stale vertex array, and its error cases. Every run compares the
# MPI_COMM_WORLD result with an MPI_COMM_SELF reference computed in the same executable.
ms_add_gtest(NAME          test_curved_determinism
             SOURCES       test_curved_determinism.cpp
             DEPENDS_ON    ms_labeling ms_seeding ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
