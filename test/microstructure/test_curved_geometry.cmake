# Curved-element geometry: surface clearance on warped bilinear and genuinely curved
# (bulged order-2) faces against brute-force and analytic distances, element volumes after serial,
# partitioned and ParPrint round trips of a bulged mesh, point-in-material classification next to
# a strongly curved face against a brute-force reference, the sampled curved element boxes, the
# preflight lattice check that catches an element folded between its quadrature points, and the
# high-order ParaView output. Runs at 1-4 MPI ranks.
ms_add_gtest(NAME          test_curved_geometry
             SOURCES       test_curved_geometry.cpp
             DEPENDS_ON    ms_seeding ms_mesh ms_io
             NUM_MPI_TASKS 1 2 3 4)
