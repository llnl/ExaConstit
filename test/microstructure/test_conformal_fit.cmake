# Conformal fitting, Stage 2d: spring-balanced pseudo-elastic fitting and element quality
# (ms_conformal: conformal/conformal_fitter, conformal/quality).
#
# test_conformal_fit checks the quality measures on simple elements (scaled Jacobian, aspect ratio
# relative to the ideal element); fits an inclined Voronoi bicrystal on hex and tet meshes (quality
# limits, box face coordinates kept bit for bit, smaller deviation, analytic grain volume, WORLD vs
# SELF positions within the solver tolerance), a 20-grain hex polycrystal, a box with a domed top
# and tilted sides (boundary nodes slide on their surfaces and stay on them), thin grains (a two-
# element-thick grain keeps the limits and its volume, a one-element-thick grain is held as under-
# resolved), a jittered (unstructured-like) tet polycrystal with input tets below the floor, and an
# order-2 hex polycrystal (every nodal point fitted, no folding on a denser grid), at np = 1, 2, 3,
# 4. ConformalFit.DumpSlices writes slice data for plots only when CONFORMAL_DUMP_DIR is set.
ms_add_gtest(NAME          test_conformal_fit
             SOURCES       test_conformal_fit.cpp
             DEPENDS_ON    ms_conformal ms_geometry ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
