# Conformal fitting: spring-balanced pseudo-elastic fitting and element quality
# (ms_conformal: conformal/conformal_fitter, conformal/quality).
#
# test_conformal_fit checks the quality measures on simple elements (scaled Jacobian, aspect ratio
# relative to the ideal element); fits an inclined Voronoi bicrystal on hex and tet meshes (quality
# limits, box face coordinates kept bit for bit, smaller deviation, analytic grain volume, WORLD vs
# SELF positions within the solver tolerance), a 120° tricrystal (triple line nodes reach the
# analytic line, hex and tet), an Additive bicrystal against its exact grain volume and boundary
# area (fitted vs staircase, two resolutions), a 20-grain hex polycrystal, a box with a domed top
# and tilted sides (boundary nodes slide on their surfaces and stay on them, also along the rims
# where the dome meets the sides), a box with a bulged side wall (grain boundary nodes on the rims
# of the curved wall slide along them to the bisector), thin grains (a grain two elements thick
# keeps the limits and its volume, a grain one element thick is held as under-resolved), a
# jittered (unstructured-like) tet polycrystal with input tets below the floor, and an order-2 hex
# polycrystal (every nodal point fitted, no folding on a denser grid), at np = 1, 2, 3, 4.
# ConformalFit.DumpSlices writes slice data for plots only when CONFORMAL_DUMP_DIR is set.
ms_add_gtest(NAME          test_conformal_fit
             SOURCES       test_conformal_fit.cpp
             DEPENDS_ON    ms_conformal ms_geometry ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
