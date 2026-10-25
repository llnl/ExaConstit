# Conformal fitting, Stage 2d: spring-balanced pseudo-elastic fitting and element quality
# (ms_conformal: conformal/conformal_fitter, conformal/quality).
#
# test_conformal_fit checks the quality measures on simple elements; fits an inclined Voronoi
# bicrystal on hex and tet meshes (quality floor, box face coordinates kept bit for bit, smaller
# deviation, analytic grain volume, WORLD vs SELF positions within the solver tolerance) and a
# 20-grain hex polycrystal (floor, almost no element below 0.5, smaller deviation), at np = 1, 2,
# 3, 4. ConformalFit.DumpSlices writes slice data for plots only when CONFORMAL_DUMP_DIR is set.
ms_add_gtest(NAME          test_conformal_fit
             SOURCES       test_conformal_fit.cpp
             DEPENDS_ON    ms_conformal ms_geometry ms_labeling ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
