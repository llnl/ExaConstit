# Sampling modes of the element labeling: quadrature-point voting (labeling/quadrature_vote) and
# periodic seed images (geometry/periodic_images).
#
# test_sampling_modes checks the vote against a brute-force loop over all seeds and quadrature
# points, that the vote assigns grain volumes closer to the analytic tessellation than centroid
# labeling on coarse meshes, the image counts / wrapping / layer bound of the periodic images,
# periodic shift invariance, that a grain crossing a periodic face stays one grain, and rank-count
# independence (each run compares the distributed result with a single-rank reference computed
# in the same run). Runs at 1-4 ranks.
ms_add_gtest(NAME          test_sampling_modes
             SOURCES       test_sampling_modes.cpp
             DEPENDS_ON    ms_labeling ms_geometry ms_mesh
             NUM_MPI_TASKS 1 2 3 4)

# cube_tet.mesh lives in the source tree; its path is resolved from here
target_compile_definitions(test_sampling_modes PRIVATE
                           MS_SAMPLING_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")
