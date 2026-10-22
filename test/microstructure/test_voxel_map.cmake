# Voxel maps (Stage 1d): the HDF5 voxel map reader (io/voxel_map_io) and the voxel labeler
# (labeling/voxel_labeler).
#
# test_voxel_map writes its own HDF5 fixtures (canonical, DREAM.3D SIMPL v6 and DREAM3D-NX-like
# layouts of a voxelized 6-seed Voronoi tessellation with a hole of unassigned voxels) into the
# test's working directory in the build tree, then checks: grid/feature reading, hyperslab
# blocks, quaternion layout and conjugation declarations, the name search and its errors,
# geometry overrides and unit transforms, the 2^31 id limit, exact labels on a voxel-aligned hex
# mesh (hole filled), multi-round fill outside the grid, vote labels on coarse hex and tet meshes
# against direct Voronoi labeling away from grain boundaries, curved (quadratic) elements whose
# vote points lie beyond their nodes, exact vote ties, negative ids, a spatially split partition
# with an empty rank that reads only its own voxel tiles (the read volume is printed), the
# per-feature search restricted to the chosen data container, and the feature-id compaction.
# Rank-count independence: every distributed result is compared with the expected serial answer.
# Runs at 1-4 ranks.
ms_add_gtest(NAME          test_voxel_map
             SOURCES       test_voxel_map.cpp
             DEPENDS_ON    ms_labeling ms_io ms_mesh
             NUM_MPI_TASKS 1 2 3 4)
