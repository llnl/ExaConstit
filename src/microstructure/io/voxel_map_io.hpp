#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file voxel_map_io.hpp
 * @brief Reads a voxelized microstructure (a 3D image whose voxels hold grain ids) from an HDF5
 * file: the canonical "ExaConstit voxel map v1" layout or a DREAM.3D file.
 *
 * **What a voxel map is.** Measured microstructures (near-field HEDM, serial-section EBSD,
 * DREAM.3D or ExaCA output) are usually stored as a regular 3D grid of box-shaped voxels, each
 * holding the id of the grain (DREAM.3D: "feature") it belongs to, plus a small per-grain table
 * with the crystal orientation and phase. The voxel labeler (labeling/voxel_labeler.hpp) looks
 * up the voxel under each sample point of each element to give elements their grains. This file
 * only reads the data.
 *
 * **Grid convention (both formats).**
 * - The voxel ids are stored as a 3D array of shape (nz, ny, nx), or (nz, ny, nx, 1), with x
 *   varying fastest in memory (the DREAM.3D convention). Voxel (i, j, k) (0-based, i along x)
 *   is entry k*ny*nx + j*nx + i.
 * - `origin` is the **lower corner** of voxel (0,0,0), `spacing` the voxel edge lengths. Voxel
 *   (i, j, k) is the box [origin + (i, j, k) * spacing, origin + (i+1, j+1, k+1) * spacing]
 *   (componentwise). So the point x lies in voxel floor((x - origin) / spacing) per axis.
 * - Ids <= 0 mean "unassigned" (void, unindexed or a measurement hole). Such voxels do not vote
 *   for any grain; the labeler fills the elements that receive no vote from their neighbors.
 *
 * **Canonical format** (`VoxelFormat::EXACONSTIT`):
 * - `/grain_ids`: any HDF5 integer type (read as int64), shape (nz, ny, nx) or (nz, ny, nx, 1);
 * - `/origin`, `/spacing`: float64[3] (x, y, z), spacing > 0;
 * - optional per-grain table: `/features/id` (int64[nf], unique, >= 1), `/features/quat`
 *   (float64[nf, 4], w x y z, crystal-to-sample passive), `/features/phase` (int32[nf]).
 *   The quaternion storage is fixed by the format, so no `quat_layout` / `conjugate`
 *   declaration is needed (declaring xyzw or conjugate = true is refused, see below).
 *
 * **DREAM.3D** (`VoxelFormat::DREAM3D`):
 * - SIMPL v6 files (DREAM.3D 6.x): `FeatureIds` (nz, ny, nx, 1) int32 per voxel, per-feature
 *   `AvgQuats` (nf+1, 4) float32 and `Phases` (nf+1, 1) int32 where row r belongs to feature id
 *   r (row 0 unused), geometry from the `_SIMPL_GEOMETRY` group of the data container:
 *   DIMENSIONS (x, y, z counts), ORIGIN and SPACING datasets.
 * - DREAM3D-NX (v7, root group `DataStructure`): the same arrays; the image geometry group
 *   carries attributes `_DIMENSIONS` (size_t[3]), `_ORIGIN` and `_SPACING` (float32[3]). These
 *   names are those of simplnx, the DREAM3D-NX library (IOConstants k_H5_DIMENSIONS,
 *   k_H5_ORIGIN, k_H5_SPACING in src/simplnx/DataStructure/IO/Generic/IOConstants.hpp, read and
 *   written in src/simplnx/DataStructure/IO/HDF5/ImageGeomIO.cpp, BlueQuartzSoftware/simplnx on
 *   GitHub, develop branch, checked 2026-10-22). They were not verified against a local NX
 *   file, so when they are not found the user must give `origin` and `spacing` (TOML values,
 *   VoxelMapParameters::has_origin / has_spacing), which always override the file anyway.
 * - **Locating the arrays.** Unless an explicit path is given (`feature_ids_path`,
 *   `quats_path`, `phases_path`), the whole file is searched recursively for datasets *named*
 *   `FeatureIds`, `AvgQuats` and `Phases`. No match, or several, is an error that lists the
 *   candidate paths and names the override key.
 * - **Geometry and data container.** Starting at the group that holds `FeatureIds` and walking
 *   up towards the root, the first group that has either a `_SIMPL_GEOMETRY` child group
 *   (SIMPL v6) or both `_ORIGIN` and `_SPACING` attributes (NX) gives origin and spacing (and
 *   dimensions, which are cross-checked against the `FeatureIds` shape). That group is the
 *   *data container* of the voxel ids; without geometry it is the grandparent group of
 *   `FeatureIds` (DC/CellData/FeatureIds -> DC), or the root.
 * - **Per-feature arrays belong to the same data container.** The search for `AvgQuats` and
 *   `Phases` only accepts datasets inside the data container of the chosen `FeatureIds`, and
 *   skips datasets with the per-voxel shape (DREAM.3D files often hold a per-voxel `Phases` next
 *   to `FeatureIds`). So with two containers and `feature_ids_path` pointing into B, A's
 *   quaternions are never used silently: if B has none, the error lists A's as candidates and
 *   asks for `quats_path`.
 * - **Quaternion storage declarations** are required whenever quaternions are read: `quat_layout`
 *   (wxyz or xyzw: the order of the 4 stored numbers) and `conjugate` (whether the stored
 *   quaternion must be conjugated - x, y, z negated - to give the crystal-to-sample passive
 *   rotation ExaConstit expects). The reader reorders to w x y z and conjugates when asked, and
 *   does nothing else: no normalization, no sign flip, no Euler angles. (The generator
 *   normalizes the orientations it writes to unit length, keeping convention, sign and order.)
 *
 * **Units.** File positions are converted to mesh units by
 *   x_mesh = units_scale * x_file + translate
 * (no rotation, ever), so origin_mesh = units_scale * origin_file + translate and
 * spacing_mesh = units_scale * spacing_file. TOML `origin` / `spacing` values are in file units
 * and replace the file's values before the conversion.
 *
 * **Parallel reading.** HDF5 files are read with the serial HDF5 library: several ranks may open
 * the same file read-only at the same time. Files are opened with HDF5 file locking turned off
 * (H5Pset_file_locking(fapl, false, true)): locking fails on some parallel file systems and is
 * not needed for read-only access. Small data (grid description, per-grain table) is read by
 * rank 0 and broadcast (the "collective" functions below; every rank must call them). The voxel
 * ids themselves are read per rank, each rank reading only the boxes of voxels under its own
 * sample points (ReadVoxelBoxes: one HDF5 "hyperslab" read per box, i.e. a rectangular sub-block
 * of a dataset read directly from the file). No function reads the whole voxel dataset.
 *
 * **Without HDF5.** When Axom was built without HDF5 (AXOM_USE_HDF5 undefined), every reader
 * throws std::runtime_error("... built without HDF5 ...") on every rank.
 *
 * Everything here runs on the host.
 */

namespace exaconstit::microstructure {

/** @brief File layout of a voxel map (TOML `format = "exaconstit" | "dream3d"`). */
enum class VoxelFormat {
    EXACONSTIT, ///< canonical "ExaConstit voxel map v1" layout (see the file comment)
    DREAM3D     ///< DREAM.3D SIMPL v6 or DREAM3D-NX file
};

/** @brief Order of the four numbers of a stored quaternion (DREAM.3D declaration). */
enum class QuatLayout {
    UNSET, ///< not declared (an error when DREAM.3D quaternions are read)
    WXYZ,  ///< scalar part first: w x y z
    XYZW   ///< scalar part last: x y z w
};

/**
 * @brief Where and how to read a voxel map (plain struct; the integration code fills it from the
 * user options). Every rank must pass the same values.
 */
struct VoxelMapParameters {
    /** @brief File layout. */
    VoxelFormat format = VoxelFormat::EXACONSTIT;
    /** @brief The HDF5 file. */
    std::filesystem::path file;
    /**
     * @brief HDF5 path of the per-voxel id dataset ("/group/dataset"). Empty: `/grain_ids`
     * (canonical) or a search for datasets named `FeatureIds` (DREAM.3D).
     */
    std::string feature_ids_path;
    /**
     * @brief HDF5 path of the per-grain quaternion dataset. Empty: `/features/quat` (canonical)
     * or a search for `AvgQuats` (DREAM.3D).
     */
    std::string quats_path;
    /**
     * @brief HDF5 path of the per-grain phase dataset. Empty: `/features/phase` (canonical) or a
     * search for `Phases` (DREAM.3D).
     */
    std::string phases_path;
    /** @brief ReadVoxelFeatures reads the quaternions. */
    bool read_quats = false;
    /** @brief ReadVoxelFeatures reads the phases. */
    bool read_phases = false;
    /** @brief Stored quaternion order (required for DREAM.3D quaternions). */
    QuatLayout quat_layout = QuatLayout::UNSET;
    /**
     * @brief Conjugate the stored quaternions: -1 unset (an error for DREAM.3D quaternions),
     * 0 no, 1 yes.
     */
    int conjugate = -1;
    /** @brief Use `origin` (file units) instead of the file's value. */
    bool has_origin = false;
    /** @brief Use `spacing` (file units) instead of the file's value. */
    bool has_spacing = false;
    /** @brief Lower corner of voxel (0,0,0), file units (used when has_origin). */
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    /** @brief Voxel edge lengths, file units, > 0 (used when has_spacing). */
    std::array<double, 3> spacing = {0.0, 0.0, 0.0};
    /** @brief File-to-mesh length factor, > 0. */
    double units_scale = 1.0;
    /** @brief Added after scaling (mesh units). */
    std::array<double, 3> translate = {0.0, 0.0, 0.0};
};

/**
 * @brief Description of the voxel grid, in mesh units. Identical on every rank after
 * ReadVoxelGridInfo.
 */
struct VoxelGridInfo {
    /** @brief Voxel counts (nx, ny, nz), each > 0. */
    std::array<std::int64_t, 3> dims = {0, 0, 0};
    /** @brief Lower corner of voxel (0,0,0), mesh units (after units_scale and translate). */
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    /** @brief Voxel edge lengths, mesh units, > 0. */
    std::array<double, 3> spacing = {0.0, 0.0, 0.0};
    /** @brief HDF5 path of the per-voxel id dataset actually used (after the name search). */
    std::string feature_ids_path;
};

/**
 * @brief A box of voxel indices [lo, hi) (0-based, half-open, x y z), 0 <= lo[d] <= hi[d] <=
 * dims[d].
 */
struct VoxelBox {
    /** @brief First voxel index per axis. */
    std::array<std::int64_t, 3> lo = {0, 0, 0};
    /** @brief One past the last voxel index per axis. */
    std::array<std::int64_t, 3> hi = {0, 0, 0};
};

/**
 * @brief The voxel ids of a rectangular block of voxels.
 *
 * `ids` has (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]) entries, x fastest: voxel (i, j, k)
 * of the grid (lo <= (i,j,k) < hi) is entry
 *   ((k - lo[2]) * (hi[1] - lo[1]) + (j - lo[1])) * (hi[0] - lo[0]) + (i - lo[0]).
 * Values are as stored (<= 0 = unassigned).
 */
struct VoxelBlock {
    /** @brief The grid the block belongs to. */
    VoxelGridInfo grid;
    /** @brief First voxel index of the box per axis (0-based, x y z). */
    std::array<std::int64_t, 3> lo = {0, 0, 0};
    /** @brief One past the last voxel index of the box per axis (half-open box [lo, hi)). */
    std::array<std::int64_t, 3> hi = {0, 0, 0};
    /** @brief Voxel ids of the block (layout above). */
    std::vector<std::int64_t> ids;
};

/**
 * @brief Per-grain (per-feature) data, sorted by increasing id. Identical on every rank after
 * ReadVoxelFeatures.
 */
struct VoxelFeatureTable {
    /** @brief Feature ids, strictly increasing, each >= 1. */
    std::vector<std::int64_t> id;
    /**
     * @brief quat[4*r + c], c = 0..3 -> w x y z of feature id[r] (crystal-to-sample, passive,
     * as stored after the declared reorder/conjugation). Empty unless read_quats.
     */
    std::vector<double> quat;
    /** @brief phase[r] of feature id[r]. Empty unless read_phases. */
    std::vector<int> phase;
};

/**
 * @brief Reads and checks the grid description: dimensions, origin, spacing and the id dataset's
 * path. Metadata only: no voxel id is read.
 *
 * Steps (rank 0): open the file; find the id dataset (explicit path, `/grain_ids`, or the
 * DREAM.3D name search); check it stores integers; take (nx, ny, nz) from its shape; read origin
 * and spacing (file values, replaced by the TOML values when given; DREAM.3D geometry from
 * `_SIMPL_GEOMETRY` or the NX attributes, cross-checked against the shape); convert to mesh
 * units. The result (or the error message) is broadcast.
 *
 * Checks: units_scale > 0 and finite; finite translate and origin; spacing > 0 and finite both
 * in file units and after scaling (a tiny units_scale could underflow to 0); integer id dataset
 * of rank 3, or rank 4 with last extent 1; dimensions > 0 and equal to the file's declared
 * dimensions. The "ids < 2^31" check is done on the voxels each rank actually reads (by the
 * voxel labeler), not here.
 *
 * **Collective** over `comm`. Host only.
 *
 * @throws std::runtime_error on every rank with rank 0's message for any failure (missing or
 *         unreadable file, missing or ambiguous dataset, bad shape or geometry, or HDF5 not
 *         available).
 */
VoxelGridInfo ReadVoxelGridInfo(const VoxelMapParameters& params, MPI_Comm comm);

/**
 * @brief Reads the ids of several voxel boxes from `grid.feature_ids_path`, opening the file
 * once and doing one hyperslab read per box.
 *
 * The result holds the boxes back to back in the order given, each box x fastest: voxel
 * (i, j, k) of box b is entry offset_b + ((k - lo_z) * ny_b + (j - lo_y)) * nx_b + (i - lo_x),
 * where offset_b is the total voxel count of the boxes before b and nx_b, ny_b are the box
 * extents. Values are as stored, converted to int64 (no range check here). Empty boxes
 * contribute nothing; with no voxels in total the file is not opened.
 *
 * **Not collective**: each rank calls it for its own boxes. Host only. Memory: 8 bytes per voxel
 * of all boxes.
 *
 * @param params as passed to ReadVoxelGridInfo (only `file` is used).
 * @param grid   result of ReadVoxelGridInfo.
 * @param boxes  voxel boxes inside the grid.
 * @throws std::invalid_argument for a box outside the grid; std::runtime_error for HDF5 errors
 *         (local to this rank; callers that need a collective error must reduce it).
 */
std::vector<std::int64_t> ReadVoxelBoxes(const VoxelMapParameters& params,
                                         const VoxelGridInfo& grid,
                                         const std::vector<VoxelBox>& boxes);

/**
 * @brief Reads the ids of one voxel box [lo, hi) (ReadVoxelBoxes with a single box), returned
 * with the box description.
 *
 * **Not collective.** Host only. Same errors as ReadVoxelBoxes.
 */
VoxelBlock ReadVoxelBlock(const VoxelMapParameters& params,
                          const VoxelGridInfo& grid,
                          const std::array<std::int64_t, 3>& lo,
                          const std::array<std::int64_t, 3>& hi);

/**
 * @brief Reads the per-grain table (ids, and the quaternions / phases that were requested) and
 * checks it against the feature ids the mesh actually uses.
 *
 * - Canonical: `/features/id` plus `/features/quat` and/or `/features/phase` (or the override
 *   paths). Ids must be unique and >= 1; rows are sorted by id.
 * - DREAM.3D: rows are indexed by feature id (row 0 unused), so the table holds ids
 *   1..rows-1. AvgQuats is reordered to w x y z according to `quat_layout` and conjugated when
 *   `conjugate` = 1. The datasets are searched inside the data container of the voxel ids (see
 *   the file comment).
 * - With nothing requested (read_quats and read_phases false) the table is empty and the file is
 *   not opened.
 *
 * Checks: every id of `used_ids` has a row; quaternion norms satisfy | |q| - 1 | <= 1e-3
 * (canonical: every row; DREAM.3D: the rows of `used_ids`, since unused DREAM.3D rows, such as
 * row 0, may hold zeros); the DREAM.3D declarations are present when quaternions are read; the
 * canonical format refuses quat_layout = XYZW and conjugate = 1 (its storage is fixed). Memory
 * is proportional to the table and to `used_ids`, whatever the size of the ids (sparse ids up to
 * 2^31 - 1 cost nothing extra).
 *
 * **Collective** over `comm` (rank 0 reads and checks, then broadcasts). Host only.
 *
 * @param params   what to read.
 * @param grid     result of ReadVoxelGridInfo (dimensions and id dataset).
 * @param used_ids feature ids used by the mesh, strictly increasing, each >= 1, identical on
 *                 every rank: the `grain_feature` list of CompactFeatureLabels
 *                 (labeling/voxel_labeler.hpp).
 * @throws std::runtime_error on every rank with rank 0's message for any failure, including
 *         `used_ids` not strictly increasing or holding an id < 1.
 */
VoxelFeatureTable ReadVoxelFeatures(const VoxelMapParameters& params,
                                    const VoxelGridInfo& grid,
                                    const std::vector<std::int64_t>& used_ids,
                                    MPI_Comm comm);

} // namespace exaconstit::microstructure
