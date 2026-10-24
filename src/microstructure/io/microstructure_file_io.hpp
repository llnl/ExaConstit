#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

/**
 * @file microstructure_file_io.hpp
 * @brief The "ExaConstit microstructure file": one HDF5 file that holds a grain table and,
 * optionally, labeled points on a regular grid. It is read as a seed list (grain table only),
 * read as a voxel map (grain table + grid), and written by the generator.
 *
 * **Layout** (format_version 1). Every value is an HDF5 *dataset*, never an HDF5 attribute, so
 * Conduit (relay::io, which never reads attributes) loads the whole file into a conduit::Node, and
 * the labeled points form a Conduit Blueprint mesh. Paths are relative to the file root:
 * ```
 * format                        string "ExaConstit microstructure"
 * format_version                integer 1
 * grains/id                     integer[G], >= 1, unique (required)
 * grains/quat                   float[4G]  w x y z per grain, crystal-to-sample (passive)
 * grains/phase                  integer[G], >= 1
 * grains/position               float[3G]  x y z per grain (seed point)
 * grains/volume                 float[G], > 0 (target grain size)
 * grains/weight                 float[G]   laguerre / additive weight
 * grains/shape                  float[6G]  m11 m22 m33 m23 m13 m12 per grain (shape tensor M)
 * mesh/coordsets/coords/type    string "uniform"
 * mesh/coordsets/coords/dims/i, j, k              integer: grid POINTS per axis (cells + 1)
 * mesh/coordsets/coords/origin/x, y, z            float: lower corner of cell (0, 0, 0)
 * mesh/coordsets/coords/spacing/dx, dy, dz        float: cell edge lengths, > 0
 * mesh/topologies/topo/type     string "uniform";  mesh/topologies/topo/coordset  "coords"
 * mesh/fields/grain_id/association  string "element"; .../topology  string "topo"
 * mesh/fields/grain_id/values   integer[nx ny nz]: x fastest, then y, then z; <= 0 = unassigned
 * ```
 * - Every grains/ dataset except id is optional. Missing means "not given", never a default.
 * - Per-grain arrays are stored one row per grain: shape (c G) flat (as Conduit writes it) or
 *   (G, c) (as h5py writes a list of rows), where c is the number of values per grain. The
 *   grid values are (nx ny nz) flat or (nz, ny, nx), optionally with a trailing 1. Transposed
 *   arrays, e.g. (4, G) or (nx, ny, nz) with nx != nz, are refused instead of being misread.
 * - Any HDF5 integer or floating-point type is accepted where the table says integer or float
 *   (converted on read). Strings may be fixed or variable length (as h5py writes them) or an
 *   array of 1-byte characters ending in a NUL (as Conduit writes them, also in the files the
 *   generator writes).
 * - Unknown groups and datasets are ignored. A file whose format_version is larger than the one
 *   this reader knows (k_microstructure_format_version) is refused; additions that do not change
 *   the meaning of existing data never raise the version.
 * - Units are the file's own; readers convert with units_scale and translate (no rotation, ever).
 *   `weight` is NOT scaled (as in the seed table CSV).
 * - Grain ids are below 2^31 (the generator stores grains in C ints).
 * - Origin and spacing are required here, although Blueprint itself would default them.
 *
 * Who reads what: a seed list goes through ReadSeedTable (seed_table_io.hpp), which detects an
 * HDF5 file and converts its grain table with the seed table's rules; a voxel map goes through
 * voxel_map_io.hpp (VoxelFormat::EXACONSTIT), which reads the grid in tiles. This file holds the
 * format constants, the grain table reader and writer and the grid description reader.
 *
 * **Without HDF5.** When Axom was built without Conduit or HDF5, every function that touches a
 * file throws std::runtime_error("... built without ...").
 *
 * Everything here runs on the host, on the calling rank only (not collective).
 */

namespace exaconstit::microstructure {

/** @brief Value of the `format` dataset. */
inline constexpr const char* k_microstructure_format_name = "ExaConstit microstructure";
/** @brief The format_version this code writes and the largest it reads. */
inline constexpr std::int64_t k_microstructure_format_version = 1;

/**
 * @brief The grain table of a microstructure file, in the file's own units. Grain r is entry r
 * of every array. Optional arrays are empty when the file does not have them.
 */
struct MicrostructureGrainTable {
    /** @brief Grain ids, >= 1, unique, < 2^31. */
    std::vector<std::int64_t> id;
    /** @brief 4 per grain, w x y z, crystal-to-sample passive (stored values, not normalized). */
    std::vector<double> quat;
    /** @brief Phase per grain, >= 1. */
    std::vector<int> phase;
    /** @brief 3 per grain: seed point x y z. */
    std::vector<double> position;
    /** @brief Target volume per grain, > 0. */
    std::vector<double> volume;
    /** @brief Metric weight per grain (laguerre w, additive a). */
    std::vector<double> weight;
    /** @brief 6 per grain: shape tensor M = m11 m22 m33 m23 m13 m12 (symmetric positive
     * definite, length^2). */
    std::vector<double> shape;

    /** @brief Number of grains. */
    std::size_t Size() const {
        return id.size();
    }
};

/**
 * @brief The uniform grid of a microstructure file, in file units, as stored (no TOML
 * overrides, no unit conversion).
 */
struct MicrostructureGrid {
    /** @brief Cell counts (nx, ny, nz), each > 0 (the stored point counts minus 1). */
    std::array<std::int64_t, 3> dims = {0, 0, 0};
    /** @brief Lower corner of cell (0, 0, 0). */
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    /** @brief Cell edge lengths (finite and > 0 is checked by the voxel reader, after the TOML
     * overrides). */
    std::array<double, 3> spacing = {0.0, 0.0, 0.0};
    /** @brief HDF5 path of the per-cell grain id dataset. */
    std::string values_path = "/mesh/fields/grain_id/values";
};

/**
 * @brief True when the file starts with the HDF5 signature (at byte 0, or at 512, 1024, 2048,
 * ... when the file has a user block). Reads only the signature bytes; needs no HDF5 library.
 * False for a missing or unreadable file.
 */
bool IsHdf5File(const std::filesystem::path& path);

/**
 * @brief Checks `format` and `format_version` and returns the version. Throws
 * std::runtime_error naming the file when the file cannot be opened, is not an ExaConstit
 * microstructure file, or has a newer version than this reader knows.
 */
std::int64_t CheckMicrostructureFileHeader(const std::filesystem::path& path);

/** @brief True when the file has labeled points (a `mesh` group). Checks the header first. */
bool MicrostructureFileHasLabeledPoints(const std::filesystem::path& path);

/**
 * @brief Reads and checks the grain table: header; grains/id present; every optional dataset
 * has the right element count; ids >= 1, < 2^31 and unique; phases >= 1; volumes > 0 and
 * finite; positions, weights and shapes finite; quaternion norms within 1e-3 of 1. Values are
 * returned as stored (file units).
 * @throws std::runtime_error naming the file and the dataset on the first problem.
 */
MicrostructureGrainTable ReadMicrostructureGrainTable(const std::filesystem::path& path);

/**
 * @brief Reads the uniform grid description under `mesh/` (Blueprint coordset "coords",
 * topology "topo", field "grain_id"): types, point counts, origin and spacing, association and
 * topology name. The values dataset itself is not read (the voxel reader checks its type and
 * element count and reads it in tiles).
 * @throws std::runtime_error naming the file and the path on the first problem (including a
 *         file without `mesh/`, or with a coordset or topology that is not "uniform").
 */
MicrostructureGrid ReadMicrostructureGrid(const std::filesystem::path& path);

/**
 * @brief Writes a microstructure file holding the header and the grain table (no labeled
 * points), replacing any existing file. Optional arrays are written only when not empty. Uses
 * Conduit relay, so every array is a 1-D dataset.
 * @throws std::invalid_argument when an array has the wrong size or an id is < 1;
 *         std::runtime_error when the file cannot be written.
 */
void WriteMicrostructureFile(const std::filesystem::path& path,
                             const MicrostructureGrainTable& grains);

} // namespace exaconstit::microstructure
