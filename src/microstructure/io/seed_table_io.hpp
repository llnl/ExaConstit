#pragma once

#include "microstructure/microstructure_types.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file seed_table_io.hpp
 * @brief Reads and writes seed tables: plain-text CSV files that list grain seeds measured by an
 * experiment (HEDM, EBSD reconstructions, ...) or produced by another code, in the format
 * "ExaConstit seed table v1".
 *
 * **What a seed table holds.** One row per grain seed: its position and, optionally, a stable id,
 * a crystal orientation, a phase, a metric weight, a grain size and a grain shape tensor. The
 * generator turns the rows into a SeedSet (microstructure_types.hpp), so measured grains can be
 * grown into the mesh with any of the tessellation metrics.
 *
 * **File syntax.**
 * - A UTF-8 byte-order mark at the start of the file (written by Excel's "CSV UTF-8" export)
 *   and Windows (CRLF) line endings are accepted.
 * - Blank lines and lines whose first non-blank character is `#` are skipped (comments).
 * - The first remaining line is the header: comma-separated column names, case-insensitive,
 *   surrounding whitespace ignored, in any order.
 * - Every following line is one seed with exactly as many comma-separated fields as the header.
 *   Numbers are parsed with the C library (strtod / strtoll in the default "C" locale, so the
 *   decimal separator is always a point); the whole field must be a finite decimal number
 *   (hexadecimal floats such as 0x1p3, inf and nan are refused); any error message
 *   names the line number (1-based, counting every line of the file, comments included).
 *
 * **Columns** (table units are the units of the file, s = units_scale, t = translate):
 * - `x, y, z` (required): position; mesh position = s * x + t.
 * - `id` (optional): int64 in [0, 2^63 - 1], unique; becomes SeedSet::source_id. Default: the
 *   0-based row index. Limit with ROI filler grains (seeding/roi_filler.hpp): the filler ids
 *   start at max id + 1 and add the generator's candidate index (< 2^40), so every id must then
 *   be below 2^63 - 1 - 2^40 (FillOutsideRoi throws otherwise).
 * - `qw, qx, qy, qz` (optional, all four or none): crystal-to-sample passive orientation. The
 *   reader returns the stored values unchanged (never reordered or sign-flipped); the generator
 *   normalizes the orientations it writes to unit length (q / |q|, convention, sign and order
 *   kept). A row with | |q| - 1 | > 1e-3 is an error (usually swapped columns or Euler angles).
 * - `phase` (optional): integer >= 0.
 * - `weight` (optional): metric weight (Laguerre/Quadric w, Additive a), already in mesh units,
 *   so it is NOT scaled.
 * - `volume` / `radius` / `diameter` (optional, at most one): grain size > 0, stored as a volume
 *   in mesh units: V = s^3 volume, or V = 4/3 pi (s r)^3 with r = radius or diameter / 2.
 * - `m11, m22, m33, m23, m13, m12` (optional, all six or none): grain shape tensor M (symmetric
 *   positive definite, length^2), scaled to s^2 M and stored as the Quadric metric tensor
 *   A = (s^2 M)^-1 in SeedSet::shape.
 *
 * The shape tensor M of an ellipsoidal grain with semi-axes a, b, c along the columns of a
 * rotation R is M = R diag(a^2, b^2, c^2) R^T. Its inverse A satisfies (x - s)^T A (x - s) = 1
 * on the ellipsoid surface, which is exactly the Quadric metric g(x) = (x - s)^T A (x - s), so a
 * Quadric tessellation with these tensors grows grains of the measured shapes.
 *
 * Ambiguous quaternion columns (`q0` .. `q3`, any name starting with `quat`) are an error too:
 * their component order (w first or last) is not known, so the user must name them
 * `qw, qx, qy, qz`.
 *
 * Columns named like another orientation description (`phi1`, `phi`, `phi2`, any name starting
 * with `euler`, `bunge`, `rodrigues` or `expmap`, or a rotation-matrix entry `r11` .. `r33`) are
 * an error: the generator never converts orientation conventions, so the user must supply
 * `qw, qx, qy, qz`. Any other unknown column is ignored with one warning printed by rank 0.
 *
 * Positions are only scaled and translated, never rotated. Duplicate positions are allowed (with
 * a warning; the seed search keeps the lowest seed index for coincident seeds), duplicate ids are
 * an error.
 */

namespace exaconstit::microstructure {

/**
 * @brief Where the seed table is and how its coordinates map to the mesh.
 */
struct SeedTableParameters {
    /** @brief Path of the CSV file (read on rank 0 only). */
    std::filesystem::path path;
    /**
     * @brief Length scale s from table units to mesh units (> 0 and finite): positions are
     * multiplied by s, volumes by s^3, radii and diameters by s, shape tensors M by s^2.
     */
    double units_scale = 1.0;
    /** @brief Shift added to every position after scaling (mesh units, finite). */
    std::array<double, 3> translate{};
};

/**
 * @brief The contents of a seed table, converted to mesh units. Identical on every rank after
 * ReadSeedTable.
 *
 * Seed i is entry i (0-based, file row order) of every per-seed array.
 */
struct SeedTable {
    /**
     * @brief Positions (mesh units) and source_id always; quat (4 per seed, w x y z, as stored),
     * phase, weight (unscaled) and shape (A = M^-1, 6 per seed, Axx Ayy Azz Ayz Axz Axy) only when
     * the matching columns are present, otherwise empty.
     */
    SeedSet seeds;
    /** @brief Grain volume per seed (mesh units^3), or empty when no size column was given. */
    std::vector<double> volume;
    /**
     * @brief The shape tensor M per seed in mesh units (length^2), 6 per seed in the order
     * m11 m22 m33 m23 m13 m12, or empty. Kept next to A = M^-1 so WriteSeedTable can write the
     * exact values read (inverting A back would not round-trip bit for bit).
     */
    std::vector<double> shape_tensor;
    bool has_quat = false;   ///< qw, qx, qy, qz were present
    bool has_phase = false;  ///< phase was present
    bool has_weight = false; ///< weight was present
    bool has_volume = false; ///< volume, radius or diameter was present
    bool has_shape = false;  ///< m11 .. m12 were present
    /** @brief Unknown columns that were ignored (lower case, in header order). */
    std::vector<std::string> ignored_columns;
};

/**
 * @brief Reads a seed table on rank 0, converts it to mesh units and gives every rank an
 * identical copy.
 *
 * Steps: rank 0 opens and parses the file and checks every rule listed in the file comment; the
 * outcome (an error message, or the packed table) is broadcast, so all ranks either return the
 * same table or throw the same exception. Warnings (ignored columns, duplicate positions) are
 * printed once, by rank 0, to std::cerr.
 *
 * **Collective** over `comm`. Host only.
 *
 * @param params file path and unit conversion; must be identical on every rank.
 * @param comm   communicator.
 * @return the table (positions, ids and optional data, mesh units).
 * @throws std::invalid_argument on every rank for invalid params (units_scale <= 0 or not finite,
 *         non-finite translate); std::runtime_error on every rank, with the same message, for an
 *         unreadable file or any format error (the message names the file and the line).
 */
SeedTable ReadSeedTable(const SeedTableParameters& params, MPI_Comm comm);

/**
 * @brief Writes a table in the canonical column order with 17 significant digits, so that
 * reading it back with units_scale = 1 and translate = 0 reproduces every value bit for bit.
 *
 * Columns written: `id, x, y, z`, then `qw, qx, qy, qz` (has_quat), `phase` (has_phase), `weight`
 * (has_weight), `volume` (has_volume) and `m11, m22, m33, m23, m13, m12` (has_shape, written from
 * shape_tensor). A `#` comment line naming the format comes first.
 *
 * Not collective (call it on one rank). Host only.
 *
 * @throws std::runtime_error if the file cannot be written, or std::invalid_argument if an array
 *         has the wrong size for its has_* flag.
 */
void WriteSeedTable(const std::filesystem::path& path, const SeedTable& table);

/**
 * @brief Removes the seeds whose position lies outside the closed box [lo, hi] (mesh units),
 * keeping the order of the others; every per-seed array (and volume, shape_tensor) is compacted
 * the same way.
 *
 * Typical use: keep only the measured seeds inside the region of interest of an HEDM scan.
 * Deterministic and purely local; calling it on every rank with the same table keeps the copies
 * identical. Not collective. Host only.
 *
 * @return the number of seeds removed.
 */
std::int64_t KeepSeedsInBox(SeedTable& table,
                            const std::array<double, 3>& lo,
                            const std::array<double, 3>& hi);

} // namespace exaconstit::microstructure
