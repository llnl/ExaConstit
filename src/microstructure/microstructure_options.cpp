/**
 * @file microstructure_options.cpp
 * @brief Reads, checks and overrides the options of the microstructure generator.
 *
 * The generator is driven by a TOML options file; the .toml files in test/data/microstructure
 * are commented examples. This file turns that file into a MicrostructureOptions struct in four
 * separate steps, which the tool calls in this order:
 *
 *   1. parse_options_file(): read the TOML file from disk (throws if it does not exist).
 *   2. from_toml():          copy every recognized key into the struct. Missing keys keep the
 *                            defaults written in microstructure_options.hpp. Unknown keys only
 *                            warn. Values that cannot be stored at all (an unknown metric name,
 *                            an array of the wrong length, a negative RNG key, an integer too
 *                            large for its option) throw.
 *   3. apply_cli_overrides(): the command-line flags (-m -n -s -o --parts) replace file values.
 *   4. validate():           check the final combination (the exec backend is built in, files
 *                            exist, counts are positive, lengths are finite, the anisotropy
 *                            tensor is a valid metric, ...). Every problem is printed, then
 *                            false is returned so the tool can stop cleanly.
 *
 * Validation is deliberately separate from parsing so that command-line overrides can fix an
 * options file before it is checked (e.g. `-m my.mesh` on a file that names a missing mesh).
 *
 * TOML access goes through toml11 (src/TOML_Reader), in the same way as ExaConstit's own options
 * readers in src/options/option_*.cpp:
 *   - `table.contains("key")` tests whether a key was written;
 *   - `toml::find<T>(table, "key")` reads it as type T and throws if the TOML type does not
 *     match (e.g. a string where an integer is expected);
 *   - `toml::find(table, "key")` returns the raw toml::value, used when more than one TOML type
 *     is accepted (integer or float, bool or string).
 *
 * MPI: nothing here communicates between ranks. Every rank reads the same file and builds the
 * same options; warnings and errors are printed by rank 0 only (WARNING_0_OPT from
 * options/option_util.hpp, which needs MPI to be initialized). Because every rank parses the
 * same input, every rank throws, or returns the same validate() result, together.
 */
#include "microstructure/microstructure_options.hpp"

#include "microstructure/exec_dispatch.hpp"
#include "options/option_util.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace exaconstit::microstructure {

// Helpers used only in this file. An unnamed ("anonymous") namespace makes them invisible to
// other source files, so their short names cannot clash with functions elsewhere.
namespace {

/**
 * @brief Warns (on rank 0) about every key of `table` that is not in `known`.
 *
 * An unknown key is not an error. It is usually either a typo (`num_sedes`) or a table/key that
 * belongs to a later stage of the generator (e.g. [Conformal], [VoxelMap], lloyd_iterations)
 * and is not read by this version. Neither should stop a run, but the user must be told that the
 * value had no effect, otherwise a typo silently falls back to the default.
 *
 * Steps: collect the keys of `table` that are not listed in `known`, sort them, and print one
 * warning per key with its full dotted name (e.g. "Mesh.Auto.ncut").
 *
 * @param table      A TOML table (must be a table; toml11 throws otherwise).
 * @param table_name Dotted name of the table for the message; "" for the top level.
 * @param known      Keys (and sub-table names) this version reads from the table.
 */
void warn_unknown_keys(const toml::value& table,
                       const std::string& table_name,
                       std::initializer_list<const char*> known) {
    // Step 1: compare every key written in the file against the list of known keys.
    std::vector<std::string> unknown;
    for (const auto& entry : table.as_table()) {
        const std::string& key = entry.first;
        bool is_known = false;
        for (const char* known_key : known) {
            if (key == known_key) {
                is_known = true;
                break;
            }
        }
        if (!is_known) {
            unknown.push_back(key);
        }
    }

    // Step 2: toml::table is a hash map, so its iteration order is arbitrary. Sorting makes the
    // warnings appear in the same order on every run and every machine.
    std::sort(unknown.begin(), unknown.end());

    // Step 3: one rank-0 warning per unknown key, with the full dotted key name.
    for (const auto& key : unknown) {
        std::string full_key = key;
        if (!table_name.empty()) {
            full_key = table_name + "." + key;
        }
        WARNING_0_OPT("Microstructure options: ignoring unknown or not yet supported key '" +
                      full_key + "'");
    }
}

/**
 * @brief Converts the string written in the file (e.g. "voronoi") into an enum value.
 *
 * Used for the options whose values are C++ enums shared with the rest of the generator
 * (placement, metric, sampling, distance, mesh_format; see microstructure_types.hpp).
 *
 * Unlike `string_to_enum` in options/option_util.hpp, which falls back to a NOTYPE value with a
 * warning, these enums have no "unset" value, and silently picking another metric or placement
 * would produce a different microstructure than the user asked for. An unknown name is therefore
 * a hard error.
 *
 * @param str     The value written in the file; matching is exact and case-sensitive.
 * @param mapping Accepted spellings and the enum value each one selects.
 * @param key     Dotted key name, used only in the error message.
 * @return The enum value for `str`.
 * @throws std::invalid_argument if `str` is not in `mapping`; the message lists the accepted
 *         spellings.
 */
template <typename EnumType>
EnumType string_to_ms_enum(const std::string& str,
                           const std::map<std::string, EnumType>& mapping,
                           const std::string& key) {
    const auto it = mapping.find(str);
    if (it != mapping.end()) {
        return it->second;
    }
    // Not found: build an error that tells the user every value that would have worked.
    std::ostringstream err;
    err << "Microstructure options: unknown value '" << str << "' for '" << key
        << "' (expected one of:";
    for (const auto& entry : mapping) {
        err << " \"" << entry.first << "\"";
    }
    err << ")";
    throw std::invalid_argument(err.str());
}

/**
 * @brief Reads a real number that may be written as a TOML integer or a TOML float.
 *
 * TOML distinguishes `1` (integer) from `1.0` (float), and toml11's `get<double>` throws on an
 * integer. Users naturally write `length = [1, 1, 1]` or `units_scale = 2`, so both forms are
 * accepted and converted to double.
 *
 * Note that TOML 1.0 also accepts the floats `nan` and `inf`; they are read as is here, and
 * validate() rejects them where a finite value is required.
 *
 * @param value A TOML value holding an integer or a float.
 * @return The value as a double.
 * @throws toml::type_error (a std::exception) if the value is neither an integer nor a float.
 */
double get_number(const toml::value& value) {
    if (value.is_integer()) {
        return static_cast<double>(value.as_integer());
    }
    return value.as_floating();
}

/**
 * @brief Reads `table[key]` as an array of exactly N real numbers (integers or floats).
 *
 * Used for fixed-size vectors such as a box size [Lx, Ly, Lz], a translation [dx, dy, dz], or
 * the 6 independent components of a symmetric 3x3 tensor.
 *
 * @param table A TOML table that contains `key` (toml11 throws if it does not).
 * @param key   Key of the array, also used in the error message.
 * @return The N values in the order written in the file.
 * @throws std::invalid_argument if the array does not have exactly N entries, so a missing
 *         component is never silently set to zero.
 */
template <std::size_t N>
std::array<double, N> get_number_array(const toml::value& table, const std::string& key) {
    const auto& array = toml::find(table, key).as_array();
    if (array.size() != N) {
        std::ostringstream err;
        err << "Microstructure options: '" << key << "' needs " << N << " values, got "
            << array.size();
        throw std::invalid_argument(err.str());
    }
    std::array<double, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = get_number(array[i]);
    }
    return result;
}

/**
 * @brief Converts a TOML integer (always 64-bit) to `int`, refusing values that do not fit.
 *
 * toml11's `get<int>` is a plain cast of the 64-bit value with no range check
 * (toml/get.hpp:92-95), so e.g. `compression = 4294967302` (2^32 + 6) would silently become 6.
 * Reading the 64-bit value and checking it first turns such a typo into an error.
 *
 * @param value    The 64-bit integer read from the file.
 * @param full_key Dotted key name, used only in the error message.
 * @return The same value as an int.
 * @throws std::invalid_argument if the value is outside the range of int.
 */
int checked_int(std::int64_t value, const std::string& full_key) {
    if (value < static_cast<std::int64_t>(std::numeric_limits<int>::min()) ||
        value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        std::ostringstream err;
        err << "Microstructure options: '" << full_key << "' = " << value
            << " is out of range for an integer option";
        throw std::invalid_argument(err.str());
    }
    return static_cast<int>(value);
}

/**
 * @brief Reads `table[key]` as an `int`, with the range check of checked_int().
 *
 * @param table    A TOML table that contains `key`.
 * @param key      Key to read.
 * @param full_key Dotted key name, used only in the error message.
 * @throws std::invalid_argument if the value does not fit in an int; toml11 errors if it is not
 *         a TOML integer.
 */
int get_int(const toml::value& table, const std::string& key, const std::string& full_key) {
    return checked_int(toml::find<std::int64_t>(table, key), full_key);
}

/**
 * @brief Chooses which of two spellings of the same key to read, warning if both are written.
 *
 * Some keys accept an alias (e.g. [Mesh.Auto] `length` or `mxyz`) so that options files can use
 * ExaConstit's spelling. If both are written, the canonical name is used, and a rank-0 warning
 * says so; otherwise the value under the alias would be dropped without the user noticing.
 *
 * @param table      The TOML table to look in.
 * @param canonical  Preferred key name.
 * @param alias      Alternative key name.
 * @param table_name Dotted table name for the warning.
 * @return The key to read: `canonical` if written, else `alias` (the caller has already checked
 *         that at least one of them is present).
 */
const char* pick_key_or_alias(const toml::value& table,
                              const char* canonical,
                              const char* alias,
                              const std::string& table_name) {
    const bool has_canonical = table.contains(canonical);
    const bool has_alias = table.contains(alias);
    if (has_canonical && has_alias) {
        WARNING_0_OPT("Microstructure options: both '" + table_name + "." + canonical +
                      "' and its alias '" + table_name + "." + alias + "' are set; using '" +
                      canonical + "'");
    }
    if (has_canonical) {
        return canonical;
    }
    return alias;
}

/** @brief True if `x` is a finite number greater than 0 (false for NaN and +-inf). */
bool is_positive_finite(double x) {
    return std::isfinite(x) && x > 0.0;
}

/** @brief True if `x` is a finite number greater than or equal to 0 (false for NaN and +-inf). */
bool is_nonnegative_finite(double x) {
    return std::isfinite(x) && x >= 0.0;
}

/**
 * @brief Reads the [Mesh] table: which mesh the grains are painted onto.
 *
 * Keys:
 *   - `type`: "file" (one serial MFEM mesh file), "parallel" (a mesh already split into one
 *     file per rank, given by its file prefix) or "auto" (a Cartesian box of hexahedra built
 *     from [Mesh.Auto]). Default "file". The spelling is checked in validate().
 *   - `floc`: path of the mesh file, or the file prefix for "parallel". Relative paths are
 *     relative to the working directory, as in ExaConstit's options.toml.
 *   - [Mesh.Auto] sub-table, used when type = "auto". It uses the same spellings as
 *     ExaConstit's own [Mesh.Auto] (MeshOptions), so the two files read alike:
 *       - `length` (alias `mxyz`): box size [Lx, Ly, Lz], in mesh length units;
 *       - `ncuts`  (alias `nxyz`): number of elements along x, y and z.
 *     If both the name and its alias are written, the first name (`length`, `ncuts`) wins and
 *     a rank-0 warning is printed.
 *
 * @throws std::invalid_argument if `length`/`ncuts` does not have 3 entries.
 */
MeshInput mesh_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input, "Mesh", {"type", "floc", "Auto"});
    MeshInput mesh;

    // Mesh source and location; checked against the disk in validate(), after CLI overrides.
    if (toml_input.contains("type")) {
        mesh.type = toml::find<std::string>(toml_input, "type");
    }
    if (toml_input.contains("floc")) {
        mesh.floc = toml::find<std::string>(toml_input, "floc");
    }

    // [Mesh.Auto]: box dimensions and element counts of the generated hex mesh.
    if (toml_input.contains("Auto")) {
        const auto& auto_section = toml::find(toml_input, "Auto");
        warn_unknown_keys(auto_section, "Mesh.Auto", {"length", "mxyz", "ncuts", "nxyz"});

        // Box size per direction (real numbers; integers are accepted too).
        if (auto_section.contains("length") || auto_section.contains("mxyz")) {
            const char* key = pick_key_or_alias(auto_section, "length", "mxyz", "Mesh.Auto");
            mesh.mxyz = get_number_array<3>(auto_section, key);
        }

        // Elements per direction (integers). Read as 64-bit and range-checked (see
        // checked_int). The values themselves (> 0) are checked in validate(); here we only
        // make sure there is one count per direction and that each fits in an int.
        if (auto_section.contains("ncuts") || auto_section.contains("nxyz")) {
            const char* key = pick_key_or_alias(auto_section, "ncuts", "nxyz", "Mesh.Auto");
            const std::string full_key = std::string("Mesh.Auto.") + key;
            const auto ncuts = toml::find<std::vector<std::int64_t>>(auto_section, key);
            if (ncuts.size() != 3) {
                throw std::invalid_argument("Microstructure options: '" + full_key +
                                            "' needs 3 values");
            }
            for (std::size_t d = 0; d < 3; ++d) {
                mesh.nxyz[d] = checked_int(ncuts[d], full_key);
            }
        }
    }
    return mesh;
}

/**
 * @brief Reads the [Seeds] table: where the grain seeds (one per grain) come from.
 *
 * Each grain grows from one seed point; the tessellation then gives every element to its
 * "closest" seed. Keys (defaults in SeedOptions, microstructure_options.hpp):
 *   - `source`: "random" (generated) or "file" (read from a seed table). Checked in validate().
 *   - `num_seeds`: number of grains to generate (random sources).
 *   - `rng_seed`: integer >= 0 that keys the random numbers. The generator uses a counter-based
 *     random number generator: every random value is computed from (rng_seed, purpose, id), not
 *     drawn from a running sequence, so the result depends only on rng_seed and never on the
 *     number of MPI ranks. Same rng_seed => same microstructure.
 *   - `placement`: how generated seeds are positioned: "rejection", "element_weighted",
 *     "lattice", "johnson_mehl" or "file" (see SeedPlacement in microstructure_types.hpp).
 *     If omitted and source = "file", it defaults to "file".
 *   - `min_distance_factor`: minimum seed spacing as a fraction of the mean grain size
 *     (V/N)^(1/3); 0 disables it.
 *   - `boundary_clearance`: minimum distance of a seed from the domain boundary; 0 disables it.
 *   - `sampling_domain`: "material" (seeds only inside the mesh) or "bbox" (anywhere in the
 *     mesh's bounding box).
 *   - `lattice_type` ("cubic", "bcc", "fcc", "hcp") and `lattice_jitter` (random offset as a
 *     fraction of the lattice spacing), for placement = "lattice".
 *   - `file`, `units_scale`, `translate`: seed table path and the scale-then-translate applied
 *     to its positions to bring them into mesh coordinates.
 *
 * @throws std::invalid_argument for a negative rng_seed, an unknown placement, or a translate
 *         that does not have 3 entries.
 */
SeedOptions seeds_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "Seeds",
                      {"source",
                       "num_seeds",
                       "rng_seed",
                       "placement",
                       "min_distance_factor",
                       "boundary_clearance",
                       "sampling_domain",
                       "lattice_type",
                       "lattice_jitter",
                       "file",
                       "units_scale",
                       "translate"});

    // Accepted spellings of `placement` and the SeedPlacement value each one selects.
    static const std::map<std::string, SeedPlacement> placement_map = {
        {"rejection", SeedPlacement::REJECTION},
        {"element_weighted", SeedPlacement::ELEMENT_WEIGHTED},
        {"lattice", SeedPlacement::LATTICE},
        {"johnson_mehl", SeedPlacement::JOHNSON_MEHL},
        {"file", SeedPlacement::FILE}};

    SeedOptions seeds;

    // Seed source and count.
    if (toml_input.contains("source")) {
        seeds.source = toml::find<std::string>(toml_input, "source");
    }
    if (toml_input.contains("num_seeds")) {
        seeds.num_seeds = toml::find<std::int64_t>(toml_input, "num_seeds");
    }

    // RNG key. TOML integers are signed 64-bit, and toml11 would convert a negative value to a
    // huge unsigned number without complaint (get<integral> is a plain cast, toml/get.hpp:92-95).
    // A negative key is almost certainly a mistake, so read it signed and reject it.
    if (toml_input.contains("rng_seed")) {
        const auto rng_seed = toml::find<std::int64_t>(toml_input, "rng_seed");
        if (rng_seed < 0) {
            throw std::invalid_argument("Microstructure options: 'Seeds.rng_seed' must be >= 0");
        }
        seeds.rng_seed = static_cast<std::uint64_t>(rng_seed);
    }

    // Placement algorithm. Writing only `source = "file"` is enough to read a seed table: file
    // placement is then implied. An explicit mismatch (e.g. source = "file" with
    // placement = "rejection") is reported by validate(), not silently corrected.
    if (toml_input.contains("placement")) {
        seeds.placement = string_to_ms_enum(
            toml::find<std::string>(toml_input, "placement"), placement_map, "Seeds.placement");
    } else if (seeds.source == "file") {
        seeds.placement = SeedPlacement::FILE;
    }

    // Spacing constraints for generated seeds (lengths or length factors; signs checked later).
    if (toml_input.contains("min_distance_factor")) {
        seeds.min_distance_factor = get_number(toml::find(toml_input, "min_distance_factor"));
    }
    if (toml_input.contains("boundary_clearance")) {
        seeds.boundary_clearance = get_number(toml::find(toml_input, "boundary_clearance"));
    }
    if (toml_input.contains("sampling_domain")) {
        seeds.sampling_domain = toml::find<std::string>(toml_input, "sampling_domain");
    }

    // Lattice placement parameters.
    if (toml_input.contains("lattice_type")) {
        seeds.lattice_type = toml::find<std::string>(toml_input, "lattice_type");
    }
    if (toml_input.contains("lattice_jitter")) {
        seeds.lattice_jitter = get_number(toml::find(toml_input, "lattice_jitter"));
    }

    // Seed table and the transform applied to its positions: x_mesh = units_scale * x_file +
    // translate. Only scale and shift are offered: orientations are passed through unchanged,
    // so rotating the seed cloud would leave the grain shapes and orientations inconsistent.
    if (toml_input.contains("file")) {
        seeds.file = toml::find<std::string>(toml_input, "file");
    }
    if (toml_input.contains("units_scale")) {
        seeds.units_scale = get_number(toml::find(toml_input, "units_scale"));
    }
    if (toml_input.contains("translate")) {
        seeds.translate = get_number_array<3>(toml_input, "translate");
    }
    return seeds;
}

/**
 * @brief Reads the [Tessellation] table: how "closest seed" is measured when labeling elements.
 *
 * Keys (defaults in TessellationOptions, microstructure_options.hpp):
 *   - `metric`: "voronoi" (plain distance, flat grain boundaries), "laguerre" (weighted, flat
 *     boundaries), "quadric" (anisotropic, elongated grains) or "additive" (curved boundaries).
 *     See MetricType in microstructure_types.hpp.
 *   - `anisotropy = {global_tensor = [Axx, Ayy, Azz, Ayz, Axz, Axy]}`: the symmetric 3x3 metric
 *     tensor A of the quadric metric, distance^2 = (x - s)^T A (x - s), in Voigt-like order
 *     (diagonal first, then yz, xz, xy). The identity reproduces Voronoi. A bare
 *     `global_tensor = [...]` directly in [Tessellation] is accepted as a shorthand; if both are
 *     written, the bare key wins (it is read last) and a rank-0 warning is printed.
 *   - `num_bins`: number of groups the seeds are split into by their weight/shape parameters
 *     when building the seed search structure (one search tree per group). Performance only.
 *   - `sampling`: "centroid" (label an element by its centroid) or "quadrature_vote" (majority
 *     vote over its quadrature points).
 *   - `periodic = [px, py, pz]`: whether the microstructure wraps around in x, y, z (grains cut
 *     by one face continue on the opposite face).
 *   - `distance`: "euclidean", "connected" or "geodesic" (see DistanceMode).
 *
 * @throws std::invalid_argument for an unknown metric/sampling/distance name, a tensor without 6
 *         entries, or `periodic` without 3 entries.
 */
TessellationOptions tessellation_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "Tessellation",
                      {"metric",
                       "anisotropy",
                       "global_tensor",
                       "num_bins",
                       "sampling",
                       "periodic",
                       "distance"});

    // Accepted spellings of the enum-valued keys.
    static const std::map<std::string, MetricType> metric_map = {
        {"voronoi", MetricType::VORONOI},
        {"laguerre", MetricType::LAGUERRE},
        {"quadric", MetricType::QUADRIC},
        {"additive", MetricType::ADDITIVE}};
    static const std::map<std::string, SamplingMode> sampling_map = {
        {"centroid", SamplingMode::CENTROID}, {"quadrature_vote", SamplingMode::QUADRATURE_VOTE}};
    static const std::map<std::string, DistanceMode> distance_map = {
        {"euclidean", DistanceMode::EUCLIDEAN},
        {"connected", DistanceMode::CONNECTED},
        {"geodesic", DistanceMode::GEODESIC}};

    TessellationOptions tess;

    // Metric family.
    if (toml_input.contains("metric")) {
        tess.metric = string_to_ms_enum(
            toml::find<std::string>(toml_input, "metric"), metric_map, "Tessellation.metric");
    }

    // Anisotropy tensor, first as an inline table, then the bare shorthand.
    // Whether the tensor is a valid metric (positive definite) is checked in validate().
    if (toml_input.contains("anisotropy")) {
        const auto& anisotropy = toml::find(toml_input, "anisotropy");
        warn_unknown_keys(anisotropy, "Tessellation.anisotropy", {"global_tensor"});
        if (anisotropy.contains("global_tensor")) {
            tess.global_tensor = get_number_array<6>(anisotropy, "global_tensor");
        }
    }
    if (toml_input.contains("global_tensor")) {
        bool inline_form_written = false;
        if (toml_input.contains("anisotropy")) {
            inline_form_written = toml::find(toml_input, "anisotropy").contains("global_tensor");
        }
        if (inline_form_written) {
            WARNING_0_OPT("Microstructure options: both 'Tessellation.anisotropy.global_tensor' "
                          "and 'Tessellation.global_tensor' are set; using "
                          "'Tessellation.global_tensor'");
        }
        tess.global_tensor = get_number_array<6>(toml_input, "global_tensor");
    }

    // Seed search structure (performance setting; checked >= 1 in validate()).
    if (toml_input.contains("num_bins")) {
        tess.num_bins = get_int(toml_input, "num_bins", "Tessellation.num_bins");
    }

    // Where each element is evaluated.
    if (toml_input.contains("sampling")) {
        tess.sampling = string_to_ms_enum(toml::find<std::string>(toml_input, "sampling"),
                                          sampling_map,
                                          "Tessellation.sampling");
    }

    // Periodicity flags, one per direction x, y, z.
    if (toml_input.contains("periodic")) {
        const auto periodic = toml::find<std::vector<bool>>(toml_input, "periodic");
        if (periodic.size() != 3) {
            throw std::invalid_argument(
                "Microstructure options: 'Tessellation.periodic' needs 3 values");
        }
        for (std::size_t d = 0; d < 3; ++d) {
            tess.periodic[d] = periodic[d];
        }
    }

    // How distance is measured through (or around) the material.
    if (toml_input.contains("distance")) {
        tess.distance = string_to_ms_enum(toml::find<std::string>(toml_input, "distance"),
                                          distance_map,
                                          "Tessellation.distance");
    }
    return tess;
}

/**
 * @brief Reads the [Output] table: which files are written and how.
 *
 * Keys (defaults in OutputOptions, microstructure_options.hpp):
 *   - `directory`, `basename`: every output file is directory/basename.<extension>.
 *   - `mesh_format`: "serial" (one mesh file), "parallel" (one file per part, readable by
 *     ExaConstit's [Mesh] type = "parallel") or "both".
 *   - `num_parts`: number of parts of the parallel output; 0 means one part per MPI rank.
 *   - `compression`: gzip level 0-9 of the mesh files; 0 writes plain text.
 *   - `grain_file`: write the per-element grain ids to a separate grain file: "auto" (only for
 *     auto meshes, which ExaConstit rebuilds itself and so cannot carry the ids inside the mesh
 *     file), "true" or "false". Both `grain_file = true` and `grain_file = "true"` are accepted.
 *   - `paraview`: boolean that switches the ParaView visualization of the grain ids on or off.
 * The tool does not write ExaConstit input decks; users write their own options.toml.
 *
 * @throws std::invalid_argument for an unknown mesh_format.
 */
OutputOptions output_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "Output",
                      {"directory",
                       "basename",
                       "mesh_format",
                       "num_parts",
                       "compression",
                       "grain_file",
                       "paraview"});

    // Accepted spellings of `mesh_format`.
    static const std::map<std::string, MeshFormat> format_map = {
        {"serial", MeshFormat::SERIAL},
        {"parallel", MeshFormat::PARALLEL},
        {"both", MeshFormat::BOTH}};

    OutputOptions output;

    // File names.
    if (toml_input.contains("directory")) {
        output.directory = toml::find<std::string>(toml_input, "directory");
    }
    if (toml_input.contains("basename")) {
        output.basename = toml::find<std::string>(toml_input, "basename");
    }

    // Mesh file layout (ranges of num_parts and compression are checked in validate()).
    if (toml_input.contains("mesh_format")) {
        output.mesh_format = string_to_ms_enum(
            toml::find<std::string>(toml_input, "mesh_format"), format_map, "Output.mesh_format");
    }
    if (toml_input.contains("num_parts")) {
        output.num_parts = get_int(toml_input, "num_parts", "Output.num_parts");
    }
    if (toml_input.contains("compression")) {
        output.compression = get_int(toml_input, "compression", "Output.compression");
    }

    // grain_file has three states ("auto", "true", "false"), so it is stored as a string. Accept
    // the natural boolean spelling as well and convert it to the matching string.
    if (toml_input.contains("grain_file")) {
        const auto& value = toml::find(toml_input, "grain_file");
        if (value.is_boolean()) {
            if (value.as_boolean()) {
                output.grain_file = "true";
            } else {
                output.grain_file = "false";
            }
        } else {
            output.grain_file = toml::find<std::string>(toml_input, "grain_file");
        }
    }

    // Optional visualization output.
    if (toml_input.contains("paraview")) {
        output.paraview = toml::find<bool>(toml_input, "paraview");
    }
    return output;
}

/**
 * @brief Records one validation error: prints it on rank 0 and marks the options invalid.
 *
 * validate() keeps going after an error so that the user sees every problem in one run instead
 * of fixing them one at a time.
 */
void report_error(bool& valid, const std::string& message) {
    WARNING_0_OPT("Error: " + message);
    valid = false;
}

/**
 * @brief Checks that a string option is one of its accepted values; reports an error otherwise.
 *
 * Used by validate() for options stored as strings rather than enums (mesh type, seed source,
 * fragments, ...). The message lists the accepted values.
 *
 * @param valid    Set to false if `value` is not accepted; left unchanged otherwise.
 * @param key      Dotted key name for the message.
 * @param value    The value to check (exact, case-sensitive match).
 * @param accepted The accepted spellings.
 */
void check_choice(bool& valid,
                  const std::string& key,
                  const std::string& value,
                  std::initializer_list<const char*> accepted) {
    for (const char* accepted_value : accepted) {
        if (value == accepted_value) {
            return;
        }
    }
    std::ostringstream err;
    err << "microstructure option '" << key << "' = \"" << value << "\" is not one of:";
    for (const char* accepted_value : accepted) {
        err << " \"" << accepted_value << "\"";
    }
    report_error(valid, err.str());
}

} // namespace

/**
 * @brief Builds the options from an already parsed TOML document.
 *
 * Reads the top-level keys, then hands each table to its reader above. Any key that is not
 * written keeps its default from microstructure_options.hpp, so an empty document gives the
 * "-n N" quick-path defaults (Voronoi, random seeds, random orientations, one region).
 *
 * Top-level keys and tables:
 *   - `exec`: where kernels run, "cpu", "openmp" or "gpu" (spelling checked in validate()).
 *   - `deterministic`: require results independent of rank count and execution space.
 *   - [Mesh], [Seeds], [Tessellation], [Output]: see the readers above.
 *   - [Source] `type`: what defines the grains. Only "tessellation" exists so far.
 *   - [Cleanup] `fragments` ("auto", "split", "merge", "none") and `min_grain_elements`: what to
 *     do with grains that end up split into disconnected pieces, or that are smaller than the
 *     given number of elements.
 *   - [Orientations] `source`: "random", "from_source" or "list".
 *   - [Regions] `mode`: "single", "from_phase" or "phase_fractions" (how grains are grouped into
 *     ExaConstit material regions).
 * The string values of [Cleanup], [Orientations] and [Regions] are checked in validate().
 *
 * Not collective; every rank may call it on its own copy of the document.
 *
 * @throws std::invalid_argument for values that cannot be stored (see the table readers) and
 *         for an unsupported [Source] type; toml11 exceptions for wrong TOML types.
 */
MicrostructureOptions MicrostructureOptions::from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "",
                      {"exec",
                       "deterministic",
                       "Mesh",
                       "Source",
                       "Seeds",
                       "Tessellation",
                       "Cleanup",
                       "Orientations",
                       "Regions",
                       "Output"});

    MicrostructureOptions options;

    // Top-level scalars.
    if (toml_input.contains("exec")) {
        options.exec = toml::find<std::string>(toml_input, "exec");
    }
    if (toml_input.contains("deterministic")) {
        options.deterministic = toml::find<bool>(toml_input, "deterministic");
    }

    // [Mesh]
    if (toml_input.contains("Mesh")) {
        options.mesh = mesh_from_toml(toml::find(toml_input, "Mesh"));
    }

    // [Source]: only tessellations are implemented. A later stage adds voxel maps (grain ids read
    // from e.g. a DREAM.3D file). If such an input were merely warned about, the tool would
    // tessellate random seeds instead and produce a completely different microstructure without
    // the user noticing, so any other type is rejected outright.
    if (toml_input.contains("Source")) {
        const auto& source = toml::find(toml_input, "Source");
        warn_unknown_keys(source, "Source", {"type"});
        if (source.contains("type")) {
            const auto type = toml::find<std::string>(source, "type");
            if (type != "tessellation") {
                throw std::invalid_argument("Microstructure options: 'Source.type' = \"" + type +
                                            "\" is not supported yet (only \"tessellation\")");
            }
        }
    }

    // [Seeds] and [Tessellation]
    if (toml_input.contains("Seeds")) {
        options.seeds = seeds_from_toml(toml::find(toml_input, "Seeds"));
    }
    if (toml_input.contains("Tessellation")) {
        options.tessellation = tessellation_from_toml(toml::find(toml_input, "Tessellation"));
    }

    // [Cleanup]: small tables, read inline.
    if (toml_input.contains("Cleanup")) {
        const auto& cleanup = toml::find(toml_input, "Cleanup");
        warn_unknown_keys(cleanup, "Cleanup", {"fragments", "min_grain_elements"});
        if (cleanup.contains("fragments")) {
            options.cleanup.fragments = toml::find<std::string>(cleanup, "fragments");
        }
        if (cleanup.contains("min_grain_elements")) {
            options.cleanup.min_grain_elements =
                get_int(cleanup, "min_grain_elements", "Cleanup.min_grain_elements");
        }
    }

    // [Orientations]
    if (toml_input.contains("Orientations")) {
        const auto& orientations = toml::find(toml_input, "Orientations");
        warn_unknown_keys(orientations, "Orientations", {"source"});
        if (orientations.contains("source")) {
            options.orientations.source = toml::find<std::string>(orientations, "source");
        }
    }

    // [Regions]
    if (toml_input.contains("Regions")) {
        const auto& regions = toml::find(toml_input, "Regions");
        warn_unknown_keys(regions, "Regions", {"mode"});
        if (regions.contains("mode")) {
            options.regions.mode = toml::find<std::string>(regions, "mode");
        }
    }

    // [Output]
    if (toml_input.contains("Output")) {
        options.output = output_from_toml(toml::find(toml_input, "Output"));
    }
    return options;
}

/**
 * @brief Reads a TOML options file from disk and builds the options from it.
 *
 * Steps: check that the file exists, parse it with toml::parse, and pass the document to
 * from_toml(). The options are NOT validated here: the tool applies its command-line overrides
 * first and then calls validate() itself.
 *
 * Not collective; every rank reads the file independently (it is small).
 *
 * @throws std::runtime_error if the file does not exist; toml11 syntax errors (with line
 *         numbers) for malformed TOML; and whatever from_toml() throws.
 */
MicrostructureOptions MicrostructureOptions::parse_options_file(const fs::path& path) {
    // Checked here so the error names the file; toml::parse would only say it cannot open it.
    if (!fs::exists(path)) {
        throw std::runtime_error("Microstructure options file '" + path.string() +
                                 "' does not exist");
    }
    return from_toml(toml::parse(path.string()));
}

/**
 * @brief Checks that the final options (file + CLI overrides) describe a run that can work.
 *
 * Each check below protects against a specific failure that would otherwise appear much later,
 * often on many ranks at once and with a less helpful message (a crash while reading a mesh, an
 * empty microstructure, a negative distance). The checks run in table order; every failed check
 * prints one "Error: ..." line on rank 0 and the function returns false at the end, after all
 * problems have been reported.
 *
 * Relative paths (`Mesh.floc`, `Seeds.file`) are checked relative to the current working
 * directory, which is how they are opened later.
 *
 * Not collective (no MPI communication), but every rank holds the same options and so returns
 * the same value; the tool can therefore abort on all ranks together.
 *
 * @return true if every check passed.
 */
bool MicrostructureOptions::validate() const {
    bool valid = true;

    // --- Execution space --------------------------------------------------------------------
    // First the spelling, then whether this build actually has that backend. The backend is the
    // mfem::Device, configured as in ExaConstit's mechanics driver ("cpu", "openmp" -> raja-omp,
    // "gpu" -> raja-cuda or raja-hip). IsExecAvailable (exec_dispatch.hpp) answers from the
    // build configuration, so asking for "gpu" in a CPU-only build is caught here, on every
    // rank, instead of when the device is configured. The availability check runs only for a
    // correct spelling, so a typo is reported once.
    bool exec_spelling_ok = true;
    check_choice(exec_spelling_ok, "exec", exec, {"cpu", "openmp", "gpu"});
    if (!exec_spelling_ok) {
        valid = false;
    } else if (!IsExecAvailable(exec)) {
        report_error(valid,
                     "exec = \"" + exec + "\" requested, but this build has no such backend");
    }

    // --- [Mesh] -----------------------------------------------------------------------------
    // The mesh must be loadable; catching a missing file here avoids every rank failing inside
    // the MFEM mesh reader.
    check_choice(valid, "Mesh.type", mesh.type, {"file", "auto", "parallel"});
    if (mesh.type == "file") {
        if (mesh.floc.empty()) {
            report_error(valid, "[Mesh] type = \"file\" needs 'floc'");
        } else if (!fs::exists(mesh.floc)) {
            report_error(valid, "mesh file '" + mesh.floc.string() + "' does not exist");
        }
    } else if (mesh.type == "parallel") {
        // floc is the prefix of a decomposition written with mfem::MakeParFilename
        // (mfem/general/globals.hpp:84), which appends the zero-padded part number, e.g.
        // "mesh." -> "mesh.000000". Part 0 always exists, so checking it catches a wrong prefix.
        if (mesh.floc.empty()) {
            report_error(valid, "[Mesh] type = \"parallel\" needs 'floc' (the file prefix)");
        } else {
            const std::string part0 = mfem::MakeParFilename(mesh.floc.string(), 0);
            if (!fs::exists(part0)) {
                report_error(valid,
                             "parallel mesh part '" + part0 + "' (prefix '" + mesh.floc.string() +
                                 "') does not exist");
            }
        }
    } else if (mesh.type == "auto") {
        // A generated box needs at least one element and a finite, positive length in every
        // direction; otherwise the mesh would be empty, inverted (negative element volumes) or
        // have NaN/infinite node coordinates (TOML accepts `nan` and `inf`).
        for (std::size_t d = 0; d < 3; ++d) {
            if (mesh.nxyz[d] <= 0 || !is_positive_finite(mesh.mxyz[d])) {
                report_error(valid,
                             "[Mesh.Auto] needs nxyz > 0 and finite mxyz > 0 in every direction");
                break;
            }
        }
    }

    // --- [Seeds] ----------------------------------------------------------------------------
    check_choice(valid, "Seeds.source", seeds.source, {"random", "file"});
    if (seeds.source == "random") {
        // Zero seeds would leave every element without a grain.
        if (seeds.num_seeds <= 0) {
            report_error(valid, "random seeds need 'Seeds.num_seeds' > 0");
        }
        // File placement reads positions from a table; with random seeds there is no table.
        if (seeds.placement == SeedPlacement::FILE) {
            report_error(valid, "Seeds.placement = \"file\" requires Seeds.source = \"file\"");
        }
    } else if (seeds.source == "file") {
        // A seed table fixes the positions, so a generating placement would be ignored or
        // contradict the table; require the two to agree instead of guessing.
        if (seeds.placement != SeedPlacement::FILE) {
            report_error(valid, "Seeds.source = \"file\" requires Seeds.placement = \"file\"");
        }
        // The number of grains is the number of rows of the table, so num_seeds is not needed,
        // but the table must exist.
        if (seeds.file.empty()) {
            report_error(valid, "Seeds.source = \"file\" needs 'Seeds.file'");
        } else if (!fs::exists(seeds.file)) {
            report_error(valid, "seed file '" + seeds.file.string() + "' does not exist");
        }
    }
    check_choice(valid, "Seeds.sampling_domain", seeds.sampling_domain, {"material", "bbox"});
    check_choice(valid, "Seeds.lattice_type", seeds.lattice_type, {"cubic", "bcc", "fcc", "hcp"});
    // Distances and jitter are magnitudes; a negative value has no physical meaning and would
    // turn the spacing tests upside down. NaN and inf (accepted by TOML) are rejected too: NaN
    // makes every comparison false and inf would reject every candidate seed.
    if (!is_nonnegative_finite(seeds.min_distance_factor)) {
        report_error(valid, "Seeds.min_distance_factor must be finite and >= 0");
    }
    if (!is_nonnegative_finite(seeds.boundary_clearance)) {
        report_error(valid, "Seeds.boundary_clearance must be finite and >= 0");
    }
    if (!is_nonnegative_finite(seeds.lattice_jitter)) {
        report_error(valid, "Seeds.lattice_jitter must be finite and >= 0");
    }
    // A zero scale collapses every seed onto one point and a negative one mirrors the table
    // (which would also mirror the grain shapes relative to their orientations); NaN or inf
    // would give meaningless seed positions.
    if (!is_positive_finite(seeds.units_scale)) {
        report_error(valid, "Seeds.units_scale must be finite and > 0");
    }
    // The shift must be a real position as well.
    for (std::size_t d = 0; d < 3; ++d) {
        if (!std::isfinite(seeds.translate[d])) {
            report_error(valid, "Seeds.translate must be finite in every direction");
            break;
        }
    }

    // --- [Tessellation] ---------------------------------------------------------------------
    // At least one group of seeds is needed to build the seed search structure.
    if (tessellation.num_bins < 1) {
        report_error(valid, "Tessellation.num_bins must be >= 1");
    }
    {
        // The anisotropy tensor A defines distance^2 = (x - s)^T A (x - s). It is a valid
        // distance only if A is positive definite (every direction has a positive length). For
        // the symmetric matrix
        //     A = [[xx, xy, xz],
        //          [xy, yy, yz],
        //          [xz, yz, zz]]
        // Sylvester's criterion says this holds exactly when the leading principal minors are
        // all positive: xx > 0, det of the top-left 2x2 > 0, and det(A) > 0.
        // Storage order of global_tensor: [xx, yy, zz, yz, xz, xy].
        const auto& tensor = tessellation.global_tensor;
        const double xx = tensor[0];
        const double yy = tensor[1];
        const double zz = tensor[2];
        const double yz = tensor[3];
        const double xz = tensor[4];
        const double xy = tensor[5];
        const double minor_2x2 = xx * yy - xy * xy;
        // det(A) by cofactor expansion along the first row.
        const double det_3x3 = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) +
                               xz * (xy * yz - yy * xz);
        // Every entry must be finite (an inf diagonal entry would pass the minor tests), and the
        // test is written as !(...) so that NaN results are rejected as well.
        bool all_finite = true;
        for (const double entry : tensor) {
            if (!std::isfinite(entry)) {
                all_finite = false;
            }
        }
        if (!(all_finite && xx > 0.0 && minor_2x2 > 0.0 && det_3x3 > 0.0)) {
            report_error(valid, "Tessellation global_tensor must be symmetric positive definite");
        }
    }

    // --- [Cleanup], [Orientations], [Regions] -----------------------------------------------
    check_choice(valid, "Cleanup.fragments", cleanup.fragments, {"auto", "split", "merge", "none"});
    // An element count below zero is meaningless (0 or 1 means no grain is too small).
    if (cleanup.min_grain_elements < 0) {
        report_error(valid, "Cleanup.min_grain_elements must be >= 0");
    }
    check_choice(
        valid, "Orientations.source", orientations.source, {"random", "from_source", "list"});
    check_choice(
        valid, "Regions.mode", regions.mode, {"single", "from_phase", "phase_fractions"});

    // --- [Output] ---------------------------------------------------------------------------
    // Every output file name is built from basename; an empty one would give nameless files
    // such as ".mesh".
    if (output.basename.empty()) {
        report_error(valid, "Output.basename must not be empty");
    }
    if (output.num_parts < 0) {
        report_error(valid, "Output.num_parts must be >= 0 (0 means the number of ranks)");
    }
    // zlib/gzip only knows levels 0 (store) to 9 (best).
    if (output.compression < 0 || output.compression > 9) {
        report_error(valid, "Output.compression must be in 0-9");
    }
    check_choice(valid, "Output.grain_file", output.grain_file, {"auto", "true", "false"});

    return valid;
}

/**
 * @brief Applies the tool's command-line flags on top of the options read from the file.
 *
 * Only flags that were given change anything (unset std::optional fields are skipped). Flags:
 *   - `-m FILE`:   use this serial mesh file (sets [Mesh] type = "file").
 *   - `-n N`:      generate N random seeds. This is the quick path, so it also switches a seed
 *                  table in the options file back to random seeds with rejection placement. If
 *                  the file already asked for random seeds, its placement is kept.
 *   - `-s SEED`:   RNG key (Seeds.rng_seed).
 *   - `-o NAME`:   output basename.
 *   - `--parts N`: number of parallel parts. Asking for parts means asking for a parallel
 *                  decomposition, so a "serial" mesh_format becomes "both" (the serial file is
 *                  kept as well). An explicit "parallel" or "both" is left as is.
 *
 * Call before validate(), so that the overridden values are the ones checked.
 */
void MicrostructureOptions::apply_cli_overrides(const CliOverrides& overrides) {
    // -m
    if (overrides.mesh) {
        mesh.type = "file";
        mesh.floc = *overrides.mesh;
    }
    // -n: random quick path, even on top of an options file with a seed table.
    if (overrides.num_seeds) {
        seeds.num_seeds = *overrides.num_seeds;
        if (seeds.source != "random") {
            seeds.source = "random";
            seeds.placement = SeedPlacement::REJECTION;
        }
    }
    // -s
    if (overrides.rng_seed) {
        seeds.rng_seed = *overrides.rng_seed;
    }
    // -o
    if (overrides.basename) {
        output.basename = *overrides.basename;
    }
    // --parts: request a decomposition, keeping the serial file if it was going to be written.
    if (overrides.num_parts) {
        output.num_parts = *overrides.num_parts;
        if (output.num_parts > 0 && output.mesh_format == MeshFormat::SERIAL) {
            output.mesh_format = MeshFormat::BOTH;
        }
    }
}

} // namespace exaconstit::microstructure
