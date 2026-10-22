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
// The compile-time constant CONNECTED_MAX_ROUNDS_LIMIT only (header-only; ms_options does not
// link ms_labeling)
#include "microstructure/labeling/connected_limits.hpp"
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
 * belongs to a later stage of the generator (e.g. [Conformal], [VoxelMap])
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
 * @brief Converts a TOML table key that names an integer (e.g. the "2" of
 * `passthrough_attributes = { "2" = 3 }`) to an int.
 *
 * TOML keys are always strings, so integer-valued keys arrive as text. The whole key (after
 * trimming nothing: keys are taken literally) must be a decimal integer that fits in an int;
 * "2a", "", "1.5" or "99999999999" are refused rather than read as a prefix or wrapped around.
 *
 * @param key      The key text.
 * @param full_key Dotted name of the table, used only in the error message.
 * @throws std::invalid_argument if the key is not an integer in the range of int.
 */
int key_to_int(const std::string& key, const std::string& full_key) {
    std::size_t pos = 0;
    long long value = 0;
    // The first character must be a digit or '-': std::stoll would also accept leading blanks
    // and a '+', which are not how an integer key is written
    bool ok = !key.empty() && ((key[0] >= '0' && key[0] <= '9') || key[0] == '-');
    if (ok) {
        try {
            value = std::stoll(key, &pos, 10);
        } catch (const std::exception&) {
            ok = false;
        }
    }
    if (!ok || pos != key.size() || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Microstructure options: key '" + key + "' of '" + full_key +
                                    "' must be an integer");
    }
    return static_cast<int>(value);
}

/**
 * @brief Reads `table[key]` as an inline table of integer -> integer entries, e.g.
 * `phase_to_region = { "0" = 1, "3" = 1, "7" = 2 }` -> {0: 1, 3: 1, 7: 2}.
 *
 * Keys are converted with key_to_int, values must be TOML integers that fit in an int. Ranges
 * (e.g. region ids >= 1) are checked in validate().
 *
 * @throws std::invalid_argument for a non-integer key or a value out of range; toml11 errors for
 *         a value that is not a TOML integer or a `key` that is not a table.
 */
std::map<int, int> get_int_map(const toml::value& table,
                               const std::string& key,
                               const std::string& full_key) {
    std::map<int, int> result;
    for (const auto& entry : toml::find(table, key).as_table()) {
        const int map_key = key_to_int(entry.first, full_key);
        result[map_key] = checked_int(entry.second.as_integer(), full_key + "." + entry.first);
    }
    return result;
}

/**
 * @brief Reads `table[key]` as an array of integers that fit in an int (e.g. a list of element
 * attributes or phase ids).
 *
 * @throws std::invalid_argument for a value out of range; toml11 errors for non-integers.
 */
std::vector<int> get_int_vector(const toml::value& table,
                                const std::string& key,
                                const std::string& full_key) {
    std::vector<int> result;
    for (const std::int64_t value : toml::find<std::vector<std::int64_t>>(table, key)) {
        result.push_back(checked_int(value, full_key));
    }
    return result;
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
 *   - `active_attributes = [..]`: element attributes that receive grains (empty = all that are
 *     not passthrough).
 *   - `passthrough_attributes = { "attr" = region }`: element attributes kept as one grain each,
 *     with the ExaConstit region id of that part.
 *
 * @throws std::invalid_argument if `length`/`ncuts` does not have 3 entries, or an attribute or
 *         region id is not an integer that fits in an int.
 */
MeshInput mesh_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "Mesh",
                      {"type", "floc", "Auto", "active_attributes", "passthrough_attributes"});
    MeshInput mesh;

    // Mesh source and location; checked against the disk in validate(), after CLI overrides.
    if (toml_input.contains("type")) {
        mesh.type = toml::find<std::string>(toml_input, "type");
    }
    if (toml_input.contains("floc")) {
        mesh.floc = toml::find<std::string>(toml_input, "floc");
    }

    // Which element attributes get grains (active) and which are kept as one grain each
    // (passthrough, attribute -> region). Ranges and overlaps are checked in validate(); whether
    // the attributes exist is only known once the mesh is read (mesh/attribute_domain.hpp).
    if (toml_input.contains("active_attributes")) {
        mesh.active_attributes =
            get_int_vector(toml_input, "active_attributes", "Mesh.active_attributes");
    }
    if (toml_input.contains("passthrough_attributes")) {
        mesh.passthrough_attributes =
            get_int_map(toml_input, "passthrough_attributes", "Mesh.passthrough_attributes");
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
 *   - `min_distance`: absolute minimum seed spacing (mesh length units); when > 0 it replaces
 *     min_distance_factor (validate() warns when both are > 0).
 *   - `boundary_clearance`: minimum distance of a seed from the domain boundary; 0 disables it.
 *   - `sampling_domain`: "material" (seeds only inside the mesh) or "bbox" (anywhere in the
 *     sampling box: the mesh's bounding box, or `sampling_box` when given).
 *   - `sampling_box = [xlo, ylo, zlo, xhi, yhi, zhi]`: draw seeds only inside this box instead
 *     of the whole mesh bounding box (mesh length units; corners checked in validate()).
 *   - `allow_fewer_seeds`: if seed generation stalls (spacing too large for the volume), keep
 *     the seeds placed so far instead of stopping with an error.
 *   - `lattice_type` ("cubic", "bcc", "fcc", "hcp") and `lattice_jitter` (random offset as a
 *     fraction of the lattice spacing), for placement = "lattice".
 *   - `file`, `units_scale`, `translate`: seed table path and the scale-then-translate applied
 *     to its positions to bring them into mesh coordinates.
 *   - `roi = {lo = [..], hi = [..]}`: region of interest of the table (mesh units); rows outside
 *     it are dropped. `fill_outside_roi` ("none", "random"), `filler_num_seeds` (0 = automatic)
 *     and `filler_min_distance_factor`: random filler seeds for the material outside the ROI.
 *   - [Seeds.Size] sub-table: `distribution` ("none", "lognormal", "from_source"),
 *     `mean_diameter` or `mean_volume`, `sigma`, `polydisperse_placement`, `exclusion_factor`
 *     (see SeedSizeOptions).
 *   - [Seeds.JohnsonMehl] sub-table: `rate`, `growth_speed`, `end_time`, `num_nuclei` (see
 *     JohnsonMehlOptions), for placement = "johnson_mehl".
 *
 * @throws std::invalid_argument for a negative rng_seed, an unknown placement, a translate
 *         that does not have 3 entries, a sampling_box that does not have 6 entries, or a `roi`
 *         without both corners of 3 entries.
 */
SeedOptions seeds_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "Seeds",
                      {"source",
                       "num_seeds",
                       "rng_seed",
                       "placement",
                       "min_distance_factor",
                       "min_distance",
                       "boundary_clearance",
                       "sampling_domain",
                       "sampling_box",
                       "allow_fewer_seeds",
                       "lattice_type",
                       "lattice_jitter",
                       "file",
                       "units_scale",
                       "translate",
                       "roi",
                       "fill_outside_roi",
                       "filler_num_seeds",
                       "filler_min_distance_factor",
                       "Size",
                       "JohnsonMehl"});

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
    if (toml_input.contains("min_distance")) {
        seeds.min_distance = get_number(toml::find(toml_input, "min_distance"));
    }
    if (toml_input.contains("boundary_clearance")) {
        seeds.boundary_clearance = get_number(toml::find(toml_input, "boundary_clearance"));
    }
    if (toml_input.contains("sampling_domain")) {
        seeds.sampling_domain = toml::find<std::string>(toml_input, "sampling_domain");
    }
    if (toml_input.contains("allow_fewer_seeds")) {
        seeds.allow_fewer_seeds = toml::find<bool>(toml_input, "allow_fewer_seeds");
    }

    // User sampling box, written as one flat array [lo corner, hi corner]. Its presence switches
    // the box on; whether hi > lo is checked in validate() so all problems are reported together.
    if (toml_input.contains("sampling_box")) {
        const std::array<double, 6> box = get_number_array<6>(toml_input, "sampling_box");
        seeds.use_sampling_box = true;
        for (std::size_t d = 0; d < 3; ++d) {
            seeds.sampling_box_lo[d] = box[d];
            seeds.sampling_box_hi[d] = box[d + 3];
        }
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

    // Region of interest of the seed table and the filler seeds outside it. The box corners
    // (hi > lo) and the combinations are checked in validate().
    if (toml_input.contains("roi")) {
        const auto& roi = toml::find(toml_input, "roi");
        warn_unknown_keys(roi, "Seeds.roi", {"lo", "hi"});
        if (!roi.contains("lo") || !roi.contains("hi")) {
            throw std::invalid_argument(
                "Microstructure options: 'Seeds.roi' needs both 'lo' and 'hi' (3 values each)");
        }
        seeds.use_roi = true;
        seeds.roi_lo = get_number_array<3>(roi, "lo");
        seeds.roi_hi = get_number_array<3>(roi, "hi");
    }
    if (toml_input.contains("fill_outside_roi")) {
        seeds.fill_outside_roi = toml::find<std::string>(toml_input, "fill_outside_roi");
    }
    if (toml_input.contains("filler_num_seeds")) {
        seeds.filler_num_seeds = toml::find<std::int64_t>(toml_input, "filler_num_seeds");
    }
    if (toml_input.contains("filler_min_distance_factor")) {
        seeds.filler_min_distance_factor =
            get_number(toml::find(toml_input, "filler_min_distance_factor"));
    }

    // [Seeds.Size]: target grain size distribution (values checked in validate())
    if (toml_input.contains("Size")) {
        const auto& size = toml::find(toml_input, "Size");
        warn_unknown_keys(size,
                          "Seeds.Size",
                          {"distribution",
                           "mean_diameter",
                           "mean_volume",
                           "sigma",
                           "polydisperse_placement",
                           "exclusion_factor"});
        if (size.contains("distribution")) {
            seeds.size.distribution = toml::find<std::string>(size, "distribution");
        }
        if (size.contains("mean_diameter")) {
            seeds.size.mean_diameter = get_number(toml::find(size, "mean_diameter"));
        }
        if (size.contains("mean_volume")) {
            seeds.size.mean_volume = get_number(toml::find(size, "mean_volume"));
        }
        if (size.contains("sigma")) {
            seeds.size.sigma = get_number(toml::find(size, "sigma"));
        }
        if (size.contains("polydisperse_placement")) {
            seeds.size.polydisperse_placement = toml::find<bool>(size, "polydisperse_placement");
        }
        if (size.contains("exclusion_factor")) {
            seeds.size.exclusion_factor = get_number(toml::find(size, "exclusion_factor"));
        }
    }

    // [Seeds.JohnsonMehl]: nucleation and growth parameters (values checked in validate())
    if (toml_input.contains("JohnsonMehl")) {
        const auto& jm = toml::find(toml_input, "JohnsonMehl");
        warn_unknown_keys(
            jm, "Seeds.JohnsonMehl", {"rate", "growth_speed", "end_time", "num_nuclei"});
        if (jm.contains("rate")) {
            seeds.johnson_mehl.rate = get_number(toml::find(jm, "rate"));
        }
        if (jm.contains("growth_speed")) {
            seeds.johnson_mehl.growth_speed = get_number(toml::find(jm, "growth_speed"));
        }
        if (jm.contains("end_time")) {
            seeds.johnson_mehl.end_time = get_number(toml::find(jm, "end_time"));
        }
        if (jm.contains("num_nuclei")) {
            seeds.johnson_mehl.num_nuclei = toml::find<std::int64_t>(jm, "num_nuclei");
        }
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
 *   - `distance`: "euclidean", "connected" or "geodesic" (see DistanceMode and
 *     TessellationOptions::distance).
 *   - `connected_max_rounds`: relabeling rounds of the connected mode (>= 0, checked in
 *     validate()).
 *   - `geodesic = {solver}`: "fim" or "graph", the path-length solver of the geodesic mode
 *     (spelling and availability checked in validate()).
 *   - `periodic_tolerance`: matching tolerance of the periodic face pairing (fraction of the
 *     smallest face edge on the periodic planes).
 *   - `vote_levels`: 0-3, resolution of quadrature voting (8^levels points per element).
 *   - `anisotropy = {kind, global_tensor, axis, normal}`: kind "none", "tensor" (implied by a
 *     written global_tensor), "columnar" (with `axis`) or "lamellar" (with `normal`).
 *   - `fit_volumes = {enabled, tolerance, max_iterations, samples}`: volume fitting.
 *   - `lloyd_iterations`, `centroidal_power`: Lloyd relaxation (optionally with volume refits).
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
                       "vote_levels",
                       "periodic",
                       "periodic_tolerance",
                       "distance",
                       "connected_max_rounds",
                       "geodesic",
                       "fit_volumes",
                       "lloyd_iterations",
                       "centroidal_power"});

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
        warn_unknown_keys(
            anisotropy, "Tessellation.anisotropy", {"kind", "global_tensor", "axis", "normal"});
        if (anisotropy.contains("global_tensor")) {
            tess.global_tensor = get_number_array<6>(anisotropy, "global_tensor");
            tess.anisotropy_kind = "tensor";
        }
        if (anisotropy.contains("axis")) {
            tess.anisotropy_axis = get_number_array<3>(anisotropy, "axis");
        }
        if (anisotropy.contains("normal")) {
            tess.anisotropy_normal = get_number_array<3>(anisotropy, "normal");
        }
        // An explicit kind wins over the "tensor" implied by a written global_tensor; its
        // spelling and consistency are checked in validate()
        if (anisotropy.contains("kind")) {
            tess.anisotropy_kind = toml::find<std::string>(anisotropy, "kind");
        }
        // A direction without its kind has no effect; say so instead of ignoring it silently
        if (anisotropy.contains("axis") && tess.anisotropy_kind != "columnar") {
            WARNING_0_OPT("Microstructure options: 'Tessellation.anisotropy.axis' is ignored "
                          "unless kind = \"columnar\"");
        }
        if (anisotropy.contains("normal") && tess.anisotropy_kind != "lamellar") {
            WARNING_0_OPT("Microstructure options: 'Tessellation.anisotropy.normal' is ignored "
                          "unless kind = \"lamellar\"");
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
        bool explicit_kind = false;
        if (toml_input.contains("anisotropy")) {
            explicit_kind = toml::find(toml_input, "anisotropy").contains("kind");
        }
        if (!explicit_kind) {
            tess.anisotropy_kind = "tensor";
        }
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

    // Distance-mode settings: the rounds cap of "connected" and the solver of "geodesic" (an
    // inline table, so later solver settings can join it). Ranges and spelling are checked in
    // validate(), so a typo is reported together with every other problem.
    if (toml_input.contains("connected_max_rounds")) {
        tess.connected_max_rounds =
            get_int(toml_input, "connected_max_rounds", "Tessellation.connected_max_rounds");
    }
    if (toml_input.contains("geodesic")) {
        const auto& geodesic = toml::find(toml_input, "geodesic");
        warn_unknown_keys(geodesic, "Tessellation.geodesic", {"solver"});
        if (geodesic.contains("solver")) {
            tess.geodesic_solver = toml::find<std::string>(geodesic, "solver");
        }
    }

    // Periodic face matching tolerance (range checked in validate()).
    if (toml_input.contains("periodic_tolerance")) {
        tess.periodic_tolerance = get_number(toml::find(toml_input, "periodic_tolerance"));
    }

    // Quadrature-vote resolution (range checked in validate()).
    if (toml_input.contains("vote_levels")) {
        tess.vote_levels = get_int(toml_input, "vote_levels", "Tessellation.vote_levels");
    }

    // Volume fitting, as an inline table.
    if (toml_input.contains("fit_volumes")) {
        const auto& fit = toml::find(toml_input, "fit_volumes");
        warn_unknown_keys(fit,
                          "Tessellation.fit_volumes",
                          {"enabled", "tolerance", "max_iterations", "samples"});
        if (fit.contains("enabled")) {
            tess.fit_volumes.enabled = toml::find<bool>(fit, "enabled");
        }
        if (fit.contains("tolerance")) {
            tess.fit_volumes.tolerance = get_number(toml::find(fit, "tolerance"));
        }
        if (fit.contains("max_iterations")) {
            tess.fit_volumes.max_iterations =
                get_int(fit, "max_iterations", "Tessellation.fit_volumes.max_iterations");
        }
        if (fit.contains("samples")) {
            tess.fit_volumes.samples = toml::find<std::string>(fit, "samples");
        }
    }

    // Lloyd relaxation.
    if (toml_input.contains("lloyd_iterations")) {
        tess.lloyd_iterations =
            get_int(toml_input, "lloyd_iterations", "Tessellation.lloyd_iterations");
    }
    if (toml_input.contains("centroidal_power")) {
        tess.centroidal_power = toml::find<bool>(toml_input, "centroidal_power");
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
 *   - `grain_adjacency`: boolean; also write <basename>_adjacency.csv (pairs of grains sharing
 *     element faces, with the shared face area and face count). Default false.
 * The tool does not write ExaConstit input decks; users write their own options.toml.
 *
 * @throws std::invalid_argument for an unknown mesh_format; toml11's type error (a
 *         std::exception) when `paraview` or `grain_adjacency` is not a boolean.
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
                       "paraview",
                       "grain_adjacency"});

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

    // Optional grain adjacency table (pairs of face-sharing grains and their shared area).
    if (toml_input.contains("grain_adjacency")) {
        output.grain_adjacency = toml::find<bool>(toml_input, "grain_adjacency");
    }
    return output;
}

/**
 * @brief Reads the [VoxelMap] table: a voxelized microstructure used with [Source] type =
 * "voxel_map" (file formats in io/voxel_map_io.hpp).
 *
 * Keys (defaults in VoxelMapOptions, microstructure_options.hpp):
 *   - `format`: "exaconstit" or "dream3d" (spelling checked in validate()).
 *   - `file`: the HDF5 file.
 *   - `feature_ids_path`, `quats_path`, `phases_path`: explicit HDF5 dataset paths (empty: the
 *     format's default, or the DREAM.3D search by dataset name).
 *   - `quat_layout` ("wxyz" or "xyzw") and `conjugate` (boolean): how the file stores its
 *     quaternions. Both are required for DREAM.3D files when orientations are read, and have no
 *     default, because guessing would silently rotate every grain.
 *   - `origin`, `spacing` (3 values each, file units): replace the file's grid geometry.
 *   - `units_scale`, `translate`: x_mesh = units_scale * x_file + translate.
 *   - `sampling` ("centroid", "quadrature_vote"), `vote_levels` (0-3), `tile_size` (voxels).
 *
 * @throws std::invalid_argument for an unknown sampling name or arrays of the wrong length;
 *         toml11 errors for wrong TOML types (e.g. `conjugate = "yes"`).
 */
VoxelMapOptions voxel_map_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "VoxelMap",
                      {"format",
                       "file",
                       "feature_ids_path",
                       "quats_path",
                       "phases_path",
                       "quat_layout",
                       "conjugate",
                       "origin",
                       "spacing",
                       "units_scale",
                       "translate",
                       "sampling",
                       "vote_levels",
                       "tile_size"});
    static const std::map<std::string, SamplingMode> sampling_map = {
        {"centroid", SamplingMode::CENTROID}, {"quadrature_vote", SamplingMode::QUADRATURE_VOTE}};

    VoxelMapOptions map;

    // File, layout and explicit dataset paths
    if (toml_input.contains("format")) {
        map.format = toml::find<std::string>(toml_input, "format");
    }
    if (toml_input.contains("file")) {
        map.file = toml::find<std::string>(toml_input, "file");
    }
    if (toml_input.contains("feature_ids_path")) {
        map.feature_ids_path = toml::find<std::string>(toml_input, "feature_ids_path");
    }
    if (toml_input.contains("quats_path")) {
        map.quats_path = toml::find<std::string>(toml_input, "quats_path");
    }
    if (toml_input.contains("phases_path")) {
        map.phases_path = toml::find<std::string>(toml_input, "phases_path");
    }

    // Quaternion storage declarations: no defaults (an absent key stays "not given")
    if (toml_input.contains("quat_layout")) {
        map.quat_layout = toml::find<std::string>(toml_input, "quat_layout");
    }
    if (toml_input.contains("conjugate")) {
        map.conjugate = toml::find<bool>(toml_input, "conjugate") ? 1 : 0;
    }

    // Grid geometry overrides and the file-to-mesh transform (values checked in validate())
    if (toml_input.contains("origin")) {
        map.has_origin = true;
        map.origin = get_number_array<3>(toml_input, "origin");
    }
    if (toml_input.contains("spacing")) {
        map.has_spacing = true;
        map.spacing = get_number_array<3>(toml_input, "spacing");
    }
    if (toml_input.contains("units_scale")) {
        map.units_scale = get_number(toml::find(toml_input, "units_scale"));
    }
    if (toml_input.contains("translate")) {
        map.translate = get_number_array<3>(toml_input, "translate");
    }

    // Element sampling
    if (toml_input.contains("sampling")) {
        map.sampling = string_to_ms_enum(
            toml::find<std::string>(toml_input, "sampling"), sampling_map, "VoxelMap.sampling");
    }
    if (toml_input.contains("vote_levels")) {
        map.vote_levels = get_int(toml_input, "vote_levels", "VoxelMap.vote_levels");
    }
    if (toml_input.contains("tile_size")) {
        map.tile_size = get_int(toml_input, "tile_size", "VoxelMap.tile_size");
    }
    return map;
}

/**
 * @brief Reads the [Regions] table: how grains are grouped into ExaConstit material regions.
 *
 * Keys (defaults in RegionOptions, microstructure_options.hpp):
 *   - `mode`: "single", "from_phase" or "phase_fractions" (checked in validate());
 *   - `non_crystalline_phases = [..]`: phases whose grains get the identity orientation;
 *   - `phase_to_region = { "phase" = region, ... }`: explicit map (keys read as integers);
 *   - `phases = [{id = .., fraction = ..}, ...]`: target phase fractions (phase_fractions);
 *   - `filler_phase`: phase of the random ROI filler grains.
 *
 * @throws std::invalid_argument for a non-integer map key, a `phases` entry without `id` or
 *         `fraction`, or integers out of range; toml11 errors for wrong TOML types.
 */
RegionOptions regions_from_toml(const toml::value& toml_input) {
    warn_unknown_keys(toml_input,
                      "Regions",
                      {"mode", "non_crystalline_phases", "phase_to_region", "phases",
                       "filler_phase"});
    RegionOptions regions;
    if (toml_input.contains("mode")) {
        regions.mode = toml::find<std::string>(toml_input, "mode");
    }
    if (toml_input.contains("non_crystalline_phases")) {
        regions.non_crystalline_phases =
            get_int_vector(toml_input, "non_crystalline_phases", "Regions.non_crystalline_phases");
    }
    if (toml_input.contains("phase_to_region")) {
        regions.phase_to_region =
            get_int_map(toml_input, "phase_to_region", "Regions.phase_to_region");
    }
    // Target fractions: an array of inline tables, each with an id and a fraction
    if (toml_input.contains("phases")) {
        for (const auto& entry : toml::find(toml_input, "phases").as_array()) {
            warn_unknown_keys(entry, "Regions.phases[]", {"id", "fraction"});
            if (!entry.contains("id") || !entry.contains("fraction")) {
                throw std::invalid_argument("Microstructure options: every 'Regions.phases' entry "
                                            "needs 'id' and 'fraction'");
            }
            regions.phase_ids.push_back(get_int(entry, "id", "Regions.phases.id"));
            regions.phase_fractions.push_back(get_number(toml::find(entry, "fraction")));
        }
    }
    if (toml_input.contains("filler_phase")) {
        regions.has_filler_phase = true;
        regions.filler_phase = get_int(toml_input, "filler_phase", "Regions.filler_phase");
    }
    return regions;
}

/**
 * @brief Where validate() also collects its error messages when its caller asked for them
 * (validate(&problems)); null otherwise. Set and cleared by validate() itself, so it is only
 * non-null during a validate() call. Not thread-safe (options are validated from one thread).
 */
std::vector<std::string>* validation_problem_sink = nullptr;

/**
 * @brief Records one validation error: prints it on rank 0, appends it to the caller's list
 * (if validate() was given one) and marks the options invalid.
 *
 * validate() keeps going after an error so that the user sees every problem in one run instead
 * of fixing them one at a time.
 */
void report_error(bool& valid, const std::string& message) {
    WARNING_0_OPT("Error: " + message);
    if (validation_problem_sink != nullptr) {
        validation_problem_sink->push_back(message);
    }
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
 *   - [Source] `type`: what defines the grains: "tessellation" (seeds) or "voxel_map".
 *   - [VoxelMap]: the voxelized microstructure for type = "voxel_map" (see voxel_map_from_toml).
 *   - [Cleanup] `fragments` ("auto", "split", "merge", "none"), `keep_component` ("largest",
 *     "seed"), `min_fragment_elements` and `min_fragment_fraction` (what "auto" calls a tiny
 *     piece) and `min_grain_elements`: what to do with grains that end up split into
 *     disconnected pieces, which piece keeps the grain, and the smallest allowed grain.
 *   - [Orientations] `source`: "random", "from_source" or "list".
 *   - [Regions]: `mode` ("single", "from_phase" or "phase_fractions": how grains are grouped into
 *     ExaConstit material regions) and its phase keys (see regions_from_toml).
 * The string values of [Cleanup], [Orientations] and [Regions] are checked in validate().
 *
 * Not collective; every rank may call it on its own copy of the document.
 *
 * @throws std::invalid_argument for values that cannot be stored (see the table readers) and
 *         for an unknown [Source] type; toml11 exceptions for wrong TOML types.
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
                       "VoxelMap",
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

    // [Source]: "tessellation" (seeds) or "voxel_map" ([VoxelMap]). Any other type is rejected
    // outright: if it were merely warned about, the tool would tessellate random seeds instead
    // and produce a completely different microstructure without the user noticing.
    if (toml_input.contains("Source")) {
        const auto& source = toml::find(toml_input, "Source");
        warn_unknown_keys(source, "Source", {"type"});
        if (source.contains("type")) {
            const auto type = toml::find<std::string>(source, "type");
            if (type != "tessellation" && type != "voxel_map") {
                throw std::invalid_argument("Microstructure options: 'Source.type' = \"" + type +
                                            "\" is not supported (expected \"tessellation\" or "
                                            "\"voxel_map\")");
            }
            options.source.type = type;
        }
    }
    // [VoxelMap]: read whenever written; validate() warns if it is unused
    if (toml_input.contains("VoxelMap")) {
        options.voxel_map = voxel_map_from_toml(toml::find(toml_input, "VoxelMap"));
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
        warn_unknown_keys(cleanup,
                          "Cleanup",
                          {"fragments",
                           "keep_component",
                           "min_grain_elements",
                           "min_fragment_elements",
                           "min_fragment_fraction"});
        if (cleanup.contains("min_fragment_elements")) {
            options.cleanup.min_fragment_elements =
                get_int(cleanup, "min_fragment_elements", "Cleanup.min_fragment_elements");
        }
        if (cleanup.contains("min_fragment_fraction")) {
            options.cleanup.min_fragment_fraction =
                get_number(toml::find(cleanup, "min_fragment_fraction"));
        }
        if (cleanup.contains("fragments")) {
            options.cleanup.fragments = toml::find<std::string>(cleanup, "fragments");
        }
        if (cleanup.contains("keep_component")) {
            options.cleanup.keep_component = toml::find<std::string>(cleanup, "keep_component");
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
        options.regions = regions_from_toml(toml::find(toml_input, "Regions"));
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
 * @param problems optional; every error message is also appended here, so a rank other than 0
 *                 (which prints nothing) can report its own problems, e.g. when only it cannot
 *                 see an input file.
 * @return true if every check passed.
 */
bool MicrostructureOptions::validate(std::vector<std::string>* problems) const {
    bool valid = true;
    // Collect messages for the caller during this call only; the guard clears the sink again on
    // every return path.
    struct SinkGuard {
        explicit SinkGuard(std::vector<std::string>* sink) {
            validation_problem_sink = sink;
        }
        ~SinkGuard() {
            validation_problem_sink = nullptr;
        }
    } sink_guard(problems);

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
    // Active / passthrough attributes: MFEM element attributes are positive, region ids start at
    // 1, and an attribute cannot be both tessellated and kept as one grain. Whether the
    // attributes exist in the mesh is checked when the mesh is read (AttributeDomain).
    for (const int attribute : mesh.active_attributes) {
        if (attribute <= 0) {
            report_error(valid, "Mesh.active_attributes must be > 0 (got " +
                                    std::to_string(attribute) + ")");
        }
    }
    for (const auto& entry : mesh.passthrough_attributes) {
        if (entry.first <= 0) {
            report_error(valid, "Mesh.passthrough_attributes: attribute " +
                                    std::to_string(entry.first) + " must be > 0");
        }
        if (entry.second < 1) {
            report_error(valid, "Mesh.passthrough_attributes: region of attribute " +
                                    std::to_string(entry.first) + " must be >= 1");
        }
        if (std::find(mesh.active_attributes.begin(), mesh.active_attributes.end(), entry.first) !=
            mesh.active_attributes.end()) {
            report_error(valid, "Mesh attribute " + std::to_string(entry.first) +
                                    " is listed in both active_attributes and "
                                    "passthrough_attributes");
        }
    }
    const bool passthrough = !mesh.passthrough_attributes.empty();
    bool any_periodic_axis = false;
    for (const bool p : tessellation.periodic) {
        any_periodic_axis = any_periodic_axis || p;
    }
    // The active part is a sub-mesh whose interface with the passthrough part is a free surface
    // of that sub-mesh (mesh/attribute_domain.hpp): a boundary clearance would push seeds away
    // from a bonded interface, and periodic-plane detection would take the interface faces for
    // periodic faces. Both are refused until they are handled explicitly.
    // (With a voxel map there are no seeds: the clearance is only warned about as ignored.)
    const bool voxel_source = (source.type == "voxel_map");
    if (passthrough && seeds.boundary_clearance > 0.0 && !voxel_source) {
        report_error(valid,
                     "Seeds.boundary_clearance > 0 cannot be combined with "
                     "Mesh.passthrough_attributes (the interface with the passthrough part would "
                     "count as an outer boundary)");
    }
    if (passthrough && any_periodic_axis) {
        report_error(valid,
                     "Tessellation.periodic cannot be combined with Mesh.passthrough_attributes");
    }

    // --- [Source] and [VoxelMap] ------------------------------------------------------------
    check_choice(valid, "Source.type", source.type, {"tessellation", "voxel_map"});
    if (voxel_source) {
        const VoxelMapOptions& map = voxel_map;
        check_choice(valid, "VoxelMap.format", map.format, {"exaconstit", "dream3d"});
        if (map.file.empty()) {
            report_error(valid, "Source.type = \"voxel_map\" needs 'VoxelMap.file'");
        } else if (!fs::exists(map.file)) {
            report_error(valid, "voxel map file '" + map.file.string() + "' does not exist");
        }
        check_choice(valid, "VoxelMap.quat_layout", map.quat_layout, {"", "wxyz", "xyzw"});
        // The canonical layout fixes its storage (w x y z, crystal-to-sample): declaring another
        // order or a conjugation contradicts the format (the reader would refuse it late)
        if (map.format == "exaconstit" && (map.quat_layout == "xyzw" || map.conjugate == 1)) {
            report_error(valid,
                         "VoxelMap.format = \"exaconstit\" stores w x y z quaternions: "
                         "quat_layout = \"xyzw\" and conjugate = true are not allowed");
        }
        // DREAM.3D files do not say how they store quaternions; guessing would rotate every
        // grain, so both declarations are required whenever orientations are read
        if (map.format == "dream3d" && orientations.source == "from_source" &&
            (map.quat_layout.empty() || map.conjugate < 0)) {
            report_error(valid,
                         "VoxelMap.format = \"dream3d\" with Orientations.source = "
                         "\"from_source\" needs both 'VoxelMap.quat_layout' (\"wxyz\" or "
                         "\"xyzw\") and 'VoxelMap.conjugate' (true or false); there are no "
                         "defaults");
        }
        if (!is_positive_finite(map.units_scale)) {
            report_error(valid, "VoxelMap.units_scale must be finite and > 0");
        }
        for (std::size_t d = 0; d < 3; ++d) {
            const bool bad_origin = map.has_origin && !std::isfinite(map.origin[d]);
            if (!std::isfinite(map.translate[d]) || bad_origin) {
                report_error(valid, "VoxelMap.translate and origin must be finite");
                break;
            }
        }
        if (map.has_spacing) {
            for (const double h : map.spacing) {
                if (!is_positive_finite(h)) {
                    report_error(valid, "VoxelMap.spacing must be finite and > 0 on every axis");
                    break;
                }
            }
        }
        if (map.vote_levels < 0 || map.vote_levels > 3) {
            report_error(valid, "VoxelMap.vote_levels must be in 0-3");
        }
        if (map.tile_size < 1) {
            report_error(valid, "VoxelMap.tile_size must be >= 1");
        }

        // Options that would change a tessellated microstructure have no meaning for a voxel map;
        // refusing them is safer than ignoring them (the user expects them to act).
        const auto refuse = [&](bool set, const std::string& what) {
            if (set) {
                report_error(valid, what + " cannot be used with Source.type = \"voxel_map\" "
                                           "(no seeds and no tessellation)");
            }
        };
        refuse(seeds.source == "file", "Seeds.source = \"file\"");
        refuse(seeds.placement == SeedPlacement::JOHNSON_MEHL,
               "Seeds.placement = \"johnson_mehl\"");
        refuse(seeds.size.distribution != "none", "Seeds.Size.distribution");
        refuse(seeds.size.polydisperse_placement, "Seeds.Size.polydisperse_placement");
        refuse(seeds.use_roi || seeds.fill_outside_roi != "none", "Seeds.roi / fill_outside_roi");
        refuse(tessellation.metric != MetricType::VORONOI, "Tessellation.metric");
        refuse(tessellation.anisotropy_kind != "none", "Tessellation.anisotropy");
        refuse(any_periodic_axis, "Tessellation.periodic");
        refuse(tessellation.fit_volumes.enabled, "Tessellation.fit_volumes");
        refuse(tessellation.lloyd_iterations > 0 || tessellation.centroidal_power,
               "Tessellation.lloyd_iterations / centroidal_power");
        refuse(cleanup.keep_component == "seed", "Cleanup.keep_component = \"seed\"");
        refuse(tessellation.distance != DistanceMode::EUCLIDEAN,
               "Tessellation.distance = \"connected\" / \"geodesic\"");
        // Harmless leftovers (e.g. from a template file, or -n on the command line) only warn
        if (seeds.num_seeds > 0 || seeds.placement != SeedPlacement::REJECTION ||
            seeds.min_distance > 0.0 || seeds.min_distance_factor > 0.0 ||
            seeds.boundary_clearance > 0.0 || seeds.use_sampling_box ||
            seeds.sampling_domain != "material" || seeds.lattice_type != "cubic" ||
            seeds.lattice_jitter > 0.0) {
            WARNING_0_OPT("Warning: the seed placement keys of [Seeds] (num_seeds, placement, "
                          "min_distance, boundary_clearance, sampling_domain/box, lattice) are "
                          "ignored with Source.type = \"voxel_map\"");
        }
        if (seeds.size.mean_diameter > 0.0 || seeds.size.mean_volume > 0.0 ||
            seeds.size.sigma > 0.0) {
            WARNING_0_OPT("Warning: Seeds.Size.mean_diameter, mean_volume and sigma are ignored "
                          "with Source.type = \"voxel_map\"");
        }
        if (tessellation.sampling != SamplingMode::CENTROID || tessellation.vote_levels != 2 ||
            tessellation.num_bins != 8) {
            WARNING_0_OPT("Warning: Tessellation.sampling, vote_levels and num_bins are ignored "
                          "with Source.type = \"voxel_map\"; use VoxelMap.sampling and "
                          "VoxelMap.vote_levels");
        }
    } else if (!voxel_map.file.empty()) {
        WARNING_0_OPT("Warning: [VoxelMap] is ignored unless Source.type = \"voxel_map\"");
    }

    // --- [Seeds] ----------------------------------------------------------------------------
    check_choice(valid, "Seeds.source", seeds.source, {"random", "file"});
    // A mean grain size (with num_seeds = 0) or Johnson-Mehl nucleation sets the seed count.
    const bool count_from_size =
        seeds.size.mean_diameter > 0.0 || seeds.size.mean_volume > 0.0;
    const bool johnson_mehl = (seeds.placement == SeedPlacement::JOHNSON_MEHL);
    if (voxel_source) {
        // No seeds: the voxel-map checks above cover [Seeds]
    } else if (seeds.source == "random") {
        // Zero seeds would leave every element without a grain.
        if (seeds.num_seeds < 0 || (seeds.num_seeds == 0 && !count_from_size && !johnson_mehl)) {
            report_error(valid,
                         "random seeds need 'Seeds.num_seeds' > 0 (or a mean grain size in "
                         "[Seeds.Size], or placement = \"johnson_mehl\")");
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
        // Moving seeds would move the measured grain positions
        if (tessellation.lloyd_iterations > 0 || tessellation.centroidal_power) {
            report_error(valid,
                         "Tessellation.lloyd_iterations / centroidal_power cannot be used with "
                         "Seeds.source = \"file\" (they would move the measured seed positions)");
        }
        if (johnson_mehl) {
            report_error(valid, "Seeds.placement = \"johnson_mehl\" cannot be used with "
                                "Seeds.source = \"file\"");
        }
    }

    // Region of interest of a seed table, and the random filler seeds outside it
    check_choice(valid, "Seeds.fill_outside_roi", seeds.fill_outside_roi, {"none", "random"});
    if (seeds.use_roi) {
        if (seeds.source != "file") {
            report_error(valid, "Seeds.roi needs Seeds.source = \"file\"");
        }
        for (std::size_t d = 0; d < 3; ++d) {
            const double lo = seeds.roi_lo[d];
            const double hi = seeds.roi_hi[d];
            if (!std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo)) {
                report_error(valid,
                             "Seeds.roi must be finite with hi > lo along every axis");
                break;
            }
        }
    }
    if (seeds.fill_outside_roi == "random" && !seeds.use_roi) {
        report_error(valid, "Seeds.fill_outside_roi = \"random\" needs Seeds.roi");
    }
    if (seeds.filler_num_seeds < 0) {
        report_error(valid, "Seeds.filler_num_seeds must be >= 0 (0 = automatic)");
    }
    if (!is_nonnegative_finite(seeds.filler_min_distance_factor)) {
        report_error(valid, "Seeds.filler_min_distance_factor must be finite and >= 0");
    }
    if (seeds.fill_outside_roi != "random" &&
        (seeds.filler_num_seeds > 0 || seeds.filler_min_distance_factor > 0.0)) {
        WARNING_0_OPT("Warning: Seeds.filler_num_seeds and filler_min_distance_factor are ignored "
                      "unless fill_outside_roi = \"random\"");
    }
    check_choice(valid, "Seeds.sampling_domain", seeds.sampling_domain, {"material", "bbox"});
    check_choice(valid, "Seeds.lattice_type", seeds.lattice_type, {"cubic", "bcc", "fcc", "hcp"});
    // Distances and jitter are magnitudes; a negative value has no physical meaning and would
    // turn the spacing tests upside down. NaN and inf (accepted by TOML) are rejected too: NaN
    // makes every comparison false and inf would reject every candidate seed.
    if (!is_nonnegative_finite(seeds.min_distance_factor)) {
        report_error(valid, "Seeds.min_distance_factor must be finite and >= 0");
    }
    if (!is_nonnegative_finite(seeds.min_distance)) {
        report_error(valid, "Seeds.min_distance must be finite and >= 0");
    }
    // Both spacing forms given: the absolute distance wins (SeedingParameters), so the factor is
    // ignored. Not an error, but worth a warning in case the user expected the factor.
    if (seeds.min_distance > 0.0 && seeds.min_distance_factor > 0.0) {
        WARNING_0_OPT("Warning: Seeds.min_distance and Seeds.min_distance_factor are both set; "
                      "min_distance is used and min_distance_factor is ignored");
    }
    if (!is_nonnegative_finite(seeds.boundary_clearance)) {
        report_error(valid, "Seeds.boundary_clearance must be finite and >= 0");
    }
    // A sampling box needs real corners and a positive extent along every axis; an empty or
    // inverted box could never receive a seed. (Whether it overlaps the material is only known
    // once the mesh is read, so the seed generator checks that.)
    if (seeds.use_sampling_box) {
        for (std::size_t d = 0; d < 3; ++d) {
            const double lo = seeds.sampling_box_lo[d];
            const double hi = seeds.sampling_box_hi[d];
            if (!std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo)) {
                report_error(valid,
                             "Seeds.sampling_box must be finite with upper corner > lower corner "
                             "along every axis ([xlo, ylo, zlo, xhi, yhi, zhi])");
                break;
            }
        }
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

    // [Seeds.Size]: a size distribution needs sensible parameters; the mean size is a length or
    // a volume, never both (they would contradict each other).
    const SeedSizeOptions& size = seeds.size;
    check_choice(valid,
                 "Seeds.Size.distribution",
                 size.distribution,
                 {"none", "lognormal", "from_source"});
    if (size.distribution == "from_source" && seeds.source != "file") {
        report_error(valid,
                     "Seeds.Size.distribution = \"from_source\" needs Seeds.source = \"file\"");
    }
    if (!is_nonnegative_finite(size.mean_diameter) || !is_nonnegative_finite(size.mean_volume)) {
        report_error(valid, "Seeds.Size.mean_diameter and mean_volume must be finite and >= 0");
    }
    if (size.mean_diameter > 0.0 && size.mean_volume > 0.0) {
        report_error(valid, "give only one of Seeds.Size.mean_diameter and mean_volume");
    }
    if (!is_nonnegative_finite(size.sigma)) {
        report_error(valid, "Seeds.Size.sigma must be finite and >= 0");
    }
    if (count_from_size && seeds.num_seeds > 0) {
        WARNING_0_OPT("Warning: Seeds.num_seeds and a mean grain size in [Seeds.Size] are both "
                      "set; num_seeds sets the grain count and the mean size is not used");
    }
    if (size.polydisperse_placement) {
        // Exclusion spheres of radius kappa r_i fill kappa^3 of the material (the target volumes
        // fill it exactly); the seed generator refuses packings above 30%.
        if (!is_positive_finite(size.exclusion_factor) ||
            size.exclusion_factor * size.exclusion_factor * size.exclusion_factor > 0.3) {
            report_error(valid,
                         "Seeds.Size.exclusion_factor must be in (0, 0.669] (kappa^3 <= 0.3)");
        }
        if (seeds.source == "file") {
            // Measured positions are given; there is nothing to place
            report_error(valid,
                         "Seeds.Size.polydisperse_placement cannot be used with Seeds.source = "
                         "\"file\" (the seed table fixes the positions)");
        } else if (seeds.placement != SeedPlacement::REJECTION &&
                   seeds.placement != SeedPlacement::ELEMENT_WEIGHTED) {
            report_error(valid,
                         "Seeds.Size.polydisperse_placement needs placement \"rejection\" or "
                         "\"element_weighted\"");
        }
        if (seeds.min_distance > 0.0 || seeds.min_distance_factor > 0.0) {
            report_error(valid,
                         "Seeds.Size.polydisperse_placement replaces Seeds.min_distance and "
                         "min_distance_factor; set those to 0");
        }
        // The target volumes (and so the exclusion radii) fill the whole material, while the
        // seeds would only be placed inside the box: the exclusion spheres could not fit (the
        // packing guard would refuse them) and the grains would be sized for the whole part.
        if (seeds.use_sampling_box) {
            report_error(valid,
                         "Seeds.Size.polydisperse_placement cannot be combined with "
                         "Seeds.sampling_box (the targets fill the whole material)");
        }
    }
    // A spread without a distribution has no effect.
    if (size.distribution == "none" && size.sigma > 0.0) {
        WARNING_0_OPT("Warning: Seeds.Size.sigma is ignored with distribution = \"none\"");
    }

    // [Seeds.JohnsonMehl]: nucleation needs a positive rate and speed, and its grains are the
    // cells of the Additive metric (weights a_i = -v t_i), so other metrics would ignore the
    // nucleation times. Size targets, fitting and relaxation would overwrite the physical
    // weights or positions, so they are refused too.
    if (johnson_mehl) {
        const JohnsonMehlOptions& jm = seeds.johnson_mehl;
        if (!is_positive_finite(jm.rate)) {
            report_error(valid, "Seeds.JohnsonMehl.rate must be finite and > 0");
        }
        if (!is_positive_finite(jm.growth_speed)) {
            report_error(valid, "Seeds.JohnsonMehl.growth_speed must be finite and > 0");
        }
        if (!is_nonnegative_finite(jm.end_time)) {
            report_error(valid, "Seeds.JohnsonMehl.end_time must be finite and >= 0");
        }
        if (jm.num_nuclei < 0) {
            report_error(valid, "Seeds.JohnsonMehl.num_nuclei must be >= 0");
        }
        if (tessellation.metric != MetricType::ADDITIVE) {
            report_error(valid,
                         "placement = \"johnson_mehl\" needs Tessellation.metric = \"additive\"");
        }
        if (size.distribution != "none" || size.polydisperse_placement ||
            tessellation.fit_volumes.enabled || tessellation.lloyd_iterations > 0) {
            report_error(valid,
                         "placement = \"johnson_mehl\" cannot be combined with [Seeds.Size], "
                         "fit_volumes or lloyd_iterations (they would replace the nucleation "
                         "weights or positions)");
        }
        // Nucleation coverage and the candidate domain are not periodic: nuclei near one face
        // would not block or be blocked by nuclei near the opposite face.
        if (tessellation.periodic[0] || tessellation.periodic[1] || tessellation.periodic[2]) {
            report_error(valid,
                         "placement = \"johnson_mehl\" is not supported with periodic axes");
        }
        if (seeds.min_distance > 0.0 || seeds.min_distance_factor > 0.0 ||
            seeds.boundary_clearance > 0.0 || seeds.sampling_domain != "material" ||
            seeds.lattice_type != "cubic" || seeds.lattice_jitter > 0.0) {
            WARNING_0_OPT("Warning: Seeds.min_distance, min_distance_factor, boundary_clearance, "
                          "sampling_domain = \"bbox\" and the lattice keys are ignored for "
                          "placement = \"johnson_mehl\" (nuclei always lie in the material)");
        }
        if (seeds.num_seeds > 0) {
            WARNING_0_OPT("Warning: Seeds.num_seeds is ignored for placement = \"johnson_mehl\" "
                          "(use Seeds.JohnsonMehl.num_nuclei to limit the count)");
        }
    }

    // --- [Tessellation] ---------------------------------------------------------------------
    // At least one group of seeds is needed to build the seed search structure.
    if (tessellation.num_bins < 1) {
        report_error(valid, "Tessellation.num_bins must be >= 1");
    }
    // Periodic face pairing: a relative tolerance well below half an edge.
    if (!(std::isfinite(tessellation.periodic_tolerance) && tessellation.periodic_tolerance > 0.0 &&
          tessellation.periodic_tolerance < 0.5)) {
        report_error(valid, "Tessellation.periodic_tolerance must be in (0, 0.5)");
    }
    // MFEM's point interpolation limits the vote to 8^3 points per element.
    if (tessellation.vote_levels < 0 || tessellation.vote_levels > 3) {
        report_error(valid, "Tessellation.vote_levels must be in 0-3");
    }
    check_choice(valid,
                 "Tessellation.anisotropy.kind",
                 tessellation.anisotropy_kind,
                 {"none", "tensor", "columnar", "lamellar", "per_seed"});
    // Per-seed tensors come from the shape columns of a seed table and only the Quadric metric
    // reads tensors (whether the table really has shape columns is checked when it is read)
    if (tessellation.anisotropy_kind == "per_seed") {
        if (tessellation.metric != MetricType::QUADRIC) {
            report_error(valid,
                         "Tessellation.anisotropy.kind = \"per_seed\" needs metric = \"quadric\"");
        }
        if (seeds.source != "file") {
            report_error(valid, "Tessellation.anisotropy.kind = \"per_seed\" needs "
                                "Seeds.source = \"file\" (a table with shape columns)");
        }
    }
    if (tessellation.anisotropy_kind == "tensor" && tessellation.metric != MetricType::QUADRIC) {
        WARNING_0_OPT("Warning: Tessellation global_tensor is ignored: only metric = "
                      "\"quadric\" uses it");
    }
    const bool semidefinite_kind = tessellation.anisotropy_kind == "columnar" ||
                                   tessellation.anisotropy_kind == "lamellar";
    if (semidefinite_kind) {
        // Columnar / lamellar tensors only mean something for the Quadric metric, and need a
        // real direction (it is normalized later, so any nonzero length is fine).
        if (tessellation.metric != MetricType::QUADRIC) {
            report_error(valid,
                         "Tessellation.anisotropy.kind = \"" + tessellation.anisotropy_kind +
                             "\" needs metric = \"quadric\"");
        }
        const std::array<double, 3>& direction = (tessellation.anisotropy_kind == "columnar")
                                                     ? tessellation.anisotropy_axis
                                                     : tessellation.anisotropy_normal;
        bool finite = true;
        double length_sq = 0.0;
        for (const double c : direction) {
            finite = finite && std::isfinite(c);
            length_sq += c * c;
        }
        if (!finite || !(length_sq > 0.0)) {
            report_error(valid,
                         "Tessellation.anisotropy axis/normal must be finite and nonzero");
        }
    }
    if (!semidefinite_kind) {
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

    // Volume fitting and Lloyd relaxation.
    const FitVolumesOptions& fit = tessellation.fit_volumes;
    check_choice(valid, "Tessellation.fit_volumes.samples", fit.samples, {"quadrature", "element"});
    if (!is_positive_finite(fit.tolerance)) {
        report_error(valid, "Tessellation.fit_volumes.tolerance must be finite and > 0");
    }
    if (fit.max_iterations < 0) {
        report_error(valid, "Tessellation.fit_volumes.max_iterations must be >= 0");
    }
    // Weights are what the fit adjusts; Voronoi has none.
    if (fit.enabled && tessellation.metric == MetricType::VORONOI) {
        report_error(valid,
                     "Tessellation.fit_volumes needs a weighted metric (laguerre, quadric or "
                     "additive), not voronoi");
    }
    if (tessellation.lloyd_iterations < 0) {
        report_error(valid, "Tessellation.lloyd_iterations must be >= 0");
    }
    if (tessellation.lloyd_iterations > 0 && tessellation.metric != MetricType::VORONOI &&
        tessellation.metric != MetricType::LAGUERRE) {
        report_error(valid, "Tessellation.lloyd_iterations needs metric voronoi or laguerre");
    }
    if (tessellation.centroidal_power &&
        (tessellation.lloyd_iterations <= 0 || tessellation.metric != MetricType::LAGUERRE)) {
        report_error(valid,
                     "Tessellation.centroidal_power needs lloyd_iterations > 0 and metric = "
                     "\"laguerre\"");
    }
    // Periodic images are copies of the seeds; moving seeds (Lloyd) or fitting weights would
    // have to move/fit every copy consistently, which these solvers do not do.
    bool any_periodic = false;
    for (const bool p : tessellation.periodic) {
        any_periodic = any_periodic || p;
    }
    if (any_periodic && (fit.enabled || tessellation.lloyd_iterations > 0)) {
        report_error(valid,
                     "fit_volumes and lloyd_iterations are not supported with periodic axes");
    }
    // Stage 1e distance modes. Both run on top of a plain (non-periodic) labeling of the active
    // material: periodic images would need paths and components that wrap around the box, and
    // Lloyd relaxation and the volume fit move seeds / weights using straight-line cells, which
    // the constrained labels would no longer match. The geodesic distance is a path length, so
    // only metrics whose value is a length qualify: Voronoi (d) and Additive (d - a_i, e.g.
    // Johnson-Mehl); Laguerre (d^2 - w) and Quadric ((x - s)^T A (x - s)) values are not lengths.
    const bool connected = (tessellation.distance == DistanceMode::CONNECTED);
    const bool geodesic = (tessellation.distance == DistanceMode::GEODESIC);
    if (connected || geodesic) {
        const std::string mode = connected ? "\"connected\"" : "\"geodesic\"";
        if (any_periodic) {
            report_error(valid, "Tessellation.distance = " + mode +
                                    " cannot be combined with periodic axes");
        }
        if (fit.enabled) {
            report_error(valid, "Tessellation.distance = " + mode +
                                    " cannot be combined with Tessellation.fit_volumes");
        }
        if (tessellation.lloyd_iterations > 0 || tessellation.centroidal_power) {
            report_error(valid, "Tessellation.distance = " + mode +
                                    " cannot be combined with Tessellation.lloyd_iterations / "
                                    "centroidal_power");
        }
    }
    if (geodesic) {
        if (tessellation.metric != MetricType::VORONOI &&
            tessellation.metric != MetricType::ADDITIVE) {
            report_error(valid, "Tessellation.distance = \"geodesic\" needs metric = \"voronoi\" "
                                "or \"additive\" (Laguerre and Quadric values are not path "
                                "lengths)");
        }
        if (tessellation.anisotropy_kind != "none") {
            report_error(valid, "Tessellation.distance = \"geodesic\" cannot be combined with "
                                "Tessellation.anisotropy (kind must be \"none\")");
        }
    }
    // Distance-mode settings, one policy for both keys: a value that is invalid in itself (an
    // unknown solver name, a rounds cap outside the labeler's range) is an error whatever the
    // distance mode, since it is a typo or a mistake either way; a valid setting of a mode that is
    // not selected only gets an "ignored" warning (compared with the default, since the options
    // do not record which keys were written). A key never gets both.
    const TessellationOptions defaults;
    bool solver_ok = true;
    check_choice(solver_ok, "Tessellation.geodesic.solver", tessellation.geodesic_solver,
                 {"fim", "graph"});
    if (!solver_ok) {
        valid = false;
    } else if (!geodesic && tessellation.geodesic_solver != defaults.geodesic_solver) {
        WARNING_0_OPT("Warning: Tessellation.geodesic is ignored unless distance = \"geodesic\"");
    }
    // The connected labeler's own bound (labeling/connected_limits.hpp; 16-64 is plenty)
    if (tessellation.connected_max_rounds < 0 ||
        tessellation.connected_max_rounds > CONNECTED_MAX_ROUNDS_LIMIT) {
        report_error(valid, "Tessellation.connected_max_rounds must be in 0.." +
                                std::to_string(CONNECTED_MAX_ROUNDS_LIMIT));
    } else if (!connected && tessellation.connected_max_rounds != defaults.connected_max_rounds) {
        WARNING_0_OPT("Warning: Tessellation.connected_max_rounds is ignored unless distance = "
                      "\"connected\"");
    }
    if (size.distribution != "none" && !size.polydisperse_placement && !fit.enabled &&
        tessellation.metric != MetricType::LAGUERRE) {
        WARNING_0_OPT("Warning: [Seeds.Size] has no effect here: use metric = \"laguerre\" "
                      "(initial weights), fit_volumes or polydisperse_placement");
    }

    // --- [Cleanup], [Orientations], [Regions] -----------------------------------------------
    check_choice(valid, "Cleanup.fragments", cleanup.fragments, {"auto", "split", "merge", "none"});
    check_choice(valid, "Cleanup.keep_component", cleanup.keep_component, {"largest", "seed"});
    // Thresholds of "auto" that separate tiny pieces (merged) from large ones (split).
    if (cleanup.min_fragment_elements < 0) {
        report_error(valid, "Cleanup.min_fragment_elements must be >= 0");
    }
    if (!(std::isfinite(cleanup.min_fragment_fraction) && cleanup.min_fragment_fraction >= 0.0 &&
          cleanup.min_fragment_fraction <= 1.0)) {
        report_error(valid, "Cleanup.min_fragment_fraction must be in [0, 1]");
    }
    // An element count below zero is meaningless (0 or 1 means no grain is too small).
    if (cleanup.min_grain_elements < 0) {
        report_error(valid, "Cleanup.min_grain_elements must be >= 0");
    }
    check_choice(
        valid, "Orientations.source", orientations.source, {"random", "from_source", "list"});
    // Orientations from the source exist only in a seed table or a voxel map; generated seeds
    // have none, and the run would otherwise fail only at the very end (after seeding, labeling
    // and cleanup). Whether the table / map really holds quaternions is checked when it is read.
    if (orientations.source == "from_source" && seeds.source != "file" && !voxel_source) {
        report_error(valid,
                     "Orientations.source = \"from_source\" needs Seeds.source = \"file\" or "
                     "Source.type = \"voxel_map\"");
    }

    // [Regions]: phases -> material regions
    check_choice(
        valid, "Regions.mode", regions.mode, {"single", "from_phase", "phase_fractions"});
    for (const int phase : regions.non_crystalline_phases) {
        if (phase < 0) {
            report_error(valid, "Regions.non_crystalline_phases must be >= 0");
            break;
        }
    }
    for (const auto& entry : regions.phase_to_region) {
        if (entry.first < 0 || entry.second < 1) {
            report_error(valid, "Regions.phase_to_region needs phase ids >= 0 and region ids >= 1 "
                                "(entry " + std::to_string(entry.first) + " = " +
                                    std::to_string(entry.second) + ")");
        }
    }
    // Phases from the data: only seed tables and voxel maps carry them
    if (regions.mode == "from_phase" && seeds.source != "file" && !voxel_source) {
        report_error(valid, "Regions.mode = \"from_phase\" needs Seeds.source = \"file\" or "
                            "Source.type = \"voxel_map\" (generated seeds have no phase)");
    }
    // Synthetic phases: one positive fraction per distinct phase id, summing to 1
    if (regions.mode == "phase_fractions") {
        if (regions.phase_ids.empty() ||
            regions.phase_ids.size() != regions.phase_fractions.size()) {
            report_error(valid, "Regions.mode = \"phase_fractions\" needs Regions.phases = "
                                "[{id = .., fraction = ..}, ...]");
        } else {
            double sum = 0.0;
            bool fractions_ok = true;
            for (const double f : regions.phase_fractions) {
                fractions_ok = fractions_ok && is_positive_finite(f);
                sum += f;
            }
            if (!fractions_ok || !(std::fabs(sum - 1.0) <= 1.0e-6)) {
                report_error(valid, "Regions.phases fractions must be > 0 and sum to 1 (within "
                                    "1e-6)");
            }
            std::vector<int> ids = regions.phase_ids;
            std::sort(ids.begin(), ids.end());
            if (ids.front() < 0 || std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
                report_error(valid, "Regions.phases ids must be >= 0 and distinct");
            }
        }
    } else if (!regions.phase_ids.empty()) {
        WARNING_0_OPT("Warning: Regions.phases is ignored unless mode = \"phase_fractions\"");
    }
    if (regions.mode == "single" &&
        (!regions.phase_to_region.empty() || !regions.non_crystalline_phases.empty())) {
        WARNING_0_OPT("Warning: Regions.phase_to_region and non_crystalline_phases are ignored "
                      "with mode = \"single\"");
    }
    // Phase of the random filler grains
    if (regions.has_filler_phase && regions.filler_phase < 0) {
        report_error(valid, "Regions.filler_phase must be >= 0");
    }
    if (regions.has_filler_phase && seeds.fill_outside_roi != "random") {
        WARNING_0_OPT("Warning: Regions.filler_phase is ignored without filler seeds "
                      "(Seeds.fill_outside_roi = \"random\")");
    }
    // Synthetic phases are known now, so an explicit phase -> region map can be checked here with
    // the rules of AssignRegionsFromPhases (regions/region_assigner.hpp): every listed phase
    // mapped, region ids exactly 1..M, no region mixing crystalline and non-crystalline phases
    if (regions.mode == "phase_fractions" && !regions.phase_to_region.empty()) {
        std::map<int, std::vector<int>> region_members; // region -> listed phases
        for (const int phase : regions.phase_ids) {
            const auto it = regions.phase_to_region.find(phase);
            if (it == regions.phase_to_region.end()) {
                report_error(valid, "Regions.phase_to_region has no entry for phase " +
                                        std::to_string(phase) + " of Regions.phases");
            } else {
                region_members[it->second].push_back(phase);
            }
        }
        if (!region_members.empty() &&
            region_members.rbegin()->first != static_cast<int>(region_members.size())) {
            report_error(valid, "Regions.phase_to_region: the regions of the listed phases must "
                                "be exactly 1..M (no unused region ids)");
        }
        for (const auto& entry : region_members) {
            int amorphous = 0;
            for (const int phase : entry.second) {
                amorphous += std::count(regions.non_crystalline_phases.begin(),
                                        regions.non_crystalline_phases.end(), phase) > 0 ? 1 : 0;
            }
            if (amorphous > 0 && amorphous < static_cast<int>(entry.second.size())) {
                report_error(valid, "Regions.phase_to_region: region " +
                                        std::to_string(entry.first) +
                                        " mixes crystalline and non-crystalline phases");
            }
        }
    }
    if (regions.mode == "from_phase" && seeds.fill_outside_roi == "random" &&
        !regions.has_filler_phase) {
        report_error(valid, "Regions.mode = \"from_phase\" with Seeds.fill_outside_roi = "
                            "\"random\" needs 'Regions.filler_phase' (filler grains have no "
                            "measured phase)");
    }
    // With a single phase region the region ids are known now: {1} plus the passthrough regions
    // must be exactly 1..R, so every [[Materials]] block is used
    if (regions.mode == "single" && passthrough) {
        std::vector<int> ids = {1};
        for (const auto& entry : mesh.passthrough_attributes) {
            ids.push_back(entry.second);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (ids.back() != static_cast<int>(ids.size())) {
            report_error(valid, "Mesh.passthrough_attributes: with Regions.mode = \"single\" the "
                                "grains are region 1, so the passthrough regions must make the "
                                "region ids exactly 1..R (no gaps)");
        }
    }

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
    // Output.paraview and Output.grain_adjacency are plain booleans: every value is valid, and a
    // value of another TOML type is already rejected by the reader (output_from_toml).

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
