#include "microstructure/io/seed_table_io.hpp"

#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

/*
 * Implementation notes (see seed_table_io.hpp for the file format).
 *
 * Reading is done by rank 0 alone, in two passes over the text: the header is turned into a
 * column map (which field index holds x, y, z, qw, ...), and every data line is then converted
 * field by field into the per-seed arrays. Every rule of the format is checked while parsing, and
 * the first violation stops the parse with a message that names the file and the line.
 *
 * Rank 0 then broadcasts either the error message or the packed table (all arrays, one MPI_Bcast
 * per array), so every rank returns the same table or throws the same exception. Collective here
 * means: every rank of the communicator must call ReadSeedTable, because the ranks exchange data
 * inside it; a rank that skipped it would leave the others waiting.
 */

namespace exaconstit::microstructure {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

/** Largest allowed deviation of |q| from 1 (more usually means swapped columns or Euler angles). */
constexpr double QUAT_NORM_TOLERANCE = 1.0e-3;

/** Largest number of entries sent in one MPI_Bcast (keeps the int count far from INT_MAX). */
constexpr std::size_t BCAST_CHUNK = std::size_t(1) << 28;

/** Removes leading and trailing whitespace (spaces, tabs, carriage returns, ...). */
std::string Trim(const std::string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        begin++;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        end--;
    }
    return text.substr(begin, end - begin);
}

/** Lower-case copy (ASCII), used to make column names case-insensitive. */
std::string ToLower(const std::string& text) {
    std::string out = text;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

/** Splits a line at every comma and trims each field. "a, b,,c" -> {"a", "b", "", "c"}. */
std::vector<std::string> SplitFields(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        if (comma == std::string::npos) {
            fields.push_back(Trim(line.substr(start)));
            break;
        }
        fields.push_back(Trim(line.substr(start, comma - start)));
        start = comma + 1;
    }
    return fields;
}

/**
 * Converts a whole field to a double with strtod; false if the field is empty or has characters
 * left over after the number ("1.5abc", "1,5"), or if it is a hexadecimal float ("0x1p3", which
 * strtod would accept): only decimal numbers are allowed. Values out of range ("1e400") come back
 * as +-inf, and "inf" / "nan" parse as such; the caller's finiteness check refuses all of them.
 */
bool ParseDouble(const std::string& field, double& out) {
    if (field.empty()) {
        return false;
    }
    if (field.find_first_of("xX") != std::string::npos) {
        return false; // hexadecimal float
    }
    char* end = nullptr;
    out = std::strtod(field.c_str(), &end);
    return end == field.c_str() + field.size();
}

/** Converts a whole field to a 64-bit integer with strtoll (base 10); false on any leftover or
 * overflow. */
bool ParseInt64(const std::string& field, std::int64_t& out) {
    if (field.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(field.c_str(), &end, 10);
    if (errno == ERANGE || end != field.c_str() + field.size()) {
        return false;
    }
    out = static_cast<std::int64_t>(value);
    return true;
}

/**
 * True for column names that describe an orientation some other way than a quaternion: Bunge
 * Euler angles (phi1, phi, phi2, euler*, bunge*), Rodrigues vectors, exponential maps and rotation
 * matrix entries r11..r33. The generator never converts conventions, so these are refused.
 */
bool IsOtherOrientationColumn(const std::string& name) {
    if (name == "phi1" || name == "phi" || name == "phi2") {
        return true;
    }
    const char* prefixes[] = {"euler", "bunge", "rodrigues", "expmap"};
    for (const char* prefix : prefixes) {
        if (name.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    // Rotation matrix entries r<i><j>, i, j in 1..3
    if (name.size() == 3 && name[0] == 'r' && name[1] >= '1' && name[1] <= '3' &&
        name[2] >= '1' && name[2] <= '3') {
        return true;
    }
    return false;
}

/**
 * True for quaternion columns whose component order is ambiguous: q0..q3 (scalar first or last
 * depends on the code that wrote them) and any name starting with "quat".
 */
bool IsAmbiguousQuaternionColumn(const std::string& name) {
    if (name.size() == 2 && name[0] == 'q' && name[1] >= '0' && name[1] <= '3') {
        return true;
    }
    return name.rfind("quat", 0) == 0;
}

/** Formats a double with 17 significant digits (enough to read back the same bits). */
std::string Format17(double value) {
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return std::string(buffer);
}

/**
 * Field index of every known column in the header (-1 when absent). The shape columns are kept
 * in the storage order m11 m22 m33 m23 m13 m12.
 */
struct ColumnMap {
    int x = -1, y = -1, z = -1, id = -1;
    int quat[4] = {-1, -1, -1, -1}; ///< qw qx qy qz
    int phase = -1, weight = -1;
    int volume = -1, radius = -1, diameter = -1;
    int shape[6] = {-1, -1, -1, -1, -1, -1}; ///< m11 m22 m33 m23 m13 m12
    std::size_t num_fields = 0;
    std::vector<std::string> ignored;
};

/*
 * ReadHeader: classifies the column names of the header line.
 *
 * Steps:
 *  1. Split, trim and lower-case the names; empty or repeated names are errors.
 *  2. Look each name up in the table of known columns; names describing another orientation
 *     parametrization are errors, anything else is recorded as ignored.
 *  3. Check the column groups: x, y, z required; the quaternion and the shape tensor complete or
 *     absent; at most one size column.
 * `where` is the "file:line" prefix for messages.
 */
ColumnMap ReadHeader(const std::string& line, const std::string& where) {
    ColumnMap map;
    const std::vector<std::string> names = SplitFields(line);
    map.num_fields = names.size();

    // Pointers to the slots of the known names, so step 2 is a single lookup
    std::map<std::string, int*> known = {
        {"x", &map.x},           {"y", &map.y},           {"z", &map.z},
        {"id", &map.id},         {"qw", &map.quat[0]},    {"qx", &map.quat[1]},
        {"qy", &map.quat[2]},    {"qz", &map.quat[3]},    {"phase", &map.phase},
        {"weight", &map.weight}, {"volume", &map.volume}, {"radius", &map.radius},
        {"diameter", &map.diameter}, {"m11", &map.shape[0]}, {"m22", &map.shape[1]},
        {"m33", &map.shape[2]},  {"m23", &map.shape[3]},  {"m13", &map.shape[4]},
        {"m12", &map.shape[5]}};

    // Steps 1 and 2: one name at a time
    std::vector<std::string> seen;
    for (std::size_t k = 0; k < names.size(); k++) {
        const std::string name = ToLower(names[k]);
        if (name.empty()) {
            throw std::runtime_error(where + ": header column " + std::to_string(k + 1) +
                                     " has an empty name");
        }
        if (std::find(seen.begin(), seen.end(), name) != seen.end()) {
            throw std::runtime_error(where + ": column '" + name +
                                     "' appears more than once in the header");
        }
        seen.push_back(name);
        const auto slot = known.find(name);
        if (slot != known.end()) {
            *slot->second = static_cast<int>(k);
        } else if (IsAmbiguousQuaternionColumn(name)) {
            throw std::runtime_error(
                where + ": column '" + name +
                "' is a quaternion component of unknown order (scalar first or last?); name the "
                "quaternion columns qw,qx,qy,qz (crystal-to-sample, passive)");
        } else if (IsOtherOrientationColumn(name)) {
            throw std::runtime_error(
                where + ": column '" + name +
                "' looks like Euler angles, a Rodrigues vector, an exponential map or a rotation "
                "matrix; seed tables take orientations only as quaternions: supply the columns "
                "qw,qx,qy,qz (crystal-to-sample, passive)");
        } else {
            map.ignored.push_back(name);
        }
    }

    // Step 3: column groups
    const char* xyz[3] = {"x", "y", "z"};
    const int xyz_index[3] = {map.x, map.y, map.z};
    for (int d = 0; d < 3; d++) {
        if (xyz_index[d] < 0) {
            throw std::runtime_error(where + ": missing required column '" + xyz[d] + "'");
        }
    }
    const char* quat_names[4] = {"qw", "qx", "qy", "qz"};
    int quat_count = 0;
    std::string quat_missing;
    for (int k = 0; k < 4; k++) {
        if (map.quat[k] >= 0) {
            quat_count++;
        } else {
            quat_missing += std::string(quat_missing.empty() ? "" : ", ") + quat_names[k];
        }
    }
    if (quat_count != 0 && quat_count != 4) {
        throw std::runtime_error(where + ": incomplete quaternion: the columns qw, qx, qy, qz "
                                         "must all be present or all absent (missing: " +
                                 quat_missing + ")");
    }
    const char* shape_names[6] = {"m11", "m22", "m33", "m23", "m13", "m12"};
    int shape_count = 0;
    std::string shape_missing;
    for (int k = 0; k < 6; k++) {
        if (map.shape[k] >= 0) {
            shape_count++;
        } else {
            shape_missing += std::string(shape_missing.empty() ? "" : ", ") + shape_names[k];
        }
    }
    if (shape_count != 0 && shape_count != 6) {
        throw std::runtime_error(where + ": incomplete shape tensor: the columns m11, m22, m33, "
                                         "m23, m13, m12 must all be present or all absent "
                                         "(missing: " +
                                 shape_missing + ")");
    }
    const int size_count =
        (map.volume >= 0 ? 1 : 0) + (map.radius >= 0 ? 1 : 0) + (map.diameter >= 0 ? 1 : 0);
    if (size_count > 1) {
        throw std::runtime_error(where + ": at most one size column is allowed (volume, radius "
                                         "or diameter), found " +
                                 std::to_string(size_count));
    }
    return map;
}

/*
 * InvertShapeTensor: A = M^-1 for a symmetric 3x3 M given as m11 m22 m33 m23 m13 m12. M is first
 * divided by its largest diagonal entry (making the test and the inverse independent of the
 * length unit), then checked by Sylvester's criterion (all leading principal minors positive:
 * m11 > 0, m11 m22 - m12^2 > 0, det M > 0). The inverse is the adjugate (matrix of cofactors)
 * divided by the determinant, written out entry by entry, and divided by the scale again. Returns false for a non-SPD or non-finite M.
 */
bool InvertShapeTensor(const double m_in[6], double a[6]) {
    // Divide by the largest diagonal entry first, so the minors are of order 1 whatever the
    // units: a tiny s^2 M (say 1e-110 m^2) would otherwise underflow det M to 0 and be refused
    const double scale = std::max({m_in[0], m_in[1], m_in[2]});
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return false; // an SPD matrix has a positive diagonal
    }
    double m[6];
    for (int k = 0; k < 6; k++) {
        m[k] = m_in[k] / scale;
    }
    const double m11 = m[0], m22 = m[1], m33 = m[2], m23 = m[3], m13 = m[4], m12 = m[5];
    // Cofactors of the symmetric matrix [[m11 m12 m13] [m12 m22 m23] [m13 m23 m33]]
    const double c11 = m22 * m33 - m23 * m23;
    const double c22 = m11 * m33 - m13 * m13;
    const double c33 = m11 * m22 - m12 * m12;
    const double c23 = m13 * m12 - m11 * m23;
    const double c13 = m12 * m23 - m22 * m13;
    const double c12 = m13 * m23 - m33 * m12;
    const double det = m11 * c11 + m12 * c12 + m13 * c13;
    if (!(m11 > 0.0) || !(c33 > 0.0) || !(det > 0.0) || !std::isfinite(det)) {
        return false;
    }
    // A = (M / scale)^-1 / scale
    a[0] = c11 / det / scale;
    a[1] = c22 / det / scale;
    a[2] = c33 / det / scale;
    a[3] = c23 / det / scale;
    a[4] = c13 / det / scale;
    a[5] = c12 / det / scale;
    for (int k = 0; k < 6; k++) {
        if (!std::isfinite(a[k])) {
            return false;
        }
    }
    return true;
}

/*
 * ParseSeedTableFile (rank 0 only, throws std::runtime_error on the first error)
 *
 * Steps:
 *  1. Read lines, skipping blank lines and comments; the first remaining line is the header.
 *  2. For every data line: check the field count, then convert each used field, check its rule
 *     and convert it to mesh units (positions s x + t, volumes s^3 V or 4/3 pi (s r)^3, shape
 *     tensors s^2 M and their inverse; quaternions and weights are copied unchanged).
 *  3. Whole-table checks: at least one seed, at most 2^31 - 1 seeds (seed indices are int),
 *     unique ids (sorted list of (id, line); equal neighbors are duplicates).
 *  4. Warnings (returned, printed by the caller on rank 0): ignored columns, duplicate positions.
 */
SeedTable ParseSeedTableFile(const SeedTableParameters& params,
                             std::vector<std::string>& warnings) {
    const std::string file = params.path.string();
    std::ifstream in(params.path);
    if (!in) {
        throw std::runtime_error("cannot open seed table '" + file + "'");
    }
    const double s = params.units_scale;

    SeedTable table;
    SeedSet& seeds = table.seeds;
    ColumnMap map;
    bool have_header = false;
    std::vector<std::int64_t> row_line; // file line of every seed, for duplicate-id messages
    std::string line;
    std::int64_t line_number = 0;

    while (std::getline(in, line)) {
        line_number++;
        // A UTF-8 byte-order mark (bytes EF BB BF, written e.g. by Excel's "CSV UTF-8" export)
        // would otherwise become part of the first column name
        if (line_number == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
            line.erase(0, 3);
        }
        const std::string where = file + ":" + std::to_string(line_number);
        const std::string trimmed = Trim(line);

        // Step 1: skip blank and comment lines; the first other line is the header
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        if (!have_header) {
            map = ReadHeader(trimmed, where);
            have_header = true;
            table.has_quat = map.quat[0] >= 0;
            table.has_phase = map.phase >= 0;
            table.has_weight = map.weight >= 0;
            table.has_volume = map.volume >= 0 || map.radius >= 0 || map.diameter >= 0;
            table.has_shape = map.shape[0] >= 0;
            table.ignored_columns = map.ignored;
            continue;
        }

        // Step 2: one seed per data line
        const std::vector<std::string> fields = SplitFields(trimmed);
        if (fields.size() != map.num_fields) {
            throw std::runtime_error(where + ": expected " + std::to_string(map.num_fields) +
                                     " comma-separated fields (one per header column), found " +
                                     std::to_string(fields.size()));
        }
        const std::int64_t row = static_cast<std::int64_t>(seeds.x.size());

        // A finite double from field `index`, named `name` in messages
        auto number = [&](int index, const char* name) {
            double value = 0.0;
            if (!ParseDouble(fields[static_cast<std::size_t>(index)], value)) {
                throw std::runtime_error(where + ": column '" + name + "': '" +
                                         fields[static_cast<std::size_t>(index)] +
                                         "' is not a number");
            }
            if (!std::isfinite(value)) {
                throw std::runtime_error(where + ": column '" + name + "': '" +
                                         fields[static_cast<std::size_t>(index)] +
                                         "' is not finite (or out of the double range)");
            }
            return value;
        };
        // A whole number from field `index`
        auto integer = [&](int index, const char* name) {
            std::int64_t value = 0;
            if (!ParseInt64(fields[static_cast<std::size_t>(index)], value)) {
                throw std::runtime_error(where + ": column '" + name + "': '" +
                                         fields[static_cast<std::size_t>(index)] +
                                         "' is not an integer");
            }
            return value;
        };

        // Position: mesh = s * table + translate
        seeds.x.push_back(s * number(map.x, "x") + params.translate[0]);
        seeds.y.push_back(s * number(map.y, "y") + params.translate[1]);
        seeds.z.push_back(s * number(map.z, "z") + params.translate[2]);

        // Id: given (>= 0) or the 0-based row index
        if (map.id >= 0) {
            const std::int64_t id = integer(map.id, "id");
            if (id < 0) {
                throw std::runtime_error(where + ": column 'id': ids must be >= 0");
            }
            seeds.source_id.push_back(id);
        } else {
            seeds.source_id.push_back(row);
        }

        // Quaternion: copied verbatim after the norm check
        if (table.has_quat) {
            const char* names[4] = {"qw", "qx", "qy", "qz"};
            double q[4];
            for (int k = 0; k < 4; k++) {
                q[k] = number(map.quat[k], names[k]);
            }
            const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (std::fabs(norm - 1.0) > QUAT_NORM_TOLERANCE) {
                throw std::runtime_error(
                    where + ": quaternion (qw, qx, qy, qz) has norm " + Format17(norm) +
                    ", not 1 within " + Format17(QUAT_NORM_TOLERANCE) +
                    "; check for swapped columns or Euler angles in the quaternion columns");
            }
            for (int k = 0; k < 4; k++) {
                seeds.quat.push_back(q[k]);
            }
        }

        // Phase: integer >= 0 that fits an int
        if (table.has_phase) {
            const std::int64_t phase = integer(map.phase, "phase");
            if (phase < 0 || phase > INT_MAX) {
                throw std::runtime_error(where + ": column 'phase': must be in [0, 2^31 - 1]");
            }
            seeds.phase.push_back(static_cast<int>(phase));
        }

        // Weight: already in mesh units, not scaled
        if (table.has_weight) {
            seeds.weight.push_back(number(map.weight, "weight"));
        }

        // Size: one of volume, radius, diameter, stored as a volume in mesh units
        if (table.has_volume) {
            const int index = (map.volume >= 0) ? map.volume
                              : (map.radius >= 0) ? map.radius
                                                  : map.diameter;
            const char* name = (map.volume >= 0)   ? "volume"
                               : (map.radius >= 0) ? "radius"
                                                   : "diameter";
            const double value = number(index, name);
            if (!(value > 0.0)) {
                throw std::runtime_error(where + ": column '" + name + "': must be > 0");
            }
            double v = 0.0;
            if (map.volume >= 0) {
                v = s * s * s * value;
            } else {
                const double r = s * ((map.radius >= 0) ? value : 0.5 * value);
                v = 4.0 / 3.0 * PI * r * r * r;
            }
            table.volume.push_back(v);
        }

        // Shape tensor: M scaled by s^2, kept, and inverted into the Quadric tensor A
        if (table.has_shape) {
            const char* names[6] = {"m11", "m22", "m33", "m23", "m13", "m12"};
            double m[6];
            for (int k = 0; k < 6; k++) {
                m[k] = s * s * number(map.shape[k], names[k]);
            }
            double a[6];
            if (!InvertShapeTensor(m, a)) {
                throw std::runtime_error(where + ": the shape tensor (m11, m22, m33, m23, m13, "
                                                 "m12) is not symmetric positive definite");
            }
            for (int k = 0; k < 6; k++) {
                table.shape_tensor.push_back(m[k]);
                seeds.shape.push_back(a[k]);
            }
        }
        row_line.push_back(line_number);
    }

    // Step 3: whole-table checks
    if (!have_header) {
        throw std::runtime_error(file + ": no header line (the file is empty or only comments)");
    }
    const std::size_t n = seeds.x.size();
    if (n == 0) {
        throw std::runtime_error(file + ": the table has a header but no seed rows");
    }
    if (n > static_cast<std::size_t>(INT_MAX)) {
        throw std::runtime_error(file + ": more than 2^31 - 1 seeds");
    }
    if (map.id >= 0) {
        std::vector<std::pair<std::int64_t, std::int64_t>> id_line(n); // (id, line)
        for (std::size_t i = 0; i < n; i++) {
            id_line[i] = {seeds.source_id[i], row_line[i]};
        }
        std::sort(id_line.begin(), id_line.end());
        for (std::size_t i = 1; i < n; i++) {
            if (id_line[i].first == id_line[i - 1].first) {
                throw std::runtime_error(file + ":" + std::to_string(id_line[i].second) +
                                         ": duplicate id " + std::to_string(id_line[i].first) +
                                         " (first used on line " +
                                         std::to_string(id_line[i - 1].second) + ")");
            }
        }
    }

    // Step 4: warnings
    for (const std::string& name : table.ignored_columns) {
        warnings.push_back("seed table '" + file + "': ignoring unknown column '" + name + "'");
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (seeds.x[a] != seeds.x[b]) {
            return seeds.x[a] < seeds.x[b];
        }
        if (seeds.y[a] != seeds.y[b]) {
            return seeds.y[a] < seeds.y[b];
        }
        if (seeds.z[a] != seeds.z[b]) {
            return seeds.z[a] < seeds.z[b];
        }
        return a < b;
    });
    std::int64_t duplicates = 0;
    for (std::size_t k = 1; k < n; k++) {
        const std::size_t a = order[k - 1];
        const std::size_t b = order[k];
        if (seeds.x[a] == seeds.x[b] && seeds.y[a] == seeds.y[b] && seeds.z[a] == seeds.z[b]) {
            duplicates++;
        }
    }
    if (duplicates > 0) {
        warnings.push_back("seed table '" + file + "': " + std::to_string(duplicates) +
                           " seed(s) share their position with another seed; the seed search "
                           "keeps the one with the lowest index");
    }
    return table;
}

/**
 * Broadcasts a vector from rank 0 (its size must already be known on every rank: `count`
 * entries), in chunks of at most BCAST_CHUNK entries so each MPI count fits an int. Collective.
 */
template <typename T>
void BroadcastVector(std::vector<T>& data, std::size_t count, MPI_Datatype type, MPI_Comm comm) {
    data.resize(count);
    for (std::size_t start = 0; start < count; start += BCAST_CHUNK) {
        const std::size_t chunk = std::min(BCAST_CHUNK, count - start);
        MPI_Bcast(data.data() + start, static_cast<int>(chunk), type, 0, comm);
    }
}

} // namespace

/*
 * ReadSeedTable (collective)
 *
 * Steps:
 *  1. Check the parameters on every rank (identical data, so every rank throws together).
 *  2. Rank 0 parses the file (ParseSeedTableFile); any exception becomes an error message.
 *  3. The message is broadcast (BroadcastString); if it is not empty, every rank throws it.
 *  4. Rank 0 prints the warnings.
 *  5. The sizes and flags, then every array, are broadcast from rank 0, so all ranks return the
 *     same table, bit for bit.
 */
SeedTable ReadSeedTable(const SeedTableParameters& params, MPI_Comm comm) {
    // Step 1
    if (!(params.units_scale > 0.0) || !std::isfinite(params.units_scale)) {
        throw std::invalid_argument("ReadSeedTable: units_scale must be positive and finite");
    }
    for (int d = 0; d < 3; d++) {
        if (!std::isfinite(params.translate[d])) {
            throw std::invalid_argument("ReadSeedTable: translate must be finite");
        }
    }
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // Step 2: parse on rank 0
    SeedTable table;
    std::vector<std::string> warnings;
    std::string error;
    if (rank == 0) {
        try {
            table = ParseSeedTableFile(params, warnings);
        } catch (const std::exception& e) {
            error = std::string("ReadSeedTable: ") + e.what();
        }
    }

    // Step 3: same outcome on every rank
    const std::string message = BroadcastString(error, 0, comm);
    if (!message.empty()) {
        throw std::runtime_error(message);
    }

    // Step 4: warnings, once
    if (rank == 0) {
        for (const std::string& warning : warnings) {
            std::cerr << "Warning: " << warning << std::endl;
        }
    }

    // Step 5: header (seed count, flags, ignored-column count), then the arrays
    std::int64_t header[7] = {0, 0, 0, 0, 0, 0, 0};
    if (rank == 0) {
        header[0] = static_cast<std::int64_t>(table.seeds.Size());
        header[1] = table.has_quat ? 1 : 0;
        header[2] = table.has_phase ? 1 : 0;
        header[3] = table.has_weight ? 1 : 0;
        header[4] = table.has_volume ? 1 : 0;
        header[5] = table.has_shape ? 1 : 0;
        header[6] = static_cast<std::int64_t>(table.ignored_columns.size());
    }
    MPI_Bcast(header, 7, MPI_INT64_T, 0, comm);
    const std::size_t n = static_cast<std::size_t>(header[0]);
    table.has_quat = header[1] != 0;
    table.has_phase = header[2] != 0;
    table.has_weight = header[3] != 0;
    table.has_volume = header[4] != 0;
    table.has_shape = header[5] != 0;

    SeedSet& seeds = table.seeds;
    BroadcastVector(seeds.x, n, MPI_DOUBLE, comm);
    BroadcastVector(seeds.y, n, MPI_DOUBLE, comm);
    BroadcastVector(seeds.z, n, MPI_DOUBLE, comm);
    BroadcastVector(seeds.source_id, n, MPI_INT64_T, comm);
    BroadcastVector(seeds.quat, table.has_quat ? 4 * n : 0, MPI_DOUBLE, comm);
    BroadcastVector(seeds.phase, table.has_phase ? n : 0, MPI_INT, comm);
    BroadcastVector(seeds.weight, table.has_weight ? n : 0, MPI_DOUBLE, comm);
    BroadcastVector(table.volume, table.has_volume ? n : 0, MPI_DOUBLE, comm);
    BroadcastVector(seeds.shape, table.has_shape ? 6 * n : 0, MPI_DOUBLE, comm);
    BroadcastVector(table.shape_tensor, table.has_shape ? 6 * n : 0, MPI_DOUBLE, comm);
    table.ignored_columns.resize(static_cast<std::size_t>(header[6]));
    for (std::string& name : table.ignored_columns) {
        name = BroadcastString(name, 0, comm);
    }
    return table;
}

/*
 * WriteSeedTable
 *
 * Checks that every array matches its has_* flag, then writes a comment line, the header and one
 * line per seed. Doubles use "%.17g" (17 significant digits), which is enough for strtod to
 * recover the same double, so ReadSeedTable(units_scale = 1, translate = 0) gives back the same
 * table: positions s x + t = 1 x + 0 = x, volumes 1^3 V = V, shape tensors 1^2 M = M (whose
 * inverse is then computed exactly as before), quaternions and weights verbatim.
 */
void WriteSeedTable(const std::filesystem::path& path, const SeedTable& table) {
    const SeedSet& seeds = table.seeds;
    const std::size_t n = seeds.Size();

    // Array sizes against the flags
    auto check = [&](bool present, std::size_t size, std::size_t per_seed, const char* name) {
        if (present && size != per_seed * n) {
            throw std::invalid_argument(std::string("WriteSeedTable: ") + name +
                                        " has the wrong size for the number of seeds");
        }
    };
    if (seeds.y.size() != n || seeds.z.size() != n || seeds.source_id.size() != n) {
        throw std::invalid_argument("WriteSeedTable: x, y, z and source_id must have equal sizes");
    }
    check(table.has_quat, seeds.quat.size(), 4, "quat");
    check(table.has_phase, seeds.phase.size(), 1, "phase");
    check(table.has_weight, seeds.weight.size(), 1, "weight");
    check(table.has_volume, table.volume.size(), 1, "volume");
    check(table.has_shape, table.shape_tensor.size(), 6, "shape_tensor");

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("WriteSeedTable: cannot open '" + path.string() +
                                 "' for writing");
    }

    // Comment and header
    out << "# ExaConstit seed table v1\n";
    out << "id,x,y,z";
    if (table.has_quat) {
        out << ",qw,qx,qy,qz";
    }
    if (table.has_phase) {
        out << ",phase";
    }
    if (table.has_weight) {
        out << ",weight";
    }
    if (table.has_volume) {
        out << ",volume";
    }
    if (table.has_shape) {
        out << ",m11,m22,m33,m23,m13,m12";
    }
    out << "\n";

    // One line per seed, columns in header order
    for (std::size_t i = 0; i < n; i++) {
        out << seeds.source_id[i] << "," << Format17(seeds.x[i]) << "," << Format17(seeds.y[i])
            << "," << Format17(seeds.z[i]);
        if (table.has_quat) {
            for (int k = 0; k < 4; k++) {
                out << "," << Format17(seeds.quat[4 * i + static_cast<std::size_t>(k)]);
            }
        }
        if (table.has_phase) {
            out << "," << seeds.phase[i];
        }
        if (table.has_weight) {
            out << "," << Format17(seeds.weight[i]);
        }
        if (table.has_volume) {
            out << "," << Format17(table.volume[i]);
        }
        if (table.has_shape) {
            for (int k = 0; k < 6; k++) {
                out << "," << Format17(table.shape_tensor[6 * i + static_cast<std::size_t>(k)]);
            }
        }
        out << "\n";
    }
    if (!out) {
        throw std::runtime_error("WriteSeedTable: error while writing '" + path.string() + "'");
    }
}

/*
 * KeepSeedsInBox
 *
 * One pass over the seeds: seed i is kept when lo[d] <= p_d <= hi[d] on every axis, and every
 * per-seed array is compacted in place with a running "write" index (multi-component arrays move
 * all components of a seed together). Arrays that are empty stay empty.
 */
std::int64_t KeepSeedsInBox(SeedTable& table,
                            const std::array<double, 3>& lo,
                            const std::array<double, 3>& hi) {
    SeedSet& s = table.seeds;
    const std::size_t n = s.Size();

    // Moves `per_seed` components of seed `from` to slot `to` when the array is in use
    auto move = [](auto& array, std::size_t per_seed, std::size_t from, std::size_t to) {
        if (array.empty()) {
            return;
        }
        for (std::size_t k = 0; k < per_seed; k++) {
            array[per_seed * to + k] = array[per_seed * from + k];
        }
    };
    auto shrink = [](auto& array, std::size_t per_seed, std::size_t kept) {
        if (!array.empty()) {
            array.resize(per_seed * kept);
        }
    };

    std::size_t kept = 0;
    for (std::size_t i = 0; i < n; i++) {
        const bool inside = s.x[i] >= lo[0] && s.x[i] <= hi[0] && s.y[i] >= lo[1] &&
                            s.y[i] <= hi[1] && s.z[i] >= lo[2] && s.z[i] <= hi[2];
        if (!inside) {
            continue;
        }
        if (kept != i) {
            move(s.x, 1, i, kept);
            move(s.y, 1, i, kept);
            move(s.z, 1, i, kept);
            move(s.weight, 1, i, kept);
            move(s.shape, 6, i, kept);
            move(s.quat, 4, i, kept);
            move(s.source_id, 1, i, kept);
            move(s.phase, 1, i, kept);
            move(s.parent_id, 1, i, kept);
            move(s.nucleation_time, 1, i, kept);
            move(table.volume, 1, i, kept);
            move(table.shape_tensor, 6, i, kept);
        }
        kept++;
    }

    // Drop the tail of every array
    shrink(s.x, 1, kept);
    shrink(s.y, 1, kept);
    shrink(s.z, 1, kept);
    shrink(s.weight, 1, kept);
    shrink(s.shape, 6, kept);
    shrink(s.quat, 4, kept);
    shrink(s.source_id, 1, kept);
    shrink(s.phase, 1, kept);
    shrink(s.parent_id, 1, kept);
    shrink(s.nucleation_time, 1, kept);
    shrink(table.volume, 1, kept);
    shrink(table.shape_tensor, 6, kept);
    return static_cast<std::int64_t>(n - kept);
}

} // namespace exaconstit::microstructure
