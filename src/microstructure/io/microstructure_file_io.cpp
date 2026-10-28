#include "microstructure/io/microstructure_file_io.hpp"

#include "axom/config.hpp"

#if defined(AXOM_USE_CONDUIT) && defined(AXOM_USE_HDF5)
#include "conduit.hpp"
#include "conduit_relay_config.h"
#if defined(CONDUIT_RELAY_IO_HDF5_ENABLED)
#include "conduit_relay_io_hdf5.hpp"
#define MS_HAVE_CONDUIT_HDF5 1
#endif
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

/*
 * Implementation notes (see microstructure_file_io.hpp for the layout).
 *
 * Everything small (header, grain table, grid description) is read and written with Conduit
 * relay. relay opens its own HDF5 handle, reads one sub-tree into a conduit::Node (hdf5_read of
 * "grains" reads only that group), and converts on demand (to_int64_array / to_float64_array
 * accept any stored integer or float type). The large per-cell grain id dataset is never read
 * here: voxel_map_io.cpp reads it in tiles with plain HDF5 hyperslabs.
 */

namespace exaconstit::microstructure {

namespace {

/** @brief Allowed deviation of a quaternion's norm from 1 (same rule as seed tables). */
constexpr double k_quat_norm_tolerance = 1.0e-3;

/** @brief "microstructure file '<path>': " for messages. */
std::string Prefix(const std::filesystem::path& path) {
    return "microstructure file '" + path.string() + "': ";
}

#ifdef MS_HAVE_CONDUIT_HDF5

/** @brief Owns a relay HDF5 file handle and closes it when destroyed. Host only. */
class RelayFile {
public:
    explicit RelayFile(const std::filesystem::path& path) : m_path(path) {
        try {
            m_id = conduit::relay::io::hdf5_open_file_for_read(path.string());
        } catch (const std::exception&) {
            throw std::runtime_error(Prefix(path) +
                                     "cannot open the file read-only (missing, unreadable or not "
                                     "an HDF5 file)");
        }
    }
    ~RelayFile() {
        try {
            conduit::relay::io::hdf5_close_file(m_id);
        } catch (...) {
        }
    }
    RelayFile(const RelayFile&) = delete;
    RelayFile& operator=(const RelayFile&) = delete;

    /** @brief True when `path` (relative to the root) exists. */
    bool Has(const std::string& path) const {
        return conduit::relay::io::hdf5_has_path(m_id, path);
    }

    /** @brief Reads the object at `path` (a dataset or a whole group) into a node. */
    conduit::Node Read(const std::string& path) const {
        if (!Has(path)) {
            throw std::runtime_error(Prefix(m_path) + "no '" + path + "'");
        }
        conduit::Node node;
        try {
            conduit::relay::io::hdf5_read(m_id, path, node);
        } catch (const std::exception& e) {
            throw std::runtime_error(Prefix(m_path) + "cannot read '" + path + "': " + e.what());
        }
        return node;
    }

    /** @brief The HDF5 shape of the dataset at `path` (slowest dimension first). */
    std::vector<std::int64_t> Shape(const std::string& path) const {
        conduit::Node info;
        try {
            conduit::relay::io::hdf5_read_info(m_id, path, info);
        } catch (const std::exception& e) {
            throw std::runtime_error(Prefix(m_path) + "cannot inspect '" + path + "': " + e.what());
        }
        return AsShape(info["num_elements"]);
    }

private:
    /** @brief The extents stored by hdf5_read_info (one per dimension). */
    static std::vector<std::int64_t> AsShape(const conduit::Node& extents) {
        conduit::Node converted;
        extents.to_int64_array(converted);
        const auto n = static_cast<std::size_t>(converted.dtype().number_of_elements());
        const conduit::int64* data = converted.as_int64_ptr();
        return std::vector<std::int64_t>(data, data + n);
    }

    std::filesystem::path m_path;
    hid_t m_id = -1;
};

/** @brief A string leaf (fixed or variable length). Throws if `node` is not a string. */
std::string
AsString(const std::filesystem::path& file, const conduit::Node& node, const std::string& path) {
    if (!node.dtype().is_string()) {
        throw std::runtime_error(Prefix(file) + "'" + path + "' must be a string");
    }
    return node.as_string();
}

/** @brief The values of an integer leaf, converted to int64, in stored (C) order. */
std::vector<std::int64_t>
AsInt64(const std::filesystem::path& file, const conduit::Node& node, const std::string& path) {
    if (!node.dtype().is_integer()) {
        throw std::runtime_error(Prefix(file) + "'" + path + "' must store integers");
    }
    conduit::Node converted;
    node.to_int64_array(converted);
    const auto n = static_cast<std::size_t>(converted.dtype().number_of_elements());
    const conduit::int64* data = converted.as_int64_ptr();
    return std::vector<std::int64_t>(data, data + n);
}

/** @brief The values of a numeric leaf (integer or float), converted to double. */
std::vector<double>
AsDouble(const std::filesystem::path& file, const conduit::Node& node, const std::string& path) {
    if (!node.dtype().is_number()) {
        throw std::runtime_error(Prefix(file) + "'" + path + "' must store numbers");
    }
    conduit::Node converted;
    node.to_float64_array(converted);
    const auto n = static_cast<std::size_t>(converted.dtype().number_of_elements());
    const conduit::float64* data = converted.as_float64_ptr();
    return std::vector<double>(data, data + n);
}

/** @brief One number from a leaf that must hold exactly one value. */
double
OneDouble(const std::filesystem::path& file, const conduit::Node& node, const std::string& path) {
    const std::vector<double> values = AsDouble(file, node, path);
    if (values.size() != 1) {
        throw std::runtime_error(Prefix(file) + "'" + path + "' must hold one number");
    }
    return values[0];
}

/** @brief One integer from a leaf that must hold exactly one value. */
std::int64_t
OneInt64(const std::filesystem::path& file, const conduit::Node& node, const std::string& path) {
    const std::vector<std::int64_t> values = AsInt64(file, node, path);
    if (values.size() != 1) {
        throw std::runtime_error(Prefix(file) + "'" + path + "' must hold one integer");
    }
    return values[0];
}

/** @brief The child `name` of `node`, or an error naming `path`/`name`. */
const conduit::Node& Child(const std::filesystem::path& file,
                           const conduit::Node& node,
                           const std::string& path,
                           const std::string& name) {
    if (!node.has_child(name)) {
        throw std::runtime_error(Prefix(file) + "no '" + path + "/" + name + "'");
    }
    return node[name];
}

/** @brief Header check on an open file (see CheckMicrostructureFileHeader). */
std::int64_t CheckHeader(const std::filesystem::path& path, const RelayFile& file) {
    if (!file.Has("format") || !file.Has("format_version")) {
        throw std::runtime_error(Prefix(path) +
                                 "not an ExaConstit microstructure file (no 'format' and "
                                 "'format_version' datasets at the root)");
    }
    const std::string name = AsString(path, file.Read("format"), "format");
    if (name != k_microstructure_format_name) {
        throw std::runtime_error(Prefix(path) + "'format' is \"" + name + "\", expected \"" +
                                 k_microstructure_format_name + "\"");
    }
    const std::int64_t version = OneInt64(path, file.Read("format_version"), "format_version");
    if (version < 1) {
        throw std::runtime_error(Prefix(path) + "format_version " + std::to_string(version) +
                                 " is not valid (versions start at 1)");
    }
    if (version > k_microstructure_format_version) {
        throw std::runtime_error(Prefix(path) + "format_version " + std::to_string(version) +
                                 " is newer than this ExaConstit reads (" +
                                 std::to_string(k_microstructure_format_version) +
                                 "); update ExaConstit");
    }
    return version;
}

/**
 * @brief Throws unless the dataset at `path` holds `per_grain` values for each of `grains`
 * grains, stored one row per grain: shape (per_grain * grains) (flat, as Conduit writes it) or
 * (grains, per_grain) (as h5py writes a list of rows); for one value per grain also (grains,
 * 1). A transposed array, (per_grain, grains), is refused instead of being misread.
 */
void CheckCount(const std::filesystem::path& file,
                const RelayFile& relay,
                const std::string& path,
                std::size_t count,
                std::size_t per_grain,
                std::size_t grains) {
    const std::vector<std::int64_t> shape = relay.Shape(path);
    const auto g = static_cast<std::int64_t>(grains);
    const auto c = static_cast<std::int64_t>(per_grain);
    const bool flat = shape.size() == 1 && shape[0] == g * c;
    const bool rows = shape.size() == 2 && shape[0] == g && shape[1] == c;
    if (count != per_grain * grains || (!flat && !rows)) {
        std::ostringstream out;
        out << Prefix(file) << "'" << path << "' has shape (";
        for (std::size_t d = 0; d < shape.size(); ++d) {
            out << (d > 0 ? ", " : "") << shape[d];
        }
        out << "); expected " << per_grain << " value(s) per grain, one row per grain: (" << g * c
            << ")";
        out << " or (" << g << ", " << c << ")";
        out << ", for the " << grains << " grains of 'grains/id'";
        throw std::runtime_error(out.str());
    }
}

/** @brief Throws unless every value is finite. */
void CheckFinite(const std::filesystem::path& file,
                 const std::string& path,
                 const std::vector<double>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error(Prefix(file) + "'" + path +
                                     "' holds a value that is not "
                                     "finite (entry " +
                                     std::to_string(i) + ")");
        }
    }
}

#endif // MS_HAVE_CONDUIT_HDF5

#ifndef MS_HAVE_CONDUIT_HDF5
/** @brief The error every file function throws without Conduit/HDF5 support. */
[[noreturn]] void ThrowNoConduit(const std::filesystem::path& path) {
    throw std::runtime_error(Prefix(path) +
                             "ExaConstit was built without Conduit HDF5 support (Axom with "
                             "Conduit and HDF5 is required for microstructure files)");
}
#endif

} // namespace

bool IsHdf5File(const std::filesystem::path& path) {
    // The 8-byte HDF5 signature, at offset 0 or at 512 * 2^k (files with a user block)
    static const unsigned char signature[8] = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    for (std::streamoff offset = 0; offset + 8 <= size; offset = (offset == 0) ? 512 : 2 * offset) {
        unsigned char bytes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        in.seekg(offset);
        if (!in.read(reinterpret_cast<char*>(bytes), 8)) {
            return false;
        }
        if (std::memcmp(bytes, signature, 8) == 0) {
            return true;
        }
    }
    return false;
}

std::int64_t CheckMicrostructureFileHeader(const std::filesystem::path& path) {
#ifdef MS_HAVE_CONDUIT_HDF5
    const RelayFile file(path);
    return CheckHeader(path, file);
#else
    ThrowNoConduit(path);
#endif
}

bool MicrostructureFileHasLabeledPoints(const std::filesystem::path& path) {
#ifdef MS_HAVE_CONDUIT_HDF5
    const RelayFile file(path);
    CheckHeader(path, file);
    return file.Has("mesh");
#else
    ThrowNoConduit(path);
#endif
}

/*
 * ReadMicrostructureGrainTable
 *
 * Steps: header; read the whole "grains" group (small: a few numbers per grain); ids (required,
 * >= 1, < 2^31, unique); each optional dataset: count, then its value rule.
 */
MicrostructureGrainTable ReadMicrostructureGrainTable(const std::filesystem::path& path) {
#ifdef MS_HAVE_CONDUIT_HDF5
    const RelayFile file(path);
    CheckHeader(path, file);
    if (!file.Has("grains/id")) {
        throw std::runtime_error(Prefix(path) + "no 'grains/id' (the grain table is required)");
    }
    const conduit::Node grains = file.Read("grains");

    MicrostructureGrainTable table;
    table.id = AsInt64(path, grains["id"], "grains/id");
    const std::size_t n = table.id.size();
    if (n == 0) {
        throw std::runtime_error(Prefix(path) + "'grains/id' is empty");
    }
    std::vector<std::int64_t> sorted = table.id;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 1 || sorted.back() > INT_MAX) {
        throw std::runtime_error(Prefix(path) + "'grains/id' must hold ids in [1, 2^31 - 1]");
    }
    const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
    if (dup != sorted.end()) {
        throw std::runtime_error(Prefix(path) + "'grains/id' holds the id " + std::to_string(*dup) +
                                 " more than once");
    }

    if (grains.has_child("quat")) {
        table.quat = AsDouble(path, grains["quat"], "grains/quat");
        CheckCount(path, file, "grains/quat", table.quat.size(), 4, n);
        for (std::size_t r = 0; r < n; ++r) {
            const double* q = &table.quat[4 * r];
            const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (!(std::fabs(norm - 1.0) <= k_quat_norm_tolerance)) {
                std::ostringstream out;
                out << Prefix(path) << "grain " << table.id[r] << ": quaternion norm " << norm
                    << " differs from 1 by more than " << k_quat_norm_tolerance
                    << " (wrong component order, Euler angles, or not a quaternion?)";
                throw std::runtime_error(out.str());
            }
        }
    }
    if (grains.has_child("phase")) {
        const std::vector<std::int64_t> phase = AsInt64(path, grains["phase"], "grains/phase");
        CheckCount(path, file, "grains/phase", phase.size(), 1, n);
        for (std::size_t r = 0; r < n; ++r) {
            if (phase[r] < 1 || phase[r] > INT_MAX) {
                throw std::runtime_error(Prefix(path) + "grain " + std::to_string(table.id[r]) +
                                         ": phase " + std::to_string(phase[r]) +
                                         " is not in [1, 2^31 - 1] (phase ids start at 1)");
            }
            table.phase.push_back(static_cast<int>(phase[r]));
        }
    }
    if (grains.has_child("position")) {
        table.position = AsDouble(path, grains["position"], "grains/position");
        CheckCount(path, file, "grains/position", table.position.size(), 3, n);
        CheckFinite(path, "grains/position", table.position);
    }
    if (grains.has_child("volume")) {
        table.volume = AsDouble(path, grains["volume"], "grains/volume");
        CheckCount(path, file, "grains/volume", table.volume.size(), 1, n);
        for (std::size_t r = 0; r < n; ++r) {
            if (!(table.volume[r] > 0.0) || !std::isfinite(table.volume[r])) {
                throw std::runtime_error(Prefix(path) + "grain " + std::to_string(table.id[r]) +
                                         ": volume must be finite and > 0");
            }
        }
    }
    if (grains.has_child("weight")) {
        table.weight = AsDouble(path, grains["weight"], "grains/weight");
        CheckCount(path, file, "grains/weight", table.weight.size(), 1, n);
        CheckFinite(path, "grains/weight", table.weight);
    }
    if (grains.has_child("shape")) {
        table.shape = AsDouble(path, grains["shape"], "grains/shape");
        CheckCount(path, file, "grains/shape", table.shape.size(), 6, n);
        CheckFinite(path, "grains/shape", table.shape);
    }
    return table;
#else
    ThrowNoConduit(path);
#endif
}

/*
 * ReadMicrostructureGrid
 *
 * Reads the coordset and topology groups (a handful of scalars) and the two string leaves of the
 * grain_id field; the field's values are left to the voxel reader.
 */
MicrostructureGrid ReadMicrostructureGrid(const std::filesystem::path& path) {
#ifdef MS_HAVE_CONDUIT_HDF5
    const RelayFile file(path);
    CheckHeader(path, file);
    if (!file.Has("mesh")) {
        throw std::runtime_error(Prefix(path) +
                                 "no labeled points ('mesh' group): the file holds only a grain "
                                 "table, which is a seed list (use it as Seeds.file with "
                                 "Seeds.source = \"file\")");
    }
    const std::string cs_path = "mesh/coordsets/coords";
    const std::string topo_path = "mesh/topologies/topo";
    const std::string field_path = "mesh/fields/grain_id";
    const conduit::Node coords = file.Read(cs_path);
    const conduit::Node topo = file.Read(topo_path);

    const std::string cs_type = AsString(
        path, Child(path, coords, cs_path, "type"), cs_path + "/type");
    if (cs_type != "uniform") {
        throw std::runtime_error(Prefix(path) + "'" + cs_path + "/type' is \"" + cs_type +
                                 "\"; only \"uniform\" grids are supported");
    }
    const std::string topo_type = AsString(
        path, Child(path, topo, topo_path, "type"), topo_path + "/type");
    const std::string topo_cs = AsString(
        path, Child(path, topo, topo_path, "coordset"), topo_path + "/coordset");
    if (topo_type != "uniform" || topo_cs != "coords") {
        throw std::runtime_error(Prefix(path) + "'" + topo_path +
                                 "' must have type \"uniform\" and coordset \"coords\"");
    }
    const std::string assoc = AsString(
        path, file.Read(field_path + "/association"), field_path + "/association");
    const std::string field_topo = AsString(
        path, file.Read(field_path + "/topology"), field_path + "/topology");
    if (assoc != "element" || field_topo != "topo") {
        throw std::runtime_error(Prefix(path) + "'" + field_path +
                                 "' must have association \"element\" (one grain id per cell) "
                                 "and topology \"topo\"");
    }
    if (!file.Has(field_path + "/values")) {
        throw std::runtime_error(Prefix(path) + "no '" + field_path + "/values'");
    }

    MicrostructureGrid grid;
    for (const char* group : {"dims", "origin", "spacing"}) {
        Child(path, coords, cs_path, group); // throws when missing
    }
    const conduit::Node& dims = coords["dims"];
    const conduit::Node& origin = coords["origin"];
    const conduit::Node& spacing = coords["spacing"];
    const char* dim_names[3] = {"i", "j", "k"};
    const char* origin_names[3] = {"x", "y", "z"};
    const char* spacing_names[3] = {"dx", "dy", "dz"};
    for (int d = 0; d < 3; ++d) {
        const std::string dpath = cs_path + "/dims/" + dim_names[d];
        const std::int64_t points = OneInt64(
            path, Child(path, dims, cs_path + "/dims", dim_names[d]), dpath);
        if (points < 2) {
            throw std::runtime_error(Prefix(path) + "'" + dpath + "' is " + std::to_string(points) +
                                     "; it counts grid POINTS (cells + 1), so it must be >= 2");
        }
        grid.dims[d] = points - 1;
        grid.origin[d] = OneDouble(path,
                                   Child(path, origin, cs_path + "/origin", origin_names[d]),
                                   cs_path + "/origin/" + origin_names[d]);
        grid.spacing[d] = OneDouble(path,
                                    Child(path, spacing, cs_path + "/spacing", spacing_names[d]),
                                    cs_path + "/spacing/" + spacing_names[d]);
    }
    return grid;
#else
    ThrowNoConduit(path);
#endif
}

/*
 * WriteMicrostructureFile
 *
 * Checks the array sizes, builds one conduit::Node (header + grains/...) that points at the
 * caller's arrays (set_external: no copy), and saves it, replacing an existing file.
 */
void WriteMicrostructureFile(const std::filesystem::path& path,
                             const MicrostructureGrainTable& grains) {
    const std::size_t n = grains.Size();
    auto check = [&](const std::vector<double>& v, std::size_t per, const char* name) {
        if (!v.empty() && v.size() != per * n) {
            throw std::invalid_argument("WriteMicrostructureFile: grains/" + std::string(name) +
                                        " must be empty or hold " + std::to_string(per) +
                                        " values per grain");
        }
    };
    check(grains.quat, 4, "quat");
    check(grains.position, 3, "position");
    check(grains.volume, 1, "volume");
    check(grains.weight, 1, "weight");
    check(grains.shape, 6, "shape");
    if (!grains.phase.empty() && grains.phase.size() != n) {
        throw std::invalid_argument("WriteMicrostructureFile: grains/phase must be empty or hold "
                                    "one value per grain");
    }
    for (const std::int64_t id : grains.id) {
        if (id < 1) {
            throw std::invalid_argument("WriteMicrostructureFile: grain ids must be >= 1");
        }
    }
#ifdef MS_HAVE_CONDUIT_HDF5
    conduit::Node node;
    node["format"] = std::string(k_microstructure_format_name);
    node["format_version"].set_int64(k_microstructure_format_version);
    const auto count = static_cast<conduit::index_t>(n);
    // set_external takes non-const pointers but only reads through them here
    node["grains/id"].set_external(const_cast<std::int64_t*>(grains.id.data()), count);
    auto add = [&](const char* name, const std::vector<double>& v) {
        if (!v.empty()) {
            node[std::string("grains/") + name].set_external(
                const_cast<double*>(v.data()), static_cast<conduit::index_t>(v.size()));
        }
    };
    add("quat", grains.quat);
    if (!grains.phase.empty()) {
        node["grains/phase"].set_external(const_cast<int*>(grains.phase.data()), count);
    }
    add("position", grains.position);
    add("volume", grains.volume);
    add("weight", grains.weight);
    add("shape", grains.shape);
    try {
        conduit::relay::io::hdf5_save(node, path.string(), "/");
    } catch (const std::exception& e) {
        throw std::runtime_error(Prefix(path) + "cannot write the file: " + e.what());
    }
#else
    ThrowNoConduit(path);
#endif
}

} // namespace exaconstit::microstructure
