/*
 * Implementation of the voxel map reader (voxel_map_io.hpp).
 *
 * Terms used in this file:
 * - HDF5 handle (hid_t): an integer that names an open HDF5 object (file, dataset, dataspace,
 *   attribute, datatype). Every handle must be closed with the matching H5?close function, on
 *   every code path; the small H5Handle class below closes it automatically when it goes out of
 *   scope (also when an exception is thrown).
 * - Dataspace: HDF5's description of an array's shape (rank and extents), used both for the
 *   array in the file and for the memory buffer it is read into.
 * - Hyperslab: a rectangular sub-block of a dataset (start + count per dimension). Selecting a
 *   hyperslab in the file dataspace and reading it moves only that block from the file.
 * - Native memory type (H5T_NATIVE_INT64, H5T_NATIVE_DOUBLE, ...): the type of our C++ buffer.
 *   H5Dread / H5Aread convert from whatever type the file stores (int32, uint16, float32, ...) to
 *   this type, so the code does not have to handle every stored type.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it. Here: rank 0 reads, then broadcasts the result or its error message.
 *
 * HDF5 API used (signatures checked against builds/hdf5/install_cpu/include, HDF5 1.14.6):
 * H5Fopen/H5Fclose (H5Fpublic.h:485,636), H5Dopen2/H5Dget_type/H5Dread/H5Dclose
 * (H5Dpublic.h:406,513,907,1644), H5Dget_space, H5Sget_simple_extent_ndims/_dims/_npoints,
 * H5Sselect_hyperslab, H5Screate_simple, H5Sclose (H5Spublic.h:740,722,759,1235,341,182),
 * H5Aexists_by_name/H5Aopen_by_name/H5Aget_space/H5Aread/H5Aclose (H5Apublic.h:371,874,573,920,
 * 95), H5Lexists (H5Lpublic.h:659), H5Oget_info_by_name3/H5Ovisit3 (H5Opublic.h:543,1220),
 * H5Tget_class/H5Tclose (H5Tpublic.h:1828,1111), H5Eget_auto2/H5Eset_auto2
 * (H5Epublic.h:613,649), H5Pcreate/H5Pset_file_locking/H5Pclose (H5Ppublic.h:833,4903,639).
 */
#include "microstructure/io/voxel_map_io.hpp"

#include "microstructure/util/collective_errors.hpp"

#include "axom/config.hpp"

#ifdef AXOM_USE_HDF5
#include "hdf5.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief Largest id that fits the `int` element labels: 2^31 - 1. */
constexpr std::int64_t k_max_label = std::numeric_limits<int>::max();

/** @brief "x, y, z" text of three numbers, for messages. */
template <typename T>
std::string Triple(const std::array<T, 3>& v) {
    std::ostringstream out;
    out.precision(17);
    out << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return out.str();
}

/** @brief Message prefix naming the file. */
std::string Prefix(const VoxelMapParameters& params) {
    return "Voxel map '" + params.file.string() + "': ";
}

#ifdef AXOM_USE_HDF5

/** @brief Allowed deviation of a quaternion's norm from 1 (same rule as seed tables). */
constexpr double k_quat_norm_tolerance = 1.0e-3;

/**
 * @brief Owns one HDF5 handle and closes it with the matching close function when destroyed.
 * Move-only. A negative id (failed open) is not closed. Host only.
 */
class H5Handle {
public:
    H5Handle() = default;
    H5Handle(hid_t id, herr_t (*close)(hid_t)) : m_id(id), m_close(close) {}
    ~H5Handle() {
        Reset();
    }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept : m_id(other.m_id), m_close(other.m_close) {
        other.m_id = -1;
    }
    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) {
            Reset();
            m_id = other.m_id;
            m_close = other.m_close;
            other.m_id = -1;
        }
        return *this;
    }
    /** @brief The raw handle (negative if the open failed). */
    hid_t Get() const {
        return m_id;
    }
    /** @brief True when the open succeeded. */
    bool Valid() const {
        return m_id >= 0;
    }

private:
    void Reset() {
        if (m_id >= 0 && m_close != nullptr) {
            m_close(m_id);
        }
        m_id = -1;
    }
    hid_t m_id = -1;
    herr_t (*m_close)(hid_t) = nullptr;
};

/**
 * @brief Turns off HDF5's automatic printing of error stacks while it is alive, and restores the
 * previous setting afterwards. Our code reports its own messages, so a probe that fails (a
 * missing attribute, a file that is not HDF5) must not print a page of HDF5 internals. Host only.
 */
class QuietHdf5Errors {
public:
    QuietHdf5Errors() {
        H5Eget_auto2(H5E_DEFAULT, &m_func, &m_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    }
    ~QuietHdf5Errors() {
        H5Eset_auto2(H5E_DEFAULT, m_func, m_data);
    }
    QuietHdf5Errors(const QuietHdf5Errors&) = delete;
    QuietHdf5Errors& operator=(const QuietHdf5Errors&) = delete;

private:
    H5E_auto2_t m_func = nullptr;
    void* m_data = nullptr;
};

/**
 * @brief Opens the file read-only with HDF5 file locking turned off; throws with a clear message
 * if it is missing or not HDF5.
 *
 * HDF5 >= 1.10 locks files on open by default, and locking fails on some parallel file systems
 * (and is pointless for read-only access by many ranks). H5Pset_file_locking(fapl, false, true)
 * (H5Ppublic.h:4903) disables it for this open; the second flag ignores the request where
 * locking is already disabled.
 */
H5Handle OpenFile(const VoxelMapParameters& params) {
    H5Handle fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    if (!fapl.Valid() || H5Pset_file_locking(fapl.Get(), false, true) < 0) {
        throw std::runtime_error(Prefix(params) + "cannot set up the HDF5 file access properties");
    }
    H5Handle file(H5Fopen(params.file.string().c_str(), H5F_ACC_RDONLY, fapl.Get()), H5Fclose);
    if (!file.Valid()) {
        throw std::runtime_error(Prefix(params) +
                                 "cannot open the file read-only (missing, unreadable or not an "
                                 "HDF5 file; on file systems without lock support also try "
                                 "setting the environment variable HDF5_USE_FILE_LOCKING=FALSE)");
    }
    return file;
}

/**
 * @brief True if the absolute HDF5 path exists. H5Lexists fails (rather than returning false)
 * when an intermediate group is missing, so the path is checked one component at a time:
 * "/a", "/a/b", "/a/b/c".
 */
bool PathExists(hid_t file, const std::string& path) {
    if (path.empty() || path[0] != '/') {
        return false;
    }
    if (path == "/") {
        return true;
    }
    std::size_t pos = 1;
    while (pos <= path.size()) {
        const std::size_t next = path.find('/', pos);
        const std::string prefix = path.substr(0, next);
        if (H5Lexists(file, prefix.c_str(), H5P_DEFAULT) <= 0) {
            return false;
        }
        if (next == std::string::npos) {
            break;
        }
        pos = next + 1;
    }
    return true;
}

/** @brief Object type (group, dataset, ...) at an existing path; UNKNOWN on failure. */
H5O_type_t ObjectType(hid_t file, const std::string& path) {
    H5O_info2_t info;
    if (H5Oget_info_by_name3(file, path.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT) < 0) {
        return H5O_TYPE_UNKNOWN;
    }
    return info.type;
}

/** @brief True if `path` exists and is a dataset. */
bool IsDataset(hid_t file, const std::string& path) {
    return PathExists(file, path) && ObjectType(file, path) == H5O_TYPE_DATASET;
}

/** @brief True if `path` exists and is a group. */
bool IsGroup(hid_t file, const std::string& path) {
    return PathExists(file, path) && ObjectType(file, path) == H5O_TYPE_GROUP;
}

/** @brief Opens a dataset (which must exist); throws on failure. */
H5Handle OpenDataset(hid_t file, const std::string& path) {
    H5Handle dset(H5Dopen2(file, path.c_str(), H5P_DEFAULT), H5Dclose);
    if (!dset.Valid()) {
        throw std::runtime_error("cannot open dataset '" + path + "'");
    }
    return dset;
}

/** @brief Extents of a dataset, slowest dimension first (HDF5 / C order). */
std::vector<hsize_t> DatasetShape(hid_t file, const std::string& path) {
    H5Handle dset = OpenDataset(file, path);
    H5Handle space(H5Dget_space(dset.Get()), H5Sclose);
    const int ndims = H5Sget_simple_extent_ndims(space.Get());
    if (ndims < 0) {
        throw std::runtime_error("cannot read the shape of dataset '" + path + "'");
    }
    std::vector<hsize_t> dims(static_cast<std::size_t>(ndims), 0);
    if (ndims > 0) {
        H5Sget_simple_extent_dims(space.Get(), dims.data(), nullptr);
    }
    return dims;
}

/** @brief "(d0, d1, ...)" text of a shape, for messages. */
std::string ShapeText(const std::vector<hsize_t>& shape) {
    std::ostringstream out;
    out << "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << ")";
    return out.str();
}

/** @brief True if the dataset stores an integer type (any size, signed or unsigned). */
bool IsIntegerDataset(hid_t file, const std::string& path) {
    H5Handle dset = OpenDataset(file, path);
    H5Handle type(H5Dget_type(dset.Get()), H5Tclose);
    return H5Tget_class(type.Get()) == H5T_INTEGER;
}

/**
 * @brief Reads a whole dataset into a vector of T, converting from the stored type to
 * `mem_type` (the native HDF5 type matching T). Returns the shape in `shape`.
 */
template <typename T>
std::vector<T> ReadWholeDataset(hid_t file, const std::string& path, hid_t mem_type,
                                std::vector<hsize_t>& shape) {
    shape = DatasetShape(file, path);
    std::size_t count = 1;
    for (hsize_t d : shape) {
        count *= static_cast<std::size_t>(d);
    }
    std::vector<T> data(count);
    if (count == 0) {
        return data;
    }
    H5Handle dset = OpenDataset(file, path);
    // H5S_ALL for both spaces: the whole dataset into a contiguous buffer of the same shape
    if (H5Dread(dset.Get(), mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
        throw std::runtime_error("cannot read dataset '" + path + "'");
    }
    return data;
}

/**
 * @brief Reads exactly three numbers (x, y, z) from a small dataset, converted to double or
 * int64 (`mem_type`). Throws if the dataset does not hold exactly 3 values.
 */
template <typename T>
std::array<T, 3> ReadTripleDataset(hid_t file, const std::string& path, hid_t mem_type) {
    std::vector<hsize_t> shape;
    const std::vector<T> data = ReadWholeDataset<T>(file, path, mem_type, shape);
    if (data.size() != 3) {
        throw std::runtime_error("dataset '" + path + "' must hold 3 values (x, y, z), it has " +
                                 std::to_string(data.size()));
    }
    return {data[0], data[1], data[2]};
}

/** @brief True if the object at `object_path` has an attribute `name`. */
bool HasAttribute(hid_t file, const std::string& object_path, const char* name) {
    return H5Aexists_by_name(file, object_path.c_str(), name, H5P_DEFAULT) > 0;
}

/**
 * @brief Reads an attribute holding exactly three numbers (x, y, z), converted to `mem_type`.
 */
template <typename T>
std::array<T, 3> ReadTripleAttribute(hid_t file, const std::string& object_path,
                                     const char* name, hid_t mem_type) {
    H5Handle attr(H5Aopen_by_name(file, object_path.c_str(), name, H5P_DEFAULT, H5P_DEFAULT),
                  H5Aclose);
    if (!attr.Valid()) {
        throw std::runtime_error("cannot open attribute '" + std::string(name) + "' of '" +
                                 object_path + "'");
    }
    H5Handle space(H5Aget_space(attr.Get()), H5Sclose);
    const hssize_t count = H5Sget_simple_extent_npoints(space.Get());
    if (count != 3) {
        throw std::runtime_error("attribute '" + std::string(name) + "' of '" + object_path +
                                 "' must hold 3 values (x, y, z)");
    }
    std::array<T, 3> out{};
    if (H5Aread(attr.Get(), mem_type, out.data()) < 0) {
        throw std::runtime_error("cannot read attribute '" + std::string(name) + "' of '" +
                                 object_path + "'");
    }
    return out;
}

/** @brief Data passed through H5Ovisit3 to NameSearchVisitor. */
struct NameSearch {
    std::string name;                ///< dataset name looked for (last path component)
    std::vector<std::string> found;  ///< absolute paths of the matches, in visiting order
};

/**
 * @brief H5Ovisit3 callback: records every dataset whose last path component equals the
 * searched name. `name` is the object's path relative to the visited group (the root here).
 * Returns 0 to continue the iteration.
 */
herr_t NameSearchVisitor(hid_t /*obj*/, const char* name, const H5O_info2_t* info, void* data) {
    auto* search = static_cast<NameSearch*>(data);
    if (info->type != H5O_TYPE_DATASET) {
        return 0;
    }
    const std::string path(name);
    const std::size_t slash = path.rfind('/');
    std::string base = path;
    if (slash != std::string::npos) {
        base = path.substr(slash + 1);
    }
    if (base == search->name) {
        search->found.push_back("/" + path);
    }
    return 0;
}

/**
 * @brief All datasets named `name` anywhere in the file (absolute paths, sorted). H5Ovisit3
 * visits each object once, in name order (H5_INDEX_NAME, H5_ITER_INC), so the list is the same
 * on every run.
 */
std::vector<std::string> FindDatasetsByName(hid_t file, const std::string& name) {
    NameSearch search;
    search.name = name;
    if (H5Ovisit3(file, H5_INDEX_NAME, H5_ITER_INC, NameSearchVisitor, &search, H5O_INFO_BASIC) <
        0) {
        throw std::runtime_error("the recursive search for datasets named '" + name + "' failed");
    }
    std::sort(search.found.begin(), search.found.end());
    return search.found;
}

/** @brief "'a', 'b'" list of paths, for messages. */
std::string PathList(const std::vector<std::string>& paths) {
    std::string out;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += "'" + paths[i] + "'";
    }
    return out;
}

/**
 * @brief Path of the per-voxel id dataset: the explicit override, `/grain_ids` (canonical), or
 * the unique dataset named `FeatureIds` (DREAM.3D name search). Throws when it is missing or,
 * for the search, ambiguous (listing the candidates and the override key).
 */
std::string ResolveIdPath(hid_t file, const VoxelMapParameters& params) {
    if (!params.feature_ids_path.empty() || params.format == VoxelFormat::EXACONSTIT) {
        std::string path = params.feature_ids_path;
        if (path.empty()) {
            path = "/grain_ids";
        }
        if (!IsDataset(file, path)) {
            throw std::runtime_error(Prefix(params) + "no dataset '" + path +
                                     "' (set feature_ids_path to the per-voxel grain id dataset)");
        }
        return path;
    }
    const std::vector<std::string> found = FindDatasetsByName(file, "FeatureIds");
    if (found.empty()) {
        throw std::runtime_error(Prefix(params) +
                                 "no dataset named 'FeatureIds' in the file; set "
                                 "feature_ids_path to the per-voxel grain id dataset");
    }
    if (found.size() > 1) {
        throw std::runtime_error(Prefix(params) + "several datasets named 'FeatureIds': " +
                                 PathList(found) + "; choose one with feature_ids_path");
    }
    return found[0];
}

/**
 * @brief Voxel counts (nx, ny, nz) from the id dataset's shape: (nz, ny, nx) or (nz, ny, nx, 1).
 * Throws for any other shape or a zero extent.
 */
std::array<std::int64_t, 3> DimsFromShape(const VoxelMapParameters& params,
                                          const std::string& path,
                                          const std::vector<hsize_t>& shape) {
    const bool rank3 = shape.size() == 3;
    const bool rank4 = shape.size() == 4 && shape[3] == 1;
    if (!rank3 && !rank4) {
        throw std::runtime_error(Prefix(params) + "dataset '" + path + "' has shape " +
                                 ShapeText(shape) +
                                 "; expected (nz, ny, nx) or (nz, ny, nx, 1)");
    }
    const std::array<std::int64_t, 3> dims = {static_cast<std::int64_t>(shape[2]),
                                              static_cast<std::int64_t>(shape[1]),
                                              static_cast<std::int64_t>(shape[0])};
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        throw std::runtime_error(Prefix(params) + "dataset '" + path + "' has shape " +
                                 ShapeText(shape) + "; every voxel count must be > 0");
    }
    return dims;
}

/**
 * @brief Reads the voxel ids of the box [lo, hi) (x y z order, 0-based, non-empty) of the id
 * dataset into `out` (x fastest; room for the box's voxel count), with one hyperslab read.
 *
 * The file dataset is (nz, ny, nx) or (nz, ny, nx, 1), so the hyperslab start and count are
 * given slowest first: (lo_z, lo_y, lo_x[, 0]) and (count_z, count_y, count_x[, 1]). The memory
 * dataspace is a contiguous array of the same counts, which in C order puts x fastest, the
 * layout VoxelBlock documents.
 */
void ReadIdBox(hid_t file, const std::string& path, const std::array<std::int64_t, 3>& lo,
               const std::array<std::int64_t, 3>& hi, std::int64_t* out) {
    const std::int64_t cx = hi[0] - lo[0];
    const std::int64_t cy = hi[1] - lo[1];
    const std::int64_t cz = hi[2] - lo[2];
    H5Handle dset = OpenDataset(file, path);
    H5Handle file_space(H5Dget_space(dset.Get()), H5Sclose);
    const int ndims = H5Sget_simple_extent_ndims(file_space.Get());
    if (ndims != 3 && ndims != 4) {
        throw std::runtime_error("dataset '" + path + "' is not 3D or 4D");
    }
    // Start and count, slowest dimension first (z, y, x[, component])
    hsize_t start[4] = {static_cast<hsize_t>(lo[2]), static_cast<hsize_t>(lo[1]),
                        static_cast<hsize_t>(lo[0]), 0};
    hsize_t count[4] = {static_cast<hsize_t>(cz), static_cast<hsize_t>(cy),
                        static_cast<hsize_t>(cx), 1};
    if (H5Sselect_hyperslab(file_space.Get(), H5S_SELECT_SET, start, nullptr, count, nullptr) <
        0) {
        throw std::runtime_error("cannot select a voxel block of dataset '" + path + "'");
    }
    H5Handle mem_space(H5Screate_simple(ndims, count, nullptr), H5Sclose);
    if (H5Dread(dset.Get(), H5T_NATIVE_INT64, mem_space.Get(), file_space.Get(), H5P_DEFAULT,
                out) < 0) {
        throw std::runtime_error("cannot read a voxel block of dataset '" + path + "'");
    }
}

/** @brief Geometry found in a DREAM.3D file (all members unset when nothing was found). */
struct FileGeometry {
    bool found = false;      ///< a geometry group / attributes were found
    std::string group;       ///< the group carrying the geometry (the data container)
    std::string where;       ///< description for messages
    bool has_dims = false;   ///< the file declares voxel counts
    std::array<std::int64_t, 3> dims = {0, 0, 0};
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    std::array<double, 3> spacing = {0.0, 0.0, 0.0};
};

/** @brief Parent group of an absolute path ("/a/b/c" -> "/a/b", "/a" -> "/"). */
std::string ParentPath(const std::string& path) {
    const std::size_t slash = path.rfind('/');
    if (slash == std::string::npos || slash == 0) {
        return "/";
    }
    return path.substr(0, slash);
}

/** @brief `group` + "/" + `name`, without a doubled slash at the root. */
std::string JoinPath(const std::string& group, const std::string& name) {
    if (group == "/") {
        return "/" + name;
    }
    return group + "/" + name;
}

/**
 * @brief DREAM.3D geometry: walks from the group holding the id dataset up to the root and
 * stops at the first group that has
 *  - a `_SIMPL_GEOMETRY` child group with ORIGIN and SPACING datasets (SIMPL v6; DIMENSIONS
 *    optional), or
 *  - both `_ORIGIN` and `_SPACING` attributes (DREAM3D-NX image geometry, simplnx IOConstants;
 *    `_DIMENSIONS` optional).
 * Values are converted to double / int64 by HDF5 (the files store float32 and int64/size_t).
 */
FileGeometry FindDream3dGeometry(hid_t file, const std::string& ids_path) {
    FileGeometry geometry;
    std::string group = ParentPath(ids_path);
    while (true) {
        // SIMPL v6: <data container>/_SIMPL_GEOMETRY/{DIMENSIONS, ORIGIN, SPACING}
        const std::string simpl = JoinPath(group, "_SIMPL_GEOMETRY");
        if (IsGroup(file, simpl) && IsDataset(file, simpl + "/ORIGIN") &&
            IsDataset(file, simpl + "/SPACING")) {
            geometry.found = true;
            geometry.group = group;
            geometry.where = "group '" + simpl + "'";
            geometry.origin = ReadTripleDataset<double>(file, simpl + "/ORIGIN",
                                                        H5T_NATIVE_DOUBLE);
            geometry.spacing = ReadTripleDataset<double>(file, simpl + "/SPACING",
                                                         H5T_NATIVE_DOUBLE);
            if (IsDataset(file, simpl + "/DIMENSIONS")) {
                geometry.has_dims = true;
                geometry.dims = ReadTripleDataset<std::int64_t>(file, simpl + "/DIMENSIONS",
                                                                H5T_NATIVE_INT64);
            }
            return geometry;
        }
        // DREAM3D-NX: attributes of the image geometry group
        if (HasAttribute(file, group, "_ORIGIN") && HasAttribute(file, group, "_SPACING")) {
            geometry.found = true;
            geometry.group = group;
            geometry.where = "attributes of '" + group + "'";
            geometry.origin = ReadTripleAttribute<double>(file, group, "_ORIGIN",
                                                          H5T_NATIVE_DOUBLE);
            geometry.spacing = ReadTripleAttribute<double>(file, group, "_SPACING",
                                                           H5T_NATIVE_DOUBLE);
            if (HasAttribute(file, group, "_DIMENSIONS")) {
                geometry.has_dims = true;
                geometry.dims = ReadTripleAttribute<std::int64_t>(file, group, "_DIMENSIONS",
                                                                  H5T_NATIVE_INT64);
            }
            return geometry;
        }
        if (group == "/") {
            return geometry;
        }
        group = ParentPath(group);
    }
}

/**
 * @brief Rank 0's part of ReadVoxelGridInfo (see the header for the steps). Throws
 * std::runtime_error with a complete message on any failure.
 */
VoxelGridInfo ReadGridInfoOnRoot(const VoxelMapParameters& params) {
    QuietHdf5Errors quiet;
    if (!(params.units_scale > 0.0) || !std::isfinite(params.units_scale)) {
        throw std::runtime_error(Prefix(params) + "units_scale must be a finite number > 0");
    }
    H5Handle file = OpenFile(params);

    // Step 1: the id dataset and the voxel counts from its shape
    VoxelGridInfo info;
    info.feature_ids_path = ResolveIdPath(file.Get(), params);
    if (!IsIntegerDataset(file.Get(), info.feature_ids_path)) {
        throw std::runtime_error(Prefix(params) + "dataset '" + info.feature_ids_path +
                                 "' must store integers (grain ids)");
    }
    const std::vector<hsize_t> shape = DatasetShape(file.Get(), info.feature_ids_path);
    info.dims = DimsFromShape(params, info.feature_ids_path, shape);

    // Step 2: origin and spacing in file units: the file's values, replaced by TOML values
    bool have_origin = false;
    bool have_spacing = false;
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    std::array<double, 3> spacing = {0.0, 0.0, 0.0};
    std::string where = "the file";
    if (params.format == VoxelFormat::EXACONSTIT) {
        if (!params.has_origin && IsDataset(file.Get(), "/origin")) {
            origin = ReadTripleDataset<double>(file.Get(), "/origin", H5T_NATIVE_DOUBLE);
            have_origin = true;
        }
        if (!params.has_spacing && IsDataset(file.Get(), "/spacing")) {
            spacing = ReadTripleDataset<double>(file.Get(), "/spacing", H5T_NATIVE_DOUBLE);
            have_spacing = true;
        }
        where = "datasets '/origin' and '/spacing'";
    } else {
        const FileGeometry geometry = FindDream3dGeometry(file.Get(), info.feature_ids_path);
        if (geometry.found) {
            where = geometry.where;
            origin = geometry.origin;
            spacing = geometry.spacing;
            have_origin = true;
            have_spacing = true;
            // Always cross-check the declared voxel counts against the FeatureIds shape
            if (geometry.has_dims && geometry.dims != info.dims) {
                throw std::runtime_error(Prefix(params) + "the dimensions in " + where + " " +
                                         Triple(geometry.dims) + " differ from the shape of '" +
                                         info.feature_ids_path + "' " + ShapeText(shape) +
                                         " (read as nx, ny, nz = " + Triple(info.dims) + ")");
            }
        }
    }
    if (params.has_origin) {
        origin = params.origin;
        have_origin = true;
    }
    if (params.has_spacing) {
        spacing = params.spacing;
        have_spacing = true;
    }
    if (!have_origin || !have_spacing) {
        throw std::runtime_error(Prefix(params) +
                                 "no voxel origin/spacing found in the file (looked for " +
                                 where + "); give origin and spacing in the options");
    }
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(origin[d])) {
            throw std::runtime_error(Prefix(params) + "the origin " + Triple(origin) +
                                     " is not finite");
        }
        if (!(spacing[d] > 0.0) || !std::isfinite(spacing[d])) {
            throw std::runtime_error(Prefix(params) + "the spacing " + Triple(spacing) +
                                     " must be finite and > 0 on every axis");
        }
    }

    // Step 3: file units -> mesh units (x_mesh = units_scale * x_file + translate). The scaled
    // spacing is checked again: a tiny units_scale can underflow it to 0.
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(params.translate[d])) {
            throw std::runtime_error(Prefix(params) + "translate " + Triple(params.translate) +
                                     " is not finite");
        }
        info.origin[d] = params.units_scale * origin[d] + params.translate[d];
        info.spacing[d] = params.units_scale * spacing[d];
        if (!(info.spacing[d] > 0.0) || !std::isfinite(info.spacing[d]) ||
            !std::isfinite(info.origin[d])) {
            throw std::runtime_error(Prefix(params) + "after units_scale and translate the "
                                     "origin " + Triple(info.origin) + " and spacing " +
                                     Triple(info.spacing) +
                                     " must be finite with spacing > 0 (units_scale too small "
                                     "or too large?)");
        }
    }
    return info;
}

/**
 * @brief The data container of the voxel id dataset: the group carrying the DREAM.3D geometry
 * (FindDream3dGeometry), or, without geometry, the grandparent group of the id dataset
 * (DC/CellData/FeatureIds -> DC; root when the path is shorter).
 */
std::string DataContainerOf(hid_t file, const std::string& ids_path) {
    const FileGeometry geometry = FindDream3dGeometry(file, ids_path);
    if (geometry.found) {
        return geometry.group;
    }
    return ParentPath(ParentPath(ids_path));
}

/** @brief True if `path` lies inside group `group` (any depth). */
bool IsInside(const std::string& path, const std::string& group) {
    if (group == "/") {
        return true;
    }
    return path.size() > group.size() && path.compare(0, group.size(), group) == 0 &&
           path[group.size()] == '/';
}

/**
 * @brief Path of a per-feature DREAM.3D array: the override, or the unique dataset named `name`
 * inside the data container `container` of the voxel ids that does not have the per-voxel shape
 * (cell arrays such as a per-voxel `Phases` are skipped).
 *
 * Candidates in other containers are never used: if the container has none, the error lists
 * them (so the user can pick one explicitly with the override key), but they are not chosen
 * silently, because another container's table belongs to another voxel grid.
 */
std::string ResolveDream3dFeaturePath(hid_t file, const VoxelMapParameters& params,
                                      const std::string& override_path, const std::string& name,
                                      const std::string& key, const std::string& container,
                                      const std::vector<hsize_t>& voxel_shape) {
    if (!override_path.empty()) {
        if (!IsDataset(file, override_path)) {
            throw std::runtime_error(Prefix(params) + "no dataset '" + override_path + "' (" +
                                     key + ")");
        }
        return override_path;
    }
    const std::vector<std::string> all = FindDatasetsByName(file, name);
    std::vector<std::string> found;     // in the container, per-feature shape
    std::vector<std::string> skipped;   // in the container, per-voxel shape
    std::vector<std::string> elsewhere; // in other containers
    for (const std::string& path : all) {
        if (!IsInside(path, container)) {
            elsewhere.push_back(path);
            continue;
        }
        // A per-voxel array has the id dataset's leading (nz, ny, nx) extents
        const std::vector<hsize_t> shape = DatasetShape(file, path);
        const bool per_voxel = shape.size() >= 3 && shape[0] == voxel_shape[0] &&
                               shape[1] == voxel_shape[1] && shape[2] == voxel_shape[2];
        if (per_voxel) {
            skipped.push_back(path);
        } else {
            found.push_back(path);
        }
    }
    if (found.empty()) {
        std::string message = Prefix(params) + "no per-feature dataset named '" + name +
                              "' in the data container '" + container + "' of the voxel ids";
        if (!skipped.empty()) {
            message += " (skipped per-voxel datasets " + PathList(skipped) + ")";
        }
        if (!elsewhere.empty()) {
            message += "; candidates in other containers (not used automatically): " +
                       PathList(elsewhere);
        }
        throw std::runtime_error(message + "; set " + key);
    }
    if (found.size() > 1) {
        throw std::runtime_error(Prefix(params) + "several per-feature datasets named '" + name +
                                 "': " + PathList(found) + "; choose one with " + key);
    }
    return found[0];
}

/**
 * @brief Number of rows of a per-feature table with `columns` values per row: shape (rows) or
 * (rows, 1) for one column, (rows, columns) otherwise. Throws for another shape.
 */
std::size_t TableRows(const VoxelMapParameters& params, const std::string& path,
                      const std::vector<hsize_t>& shape, hsize_t columns) {
    const bool ok = (shape.size() == 2 && shape[1] == columns) ||
                    (columns == 1 && shape.size() == 1);
    if (!ok) {
        throw std::runtime_error(Prefix(params) + "dataset '" + path + "' has shape " +
                                 ShapeText(shape) + "; expected (rows, " +
                                 std::to_string(columns) + ")");
    }
    return static_cast<std::size_t>(shape[0]);
}

/** @brief Throws unless | |q| - 1 | <= 1e-3 for the quaternion q[0..3] of feature `id`. */
void CheckQuatNorm(const VoxelMapParameters& params, std::int64_t id, const double* q) {
    const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(std::fabs(norm - 1.0) <= k_quat_norm_tolerance)) {
        std::ostringstream out;
        out << Prefix(params) << "feature id " << id << ": quaternion norm " << norm
            << " differs from 1 by more than " << k_quat_norm_tolerance
            << " (wrong dataset, wrong column order or not a quaternion?)";
        throw std::runtime_error(out.str());
    }
}

/**
 * @brief Rank 0's part of ReadVoxelFeatures (see the header). Throws std::runtime_error with a
 * complete message on any failure.
 *
 * Steps: declarations and `used_ids` checks; read the table (canonical: explicit ids, sorted by
 * id; DREAM.3D: row = id, rows 1..R-1, datasets searched in the voxel ids' data container);
 * check every used id has a row; reorder/conjugate DREAM.3D quaternions; norm checks. Nothing is
 * allocated per possible id value: membership tests are binary searches of sorted lists (a
 * binary search finds a value in a sorted list by repeatedly halving the range that can hold
 * it).
 */
VoxelFeatureTable ReadFeaturesOnRoot(const VoxelMapParameters& params,
                                     const VoxelGridInfo& grid,
                                     const std::vector<std::int64_t>& used_ids) {
    QuietHdf5Errors quiet;
    for (std::size_t i = 0; i < used_ids.size(); ++i) {
        if (used_ids[i] < 1 || (i > 0 && used_ids[i] <= used_ids[i - 1])) {
            throw std::runtime_error(Prefix(params) + "the list of used feature ids must be "
                                     "strictly increasing with ids >= 1 (pass grain_feature from "
                                     "CompactFeatureLabels)");
        }
    }
    // Step 1: storage declarations
    if (params.format == VoxelFormat::EXACONSTIT) {
        if (params.quat_layout == QuatLayout::XYZW || params.conjugate == 1) {
            throw std::runtime_error(Prefix(params) +
                                     "the canonical format stores w x y z crystal-to-sample "
                                     "quaternions; quat_layout = \"xyzw\" and conjugate = true "
                                     "do not apply to it");
        }
    } else if (params.read_quats) {
        if (params.quat_layout == QuatLayout::UNSET) {
            throw std::runtime_error(Prefix(params) +
                                     "reading DREAM.3D quaternions requires quat_layout "
                                     "(\"wxyz\" or \"xyzw\": the stored order)");
        }
        if (params.conjugate != 0 && params.conjugate != 1) {
            throw std::runtime_error(Prefix(params) +
                                     "reading DREAM.3D quaternions requires conjugate (true or "
                                     "false: whether the stored quaternion must be conjugated)");
        }
    }

    H5Handle file = OpenFile(params);

    // Step 2: shape of the voxel ids (to skip per-voxel arrays in the DREAM.3D search)
    const std::vector<hsize_t> voxel_shape = DatasetShape(file.Get(), grid.feature_ids_path);

    VoxelFeatureTable table;
    if (params.format == VoxelFormat::EXACONSTIT) {
        // Step 3a (canonical): explicit ids, then the requested columns
        const std::string id_path = "/features/id";
        if (!IsDataset(file.Get(), id_path)) {
            throw std::runtime_error(Prefix(params) +
                                     "per-grain data requested but there is no '/features/id'");
        }
        std::vector<hsize_t> shape;
        const std::vector<std::int64_t> ids =
            ReadWholeDataset<std::int64_t>(file.Get(), id_path, H5T_NATIVE_INT64, shape);
        const std::size_t rows = TableRows(params, id_path, shape, 1);

        std::vector<double> quat;
        if (params.read_quats) {
            const std::string path =
                params.quats_path.empty() ? std::string("/features/quat") : params.quats_path;
            if (!IsDataset(file.Get(), path)) {
                throw std::runtime_error(Prefix(params) + "no quaternion dataset '" + path + "'");
            }
            quat = ReadWholeDataset<double>(file.Get(), path, H5T_NATIVE_DOUBLE, shape);
            if (TableRows(params, path, shape, 4) != rows) {
                throw std::runtime_error(Prefix(params) + "'" + path + "' and '" + id_path +
                                         "' have different row counts");
            }
        }
        std::vector<int> phase;
        if (params.read_phases) {
            const std::string path =
                params.phases_path.empty() ? std::string("/features/phase") : params.phases_path;
            if (!IsDataset(file.Get(), path)) {
                throw std::runtime_error(Prefix(params) + "no phase dataset '" + path + "'");
            }
            phase = ReadWholeDataset<int>(file.Get(), path, H5T_NATIVE_INT, shape);
            if (TableRows(params, path, shape, 1) != rows) {
                throw std::runtime_error(Prefix(params) + "'" + path + "' and '" + id_path +
                                         "' have different row counts");
            }
        }

        // Sort rows by id; ids must be >= 1 and unique
        std::vector<std::size_t> order(rows);
        std::iota(order.begin(), order.end(), std::size_t(0));
        std::sort(order.begin(), order.end(),
                  [&ids](std::size_t a, std::size_t b) { return ids[a] < ids[b]; });
        for (std::size_t r = 0; r < rows; ++r) {
            const std::int64_t id = ids[order[r]];
            if (id < 1) {
                throw std::runtime_error(Prefix(params) + "'/features/id' holds the id " +
                                         std::to_string(id) + "; ids must be >= 1");
            }
            if (r > 0 && id == ids[order[r - 1]]) {
                throw std::runtime_error(Prefix(params) + "'/features/id' holds the id " +
                                         std::to_string(id) + " more than once");
            }
            table.id.push_back(id);
            if (params.read_quats) {
                const double* q = &quat[4 * order[r]];
                CheckQuatNorm(params, id, q);
                table.quat.insert(table.quat.end(), q, q + 4); // verbatim, w x y z
            }
            if (params.read_phases) {
                table.phase.push_back(phase[order[r]]);
            }
        }
    } else {
        // Step 3b (DREAM.3D): rows indexed by feature id, row 0 unused
        std::size_t rows = 0;
        std::vector<double> stored_quat;
        std::vector<int> phase;
        std::vector<hsize_t> shape;
        const std::string container = DataContainerOf(file.Get(), grid.feature_ids_path);
        if (params.read_quats) {
            const std::string path =
                ResolveDream3dFeaturePath(file.Get(), params, params.quats_path, "AvgQuats",
                                          "quats_path", container, voxel_shape);
            stored_quat = ReadWholeDataset<double>(file.Get(), path, H5T_NATIVE_DOUBLE, shape);
            rows = TableRows(params, path, shape, 4);
        }
        if (params.read_phases) {
            const std::string path =
                ResolveDream3dFeaturePath(file.Get(), params, params.phases_path, "Phases",
                                          "phases_path", container, voxel_shape);
            phase = ReadWholeDataset<int>(file.Get(), path, H5T_NATIVE_INT, shape);
            const std::size_t phase_rows = TableRows(params, path, shape, 1);
            if (params.read_quats && phase_rows != rows) {
                throw std::runtime_error(Prefix(params) + "the quaternion and phase tables have "
                                         "different row counts (" + std::to_string(rows) +
                                         " and " + std::to_string(phase_rows) + ")");
            }
            rows = phase_rows;
        }
        // Every used id needs a row: the largest used id < rows
        if (!used_ids.empty() && used_ids.back() >= static_cast<std::int64_t>(rows)) {
            throw std::runtime_error(Prefix(params) + "the mesh uses feature id " +
                                     std::to_string(used_ids.back()) +
                                     " but the feature table has only rows 0.." +
                                     std::to_string(rows) + " - 1");
        }
        for (std::size_t r = 1; r < rows; ++r) {
            const auto id = static_cast<std::int64_t>(r);
            table.id.push_back(id);
            if (params.read_quats) {
                // Reorder the stored numbers to w x y z, then conjugate if declared
                const double* s = &stored_quat[4 * r];
                double q[4];
                if (params.quat_layout == QuatLayout::WXYZ) {
                    q[0] = s[0];
                    q[1] = s[1];
                    q[2] = s[2];
                    q[3] = s[3];
                } else {
                    q[0] = s[3];
                    q[1] = s[0];
                    q[2] = s[1];
                    q[3] = s[2];
                }
                if (params.conjugate == 1) {
                    q[1] = -q[1];
                    q[2] = -q[2];
                    q[3] = -q[3];
                }
                // Unused rows may hold zeros; only rows of used ids must be unit quaternions
                if (std::binary_search(used_ids.begin(), used_ids.end(), id)) {
                    CheckQuatNorm(params, id, q);
                }
                table.quat.insert(table.quat.end(), q, q + 4);
            }
            if (params.read_phases) {
                table.phase.push_back(phase[r]);
            }
        }
    }

    // Step 4: every used id has a row (the table is sorted by id)
    for (const std::int64_t id : used_ids) {
        if (!std::binary_search(table.id.begin(), table.id.end(), id)) {
            throw std::runtime_error(Prefix(params) + "grain id " + std::to_string(id) +
                                     " is used by the mesh but has no row in the feature table");
        }
    }
    return table;
}

#endif // AXOM_USE_HDF5

#ifndef AXOM_USE_HDF5
/** @brief The error every reader throws when HDF5 support is missing. */
[[noreturn]] void ThrowNoHdf5(const VoxelMapParameters& params) {
    throw std::runtime_error(Prefix(params) +
                             "this build of the microstructure generator was built without "
                             "HDF5 (Axom without HDF5), so voxel maps cannot be read");
}
#endif

/** @brief Broadcasts a string from rank 0 (wrapper so the call sites read clearly). */
std::string BcastString(const std::string& text, MPI_Comm comm) {
    return BroadcastString(text, 0, comm);
}

} // namespace

/*
 * ReadVoxelGridInfo
 *
 * Rank 0 does all the reading (ReadGridInfoOnRoot) inside a try block and keeps any error
 * message; ThrowIfAnyRankFailed then makes every rank throw rank 0's message, so no rank is left
 * waiting in the broadcasts. On success the small result is broadcast field by field (MPI_Bcast
 * of 3 int64 counts, 6 doubles, and the dataset path string).
 */
VoxelGridInfo ReadVoxelGridInfo(const VoxelMapParameters& params, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // Step 1: rank 0 reads, catching every error as a message
    VoxelGridInfo info;
    std::string error;
    if (rank == 0) {
        try {
#ifdef AXOM_USE_HDF5
            info = ReadGridInfoOnRoot(params);
#else
            ThrowNoHdf5(params);
#endif
        } catch (const std::exception& e) {
            error = e.what();
            if (error.empty()) {
                error = Prefix(params) + "unknown error";
            }
        }
    }

    // Step 2: all ranks throw together if rank 0 failed
    ThrowIfAnyRankFailed(comm, error, false);

    // Step 3: broadcast the result
    std::int64_t counts[3] = {info.dims[0], info.dims[1], info.dims[2]};
    MPI_Bcast(counts, 3, MPI_INT64_T, 0, comm);
    double geometry[6] = {info.origin[0],  info.origin[1],  info.origin[2],
                          info.spacing[0], info.spacing[1], info.spacing[2]};
    MPI_Bcast(geometry, 6, MPI_DOUBLE, 0, comm);
    info.feature_ids_path = BcastString(info.feature_ids_path, comm);
    for (int d = 0; d < 3; ++d) {
        info.dims[d] = counts[d];
        info.origin[d] = geometry[d];
        info.spacing[d] = geometry[3 + d];
    }
    return info;
}

/*
 * ReadVoxelBoxes
 *
 * Checks every box against the grid and sums the voxel counts; then (unless there is nothing to
 * read) opens the file once and reads each non-empty box with one hyperslab read (ReadIdBox)
 * straight into its slot of the result. Not collective: every rank opens the file on its own,
 * which serial HDF5 allows for read-only access.
 */
std::vector<std::int64_t> ReadVoxelBoxes(const VoxelMapParameters& params,
                                         const VoxelGridInfo& grid,
                                         const std::vector<VoxelBox>& boxes) {
    // Step 1: every box inside the grid; total size
    std::int64_t total = 0;
    for (const VoxelBox& box : boxes) {
        std::int64_t count = 1;
        for (int d = 0; d < 3; ++d) {
            if (box.lo[d] < 0 || box.hi[d] < box.lo[d] || box.hi[d] > grid.dims[d]) {
                throw std::invalid_argument(Prefix(params) + "voxel box " + Triple(box.lo) +
                                            " .. " + Triple(box.hi) + " is outside the grid " +
                                            Triple(grid.dims));
            }
            count *= box.hi[d] - box.lo[d];
        }
        total += count;
    }
    std::vector<std::int64_t> ids(static_cast<std::size_t>(total), 0);
    if (total == 0) {
        return ids;
    }
    // Step 2: one hyperslab read per non-empty box, into consecutive slots
#ifdef AXOM_USE_HDF5
    QuietHdf5Errors quiet;
    H5Handle file = OpenFile(params);
    std::size_t offset = 0;
    for (const VoxelBox& box : boxes) {
        const auto count = static_cast<std::size_t>((box.hi[0] - box.lo[0]) *
                                                    (box.hi[1] - box.lo[1]) *
                                                    (box.hi[2] - box.lo[2]));
        if (count == 0) {
            continue;
        }
        try {
            ReadIdBox(file.Get(), grid.feature_ids_path, box.lo, box.hi, ids.data() + offset);
        } catch (const std::exception& e) {
            throw std::runtime_error(Prefix(params) + e.what());
        }
        offset += count;
    }
#else
    ThrowNoHdf5(params);
#endif
    return ids;
}

/*
 * ReadVoxelBlock: one box through ReadVoxelBoxes, packaged with its description.
 */
VoxelBlock ReadVoxelBlock(const VoxelMapParameters& params,
                          const VoxelGridInfo& grid,
                          const std::array<std::int64_t, 3>& lo,
                          const std::array<std::int64_t, 3>& hi) {
    VoxelBlock block;
    block.grid = grid;
    block.lo = lo;
    block.hi = hi;
    VoxelBox box;
    box.lo = lo;
    box.hi = hi;
    block.ids = ReadVoxelBoxes(params, grid, {box});
    return block;
}

/*
 * ReadVoxelFeatures
 *
 * Same pattern as ReadVoxelGridInfo: rank 0 reads and checks (ReadFeaturesOnRoot), errors become
 * one collective exception, and the table is broadcast as three arrays (sizes first, then the
 * int64 ids, the doubles and the ints). With nothing requested, every rank returns an empty
 * table without communication (the parameters are the same on every rank).
 */
VoxelFeatureTable ReadVoxelFeatures(const VoxelMapParameters& params,
                                    const VoxelGridInfo& grid,
                                    const std::vector<std::int64_t>& used_ids,
                                    MPI_Comm comm) {
    VoxelFeatureTable table;
    if (!params.read_quats && !params.read_phases) {
        return table;
    }
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // Step 1: rank 0 reads
    std::string error;
    if (rank == 0) {
        try {
#ifdef AXOM_USE_HDF5
            table = ReadFeaturesOnRoot(params, grid, used_ids);
#else
            ThrowNoHdf5(params);
#endif
        } catch (const std::exception& e) {
            error = e.what();
            if (error.empty()) {
                error = Prefix(params) + "unknown error";
            }
        }
    }

    // Step 2: collective error
    ThrowIfAnyRankFailed(comm, error, false);

    // Step 3: broadcast sizes, then the arrays
    std::int64_t sizes[3] = {static_cast<std::int64_t>(table.id.size()),
                             static_cast<std::int64_t>(table.quat.size()),
                             static_cast<std::int64_t>(table.phase.size())};
    MPI_Bcast(sizes, 3, MPI_INT64_T, 0, comm);
    // MPI counts are int: every array must fit (the same sizes on every rank, so all ranks throw
    // together)
    if (sizes[0] > k_max_label || sizes[1] > k_max_label || sizes[2] > k_max_label) {
        throw std::runtime_error(Prefix(params) + "the feature table is too large to broadcast");
    }
    table.id.resize(static_cast<std::size_t>(sizes[0]));
    table.quat.resize(static_cast<std::size_t>(sizes[1]));
    table.phase.resize(static_cast<std::size_t>(sizes[2]));
    if (sizes[0] > 0) {
        MPI_Bcast(table.id.data(), static_cast<int>(sizes[0]), MPI_INT64_T, 0, comm);
    }
    if (sizes[1] > 0) {
        MPI_Bcast(table.quat.data(), static_cast<int>(sizes[1]), MPI_DOUBLE, 0, comm);
    }
    if (sizes[2] > 0) {
        MPI_Bcast(table.phase.data(), static_cast<int>(sizes[2]), MPI_INT, 0, comm);
    }
    return table;
}

} // namespace exaconstit::microstructure
