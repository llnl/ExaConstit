#pragma once
/**
 * @file voxel_fixture.hpp
 * @brief Shared test fixture: a small voxelized Voronoi microstructure written as HDF5 voxel maps
 * in the canonical "ExaConstit voxel map v1" layout and the DREAM.3D (SIMPL v6 and NX-like)
 * layouts. Used by test_voxel_map (reader and labeler) and test_driver_stage1d (the driver's
 * voxel-map path).
 *
 * **The map.** A 20 x 16 x 12 voxel grid with voxel size h = 1/16 and origin 0 (so the box is
 * [0, 1.25] x [0, 1] x [0, 0.75], and every voxel boundary is exactly representable in floating
 * point). Every voxel carries the feature id of the seed nearest to the voxel center (a Voronoi
 * tessellation of 6 seeds, computed here), with feature ids {3, 5, 8, 13, 21, 34} (not 1..6, to
 * exercise the compaction), and a 2 x 2 x 2 hole of id 0 placed inside one grain (where the
 * surrounding 4 x 4 x 4 voxels all have the same id). Feature r has the unit quaternion
 * FeatureQuat(r) and the phase FeaturePhase(r) = r % 3.
 *
 * Everything here is host code for tests; the writers use the HDF5 C API and are meant to be
 * called on one rank (rank 0) only. Only available when Axom was built with HDF5
 * (AXOM_USE_HDF5).
 */
#include "axom/config.hpp"

#ifdef AXOM_USE_HDF5

#include "hdf5.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxel_fixture {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------
// The voxelized tessellation
// ---------------------------------------------------------------------------------------------

inline constexpr int NX = 20;
inline constexpr int NY = 16;
inline constexpr int NZ = 12;
inline constexpr double H = 0.0625; // voxel size 1/16 (exact in binary)
inline constexpr int NUM_SEEDS = 6;
/** Feature id of seed i (deliberately not 1..6). */
inline const std::array<int, NUM_SEEDS> kFeatureId = {3, 5, 8, 13, 21, 34};
/** The feature ids the mesh uses (CompactFeatureLabels' grain_feature for the full map). */
inline const std::vector<std::int64_t> kUsedIds = {3, 5, 8, 13, 21, 34};
/** Seeds inside the box [0, 1.25] x [0, 1] x [0, 0.75]; generic positions (no exact ties). */
inline const std::array<std::array<double, 3>, NUM_SEEDS> kSeeds = {{{0.23, 0.21, 0.17},
                                                              {0.97, 0.26, 0.21},
                                                              {0.31, 0.77, 0.52},
                                                              {1.01, 0.79, 0.55},
                                                              {0.62, 0.48, 0.36},
                                                              {0.58, 0.13, 0.61}}};

/** Index of the seed nearest to p (brute force; ties go to the lowest index). */
inline int NearestSeed(const std::array<double, 3>& p) {
    int best = 0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < NUM_SEEDS; ++i) {
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double t = p[d] - kSeeds[static_cast<std::size_t>(i)][d];
            d2 += t * t;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

/**
 * Distance from p to the nearest Voronoi bisector of its own cell: for the nearest seed a and
 * every other seed b, the distance to the bisector plane is (|p-b|^2 - |p-a|^2) / (2 |a-b|).
 */
inline double DistanceToBisector(const std::array<double, 3>& p) {
    const int a = NearestSeed(p);
    double best = std::numeric_limits<double>::infinity();
    for (int b = 0; b < NUM_SEEDS; ++b) {
        if (b == a) {
            continue;
        }
        double da = 0.0, db = 0.0, ab = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double ta = p[d] - kSeeds[static_cast<std::size_t>(a)][d];
            const double tb = p[d] - kSeeds[static_cast<std::size_t>(b)][d];
            const double tab = kSeeds[static_cast<std::size_t>(a)][d] -
                               kSeeds[static_cast<std::size_t>(b)][d];
            da += ta * ta;
            db += tb * tb;
            ab += tab * tab;
        }
        best = std::min(best, (db - da) / (2.0 * std::sqrt(ab)));
    }
    return best;
}

/** Voxel (i, j, k) -> linear index, x fastest (the file and VoxelBlock layout). */
inline int VoxelIndex(int i, int j, int k) {
    return (k * NY + j) * NX + i;
}

/** The voxelized tessellation: ids (x fastest), the hole voxels and the hole's grain id. */
struct VoxelMapData {
    std::vector<std::int64_t> ids;      ///< with the hole (0)
    std::vector<std::int64_t> filled;   ///< without the hole (what the fill must restore)
    std::array<int, 3> hole_lo{};       ///< first voxel of the 2 x 2 x 2 hole
    int hole_feature = 0;
};

/**
 * Builds the map: the id of every voxel is the feature id of the seed nearest to its center;
 * then the first (lexicographic) 2 x 2 x 2 block whose surrounding 4 x 4 x 4 voxels all carry
 * one id becomes the hole. Every hole voxel then has 3 face neighbors outside the hole, all in
 * that grain, so the fill must give the hole that grain.
 */
inline const VoxelMapData& MapData() {
    static VoxelMapData data;
    if (!data.ids.empty()) {
        return data;
    }
    data.filled.resize(static_cast<std::size_t>(NX * NY * NZ));
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                const std::array<double, 3> c = {(i + 0.5) * H, (j + 0.5) * H, (k + 0.5) * H};
                data.filled[static_cast<std::size_t>(VoxelIndex(i, j, k))] =
                    kFeatureId[static_cast<std::size_t>(NearestSeed(c))];
            }
        }
    }
    data.ids = data.filled;
    bool placed = false;
    for (int k = 1; k + 3 <= NZ && !placed; ++k) {
        for (int j = 1; j + 3 <= NY && !placed; ++j) {
            for (int i = 1; i + 3 <= NX && !placed; ++i) {
                const std::int64_t id =
                    data.filled[static_cast<std::size_t>(VoxelIndex(i - 1, j - 1, k - 1))];
                bool uniform = true;
                for (int c = k - 1; c < k + 3; ++c) {
                    for (int b = j - 1; b < j + 3; ++b) {
                        for (int a = i - 1; a < i + 3; ++a) {
                            uniform = uniform &&
                                      data.filled[static_cast<std::size_t>(VoxelIndex(a, b, c))] ==
                                          id;
                        }
                    }
                }
                if (uniform) {
                    placed = true;
                    data.hole_lo = {i, j, k};
                    data.hole_feature = static_cast<int>(id);
                    for (int c = k; c < k + 2; ++c) {
                        for (int b = j; b < j + 2; ++b) {
                            for (int a = i; a < i + 2; ++a) {
                                data.ids[static_cast<std::size_t>(VoxelIndex(a, b, c))] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    if (!placed) {
        throw std::logic_error("test setup: no place for the hole");
    }
    return data;
}

/** Quaternion (w x y z, unit) of feature id r: a fixed rotation family, normalized. */
inline std::array<double, 4> FeatureQuat(int r) {
    std::array<double, 4> q = {1.0, 0.1 * r, -0.05 * r, 0.02 * r};
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (double& v : q) {
        v /= n;
    }
    return q;
}

/** Phase of feature id r. */
inline int FeaturePhase(int r) {
    return r % 3;
}

// ---------------------------------------------------------------------------------------------
// HDF5 fixture writers (host, rank 0 only)
// ---------------------------------------------------------------------------------------------

/** Closes an HDF5 handle at scope exit. */
struct Closer {
    hid_t id;
    herr_t (*close)(hid_t);
    ~Closer() {
        if (id >= 0) {
            close(id);
        }
    }
};

/** Writer for one fixture file (created/truncated in the constructor). */
class H5Writer {
public:
    explicit H5Writer(const fs::path& path) {
        m_file = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (m_file < 0) {
            throw std::runtime_error("cannot create " + path.string());
        }
        // Link creation property: create missing intermediate groups automatically
        m_lcpl = H5Pcreate(H5P_LINK_CREATE);
        H5Pset_create_intermediate_group(m_lcpl, 1);
    }
    ~H5Writer() {
        H5Pclose(m_lcpl);
        H5Fclose(m_file);
    }
    H5Writer(const H5Writer&) = delete;
    H5Writer& operator=(const H5Writer&) = delete;

    /** Writes `data` (memory type mem_type) as a dataset of file type file_type and shape dims. */
    void Dataset(const std::string& path, hid_t file_type, hid_t mem_type,
                 const std::vector<hsize_t>& dims, const void* data) {
        Closer space{H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr),
                     H5Sclose};
        Closer dset{H5Dcreate2(m_file, path.c_str(), file_type, space.id, m_lcpl, H5P_DEFAULT,
                               H5P_DEFAULT),
                    H5Dclose};
        if (dset.id < 0 ||
            H5Dwrite(dset.id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
            throw std::runtime_error("cannot write dataset " + path);
        }
    }

    /** Creates a group (and its parents). */
    void Group(const std::string& path) {
        Closer group{H5Gcreate2(m_file, path.c_str(), m_lcpl, H5P_DEFAULT, H5P_DEFAULT),
                     H5Gclose};
        if (group.id < 0) {
            throw std::runtime_error("cannot create group " + path);
        }
    }

    /** Writes a 3-value attribute on an existing object. */
    void Attribute3(const std::string& object, const char* name, hid_t file_type, hid_t mem_type,
                    const void* data) {
        Closer obj{H5Oopen(m_file, object.c_str(), H5P_DEFAULT), H5Oclose};
        const hsize_t three = 3;
        Closer space{H5Screate_simple(1, &three, nullptr), H5Sclose};
        Closer attr{H5Acreate2(obj.id, name, file_type, space.id, H5P_DEFAULT, H5P_DEFAULT),
                    H5Aclose};
        if (attr.id < 0 || H5Awrite(attr.id, mem_type, data) < 0) {
            throw std::runtime_error(std::string("cannot write attribute ") + name);
        }
    }

private:
    hid_t m_file = -1;
    hid_t m_lcpl = -1;
};

/** Voxel ids as int32 (DREAM.3D type). */
inline std::vector<int> AsInt32(const std::vector<std::int64_t>& ids) {
    return std::vector<int>(ids.begin(), ids.end());
}

/** Options of a canonical fixture. */
struct CanonicalOptions {
    std::vector<std::int64_t> ids;                 ///< voxel ids (x fastest), dims[0..2]
    std::array<hsize_t, 3> dims = {NX, NY, NZ};    ///< voxel counts x, y, z
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    std::array<double, 3> spacing = {H, H, H};
    bool rank4_uint16 = false;                     ///< store as uint16 (nz, ny, nx, 1)
    bool features = true;
    std::vector<int> feature_ids = {21, 3, 34, 8, 13, 5}; ///< unsorted on purpose
    int bad_quat_id = -1;                          ///< this id gets a quaternion of norm 0.5
};

/** Writes a canonical voxel map. Ids are stored as int64 unless rank4_uint16. */
inline void WriteCanonical(const fs::path& path, const CanonicalOptions& opt) {
    H5Writer w(path);
    if (opt.rank4_uint16) {
        std::vector<std::uint16_t> ids(opt.ids.begin(), opt.ids.end());
        w.Dataset("/grain_ids", H5T_STD_U16LE, H5T_NATIVE_UINT16,
                  {opt.dims[2], opt.dims[1], opt.dims[0], 1}, ids.data());
    } else {
        w.Dataset("/grain_ids", H5T_STD_I64LE, H5T_NATIVE_INT64,
                  {opt.dims[2], opt.dims[1], opt.dims[0]}, opt.ids.data());
    }
    w.Dataset("/origin", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, {3}, opt.origin.data());
    w.Dataset("/spacing", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, {3}, opt.spacing.data());
    if (!opt.features) {
        return;
    }
    const hsize_t nf = opt.feature_ids.size();
    std::vector<std::int64_t> fid(opt.feature_ids.begin(), opt.feature_ids.end());
    std::vector<double> quat;
    std::vector<int> phase;
    for (int id : opt.feature_ids) {
        std::array<double, 4> q = FeatureQuat(id);
        if (id == opt.bad_quat_id) {
            for (double& v : q) {
                v *= 0.5;
            }
        }
        quat.insert(quat.end(), q.begin(), q.end());
        phase.push_back(FeaturePhase(id));
    }
    w.Dataset("/features/id", H5T_STD_I64LE, H5T_NATIVE_INT64, {nf}, fid.data());
    w.Dataset("/features/quat", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, {nf, 4}, quat.data());
    w.Dataset("/features/phase", H5T_STD_I32LE, H5T_NATIVE_INT, {nf}, phase.data());
}

/** Number of DREAM.3D feature rows: ids 0..34. */
inline constexpr int kRows = 35;

/**
 * Float32 per-feature quaternion table as DREAM.3D stores it: row r = feature id r, stored
 * x y z w (scalar last); unused rows (including row 0) are zero.
 */
inline std::vector<float> Dream3dQuatsXyzw() {
    std::vector<float> q(static_cast<std::size_t>(4 * kRows), 0.0f);
    for (int id : kFeatureId) {
        const std::array<double, 4> wxyz = FeatureQuat(id);
        q[static_cast<std::size_t>(4 * id + 0)] = static_cast<float>(wxyz[1]);
        q[static_cast<std::size_t>(4 * id + 1)] = static_cast<float>(wxyz[2]);
        q[static_cast<std::size_t>(4 * id + 2)] = static_cast<float>(wxyz[3]);
        q[static_cast<std::size_t>(4 * id + 3)] = static_cast<float>(wxyz[0]);
    }
    return q;
}

/** Per-feature phases (row = id), int32. */
inline std::vector<int> Dream3dPhases() {
    std::vector<int> p(kRows, 0);
    for (int id : kFeatureId) {
        p[static_cast<std::size_t>(id)] = FeaturePhase(id);
    }
    return p;
}

/**
 * DREAM.3D SIMPL v6 layout: /DataContainers/ImageDataContainer/{_SIMPL_GEOMETRY, CellData,
 * CellFeatureData}. CellData also holds a per-voxel `Phases` array (as real files often do), so
 * the per-feature name search must skip it.
 */
inline void WriteSimpl(const fs::path& path) {
    const VoxelMapData& map = MapData();
    H5Writer w(path);
    const std::string dc = "/DataContainers/ImageDataContainer";
    const std::int64_t dims[3] = {NX, NY, NZ};
    const float origin[3] = {0.0f, 0.0f, 0.0f};
    const float spacing[3] = {static_cast<float>(H), static_cast<float>(H),
                              static_cast<float>(H)};
    w.Dataset(dc + "/_SIMPL_GEOMETRY/DIMENSIONS", H5T_STD_I64LE, H5T_NATIVE_INT64, {3}, dims);
    w.Dataset(dc + "/_SIMPL_GEOMETRY/ORIGIN", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {3}, origin);
    w.Dataset(dc + "/_SIMPL_GEOMETRY/SPACING", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {3}, spacing);
    const std::vector<int> ids = AsInt32(map.ids);
    w.Dataset(dc + "/CellData/FeatureIds", H5T_STD_I32LE, H5T_NATIVE_INT, {NZ, NY, NX, 1},
              ids.data());
    std::vector<int> cell_phases(ids.size(), 1);
    w.Dataset(dc + "/CellData/Phases", H5T_STD_I32LE, H5T_NATIVE_INT, {NZ, NY, NX, 1},
              cell_phases.data());
    const std::vector<float> quats = Dream3dQuatsXyzw();
    w.Dataset(dc + "/CellFeatureData/AvgQuats", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {kRows, 4},
              quats.data());
    const std::vector<int> phases = Dream3dPhases();
    w.Dataset(dc + "/CellFeatureData/Phases", H5T_STD_I32LE, H5T_NATIVE_INT, {kRows, 1},
              phases.data());
}

/**
 * DREAM3D-NX-like layout: /DataStructure/ImageGeom with attributes _DIMENSIONS (uint64),
 * _ORIGIN, _SPACING (float32), arrays in "Cell Data" and "Cell Feature Data".
 * `geometry` = 0: no attributes; 1: correct; 2: wrong _DIMENSIONS.
 */
inline void WriteNx(const fs::path& path, int geometry) {
    const VoxelMapData& map = MapData();
    H5Writer w(path);
    const std::string geom = "/DataStructure/ImageGeom";
    w.Group(geom);
    if (geometry > 0) {
        const std::uint64_t dims[3] = {NX, NY, static_cast<std::uint64_t>(geometry == 1 ? NZ
                                                                                         : NZ + 1)};
        const float origin[3] = {0.0f, 0.0f, 0.0f};
        const float spacing[3] = {static_cast<float>(H), static_cast<float>(H),
                                  static_cast<float>(H)};
        w.Attribute3(geom, "_DIMENSIONS", H5T_STD_U64LE, H5T_NATIVE_UINT64, dims);
        w.Attribute3(geom, "_ORIGIN", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, origin);
        w.Attribute3(geom, "_SPACING", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, spacing);
    }
    const std::vector<int> ids = AsInt32(map.ids);
    w.Dataset(geom + "/Cell Data/FeatureIds", H5T_STD_I32LE, H5T_NATIVE_INT, {NZ, NY, NX, 1},
              ids.data());
    const std::vector<float> quats = Dream3dQuatsXyzw();
    w.Dataset(geom + "/Cell Feature Data/AvgQuats", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
              {kRows, 4}, quats.data());
}

/**
 * Two data containers A and B, so `FeatureIds` is ambiguous. With `a_quats_zero`, A's AvgQuats
 * are all zero (reading them for the used ids would fail the norm check, which shows that B's
 * are read when B's FeatureIds is chosen); with `b_has_quats` false, only A has AvgQuats.
 */
inline void WriteTwoContainers(const fs::path& path, bool a_quats_zero, bool b_has_quats) {
    const VoxelMapData& map = MapData();
    H5Writer w(path);
    const std::vector<int> ids = AsInt32(map.ids);
    for (const std::string dc : {"/DataContainers/A", "/DataContainers/B"}) {
        const bool is_a = dc == "/DataContainers/A";
        std::vector<float> quats = Dream3dQuatsXyzw();
        if (is_a && a_quats_zero) {
            std::fill(quats.begin(), quats.end(), 0.0f);
        }
        const std::int64_t dims[3] = {NX, NY, NZ};
        const float origin[3] = {0.0f, 0.0f, 0.0f};
        const float spacing[3] = {static_cast<float>(H), static_cast<float>(H),
                                  static_cast<float>(H)};
        w.Dataset(dc + "/_SIMPL_GEOMETRY/DIMENSIONS", H5T_STD_I64LE, H5T_NATIVE_INT64, {3}, dims);
        w.Dataset(dc + "/_SIMPL_GEOMETRY/ORIGIN", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {3}, origin);
        w.Dataset(dc + "/_SIMPL_GEOMETRY/SPACING", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {3},
                  spacing);
        w.Dataset(dc + "/CellData/FeatureIds", H5T_STD_I32LE, H5T_NATIVE_INT, {NZ, NY, NX, 1},
                  ids.data());
        if (is_a || b_has_quats) {
            w.Dataset(dc + "/CellFeatureData/AvgQuats", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
                      {kRows, 4}, quats.data());
        }
    }
}

/** A file with no FeatureIds dataset at all. */
inline void WriteNoFeatureIds(const fs::path& path) {
    H5Writer w(path);
    const int one = 1;
    w.Dataset("/Something/Else", H5T_STD_I32LE, H5T_NATIVE_INT, {1}, &one);
}

} // namespace voxel_fixture

#endif // AXOM_USE_HDF5
