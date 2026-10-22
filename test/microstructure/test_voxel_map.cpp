/**
 * @file test_voxel_map.cpp
 * @brief Tests of the voxel map reader (io/voxel_map_io.hpp) and the voxel labeler
 * (labeling/voxel_labeler.hpp).
 *
 * **Fixtures.** Rank 0 writes small HDF5 files with the HDF5 C API into
 * `test_voxel_map_data/np<N>/` under the working directory (the build tree when run by ctest;
 * never /tmp), then all ranks meet at a barrier. The main map is a 20 x 16 x 12 voxel grid with
 * voxel size h = 1/16 and origin 0 (so the box is [0, 1.25] x [0, 1] x [0, 0.75], and every voxel
 * boundary is exactly representable in floating point). Every voxel carries the feature id of the
 * seed nearest to the voxel center (a Voronoi tessellation of 6 seeds, computed in the test),
 * with feature ids {3, 5, 8, 13, 21, 34} (not 1..6, to exercise the compaction), and a 2 x 2 x 2
 * hole of id 0 placed inside one grain (where the surrounding 4 x 4 x 4 voxels all have the same
 * id). The map is written in the canonical layout and in DREAM.3D SIMPL v6 and NX-like layouts.
 *
 * **Rank-count independence.** Distributed labels are gathered into serial element order and
 * compared with the expected serial answer, so running the executable at np = 1..4
 * (test_voxel_map.cmake) checks that every rank count gives the same labels, with voxel blocks
 * split between ranks.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever. Error
 * tests call the throwing function on every rank; if any rank did not throw, the test would hang
 * or fail, so a pass means the error was collective.
 */
#include "microstructure/io/voxel_map_io.hpp"
#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/labeling/voxel_labeler.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/serial_order.hpp"

#include "common/voxel_fixture.hpp"

#include "axom/config.hpp"
#include "mfem.hpp"

#ifdef AXOM_USE_HDF5
#include "hdf5.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

int Rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int NumRanks() {
    int n = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

/**
 * Runs `f` (expected to throw on every rank) and returns the exception message, or "" if it did
 * not throw. Every rank must call it.
 */
std::string ErrorOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/** True if `text` contains `part`. */
bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

} // namespace

#ifdef AXOM_USE_HDF5

namespace {

// The voxelized tessellation and the HDF5 fixture writers live in common/voxel_fixture.hpp
// (shared with test_driver_stage1d).
using namespace voxel_fixture;

/** Directory of this run's fixtures (one per rank count, so concurrent np runs never collide). */
fs::path DataDir() {
    return fs::path("test_voxel_map_data") / ("np" + std::to_string(NumRanks()));
}

/** Path of fixture `name`. */
fs::path Fixture(const std::string& name) {
    return DataDir() / name;
}

/** File-unit description of voxel_scaled.h5 (units_scale 1e-3, see WriteFixtures). */
const std::array<double, 3> kScaledOrigin = {-10.0, 5.0, 2.0};
const std::array<double, 3> kScaledSpacing = {62.5, 62.5, 62.5};
const std::array<double, 3> kScaledTranslate = {0.01, -0.005, -0.002};

/** Writes every fixture once (rank 0), then synchronizes (collective). */
void WriteFixtures() {
    static bool written = false;
    if (written) {
        return;
    }
    written = true;
    std::string error;
    if (Rank() == 0) {
        try {
            fs::create_directories(DataDir());
            const VoxelMapData& map = MapData();
            CanonicalOptions base;
            base.ids = map.ids;
            WriteCanonical(Fixture("canonical.h5"), base);

            CanonicalOptions r4 = base;
            r4.rank4_uint16 = true;
            r4.features = false;
            WriteCanonical(Fixture("canonical_rank4.h5"), r4);

            CanonicalOptions scaled = base;
            scaled.origin = kScaledOrigin;
            scaled.spacing = kScaledSpacing;
            WriteCanonical(Fixture("scaled.h5"), scaled);

            CanonicalOptions wrong = base;
            wrong.origin = {5.0, 5.0, 5.0};
            wrong.spacing = {1.0, 1.0, 1.0};
            WriteCanonical(Fixture("wrong_geometry.h5"), wrong);

            CanonicalOptions big = base;
            big.ids[7] = std::int64_t(1) << 31;
            WriteCanonical(Fixture("big_id.h5"), big);

            CanonicalOptions empty = base;
            std::fill(empty.ids.begin(), empty.ids.end(), 0);
            WriteCanonical(Fixture("empty.h5"), empty);

            CanonicalOptions badquat = base;
            badquat.bad_quat_id = 13;
            WriteCanonical(Fixture("bad_quat.h5"), badquat);

            CanonicalOptions missing = base;
            missing.feature_ids = {3, 5, 8, 13, 21}; // 34 occurs in the voxels but has no row
            WriteCanonical(Fixture("missing_feature.h5"), missing);

            WriteSimpl(Fixture("simpl.h5"));
            WriteNx(Fixture("nx.h5"), 1);
            WriteNx(Fixture("nx_nogeom.h5"), 0);
            WriteNx(Fixture("nx_baddims.h5"), 2);
            WriteTwoContainers(Fixture("ambiguous.h5"), true, true);
            WriteTwoContainers(Fixture("only_a_quats.h5"), false, false);

            // Every element of a 2-voxel-wide mesh covers one voxel of id 9 (even i) and one of
            // id 4 (odd i): an exact vote tie
            CanonicalOptions tie = base;
            for (int k = 0; k < NZ; ++k) {
                for (int j = 0; j < NY; ++j) {
                    for (int i = 0; i < NX; ++i) {
                        tie.ids[static_cast<std::size_t>(VoxelIndex(i, j, k))] =
                            (i % 2 == 0) ? 9 : 4;
                    }
                }
            }
            tie.features = false;
            WriteCanonical(Fixture("tie.h5"), tie);

            // The hole stored as -5 instead of 0: negative ids are unassigned too
            CanonicalOptions negative = base;
            for (std::int64_t& id : negative.ids) {
                if (id == 0) {
                    id = -5;
                }
            }
            WriteCanonical(Fixture("negative.h5"), negative);

            // Grid for the curved-mesh test: [0, 1] x [0, 1] x [-0.25, 2], h = 1/16; id 7 below
            // z = 0, id 9 above (by voxel center)
            CanonicalOptions curved;
            curved.features = false;
            curved.origin = {0.0, 0.0, -0.25};
            curved.dims = {16, 16, 36};
            curved.ids.resize(16 * 16 * 36);
            for (int k = 0; k < 36; ++k) {
                for (int v = 0; v < 16 * 16; ++v) {
                    const double zc = -0.25 + (k + 0.5) * H;
                    curved.ids[static_cast<std::size_t>(k * 256 + v)] = (zc < 0.0) ? 7 : 9;
                }
            }
            WriteCanonical(Fixture("curved.h5"), curved);
            WriteNoFeatureIds(Fixture("no_feature_ids.h5"));
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    int failed = error.empty() ? 0 : 1;
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (failed != 0) {
        throw std::runtime_error("fixture writing failed on rank 0: " + error);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/** Canonical parameters for fixture `name`. */
ms::VoxelMapParameters Canonical(const std::string& name) {
    ms::VoxelMapParameters p;
    p.format = ms::VoxelFormat::EXACONSTIT;
    p.file = Fixture(name);
    return p;
}

/** DREAM.3D parameters for fixture `name`. */
ms::VoxelMapParameters Dream3d(const std::string& name) {
    ms::VoxelMapParameters p;
    p.format = ms::VoxelFormat::DREAM3D;
    p.file = Fixture(name);
    return p;
}

// ---------------------------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------------------------

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/**
 * Partitions `serial` over `comm` (collective), keeping local -> serial element ids. With
 * MPI_COMM_SELF every rank holds the whole mesh in serial order: the np = 1 reference.
 */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm = MPI_COMM_WORLD) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** Gathers per-element ints to rank 0 in serial element order (collective over `comm`). */
std::vector<int> Gather(const Distributed& d, const mfem::Array<int>& local,
                        MPI_Comm comm = MPI_COMM_WORLD) {
    local.HostRead();
    return ms::GatherToSerial(comm, d.order, local, 0);
}

/**
 * A partition given as an owner rank per serial element (collective): builds the local -> serial
 * map and the ParMesh. Owners must be valid ranks; a rank may own nothing.
 */
Distributed DistributeWithOwners(mfem::Mesh& serial, const std::vector<int>& owner) {
    const int rank = Rank();
    Distributed d;
    d.order.partitioning.SetSize(serial.GetNE());
    int num_local = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        d.order.partitioning[e] = owner[static_cast<std::size_t>(e)];
        num_local += (owner[static_cast<std::size_t>(e)] == rank) ? 1 : 0;
    }
    d.order.local_to_serial.SetSize(num_local);
    int next = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        if (owner[static_cast<std::size_t>(e)] == rank) {
            d.order.local_to_serial[next++] = e;
        }
    }
    d.pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, d.order);
    return d;
}

/**
 * The labeler with the grid description read first, as the integration code will call it
 * (collective).
 */
ms::VoxelLabels Label(mfem::ParMesh& pmesh, MPI_Comm comm, const ms::VoxelMapParameters& map,
                      const ms::VoxelLabelParameters& params) {
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(map, comm);
    return ms::LabelElementsFromVoxels(pmesh, comm, map, grid, params);
}

/** Box mesh of nx x ny x nz elements over [0, sx] x [0, sy] x [0, sz], lexicographic order. */
mfem::Mesh BoxMesh(int nx, int ny, int nz, mfem::Element::Type type, double sx, double sy,
                   double sz) {
    return mfem::Mesh::MakeCartesian3D(nx, ny, nz, type, sx, sy, sz, false);
}

/**
 * Vertex average c and radius r = max |v - c| over the vertices v of serial element e. For an
 * affine (hex box or tet) element every point of the element is within r of c (the element is
 * the convex hull of its vertices).
 */
void ElementCenterAndRadius(const mfem::Mesh& mesh, int e, std::array<double, 3>& center,
                            double& radius) {
    mfem::Array<int> v;
    mesh.GetElementVertices(e, v);
    center = {0.0, 0.0, 0.0};
    for (int a = 0; a < v.Size(); ++a) {
        const mfem::real_t* p = mesh.GetVertex(v[a]);
        for (int d = 0; d < 3; ++d) {
            center[static_cast<std::size_t>(d)] += p[d] / v.Size();
        }
    }
    radius = 0.0;
    for (int a = 0; a < v.Size(); ++a) {
        const mfem::real_t* p = mesh.GetVertex(v[a]);
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double t = p[d] - center[static_cast<std::size_t>(d)];
            d2 += t * t;
        }
        radius = std::max(radius, std::sqrt(d2));
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Reader tests
// ---------------------------------------------------------------------------------------------

/**
 * Canonical file: dimensions come from the (nz, ny, nx) shape, origin/spacing from the file, the
 * largest id is 34; a hyperslab block returns exactly the stored ids of that box; the feature
 * table is sorted by id (the file stores the ids unsorted) with quaternions bitwise verbatim and
 * the phases. The rank-4 uint16 variant gives the same grid and blocks (any integer type works).
 */
TEST(VoxelMapReader, CanonicalGridBlockAndFeatures) {
    WriteFixtures();
    const VoxelMapData& map = MapData();
    for (const std::string name : {"canonical.h5", "canonical_rank4.h5"}) {
        const ms::VoxelMapParameters params = Canonical(name);
        const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD);
        EXPECT_EQ(grid.dims, (std::array<std::int64_t, 3>{NX, NY, NZ}));
        EXPECT_EQ(grid.origin, (std::array<double, 3>{0.0, 0.0, 0.0}));
        EXPECT_EQ(grid.spacing, (std::array<double, 3>{H, H, H}));
        EXPECT_EQ(grid.feature_ids_path, "/grain_ids");

        // Block [3, 11) x [2, 9) x [4, 7): compare every voxel with the stored map
        const std::array<std::int64_t, 3> lo = {3, 2, 4};
        const std::array<std::int64_t, 3> hi = {11, 9, 7};
        const ms::VoxelBlock block = ms::ReadVoxelBlock(params, grid, lo, hi);
        EXPECT_EQ(block.ids.size(), std::size_t(8 * 7 * 3));
        bool same = block.ids.size() == std::size_t(8 * 7 * 3);
        for (int k = 4; k < 7 && same; ++k) {
            for (int j = 2; j < 9; ++j) {
                for (int i = 3; i < 11; ++i) {
                    const std::size_t b = static_cast<std::size_t>(((k - 4) * 7 + (j - 2)) * 8 +
                                                                   (i - 3));
                    same = same &&
                           block.ids[b] == map.ids[static_cast<std::size_t>(VoxelIndex(i, j, k))];
                }
            }
        }
        EXPECT_TRUE(same) << name;

        // Empty and out-of-range boxes
        EXPECT_TRUE(ms::ReadVoxelBlock(params, grid, {0, 0, 0}, {0, 5, 5}).ids.empty());
        EXPECT_THROW(ms::ReadVoxelBlock(params, grid, {0, 0, 0}, {NX + 1, 1, 1}),
                     std::invalid_argument);
    }

    // Feature table: sorted ids, verbatim quaternions, phases
    ms::VoxelMapParameters params = Canonical("canonical.h5");
    params.read_quats = true;
    params.read_phases = true;
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD);
    const ms::VoxelFeatureTable table = ms::ReadVoxelFeatures(params, grid, kUsedIds, MPI_COMM_WORLD);
    EXPECT_EQ(table.id, (std::vector<std::int64_t>{3, 5, 8, 13, 21, 34}));
    EXPECT_EQ(table.quat.size(), std::size_t(24));
    EXPECT_EQ(table.phase.size(), std::size_t(6));
    for (std::size_t r = 0; r < table.id.size() && table.quat.size() == 24; ++r) {
        const std::array<double, 4> q = FeatureQuat(static_cast<int>(table.id[r]));
        for (int c = 0; c < 4; ++c) {
            const auto cc = static_cast<std::size_t>(c);
            EXPECT_EQ(table.quat[4 * r + cc], q[cc]);
        }
        EXPECT_EQ(table.phase[r], FeaturePhase(static_cast<int>(table.id[r])));
    }
}

/**
 * DREAM.3D SIMPL v6: FeatureIds is found by name, the geometry comes from _SIMPL_GEOMETRY. The
 * per-feature `Phases` search skips the per-voxel `CellData/Phases` (same leading shape as the
 * voxels) and finds `CellFeatureData/Phases`. The quaternions are stored x y z w as float32:
 * - quat_layout = XYZW, conjugate = false gives w x y z = the stored (x y z w) moved, bitwise
 *   equal to the float32 values;
 * - conjugate = true negates x, y, z;
 * - quat_layout = WXYZ takes the stored order as is (here that is wrong physically, but the
 *   reader must do exactly what is declared).
 * Table ids are 1..34 (rows indexed by feature id, row 0 skipped).
 */
TEST(VoxelMapReader, Dream3dSimplLayoutAndQuatDeclarations) {
    WriteFixtures();
    ms::VoxelMapParameters params = Dream3d("simpl.h5");
    params.read_quats = true;
    params.read_phases = true;
    params.quat_layout = ms::QuatLayout::XYZW;
    params.conjugate = 0;
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD);
    EXPECT_EQ(grid.feature_ids_path, "/DataContainers/ImageDataContainer/CellData/FeatureIds");
    EXPECT_EQ(grid.dims, (std::array<std::int64_t, 3>{NX, NY, NZ}));
    EXPECT_EQ(grid.spacing, (std::array<double, 3>{H, H, H}));

    const std::vector<float> stored = Dream3dQuatsXyzw();
    const ms::VoxelFeatureTable plain = ms::ReadVoxelFeatures(params, grid, kUsedIds, MPI_COMM_WORLD);
    EXPECT_EQ(plain.id.size(), std::size_t(kRows - 1));
    EXPECT_EQ(plain.id.front(), 1);
    EXPECT_EQ(plain.id.back(), kRows - 1);
    params.conjugate = 1;
    const ms::VoxelFeatureTable conj = ms::ReadVoxelFeatures(params, grid, kUsedIds, MPI_COMM_WORLD);
    params.conjugate = 0;
    params.quat_layout = ms::QuatLayout::WXYZ;
    params.read_phases = false;
    // WXYZ reads the stored numbers in place; the norm is the same, so no error
    const ms::VoxelFeatureTable raw = ms::ReadVoxelFeatures(params, grid, kUsedIds, MPI_COMM_WORLD);

    for (int id : kFeatureId) {
        const std::size_t r = static_cast<std::size_t>(id - 1); // table row of feature id
        const std::size_t s = static_cast<std::size_t>(4 * id); // stored row
        // xyzw -> wxyz
        EXPECT_EQ(plain.quat[4 * r + 0], static_cast<double>(stored[s + 3]));
        EXPECT_EQ(plain.quat[4 * r + 1], static_cast<double>(stored[s + 0]));
        EXPECT_EQ(plain.quat[4 * r + 2], static_cast<double>(stored[s + 1]));
        EXPECT_EQ(plain.quat[4 * r + 3], static_cast<double>(stored[s + 2]));
        // conjugate: same w, negated vector part
        EXPECT_EQ(conj.quat[4 * r + 0], plain.quat[4 * r + 0]);
        EXPECT_EQ(conj.quat[4 * r + 1], -plain.quat[4 * r + 1]);
        EXPECT_EQ(conj.quat[4 * r + 2], -plain.quat[4 * r + 2]);
        EXPECT_EQ(conj.quat[4 * r + 3], -plain.quat[4 * r + 3]);
        // WXYZ: stored order unchanged
        for (int c = 0; c < 4; ++c) {
            EXPECT_EQ(raw.quat[4 * r + static_cast<std::size_t>(c)],
                      static_cast<double>(stored[s + static_cast<std::size_t>(c)]));
        }
        EXPECT_EQ(plain.phase[r], FeaturePhase(id));
    }
    EXPECT_TRUE(raw.phase.empty());

    // Phases only: no quaternion declarations needed
    ms::VoxelMapParameters phases_only = Dream3d("simpl.h5");
    phases_only.read_phases = true;
    const ms::VoxelFeatureTable phases =
        ms::ReadVoxelFeatures(phases_only, grid, kUsedIds, MPI_COMM_WORLD);
    EXPECT_TRUE(phases.quat.empty());
    EXPECT_EQ(phases.phase.size(), std::size_t(kRows - 1));
    for (int id : kFeatureId) {
        EXPECT_EQ(phases.phase[static_cast<std::size_t>(id - 1)], FeaturePhase(id));
    }
}

/**
 * Missing DREAM.3D declarations are errors on every rank (no defaults: the declarations describe
 * the file, and guessing would silently rotate every grain). The canonical format refuses
 * declarations that contradict its fixed w x y z storage.
 */
TEST(VoxelMapReader, QuatDeclarationsRequired) {
    WriteFixtures();
    ms::VoxelMapParameters params = Dream3d("simpl.h5");
    params.read_quats = true;
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD);

    params.conjugate = 0; // layout missing
    std::string error = ErrorOf([&] { ms::ReadVoxelFeatures(params, grid, kUsedIds, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "quat_layout")) << error;

    params.quat_layout = ms::QuatLayout::XYZW;
    params.conjugate = -1; // conjugate missing
    error = ErrorOf([&] { ms::ReadVoxelFeatures(params, grid, kUsedIds, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "conjugate")) << error;

    ms::VoxelMapParameters canonical = Canonical("canonical.h5");
    canonical.read_quats = true;
    canonical.quat_layout = ms::QuatLayout::XYZW;
    const ms::VoxelGridInfo cgrid = ms::ReadVoxelGridInfo(canonical, MPI_COMM_WORLD);
    error = ErrorOf([&] { ms::ReadVoxelFeatures(canonical, cgrid, kUsedIds, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "canonical")) << error;
}

/**
 * Name search: two datasets named FeatureIds are an error listing both paths and naming the
 * override key; no FeatureIds at all names the key too; an explicit path resolves the ambiguity
 * (and then the geometry is found next to the chosen array). Per-feature arrays come from the
 * data container of the chosen FeatureIds: with B chosen, B's AvgQuats are read automatically
 * (A's are all zero, so reading them would fail the norm check); when only A has AvgQuats, the
 * read fails with A listed as a candidate from another container, instead of silently using
 * A's table; an explicit quats_path still allows it.
 */
TEST(VoxelMapReader, NameSearchErrorsAndOverrides) {
    WriteFixtures();
    ms::VoxelMapParameters params = Dream3d("ambiguous.h5");
    std::string error = ErrorOf([&] { ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "/DataContainers/A/CellData/FeatureIds")) << error;
    EXPECT_TRUE(Contains(error, "/DataContainers/B/CellData/FeatureIds")) << error;
    EXPECT_TRUE(Contains(error, "feature_ids_path")) << error;

    params.feature_ids_path = "/DataContainers/B/CellData/FeatureIds";
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD);
    EXPECT_EQ(grid.feature_ids_path, params.feature_ids_path);
    EXPECT_EQ(grid.dims, (std::array<std::int64_t, 3>{NX, NY, NZ}));

    params.read_quats = true;
    params.quat_layout = ms::QuatLayout::XYZW;
    params.conjugate = 0;
    const ms::VoxelFeatureTable from_b =
        ms::ReadVoxelFeatures(params, grid, kUsedIds, MPI_COMM_WORLD);
    EXPECT_EQ(from_b.id.size(), std::size_t(kRows - 1));

    ms::VoxelMapParameters only_a = Dream3d("only_a_quats.h5");
    only_a.feature_ids_path = "/DataContainers/B/CellData/FeatureIds";
    only_a.read_quats = true;
    only_a.quat_layout = ms::QuatLayout::XYZW;
    only_a.conjugate = 0;
    const ms::VoxelGridInfo grid_b = ms::ReadVoxelGridInfo(only_a, MPI_COMM_WORLD);
    error = ErrorOf([&] { ms::ReadVoxelFeatures(only_a, grid_b, kUsedIds, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "/DataContainers/A/CellFeatureData/AvgQuats")) << error;
    EXPECT_TRUE(Contains(error, "other containers")) << error;
    EXPECT_TRUE(Contains(error, "quats_path")) << error;
    only_a.quats_path = "/DataContainers/A/CellFeatureData/AvgQuats";
    EXPECT_EQ(ms::ReadVoxelFeatures(only_a, grid_b, kUsedIds, MPI_COMM_WORLD).id.size(),
              std::size_t(kRows - 1));

    const ms::VoxelMapParameters none = Dream3d("no_feature_ids.h5");
    error = ErrorOf([&] { ms::ReadVoxelGridInfo(none, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "FeatureIds")) << error;
    EXPECT_TRUE(Contains(error, "feature_ids_path")) << error;

    const ms::VoxelMapParameters missing_file = Canonical("does_not_exist.h5");
    error = ErrorOf([&] { ms::ReadVoxelGridInfo(missing_file, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "cannot open")) << error;
}

/**
 * DREAM3D-NX-like files: geometry from the _ORIGIN/_SPACING/_DIMENSIONS attributes of the
 * geometry group (simplnx names). Without them, origin and spacing must come from the options;
 * with wrong _DIMENSIONS the cross-check against the FeatureIds shape fails.
 */
TEST(VoxelMapReader, Dream3dNxGeometry) {
    WriteFixtures();
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(Dream3d("nx.h5"), MPI_COMM_WORLD);
    EXPECT_EQ(grid.feature_ids_path, "/DataStructure/ImageGeom/Cell Data/FeatureIds");
    EXPECT_EQ(grid.dims, (std::array<std::int64_t, 3>{NX, NY, NZ}));
    EXPECT_EQ(grid.spacing, (std::array<double, 3>{H, H, H}));

    ms::VoxelMapParameters nogeom = Dream3d("nx_nogeom.h5");
    std::string error = ErrorOf([&] { ms::ReadVoxelGridInfo(nogeom, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "origin and spacing")) << error;
    nogeom.has_origin = true;
    nogeom.has_spacing = true;
    nogeom.origin = {1.0, 2.0, 3.0};
    nogeom.spacing = {0.5, 0.25, 0.125};
    const ms::VoxelGridInfo given = ms::ReadVoxelGridInfo(nogeom, MPI_COMM_WORLD);
    EXPECT_EQ(given.origin, nogeom.origin);
    EXPECT_EQ(given.spacing, nogeom.spacing);

    error = ErrorOf([&] { ms::ReadVoxelGridInfo(Dream3d("nx_baddims.h5"), MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "dimensions")) << error;
}

/**
 * Validation: ids >= 2^31 are refused (element labels are int); a quaternion of norm 0.5 is an
 * error naming its feature id; an id that occurs in the voxels without a feature row is an error
 * naming the id; a non-positive spacing override is refused.
 */
TEST(VoxelMapReader, ValidationErrors) {
    WriteFixtures();
    // The id >= 2^31 sits in voxel (7, 0, 0); the grid description (metadata only) reads fine,
    // the labeler refuses it on the rank(s) whose tiles contain it, and all ranks throw
    mfem::Mesh serial = BoxMesh(NX, NY, NZ, mfem::Element::HEXAHEDRON, NX * H, NY * H, NZ * H);
    Distributed dist = Distribute(serial);
    std::string error = ErrorOf([&] {
        Label(*dist.pmesh, MPI_COMM_WORLD, Canonical("big_id.h5"), ms::VoxelLabelParameters{});
    });
    EXPECT_TRUE(Contains(error, "2^31")) << error;

    // Used-id list must be strictly increasing and >= 1; sparse large ids cost nothing extra
    // (the check reports the missing id instead of allocating one flag per possible id)
    ms::VoxelMapParameters canon = Canonical("canonical.h5");
    canon.read_phases = true;
    const ms::VoxelGridInfo canon_grid = ms::ReadVoxelGridInfo(canon, MPI_COMM_WORLD);
    error = ErrorOf([&] {
        ms::ReadVoxelFeatures(canon, canon_grid, {5, 3}, MPI_COMM_WORLD);
    });
    EXPECT_TRUE(Contains(error, "strictly increasing")) << error;
    error = ErrorOf([&] {
        ms::ReadVoxelFeatures(canon, canon_grid, {3, 2147483647}, MPI_COMM_WORLD);
    });
    EXPECT_TRUE(Contains(error, "grain id 2147483647")) << error;
    ms::VoxelMapParameters simpl = Dream3d("simpl.h5");
    simpl.read_phases = true;
    const ms::VoxelGridInfo simpl_grid = ms::ReadVoxelGridInfo(simpl, MPI_COMM_WORLD);
    error = ErrorOf([&] {
        ms::ReadVoxelFeatures(simpl, simpl_grid, {3, 2147483647}, MPI_COMM_WORLD);
    });
    EXPECT_TRUE(Contains(error, "2147483647")) << error;

    ms::VoxelMapParameters bad = Canonical("bad_quat.h5");
    bad.read_quats = true;
    const ms::VoxelGridInfo bad_grid = ms::ReadVoxelGridInfo(bad, MPI_COMM_WORLD);
    error = ErrorOf([&] { ms::ReadVoxelFeatures(bad, bad_grid, kUsedIds, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "feature id 13")) << error;

    ms::VoxelMapParameters missing = Canonical("missing_feature.h5");
    missing.read_phases = true;
    const ms::VoxelGridInfo missing_grid = ms::ReadVoxelGridInfo(missing, MPI_COMM_WORLD);
    error = ErrorOf([&] { ms::ReadVoxelFeatures(missing, missing_grid, kUsedIds, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "grain id 34")) << error;
    // With a used-id list without 34 (a mesh that never touches grain 34) the table is fine
    EXPECT_EQ(ms::ReadVoxelFeatures(missing, missing_grid, {3, 5, 8}, MPI_COMM_WORLD).id.size(),
              std::size_t(5));

    ms::VoxelMapParameters zero = Canonical("canonical.h5");
    zero.has_spacing = true;
    zero.spacing = {H, 0.0, H};
    error = ErrorOf([&] { ms::ReadVoxelGridInfo(zero, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "spacing")) << error;

    // A units_scale so small that the scaled spacing underflows to 0, and a non-finite translate
    ms::VoxelMapParameters tiny = Canonical("canonical.h5");
    tiny.units_scale = 1.0e-323; // times spacing 1/16 rounds to 0
    error = ErrorOf([&] { ms::ReadVoxelGridInfo(tiny, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "after units_scale")) << error;
    ms::VoxelMapParameters nan_translate = Canonical("canonical.h5");
    nan_translate.translate = {0.0, std::nan(""), 0.0};
    error = ErrorOf([&] { ms::ReadVoxelGridInfo(nan_translate, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "translate")) << error;
}

/**
 * Transforms: the scaled fixture stores origin (-10, 5, 2) and spacing 62.5 in file units; with
 * units_scale = 1e-3 and translate = (0.01, -0.005, -0.002) the mesh-unit grid is origin 0,
 * spacing 1/16 (to round-off). TOML origin/spacing replace wrong file values before the
 * transform.
 */
TEST(VoxelMapReader, UnitsAndOverrides) {
    WriteFixtures();
    ms::VoxelMapParameters scaled = Canonical("scaled.h5");
    scaled.units_scale = 1.0e-3;
    scaled.translate = kScaledTranslate;
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(scaled, MPI_COMM_WORLD);
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(grid.origin[static_cast<std::size_t>(d)], 0.0, 1e-15);
        EXPECT_NEAR(grid.spacing[static_cast<std::size_t>(d)], H, 1e-15);
    }

    ms::VoxelMapParameters wrong = Canonical("wrong_geometry.h5");
    wrong.has_origin = true;
    wrong.has_spacing = true;
    wrong.origin = {0.0, 0.0, 0.0};
    wrong.spacing = {H, H, H};
    const ms::VoxelGridInfo fixed = ms::ReadVoxelGridInfo(wrong, MPI_COMM_WORLD);
    EXPECT_EQ(fixed.origin, (std::array<double, 3>{0.0, 0.0, 0.0}));
    EXPECT_EQ(fixed.spacing, (std::array<double, 3>{H, H, H}));
}

// ---------------------------------------------------------------------------------------------
// Labeler tests
// ---------------------------------------------------------------------------------------------

namespace {

/**
 * Labels the voxel-aligned 20 x 16 x 12 hex mesh (one element per voxel, same lexicographic
 * order as the voxels) and checks the gathered labels against the hole-free map. Every centroid
 * and every subcell voting point lies strictly inside its own voxel, so each element sees exactly
 * its voxel's id; the 8 hole elements get no vote and are filled in one round from their 3
 * outside face neighbors, which all carry the hole's grain.
 */
void ExpectAlignedExact(const ms::VoxelMapParameters& map, const ms::VoxelLabelParameters& lp,
                        const std::string& what) {
    mfem::Mesh serial = BoxMesh(NX, NY, NZ, mfem::Element::HEXAHEDRON, NX * H, NY * H, NZ * H);
    Distributed dist = Distribute(serial);
    const ms::VoxelLabels labels = Label(*dist.pmesh, MPI_COMM_WORLD, map,
                                                               lp);
    EXPECT_EQ(labels.num_filled, 8) << what;
    EXPECT_EQ(labels.fill_rounds, 1) << what;
    EXPECT_EQ(labels.num_outside_points, 0) << what;
    const std::vector<int> all = Gather(dist, labels.element_feature);
    if (Rank() == 0) {
        const VoxelMapData& data = MapData();
        EXPECT_EQ(all.size(), data.filled.size());
        int mismatches = 0;
        for (std::size_t e = 0; e < all.size(); ++e) {
            mismatches += (all[e] != data.filled[e]) ? 1 : 0;
        }
        EXPECT_EQ(mismatches, 0) << what;
    }
}

} // namespace

/**
 * Exact labels on the voxel-aligned mesh, for the canonical and SIMPL files and for centroid and
 * vote sampling (see ExpectAlignedExact). Comparing with the serial map at every rank count
 * checks rank-count independence, including blocks split between ranks.
 */
TEST(VoxelLabeler, AlignedHexMeshMatchesVoxels) {
    WriteFixtures();
    ms::VoxelLabelParameters centroid;
    ms::VoxelLabelParameters vote;
    vote.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    vote.vote_levels = 2;
    vote.max_points_per_chunk = 1000; // several chunks per rank
    ExpectAlignedExact(Canonical("canonical.h5"), centroid, "canonical centroid");
    ExpectAlignedExact(Canonical("canonical.h5"), vote, "canonical vote");
    ExpectAlignedExact(Dream3d("simpl.h5"), centroid, "simpl centroid");
    ExpectAlignedExact(Dream3d("nx.h5"), vote, "nx vote");

    // Transforms: file-unit geometry made to match the mesh by units_scale/translate, and wrong
    // file geometry replaced by the TOML values
    ms::VoxelMapParameters scaled = Canonical("scaled.h5");
    scaled.units_scale = 1.0e-3;
    scaled.translate = kScaledTranslate;
    ExpectAlignedExact(scaled, centroid, "scaled");
    ms::VoxelMapParameters wrong = Canonical("wrong_geometry.h5");
    wrong.has_origin = true;
    wrong.has_spacing = true;
    wrong.origin = {0.0, 0.0, 0.0};
    wrong.spacing = {H, H, H};
    ExpectAlignedExact(wrong, vote, "override");

    // Negative ids are unassigned like 0: the hole stored as -5 is filled the same way
    ExpectAlignedExact(Canonical("negative.h5"), centroid, "negative ids");

    // Small tiles (many tiles per rank): same labels
    ms::VoxelLabelParameters small_tiles = vote;
    small_tiles.tile_size = 3;
    ExpectAlignedExact(Canonical("canonical.h5"), small_tiles, "tile_size 3");
}

/**
 * Exact vote tie: every element of a 10 x 16 x 12 hex mesh spans two voxels along x, one with
 * id 9 (even i) and one with id 4 (odd i). The vote points are symmetric (level 2: 4 points per
 * direction, 2 in each voxel) with equal weights on an affine element, so both ids collect
 * exactly half of the element volume and the lower id, 4, must win everywhere.
 */
TEST(VoxelLabeler, ExactVoteTieGoesToLowestId) {
    WriteFixtures();
    mfem::Mesh serial = BoxMesh(NX / 2, NY, NZ, mfem::Element::HEXAHEDRON, NX * H, NY * H,
                                NZ * H);
    Distributed dist = Distribute(serial);
    ms::VoxelLabelParameters vote;
    vote.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    vote.vote_levels = 2;
    const ms::VoxelLabels labels = Label(*dist.pmesh, MPI_COMM_WORLD, Canonical("tie.h5"), vote);
    const std::vector<int> all = Gather(dist, labels.element_feature);
    if (Rank() == 0) {
        EXPECT_EQ(all, std::vector<int>(all.size(), 4));
    }
}

/**
 * Curved elements (the reviewer's case): a 2 x 2 x 2 hex mesh of the unit cube made quadratic
 * (SetCurvature(2)) and mapped by z' = z + x(2x - 1). The map is quadratic, so the quadratic
 * nodes reproduce it exactly; det J = 1 (volume preserving). Every node has z' >= 0 (at the node
 * x values 0, 1/2, 1 the term x(2x-1) is 0, 0, 1), but between the nodes the bottom face dips
 * to z' = -1/8 at x = 1/4, so the vote points reach below the node bounding box by more than a
 * voxel (h = 1/16). The grid spans z in [-0.25, 2] with id 7 below z = 0 and id 9 above. The
 * labeler must not stop (it reads the tiles under the actual points), every point lies in the
 * grid, and the labels must equal a single-rank reference. Run with small tiles so the tiles
 * below z = 0 are separate tiles that the vertex/node box would have missed.
 */
TEST(VoxelLabeler, CurvedElementsBeyondTheirNodes) {
    WriteFixtures();
    mfem::Mesh serial = BoxMesh(2, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0);
    serial.SetCurvature(2);
    serial.Transform([](const mfem::Vector& x, mfem::Vector& y) {
        y = x;
        y(2) = x(2) + x(0) * (2.0 * x(0) - 1.0);
    });
    ms::VoxelLabelParameters vote;
    vote.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    vote.vote_levels = 3;
    vote.tile_size = 2;

    // The scenario: voting points below the lowest node by more than one voxel
    ms::QuadraturePointSamples points;
    ms::ComputeVotePoints(serial, 3, 0, serial.GetNE(), points);
    const double* z = points.z.HostRead();
    double min_z = 0.0;
    for (int i = 0; i < points.z.Size(); ++i) {
        min_z = std::min(min_z, z[i]);
    }
    EXPECT_LT(min_z, -H);

    Distributed dist = Distribute(serial);
    const ms::VoxelLabels labels = Label(*dist.pmesh, MPI_COMM_WORLD, Canonical("curved.h5"), vote);
    EXPECT_EQ(labels.num_outside_points, 0);
    EXPECT_EQ(labels.num_filled, 0);
    const std::vector<int> all = Gather(dist, labels.element_feature);

    Distributed self = Distribute(serial, MPI_COMM_SELF);
    const ms::VoxelLabels reference =
        Label(*self.pmesh, MPI_COMM_SELF, Canonical("curved.h5"), vote);
    const std::vector<int> reference_all = Gather(self, reference.element_feature, MPI_COMM_SELF);
    if (Rank() == 0) {
        EXPECT_EQ(all, reference_all);
        // Most of every element lies above z = 0
        EXPECT_EQ(all, std::vector<int>(all.size(), 9));
    }
}

/**
 * Tiles follow the elements, not the partition's bounding box: on the voxel-aligned mesh, rank
 * 0 owns only the two x-layers at each end of the grid (i in {0, 1, 18, 19}), a partition whose
 * bounding box is the whole grid; the other elements are dealt round-robin to ranks
 * 1 .. nranks-2, so with np >= 3 the last rank owns no element at all (and reads nothing). With
 * tile_size 4, rank 0 needs the x-tiles 0 and 4 only: 2 x 4 x 3 = 24 tiles of 64 voxels = 1536
 * voxels, against 3840 for the whole grid (np = 1: rank 0 owns everything and reads all 60
 * tiles). Labels must still be exact.
 */
TEST(VoxelLabeler, SplitPartitionReadsOnlyItsTiles) {
    WriteFixtures();
    mfem::Mesh serial = BoxMesh(NX, NY, NZ, mfem::Element::HEXAHEDRON, NX * H, NY * H, NZ * H);
    const int nranks = NumRanks();
    std::vector<int> owner(static_cast<std::size_t>(serial.GetNE()), 0);
    int dealt = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        const int i = e % NX;
        const bool end_layer = i < 2 || i >= NX - 2;
        if (!end_layer && nranks > 1) {
            const int others = (nranks > 2) ? nranks - 2 : 1; // ranks 1 .. nranks-2 (or 1)
            owner[static_cast<std::size_t>(e)] = 1 + (dealt++ % others);
        }
    }
    Distributed dist = DistributeWithOwners(serial, owner);
    ms::VoxelLabelParameters params;
    params.tile_size = 4;
    const ms::VoxelLabels labels =
        Label(*dist.pmesh, MPI_COMM_WORLD, Canonical("canonical.h5"), params);
    EXPECT_EQ(labels.num_filled, 8);
    if (Rank() == 0) {
        std::cout << "  split partition, np " << nranks << ": rank 0 read "
                  << labels.num_tiles_read << " tiles, " << labels.num_voxels_read
                  << " voxels (grid: " << NX * NY * NZ << ")" << std::endl;
        if (nranks > 1) {
            EXPECT_EQ(labels.num_tiles_read, 24);
            EXPECT_EQ(labels.num_voxels_read, 1536);
        } else {
            EXPECT_EQ(labels.num_voxels_read, NX * NY * NZ);
        }
    }
    if (nranks >= 3 && Rank() == nranks - 1) {
        EXPECT_EQ(dist.pmesh->GetNE(), 0);
        EXPECT_EQ(labels.num_tiles_read, 0);
        EXPECT_EQ(labels.element_feature.Size(), 0);
    }
    const std::vector<int> all = Gather(dist, labels.element_feature);
    if (Rank() == 0) {
        const VoxelMapData& data = MapData();
        int mismatches = 0;
        for (std::size_t e = 0; e < all.size(); ++e) {
            mismatches += (all[e] != data.filled[e]) ? 1 : 0;
        }
        EXPECT_EQ(mismatches, 0);
    }

    // Compaction and the feature table with a rank that owns nothing
    mfem::Array<int> element_grain;
    std::vector<std::int64_t> grain_feature;
    ms::CompactFeatureLabels(MPI_COMM_WORLD, labels.element_feature, element_grain,
                             grain_feature);
    EXPECT_EQ(grain_feature, kUsedIds);
    ms::VoxelMapParameters map = Canonical("canonical.h5");
    map.read_quats = true;
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(map, MPI_COMM_WORLD);
    EXPECT_EQ(ms::ReadVoxelFeatures(map, grid, grain_feature, MPI_COMM_WORLD).id, kUsedIds);
}

/**
 * A mesh three element layers longer than the grid in +x (23 x 16 x 12 voxel-sized hexes): the
 * 3 x 16 x 12 = 576 elements beyond x = 1.25 have their centroid outside the grid (576 outside
 * points). Fill rounds are synchronous, so layer x = 20 copies layer 19 in round 1 (its only
 * assigned neighbor), layer 21 copies 20 in round 2 and layer 22 in round 3: 3 rounds, and the
 * 8 hole elements are filled in round 1, so 584 elements are filled in total.
 */
TEST(VoxelLabeler, FillRoundsOutsideTheGrid) {
    WriteFixtures();
    constexpr int NXE = NX + 3;
    mfem::Mesh serial = BoxMesh(NXE, NY, NZ, mfem::Element::HEXAHEDRON, NXE * H, NY * H, NZ * H);
    Distributed dist = Distribute(serial);
    const ms::VoxelLabels labels = Label(
        *dist.pmesh, MPI_COMM_WORLD, Canonical("canonical.h5"), ms::VoxelLabelParameters{});
    EXPECT_EQ(labels.fill_rounds, 3);
    EXPECT_EQ(labels.num_filled, 3 * NY * NZ + 8);
    EXPECT_EQ(labels.num_outside_points, 3 * NY * NZ);
    const std::vector<int> all = Gather(dist, labels.element_feature);
    if (Rank() == 0) {
        const VoxelMapData& data = MapData();
        int mismatches = 0;
        for (int k = 0; k < NZ; ++k) {
            for (int j = 0; j < NY; ++j) {
                for (int i = 0; i < NXE; ++i) {
                    const int e = (k * NY + j) * NXE + i;
                    const int voxel = VoxelIndex(std::min(i, NX - 1), j, k); // copy of layer 19
                    const auto expected = data.filled[static_cast<std::size_t>(voxel)];
                    mismatches += (all[static_cast<std::size_t>(e)] != expected) ? 1 : 0;
                }
            }
        }
        EXPECT_EQ(mismatches, 0);
    }
}

/**
 * Coarse meshes not aligned with the voxels (9 x 7 x 5 hexes, and 6 x 5 x 4 boxes of 6 tets
 * each), vote sampling: away from grain boundaries the voxel labels must equal direct Voronoi
 * labeling of the 6 seeds. Why: every voting point of an element lies within the element radius
 * r of its center c, and the center of the voxel holding the point lies within half a voxel
 * diagonal of the point. If c is farther than r + (voxel diagonal)/2 from every bisector of its
 * Voronoi cell, every such voxel center lies in that cell, so every voxel the element can see
 * carries the cell's feature id (voxel id = nearest seed of the voxel center), except the hole,
 * which does not vote; the element must get the cell's id. (This is the work package's
 * "voxel diagonal + element diameter" band, sharpened to radius + half diagonal so that enough
 * elements of this small 6-grain map are checked.) Closer elements are skipped; at least 10
 * elements per mesh must be checked. In addition, *all* labels (including the boundary band and
 * the hole) must equal a single-rank reference computed in the same run on the whole mesh with
 * MPI_COMM_SELF: the rank-count independence check for non-aligned meshes.
 */
TEST(VoxelLabeler, CoarseMeshesMatchVoronoiAwayFromBoundaries) {
    WriteFixtures();
    ms::VoxelLabelParameters vote;
    vote.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    vote.vote_levels = 2;
    const double half_voxel_diagonal = 0.5 * H * std::sqrt(3.0);
    struct Case {
        mfem::Element::Type type;
        int nx, ny, nz;
    };
    for (const Case c : {Case{mfem::Element::HEXAHEDRON, 9, 7, 5},
                         Case{mfem::Element::TETRAHEDRON, 6, 5, 4}}) {
        mfem::Mesh serial = BoxMesh(c.nx, c.ny, c.nz, c.type, NX * H, NY * H, NZ * H);
        Distributed dist = Distribute(serial);
        const ms::VoxelLabels labels = Label(
            *dist.pmesh, MPI_COMM_WORLD, Canonical("canonical.h5"), vote);
        EXPECT_EQ(labels.num_outside_points, 0);
        const std::vector<int> all = Gather(dist, labels.element_feature);

        // Single-rank reference (every rank computes it on its own copy of the whole mesh)
        Distributed self = Distribute(serial, MPI_COMM_SELF);
        const ms::VoxelLabels reference = Label(
            *self.pmesh, MPI_COMM_SELF, Canonical("canonical.h5"), vote);
        const std::vector<int> reference_all =
            Gather(self, reference.element_feature, MPI_COMM_SELF);
        EXPECT_EQ(labels.num_filled, reference.num_filled);
        if (Rank() == 0) {
            EXPECT_EQ(all, reference_all);
            int checked = 0;
            int mismatches = 0;
            for (int e = 0; e < serial.GetNE(); ++e) {
                std::array<double, 3> center{};
                double radius = 0.0;
                ElementCenterAndRadius(serial, e, center, radius);
                if (DistanceToBisector(center) <= radius + half_voxel_diagonal) {
                    continue;
                }
                ++checked;
                const int expected = kFeatureId[static_cast<std::size_t>(NearestSeed(center))];
                mismatches += (all[static_cast<std::size_t>(e)] != expected) ? 1 : 0;
            }
            EXPECT_GT(checked, 10) << serial.GetNE() << " elements";
            EXPECT_EQ(mismatches, 0);
            std::cout << "  coarse mesh: " << serial.GetNE() << " elements, " << checked
                      << " checked" << std::endl;
        }
    }
}

/**
 * A map with no assigned voxel: no element gets a vote and nothing can be filled, so every rank
 * throws the same error with the global element count. A mesh entirely outside the grid fails
 * the same way.
 */
TEST(VoxelLabeler, UnfillableElementsThrowEverywhere) {
    WriteFixtures();
    mfem::Mesh serial = BoxMesh(5, 4, 3, mfem::Element::HEXAHEDRON, NX * H, NY * H, NZ * H);
    Distributed dist = Distribute(serial);
    std::string error = ErrorOf([&] {
        Label(*dist.pmesh, MPI_COMM_WORLD, Canonical("empty.h5"),
                                    ms::VoxelLabelParameters{});
    });
    EXPECT_TRUE(Contains(error, "60 element(s)")) << error;

    ms::VoxelMapParameters far = Canonical("canonical.h5");
    far.translate = {100.0, 0.0, 0.0};
    error = ErrorOf([&] {
        Label(*dist.pmesh, MPI_COMM_WORLD, far, ms::VoxelLabelParameters{});
    });
    EXPECT_TRUE(Contains(error, "60 element(s)")) << error;

    ms::VoxelLabelParameters bad;
    bad.sampling = ms::SamplingMode::QUADRATURE_VOTE;
    bad.vote_levels = 4;
    EXPECT_THROW(Label(*dist.pmesh, MPI_COMM_WORLD,
                                             Canonical("canonical.h5"), bad),
                 std::invalid_argument);
}

/**
 * Compaction: on the aligned mesh the six feature ids {3, 5, 8, 13, 21, 34} become grains
 * 1..6 in ascending feature id, identically on every rank. Directly: rank r holds features
 * {100 + r, 7} (a different set per rank), so the grains are 7 -> 1, 100 -> 2, ...; a feature id
 * of 0 on any rank is refused on every rank.
 */
TEST(VoxelLabeler, CompactFeatureLabels) {
    WriteFixtures();
    mfem::Mesh serial = BoxMesh(NX, NY, NZ, mfem::Element::HEXAHEDRON, NX * H, NY * H, NZ * H);
    Distributed dist = Distribute(serial);
    const ms::VoxelLabels labels = Label(
        *dist.pmesh, MPI_COMM_WORLD, Canonical("canonical.h5"), ms::VoxelLabelParameters{});
    mfem::Array<int> element_grain;
    std::vector<std::int64_t> grain_feature;
    ms::CompactFeatureLabels(MPI_COMM_WORLD, labels.element_feature, element_grain,
                             grain_feature);
    EXPECT_EQ(grain_feature, (std::vector<std::int64_t>{3, 5, 8, 13, 21, 34}));
    const int* feature = labels.element_feature.HostRead();
    const int* grain = element_grain.HostRead();
    for (int e = 0; e < element_grain.Size(); ++e) {
        EXPECT_EQ(grain_feature[static_cast<std::size_t>(grain[e] - 1)], feature[e]);
    }

    const int rank = Rank();
    mfem::Array<int> mine({100 + rank, 7, 100 + rank});
    ms::CompactFeatureLabels(MPI_COMM_WORLD, mine, element_grain, grain_feature);
    std::vector<std::int64_t> expected = {7};
    for (int r = 0; r < NumRanks(); ++r) {
        expected.push_back(100 + r);
    }
    EXPECT_EQ(grain_feature, expected);
    const int* g = element_grain.HostRead();
    EXPECT_EQ(g[0], rank + 2);
    EXPECT_EQ(g[1], 1);
    EXPECT_EQ(g[2], rank + 2);

    mfem::Array<int> invalid({rank == NumRanks() - 1 ? 0 : 5});
    EXPECT_THROW(ms::CompactFeatureLabels(MPI_COMM_WORLD, invalid, element_grain, grain_feature),
                 std::invalid_argument);
}

#else // AXOM_USE_HDF5

/** Without HDF5 every reader refuses with a clear message, on every rank. */
TEST(VoxelMapReader, BuiltWithoutHdf5) {
    ms::VoxelMapParameters params;
    params.file = "anything.h5";
    const std::string error =
        ErrorOf([&] { ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD); });
    EXPECT_TRUE(Contains(error, "without HDF5")) << error;
}

#endif // AXOM_USE_HDF5

/** Initializes MPI (mfem::Mpi finalizes it at exit) and runs every test on every rank. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
