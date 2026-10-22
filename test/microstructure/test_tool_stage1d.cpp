/**
 * @file test_tool_stage1d.cpp
 * @brief Input writer of the Stage 1d end-to-end chain (test_tool_stage1d.cmake).
 *
 * The chain runs the microstructure_generator tool on data-driven inputs and then ExaConstit's
 * mechanics on the results. Three of its inputs are written here, at test time, into the
 * working directory of this test (<build>/test/microstructure/end_to_end_stage1d/inputs), instead
 * of being checked in:
 *   - e2e_two_phase.dream3d: a DREAM.3D (SIMPL v6 layout) voxel map with two phases and its
 *     per-feature quaternions stored x y z w (scalar last), as DREAM.3D writes them;
 *   - e2e_plate_box.mesh: a hex box whose two bottom element layers carry attribute 2 (a build
 *     plate) and the rest attribute 1 (the part), with ExaConstit's auto-mesh boundary ids;
 *   - e2e_spparks.dump: a tiny SPPARKS text site dump (simple cubic lattice) for the converter
 *     step (scripts/microstructure/spparks_to_voxels.py).
 * Each writer is one gtest, which also reads its file back and checks it, so a broken fixture
 * stops the chain at this step (the later steps require this one through a CTest fixture).
 *
 * Registered at 1 MPI rank only (it is the setup step of the chain), but it runs at any rank
 * count: rank 0 writes, then broadcasts whether the write succeeded (RootWriteSucceeded), so
 * every rank fails together if it did not.
 */
#include "microstructure/io/voxel_map_io.hpp"
#include "utilities/auto_mesh.hpp"

#include "common/voxel_fixture.hpp"

#include "axom/config.hpp"
#include "mfem.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <mpi.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

int Rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/**
 * Shares rank 0's "the file was written" verdict with every rank (MPI_Bcast, which also keeps
 * the other ranks from reading the file before rank 0 has finished it). Each test records its
 * write result in a bool on rank 0, calls this outside the rank-0 branch, and only then asserts:
 * an ASSERT inside the rank-0 branch would return from the test on rank 0 alone and leave the
 * other ranks waiting forever in the next collective call. Collective.
 */
bool RootWriteSucceeded(bool ok_on_root) {
    int ok = ok_on_root ? 1 : 0;
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
}

/**
 * Phase of feature id r in the two-phase DREAM.3D fixture: 1 + (r mod 2). The fixture's feature
 * ids {3, 5, 8, 13, 21, 34} give phases {2, 2, 1, 2, 2, 1}: phase 1 for features 8 and 34,
 * phase 2 for the other four. (DREAM.3D numbers phases from 1; phase 0 is its "unknown" phase.)
 */
int TwoPhase(int r) {
    return 1 + r % 2;
}

#ifdef AXOM_USE_HDF5
/**
 * Writes the two-phase DREAM.3D fixture (SIMPL v6 layout) described at TwoPhaseDream3d below.
 * Host, one rank. Throws std::runtime_error on HDF5 errors (voxel_fixture::H5Writer).
 */
void WriteTwoPhaseDream3d(const fs::path& path) {
    namespace vf = voxel_fixture;
    const vf::VoxelMapData& map = vf::MapData();
    vf::H5Writer w(path);
    const std::string dc = "/DataContainers/ImageDataContainer";
    const std::int64_t dims[3] = {vf::NX, vf::NY, vf::NZ};
    const float origin[3] = {0.0f, 0.0f, 0.0f};
    const float spacing[3] = {static_cast<float>(vf::H), static_cast<float>(vf::H),
                              static_cast<float>(vf::H)};
    w.Dataset(dc + "/_SIMPL_GEOMETRY/DIMENSIONS", H5T_STD_I64LE, H5T_NATIVE_INT64, {3}, dims);
    w.Dataset(dc + "/_SIMPL_GEOMETRY/ORIGIN", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {3}, origin);
    w.Dataset(dc + "/_SIMPL_GEOMETRY/SPACING", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {3},
              spacing);
    // Per-voxel feature ids (shape nz, ny, nx, 1; x fastest), with the hole (id 0)
    const std::vector<int> ids = vf::AsInt32(map.ids);
    w.Dataset(dc + "/CellData/FeatureIds", H5T_STD_I32LE, H5T_NATIVE_INT,
              {vf::NZ, vf::NY, vf::NX, 1}, ids.data());
    // Per-feature rows: row r is feature id r, row 0 unused
    const std::vector<float> quats = vf::Dream3dQuatsXyzw();
    w.Dataset(dc + "/CellFeatureData/AvgQuats", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
              {vf::kRows, 4}, quats.data());
    std::vector<int> phases(vf::kRows, 0);
    for (const int id : vf::kFeatureId) {
        phases[static_cast<std::size_t>(id)] = TwoPhase(id);
    }
    w.Dataset(dc + "/CellFeatureData/Phases", H5T_STD_I32LE, H5T_NATIVE_INT, {vf::kRows, 1},
              phases.data());
}
#endif

} // namespace

// The two-phase DREAM.3D voxel map: the shared 20 x 16 x 12 voxelized Voronoi map of
// common/voxel_fixture.hpp (six features, a 2 x 2 x 2 hole of id 0, voxel size 1/16, origin 0),
// written in the SIMPL v6 layout like voxel_fixture::WriteSimpl, but with two phases
// (TwoPhase) instead of three, so the generator's "from_phase" regions give exactly two
// ExaConstit material regions. The quaternions are float32 x y z w (scalar last), which the
// generator input declares with quat_layout = "xyzw", conjugate = false. The read-back goes
// through the generator's own reader (ReadVoxelGridInfo / ReadVoxelFeatures) with those
// declarations, so it checks that the file says what the end-to-end input assumes: the grid,
// and for every feature its phase and its quaternion reordered to w x y z (float32 precision).
TEST(EndToEndInputs, TwoPhaseDream3d) {
#ifdef AXOM_USE_HDF5
    namespace vf = voxel_fixture;
    const fs::path path = fs::current_path() / "e2e_two_phase.dream3d";
    bool written = true;
    std::string write_error;
    if (Rank() == 0) {
        // The fixture writer throws on HDF5 errors; catch them so rank 0 still reaches the
        // collective verdict below
        try {
            WriteTwoPhaseDream3d(path);
        } catch (const std::exception& e) {
            written = false;
            write_error = e.what();
        }
    }
    ASSERT_TRUE(RootWriteSucceeded(written)) << "rank 0 could not write " << path << " "
                                             << write_error;

    // Read back with the declarations the end-to-end input uses
    ms::VoxelMapParameters params;
    params.format = ms::VoxelFormat::DREAM3D;
    params.file = path;
    params.read_quats = true;
    params.read_phases = true;
    params.quat_layout = ms::QuatLayout::XYZW;
    params.conjugate = 0;
    const ms::VoxelGridInfo grid = ms::ReadVoxelGridInfo(params, MPI_COMM_WORLD);
    EXPECT_EQ(grid.dims, (std::array<std::int64_t, 3>{vf::NX, vf::NY, vf::NZ}));
    EXPECT_EQ(grid.spacing, (std::array<double, 3>{vf::H, vf::H, vf::H}));
    EXPECT_EQ(grid.origin, (std::array<double, 3>{0.0, 0.0, 0.0}));
    const ms::VoxelFeatureTable features =
        ms::ReadVoxelFeatures(params, grid, vf::kUsedIds, MPI_COMM_WORLD);
    std::map<std::int64_t, std::size_t> row;
    for (std::size_t k = 0; k < features.id.size(); ++k) {
        row[features.id[k]] = k;
    }
    int phase1 = 0;
    int phase2 = 0;
    for (const int id : vf::kFeatureId) {
        ASSERT_EQ(row.count(id), 1u) << "feature " << id;
        const std::size_t k = row[id];
        EXPECT_EQ(features.phase[k], TwoPhase(id));
        phase1 += (features.phase[k] == 1) ? 1 : 0;
        phase2 += (features.phase[k] == 2) ? 1 : 0;
        const std::array<double, 4> q = vf::FeatureQuat(id);
        for (std::size_t c = 0; c < 4; ++c) {
            // Stored as float32, so the reordered value is the float rounding of the original
            EXPECT_EQ(features.quat[4 * k + c], static_cast<double>(static_cast<float>(q[c])))
                << "feature " << id << " component " << c;
        }
    }
    EXPECT_EQ(phase1, 2);
    EXPECT_EQ(phase2, 4);
#else
    GTEST_SKIP() << "built without HDF5: the DREAM.3D end-to-end input cannot be written";
#endif
}

// The build-plate mesh: ExaConstit's auto box (MakeAutoHexMesh) of 8 x 8 x 10 hexes on
// [0, 1] x [0, 1] x [0, 1.25], with ExaConstit's auto-mesh boundary ids (1 bottom, 2 left,
// 3 front, 4 top, ...; ApplyAutoMeshBoundaryAttributes), so the mechanics deck uses the same
// boundary ids as the other end-to-end decks. Elements in the two bottom layers (z < 0.25) get
// attribute 2 (the plate, 8 x 8 x 2 = 128 elements), the rest attribute 1 (the part, 512
// elements). Written with 17 significant digits and read back to check the counts.
TEST(EndToEndInputs, BuildPlateMesh) {
    const fs::path path = fs::current_path() / "e2e_plate_box.mesh";
    bool written = true;
    if (Rank() == 0) {
        mfem::Mesh mesh = exaconstit::MakeAutoHexMesh({8, 8, 10}, {1.0, 1.0, 1.25});
        exaconstit::ApplyAutoMeshBoundaryAttributes(mesh);
        for (int e = 0; e < mesh.GetNE(); ++e) {
            // Element centroid height from its vertices (layers are 0.125 thick)
            mfem::Array<int> vertices;
            mesh.GetElementVertices(e, vertices);
            double z = 0.0;
            for (int k = 0; k < vertices.Size(); ++k) {
                z += mesh.GetVertex(vertices[k])[2];
            }
            z /= vertices.Size();
            mesh.SetAttribute(e, (z < 0.25) ? 2 : 1);
        }
        mesh.SetAttributes();
        std::ofstream out(path);
        out.precision(17);
        mesh.Print(out);
        written = out.good();
    }
    ASSERT_TRUE(RootWriteSucceeded(written)) << "rank 0 could not write " << path;

    // Read back the way the generator reads a mesh file
    mfem::Mesh read(path.string().c_str(), 1, 1, true);
    int plate = 0;
    int part = 0;
    for (int e = 0; e < read.GetNE(); ++e) {
        plate += (read.GetAttribute(e) == 2) ? 1 : 0;
        part += (read.GetAttribute(e) == 1) ? 1 : 0;
    }
    EXPECT_EQ(plate, 128);
    EXPECT_EQ(part, 512);
    EXPECT_EQ(read.bdr_attributes.Max(), 6);
}

// The SPPARKS site dump: a simple cubic lattice of 8 x 8 x 8 sites at integer positions
// 0..7 (lattice spacing 1, box [0, 8]^3, as SPPARKS writes an sc/26n lattice of 8 cells), in the
// text format of SPPARKS' `dump text` command (ITEM: TIMESTEP / NUMBER OF ATOMS / BOX BOUNDS /
// ATOMS id type x y z; "type" is the site's spin). Spins: four columns along z,
// spin = 1 + (x >= 4) + 2 (y >= 4), so four grains; plus two defects the generator must repair:
//   - site (1, 1, 1) has spin 0 (unassigned): the voxel labeler fills that element from its
//     neighbors (grain 1);
//   - site (1, 1, 5) has spin 4, a one-voxel island of grain 4 inside grain 1 (a Potts model
//     with few spin values produces such repeated spins): the fragment cleanup merges it into
//     grain 1 (data-driven sources merge every extra piece).
// So the generator must report exactly 4 grains. The read-back counts the sites per spin.
TEST(EndToEndInputs, SpparksDump) {
    const fs::path path = fs::current_path() / "e2e_spparks.dump";
    const int n = 8;
    auto spin = [](int i, int j, int k) {
        if (i == 1 && j == 1 && k == 1) {
            return 0;
        }
        if (i == 1 && j == 1 && k == 5) {
            return 4;
        }
        return 1 + (i >= 4 ? 1 : 0) + 2 * (j >= 4 ? 1 : 0);
    };
    bool written = true;
    if (Rank() == 0) {
        std::ofstream out(path);
        out << "ITEM: TIMESTEP\n0 0\nITEM: NUMBER OF ATOMS\n" << n * n * n
            << "\nITEM: BOX BOUNDS\n0 8\n0 8\n0 8\nITEM: ATOMS id type x y z\n";
        int id = 1;
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    out << id++ << " " << spin(i, j, k) << " " << i << " " << j << " " << k
                        << "\n";
                }
            }
        }
        written = out.good();
    }
    ASSERT_TRUE(RootWriteSucceeded(written)) << "rank 0 could not write " << path;

    // Read back: 512 site lines after the header, 128 sites per column grain minus the defects
    std::ifstream in(path);
    std::string line;
    for (int k = 0; k < 9; ++k) {
        std::getline(in, line);
    }
    std::map<int, int> count;
    int id = 0;
    int s = 0;
    double x = 0.0, y = 0.0, z = 0.0;
    while (in >> id >> s >> x >> y >> z) {
        count[s] += 1;
    }
    EXPECT_EQ(count, (std::map<int, int>{{0, 1}, {1, 126}, {2, 128}, {3, 128}, {4, 129}}));
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
