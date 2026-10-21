#include "microstructure/exec_dispatch.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"
#include "utilities/auto_mesh.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

namespace ms = exaconstit::microstructure;

namespace {

/**
 * @brief The process's mfem::Device. MFEM allows one Device object to be configured, once, so
 * all tests share this one (it lives until exit).
 */
mfem::Device& TestDevice() {
    static mfem::Device device;
    return device;
}

} // namespace

/*
 * Build-system smoke test of the header-only ms_core contracts (exec dispatch, options and
 * result defaults) and of the auto-mesh helper shared with ExaConstit's makeMesh. If this
 * executable compiles, links and runs, the microstructure CMake wiring is sound.
 */

// The `exec` option values are recognized: "cpu" is always available, unknown names (including
// MFEM's own device strings such as "raja-omp") are not, and "openmp"/"gpu" follow the build's
// MFEM/Axom configuration. ConfigureMfemDevice refuses an unavailable value before touching the
// device, instead of silently running on the CPU.
TEST(MicrostructureSkeleton, ExecAvailability) {
    EXPECT_TRUE(ms::IsExecAvailable("cpu"));
    EXPECT_FALSE(ms::IsExecAvailable("seq"));
    EXPECT_FALSE(ms::IsExecAvailable("raja-omp"));
    EXPECT_EQ(ms::IsExecAvailable("openmp"), ms::EXEC_OPENMP_AVAILABLE);
    EXPECT_EQ(ms::IsExecAvailable("gpu"), ms::EXEC_GPU_AVAILABLE);

    // These throw before the device is touched, so TestDevice() stays unconfigured
    EXPECT_THROW(ms::ConfigureMfemDevice(TestDevice(), "bogus"), std::invalid_argument);
    if (!ms::EXEC_GPU_AVAILABLE) {
        EXPECT_THROW(ms::ConfigureMfemDevice(TestDevice(), "gpu"), std::invalid_argument);
    }
}

// With the mfem::Device configured for "cpu", BVH dispatch hands the generic lambda
// axom::SEQ_EXEC, and BVH arrays use Axom's host allocator (the id Axom reports for SEQ_EXEC).
TEST(MicrostructureSkeleton, DispatchAxomExecOnCpuDevice) {
    ms::ConfigureMfemDevice(TestDevice(), "cpu");
    EXPECT_TRUE(mfem::Device::Allows(mfem::Backend::CPU));

    bool called_seq = false;
    ms::DispatchAxomExec([&](auto exec) {
        called_seq = std::is_same_v<decltype(exec), axom::SEQ_EXEC>;
    });
    EXPECT_TRUE(called_seq);
    EXPECT_EQ(ms::AxomAllocatorIdForCurrentDevice(),
              axom::execution_space<axom::SEQ_EXEC>::allocatorID());
}

// The documented defaults of the shared contracts (the "-n N" quick path: Voronoi, rejection
// seeds, serial output, CPU) hold, so later packages can rely on them.
TEST(MicrostructureSkeleton, ContractDefaults) {
    const ms::MicrostructureOptions options;
    EXPECT_EQ(options.exec, "cpu");
    EXPECT_EQ(options.tessellation.metric, ms::MetricType::VORONOI);
    EXPECT_EQ(options.seeds.placement, ms::SeedPlacement::REJECTION);
    EXPECT_EQ(options.output.mesh_format, ms::MeshFormat::SERIAL);

    ms::SeedSet seeds;
    seeds.x = {0.0, 1.0};
    EXPECT_EQ(seeds.Size(), 2u);

    const ms::MicrostructureResult result;
    EXPECT_EQ(result.num_grains, 0);
    EXPECT_EQ(result.num_regions, 1);
}

// The auto mesh is a 2x3x4 hex box over [0,1]x[0,2]x[0,3] in lexicographic element order (element
// 0 at the origin, x fastest), and after the remap every boundary face carries the ExaConstit
// attribute of its side (1 bottom, 2 left, 3 front, 4 top, 5 right, 6 back).
TEST(MicrostructureSkeleton, AutoMesh) {
    mfem::Mesh mesh = exaconstit::MakeAutoHexMesh({2, 3, 4}, {1.0, 2.0, 3.0});
    EXPECT_EQ(mesh.GetNE(), 24);
    EXPECT_EQ(mesh.GetElementType(0), mfem::Element::HEXAHEDRON);

    // Lexicographic order: element 0 sits at the origin corner, x varies fastest. The element
    // size in x is 1/2, so the first two centers are x = 0.25 and x = 0.75 at the same y.
    mfem::Vector c0(3), c1(3);
    mesh.GetElementCenter(0, c0);
    mesh.GetElementCenter(1, c1);
    EXPECT_DOUBLE_EQ(c0(0), 0.25);
    EXPECT_DOUBLE_EQ(c1(0), 0.75);
    EXPECT_DOUBLE_EQ(c1(1), c0(1));

    // MFEM face order bottom, front, right, back, left, top -> ExaConstit 1, 3, 5, 6, 2, 4;
    // check by the face position after the remap. The face center (mean of its vertices) lies
    // exactly on one of the six box planes; the box corner coordinates are exact in binary.
    exaconstit::ApplyAutoMeshBoundaryAttributes(mesh);
    mfem::Array<int> verts;
    for (int i = 0; i < mesh.GetNBE(); ++i) {
        mesh.GetBdrElementVertices(i, verts);
        double mean[3] = {0.0, 0.0, 0.0};
        for (int v : verts) {
            for (int d = 0; d < 3; ++d) {
                mean[d] += mesh.GetVertex(v)[d] / verts.Size();
            }
        }
        int expected = 0;
        if (mean[2] == 0.0) {
            expected = 1; // bottom (z = 0)
        } else if (mean[0] == 0.0) {
            expected = 2; // left (x = 0)
        } else if (mean[1] == 0.0) {
            expected = 3; // front (y = 0)
        } else if (mean[2] == 3.0) {
            expected = 4; // top
        } else if (mean[0] == 1.0) {
            expected = 5; // right
        } else if (mean[1] == 2.0) {
            expected = 6; // back
        }
        EXPECT_EQ(mesh.GetBdrAttribute(i), expected) << "boundary element " << i;
    }
}

// gtest driver: MFEM's MPI wrapper initializes MPI (the ms_* code is MPI-aware) and finalizes it at
// exit.
int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
